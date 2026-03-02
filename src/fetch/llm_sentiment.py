"""
Claude CLI を使った市場センチメント分析。

yfinance からマクロ市場のニュースを取得し、Claude CLI に渡して
翌日の市場センチメントスコアと除外推奨銘柄を取得する。

出力:
  - market_sentiment: float (-1.0 〜 +1.0)  弱気 〜 強気
  - confidence: float (0.0 〜 1.0)
  - risk_tickers: list[str]  除外推奨銘柄コード（4桁）
  - reasoning: str
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

# マクロ指標のニュースソース
MACRO_TICKERS = ["^N225", "^GSPC", "^VIX", "USDJPY=X", "^IXIC"]

# Claude CLI タイムアウト（秒）
CLAUDE_TIMEOUT = 120


def _fetch_news_headlines(max_per_ticker: int = 5) -> list[dict]:
    """
    マクロ指標ティッカーからニュースヘッドラインを取得する。

    Returns
    -------
    list[dict]
        {"title": str, "summary": str, "pubDate": str} のリスト
    """
    headlines = []
    seen_titles: set[str] = set()

    for symbol in MACRO_TICKERS:
        try:
            news = yf.Ticker(symbol).news or []
            for item in news[:max_per_ticker]:
                content = item.get("content", {})
                title = content.get("title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                headlines.append({
                    "title": title,
                    "summary": content.get("summary", "").strip(),
                    "pubDate": content.get("pubDate", ""),
                })
        except Exception as exc:
            logger.warning("%s のニュース取得失敗: %s", symbol, exc)

    # 新しい順にソート
    headlines.sort(key=lambda x: x.get("pubDate", ""), reverse=True)
    return headlines


def _build_prompt(
    market_data: dict,
    headlines: list[dict],
    top_tickers: list[str] | None = None,
) -> str:
    """
    Claude CLI に渡すプロンプトを構築する。

    Parameters
    ----------
    market_data : dict
        当日の市場数値。例: {"nikkei_ret": -1.2, "vix": 18.5, ...}
    headlines : list[dict]
        ニュースヘッドラインのリスト
    top_tickers : list[str] | None
        LGBMが選んだ上位銘柄コード（4桁）のリスト
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 市場数値セクション
    mkt_lines = []
    if "nikkei_ret" in market_data:
        mkt_lines.append(f"- 日経225: {market_data['nikkei_ret']:+.2f}%")
    if "topix_ret" in market_data:
        mkt_lines.append(f"- TOPIX: {market_data['topix_ret']:+.2f}%")
    if "vix" in market_data:
        mkt_lines.append(f"- VIX: {market_data['vix']:.1f}")
    if "usdjpy" in market_data:
        mkt_lines.append(f"- USD/JPY: {market_data['usdjpy']:.2f}")
    if "sp500_ret" in market_data:
        mkt_lines.append(f"- S&P500: {market_data['sp500_ret']:+.2f}%")
    mkt_section = "\n".join(mkt_lines) if mkt_lines else "（データなし）"

    # ニュースセクション
    news_lines = []
    for i, h in enumerate(headlines[:15], 1):
        summary = f" — {h['summary']}" if h["summary"] else ""
        news_lines.append(f"{i}. {h['title']}{summary}")
    news_section = "\n".join(news_lines) if news_lines else "（ニュースなし）"

    # 推薦銘柄セクション
    tickers_section = ""
    if top_tickers:
        tickers_section = f"""
## LGBMモデルの上位推薦銘柄（4桁コード）
{", ".join(top_tickers[:30])}

上記銘柄の中で、決算発表直前・不祥事・業績悪化ニュース等のリスクがある銘柄があれば
risk_tickers に含めてください（情報がない場合は空リストで構いません）。
"""

    prompt = f"""あなたは日本株デイトレードの市場アナリストです。
以下の情報を基に、**明日（{today}の翌営業日）の東証市場**の方向性を分析してください。

## 本日の市場概況
{mkt_section}

## 主要ニュースヘッドライン
{news_section}
{tickers_section}
## 回答形式（必ずこのJSONのみを返してください）
```json
{{
  "market_sentiment": <-1.0〜1.0の数値。-1.0=強い弱気、0=中立、1.0=強い強気>,
  "confidence": <0.0〜1.0の確信度>,
  "risk_tickers": [<除外推奨の4桁銘柄コード文字列のリスト>],
  "reasoning": "<100字以内の判断理由>"
}}
```"""
    return prompt


def _call_claude_cli(prompt: str) -> str:
    """
    Claude CLI を呼び出してレスポンスを返す。
    CLAUDECODE 環境変数を unset して実行する（ネスト制限を回避）。
    """
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI エラー: {result.stderr[:200]}")
    return result.stdout.strip()


def _parse_response(raw: str) -> dict:
    """
    Claude のレスポンスから JSON を抽出してパースする。
    """
    # ```json ... ``` ブロックを優先して抽出
    m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    json_str = m.group(1) if m else raw

    # JSONブロックが見つからない場合は {} 部分を探す
    if not m:
        m2 = re.search(r"\{.*\}", json_str, re.DOTALL)
        if m2:
            json_str = m2.group(0)

    data = json.loads(json_str)

    return {
        "market_sentiment": float(data.get("market_sentiment", 0.0)),
        "confidence": float(data.get("confidence", 0.5)),
        "risk_tickers": [str(t) for t in data.get("risk_tickers", [])],
        "reasoning": str(data.get("reasoning", "")),
    }


def get_llm_sentiment(
    market_data: dict | None = None,
    top_tickers: list[str] | None = None,
) -> dict:
    """
    Claude CLI を使って市場センチメントを取得する。

    Parameters
    ----------
    market_data : dict | None
        当日の市場数値（なければ空辞書）
    top_tickers : list[str] | None
        LGBMの上位推薦銘柄コード（4桁）

    Returns
    -------
    dict
        {
            "market_sentiment": float,  # -1.0 〜 1.0
            "confidence": float,        # 0.0 〜 1.0
            "risk_tickers": list[str],  # 除外推奨銘柄
            "reasoning": str,
        }
        エラー時はデフォルト値（sentiment=0, confidence=0）を返す。
    """
    _default = {
        "market_sentiment": 0.0,
        "confidence": 0.0,
        "risk_tickers": [],
        "reasoning": "取得失敗",
    }

    try:
        logger.info("ニュースヘッドラインを取得します...")
        headlines = _fetch_news_headlines()
        logger.info("ニュース %d 件取得", len(headlines))

        prompt = _build_prompt(
            market_data=market_data or {},
            headlines=headlines,
            top_tickers=top_tickers,
        )

        logger.info("Claude CLI を呼び出します...")
        raw = _call_claude_cli(prompt)
        logger.debug("Claude レスポンス: %s", raw[:300])

        result = _parse_response(raw)
        logger.info(
            "センチメント: %.2f (確信度: %.2f) — %s",
            result["market_sentiment"],
            result["confidence"],
            result["reasoning"],
        )
        return result

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI タイムアウト（%d秒）", CLAUDE_TIMEOUT)
        return _default
    except json.JSONDecodeError as exc:
        logger.warning("Claude レスポンスのJSON解析失敗: %s", exc)
        return _default
    except Exception as exc:
        logger.warning("LLMセンチメント取得失敗: %s", exc)
        return _default
