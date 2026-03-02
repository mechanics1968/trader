"""
市場センチメント分析を単独で実行するスクリプト。

使い方:
  python scripts/run_sentiment.py
  python scripts/run_sentiment.py --sp500 -0.5 --vix 18.5 --usdjpy 149.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetch.llm_sentiment import get_llm_sentiment
from src.fetch.us_market import fetch_us_market


def main() -> None:
    parser = argparse.ArgumentParser(description="市場センチメント分析")
    parser.add_argument("--sp500", type=float, default=None, help="S&P500前日比(%)")
    parser.add_argument("--vix", type=float, default=None, help="VIX水準")
    parser.add_argument("--usdjpy", type=float, default=None, help="USD/JPY水準")
    parser.add_argument("--top-tickers", type=str, default=None,
                        help="LGBMの上位推薦銘柄（カンマ区切り4桁コード）")
    args = parser.parse_args()

    # 市場数値の組み立て
    market_data: dict = {}

    # 引数が渡された場合はそれを使う、なければキャッシュから自動取得
    if any([args.sp500 is not None, args.vix is not None, args.usdjpy is not None]):
        if args.sp500 is not None:
            market_data["sp500_ret"] = args.sp500
        if args.vix is not None:
            market_data["vix"] = args.vix
        if args.usdjpy is not None:
            market_data["usdjpy"] = args.usdjpy
    else:
        try:
            us = fetch_us_market(refresh=False)
            if us is not None and not us.empty:
                last = us.iloc[-1]
                market_data["sp500_ret"] = float(last.get("sp500_ret", 0) * 100)
                market_data["vix"] = float(last.get("vix_level", 0))
                market_data["usdjpy"] = float(last.get("usdjpy_level", 0))
        except Exception:
            pass  # データなしでも続行

    top_tickers = None
    if args.top_tickers:
        top_tickers = [t.strip() for t in args.top_tickers.split(",") if t.strip()]

    print("=" * 60)
    print("  市場センチメント分析")
    print("=" * 60)
    if market_data:
        print("【入力データ】")
        for k, v in market_data.items():
            print(f"  {k}: {v}")
    else:
        print("【入力データ】なし（ニュースのみで判断）")
    print()

    result = get_llm_sentiment(market_data=market_data, top_tickers=top_tickers)

    sentiment = result["market_sentiment"]
    confidence = result["confidence"]
    risk_tickers = result["risk_tickers"]
    reasoning = result["reasoning"]

    # センチメントを視覚化
    bar_len = int(abs(sentiment) * 20)
    if sentiment >= 0:
        bar = "+" * bar_len
        direction = "強気"
    else:
        bar = "-" * bar_len
        direction = "弱気"

    print("【分析結果】")
    print(f"  センチメント : {sentiment:+.2f}  [{bar:<20}]  {direction}")
    print(f"  確信度       : {confidence:.2f}")
    print(f"  判断理由     : {reasoning}")
    if risk_tickers:
        print(f"  リスク銘柄   : {', '.join(risk_tickers)}")
    else:
        print("  リスク銘柄   : なし")

    print()
    print("【推薦への影響】")
    llm_up_prob = (sentiment + 1.0) / 2.0
    print(f"  LLM上昇確率  : {llm_up_prob:.3f}")
    print()

    # JSON出力（パイプ連携用）
    print("【JSON】")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
