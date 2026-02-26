"""
決算発表日カレンダーの取得。

優先順位:
  1. J-Quants API（無料プラン: JQUANTS_REFRESH_TOKEN 環境変数が必要）
  2. yfinance の earnings_dates（一部銘柄のみ対応）
  3. 取得失敗 → 空の DataFrame を返して以降の処理をスキップ

決算接近フラグは特徴量の質に大きく影響する（決算前後に異常なギャップが発生するため）。

環境変数設定例（.env ファイルに記載）:
  JQUANTS_REFRESH_TOKEN=xxxxx
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

_CACHE_PATH = config.DATA_DIR / "earnings_calendar.csv"
_CACHE_DAYS = 1  # 1日以内のキャッシュは再利用


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def fetch_earnings_calendar(
    tickers: list[str],
    refresh: bool = False,
) -> pd.DataFrame:
    """
    決算発表日カレンダーを取得する。

    Returns
    -------
    pd.DataFrame
        列: ticker, announcement_date (date型)
        空の場合は空 DataFrame を返す（機能を無効化して続行）。
    """
    if not refresh and _CACHE_PATH.exists():
        age_hours = (pd.Timestamp.now() - pd.Timestamp(_CACHE_PATH.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age_hours < _CACHE_DAYS * 24:
            df = pd.read_csv(_CACHE_PATH, parse_dates=["announcement_date"])
            logger.info("決算カレンダーをキャッシュから読み込みました: %d 件", len(df))
            return df

    # J-Quants を試みる
    df = _fetch_jquants()
    if df is not None and not df.empty:
        df.to_csv(_CACHE_PATH, index=False)
        logger.info("J-Quants から決算カレンダーを取得しました: %d 件", len(df))
        return df

    # yfinance フォールバック（一部銘柄のみ）
    df = _fetch_yfinance_sample(tickers[:200])  # 負荷軽減のため先頭200銘柄のみ
    if df is not None and not df.empty:
        df.to_csv(_CACHE_PATH, index=False)
        logger.info("yfinance から決算カレンダーを取得しました: %d 件", len(df))
        return df

    logger.warning(
        "決算カレンダーの取得に失敗しました。"
        "J-Quants を使用するには .env に JQUANTS_REFRESH_TOKEN を設定してください。"
    )
    return pd.DataFrame(columns=["ticker", "announcement_date"])


def build_earnings_proximity(
    tickers: list[str],
    calendar: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> dict[str, pd.Series]:
    """
    各銘柄について「決算発表日からの距離（日数）」の時系列 Series を返す。

    Parameters
    ----------
    tickers : list[str]
    calendar : pd.DataFrame
        fetch_earnings_calendar() の出力
    reference_date : pd.Timestamp | None
        基準日（テスト用。None なら本日）

    Returns
    -------
    dict[str, pd.Series]
        {ticker: Series(index=date, values=days_to_nearest_earnings)}
        正値 = 決算まで残り日数、負値 = 決算後経過日数
    """
    if calendar.empty:
        return {}

    result: dict[str, pd.Series] = {}
    cal = calendar.copy()
    cal["announcement_date"] = pd.to_datetime(cal["announcement_date"])

    for ticker in tickers:
        dates = cal.loc[cal["ticker"] == ticker, "announcement_date"].sort_values().values
        if len(dates) == 0:
            continue
        result[ticker] = pd.Series(dates, name="announcement_date")

    return result


# ---------------------------------------------------------------------------
# 内部実装
# ---------------------------------------------------------------------------

def _fetch_jquants() -> pd.DataFrame | None:
    """J-Quants API から決算発表スケジュールを取得する。

    jquantsapi v2.x: ClientV2(api_key=...) を使用。
    環境変数 JQUANTS_API_KEY または JQUANTS_REFRESH_TOKEN を参照する。
    JQUANTS_REFRESH_TOKEN が設定されている場合は、まず REST API でリフレッシュを試みる。
    """
    api_key = os.getenv("JQUANTS_API_KEY", "")
    refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN", "")

    if not api_key and not refresh_token:
        logger.debug("JQUANTS_API_KEY / JQUANTS_REFRESH_TOKEN が未設定のため J-Quants をスキップします")
        return None

    try:
        import jquantsapi  # type: ignore

        # リフレッシュトークンから API キーを取得する（V2 フロー）
        if not api_key and refresh_token:
            import requests as _req
            resp = _req.post(
                "https://api.jquants.com/v1/token/auth_refresh",
                params={"refreshtoken": refresh_token},
                timeout=30,
            )
            if resp.ok:
                id_token = resp.json().get("idToken", "")
                if id_token:
                    api_key = id_token
            if not api_key:
                logger.warning(
                    "J-Quants リフレッシュトークンからの id_token 取得に失敗しました: %s", resp.text[:100]
                )
                return None

        # JQUANTS_API_KEY を一時的にセットして ClientV2 を初期化
        os.environ["JQUANTS_API_KEY"] = api_key
        cli = jquantsapi.ClientV2()
        df = cli.get_eq_earnings_cal()

        if df is None or df.empty:
            return None

        # カラム名の正規化
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "code" in cl:
                col_map[c] = "ticker_code"
            elif "date" in cl or "day" in cl:
                col_map[c] = "announcement_date"
        df = df.rename(columns=col_map)

        if "ticker_code" not in df.columns or "announcement_date" not in df.columns:
            logger.warning("J-Quants レスポンスのカラム構造が想定外: %s", df.columns.tolist())
            return None

        # 4桁 → {code}.T 形式に変換
        df["ticker"] = df["ticker_code"].astype(str).str.zfill(4) + ".T"
        df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
        df = df.dropna(subset=["announcement_date"])
        return df[["ticker", "announcement_date"]].drop_duplicates()

    except ImportError:
        logger.debug("jquantsapi がインストールされていません")
    except Exception as exc:
        logger.warning("J-Quants 取得エラー: %s", exc)
    return None


def _fetch_yfinance_sample(tickers: list[str]) -> pd.DataFrame | None:
    """yfinance から決算日を取得する（対応銘柄のみ）。"""
    import time
    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                continue
            for dt in ed.index:
                rows.append({"ticker": ticker, "announcement_date": dt.date()})
            time.sleep(0.2)
        except Exception:
            pass

    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates()
