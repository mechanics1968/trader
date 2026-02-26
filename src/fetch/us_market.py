"""
米国市場データ取得モジュール

日本株の翌営業日始値は前日の米国市場動向に強く連動する。
S&P500・NASDAQ・VIX・ドル円・CME日経先物の日次データを取得し、
特徴量として提供する。

取得ティッカー:
  ^GSPC   : S&P 500
  ^IXIC   : NASDAQ Composite
  ^VIX    : CBOE Volatility Index
  USDJPY=X: ドル円レート
  NKD=F   : CME Nikkei 225 Futures (USD)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/us_market.csv")
US_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
    "usdjpy": "USDJPY=X",
    "nk_futures": "NKD=F",
}


def fetch_us_market(refresh: bool = False) -> pd.DataFrame:
    """米国市場データを取得して日次特徴量 DataFrame を返す。

    Returns
    -------
    pd.DataFrame
        index: Date（日本時間の営業日に合わせた日付）
        columns:
          sp500_ret      : S&P500 前日比リターン
          nasdaq_ret     : NASDAQ 前日比リターン
          vix_level      : VIX 終値レベル
          vix_change     : VIX 前日差
          usdjpy_ret     : ドル円前日比リターン
          nk_futures_ret : CME日経先物前日比リターン
    """
    if not refresh and CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        logger.info("米国市場データをキャッシュから読み込みました: %d 行", len(df))
        return df

    frames: dict[str, pd.Series] = {}
    for name, ticker in US_TICKERS.items():
        try:
            raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if raw.empty:
                logger.warning("米国市場データ取得失敗: %s (%s)", name, ticker)
                continue
            close = raw["Close"].squeeze()
            frames[name] = close
            logger.debug("取得: %s (%s) %d 行", name, ticker, len(close))
        except Exception as exc:
            logger.warning("米国市場データ取得エラー %s: %s", ticker, exc)

    if not frames:
        logger.error("米国市場データがひとつも取得できませんでした")
        return pd.DataFrame()

    df = pd.DataFrame(frames).sort_index()

    # 特徴量計算
    result = pd.DataFrame(index=df.index)

    if "sp500" in df.columns:
        result["sp500_ret"] = df["sp500"].pct_change(1)
        result["sp500_ret_5d"] = df["sp500"].pct_change(5)

    if "nasdaq" in df.columns:
        result["nasdaq_ret"] = df["nasdaq"].pct_change(1)

    if "vix" in df.columns:
        result["vix_level"] = df["vix"]
        result["vix_change"] = df["vix"].diff(1)
        # VIX が20超 = 高ボラ局面フラグ
        result["vix_high"] = (df["vix"] > 20).astype(float)

    if "usdjpy" in df.columns:
        result["usdjpy_ret"] = df["usdjpy"].pct_change(1)
        result["usdjpy_level"] = df["usdjpy"]

    if "nk_futures" in df.columns:
        result["nk_futures_ret"] = df["nk_futures"].pct_change(1)

    result = result.dropna(how="all")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(CACHE_PATH)
    logger.info("米国市場データを保存しました: %s (%d 行)", CACHE_PATH, len(result))
    return result
