"""
Yahoo Finance から東証銘柄の OHLCV データを取得し、
data/raw/{ticker}.csv に保存する。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str],
    period: str = config.PRICE_PERIOD,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    複数銘柄の OHLCV を並列取得する。

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance 形式のティッカーリスト（例: ["7203.T", ...]）
    period : str
        yfinance の period 文字列（デフォルト: "1y"）
    refresh : bool
        True の場合はキャッシュを無視して再取得する

    Returns
    -------
    dict[str, pd.DataFrame]
        {ticker: DataFrame(Open, High, Low, Close, Volume)} の辞書
    """
    results: dict[str, pd.DataFrame] = {}
    targets = tickers if refresh else _filter_uncached(tickers)

    if not targets:
        logger.info("すべてキャッシュ済みです。キャッシュから読み込みます。")
    else:
        logger.info("%d 銘柄を取得します（並列数: %d）", len(targets), config.FETCH_MAX_WORKERS)
        _fetch_parallel(targets, period, results)

    # キャッシュ済みのものをマージ
    for ticker in tickers:
        if ticker not in results:
            df = _load_cache(ticker)
            if df is not None:
                results[ticker] = df

    return results


def fetch_single(ticker: str, period: str = config.PRICE_PERIOD) -> pd.DataFrame | None:
    """
    1銘柄の OHLCV を取得してキャッシュに保存する。

    Returns
    -------
    pd.DataFrame | None
        取得成功時は DataFrame、失敗時は None
    """
    for attempt in range(1, config.FETCH_RETRY + 1):
        try:
            time.sleep(config.FETCH_SLEEP_SEC)
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, auto_adjust=True)

            if df.empty:
                logger.warning("%s: データが空です", ticker)
                return None

            df = _normalize(df)
            _save_cache(ticker, df)
            return df

        except Exception as exc:
            logger.warning(
                "%s: 取得失敗 (試行 %d/%d): %s",
                ticker, attempt, config.FETCH_RETRY, exc,
            )
            if attempt < config.FETCH_RETRY:
                time.sleep(2 ** attempt)

    logger.error("%s: %d 回試行しましたが取得できませんでした", ticker, config.FETCH_RETRY)
    return None


def _fetch_parallel(
    tickers: list[str],
    period: str,
    results: dict[str, pd.DataFrame],
) -> None:
    """ThreadPoolExecutor で並列取得する。"""
    with ThreadPoolExecutor(max_workers=config.FETCH_MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_single, t, period): t for t in tickers
        }
        for future in tqdm(
            as_completed(future_to_ticker),
            total=len(tickers),
            desc="株価取得",
            unit="銘柄",
        ):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None:
                    results[ticker] = df
            except Exception as exc:
                logger.error("%s: 予期しないエラー: %s", ticker, exc)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """列名を統一し、不要な列を除去する。"""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    df.columns = [c.lower() for c in df.columns]

    # 異常値除去（価格が 0 以下の行を削除）
    price_cols = ["open", "high", "low", "close"]
    df = df[(df[price_cols] > 0).all(axis=1)]

    return df.sort_index()


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace(".", "_")
    return config.RAW_DIR / f"{safe}.csv"


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(ticker))


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


def _filter_uncached(tickers: list[str]) -> list[str]:
    """キャッシュが存在しない銘柄のみを返す。"""
    return [t for t in tickers if not _cache_path(t).exists()]
