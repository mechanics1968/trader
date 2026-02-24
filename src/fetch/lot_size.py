"""
各銘柄の単元株数（売買単位）を取得する。

2018年10月の東証売買単位統一により、上場銘柄の売買単位はすべて100株に統一済み。
JPX の上場銘柄一覧 Excel に「売買単位」列があれば優先して使用し、
なければ一律 100 株をデフォルトとする。

yfinance の info["lotSize"] による個別 API 取得は行わない
（3,000銘柄超ではレートリミットで実質不可能なため）。
"""
from __future__ import annotations

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)

DEFAULT_LOT_SIZE = 100
CACHE_PATH = config.TICKERS_DIR / "lot_sizes.csv"

# JPX の Excel で使われる売買単位の列名候補
_LOT_COL_CANDIDATES = ["売買単位", "Unit", "unit", "lot_size", "LotSize"]


def fetch_lot_sizes(
    tickers: list[str],
    refresh: bool = False,
) -> dict[str, int]:
    """
    複数銘柄の単元株数を返す。

    1. キャッシュが存在する場合はキャッシュを使用する
    2. JPX 銘柄一覧の Excel に売買単位列があればそこから取得する
    3. 上記いずれもない場合は一律 100 株を返す

    Returns
    -------
    dict[str, int]
        {ticker: lot_size} の辞書
    """
    if not refresh and CACHE_PATH.exists():
        cached = _load_cache()
        missing = [t for t in tickers if t not in cached]
        if not missing:
            return {t: cached.get(t, DEFAULT_LOT_SIZE) for t in tickers}
        # 不足分をデフォルト値で補完
        for t in missing:
            cached[t] = DEFAULT_LOT_SIZE
        return {t: cached.get(t, DEFAULT_LOT_SIZE) for t in tickers}

    # JPX データから取得を試みる
    data = _fetch_from_jpx_cache(tickers)
    _save_cache(data)
    logger.info("単元株数を設定しました（デフォルト %d 株）: %d 銘柄", DEFAULT_LOT_SIZE, len(data))
    return data


def _fetch_from_jpx_cache(tickers: list[str]) -> dict[str, int]:
    """
    JPX 銘柄一覧キャッシュから売買単位を取得する。
    列が存在しない場合はすべて DEFAULT_LOT_SIZE で返す。
    """
    ticker_cache = config.TICKERS_DIR / "tse_tickers.csv"
    if not ticker_cache.exists():
        logger.warning("銘柄一覧キャッシュが見つかりません。全銘柄を %d 株に設定します。", DEFAULT_LOT_SIZE)
        return {t: DEFAULT_LOT_SIZE for t in tickers}

    df = pd.read_csv(ticker_cache, dtype=str)

    # 売買単位列を探す
    lot_col = next((c for c in _LOT_COL_CANDIDATES if c in df.columns), None)

    if lot_col is None:
        logger.info(
            "JPX データに売買単位列が見つかりません。全銘柄を %d 株に設定します。",
            DEFAULT_LOT_SIZE,
        )
        return {t: DEFAULT_LOT_SIZE for t in tickers}

    # ticker → lot_size のマッピングを作成
    ticker_to_lot: dict[str, int] = {}
    if "ticker" in df.columns:
        for _, row in df.iterrows():
            try:
                ticker_to_lot[row["ticker"]] = int(row[lot_col])
            except (ValueError, TypeError):
                pass

    return {t: ticker_to_lot.get(t, DEFAULT_LOT_SIZE) for t in tickers}


def _save_cache(data: dict[str, int]) -> None:
    pd.Series(data, name="lot_size").rename_axis("ticker").to_csv(CACHE_PATH)


def _load_cache() -> dict[str, int]:
    df = pd.read_csv(CACHE_PATH, index_col="ticker")
    return df["lot_size"].astype(int).to_dict()
