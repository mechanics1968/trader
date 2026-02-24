"""
東証全銘柄コードを JPX 公開 Excel から取得する。

取得結果を data/tickers/tse_tickers.csv に保存し、
次回以降はキャッシュを利用する（--refresh フラグで強制再取得）。
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

CACHE_PATH = config.TICKERS_DIR / "tse_tickers.csv"

# JPX が公開している上場銘柄一覧 Excel の列名マッピング
_COL_MAP = {
    "コード": "code",
    "銘柄名": "name",
    "市場・商品区分": "market",
    "33業種コード": "sector33_code",
    "33業種区分": "sector33",
    "17業種コード": "sector17_code",
    "17業種区分": "sector17",
    "規模コード": "scale_code",
    "規模区分": "scale",
}


def fetch_tickers(refresh: bool = False) -> pd.DataFrame:
    """
    東証全銘柄コード一覧を返す。

    Parameters
    ----------
    refresh : bool
        True の場合は JPX サイトから再取得してキャッシュを更新する。

    Returns
    -------
    pd.DataFrame
        列: code (str), name, market, sector33, sector17, scale
    """
    if not refresh and CACHE_PATH.exists():
        logger.info("キャッシュから銘柄一覧を読み込みます: %s", CACHE_PATH)
        return _load_cache()

    logger.info("JPX から銘柄一覧を取得します: %s", config.JPX_LIST_URL)
    df = _download_from_jpx()
    df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    logger.info("銘柄一覧を保存しました: %s (%d 件)", CACHE_PATH, len(df))
    return df


def _download_from_jpx() -> pd.DataFrame:
    """JPX の Excel を取得してパースする。"""
    for attempt in range(1, config.FETCH_RETRY + 1):
        try:
            resp = requests.get(config.JPX_LIST_URL, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            logger.warning("取得失敗 (試行 %d/%d): %s", attempt, config.FETCH_RETRY, exc)
            if attempt == config.FETCH_RETRY:
                raise
            time.sleep(2 ** attempt)

    df = pd.read_excel(io.BytesIO(resp.content), header=0, dtype=str)
    df = df.rename(columns=_COL_MAP)

    # 必要な列だけ保持
    keep = [v for v in _COL_MAP.values() if v in df.columns]
    df = df[keep].copy()

    # 対象市場に絞り込む
    df = _filter_target_markets(df)

    # code を4桁ゼロ埋め文字列に統一
    df["code"] = df["code"].str.strip().str.zfill(4)

    # Yahoo Finance 形式のティッカー（例: "7203.T"）
    df["ticker"] = df["code"] + ".T"

    return df.reset_index(drop=True)


def _filter_target_markets(df: pd.DataFrame) -> pd.DataFrame:
    """対象市場（プライム / スタンダード / グロース）のみに絞り込む。"""
    if "market" not in df.columns:
        logger.warning("'market' 列が見つかりません。全銘柄を対象にします。")
        return df

    mask = df["market"].str.contains(
        "|".join(config.TARGET_MARKETS), na=False
    )
    filtered = df[mask].copy()
    logger.info(
        "市場フィルタ後: %d 件 / 全 %d 件", len(filtered), len(df)
    )
    return filtered


def _load_cache() -> pd.DataFrame:
    return pd.read_csv(CACHE_PATH, dtype={"code": str, "ticker": str})


def get_ticker_list(refresh: bool = False) -> list[str]:
    """ティッカー文字列のリスト（例: ["7203.T", ...]）を返す。"""
    df = fetch_tickers(refresh=refresh)
    return df["ticker"].tolist()
