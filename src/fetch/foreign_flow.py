"""
JPX 投資部門別売買状況から外国人投資家の売買動向を取得する。

データソース:
  https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html
  週次 Excel ファイル（現物株）

提供データ:
  - 週次の外国人・個人・法人・証券会社別の買い越し/売り越し額
  - 市場全体のシグナルとして使用（個別銘柄への帰属は行わない）

使い方:
  foreign_flow = fetch_foreign_flow(refresh=False)
  # → pd.DataFrame(index=date, columns=[foreign_net_buy, foreign_net_buy_4wk, ...])
  # 全銘柄の特徴量 DataFrame に join する（市場レジームシグナル）
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

_CACHE_PATH = config.DATA_DIR / "foreign_flow.csv"

# JPX 投資部門別売買状況（週次現物）の Excel URL
# 毎週更新されるため、インデックスページから最新ファイルを探す
_INDEX_URL = "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; trader-bot/1.0)"}


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def fetch_foreign_flow(refresh: bool = False) -> pd.DataFrame:
    """
    外国人投資家の週次売買動向を返す。

    Returns
    -------
    pd.DataFrame
        index: date（週の最終営業日）
        列:
          foreign_net_buy     : 外国人の週次純買い額（円換算・正=買い越し）
          foreign_net_buy_4wk : 直近4週累計（買いトレンド判定用）
          foreign_buy_regime  : 外国人が直近4週買い越しなら1、それ以外は0
        失敗時は空 DataFrame を返す
    """
    if not refresh and _CACHE_PATH.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(_CACHE_PATH.stat().st_mtime, unit="s")).days
        if age_days < 7:  # 週次データなので7日以内はキャッシュ再利用
            df = pd.read_csv(_CACHE_PATH, index_col="date", parse_dates=True)
            logger.info("外資売買動向をキャッシュから読み込みました: %d 週分", len(df))
            return df

    df = _download_jpx_investor_type()
    if df is None or df.empty:
        logger.warning(
            "外資売買動向の取得に失敗しました。この特徴量はスキップされます。"
        )
        return pd.DataFrame()

    df.to_csv(_CACHE_PATH)
    logger.info("外資売買動向を取得しました: %d 週分", len(df))
    return df


def align_foreign_flow_to_daily(
    foreign_flow: pd.DataFrame,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    週次の外資売買動向を日次インデックスに forward-fill で展開する。

    Parameters
    ----------
    foreign_flow : pd.DataFrame
        fetch_foreign_flow() の出力
    date_index : pd.DatetimeIndex
        銘柄の日次インデックス（特徴量 DataFrame のインデックス）

    Returns
    -------
    pd.DataFrame
        date_index に合わせた日次 DataFrame
    """
    if foreign_flow.empty:
        return pd.DataFrame(index=date_index)

    # 週次 → 日次に forward-fill
    combined_idx = date_index.union(foreign_flow.index).sort_values()
    aligned = foreign_flow.reindex(combined_idx).ffill()
    return aligned.reindex(date_index)


# ---------------------------------------------------------------------------
# 内部実装
# ---------------------------------------------------------------------------

def _download_jpx_investor_type() -> pd.DataFrame | None:
    """JPX の投資部門別売買状況 Excel を取得してパースする。"""
    try:
        # インデックスページから最新ファイルのリンクを取得
        resp = requests.get(_INDEX_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()

        # Excel ファイルのリンクを探す
        from html.parser import HTMLParser

        class LinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links: list[str] = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for k, v in attrs:
                        if k == "href" and v and v.endswith(".xls") and "investor" in v.lower():
                            self.links.append(v)

        parser = LinkParser()
        parser.feed(resp.text)

        if not parser.links:
            # より広い検索
            import re
            matches = re.findall(r'href="([^"]*\.xls[x]?)"', resp.text)
            parser.links = [m for m in matches if "investor" in m.lower() or "gyoshu" in m.lower() or "kaishime" in m.lower()]

        if not parser.links:
            logger.warning("JPX 投資部門別 Excel ファイルのリンクが見つかりませんでした")
            return None

        # 最初のリンクをダウンロード
        xl_url = parser.links[0]
        if not xl_url.startswith("http"):
            xl_url = "https://www.jpx.co.jp" + xl_url

        xl_resp = requests.get(xl_url, headers=_HEADERS, timeout=60)
        xl_resp.raise_for_status()

        df = _parse_investor_type_excel(xl_resp.content)
        return df

    except Exception as exc:
        logger.warning("JPX 外資売買動向取得エラー: %s", exc)
        return None


def _parse_investor_type_excel(content: bytes) -> pd.DataFrame | None:
    """
    投資部門別売買状況 Excel をパースして外国人純買いデータを抽出する。

    JPX の Excel は複数シートと複雑なヘッダを持つため、
    「外国人」行と「買い越し額」列を柔軟に検索する。
    """
    try:
        xl = pd.ExcelFile(io.BytesIO(content))
        for sheet in xl.sheet_names:
            try:
                raw = pd.read_excel(xl, sheet_name=sheet, header=None)
                result = _extract_foreign_net(raw)
                if result is not None:
                    return result
            except Exception:
                continue
        return None
    except Exception as exc:
        logger.debug("Excel パースエラー: %s", exc)
        return None


def _extract_foreign_net(raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    生の Excel DataFrame から外国人の純買い額を抽出する。

    JPX の形式は年度ごとに変わることがあるため、ヒューリスティックに検索する。
    """
    import numpy as np

    # 日付行を探す（数値の日付またはdatetime型が並ぶ行）
    date_row_idx = None
    for i, row in raw.iterrows():
        vals = row.dropna().values
        try:
            dates = pd.to_datetime(vals, errors="coerce")
            if dates.notna().sum() >= 5:
                date_row_idx = i
                break
        except Exception:
            pass

    if date_row_idx is None:
        return None

    # 外国人行を探す（「外国人」「外国」「海外」などを含む行）
    foreign_keywords = ["外国人", "外国", "海外", "Foreign"]
    foreign_rows = []
    for i, row in raw.iterrows():
        if i <= date_row_idx:
            continue
        for val in row.values:
            if isinstance(val, str) and any(kw in val for kw in foreign_keywords):
                foreign_rows.append(i)
                break

    if not foreign_rows:
        return None

    # 日付を取得
    date_series = pd.to_datetime(raw.iloc[date_row_idx], errors="coerce")
    date_cols = date_series.dropna().index.tolist()

    rows = []
    for row_idx in foreign_rows[:1]:  # 最初の外国人行のみ使用
        vals = raw.iloc[row_idx, date_cols].values
        for col, val in zip(date_cols, vals):
            dt = date_series[col]
            if pd.notna(dt) and isinstance(val, (int, float)) and not np.isnan(val):
                rows.append({"date": dt, "foreign_net_buy": float(val)})

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["foreign_net_buy_4wk"] = df["foreign_net_buy"].rolling(4, min_periods=1).sum()
    df["foreign_buy_regime"] = (df["foreign_net_buy_4wk"] > 0).astype(float)
    return df
