"""
予測値から推薦銘柄リストを生成する。

フィルタリング条件:
  - 出来高 >= MIN_VOLUME
  - 期待上昇率 >= MIN_EXPECTED_GAIN_PCT
  - 1単元の購入額 <= MAX_PRICE_PER_UNIT
  - 前日比騰落率の絶対値 <= MAX_DAILY_CHANGE_PCT（ギャップアップ/ダウン除外）
"""
from __future__ import annotations

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def build_recommendations(
    predictions: pd.DataFrame,
    ticker_info: pd.DataFrame,
    lot_sizes: dict[str, int],
    market_up_prob: float = 0.5,
    risk_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    予測 DataFrame から推薦銘柄リストを生成する。

    Parameters
    ----------
    predictions : pd.DataFrame
        predict_next_day() の出力
        列: ticker, last_close, last_volume, last_return_pct,
            pred_open, pred_close, expected_gain_pct
    ticker_info : pd.DataFrame
        fetch_tickers() の出力
        列: code, name, market, ticker, ...
    lot_sizes : dict[str, int]
        {ticker: lot_size}

    Returns
    -------
    pd.DataFrame
        フィルタ・ソート済みの推薦銘柄リスト
    """
    df = predictions.copy()

    # LLM指定のリスク銘柄を除外（4桁コード → "XXXX.T" 形式に変換して照合）
    if risk_tickers and config.LLM_RISK_TICKER_EXCLUDE:
        risk_set = {f"{t}.T" for t in risk_tickers}
        before = len(df)
        df = df[~df["ticker"].isin(risk_set)]
        excluded = before - len(df)
        if excluded:
            logger.info("LLMリスク銘柄を %d 件除外しました: %s", excluded, risk_tickers)

    # 単元株数をマージ
    df["lot_size"] = df["ticker"].map(lot_sizes).fillna(100).astype(int)

    # 1単元あたりの必要資金（予測始値 × 単元株数）
    df["required_amount"] = (df["pred_open"] * df["lot_size"]).round(0).astype(int)

    # 1単元あたりの期待利益額
    # バイナリモードでは pred_close が存在しないため expected_gain_pct × pred_open で近似
    if "pred_close" in df.columns:
        profit_base = (df["pred_close"] - df["pred_open"]) * df["lot_size"]
    else:
        profit_base = df["pred_open"] * (df["expected_gain_pct"] / 100) * df["lot_size"]
    df["expected_profit_per_lot"] = profit_base.round(0).astype(int)

    # 銘柄名・市場区分をマージ
    info_cols = ["ticker", "code", "name", "market"]
    available_cols = [c for c in info_cols if c in ticker_info.columns]
    df = df.merge(ticker_info[available_cols], on="ticker", how="left")

    # ------------------------------------------------------------------ #
    # フィルタリング
    # ------------------------------------------------------------------ #
    before = len(df)

    # 1) 出来高フィルタ
    df = df[df["last_volume"] >= config.MIN_VOLUME]

    # 2) 期待上昇率フィルタ（固定閾値 — 市場動向に関係なく常に同じ基準）
    # Phase B 分類モードでは expected_gain_pct = (確率 - 0.5) × 100
    if config.USE_BINARY_CLOSE and config.USE_INTRADAY_TARGET:
        df = df[df["expected_gain_pct"] > 0]
    else:
        df = df[df["expected_gain_pct"] >= config.MIN_EXPECTED_GAIN_PCT]

    # 3) 購入可能金額フィルタ（1単元が元手の20%以内）
    df = df[df["required_amount"] <= config.MAX_PRICE_PER_UNIT]

    # 4) 異常な前日変動を除外（ギャップアップ/ダウン）
    df = df[df["last_return_pct"].abs() <= config.MAX_DAILY_CHANGE_PCT]

    # 5) 市場モメンタムフィルタ（前日の等加重市場リターンが閾値未満なら全推薦を除外）
    if "last_mkt_return" in df.columns and len(df) > 0:
        mkt_prev_pct = df["last_mkt_return"].iloc[0]   # 全銘柄で同値
        threshold_pct = config.MARKET_DECLINE_THRESHOLD * 100
        if mkt_prev_pct < threshold_pct:
            logger.info(
                "市場モメンタムフィルタ: 前日市場 %.3f%% < 閾値 %.3f%% → 推薦なし",
                mkt_prev_pct, threshold_pct,
            )
            df = df.iloc[:0]

    after = len(df)
    logger.info("フィルタリング: %d → %d 銘柄", before, after)

    # ------------------------------------------------------------------ #
    # ソート・整形
    # ------------------------------------------------------------------ #
    df = df.sort_values("expected_gain_pct", ascending=False).head(
        config.TOP_N_RECOMMENDATIONS
    )

    df = df.reset_index(drop=True)
    df.index += 1  # 推薦順位を1始まりにする

    # ------------------------------------------------------------------ #
    # 市場評価レーティング（★1〜5）付与
    # ------------------------------------------------------------------ #
    df["market_rating"] = _compute_market_ratings(df["expected_gain_pct"], market_up_prob)
    df["market_rating_label"] = df["market_rating"].map(lambda r: "★" * r)
    logger.info(
        "市場評価レーティング付与 (market_up_prob=%.3f): %s",
        market_up_prob,
        df["market_rating_label"].value_counts().to_dict(),
    )

    # 出力列の順序
    output_cols = [
        "code", "name", "market",
        "market_rating_label",
        "pred_open", "pred_close", "close_up_prob",
        "expected_gain_pct", "expected_profit_per_lot",
        "lot_size", "required_amount",
        "last_close", "last_volume",
        "pred_open_return_pct", "pred_close_return_pct",
        "ticker",
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    return df[output_cols]


def _compute_market_ratings(gain_series: pd.Series, market_up_prob: float) -> pd.Series:
    """
    フィルタ・ソート後の推薦候補に市場評価レーティング（★1〜5）を付与する。

    算出ロジック:
      1. individual_rank = expected_gain_pct のパーセンタイル順位（0〜1）
         → 候補内で相対的に上位ほど 1.0 に近い
      2. composite = market_up_prob × w + individual_rank × (1 - w)
         → 市場強気 + 個別スコア高い ほど 1.0 に近い
      3. config.MARKET_RATING_THRESHOLDS で composite を ★1〜5 にマッピング

    Parameters
    ----------
    gain_series : pd.Series
        expected_gain_pct の列（フィルタ後）
    market_up_prob : float
        翌日市場上昇確率（0〜1）

    Returns
    -------
    pd.Series[int]
        各銘柄の市場評価レーティング（1〜5）
    """
    if gain_series.empty:
        return gain_series.astype(int)

    w = config.MARKET_RATING_MARKET_WEIGHT
    individual_rank = gain_series.rank(pct=True)
    composite = market_up_prob * w + individual_rank * (1 - w)
    thresholds = config.MARKET_RATING_THRESHOLDS  # [0.25, 0.40, 0.60, 0.75]

    def to_rating(c: float) -> int:
        for rating, threshold in enumerate(thresholds, start=1):
            if c < threshold:
                return rating
        return 5

    return composite.map(to_rating)
