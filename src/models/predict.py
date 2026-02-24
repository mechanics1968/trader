"""
学習済みモデルで翌日の始値・終値変化率を予測する。
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from src.features.engineer import get_feature_columns

logger = logging.getLogger(__name__)


def predict_next_day(
    features: dict[str, pd.DataFrame],
    model_open: lgb.Booster,
    model_close: lgb.Booster,
) -> pd.DataFrame:
    """
    全銘柄の翌日の始値・終値を予測する。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}
    model_open : lgb.Booster
        始値変化率予測モデル
    model_close : lgb.Booster
        終値変化率予測モデル

    Returns
    -------
    pd.DataFrame
        列: ticker, last_close, pred_open, pred_close, expected_gain_pct
    """
    rows = []

    for ticker, df in features.items():
        if df.empty:
            continue

        # 最新行（= 本日のデータ）を使って翌日を予測
        latest = df.iloc[[-1]]
        feat_cols = get_feature_columns(df)
        X = latest[feat_cols].replace([np.inf, -np.inf], np.nan)

        if X.isna().any(axis=1).values[0]:
            logger.debug("%s: NaN が含まれるためスキップ", ticker)
            continue

        last_close = float(latest["close"].values[0])
        last_volume = float(latest["volume"].values[0])
        last_return = float(latest.get("return_1d", pd.Series([0])).values[0])
        # 前日の等加重市場リターン（市場フィルタ用）
        last_mkt_return = float(
            latest["mkt_return_1d"].values[0]
            if "mkt_return_1d" in latest.columns
            else 0.0
        )

        # モデル出力を取得
        # USE_CS_TARGET=True  → z-score（クロスセクション内の相対順位）
        # USE_CS_TARGET=False → 超過リターン（α）または絶対リターン
        score_open  = float(model_open.predict(X)[0])
        score_close = float(model_close.predict(X)[0])

        if config.USE_CS_TARGET:
            # CS モード: モデル出力は z-score。価格予測は last_close をそのまま使う
            # （z-score を return として解釈すると pred_open/pred_close が乖離するため）
            pred_open  = last_close
            pred_close = last_close
            # 推薦スコア: 終値 z-score × 100（大きいほど全銘柄の中で高い終値を予測）
            expected_gain_pct = score_close * 100
        else:
            # Alpha / 絶対リターンモード: return として解釈して絶対価格を計算
            pred_open  = last_close * (1 + score_open)
            pred_close = last_close * (1 + score_close)
            expected_gain_pct = (score_close - score_open) * 100

        rows.append({
            "ticker": ticker,
            "last_close": round(last_close, 1),
            "last_volume": int(last_volume),
            "last_return_pct": round(last_return * 100, 2),
            "last_mkt_return": round(last_mkt_return * 100, 4),
            "pred_open": round(pred_open, 1),
            "pred_close": round(pred_close, 1),
            "pred_open_return_pct": round(score_open * 100, 2),
            "pred_close_return_pct": round(score_close * 100, 2),
            "expected_gain_pct": round(expected_gain_pct, 2),
        })

    result = pd.DataFrame(rows)
    logger.info("予測完了: %d 銘柄", len(result))
    return result
