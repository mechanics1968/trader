"""
予測精度の評価・バックテスト。

評価指標:
  - 方向的中率（上昇/下落の方向が合っていたか）
  - RMSE（価格変化率の誤差）
  - 期待収益率（バックテスト期間の平均日次リターン）
  - 最大ドローダウン
  - Sharpe Ratio
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def evaluate_predictions(
    features: dict[str, pd.DataFrame],
    model_open,
    model_close,
) -> pd.DataFrame:
    """
    バリデーションセット（末尾 VALIDATION_DAYS）で精度評価する。

    Returns
    -------
    pd.DataFrame
        指標をまとめた DataFrame
    """
    from src.features.engineer import get_feature_columns

    from src.models.tft_model import TFTModelWrapper
    if isinstance(model_open, TFTModelWrapper):
        return _evaluate_tft(features, model_open, model_close)

    all_actual_open: list[float] = []
    all_pred_open: list[float] = []
    all_actual_close: list[float] = []
    all_pred_close: list[float] = []

    n_val = config.VALIDATION_DAYS

    for ticker, df in features.items():
        if len(df) <= n_val:
            continue

        val_df = df.iloc[-n_val:]
        feat_cols = get_feature_columns(df)
        X_val = val_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if X_val.empty:
            continue

        valid_idx = X_val.index
        y_actual_open = val_df.loc[valid_idx, "target_open_return"].values
        y_actual_close = val_df.loc[valid_idx, "target_close_return"].values

        y_pred_open = model_open.predict(X_val)
        y_pred_close = model_close.predict(X_val)

        all_actual_open.extend(y_actual_open)
        all_pred_open.extend(y_pred_open)
        all_actual_close.extend(y_actual_close)
        all_pred_close.extend(y_pred_close)

    metrics = {
        "始値_方向的中率": _directional_accuracy(all_actual_open, all_pred_open),
        "終値_方向的中率": _directional_accuracy(all_actual_close, all_pred_close),
        "始値_RMSE": _rmse(all_actual_open, all_pred_open),
        "終値_RMSE": _rmse(all_actual_close, all_pred_close),
    }

    # デイトレ収益シミュレーション（終値 - 始値）
    strategy_returns = _simulate_strategy(all_pred_open, all_pred_close, all_actual_open, all_actual_close)
    metrics["期待収益率(平均日次)"] = float(np.mean(strategy_returns))
    metrics["最大ドローダウン"] = float(_max_drawdown(strategy_returns))
    metrics["Sharpe_Ratio"] = float(_sharpe_ratio(strategy_returns))

    result = pd.DataFrame([metrics])
    logger.info("バックテスト結果:\n%s", result.to_string(index=False))
    return result


def _evaluate_tft(
    features: dict,
    model_open,
    model_close,
) -> pd.DataFrame:
    """TFT モデル向けのバックテスト評価。"""
    p_open, a_open = model_open.compute_val_predictions(features, config.VALIDATION_DAYS)
    p_close, a_close = model_close.compute_val_predictions(features, config.VALIDATION_DAYS)

    all_actual_open = list(a_open)
    all_pred_open = list(p_open)
    all_actual_close = list(a_close)
    all_pred_close = list(p_close)

    metrics = {
        "始値_方向的中率": _directional_accuracy(all_actual_open, all_pred_open),
        "終値_方向的中率": _directional_accuracy(all_actual_close, all_pred_close),
        "始値_RMSE": _rmse(all_actual_open, all_pred_open),
        "終値_RMSE": _rmse(all_actual_close, all_pred_close),
    }

    strategy_returns = _simulate_strategy(
        all_pred_open, all_pred_close, all_actual_open, all_actual_close
    )
    metrics["期待収益率(平均日次)"] = float(np.mean(strategy_returns))
    metrics["最大ドローダウン"] = float(_max_drawdown(strategy_returns))
    metrics["Sharpe_Ratio"] = float(_sharpe_ratio(strategy_returns))

    result = pd.DataFrame([metrics])
    logger.info("[TFT] バックテスト結果:\n%s", result.to_string(index=False))
    return result


def _directional_accuracy(actual: list[float], pred: list[float]) -> float:
    """方向的中率を計算する。"""
    if not actual:
        return float("nan")
    a = np.array(actual)
    p = np.array(pred)
    correct = np.sign(a) == np.sign(p)
    return float(correct.mean())


def _rmse(actual: list[float], pred: list[float]) -> float:
    if not actual:
        return float("nan")
    a = np.array(actual)
    p = np.array(pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _simulate_strategy(
    pred_open: list[float],
    pred_close: list[float],
    actual_open: list[float],
    actual_close: list[float],
) -> np.ndarray:
    """
    予測で「上昇」と判定した銘柄のみ投資した場合のリターン列を返す。
    （始値で買い → 終値で売り）
    """
    p_open = np.array(pred_open)
    p_close = np.array(pred_close)
    a_open = np.array(actual_open)
    a_close = np.array(actual_close)

    # 予測で終値 > 始値 のとき投資
    invest_mask = p_close > p_open
    actual_gain = np.where(invest_mask, a_close - a_open, 0.0)
    return actual_gain


def _max_drawdown(returns: np.ndarray) -> float:
    """最大ドローダウン（累積リターンのピークからの最大下落率）を返す。"""
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    return float(drawdown.min())


def _sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """年率 Sharpe Ratio（日次リターン × √252）。"""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free
    return float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))
