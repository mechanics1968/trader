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

    # 日付ごとに集計するためにDataFrameで蓄積する
    records: list[dict] = []

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
        # 終値モデルのターゲット選択（αモード > 日中騰落率 > 通常）
        if config.USE_INTRADAY_ALPHA and "target_intraday_alpha" in val_df.columns:
            close_target_col = "target_intraday_alpha"
        elif config.USE_INTRADAY_TARGET and "target_intraday_return" in val_df.columns:
            close_target_col = "target_intraday_return"
        else:
            close_target_col = "target_close_return"
        y_actual_close = val_df.loc[valid_idx, close_target_col].values

        # 収益シミュレーション用の実際の日中騰落率
        if config.USE_INTRADAY_ALPHA and "target_intraday_return" in val_df.columns:
            y_actual_intraday = val_df.loc[valid_idx, "target_intraday_return"].values
        else:
            y_actual_intraday = y_actual_close

        y_pred_open = model_open.predict(X_val)
        y_pred_close = model_close.predict(X_val)

        dates = valid_idx.normalize() if hasattr(valid_idx, "normalize") else valid_idx
        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "pred_open": float(y_pred_open[i]),
                "pred_close": float(y_pred_close[i]),
                "actual_open": float(y_actual_open[i]),
                "actual_close": float(y_actual_close[i]),
                "actual_intraday": float(y_actual_intraday[i]),
            })

    if not records:
        return pd.DataFrame()

    rec_df = pd.DataFrame(records)

    all_pred_open = rec_df["pred_open"].tolist()
    all_actual_close = rec_df["actual_close"].tolist()
    all_pred_close = rec_df["pred_close"].tolist()

    is_binary = (
        config.USE_BINARY_CLOSE
        and config.USE_INTRADAY_ALPHA
    )

    if config.USE_INTRADAY_ALPHA:
        close_label = "日中α2値" if is_binary else "日中α"
    elif config.USE_INTRADAY_TARGET:
        close_label = "日中騰落率"
    else:
        close_label = "終値"

    if is_binary:
        # 分類モード: pred_close は確率（0〜1）。0.5 超を正例（市場超過）とみなす
        pred_close_dir = [p - 0.5 for p in all_pred_close]   # dir_acc 計算用にゼロ基準に変換
        actual_close_dir = all_actual_close                   # actual は連続値のまま（符号で判定）
        close_dir_acc = _directional_accuracy(actual_close_dir, pred_close_dir)
        # AUC も計算
        from sklearn.metrics import roc_auc_score
        actual_bin = [1 if a > 0 else 0 for a in all_actual_close]
        try:
            auc = float(roc_auc_score(actual_bin, all_pred_close))
        except Exception:
            auc = float("nan")
        metrics = {
            "始値_方向的中率": _directional_accuracy(rec_df["actual_open"].tolist(), all_pred_open),
            f"{close_label}_方向的中率": close_dir_acc,
            "始値_RMSE": _rmse(rec_df["actual_open"].tolist(), all_pred_open),
            f"{close_label}_AUC": auc,
        }
    else:
        metrics = {
            "始値_方向的中率": _directional_accuracy(rec_df["actual_open"].tolist(), all_pred_open),
            f"{close_label}_方向的中率": _directional_accuracy(all_actual_close, all_pred_close),
            "始値_RMSE": _rmse(rec_df["actual_open"].tolist(), all_pred_open),
            f"{close_label}_RMSE": _rmse(all_actual_close, all_pred_close),
        }

    # 日次ポートフォリオリターンを計算してからSharpeを算出
    daily_returns = _compute_daily_returns(rec_df)
    metrics["期待収益率(平均日次)"] = float(np.mean(daily_returns))
    metrics["最大ドローダウン"] = float(_max_drawdown(daily_returns))
    metrics["Sharpe_Ratio"] = float(_sharpe_ratio(daily_returns))

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


def _compute_daily_returns(rec_df: pd.DataFrame) -> np.ndarray:
    """
    銘柄×日付のレコードから、日次ポートフォリオリターンの時系列を返す。

    各日付で「上昇予測」銘柄のみ投資し、その日の平均実績リターンを1点として集計。
    これによりSharpe計算の分母が「日次リターンのstd」となり正しい値になる。

    投資判定:
      αモード / 日中モード: pred_close > 0
      通常モード: pred_close > pred_open
    実績リターン:
      αモード: actual_intraday（実際の日中騰落率）
      日中/通常モード: actual_intraday（= actual_close）
    """
    if config.USE_BINARY_CLOSE and config.USE_INTRADAY_ALPHA:
        # 分類モード: pred_close は NaN、close_up_prob に確率（0〜1）が入る
        invest_mask = rec_df["pred_close"] > 0.5
    elif config.USE_INTRADAY_ALPHA or config.USE_INTRADAY_TARGET:
        invest_mask = rec_df["pred_close"] > 0
    else:
        invest_mask = rec_df["pred_close"] > rec_df["pred_open"]

    rec_df = rec_df.copy()
    rec_df["gain"] = np.where(invest_mask, rec_df["actual_intraday"], np.nan)

    # 日付ごとに推薦銘柄の平均リターンを集計
    daily = (
        rec_df[invest_mask]
        .groupby("date")["gain"]
        .mean()
        .sort_index()
    )

    if daily.empty:
        return np.array([0.0])
    return daily.to_numpy()


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
