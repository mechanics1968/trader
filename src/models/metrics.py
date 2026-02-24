"""
予測誤差メトリクスの計算。

ウォークフォワード評価:
  - 前日までの特徴量でモデルが当日の始値・終値変化率を予測
  - 実際の当日値動きと比較し、誤差を算出
  - この誤差が多目的ハイパーパラメータ最適化の目的関数になる

主な誤差指標:
  rmse_open    : 始値変化率の RMSE（最小化）
  rmse_close   : 終値変化率の RMSE（最小化）
  dir_acc_open : 始値方向的中率（最大化 → 1-值を最小化）
  dir_acc_close: 終値方向的中率（最大化 → 1-値を最小化）
  rmse_gain    : 期待利益幅（終値-始値 変化率）の RMSE（最小化）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import config
from src.features.engineer import get_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class PredictionErrors:
    """1評価期間分の予測誤差をまとめたデータクラス。"""

    # --- スカラー指標 ---
    rmse_open: float = float("nan")       # 始値変化率 RMSE
    rmse_close: float = float("nan")      # 終値変化率 RMSE
    rmse_gain: float = float("nan")       # 期待利益幅（close-open）RMSE
    dir_acc_open: float = float("nan")    # 始値方向的中率  (0〜1, 高いほど良い)
    dir_acc_close: float = float("nan")   # 終値方向的中率  (0〜1, 高いほど良い)
    mae_open: float = float("nan")        # 始値変化率 MAE
    mae_close: float = float("nan")       # 終値変化率 MAE
    n_samples: int = 0                    # 評価に使ったサンプル数

    # --- 多目的最適化の目的関数値（最小化方向に統一）---
    # Optuna はデフォルトで minimize なので方向的中率は (1 - 値) で返す
    @property
    def objectives(self) -> dict[str, float]:
        return {
            "rmse_open": self.rmse_open,
            "rmse_close": self.rmse_close,
            "1_minus_dir_acc_open": 1.0 - self.dir_acc_open,
            "1_minus_dir_acc_close": 1.0 - self.dir_acc_close,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def compute_errors(
    features: dict[str, pd.DataFrame],
    model_open,
    model_close,
    n_val_days: int = config.VALIDATION_DAYS,
) -> PredictionErrors:
    """
    全銘柄のウォークフォワード予測誤差を計算する。

    評価期間: 各銘柄の末尾 n_val_days 日間
    評価ロジック:
        - その日の特徴量（前日までのデータから計算済み）を使って
          当日の始値・終値変化率を予測
        - target_open_return / target_close_return（実績）と比較

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}（target_open_return / target_close_return 含む）
    model_open : lgb.Booster
        始値変化率予測モデル
    model_close : lgb.Booster
        終値変化率予測モデル
    n_val_days : int
        評価に使うバリデーション期間の日数

    Returns
    -------
    PredictionErrors
        誤差指標をまとめたデータクラス
    """
    from src.models.tft_model import TFTModelWrapper
    if isinstance(model_open, TFTModelWrapper):
        return _compute_errors_tft(features, model_open, model_close, n_val_days)

    actual_open_list: list[float] = []
    pred_open_list: list[float] = []
    actual_close_list: list[float] = []
    pred_close_list: list[float] = []

    for ticker, df in features.items():
        if len(df) <= n_val_days:
            continue

        val_df = df.iloc[-n_val_days:]
        feat_cols = get_feature_columns(df)

        X_val = (
            val_df[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if X_val.empty:
            continue

        valid_idx = X_val.index
        y_actual_open = val_df.loc[valid_idx, "target_open_return"].to_numpy()
        y_actual_close = val_df.loc[valid_idx, "target_close_return"].to_numpy()

        y_pred_open = model_open.predict(X_val)
        y_pred_close = model_close.predict(X_val)

        actual_open_list.extend(y_actual_open)
        pred_open_list.extend(y_pred_open)
        actual_close_list.extend(y_actual_close)
        pred_close_list.extend(y_pred_close)

    if not actual_open_list:
        logger.warning("評価サンプルが 0 件です")
        return PredictionErrors()

    a_open = np.array(actual_open_list)
    p_open = np.array(pred_open_list)
    a_close = np.array(actual_close_list)
    p_close = np.array(pred_close_list)

    # 実績・予測それぞれの「当日内上昇幅（終値-始値）」
    a_gain = a_close - a_open
    p_gain = p_close - p_open

    errors = PredictionErrors(
        rmse_open=_rmse(a_open, p_open),
        rmse_close=_rmse(a_close, p_close),
        rmse_gain=_rmse(a_gain, p_gain),
        dir_acc_open=_dir_acc(a_open, p_open),
        dir_acc_close=_dir_acc(a_close, p_close),
        mae_open=_mae(a_open, p_open),
        mae_close=_mae(a_close, p_close),
        n_samples=len(a_open),
    )

    _log_summary(errors)
    return errors


def compute_errors_per_day(
    features: dict[str, pd.DataFrame],
    model_open,
    model_close,
    n_val_days: int = config.VALIDATION_DAYS,
) -> pd.DataFrame:
    """
    日付ごとの予測誤差を返す（可視化・詳細分析用）。

    Returns
    -------
    pd.DataFrame
        列: date, ticker, actual_open, pred_open, err_open,
                         actual_close, pred_close, err_close,
                         actual_gain, pred_gain, err_gain,
                         dir_correct_open, dir_correct_close
    """
    from src.models.tft_model import TFTModelWrapper
    if isinstance(model_open, TFTModelWrapper):
        return _compute_errors_per_day_tft(features, model_open, model_close, n_val_days)

    rows: list[dict] = []

    for ticker, df in features.items():
        if len(df) <= n_val_days:
            continue

        val_df = df.iloc[-n_val_days:]
        feat_cols = get_feature_columns(df)

        X_val = (
            val_df[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if X_val.empty:
            continue

        valid_idx = X_val.index
        y_pred_open = model_open.predict(X_val)
        y_pred_close = model_close.predict(X_val)

        for i, date in enumerate(valid_idx):
            a_open = float(val_df.loc[date, "target_open_return"])
            a_close = float(val_df.loc[date, "target_close_return"])
            p_open = float(y_pred_open[i])
            p_close = float(y_pred_close[i])

            rows.append({
                "date": date,
                "ticker": ticker,
                "actual_open": a_open,
                "pred_open": p_open,
                "err_open": p_open - a_open,          # 正: 過大予測, 負: 過小予測
                "actual_close": a_close,
                "pred_close": p_close,
                "err_close": p_close - a_close,
                "actual_gain": a_close - a_open,
                "pred_gain": p_close - p_open,
                "err_gain": (p_close - p_open) - (a_close - a_open),
                "dir_correct_open": int(np.sign(a_open) == np.sign(p_open)),
                "dir_correct_close": int(np.sign(a_close) == np.sign(p_close)),
            })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "日次誤差計算完了: %d 行 / %d 銘柄 / %d 日間",
        len(df_out),
        df_out["ticker"].nunique(),
        df_out["date"].nunique(),
    )
    return df_out


# ---------------------------------------------------------------------------
# 内部関数
# ---------------------------------------------------------------------------

def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _dir_acc(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.sign(actual) == np.sign(pred)))


def _compute_errors_tft(
    features: dict[str, pd.DataFrame],
    model_open,
    model_close,
    n_val_days: int,
) -> PredictionErrors:
    """TFT モデル向けの誤差計算。"""
    p_open, a_open = model_open.compute_val_predictions(features, n_val_days)
    p_close, a_close = model_close.compute_val_predictions(features, n_val_days)

    a_gain = a_close - a_open
    p_gain = p_close - p_open

    errors = PredictionErrors(
        rmse_open=_rmse(a_open, p_open),
        rmse_close=_rmse(a_close, p_close),
        rmse_gain=_rmse(a_gain, p_gain),
        dir_acc_open=_dir_acc(a_open, p_open),
        dir_acc_close=_dir_acc(a_close, p_close),
        mae_open=_mae(a_open, p_open),
        mae_close=_mae(a_close, p_close),
        n_samples=len(a_open),
    )
    _log_summary(errors)
    return errors


def _compute_errors_per_day_tft(
    features: dict[str, pd.DataFrame],
    model_open,
    model_close,
    n_val_days: int,
) -> pd.DataFrame:
    """TFT モデル向けの日次誤差計算。"""
    p_open, a_open = model_open.compute_val_predictions(features, n_val_days)
    p_close, a_close = model_close.compute_val_predictions(features, n_val_days)

    rows = []
    for i in range(len(a_open)):
        rows.append({
            "actual_open": float(a_open[i]),
            "pred_open": float(p_open[i]),
            "err_open": float(p_open[i] - a_open[i]),
            "actual_close": float(a_close[i]),
            "pred_close": float(p_close[i]),
            "err_close": float(p_close[i] - a_close[i]),
            "actual_gain": float(a_close[i] - a_open[i]),
            "pred_gain": float(p_close[i] - p_open[i]),
            "err_gain": float((p_close[i] - p_open[i]) - (a_close[i] - a_open[i])),
            "dir_correct_open": int(np.sign(a_open[i]) == np.sign(p_open[i])),
            "dir_correct_close": int(np.sign(a_close[i]) == np.sign(p_close[i])),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


def _log_summary(e: PredictionErrors) -> None:
    logger.info(
        "予測誤差サマリ (n=%d)\n"
        "  始値 RMSE=%.6f  MAE=%.6f  方向的中率=%.3f\n"
        "  終値 RMSE=%.6f  MAE=%.6f  方向的中率=%.3f\n"
        "  利益幅 RMSE=%.6f",
        e.n_samples,
        e.rmse_open, e.mae_open, e.dir_acc_open,
        e.rmse_close, e.mae_close, e.dir_acc_close,
        e.rmse_gain,
    )
