"""
LightGBM モデルの学習。

全銘柄の特徴量データを結合して共通モデルを学習する。
始値予測・終値予測の 2 モデルを別々に学習する。
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
from src.features.engineer import get_feature_columns

logger = logging.getLogger(__name__)

MODEL_OPEN_PATH = config.MODELS_DIR / "lgbm_open.pkl"
MODEL_CLOSE_PATH = config.MODELS_DIR / "lgbm_close.pkl"


def train(
    features: dict[str, pd.DataFrame],
    force: bool = False,
    params: dict | None = None,
    num_rounds: int = config.LGBM_NUM_ROUNDS,
    early_stopping: int = config.LGBM_EARLY_STOPPING,
    save: bool = True,
) -> tuple[lgb.Booster, lgb.Booster]:
    """
    全銘柄の特徴量を結合して LightGBM モデルを学習する。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}
    force : bool
        True の場合、保存済みモデルがあっても再学習する
    params : dict | None
        LightGBM パラメータ。None の場合は config.LGBM_PARAMS を使用。
    num_rounds : int
        最大ブーストラウンド数。
    early_stopping : int
        Early Stopping のラウンド数。
    save : bool
        True の場合、学習後にモデルをファイルに保存する（最適化中は False を推奨）。

    Returns
    -------
    tuple[lgb.Booster, lgb.Booster]
        (始値予測モデル, 終値予測モデル)
    """
    if not force and save and MODEL_OPEN_PATH.exists() and MODEL_CLOSE_PATH.exists():
        logger.info("保存済みモデルを読み込みます")
        return _load_models()

    logger.info("モデル学習を開始します")
    combined = _combine_features(features)

    feat_cols = get_feature_columns(combined)
    X = combined[feat_cols]

    # ターゲット変数の選択: CS z-score > 残差α > 絶対リターン の優先順位
    if "target_cs_open" in combined.columns and "target_cs_close" in combined.columns:
        y_open  = combined["target_cs_open"]
        y_close = combined["target_cs_close"]
        label_open  = "始値(CS-z)"
        label_close = "終値(CS-z)"
        target_cols = ["target_cs_open", "target_cs_close"]
        logger.info("クロスセクション標準化ターゲット（z-スコア）を使用します")
    elif "target_alpha_open" in combined.columns and "target_alpha_close" in combined.columns:
        y_open  = combined["target_alpha_open"]
        y_close = combined["target_alpha_close"]
        label_open  = "始値(超過α)"
        label_close = "終値(超過α)"
        target_cols = ["target_alpha_open", "target_alpha_close"]
        logger.info("残差ターゲット（超過リターン）を使用します")
    else:
        y_open  = combined["target_open_return"]
        y_close = combined["target_close_return"]
        label_open  = "始値"
        label_close = "終値"
        target_cols = ["target_open_return", "target_close_return"]
        logger.warning("残差ターゲット列が見つかりません。通常ターゲットにフォールバックします")

    date_col = combined["date"] if "date" in combined.columns else None
    model_open = _train_single(X, y_open, label=label_open, params=params,
                               num_rounds=num_rounds, early_stopping=early_stopping,
                               date_col=date_col)
    model_close = _train_single(X, y_close, label=label_close, params=params,
                                num_rounds=num_rounds, early_stopping=early_stopping,
                                date_col=date_col)

    if save:
        _save_models(model_open, model_close)
    return model_open, model_close


def _train_single(
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "",
    params: dict | None = None,
    num_rounds: int = config.LGBM_NUM_ROUNDS,
    early_stopping: int = config.LGBM_EARLY_STOPPING,
    date_col: pd.Series | None = None,
) -> lgb.Booster:
    """
    1ターゲット分の LightGBM モデルを学習する。

    Parameters
    ----------
    params : dict | None
        LightGBM パラメータ。None の場合は config.LGBM_PARAMS を使用。
    num_rounds : int
        最大ブーストラウンド数。
    early_stopping : int
        Early Stopping のラウンド数。
    date_col : pd.Series | None
        日付列（combined["date"]）。提供された場合は末尾 VALIDATION_DAYS 日の
        全銘柄行をValセットとして使用する（位置ベース分割より大きく代表的なValセット）。
        None の場合は従来の位置ベース分割にフォールバック。
    """
    lgbm_params = params if params is not None else config.LGBM_PARAMS

    n_val = config.VALIDATION_DAYS
    if date_col is not None:
        # 日付ベース分割: 末尾 n_val 日 × 全銘柄 をValセットとする
        all_dates = sorted(date_col.unique())
        val_dates = set(all_dates[-n_val:])
        val_mask = date_col.isin(val_dates)
        X_train, X_val = X[~val_mask], X[val_mask]
        y_train, y_val = y[~val_mask], y[val_mask]
    else:
        # フォールバック: 位置ベース分割（date 列なし）
        X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    callbacks = [lgb.log_evaluation(period=100)]
    if early_stopping is not None and early_stopping > 0:
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=early_stopping, verbose=False))

    model = lgb.train(
        params=lgbm_params,
        train_set=dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    logger.info(
        "[%s] 最適ラウンド数: %d / RMSE(val): %.6f",
        label,
        model.best_iteration,
        model.best_score["valid_0"]["rmse"],
    )
    return model


def _combine_features(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """全銘柄の特徴量 DataFrame を縦に結合する。"""
    dfs = []
    for ticker, df in features.items():
        tmp = df.copy()
        # date がインデックスにある場合はカラムに変換（CS 計算の groupby で使用）
        if tmp.index.name == "date" or (
            hasattr(tmp.index, "dtype") and str(tmp.index.dtype).startswith("datetime")
        ):
            tmp = tmp.reset_index()
            if "index" in tmp.columns and "date" not in tmp.columns:
                tmp = tmp.rename(columns={"index": "date"})
        tmp["ticker"] = ticker
        dfs.append(tmp)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    # NOTE: date でのソートは行わない。
    # ソートすると iloc[-VALIDATION_DAYS:] が「最終日の20銘柄のみ」という
    # 極小バリデーションセットになり early stopping が誤動作するため。
    # date カラムは CS 計算の groupby にのみ使用。

    # 無限大・NaN を除去
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    # ターゲット変数の外れ値をクリップ（±20%超は除外）
    # 残差ターゲットが存在する場合はそちらを優先してクリップ
    clip = 0.20
    clip_cols = (
        ["target_alpha_open", "target_alpha_close"]
        if "target_alpha_open" in combined.columns
        else ["target_open_return", "target_close_return"]
    )
    for col in clip_cols:
        if col in combined.columns:
            combined = combined[combined[col].between(-clip, clip)]

    # クロスセクション標準化ターゲット（date ごとに全銘柄の alpha を z-score 化）
    # ±20% クリップ後のデータで計算するため分布が安定している
    if config.USE_CS_TARGET and "date" in combined.columns:
        _src_open  = "target_alpha_open"  if "target_alpha_open"  in combined.columns else "target_open_return"
        _src_close = "target_alpha_close" if "target_alpha_close" in combined.columns else "target_close_return"
        combined["target_cs_open"] = (
            combined.groupby("date")[_src_open]
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            .clip(-3.0, 3.0)
        )
        combined["target_cs_close"] = (
            combined.groupby("date")[_src_close]
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            .clip(-3.0, 3.0)
        )
        logger.info("クロスセクション標準化ターゲットを計算しました（z-score ±3 クリップ）")

    logger.info("学習データ: %d 行 × %d 列（ターゲットクリップ ±%.0f%%）", *combined.shape, clip * 100)
    return combined


def _save_models(model_open: lgb.Booster, model_close: lgb.Booster) -> None:
    with open(MODEL_OPEN_PATH, "wb") as f:
        pickle.dump(model_open, f)
    with open(MODEL_CLOSE_PATH, "wb") as f:
        pickle.dump(model_close, f)
    logger.info("モデルを保存しました: %s, %s", MODEL_OPEN_PATH, MODEL_CLOSE_PATH)


def _load_models() -> tuple[lgb.Booster, lgb.Booster]:
    with open(MODEL_OPEN_PATH, "rb") as f:
        model_open = pickle.load(f)
    with open(MODEL_CLOSE_PATH, "rb") as f:
        model_close = pickle.load(f)
    return model_open, model_close
