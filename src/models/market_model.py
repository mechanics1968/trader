"""
Phase C: 市場方向予測モデル

翌日の市場全体の日中リターン（始値→終値）が正か否かを2値分類で予測する。
入力: 純粋なマクロ指標（米国市場・VIX・ドル円・日経先物・国内市場統計）のみ
出力: 翌日市場上昇確率（0〜1）

このモデルは個別銘柄モデルとは独立して学習・推論され、
推薦フィルタの閾値調整に使用する。
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

MODEL_PATH = config.MODELS_DIR / "market_model.pkl"

# マクロ特徴量として使用する列（us_market + 国内市場統計）
MACRO_FEATURE_COLS = [
    "sp500_ret", "sp500_ret_5d", "nasdaq_ret",
    "vix_level", "vix_change", "vix_high",
    "usdjpy_ret", "usdjpy_level",
    "nk_futures_ret",
    "mkt_return_1d", "mkt_return_5d", "mkt_intraday_return",
]

# ラグ特徴量のウィンドウ（前 N 日分を特徴量として使用）
LAG_DAYS = [1, 2, 3, 5]


def build_market_features(
    market_returns: pd.DataFrame,
    us_market: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    市場方向予測モデル用の特徴量 DataFrame を構築する。

    Parameters
    ----------
    market_returns : pd.DataFrame
        _compute_market_returns() の出力（mkt_return_1d, mkt_return_5d, mkt_intraday_return）
    us_market : pd.DataFrame | None
        fetch_us_market() の出力（sp500_ret, vix_level, ... など）

    Returns
    -------
    pd.DataFrame
        index: date, columns: ラグ付きマクロ特徴量 + ターゲット列 target_mkt_up
    """
    # ベース DataFrame を構築
    df = market_returns.copy()

    # 米国市場特徴量をマージ
    if us_market is not None and not us_market.empty:
        us_idx = us_market.copy()
        us_idx.index = pd.to_datetime(us_idx.index).normalize()
        df = df.join(us_idx, how="left")

    # 利用可能な列のみ使用
    available = [c for c in MACRO_FEATURE_COLS if c in df.columns]

    # ラグ特徴量を生成（過去 N 日の値を今日の特徴量とする）
    feat_frames = []
    for col in available:
        for lag in LAG_DAYS:
            feat_frames.append(df[col].shift(lag).rename(f"{col}_lag{lag}"))

    feat_df = pd.concat(feat_frames, axis=1)

    # ターゲット: 翌日の市場日中リターンが正か（ルックアヘッドに注意: shift(-1) は翌日）
    feat_df["target_mkt_up"] = (df["mkt_intraday_return"].shift(-1) > 0).astype(int)

    # 先頭のNaN行と末尾のターゲットNaN行を除去
    feat_df = feat_df.dropna()
    return feat_df


def train_market_model(
    market_returns: pd.DataFrame,
    us_market: pd.DataFrame | None,
    save: bool = True,
) -> lgb.Booster:
    """
    市場方向予測モデルを学習する。

    バリデーション: 末尾 20 日をテストセットとして保留
    """
    feat_df = build_market_features(market_returns, us_market)
    if len(feat_df) < 40:
        logger.warning("市場モデル学習データが不足しています（%d 行）", len(feat_df))
        return None

    feature_cols = [c for c in feat_df.columns if c != "target_mkt_up"]
    X = feat_df[feature_cols]
    y = feat_df["target_mkt_up"]

    n_val = 20
    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
    y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=9999),
        ],
    )

    # バリデーション精度
    val_preds = model.predict(X_val)
    val_dir = ((val_preds > 0.5) == y_val.values).mean()
    logger.info(
        "市場方向モデル学習完了: rounds=%d, val_dir_acc=%.3f",
        model.best_iteration, val_dir,
    )

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("市場方向モデルを保存しました: %s", MODEL_PATH)

    return model


def predict_market_direction(
    market_returns: pd.DataFrame,
    us_market: pd.DataFrame | None,
    model: lgb.Booster | None = None,
) -> float:
    """
    翌日の市場上昇確率を返す。

    Parameters
    ----------
    model : lgb.Booster | None
        None の場合は保存済みモデルをロードする

    Returns
    -------
    float
        翌日市場上昇確率（0〜1）。モデルがない場合は 0.5（中立）
    """
    if model is None:
        if not MODEL_PATH.exists():
            logger.warning("市場方向モデルが見つかりません。中立（0.5）を返します")
            return 0.5
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    feat_df = build_market_features(market_returns, us_market)
    if feat_df.empty:
        return 0.5

    feature_cols = [c for c in feat_df.columns if c != "target_mkt_up"]
    # 最新行（翌日予測用）: ターゲットは不要なので最後の行を使う
    # ただし build_market_features は dropna しているので末尾が使えない場合がある
    # → 末尾 1 行を取得（target_mkt_up は予測対象なので NaN でも無視）
    latest_feats = feat_df[feature_cols].iloc[[-1]]
    available = [c for c in feature_cols if c in latest_feats.columns]
    # モデルの特徴量に合わせる
    model_feats = model.feature_name()
    row = latest_feats.reindex(columns=model_feats, fill_value=np.nan)

    prob = float(model.predict(row)[0])
    logger.info("翌日市場上昇確率: %.3f", prob)
    return prob
