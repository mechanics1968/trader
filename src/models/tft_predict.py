"""
学習済み TFT モデルで翌日の始値・終値変化率を予測する。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from src.models.tft_model import TFTModelWrapper, build_time_series_dataset

logger = logging.getLogger(__name__)


def predict_next_day_tft(
    features: dict[str, pd.DataFrame],
    model_open: TFTModelWrapper,
    model_close: TFTModelWrapper,
) -> pd.DataFrame:
    """
    全銘柄の翌日の始値・終値を TFT で予測する。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}
    model_open : TFTModelWrapper
        始値変化率予測モデル
    model_close : TFTModelWrapper
        終値変化率予測モデル

    Returns
    -------
    pd.DataFrame
        列: ticker, last_close, last_volume, last_return_pct,
            pred_open, pred_close, pred_open_return_pct, pred_close_return_pct,
            expected_gain_pct
    """
    # 予測用 long_df 構築（末尾 ENCODER_LENGTH 行をコンテキストとして使用）
    pred_long_df = _build_predict_df(features, model_open.feat_cols, model_open.target_col)

    # 始値・終値それぞれ予測
    open_returns = _run_tft_predict(model_open, pred_long_df)
    close_returns = _run_tft_predict(model_close, pred_long_df)

    # 直近クローズ・出来高・リターンを取得
    rows = []
    for ticker, df in features.items():
        if df.empty or ticker not in open_returns or ticker not in close_returns:
            continue

        latest = df.iloc[-1]
        last_close = float(latest["close"])
        last_volume = float(latest["volume"])
        last_return = float(latest.get("return_1d", 0.0))

        open_return = float(open_returns[ticker])
        close_return = float(close_returns[ticker])

        pred_open = last_close * (1 + open_return)
        pred_close = last_close * (1 + close_return)
        expected_gain_pct = (pred_close - pred_open) / (pred_open + 1e-9) * 100

        rows.append({
            "ticker": ticker,
            "last_close": round(last_close, 1),
            "last_volume": int(last_volume),
            "last_return_pct": round(last_return * 100, 2),
            "pred_open": round(pred_open, 1),
            "pred_close": round(pred_close, 1),
            "pred_open_return_pct": round(open_return * 100, 2),
            "pred_close_return_pct": round(close_return * 100, 2),
            "expected_gain_pct": round(expected_gain_pct, 2),
        })

    result = pd.DataFrame(rows)
    logger.info("[TFT] 予測完了: %d 銘柄", len(result))
    return result


# ---------------------------------------------------------------------------
# 内部関数
# ---------------------------------------------------------------------------

def _build_predict_df(
    features: dict[str, pd.DataFrame],
    feat_cols: list[str],
    target_col: str,
    context_len: int = config.TFT_ENCODER_LENGTH,
) -> pd.DataFrame:
    """
    予測用 long_df を構築する。

    各銘柄の末尾 context_len 行を取り出す。
    TFT はエンコーダで過去の target 値を入力として使うため、
    encoder 窓（time_idx 0..context_len-1）の target_col は実際の履歴値をそのまま保持する。
    末尾に追加する dummy 行（time_idx=context_len）だけ target_col=0.0 にする。
    """
    dfs = []
    min_len = context_len + 1

    for ticker, df in features.items():
        if len(df) < min_len:
            continue

        tmp = df.iloc[-context_len:].copy()
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)

        if len(tmp) < config.TFT_ENCODER_LENGTH:
            continue

        # target_col がない場合のみ 0.0 で初期化。ある場合は実際の履歴値を保持。
        if target_col not in tmp.columns:
            tmp[target_col] = 0.0

        tmp = tmp.reset_index(drop=False)
        if "index" in tmp.columns:
            tmp = tmp.rename(columns={"index": "date"})
        elif tmp.index.name:
            tmp = tmp.rename(columns={tmp.index.name: "date"})

        tmp["ticker"] = ticker
        tmp["time_idx"] = range(len(tmp))

        # TFT は min_sequence_length = encoder_length + prediction_length を要求するため、
        # 末尾に dummy 行（time_idx = encoder_length）を追加して decoder 窓を確保する。
        # time_varying_unknown_reals は decoder では使用されないため値は任意。
        dummy = tmp.iloc[[-1]].copy()
        dummy["time_idx"] = len(tmp)
        dummy[target_col] = 0.0
        tmp = pd.concat([tmp, dummy], ignore_index=True)

        dfs.append(tmp)

    if not dfs:
        raise ValueError("予測用データが 0 件です")

    return pd.concat(dfs, axis=0, ignore_index=True)


def _run_tft_predict(
    wrapper: TFTModelWrapper,
    pred_long_df: pd.DataFrame,
) -> dict[str, float]:
    """
    predict モードで TFT 推論し、{ticker: return_value} を返す。
    """
    # モデルが学習時に見た銘柄のみに絞る（未知銘柄は NaNLabelEncoder により除去される）
    known_tickers = set(wrapper.training_dataset.decoded_index["ticker"].unique())
    filtered_df = pred_long_df[pred_long_df["ticker"].isin(known_tickers)]

    if filtered_df.empty:
        logger.warning("[TFT] モデルが知っている銘柄が予測用データに 0 件です")
        return {}

    n_known = filtered_df["ticker"].nunique()
    logger.info("[TFT] 予測対象: %d / %d 銘柄（モデル学習銘柄数: %d）",
                n_known, pred_long_df["ticker"].nunique(), len(known_tickers))

    pred_dataset = build_time_series_dataset(
        filtered_df,
        target_col=wrapper.target_col,
        feat_cols=wrapper.feat_cols,
        mode="predict",
        training_dataset=wrapper.training_dataset,
    )

    pred_dl = pred_dataset.to_dataloader(
        train=False,
        batch_size=config.TFT_BATCH_SIZE,
        num_workers=0,
    )

    import torch as _torch
    _acc = "gpu" if _torch.cuda.is_available() else "cpu"
    predictions = wrapper.model.predict(
        pred_dl,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"accelerator": _acc, "devices": 1},
    )

    # predictions.index は DataFrame（ticker 列を含む）
    # predictions.output は shape=(N, PREDICTION_LENGTH) のテンソル
    index_df = predictions.index
    output = predictions.output.squeeze(-1)

    if hasattr(output, "numpy"):
        output = output.numpy()

    result: dict[str, float] = {}
    for i, row in index_df.iterrows():
        ticker = row["ticker"]
        result[ticker] = float(output[i])

    return result
