"""
TFT (Temporal Fusion Transformer) モデルのラッパーとデータ変換ユーティリティ。

データ変換フロー:
  {ticker: DatetimeIndex DataFrame}
      → features_to_long_df()
      → long-format DataFrame（ticker, time_idx, date, ...）
      → build_time_series_dataset()
      → pytorch_forecasting.TimeSeriesDataSet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

import config

logger = logging.getLogger(__name__)

# 最小データ長（エンコーダ + 予測 + バッファ1行）
MIN_TICKER_LEN = config.TFT_ENCODER_LENGTH + config.TFT_PREDICTION_LENGTH + 1


# ---------------------------------------------------------------------------
# データ変換
# ---------------------------------------------------------------------------

def features_to_long_df(
    features: dict[str, pd.DataFrame],
    target_col: str,
    clip_target: bool = True,
) -> pd.DataFrame:
    """
    {ticker: DatetimeIndex DataFrame} を long-format に変換する。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}（target_open_return / target_close_return 含む）
    target_col : str
        ターゲット列名（"target_open_return" or "target_close_return"）
    clip_target : bool
        True の場合、ターゲットを ±20% でクリップする

    Returns
    -------
    pd.DataFrame
        列: ticker, time_idx, date, <feat_cols>, <target_col>
    """
    from src.features.engineer import get_feature_columns

    dfs = []
    skipped = 0

    for ticker, df in features.items():
        if df.empty or target_col not in df.columns:
            skipped += 1
            continue

        tmp = df.copy()
        # float32 でメモリ削減
        feat_cols = get_feature_columns(tmp)
        tmp[feat_cols] = tmp[feat_cols].astype("float32")

        # NaN / Inf を除去
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])

        if len(tmp) < MIN_TICKER_LEN:
            skipped += 1
            continue

        # ターゲットのクリップ
        if clip_target:
            clip = 0.20
            tmp = tmp[tmp[target_col].between(-clip, clip)]
            if len(tmp) < MIN_TICKER_LEN:
                skipped += 1
                continue

        # time_idx: dropna 後に 0 始まり連番（連続性を保証）
        tmp = tmp.reset_index(drop=False)
        if "index" in tmp.columns:
            tmp = tmp.rename(columns={"index": "date"})
        elif tmp.index.name:
            tmp = tmp.rename(columns={tmp.index.name: "date"})

        tmp["ticker"] = ticker
        tmp["time_idx"] = range(len(tmp))

        dfs.append(tmp)

    if not dfs:
        raise ValueError("有効な銘柄データが 0 件です（データが短すぎるか特徴量が不足）")

    long_df = pd.concat(dfs, axis=0, ignore_index=True)

    if skipped > 0:
        logger.info("データ不足でスキップした銘柄: %d 件", skipped)
    logger.info(
        "long_df 生成完了: %d 行 × %d 列 / %d 銘柄",
        len(long_df), long_df.shape[1], long_df["ticker"].nunique(),
    )
    return long_df


def build_time_series_dataset(
    long_df: pd.DataFrame,
    target_col: str,
    feat_cols: list[str],
    encoder_length: int = config.TFT_ENCODER_LENGTH,
    prediction_length: int = config.TFT_PREDICTION_LENGTH,
    mode: Literal["train", "val", "predict"] = "train",
    training_dataset=None,
):
    """
    pytorch_forecasting.TimeSeriesDataSet を構築する。

    Parameters
    ----------
    long_df : pd.DataFrame
        features_to_long_df() の出力
    target_col : str
        ターゲット列名
    feat_cols : list[str]
        時変連続特徴量の列名リスト
    mode : "train" | "val" | "predict"
        "train" → 新規 dataset 作成
        "val" / "predict" → training_dataset から normalizer を引き継ぐ
    training_dataset : TimeSeriesDataSet | None
        mode != "train" の場合に必須

    Returns
    -------
    pytorch_forecasting.TimeSeriesDataSet
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder

    if mode == "train":
        max_encoder_length = encoder_length
        max_prediction_length = prediction_length

        dataset = TimeSeriesDataSet(
            long_df,
            time_idx="time_idx",
            target=target_col,
            group_ids=["ticker"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["ticker"],
            # add_nan=True: 学習時に存在しなかった銘柄を予測する際も
            # "unknown" 埋め込みとして扱えるようにする
            categorical_encoders={"ticker": NaNLabelEncoder(add_nan=True)},
            time_varying_unknown_reals=feat_cols,  # target_col は target= で自動管理
            # 正規化なし: リターン値は既に -0.2〜+0.2 に収まっており追加正規化は不要。
            # GroupNormalizer(softplus) は非負値向け、EncoderNormalizer はゼロ近傍値を
            # 全サンプルゼロに潰してしまうため不適。
            target_normalizer=None,
            allow_missing_timesteps=True,
        )
        return dataset
    else:
        if training_dataset is None:
            raise ValueError("mode='val'/'predict' の場合は training_dataset が必須です")
        predict_mode = mode == "predict"
        dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            long_df,
            predict=predict_mode,
            stop_randomization=True,
        )
        return dataset


# ---------------------------------------------------------------------------
# バリデーション用コンテキスト抽出
# ---------------------------------------------------------------------------

def _extract_val_context(
    features: dict[str, pd.DataFrame],
    target_col: str,
    feat_cols: list[str],
    n_val_days: int,
    context_len: int = config.TFT_ENCODER_LENGTH,
) -> pd.DataFrame:
    """
    バリデーション用の long_df を構築する。

    各銘柄の末尾 (context_len + n_val_days + PREDICTION_LENGTH) 行を取り出し、
    time_idx を 0 から振り直す。TFT がエンコーダのコンテキストとして
    前の日数も必要なため、val 期間だけでは不足する。
    """
    n_context = context_len + n_val_days + config.TFT_PREDICTION_LENGTH
    dfs = []

    for ticker, df in features.items():
        if len(df) < n_context + 1:
            continue

        tmp = df.iloc[-n_context:].copy()
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])

        if len(tmp) < config.TFT_ENCODER_LENGTH + config.TFT_PREDICTION_LENGTH + 1:
            continue

        tmp = tmp.reset_index(drop=False)
        if "index" in tmp.columns:
            tmp = tmp.rename(columns={"index": "date"})
        elif tmp.index.name:
            tmp = tmp.rename(columns={tmp.index.name: "date"})

        tmp["ticker"] = ticker
        tmp["time_idx"] = range(len(tmp))
        dfs.append(tmp)

    if not dfs:
        raise ValueError("バリデーション用のデータが 0 件です")

    return pd.concat(dfs, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# TFTModelWrapper
# ---------------------------------------------------------------------------

class TFTModelWrapper:
    """
    学習済み TFT モデルと TimeSeriesDataSet をまとめるラッパー。

    Attributes
    ----------
    model : TemporalFusionTransformer
        学習済み TFT モデル
    training_dataset : TimeSeriesDataSet
        学習時に使用した TimeSeriesDataSet（normalizer の引き継ぎに必要）
    target_col : str
        ターゲット列名
    feat_cols : list[str]
        時変連続特徴量の列名リスト
    """

    def __init__(self, model, training_dataset, target_col: str, feat_cols: list[str]) -> None:
        self.model = model
        self.training_dataset = training_dataset
        self.target_col = target_col
        self.feat_cols = feat_cols

    # ------------------------------------------------------------------
    # 保存・読み込み
    # ------------------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """training_dataset を pickle で、メタ情報を meta.json で保存する。"""
        import pickle

        dir_path.mkdir(parents=True, exist_ok=True)

        with open(dir_path / "dataset.pkl", "wb") as f:
            pickle.dump(self.training_dataset, f)

        meta = {
            "target_col": self.target_col,
            "feat_cols": self.feat_cols,
        }
        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("TFTModelWrapper を保存しました: %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "TFTModelWrapper":
        """保存済みの TFTModelWrapper を読み込む。"""
        import pickle
        from pytorch_forecasting import TemporalFusionTransformer

        ckpt_path = dir_path / "best_model.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

        meta_path = dir_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"メタデータが見つかりません: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        pkl_path = dir_path / "dataset.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"dataset.pkl が見つかりません: {pkl_path}\n"
                "古い dataset.pt 形式は非対応です。--retrain で再学習してください。"
            )

        with open(pkl_path, "rb") as f:
            training_dataset = pickle.load(f)

        model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
        model.eval()

        logger.info("TFTModelWrapper を読み込みました: %s", dir_path)
        return cls(model, training_dataset, meta["target_col"], meta["feat_cols"])

    # ------------------------------------------------------------------
    # バリデーション予測
    # ------------------------------------------------------------------

    def compute_val_predictions(
        self,
        features: dict[str, pd.DataFrame],
        n_val_days: int = config.VALIDATION_DAYS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        バリデーション期間の予測値と実績値を返す。

        Parameters
        ----------
        features : dict[str, pd.DataFrame]
            {ticker: 特徴量 DataFrame}
        n_val_days : int
            バリデーション日数

        Returns
        -------
        (y_pred, y_actual) : tuple[np.ndarray, np.ndarray]
            どちらも shape=(N,) の 1D 配列
        """
        # 学習時に見た銘柄のみで評価（未知銘柄は NaNLabelEncoder で壊れた予測が生成されるため除外）
        known_tickers = set(self.training_dataset.decoded_index["ticker"].unique())
        features_known = {k: v for k, v in features.items() if k in known_tickers}
        if not features_known:
            logger.warning("compute_val_predictions: 既知銘柄が 0 件です")
            return np.array([]), np.array([])
        logger.info("compute_val_predictions: %d / %d 銘柄を評価", len(features_known), len(features))

        val_long_df = _extract_val_context(
            features_known,
            self.target_col,
            self.feat_cols,
            n_val_days,
            context_len=config.TFT_ENCODER_LENGTH,
        )

        val_dataset = build_time_series_dataset(
            val_long_df,
            self.target_col,
            self.feat_cols,
            mode="val",
            training_dataset=self.training_dataset,
        )

        val_dl = val_dataset.to_dataloader(
            train=False,
            batch_size=config.TFT_BATCH_SIZE,
            num_workers=0,
        )

        import torch as _torch
        _acc = "gpu" if _torch.cuda.is_available() else "cpu"
        predictions = self.model.predict(
            val_dl,
            return_y=True,
            mode="prediction",
            trainer_kwargs={"accelerator": _acc, "devices": 1},
        )

        y_pred = predictions.output.squeeze(-1).cpu().numpy()
        y_actual = predictions.y[0].squeeze(-1).cpu().numpy()

        return y_pred, y_actual
