"""
TFT (Temporal Fusion Transformer) モデルの学習。

全銘柄の特徴量を TimeSeriesDataSet に変換し、グローバルモデルとして学習する。
始値予測・終値予測の 2 モデルを個別に学習する。

M1 Mac MPS 対応:
  - PYTORCH_ENABLE_MPS_FALLBACK=1 を設定（一部 ops が MPS 未実装）
  - DataLoader(num_workers=0) を使用（MPS と fork の相性問題）
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

import config
from src.features.engineer import get_feature_columns
from src.models.tft_model import TFTModelWrapper, features_to_long_df, build_time_series_dataset

logger = logging.getLogger(__name__)


def _detect_accelerator() -> str:
    """利用可能なアクセラレータを自動検出する。"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def train_tft(
    features: dict,
    target_col: str,
    force: bool = False,
    save: bool = True,
) -> TFTModelWrapper:
    """
    TFT モデルを学習して TFTModelWrapper を返す。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}
    target_col : str
        ターゲット列名（"target_open_return" or "target_close_return"）
    force : bool
        True の場合、保存済みモデルがあっても再学習する
    save : bool
        True の場合、学習後にモデルをファイルに保存する

    Returns
    -------
    TFTModelWrapper
        学習済みモデルのラッパー
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import MAE

    # ディレクトリ決定
    if target_col == "target_open_return":
        save_dir = config.TFT_MODEL_OPEN_DIR
    else:
        save_dir = config.TFT_MODEL_CLOSE_DIR

    # キャッシュチェック
    ckpt_path = save_dir / "best_model.ckpt"
    if not force and save and ckpt_path.exists():
        try:
            logger.info("保存済み TFT モデルを読み込みます: %s", save_dir)
            return TFTModelWrapper.load(save_dir)
        except FileNotFoundError as e:
            logger.warning("キャッシュ読み込み失敗（%s）。再学習します。", e)

    # M1 MPS フォールバック設定
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logger.info("[TFT] モデル学習を開始します (target=%s)", target_col)

    # 特徴量列取得（任意の銘柄から）
    sample_df = next(iter(features.values()))
    feat_cols = get_feature_columns(sample_df)

    # long_df 生成
    long_df = features_to_long_df(features, target_col, clip_target=True)

    # バリデーション分割
    # val_df はエンコーダコンテキスト（60行）＋バリデーション期間（20行）を含む必要がある。
    # val_context_start = val_cutoff - encoder_length + 1
    # → 最初のデコーダ位置が val_cutoff + 1（バリデーション初日）になる。
    max_idx_per_ticker = long_df.groupby("ticker")["time_idx"].max()
    val_cutoff_per_ticker = max_idx_per_ticker - config.VALIDATION_DAYS
    val_context_start_per_ticker = val_cutoff_per_ticker - config.TFT_ENCODER_LENGTH + 1

    long_df = long_df.merge(val_cutoff_per_ticker.rename("val_cutoff"), left_on="ticker", right_index=True)
    long_df = long_df.merge(val_context_start_per_ticker.rename("val_context_start"), left_on="ticker", right_index=True)

    train_df = long_df[long_df["time_idx"] <= long_df["val_cutoff"]].drop(columns=["val_cutoff", "val_context_start"])
    val_df = long_df[long_df["time_idx"] >= long_df["val_context_start"]].drop(columns=["val_cutoff", "val_context_start"])
    del long_df

    logger.info(
        "学習データ: %d 行 / バリデーション: %d 行",
        len(train_df), len(val_df),
    )

    # TimeSeriesDataSet 構築
    training_dataset = build_time_series_dataset(
        train_df,
        target_col=target_col,
        feat_cols=feat_cols,
        mode="train",
    )
    val_dataset = build_time_series_dataset(
        val_df,
        target_col=target_col,
        feat_cols=feat_cols,
        mode="val",
        training_dataset=training_dataset,
    )

    # DataLoader（MPS/fork 相性問題のため num_workers=0）
    train_dl = training_dataset.to_dataloader(
        train=True,
        batch_size=config.TFT_BATCH_SIZE,
        num_workers=0,
    )
    val_dl = val_dataset.to_dataloader(
        train=False,
        batch_size=config.TFT_BATCH_SIZE,
        num_workers=0,
    )

    # モデル構築
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.TFT_LEARNING_RATE,
        hidden_size=config.TFT_HIDDEN_SIZE,
        attention_head_size=config.TFT_ATTENTION_HEADS,
        dropout=config.TFT_DROPOUT,
        hidden_continuous_size=config.TFT_HIDDEN_CONTINUOUS_SIZE,
        loss=MAE(),
        log_interval=-1,
    )
    logger.info(
        "TFT パラメータ数: %s",
        f"{sum(p.numel() for p in tft.parameters()):,}",
    )

    # コールバック
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="best_model",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.TFT_EARLY_STOPPING_PATIENCE,
        mode="min",
    )

    # Trainer
    accelerator = _detect_accelerator()
    logger.info("アクセラレータ: %s", accelerator)

    csv_logger = CSVLogger(
        save_dir=str(save_dir),
        name="logs",
        version=0,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=config.TFT_MAX_EPOCHS,
        gradient_clip_val=config.TFT_GRADIENT_CLIP,
        callbacks=[ckpt_callback, early_stopping],
        enable_progress_bar=True,
        logger=csv_logger,
    )

    trainer.fit(tft, train_dl, val_dl)

    # ベストモデル読み込み
    best_ckpt = ckpt_callback.best_model_path
    logger.info("ベストチェックポイント: %s", best_ckpt)
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    best_model.eval()

    wrapper = TFTModelWrapper(best_model, training_dataset, target_col, feat_cols)

    if save:
        # best_model.ckpt に上書きコピーして load() が常に最新を読めるようにする
        import shutil
        canonical_ckpt = save_dir / "best_model.ckpt"
        if Path(best_ckpt).resolve() != canonical_ckpt.resolve():
            shutil.copy2(best_ckpt, str(canonical_ckpt))
            logger.info("best_model.ckpt を更新しました: %s", canonical_ckpt)
        wrapper.save(save_dir)

    return wrapper
