"""
TFT 学習ログ（CSVLogger 出力）から損失推移を表示・保存する。

使い方:
  conda run -n trade python scripts/plot_tft_loss.py
  conda run -n trade python scripts/plot_tft_loss.py --show   # 画面表示も行う
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

LOG_PATHS = {
    "始値モデル (tft_open)": config.TFT_MODEL_OPEN_DIR / "logs" / "version_0" / "metrics.csv",
    "終値モデル (tft_close)": config.TFT_MODEL_CLOSE_DIR / "logs" / "version_0" / "metrics.csv",
}


def load_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] ログが見つかりません: {path}")
        return None
    df = pd.read_csv(path)
    # train_loss と val_loss を epoch 単位で集約
    df = df.dropna(subset=["epoch"])
    df["epoch"] = df["epoch"].astype(int)
    return df


def plot_loss(show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TFT 学習損失推移", fontsize=14)

    any_plotted = False

    for ax, (label, log_path) in zip(axes, LOG_PATHS.items()):
        df = load_metrics(log_path)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MAE)")
        ax.grid(True, alpha=0.3)

        if df is None:
            ax.text(0.5, 0.5, "ログなし", ha="center", va="center", transform=ax.transAxes)
            continue

        # train_loss: step 単位で記録されるため epoch ごとの最後の値を使う
        if "train_loss_step" in df.columns:
            train = (
                df.dropna(subset=["train_loss_step"])
                .groupby("epoch")["train_loss_step"]
                .last()
            )
            ax.plot(train.index, train.values, label="train_loss", marker="o", markersize=3)
            any_plotted = True

        if "val_loss" in df.columns:
            val = df.dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].last()
            ax.plot(val.index, val.values, label="val_loss", marker="s", markersize=3)
            any_plotted = True

        ax.legend()

        # 最終エポックの損失を表示
        last_epoch = df["epoch"].max()
        print(f"\n=== {label} (最終エポック: {last_epoch}) ===")
        last = df[df["epoch"] == last_epoch]
        for col in ["train_loss_step", "val_loss"]:
            if col in last.columns:
                val_last = last[col].dropna()
                if not val_last.empty:
                    print(f"  {col}: {val_last.values[-1]:.6f}")

    if not any_plotted:
        print("\nまだ学習ログがありません。--retrain で再学習後に実行してください。")
        plt.close()
        return

    out_path = config.RESULTS_DIR / "tft_loss.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n損失グラフを保存しました: {out_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="画面にも表示する")
    args = parser.parse_args()
    plot_loss(show=args.show)
