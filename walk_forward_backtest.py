"""
ウォークフォワード・バックテスト

各ターゲット日 T に対して:
  1. 日付 <= T-2 の特徴量でモデルを再学習（情報漏洩なし）
  2. T-1 の特徴量で翌日（T）を予測
  3. raw データの T の始値・終値と比較

実行例:
  conda run -n trade python walk_forward_backtest.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import json
import config
from src.models.train import train, MODEL_OPEN_PATH, MODEL_CLOSE_PATH
from src.models.predict import predict_next_day

# best_params.json が存在すればロードしてウォークフォワードに使用する
_best_params_path = config.BASE_DIR / "best_params.json"
if _best_params_path.exists():
    _best_params = json.loads(_best_params_path.read_text(encoding="utf-8"))
    print(f"best_params.json を使用します: {_best_params_path}")
else:
    _best_params = None
    print("best_params.json が見つかりません。デフォルトパラメータを使用します。")

# ---------------------------------------------------------------------------
# 1. 全特徴量・rawデータを一括読み込み
# ---------------------------------------------------------------------------
print("特徴量読み込み中...")
proc_dir = config.DATA_DIR / "processed"
raw_dir  = config.DATA_DIR / "raw"

all_features: dict[str, pd.DataFrame] = {}
raw_prices:   dict[str, pd.DataFrame] = {}

for f in proc_dir.glob("*.csv"):
    ticker = f.stem[:-2] + ".T" if f.stem.endswith("_T") else f.stem.replace("_", ".")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    if len(df) > 0:
        all_features[ticker] = df
    rf = raw_dir / f.name
    if rf.exists():
        raw_prices[ticker] = pd.read_csv(rf, index_col=0, parse_dates=True)

print(f"特徴量: {len(all_features)} 銘柄 / raw: {len(raw_prices)} 銘柄")

# ---------------------------------------------------------------------------
# 2. バックテスト対象日を決定
#    raw データに存在する取引日を使い、直近 N 日をターゲットとする
# ---------------------------------------------------------------------------
BACKTEST_DAYS = 50  # 直近何日分をテストするか

sample_raw = next(iter(raw_prices.values()))
all_trading_dates = sorted(sample_raw.index.tolist())
target_dates = all_trading_dates[-BACKTEST_DAYS:]  # 直近 N 日

print(f"\nバックテスト対象日 ({len(target_dates)} 日):")
for d in target_dates:
    print(f"  {d.date()}")
print()

# ---------------------------------------------------------------------------
# 3. ウォークフォワード実行
# ---------------------------------------------------------------------------
daily_results: list[dict] = []

for i, target_date in enumerate(target_dates):
    idx = all_trading_dates.index(target_date)
    if idx < 2:
        continue  # T-1, T-2 が存在しない場合はスキップ

    prev_date      = all_trading_dates[idx - 1]   # T-1: 予測に使う特徴量の最終日
    prev_prev_date = all_trading_dates[idx - 2]   # T-2: 学習データの最終日

    # ---- 学習用: date <= T-2 の行のみ ----
    train_features: dict[str, pd.DataFrame] = {}
    for ticker, df in all_features.items():
        sub = df[df.index <= prev_prev_date]
        if len(sub) >= 62:   # 最低限の行数（TFT不要、LGBMは余裕あり）
            train_features[ticker] = sub

    if not train_features:
        logger.warning("%s: 学習データなし", target_date.date())
        continue

    # ---- モデル再学習 ----
    model_open, model_close = train(
        train_features,
        force=True,
        save=False,
        params=_best_params,  # None の場合は config.LGBM_PARAMS を使用
    )

    # ---- 予測用: date <= T-1 の行のみ（最終行が T-1 になる） ----
    pred_features: dict[str, pd.DataFrame] = {}
    for ticker, df in all_features.items():
        sub = df[df.index <= prev_date]
        if len(sub) > 0:
            pred_features[ticker] = sub

    predictions = predict_next_day(pred_features, model_open, model_close)

    # ---- 実績取得 ----
    actuals = []
    for ticker, rdf in raw_prices.items():
        if target_date in rdf.index:
            row = rdf.loc[target_date]
            actuals.append({
                "ticker": ticker,
                "actual_open":  float(row["open"]),
                "actual_close": float(row["close"]),
            })
    actual_df = pd.DataFrame(actuals)

    if actual_df.empty:
        logger.warning("%s: 実績データなし", target_date.date())
        continue

    merged = predictions.merge(actual_df, on="ticker", how="inner")
    merged["actual_open_return_pct"]  = (merged["actual_open"]  / merged["last_close"] - 1) * 100
    merged["actual_close_return_pct"] = (merged["actual_close"] / merged["last_close"] - 1) * 100
    merged["actual_gain_pct"]         = (
        (merged["actual_close"] - merged["actual_open"]) / merged["actual_open"] * 100
    )

    open_acc  = (np.sign(merged["pred_open_return_pct"])  == np.sign(merged["actual_open_return_pct"])).mean()
    close_acc = (np.sign(merged["pred_close_return_pct"]) == np.sign(merged["actual_close_return_pct"])).mean()
    eg_acc    = (np.sign(merged["expected_gain_pct"])     == np.sign(merged["actual_gain_pct"])).mean()

    # 市場全体リターン（±50% 超の異常値を除外して計算）
    mkt_open  = merged["actual_open_return_pct"].clip(-50, 50).mean()
    mkt_close = merged["actual_close_return_pct"].clip(-50, 50).mean()
    rmse_close = np.sqrt(((merged["pred_close_return_pct"] - merged["actual_close_return_pct"].clip(-50, 50))**2).mean())

    # 前日の等加重市場リターン（recommendations の市場フィルタ用）
    mkt_return_prev = (
        merged["last_mkt_return"].mean()
        if "last_mkt_return" in merged.columns else 0.0
    )

    # 推薦銘柄: フィルタ後、期待上昇率の高い上位 TOP_N 件に絞る
    TOP_N = 20
    _base_filter = (
        (merged["expected_gain_pct"] >= config.MIN_EXPECTED_GAIN_PCT) &
        (merged["last_volume"] >= config.MIN_VOLUME) &
        (merged["last_return_pct"].abs() <= config.MAX_DAILY_CHANGE_PCT)
    )

    # フィルタなし（市場条件を無視した推薦）
    rec_nofilter = merged[_base_filter].nlargest(TOP_N, "expected_gain_pct").copy()

    # フィルタあり（市場下落日は推薦しない）
    mkt_filtered_out = mkt_return_prev < config.MARKET_DECLINE_THRESHOLD * 100
    rec = rec_nofilter if not mkt_filtered_out else merged.iloc[:0].copy()

    n_rec = len(rec)
    n_win = int((rec["actual_gain_pct"] > 0).sum()) if n_rec > 0 else 0
    win_rate = n_win / n_rec if n_rec > 0 else float("nan")

    # フィルタなし勝率（比較用）
    n_rec_nf = len(rec_nofilter)
    n_win_nf = int((rec_nofilter["actual_gain_pct"] > 0).sum()) if n_rec_nf > 0 else 0

    day_result = {
        "target_date":      target_date.date(),
        "n_stocks":         len(merged),
        "open_acc":         open_acc,
        "close_acc":        close_acc,
        "eg_acc":           eg_acc,
        "rmse_close":       rmse_close,
        "mkt_prev":         mkt_return_prev,
        "mkt_filtered_out": mkt_filtered_out,
        "n_rec":            n_rec,
        "n_win":            n_win,
        "win_rate":         win_rate,
        "n_rec_nf":         n_rec_nf,
        "n_win_nf":         n_win_nf,
        "mkt_open":         mkt_open,
        "mkt_close":        mkt_close,
    }
    daily_results.append(day_result)

    mkt_tag = " [市場フィルタで除外]" if mkt_filtered_out else ""
    status = f"{n_win}/{n_rec}={win_rate:.0%}" if n_rec > 0 else "推薦なし"
    print(
        f"  {target_date.date()}  "
        f"終値方向的中率:{close_acc:.1%}  "
        f"期待上昇的中率:{eg_acc:.1%}  "
        f"推薦勝率:{status:>12}  "
        f"市場前日:{mkt_return_prev:+.2f}%  "
        f"市場当日:{mkt_close:+.2f}%"
        f"{mkt_tag}"
    )

# ---------------------------------------------------------------------------
# 4. 集計・サマリー出力
# ---------------------------------------------------------------------------
result_df = pd.DataFrame(daily_results)

print()
print("=" * 70)
print("  ウォークフォワード・バックテスト サマリー")
print("=" * 70)
print(f"テスト日数       : {len(result_df)} 日")
print(f"平均 終値方向的中率: {result_df['close_acc'].mean():.1%}  (std: {result_df['close_acc'].std():.1%})")
print(f"平均 期待上昇的中率: {result_df['eg_acc'].mean():.1%}  (std: {result_df['eg_acc'].std():.1%})")
print(f"平均 終値 RMSE   : {result_df['rmse_close'].mean():.4f}%")
print(f"平均 市場リターン: {result_df['mkt_close'].mean():+.3f}%/日")
print()

# ---- フィルタあり 集計 ----
rec_days = result_df[result_df["n_rec"] > 0]
if not rec_days.empty:
    total_rec = int(rec_days["n_rec"].sum())
    total_win = int(rec_days["n_win"].sum())
    total_win_rate = total_win / total_rec
    n_filtered = int(result_df["mkt_filtered_out"].sum())
    print(f"市場フィルタで除外  : {n_filtered} 日 / {len(result_df)} 日")
    print(f"推薦あり日数     : {len(rec_days)} 日 / {len(result_df)} 日")
    print(f"総推薦銘柄数     : {total_rec} 件")
    print(f"総勝利数         : {total_win} 件")
    print(f"総合 推薦勝率    : {total_win}/{total_rec} = {total_win_rate:.1%}")
else:
    print("推薦銘柄が0件の日のみでした")

# ---- フィルタなし 集計（比較用）----
total_rec_nf = int(result_df["n_rec_nf"].sum())
total_win_nf = int(result_df["n_win_nf"].sum())
if total_rec_nf > 0:
    print()
    print(f"【参考: フィルタなし】推薦勝率: {total_win_nf}/{total_rec_nf} = {total_win_nf/total_rec_nf:.1%}")

print()
print("--- 推薦あり日の詳細（フィルタあり）---")
for _, row in result_df.iterrows():
    if row["mkt_filtered_out"]:
        nf_rate = f"{int(row['n_win_nf'])}/{int(row['n_rec_nf'])}={row['n_win_nf']/row['n_rec_nf']:.0%}" if row['n_rec_nf'] > 0 else "推薦なし"
        print(
            f"  {row['target_date']}  [市場フィルタ除外]  "
            f"(フィルタなし: {nf_rate})  "
            f"市場前日:{row['mkt_prev']:+.2f}%  市場当日:{row['mkt_close']:+.2f}%"
        )
    elif row["n_rec"] > 0:
        print(
            f"  {row['target_date']}  "
            f"{int(row['n_win'])}/{int(row['n_rec'])}={row['win_rate']:.0%}  "
            f"市場前日:{row['mkt_prev']:+.2f}%  市場当日:{row['mkt_close']:+.2f}%  "
            f"終値方向的中率{row['close_acc']:.1%}"
        )

print()
print("--- 1日ごと詳細 ---")
cols = ["target_date", "close_acc", "eg_acc", "n_rec", "n_win", "win_rate", "mkt_prev", "mkt_close"]
print(result_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
