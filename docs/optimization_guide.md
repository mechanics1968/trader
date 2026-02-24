# 多目的ハイパーパラメータ最適化 手順書

## 概要

翌日の株価予測精度を高めるため、LightGBM のハイパーパラメータを
**Optuna NSGA-II（多目的遺伝的アルゴリズム）** で自動探索する。

予測誤差（目的関数）を最小化するパラメータの集合（Pareto 最前線）を求め、
最終的に本番モデルへ適用する。

---

## 目的関数

4 つの指標をすべて **最小化** する方向で最適化する。

| 変数名 | 意味 | 計算式 |
|---|---|---|
| `rmse_open` | 翌日始値変化率の予測誤差 | √(Σ(予測始値変化率 − 実績始値変化率)²) |
| `rmse_close` | 翌日終値変化率の予測誤差 | √(Σ(予測終値変化率 − 実績終値変化率)²) |
| `1_minus_dir_acc_open` | 始値の上昇/下落方向の誤的中率 | 1 − (方向が一致した日数 / 総日数) |
| `1_minus_dir_acc_close` | 終値の上昇/下落方向の誤的中率 | 1 − (方向が一致した日数 / 総日数) |

> **なぜ 1 − 方向的中率か？**
> Optuna は最小化を前提とするため、「最大化したい指標」は符号を反転する。
> 方向的中率 0.6 → `1 − 0.6 = 0.4` を最小化することで的中率を上げる。

### 誤差の計算ロジック

```
データ期間（1年 = 約250営業日）
│
├── 学習セット: 先頭 230日（VALIDATION_DAYS=20 を除いた部分）
│   └── モデルを学習
│
└── バリデーションセット: 末尾 20日
    ├── 各日: 前日までの特徴量 → 当日の始値・終値変化率を予測
    └── 予測値 − 実績値 → 誤差指標を計算
```

実装: `src/models/metrics.py`

---

## 探索空間（ハイパーパラメータ）

| パラメータ | 探索範囲 | スケール | 意味 |
|---|---|---|---|
| `learning_rate` | 0.01 〜 0.3 | 対数 | 1 ステップあたりの学習量 |
| `num_leaves` | 15 〜 255 | 整数 | 木の葉の数（複雑さ） |
| `max_depth` | 3 〜 12 | 整数 | 木の最大深さ |
| `min_child_samples` | 10 〜 100 | 整数 | 葉に必要な最低サンプル数（過学習抑制） |
| `min_gain_to_split` | 0.0 〜 1.0 | 連続 | 分岐に必要な最低利得 |
| `feature_fraction` | 0.4 〜 1.0 | 連続 | 各ツリーで使う特徴量の割合 |
| `bagging_fraction` | 0.4 〜 1.0 | 連続 | 各ツリーで使うデータの割合 |
| `bagging_freq` | 1 〜 10 | 整数 | バギングの実施頻度 |
| `lambda_l1` | 1e-8 〜 10.0 | 対数 | L1 正則化係数 |
| `lambda_l2` | 1e-8 〜 10.0 | 対数 | L2 正則化係数 |

---

## 実行手順

### ステップ 1: データ準備（初回のみ）

```bash
# 株価データの取得・特徴量生成・ベースラインモデルの学習
conda activate trade
python main.py
```

### ステップ 2: 多目的最適化の実行

```bash
# デフォルト（全銘柄・50試行）
python main.py --optimize

# 速度重視（300銘柄に絞る、所要時間: 約10〜30分）
python main.py --optimize --n-trials 50 --n-tickers 300

# より精度重視（全銘柄・100試行、所要時間: 数時間）
python main.py --optimize --n-trials 100

# 結果を SQLite に保存し、後から再実行・再開できる
python main.py --optimize --n-trials 100 --n-tickers 300 \
    --storage "sqlite:///optuna.db"
```

**`--n-tickers` の目安**

| 銘柄数 | 1 Trial の所要時間 | 推奨用途 |
|---|---|---|
| 50〜100 | 約 0.2 秒 | 動作確認・スモークテスト |
| 300 | 約 2〜5 秒 | 通常の最適化（推奨） |
| 全銘柄（3,776） | 約 30〜60 秒 | 精度重視の本番最適化 |

### ステップ 3: 結果の確認

最適化完了後、以下のファイルが生成される。

```
trader/
├── optuna_results/
│   ├── all_trials.csv     # 全 Trial の目的関数値・パラメータ一覧
│   └── pareto_front.csv   # Pareto 最前線の点のみ抽出
└── best_params.json       # 推薦パラメータ（rmse_open+rmse_close 最小の1点）
```

**Pareto 最前線とは？**

4 つの目的関数すべてにおいて、他のどの Trial にも支配されない点の集合。
例えば：

```
Trial A: rmse_open=0.010, rmse_close=0.015  ← どちらも小さい → Pareto 最前線
Trial B: rmse_open=0.012, rmse_close=0.012  ← A より close は良いが open は悪い → Pareto 最前線
Trial C: rmse_open=0.015, rmse_close=0.020  ← A にも B にも支配される → 除外
```

`best_params.json` はこの Pareto 最前線の中から `rmse_open + rmse_close` が
最小の 1 点を自動選択したもの。

**`pareto_front.csv` の列**

| 列名 | 意味 |
|---|---|
| `obj_rmse_open` | 始値 RMSE |
| `obj_rmse_close` | 終値 RMSE |
| `obj_1_minus_dir_acc_open` | 始値誤的中率（小さいほど良い） |
| `obj_1_minus_dir_acc_close` | 終値誤的中率（小さいほど良い） |
| `param_*` | 対応するハイパーパラメータ値 |
| `trial_number` | Optuna の Trial 番号 |

### ステップ 4: 本番モデルへの適用

```bash
# best_params.json のパラメータで LightGBM を全銘柄再学習 → 保存
python main.py --apply-best

# 適用後すぐに推薦を実行したい場合
python main.py --apply-best --backtest
```

---

## ファイル・クラス対応表

| 処理 | ファイル | 主な関数 / クラス |
|---|---|---|
| 誤差計算 | `src/models/metrics.py` | `compute_errors()` / `PredictionErrors` |
| 最適化実行 | `src/models/optimize.py` | `run_optimization()` |
| パラメータ適用 | `src/models/optimize.py` | `apply_best_params()` |
| モデル学習 | `src/models/train.py` | `train()` / `_train_single()` |

---

## 典型的な最適化サイクル

```
毎週末（金曜引け後）に実施することを推奨

[引け後 15:30]
  └── python main.py              # 通常の推薦実行

[週末]
  ├── python main.py --optimize --n-trials 100 --n-tickers 300
  │     # Pareto 最前線の確認 → best_params.json を確認・必要なら手動調整
  └── python main.py --apply-best # 翌週から最適化済みモデルで運用

[翌週月曜〜]
  └── python main.py              # 最適化済みパラメータで推薦
```

---

## 注意事項

- **銘柄数が少ないと目的関数値が収束しない**
  50 銘柄程度では Trial ごとの差が出にくい。300 銘柄以上を推奨。

- **試行回数が少ないと Pareto 最前線が粗い**
  NSGA-II は進化的アルゴリズムのため、50〜100 trial 以上で有意な探索が進む。

- **Pareto 最前線は「トレードオフの地図」**
  `rmse_open` と `dir_acc_close` は必ずしも同時に改善しない。
  何を最重視するかはトレーダー自身の判断で `best_params.json` を編集して良い。

- **方向的中率 55% 以上が実用ライン**
  `1_minus_dir_acc_close < 0.45` となる Trial を優先的に採用することを推奨。
