# ベータ依存性排除 実装プラン

作成日: 2026-02-27

---

## 1. 問題の本質

### 現状の構造的欠陥

現在のモデルは「翌日の日中騰落率（絶対値）」を予測している。
この設計には以下の根本的な問題がある。

```
市場上昇日: ほぼ全銘柄が上昇 → モデルが「上昇」と予測 → 推薦勝率 75〜85%
市場下落日: ほぼ全銘柄が下落 → モデルが「上昇」と予測 → 推薦勝率  5〜15%
→ 最大 80pp の乖離（development_report.md 6.3節より）
```

**モデルが予測しているのは「個別銘柄の固有の動き」ではなく「市場全体の動きに乗った絶対リターン」である。**

---

## 2. 過去の失敗（試行#2）との違い

development_report.md の試行#2（推薦勝率 29.6%）では、
`target_close_return`（前日終値→翌日終値）にのみ市場リターンを引いたため、
`target_open_return`（前日終値→翌日始値）との定義が不一致になり崩壊した。

```python
# 試行#2の失敗パターン
target_open_return  = open.shift(-1) / close - 1          # 市場込み
target_close_return = close.shift(-1) / close - 1 - mkt   # 市場除き
# → expected_gain = pred_close - pred_open が意味を失う
```

**今回は日中騰落率モード（`USE_INTRADAY_TARGET=True`）を採用済みなので、
`model_close` のターゲット1つだけを変更すれば整合が取れる。**

```
model_open  : target_open_return（オーバーナイトギャップ → 変更なし）
model_close : target_intraday_alpha（日中α = 日中騰落率 - 市場日中騰落率）
推薦スコア  : predicted_intraday_alpha（αがプラス = 市場より上がる銘柄）
```

---

## 3. 現状の特徴量分析

### 良い点

#### 市場・マクロ系特徴量が支配的（これは正しい）

feature importance（gain）上位が米国市場・マクロ系で占められており、
「翌日の始値は米国市場・マクロで決まる」という仮説が実証されている。

| 特徴量 | gain（始値モデル） | 解釈 |
|---|---|---|
| `beta_20d` | 18.9% | 市場感応度（最重要） |
| `vix_change` | 9.8% | リスクオン/オフ |
| `mkt_return_1d` | 9.0% | 市場方向 |
| `nasdaq_ret` | 7.9% | 米国ハイテク |
| `usdjpy_level` | 7.8% | 円安/円高 |
| `usdjpy_ret` | 6.0% | ドル円変化率 |
| `nk_futures_ret` | 5.1% | 夜間先物 |

#### 比率ベース統一
3,776銘柄を1モデルで扱うため、全特徴量を比率ベースに統一している設計は正しい。

#### クロスセクショナルランク
`cs_rank_amihud`（1.96%）など銘柄間の相対的な位置づけが日中モデルで効いており、
グローバルモデルの弱点（銘柄固有特性を捉えにくい）を補完できている。

#### Amihud非流動性
`amihud`（始値モデル 5.0%）が明確に有効。
流動性の低い銘柄は価格インパクトが大きくギャップが生じやすい仮説を実証。

### 悪い点・課題

#### 1. 市場ベータ依存性（最大の問題）

特徴量として `beta_20d` や `alpha_1d` を入力しても、
**ターゲットが絶対リターンである限りベータ依存性は解消されない。**
市場全体が下落する日は全銘柄が下落するため、絶対値予測では回避不能。

#### 2. ギャップ特徴量の寄与がまだ微弱

今回追加した `gap_abs`（0.23%）、`gap_vs_atr`（0.18%）は効いているが小さい。
ギャップとその後の値動きの関係はより非線形なパターンがある可能性があり、
αターゲット化後に再評価が必要。

#### 3. テクニカル指標が日中騰落率モデルでほぼ効いていない

SMA/EMA/MACD/ADXなどの中長期トレンド系指標はいずれも日中モデルで gain が低い。
日中の動きは前日までのトレンドより当日の市場環境で決まることを示しており、
これらの特徴量は冗長な可能性がある。

#### 4. データ量の壁

- 1年データ（250日）が最適と確認済み（試行#8: 3年データで悪化 42.8%）
- 3年データが悪化した原因: 2022年金利上昇・2023年回復・2024年上昇相場など
  異なる市場レジームが混在し、モデルが「平均化」されてしまう
- **解決策**: データ量増加ではなく time-decay weighting（直近データへの重みづけ）が有望

#### 5. 日中騰落率はランダム性が高い

`model_close` が32〜83ラウンドと少ないラウンドで収束する傾向がある。
αターゲット化でノイズを減らせれば収束ラウンドが増え精度向上が期待できる。

---

## 4. 具体的な実装計画

### Phase A: αターゲット化（最優先）

#### Step 1: 市場日中リターン特徴量の追加

```python
# src/features/engineer.py の _compute_market_returns に追加
# 市場の日中リターン（当日の等加重平均: open→close）
mkt_open_df  = pd.DataFrame({t: d["open"]  for t, d in price_data.items()})
mkt_close_df = pd.DataFrame({t: d["close"] for t, d in price_data.items()})
mkt_intraday_raw = mkt_close_df / mkt_open_df - 1
clip = 0.20  # ストップ高/安の異常値を除外
mkt_intraday = mkt_intraday_raw.where(mkt_intraday_raw.abs() <= clip).mean(axis=1)
# → mkt_intraday_return として DataFrame に追加
```

各銘柄の特徴量に追加（前日値を特徴量として使用）:
```python
df["mkt_intraday_return"] = mkt_intraday.reindex(df.index).shift(1)
```

#### Step 2: ターゲット変数の変更

```python
# src/features/engineer.py
# 変更前
df["target_intraday_return"] = close.shift(-1) / open.shift(-1) - 1

# 変更後
intraday_return = close.shift(-1) / open.shift(-1) - 1
mkt_intraday_next = mkt_intraday.reindex(df.index).shift(-1)  # 翌日の市場日中リターン
df["target_intraday_alpha"] = intraday_return - mkt_intraday_next
df["target_intraday_return"] = intraday_return  # 参照用に残す
```

#### Step 3: train.py のターゲット選択を変更

```python
# src/models/train.py
if "target_intraday_alpha" in combined.columns:
    y_close = combined["target_intraday_alpha"]
    label_close = "日中αリターン"
```

#### Step 4: predict.py の推薦スコア変更

```python
# src/models/predict.py
# USE_INTRADAY_TARGET=True かつ αモードの場合
# score_close = predicted_intraday_alpha（市場超過リターン予測値）
pred_open  = last_close * (1 + score_open)
pred_close = pred_open  * (1 + score_close)  # αを加えた終値
expected_gain_pct = score_close * 100  # α（市場超過分）
```

#### Step 5: 推薦フィルタの変更

```python
# src/strategy/recommend.py
# αがプラス = 市場より上がると予測 → 推薦対象
filters["min_expected_gain_pct"] = 0.0  # 0%超（市場を上回る予測）
```

#### Step 6: 評価指標の追加

```python
# src/models/evaluate.py
# α方向的中率: 予測αがプラスの銘柄が実際に市場を上回ったか
alpha_dir_acc = np.mean(
    (pred_alpha > 0) == (actual_alpha > 0)
)
```

---

### Phase B: 分類モデル化（Phase A結果を見てから）

Phase Aでα方向的中率が改善しても実用ラインに届かない場合、
回帰（αの絶対値予測）から分類（市場を上回るか否かの2値予測）に変更する。

```python
# LightGBM の objective を変更
params["objective"] = "binary"
params["metric"]    = "binary_logloss"

# ターゲット
y_close = (target_intraday_alpha > 0).astype(int)  # 1: 市場超過, 0: 市場以下
```

メリット:
- 方向的中率を直接最適化できる
- RMSEではなくlog-lossで学習するため予測の確信度が適切に反映される

---

### Phase C: 市場方向予測モデルの追加（中長期）

個別銘柄モデルとは別に、翌日の市場全体方向を予測するモデルを構築する。

入力特徴量: VIX、米国市場指数、ドル円、日経先物、債券金利など純粋なマクロ指標
出力: 翌日の市場日中リターン（回帰 or 分類）
活用方法: 市場下落確率が高い日は推薦数を絞る（ただし閾値フィルタは試行#6で失敗済みなので注意）

---

## 5. 実装上の注意点

### ルックアヘッドバイアスの防止

| 変数 | 使用方法 | ルックアヘッド |
|---|---|---|
| `mkt_intraday_return` (特徴量) | `.shift(1)` = 前日の市場日中リターン | なし ✅ |
| `mkt_intraday_next` (ターゲット) | `.shift(-1)` = 翌日の実績値 | ターゲット用なので許容 ✅ |

### 試行#2との根本的な違い

| 項目 | 試行#2（失敗） | 今回（Phase A） |
|---|---|---|
| model_open ターゲット | target_open_return（市場込み） | target_open_return（変更なし） |
| model_close ターゲット | close_return - 市場（不整合） | **intraday_return - 市場intraday** |
| 推薦スコア | expected_gain = pred_close - pred_open | **predicted_intraday_alpha** |
| 整合性 | ✗ 破綻 | ✅ 一貫 |

### キャッシュの扱い

ターゲット変数を変更するため、`data/processed/` のキャッシュを削除してから実行する必要がある。

---

## 6. 成功の判断基準

| 指標 | 現状 | Phase A目標 |
|---|---|---|
| 日中α方向的中率 | 未計測 | **55%以上** |
| 市場上昇日 推薦勝率 | ~80% | 75%以上を維持 |
| 市場下落日 推薦勝率 | ~10% | **45%以上**（最重要） |
| Sharpe Ratio | 1.618 | 1.5以上を維持 |
| 上昇日/下落日の勝率乖離 | ~70pp | **30pp以内**に圧縮 |

---

## 7. 優先度まとめ

| 優先度 | 施策 | 期待効果 | 実装コスト |
|---|---|---|---|
| ★★★ | αターゲット化（Phase A） | 市場下落日の勝率改善 | 低 |
| ★★★ | 分類モデル化（Phase B） | 方向的中率の直接最適化 | 中 |
| ★★ | 冗長テクニカル指標の削除 | 過学習抑制、速度改善 | 低 |
| ★★ | time-decay weighting | 直近市場環境への適応 | 中 |
| ★ | 市場方向予測モデル（Phase C） | エントリー判断の精度向上 | 高 |
| ★ | 信用倍率・空売り比率 | 需給の先行指標 | 高（データ取得） |
