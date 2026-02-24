# AI株価予測トレンドとの比較分析

**作成日**: 2026-02-24

---

## 1. 今回開発したシステムの位置付け

### 世界標準との比較表

| 観点 | 世界のトレンド（2024-25年） | 今回の実装 | 評価 |
|---|---|---|---|
| **モデル** | iTransformer, PatchTST, GNN, LLM統合 | LightGBM（Phase 1） | 妥当（後述） |
| **特徴量** | 代替データ・ニュースセンチメント・GNNによる銘柄関係 | テクニカル指標 + 市場/セクター相対値 | 標準的 |
| **ターゲット** | 残差リターン（市場ベータ除去）がトレンド | 絶対リターン（close_return） | **遅れている** |
| **バックテスト** | Walk-forward（ローリング型）が標準 | Walk-forward 50日 | **業界標準を満たす** |
| **勝率水準** | 55%が実用ライン、60%以上が優秀 | **47%**（目標未達） | 要改善 |
| **日本株対応** | PFN が TOPIX500 で TimesFM ファインチューニング（最先端） | 東証全銘柄対象・セクター対応 | 先行事例と同等の対象範囲 |

---

## 2. 優位性・独自性

### ✅ 強み

#### （a）東証全銘柄（3,776 銘柄）のグローバルモデル
- 多くの研究は少数銘柄（50〜500）を対象。全銘柄を1モデルで扱うアプローチは実用的かつ先進的
- クロスセクション情報（「この銘柄が他と比べてどう動くか」）が学習データに暗黙的に含まれる

#### （b）業界標準のウォークフォワード・バックテスト
- 情報漏洩なし、毎回再学習の厳密な実装
- 実際の研究でも「データリーク防止の困難さ」が指摘されており、正しく実装できているのは強み

#### （c）セクター相対リターン・市場ベータ特徴量
- `beta_20d`・`alpha_1d`・`sector_return`・`mkt_return` を特徴量として実装済み
- ベータ中立化を「特徴量」として取り込む方向性は正しく、最新研究の FactorGCL（2025年2月）とも方向性が一致

---

## 3. 最大の弱点と世界の解法：市場ベータ依存問題

### なぜ今回の方法では解決できなかったか

今回の実装では市場ベータを「**特徴量**」として入力したが、ターゲット変数は依然として**絶対リターン**のまま。
モデルは特徴量としてベータを認識していても、「上昇市場では全銘柄が上がる」というパターンを学習してしまう。

```
現在の構造:
  特徴量: [テクニカル指標, beta_20d, alpha_1d, mkt_return_1d, ...]
  ターゲット: target_close_return（絶対リターン）
  ↓
  モデルが学習するもの: 「市場が上がる日は予測も上がる」
```

### 世界の解法：残差リターン（アルファ）をターゲットにする

最先端研究（FactorGCL, arXiv 2502.05218, 2025年2月）が実践しているアプローチ：

```python
# ターゲットをアルファ成分に変更（open/close 両モデルで統一）
target_alpha_open  = target_open_return  - beta_20d * mkt_return_1d.shift(-1)
target_alpha_close = target_close_return - beta_20d * mkt_return_1d.shift(-1)
```

これにより：
- モデルは「市場より何%多く動くか」を学習する
- 市場下落日でも「市場平均より上回った銘柄」を推薦できる
- 評価指標も「市場超過リターンがプラスだった割合」に変更する必要がある

> **注意**: 試行 #2「市場超過リターンターゲット」で失敗したのは `target_open_return`（市場込み）との不整合が原因。
> **open/close 両モデルを残差リターンに統一する**ことが必須条件。

### ポートフォリオ構築段階での対処（ロング・ショート前提）

CAIA（2024）等の研究では「ベータ中立ポートフォリオ」として：
```
ロング銘柄のβ合計 ≈ ショート銘柄のβ合計
```
を満たすよう調整するが、本プロジェクトはロングオンリー・当日精算のため直接適用は難しい。

### クロスセクショナル・ランキング予測

「各銘柄の絶対リターンを予測」するのではなく「**全銘柄の中での相対順位**を予測」するアプローチ：

```python
# ターゲットをクロスセクション標準化（その日の全銘柄内での偏差値）
target = (return_1d - return_1d.mean()) / return_1d.std()
```

市場全体が下落する日でも「相対的に下がりにくい上位銘柄」を選べる。
デイトレードのロングオンリーでも有効な可能性が高い。

---

## 4. 勝率 47% の世界的な位置付け

| 水準 | 推薦勝率 | 評価 |
|---|:---:|---|
| ランダム | 50% | ベースライン（ランダム選択） |
| 実用ライン | 55% | 実際の取引に耐えうる最低水準 |
| 優秀 | 60%+ | 研究論文で「有意な予測力あり」と見なされる水準 |
| **今回** | **47%** | **ランダム以下（市場依存性が足を引っ張る）** |

今回の 47% は「上昇日 ~70%・下落日 ~20%」という構造であり、市場方向を除けば個別銘柄の予測は機能している可能性がある。
ランダムでは「市場下落日でも 50% 勝てる」はずなので、47% が出るのは市場上昇バイアスが混入しているから。

---

## 5. 世界のトレンド詳細

### モデルアーキテクチャ（2024-2025年）

| モデル | 特徴 | 参考 |
|---|---|---|
| **iTransformer**（ICLR 2024） | 時間軸ではなく変数軸でAttention → 多銘柄・多特徴量に適合 | [解説記事](https://www.datasciencewithmarco.com/blog/itransformer-the-latest-breakthrough-in-time-series-forecasting) |
| **PatchTST**（ICLR 2023） | 時系列をパッチに分割してBERTライク処理 | [GitHub](https://github.com/yuqinie98/PatchTST) |
| **FactorGCL**（2025年2月） | beta→hidden beta→alpha の3段階残差抽出 | [arXiv 2502.05218](https://arxiv.org/html/2502.05218v1) |
| **LLM統合** | ニュース・決算テキストをLLMで処理しシグナル化 | [論文](https://ojs.bonviewpress.com/index.php/FSI/article/view/5703) |
| **TimesFM（PFN事例）** | TOPIX500でファインチューニング済み | [PFN Tech Blog](https://tech.preferred.jp/en/blog/timesfm/) |

### LightGBM vs Transformer 現時点の結論

- **LightGBM**: 約250日（1年分）の学習データではTransformerより安定して高精度。M5コンペ優勝の実績
- **Transformer**: データ量が多い（数年・多銘柄）場合に真価を発揮
- **現実的結論**: データ量が限られたPhase 1ではLightGBMが最堅実な選択
- 出典: [AAAI 2023 "Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504)

### 日本株特有の先行事例

- **PFN（Preferred Networks）**: TOPIX500 + S&P500 の日次・時間足データ計1億時点でTimesFMをファインチューニング
  → 生の事前学習モデルでは不満足だったが、ファインチューニング後に大幅改善
  → [PFN Tech Blog](https://tech.preferred.jp/en/blog/timesfm/)
- **ExtractAlpha Japan News Signal**（2024年4月）: 日経Flash Newsベースの日本株専用NLPシグナル
  → [発表記事](https://extractalpha.com/2024/04/16/extractalpha-introduces-japan-news-signal-a-breakthrough-in-nlp-based-stock-prediction/)

---

## 6. 現実的な次のステップ（研究トレンドを踏まえて）

| 優先度 | 施策 | 根拠 |
|:---:|---|---|
| **最高** | **残差リターンをターゲットに（open/close 両モデル統一）** | FactorGCL等の最新研究・市場ベータ問題の本質的解法 |
| 高 | **クロスセクション・ランキングスコアでの推薦** | 市場方向に依存しない相対的な強弱を選ぶ |
| 中 | **Phase 2: iTransformer への移行** | 多変量・多銘柄の依存関係を直接モデル化 |
| 中 | **日経先物（前場寄り前）を特徴量追加** | 当日の市場方向をリアルタイムで組み込む（最もシンプルな対処） |
| 低 | **TimesFM / Chronos のファインチューニング** | PFNの事例あり。データ量不足時の汎化向上 |

---

## 主要参考文献

- [FactorGCL: Hypergraph-Based Factor Model (arXiv 2502.05218)](https://arxiv.org/html/2502.05218v1)
- [Are Transformers Effective for Time Series Forecasting? (AAAI 2023)](https://arxiv.org/abs/2205.13504)
- [iTransformer (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf)
- [TimesFM for Stock Price Prediction - PFN Tech Blog](https://tech.preferred.jp/en/blog/timesfm/)
- [Equity-Market-Neutral Strategy using LSTM (MDPI)](https://www.mdpi.com/2227-7072/11/2/57)
- [Demystifying Equity Market Neutral Investing (CAIA 2024)](https://caia.org/blog/2024/03/17/demystifying-equity-market-neutral-investing)
- [An Explainable Walk-Forward Framework (SSRN)](https://papers.ssrn.com/sol3/Delivery.cfm/5351507.pdf?abstractid=5351507&mirid=1)
- [ExtractAlpha Japan News Signal 2024](https://extractalpha.com/2024/04/16/extractalpha-introduces-japan-news-signal-a-breakthrough-in-nlp-based-stock-prediction/)

---

*本分析は 2026-02-24 時点の調査に基づきます。*
