# Claude Code Skills 実装レポート

**作成日**: 2026年3月2日
**対象システム**: 東証デイトレーダー支援システム
**スキル数**: 4

---

## 概要

Claude Code の Skills 機能を活用し、デイトレ支援システムの主要操作をスラッシュコマンドとして実装した。
スキル化により、以下の効果が得られた。

- **操作の簡略化**: `python main.py --backtest` 等の長いコマンドを `/trader-review` 一言で実行できる
- **解釈の自動化**: 数値の羅列（AUC, Sharpe等）をAIが自然言語で解釈し、「運用継続 / 要監視 / 再学習推奨」の判断を提示する
- **連携フローの明示**: 各スキルが次のアクションを示すことで、毎朝の運用手順が標準化される

---

## スキル一覧

### 1. `/trader-sentiment`

**用途**: 翌営業日の市場センチメント分析
**実装日**: 2026年3月2日
**コミット**: `7ef4f7a`

#### ファイル構成

| ファイル | 役割 |
|---------|------|
| `.claude/skills/trader-sentiment/SKILL.md` | スキル定義・解釈指示 |
| `scripts/run_sentiment.py` | センチメント分析の単独実行スクリプト |
| `src/fetch/llm_sentiment.py` | Claude CLIによるLLMセンチメント取得モジュール |

#### 動作フロー

1. `yfinance` から日経225・S&P500・VIX・USD/JPYのニュースヘッドラインを取得
2. 市場数値（VIX水準・為替・前日リターン等）と合わせてプロンプトを構築
3. Claude CLIを呼び出し、市場センチメントスコア（-1.0〜+1.0）・確信度・リスク銘柄・判断理由をJSON形式で取得
4. 結果をスコア段階別（強気/やや強気/中立/やや弱気/強い弱気）に解釈して提示

#### 出力例（2026年3月2日）

```
センチメント : -0.75  弱気
確信度       : 0.82
判断理由     : 米イスラエルによるイラン攻撃でホルムズ海峡リスク高まり、
               日経先物-2%・原油+8〜13%急騰。エネルギー輸入依存の日本株全般に下押し圧力。
リスク銘柄   : 5020(ENEOS), 5019(出光), 9202(ANA), 9201(JAL) 他
```

#### 引数

```
/trader-sentiment                    # 引数なし（全マクロ指標を自動取得）
/trader-sentiment --sp500 --vix      # 特定指標を重点分析
/trader-sentiment --top-tickers 9513,3401,4208   # 推薦候補銘柄のリスク判定
```

---

### 2. `/trader-recommend`

**用途**: 翌営業日の推薦銘柄表示・解釈
**実装日**: 2026年3月2日
**コミット**: `2cd2d63`

#### ファイル構成

| ファイル | 役割 |
|---------|------|
| `.claude/skills/trader-recommend/SKILL.md` | スキル定義・解釈指示 |
| `scripts/show_recommend.py` | キャッシュ読み込み・フィルタリングスクリプト |

#### 動作フロー

1. `results/YYYY-MM-DD/recommendations.md` の存在を確認（キャッシュ優先）
2. キャッシュがあれば `scripts/show_recommend.py` で即座に読み込み（高速）
3. キャッシュがない or `--refresh`/`--retrain` 指定時は `python main.py` を再実行
4. ★レーティング分布・注目銘柄・ポートフォリオ案・取引アドバイスを提示

#### 引数

```
/trader-recommend                    # キャッシュを読んで即時表示
/trader-recommend --refresh          # データ再取得 → 再予測
/trader-recommend --retrain          # モデル再学習 → 再予測
/trader-recommend --min-rating 4     # ★4以上に絞り込み
/trader-recommend --max-amount 200000  # 必要資金20万円以下に絞り込み
```

#### 推薦フィルタリング機能（`show_recommend.py`）

- `--min-rating N`: `market_rating_label` 列の★の数でフィルタ
- `--max-amount 円`: `required_amount` 列で上限フィルタ
- フィルタなし時はMarkdownをそのまま表示

---

### 3. `/trader-review`

**用途**: バックテスト・ウォークフォワード結果の精度確認・運用判断
**実装日**: 2026年3月2日
**コミット**: `c4738ce`

#### ファイル構成

| ファイル | 役割 |
|---------|------|
| `.claude/skills/trader-review/SKILL.md` | スキル定義・解釈指示 |

#### 動作フロー

1. `results/YYYY-MM-DD_backtest.log` / `_walkforward.log` の存在を確認
2. キャッシュがあればログを `grep` で抽出（再実行なし）
3. `--backtest` 指定時は `python main.py --backtest` を再実行
4. 精度指標を評価基準と照合し「運用継続 / 要監視 / 再学習推奨」を判定

#### 評価基準

| 指標 | 良好 | 許容範囲 | 要注意 |
|------|------|---------|-------|
| 終値 方向的中率 | ≥ 55% | 52〜55% | < 52% |
| 日中α AUC | ≥ 0.55 | 0.52〜0.55 | < 0.52 |
| 推薦銘柄勝率（WF） | ≥ 60% | 45〜60% | < 45% |

#### 引数

```
/trader-review               # キャッシュからログを読んで評価
/trader-review --backtest    # バックテストを再実行してから評価
/trader-review --walkforward # ウォークフォワード結果を重点確認
```

---

### 4. `/trader-retrain`

**用途**: LightGBMモデルの強制再学習・精度回復
**実装日**: 2026年3月2日
**コミット**: `e0eca2b`

#### ファイル構成

| ファイル | 役割 |
|---------|------|
| `.claude/skills/trader-retrain/SKILL.md` | スキル定義・実行指示 |

#### 動作フロー

1. 再学習前の精度をログから確認
2. `python main.py --retrain --backtest` を実行（保存済みモデルを無視して再学習）
3. 再学習後のバックテスト結果を取得
4. 再学習前後の精度を比較し「改善 / 横ばい / 悪化」を判定

#### 引数

```
/trader-retrain              # 通常の再学習（--retrain --backtest）
/trader-retrain --optimize   # Optuna最適化 → apply-best → backtest（長時間）
/trader-retrain --backtest   # バックテストのみ再実行（再学習なし）
```

#### 効果判定基準

| 判定 | 条件 | 次のアクション |
|------|------|--------------|
| 改善 | 方向的中率 +1%以上 or AUC +0.01以上 | `/trader-recommend` で推薦を更新 |
| 横ばい | ±1%以内 | 1週間後に再評価 |
| 悪化 | 方向的中率 -1%以上 | `--optimize` でハイパーパラメータ最適化を実行 |

---

## スキル連携フロー

毎朝の標準的な運用手順と、問題発生時の対処フローを以下に示す。

### 通常の朝の運用手順

```
① /trader-sentiment          # 市場の方向性を確認（所要: 約30秒）
② /trader-recommend          # 推薦銘柄を取得・解釈（所要: 即時〜数秒）
```

### 精度が落ちていると感じた場合

```
① /trader-review             # 精度指標を確認
   ↓ 方向的中率 < 52% / AUC < 0.52 の場合
② /trader-retrain            # 再学習（所要: 数分〜十数分）
   ↓ 改善しない場合
③ /trader-retrain --optimize # Optuna最適化（所要: 数十分〜数時間）
   ↓ 精度回復後
④ /trader-recommend          # 更新済みモデルで推薦を出す
```

### 週次メンテナンス（推奨: 週1回）

```
① /trader-review --walkforward   # 直近50日のトレンドを確認
② /trader-retrain                # 定期再学習（新しいデータを取り込む）
③ /trader-recommend              # 翌週分の推薦を更新
```

---

## 技術仕様

### スキルの共通設定

| 項目 | 値 |
|------|---|
| Skills ディレクトリ | `.claude/skills/` |
| 定義ファイル | `SKILL.md`（各スキルディレクトリ内） |
| 実行権限 | `allowed-tools` で `Bash(python *)`, `Read` 等を明示 |

### ヘルパースクリプト

| スクリプト | 用途 | 引数 |
|-----------|------|------|
| `scripts/run_sentiment.py` | センチメント分析の単独実行 | `--sp500`, `--vix`, `--usdjpy`, `--top-tickers` |
| `scripts/show_recommend.py` | 最新推薦キャッシュの読み込み | `--min-rating`, `--max-amount` |

### 制約・注意事項

- `trader-sentiment` はClaude CLIをサブプロセスとして呼び出す。Claude Code内から実行する場合は `CLAUDECODE` 環境変数を unset して実行（ネスト制限を回避）
- `trader-retrain --optimize` はOptunaによる多目的最適化のため数十分〜数時間かかる
- キャッシュは当日分のみ有効。翌日以降は自動的に再実行が必要

---

## 今後の拡張候補

| スキル名 | 内容 | 優先度 |
|---------|------|-------|
| `/trader-watchlist` | 必要資金・★レーティングでインタラクティブに絞り込む | 中 |
| `/trader-walkforward` | ウォークフォワードバックテストの単独実行 | 低 |

---

> 本レポートは2026年3月2日時点の実装状況をまとめたものです。
