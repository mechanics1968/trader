---
name: trader-retrain
description: >
  LightGBMモデルを強制再学習し、再学習後の精度を確認する。
  「再学習して」「モデル更新して」「精度が落ちたので学習し直して」のような
  問いかけに対して自動的に使用する。
allowed-tools: Bash(python *), Bash(grep *), Bash(tail *)
argument-hint: "[--optimize] [--backtest]"
---

以下の手順でモデルを再学習し、結果を日本語で報告してください。

## ステップ1: 事前確認

再学習前に現在のモデルの状態をログから確認する。

```bash
grep -E "(val_dir_acc|バックテスト結果|方向的中率|AUC)" results/2026-*_backtest.log 2>/dev/null | tail -5
```

## ステップ2: 再学習の実行

`$ARGUMENTS` に応じて実行コマンドを選択する。

### 通常の再学習（デフォルト）

```bash
python main.py --retrain --backtest
```

- `--retrain`: 保存済みモデルを無視して強制再学習
- `--backtest`: 再学習後にバックテストを実行して精度を確認

### `--optimize` が指定された場合: ハイパーパラメータ最適化 → 再学習

```bash
python main.py --optimize && python main.py --apply-best --backtest
```

- Optuna で多目的最適化（時間がかかるため注意）
- `--apply-best` で最適パラメータを本番モデルに適用

実行中は進捗をそのまま伝えてください（完了まで数分〜十数分かかる場合があります）。

## ステップ3: 再学習結果の確認

```bash
grep -A 3 "バックテスト結果" results/$(date +%Y-%m-%d)_backtest.log 2>/dev/null | tail -5
grep -E "(学習完了|val_dir_acc|rounds)" results/$(date +%Y-%m-%d).log 2>/dev/null | tail -10
```

## ステップ4: 結果の解釈と報告

### 報告すべき内容

1. **再学習前後の精度比較**

   | 指標 | 再学習前 | 再学習後 | 変化 |
   |------|---------|---------|------|
   | 終値 方向的中率 | X.X% | X.X% | ↑ or ↓ |
   | 日中α AUC | X.XXX | X.XXX | ↑ or ↓ |
   | Sharpe Ratio | X.XX | X.XX | ↑ or ↓ |

2. **再学習の効果判定**

   | 判定 | 条件 | 次のアクション |
   |------|------|--------------|
   | 改善 | 方向的中率 +1%以上 or AUC +0.01以上 | 運用継続。`/trader-recommend` で推薦を更新 |
   | 横ばい | ±1%以内 | 現状維持。1週間後に再評価 |
   | 悪化 | 方向的中率 -1%以上 | 特徴量の見直しが必要。`--optimize` の実行を検討 |

3. **総合コメント**（1〜2文）
   - 再学習の効果を端的に述べ、次のアクションを提示する

### 注意事項

- 再学習は数分〜十数分かかる。完了まで待ってから結果を報告すること
- 本分析はAIによる参考情報であり、最終判断はトレーダー本人が行う
