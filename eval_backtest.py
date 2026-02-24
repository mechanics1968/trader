"""2/19特徴量 → 2/20予測 vs 実績 バックテスト比較スクリプト"""
import sys
sys.path.insert(0, '.')
import config
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from src.models.train import _load_models
from src.models.predict import predict_next_day

proc_dir = config.DATA_DIR / 'processed'
raw_dir  = config.DATA_DIR / 'raw'
features, raw_prices = {}, {}

for f in proc_dir.glob('*.csv'):
    ticker = f.stem[:-2] + '.T' if f.stem.endswith('_T') else f.stem.replace('_', '.')
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    if len(df) > 0 and str(df.index[-1].date()) == '2026-02-19':
        features[ticker] = df
    rf = raw_dir / f.name
    if rf.exists():
        raw_prices[ticker] = pd.read_csv(rf, index_col=0, parse_dates=True)

print(f'特徴量: {len(features)} 銘柄')
model_open, model_close = _load_models()
predictions = predict_next_day(features, model_open, model_close)

TARGET_DATE = pd.Timestamp('2026-02-20')
actuals = []
for ticker, rdf in raw_prices.items():
    if TARGET_DATE in rdf.index:
        row = rdf.loc[TARGET_DATE]
        actuals.append({
            'ticker': ticker,
            'actual_open': float(row['open']),
            'actual_close': float(row['close']),
        })
actual_df = pd.DataFrame(actuals)

merged = predictions.merge(actual_df, on='ticker', how='inner')
merged['actual_open_return_pct']  = (merged['actual_open']  / merged['last_close'] - 1) * 100
merged['actual_close_return_pct'] = (merged['actual_close'] / merged['last_close'] - 1) * 100
merged['actual_gain_pct'] = (
    (merged['actual_close'] - merged['actual_open']) / merged['actual_open'] * 100
)

open_acc  = (np.sign(merged['pred_open_return_pct'])  == np.sign(merged['actual_open_return_pct'])).mean()
close_acc = (np.sign(merged['pred_close_return_pct']) == np.sign(merged['actual_close_return_pct'])).mean()
eg_acc    = (np.sign(merged['expected_gain_pct'])     == np.sign(merged['actual_gain_pct'])).mean()
rmse_open  = np.sqrt(((merged['pred_open_return_pct']  - merged['actual_open_return_pct'])**2).mean())
rmse_close = np.sqrt(((merged['pred_close_return_pct'] - merged['actual_close_return_pct'])**2).mean())
close_std  = merged['pred_close_return_pct'].std()

rec = merged[merged['expected_gain_pct'] >= config.MIN_EXPECTED_GAIN_PCT].copy()
wins = int((rec['actual_gain_pct'] > 0).sum())

print()
print('==== 2/19->2/20 予測精度比較 ====')
print(f'{"指標":<24} {"旧モデル":>10} {"②のみ":>10} {"②+セクター":>12}')
print('-' * 60)
print(f'{"始値 方向的中率":<24} {"24.1%":>10} {"24.2%":>10} {open_acc:>11.1%}')
print(f'{"終値 方向的中率":<24} {"24.5%":>10} {"25.1%":>10} {close_acc:>11.1%}')
print(f'{"期待上昇 的中率":<24} {"50.3%":>10} {"49.9%":>10} {eg_acc:>11.1%}')
print(f'{"始値 RMSE":<24} {"1.4162%":>10} {"1.4020%":>10} {rmse_open:>10.4f}%')
print(f'{"終値 RMSE":<24} {"2.4876%":>10} {"2.4630%":>10} {rmse_close:>10.4f}%')
print(f'{"推薦銘柄数":<24} {"7":>10} {"9":>10} {len(rec):>11}')
if len(rec) > 0:
    print(f'{"推薦勝率":<24} {"3/7=43%":>10} {"6/9=67%":>10} {wins}/{len(rec)}={wins/len(rec):>6.0%}')
print(f'{"close予測 std":<24} {"0.026%":>10} {"0.093%":>10} {close_std:>10.4f}%')
print(f'{"終値モデル rounds":<24} {"5":>10} {"5":>10} {"161":>11}')

print()
print(f'--- 推薦銘柄詳細 ({len(rec)} 銘柄) ---')
if not rec.empty:
    for _, row in rec.iterrows():
        r = '✓ 利益' if row['actual_gain_pct'] > 0 else '✗ 損失'
        print(f"  {row['ticker']:8s}  期待{row['expected_gain_pct']:+.2f}%  実際{row['actual_gain_pct']:+.2f}%  {r}")
