"""
最新の推薦銘柄ファイルを表示するスクリプト。

使い方:
  python scripts/show_recommend.py
  python scripts/show_recommend.py --min-rating 3
  python scripts/show_recommend.py --max-amount 200000
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main() -> None:
    parser = argparse.ArgumentParser(description="最新の推薦銘柄を表示")
    parser.add_argument("--min-rating", type=int, default=None, help="★最低レーティング（1〜5）")
    parser.add_argument("--max-amount", type=int, default=None, help="必要資金の上限（円）")
    args = parser.parse_args()

    # 最新の recommendations.md を探す
    results_dir = config.RESULTS_DIR
    today_str = date.today().strftime("%Y-%m-%d")

    # 本日分を優先、なければ最新日付
    candidate = results_dir / today_str / "recommendations.md"
    if not candidate.exists():
        dirs = sorted(
            [d for d in results_dir.iterdir() if d.is_dir() and (d / "recommendations.md").exists()],
            reverse=True,
        )
        if not dirs:
            print("推薦ファイルが見つかりません。`python main.py` を実行してください。")
            sys.exit(1)
        candidate = dirs[0] / "recommendations.md"

    print(f"読み込み: {candidate}")
    print()

    # フィルタなしの場合はそのまま表示
    if args.min_rating is None and args.max_amount is None:
        print(candidate.read_text(encoding="utf-8"))
        return

    # フィルタありの場合はCSVを読んでフィルタ
    csv_path = candidate.with_suffix(".csv")
    if not csv_path.exists():
        print("CSVファイルが見つからないためフィルタ適用不可。Markdownをそのまま表示します。")
        print(candidate.read_text(encoding="utf-8"))
        return

    import pandas as pd
    df = pd.read_csv(csv_path)

    if args.min_rating is not None and "market_rating_label" in df.columns:
        before = len(df)
        df = df[df["market_rating_label"].str.count("★") >= args.min_rating]
        print(f"★{args.min_rating}以上でフィルタ: {before} → {len(df)} 銘柄")

    if args.max_amount is not None and "required_amount" in df.columns:
        before = len(df)
        df = df[df["required_amount"] <= args.max_amount]
        print(f"必要資金 {args.max_amount:,}円以下でフィルタ: {before} → {len(df)} 銘柄")

    print()
    if df.empty:
        print("条件に合う銘柄がありませんでした。")
        return

    # 表示列を絞る
    show_cols = [c for c in [
        "code", "name", "market_rating_label", "close_up_prob",
        "expected_gain_pct", "expected_profit_per_lot", "required_amount",
    ] if c in df.columns]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
