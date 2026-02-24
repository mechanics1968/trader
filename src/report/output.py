"""
推薦結果を CSV と Markdown で出力する。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def save_recommendations(
    recommendations: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    target_date: date | None = None,
) -> Path:
    """
    推薦結果を results/YYYY-MM-DD/ に保存する。

    Parameters
    ----------
    recommendations : pd.DataFrame
        推薦銘柄リスト
    metrics : pd.DataFrame | None
        バックテスト評価指標（省略可）
    target_date : date | None
        対象日付（省略時は今日）

    Returns
    -------
    Path
        出力ディレクトリのパス
    """
    target_date = target_date or date.today()
    out_dir = config.RESULTS_DIR / target_date.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV 保存
    csv_path = out_dir / "recommendations.csv"
    recommendations.to_csv(csv_path, encoding="utf-8-sig")
    logger.info("CSV を保存しました: %s", csv_path)

    # Markdown 保存
    md_path = out_dir / "recommendations.md"
    _save_markdown(recommendations, metrics, md_path, target_date)
    logger.info("Markdown を保存しました: %s", md_path)

    return out_dir


def _save_markdown(
    df: pd.DataFrame,
    metrics: pd.DataFrame | None,
    path: Path,
    target_date: date,
) -> None:
    lines: list[str] = []

    lines.append(f"# デイトレ推薦銘柄 — {target_date.strftime('%Y年%m月%d日')}")
    lines.append("")
    lines.append(f"> 元手: {config.CAPITAL:,} 円 / "
                 f"1銘柄上限: {config.MAX_POSITION_AMOUNT:,} 円 / "
                 f"推薦数: {len(df)} 銘柄")
    lines.append("")

    # 推薦テーブル
    lines.append("## 推薦銘柄一覧（期待上昇率順）")
    lines.append("")

    display_cols = {
        "code": "コード",
        "name": "銘柄名",
        "market": "市場",
        "pred_open": "予測始値",
        "pred_close": "予測終値",
        "expected_gain_pct": "期待上昇率(%)",
        "expected_profit_per_lot": "期待利益/単元(円)",
        "lot_size": "単元株数",
        "required_amount": "必要資金(円)",
        "last_volume": "前日出来高",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    table = df[list(available.keys())].rename(columns=available)

    lines.append(_df_to_md(table))
    lines.append("")

    # バックテスト評価指標
    if metrics is not None and not metrics.empty:
        lines.append("## モデル評価指標（直近バックテスト）")
        lines.append("")
        lines.append(_df_to_md(metrics.round(4)))
        lines.append("")

    # 免責事項
    lines.append("---")
    lines.append(
        "> **免責**: 本推薦は AI による予測結果であり、"
        "投資判断の最終責任はトレーダー本人が負います。"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    """DataFrame を Markdown テーブル文字列に変換する。"""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=True)
    ]
    return "\n".join([header, separator] + rows)


def print_recommendations(recommendations: pd.DataFrame) -> None:
    """推薦結果をターミナルに表示する。"""
    display_cols = {
        "code": "コード",
        "name": "銘柄名",
        "market": "市場",
        "pred_open": "予測始値",
        "pred_close": "予測終値",
        "expected_gain_pct": "期待上昇率(%)",
        "required_amount": "必要資金(円)",
    }
    available = {k: v for k, v in display_cols.items() if k in recommendations.columns}
    table = recommendations[list(available.keys())].rename(columns=available)

    print("\n" + "=" * 70)
    print(f"  デイトレ推薦銘柄 TOP {len(table)}")
    print("=" * 70)
    print(table.to_string())
    print("=" * 70 + "\n")
