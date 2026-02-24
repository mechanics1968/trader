#!/usr/bin/env bash
# daily_run.sh — 毎営業日の引け後（15:30 以降）に実行する
#
# cron 設定例（月〜金 15:35 に実行）:
#   35 15 * * 1-5 /Users/inoueakira/work/trader/scripts/daily_run.sh
#
# 実行前に venv を有効化しているか、または絶対パスで python を指定すること。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/results/logs"
DATE="$(date +%Y-%m-%d)"

mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] daily_run.sh 開始" | tee -a "$LOG_DIR/$DATE.log"

cd "$PROJECT_DIR"

# 仮想環境が存在する場合は有効化
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 月曜日は銘柄コード一覧を再取得（週次更新）
DOW="$(date +%u)"  # 1=月, 7=日
if [ "$DOW" -eq 1 ]; then
    TICKERS_FLAG="--tickers-refresh"
else
    TICKERS_FLAG=""
fi

# メイン実行（バックテストは週次で実行）
if [ "$DOW" -eq 5 ]; then  # 金曜日はバックテストも実行
    python main.py $TICKERS_FLAG --backtest 2>&1 | tee -a "$LOG_DIR/$DATE.log"
else
    python main.py $TICKERS_FLAG 2>&1 | tee -a "$LOG_DIR/$DATE.log"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] daily_run.sh 完了" | tee -a "$LOG_DIR/$DATE.log"
