"""
プロジェクト全体の設定値。
環境依存の値は .env ファイルで上書きできる。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TICKERS_DIR = DATA_DIR / "tickers"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models_saved"

for _dir in (RAW_DIR, PROCESSED_DIR, TICKERS_DIR, RESULTS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 資金・リスク管理
# ---------------------------------------------------------------------------
CAPITAL = 2_000_000          # 元手（円）
MAX_POSITION_RATIO = 0.20    # 1銘柄への最大投資比率
MAX_POSITION_AMOUNT = int(CAPITAL * MAX_POSITION_RATIO)  # = 400,000 円
STOP_LOSS_RATIO = -0.02      # 損切りライン（始値比 -2%）

# ---------------------------------------------------------------------------
# 対象市場
# ---------------------------------------------------------------------------
TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]

# JPX 上場銘柄一覧（Excel）の公開ページ
JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/"
    "data_j.xls"
)

# ---------------------------------------------------------------------------
# データ取得
# ---------------------------------------------------------------------------
PRICE_PERIOD = "1y"          # Yahoo Finance から取得する期間
FETCH_MAX_WORKERS = 10       # 並列取得のスレッド数（レートリミット対策）
FETCH_SLEEP_SEC = 1.5        # リクエスト間のスリープ（秒）
FETCH_RETRY = 3              # リトライ回数

# ---------------------------------------------------------------------------
# 特徴量エンジニアリング
# ---------------------------------------------------------------------------
SMA_WINDOWS = [5, 20, 60]
EMA_WINDOWS = [12, 26]
RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
LAG_DAYS = [1, 2, 3, 5]     # ラグ特徴量の日数

# ---------------------------------------------------------------------------
# モデル（LightGBM）
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
}
LGBM_NUM_ROUNDS = 1000
LGBM_EARLY_STOPPING = 50

# バリデーション（ウォークフォワード）
VALIDATION_DAYS = 20         # テストセットとして保持する日数
RETRAIN_THRESHOLD = 0.52     # 方向的中率がこれを下回ったら再学習

# ---------------------------------------------------------------------------
# 推薦フィルタ
# ---------------------------------------------------------------------------
MIN_VOLUME = 100_000                         # 最低出来高（株）
MIN_EXPECTED_GAIN_PCT = 0.5                  # 最低期待上昇率（%）
MAX_PRICE_PER_UNIT = MAX_POSITION_AMOUNT     # 1単元の最大購入金額（円）
MAX_DAILY_CHANGE_PCT = 10.0                  # 前日比騰落率の除外上限（%）
TOP_N_RECOMMENDATIONS = 20                  # 推薦銘柄数
# 市場モメンタムフィルタ: 前日の等加重市場リターンがこれ未満なら推薦しない
# -1.0 = 事実上無効（前日市場-100%以下は存在しない）
# ※ 単純な前日リターン閾値はリバウンド日を除外してしまうため逆効果と判明
MARKET_DECLINE_THRESHOLD: float = -1.0

# クロスセクション標準化ターゲット
# True: 日付ごとに全銘柄の alpha を z-score 標準化した target_cs_* を学習ターゲットとして使用
# False: target_alpha_* （残差リターン）をそのまま使用
USE_CS_TARGET: bool = False      # CS z-score は絶対勝率を改善しないため無効化（実験済み）
USE_INTRADAY_TARGET: bool = True  # 終値モデルのターゲットを日中騰落率（始値→終値）に変更

# ---------------------------------------------------------------------------
# モデル（TFT: Temporal Fusion Transformer）
# ---------------------------------------------------------------------------
TFT_ENCODER_LENGTH: int = 60          # エンコーダ参照日数（過去約3ヶ月）
TFT_PREDICTION_LENGTH: int = 1        # 予測ホライズン（翌日固定）
TFT_HIDDEN_SIZE: int = 64             # 隠れ層サイズ
TFT_ATTENTION_HEADS: int = 4          # アテンションヘッド数
TFT_DROPOUT: float = 0.1
TFT_HIDDEN_CONTINUOUS_SIZE: int = 16
TFT_MAX_EPOCHS: int = 30
TFT_BATCH_SIZE: int = 256
TFT_LEARNING_RATE: float = 1e-3
TFT_GRADIENT_CLIP: float = 0.1
TFT_EARLY_STOPPING_PATIENCE: int = 5
TFT_MODEL_OPEN_DIR: Path = MODELS_DIR / "tft_open"
TFT_MODEL_CLOSE_DIR: Path = MODELS_DIR / "tft_close"
# DataLoader の worker 数（0=メインプロセスのみ、>0 で並列ロード）
# MPS 環境では 0 固定。CUDA 環境では 4〜8 推奨。
TFT_NUM_WORKERS: int = int(os.getenv("TFT_NUM_WORKERS", "4"))
# GPU 数に比例して学習率をスケールアップするか（Linear Scaling Rule）
TFT_LR_SCALE_WITH_GPUS: bool = True

# ---------------------------------------------------------------------------
# ログ
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
