"""
デイトレーダー支援システム — メインエントリポイント。

使い方:
  python main.py                         # 通常実行（キャッシュ活用）
  python main.py --refresh               # データを強制再取得
  python main.py --retrain               # モデルを強制再学習
  python main.py --backtest              # バックテスト結果も表示
  python main.py --model tft             # TFT モデルで予測
  python main.py --model tft --retrain   # TFT を再学習
  python main.py --optimize              # 多目的ハイパーパラメータ最適化を実行（lgbm 専用）
  python main.py --optimize --n-trials 100 --n-tickers 500
  python main.py --apply-best            # 最適化済みパラメータで本番モデルを再学習（lgbm 専用）
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import config
from src.fetch.tickers import fetch_tickers, get_ticker_list
from src.fetch.prices import fetch_prices
from src.fetch.lot_size import fetch_lot_sizes
from src.features.engineer import build_features_all
from src.models.train import train
from src.models.predict import predict_next_day
from src.models.evaluate import evaluate_predictions
from src.models.optimize import run_optimization, apply_best_params
from src.models.tft_train import train_tft
from src.models.tft_predict import predict_next_day_tft
from src.strategy.recommend import build_recommendations
from src.report.output import save_recommendations, print_recommendations

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            config.RESULTS_DIR / f"{date.today()}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="デイトレーダー支援システム")
    parser.add_argument("--refresh", action="store_true", help="データを強制再取得")
    parser.add_argument("--retrain", action="store_true", help="モデルを強制再学習")
    parser.add_argument("--backtest", action="store_true", help="バックテスト評価を実行")
    parser.add_argument("--optimize", action="store_true", help="多目的ハイパーパラメータ最適化を実行")
    parser.add_argument("--n-trials", type=int, default=50, help="最適化の試行回数（デフォルト: 50）")
    parser.add_argument(
        "--n-tickers", type=int, default=0,
        help="最適化に使う銘柄数（デフォルト: 0=全銘柄、速度重視なら 300 を指定）"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="最適化の並列 Trial 数（デフォルト: 1）。CPU コア数の半分程度を推奨。"
    )
    parser.add_argument("--apply-best", action="store_true",
                        help="最適化済みパラメータで本番モデルを再学習")
    parser.add_argument(
        "--tickers-refresh",
        action="store_true",
        help="銘柄コード一覧を JPX から再取得",
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "tft"],
        default="lgbm",
        help="使用するモデル（デフォルト: lgbm）。--optimize / --apply-best は lgbm 専用。",
    )
    parser.add_argument(
        "--tft-n-tickers", type=int, default=0,
        help="TFT 学習に使う銘柄数（デフォルト: 0=全銘柄）。デバッグ時は 50〜100 を推奨。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== デイトレーダー支援システム 起動 ===")

    # ------------------------------------------------------------------ #
    # Step 1: 銘柄コード取得
    # ------------------------------------------------------------------ #
    logger.info("[Step 1] 東証銘柄コードを取得します")
    ticker_info = fetch_tickers(refresh=args.tickers_refresh)
    tickers = ticker_info["ticker"].tolist()
    logger.info("対象銘柄数: %d", len(tickers))

    # ------------------------------------------------------------------ #
    # Step 2: 株価データ取得（直近1年）
    # ------------------------------------------------------------------ #
    logger.info("[Step 2] Yahoo Finance から株価データを取得します")
    price_data = fetch_prices(tickers, refresh=args.refresh)
    logger.info("取得成功: %d 銘柄", len(price_data))

    # ------------------------------------------------------------------ #
    # Step 3: 特徴量エンジニアリング
    # ------------------------------------------------------------------ #
    logger.info("[Step 3] 特徴量を生成します")
    features = build_features_all(price_data, ticker_info)

    # ------------------------------------------------------------------ #
    # Step 4a: 多目的ハイパーパラメータ最適化（lgbm 専用）
    # ------------------------------------------------------------------ #
    if args.optimize:
        if args.model != "lgbm":
            logger.error("--optimize は lgbm 専用です。--model lgbm を指定してください")
            sys.exit(1)
        logger.info("[Step 4a] 多目的ハイパーパラメータ最適化を実行します")
        n_tickers = args.n_tickers if args.n_tickers > 0 else None
        run_optimization(features, n_trials=args.n_trials, n_tickers=n_tickers, n_jobs=args.n_jobs)
        logger.info("最適化完了。結果: optuna_results/ / 推薦パラメータ: best_params.json")

    # ------------------------------------------------------------------ #
    # Step 4b: 最適化済みパラメータで再学習（lgbm 専用）
    # ------------------------------------------------------------------ #
    if args.apply_best:
        if args.model != "lgbm":
            logger.error("--apply-best は lgbm 専用です。--model lgbm を指定してください")
            sys.exit(1)
        logger.info("[Step 4b] 最適化済みパラメータで本番モデルを再学習します")
        model_open, model_close = apply_best_params(features)
    elif args.model == "tft":
        # ------------------------------------------------------------------ #
        # Step 4 (TFT): TFT モデル学習
        # ------------------------------------------------------------------ #
        logger.info("[Step 4] TFT モデルを学習します")
        tft_features = features
        if args.tft_n_tickers > 0:
            import random as _random
            sampled = _random.sample(list(features.keys()), min(args.tft_n_tickers, len(features)))
            tft_features = {k: features[k] for k in sampled}
            logger.info("TFT デバッグモード: %d / %d 銘柄を使用", len(tft_features), len(features))
        model_open = train_tft(tft_features, "target_open_return", force=args.retrain)
        model_close = train_tft(tft_features, "target_close_return", force=args.retrain)
    else:
        # ------------------------------------------------------------------ #
        # Step 4: LightGBM モデル学習（通常）
        # ------------------------------------------------------------------ #
        logger.info("[Step 4] モデルを学習します")
        model_open, model_close = train(features, force=args.retrain)

    # ------------------------------------------------------------------ #
    # Step 5: 翌日予測
    # ------------------------------------------------------------------ #
    logger.info("[Step 5] 翌日の株価を予測します")
    if args.model == "tft":
        predictions = predict_next_day_tft(features, model_open, model_close)
    else:
        predictions = predict_next_day(features, model_open, model_close)

    # ------------------------------------------------------------------ #
    # Step 6: バックテスト評価（オプション）
    # ------------------------------------------------------------------ #
    metrics = None
    if args.backtest:
        logger.info("[Step 6] バックテスト評価を実行します")
        metrics = evaluate_predictions(features, model_open, model_close)

    # ------------------------------------------------------------------ #
    # Step 7: 単元株数取得
    # ------------------------------------------------------------------ #
    logger.info("[Step 7] 単元株数を取得します")
    lot_sizes = fetch_lot_sizes(tickers, refresh=args.refresh)

    # ------------------------------------------------------------------ #
    # Step 8: 推薦銘柄リスト生成
    # ------------------------------------------------------------------ #
    logger.info("[Step 8] 推薦銘柄リストを生成します")
    recommendations = build_recommendations(predictions, ticker_info, lot_sizes)

    # ------------------------------------------------------------------ #
    # Step 9: 出力
    # ------------------------------------------------------------------ #
    logger.info("[Step 9] 結果を出力します")
    print_recommendations(recommendations)
    out_dir = save_recommendations(recommendations, metrics)
    logger.info("結果を保存しました: %s", out_dir)
    logger.info("=== 完了 ===")


if __name__ == "__main__":
    main()
