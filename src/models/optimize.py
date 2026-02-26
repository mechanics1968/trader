"""
多目的ハイパーパラメータ最適化（Optuna NSGA-II）。

目的関数（すべて最小化）:
  1. rmse_open               : 翌日始値変化率の RMSE
  2. rmse_close              : 翌日終値変化率の RMSE
  3. 1_minus_dir_acc_open    : 始値方向的中率の補数（= 1 - 的中率）
  4. 1_minus_dir_acc_close   : 終値方向的中率の補数
  5. neg_n_recommendations   : 推薦通過銘柄数の負値（最小化 = 推薦数を最大化）

探索空間:
  LightGBM モデルパラメータ（学習率・木の複雑さ・正則化など）

実行例:
  python -m src.models.optimize --n-trials 50 --n-tickers 300
  python -m src.models.optimize --n-trials 300 --n-tickers 300 --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
from pathlib import Path

import optuna
import pandas as pd

import config
from src.features.engineer import load_processed
from src.models.train import train, _save_models
from src.models.metrics import compute_errors, PredictionErrors
from src.models.predict import predict_next_day

logger = logging.getLogger(__name__)

# 最適化結果の保存先
OPTUNA_RESULTS_DIR = config.BASE_DIR / "optuna_results"
BEST_PARAMS_PATH = config.BASE_DIR / "best_params.json"

# 最適化中の学習設定
# Early Stopping を無効化して固定ラウンド数で評価する。
# 理由: αターゲットは RMSE が round 1〜5 でピークを迎えた後に方向的中率が round 30 で
#        ピークを迎える（RMSE と方向的中率が相反する）。Early Stopping を使うと
#        全 Trial が「null モデル」で止まり、ハイパーパラメータの差が検出できない。
OPT_NUM_ROUNDS = 50    # 固定ラウンド数（方向的中率がピークに近い範囲）
OPT_EARLY_STOPPING = None  # None = Early Stopping 無効


# ---------------------------------------------------------------------------
# 探索空間の定義
# ---------------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial, lgbm_n_jobs: int = 1) -> dict:
    """Optuna の Trial からハイパーパラメータをサンプリングする。"""
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "n_jobs": lgbm_n_jobs,
        # --- 学習率 ---
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # --- 木の複雑さ ---
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        # 注意: αターゲットは分散が小さく min_gain_to_split >= 0.1 で split が一切
        # 起きなくなる（定数予測 = null model）。対数スケールで小さな値を探索する。
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 0.05, log=True),
        # --- サンプリング ---
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        # --- 正則化 ---
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# 推薦数カウント
# ---------------------------------------------------------------------------

def _count_recommendations(
    features: dict[str, pd.DataFrame],
    model_open,
    model_close,
) -> int:
    """
    現在の特徴量に対して MIN_EXPECTED_GAIN_PCT フィルタを通過する銘柄数を返す。

    predict_next_day は各銘柄の最終行を使って翌日を予測する。
    推薦フィルタのうちモデルの予測値に依存する expected_gain_pct のみを適用する
    （出来高・騰落率は銘柄固有のデータであり、モデルパラメータに依存しない）。
    """
    try:
        preds = predict_next_day(features, model_open, model_close)
        n_recs = int(
            (preds["expected_gain_pct"] >= config.MIN_EXPECTED_GAIN_PCT).sum()
        )
        return n_recs
    except Exception as exc:
        logger.debug("推薦数カウント中にエラー: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# 目的関数
# ---------------------------------------------------------------------------

def _objective(
    trial: optuna.Trial,
    features: dict[str, pd.DataFrame],
    lgbm_n_jobs: int = 1,
) -> tuple[float, float, float, float, float]:
    """
    1 Trial の評価。

    1. ハイパーパラメータをサンプリング
    2. モデルを学習（保存しない）
    3. バリデーション期間の誤差を計算
    4. 推薦フィルタを通過する銘柄数を計算

    Returns
    -------
    tuple[float, float, float, float, float]
        (rmse_open, rmse_close, 1-dir_acc_open, 1-dir_acc_close, neg_n_recommendations)
        neg_n_recommendations: 推薦数の負値（最小化 = 推薦数を最大化）
    """
    params = _suggest_params(trial, lgbm_n_jobs=lgbm_n_jobs)

    try:
        model_open, model_close = train(
            features,
            force=True,
            params=params,
            num_rounds=OPT_NUM_ROUNDS,
            early_stopping=OPT_EARLY_STOPPING,
            save=False,
        )
        errors = compute_errors(features, model_open, model_close)

        # NaN が出た場合は最悪値を返して枝刈り
        if any(v != v for v in errors.objectives.values()):  # nan check
            raise optuna.TrialPruned()

        n_recs = _count_recommendations(features, model_open, model_close)

        objs = errors.objectives
        return (
            objs["rmse_open"],
            objs["rmse_close"],
            objs["1_minus_dir_acc_open"],
            objs["1_minus_dir_acc_close"],
            -float(n_recs),  # 負値 = 推薦数最大化
        )

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        logger.warning("Trial %d 失敗: %s", trial.number, exc)
        raise optuna.TrialPruned()


# ---------------------------------------------------------------------------
# 並列ワーカー（spawn プロセスから呼び出される）
# ---------------------------------------------------------------------------

def _worker(
    study_name: str,
    storage: str,
    opt_features: dict,
    lgbm_n_jobs: int,
    n_trials: int,
) -> None:
    """
    spawn された子プロセス内で study に接続し、n_trials 回の最適化を実行する。
    fork + OpenMP segfault を回避するため、このプロセスは spawn で起動される。
    """
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _study = _optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize", "minimize", "minimize"],
        sampler=_optuna.samplers.NSGAIISampler(seed=42),
    )
    _study.optimize(
        lambda trial: _objective(trial, opt_features, lgbm_n_jobs=lgbm_n_jobs),
        n_trials=n_trials,
        n_jobs=1,
        catch=(Exception,),  # 子プロセス内の例外を FAIL 状態にして継続
    )


# ---------------------------------------------------------------------------
# 最適化の実行
# ---------------------------------------------------------------------------

def run_optimization(
    features: dict[str, pd.DataFrame],
    n_trials: int = 50,
    n_tickers: int | None = None,
    study_name: str = "trader_lgbm",
    storage: str | None = None,
    n_jobs: int = 1,
) -> optuna.Study:
    """
    多目的ハイパーパラメータ最適化を実行する。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}（全銘柄）
    n_trials : int
        試行回数
    n_tickers : int | None
        最適化に使う銘柄数（None = 全銘柄）。
        全銘柄だと1 Trial が重いため、速度重視なら 300〜500 を推奨。
    study_name : str
        Optuna Study の名前（ストレージを使う場合に識別子になる）
    storage : str | None
        Optuna のストレージ URL（例: "sqlite:///optuna.db"）。
        None の場合はインメモリで実行（結果は揮発する）。
    n_jobs : int
        Optuna の並列 Trial 数。各 Trial の LightGBM スレッド数を自動調整する。
        cpu_count // n_jobs コアを LightGBM に割り当てる（デフォルト: 1）。

    Returns
    -------
    optuna.Study
        完了した Study オブジェクト
    """
    OPTUNA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 銘柄数を絞る（速度重視モード）
    if n_tickers is not None and n_tickers < len(features):
        sampled_keys = random.sample(list(features.keys()), n_tickers)
        opt_features = {k: features[k] for k in sampled_keys}
        logger.info("最適化用に %d / %d 銘柄をサンプリングします", n_tickers, len(features))
    else:
        opt_features = features

    # 並列 Trial 数に応じて LightGBM の per-Trial スレッド数を決定
    # 例: 8コア / n_jobs=4 → 各 Trial に 2 スレッド → 合計 8 コアをフル活用
    cpu_count = os.cpu_count() or 4
    lgbm_n_jobs = max(1, cpu_count // max(1, n_jobs))

    # fork + OpenMP の競合による segfault 対策
    # LightGBM が内部で使う OpenMP スレッド数を明示的に制限する
    os.environ["OMP_NUM_THREADS"] = str(lgbm_n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(lgbm_n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(lgbm_n_jobs)
    logger.info(
        "並列 Trial: %d / CPU: %d コア / LightGBM per-Trial スレッド: %d",
        n_jobs, cpu_count, lgbm_n_jobs,
    )

    # n_jobs > 1 の場合はストレージが必須（NSGAIISampler の並列安全性のため）
    if n_jobs > 1 and storage is None:
        # spawn プロセスでも確実に書き込める /tmp を使用
        db_path = "/tmp/trader_optuna.db"
        storage = f"sqlite:///{db_path}"
        logger.warning(
            "n_jobs > 1 には永続ストレージが必要です。自動的に SQLite を使用します: %s", storage
        )

    # Study の作成
    sampler = optuna.samplers.NSGAIISampler(seed=42)
    directions = ["minimize", "minimize", "minimize", "minimize", "minimize"]

    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        storage=storage,
        load_if_exists=(storage is not None),
    )

    objective_names = [
        "rmse_open",
        "rmse_close",
        "1_minus_dir_acc_open",
        "1_minus_dir_acc_close",
        "neg_n_recommendations",  # 負値（小さいほど推薦数が多い）
    ]
    study.set_metric_names(objective_names)

    logger.info(
        "多目的最適化を開始します（%d trials / %d 銘柄 / NSGA-II / n_jobs=%d）",
        n_trials, len(opt_features), n_jobs,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if n_jobs <= 1:
        # シングルプロセス: そのまま実行
        study.optimize(
            lambda trial: _objective(trial, opt_features, lgbm_n_jobs=lgbm_n_jobs),
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1,
        )
    else:
        # マルチプロセス: spawn した子プロセスを n_jobs 個起動し、
        # 各プロセスが同じ SQLite study に n_jobs=1 で接続して並列実行する。
        # （fork + OpenMP の組み合わせによる segfault を回避するため spawn を使用）
        logger.info(
            "並列モード: %d プロセスを spawn して最適化します（study: %s）",
            n_jobs, storage,
        )
        ctx = multiprocessing.get_context("spawn")
        procs = []
        for _ in range(n_jobs):
            p = ctx.Process(
                target=_worker,
                args=(study_name, storage, opt_features, lgbm_n_jobs, n_trials),
                daemon=True,
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        # join 後に study を再ロードして最新状態を取得
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
        )

    # 結果の保存と表示
    _save_results(study, objective_names)
    _log_pareto_front(study, objective_names)

    return study


# ---------------------------------------------------------------------------
# 結果の保存・表示
# ---------------------------------------------------------------------------

def _save_results(study: optuna.Study, objective_names: list[str]) -> None:
    """全 Trial の結果と Pareto 最前線を CSV に保存する。"""
    # 全 Trial
    all_trials_path = OPTUNA_RESULTS_DIR / "all_trials.csv"
    trials_df = study.trials_dataframe()
    trials_df.to_csv(all_trials_path, index=False)
    logger.info("全 Trial 結果を保存しました: %s", all_trials_path)

    # Pareto 最前線
    pareto = _build_pareto_df(study, objective_names)
    pareto_path = OPTUNA_RESULTS_DIR / "pareto_front.csv"
    pareto.to_csv(pareto_path, index=False)
    logger.info("Pareto 最前線を保存しました: %s (%d 件)", pareto_path, len(pareto))

    # 推薦パラメータ（rmse_open + rmse_close の合計が最小の1点）
    best = _select_best_params(pareto, objective_names)
    if best is not None:
        BEST_PARAMS_PATH.write_text(
            json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("推薦パラメータを保存しました: %s", BEST_PARAMS_PATH)


def _build_pareto_df(
    study: optuna.Study, objective_names: list[str]
) -> pd.DataFrame:
    """Pareto 最前線の Trial を DataFrame に変換する。"""
    rows = []
    for t in study.best_trials:
        row = {f"obj_{name}": v for name, v in zip(objective_names, t.values)}
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row["trial_number"] = t.number
        rows.append(row)
    return pd.DataFrame(rows)


def _select_best_params(
    pareto: pd.DataFrame, objective_names: list[str]
) -> dict | None:
    """
    Pareto 最前線から最適パラメータを選択する。

    選択基準:
      1. 推薦数 >= MIN_RECS_THRESHOLD を満たす Trial を優先
      2. その中で rmse_open + rmse_close が最小の点を選ぶ
      3. 該当なければ全 Pareto 最前線から推薦数を加味した複合スコアで選ぶ
    """
    if pareto.empty:
        return None

    # neg_n_recommendations が存在する場合は推薦数フィルタを適用
    MIN_RECS_THRESHOLD = 5  # 最低限この銘柄数以上の推薦が欲しい

    pareto = pareto.copy()
    if "obj_neg_n_recommendations" in pareto.columns:
        pareto["n_recs"] = (-pareto["obj_neg_n_recommendations"]).clip(lower=0)
        qualified = pareto[pareto["n_recs"] >= MIN_RECS_THRESHOLD]
        pool = qualified if not qualified.empty else pareto
        logger.info(
            "Pareto 最前線: %d 点中 推薦数 >= %d の点: %d 点",
            len(pareto), MIN_RECS_THRESHOLD, len(qualified),
        )
    else:
        pool = pareto

    score_col = "score"
    pool = pool.copy()
    pool[score_col] = pool["obj_rmse_open"] + pool["obj_rmse_close"]
    best_row = pool.loc[pool[score_col].idxmin()]

    # param_ プレフィックスを除いて返す
    params = {
        k.removeprefix("param_"): v
        for k, v in best_row.items()
        if k.startswith("param_")
    }
    # LightGBM に必要な固定キーを追加
    params.update({"objective": "regression", "metric": "rmse",
                   "verbose": -1, "n_jobs": -1})
    # int キャスト
    for key in ("num_leaves", "max_depth", "min_child_samples", "bagging_freq"):
        if key in params:
            params[key] = int(params[key])
    return params


def _log_pareto_front(study: optuna.Study, objective_names: list[str]) -> None:
    """Pareto 最前線のサマリをログに出力する。"""
    pareto = _build_pareto_df(study, objective_names)
    if pareto.empty:
        logger.warning("Pareto 最前線が空です")
        return

    obj_cols = [f"obj_{n}" for n in objective_names]
    logger.info(
        "\n=== Pareto 最前線 (%d 点) ===\n%s",
        len(pareto),
        pareto[["trial_number"] + obj_cols].round(6).to_string(index=False),
    )


# ---------------------------------------------------------------------------
# 最適化済みパラメータの適用
# ---------------------------------------------------------------------------

def apply_best_params(
    features: dict[str, pd.DataFrame],
    params_path: Path = BEST_PARAMS_PATH,
) -> tuple:
    """
    保存済みの推薦パラメータでモデルを再学習して保存する。

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        全銘柄の特徴量
    params_path : Path
        best_params.json のパス

    Returns
    -------
    tuple[lgb.Booster, lgb.Booster]
    """
    if not params_path.exists():
        raise FileNotFoundError(f"推薦パラメータファイルが見つかりません: {params_path}")

    params = json.loads(params_path.read_text(encoding="utf-8"))
    logger.info("推薦パラメータを適用して再学習します: %s", params)

    model_open, model_close = train(
        features,
        force=True,
        params=params,
        save=True,
    )
    return model_open, model_close


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多目的ハイパーパラメータ最適化")
    p.add_argument("--n-trials", type=int, default=50, help="試行回数（デフォルト: 50）")
    p.add_argument(
        "--n-tickers", type=int, default=0,
        help="最適化に使う銘柄数（デフォルト: 0=全銘柄）。速度重視なら 300 を指定。"
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="並列 Trial 数（デフォルト: 1）。CPU コア数の半分程度を推奨。"
    )
    p.add_argument(
        "--storage", type=str, default=None,
        help='Optuna ストレージ URL（例: "sqlite:///optuna.db"）'
    )
    p.add_argument(
        "--apply", action="store_true",
        help="最適化後に推薦パラメータで本番モデルを再学習する"
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = _parse_args()

    # 特徴量をすべてロード
    import os
    from src.features.engineer import load_processed

    logger.info("特徴量を読み込みます ...")
    features: dict[str, pd.DataFrame] = {}
    for fname in os.listdir(config.PROCESSED_DIR):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace("_T.csv", ".T")
        df = load_processed(ticker)
        if df is not None:
            features[ticker] = df
    logger.info("%d 銘柄の特徴量を読み込みました", len(features))

    n_tickers = args.n_tickers if args.n_tickers > 0 else None

    study = run_optimization(
        features,
        n_trials=args.n_trials,
        n_tickers=n_tickers,
        storage=args.storage,
        n_jobs=args.n_jobs,
    )

    if args.apply:
        logger.info("推薦パラメータで本番モデルを再学習します ...")
        apply_best_params(features)
        logger.info("完了")
