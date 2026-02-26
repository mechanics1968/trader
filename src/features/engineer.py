"""
OHLCV データからテクニカル指標・特徴量を生成する。

生成した特徴量 DataFrame は data/processed/{ticker}.csv に保存する。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ta.momentum as ta_mom
import ta.trend as ta_trend
import ta.volatility as ta_vol
import ta.volume as ta_vol2

import config

logger = logging.getLogger(__name__)


def _rolling_ts_rank(series: pd.Series, window: int) -> pd.Series:
    """時系列ランク: 直近 window 日中の現在値の百分位（0〜1）を返す。"""
    return series.rolling(window, min_periods=window // 2).apply(
        lambda x: float((x[:-1] <= x[-1]).mean()), raw=True
    )


def build_features(
    ticker: str,
    df: pd.DataFrame,
    sector_returns: pd.DataFrame | None = None,
    market_returns: pd.DataFrame | None = None,
    foreign_flow: pd.DataFrame | None = None,
    us_market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    1銘柄分の OHLCV から特徴量を生成して保存する。

    Parameters
    ----------
    ticker : str
        銘柄ティッカー（例: "7203.T"）
    df : pd.DataFrame
        列: open, high, low, close, volume / index: date
    foreign_flow : pd.DataFrame | None
        align_foreign_flow_to_daily() の出力（日次外資売買動向）

    Returns
    -------
    pd.DataFrame
        特徴量を付加した DataFrame（NaN 行を除去済み）
    """
    df = df.copy().sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ------------------------------------------------------------------ #
    # 価格変化率
    # ------------------------------------------------------------------ #
    df["return_1d"] = close.pct_change(1)
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)
    df["open_close_ratio"] = (close - df["open"]) / df["open"]
    df["high_low_ratio"] = (high - low) / low
    df["gap_ratio"] = (df["open"] - close.shift(1)) / close.shift(1)

    # ------------------------------------------------------------------ #
    # トレンド指標
    # ------------------------------------------------------------------ #
    for w in config.SMA_WINDOWS:
        sma = ta_trend.SMAIndicator(close, window=w).sma_indicator()
        df[f"sma_{w}"] = sma
        df[f"close_sma{w}_ratio"] = close / (sma + 1e-9) - 1

    for w in config.EMA_WINDOWS:
        df[f"ema_{w}"] = ta_trend.EMAIndicator(close, window=w).ema_indicator()

    macd_ind = ta_trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    adx_ind = ta_trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_ind.adx()

    # ------------------------------------------------------------------ #
    # モメンタム指標
    # ------------------------------------------------------------------ #
    df["rsi"] = ta_mom.RSIIndicator(close, window=config.RSI_PERIOD).rsi()

    stoch = ta_mom.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["willr"] = ta_mom.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    # ------------------------------------------------------------------ #
    # ボラティリティ指標
    # ------------------------------------------------------------------ #
    atr_ind = ta_vol.AverageTrueRange(high, low, close, window=config.ATR_PERIOD)
    df["atr"] = atr_ind.average_true_range()
    df["atr_ratio"] = df["atr"] / (close + 1e-9)

    bb = ta_vol.BollingerBands(close, window=config.BB_PERIOD, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_position"] = bb.bollinger_pband()

    df["std_5"] = close.rolling(5).std() / (close + 1e-9)
    df["std_20"] = close.rolling(20).std() / (close + 1e-9)

    # ------------------------------------------------------------------ #
    # 出来高指標（すべて比率ベース）
    # ------------------------------------------------------------------ #
    volume_ma5 = volume.rolling(5).mean()
    volume_ma20 = volume.rolling(20).mean()
    df["volume_ratio_5"] = volume / (volume_ma5 + 1)       # 5日平均比
    df["volume_ratio_20"] = volume / (volume_ma20 + 1)     # 20日平均比
    df["volume_change"] = volume.pct_change(1)              # 前日比

    obv = ta_vol2.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_change"] = obv.pct_change(5)                    # OBV の5日変化率

    # ------------------------------------------------------------------ #
    # ラグ特徴量（すべて比率ベース・絶対値は使わない）
    # ------------------------------------------------------------------ #
    for lag in config.LAG_DAYS:
        # 価格ラグ: 当日終値に対する比率
        df[f"close_lag{lag}_ratio"] = close.shift(lag) / (close + 1e-9) - 1
        # リターンラグ: そのまま変化率
        df[f"return_lag{lag}"] = df["return_1d"].shift(lag)
        # 出来高ラグ: 5日平均比
        df[f"volume_lag{lag}_ratio"] = volume.shift(lag) / (volume_ma5 + 1)

    # ------------------------------------------------------------------ #
    # セクター相対リターン（ticker_info が渡された場合のみ）
    # ------------------------------------------------------------------ #
    if sector_returns is not None:
        # sector_returns は {date: sector_return_1d, ...} の Series
        sec_1d = sector_returns["sector_return_1d"].reindex(df.index)
        sec_5d = sector_returns["sector_return_5d"].reindex(df.index)
        df["sector_return_1d"] = sec_1d
        df["sector_return_5d"] = sec_5d
        df["rel_return_1d"] = df["return_1d"] - sec_1d   # 業種平均との乖離（当日）
        df["rel_return_5d"] = df["return_5d"] - sec_5d   # 業種平均との乖離（5日）

    # ------------------------------------------------------------------ #
    # 市場相対特徴量（market_returns が渡された場合のみ）
    # ------------------------------------------------------------------ #
    if market_returns is not None:
        mkt_1d = market_returns["mkt_return_1d"].reindex(df.index)
        mkt_5d = market_returns["mkt_return_5d"].reindex(df.index)

        df["mkt_return_1d"] = mkt_1d                        # 市場平均リターン（当日）
        df["mkt_return_5d"] = mkt_5d                        # 市場平均リターン（5日）
        df["rel_to_mkt_1d"] = df["return_1d"] - mkt_1d     # 市場超過リターン（当日）
        df["rel_to_mkt_5d"] = df["return_5d"] - mkt_5d     # 市場超過リターン（5日）

        # ローリングベータ（20日）: 市場1%動いたときこの銘柄が何%動くか
        cov_20 = df["return_1d"].rolling(20).cov(mkt_1d)
        var_20 = mkt_1d.rolling(20).var()
        df["beta_20d"] = cov_20 / (var_20 + 1e-9)

        # 残差アルファ: ベータ成分を除いた当日の固有リターン
        df["alpha_1d"] = df["return_1d"] - df["beta_20d"] * mkt_1d

    # ------------------------------------------------------------------ #
    # 追加特徴量（既存）
    # ------------------------------------------------------------------ #
    # 当日の日中変動（始値→終値）
    df["intraday_return"] = (close - df["open"]) / (df["open"] + 1e-9)
    # 前日の日中変動（当日の寄り付き方向の参考）
    df["intraday_return_lag1"] = df["intraday_return"].shift(1)

    # 出来高急増フラグ（20日平均の3倍超 = 異常出来高）
    df["vol_spike_flag"] = (volume / (volume_ma20 + 1) > 3.0).astype(float)

    # 累積高値・安値からの距離（データ期間内の相対位置）
    df["dist_running_high"] = (high.expanding().max() - close) / (close + 1e-9)
    df["dist_running_low"] = (close - low.expanding().min()) / (close + 1e-9)

    # ------------------------------------------------------------------ #
    # 追加特徴量（新規: 世界標準シグナル）
    # ------------------------------------------------------------------ #

    # --- 中期モメンタム（Gu et al. 2020 第2・第5位因子） ---
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    # --- Amihud 非流動性（同論文 第3位因子） ---
    # |日次リターン| / 売買代金。非流動性が高いほど次日リバウンドしやすい
    dollar_vol = volume * close
    df["amihud"] = df["return_1d"].abs() / (dollar_vol + 1.0)
    df["amihud_20"] = df["amihud"].rolling(20).mean()

    # --- WorldQuant Alpha#101: intraday body ratio ---
    # (close - open) / (high - low)。日中方向の強さを高値安値レンジで正規化
    df["body_ratio"] = (close - df["open"]) / (high - low + 1e-6)

    # --- WorldQuant Alpha#12: 出来高-価格乖離シグナル ---
    # 出来高が増えたのに価格が下がった(or逆)場合は次日に反転しやすい
    df["vol_price_div"] = np.sign(volume.diff(1)) * (-close.diff(1) / (close + 1e-9))

    # --- MAX effect（過去20日の最大・最小日次リターン） ---
    # 直近で大きく上がった銘柄は翌日反転しやすい（宝くじ効果）
    df["max_ret_20d"] = df["return_1d"].rolling(20).max()
    df["min_ret_20d"] = df["return_1d"].rolling(20).min()

    # --- 52週高値・安値からの距離（ローリング252日） ---
    # 高値圏にある銘柄はモメンタム継続、安値圏は反転候補
    df["high_52w_ratio"] = close / (high.rolling(252, min_periods=60).max() + 1e-9) - 1
    df["low_52w_ratio"] = close / (low.rolling(252, min_periods=60).min() + 1e-9) - 1

    # --- MACD ヒストグラム変化量（モメンタムの加速度） ---
    df["macd_hist_change"] = df["macd_hist"].diff(1)

    # --- ボラティリティレジーム（短期/長期ボラの比率） ---
    # >1 なら最近ボラが高まっている（ブレイクアウトまたは混乱局面）
    df["vol_regime"] = df["std_5"] / (df["std_20"] + 1e-9)

    # --- Overnight gap の持続性（直近5日の平均ギャップ） ---
    df["gap_ratio_5d_mean"] = df["gap_ratio"].rolling(5).mean()

    # --- Alpha#6: 始値と出来高の10日相関（逆符号） ---
    # 始値が高いときに出来高が多い = 分布売り → 次日下落シグナル
    df["open_vol_corr_10"] = -df["open"].rolling(10).corr(volume)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # 外資売買動向（foreign_flow が渡された場合のみ）
    # ------------------------------------------------------------------ #
    if foreign_flow is not None and not foreign_flow.empty:
        for col in ["foreign_net_buy", "foreign_net_buy_4wk", "foreign_buy_regime"]:
            if col in foreign_flow.columns:
                df[col] = foreign_flow[col].reindex(df.index)
        # 欠損を中立値（0.5）で補完
        df["foreign_buy_regime"] = df["foreign_buy_regime"].fillna(0.5)
    else:
        # 外資データなし: 0.5 = 中立（買い越しでも売り越しでもない）
        df["foreign_buy_regime"] = 0.5

    # ------------------------------------------------------------------ #
    # 米国市場特徴量（us_market が渡された場合のみ）
    # 日本株の翌営業日始値は前日の米国市場動向に強く連動する
    # ------------------------------------------------------------------ #
    if us_market is not None and not us_market.empty:
        # 日付インデックスを正規化して結合
        us_idx = us_market.copy()
        us_idx.index = pd.to_datetime(us_idx.index).normalize()
        df_idx = pd.to_datetime(df.index).normalize()
        for col in us_idx.columns:
            df[col] = us_idx[col].reindex(df_idx).values
        logger.debug("米国市場特徴量を追加: %s", us_idx.columns.tolist())

    # ------------------------------------------------------------------ #
    # ターゲット変数
    # ------------------------------------------------------------------ #
    # 翌日の始値変化率（前日終値比）
    df["target_open_return"] = df["open"].shift(-1) / close - 1
    # 翌日の終値変化率（前日終値比）
    df["target_close_return"] = close.shift(-1) / close - 1
    # 翌日の日中騰落率（始値→終値）— デイトレの実損益に直結する予測ターゲット
    df["target_intraday_return"] = close.shift(-1) / df["open"].shift(-1) - 1

    # 残差ターゲット変数（市場ベータを除去した超過リターン）
    # 翌日（T+1）の市場リターンをベータ分だけ引いてアルファ成分を抽出する。
    # mkt_return_1d.shift(-1) = T+1 の等加重市場リターン（ターゲット日の市場分）
    # beta_20d は直近 20 日の推定ベータ（±3 でクリップして外れ値を防止）
    if "mkt_return_1d" in df.columns and "beta_20d" in df.columns:
        mkt_next = df["mkt_return_1d"].shift(-1)
        beta_clipped = df["beta_20d"].clip(-3.0, 3.0)
        df["target_alpha_open"] = df["target_open_return"] - beta_clipped * mkt_next
        df["target_alpha_close"] = df["target_close_return"] - beta_clipped * mkt_next

    # NaN を除去
    df = df.dropna()

    # 保存
    _save(ticker, df)

    return df


def build_features_all(
    price_data: dict[str, pd.DataFrame],
    ticker_info: pd.DataFrame | None = None,
    foreign_flow: "pd.DataFrame | None" = None,
    us_market: "pd.DataFrame | None" = None,
) -> dict[str, pd.DataFrame]:
    """
    全銘柄の特徴量を一括生成する。

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame]
        {ticker: OHLCV DataFrame}
    ticker_info : pd.DataFrame | None
        fetch_tickers() の出力。sector17 列を持つ場合、セクター相対リターンを追加する。
    foreign_flow : pd.DataFrame | None
        fetch_foreign_flow() の出力（週次外資売買動向）

    Returns
    -------
    dict[str, pd.DataFrame]
        {ticker: 特徴量 DataFrame}
    """
    from src.fetch.foreign_flow import align_foreign_flow_to_daily

    # 市場平均リターンを事前計算（全銘柄共通）
    market_returns = _compute_market_returns(price_data)
    logger.info("市場平均リターンを計算しました")

    # セクター相対リターンを事前計算
    sector_return_map: dict[str, pd.DataFrame] = {}
    if ticker_info is not None and "sector17" in ticker_info.columns:
        sector_return_map = _compute_sector_returns(price_data, ticker_info)
        logger.info("セクター相対リターンを計算しました（%d 銘柄）", len(sector_return_map))

    # 外資売買動向: 代表的な日次インデックスに合わせて展開（最初の銘柄を使用）
    foreign_flow_daily: pd.DataFrame | None = None
    if foreign_flow is not None and not foreign_flow.empty:
        sample_dates = next(iter(price_data.values())).index
        foreign_flow_daily = align_foreign_flow_to_daily(foreign_flow, pd.DatetimeIndex(sample_dates))
        logger.info("外資売買動向を日次展開しました")

    features: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        try:
            sec_ret = sector_return_map.get(ticker)
            features[ticker] = build_features(
                ticker, df,
                sector_returns=sec_ret,
                market_returns=market_returns,
                foreign_flow=foreign_flow_daily,
                us_market=us_market,
            )
        except Exception as exc:
            logger.warning("%s: 特徴量生成失敗: %s", ticker, exc)

    # クロスセクションランク特徴量を追加（全銘柄処理後に実施）
    features = _add_cross_sectional_ranks(features)
    logger.info("特徴量生成完了: %d 銘柄", len(features))
    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """ターゲット列・価格実値列・文字列列を除いた特徴量列名を返す。"""
    exclude = {
        "open", "high", "low", "close", "volume",
        "target_open_return", "target_close_return", "target_intraday_return",
        "target_alpha_open", "target_alpha_close",
        "target_cs_open", "target_cs_close",
        "ticker", "date",
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _add_cross_sectional_ranks(features: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    全銘柄処理後にクロスセクションランク特徴量を追加する。

    各日付について全銘柄の特徴量を百分位ランク化する（0=最低、1=最高）。
    LightGBM のクロスセクション予測において、銘柄間のスケール差を吸収しながら
    相対的な強弱シグナルを表現する最も有効な方法の1つ（Gu et al. 2020）。
    """
    # ランク化対象列: (元の列名, 新しい列名)
    rank_targets = [
        ("return_1d",       "cs_rank_ret1d"),   # 短期リターン（逆張りシグナル）
        ("return_5d",       "cs_rank_ret5d"),   # 5日リターン（短期モメンタム）
        ("return_20d",      "cs_rank_ret20d"),  # 20日リターン（中期モメンタム）
        ("amihud_20",       "cs_rank_amihud"),  # 非流動性（Amihud）
        ("volume_ratio_20", "cs_rank_vol20"),   # 出来高異常度
        ("vol_regime",      "cs_rank_volreg"),  # ボラティリティレジーム
    ]

    for src_col, dst_col in rank_targets:
        # 日付×銘柄のワイドフォーマットに変換
        wide = pd.DataFrame(
            {ticker: df[src_col] for ticker, df in features.items() if src_col in df.columns}
        )
        if wide.empty:
            continue

        # 各日付のクロスセクションで百分位ランク（NaN は無視）
        cs_rank = wide.rank(axis=1, pct=True, na_option="keep")

        for ticker in features:
            if ticker in cs_rank.columns:
                features[ticker][dst_col] = cs_rank[ticker].reindex(features[ticker].index)

    return features


def _save(ticker: str, df: pd.DataFrame) -> None:
    path = config.PROCESSED_DIR / f"{ticker.replace('.', '_')}.csv"
    df.to_csv(path)


def load_processed(ticker: str) -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


def _compute_market_returns(
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    全銘柄の終値から等加重市場平均リターンを計算して返す。

    Returns
    -------
    pd.DataFrame
        列: mkt_return_1d, mkt_return_5d（インデックス: date）
    """
    close_df = pd.DataFrame({t: d["close"] for t, d in price_data.items()})
    ret_1d_raw = close_df.pct_change(1, fill_method=None)
    ret_5d_raw = close_df.pct_change(5, fill_method=None)
    # ±50% 超の外れ値（データ異常・上場廃止等）を除外してから平均
    clip = 0.50
    ret_1d = ret_1d_raw.where(ret_1d_raw.abs() <= clip).mean(axis=1)
    ret_5d = ret_5d_raw.where(ret_5d_raw.abs() <= clip).mean(axis=1)
    return pd.DataFrame({"mkt_return_1d": ret_1d, "mkt_return_5d": ret_5d})


def _compute_sector_returns(
    price_data: dict[str, pd.DataFrame],
    ticker_info: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    全銘柄の終値から業種別平均リターンを計算し、
    {ticker: DataFrame(sector_return_1d, sector_return_5d)} を返す。

    ticker_info には ticker 列と sector17 列が必要。
    sector17 が "-" の銘柄はセクター情報なしとして扱う。
    """
    # ticker → sector17 の辞書
    valid = ticker_info[ticker_info["sector17"] != "-"][["ticker", "sector17"]]
    sector_map = valid.set_index("ticker")["sector17"].to_dict()

    # 全銘柄の終値を日付×銘柄の DataFrame に変換
    close_df = pd.DataFrame(
        {t: d["close"] for t, d in price_data.items() if t in sector_map}
    )
    ret_1d_raw = close_df.pct_change(1, fill_method=None)
    ret_5d_raw = close_df.pct_change(5, fill_method=None)
    # ±50% 超の外れ値を除外
    clip = 0.50
    ret_1d = ret_1d_raw.where(ret_1d_raw.abs() <= clip)
    ret_5d = ret_5d_raw.where(ret_5d_raw.abs() <= clip)

    # 業種ごとの平均リターンを計算
    sectors = valid["sector17"].unique()
    sector_ret_1d: dict[str, pd.Series] = {}
    sector_ret_5d: dict[str, pd.Series] = {}
    for sec in sectors:
        members = [t for t, s in sector_map.items() if s == sec and t in ret_1d.columns]
        if members:
            sector_ret_1d[sec] = ret_1d[members].mean(axis=1)
            sector_ret_5d[sec] = ret_5d[members].mean(axis=1)

    # 各 ticker にセクターリターンを対応付け
    result: dict[str, pd.DataFrame] = {}
    for ticker in price_data:
        sec = sector_map.get(ticker)
        if sec and sec in sector_ret_1d:
            result[ticker] = pd.DataFrame({
                "sector_return_1d": sector_ret_1d[sec],
                "sector_return_5d": sector_ret_5d[sec],
            })

    return result
