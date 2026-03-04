"""Historical-only backtesting Gymnasium environment for Kraken ETH/USD data."""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def _fast_lr_channel(close_arr, w):
    if w > 2000:
        return _lr_channel_large_window(close_arr, w)

    n = len(close_arr)
    x = np.arange(w, dtype=np.float64)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    x_norm = x - x_mean

    m = n - w + 1  # number of windows
    predicted_last = np.empty(m)
    stds = np.empty(m)

    chunk_size = max(1, int(100_000_000 / (w * 8)))
    for i in range(0, m, chunk_size):
        chunk = np.lib.stride_tricks.sliding_window_view(close_arr[i:i+chunk_size+w-1], w)
        if len(chunk) == 0:
            continue
        y_means = chunk.mean(axis=1)
        ss_xy = ((chunk - y_means[:,None]) * x_norm).sum(axis=1)
        slopes = ss_xy / ss_xx
        intercepts = y_means - slopes * x_mean
        predicted_last[i:i+len(chunk)] = slopes * (w-1) + intercepts
        residuals = (chunk - y_means[:,None]) - slopes[:,None] * x_norm
        stds[i:i+len(chunk)] = residuals.std(axis=1)

    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid[w-1:] = (predicted_last - close_arr[w-1:]) / close_arr[w-1:]
    upper[w-1:] = (predicted_last + 2*stds - close_arr[w-1:]) / close_arr[w-1:]
    lower[w-1:] = (predicted_last - 2*stds - close_arr[w-1:]) / close_arr[w-1:]
    return mid, upper, lower


def _lr_channel_large_window(close_arr, w):
    n = len(close_arr)

    x = np.arange(w, dtype=np.float64)
    x_mean = (w - 1) / 2.0
    ss_xx = w * (w - 1) * (2 * w - 1) / 6 - w * x_mean**2
    weights = x - x_mean

    cs_y = np.concatenate([[0], np.cumsum(close_arr)])
    sum_y = cs_y[w:] - cs_y[:-w]
    y_means = sum_y / w

    ss_xy = np.convolve(close_arr, weights[::-1], mode="valid")

    slopes = ss_xy / ss_xx
    intercepts = y_means - slopes * x_mean
    predicted_last = slopes * (w - 1) + intercepts

    cs_y2 = np.concatenate([[0], np.cumsum(close_arr**2)])
    sum_y2 = cs_y2[w:] - cs_y2[:-w]
    ss_yy = sum_y2 - w * y_means**2

    ss_residuals = ss_yy - ss_xy**2 / ss_xx
    ss_residuals = np.maximum(ss_residuals, 0)
    stds = np.sqrt(ss_residuals / w)

    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid[w-1:] = (predicted_last - close_arr[w-1:]) / close_arr[w-1:]
    upper[w-1:] = (predicted_last + 2 * stds - close_arr[w-1:]) / close_arr[w-1:]
    lower[w-1:] = (predicted_last - 2 * stds - close_arr[w-1:]) / close_arr[w-1:]
    return mid, upper, lower

from env import (
    FEATURE_COLUMNS,
    LAG_VALUES,
    MAKER_FEE,
    REWARD_FEE_RATE,
    OBSERVATION_COLUMNS,
    OBSERVATION_SIZE,
    SCALAR_COLUMNS,
    TREND_LAG_VALUES,
    TREND_PREFIXES,
)


class KrakenBacktestEnv(gym.Env):
    """Gymnasium-compatible historical backtest environment with live-env feature parity."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        candle_interval: int = 5,
        episode_length: int = 8000,
        start_idx: int = None,
        initial_usd: float = 50.0,
        initial_eth: float = 0.025,
        max_buffer_rows: int = 40000,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.candle_interval = int(candle_interval)
        self.episode_length = int(episode_length)
        self.start_idx = int(start_idx) if start_idx is not None else None
        self.initial_usd = float(initial_usd)
        self.initial_eth = float(initial_eth)
        self.max_buffer_rows = int(max_buffer_rows)
        self.min_order_eth = 0.001
        self.btc_prices: deque[float] = deque(maxlen=20)
        self.btc_prices_rsi: deque[float] = deque(maxlen=30)
        self.btc_prices_lr: deque[float] = deque(maxlen=12)
        self.btc_df: Optional[pd.DataFrame] = None
        self.btc_available = False
        self.btc_feature_cols = [
            "btc_return_1",
            "btc_return_4",
            "btc_return_16",
            "btc_ema20_dev",
            "btc_rsi14_norm",
            "btc_lr12_mid",
            "btc_lr12_upper",
        ]

        load_start_time = time.perf_counter()
        self.full_df = pd.read_csv(
            self.csv_path,
            header=None,
            names=["ts", "open", "high", "low", "close", "vol", "trades"],
            dtype={
                "ts": np.int64,
                "open": np.float32,
                "high": np.float32,
                "low": np.float32,
                "close": np.float32,
                "vol": np.float32,
                "trades": np.int32,
            },
            usecols=["ts", "open", "high", "low", "close", "vol"],
        )
        ts_unit = self._timestamp_unit(self.full_df["ts"])
        two_years_delta = self._two_year_cutoff_delta(ts_unit)
        max_ts_raw = int(self.full_df["ts"].max())
        sample_cutoff_raw = max_ts_raw - two_years_delta
        self.full_df["ts"] = self._normalize_timestamp_to_ms(self.full_df["ts"])
        total_rows_loaded = len(self.full_df)
        load_duration_seconds = time.perf_counter() - load_start_time
        self.sample_cutoff_ts = sample_cutoff_raw * 1000 if ts_unit == "seconds" else sample_cutoff_raw
        self.sample_start_idx = int(self.full_df["ts"].searchsorted(self.sample_cutoff_ts, side="left"))
        cutoff_iso = datetime.fromtimestamp(self.sample_cutoff_ts / 1000.0, tz=timezone.utc).isoformat()
        sample_rows = total_rows_loaded - self.sample_start_idx
        print(
            f"[KrakenBacktestEnv] Completed load: {total_rows_loaded:,} rows "
            f"in {load_duration_seconds:.2f}s"
        )
        print(
            f"[KrakenBacktestEnv] timestamp_unit={ts_unit}; "
            f"2y cutoff={self.sample_cutoff_ts} ({cutoff_iso}), "
            f"available_sample_rows={sample_rows:,}"
        )

        btc_csv_path = f"D:/BTCUSD_{self.candle_interval}.csv"
        if os.path.exists(btc_csv_path):
            self.btc_df = pd.read_csv(btc_csv_path, header=0)
            original_btc_columns = list(self.btc_df.columns)

            normalized_map = {}
            for column in original_btc_columns:
                col_norm = str(column).strip().lower()
                if col_norm in {"ts", "timestamp", "time", "datetime", "date"}:
                    normalized_map[column] = "ts"
                elif col_norm in {"open", "o"}:
                    normalized_map[column] = "open"
                elif col_norm in {"high", "h"}:
                    normalized_map[column] = "high"
                elif col_norm in {"low", "l"}:
                    normalized_map[column] = "low"
                elif col_norm in {"close", "c", "last"}:
                    normalized_map[column] = "close"
                elif col_norm in {"vol", "volume", "v"}:
                    normalized_map[column] = "vol"

            self.btc_df = self.btc_df.rename(columns=normalized_map)

            expected_cols = ["ts", "open", "high", "low", "close", "vol"]
            missing = [col for col in expected_cols if col not in self.btc_df.columns]
            if missing:
                fallback_cols = list(self.btc_df.columns[:6])
                if len(fallback_cols) == 6:
                    self.btc_df = self.btc_df[fallback_cols].copy()
                    self.btc_df.columns = expected_cols
                else:
                    raise ValueError(
                        f"BTC CSV missing expected columns {missing} and does not have at least 6 columns for fallback mapping."
                    )
            else:
                self.btc_df = self.btc_df[expected_cols].copy()

            self.btc_df = self.btc_df.astype(
                {
                    "ts": np.int64,
                    "open": np.float32,
                    "high": np.float32,
                    "low": np.float32,
                    "close": np.float32,
                    "vol": np.float32,
                }
            )
            self.btc_df["ts"] = self._normalize_timestamp_to_ms(self.btc_df["ts"])
            self.btc_available = len(self.btc_df) > 0
            if self.btc_available:
                btc_close = self.btc_df["close"].astype(np.float64)
                self.btc_df["btc_return_1"] = btc_close.pct_change(1).fillna(0.0)
                self.btc_df["btc_return_4"] = btc_close.pct_change(4).fillna(0.0)
                self.btc_df["btc_return_16"] = btc_close.pct_change(16).fillna(0.0)
                btc_ema20 = EMAIndicator(close=btc_close, window=20).ema_indicator()
                self.btc_df["btc_ema20_dev"] = ((btc_close / (btc_ema20 + 1e-8)) - 1.0).fillna(0.0)
                btc_rsi14 = RSIIndicator(close=btc_close, window=14).rsi()
                self.btc_df["btc_rsi14_norm"] = ((btc_rsi14 - 50.0) / 50.0).fillna(0.0)
                btc_mid, btc_upper, _ = self._rolling_lr_channel(btc_close, 12)
                self.btc_df["btc_lr12_mid"] = btc_mid.fillna(0.0)
                self.btc_df["btc_lr12_upper"] = btc_upper.fillna(0.0)
                self.btc_df = self.btc_df.sort_values("ts").reset_index(drop=True)

        self._validate_data_requirements()

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
        self.window_start_idx = 0
        self.window_end_idx = 0
        self.current_pos = 0
        self.current_idx = 0
        self.steps_in_episode = 0
        self.step_count = 0
        self.last_action = "hold"
        self.last_filled_trade_step = 0
        self.position_entry_price: Optional[float] = None
        self.position_entry_step: Optional[int] = None
        self.usd_balance = self.initial_usd
        self.eth_balance = self.initial_eth
        self.starting_portfolio_usd = 0.0
        self.last_balance = 0.0
        self.last_obs = np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        self.prev_price = 0.0

    def _validate_data_requirements(self) -> None:
        min_start_idx = max(self.max_buffer_rows - 1, self.sample_start_idx)
        max_start_idx = len(self.full_df) - self.episode_length - 1
        if max_start_idx < min_start_idx:
            raise ValueError(
                "CSV does not have enough rows in the 2-year sample zone for 40k warmup plus episode_length. "
                f"rows={len(self.full_df)}, sample_start_idx={self.sample_start_idx}, "
                f"required_start_idx<={max_start_idx}"
            )

        if self.start_idx is not None and not (min_start_idx <= self.start_idx <= max_start_idx):
            raise ValueError(
                f"start_idx must be in [{min_start_idx}, {max_start_idx}], got {self.start_idx}."
            )

    def _get_portfolio_value(self, eth_balance: float, usd_balance: float, price: float) -> float:
        return float((eth_balance * price) + usd_balance)

    def _lag_price_vol(self, n: int, close_now: float, vol_now: float) -> tuple[float, float]:
        if self.current_pos < n:
            return 0.0, 0.0
        c_prev = float(self.df["close"].iloc[self.current_pos - n])
        v_prev = float(self.df["vol"].iloc[self.current_pos - n])
        ret = np.tanh(((close_now - c_prev) / (c_prev + 1e-8)) * 5.0)
        vol_ret = np.tanh(((vol_now - v_prev) / (v_prev + 1e-8)) * 5.0)
        return float(ret), float(vol_ret)

    def _series_lag_value(self, column: str, n: int, transform: str = "none") -> float:
        if self.current_pos < n:
            return 0.0
        value = float(self.df[column].iloc[self.current_pos - n])
        if transform == "tanh":
            return float(np.tanh(value * 5.0))
        return value

    def _fast_adx(self, high, low, close, window: int = 14) -> pd.Series:
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        dm_pos = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0,
        )
        dm_neg = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0,
        )
        atr = pd.Series(tr).ewm(span=window, adjust=False).mean()
        di_pos = 100 * pd.Series(dm_pos).ewm(span=window, adjust=False).mean() / (atr + 1e-8)
        di_neg = 100 * pd.Series(dm_neg).ewm(span=window, adjust=False).mean() / (atr + 1e-8)
        dx = 100 * np.abs(di_pos - di_neg) / (di_pos + di_neg + 1e-8)
        adx = dx.ewm(span=window, adjust=False).mean()
        result = pd.Series(np.nan, index=range(len(close)))
        result.iloc[1:] = adx.values
        return result / 100.0

    def _rolling_lr_channel(self, series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        values = series.to_numpy(dtype=np.float64)
        mid, upper, lower = _fast_lr_channel(values, window)
        return (
            pd.Series(mid, index=series.index),
            pd.Series(upper, index=series.index),
            pd.Series(lower, index=series.index),
        )

    def _timestamp_unit(self, ts_series: pd.Series) -> str:
        median_ts = int(ts_series.astype(np.int64).median()) if len(ts_series) else 0
        return "seconds" if median_ts < 10**12 else "milliseconds"

    def _normalize_timestamp_to_ms(self, ts_series: pd.Series) -> pd.Series:
        ts_int = ts_series.astype(np.int64)
        if self._timestamp_unit(ts_int) == "seconds":
            return ts_int * 1000
        return ts_int

    def _two_year_cutoff_delta(self, ts_unit: str) -> int:
        if ts_unit == "seconds":
            return int(2 * 365 * 24 * 60 * 60)
        return int(2 * 365 * 24 * 60 * 60 * 1000)

    def _compute_cci_williams_fast(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        cci_window: int = 20,
        willr_window: int = 14,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        high_arr = high.to_numpy(dtype=np.float64)
        low_arr = low.to_numpy(dtype=np.float64)
        close_arr = close.to_numpy(dtype=np.float64)
        typical = (high_arr + low_arr + close_arr) / 3.0

        cci = np.full(len(typical), np.nan, dtype=np.float64)
        if len(typical) >= cci_window:
            cci_windows = np.lib.stride_tricks.sliding_window_view(typical, cci_window)
            sma = cci_windows.mean(axis=1)
            mad = np.mean(np.abs(cci_windows - sma[:, None]), axis=1)
            cci_values = (typical[cci_window - 1 :] - sma) / ((0.015 * mad) + 1e-8)
            cci[cci_window - 1 :] = np.clip(cci_values / 200.0, -1.0, 1.0)

        willr = np.full(len(close_arr), np.nan, dtype=np.float64)
        if len(close_arr) >= willr_window:
            high_windows = np.lib.stride_tricks.sliding_window_view(high_arr, willr_window)
            low_windows = np.lib.stride_tricks.sliding_window_view(low_arr, willr_window)
            highest_high = high_windows.max(axis=1)
            lowest_low = low_windows.min(axis=1)
            willr_values = ((highest_high - close_arr[willr_window - 1 :]) / ((highest_high - lowest_low) + 1e-8)) * -100.0
            willr[willr_window - 1 :] = (willr_values + 100.0) / 100.0

        elapsed = time.perf_counter() - t0
        return cci, willr, elapsed

    def _precompute_indicators(self) -> None:
        if self.df.empty:
            return

        close = self.df["close"]
        vol = self.df["vol"]
        high = self.df["high"]
        low = self.df["low"]

        self.df["rsi_7_norm"] = RSIIndicator(close=close, window=7).rsi() / 100.0
        self.df["rsi_14_norm"] = RSIIndicator(close=close, window=14).rsi() / 100.0
        self.df["rsi_21_norm"] = RSIIndicator(close=close, window=21).rsi() / 100.0

        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        self.df["stoch_k_norm"] = stoch.stoch() / 100.0
        self.df["stoch_d_norm"] = stoch.stoch_signal() / 100.0

        cci_20_clipped, willr_14_norm, cci_willr_seconds = self._compute_cci_williams_fast(
            high=high,
            low=low,
            close=close,
            cci_window=20,
            willr_window=14,
        )
        self.df["cci_20_clipped"] = np.nan_to_num(cci_20_clipped, nan=0.0)
        self.df["willr_14_norm"] = np.nan_to_num(willr_14_norm, nan=0.0)
        print(
            f"[KrakenBacktestEnv] cci_williams_r block: {cci_willr_seconds:.4f}s "
            f"(target < 0.1000s)"
        )

        self.df["adx_14_norm"] = self._fast_adx(high, low, close, window=14)

        bb20 = BollingerBands(close=close, window=20, window_dev=2)
        bb50 = BollingerBands(close=close, window=50, window_dev=2)
        self.df["bb20_p"] = bb20.bollinger_pband()

        atr14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        self.df["atr_14_over_price"] = atr14 / (close + 1e-8)
        self.df["realized_vol_20_norm"] = close.pct_change().rolling(20).std() / 0.02

        ema9 = EMAIndicator(close=close, window=9).ema_indicator()
        ema20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema50 = EMAIndicator(close=close, window=50).ema_indicator()
        ema200 = EMAIndicator(close=close, window=200).ema_indicator()
        macd_hist = MACD(close=close, window_fast=12, window_slow=26, window_sign=9).macd_diff()

        obv = OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()

        self.df["ema9"] = (close / (ema9 + 1e-8)) - 1.0
        self.df["ema20"] = (close / (ema20 + 1e-8)) - 1.0
        self.df["ema50"] = (close / (ema50 + 1e-8)) - 1.0
        self.df["ema200"] = (close / (ema200 + 1e-8)) - 1.0
        self.df["macd_hist_atr"] = macd_hist / (atr14 + 1e-8)
        self.df["bb20_width_price"] = (bb20.bollinger_hband() - bb20.bollinger_lband()) / (close + 1e-8)
        self.df["bb50_width_price"] = (bb50.bollinger_hband() - bb50.bollinger_lband()) / (close + 1e-8)
        self.df["obv_pct_change"] = obv.pct_change()

        close_np = close.to_numpy(dtype=np.float64)
        # 5-minute bars: 12 bars = 1hr, 288 bars = 1 day, 2016 bars = 1 week, 8000 bars ≈ 28 days.
        for lr_window in [12, 288, 2016, 8000]:
            lr_mid, lr_upper, lr_lower = _fast_lr_channel(close_np, lr_window)
            self.df[f"lr{lr_window}_mid"] = np.nan_to_num(lr_mid, nan=0.0)
            self.df[f"lr{lr_window}_upper"] = np.nan_to_num(lr_upper, nan=0.0)
            self.df[f"lr{lr_window}_lower"] = np.nan_to_num(lr_lower, nan=0.0)

        vol20 = vol.rolling(20).mean().replace(0.0, np.nan)
        spread = (high - low) / (close + 1e-8)
        spread_mean20 = spread.rolling(20).mean().replace(0.0, np.nan)
        self.df["bid_ask_spread_frac"] = (spread / (spread_mean20 + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
        self.df["bid_depth_5_over_vol20"] = ((vol * 0.5) / (vol20 + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
        self.df["ask_depth_5_over_vol20"] = ((vol * 0.5) / (vol20 + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
        imbalance = (((close - low) / ((high - low) + 1e-8)) * 2.0) - 1.0
        self.df["bid_ask_imbalance"] = imbalance.clip(-1.0, 1.0).fillna(0.0)
        self.df["price_dist_best_bid"] = ((close - low) / (close + 1e-8)).clip(-1.0, 1.0).fillna(0.0)

        self.df["eth_return_1"] = close.pct_change(1).fillna(0.0)
        self.df["eth_return_4"] = close.pct_change(4).fillna(0.0)
        self.df["eth_return_16"] = close.pct_change(16).fillna(0.0)

        if self.btc_available and self.btc_df is not None:
            btc_features = self.btc_df[["ts", *self.btc_feature_cols]]
            self.df = pd.merge_asof(
                self.df.sort_values("ts"),
                btc_features,
                on="ts",
                direction="backward",
            ).sort_index()
            self.df[self.btc_feature_cols] = self.df[self.btc_feature_cols].fillna(0.0)
        else:
            for col in self.btc_feature_cols:
                self.df[col] = 0.0

    def _get_btc_features(self) -> list[float]:
        eth_row = self.df.iloc[self.current_pos]
        return [
            float(eth_row.get("btc_return_1", 0.0)),
            float(eth_row.get("btc_return_4", 0.0)),
            float(eth_row.get("btc_return_16", 0.0)),
            float(eth_row.get("btc_ema20_dev", 0.0)),
            float(eth_row.get("btc_rsi14_norm", 0.0)),
            float(eth_row.get("btc_return_1", 0.0) - eth_row.get("eth_return_1", 0.0)),
            float(eth_row.get("btc_return_4", 0.0) - eth_row.get("eth_return_4", 0.0)),
            float(eth_row.get("btc_return_16", 0.0) - eth_row.get("eth_return_16", 0.0)),
            float(eth_row.get("btc_lr12_mid", 0.0)),
            float(eth_row.get("btc_lr12_upper", 0.0)),
        ]

    def _compute_observation(self, usd_balance: float, eth_balance: float) -> np.ndarray:
        if self.df.empty:
            return np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        row = self.df.iloc[self.current_pos]
        current_price = float(row["close"])
        current_vol = float(row["vol"])

        price_lags, vol_lags = [], []
        for n in LAG_VALUES:
            p, v = self._lag_price_vol(n, current_price, current_vol)
            price_lags.append(p)
            vol_lags.append(v)

        scalar_features = [
            float(row["rsi_7_norm"]),
            float(row["rsi_14_norm"]),
            float(row["rsi_21_norm"]),
            float(row["stoch_k_norm"]),
            float(row["stoch_d_norm"]),
            float(row["cci_20_clipped"]),
            float(row["willr_14_norm"]),
            float(row["adx_14_norm"]),
            float(row["bb20_p"]),
            float(row["atr_14_over_price"]),
            float(row["realized_vol_20_norm"]),
        ]

        trend_lags: list[float] = []
        for prefix in TREND_PREFIXES:
            for n in TREND_LAG_VALUES:
                trend_lags.append(self._series_lag_value(prefix, n, transform="tanh"))

        lr_channel_features = [
            float(row.get("lr12_mid", 0.0)), float(row.get("lr12_upper", 0.0)), float(row.get("lr12_lower", 0.0)),
            float(row.get("lr288_mid", 0.0)), float(row.get("lr288_upper", 0.0)), float(row.get("lr288_lower", 0.0)),
            float(row.get("lr2016_mid", 0.0)), float(row.get("lr2016_upper", 0.0)), float(row.get("lr2016_lower", 0.0)),
            float(row.get("lr8000_mid", 0.0)), float(row.get("lr8000_upper", 0.0)), float(row.get("lr8000_lower", 0.0)),
        ]

        order_book_features = [
            float(row.get("bid_ask_spread_frac", 0.0)),
            float(row.get("bid_depth_5_over_vol20", 0.0)),
            float(row.get("ask_depth_5_over_vol20", 0.0)),
            float(row.get("bid_ask_imbalance", 0.0)),
            float(row.get("price_dist_best_bid", 0.0)),
        ]

        portfolio_total = max(self._get_portfolio_value(eth_balance, usd_balance, current_price), 1e-8)
        portfolio_features = [
            (eth_balance * current_price) / portfolio_total,
            usd_balance / portfolio_total,
        ]

        last_action_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
        bars_since_trade = np.clip((self.step_count - self.last_filled_trade_step) / 100.0, 0.0, 1.0)
        if self.position_entry_price and self.position_entry_price > 0:
            pos_ret = np.tanh(((current_price - self.position_entry_price) / self.position_entry_price) * 5.0)
        else:
            pos_ret = 0.0

        self_awareness = [last_action_map.get(self.last_action, 0.0), float(bars_since_trade), float(pos_ret)]

        ts = pd.to_datetime(int(row["ts"]), unit="ms", utc=True)
        hour = ts.hour
        dow = ts.dayofweek
        dom = ts.day
        time_features = [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * dom / 31.0),
            np.cos(2 * np.pi * dom / 31.0),
        ]

        btc_features = self._get_btc_features()

        obs = np.array(
            price_lags
            + vol_lags
            + scalar_features
            + trend_lags
            + lr_channel_features
            + order_book_features
            + portfolio_features
            + self_awareness
            + time_features
            + btc_features,
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        min_start_idx = max(self.max_buffer_rows - 1, self.sample_start_idx)
        max_start_idx = len(self.full_df) - self.episode_length - 1

        if self.start_idx is not None:
            start_idx = self.start_idx
        else:
            start_idx = int(self.np_random.integers(min_start_idx, max_start_idx + 1))

        self.current_idx = start_idx
        self.window_start_idx = start_idx - self.max_buffer_rows + 1
        self.window_end_idx = start_idx + self.episode_length
        self.df = self.full_df.iloc[self.window_start_idx : self.window_end_idx + 1].copy().reset_index(drop=True)
        self.current_pos = start_idx - self.window_start_idx
        precompute_start_time = time.perf_counter()
        self._precompute_indicators()
        precompute_duration_seconds = time.perf_counter() - precompute_start_time
        print(f"[KrakenBacktestEnv] Precompute completed in {precompute_duration_seconds:.2f}s")

        self.usd_balance = self.initial_usd
        self.eth_balance = self.initial_eth

        self.steps_in_episode = 0
        self.step_count = 0
        self.last_action = "hold"
        self.last_filled_trade_step = 0
        self.position_entry_price = None
        self.position_entry_step = None
        self.btc_prices.clear()
        self.btc_prices_rsi.clear()
        self.btc_prices_lr.clear()
        current_price = float(self.df.iloc[self.current_pos]["close"])
        self.starting_portfolio_usd = self._get_portfolio_value(self.eth_balance, self.usd_balance, current_price)
        self.last_balance = self.starting_portfolio_usd
        self.prev_price = current_price

        obs = self._compute_observation(self.usd_balance, self.eth_balance)
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        action_raw = float(np.array(action).reshape(-1)[0])
        self.last_action = "hold"

        next_idx = self.current_idx + 1
        if next_idx > self.window_end_idx or next_idx >= len(self.full_df):
            return self.last_obs.copy(), 0.0, True, False, {"action_taken": "hold", "portfolio_usd": self.last_balance}

        self.current_idx = next_idx
        self.current_pos += 1
        next_bar = self.df.iloc[self.current_pos]
        execution_price = float(next_bar["open"])

        trade_filled = False
        filled_price = None
        trade_value = 0.0

        target_eth_alloc: Optional[float] = None
        if action_raw > 0.3:
            target_eth_alloc = 1.0
        elif action_raw < -0.3:
            target_eth_alloc = 0.0

        eth_value = self.eth_balance * execution_price
        portfolio_before = eth_value + self.usd_balance
        eth_value_gap = 0.0
        if target_eth_alloc is not None:
            target_eth_value = portfolio_before * target_eth_alloc
            eth_value_gap = target_eth_value - eth_value
            alloc_diff_threshold = 0.10 * portfolio_before

        if target_eth_alloc is not None and portfolio_before > 0 and abs(eth_value_gap) > alloc_diff_threshold:
            if eth_value_gap > 0:
                buy_eth = eth_value_gap / max(execution_price, 1e-8)
                max_affordable_eth = self.usd_balance / max(execution_price * (1.0 + MAKER_FEE), 1e-8)
                buy_eth = min(buy_eth, max_affordable_eth)
                if buy_eth >= self.min_order_eth:
                    required = buy_eth * execution_price * (1.0 + MAKER_FEE)
                    self.usd_balance -= required
                    self.eth_balance += buy_eth
                    trade_filled = True
                    filled_price = execution_price
                    trade_value = buy_eth * execution_price
                    self.last_action = "buy"
                    self.position_entry_price = execution_price
                    self.position_entry_step = self.step_count + 1
                    self.last_filled_trade_step = self.step_count + 1
            else:
                sell_eth = min(abs(eth_value_gap) / max(execution_price, 1e-8), self.eth_balance)
                if sell_eth >= self.min_order_eth:
                    proceeds = sell_eth * execution_price * (1.0 - MAKER_FEE)
                    self.eth_balance -= sell_eth
                    self.usd_balance += proceeds
                    trade_filled = True
                    filled_price = execution_price
                    trade_value = sell_eth * execution_price
                    self.last_action = "sell"
                    self.position_entry_price = None
                    self.position_entry_step = None
                    self.last_filled_trade_step = self.step_count + 1

        current_price = float(next_bar["close"])
        obs = self._compute_observation(self.usd_balance, self.eth_balance)
        portfolio_usd = self._get_portfolio_value(self.eth_balance, self.usd_balance, current_price)

        prev_price = max(self.prev_price, 1e-8)
        price_change = (current_price - prev_price) / prev_price
        eth_allocation = (self.eth_balance * current_price / portfolio_usd) if portfolio_usd > 0 else 0.0
        fee_if_traded = 0.0
        if trade_filled:
            fee_if_traded = REWARD_FEE_RATE * trade_value / max(portfolio_usd, 1e-8)
        reward = (2.0 * eth_allocation - 1.0) * price_change - fee_if_traded

        self.last_balance = portfolio_usd
        self.prev_price = current_price
        self.last_obs = obs
        self.step_count += 1
        self.steps_in_episode += 1

        terminated = self.steps_in_episode >= self.episode_length or (
            portfolio_usd < 0.5 * self.starting_portfolio_usd
        )

        info = {
            "action_taken": self.last_action,
            "portfolio_usd": portfolio_usd,
            "fill_price": filled_price,
            "eth_allocation": (self.eth_balance * current_price / portfolio_usd) if portfolio_usd > 0 else 0.0,
        }
        return obs, float(reward), terminated, False, info


# Import consistency checks requested for feature contracts.
assert len(FEATURE_COLUMNS) == OBSERVATION_SIZE
assert FEATURE_COLUMNS == OBSERVATION_COLUMNS
assert len(SCALAR_COLUMNS) == 11
