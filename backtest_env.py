"""Historical-only backtesting Gymnasium environment for Kraken ETH/USD data."""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from env import (
    FEATURE_COLUMNS,
    LAG_VALUES,
    MAKER_FEE,
    OBSERVATION_COLUMNS,
    OBSERVATION_SIZE,
    SCALAR_COLUMNS,
    SHAPE_BONUS,
    TREND_PREFIXES,
)


class KrakenBacktestEnv(gym.Env):
    """Gymnasium-compatible historical backtest environment with live-env feature parity."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        episode_length: int = 5000,
        start_idx: int = None,
        initial_usd: float = 50.0,
        initial_eth: float = 0.025,
        max_buffer_rows: int = 40000,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.episode_length = int(episode_length)
        self.start_idx = int(start_idx) if start_idx is not None else None
        self.initial_usd = float(initial_usd)
        self.initial_eth = float(initial_eth)
        self.max_buffer_rows = int(max_buffer_rows)
        self.trade_size_eth = 0.001

        print(f"[KrakenBacktestEnv] Starting init for CSV: {self.csv_path}")
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
        self.full_df["ts"] = self.full_df["ts"] * 1000
        total_rows_loaded = len(self.full_df)
        load_duration_seconds = time.perf_counter() - load_start_time
        print(
            f"[KrakenBacktestEnv] Completed load: {total_rows_loaded:,} rows "
            f"in {load_duration_seconds:.2f}s"
        )

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

    def _validate_data_requirements(self) -> None:
        min_start_idx = self.max_buffer_rows - 1
        max_start_idx = len(self.full_df) - self.episode_length - 1
        if max_start_idx < min_start_idx:
            raise ValueError(
                "CSV does not have enough rows for 40k warmup plus episode_length. "
                f"rows={len(self.full_df)}, required>={self.max_buffer_rows + self.episode_length + 1}"
            )

        if self.start_idx is not None and not (min_start_idx <= self.start_idx <= max_start_idx):
            raise ValueError(
                f"start_idx must be in [{min_start_idx}, {max_start_idx}], got {self.start_idx}."
            )

    def _get_portfolio_value(self, eth_balance: float, usd_balance: float, price: float) -> float:
        return float((eth_balance * price) + usd_balance)

    def _lag_price_vol(self, n: int, close_now: float, vol_now: float) -> tuple[float, float]:
        if len(self.df) < n + 1:
            return 0.0, 0.0
        c_prev = float(self.df["close"].iloc[-(n + 1)])
        v_prev = float(self.df["vol"].iloc[-(n + 1)])
        ret = np.tanh(((close_now - c_prev) / (c_prev + 1e-8)) * 5.0)
        vol_ret = np.tanh(((vol_now - v_prev) / (v_prev + 1e-8)) * 5.0)
        return float(ret), float(vol_ret)

    def _series_lag_value(self, series: pd.Series, n: int, transform: str = "none") -> float:
        if len(series) < n + 1:
            return 0.0
        value = float(series.iloc[-(n + 1)])
        if transform == "tanh":
            return float(np.tanh(value * 5.0))
        return value

    def _compute_observation(self, usd_balance: float, eth_balance: float) -> np.ndarray:
        df = self.df.copy()
        if df.empty:
            return np.zeros((OBSERVATION_SIZE,), dtype=np.float32)

        close = df["close"]
        vol = df["vol"]
        high = df["high"]
        low = df["low"]

        rsi7 = RSIIndicator(close=close, window=7).rsi() / 100.0
        rsi14 = RSIIndicator(close=close, window=14).rsi() / 100.0
        rsi21 = RSIIndicator(close=close, window=21).rsi() / 100.0
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        stoch_k = stoch.stoch() / 100.0
        stoch_d = stoch.stoch_signal() / 100.0
        cci20 = (CCIIndicator(high=high, low=low, close=close, window=20).cci() / 200.0).clip(-1.0, 1.0)
        willr = (WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r() + 100.0) / 100.0
        adx14 = ADXIndicator(high=high, low=low, close=close, window=14).adx() / 100.0
        bb20 = BollingerBands(close=close, window=20, window_dev=2)
        bb50 = BollingerBands(close=close, window=50, window_dev=2)
        bb20_p = bb20.bollinger_pband()

        atr14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        atr14_over_price = atr14 / (close + 1e-8)
        one_ret = close.pct_change()
        realized_vol_20 = one_ret.rolling(20).std() / 0.02

        ema9 = EMAIndicator(close=close, window=9).ema_indicator()
        ema20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema50 = EMAIndicator(close=close, window=50).ema_indicator()
        ema200 = EMAIndicator(close=close, window=200).ema_indicator()
        macd_hist = MACD(close=close, window_fast=12, window_slow=26, window_sign=9).macd_diff()
        bb20_width = (bb20.bollinger_hband() - bb20.bollinger_lband()) / (close + 1e-8)
        bb50_width = (bb50.bollinger_hband() - bb50.bollinger_lband()) / (close + 1e-8)
        obv = OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
        obv_pct = obv.pct_change()

        current_price = float(close.iloc[-1])
        current_vol = float(vol.iloc[-1])

        price_lags, vol_lags = [], []
        for n in LAG_VALUES:
            p, v = self._lag_price_vol(n, current_price, current_vol)
            price_lags.append(p)
            vol_lags.append(v)

        scalar_features = [
            float(rsi7.iloc[-1]),
            float(rsi14.iloc[-1]),
            float(rsi21.iloc[-1]),
            float(stoch_k.iloc[-1]),
            float(stoch_d.iloc[-1]),
            float(cci20.iloc[-1]),
            float(willr.iloc[-1]),
            float(adx14.iloc[-1]),
            float(bb20_p.iloc[-1]),
            float(atr14_over_price.iloc[-1]),
            float(realized_vol_20.iloc[-1]),
        ]

        series_map = {
            "ema9": (close / (ema9 + 1e-8)) - 1.0,
            "ema20": (close / (ema20 + 1e-8)) - 1.0,
            "ema50": (close / (ema50 + 1e-8)) - 1.0,
            "ema200": (close / (ema200 + 1e-8)) - 1.0,
            "macd_hist_atr": macd_hist / (atr14 + 1e-8),
            "bb20_width_price": bb20_width,
            "bb50_width_price": bb50_width,
            "obv_pct_change": obv_pct,
        }

        trend_lags: list[float] = []
        for prefix in TREND_PREFIXES:
            series = series_map[prefix]
            for n in LAG_VALUES:
                trend_lags.append(self._series_lag_value(series, n, transform="tanh"))

        # Historical-only backtest: no order book API calls, so keep these as zeros.
        order_book_features = [0.0] * 5

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

        ts = pd.to_datetime(int(df.iloc[-1]["ts"]), unit="ms", utc=True)
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

        obs = np.array(
            price_lags
            + vol_lags
            + scalar_features
            + trend_lags
            + order_book_features
            + portfolio_features
            + self_awareness
            + time_features,
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        min_start_idx = self.max_buffer_rows - 1
        max_start_idx = len(self.full_df) - self.episode_length - 1

        if self.start_idx is not None:
            start_idx = self.start_idx
        else:
            start_idx = int(self.np_random.integers(min_start_idx, max_start_idx + 1))

        self.current_idx = start_idx
        self.df = self.full_df.iloc[start_idx - self.max_buffer_rows + 1 : start_idx + 1].copy().reset_index(drop=True)

        self.usd_balance = self.initial_usd
        self.eth_balance = self.initial_eth

        self.steps_in_episode = 0
        self.step_count = 0
        self.last_action = "hold"
        self.last_filled_trade_step = 0
        self.position_entry_price = None
        self.position_entry_step = None

        current_price = float(self.df.iloc[-1]["close"])
        self.starting_portfolio_usd = self._get_portfolio_value(self.eth_balance, self.usd_balance, current_price)
        self.last_balance = self.starting_portfolio_usd

        obs = self._compute_observation(self.usd_balance, self.eth_balance)
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        action_raw = float(np.array(action).reshape(-1)[0])
        self.last_action = "hold"

        next_idx = self.current_idx + 1
        if next_idx >= len(self.full_df):
            return self.last_obs.copy(), 0.0, True, False, {"action_taken": "hold", "portfolio_usd": self.last_balance}

        next_bar = self.full_df.iloc[next_idx]
        execution_price = float(next_bar["open"])

        trade_filled = False
        filled_price = None

        if action_raw > 0.85:
            required = self.trade_size_eth * execution_price * (1.0 + MAKER_FEE)
            if self.usd_balance >= required:
                self.usd_balance -= required
                self.eth_balance += self.trade_size_eth
                trade_filled = True
                filled_price = execution_price
                self.last_action = "buy"
                self.position_entry_price = execution_price
                self.position_entry_step = self.step_count + 1
                self.last_filled_trade_step = self.step_count + 1
        elif action_raw < -0.85 and self.eth_balance >= self.trade_size_eth:
            proceeds = self.trade_size_eth * execution_price * (1.0 - MAKER_FEE)
            self.eth_balance -= self.trade_size_eth
            self.usd_balance += proceeds
            trade_filled = True
            filled_price = execution_price
            self.last_action = "sell"
            self.position_entry_price = None
            self.position_entry_step = None
            self.last_filled_trade_step = self.step_count + 1

        self.current_idx = next_idx
        self.df = pd.concat([self.df, pd.DataFrame([next_bar])], ignore_index=True).tail(self.max_buffer_rows).reset_index(drop=True)

        current_price = float(self.df.iloc[-1]["close"])
        obs = self._compute_observation(self.usd_balance, self.eth_balance)
        portfolio_usd = self._get_portfolio_value(self.eth_balance, self.usd_balance, current_price)

        prev_balance = max(self.last_balance, 1e-8)
        reward = (portfolio_usd - prev_balance) / prev_balance
        if trade_filled:
            reward -= MAKER_FEE

        if len(self.df) >= 6:
            price_change_5 = current_price - float(self.df["close"].iloc[-6])
            if price_change_5 > 0 and self.eth_balance > 0:
                reward += SHAPE_BONUS
            elif price_change_5 < 0 and self.usd_balance > 0:
                reward += SHAPE_BONUS

        scaled_reward = reward * 100.0

        self.last_balance = portfolio_usd
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
        }
        return obs, float(scaled_reward), terminated, False, info


# Import consistency checks requested for feature contracts.
assert len(FEATURE_COLUMNS) == OBSERVATION_SIZE
assert FEATURE_COLUMNS == OBSERVATION_COLUMNS
assert len(SCALAR_COLUMNS) == 11
