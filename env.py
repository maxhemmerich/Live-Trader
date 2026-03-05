"""Kraken live trading environment with expanded observation/reward/trading logic."""

from __future__ import annotations

import glob
import logging
from collections import deque
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gymnasium import spaces
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

BASE_OHLCV_COLUMNS = ["ts", "open", "high", "low", "close", "vol"]
MAKER_FEE = 0.0016
REWARD_FEE_RATE = 0.0026
LAG_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    65536,
    131072,
    262144,
    524288,
    1048576,
]
DISTANT_ANCHORS = [65536, 131072, 262144, 524288, 1048576]
ANCHOR_REFRESH_SECONDS = {
    65536: 24 * 3600,
    131072: 24 * 3600,
    262144: 24 * 3600,
    524288: 7 * 24 * 3600,
    1048576: 7 * 24 * 3600,
}


PRICE_LAG_COLUMNS = [f"return_lag_{n}" for n in LAG_VALUES]
VOL_LAG_COLUMNS = [f"vol_lag_{n}" for n in LAG_VALUES]
SCALAR_COLUMNS = [
    "rsi_7_norm",
    "rsi_14_norm",
    "rsi_21_norm",
    "stoch_k_norm",
    "stoch_d_norm",
    "cci_20_clipped",
    "willr_14_norm",
    "adx_14_norm",
    "bb20_p",
    "atr_14_over_price",
    "realized_vol_20_norm",
]
TREND_PREFIXES = [
    "ema9",
    "ema20",
    "ema50",
    "ema200",
    "macd_hist_atr",
    "bb20_width_price",
    "bb50_width_price",
    "obv_pct_change",
]
TREND_LAG_VALUES = [1, 4, 16, 64, 256]
TREND_LAG_COLUMNS = [f"{prefix}_lag_{n}" for prefix in TREND_PREFIXES for n in TREND_LAG_VALUES]
LR_WINDOWS = [12, 288, 2016, 8000]
LR_CHANNEL_COLUMNS = [f"lr{n}_{suffix}" for n in LR_WINDOWS for suffix in ("mid", "upper", "lower")]
ORDER_BOOK_COLUMNS = [
    "bid_ask_spread_frac",
    "bid_depth_5_over_vol20",
    "ask_depth_5_over_vol20",
    "bid_ask_imbalance",
    "price_dist_best_bid",
]
PORTFOLIO_COLUMNS = ["btc_value_weight", "usd_value_weight"]
SELF_AWARE_COLUMNS = [
    "last_action_was_buy",
    "bars_since_last_trade",
    "current_position_return",
]
TIME_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_sin", "dom_cos"]
ETH_FEATURE_COLUMNS = [
    "eth_return_1",
    "eth_return_4",
    "eth_return_16",
    "eth_ema20_dev",
    "eth_rsi14_norm",
    "eth_btc_return_diff_1",
    "eth_btc_return_diff_4",
    "eth_btc_return_diff_16",
    "eth_lr12_mid",
    "eth_lr12_upper",
]

FEATURE_COLUMNS = (
    PRICE_LAG_COLUMNS
    + VOL_LAG_COLUMNS
    + SCALAR_COLUMNS
    + TREND_LAG_COLUMNS
    + LR_CHANNEL_COLUMNS
    + ORDER_BOOK_COLUMNS
    + PORTFOLIO_COLUMNS
    + SELF_AWARE_COLUMNS
    + TIME_COLUMNS
    + ETH_FEATURE_COLUMNS
)
OBSERVATION_COLUMNS = FEATURE_COLUMNS
OBSERVATION_SIZE = len(FEATURE_COLUMNS)


class KrakenLiveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol: str = "XBT/USD",
        timeframe: str = "1m",
        candle_interval: int = 1,
        candle_limit: int = 720,
        max_buffer_rows: int = 40000,
        checkpoint_dir: str = "./checkpoints/",
        trading_log_path: str = "trading_log.csv",
    ) -> None:
        super().__init__()
        load_dotenv()

        self.symbol = symbol
        self.candle_interval = candle_interval
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self.max_buffer_rows = max_buffer_rows
        self.checkpoint_dir = checkpoint_dir
        self.trading_log_path = trading_log_path
        self.min_order_btc = 0.0001

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.exchange = ccxt.kraken(
            {
                "apiKey": os.getenv("KRAKEN_API_KEY", ""),
                "secret": os.getenv("KRAKEN_API_SECRET", ""),
                "enableRateLimit": True,
            }
        )

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

        self.consecutive_errors = 0
        self.kill_switch = False
        self.last_action = "hold"
        self.last_filled_trade_step = 0
        self.position_entry_price: Optional[float] = None
        self.position_entry_step: Optional[int] = None
        self.pending_order_id: Optional[str] = None

        self.eth_prices: deque[float] = deque(maxlen=20)
        self.eth_prices_rsi: deque[float] = deque(maxlen=30)
        self.eth_prices_lr: deque[float] = deque(maxlen=12)

        self.df = self._initialize_candle_buffer()
        self.distant_anchors = self._init_distant_anchors()

        self.last_obs = np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.starting_portfolio_usd: Optional[float] = None
        self.last_balance = 0.0
        self.prev_price = 0.0

        print("[init] Primary asset: XBT/USD | Secondary feature: ETH/USD | Candle interval: 1min")

        self._rotate_log_if_needed()
        self._init_log_file()

    def _rotate_log_if_needed(self) -> None:
        if not os.path.exists(self.trading_log_path):
            return
        created_at = os.path.getctime(self.trading_log_path)
        if time.time() - created_at <= 7 * 24 * 3600:
            return
        created_date = datetime.fromtimestamp(created_at, tz=timezone.utc).strftime("%Y-%m-%d")
        archived = f"trading_log_{created_date}.csv"
        idx = 1
        while os.path.exists(archived):
            archived = f"trading_log_{created_date}_{idx}.csv"
            idx += 1
        os.rename(self.trading_log_path, archived)
        self.logger.info("Rotated stale trading log to %s", archived)

    def _initialize_candle_buffer(self) -> pd.DataFrame:
        csv_path = f"D:/XBTUSD_{self.candle_interval}.csv"

        def _fetch_live_candles() -> list[list[float]]:
            try:
                bars = self.exchange.fetch_ohlcv(
                    self.symbol,
                    f"{self.candle_interval}m",
                    limit=self.candle_limit,
                    params={"interval": self.candle_interval},
                )
                self.consecutive_errors = 0
                return bars
            except Exception as exc:
                self._record_api_error(exc)
                return []

        if os.path.exists(csv_path):
            csv_df = pd.read_csv(
                csv_path,
                header=0,
                names=["ts", "open", "high", "low", "close", "vol", "trades"],
            )
            csv_df = csv_df.drop(columns=["trades"])
            csv_df["ts"] = csv_df["ts"] * 1000

            live_bars = _fetch_live_candles()
            live_df = pd.DataFrame(live_bars, columns=BASE_OHLCV_COLUMNS)

            merged = pd.concat([csv_df, live_df], ignore_index=True)
            merged.drop_duplicates(subset=["ts"], inplace=True)
            merged.sort_values("ts", inplace=True)
            trimmed = merged.tail(self.max_buffer_rows).reset_index(drop=True)
            print(
                f"[INIT] Loaded {len(csv_df)} rows from CSV + {len(live_df)} live candles = {len(trimmed)} total rows"
            )
            return trimmed

        print(f"[INIT][WARNING] CSV not found at {csv_path}, loading API candles only.")
        live_bars = _fetch_live_candles()
        live_df = pd.DataFrame(live_bars, columns=BASE_OHLCV_COLUMNS)
        live_df.drop_duplicates(subset=["ts"], inplace=True)
        live_df.sort_values("ts", inplace=True)
        trimmed = live_df.tail(self.max_buffer_rows).reset_index(drop=True)
        print(f"[INIT] Loaded 0 rows from CSV + {len(live_df)} live candles = {len(trimmed)} total rows")
        return trimmed

    def _init_distant_anchors(self) -> dict[int, dict[str, float]]:
        anchors: dict[int, dict[str, float]] = {}
        now = time.time()
        for n in DISTANT_ANCHORS:
            close_val, vol_val = self._fetch_anchor_value(n)
            anchors[n] = {"close": close_val, "vol": vol_val, "fetched_at": now}
        return anchors

    def _fetch_anchor_value(self, n: int) -> tuple[float, float]:
        try:
            since_ms = int(time.time() * 1000) - (n * self.candle_interval * 60000)
            data = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                since=since_ms,
                limit=2,
                params={"interval": self.candle_interval},
            )
            if data:
                bar = data[-1]
                return float(bar[4]), float(bar[5])
        except Exception as exc:
            self.logger.warning("Failed to fetch anchor %s: %s", n, exc)
        return 0.0, 0.0

    def _refresh_distant_anchors_if_needed(self) -> None:
        now = time.time()
        for n in DISTANT_ANCHORS:
            anchor = self.distant_anchors.get(n)
            if anchor is None:
                self.distant_anchors[n] = {"close": 0.0, "vol": 0.0, "fetched_at": 0.0}
                anchor = self.distant_anchors[n]
            if now - float(anchor.get("fetched_at", 0.0)) < ANCHOR_REFRESH_SECONDS[n]:
                continue
            try:
                close_val, vol_val = self._fetch_anchor_value(n)
                if close_val > 0.0:
                    anchor["close"] = close_val
                if vol_val > 0.0:
                    anchor["vol"] = vol_val
                anchor["fetched_at"] = now
            except Exception:
                pass

    def _record_api_error(self, exc: Exception) -> bool:
        msg = str(exc)
        now = datetime.now(timezone.utc).isoformat()
        maintenance_tokens = ("maintenance", "unavailable", "Service unavailable")
        if any(token.lower() in msg.lower() for token in maintenance_tokens):
            self.logger.warning("%s | Kraken maintenance/unavailable: %s", now, msg)
            time.sleep(600)
            return True

        self.consecutive_errors += 1
        self.logger.error("%s | API error: %s", now, msg)
        if self.consecutive_errors >= 5:
            self.logger.critical("Kill switch triggered after 5 consecutive API errors.")
            self.kill_switch = True
        return False

    def _safe_fetch_balance(self) -> Optional[dict]:
        try:
            bal = self.exchange.fetch_balance()
            self.consecutive_errors = 0
            return bal
        except Exception as exc:
            maintenance = self._record_api_error(exc)
            if maintenance:
                try:
                    bal = self.exchange.fetch_balance()
                    self.consecutive_errors = 0
                    return bal
                except Exception as exc_retry:
                    self._record_api_error(exc_retry)
            return None

    @staticmethod
    def _extract_balances(balance: dict) -> tuple[float, float]:
        usd_balance = float(balance.get("free", {}).get("USD", 0.0))
        btc_balance = float(balance.get("free", {}).get("XBT", balance.get("free", {}).get("BTC", 0.0)))
        if usd_balance == 0.0:
            usd_balance = float(balance.get("total", {}).get("USD", 0.0))
        if btc_balance == 0.0:
            btc_balance = float(balance.get("total", {}).get("XBT", balance.get("total", {}).get("BTC", 0.0)))
        return usd_balance, btc_balance

    def _get_portfolio_value(self, btc_balance: float, usd_balance: float, price: float) -> float:
        return float((btc_balance * price) + usd_balance)

    def _lag_price_vol(self, n: int, close_now: float, vol_now: float) -> tuple[float, float]:
        if n <= 16384:
            idx = -(n + 1)
            if len(self.df) >= n + 1:
                c_prev = float(self.df["close"].iloc[idx])
                v_prev = float(self.df["vol"].iloc[idx])
            else:
                return 0.0, 0.0
        else:
            anchor = self.distant_anchors.get(n, {"close": 0.0, "vol": 0.0})
            c_prev = float(anchor.get("close", 0.0))
            v_prev = float(anchor.get("vol", 0.0))
            if c_prev == 0.0 and v_prev == 0.0:
                return 0.0, 0.0

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

    def _compute_lr_channel_features(self, prices: pd.Series, current_price: float) -> list[float]:
        features: list[float] = []
        for window in LR_WINDOWS:
            if len(prices) < window:
                features.extend([0.0, 0.0, 0.0])
                continue
            y = prices.iloc[-window:].to_numpy(dtype=np.float64)
            x = np.arange(window, dtype=np.float64)
            slope, intercept = np.polyfit(x, y, 1)
            fit = (slope * x) + intercept
            reg_now = float((slope * (window - 1)) + intercept)
            resid_std = float(np.std(y - fit))
            upper = reg_now + (2.0 * resid_std)
            lower = reg_now - (2.0 * resid_std)
            denom = current_price + 1e-8
            features.extend([
                (reg_now - current_price) / denom,
                (upper - current_price) / denom,
                (lower - current_price) / denom,
            ])
        return features

    def _get_eth_features(self) -> list[float]:
        try:
            ticker = self.exchange.fetch_ticker("ETH/USD")
            eth_price = float(ticker.get("last") or 0.0)
            if eth_price <= 0.0:
                return [0.0] * len(ETH_FEATURE_COLUMNS)

            self.eth_prices.append(eth_price)
            self.eth_prices_rsi.append(eth_price)
            self.eth_prices_lr.append(eth_price)

            eth_return_1 = 0.0
            eth_return_4 = 0.0
            eth_return_16 = 0.0
            if len(self.eth_prices) >= 2:
                prev = float(self.eth_prices[-2])
                eth_return_1 = (eth_price - prev) / (prev + 1e-8)
            if len(self.eth_prices) >= 5:
                prev = float(self.eth_prices[-5])
                eth_return_4 = (eth_price - prev) / (prev + 1e-8)
            if len(self.eth_prices) >= 17:
                prev = float(self.eth_prices[-17])
                eth_return_16 = (eth_price - prev) / (prev + 1e-8)

            eth_series = pd.Series(list(self.eth_prices), dtype=np.float64)
            eth_ema20 = float(EMAIndicator(close=eth_series, window=max(1, min(20, len(eth_series)))).ema_indicator().iloc[-1])
            eth_ema20_dev = (eth_price / (eth_ema20 + 1e-8)) - 1.0

            eth_rsi_norm = 0.0
            if len(self.eth_prices_rsi) >= 14:
                rsi_series = RSIIndicator(close=pd.Series(list(self.eth_prices_rsi), dtype=np.float64), window=14).rsi()
                rsi_last = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
                eth_rsi_norm = (rsi_last - 50.0) / 50.0

            def btc_ret(lag: int) -> float:
                if len(self.df) < lag + 1:
                    return 0.0
                prev_btc = float(self.df["close"].iloc[-(lag + 1)])
                curr_btc = float(self.df["close"].iloc[-1])
                return (curr_btc - prev_btc) / (prev_btc + 1e-8)

            eth_lr12_mid = 0.0
            eth_lr12_upper = 0.0
            if len(self.eth_prices_lr) >= 12:
                y = np.array(list(self.eth_prices_lr)[-12:], dtype=np.float64)
                x = np.arange(12, dtype=np.float64)
                slope, intercept = np.polyfit(x, y, 1)
                fit = (slope * x) + intercept
                reg_now = float((slope * 11) + intercept)
                resid_std = float(np.std(y - fit))
                eth_lr12_mid = (reg_now - eth_price) / (eth_price + 1e-8)
                eth_lr12_upper = ((reg_now + 2.0 * resid_std) - eth_price) / (eth_price + 1e-8)

            return [
                eth_return_1,
                eth_return_4,
                eth_return_16,
                eth_ema20_dev,
                eth_rsi_norm,
                eth_return_1 - btc_ret(1),
                eth_return_4 - btc_ret(4),
                eth_return_16 - btc_ret(16),
                eth_lr12_mid,
                eth_lr12_upper,
            ]
        except Exception:
            return [0.0] * len(ETH_FEATURE_COLUMNS)

    def _compute_observation(self, usd_balance: float, btc_balance: float) -> np.ndarray:
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
            for n in TREND_LAG_VALUES:
                trend_lags.append(self._series_lag_value(series, n, transform="tanh"))

        lr_channel_features = self._compute_lr_channel_features(close, current_price)

        order_book_features = [0.0] * 5
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=20)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2.0
                bid_depth_5 = float(sum(level[1] for level in bids[:5]))
                ask_depth_5 = float(sum(level[1] for level in asks[:5]))
                vol20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
                vol20 = max(vol20, 1e-8)
                order_book_features = [
                    (best_ask - best_bid) / (mid + 1e-8),
                    bid_depth_5 / vol20,
                    ask_depth_5 / vol20,
                    (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1e-8),
                    (current_price - best_bid) / (current_price + 1e-8),
                ]
        except Exception:
            order_book_features = [0.0] * 5

        portfolio_total = max(self._get_portfolio_value(btc_balance, usd_balance, current_price), 1e-8)
        portfolio_features = [
            (btc_balance * current_price) / portfolio_total,
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

        eth_features = self._get_eth_features()

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
            + eth_features,
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _init_log_file(self) -> None:
        if os.path.exists(self.trading_log_path):
            return
        headers = [
            "timestamp",
            "step",
            "action_raw",
            "action_taken",
            "reward",
            "cumulative_reward",
            "btc_price",
            "portfolio_usd",
            "btc_balance",
            "usd_balance",
        ] + FEATURE_COLUMNS
        with open(self.trading_log_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")

    def _append_log_row(self, action_raw: float, reward: float, btc_price: float, portfolio_usd: float, btc_balance: float, usd_balance: float, obs: np.ndarray) -> None:
        row = [
            datetime.now(timezone.utc).isoformat(),
            str(self.step_count),
            f"{action_raw:.8f}",
            self.last_action,
            f"{reward:.10f}",
            f"{self.cumulative_reward:.10f}",
            f"{btc_price:.8f}",
            f"{portfolio_usd:.8f}",
            f"{btc_balance:.8f}",
            f"{usd_balance:.8f}",
        ] + [f"{float(x):.8f}" for x in obs]
        with open(self.trading_log_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")

    def _extract_fill_price(self, order: dict, fallback_price: float) -> float:
        fill_price = order.get("average") or order.get("price")
        if fill_price is None:
            self.logger.warning("Order fill price unavailable; falling back to quoted price %.8f", fallback_price)
            return float(fallback_price)
        return float(fill_price)

    def _execute_limit_order(self, side: str, quoted_price: float, amount_btc: float) -> tuple[bool, float, bool]:
        """Return (filled, fill_price, canceled)."""
        try:
            if side == "buy":
                order = self.exchange.create_limit_buy_order(self.symbol, amount_btc, quoted_price)
            else:
                order = self.exchange.create_limit_sell_order(self.symbol, amount_btc, quoted_price)

            order_id = order.get("id")
            self.pending_order_id = order_id
            latest_order = order
            for _ in range(6):
                status = str(latest_order.get("status", "")).lower()
                if status in {"closed", "filled"}:
                    self.pending_order_id = None
                    return True, self._extract_fill_price(latest_order, quoted_price), False
                time.sleep(5)
                if hasattr(self.exchange, "fetch_order") and order_id:
                    latest_order = self.exchange.fetch_order(order_id, self.symbol)

            if self.pending_order_id and hasattr(self.exchange, "fetch_order"):
                latest_order = self.exchange.fetch_order(self.pending_order_id, self.symbol)
                status = str(latest_order.get("status", "")).lower()
                if status in {"closed", "filled"}:
                    self.logger.debug(
                        "Skipping cancel for order %s because it is already %s.",
                        self.pending_order_id,
                        status,
                    )
                    self.pending_order_id = None
                    return True, self._extract_fill_price(latest_order, quoted_price), False
                if status == "open" and hasattr(self.exchange, "cancel_order"):
                    try:
                        self.exchange.cancel_order(self.pending_order_id, self.symbol)
                        self.pending_order_id = None
                        return False, 0.0, True
                    except Exception as exc:
                        if "eorder:unknown order" in str(exc).lower():
                            self.logger.debug(
                                "Ignoring cancel for unknown order %s (likely already filled): %s",
                                self.pending_order_id,
                                exc,
                            )
                            self.pending_order_id = None
                            return False, 0.0, False
                        raise

            self.pending_order_id = None
            return False, 0.0, False
        except Exception as exc:
            self.logger.error("Limit order execution failed: %s", exc)
            self.pending_order_id = None
            return False, 0.0, False

    def _maintenance_retry_fetch_ohlcv(self, **kwargs):
        try:
            data = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                params={"interval": self.candle_interval},
                **kwargs,
            )
            self.consecutive_errors = 0
            return data, False
        except Exception as exc:
            maintenance = self._record_api_error(exc)
            if maintenance:
                try:
                    data = self.exchange.fetch_ohlcv(
                        self.symbol,
                        timeframe=self.timeframe,
                        params={"interval": self.candle_interval},
                        **kwargs,
                    )
                    self.consecutive_errors = 0
                    return data, False
                except Exception as retry_exc:
                    self._record_api_error(retry_exc)
                    return None, True
            return None, True

    def save_checkpoint(self, model, step: int) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path_no_ext = os.path.join(self.checkpoint_dir, f"sac_step_{step}_{timestamp}")
        model.save(path_no_ext)
        checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "sac_step_*.zip")),
            key=self._checkpoint_sort_key,
            reverse=True,
        )
        for old in checkpoints[5:]:
            os.remove(old)
        return f"{path_no_ext}.zip"

    @staticmethod
    def _checkpoint_sort_key(path: str) -> int:
        match = re.search(r"sac_step_(\d+)_\d{8}_\d{6}\.zip$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "sac_step_*.zip"))
        if not checkpoints:
            return None
        checkpoints.sort(key=self._checkpoint_sort_key, reverse=True)
        return checkpoints[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rotate_log_if_needed()
        self._init_log_file()

        balance = self._safe_fetch_balance()
        if balance is None:
            return self.last_obs.copy(), {}
        usd_balance, btc_balance = self._extract_balances(balance)

        obs = self._compute_observation(usd_balance, btc_balance)
        self.last_obs = obs

        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0
        total_usd = self._get_portfolio_value(btc_balance, usd_balance, price)
        self.last_balance = total_usd
        self.prev_price = price
        self.starting_portfolio_usd = total_usd
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.last_action = "hold"
        self.last_filled_trade_step = 0
        self.position_entry_price = None
        self.position_entry_step = None
        self.pending_order_id = None
        self.eth_prices.clear()
        self.eth_prices_rsi.clear()
        self.eth_prices_lr.clear()
        self.kill_switch = False
        self.consecutive_errors = 0

        return obs, {}

    def step(self, action):
        time.sleep(self.candle_interval * 60)
        self._rotate_log_if_needed()
        self._init_log_file()
        self._refresh_distant_anchors_if_needed()

        action_raw = float(np.array(action).reshape(-1)[0])
        self.last_action = "hold"

        if self.kill_switch:
            return self.last_obs.copy(), 0.0, True, False, {"kill_switch": True, "action_taken": "hold"}

        recent, truncated = self._maintenance_retry_fetch_ohlcv(limit=3)
        if truncated:
            return self.last_obs.copy(), 0.0, self.kill_switch, True, {"action_taken": "hold", "api_error": True}

        if recent:
            recent_df = pd.DataFrame(recent, columns=BASE_OHLCV_COLUMNS)
            self.df = pd.concat([self.df, recent_df], ignore_index=True)
            self.df = self.df.drop_duplicates(subset=["ts"]).sort_values("ts").tail(self.max_buffer_rows).reset_index(drop=True)

        balance_before = self._safe_fetch_balance()
        if balance_before is None:
            return self.last_obs.copy(), 0.0, self.kill_switch, True, {"action_taken": "hold", "api_error": True}
        usd_balance, btc_balance = self._extract_balances(balance_before)
        current_price = float(self.df.iloc[-1]["close"])

        trade_filled = False
        filled_price = 0.0
        trade_value = 0.0
        forced_hold_reward = False

        try:
            order_book = self.exchange.fetch_order_book(self.symbol, limit=20)
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
        except Exception:
            bids, asks = [], []

        if not self.kill_switch and bids and asks:
            target_btc_alloc: Optional[float] = None
            if action_raw > 0.3:
                target_btc_alloc = 1.0
            elif action_raw < -0.3:
                target_btc_alloc = 0.0

            btc_value = btc_balance * current_price
            portfolio_usd_before = btc_value + usd_balance
            btc_value_gap = 0.0
            if target_btc_alloc is not None:
                alloc_diff_threshold = 0.10 * portfolio_usd_before
                target_btc_value = portfolio_usd_before * target_btc_alloc
                btc_value_gap = target_btc_value - btc_value

            if target_btc_alloc is not None and portfolio_usd_before > 0 and abs(btc_value_gap) > alloc_diff_threshold:
                if btc_value_gap > 0:
                    quote_price = float(bids[0][0])
                    order_btc = btc_value_gap / max(quote_price, 1e-8)
                    max_affordable_btc = usd_balance / max(quote_price * (1.0 + MAKER_FEE), 1e-8)
                    order_btc = min(order_btc, max_affordable_btc)
                    if order_btc >= self.min_order_btc:
                        trade_filled, filled_price, canceled = self._execute_limit_order("buy", quote_price, order_btc)
                        if trade_filled:
                            trade_value = order_btc * max(filled_price, 1e-8)
                            self.last_action = "buy"
                            self.position_entry_price = filled_price
                            self.position_entry_step = self.step_count + 1
                            self.last_filled_trade_step = self.step_count + 1
                        elif canceled:
                            forced_hold_reward = True
                else:
                    quote_price = float(asks[0][0])
                    desired_sell_btc = abs(btc_value_gap) / max(quote_price, 1e-8)
                    order_btc = min(desired_sell_btc, btc_balance)
                    if order_btc >= self.min_order_btc:
                        trade_filled, filled_price, canceled = self._execute_limit_order("sell", quote_price, order_btc)
                        if trade_filled:
                            trade_value = order_btc * max(filled_price, 1e-8)
                            self.last_action = "sell"
                            self.position_entry_price = None
                            self.position_entry_step = None
                            self.last_filled_trade_step = self.step_count + 1
                        elif canceled:
                            forced_hold_reward = True

        balance_after = self._safe_fetch_balance()
        if balance_after is None:
            return self.last_obs.copy(), 0.0, self.kill_switch, True, {"action_taken": self.last_action, "api_error": True}
        usd_balance, btc_balance = self._extract_balances(balance_after)

        obs = self._compute_observation(usd_balance, btc_balance)
        portfolio_usd = self._get_portfolio_value(btc_balance, usd_balance, current_price)

        reward = 0.0
        if not forced_hold_reward:
            prev_price = max(self.prev_price, 1e-8)
            price_change = (current_price - prev_price) / prev_price
            btc_allocation = (btc_balance * current_price / portfolio_usd) if portfolio_usd > 0 else 0.0
            fee_if_traded = 0.0
            if trade_filled:
                fee_if_traded = REWARD_FEE_RATE * trade_value / max(portfolio_usd, 1e-8)
            reward = (2.0 * btc_allocation - 1.0) * price_change - fee_if_traded
        else:
            self.last_action = "hold"

        self.last_balance = portfolio_usd
        self.prev_price = current_price
        self.cumulative_reward += float(reward)
        self.last_obs = obs
        self.step_count += 1

        terminated = False
        if self.starting_portfolio_usd and portfolio_usd < 0.05 * self.starting_portfolio_usd:
            self.logger.critical(
                "Kill switch triggered: portfolio %.4f below 5%% of start %.4f",
                portfolio_usd,
                self.starting_portfolio_usd,
            )
            self.kill_switch = True
            terminated = True
            self._append_log_row(action_raw, float(reward), current_price, portfolio_usd, btc_balance, usd_balance, obs)
            sys.exit(1)

        self._append_log_row(action_raw, float(reward), current_price, portfolio_usd, btc_balance, usd_balance, obs)

        return obs, float(reward), terminated, False, {
            "action_taken": self.last_action,
            "portfolio_usd": portfolio_usd,
            "kill_switch": self.kill_switch,
            "fill_price": filled_price if trade_filled else None,
            "btc_allocation": (btc_balance * current_price / portfolio_usd) if portfolio_usd > 0 else 0.0,
        }
