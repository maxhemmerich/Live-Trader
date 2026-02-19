"""Kraken live trading environment using indicators from the ta library."""

import glob
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import gymnasium as gym
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from dotenv import load_dotenv
from gymnasium import spaces


BASE_OHLCV_COLUMNS = ["ts", "open", "high", "low", "close", "vol"]
FEATURE_COLUMNS = [
    "rsi_14_norm",
    "rsi_7_norm",
    "rsi_21_norm",
    "stoch_k_norm",
    "stoch_d_norm",
    "cci_20_clipped",
    "willr_14_norm",
    "close_vs_ema_9",
    "close_vs_ema_20",
    "close_vs_ema_50",
    "close_vs_ema_200",
    "macd_hist_over_atr_14",
    "adx_14_norm",
    "bb20_width_over_close",
    "bb50_width_over_close",
    "bb20_position",
    "atr_14_over_close",
    "vol_over_vol_sma_20",
    "obv_pct_change",
    "return_10_clipped",
    "return_50_clipped",
    "return_200_clipped",
    "realized_vol_20_norm",
    "bid_ask_spread_frac",
    "bid_depth_5_over_vol_sma_20",
    "ask_depth_5_over_vol_sma_20",
    "bid_ask_imbalance",
    "price_vs_best_bid",
    "eth_value_weight",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
    "usd_value_weight",
]
OBSERVATION_COLUMNS = FEATURE_COLUMNS
OBSERVATION_SIZE = len(FEATURE_COLUMNS)


class KrakenLiveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol: str = "ETH/USD",
        timeframe: str = "1m",
        candle_limit: int = 720,
        max_buffer_rows: int = 40_000,
        checkpoint_dir: str = "./checkpoints/",
        trading_log_path: str = "trading_log.csv",
    ) -> None:
        super().__init__()
        load_dotenv()

        self.symbol = symbol
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self.max_buffer_rows = max_buffer_rows
        self.checkpoint_dir = checkpoint_dir
        self.trading_log_path = trading_log_path
        self.trade_size_eth = 0.001
        self.taker_fee = 0.0026

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        api_key = os.getenv("KRAKEN_API_KEY", "")
        api_secret = os.getenv("KRAKEN_API_SECRET", "")
        self.exchange = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
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
        self.df = self._initialize_candle_buffer()
        self.last_obs = np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        self.last_action = "hold"
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.starting_portfolio_usd: Optional[float] = None
        self.last_balance = 0.0

        self._init_log_file()

    def _initialize_candle_buffer(self) -> pd.DataFrame:
        all_bars = []
        since = None

        while len(all_bars) < self.max_buffer_rows:
            try:
                batch = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    since=since,
                    limit=self.candle_limit,
                )
                self.consecutive_errors = 0
            except Exception as exc:
                self._record_api_error(exc)
                break

            if not batch:
                break

            all_bars = batch + all_bars
            since = batch[0][0] - 60_000

            time.sleep(1)

            if len(all_bars) % (self.candle_limit * 10) == 0:
                print(f"[INIT] Collected {len(all_bars)} bars so far...")

        df = pd.DataFrame(all_bars, columns=BASE_OHLCV_COLUMNS)
        df.drop_duplicates(subset=["ts"], inplace=True)
        df.sort_values("ts", inplace=True)
        df = df.tail(self.max_buffer_rows)
        df.reset_index(drop=True, inplace=True)
        print(f"[INIT] Buffer ready: {len(df)} bars")
        return df

    def _record_api_error(self, exc: Exception) -> None:
        self.consecutive_errors += 1
        now = datetime.now(timezone.utc).isoformat()
        self.logger.error("%s | API error: %s", now, str(exc))
        if self.consecutive_errors >= 5:
            self.logger.critical("Kill switch triggered after 5 consecutive API errors.")
            self.kill_switch = True

    def _safe_fetch_balance(self) -> Optional[dict]:
        try:
            balance = self.exchange.fetch_balance()
            self.consecutive_errors = 0
            return balance
        except Exception as exc:
            self._record_api_error(exc)
            return None

    @staticmethod
    def _extract_balances(balance: dict) -> tuple[float, float]:
        usd_balance = float(balance.get("free", {}).get("USD", 0.0))
        eth_balance = float(balance.get("free", {}).get("ETH", 0.0))

        if usd_balance == 0.0:
            usd_balance = float(balance.get("total", {}).get("USD", 0.0))
        if eth_balance == 0.0:
            eth_balance = float(balance.get("total", {}).get("ETH", 0.0))

        return usd_balance, eth_balance

    def _get_total_balance(
        self,
        eth_balance: float,
        usd_balance: float,
        eth_price: float,
        context: str,
    ) -> float:
        total_balance = (eth_balance * eth_price) + usd_balance
        print(
            "[BALANCE DEBUG] "
            f"context={context} | "
            f"eth_balance={eth_balance:.8f}, "
            f"usd_balance={usd_balance:.8f}, "
            f"eth_price={eth_price:.8f}, "
            f"total_usd={total_balance:.8f}"
        )
        return total_balance

    def _compute_observation(self, usd_balance: float, eth_balance: float) -> np.ndarray:
        df = self.df.copy()

        df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
        df["rsi_7"] = RSIIndicator(close=df["close"], window=7).rsi()
        df["rsi_21"] = RSIIndicator(close=df["close"], window=21).rsi()

        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
            smooth_window=3,
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["cci_20"] = CCIIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=20,
        ).cci()
        df["willr_14"] = WilliamsRIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            lbp=14,
        ).williams_r()

        df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
        df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()

        df["macd_hist"] = MACD(
            close=df["close"],
            window_fast=12,
            window_slow=26,
            window_sign=9,
        ).macd_diff()

        df["adx_14"] = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
        ).adx()

        bb20 = BollingerBands(close=df["close"], window=20, window_dev=2)
        bb50 = BollingerBands(close=df["close"], window=50, window_dev=2)

        df["bb20_u"] = bb20.bollinger_hband()
        df["bb20_l"] = bb20.bollinger_lband()
        df["bb20_p"] = bb20.bollinger_pband()

        df["bb50_u"] = bb50.bollinger_hband()
        df["bb50_l"] = bb50.bollinger_lband()

        df["atr_14"] = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
        ).average_true_range()

        df["vol_sma_20"] = df["vol"].rolling(20).mean()
        df["obv"] = OnBalanceVolumeIndicator(
            close=df["close"],
            volume=df["vol"],
        ).on_balance_volume()
        df["obv_pct"] = df["obv"].pct_change(periods=1)

        last = df.iloc[-1]
        price = float(last["close"])
        total_usd = self._get_total_balance(
            eth_balance=eth_balance,
            usd_balance=usd_balance,
            eth_price=price,
            context="compute_observation",
        )
        total_usd = max(total_usd, 1e-9)

        ts = pd.to_datetime(int(last["ts"]), unit="ms", utc=True)
        hour = ts.hour
        day_of_week = ts.dayofweek
        day_of_month = min(ts.day, 30)

        return_10 = np.nan_to_num(
            np.clip((price - float(df["close"].iloc[-11])) / max(float(df["close"].iloc[-11]), 1e-9), -0.1, 0.1),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return_50 = np.nan_to_num(
            np.clip((price - float(df["close"].iloc[-51])) / max(float(df["close"].iloc[-51]), 1e-9), -0.2, 0.2),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return_200 = np.nan_to_num(
            np.clip((price - float(df["close"].iloc[-201])) / max(float(df["close"].iloc[-201]), 1e-9), -0.5, 0.5),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        one_period_returns = df["close"].pct_change().tail(20)
        realized_vol_20 = np.nan_to_num(
            float(one_period_returns.std()) / 0.02,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        bid_ask_spread_frac = 0.0
        bid_depth_5_over_vol_sma_20 = 0.0
        ask_depth_5_over_vol_sma_20 = 0.0
        bid_ask_imbalance = 0.0
        price_vs_best_bid = 0.0
        try:
            order_book = self.exchange.fetch_order_book("ETH/USD", limit=20)
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if bids and asks:
                best_bid_price = float(bids[0][0])
                best_ask_price = float(asks[0][0])
                mid_price = (best_bid_price + best_ask_price) / 2.0

                bid_depth_5 = float(sum(level[1] for level in bids[:5]))
                ask_depth_5 = float(sum(level[1] for level in asks[:5]))
                vol_sma_20 = max(float(last["vol_sma_20"]), 1e-9)

                bid_ask_spread_frac = np.nan_to_num(
                    (best_ask_price - best_bid_price) / max(mid_price, 1e-9),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                bid_depth_5_over_vol_sma_20 = np.nan_to_num(
                    bid_depth_5 / vol_sma_20,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                ask_depth_5_over_vol_sma_20 = np.nan_to_num(
                    ask_depth_5 / vol_sma_20,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                bid_ask_imbalance = np.nan_to_num(
                    (bid_depth_5 - ask_depth_5) / max(bid_depth_5 + ask_depth_5, 1e-9),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                price_vs_best_bid = np.nan_to_num(
                    (price - best_bid_price) / max(price, 1e-9),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
        except Exception:
            pass

        obs = np.array(
            [
                float(last["rsi_14"]) / 100.0,
                float(last["rsi_7"]) / 100.0,
                float(last["rsi_21"]) / 100.0,
                float(last["stoch_k"]) / 100.0,
                float(last["stoch_d"]) / 100.0,
                float(np.clip(float(last["cci_20"]) / 200.0, -1.0, 1.0)),
                (float(last["willr_14"]) + 100.0) / 100.0,
                (price / float(last["ema_9"])) - 1.0,
                (price / float(last["ema_20"])) - 1.0,
                (price / float(last["ema_50"])) - 1.0,
                (price / float(last["ema_200"])) - 1.0,
                float(last["macd_hist"]) / max(float(last["atr_14"]), 1e-9),
                float(last["adx_14"]) / 100.0,
                (float(last["bb20_u"]) - float(last["bb20_l"])) / max(price, 1e-9),
                (float(last["bb50_u"]) - float(last["bb50_l"])) / max(price, 1e-9),
                float(last["bb20_p"]),
                float(last["atr_14"]) / max(price, 1e-9),
                float(last["vol"]) / max(float(last["vol_sma_20"]), 1e-9),
                float(last["obv_pct"]),
                float(return_10),
                float(return_50),
                float(return_200),
                float(realized_vol_20),
                float(bid_ask_spread_frac),
                float(bid_depth_5_over_vol_sma_20),
                float(ask_depth_5_over_vol_sma_20),
                float(bid_ask_imbalance),
                float(price_vs_best_bid),
                (eth_balance * price) / total_usd,
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.sin(2 * np.pi * day_of_week / 7.0),
                np.cos(2 * np.pi * day_of_week / 7.0),
                np.sin(2 * np.pi * day_of_month / 30.0),
                np.cos(2 * np.pi * day_of_month / 30.0),
                usd_balance / total_usd,
            ],
            dtype=np.float32,
        )

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return obs

    def _init_log_file(self) -> None:
        if os.path.exists(self.trading_log_path):
            return

        header = [
            "timestamp",
            "step",
            "action_raw",
            "action_taken",
            "reward",
            "cumulative_reward",
            "eth_price",
            "portfolio_usd",
            "eth_balance",
            "usd_balance",
        ] + FEATURE_COLUMNS

        with open(self.trading_log_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

    def _append_log_row(
        self,
        action_raw: float,
        reward: float,
        eth_price: float,
        portfolio_usd: float,
        eth_balance: float,
        usd_balance: float,
        obs: np.ndarray,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        row = [
            now,
            str(self.step_count),
            f"{action_raw:.8f}",
            self.last_action,
            f"{reward:.10f}",
            f"{self.cumulative_reward:.10f}",
            f"{eth_price:.8f}",
            f"{portfolio_usd:.8f}",
            f"{eth_balance:.8f}",
            f"{usd_balance:.8f}",
        ] + [f"{float(x):.8f}" for x in obs]

        with open(self.trading_log_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")

    def save_checkpoint(self, model, step: int) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path_no_ext = os.path.join(self.checkpoint_dir, f"sac_step_{step}")
        model.save(path_no_ext)

        checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "sac_step_*.zip")),
            key=os.path.getmtime,
            reverse=True,
        )
        for old in checkpoints[5:]:
            os.remove(old)

        return f"{path_no_ext}.zip"

    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "sac_step_*.zip")),
            key=os.path.getmtime,
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        balance = self._safe_fetch_balance()
        if balance is None:
            obs = self.last_obs.copy()
            return obs, {}

        usd_balance, eth_balance = self._extract_balances(balance)

        if self.df.empty:
            obs = np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        else:
            obs = self._compute_observation(usd_balance, eth_balance)

        self.last_obs = obs

        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0
        total_usd = self._get_total_balance(
            eth_balance=eth_balance,
            usd_balance=usd_balance,
            eth_price=price,
            context="reset",
        )

        self.last_balance = total_usd
        self.starting_portfolio_usd = (eth_balance * price) + usd_balance
        print(
            "[INIT] Starting portfolio value: "
            f"${self.starting_portfolio_usd:.2f} "
            f"(ETH: {eth_balance:.4f} × ${price:.2f} + USD: ${usd_balance:.2f})"
        )
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.last_action = "hold"
        self.kill_switch = False
        self.consecutive_errors = 0

        return obs, {}

    def step(self, action):
        time.sleep(60)

        action_raw = float(np.array(action).reshape(-1)[0])
        self.last_action = "hold"
        if self.kill_switch:
            return self.last_obs.copy(), 0.0, True, False, {
                "kill_switch": True,
                "action_taken": self.last_action,
            }

        try:
            recent = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                limit=3,
            )
            self.consecutive_errors = 0
        except Exception as exc:
            self._record_api_error(exc)
            terminated = self.kill_switch
            return self.last_obs.copy(), 0.0, terminated, True, {
                "api_error": True,
                "action_taken": self.last_action,
            }

        if recent:
            recent_df = pd.DataFrame(
                recent, columns=BASE_OHLCV_COLUMNS
            )
            self.df = pd.concat([self.df, recent_df], ignore_index=True)
            self.df = (
                self.df.drop_duplicates(subset=["ts"])
                .sort_values("ts")
                .tail(self.max_buffer_rows)
                .reset_index(drop=True)
            )

        balance = self._safe_fetch_balance()
        if balance is None:
            terminated = self.kill_switch
            return self.last_obs.copy(), 0.0, terminated, True, {
                "api_error": True,
                "action_taken": self.last_action,
            }

        usd_balance, eth_balance = self._extract_balances(balance)
        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0

        trade_executed = False
        if not self.kill_switch:
            if action_raw > 0.5:
                required = self.trade_size_eth * price * (1.0 + self.taker_fee)
                if usd_balance >= required:
                    try:
                        self.exchange.create_market_buy_order(self.symbol, self.trade_size_eth)
                        self.consecutive_errors = 0
                        self.last_action = "buy"
                        trade_executed = True
                    except Exception as exc:
                        self._record_api_error(exc)
                else:
                    self.last_action = "hold"
            elif action_raw < -0.5 and eth_balance >= self.trade_size_eth:
                try:
                    self.exchange.create_market_sell_order(self.symbol, self.trade_size_eth)
                    self.consecutive_errors = 0
                    self.last_action = "sell"
                    trade_executed = True
                except Exception as exc:
                    self._record_api_error(exc)

        balance_after = self._safe_fetch_balance()
        if balance_after is None:
            terminated = self.kill_switch
            return self.last_obs.copy(), 0.0, terminated, True, {
                "api_error": True,
                "action_taken": self.last_action,
            }

        usd_balance, eth_balance = self._extract_balances(balance_after)
        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0

        obs = self._compute_observation(usd_balance, eth_balance)
        portfolio_usd = self._get_total_balance(
            eth_balance=eth_balance,
            usd_balance=usd_balance,
            eth_price=price,
            context=f"step_{self.step_count + 1}",
        )

        prev_balance = max(self.last_balance, 1e-9)
        reward = (portfolio_usd - prev_balance) / prev_balance
        if trade_executed:
            reward -= self.taker_fee

        self.last_balance = portfolio_usd
        self.cumulative_reward += float(reward)
        self.last_obs = obs
        self.step_count += 1

        terminated = False
        if (
            self.starting_portfolio_usd is not None
            and portfolio_usd < 0.6 * self.starting_portfolio_usd
        ):
            self.logger.critical(
                "Kill switch triggered: portfolio %.4f below 60%% of start %.4f",
                portfolio_usd,
                self.starting_portfolio_usd,
            )
            self.kill_switch = True
            terminated = True

        if self.kill_switch and self.consecutive_errors >= 5:
            terminated = True

        self._append_log_row(
            action_raw=action_raw,
            reward=float(reward),
            eth_price=price,
            portfolio_usd=portfolio_usd,
            eth_balance=eth_balance,
            usd_balance=usd_balance,
            obs=obs,
        )

        return obs, float(reward), terminated, False, {
            "action_taken": self.last_action,
            "portfolio_usd": portfolio_usd,
            "kill_switch": self.kill_switch,
        }
