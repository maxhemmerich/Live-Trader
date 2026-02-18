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
import pandas_ta as ta
from dotenv import load_dotenv
from gymnasium import spaces


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
            shape=(27,),
            dtype=np.float32,
        )

        self.df = self._initialize_candle_buffer()
        self.last_obs = np.zeros((27,), dtype=np.float32)
        self.last_action = "hold"
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.kill_switch = False
        self.consecutive_errors = 0
        self.starting_portfolio_usd: Optional[float] = None
        self.last_balance = 0.0

        self._init_log_file()

    def _initialize_candle_buffer(self) -> pd.DataFrame:
        pages = []
        page_num = 0

        try:
            latest = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                limit=self.candle_limit,
            )
            self.consecutive_errors = 0
        except Exception as exc:
            self._record_api_error(exc)
            return pd.DataFrame(
                columns=["ts", "open", "high", "low", "close", "vol"]
            )

        if not latest:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])

        pages.append(latest)
        earliest_ts = latest[0][0]
        page_num += 1

        while sum(len(p) for p in pages) < self.max_buffer_rows:
            since = earliest_ts - (self.candle_limit * 60 * 1000)
            try:
                older = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=self.candle_limit,
                )
                self.consecutive_errors = 0
            except Exception as exc:
                self._record_api_error(exc)
                break

            if not older:
                break

            older_only = [row for row in older if row[0] < earliest_ts]
            if not older_only:
                break

            pages.append(older_only)
            earliest_ts = older_only[0][0]
            page_num += 1

            if page_num % 10 == 0:
                current_rows = sum(len(p) for p in pages)
                print(
                    f"Init fetch progress: pages={page_num}, rows={current_rows}, earliest_ts={earliest_ts}"
                )

            time.sleep(1)

        all_rows = [row for page in pages for row in page]
        df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "vol"])
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").tail(self.max_buffer_rows)
        df = df.reset_index(drop=True)
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

    def _compute_observation(self, usd_balance: float, eth_balance: float) -> np.ndarray:
        df = self.df.copy()

        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)
        df["rsi_21"] = ta.rsi(df["close"], length=21)

        stoch = ta.stoch(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

        df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        df["willr_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd_hist"] = macd["MACDh_12_26_9"]

        df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

        bb20 = ta.bbands(df["close"], length=20, std=2.0)
        bb50 = ta.bbands(df["close"], length=50, std=2.0)

        df["bb20_u"] = bb20["BBU_20_2.0"]
        df["bb20_l"] = bb20["BBL_20_2.0"]
        df["bb20_p"] = bb20["BBP_20_2.0"]

        df["bb50_u"] = bb50["BBU_50_2.0"]
        df["bb50_l"] = bb50["BBL_50_2.0"]

        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        df["vol_sma_20"] = df["vol"].rolling(20).mean()
        df["obv"] = ta.obv(df["close"], df["vol"])
        df["obv_pct"] = df["obv"].pct_change(periods=1)

        last = df.iloc[-1]
        price = float(last["close"])
        total_usd = eth_balance * price + usd_balance
        total_usd = max(total_usd, 1e-9)

        ts = pd.to_datetime(int(last["ts"]), unit="ms", utc=True)
        hour = ts.hour
        day_of_week = ts.dayofweek
        day_of_month = min(ts.day, 30)

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
        ] + [f"obs_{i}" for i in range(27)]

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
            obs = np.zeros((27,), dtype=np.float32)
        else:
            obs = self._compute_observation(usd_balance, eth_balance)

        self.last_obs = obs

        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0
        total_usd = eth_balance * price + usd_balance

        self.last_balance = total_usd
        self.starting_portfolio_usd = total_usd
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.last_action = "hold"
        self.kill_switch = False
        self.consecutive_errors = 0

        return obs, {}

    def step(self, action):
        time.sleep(60)

        action_raw = float(np.array(action).reshape(-1)[0])
        if self.kill_switch:
            return self.last_obs.copy(), 0.0, True, False, {"kill_switch": True}

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
            return self.last_obs.copy(), 0.0, terminated, True, {"api_error": True}

        if recent:
            recent_df = pd.DataFrame(
                recent, columns=["ts", "open", "high", "low", "close", "vol"]
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
            return self.last_obs.copy(), 0.0, terminated, True, {"api_error": True}

        usd_balance, eth_balance = self._extract_balances(balance)
        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0

        self.last_action = "hold"
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
            return self.last_obs.copy(), 0.0, terminated, True, {"api_error": True}

        usd_balance, eth_balance = self._extract_balances(balance_after)
        price = float(self.df.iloc[-1]["close"]) if not self.df.empty else 0.0

        obs = self._compute_observation(usd_balance, eth_balance)
        portfolio_usd = eth_balance * price + usd_balance

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
