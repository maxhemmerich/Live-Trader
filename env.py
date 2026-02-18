"""
env.py - Custom Gymnasium Trading Environment for SAC Live Trader.

Wraps OHLCV data fetched from Kraken via ccxt.
Observation space: recent OHLCV + RSI, MACD, Bollinger Bands.
Action space: continuous [-1, 1] (short=-1, neutral=0, long=+1).
Reward: step-by-step portfolio log return (PnL-based).
"""

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
from gymnasium import spaces
import ccxt
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────
SYMBOL       = "BTC/USD"
TIMEFRAME    = "1h"
WINDOW       = 20        # Number of past candles in each observation
INITIAL_CASH = 10_000.0
TRADING_FEE  = 0.0026    # Kraken taker fee (0.26 %)
FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_mid",
    "bb_lower",
]


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for SAC-based crypto trading.

    Observation:
        Flattened array of shape (WINDOW * n_features,) where n_features is
        the number of OHLCV + indicator columns after dropping NaNs.

    Action:
        Box([-1], [1], float32) — target portfolio fraction.
        -1 = 100 % short, 0 = flat, +1 = 100 % long.

    Reward:
        Logarithmic portfolio return for the step, penalised by transaction
        cost proportional to position change.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window: int = WINDOW):
        super().__init__()

        if list(df.columns) != FEATURE_COLUMNS:
            raise ValueError(
                "TradingEnv expects exact feature columns/order: "
                f"{FEATURE_COLUMNS}; got {list(df.columns)}"
            )

        self.df     = df.reset_index(drop=True)
        self.window = window

        self.n_features = len(self.df.columns)

        # ── Spaces ─────────────────────────────────────────────────────────
        obs_dim = self.window * self.n_features
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (obs_dim,),
            dtype = np.float32,
        )
        self.action_space = spaces.Box(
            low   = np.array([-1.0], dtype=np.float32),
            high  = np.array([ 1.0], dtype=np.float32),
            dtype = np.float32,
        )

        # ── Portfolio state ────────────────────────────────────────────────
        self.portfolio    = INITIAL_CASH
        self.position     = 0.0
        self.current_step = self.window

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return flattened window of features ending at current_step."""
        window_df = self.df.iloc[self.current_step - self.window : self.current_step]
        return window_df.values.astype(np.float32).flatten()

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.portfolio    = INITIAL_CASH
        self.position     = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        new_position = float(np.clip(action[0], -1.0, 1.0))

        price_now  = float(self.df.iloc[self.current_step]["close"])
        price_prev = float(self.df.iloc[self.current_step - 1]["close"])

        # Transaction cost on position change
        position_delta = abs(new_position - self.position)
        fee = position_delta * TRADING_FEE * self.portfolio

        # PnL from holding the previous position through this candle
        price_return = (price_now - price_prev) / price_prev
        pnl          = self.position * price_return * self.portfolio
        prev_value   = self.portfolio
        self.portfolio = self.portfolio + pnl - fee

        # Log-return reward
        reward = float(np.log(max(self.portfolio, 1e-8) / max(prev_value, 1e-8)))

        self.position = new_position
        self.current_step += 1

        terminated = self.current_step >= len(self.df)
        truncated  = False

        obs = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._get_obs()
        )
        info = {
            "portfolio_value": self.portfolio,
            "position":        self.position,
            "price":           price_now,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step {self.current_step:5d} | "
            f"Portfolio: ${self.portfolio:10.2f} | "
            f"Position: {self.position:+.2f}"
        )

    def close(self):
        pass


# ── Data helper ────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str    = SYMBOL,
    timeframe: str = TIMEFRAME,
    limit: int     = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Kraken via ccxt and compute technical indicators.

    Returns a cleaned DataFrame with columns:
        open, high, low, close, volume,
        rsi_14,
        macd, macd_signal, macd_hist,
        bb_upper, bb_mid, bb_lower

    All rows with NaN values (indicator warm-up period) are dropped.
    """
    api_key    = os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET")

    exchange = ccxt.kraken({
        "apiKey":          api_key,
        "secret":          api_secret,
        "enableRateLimit": True,
    })

    raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df  = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # RSI (14-period)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # MACD (12, 26, 9)
    macd_df           = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd_df["MACD_12_26_9"]
    df["macd_signal"] = macd_df["MACDs_12_26_9"]
    df["macd_hist"]   = macd_df["MACDh_12_26_9"]

    # Bollinger Bands (20-period, 2 std)
    bb_df          = ta.bbands(df["close"], length=20, std=2.0)
    df["bb_upper"] = bb_df["BBU_20_2.0"]
    df["bb_mid"]   = bb_df["BBM_20_2.0"]
    df["bb_lower"] = bb_df["BBL_20_2.0"]

    df.dropna(inplace=True)
    df = df[FEATURE_COLUMNS].copy()
    df.reset_index(drop=True, inplace=True)

    return df
