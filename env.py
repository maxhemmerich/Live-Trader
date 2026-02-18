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
import ta
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
        rsi_14, stoch_k, stoch_d, cci_20, williams_r_14, ema_20,
        macd, macd_signal, macd_hist,
        adx_14, bb_upper, bb_mid, bb_lower, atr_14, obv

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
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # Stochastic Oscillator (14, 3)
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # CCI (20-period)
    df["cci_20"] = ta.trend.CCIIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=20
    ).cci()

    # Williams %R (14-period)
    df["williams_r_14"] = ta.momentum.WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"], lbp=14
    ).williams_r()

    # EMA (20-period)
    df["ema_20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ADX (14-period)
    df["adx_14"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()

    # Bollinger Bands (20-period, 2 std)
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # ATR (14-period)
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # OBV
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
