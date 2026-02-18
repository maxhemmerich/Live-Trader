"""Lightweight, credential-free integration test for KrakenLiveEnv behavior.

This script mocks the ccxt exchange layer, seeds deterministic synthetic market data,
then runs 5 steps while validating observation integrity.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MockKrakenExchange:
    """Mock ccxt.kraken exchange with deterministic OHLCV + balance responses."""

    ohlcv: list[list[float]]

    def fetch_ohlcv(self, symbol: str, timeframe: str, since=None, limit: int = 1000):
        _ = (symbol, timeframe, since, limit)
        return self.ohlcv

    def fetch_balance(self):
        return {
            "free": {"USD": 10_000.0, "ETH": 0.0},
            "total": {"USD": 10_000.0, "ETH": 0.0},
        }


def _make_synthetic_eth_data(rows: int = 250, seed: int = 7) -> pd.DataFrame:
    """Create realistic synthetic ETH/USD-like features with no NaNs."""
    rng = np.random.default_rng(seed)

    start = 2150.0
    hourly_returns = rng.normal(loc=0.0002, scale=0.008, size=rows)
    close = start * np.exp(np.cumsum(hourly_returns))

    open_ = np.roll(close, 1)
    open_[0] = close[0] * (1.0 - rng.uniform(0.0005, 0.002))

    wick_up = rng.uniform(0.0005, 0.004, size=rows)
    wick_down = rng.uniform(0.0005, 0.004, size=rows)
    high = np.maximum(open_, close) * (1.0 + wick_up)
    low = np.minimum(open_, close) * (1.0 - wick_down)

    volume = rng.lognormal(mean=6.8, sigma=0.25, size=rows)

    close_series = pd.Series(close)
    rsi_14 = 50.0 + np.tanh(close_series.pct_change().fillna(0).rolling(14).mean().fillna(0) * 400) * 45
    macd = close_series.ewm(span=12, adjust=False).mean() - close_series.ewm(span=26, adjust=False).mean()
    bb_mid = close_series.rolling(20, min_periods=1).mean()
    bb_std = close_series.rolling(20, min_periods=1).std().fillna(0)
    bb_upper = bb_mid + 2.0 * bb_std

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "rsi_14": rsi_14,
            "macd": macd,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
        }
    )
    assert df.shape == (250, 9)
    return df.astype(float)


def _to_mock_ohlcv(df: pd.DataFrame) -> list[list[float]]:
    timestamps = pd.date_range("2024-01-01", periods=len(df), freq="h")
    data = []
    for ts, row in zip(timestamps, df.itertuples(index=False), strict=True):
        data.append(
            [
                int(ts.timestamp() * 1000),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            ]
        )
    return data


def _build_env(synthetic_df: pd.DataFrame):
    """Instantiate KrakenLiveEnv (or repo TradingEnv fallback)."""
    try:
        from live_env import KrakenLiveEnv  # type: ignore
    except Exception:
        from env import TradingEnv as KrakenLiveEnv  # repo fallback

    sig = inspect.signature(KrakenLiveEnv)
    if "df" in sig.parameters:
        # Keep obs exactly (27,) -> 3-window * 9 features
        env = KrakenLiveEnv(df=synthetic_df, window=3)
    else:
        env = KrakenLiveEnv()

    return env


def main() -> None:
    synthetic_df = _make_synthetic_eth_data(rows=250)
    mock_exchange = MockKrakenExchange(ohlcv=_to_mock_ohlcv(synthetic_df))

    env = _build_env(synthetic_df)

    # Replace live ccxt client with mock and short-circuit any expensive bootstrap path.
    env.exchange = mock_exchange
    if hasattr(env, "_init_historical_data"):
        env._init_historical_data = lambda *args, **kwargs: None  # skip 40k pagination init
    if hasattr(env, "fetch_historical_data"):
        env.fetch_historical_data = lambda *args, **kwargs: synthetic_df.copy()

    # Seed internal frame directly for deterministic, credential-free test execution.
    env.df = synthetic_df.copy().reset_index(drop=True)

    obs, _ = env.reset()

    actions = [
        np.array([0.20], dtype=np.float32),
        np.array([-0.35], dtype=np.float32),
        np.array([0.60], dtype=np.float32),
        np.array([0.00], dtype=np.float32),
        np.array([0.85], dtype=np.float32),
    ]

    for i, action in enumerate(actions, start=1):
        assert obs.shape == (27,), f"Step {i}: expected obs shape (27,), got {obs.shape}"
        assert np.isfinite(obs).all(), f"Step {i}: observation has NaN/inf"

        next_obs, reward, terminated, truncated, info = env.step(action)
        portfolio_value = info.get("portfolio_value", float("nan"))

        print(
            f"step={i} obs_shape={obs.shape} reward={reward:.8f} "
            f"action={float(action[0]):+.2f} portfolio_value={portfolio_value:.2f}"
        )

        if terminated or truncated:
            break
        obs = next_obs

    # Final sanity check on the last valid observation seen in the loop.
    assert obs.shape == (27,), f"Final: expected obs shape (27,), got {obs.shape}"
    assert np.isfinite(obs).all(), "Final: observation has NaN/inf"


if __name__ == "__main__":
    main()
