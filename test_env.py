"""Basic smoke entrypoint for the KrakenLiveEnv environment."""

import numpy as np
import pandas as pd

import types
import sys

try:
    import env as env_module
except ModuleNotFoundError as exc:
    if "lighter_client" not in str(exc):
        raise
    ccxt_stub = types.ModuleType("ccxt")

    class _StubKraken:
        def __init__(self, *_args, **_kwargs):
            pass

    ccxt_stub.kraken = _StubKraken
    sys.modules["ccxt"] = ccxt_stub
    import env as env_module

from env import (
    ETH_FEATURE_COLUMNS,
    LAG_VALUES,
    LR_CHANNEL_COLUMNS,
    OBSERVATION_SIZE,
    SCALAR_COLUMNS,
    TREND_LAG_COLUMNS,
    TREND_PREFIXES,
    KrakenLiveEnv,
)


def _print_observation_size_delta() -> None:
    old_trend_lags = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072, 262144, 524288, 1048576]
    old_counts = {
        "price_lags": len(LAG_VALUES),
        "vol_lags": len(LAG_VALUES),
        "scalar": len(SCALAR_COLUMNS),
        "trend_lags": len(TREND_PREFIXES) * len(old_trend_lags),
        "lr_channels": 0,
        "order_book": 5,
        "portfolio": 2,
        "self_awareness": 3,
        "time": 6,
        "eth": 3,
    }
    new_counts = {
        "price_lags": len(LAG_VALUES),
        "vol_lags": len(LAG_VALUES),
        "scalar": len(SCALAR_COLUMNS),
        "trend_lags": len(TREND_LAG_COLUMNS),
        "lr_channels": len(LR_CHANNEL_COLUMNS),
        "order_book": 5,
        "portfolio": 2,
        "self_awareness": 3,
        "time": 6,
        "eth": len(ETH_FEATURE_COLUMNS),
    }
    old_size = sum(old_counts.values())
    new_size = sum(new_counts.values())
    print(f"old observation size: {old_size}")
    print(f"new observation size: {new_size}")
    print("feature breakdown delta (new - old):")
    for key in old_counts:
        delta = new_counts[key] - old_counts[key]
        print(f"  {key}: {new_counts[key]} (delta {delta:+d})")



def main() -> None:
    env_module.time.sleep = lambda *_args, **_kwargs: None

    _print_observation_size_delta()

    env = KrakenLiveEnv(candle_interval=1, timeframe="1m")

    synthetic_df = pd.DataFrame(
        [
            [1700000000000 + i * 60000, 2000.0 + i, 2005.0 + i, 1995.0 + i, 2002.0 + i, 10.0 + i]
            for i in range(500)
        ],
        columns=env_module.BASE_OHLCV_COLUMNS,
    )

    class MockExchange:
        def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=3, params=None):
            del symbol, timeframe, since, params
            rows = synthetic_df[env_module.BASE_OHLCV_COLUMNS].values.tolist()
            return rows[-limit:]

        def fetch_balance(self):
            return {
                "free": {"XBT": 0.05, "USD": 135.0},
                "total": {"XBT": 0.05, "USD": 135.0},
            }

        def fetch_order_book(self, symbol, limit=20):
            del symbol
            mid = float(synthetic_df.iloc[-1]["close"])
            bid_levels = [[mid - (i + 1) * 0.5, 1.5 + i * 0.2] for i in range(limit)]
            ask_levels = [[mid + (i + 1) * 0.5, 1.4 + i * 0.2] for i in range(limit)]
            return {"bids": bid_levels, "asks": ask_levels}

        def create_limit_buy_order(self, symbol, amount, price):
            del symbol, amount, price
            return {"id": "mock_order", "status": "closed", "average": 1955.00, "price": 1955.00}

        def create_limit_sell_order(self, symbol, amount, price):
            del symbol, amount, price
            return {"id": "mock_order", "status": "closed", "average": 1955.00, "price": 1955.00}

        def fetch_order(self, order_id, symbol):
            del order_id, symbol
            return {"id": "mock_order", "status": "closed", "average": 1955.00, "price": 1955.00}

        def cancel_order(self, order_id, symbol):
            del order_id, symbol
            return {"id": "mock_order", "status": "canceled"}

    env.exchange = MockExchange()
    env.df = synthetic_df.copy()

    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Reset info: {info}")

    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.shape == env.observation_space.shape
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()

    for step_num in range(1, 6):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        has_nan = bool(np.isnan(obs).any())
        has_inf = bool(np.isinf(obs).any())
        action_taken = info.get("action_taken", "unknown")

        print(
            f"Step {step_num}: "
            f"obs_shape={obs.shape}, "
            f"has_nan_or_inf={has_nan or has_inf}, "
            f"reward={reward:.8f}, "
            f"action_taken={action_taken}"
        )

        assert obs.shape == (OBSERVATION_SIZE,)
        assert obs.shape == env.observation_space.shape
        assert not has_nan, f"NaN detected in observation at step {step_num}"
        assert not has_inf, f"Inf detected in observation at step {step_num}"
        assert isinstance(reward, float)
        assert isinstance(action_taken, str)

        if terminated or truncated:
            break

    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
