"""Basic smoke entrypoint for the KrakenLiveEnv environment."""

import numpy as np
import pandas as pd

import env as env_module
from env import OBSERVATION_SIZE, KrakenLiveEnv


def main() -> None:
    env_module.time.sleep = lambda *_args, **_kwargs: None

    env = KrakenLiveEnv()

    synthetic_df = pd.DataFrame(
        [
            [1700000000000 + i * 60000, 2000.0 + i, 2005.0 + i, 1995.0 + i, 2002.0 + i, 10.0 + i]
            for i in range(500)
        ],
        columns=env_module.BASE_OHLCV_COLUMNS,
    )

    class MockExchange:
        def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=3):
            del symbol, timeframe, since
            rows = synthetic_df[env_module.BASE_OHLCV_COLUMNS].values.tolist()
            return rows[-limit:]

        def fetch_balance(self):
            return {
                "free": {"ETH": 0.05, "USD": 135.0},
                "total": {"ETH": 0.05, "USD": 135.0},
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
