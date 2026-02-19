"""
train.py - Contract checks and live training runner.

Usage:
    python train.py

This script validates that observation shape, feature count, and
column names are consistent with `KrakenLiveEnv` in env.py, then starts
live training with step-by-step console output.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from env import (
    BASE_OHLCV_COLUMNS,
    OBSERVATION_COLUMNS,
    OBSERVATION_SIZE,
    KrakenLiveEnv,
)

# Canonical column contracts used by the training/evaluation pipeline.
TRAIN_OHLCV_COLUMNS = ["ts", "open", "high", "low", "close", "vol"]
TRAIN_OBSERVATION_COLUMNS = [
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


class StepConsoleLogger(BaseCallback):
    """Print one console line for every environment step."""

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info: dict[str, Any] = self.locals["infos"][0]
        action_taken = info.get("action_taken", "unknown")
        portfolio = info.get("portfolio_usd", "n/a")

        print(
            "[train] "
            f"step={self.num_timesteps} "
            f"reward={reward:+.8f} "
            f"action={action_taken} "
            f"portfolio_usd={portfolio} "
            f"done={done}"
        )
        return True


def review_contracts() -> None:
    """Fail fast if any training contract drifts away from env.py."""
    if TRAIN_OHLCV_COLUMNS != BASE_OHLCV_COLUMNS:
        raise ValueError(
            "OHLCV column mismatch. "
            f"train.py={TRAIN_OHLCV_COLUMNS}, env.py={BASE_OHLCV_COLUMNS}"
        )

    if TRAIN_OBSERVATION_COLUMNS != OBSERVATION_COLUMNS:
        raise ValueError(
            "Observation column mismatch. "
            f"train.py={TRAIN_OBSERVATION_COLUMNS}, env.py={OBSERVATION_COLUMNS}"
        )

    if len(TRAIN_OBSERVATION_COLUMNS) != OBSERVATION_SIZE:
        raise ValueError(
            "Observation feature count mismatch. "
            f"train.py={len(TRAIN_OBSERVATION_COLUMNS)}, env.py={OBSERVATION_SIZE}"
        )

    # env.py binds observation_space shape to OBSERVATION_SIZE in KrakenLiveEnv.
    expected_shape = (OBSERVATION_SIZE,)
    if expected_shape != (len(OBSERVATION_COLUMNS),):
        raise ValueError(
            "Observation shape mismatch. "
            f"Expected {expected_shape}, got {(len(OBSERVATION_COLUMNS),)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live SAC training on KrakenLiveEnv")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training steps to run (default: 100000)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N steps (default: 10)",
    )
    return parser.parse_args()


def run_training(total_timesteps: int, checkpoint_every: int) -> None:
    print("[train] Creating KrakenLiveEnv (this performs 40k historical bar initialization)...")
    env = KrakenLiveEnv(max_buffer_rows=40_000)
    print(f"[train] Candle buffer initialized with {len(env.df)} rows.")

    latest_checkpoint = env.get_latest_checkpoint()
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"[train] Loading latest checkpoint: {latest_checkpoint}")
        model = SAC.load(latest_checkpoint, env=env)
    else:
        print("[train] No checkpoint found; creating new SAC model.")
        model = SAC("MlpPolicy", env, verbose=0)

    callback = StepConsoleLogger()
    print(f"[train] Starting training loop for {total_timesteps} timesteps...")

    for step in range(1, total_timesteps + 1):
        model.learn(total_timesteps=1, reset_num_timesteps=False, callback=callback)

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = env.save_checkpoint(model, step)
            print(f"[train] checkpoint saved: {ckpt_path}")


def main() -> None:
    args = parse_args()
    review_contracts()
    print("✅ env.py and train.py contracts match exactly.")
    run_training(total_timesteps=args.timesteps, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
