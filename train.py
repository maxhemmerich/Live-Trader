"""
train.py - Contract checks between this file and env.py.

Usage:
    python train.py

This script intentionally validates that observation shape, feature count, and
column names are consistent with `KrakenLiveEnv` in env.py.
"""

from __future__ import annotations

from env import (
    BASE_OHLCV_COLUMNS,
    OBSERVATION_COLUMNS,
    OBSERVATION_SIZE,
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
    "eth_value_weight",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
    "usd_value_weight",
]


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


if __name__ == "__main__":
    review_contracts()
    print("✅ env.py and train.py contracts match exactly.")
