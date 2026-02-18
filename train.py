"""
train.py - SAC training script for the Live Trader.

Usage:
    python train.py

Steps:
    1. Load .env credentials.
    2. Fetch historical OHLCV data from Kraken.
    3. Build the TradingEnv (80 % train / 20 % eval split).
    4. Validate the environment with SB3's check_env.
    5. Train SAC with periodic checkpoint and eval callbacks.
    6. Save the final model to checkpoints/.
    7. Run a deterministic evaluation episode and write trading_log.csv.
"""

import os
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env import FEATURE_COLUMNS, TradingEnv, fetch_ohlcv

load_dotenv()

# ── Hyperparameters ────────────────────────────────────────────────────────
SYMBOL          = "BTC/USD"
TIMEFRAME       = "1h"
LIMIT           = 1000         # Historical candles to fetch
TOTAL_TIMESTEPS = 100_000      # SAC training steps
CHECKPOINT_FREQ = 10_000       # Save model every N steps
EVAL_FREQ       = 5_000        # Evaluate every N steps
CHECKPOINT_DIR  = "checkpoints"
LOG_CSV         = "trading_log.csv"


def main():
    print("=== SAC Live Trader — Training ===")
    print(f"Symbol: {SYMBOL}  |  Timeframe: {TIMEFRAME}  |  Candles: {LIMIT}")

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    print("\n[1/5] Fetching OHLCV data from Kraken...")
    df = fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    print(f"      Fetched {len(df)} candles with {len(df.columns)} features.")

    # ── 2. Train / eval split ──────────────────────────────────────────────
    split    = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    # ── 3. Build environments ──────────────────────────────────────────────
    print("[2/5] Building environments...")
    train_env = TradingEnv(train_df)
    eval_env  = TradingEnv(eval_df)
    _validate_feature_contract(df, train_env, eval_env)

    # ── 4. Validate environment ────────────────────────────────────────────
    print("[3/5] Validating environment...")
    check_env(train_env, warn=True)
    print("      Environment OK.")

    # ── 5. Configure callbacks ─────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "sac_trader",
        verbose     = 1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = CHECKPOINT_DIR,
        log_path             = CHECKPOINT_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 3,
        deterministic        = True,
        verbose              = 1,
    )

    # ── 6. Instantiate SAC ─────────────────────────────────────────────────
    print("[4/5] Initialising SAC model...")
    model = SAC(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        buffer_size     = 100_000,
        batch_size      = 256,
        tau             = 0.005,
        gamma           = 0.99,
        train_freq      = 1,
        gradient_steps  = 1,
        verbose         = 1,
        tensorboard_log = os.path.join(CHECKPOINT_DIR, "tb_logs"),
    )

    # ── 7. Train ───────────────────────────────────────────────────────────
    print(f"[5/5] Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_callback, eval_callback],
        progress_bar    = True,
    )

    # ── 8. Save final model ────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "sac_trader_final")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")

    # ── 9. Evaluation run → trading_log.csv ───────────────────────────────
    print(f"\nRunning evaluation and writing {LOG_CSV} ...")
    _run_eval_and_log(model, eval_env, LOG_CSV)
    print(f"Log written to {LOG_CSV}")


def _validate_feature_contract(df, train_env: TradingEnv, eval_env: TradingEnv) -> None:
    """Fail fast if data/env feature contract drifts across files."""
    if list(df.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Fetched dataframe columns do not match expected feature contract. "
            f"Expected {FEATURE_COLUMNS}, got {list(df.columns)}"
        )

    expected_features = len(FEATURE_COLUMNS)
    if train_env.n_features != expected_features or eval_env.n_features != expected_features:
        raise ValueError(
            "Feature count mismatch. "
            f"Expected {expected_features}, got train={train_env.n_features}, "
            f"eval={eval_env.n_features}"
        )

    expected_obs_shape = (train_env.window * expected_features,)
    if train_env.observation_space.shape != expected_obs_shape:
        raise ValueError(
            "Train observation shape mismatch. "
            f"Expected {expected_obs_shape}, got {train_env.observation_space.shape}"
        )
    if eval_env.observation_space.shape != expected_obs_shape:
        raise ValueError(
            "Eval observation shape mismatch. "
            f"Expected {expected_obs_shape}, got {eval_env.observation_space.shape}"
        )


def _run_eval_and_log(model: SAC, env: TradingEnv, log_path: str) -> None:
    """
    Run one deterministic episode on env and write per-step records to CSV.
    The CSV is consumed by monitor.py for the Streamlit dashboard.
    """
    fieldnames = [
        "timestamp", "step", "price", "action",
        "position", "portfolio_value", "reward",
    ]

    obs, _ = env.reset()
    done   = False
    step   = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            writer.writerow({
                "timestamp":       datetime.now(timezone.utc).isoformat(),
                "step":            step,
                "price":           round(info["price"],           2),
                "action":          round(float(action[0]),        4),
                "position":        round(info["position"],        4),
                "portfolio_value": round(info["portfolio_value"], 2),
                "reward":          round(float(reward),           6),
            })
            step += 1


if __name__ == "__main__":
    main()
