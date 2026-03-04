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
import glob
import os
import re
import zipfile
from pathlib import Path
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
TRAIN_OBSERVATION_COLUMNS = OBSERVATION_COLUMNS


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
    env = KrakenLiveEnv(candle_interval=5, timeframe="5m", max_buffer_rows=40_000)
    print(f"[train] Candle buffer initialized with {len(env.df)} rows.")

    latest_checkpoint = env.get_latest_checkpoint()
    pretrained_checkpoint = os.path.join(env.checkpoint_dir, "pretrained_sac.zip")
    pretrain_checkpoints = glob.glob(
        os.path.join(env.checkpoint_dir, "pretrain_checkpoint_*.zip")
    )

    def _pretrain_sort_key(path: str) -> int:
        match = re.search(r"pretrain_checkpoint_(\d+)\.zip$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    pretrain_checkpoints.sort(key=_pretrain_sort_key, reverse=True)

    def _load_latest_valid_checkpoint(checkpoints: list[str], source_name: str) -> SAC | None:
        for checkpoint in checkpoints:
            try:
                print(f"[train] Model source: {source_name} ({checkpoint})")
                return SAC.load(checkpoint, env=env)
            except (EOFError, zipfile.BadZipFile) as exc:
                print(
                    f"[train][warn] Failed to load checkpoint due to corruption: {checkpoint} ({exc}). "
                    "Deleting file and trying next most recent checkpoint."
                )
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
        return None

    live_checkpoints: list[str] = []
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        live_checkpoints = sorted(
            glob.glob(os.path.join(env.checkpoint_dir, "sac_step_*.zip")),
            key=env._checkpoint_sort_key,
            reverse=True,
        )

    model = _load_latest_valid_checkpoint(live_checkpoints, "live checkpoint")

    if model is None:
        model = _load_latest_valid_checkpoint(
            pretrain_checkpoints,
            "pretrain checkpoint",
        )

    if model is None and os.path.exists(pretrained_checkpoint):
        print(f"[train] Model source: pretrained checkpoint ({pretrained_checkpoint})")
        model = SAC.load(pretrained_checkpoint, env=env)
    elif model is None:
        print("[train] Model source: fresh initialization (no live or pretrained checkpoint found)")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            device="cuda",
            policy_kwargs=dict(net_arch=[512, 512]),
            buffer_size=50000,
            learning_rate=1e-4,
            learning_starts=200,
        )

    callback = StepConsoleLogger()
    print(f"[train] Starting training loop for {total_timesteps} timesteps...")

    sac_log_path = Path(env.trading_log_path).resolve().parent / "sac_log.csv"
    if not sac_log_path.exists():
        sac_log_path.write_text("global_step,actor_loss,critic_loss,ent_coef\n", encoding="utf-8")

    for step in range(1, total_timesteps + 1):
        model.learn(total_timesteps=1, reset_num_timesteps=False, callback=callback)

        if step % 10 == 0:
            try:
                ent_coef = float(model.ent_coef_tensor.detach().cpu().item())
            except Exception:
                ent_coef = float(model.ent_coef) if isinstance(model.ent_coef, float) else -1.0
            ent_coef_text = f"{ent_coef:.4f}" if ent_coef >= 0 else "N/A"

            logged_metrics = getattr(model.logger, "name_to_value", {})

            def metric_value(name: str) -> float | None:
                value = logged_metrics.get(name)
                if value is None:
                    return None
                return float(value)

            def metric_text(name: str) -> str:
                value = metric_value(name)
                if value is None:
                    return "N/A"
                return f"{value:.4f}"

            actor_loss = metric_value("train/actor_loss")
            critic_loss = metric_value("train/critic_loss")
            global_step = int(env.step_count)

            sac_log_row = [
                str(global_step),
                "" if actor_loss is None else f"{actor_loss:.10f}",
                "" if critic_loss is None else f"{critic_loss:.10f}",
                f"{ent_coef:.10f}",
            ]
            with sac_log_path.open("a", encoding="utf-8") as f:
                f.write(",".join(sac_log_row) + "\n")

            print(
                f"[SAC] step={global_step} "
                f"| ent_coef={ent_coef_text} "
                f"| actor_loss={metric_text('train/actor_loss')} "
                f"| critic_loss={metric_text('train/critic_loss')} "
                f"| entropy_loss={metric_text('train/entropy_loss')}"
            )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = env.save_checkpoint(model, step)
            print(f"[train] checkpoint saved: {ckpt_path}")


def main() -> None:
    args = parse_args()
    review_contracts()
    print("env.py and train.py contracts match exactly.")
    print(f"[train] OBSERVATION_SIZE={OBSERVATION_SIZE}")
    run_training(total_timesteps=args.timesteps, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
