"""Fast historical pretraining script for SAC using KrakenBacktestEnv."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from backtest_env import KrakenBacktestEnv


class BacktestProgressCallback(BaseCallback):
    """Print summary statistics every N steps during pretraining."""

    def __init__(self, print_every: int = 10_000) -> None:
        super().__init__()
        self.print_every = int(print_every)
        self.start_time = 0.0
        self.episode_count = 0
        self.completed_episode_rewards: list[float] = []
        self.running_episode_reward = 0.0
        self.latest_portfolio = float("nan")

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info: dict[str, Any] = self.locals["infos"][0]

        self.running_episode_reward += reward
        if "portfolio_usd" in info:
            self.latest_portfolio = float(info["portfolio_usd"])

        if done:
            self.episode_count += 1
            self.completed_episode_rewards.append(self.running_episode_reward)
            self.running_episode_reward = 0.0

        if self.num_timesteps % self.print_every == 0:
            elapsed_seconds = time.perf_counter() - self.start_time
            mean_reward = (
                float(np.mean(self.completed_episode_rewards))
                if self.completed_episode_rewards
                else 0.0
            )
            print(
                f"[pretrain] steps_completed={self.num_timesteps} "
                f"time_elapsed={elapsed_seconds:.1f}s "
                f"mean_reward={mean_reward:+.6f}"
            )

        return True


class RotatingCheckpointCallback(CheckpointCallback):
    """Save periodic checkpoints and keep only the most recent files."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, max_checkpoints: int = 3) -> None:
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        self.max_checkpoints = int(max_checkpoints)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        return os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}.{extension}")

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            checkpoints = sorted(
                [
                    file_name
                    for file_name in os.listdir(self.save_path)
                    if file_name.startswith(f"{self.name_prefix}_") and file_name.endswith(".zip")
                ],
                key=lambda file_name: int(file_name.removesuffix(".zip").split("_")[-1]),
            )
            old_checkpoints = checkpoints[:-self.max_checkpoints]
            for checkpoint in old_checkpoints:
                os.remove(os.path.join(self.save_path, checkpoint))

        return continue_training


def main() -> None:
    env = DummyVecEnv([
        lambda: KrakenBacktestEnv(
            csv_path="D:/ETHUSD_1.csv",
            episode_length=5000,
            start_idx=None,
        )
    ])

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

    callback = CallbackList([
        BacktestProgressCallback(print_every=1_000),
        RotatingCheckpointCallback(
            save_freq=10_000,
            save_path="./checkpoints",
            name_prefix="pretrain_checkpoint",
            max_checkpoints=3,
        ),
    ])
    model.learn(total_timesteps=500_000, callback=callback)

    os.makedirs("./checkpoints", exist_ok=True)
    model.save("./checkpoints/pretrained_sac.zip")


if __name__ == "__main__":
    main()
