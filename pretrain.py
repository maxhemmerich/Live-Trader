"""Fast historical pretraining script for SAC using KrakenBacktestEnv."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
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
        policy_kwargs=dict(net_arch=[64, 64]),
        buffer_size=50000,
        learning_rate=1e-4,
        learning_starts=200,
    )

    callback = BacktestProgressCallback(print_every=1_000)
    model.learn(total_timesteps=500_000, callback=callback)

    os.makedirs("./checkpoints", exist_ok=True)
    model.save("./checkpoints/pretrained_sac.zip")


if __name__ == "__main__":
    main()
