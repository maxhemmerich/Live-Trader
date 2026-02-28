"""Fast historical pretraining script for SAC using KrakenBacktestEnv."""

from __future__ import annotations

import os
import time
import csv
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from backtest_env import KrakenBacktestEnv


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = float("-inf")
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak <= 0:
            continue
        drawdown = (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown * 100.0


def _sharpe_ratio_approx(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = np.array(returns, dtype=np.float64)
    std = float(np.std(returns_array))
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(len(returns_array)) * np.mean(returns_array) / std)


def _evaluate_pretrained_model(model: SAC, num_episodes: int = 20) -> str:
    eval_env = KrakenBacktestEnv(
        csv_path="D:/ETHUSD_1.csv",
        episode_length=5000,
        start_idx=None,
    )

    episode_returns: list[float] = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        initial_portfolio = float(eval_env.starting_portfolio_usd)
        last_portfolio = initial_portfolio

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            last_portfolio = float(info.get("portfolio_usd", last_portfolio))

        episode_return = (last_portfolio / max(initial_portfolio, 1e-8)) - 1.0
        episode_returns.append(float(episode_return))

    mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_ret = float(np.std(episode_returns)) if episode_returns else 0.0
    best_ret = float(np.max(episode_returns)) if episode_returns else 0.0
    worst_ret = float(np.min(episode_returns)) if episode_returns else 0.0
    sharpe = 0.0 if std_ret <= 1e-12 else float(np.sqrt(len(episode_returns)) * mean_ret / std_ret)

    return (
        "Deterministic evaluation (20 episodes)\n"
        f"mean return: {mean_ret:.6f}\n"
        f"std return: {std_ret:.6f}\n"
        f"best episode: {best_ret:.6f}\n"
        f"worst episode: {worst_ret:.6f}\n"
        f"approximate Sharpe ratio: {sharpe:.6f}"
    )


class BacktestProgressCallback(BaseCallback):
    """Print summary statistics every N steps during pretraining."""

    def __init__(self, print_every: int = 10_000, log_path: str = "pretrain_log.csv") -> None:
        super().__init__()
        self.print_every = int(print_every)
        self.log_path = log_path
        self.start_time = 0.0
        self.episode_count = 0
        self.completed_episode_rewards: list[float] = []
        self.running_episode_reward = 0.0
        self.latest_portfolio = float("nan")

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        log_exists = os.path.exists(self.log_path)
        should_write_header = (not log_exists) or os.path.getsize(self.log_path) == 0
        if should_write_header:
            with open(self.log_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["steps_completed", "time_elapsed", "mean_reward"])

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

            with open(self.log_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    int(self.num_timesteps),
                    round(elapsed_seconds, 3),
                    round(mean_reward, 6),
                ])

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
            save_freq=100_000,
            save_path="./checkpoints",
            name_prefix="pretrain_checkpoint",
            max_checkpoints=3,
        ),
    ])
    model.learn(total_timesteps=1_000_000, callback=callback)

    os.makedirs("./checkpoints", exist_ok=True)
    model.save("./checkpoints/pretrained_sac.zip")

    evaluation_summary = _evaluate_pretrained_model(model=model, num_episodes=20)
    print(evaluation_summary)
    with open("./checkpoints/pretrain_stats.txt", "w", encoding="utf-8") as stats_file:
        stats_file.write(evaluation_summary + "\n")


if __name__ == "__main__":
    main()
