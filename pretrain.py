"""Fast historical pretraining script for SAC using KrakenBacktestEnv."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
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
        csv_path="D:/XBTUSD_1.csv",
        candle_interval=1,
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


def _save_and_verify_pretrained_model(model: SAC, checkpoint_dir: str = "./checkpoints") -> str:
    """Persist pretrained model with verification and timestamped backup copy."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    canonical_path = os.path.abspath(os.path.join(checkpoint_dir, "pretrained_sac.zip"))

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            model.save(canonical_path)
            if not os.path.exists(canonical_path) or os.path.getsize(canonical_path) <= 0:
                raise RuntimeError("saved file missing or empty")
            _ = SAC.load(canonical_path, env=None)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.abspath(
                os.path.join(checkpoint_dir, f"pretrained_sac_{timestamp}.zip")
            )
            shutil.copy2(canonical_path, backup_path)
            print(f"[pretrain] Saved final model: {canonical_path}")
            print(f"[pretrain] Saved backup model: {backup_path}")
            return canonical_path
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            print(f"[pretrain][warn] Save attempt {attempt}/3 failed: {exc}")
            time.sleep(1.0)

    raise RuntimeError(f"Failed to save pretrained model after 3 attempts: {last_error}")


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


class PlateauStopCallback(BaseCallback):
    """Stop pretraining when episodic rewards flatten after a warmup period."""

    def __init__(self, min_steps: int = 400_000, recent_episodes: int = 5) -> None:
        super().__init__()
        self.min_steps = int(min_steps)
        self.recent_episodes = int(recent_episodes)
        self.completed_episode_rewards: list[float] = []
        self.running_episode_reward = 0.0
        self.stopped_by_plateau = False
        self.stop_reason = ""

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        self.running_episode_reward += reward

        if not done:
            return True

        self.completed_episode_rewards.append(self.running_episode_reward)
        self.running_episode_reward = 0.0

        if len(self.completed_episode_rewards) < (self.recent_episodes * 2):
            return True
        if self.num_timesteps <= self.min_steps:
            return True

        previous_window = self.completed_episode_rewards[-(self.recent_episodes * 2):-self.recent_episodes]
        recent_window = self.completed_episode_rewards[-self.recent_episodes:]

        previous_mean = float(np.mean(previous_window))
        recent_mean = float(np.mean(recent_window))
        improvement = recent_mean - previous_mean
        plateau_threshold = 0.01 * max(abs(recent_mean), 1e-8)

        if abs(improvement) < plateau_threshold and recent_mean > -0.5:
            self.stopped_by_plateau = True
            self.stop_reason = (
                f"plateau detected (improvement={improvement:+.6f}, "
                f"threshold={plateau_threshold:.6f}, "
                f"recent_mean={recent_mean:+.6f})"
            )
            print(f"[pretrain] Stopping early: {self.stop_reason}")
            return False

        return True


def main() -> None:
    os.makedirs("./checkpoints", exist_ok=True)

    env = DummyVecEnv([
        lambda: KrakenBacktestEnv(
            csv_path="D:/XBTUSD_1.csv",
            candle_interval=1,
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

    plateau_callback = PlateauStopCallback(min_steps=400_000, recent_episodes=5)
    callback = CallbackList([
        BacktestProgressCallback(print_every=1_000),
        RotatingCheckpointCallback(
            save_freq=100_000,
            save_path="./checkpoints",
            name_prefix="pretrain_checkpoint",
            max_checkpoints=3,
        ),
        plateau_callback,
    ])
    max_timesteps = 1_000_000
    model.learn(total_timesteps=max_timesteps, callback=callback)

    if plateau_callback.stopped_by_plateau:
        print(f"[pretrain] Stop reason: {plateau_callback.stop_reason}")
    else:
        print(f"[pretrain] Stop reason: max steps reached ({max_timesteps})")

    _save_and_verify_pretrained_model(model=model, checkpoint_dir="./checkpoints")

    evaluation_summary = _evaluate_pretrained_model(model=model, num_episodes=20)
    print(evaluation_summary)
    with open("./checkpoints/pretrain_stats.txt", "w", encoding="utf-8") as stats_file:
        stats_file.write(evaluation_summary + "\n")

    if len(plateau_callback.completed_episode_rewards) >= plateau_callback.recent_episodes:
        recent_mean = float(
            np.mean(plateau_callback.completed_episode_rewards[-plateau_callback.recent_episodes:])
        )
    elif plateau_callback.completed_episode_rewards:
        recent_mean = float(np.mean(plateau_callback.completed_episode_rewards))
    else:
        recent_mean = 0.0

    if recent_mean <= 0:
        print(
            "[pretrain][warn] Recent mean reward is not positive "
            f"(recent_mean={recent_mean:+.6f})."
        )
        print(
            "Pretraining did not reach positive reward. "
            "Run pretrain.py again or launch train.py manually when ready."
        )
        return

    print('[pretrain] Launching train.py...')
    train_log_path = os.path.abspath("./checkpoints/train_autostart.log")
    with open(train_log_path, "a", encoding="utf-8") as train_log:
        process = subprocess.Popen(
            [sys.executable, "train.py"],
            stdout=train_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print(
        "[pretrain] train.py started "
        f"(pid={process.pid}, output redirected to {train_log_path})"
    )


if __name__ == "__main__":
    main()
