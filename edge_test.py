"""Quick edge test: do engineered features predict next-bar direction?"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from backtest_env import KrakenBacktestEnv


BASE_COLUMNS = {"ts", "open", "high", "low", "close", "vol"}


@dataclass
class EdgeResult:
    csv_path: str
    timeframe_label: str
    samples: int
    train_samples: int
    test_samples: int
    accuracy: float
    top_features: list[tuple[str, float]]


def build_dataset(csv_path: str, candle_interval: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    env = KrakenBacktestEnv(
        csv_path=csv_path,
        candle_interval=candle_interval,
        episode_length=1,
        max_buffer_rows=1,
    )

    two_year_df = env.full_df.iloc[env.sample_start_idx :].copy().reset_index(drop=True)
    env.df = two_year_df
    env._precompute_indicators()

    feature_columns = [
        col
        for col in env.df.columns
        if col not in BASE_COLUMNS
    ]

    features_df = env.df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    next_close = env.df["close"].shift(-1)
    labels = (next_close > env.df["close"]).astype(int)

    features_df = features_df.iloc[:-1].reset_index(drop=True)
    labels = labels.iloc[:-1].to_numpy(dtype=np.int32)

    return features_df.to_numpy(dtype=np.float32), labels, feature_columns


def _format_timeframe_label(candle_interval: int) -> str:
    if candle_interval >= 1440:
        days = candle_interval // 1440
        return f"{days}day" if days == 1 else f"{days}days"
    if candle_interval >= 60 and candle_interval % 60 == 0:
        hours = candle_interval // 60
        return f"{hours}hour" if hours == 1 else f"{hours}hours"
    return f"{candle_interval}min"


def _extract_timeframe_token(path: str) -> str | None:
    stem = Path(path).stem
    match = re.search(r"_(\d+)$", stem)
    return match.group(1) if match else None


def parse_timeframe(timeframe: str | None, csv_path: str | None = None) -> tuple[int, str]:
    if timeframe:
        token = timeframe.strip().lower()
    else:
        token = _extract_timeframe_token(csv_path or "")
        if token is None:
            raise ValueError(f"Unable to infer timeframe from filename: {csv_path}")

    normalized = re.sub(r"[^0-9a-z]", "", token)
    match = re.match(r"^(\d+)(min|m|hour|h|day|d)?$", normalized)
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe or token}")

    value = int(match.group(1))
    unit = match.group(2) or "min"
    if unit in {"m", "min"}:
        minutes = value
    elif unit in {"h", "hour"}:
        minutes = value * 60
    else:
        minutes = value * 1440

    return minutes, _format_timeframe_label(minutes)


def run_edge_test(csv_path: str, candle_interval: int, timeframe_label: str) -> EdgeResult:
    X, y, feature_columns = build_dataset(csv_path, candle_interval)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    importance_idx = np.argsort(clf.feature_importances_)[::-1][:5]
    top_features = [(feature_columns[idx], float(clf.feature_importances_[idx])) for idx in importance_idx]

    print("=" * 88)
    print(f"CSV: {csv_path}")
    print(f"Timeframe: {timeframe_label} (candle_interval={candle_interval})")
    print(f"Samples: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Test accuracy: {acc:.4f}")
    print("Top 5 feature importances:")
    for rank, (name, importance) in enumerate(top_features, start=1):
        print(f"{rank:2d}. {name}: {importance:.6f}")

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return EdgeResult(
        csv_path=csv_path,
        timeframe_label=timeframe_label,
        samples=len(X),
        train_samples=len(X_train),
        test_samples=len(X_test),
        accuracy=float(acc),
        top_features=top_features,
    )


def find_default_csv_jobs(default_dir: str = "D:/") -> list[tuple[str, int, str]]:
    base = Path(default_dir)
    candidates = sorted(base.glob("*USD*.csv"))
    jobs: list[tuple[str, int, str]] = []
    for path in candidates:
        token = _extract_timeframe_token(str(path))
        if token is None:
            continue
        candle_interval, label = parse_timeframe(token)
        jobs.append((str(path), candle_interval, label))
    return jobs


def print_ranked_summary(results: list[EdgeResult]) -> None:
    if not results:
        print("No successful runs to summarize.")
        return

    ranked = sorted(results, key=lambda r: r.accuracy, reverse=True)
    print("\n" + "#" * 88)
    print("Final ranking by test accuracy (descending)")
    print("#" * 88)
    print(f"{'Rank':<6}{'Accuracy':<12}{'Timeframe':<12}{'Samples':<12}{'CSV'}")
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:<6}{result.accuracy:<12.4f}{result.timeframe_label:<12}"
            f"{result.samples:<12,}{result.csv_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictive edge of engineered features.")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV file for a single run. If omitted, runs a sweep over D:/*USD*.csv.",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Timeframe for --csv (e.g. 5min, 1min, 1day). If omitted, inferred from filename.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_results: list[EdgeResult] = []

    if args.csv:
        interval, label = parse_timeframe(args.timeframe, args.csv)
        all_results.append(run_edge_test(args.csv, interval, label))
    else:
        jobs = find_default_csv_jobs("D:/")
        if not jobs:
            raise FileNotFoundError(
                "No files matching D:/*USD*.csv with timeframe suffix were found."
            )
        for csv_path, interval, label in jobs:
            all_results.append(run_edge_test(csv_path, interval, label))

    print_ranked_summary(all_results)
