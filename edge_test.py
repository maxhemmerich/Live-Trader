"""Quick edge test: do engineered features predict next-bar direction?"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from backtest_env import KrakenBacktestEnv


BASE_COLUMNS = {"ts", "open", "high", "low", "close", "vol"}


def build_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    env = KrakenBacktestEnv(
        csv_path=csv_path,
        candle_interval=5,
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


def run_edge_test(csv_path: str) -> None:
    X, y, feature_columns = build_dataset(csv_path)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    importance_idx = np.argsort(clf.feature_importances_)[::-1][:10]

    print(f"Samples: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Test accuracy: {acc:.4f}")
    print("Top 10 feature importances:")
    for rank, idx in enumerate(importance_idx, start=1):
        print(f"{rank:2d}. {feature_columns[idx]}: {clf.feature_importances_[idx]:.6f}")

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictive edge of engineered features.")
    parser.add_argument(
        "--csv-path",
        default="D:/ETHUSD_5.csv",
        help="Path to ETHUSD csv file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_edge_test(args.csv_path)
