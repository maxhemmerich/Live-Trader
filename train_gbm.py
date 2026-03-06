from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from edge_test import BASE_COLUMNS, build_lightweight_features

CSV_PATH = "D:/XBTUSD_1.csv"
MODEL_PATH = Path("./checkpoints/gbm_model.pkl")


def build_training_frame(csv_path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    features_source = build_lightweight_features(csv_path, quote_currency="USD")
    print(f"[gbm] Loaded {len(features_source)} rows")

    start_2024 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    features_source = features_source[features_source["ts"] >= start_2024].copy().reset_index(drop=True)

    feature_columns = [col for col in features_source.columns if col not in BASE_COLUMNS]
    features_df = features_source[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    next_close = features_source["close"].shift(-1)
    labels = (next_close > features_source["close"]).astype(int)

    features_df = features_df.iloc[:-1].reset_index(drop=True)
    labels = labels.iloc[:-1].reset_index(drop=True)
    print(f"[gbm] Features built: {len(features_df)} samples, {len(feature_columns)} features")
    return features_df, labels, feature_columns


def main() -> None:
    X_df, y, feature_columns = build_training_frame(CSV_PATH)
    if len(X_df) < 200:
        raise RuntimeError(f"Not enough rows to train ({len(X_df)} rows after 2024 filter).")

    split_idx = int(len(X_df) * 0.8)
    X_train = X_df.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X_df.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    print(f"[gbm] Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    top10 = importances[:10]

    test_probs = model.predict_proba(X_test)[:, 1]
    up_mask = test_probs > 0.60
    if up_mask.any():
        calibration = float((y_test.to_numpy()[up_mask] == 1).mean())
        coverage = int(up_mask.sum())
    else:
        calibration = float("nan")
        coverage = 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns}, MODEL_PATH)
    print("[gbm] Model saved to ./checkpoints/gbm_model.pkl")

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print("Top 10 feature importances:")
    for rank, (name, score) in enumerate(top10, start=1):
        print(f"  {rank:2d}. {name:<28} {score:.6f}")

    if np.isnan(calibration):
        print("Calibration @ prob_up > 0.60: no qualifying predictions")
    else:
        print(
            "Calibration @ prob_up > 0.60: "
            f"{calibration * 100:.2f}% up on {coverage} bars"
        )


if __name__ == "__main__":
    main()
