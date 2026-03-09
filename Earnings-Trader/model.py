from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

MODEL_PATH = Path("model.pkl")
LOGGER = logging.getLogger(__name__)


class EarningsBeatModel:
    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        self.feature_columns = []

    def train(self, df: pd.DataFrame) -> None:
        train_df, test_df = self._walk_forward_split(df)
        X_train, y_train = self._xy(train_df)
        X_test, y_test = self._xy(test_df)
        self.model.fit(X_train, y_train)
        self.feature_columns = list(X_train.columns)
        self.evaluate(X_test, y_test)

    def predict(self, features: pd.DataFrame) -> float:
        if self.feature_columns:
            features = features.reindex(columns=self.feature_columns, fill_value=0)
        proba = self.model.predict_proba(features)[:, 1][0]
        return float(proba)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "brier_score": brier_score_loss(y_test, probs),
        }
        LOGGER.info("Eval metrics: %s", metrics)
        importances = pd.Series(self.model.feature_importances_, index=X_test.columns)
        LOGGER.info("Top feature importances:\n%s", importances.sort_values(ascending=False).head(15))
        return metrics

    def save(self, path: Path = MODEL_PATH) -> None:
        payload = {"model": self.model, "feature_columns": self.feature_columns}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "EarningsBeatModel":
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.feature_columns = payload.get("feature_columns", [])
        return obj

    @staticmethod
    def _walk_forward_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfx = df.sort_values(["ticker", "event_date"]).copy()
        dfx["quarter_idx"] = dfx.groupby("ticker").cumcount() + 1
        train_df = dfx[dfx["quarter_idx"].between(1, 16)]
        test_df = dfx[dfx["quarter_idx"].between(17, 20)]
        if test_df.empty:
            split = int(len(dfx) * 0.8)
            train_df, test_df = dfx.iloc[:split], dfx.iloc[split:]
        return train_df, test_df

    @staticmethod
    def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        features = df.drop(columns=["target", "ticker", "event_date", "quarter_idx"], errors="ignore")
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        target = df["target"].astype(int)
        return features, target
