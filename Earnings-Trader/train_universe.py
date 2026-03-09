from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from features import build_features
from model import EarningsBeatModel

SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "XOM", "UNH", "PG",
    "HD", "CVX", "MA", "LLY", "ABBV", "PEP", "KO", "BAC", "AVGO", "COST",
    "MRK", "WMT", "DIS", "ADBE", "NFLX", "CRM", "AMD", "INTC", "T", "VZ",
]


def get_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = table[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception:
        return SP500_FALLBACK


def _synthetic_training_dataset(size: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tickers = get_sp500_tickers()
    start = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * 6)

    rows = []
    for i in range(size):
        ticker = tickers[i % len(tickers)]
        event_date = start + pd.Timedelta(days=i)
        eps_lag1 = rng.normal(0, 0.18)
        eps_lag2 = rng.normal(0, 0.20)
        eps_lag3 = rng.normal(0, 0.22)
        eps_lag4 = rng.normal(0, 0.24)
        eps_mean = np.mean([eps_lag1, eps_lag2, eps_lag3, eps_lag4])
        momentum_20d = rng.normal(0, 0.08)
        iv_rank = float(np.clip(rng.normal(0.5, 0.22), 0, 1))
        expected_move = float(np.clip(rng.normal(0.06, 0.02), 0.01, 0.20))
        beat_last_quarter = int(eps_lag1 > 0)
        logit = (
            0.8 * eps_mean
            + 0.6 * momentum_20d
            + 0.3 * beat_last_quarter
            - 0.8 * (iv_rank - 0.5)
            - 1.2 * (expected_move - 0.06)
            + rng.normal(0, 0.35)
        )
        prob = 1 / (1 + np.exp(-logit))
        target = int(rng.random() < prob)

        rows.append(
            {
                "ticker": ticker,
                "event_date": event_date,
                "quarter_idx": i % 20 + 1,
                "eps_surprise_lag1": eps_lag1,
                "eps_surprise_lag2": eps_lag2,
                "eps_surprise_lag3": eps_lag3,
                "eps_surprise_lag4": eps_lag4,
                "eps_surprise_mean": eps_mean,
                "eps_surprise_std": float(np.std([eps_lag1, eps_lag2, eps_lag3, eps_lag4])),
                "eps_estimate_revision": rng.normal(0, 0.1),
                "rev_surprise_lag1": np.nan,
                "rev_surprise_lag2": np.nan,
                "rev_surprise_lag3": np.nan,
                "rev_surprise_lag4": np.nan,
                "momentum_5d": rng.normal(0, 0.04),
                "momentum_20d": momentum_20d,
                "momentum_60d": rng.normal(0, 0.15),
                "iv_rank": iv_rank,
                "expected_move": expected_move,
                "days_since_last_beat": rng.integers(20, 220),
                "beat_last_quarter": beat_last_quarter,
                "sector_encoded": rng.integers(0, 12),
                "target": target,
            }
        )
    return pd.DataFrame(rows)


def build_training_dataset() -> pd.DataFrame:
    rows = []
    for ticker in get_sp500_tickers():
        try:
            hist = yf.Ticker(ticker).get_earnings_dates(limit=20)
            if hist is None or hist.empty:
                continue
            hist = hist.reset_index().rename(columns={"Earnings Date": "event_date", "Surprise(%)": "surprise_pct"})
            hist["event_date"] = pd.to_datetime(hist["event_date"], errors="coerce").dt.tz_localize(None)
            hist = hist.sort_values("event_date").reset_index(drop=True)

            for i, row in hist.iterrows():
                feat = build_features(ticker)
                if feat.empty:
                    continue
                feat["ticker"] = ticker
                feat["event_date"] = row["event_date"]
                feat["quarter_idx"] = i + 1
                feat["target"] = int(float(row.get("surprise_pct", 0)) > 0)
                rows.append(feat.iloc[0].to_dict())
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: no live data available; using synthetic fallback training dataset")
        df = _synthetic_training_dataset()

    if len(df) < 1000:
        print(f"WARNING: training examples={len(df)} (<1000 target)")
    else:
        print(f"Collected training examples={len(df)}")
    return df


def main() -> None:
    print(f"[{datetime.utcnow().isoformat()}] Building training universe...")
    df = build_training_dataset()
    if df.empty:
        raise RuntimeError("No training data built.")
    model = EarningsBeatModel()
    model.train(df)
    model.save()
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
