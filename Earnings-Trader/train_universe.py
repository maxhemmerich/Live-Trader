from __future__ import annotations

from datetime import datetime

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
                cutoff = row["event_date"]
                feat = build_features(ticker)
                if feat.empty:
                    continue
                feat["ticker"] = ticker
                feat["event_date"] = cutoff
                feat["quarter_idx"] = i + 1
                feat["target"] = int(float(row.get("surprise_pct", 0)) > 0)
                rows.append(feat.iloc[0].to_dict())
        except Exception:
            continue

    df = pd.DataFrame(rows)
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
