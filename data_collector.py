from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from textblob import TextBlob

MARKETS_PATH = Path("polymarket_markets.json")
OUTPUT_PATH = Path("market_signals.json")
GBM_MODEL_PATH = Path("checkpoints/gbm_model.pkl")


load_dotenv()


def request_with_retries(url: str, params: dict[str, Any] | None = None, max_retries: int = 3) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[collector] GET {url} (attempt {attempt}/{max_retries})")
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return r.json()
            return r.text
        except requests.RequestException as exc:
            print(f"[collector] Request failed: {exc}")
            if attempt == max_retries:
                return None
            time.sleep(2 * attempt)


def market_type(question: str, category: str) -> str:
    text = f"{question} {category}".lower()
    if any(k in text for k in ["election", "president", "senate", "governor", "vote", "poll"]):
        return "politics"
    if any(k in text for k in ["bitcoin", "crypto", "ethereum", "solana", "btc", "eth"]):
        return "crypto"
    if any(k in text for k in ["inflation", "gdp", "fed", "interest rate", "cpi", "unemployment"]):
        return "economics"
    return "general"


def collect_poll_signal() -> dict[str, float | None]:
    # 538 poll csv endpoint (national general election)
    csv_url = "https://projects.fivethirtyeight.com/polls-page/data/president_polls.csv"
    text = request_with_retries(csv_url)
    if text is None:
        return {"poll_average": None}
    try:
        from io import StringIO

        df = pd.read_csv(StringIO(text))
        recent = df.sort_values("end_date").tail(100)
        if "pct" in recent.columns:
            poll_avg = float(recent["pct"].mean() / 100.0)
            return {"poll_average": max(0.0, min(1.0, poll_avg))}
    except Exception as exc:
        print(f"[collector] Failed parsing polling data: {exc}")
    return {"poll_average": None}


def collect_crypto_signal() -> dict[str, float | None]:
    if not GBM_MODEL_PATH.exists():
        print("[collector] GBM model not found; crypto signal unavailable.")
        return {"gbm_signal": None}
    try:
        payload = joblib.load(GBM_MODEL_PATH)
        model = payload["model"]
        n_features = max(1, len(payload.get("feature_columns", [])))
        sample = np.zeros((1, n_features))
        prob = float(model.predict_proba(sample)[0][1])
        return {"gbm_signal": prob}
    except Exception as exc:
        print(f"[collector] Could not read GBM signal: {exc}")
        return {"gbm_signal": None}


def collect_fred_signal(question: str) -> dict[str, float | None]:
    lower = question.lower()
    series = "CPIAUCSL"
    if "gdp" in lower:
        series = "GDP"
    elif "fed" in lower or "interest" in lower or "rate" in lower:
        series = "FEDFUNDS"
    elif "unemployment" in lower:
        series = "UNRATE"

    csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    text = request_with_retries(csv_url)
    if text is None:
        return {"fred_series": series, "fred_latest": None, "fred_trend": None}

    try:
        from io import StringIO

        df = pd.read_csv(StringIO(text))
        value_col = [c for c in df.columns if c != "DATE"][0]
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col]).tail(12)
        latest = float(df[value_col].iloc[-1])
        trend = float(df[value_col].pct_change().mean())
        return {"fred_series": series, "fred_latest": latest, "fred_trend": trend}
    except Exception as exc:
        print(f"[collector] Could not parse FRED data: {exc}")
        return {"fred_series": series, "fred_latest": None, "fred_trend": None}


def collect_news_sentiment(query: str) -> dict[str, Any]:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return {"news_headlines": [], "news_sentiment": 0.0}

    payload = request_with_retries(
        "https://newsapi.org/v2/everything",
        params={
            "q": query[:120],
            "language": "en",
            "pageSize": 10,
            "sortBy": "publishedAt",
            "apiKey": api_key,
        },
    )
    if not payload or "articles" not in payload:
        return {"news_headlines": [], "news_sentiment": 0.0}

    headlines = [a.get("title", "") for a in payload["articles"] if a.get("title")]
    if not headlines:
        return {"news_headlines": [], "news_sentiment": 0.0}

    polarities = [TextBlob(h).sentiment.polarity for h in headlines]
    sentiment = float(np.mean(polarities)) if polarities else 0.0
    return {"news_headlines": headlines, "news_sentiment": sentiment}


def main() -> None:
    if not MARKETS_PATH.exists():
        raise FileNotFoundError("polymarket_markets.json not found. Run polymarket_explorer.py first.")

    markets = json.loads(MARKETS_PATH.read_text(encoding="utf-8"))
    results = []

    for idx, market in enumerate(markets, start=1):
        q = market.get("question", "")
        cat = str(market.get("category", ""))
        mtype = market_type(q, cat)
        print(f"[collector] {idx}/{len(markets)} - Collecting signals for: {q[:80]}")

        signals: dict[str, Any] = {
            "question": q,
            "category": cat,
            "market_type": mtype,
            "market_prob": float(market.get("yes_probability", 0.5)),
            "volume": float(market.get("volume", 0.0)),
            "end_date": market.get("end_date"),
        }

        if mtype == "politics":
            signals.update(collect_poll_signal())
        elif mtype == "crypto":
            signals.update(collect_crypto_signal())
        elif mtype == "economics":
            signals.update(collect_fred_signal(q))

        signals.update(collect_news_sentiment(q))
        results.append(signals)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[collector] Saved signals for {len(results)} markets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
