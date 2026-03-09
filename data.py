"""Data helpers for earnings options screening using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf


DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "NFLX", "INTC",
    "CRM", "ORCL", "ADBE", "PYPL", "UBER", "SHOP", "SQ", "ROKU", "SNOW", "PLTR",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "COIN", "HOOD", "RBLX",
    "DIS", "CMCSA", "PARA", "WBD", "SPOT", "PINS", "SNAP", "BABA", "JD", "PDD",
    "XOM", "CVX", "OXY", "SLB", "COP", "PFE", "MRNA", "LLY", "UNH", "JNJ",
]


def _parse_earnings_date(calendar: pd.DataFrame | dict | None) -> pd.Timestamp | None:
    """Extract a single upcoming earnings date from yfinance calendar output."""
    if calendar is None:
        return None

    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        if "Earnings Date" in calendar.index and 0 in calendar.columns:
            value = calendar.loc["Earnings Date", 0]
        elif "Earnings Date" in calendar.columns:
            value = calendar["Earnings Date"].iloc[0]
        else:
            value = None
    elif isinstance(calendar, dict):
        value = calendar.get("Earnings Date")
    else:
        value = None

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        value = value[0] if value else None

    if value is None:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_localize(None) if parsed.tzinfo else parsed


def get_upcoming_earnings(days_ahead: int = 5, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Return earnings events in the next `days_ahead` days for a ticker universe."""
    symbols = list(tickers) if tickers is not None else DEFAULT_UNIVERSE
    now = pd.Timestamp(datetime.utcnow().date())
    cutoff = now + timedelta(days=days_ahead)

    rows: list[dict] = []
    for symbol in symbols:
        try:
            calendar = yf.Ticker(symbol).calendar
            earnings_date = _parse_earnings_date(calendar)
        except Exception:
            earnings_date = None

        if earnings_date is None:
            continue

        earnings_day = pd.Timestamp(earnings_date.date())
        if now <= earnings_day <= cutoff:
            rows.append({"ticker": symbol, "earnings_date": earnings_day})

    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date"])

    return pd.DataFrame(rows).drop_duplicates().sort_values("earnings_date").reset_index(drop=True)


def get_earnings_history(ticker: str, n: int = 12) -> pd.DataFrame:
    """Return historical EPS estimate/actual surprises from yfinance earnings dates."""
    try:
        hist = yf.Ticker(ticker).get_earnings_dates(limit=n)
        if hist is None or hist.empty:
            return pd.DataFrame(columns=["quarter", "eps_estimate", "eps_actual", "surprise_pct"])

        df = hist.reset_index().rename(
            columns={
                "Earnings Date": "quarter",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "eps_actual",
                "Surprise(%)": "surprise_pct",
                "Revenue Estimate": "revenue_estimate",
                "Reported Revenue": "revenue_actual",
                "Revenue Surprise(%)": "revenue_surprise_pct",
            }
        )
        print(f"{ticker} earnings columns: {list(df.columns)}")

        preferred_cols = [
            "quarter",
            "eps_estimate",
            "eps_actual",
            "surprise_pct",
            "revenue_estimate",
            "revenue_actual",
            "revenue_surprise_pct",
        ]
        keep_cols = [col for col in preferred_cols if col in df.columns]

        df = df[keep_cols].copy()
        if "quarter" in df.columns:
            df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
        if "surprise_pct" in df.columns:
            df["surprise_pct"] = pd.to_numeric(df["surprise_pct"], errors="coerce")
        if "revenue_surprise_pct" in df.columns:
            df["revenue_surprise_pct"] = pd.to_numeric(df["revenue_surprise_pct"], errors="coerce")

        if "quarter" in df.columns:
            return df.sort_values("quarter", ascending=False).reset_index(drop=True)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_options_chain(ticker: str) -> pd.DataFrame:
    """Fetch full options chain (calls + puts for each expiration) from yfinance."""
    tk = yf.Ticker(ticker)

    rows: list[pd.DataFrame] = []
    for expiry in tk.options:
        try:
            chain = tk.option_chain(expiry)
        except Exception:
            continue

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["option_type"] = "call"
        puts["option_type"] = "put"
        calls["expiry"] = pd.to_datetime(expiry)
        puts["expiry"] = pd.to_datetime(expiry)
        rows.extend([calls, puts])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if "lastPrice" in out.columns:
        out = out.rename(columns={"lastPrice": "premium"})
    out["premium"] = pd.to_numeric(out.get("premium"), errors="coerce")
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    return out
