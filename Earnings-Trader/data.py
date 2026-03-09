from __future__ import annotations
from tickers import ALL_TICKERS

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)



_CACHE_FILE = os.path.join(os.path.dirname(__file__), "tickers_cache.json")
_CACHE_TTL_HOURS = 24


def get_active_tickers() -> List[str]:
    return ALL_TICKERS


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_ticker_earnings(ticker: str, today, end_date):
    try:
        calendar = yf.Ticker(ticker).calendar
    except Exception:
        return None

    value = None
    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        if "Earnings Date" in calendar.index and 0 in calendar.columns:
            value = calendar.loc["Earnings Date", 0]
        elif "Earnings Date" in calendar.columns:
            value = calendar["Earnings Date"].iloc[0]
    elif isinstance(calendar, dict):
        value = calendar.get("Earnings Date")

    if isinstance(value, (list, tuple)):
        value = value[0] if value else None

    earnings_date = pd.to_datetime(value, errors="coerce")
    if pd.isna(earnings_date):
        return None

    earnings_day = earnings_date.tz_localize(None).date() if getattr(earnings_date, "tzinfo", None) else earnings_date.date()
    if not (today <= earnings_day <= end_date):
        return None

    return {"ticker": ticker, "earnings_date": earnings_day.isoformat()}


def get_upcoming_earnings(days_ahead: int = 3) -> List[Dict[str, str]]:
    today = datetime.utcnow().date()
    end_date = today + timedelta(days=days_ahead)
    tickers = get_active_tickers()
    events = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_check_ticker_earnings, t, today, end_date): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                events.append(result)

    return events


def get_earnings_history(ticker: str, n: int = 20) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.get_earnings_dates(limit=max(n, 40))
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "eps_estimate", "eps_actual", "surprise_pct"])

    df = df.rename(columns={
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)": "surprise_pct",
    }).copy()

    df = df.reset_index().rename(columns={"Earnings Date": "date"})
    available = [c for c in ["date", "eps_estimate", "eps_actual", "surprise_pct"] if c in df.columns]
    out = df[available].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["eps_actual"])
    out = out.sort_values("date", ascending=False).head(n).reset_index(drop=True)
    return out


def get_price_history(ticker: str, days: int = 90) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    period_days = max(days + 30, 120)
    hist = tk.history(period=f"{period_days}d", auto_adjust=False)
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    hist = hist.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)


def get_options_chain(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    rows = []
    for expiry in tk.options:
        chain = tk.option_chain(expiry)
        for right, side_df in (("C", chain.calls), ("P", chain.puts)):
            if side_df is None or side_df.empty:
                continue
            cpy = side_df.copy()
            cpy["expiry"] = pd.to_datetime(expiry)
            cpy["right"] = right
            rows.append(cpy)

    if not rows:
        return pd.DataFrame(columns=[
            "contractSymbol", "strike", "lastPrice", "bid", "ask",
            "volume", "openInterest", "impliedVolatility", "expiry", "right", "inTheMoney",
        ])

    all_opts = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "contractSymbol", "strike", "lastPrice", "bid", "ask",
        "volume", "openInterest", "impliedVolatility", "expiry", "right", "inTheMoney",
    ]
    for col in keep_cols:
        if col not in all_opts:
            all_opts[col] = None
    return all_opts[keep_cols].copy()
