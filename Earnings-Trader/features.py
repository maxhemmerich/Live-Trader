from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from data import get_earnings_history, get_options_chain, get_price_history


SECTOR_MAP: Dict[str, int] = {}


def _sector_to_int(sector: str) -> int:
    if not sector:
        return -1
    if sector not in SECTOR_MAP:
        SECTOR_MAP[sector] = len(SECTOR_MAP)
    return SECTOR_MAP[sector]


def _atm_straddle_ratio(price: float, options_df: pd.DataFrame) -> float:
    if options_df.empty or price <= 0:
        return np.nan
    earliest = options_df["expiry"].min()
    near = options_df[options_df["expiry"] == earliest].copy()
    if near.empty:
        return np.nan
    near["distance"] = (near["strike"] - price).abs()
    atm_strike = near.sort_values("distance").iloc[0]["strike"]
    call = near[(near["right"] == "C") & (near["strike"] == atm_strike)]
    put = near[(near["right"] == "P") & (near["strike"] == atm_strike)]
    if call.empty or put.empty:
        return np.nan

    def mid(df: pd.DataFrame) -> float:
        b, a, lp = df.iloc[0]["bid"], df.iloc[0]["ask"], df.iloc[0]["lastPrice"]
        if pd.notna(b) and pd.notna(a) and a > 0:
            return float((b + a) / 2)
        return float(lp) if pd.notna(lp) else np.nan

    straddle = mid(call) + mid(put)
    return straddle / price if price > 0 else np.nan


def _atm_implied_volatility(price: float, options_df: pd.DataFrame) -> float:
    if options_df.empty or price <= 0 or "impliedVolatility" not in options_df.columns:
        return np.nan
    earliest = options_df["expiry"].min()
    near = options_df[options_df["expiry"] == earliest].copy()
    if near.empty:
        return np.nan
    near = near[pd.notna(near["impliedVolatility"])].copy()
    if near.empty:
        return np.nan
    near["distance"] = (near["strike"] - price).abs()
    atm_strike = near.sort_values("distance").iloc[0]["strike"]
    atm_rows = near[near["strike"] == atm_strike]
    if atm_rows.empty:
        return np.nan
    return float(atm_rows["impliedVolatility"].astype(float).mean())


def build_features(ticker: str) -> pd.DataFrame:
    earnings = get_earnings_history(ticker, n=24)
    prices = get_price_history(ticker, days=260)
    options_df = get_options_chain(ticker)

    if earnings.empty or prices.empty:
        return pd.DataFrame()

    if ticker.upper() == "ADBE":
        raw_earnings = yf.Ticker(ticker).get_earnings_dates(limit=24)
        print("ADBE raw earnings dataframe:")
        print(raw_earnings)

    latest_price = float(prices.iloc[-1]["close"])

    surprises = earnings["surprise_pct"].astype(float)
    rev_surprises = earnings["revenue_surprise_pct"].astype(float) if "revenue_surprise_pct" in earnings.columns else pd.Series([], dtype=float)

    def lag(series: pd.Series, n: int) -> float:
        return float(series.iloc[n - 1]) if len(series) >= n else np.nan

    mean_surprise = float(surprises.mean()) if not surprises.empty else np.nan
    std_surprise = float(surprises.std(ddof=0)) if len(surprises) > 1 else np.nan

    est_now = earnings.iloc[0]["eps_estimate"] if len(earnings) >= 1 else np.nan
    est_30d = earnings.iloc[1]["eps_estimate"] if len(earnings) >= 2 else np.nan
    est_revision = np.nan
    if pd.notna(est_now) and pd.notna(est_30d) and abs(est_now) > 1e-9:
        est_revision = (est_now - est_30d) / abs(est_now)

    closes = prices["close"].astype(float).reset_index(drop=True)

    def ret(window: int) -> float:
        if len(closes) <= window:
            return np.nan
        return float(closes.iloc[-1] / closes.iloc[-1 - window] - 1)

    iv_proxy = _atm_straddle_ratio(latest_price, options_df)

    atm_iv = _atm_implied_volatility(latest_price, options_df)
    iv_rank = np.nan
    if pd.notna(atm_iv):
        iv_rank = min(max(float(atm_iv), 0.0) / 2.0, 1.0)

    beats = (surprises > 0).astype(int)
    last_beat = int(beats.iloc[0]) if len(beats) else 0
    days_since_beat = np.nan
    for _, row in earnings.iterrows():
        if pd.notna(row["surprise_pct"]) and float(row["surprise_pct"]) > 0:
            days_since_beat = (datetime.utcnow() - row["date"]).days
            break

    info = yf.Ticker(ticker).info or {}
    sector_int = _sector_to_int(str(info.get("sector", "")))

    feat = {
        "ticker": ticker,
        "eps_surprise_lag1": lag(surprises, 1),
        "eps_surprise_lag2": lag(surprises, 2),
        "eps_surprise_lag3": lag(surprises, 3),
        "eps_surprise_lag4": lag(surprises, 4),
        "eps_surprise_mean": mean_surprise,
        "eps_surprise_std": std_surprise,
        "eps_estimate_revision": est_revision,
        "rev_surprise_lag1": np.nan,
        "rev_surprise_lag2": np.nan,
        "rev_surprise_lag3": np.nan,
        "rev_surprise_lag4": np.nan,
        "momentum_5d": ret(5),
        "momentum_20d": ret(20),
        "momentum_60d": ret(60),
        "iv_rank": iv_rank,
        "expected_move": iv_proxy,
        "days_since_last_beat": days_since_beat,
        "beat_last_quarter": last_beat,
        "sector_encoded": sector_int,
    }
    return pd.DataFrame([feat])
