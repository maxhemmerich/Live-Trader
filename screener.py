"""Earnings options screener focused on low-cost, high-leverage contracts."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf

from data import get_earnings_history, get_options_chain, get_upcoming_earnings

BUDGET_USD = 90
MIN_PREMIUM = 0.05
MAX_PREMIUM = 0.20
MIN_EXPECTED_MOVE = 0.03
EARNINGS_LOOKAHEAD_DAYS = 3
EXPIRY_WINDOW_DAYS = 7
OUTPUT_CSV = "screener_output.csv"


def _mid_price(row: pd.Series) -> float:
    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float((bid + ask) / 2)
    premium = pd.to_numeric(row.get("premium"), errors="coerce")
    return float(premium) if pd.notna(premium) else float("nan")


def _expected_move_pct(ticker: str, earnings_date: pd.Timestamp, chain: pd.DataFrame) -> float:
    expiries = chain["expiry"].dropna().sort_values().unique()
    valid_expiries = [e for e in expiries if pd.Timestamp(e) >= earnings_date]
    if not valid_expiries:
        return float("nan")

    expiry = pd.Timestamp(valid_expiries[0])
    one_exp = chain[chain["expiry"] == expiry].copy()
    calls = one_exp[one_exp["option_type"] == "call"].copy()
    puts = one_exp[one_exp["option_type"] == "put"].copy()
    if calls.empty or puts.empty:
        return float("nan")

    spot = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
    if spot <= 0:
        return float("nan")

    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()
    atm_call = calls.sort_values("dist").iloc[0]
    atm_put = puts.sort_values("dist").iloc[0]

    straddle = _mid_price(atm_call) + _mid_price(atm_put)
    if pd.isna(straddle) or straddle <= 0:
        return float("nan")
    return float(straddle / spot)


def _signal_from_history(hist: pd.DataFrame) -> tuple[str, float]:
    last8 = hist.head(8).copy()
    if last8.empty:
        return "neutral", 0.0

    valid = last8.dropna(subset=["eps_estimate", "eps_actual"]).copy()
    if valid.empty:
        return "neutral", 0.0

    beat_rate = (valid["eps_actual"] > valid["eps_estimate"]).mean()
    miss_rate = (valid["eps_actual"] < valid["eps_estimate"]).mean()

    if beat_rate > 0.70:
        return "call", float(beat_rate)
    if miss_rate > 0.70:
        return "put", float(beat_rate)
    return "neutral", float(beat_rate)


def run_screener() -> pd.DataFrame:
    earnings = get_upcoming_earnings(days_ahead=EARNINGS_LOOKAHEAD_DAYS)
    opportunities: list[dict] = []

    for row in earnings.itertuples(index=False):
        ticker = row.ticker
        earnings_date = pd.Timestamp(row.earnings_date)

        chain = get_options_chain(ticker)
        if chain.empty:
            continue

        chain = chain.copy()
        chain["expiry"] = pd.to_datetime(chain["expiry"], errors="coerce")
        chain["premium"] = chain.apply(_mid_price, axis=1)

        window_end = earnings_date + timedelta(days=EXPIRY_WINDOW_DAYS)
        filtered = chain[
            (chain["premium"] >= MIN_PREMIUM)
            & (chain["premium"] <= MAX_PREMIUM)
            & (chain["expiry"] >= earnings_date)
            & (chain["expiry"] <= window_end)
        ].copy()
        if filtered.empty:
            continue

        expected_move = _expected_move_pct(ticker, earnings_date, chain)
        if pd.isna(expected_move) or expected_move < MIN_EXPECTED_MOVE:
            continue

        hist = get_earnings_history(ticker, n=8)
        direction, beat_rate = _signal_from_history(hist)

        if direction in {"call", "put"}:
            candidates = filtered[filtered["option_type"] == direction]
        else:
            spot = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
            filtered["dist"] = (filtered["strike"] - spot).abs()
            candidates = filtered.sort_values(["dist", "premium"])

        if candidates.empty:
            continue

        pick = candidates.sort_values("premium").iloc[0]
        contract_cost = float(pick["premium"] * 100)
        if contract_cost > BUDGET_USD:
            continue

        opportunities.append(
            {
                "ticker": ticker,
                "earnings_date": earnings_date.date().isoformat(),
                "direction": direction,
                "strike": round(float(pick["strike"]), 2),
                "expiry": pd.Timestamp(pick["expiry"]).date().isoformat(),
                "premium": round(float(pick["premium"]), 3),
                "contract_cost": round(contract_cost, 2),
                "expected_move": round(expected_move * 100, 2),
                "historical_beat_rate": round(beat_rate * 100, 1),
            }
        )

    cols = [
        "ticker",
        "earnings_date",
        "direction",
        "strike",
        "expiry",
        "premium",
        "contract_cost",
        "expected_move",
        "historical_beat_rate",
    ]
    result = pd.DataFrame(opportunities, columns=cols).sort_values("contract_cost", ascending=True)
    result.to_csv(OUTPUT_CSV, index=False)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    out = run_screener()
    if out.empty:
        print("No opportunities found with current filters.")
    else:
        print(out.to_string(index=False))
        print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")
