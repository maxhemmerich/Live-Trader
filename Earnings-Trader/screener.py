from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data import get_options_chain, get_upcoming_earnings
from features import build_features
from model import EarningsBeatModel

load_dotenv()


def _option_mid(row: pd.Series) -> float:
    if pd.notna(row.get("bid")) and pd.notna(row.get("ask")) and row["ask"] > 0:
        return float((row["bid"] + row["ask"]) / 2)
    return float(row.get("lastPrice", np.nan))


def _pick_option(ticker: str, right: str, max_days: int = 7) -> Optional[Dict]:
    opts = get_options_chain(ticker)
    if opts.empty:
        return None

    today = pd.Timestamp.utcnow().tz_localize(None)
    opts = opts.copy()
    opts["dte"] = (pd.to_datetime(opts["expiry"]) - today).dt.days
    opts = opts[(opts["dte"] >= 0) & (opts["dte"] <= max_days) & (opts["right"] == right)]
    if opts.empty:
        return None

    opts["premium"] = opts.apply(_option_mid, axis=1)
    opts = opts[(opts["premium"] > 0) & (opts["premium"] <= 0.50)]
    if opts.empty:
        return None

    # Prefer cheap contracts near requested premium range.
    opts["range_distance"] = np.where(
        opts["premium"].between(0.05, 0.30), 0, np.minimum(abs(opts["premium"] - 0.05), abs(opts["premium"] - 0.30))
    )
    chosen = opts.sort_values(["range_distance", "premium", "dte"]).iloc[0]
    return {
        "option_strike": float(chosen["strike"]),
        "option_expiry": pd.to_datetime(chosen["expiry"]).date().isoformat(),
        "premium": float(chosen["premium"]),
        "right": right,
    }


def kelly_fraction(p: float, premium: float, target_mult: float = 3.0) -> float:
    if premium <= 0:
        return 0.0
    target_price = premium * target_mult
    b = (target_price - premium) / premium
    if b <= 0:
        return 0.0
    f = (p * (b + 1) - 1) / b
    return max(0.0, min(float(f), 1.0))


def run_screener() -> pd.DataFrame:
    events = get_upcoming_earnings(days_ahead=3)
    model = EarningsBeatModel.load()
    rows = []

    for event in events:
        ticker = event["ticker"]
        feat = build_features(ticker)
        if feat.empty:
            continue
        p_beat = model.predict(feat.drop(columns=["ticker"], errors="ignore"))
        expected_move = float(feat.iloc[0]["expected_move"]) if pd.notna(feat.iloc[0]["expected_move"]) else 0
        if expected_move < 0.03:
            continue

        recommendation = "SKIP"
        option = None
        if p_beat > 0.65:
            recommendation = "BUY_CALL"
            option = _pick_option(ticker, "C")
        elif p_beat < 0.35:
            recommendation = "BUY_PUT"
            option = _pick_option(ticker, "P")

        if not option:
            continue

        kelly = kelly_fraction(p_beat if recommendation == "BUY_CALL" else 1 - p_beat, option["premium"])
        bankroll = 300
        suggested = bankroll * kelly

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": event["earnings_date"],
                "P(beat)": round(p_beat, 4),
                "recommended_action": recommendation,
                "option_strike": option["option_strike"],
                "option_expiry": option["option_expiry"],
                "premium": round(option["premium"], 3),
                "kelly_fraction": round(kelly, 4),
                "suggested_$at$300_bankroll": round(suggested, 2),
            }
        )

    out = pd.DataFrame(rows).sort_values("P(beat)", ascending=False)
    if not out.empty:
        print(out.to_string(index=False))
    else:
        print(f"[{datetime.utcnow().isoformat()}] No qualifying setups today.")
    return out


if __name__ == "__main__":
    run_screener()
