from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data import get_earnings_history, get_options_chain, get_upcoming_earnings
from features import build_features
from model import EarningsBeatModel

load_dotenv()


def _option_mid(row: pd.Series) -> float:
    if pd.notna(row.get("bid")) and pd.notna(row.get("ask")) and row["ask"] > 0:
        return float((row["bid"] + row["ask"]) / 2)
    return float(row.get("lastPrice", np.nan))


def _pick_option(ticker: str, right: str, max_days: int = 14, debug: bool = False) -> Optional[Dict]:
    opts = get_options_chain(ticker)
    if debug:
        print(f"{ticker} option chain size before filtering: {len(opts)}")
    if opts.empty:
        return None

    today = pd.Timestamp.utcnow().tz_localize(None)
    opts = opts.copy()
    opts["dte"] = (pd.to_datetime(opts["expiry"]) - today).dt.days
    opts = opts[(opts["dte"] >= 0) & (opts["dte"] <= max_days) & (opts["right"] == right)]
    if opts.empty:
        return None

    opts = opts[
        opts["bid"].notna()
        & opts["ask"].notna()
        & (opts["bid"] >= 0)
        & (opts["ask"] > 0)
    ]
    if opts.empty:
        return None

    opts["spread_pct"] = (opts["ask"] - opts["bid"]) / opts["ask"]
    opts = opts[opts["spread_pct"] <= 0.50]
    if opts.empty:
        return None

    opts["premium"] = opts.apply(_option_mid, axis=1)
    opts = opts[(opts["premium"] >= 0.05) & (opts["premium"] <= 2.00)]
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


def _historical_beat_rate_signal(ticker: str, n: int = 8) -> float:
    history = get_earnings_history(ticker, n=n)
    if history.empty:
        return 0.5
    valid = history.dropna(subset=["eps_estimate", "eps_actual"])
    if valid.empty:
        return 0.5
    return float((valid["eps_actual"] > valid["eps_estimate"]).mean())


def run_screener() -> pd.DataFrame:
    events = get_upcoming_earnings(days_ahead=7)
    try:
        model = EarningsBeatModel.load()
    except Exception:
        model = None
    rows = []

    for event in events:
        ticker = event["ticker"]
        try:
            feat = build_features(ticker)
        except Exception:
            continue
        if feat.empty:
            continue
        if model is None:
            p_beat = _historical_beat_rate_signal(ticker)
        else:
            p_beat = model.predict(feat.drop(columns=["ticker"], errors="ignore"))
        expected_move = float(feat.iloc[0]["expected_move"]) if pd.notna(feat.iloc[0]["expected_move"]) else 0
        if expected_move < 0.03:
            print(f"{ticker} SKIP: expected_move={expected_move:.3f}")
            continue
        iv_rank = float(feat.iloc[0]["iv_rank"]) if pd.notna(feat.iloc[0]["iv_rank"]) else np.nan
        if pd.isna(iv_rank) or iv_rank >= 0.60:
            continue

        recommendation = "SKIP"
        option = None
        print(f"{ticker} p_beat={p_beat:.3f} iv_rank={feat.iloc[0]['iv_rank']:.3f} expected_move={expected_move:.3f}")
        if p_beat > 0.65:
            recommendation = "BUY_CALL"
            option = _pick_option(ticker, "C", debug=(len(rows) == 0))
        elif p_beat < 0.35:
            recommendation = "BUY_PUT"
            option = _pick_option(ticker, "P", debug=(len(rows) == 0))

        if not option:
            continue

        kelly = kelly_fraction(p_beat if recommendation == "BUY_CALL" else 1 - p_beat, option["premium"])
        bankroll = 90
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
                "suggested_$at$100_bankroll": round(bankroll * kelly, 2),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("P(beat)", ascending=False)
    if not out.empty:
        print(out.to_string(index=False))
    else:
        print(f"[{datetime.utcnow().isoformat()}] No qualifying setups today.")
    return out


if __name__ == "__main__":
    run_screener()
