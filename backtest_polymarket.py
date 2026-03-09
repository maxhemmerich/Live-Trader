from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

HIST_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=500&active=false&closed=true&order=volume&ascending=false"
)
OUTPUT_PATH = Path("backtest_results.json")


def request_with_retries(url: str, max_retries: int = 3) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[backtest] Fetching resolved markets ({attempt}/{max_retries})")
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            print(f"[backtest] Error: {exc}")
            if attempt == max_retries:
                return []
            time.sleep(2 * attempt)


def infer_resolution_yes(market: dict[str, Any]) -> float | None:
    winner = str(market.get("winner") or "").upper()
    if winner == "YES":
        return 1.0
    if winner == "NO":
        return 0.0
    return None


def reconstruct_model_probability(market: dict[str, Any], days_before: int) -> float:
    base = 0.5
    question = str(market.get("question") or "").lower()
    if any(k in question for k in ["bitcoin", "crypto", "eth"]):
        base += 0.03
    if any(k in question for k in ["election", "president", "vote"]):
        base += 0.02

    volume = float(market.get("volume") or 0.0)
    liquidity_term = min(0.05, math.log10(volume + 1) / 20)
    time_term = (30 - min(days_before, 30)) / 1000

    return max(0.01, min(0.99, base + liquidity_term + time_term))


def sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * np.sqrt(252)


def main() -> None:
    markets = request_with_retries(HIST_URL)
    portfolio = 0.0
    stake_base = 1.0
    returns: list[float] = []
    edges: list[float] = []
    wins = 0
    bets = []

    for market in markets:
        result_yes = infer_resolution_yes(market)
        if result_yes is None:
            continue

        market_prob = float(market.get("probability") or 0.5)

        # Reconstruct model at multiple horizons and average
        model_probs = [reconstruct_model_probability(market, d) for d in (30, 14, 7, 3, 1)]
        model_prob = float(np.mean(model_probs))
        edge = model_prob - market_prob
        edges.append(edge)

        if abs(edge) <= 0.10:
            continue

        direction_yes = edge > 0
        denom = (1 - market_prob) if direction_yes else market_prob
        kelly = max(0.0, min(1.0, abs(edge) / max(denom, 0.01)))
        stake = stake_base * kelly

        win = (result_yes == 1.0 and direction_yes) or (result_yes == 0.0 and not direction_yes)
        pnl = stake if win else -stake
        portfolio += pnl
        returns.append(pnl)
        wins += int(win)

        bets.append(
            {
                "question": market.get("question"),
                "edge": edge,
                "stake": stake,
                "pnl": pnl,
                "win": win,
            }
        )

    total_bets = len(bets)
    win_rate = wins / total_bets if total_bets else 0.0
    avg_edge = float(np.mean(edges)) if edges else 0.0
    best_bet = max(bets, key=lambda x: x["pnl"], default=None)
    worst_bet = min(bets, key=lambda x: x["pnl"], default=None)
    result = {
        "total_return": portfolio,
        "sharpe_ratio": sharpe_ratio(returns),
        "win_rate": win_rate,
        "average_edge": avg_edge,
        "bets_count": total_bets,
        "best_bet": best_bet,
        "worst_bet": worst_bet,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n[backtest] Backtest complete")
    print(f"[backtest] total return: {result['total_return']:.2f}")
    print(f"[backtest] sharpe ratio: {result['sharpe_ratio']:.2f}")
    print(f"[backtest] win rate: {result['win_rate']:.2%}")
    print(f"[backtest] average edge: {result['average_edge']:+.3f}")
    print(f"[backtest] best bet: {best_bet}")
    print(f"[backtest] worst bet: {worst_bet}")


if __name__ == "__main__":
    main()
