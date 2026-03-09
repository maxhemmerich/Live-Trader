from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests

INPUT_PATH = Path("market_signals.json")
OUTPUT_PATH = Path("model_predictions.json")
HIST_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=500&active=false&closed=true&order=volume&ascending=false"
)


def request_with_retries(url: str, max_retries: int = 3) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[model] Fetching historical markets {attempt}/{max_retries}")
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            print(f"[model] Historical request failed: {exc}")
            if attempt == max_retries:
                return []
            time.sleep(2 * attempt)


def extract_base_rates() -> dict[str, float]:
    resolved = request_with_retries(HIST_URL)
    buckets: dict[str, list[float]] = defaultdict(list)
    for m in resolved:
        category = str(m.get("category") or "uncategorized").lower()
        outcomes = m.get("outcomes")
        winner = str(m.get("winner") or "").upper()
        yes_resolved = None
        if winner:
            yes_resolved = 1.0 if winner == "YES" else 0.0
        elif isinstance(outcomes, list) and any(str(x).upper() == "YES" for x in outcomes):
            # fallback heuristic when winner missing
            yes_resolved = 0.5
        if yes_resolved is not None:
            buckets[category].append(yes_resolved)

    rates = {k: float(np.mean(v)) for k, v in buckets.items() if v}
    rates["default"] = float(np.mean(list(rates.values()))) if rates else 0.5
    print(f"[model] Computed base rates for {len(rates)-1} categories")
    return rates


def clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


def build_model_probability(signal: dict[str, Any], base_rates: dict[str, float]) -> tuple[float, float]:
    category = str(signal.get("category", "uncategorized")).lower()
    base = base_rates.get(category, base_rates.get("default", 0.5))

    components = [base]
    weights = [0.4]

    if signal.get("news_sentiment") is not None:
        news_component = 0.5 + float(signal["news_sentiment"]) * 0.35
        components.append(clamp(news_component))
        weights.append(0.2)

    if signal.get("poll_average") is not None:
        components.append(clamp(float(signal["poll_average"])))
        weights.append(0.25)

    if signal.get("gbm_signal") is not None:
        components.append(clamp(float(signal["gbm_signal"])))
        weights.append(0.25)

    if signal.get("fred_trend") is not None:
        fred_component = 0.5 + float(signal["fred_trend"]) * 2
        components.append(clamp(fred_component))
        weights.append(0.15)

    vol = float(signal.get("volume") or 0.0)
    confidence = clamp(0.45 + min(0.25, math.log10(vol + 1) / 10))

    model_prob = float(np.average(components, weights=weights))
    return clamp(model_prob), confidence


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("market_signals.json not found. Run data_collector.py first.")

    signals = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    base_rates = extract_base_rates()

    predictions = []
    for signal in signals:
        market_prob = float(signal.get("market_prob", 0.5))
        model_prob, confidence = build_model_probability(signal, base_rates)
        edge = model_prob - market_prob
        direction = "YES" if edge > 0 else "NO"
        denom = (1 - market_prob) if direction == "YES" else market_prob
        bet_fraction = max(0.0, abs(edge) / max(denom, 0.01))
        high_conviction = abs(edge) > 0.10

        predictions.append(
            {
                "question": signal.get("question"),
                "market_prob": market_prob,
                "model_prob": model_prob,
                "edge": edge,
                "confidence": confidence,
                "high_conviction": high_conviction,
                "recommended_bet_direction": direction,
                "recommended_bet_size": round(min(1.0, bet_fraction), 4),
            }
        )

    predictions.sort(key=lambda x: abs(x["edge"]), reverse=True)
    OUTPUT_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"[model] Saved {len(predictions)} predictions to {OUTPUT_PATH}")

    print("\n[model] Top 10 highest-edge opportunities:")
    for i, pred in enumerate(predictions[:10], start=1):
        print(
            f"  {i}. edge={pred['edge']:+.3f}, mkt={pred['market_prob']:.3f}, "
            f"model={pred['model_prob']:.3f}, dir={pred['recommended_bet_direction']} | "
            f"{pred['question'][:80]}"
        )


if __name__ == "__main__":
    main()
