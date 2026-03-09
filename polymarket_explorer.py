from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

MARKETS_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=100&active=true&closed=false&order=volume&ascending=false"
)
OUTPUT_PATH = Path("polymarket_markets.json")


def request_with_retries(url: str, max_retries: int = 3, timeout: int = 20) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[explorer] Request attempt {attempt}/{max_retries}: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            print(f"[explorer] Request failed on attempt {attempt}: {exc}")
            if attempt == max_retries:
                print("[explorer] Max retries reached; returning empty market list.")
                return []
            time.sleep(2 * attempt)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_yes_probability(market: dict[str, Any]) -> float:
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")

    if isinstance(outcomes, list) and isinstance(prices, list):
        for idx, outcome in enumerate(outcomes):
            if str(outcome).strip().upper() == "YES" and idx < len(prices):
                return _to_float(prices[idx], 0.5)

    if isinstance(outcomes, str) and isinstance(prices, str):
        try:
            outcomes_list = json.loads(outcomes)
            prices_list = json.loads(prices)
            for idx, outcome in enumerate(outcomes_list):
                if str(outcome).strip().upper() == "YES" and idx < len(prices_list):
                    return _to_float(prices_list[idx], 0.5)
        except json.JSONDecodeError:
            pass

    return _to_float(market.get("probability"), 0.5)


def extract_category(market: dict[str, Any]) -> str:
    tags = market.get("tags") or []
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, dict):
            return str(first.get("label") or first.get("name") or "uncategorized")
        return str(first)
    return str(market.get("category") or "uncategorized")


def main() -> None:
    print("[explorer] Fetching top 100 active Polymarket markets...")
    raw_markets = request_with_retries(MARKETS_URL)

    parsed_markets: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for market in raw_markets:
        parsed = {
            "question": market.get("question", ""),
            "yes_probability": extract_yes_probability(market),
            "volume": _to_float(market.get("volume"), 0.0),
            "end_date": market.get("endDate") or market.get("end_date"),
            "category": extract_category(market),
            "tags": market.get("tags") or [],
            "market_id": market.get("id"),
            "slug": market.get("slug"),
        }
        parsed_markets.append(parsed)
        category_counts[parsed["category"]] += 1

    OUTPUT_PATH.write_text(json.dumps(parsed_markets, indent=2), encoding="utf-8")
    print(f"[explorer] Saved {len(parsed_markets)} markets to {OUTPUT_PATH}")

    print("\n[explorer] Market summary by category:")
    for category, count in category_counts.most_common():
        print(f"  - {category}: {count}")

    print("\n[explorer] First 5 markets:")
    for idx, market in enumerate(parsed_markets[:5], start=1):
        print(
            f"  {idx}. {market['question'][:90]} | "
            f"YES={market['yes_probability']:.3f} | "
            f"volume={market['volume']:.2f}"
        )


if __name__ == "__main__":
    main()
