from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

PRIMARY_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=100&active=true&closed=false&order=volume&ascending=false"
)
ALTERNATIVE_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=10&active=true&closed=false&order=volume24hr&ascending=false"
)
CLOB_HIGH_VOLUME_URL = "https://clob.polymarket.com/markets?next_cursor=&limit=10"
OUTPUT_PATH = Path("polymarket_markets.json")
VOLUME_CANDIDATE_KEYS = (
    "volume",
    "volumeNum",
    "usdcVolume",
    "volume24hr",
    "liquidity",
)
CATEGORY_CANDIDATE_KEYS = (
    "category",
    "groupItemTitle",
    "groupTitle",
)


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


def extract_volume(market: dict[str, Any]) -> float:
    for key in VOLUME_CANDIDATE_KEYS:
        if key in market and market.get(key) is not None:
            return _to_float(market.get(key), 0.0)

    for key, value in market.items():
        if "volume" in str(key).lower() and value is not None:
            return _to_float(value, 0.0)

    return 0.0


def extract_category(market: dict[str, Any]) -> str:
    tags = market.get("tags") or []
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, dict):
            return str(first.get("label") or first.get("name") or "uncategorized")
        return str(first)

    for key in CATEGORY_CANDIDATE_KEYS:
        value = market.get(key)
        if value:
            return str(value)

    return "uncategorized"


def print_first_market_raw(markets: list[dict[str, Any]], label: str) -> None:
    if not markets:
        print(f"\n[explorer] {label}: no markets returned.")
        return

    print(f"\n[explorer] {label}: first market raw JSON")
    print(json.dumps(markets[0], indent=2))
    print(f"[explorer] {label}: first market field count = {len(markets[0])}")


def normalize_markets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [market for market in payload if isinstance(market, dict)]

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [market for market in payload["data"] if isinstance(market, dict)]
        if isinstance(payload.get("markets"), list):
            return [market for market in payload["markets"] if isinstance(market, dict)]

    return []


def print_top_markets(markets: list[dict[str, Any]], label: str, limit: int = 10) -> None:
    if not markets:
        print(f"\n[explorer] {label}: no markets to display.")
        return

    sorted_markets = sorted(markets, key=extract_volume, reverse=True)

    print(f"\n[explorer] {label}: top {min(limit, len(sorted_markets))} markets by detected volume")
    for idx, market in enumerate(sorted_markets[:limit], start=1):
        market_with_debug = {
            "_detected_volume": extract_volume(market),
            "_detected_category": extract_category(market),
            "_available_volume_fields": {
                k: market.get(k)
                for k in market.keys()
                if "volume" in str(k).lower()
            },
            **market,
        }
        print(f"\n[{label} #{idx}]")
        print(json.dumps(market_with_debug, indent=2, sort_keys=True, default=str))


def main() -> None:
    print("[explorer] Fetching Polymarket markets from primary endpoint...")
    primary_markets = request_with_retries(PRIMARY_URL)
    primary_markets = normalize_markets(primary_markets)

    print("\n[explorer] Fetching Polymarket markets from alternative endpoint...")
    alternative_markets = request_with_retries(ALTERNATIVE_URL)
    alternative_markets = normalize_markets(alternative_markets)

    print("\n[explorer] Fetching Polymarket markets from CLOB higher-volume endpoint...")
    clob_markets = request_with_retries(CLOB_HIGH_VOLUME_URL)
    clob_markets = normalize_markets(clob_markets)

    print_first_market_raw(primary_markets, "Primary endpoint")
    print_first_market_raw(alternative_markets, "Alternative endpoint")
    print_first_market_raw(clob_markets, "CLOB higher-volume endpoint")

    print_top_markets(primary_markets, "Primary endpoint", limit=10)
    print_top_markets(alternative_markets, "Alternative endpoint", limit=10)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "primary_endpoint": primary_markets,
                "alternative_endpoint": alternative_markets,
                "clob_higher_volume_endpoint": clob_markets,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[explorer] Saved raw market payloads to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
