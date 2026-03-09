from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PREDICTIONS_PATH = Path("model_predictions.json")
LOG_PATH = Path("bet_log.csv")
CLOB_ORDER_URL = "https://clob.polymarket.com/orders"

load_dotenv()


def request_with_retries(method: str, url: str, **kwargs: Any) -> requests.Response | None:
    for attempt in range(1, 4):
        try:
            print(f"[executor] {method} {url} (attempt {attempt}/3)")
            resp = requests.request(method, url, timeout=20, **kwargs)
            if resp.status_code >= 400:
                print(f"[executor] HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            print(f"[executor] Request failed: {exc}")
            if attempt == 3:
                return None
            time.sleep(2 * attempt)


def load_predictions() -> list[dict[str, Any]]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("model_predictions.json not found. Run probability_model.py first.")
    return json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))


def daily_loss_exceeded(portfolio_value: float) -> bool:
    if not LOG_PATH.exists():
        return False
    today = dt.datetime.utcnow().date().isoformat()
    daily_pnl = 0.0
    with LOG_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("timestamp", "")).startswith(today):
                daily_pnl += float(row.get("pnl", 0.0))
    drawdown = -daily_pnl / max(portfolio_value, 1e-6)
    return drawdown > 0.20


def place_clob_order(question: str, direction: str, amount_usd: float) -> tuple[bool, str]:
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    wallet = os.getenv("POLYMARKET_WALLET_ADDRESS")

    if not private_key or not wallet:
        return False, "Missing wallet credentials in .env (POLYMARKET_PRIVATE_KEY/POLYMARKET_WALLET_ADDRESS)."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {private_key}",
    }
    payload = {
        "wallet": wallet,
        "question": question,
        "side": direction,
        "amount": round(float(amount_usd), 2),
    }

    response = request_with_retries("POST", CLOB_ORDER_URL, headers=headers, json=payload)
    if response is None:
        return False, "Order request failed after retries"
    return True, response.text[:200]


def append_log(row: dict[str, Any]) -> None:
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "question", "direction", "amount_usd", "edge", "kelly_size", "status", "details", "pnl"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    portfolio_value = float(os.getenv("PORTFOLIO_VALUE", "1000"))
    if daily_loss_exceeded(portfolio_value):
        print("[executor] Circuit breaker active: daily loss exceeded 20% of portfolio. Stopping betting.")
        return

    preds = load_predictions()
    candidates = [
        p for p in preds if p.get("edge", 0) > 0.15 and float(p.get("recommended_bet_size", 0)) * portfolio_value > 1
    ]

    print(f"[executor] Found {len(candidates)} candidates with edge > 0.15")

    for p in candidates:
        amount = float(p["recommended_bet_size"]) * portfolio_value
        success, details = place_clob_order(p["question"], p["recommended_bet_direction"], amount)
        status = "placed" if success else "failed"

        append_log(
            {
                "timestamp": dt.datetime.utcnow().isoformat(),
                "question": p["question"],
                "direction": p["recommended_bet_direction"],
                "amount_usd": round(amount, 2),
                "edge": p["edge"],
                "kelly_size": p["recommended_bet_size"],
                "status": status,
                "details": details,
                "pnl": 0.0,
            }
        )
        print(f"[executor] {status.upper()} | {p['question'][:70]} | amount=${amount:.2f}")


if __name__ == "__main__":
    main()
