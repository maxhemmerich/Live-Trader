from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from edge_test import BASE_COLUMNS, build_lightweight_features

SYMBOL = "XBT/USD"
CANDLE_INTERVAL = 1
TIMEFRAME = "1m"
CSV_BUFFER_PATH = Path("./checkpoints/gbm_live_buffer.csv")
MODEL_PATH = Path("./checkpoints/gbm_model.pkl")
LOG_PATH = Path("trading_log.csv")
MIN_ORDER_BTC = 0.0001


def init_exchange() -> ccxt.kraken:
    load_dotenv()
    return ccxt.kraken(
        {
            "apiKey": os.getenv("KRAKEN_API_KEY", ""),
            "secret": os.getenv("KRAKEN_API_SECRET", ""),
            "enableRateLimit": True,
        }
    )


def extract_balances(balance: dict) -> tuple[float, float]:
    usd_balance = float(balance.get("free", {}).get("USD", 0.0))
    btc_balance = float(balance.get("free", {}).get("XBT", balance.get("free", {}).get("BTC", 0.0)))
    if usd_balance == 0.0:
        usd_balance = float(balance.get("total", {}).get("USD", 0.0))
    if btc_balance == 0.0:
        btc_balance = float(balance.get("total", {}).get("XBT", balance.get("total", {}).get("BTC", 0.0)))
    return usd_balance, btc_balance


def ensure_log_file(path: Path) -> None:
    if path.exists():
        return
    headers = [
        "timestamp",
        "step",
        "action_raw",
        "action_taken",
        "reward",
        "cumulative_reward",
        "btc_price",
        "portfolio_usd",
        "btc_balance",
        "usd_balance",
        "prob_up",
    ]
    path.write_text(",".join(headers) + "\n", encoding="utf-8")


def fetch_and_cache_bars(exchange: ccxt.kraken, path: Path, limit: int = 2500) -> pd.DataFrame:
    bars = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit, params={"interval": CANDLE_INTERVAL})
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "vol"])
    df.to_csv(path, header=False, index=False)
    return df


def predict_prob_up(model, feature_columns: list[str], csv_path: str) -> tuple[float, float]:
    feats = build_lightweight_features(csv_path, quote_currency="USD")
    if feats.empty:
        raise RuntimeError("No features built from live buffer")
    X_last = feats[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).iloc[[-1]]
    prob_up = float(model.predict_proba(X_last)[0, 1])
    last_price = float(feats.iloc[-1]["close"])
    return prob_up, last_price


def execute_limit_order(exchange: ccxt.kraken, side: str, quoted_price: float, amount_btc: float) -> bool:
    try:
        if side == "buy":
            order = exchange.create_limit_buy_order(SYMBOL, amount_btc, quoted_price)
        else:
            order = exchange.create_limit_sell_order(SYMBOL, amount_btc, quoted_price)

        order_id = order.get("id")
        latest_order = order
        for _ in range(6):
            status = str(latest_order.get("status", "")).lower()
            if status in {"closed", "filled"}:
                return True
            time.sleep(5)
            if hasattr(exchange, "fetch_order") and order_id:
                latest_order = exchange.fetch_order(order_id, SYMBOL)

        if order_id and hasattr(exchange, "fetch_order"):
            latest_order = exchange.fetch_order(order_id, SYMBOL)
            status = str(latest_order.get("status", "")).lower()
            if status in {"closed", "filled"}:
                return True
            if status == "open" and hasattr(exchange, "cancel_order"):
                exchange.cancel_order(order_id, SYMBOL)
        return False
    except Exception as exc:
        print(f"[order] failed: {exc}")
        return False


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    packed = joblib.load(MODEL_PATH)
    model = packed["model"]
    feature_columns: list[str] = packed["feature_columns"]

    exchange = init_exchange()
    ensure_log_file(LOG_PATH)

    cumulative_reward = 0.0
    step = 0
    prev_portfolio = None

    print("[gbm_trader] started")
    while True:
        step += 1
        action_taken = "hold"
        action_raw = 0.0

        try:
            fetch_and_cache_bars(exchange, CSV_BUFFER_PATH)
            prob_up, last_price = predict_prob_up(model, feature_columns, str(CSV_BUFFER_PATH))

            balance = exchange.fetch_balance()
            usd_balance, btc_balance = extract_balances(balance)
            portfolio_usd = (btc_balance * last_price) + usd_balance

            if prob_up > 0.58:
                target_btc_alloc = 1.0
                action_raw = 1.0
            elif prob_up < 0.42:
                target_btc_alloc = 0.0
                action_raw = -1.0
            else:
                target_btc_alloc = None

            if target_btc_alloc is not None and portfolio_usd > 0:
                target_btc_value = portfolio_usd * target_btc_alloc
                current_btc_value = btc_balance * last_price
                value_gap = target_btc_value - current_btc_value

                if abs(value_gap) > 0.10 * portfolio_usd:
                    if value_gap > 0:
                        asks = exchange.fetch_order_book(SYMBOL, limit=5).get("asks", [])
                        quote_price = float(asks[0][0]) if asks else last_price
                        amount_btc = max(0.0, value_gap / (quote_price + 1e-8))
                        if amount_btc >= MIN_ORDER_BTC and execute_limit_order(exchange, "buy", quote_price, amount_btc):
                            action_taken = "buy"
                    else:
                        bids = exchange.fetch_order_book(SYMBOL, limit=5).get("bids", [])
                        quote_price = float(bids[0][0]) if bids else last_price
                        amount_btc = min(btc_balance, abs(value_gap) / (quote_price + 1e-8))
                        if amount_btc >= MIN_ORDER_BTC and execute_limit_order(exchange, "sell", quote_price, amount_btc):
                            action_taken = "sell"

            balance_after = exchange.fetch_balance()
            usd_balance, btc_balance = extract_balances(balance_after)
            portfolio_usd = (btc_balance * last_price) + usd_balance
            reward = 0.0 if prev_portfolio is None else (portfolio_usd - prev_portfolio)
            prev_portfolio = portfolio_usd
            cumulative_reward += reward

            row = [
                datetime.now(timezone.utc).isoformat(),
                str(step),
                f"{action_raw:.8f}",
                action_taken,
                f"{reward:.10f}",
                f"{cumulative_reward:.10f}",
                f"{last_price:.8f}",
                f"{portfolio_usd:.8f}",
                f"{btc_balance:.8f}",
                f"{usd_balance:.8f}",
                f"{prob_up:.8f}",
            ]
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(",".join(row) + "\n")

            print(f"step={step} prob_up={prob_up:.4f} action={action_taken} portfolio_usd={portfolio_usd:.2f}")

        except Exception as exc:
            print(f"[gbm_trader] step={step} error: {exc}")

        time.sleep(60)


if __name__ == "__main__":
    main()
