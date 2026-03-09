from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

load_dotenv()


@dataclass
class IBConfig:
    host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    port: int = int(os.getenv("IBKR_PORT", "7497"))
    live: bool = os.getenv("LIVE", "false").lower() == "true"


class IBApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.next_order_id = None
        self.positions: List[dict] = []

    def nextValidId(self, orderId: int):
        self.next_order_id = orderId

    def position(self, account, contract, pos, avgCost):
        self.positions.append(
            {
                "account": account,
                "symbol": contract.symbol,
                "right": contract.right,
                "strike": contract.strike,
                "expiry": contract.lastTradeDateOrContractMonth,
                "position": pos,
                "avg_cost": avgCost,
            }
        )


def _option_contract(ticker: str, strike: float, expiry: str, right: str) -> Contract:
    c = Contract()
    c.symbol = ticker
    c.secType = "OPT"
    c.exchange = "SMART"
    c.currency = "USD"
    c.lastTradeDateOrContractMonth = expiry.replace("-", "")
    c.strike = float(strike)
    c.right = right
    c.multiplier = "100"
    return c


def _market_order(action: str, quantity: int) -> Order:
    o = Order()
    o.action = action
    o.orderType = "MKT"
    o.totalQuantity = quantity
    return o


def connect_app() -> IBApp:
    cfg = IBConfig()
    if cfg.live and cfg.port != 7496:
        raise ValueError("LIVE=true requires IBKR_PORT=7496 for live trading.")

    app = IBApp()
    # TODO: Validate TWS/Gateway is running and API access is enabled before connect in prod.
    app.connect(cfg.host, cfg.port, clientId=1)
    return app


def place_option_order(ticker: str, strike: float, expiry: str, right: str, contracts: int = 1) -> None:
    app = connect_app()
    # TODO: Replace fixed sleep/event handling with robust async run loop + callbacks.
    app.reqIds(-1)
    contract = _option_contract(ticker, strike, expiry, right)
    action = "BUY"
    order = _market_order(action, contracts)
    oid = app.next_order_id or 1
    app.placeOrder(oid, contract, order)
    app.disconnect()


def get_positions() -> List[dict]:
    app = connect_app()
    # TODO: Wait for positionEnd callback before returning in live integration.
    app.reqPositions()
    app.disconnect()
    return app.positions


def close_position(ticker: str) -> None:
    positions = [p for p in get_positions() if p["symbol"] == ticker and p["position"] > 0]
    for p in positions:
        place_option_order(
            ticker=ticker,
            strike=p["strike"],
            expiry=p["expiry"],
            right=p["right"],
            contracts=int(abs(p["position"])),
        )
