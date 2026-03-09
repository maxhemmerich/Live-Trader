"""Data helpers for earnings options screening using yfinance."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


CACHE_FILE = Path(__file__).with_name("tickers_cache.json")
CACHE_TTL_HOURS = 24
SUPPLEMENTAL_TICKERS = [
    "PLTR", "RIVN", "LCID", "SOFI", "HOOD", "COIN", "MSTR", "RBLX", "SNAP", "UBER",
    "LYFT", "DASH", "ABNB", "GME", "AMC", "BBBY", "SNDL", "CLOV", "WISH", "BB",
    "NOK", "SPCE", "TLRY", "SFIX", "OPEN", "DKNG", "PENN", "CHPT", "NKLA", "WKHS",
    "RIDE", "FSR", "GOEV", "HYLN", "BLNK", "EVGO", "PLUG", "FCEL", "BLDP", "BE",
    "CLNE", "RUN", "NOVA", "ENPH", "SEDG", "FSLR", "CSIQ", "SPWR", "ARRY", "MARA",
    "RIOT", "BITF", "HUT", "CIFR", "CLSK", "WULF", "IREN", "CAN", "BTBT", "MVIS",
    "ASTS", "SOUN", "IONQ", "QBTS", "RGTI", "AI", "BBAI", "UPST", "AFRM", "CVNA",
    "CAR", "AAL", "UAL", "DAL", "CCL", "NCLH", "RCL", "TGTX", "SAVA", "BYND",
    "PTON", "CHWY", "ETSY", "ROKU", "PINS", "SHOP", "SQ", "PYPL", "NIO", "XPEV",
    "LI", "BILI", "IQ", "KWEB", "TQQQ", "SQQQ", "ARKK", "SMCI", "ARM", "TEM",
]



def _dedupe_tickers(symbols: Iterable[str]) -> list[str]:
    cleaned = [str(symbol).strip().upper().replace(".", "-") for symbol in symbols if symbol]
    return list(dict.fromkeys(cleaned))


def _read_cached_tickers() -> list[str] | None:
    if not CACHE_FILE.exists():
        return None

    try:
        payload = json.loads(CACHE_FILE.read_text())
        fetched_at = pd.to_datetime(payload.get("fetched_at"), errors="coerce")
        if pd.isna(fetched_at):
            return None

        age = datetime.utcnow() - fetched_at.to_pydatetime()
        if age > timedelta(hours=CACHE_TTL_HOURS):
            return None

        symbols = payload.get("tickers", [])
        return _dedupe_tickers(symbols)
    except Exception:
        return None


def _write_ticker_cache(tickers: list[str]) -> None:
    payload = {
        "fetched_at": datetime.utcnow().isoformat(),
        "tickers": tickers,
    }
    try:
        CACHE_FILE.write_text(json.dumps(payload))
    except Exception:
        pass


def _fetch_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    except ImportError:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="bs4")[0]
    return _dedupe_tickers(table.get("Symbol", []).tolist())


def _fetch_russell1000_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index")
    except ImportError:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index", flavor="bs4")
    collected: list[str] = []
    for table in tables:
        for column in ("Symbol", "Ticker", "Ticker symbol"):
            if column in table.columns:
                collected.extend(table[column].tolist())
    return _dedupe_tickers(collected)


def get_active_tickers() -> list[str]:
    """Return an actively traded ticker universe with 24-hour caching."""
    cached = _read_cached_tickers()
    if cached:
        return cached

    try:
        sp500 = _fetch_sp500_tickers()
    except Exception:
        sp500 = []
    try:
        russell = _fetch_russell1000_tickers()
    except Exception:
        russell = []

    tickers = _dedupe_tickers([*sp500, *russell, *SUPPLEMENTAL_TICKERS])
    _write_ticker_cache(tickers)
    return tickers


def _parse_earnings_date(calendar: pd.DataFrame | dict | None) -> pd.Timestamp | None:
    """Extract a single upcoming earnings date from yfinance calendar output."""
    if calendar is None:
        return None

    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        if "Earnings Date" in calendar.index and 0 in calendar.columns:
            value = calendar.loc["Earnings Date", 0]
        elif "Earnings Date" in calendar.columns:
            value = calendar["Earnings Date"].iloc[0]
        else:
            value = None
    elif isinstance(calendar, dict):
        value = calendar.get("Earnings Date")
    else:
        value = None

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        value = value[0] if value else None

    if value is None:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_localize(None) if parsed.tzinfo else parsed


def get_upcoming_earnings(days_ahead: int = 5, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Return earnings events in the next `days_ahead` days for a ticker universe."""
    symbols = list(tickers) if tickers is not None else get_active_tickers()
    now = pd.Timestamp(datetime.utcnow().date())
    cutoff = now + timedelta(days=days_ahead)

    def fetch_symbol(symbol: str) -> dict | None:
        try:
            calendar = yf.Ticker(symbol).calendar
            earnings_date = _parse_earnings_date(calendar)
        except Exception:
            return None

        if earnings_date is None:
            return None

        earnings_day = pd.Timestamp(earnings_date.date())
        if now <= earnings_day <= cutoff:
            return {"ticker": symbol, "earnings_date": earnings_day}
        return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        rows = [row for row in executor.map(fetch_symbol, symbols) if row is not None]

    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date"])

    return pd.DataFrame(rows).drop_duplicates().sort_values("earnings_date").reset_index(drop=True)


def get_earnings_history(ticker: str, n: int = 12) -> pd.DataFrame:
    """Return historical EPS estimate/actual surprises from yfinance earnings dates."""
    try:
        hist = yf.Ticker(ticker).get_earnings_dates(limit=n)
        if hist is None or hist.empty:
            return pd.DataFrame(columns=["quarter", "eps_estimate", "eps_actual", "surprise_pct"])

        df = hist.reset_index()
        rename_map = {
            "Earnings Date": "quarter",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "surprise_pct",
            "Revenue Estimate": "revenue_estimate",
            "Reported Revenue": "revenue_actual",
            "Revenue Actual": "revenue_actual",
            "Revenue Surprise(%)": "revenue_surprise_pct",
        }
        df = df.rename(columns={src: dst for src, dst in rename_map.items() if src in df.columns})

        preferred_cols = [
            "quarter",
            "eps_estimate",
            "eps_actual",
            "surprise_pct",
            "revenue_estimate",
            "revenue_actual",
            "revenue_surprise_pct",
        ]
        keep_cols = [col for col in preferred_cols if col in df.columns]
        print(df.columns)

        df = df[keep_cols].copy()
        if "quarter" in df.columns:
            df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
        if "surprise_pct" in df.columns:
            df["surprise_pct"] = pd.to_numeric(df["surprise_pct"], errors="coerce")
        if "revenue_surprise_pct" in df.columns:
            df["revenue_surprise_pct"] = pd.to_numeric(df["revenue_surprise_pct"], errors="coerce")

        if "quarter" in df.columns:
            return df.sort_values("quarter", ascending=False).reset_index(drop=True)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_options_chain(ticker: str) -> pd.DataFrame:
    """Fetch full options chain (calls + puts for each expiration) from yfinance."""
    tk = yf.Ticker(ticker)

    rows: list[pd.DataFrame] = []
    for expiry in tk.options:
        try:
            chain = tk.option_chain(expiry)
        except Exception:
            continue

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["option_type"] = "call"
        puts["option_type"] = "put"
        calls["expiry"] = pd.to_datetime(expiry)
        puts["expiry"] = pd.to_datetime(expiry)
        rows.extend([calls, puts])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if "lastPrice" in out.columns:
        out = out.rename(columns={"lastPrice": "premium"})
    out["premium"] = pd.to_numeric(out.get("premium"), errors="coerce")
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    return out
