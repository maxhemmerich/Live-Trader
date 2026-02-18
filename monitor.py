"""
monitor.py - Streamlit live monitoring dashboard for the SAC Live Trader.

Usage:
    streamlit run monitor.py

Features:
    - Auto-refreshes every 10 seconds (configurable via sidebar)
    - Loads and displays trading_log.csv produced by train.py
    - KPI metrics: portfolio value, total PnL, current position, last price
    - Line charts: portfolio value, agent position, price over time
    - Recent trades table (last 20 rows)
    - Summary statistics for reward and position
"""

import os
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Page configuration (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title = "SAC Live Trader Monitor",
    page_icon  = "📈",
    layout     = "wide",
)

LOG_CSV      = "trading_log.csv"
INITIAL_CASH = 10_000.0

# ── Sidebar controls ───────────────────────────────────────────────────────
st.sidebar.title("Dashboard Controls")
refresh_interval = st.sidebar.slider(
    "Refresh interval (seconds)",
    min_value = 5,
    max_value = 60,
    value     = 10,
    step      = 5,
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Log file: `{LOG_CSV}`")

# ── Auto-refresh ───────────────────────────────────────────────────────────
refresh_count = st_autorefresh(
    interval = refresh_interval * 1000,
    limit    = None,
    key      = "sac_monitor_refresh",
)

# ── Title ──────────────────────────────────────────────────────────────────
st.title("SAC Live Trader — Performance Monitor")
st.caption(f"Auto-refreshing every {refresh_interval}s  |  Refresh count: {refresh_count}")

# ── Load data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=refresh_interval)
def load_log(path: str) -> pd.DataFrame | None:
    """Load trading_log.csv with TTL caching. Returns None if not found."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as e:
        st.error(f"Failed to load log: {e}")
        return None


df = load_log(LOG_CSV)

if df is None or df.empty:
    st.warning(
        f"No data found. Run `python train.py` first to generate `{LOG_CSV}`.",
        icon="⚠️",
    )
    st.stop()

# ── KPI metrics ────────────────────────────────────────────────────────────
latest           = df.iloc[-1]
portfolio_value  = latest["portfolio_value"]
total_pnl        = portfolio_value - INITIAL_CASH
pnl_pct          = (total_pnl / INITIAL_CASH) * 100.0
current_position = latest["position"]
current_price    = latest["price"]
total_steps      = len(df)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label = "Portfolio Value",
    value = f"${portfolio_value:,.2f}",
    delta = f"${total_pnl:+,.2f}",
)
col2.metric(
    label = "Total PnL",
    value = f"${total_pnl:+,.2f}",
    delta = f"{pnl_pct:+.2f}%",
)
col3.metric(
    label = "Current Position",
    value = f"{current_position:+.4f}",
    help  = "-1 = full short  |  0 = flat  |  +1 = full long",
)
col4.metric(
    label = "Last Price (BTC/USD)",
    value = f"${current_price:,.2f}",
)
col5.metric(
    label = "Total Steps",
    value = f"{total_steps:,}",
)

st.markdown("---")

# ── Line charts ────────────────────────────────────────────────────────────
st.subheader("Portfolio Value Over Time")
st.line_chart(df.set_index("timestamp")[["portfolio_value"]], use_container_width=True)

st.subheader("Agent Position Over Time")
st.line_chart(df.set_index("timestamp")[["position"]], use_container_width=True)

st.subheader("BTC/USD Price Over Time")
st.line_chart(df.set_index("timestamp")[["price"]], use_container_width=True)

# ── Recent trades ──────────────────────────────────────────────────────────
st.subheader("Recent Trades (Last 20 Steps)")
recent = (
    df[["timestamp", "step", "price", "action", "position", "portfolio_value", "reward"]]
    .tail(20)
    .sort_values("step", ascending=False)
    .copy()
)
recent["price"]           = recent["price"].map("${:,.2f}".format)
recent["portfolio_value"] = recent["portfolio_value"].map("${:,.2f}".format)
recent["action"]          = recent["action"].map("{:+.4f}".format)
recent["position"]        = recent["position"].map("{:+.4f}".format)
recent["reward"]          = recent["reward"].map("{:.6f}".format)

st.dataframe(recent, use_container_width=True, hide_index=True)

# ── Summary statistics ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Summary Statistics")

stat_col1, stat_col2 = st.columns(2)
with stat_col1:
    st.write("**Reward**")
    st.dataframe(df["reward"].describe().rename("value").to_frame(), use_container_width=True)
with stat_col2:
    st.write("**Position**")
    st.dataframe(df["position"].describe().rename("value").to_frame(), use_container_width=True)
