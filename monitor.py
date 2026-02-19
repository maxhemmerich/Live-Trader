"""
monitor.py - Streamlit live monitoring dashboard for the SAC Live Trader.

Usage:
    streamlit run monitor.py

Features:
    - Auto-refreshes every 10 seconds (configurable via sidebar)
    - Loads and displays trading_log.csv produced by train.py
    - KPI metrics: portfolio value, total PnL, latest action, last ETH price
    - Line charts: portfolio value, action signal, ETH price over time
    - Recent trades table (last 20 rows)
    - Summary statistics for reward and action signal
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
portfolio_value      = latest["portfolio_usd"]
initial_portfolio    = float(df.iloc[0]["portfolio_usd"])
total_pnl            = portfolio_value - initial_portfolio
pnl_pct              = (total_pnl / initial_portfolio) * 100.0 if initial_portfolio != 0 else 0.0
current_action   = latest["action_taken"]
current_price    = latest["eth_price"]
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
    label = "Latest Action",
    value = f"{current_action}",
)
col4.metric(
    label = "Last Price (ETH/USD)",
    value = f"${current_price:,.2f}",
)
col5.metric(
    label = "Total Steps",
    value = f"{total_steps:,}",
)

st.markdown("---")

# ── Line charts ────────────────────────────────────────────────────────────
st.subheader("Portfolio Value Over Time")
st.line_chart(df.set_index("timestamp")[["portfolio_usd"]], width='stretch')

st.subheader("Action Signal Over Time")
st.line_chart(df.set_index("timestamp")[["action_raw"]], width='stretch')

st.subheader("ETH/USD Price Over Time")
st.line_chart(df.set_index("timestamp")[["eth_price"]], width='stretch')

# ── Recent trades ──────────────────────────────────────────────────────────
st.subheader("Recent Trades (Last 20 Steps)")
recent = (
    df[["timestamp", "step", "eth_price", "action_raw", "action_taken", "portfolio_usd", "reward"]]
    .tail(20)
    .sort_values("step", ascending=False)
    .copy()
)
recent["eth_price"]      = recent["eth_price"].map("${:,.2f}".format)
recent["portfolio_usd"]  = recent["portfolio_usd"].map("${:,.2f}".format)
recent["action_raw"]     = recent["action_raw"].map("{:+.4f}".format)
recent["reward"]         = recent["reward"].map("{:.6f}".format)

st.dataframe(recent, width='stretch', hide_index=True)

# ── Summary statistics ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Summary Statistics")

stat_col1, stat_col2 = st.columns(2)
with stat_col1:
    st.write("**Reward**")
    st.dataframe(df["reward"].describe().rename("value").to_frame(), width='stretch')
with stat_col2:
    st.write("**Action Signal (Raw)**")
    st.dataframe(df["action_raw"].describe().rename("value").to_frame(), width='stretch')
