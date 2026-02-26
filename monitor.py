"""
monitor.py - Streamlit live monitoring dashboard for the SAC Live Trader.

Usage:
    streamlit run monitor.py

Features:
    - Auto-refreshes every 10 seconds (configurable via sidebar)
    - Loads and displays trading_log.csv produced by train.py
    - KPI metrics: portfolio value, total PnL, trend duration, today's PnL, latest action, last ETH price
    - Line charts: portfolio value and ETH overlays for allocation + action signal
    - Recent trades table (last 20 buy/sell actions)
    - Summary statistics for reward and action signal
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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
eth_balance      = latest["eth_balance"]
usd_balance      = latest["usd_balance"]

if portfolio_value > 0:
    eth_allocation_pct = ((eth_balance * current_price) / portfolio_value) * 100.0
    usd_allocation_pct = (usd_balance / portfolio_value) * 100.0
else:
    eth_allocation_pct = 0.0
    usd_allocation_pct = 0.0

time_span_hours = 0.0
if len(df) > 1:
    time_span_seconds = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    time_span_hours = time_span_seconds / 3600 if time_span_seconds > 0 else 0.0

trade_count = int((df["action_taken"] != "hold").sum())
trade_frequency = (trade_count / time_span_hours) if time_span_hours > 0 else 0.0

if "global_step" not in df.columns:
    df["global_step"] = range(len(df))

current_val = df["portfolio_usd"].iloc[-1]
current_global_step = int(df["global_step"].iloc[-1])
lower_steps = df[df["portfolio_usd"] < current_val]

if lower_steps.empty:
    trend_steps = 0
    trend_metric_label = "Downtrend Duration"
    trend_metric_value = f"all {current_global_step} steps"
    trend_metric_delta = None
else:
    earliest_lower = int(lower_steps["global_step"].min())
    trend_steps = current_global_step - earliest_lower
    trend_metric_label = "Above Low Duration"
    trend_metric_value = f"{trend_steps} steps"
    trend_metric_delta = f"Above low for: {trend_steps} steps"

today = pd.Timestamp.now().date()
today_rows = df[df["timestamp"].dt.date == today]
if today_rows.empty:
    pnl_today = 0.0
else:
    first_today_portfolio = float(today_rows.iloc[0]["portfolio_usd"])
    pnl_today = portfolio_value - first_today_portfolio

action_counts = (
    df["action_taken"]
    .value_counts()
    .reindex(["buy", "sell", "hold"], fill_value=0)
)
action_percentages = ((action_counts / total_steps) * 100.0) if total_steps > 0 else action_counts * 0

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

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

col6, col7, col8, col9, col10, col11, col12, col13 = st.columns(8)

col6.metric(
    label = "Trade Frequency",
    value = f"{trade_frequency:.1f} trades/hr",
)
col7.metric(
    label = "Buy",
    value = f"{action_counts['buy']:,} ({action_percentages['buy']:.0f}%)",
)
col8.metric(
    label = "Sell",
    value = f"{action_counts['sell']:,} ({action_percentages['sell']:.0f}%)",
)
col9.metric(
    label = "Hold",
    value = f"{action_counts['hold']:,} ({action_percentages['hold']:.0f}%)",
)
col10.metric(
    label = "ETH Allocation",
    value = f"{eth_allocation_pct:.2f}%",
)
col11.metric(
    label = "USD Allocation",
    value = f"{usd_allocation_pct:.2f}%",
)
col12.metric(
    label = trend_metric_label,
    value = trend_metric_value,
    delta = trend_metric_delta,
)
col13.metric(
    label = "PnL Today",
    value = f"${pnl_today:+,.2f}",
)

st.markdown("---")

# ── Line charts ────────────────────────────────────────────────────────────
df_plot = df.copy()
if "global_step" not in df_plot.columns:
    df_plot["global_step"] = range(len(df_plot))
df_plot["eth_allocation_pct"] = (
    (df_plot["eth_balance"] * df_plot["eth_price"]) / df_plot["portfolio_usd"].replace(0, pd.NA)
) * 100.0
df_plot["eth_allocation_pct"] = df_plot["eth_allocation_pct"].fillna(0.0)
df_plot["action_smoothed"] = df_plot["action_raw"].rolling(window=20, min_periods=1).mean()

fig1 = px.line(df_plot, x="global_step", y="portfolio_usd", title="Portfolio Value Over Time")
fig1.update_layout(yaxis=dict(range=[df["portfolio_usd"].min(), df["portfolio_usd"].max()]))
st.plotly_chart(fig1, width="stretch")

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["eth_price"],
        name="ETH/USD",
        line=dict(color="#1f77b4", width=2),
    ),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["eth_allocation_pct"],
        name="ETH Allocation %",
        line=dict(color="#ff7f0e", width=2),
    ),
    secondary_y=True,
)
fig2.update_layout(title="ETH Price vs Portfolio ETH Allocation")
fig2.update_xaxes(title_text="Global Step")
fig2.update_yaxes(title_text="ETH/USD Price", secondary_y=False)
fig2.update_yaxes(title_text="ETH Allocation (%)", secondary_y=True)
st.plotly_chart(fig2, width="stretch")

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["eth_price"],
        name="ETH/USD",
        line=dict(color="#1f77b4", width=2),
    ),
    secondary_y=False,
)
fig3.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["action_raw"],
        name="Action Raw",
        line=dict(color="#d62728", width=1),
        opacity=0.35,
    ),
    secondary_y=True,
)
fig3.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["action_smoothed"],
        name="Action Raw (Rolling Mean, 20)",
        line=dict(color="#d62728", width=3),
    ),
    secondary_y=True,
)
fig3.update_layout(title="ETH Price vs Action Signal")
fig3.update_xaxes(title_text="Global Step")
fig3.update_yaxes(title_text="ETH/USD Price", secondary_y=False)
fig3.update_yaxes(title_text="Action Signal", secondary_y=True, range=[-1, 1])
st.plotly_chart(fig3, width="stretch")

fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["eth_price"],
        name="ETH Price",
        line=dict(color="#1f77b4", width=2),
    ),
    secondary_y=False,
)
fig4.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["portfolio_usd"],
        name="Portfolio Value",
        line=dict(color="#2ca02c", width=2),
    ),
    secondary_y=True,
)
fig4.add_trace(
    go.Scatter(
        x=df_plot["global_step"],
        y=df_plot["eth_allocation_pct"],
        name="ETH Allocation %",
        line=dict(color="#ff7f0e", width=2),
    ),
    secondary_y=True,
)
fig4.update_layout(
    title="ETH Price, Portfolio Value & ETH Allocation",
    showlegend=True,
)
fig4.update_xaxes(title_text="Global Step")
fig4.update_yaxes(title_text="ETH/USD Price", secondary_y=False)
fig4.update_yaxes(title_text="Portfolio Value (USD) / ETH Allocation (%)", secondary_y=True)
st.plotly_chart(fig4, width="stretch")

# ── Recent trades ──────────────────────────────────────────────────────────
st.subheader("Recent Trades (Last 20 Buy/Sell Actions)")
recent = (
    df[df["action_taken"].isin(["buy", "sell"])][["timestamp", "step", "eth_price", "action_raw", "action_taken", "portfolio_usd", "reward"]]
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
