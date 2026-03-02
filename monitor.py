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
import numpy as np
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
SAC_LOG_CSV  = "sac_log.csv"
PRETRAIN_LOG_CSV = "pretrain_log.csv"
PRETRAINED_MODEL_PATH = "./checkpoints/pretrained_sac.zip"
TRAIN_AUTOSTART_LOG_PATH = "./checkpoints/train_autostart.log"

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
st.sidebar.caption(f"SAC log file: `{SAC_LOG_CSV}`")
st.sidebar.caption(f"Pretrain log file: `{PRETRAIN_LOG_CSV}`")
st.sidebar.caption(f"Pretrained model: `{PRETRAINED_MODEL_PATH}`")

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


@st.cache_data(ttl=refresh_interval)
def load_sac_log(path: str) -> pd.DataFrame | None:
    """Load SAC metric log if present. Returns None when unavailable."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to load SAC metrics log: {e}", icon="⚠️")
        return None


@st.cache_data(ttl=refresh_interval)
def load_pretrain_log(path: str) -> pd.DataFrame | None:
    """Load pretraining progress log if present."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        for col in ["steps_completed", "time_elapsed", "mean_reward"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["steps_completed", "time_elapsed", "mean_reward"])
    except Exception as e:
        st.warning(f"Failed to load pretraining log: {e}", icon="⚠️")
        return None


df = load_log(LOG_CSV)
sac_df = load_sac_log(SAC_LOG_CSV)
pretrain_df = load_pretrain_log(PRETRAIN_LOG_CSV)

trading_log_exists = os.path.exists(LOG_CSV)
pretrain_log_exists = os.path.exists(PRETRAIN_LOG_CSV)
pretrained_model_exists = os.path.exists(PRETRAINED_MODEL_PATH)
train_autostart_log_exists = os.path.exists(TRAIN_AUTOSTART_LOG_PATH)

show_pretrain_section = pretrain_log_exists
show_live_section = trading_log_exists

if not show_pretrain_section and not show_live_section:
    st.warning(
        f"No data found. Run `python pretrain.py` to generate `{PRETRAIN_LOG_CSV}` or `python train.py` to generate `{LOG_CSV}`.",
        icon="⚠️",
    )
    st.stop()

if show_pretrain_section:
    if show_live_section:
        st.header("Pretraining")
    else:
        st.subheader("Pretraining")

    if pretrain_df is None or pretrain_df.empty:
        st.warning(f"`{PRETRAIN_LOG_CSV}` exists but no readable rows were found.", icon="⚠️")
    else:
        pretrain_plot = pretrain_df.sort_values("steps_completed").copy()
        latest_pretrain = pretrain_plot.iloc[-1]
        completed_steps = int(latest_pretrain["steps_completed"])
        elapsed_seconds = float(latest_pretrain["time_elapsed"])

        steps_per_second = (completed_steps / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        total_pretrain_steps = 1_000_000
        steps_remaining = max(total_pretrain_steps - completed_steps, 0)
        estimated_remaining_seconds = (steps_remaining / steps_per_second) if steps_per_second > 0 else float("inf")

        pretrain_col1, pretrain_col2, pretrain_col3 = st.columns(3)
        pretrain_col1.metric("Steps Completed", f"{completed_steps:,}")
        pretrain_col2.metric("Time Elapsed", f"{elapsed_seconds:,.1f}s")
        if np.isfinite(estimated_remaining_seconds):
            remaining_s = int(estimated_remaining_seconds)
            hours = remaining_s // 3600
            minutes = (remaining_s % 3600) // 60
            seconds = remaining_s % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            pretrain_col3.metric("Estimated Time Remaining", time_str)
        else:
            pretrain_col3.metric("Estimated Time Remaining", "N/A")

        progress_value = min(max(completed_steps / total_pretrain_steps, 0.0), 1.0)
        st.progress(progress_value, text=f"Pretraining progress: {progress_value * 100:.2f}%")

        pretrain_fig = px.line(
            pretrain_plot,
            x="steps_completed",
            y="mean_reward",
            title="Pretraining Mean Reward Over Steps",
        )
        pretrain_fig.update_xaxes(title_text="Steps Completed")
        pretrain_fig.update_yaxes(title_text="Mean Reward")
        st.plotly_chart(pretrain_fig, width="stretch")

if show_pretrain_section and not show_live_section and pretrained_model_exists:
    st.info(
        "Pretraining artifacts were found, but live trading logs are not available yet. "
        "`train.py` may still be initializing historical candles or may have failed to start. "
        "Check `./checkpoints/train_autostart.log` for startup output.",
        icon="ℹ️",
    )

    if train_autostart_log_exists:
        try:
            with open(TRAIN_AUTOSTART_LOG_PATH, "r", encoding="utf-8") as f:
                recent_lines = f.readlines()[-20:]
            st.caption("Recent auto-start log lines")
            st.code("".join(recent_lines) if recent_lines else "(log exists but is empty)")
        except Exception as e:
            st.warning(f"Could not read `{TRAIN_AUTOSTART_LOG_PATH}`: {e}", icon="⚠️")

if show_live_section and show_pretrain_section:
    st.markdown("---")

if show_live_section and not show_pretrain_section:
    st.subheader("Live Trading")
elif show_live_section and show_pretrain_section:
    st.header("Live Trading")

if show_live_section and (df is None or df.empty):
    st.warning(
        f"`{LOG_CSV}` exists but no readable rows were found.",
        icon="⚠️",
    )
    st.stop()

if not show_live_section:
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

learning_health = "No Data"
learning_health_detail = "Need at least 10 SAC actor-loss points"
if sac_df is not None and not sac_df.empty and "actor_loss" in sac_df.columns:
    actor_series = pd.to_numeric(sac_df["actor_loss"], errors="coerce").dropna().tail(10)
    if len(actor_series) >= 2:
        x_idx = np.arange(len(actor_series), dtype=float)
        actor_slope = float(np.polyfit(x_idx, actor_series.to_numpy(dtype=float), 1)[0])
        latest_actor_loss = float(actor_series.iloc[-1])
        if actor_slope < -1e-3:
            learning_health = "Improving"
        elif actor_slope > 1e-3 and latest_actor_loss > 2:
            learning_health = "Diverging"
        else:
            learning_health = "Stalled"
        learning_health_detail = f"actor slope (last 10): {actor_slope:+.5f}"

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

st.metric(
    label="Learning Health",
    value=learning_health,
    delta=learning_health_detail,
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

if sac_df is not None and not sac_df.empty:
    sac_plot = sac_df.copy()
    for col in ["global_step", "actor_loss", "critic_loss", "ent_coef"]:
        if col in sac_plot.columns:
            sac_plot[col] = pd.to_numeric(sac_plot[col], errors="coerce")
    sac_plot = sac_plot.dropna(subset=["global_step", "actor_loss", "critic_loss"])

    if not sac_plot.empty:
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(
            go.Scatter(
                x=sac_plot["global_step"],
                y=sac_plot["actor_loss"],
                name="Actor Loss",
                line=dict(color="#9467bd", width=2),
            ),
            secondary_y=False,
        )
        fig5.add_trace(
            go.Scatter(
                x=sac_plot["global_step"],
                y=sac_plot["critic_loss"],
                name="Critic Loss",
                line=dict(color="#8c564b", width=2),
            ),
            secondary_y=True,
        )
        if "ent_coef" in sac_plot.columns:
            ent_coef_plot = sac_plot.dropna(subset=["ent_coef"])
            if not ent_coef_plot.empty:
                fig5.add_trace(
                    go.Scatter(
                        x=ent_coef_plot["global_step"],
                        y=ent_coef_plot["ent_coef"],
                        name="Entropy Coef",
                        line=dict(color="#17becf", width=2, dash="dot"),
                    ),
                    secondary_y=False,
                )
        fig5.add_hline(
            y=0,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="actor loss = 0",
            annotation_position="top left",
        )
        fig5.update_layout(title="Actor/Critic Loss & Entropy Coef Over Global Step")
        fig5.update_xaxes(title_text="Global Step")
        fig5.update_yaxes(title_text="Actor Loss", secondary_y=False)
        fig5.update_yaxes(title_text="Critic Loss", secondary_y=True)
        st.plotly_chart(fig5, width="stretch")

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
