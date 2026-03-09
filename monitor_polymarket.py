from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PREDICTIONS_PATH = Path("model_predictions.json")
BACKTEST_PATH = Path("backtest_results.json")
BET_LOG_PATH = Path("bet_log.csv")

st.set_page_config(page_title="Polymarket Trader Monitor", page_icon="📊", layout="wide")
st.title("Polymarket Prediction Market Dashboard")
st_autorefresh(interval=5 * 60 * 1000, limit=None, key="poly_refresh")

if not PREDICTIONS_PATH.exists():
    st.warning("model_predictions.json not found. Run probability_model.py first.")
    st.stop()

preds = pd.DataFrame(json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8")))
if preds.empty:
    st.warning("No predictions available.")
    st.stop()

preds = preds.sort_values("edge", key=lambda s: s.abs(), ascending=False)
top10 = preds.head(10)

st.subheader("Top 10 High-Edge Opportunities")
st.dataframe(top10[["question", "market_prob", "model_prob", "edge", "confidence", "recommended_bet_direction", "recommended_bet_size"]], width="stretch")

fig = px.bar(
    top10,
    x="question",
    y=["market_prob", "model_prob"],
    barmode="group",
    title="Market Probability vs Model Probability",
)
fig.update_layout(xaxis_title="Market", yaxis_title="Probability", xaxis_tickangle=30)
st.plotly_chart(fig, width="stretch")

st.subheader("Historical Bet Performance")
if BACKTEST_PATH.exists():
    backtest = json.loads(BACKTEST_PATH.read_text(encoding="utf-8"))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"${backtest.get('total_return', 0):.2f}")
    col2.metric("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
    col3.metric("Win Rate", f"{backtest.get('win_rate', 0):.2%}")
    col4.metric("Average Edge", f"{backtest.get('average_edge', 0):+.3f}")
else:
    st.info("Run backtest_polymarket.py to populate historical performance.")

st.subheader("Current Open Positions")
if BET_LOG_PATH.exists():
    bets = pd.read_csv(BET_LOG_PATH)
    st.dataframe(bets.tail(50), width="stretch")
    pnl = bets["pnl"].sum() if "pnl" in bets.columns else 0.0
    st.metric("Total PnL", f"${pnl:.2f}")
else:
    st.info("No bet_log.csv found yet.")
    st.metric("Total PnL", "$0.00")
