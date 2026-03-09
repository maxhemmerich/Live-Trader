from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from executor import get_positions
from model import EarningsBeatModel
from screener import run_screener

load_dotenv()
TRADES_LOG = Path("trades_log.csv")

st.set_page_config(page_title="Earnings Options Trader", layout="wide")
st.title("IBKR Earnings Options Dashboard")


tab1, tab2, tab3, tab4 = st.tabs([
    "Today's Screener",
    "Open Positions",
    "Historical Trades",
    "Model Performance",
])

with tab1:
    if st.button("Refresh Screener"):
        st.cache_data.clear()

    @st.cache_data(ttl=1800)
    def _screen() -> pd.DataFrame:
        return run_screener()

    st.dataframe(_screen(), use_container_width=True)

with tab2:
    positions = pd.DataFrame(get_positions())
    if positions.empty:
        st.info("No open positions.")
    else:
        positions["recommended_action"] = "HOLD"
        st.dataframe(positions, use_container_width=True)

with tab3:
    if TRADES_LOG.exists():
        trades = pd.read_csv(TRADES_LOG)
        win_rate = (trades["pnl"] > 0).mean() if not trades.empty else 0
        avg_return = trades["return_pct"].mean() if "return_pct" in trades else 0
        total_pnl = trades["pnl"].sum() if "pnl" in trades else 0
        st.metric("Win Rate", f"{win_rate:.1%}")
        st.metric("Avg Return", f"{avg_return:.2%}")
        st.metric("Total P&L", f"${total_pnl:,.2f}")
        st.dataframe(trades.tail(200), use_container_width=True)
    else:
        st.info("No historical trades log yet. Create trades_log.csv after first trade.")

with tab4:
    if Path("model.pkl").exists():
        m = EarningsBeatModel.load()
        importances = pd.Series(m.model.feature_importances_, index=m.feature_columns).sort_values(ascending=False)
        chart = px.bar(importances.head(20), title="Top Feature Importances")
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning("model.pkl not found. Run train_universe.py first.")

st.caption(
    f"LIVE mode: {os.getenv('LIVE', 'false')} | IBKR: {os.getenv('IBKR_HOST', '127.0.0.1')}:{os.getenv('IBKR_PORT', '7497')}"
)
