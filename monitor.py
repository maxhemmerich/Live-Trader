"""Streamlit dashboard for the earnings options screener."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

OUTPUT_CSV = "screener_output.csv"

st.set_page_config(page_title="Earnings Options Screener", page_icon="🧮", layout="wide")
st.title("Earnings Options Screener")
st.caption("Cheap earnings options candidates ($5–$20 premium per contract).")

if st.button("🔄 Refresh screener output"):
    st.cache_data.clear()

@st.cache_data(ttl=60)
def load_output(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


df = load_output(OUTPUT_CSV)

if df.empty:
    st.warning(
        "No screener results found yet. Run `python screener.py` first to generate `screener_output.csv`."
    )
else:
    st.subheader("Ranked opportunities")
    st.dataframe(df, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", len(df))
    c2.metric("Cheapest contract", f"${df['contract_cost'].min():.2f}")
    c3.metric("Average expected move", f"{df['expected_move'].mean():.2f}%")
