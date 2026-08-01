"""Streamlit UI for the Financial Analyst Agent.

Run:
    streamlit run app.py
"""
import streamlit as st

from src import config
from src.agent import run_agent

st.set_page_config(page_title="Financial Analyst Agent", page_icon="🤖", layout="wide")

st.title("🤖 Financial Analyst Agent")
st.caption(
    "Ask a question about a stock. The agent (Claude) autonomously calls market-data "
    "and quant-risk tools, then writes an analysis. Not investment advice."
)

st.markdown("**Try:** *Analyze the risk profile of NVDA* · *Compare the volatility of "
            "AAPL and TSLA* · *What's the 99% 1-day VaR for MSFT?*")

question = st.text_input("Your question", placeholder="Analyze the risk profile of NVDA")

if st.button("Run analysis", type="primary") and question.strip():
    with st.spinner("Agent is calling tools and analyzing…"):
        result = run_agent(question.strip())

    st.markdown("### Analysis")
    st.markdown(result.answer)

    if result.tool_calls:
        st.markdown("### 🔧 Tools the agent called")
        for i, call in enumerate(result.tool_calls, 1):
            with st.expander(f"{i}. `{call['name']}` — {call['input']}"):
                st.code(call["output"])

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. You ask a question.\n"
        "2. Claude decides which tools to call.\n"
        "3. Tools fetch live market data and compute risk metrics "
        "(VaR, volatility, drawdown, Sharpe).\n"
        "4. Claude synthesizes the results into an analysis."
    )
    st.caption(f"Model: `{config.AGENT_MODEL}`")
