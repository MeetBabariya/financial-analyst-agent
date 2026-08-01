# 🤖 Financial Analyst Agent

An **AI agent** that answers questions about stocks by autonomously calling tools —
fetching live market data and running quantitative risk calculations (Value-at-Risk,
volatility, drawdown, Sharpe ratio) — then synthesizing the results into a written
analysis. Powered by **Claude** with **tool use (function calling)**.

Built to demonstrate **agentic AI engineering** — the request → tool-call → execute →
feed-back loop — over a real **quantitative-finance** domain.

> ⚠️ Educational project. Not investment advice.

---

## What makes this an *agent*

A single LLM call can't fetch a live stock price or compute VaR — it would hallucinate
the numbers. This project gives Claude a set of **tools** and lets it decide, on its own,
which to call and in what order:

> **You:** "Analyze the risk profile of NVDA."
>
> **Agent:** *(calls `get_stock_snapshot` → `calculate_volatility` → `calculate_var` →
> `calculate_max_drawdown`)* then writes an analysis grounded in the real numbers those
> tools returned.

The loop runs until Claude has gathered enough to answer — that autonomy is what
separates an *agent* from a one-shot prompt.

## How it works

```
   User question
        │
        ▼
  ┌───────────────┐   tool_use    ┌──────────────────────────────┐
  │    Claude     │ ────────────▶ │  Finance tools (this repo)    │
  │  (agent loop) │ ◀──────────── │  market data + VaR/vol/Sharpe │
  └───────┬───────┘  tool_result  └──────────────────────────────┘
          │  (repeats until Claude has enough)
          ▼
   Written analysis + the numbers it used
```

The agent loop is **hand-written** (not the SDK's built-in tool runner) so the
request → `tool_use` → execute → `tool_result` → repeat cycle is explicit and easy to read
in [`src/agent.py`](src/agent.py).

## The tools (Claude's capabilities)

| Tool | What it computes |
|------|------------------|
| `get_stock_snapshot`     | Latest price, 1-month & 1-year return, 52-week range |
| `calculate_var`          | Value-at-Risk (historical + parametric) at a confidence level |
| `calculate_volatility`   | Daily and annualized volatility |
| `calculate_max_drawdown` | Largest peak-to-trough decline |
| `calculate_performance`  | Annualized return, volatility, and Sharpe ratio |

Market data comes from **Yahoo Finance** (`yfinance`); the risk math is standard
quantitative finance implemented with `numpy`/`pandas`.

## Tech stack

| Layer        | Choice                               |
|--------------|--------------------------------------|
| LLM + agent  | **Claude** (Anthropic API, tool use) |
| Market data  | `yfinance` (Yahoo Finance)           |
| Quant math   | `numpy`, `pandas`                     |
| UI           | **Streamlit**                        |

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env      # then edit .env and add your ANTHROPIC_API_KEY

# 3a. Run from the command line (shows each tool call)
python -m src.agent "Analyze the risk profile of NVDA"

# 3b. …or launch the web app
streamlit run app.py
```

## Example questions

- "Analyze the risk profile of NVDA."
- "Compare the volatility of AAPL and TSLA."
- "What is the 99% one-day Value-at-Risk for MSFT?"
- "Which is riskier over the past year, GOOGL or AMZN? Explain."

## Project structure

```
financial-analyst-agent/
├── app.py            # Streamlit UI
├── src/
│   ├── config.py     # model, turn limit
│   ├── tools.py      # finance tools + schemas + dispatch
│   └── agent.py      # the Claude agentic loop
├── requirements.txt
└── .env.example
```

## Design decisions

- **Explicit agent loop.** Written by hand to show the tool-use mechanics clearly, with a
  `MAX_TURNS` safety cap so the agent can't loop forever.
- **Tools return strings, errors included.** A failed tool returns an error message
  rather than crashing, so the agent can adapt (e.g. try a different ticker).
- **Domain logic is real quant finance.** VaR, volatility, drawdown, and Sharpe are the
  same calculations used in risk management — the agent just orchestrates them.
- **No personalized advice.** The system prompt keeps the agent to analysis, not
  investment recommendations.

## Possible extensions

- Add a `compare_tickers` tool for side-by-side analysis
- Stream the agent's reasoning and tool calls live in the UI
- Add fundamentals tools (P/E, revenue growth) alongside the risk tools
- Swap the hand-written loop for the Anthropic SDK's tool runner

---

*Built by [Meet Babariya](https://github.com/MeetBabariya). Educational use only — not
financial advice.*
