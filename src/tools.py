"""Finance tools the agent can call.

Each tool fetches its own market data (via yfinance) and returns a compact,
model-friendly string. The quant calculations — Value-at-Risk, annualized
volatility, max drawdown, performance stats — are the domain logic the agent
orchestrates.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import yfinance as yf

from . import config


# --------------------------------------------------------------------------- #
# Data helper
# --------------------------------------------------------------------------- #
def _daily_returns(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch adjusted close prices and return daily percentage returns."""
    data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No price data found for ticker {ticker!r}.")
    return data["Close"].pct_change().dropna()


def _prices(ticker: str, period: str = "1y") -> pd.Series:
    data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No price data found for ticker {ticker!r}.")
    return data["Close"]


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def get_stock_snapshot(ticker: str) -> str:
    prices = _prices(ticker, "1y")
    latest = prices.iloc[-1]
    ret_1m = (latest / prices.iloc[-21] - 1) * 100 if len(prices) > 21 else float("nan")
    ret_1y = (latest / prices.iloc[0] - 1) * 100
    return (
        f"{ticker.upper()} snapshot:\n"
        f"- Latest close: ${latest:,.2f}\n"
        f"- 1-month return: {ret_1m:+.2f}%\n"
        f"- 1-year return: {ret_1y:+.2f}%\n"
        f"- 52-week range: ${prices.min():,.2f} – ${prices.max():,.2f}"
    )


def calculate_var(ticker: str, confidence_level: float = 0.95, horizon_days: int = 1) -> str:
    returns = _daily_returns(ticker, "1y")
    alpha = 1 - confidence_level
    scale = np.sqrt(horizon_days)
    # Historical VaR: empirical quantile of the loss distribution.
    hist_var = -np.percentile(returns, alpha * 100) * scale
    # Parametric (normal) VaR: mean + z*sigma, z = standard-normal quantile at alpha.
    z = NormalDist().inv_cdf(alpha)  # negative for alpha < 0.5
    param_var = -(returns.mean() + z * returns.std()) * scale
    return (
        f"{ticker.upper()} Value-at-Risk ({confidence_level:.0%} confidence, "
        f"{horizon_days}-day horizon):\n"
        f"- Historical VaR: {hist_var * 100:.2f}% of position value\n"
        f"- Parametric (normal) VaR: {param_var * 100:.2f}%\n"
        f"Interpretation: on a normal day, losses are not expected to exceed this "
        f"with {confidence_level:.0%} confidence."
    )


def calculate_volatility(ticker: str) -> str:
    returns = _daily_returns(ticker, "1y")
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(config.TRADING_DAYS)
    return (
        f"{ticker.upper()} volatility (trailing 1y):\n"
        f"- Daily volatility: {daily_vol * 100:.2f}%\n"
        f"- Annualized volatility: {annual_vol * 100:.2f}%"
    )


def calculate_max_drawdown(ticker: str) -> str:
    prices = _prices(ticker, "1y")
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    mdd = drawdown.min()
    return (
        f"{ticker.upper()} maximum drawdown (trailing 1y): {mdd * 100:.2f}%\n"
        f"(largest peak-to-trough decline over the period)."
    )


def calculate_performance(ticker: str) -> str:
    returns = _daily_returns(ticker, "1y")
    annual_return = returns.mean() * config.TRADING_DAYS
    annual_vol = returns.std() * np.sqrt(config.TRADING_DAYS)
    sharpe = annual_return / annual_vol if annual_vol else float("nan")
    return (
        f"{ticker.upper()} risk-adjusted performance (trailing 1y):\n"
        f"- Annualized return: {annual_return * 100:.2f}%\n"
        f"- Annualized volatility: {annual_vol * 100:.2f}%\n"
        f"- Sharpe ratio (rf=0): {sharpe:.2f}"
    )


# --------------------------------------------------------------------------- #
# Tool schemas (what Claude sees) + dispatch table
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {
        "name": "get_stock_snapshot",
        "description": "Get a quick snapshot of a stock: latest price, 1-month and "
        "1-year returns, and 52-week range. Call this first to orient on a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_var",
        "description": "Calculate Value-at-Risk (VaR) for a stock — the loss threshold "
        "not expected to be exceeded at a given confidence level. Use for risk questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
                "confidence_level": {"type": "number", "description": "e.g. 0.95 or 0.99"},
                "horizon_days": {"type": "integer", "description": "Holding period in days"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_volatility",
        "description": "Calculate daily and annualized volatility (standard deviation of "
        "returns) for a stock over the trailing year.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_max_drawdown",
        "description": "Calculate the maximum drawdown (largest peak-to-trough decline) "
        "of a stock over the trailing year.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_performance",
        "description": "Calculate risk-adjusted performance: annualized return, "
        "annualized volatility, and Sharpe ratio for a stock over the trailing year.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker"}},
            "required": ["ticker"],
        },
    },
]

_DISPATCH = {
    "get_stock_snapshot": get_stock_snapshot,
    "calculate_var": calculate_var,
    "calculate_volatility": calculate_volatility,
    "calculate_max_drawdown": calculate_max_drawdown,
    "calculate_performance": calculate_performance,
}


def run_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name, returning a result string (or an error message)."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}."
    try:
        return fn(**tool_input)
    except Exception as exc:  # noqa: BLE001 - return errors so the agent can adapt
        return f"Error running {name}: {exc}"
