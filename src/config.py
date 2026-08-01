"""Configuration for the Financial Analyst Agent."""
import os

from dotenv import load_dotenv

load_dotenv()

# Claude model driving the agent loop. Swap tiers via AGENT_MODEL in .env:
#   claude-opus-5    -> highest quality
#   claude-sonnet-5  -> balanced (default)
#   claude-haiku-4-5 -> fastest / cheapest
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5")

# Safety cap on how many tool-calling turns the agent may take before stopping.
MAX_TURNS = 8

# Default trading days used to annualize volatility.
TRADING_DAYS = 252
