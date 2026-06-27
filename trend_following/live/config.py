"""Configuration for live trend-following paper trading."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the shared core module importable regardless of how we're invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import ALL_ETFS, CASH_ETF, MA_MONTHS, RISK_ETFS  # noqa: E402,F401

load_dotenv(Path(__file__).parent / ".env")

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_PATH = DATA_DIR / "portfolio_state.json"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# --- Alpaca paper credentials (separate account from the factor strategy) ---
ALPACA_KEY = os.environ.get("ALPACA_TREND_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_TREND_SECRET", "")

# --- Behaviour ---
DRY_RUN: bool = os.environ.get("DRY_RUN", "false").lower() == "true"
# Smallest dollar move worth trading — avoids churn and sub-$1 Alpaca rejections.
MIN_TRADE_USD: float = 20.0
