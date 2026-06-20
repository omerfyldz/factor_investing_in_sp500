"""Centralised config for the paper-trading deployment."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Paths (read-only references to the parent repo's data)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
CONSTITUENTS_PATH = REPO_ROOT / "data" / "raw" / "sp500_constituents.csv"
PANEL_PATH = REPO_ROOT / "data" / "processed" / "factor_panel.csv"
INDEX_MONTHLY_PATH = REPO_ROOT / "data" / "processed" / "monthly_sp500_index.csv"
STOCK_MA_PATH = REPO_ROOT / "data" / "processed" / "monthly_stock_ma_signals.csv"
INDEX_MA_PATH = REPO_ROOT / "data" / "processed" / "monthly_index_ma_signals.csv"

STATE_PATH = Path(__file__).parent / "data" / "portfolio_state.json"
LOG_DIR = Path(__file__).parent / "logs"

# ---------------------------------------------------------------------------
# Alpaca credentials (from env / .env file)
# ---------------------------------------------------------------------------
ALPACA_IMP4_KEY = os.environ.get("ALPACA_IMP4_KEY", "")
ALPACA_IMP4_SECRET = os.environ.get("ALPACA_IMP4_SECRET", "")

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
STRATEGIES = {
    "improved_4": {
        "top_n": 10,
        "stop_pct": 0.05,
        "take_pct": 0.30,
        "sizing": "fixed_cash",
        "cash_per_trade": 100_000,
        "alpaca_key": ALPACA_IMP4_KEY,
        "alpaca_secret": ALPACA_IMP4_SECRET,
    },
}

# MA windows must match those used in project_core.py
MA_WINDOWS = [3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
MA_COLS = [f"ma_dev_{w}" for w in MA_WINDOWS]

# Minimum months of index history before the expanding trend regression is valid
TREND_MIN_OBS = 72

# Run in dry-run mode (print signals only, no orders submitted)
DRY_RUN: bool = os.environ.get("DRY_RUN", "false").lower() == "true"
