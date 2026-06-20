"""
Monthly rebalance entry point.

Pipeline:
  1. Load S&P 500 universe from existing constituents file (read-only)
  2. Fetch live prices + fundamentals from yfinance
  3. Fetch / load historical index data for trend regression
  4. Build current-month factor panel
  5. Strategy (improved_4):
     a. Score and rank stocks → top-N target tickers
     b. Connect to Alpaca paper account
     c. Diff current vs target holdings
     d. Execute rebalance (or dry-run: print only)
  6. Save updated portfolio state

Usage:
    python run_monthly.py              # live rebalance
    DRY_RUN=true python run_monthly.py # dry run (no orders submitted)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    CONSTITUENTS_PATH,
    DRY_RUN,
    INDEX_MA_PATH,
    INDEX_MONTHLY_PATH,
    LOG_DIR,
    STATE_PATH,
    STRATEGIES,
)
from fetcher import build_live_panel, fetch_index_ma_signals, fetch_index_monthly
from signals import build_live_panel as build_signal_panel
from signals import get_target_tickers, print_factor_summary
from state import load_state, save_state, update_holdings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"rebalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)


def load_index_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical S&P 500 index monthly data and MA signals.
    Prefer the pre-computed processed files; fall back to live yfinance fetch.
    """
    if INDEX_MONTHLY_PATH.exists() and INDEX_MA_PATH.exists():
        logger.info("Loading historical index data from processed files...")
        idx_monthly = pd.read_csv(INDEX_MONTHLY_PATH, parse_dates=["month"])
        idx_ma = pd.read_csv(INDEX_MA_PATH, parse_dates=["month"])
    else:
        logger.info("Processed index files not found — fetching from yfinance...")
        idx_monthly = fetch_index_monthly()
        idx_ma = fetch_index_ma_signals()
    return idx_monthly, idx_ma


def main() -> None:
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    logger.info(f"=== Monthly Rebalance [{mode}] — {datetime.now().isoformat()} ===")

    # ------------------------------------------------------------------
    # 1. Universe
    # ------------------------------------------------------------------
    logger.info("Loading S&P 500 universe...")
    constituents = pd.read_csv(CONSTITUENTS_PATH)
    tickers = constituents["Symbol"].dropna().tolist()
    logger.info(f"  Universe: {len(tickers)} tickers")

    # ------------------------------------------------------------------
    # 2. Fetch live data
    # ------------------------------------------------------------------
    logger.info("Fetching live price and fundamental data...")
    raw_panel = build_live_panel(tickers, years_back=5)

    # ------------------------------------------------------------------
    # 3. Historical index data for trend regression
    # ------------------------------------------------------------------
    idx_monthly, idx_ma = load_index_data()

    # ------------------------------------------------------------------
    # 4. Build factor panel for current month
    # ------------------------------------------------------------------
    logger.info("Building factor panel and computing signals...")
    live_panel = build_signal_panel(raw_panel, idx_monthly, idx_ma)
    print_factor_summary(live_panel)

    # ------------------------------------------------------------------
    # 5. Per-strategy rebalance
    # ------------------------------------------------------------------
    state = load_state()

    for strategy_name, cfg in STRATEGIES.items():
        logger.info(f"\n--- Strategy: {strategy_name} ---")

        # Generate target tickers
        target = get_target_tickers(live_panel, top_n=cfg["top_n"])

        if DRY_RUN:
            logger.info(f"  [DRY RUN] Would hold: {target}")
            state = update_holdings(state, strategy_name, target)
            continue

        # Check credentials
        if not cfg["alpaca_key"] or not cfg["alpaca_secret"]:
            logger.warning(
                f"  Missing Alpaca credentials for {strategy_name}. "
                "Set ALPACA_IMP4_KEY/SECRET or ALPACA_IMP8_KEY/SECRET env vars. Skipping."
            )
            continue

        # Connect and rebalance
        try:
            from executor import make_client, get_current_holdings, rebalance

            client = make_client(cfg["alpaca_key"], cfg["alpaca_secret"])
            current = get_current_holdings(client)
            logger.info(f"  Connected to Alpaca. Current holdings: {sorted(current)}")

            final_holdings = rebalance(client, current, target, cfg, dry_run=False)
            state = update_holdings(state, strategy_name, final_holdings)
            logger.info(f"  Rebalance complete. Holdings: {sorted(final_holdings)}")

        except Exception as exc:
            logger.error(f"  ERROR during {strategy_name} rebalance: {exc}")

    # ------------------------------------------------------------------
    # 6. Persist state
    # ------------------------------------------------------------------
    save_state(state)
    logger.info(f"\nState saved to {STATE_PATH}")
    logger.info(f"Log: {log_file}")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
