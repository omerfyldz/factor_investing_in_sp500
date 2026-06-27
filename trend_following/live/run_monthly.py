"""Monthly rebalance entry point for the trend-following strategy.

Pipeline:
  1. Fetch month-end adjusted closes for the 10 ETFs (yfinance).
  2. Compute target weights from the shared core (10-month SMA; OFF -> BIL).
  3. If Alpaca credentials are present, rebalance the paper account to those
     dollar weights with notional orders (or print-only when DRY_RUN).
  4. Persist target weights + rebalance date to portfolio_state.json.

Run from the trend_following/ directory:
    python -m live.run_monthly                 # live (needs creds)
    DRY_RUN=true python -m live.run_monthly    # print plan, submit nothing
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import CASH_ETF, RISK_ETFS  # noqa: E402
from live.config import (  # noqa: E402
    ALPACA_KEY, ALPACA_SECRET, DRY_RUN, LOG_DIR, MIN_TRADE_USD, STATE_PATH,
)
from live.fetcher import fetch_monthly_closes  # noqa: E402
from live.signals import compute_target_weights  # noqa: E402
from live.state import save_state  # noqa: E402

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    log.info(f"=== Trend Rebalance [{mode}] — {datetime.now().isoformat()} ===")

    log.info("Fetching month-end closes from yfinance...")
    monthly = fetch_monthly_closes()
    signal_month = monthly.index[-1].date()
    log.info(f"  Signal month: {signal_month}  ({len(monthly)} months of history)")

    weights = compute_target_weights(monthly)
    on = [t for t in RISK_ETFS if weights.get(t, 0) > 1e-9]
    log.info(f"  In-trend sleeves ({len(on)}/{len(RISK_ETFS)}): {on}")
    log.info(f"  BIL (cash) weight: {weights.get(CASH_ETF, 0):.1%}")
    for tkr, w in weights.items():
        if w > 1e-9:
            log.info(f"    {tkr:<5} {w:6.2%}")

    if not ALPACA_KEY or not ALPACA_SECRET:
        log.warning("  No Alpaca credentials (ALPACA_TREND_KEY/SECRET) — "
                    "computed weights only, no account action.")
    else:
        from live.executor import make_client, rebalance
        try:
            client = make_client(ALPACA_KEY, ALPACA_SECRET)
            log.info(f"  Connected to Alpaca paper account. Rebalancing ({mode})...")
            rebalance(client, weights.to_dict(), min_trade_usd=MIN_TRADE_USD, dry_run=DRY_RUN)
        except Exception as exc:
            log.error(f"  ERROR during rebalance: {exc}")

    save_state(weights.to_dict())
    log.info(f"State saved to {STATE_PATH}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
