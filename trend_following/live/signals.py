"""Live target weights — a thin wrapper over ``core``.

Because this delegates to ``core.latest_target_weights`` (the exact function the
backtest uses for its last row), the live allocation is guaranteed identical to
what the backtest validated. ``tests/test_signal_parity.py`` locks this in.
"""
from __future__ import annotations

import pandas as pd

from core import MA_MONTHS, latest_target_weights


def compute_target_weights(monthly_closes: pd.DataFrame, ma_months: int = MA_MONTHS) -> pd.Series:
    """Target portfolio weights for the most recent fully-closed month.

    Returns a Series indexed by ticker that sums to 1.0 (BIL absorbs the weight
    of every sleeve currently below its 10-month SMA).
    """
    return latest_target_weights(monthly_closes, ma_months)
