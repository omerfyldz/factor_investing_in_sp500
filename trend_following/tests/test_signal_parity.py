"""Backtest/live signal parity — the live trader must allocate EXACTLY what the
backtest validated. Both derive from ``core``; this test locks that in so a
future edit to the live wrapper can't silently diverge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # trend_following/

from core import ALL_ETFS, MA_MONTHS, signals, target_weights  # noqa: E402
from live.signals import compute_target_weights  # noqa: E402


def _prices(n_months: int) -> pd.DataFrame:
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(index=idx, columns=ALL_ETFS, dtype=float)
    for i, col in enumerate(ALL_ETFS):
        steps = rng.normal(0.3 * (i - 4), 1.0, n_months)  # varied trends/sign
        df[col] = 100.0 + np.cumsum(steps)
    df[df < 1.0] = 1.0  # keep positive
    return df


def test_live_matches_backtest_last_row():
    prices = _prices(40)
    backtest_last = target_weights(signals(prices, MA_MONTHS)).iloc[-1]
    live = compute_target_weights(prices, MA_MONTHS)
    np.testing.assert_allclose(
        live.reindex(backtest_last.index).to_numpy(),
        backtest_last.to_numpy(),
        rtol=1e-9,
        err_msg="live target weights diverged from the backtest's last-row decision",
    )
    assert abs(float(live.sum()) - 1.0) < 1e-9
    print("test_live_matches_backtest_last_row: OK")


if __name__ == "__main__":
    test_live_matches_backtest_last_row()
