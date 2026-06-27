"""Invariants for the shared signal/weight core (used by backtest AND live)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # trend_following/

from core import (  # noqa: E402
    ALL_ETFS, CASH_ETF, MA_MONTHS, RISK_ETFS,
    latest_target_weights, signals, target_weights,
)


def _prices(n_months: int, trend: dict[str, float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    df = pd.DataFrame(100.0, index=idx, columns=ALL_ETFS, dtype=float)
    df[CASH_ETF] = 100.0
    for col, slope in (trend or {}).items():
        df[col] = 100.0 + slope * np.arange(n_months)
    return df


def test_weights_sum_to_one():
    w = target_weights(signals(_prices(24, {"SPY": 2.0, "EEM": 1.0}), MA_MONTHS))
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 1.0, rtol=1e-9)
    print("test_weights_sum_to_one: OK")


def test_warmup_sits_in_bil():
    # Before a full 10-month SMA window exists, everything is OFF -> all BIL.
    w = target_weights(signals(_prices(MA_MONTHS, {"SPY": 5.0}), MA_MONTHS))
    assert (w[CASH_ETF].iloc[: MA_MONTHS - 1] == 1.0).all(), "warm-up months must be 100% BIL"
    print("test_warmup_sits_in_bil: OK")


def test_all_on_equal_weight():
    # Every risk ETF in a clean uptrend -> each gets 1/N, BIL gets 0.
    n = len(RISK_ETFS)
    trend = {t: 3.0 for t in RISK_ETFS}
    w = target_weights(signals(_prices(30, trend), MA_MONTHS)).iloc[-1]
    np.testing.assert_allclose(w[RISK_ETFS].to_numpy(), 1.0 / n, rtol=1e-9)
    assert abs(w[CASH_ETF]) < 1e-12
    print("test_all_on_equal_weight: OK")


def test_all_off_full_bil():
    # Flat prices -> close never strictly above SMA -> 100% BIL.
    w = target_weights(signals(_prices(30), MA_MONTHS)).iloc[-1]
    assert abs(w[CASH_ETF] - 1.0) < 1e-12, "flat market must be fully in BIL"
    print("test_all_off_full_bil: OK")


def test_latest_matches_last_row():
    prices = _prices(30, {"SPY": 2.0, "TLT": -1.0})
    full = target_weights(signals(prices, MA_MONTHS)).iloc[-1]
    latest = latest_target_weights(prices, MA_MONTHS)
    np.testing.assert_allclose(full.to_numpy(), latest.reindex(full.index).to_numpy(), rtol=1e-9)
    print("test_latest_matches_last_row: OK")


if __name__ == "__main__":
    test_weights_sum_to_one()
    test_warmup_sits_in_bil()
    test_all_on_equal_weight()
    test_all_off_full_bil()
    test_latest_matches_last_row()
    print("all core tests: OK")
