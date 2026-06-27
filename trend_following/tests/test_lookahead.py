"""Look-ahead guard test — the single most important correctness check.

Construct a scenario where SPY sits flat (signal OFF, parked in BIL), then jumps
hard in month T (turning the 10-month-SMA signal ON at month-end T). An honest
backtest must NOT capture the T jump — the signal is only *known* at the close
of T, so the earliest you can be invested is T+1. A cheating (un-shifted)
backtest would capture the T jump. We assert:

  1. the real engine's month-T return is ~BIL-only (it misses the jump),
  2. a deliberately un-shifted "cheat" calc DOES capture the jump (real < cheat),
  3. the real engine captures SPY only from T+1 onward,
  4. engine weights equal the previous month's target (the .shift(1)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # trend_following/

from backtest.engine import run_backtest          # noqa: E402
from core import ALL_ETFS, RISK_ETFS, signals, target_weights  # noqa: E402


def _synthetic_prices() -> pd.DataFrame:
    """Month-end indexed frame; SPY flat at 100 then jumps at 2020-01-31."""
    idx = pd.date_range("2018-01-31", "2020-06-30", freq="ME")
    df = pd.DataFrame(index=idx, columns=ALL_ETFS, dtype=float)
    # All risk ETFs flat at 100 (=> below/equal SMA => OFF => sit in BIL).
    for col in RISK_ETFS:
        df[col] = 100.0
    # BIL drifts up slowly (~0.3%/month) — the T-bill yield.
    df["BIL"] = 100.0 * (1.003 ** np.arange(len(idx)))
    # SPY jumps at 2020-01-31 and keeps rising — signal turns ON at month-end T.
    jump = pd.Timestamp("2020-01-31")
    mask = df.index >= jump
    df.loc[mask, "SPY"] = 130.0 + 5.0 * np.arange(mask.sum())  # 130,135,140,...
    return df


def test_lookahead_guard():
    prices = _synthetic_prices()
    T = pd.Timestamp("2020-01-31")
    Tplus1 = pd.Timestamp("2020-02-29")

    res = run_backtest(prices, ma_months=10, cost_bps=0.0)
    real = res["monthly_returns"]

    # --- "cheat" calc: un-shifted weights against same-month returns ---
    monthly_ret = prices.pct_change()
    w = target_weights(signals(prices, 10))
    cheat = (w * monthly_ret).sum(axis=1, min_count=1)

    # (1) real month-T return misses the SPY jump (≈ BIL-only, tiny)
    assert real.loc[T] < 0.01, f"real month-T return {real.loc[T]:.4f} should miss the jump"
    # (2) cheat captures the jump (SPY +30% * 1/9 ≈ +3.3%)
    assert cheat.loc[T] > 0.02, f"cheat month-T {cheat.loc[T]:.4f} should capture the jump"
    assert real.loc[T] < cheat.loc[T], "shifted (honest) must underperform the cheat in month T"
    # (3) real captures SPY from T+1 (135/130-1 = +3.8% * 1/9 ≈ +0.43%)
    assert real.loc[Tplus1] > 0.003, f"real T+1 {real.loc[Tplus1]:.4f} should reflect SPY exposure"

    # (4) engine weights at T+1 equal target weights decided at T (.shift(1))
    np.testing.assert_allclose(
        res["weights"].loc[Tplus1, "SPY"], w.loc[T, "SPY"], rtol=1e-9,
        err_msg="traded weights must equal previous month's target",
    )
    print("test_lookahead_guard: OK")


if __name__ == "__main__":
    test_lookahead_guard()
