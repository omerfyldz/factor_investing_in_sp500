"""Vectorised monthly backtest with explicit look-ahead protection and costs.

Timing convention (the part that prevents look-ahead bias)
----------------------------------------------------------
Target weights ``w[t]`` are decided at month-end ``t`` using only closes up to
and including ``t``. We then EARN those weights over month ``t+1``. In vector
form that is ``w.shift(1)`` applied to next month's returns:

    portfolio_return[s] = sum_i( w[s-1, i] * monthly_return[s, i] )

Using ``w[s]`` (un-shifted) against ``return[s]`` would mean trading on a price
we only learn at the close that generates the signal — the classic look-ahead
trap. The ``.shift(1)`` is the single line that makes this honest; it is
covered by ``tests/test_lookahead.py``.
"""
from __future__ import annotations

import pandas as pd

from core import ALL_ETFS, MA_MONTHS, signals, target_weights, to_month_end


def run_backtest(
    daily_close: pd.DataFrame,
    ma_months: int = MA_MONTHS,
    cost_bps: float = 5.0,
) -> dict:
    """Run the Faber GTAA backtest on a daily adjusted-close frame.

    Parameters
    ----------
    daily_close : wide frame (index=date, cols include ALL_ETFS)
    ma_months   : SMA lookback in months (10 = Faber canonical)
    cost_bps    : one-way transaction cost in basis points applied to turnover
                  (5 bps is conservative for these highly liquid ETFs)

    Returns a dict with monthly net returns, NAV, traded weights, turnover.
    """
    monthly = to_month_end(daily_close[ALL_ETFS])

    # Start only once the FULL universe (incl. the BIL cash leg, inception
    # 2007-05) has data. Otherwise the 2005-2007 window would route "off"
    # sleeves into a non-existent BIL that silently earns 0% instead of the
    # real T-bill yield, and run an incomplete universe. Trimming to the first
    # all-present month makes the cash leg always real and the universe whole.
    full = monthly.notna().all(axis=1)
    if full.any():
        monthly = monthly.loc[full.idxmax():]

    monthly_ret = monthly.pct_change()

    w = target_weights(signals(monthly, ma_months))   # decided at month-end t
    w_traded = w.shift(1)                              # LOOK-AHEAD GUARD: held in t+1

    gross = (w_traded * monthly_ret).sum(axis=1, min_count=1)

    # Turnover = sum of |change in target weight| month-over-month (simple
    # approximation that ignores intramonth drift; conservative and standard
    # for low-frequency TAA). First rebalance pays cost on the full allocation.
    turnover = (w_traded.fillna(0.0) - w_traded.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (cost_bps / 1e4)

    net = (gross - cost).dropna()
    nav = (1.0 + net).cumprod()

    return {
        "monthly_returns": net,
        "gross_returns": gross.reindex(net.index),
        "nav": nav,
        "weights": w_traded.reindex(net.index),
        "turnover": turnover.reindex(net.index),
        "monthly_close": monthly,
        "monthly_asset_returns": monthly_ret,
    }


def benchmark_spy(daily_close: pd.DataFrame) -> dict:
    """SPY buy-and-hold, monthly, total return."""
    monthly_ret = to_month_end(daily_close[["SPY"]]).pct_change()["SPY"]
    return _series_to_result(monthly_ret)


def benchmark_60_40(daily_close: pd.DataFrame) -> dict:
    """Static 60% SPY / 40% IEF, rebalanced monthly."""
    mret = to_month_end(daily_close[["SPY", "IEF"]]).pct_change()
    blended = 0.6 * mret["SPY"] + 0.4 * mret["IEF"]
    return _series_to_result(blended)


def _series_to_result(monthly_ret: pd.Series) -> dict:
    net = monthly_ret.dropna()
    return {"monthly_returns": net, "nav": (1.0 + net).cumprod()}


def align_common(*results: dict) -> list[dict]:
    """Trim a set of results to their common month index for fair comparison."""
    idx = None
    for r in results:
        idx = r["monthly_returns"].index if idx is None else idx.intersection(
            r["monthly_returns"].index
        )
    out = []
    for r in results:
        ret = r["monthly_returns"].loc[idx]
        out.append({**r, "monthly_returns": ret, "nav": (1.0 + ret).cumprod()})
    return out
