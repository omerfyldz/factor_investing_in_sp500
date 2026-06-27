"""
Core trend-following signal logic — the SINGLE SOURCE OF TRUTH.

Both the backtest (``backtest/engine.py``) and the live trader
(``live/signals.py``) import these functions, so the strategy that is
validated historically is byte-for-byte the strategy that trades live.
No drift between backtest and production is possible.

Strategy — Faber (2007) GTAA timing rule
----------------------------------------
At each month-end, for each risk ETF: hold it if its month-end close is above
its trailing 10-month simple moving average; otherwise move that sleeve's
capital to T-bills (BIL). Equal weight across the risk sleeves (1/N each);
any OFF sleeve's 1/N budget is parked in BIL.

Reference: Mebane T. Faber, "A Quantitative Approach to Tactical Asset
Allocation," The Journal of Wealth Management (2007). SSRN id=962461.
The 10-month SMA is Faber's canonical, pre-registered parameter (it
corresponds closely to the 200-day MA); using it as-is is what makes the
rule hard to overfit.
"""
from __future__ import annotations

import pandas as pd

# --- Universe: one clean representative per asset class ------------------
# (QQQ/XLK/XLE/XLF deliberately excluded as redundant with SPY; USO excluded
#  for contango/roll drag — DBC used for broad commodities instead.)
RISK_ETFS = ["SPY", "EFA", "EEM", "TLT", "IEF", "HYG", "GLD", "DBC", "VNQ"]
CASH_ETF = "BIL"                      # T-bill leg; same ticker in backtest + live
ALL_ETFS = RISK_ETFS + [CASH_ETF]

MA_MONTHS = 10                        # Faber's 10-month SMA (~200-day MA)


def to_month_end(daily_close: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily adjusted-close frame to month-end last observations."""
    return daily_close.resample("ME").last()


def sma(monthly_close: pd.DataFrame, ma_months: int = MA_MONTHS) -> pd.DataFrame:
    """Trailing simple moving average of month-end closes.

    ``min_periods=ma_months`` means the SMA is NaN until a full window of
    history exists (the warm-up period), preventing a signal from firing on
    partial data.
    """
    return monthly_close.rolling(ma_months, min_periods=ma_months).mean()


def signals(monthly_close: pd.DataFrame, ma_months: int = MA_MONTHS) -> pd.DataFrame:
    """Boolean 'in-trend' signal per risk ETF per month: close > 10-month SMA.

    Only ``RISK_ETFS`` are signalled (BIL is the destination, never signalled).
    During the SMA warm-up (NaN), the signal is False — i.e. the sleeve sits in
    T-bills until enough history exists to judge its trend.
    """
    risk = monthly_close[RISK_ETFS]
    ma = sma(risk, ma_months)
    sig = risk > ma
    return sig.where(ma.notna(), other=False)


def target_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean signals to target portfolio weights (each row sums to 1).

    Each of the N = len(RISK_ETFS) sleeves carries an equal 1/N budget:
      * sleeve ON  -> its 1/N is invested in the sleeve's ETF
      * sleeve OFF -> its 1/N is parked in BIL
    So ``weight[BIL] = (number of OFF sleeves) / N``.
    """
    n = len(RISK_ETFS)
    on = sig[RISK_ETFS].astype(float)
    w = pd.DataFrame(0.0, index=sig.index, columns=ALL_ETFS)
    w[RISK_ETFS] = on / n
    w[CASH_ETF] = (n - on.sum(axis=1)) / n
    return w


def latest_target_weights(
    monthly_close: pd.DataFrame, ma_months: int = MA_MONTHS
) -> "pd.Series":
    """Target weights for the most recent fully-closed month (live use).

    Returns a Series indexed by ticker that sums to 1.0. This is exactly the
    last row of ``target_weights(signals(...))`` — the live trader and the
    backtest therefore compute identical allocations from identical data.
    """
    w = target_weights(signals(monthly_close, ma_months))
    return w.iloc[-1]
