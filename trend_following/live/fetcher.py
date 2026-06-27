"""Fetch daily -> month-end adjusted closes for the trend universe via yfinance.

Small and fast: only the 10 ETFs in ALL_ETFS. Uses auto_adjust=True so closes
are dividend-adjusted (total return), matching the backtest and Faber.
"""
from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

from core import ALL_ETFS, to_month_end


def fetch_monthly_closes(years_back: int = 20) -> pd.DataFrame:
    """Return a month-end adjusted-close frame (index=month-end, cols=ALL_ETFS).

    20 years is far more than the 10-month SMA needs; it just guarantees a full
    warm-up window even if a provider trims early history.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ALL_ETFS, period=f"{years_back}y", auto_adjust=True,
            progress=False, threads=True,
        )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": ALL_ETFS[0]})
    close.index = pd.to_datetime(close.index)
    monthly = to_month_end(close.sort_index())
    missing = [t for t in ALL_ETFS if t not in monthly.columns]
    if missing:
        raise RuntimeError(f"yfinance did not return data for: {missing}")
    return monthly[ALL_ETFS]
