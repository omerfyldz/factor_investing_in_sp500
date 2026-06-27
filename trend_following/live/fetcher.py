"""Fetch daily -> month-end adjusted closes for the trend universe via yfinance.

Small and fast: only the 10 ETFs in ALL_ETFS. Uses auto_adjust=True so closes
are dividend-adjusted (total return), matching the backtest and Faber.

Robustness: a transient yfinance failure for one ticker must NOT be silently
treated as "that sleeve is out of trend" (which would wrongly dump it to cash).
We retry, and if the signal month is still missing any ticker we raise — better
to error the run (and re-trigger) than to mis-allocate on incomplete data.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd
import yfinance as yf

from core import ALL_ETFS, to_month_end


def _download_monthly(years_back: int) -> pd.DataFrame:
    """One download attempt -> completed-month closes for ALL_ETFS."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # threads=False avoids yfinance's sqlite-cache "database is locked"
        # contention; 10 tickers download fast enough serially.
        raw = yf.download(
            ALL_ETFS, period=f"{years_back}y", auto_adjust=True,
            progress=False, threads=False,
        )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": ALL_ETFS[0]})
    close.index = pd.to_datetime(close.index)
    monthly = to_month_end(close.sort_index())

    # Use only COMPLETED months. Mid-month, the latest resample bucket is
    # labelled with the month-end date but holds a partial-month price, so
    # acting on it would trade on an unfinished month. Dropping the current
    # (unfinished) month makes every run — a manual dry/live run OR the
    # 1st-of-month cron — use the last fully-closed month, matching Faber's
    # "last trading day of the month" rule.
    current_month_start = pd.Timestamp.today().normalize().replace(day=1)
    return monthly[monthly.index < current_month_start]


def fetch_monthly_closes(years_back: int = 20, retries: int = 3) -> pd.DataFrame:
    """Return a month-end adjusted-close frame (index=completed month-ends,
    cols=ALL_ETFS), guaranteed to have a fully-populated final (signal) row.

    Raises RuntimeError if, after ``retries`` attempts, any ticker is missing
    data for the signal month — so the rebalance fails loudly instead of
    treating a failed download as a (false) "out of trend" signal.
    """
    last_err = ""
    for attempt in range(retries):
        try:
            monthly = _download_monthly(years_back)
        except Exception as exc:  # network / parsing blip
            last_err = f"download error: {exc}"
            time.sleep(2 * (attempt + 1))
            continue

        missing_cols = [t for t in ALL_ETFS if t not in monthly.columns]
        if missing_cols or monthly.empty:
            last_err = f"missing columns {missing_cols} / empty={monthly.empty}"
        else:
            signal_row = monthly[ALL_ETFS].iloc[-1]
            nan_tickers = [t for t in ALL_ETFS if pd.isna(signal_row[t])]
            if not nan_tickers:
                return monthly[ALL_ETFS]
            last_err = (f"no data for signal month "
                        f"{monthly.index[-1].date()}: {nan_tickers}")

        time.sleep(2 * (attempt + 1))  # transient (e.g. 'database is locked')

    raise RuntimeError(
        f"fetch_monthly_closes failed after {retries} attempts — {last_err}. "
        "Refusing to trade on incomplete data; re-run the workflow."
    )
