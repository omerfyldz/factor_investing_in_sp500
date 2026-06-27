"""Download daily adjusted-close ETF prices via yfinance and cache to CSV.

Network note: this dev sandbox blocks Yahoo/Stooq by policy, so downloads
happen in GitHub Actions (open network). Once ``data/prices.csv`` is cached
(committed by the Actions run), the backtest can be re-run anywhere offline.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from core import ALL_ETFS

CACHE = Path(__file__).resolve().parent.parent / "data" / "prices.csv"


def download_prices(start: str = "2005-01-01", tickers: list[str] | None = None) -> pd.DataFrame:
    """Daily auto-adjusted (total-return) closes, wide frame: index=date, cols=ticker."""
    tickers = tickers or ALL_ETFS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers, start=start, auto_adjust=True, progress=False, threads=True
        )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:  # single ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index)
    # Keep a stable column order matching ALL_ETFS where present.
    cols = [t for t in tickers if t in close.columns]
    return close[cols].sort_index()


def load_or_download(start: str = "2005-01-01", force: bool = False) -> pd.DataFrame:
    """Return cached prices if present (and not ``force``), else download + cache."""
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    df = download_prices(start)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE)
    return df


def inception_report(daily_close: pd.DataFrame) -> pd.DataFrame:
    """First valid date per ticker — documents what binds the backtest start."""
    rows = []
    for col in daily_close.columns:
        first = daily_close[col].first_valid_index()
        rows.append({"ticker": col, "first_date": None if first is None else first.date()})
    return pd.DataFrame(rows).sort_values("first_date").reset_index(drop=True)
