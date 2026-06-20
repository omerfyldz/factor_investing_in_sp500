"""Fetch live prices and fundamentals from Yahoo Finance for all S&P 500 tickers."""
from __future__ import annotations

import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from config import MA_WINDOWS

MA_COLS = [f"ma_dev_{w}" for w in MA_WINDOWS]


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(0)


def fetch_live_prices(tickers: list[str], years_back: int = 5) -> pd.DataFrame:
    """
    Download daily adjusted close prices for all tickers via yfinance.
    Returns a wide DataFrame: index=date, columns=ticker.
    """
    period = f"{years_back}y"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def _compute_ma_deviations(daily_prices: pd.Series, windows: list[int]) -> dict[str, float]:
    """Compute MA deviation signals for a single stock's daily price series."""
    result = {}
    last_price = daily_prices.iloc[-1]
    for w in windows:
        if len(daily_prices) >= w:
            ma = daily_prices.iloc[-w:].mean()
            result[f"ma_dev_{w}"] = float((ma / last_price) - 1.0) if last_price > 0 else np.nan
        else:
            result[f"ma_dev_{w}"] = np.nan
    return result


def fetch_fundamentals(tickers: list[str], delay: float = 0.05) -> pd.DataFrame:
    """
    Fetch trailing P/E and ROE for each ticker via yfinance info dict.
    Returns a DataFrame with columns: ticker, pe_ratio, roe.
    Note: ROE from yfinance is expressed as a decimal (e.g. 0.35 = 35%).
          The backtest uses ROE in decimal form too (from Tiingo statements).
    """
    rows = []
    print(f"  Fetching fundamentals for {len(tickers)} tickers (may take ~{len(tickers) * delay:.0f}s)...")
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE", None)
            roe = info.get("returnOnEquity", None)
            rows.append({
                "ticker": ticker,
                "pe_ratio": float(pe) if pe is not None and np.isfinite(float(pe)) else np.nan,
                "roe": float(roe) if roe is not None and np.isfinite(float(roe)) else np.nan,
            })
        except Exception:
            rows.append({"ticker": ticker, "pe_ratio": np.nan, "roe": np.nan})
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(tickers)} done...")
        time.sleep(delay)
    return pd.DataFrame(rows)


def build_live_panel(tickers: list[str], years_back: int = 5) -> pd.DataFrame:
    """
    Fetch prices + fundamentals and assemble a single-month panel for the
    most recent month-end.  Returns a DataFrame with one row per ticker.

    Columns: ticker, month, close, momentum_12m, pe_ratio, roe,
             ma_dev_3, ..., ma_dev_1000
    """
    print("Fetching live price data from Yahoo Finance...")
    daily_prices = fetch_live_prices(tickers, years_back=years_back)

    # Month-end close prices (for momentum and MA signal extraction)
    monthly_close = daily_prices.resample("ME").last()

    # Current month-end (last available)
    live_month = _month_end(monthly_close.index[-1])
    print(f"  Signal month: {live_month.date()}")

    rows = []
    for ticker in tickers:
        if ticker not in daily_prices.columns:
            continue
        series = daily_prices[ticker].dropna()
        if len(series) < 13:
            continue

        close_now = series.iloc[-1]

        # 12-month momentum: last monthly close vs 12 months ago
        mo_series = monthly_close[ticker].dropna()
        momentum_12m = np.nan
        if len(mo_series) >= 13:
            momentum_12m = float(mo_series.iloc[-1] / mo_series.iloc[-13] - 1)

        # MA deviations on daily prices
        ma_devs = _compute_ma_deviations(series, MA_WINDOWS)

        row = {
            "ticker": ticker,
            "month": live_month,
            "close": float(close_now),
            "momentum_12m": momentum_12m,
            **ma_devs,
        }
        rows.append(row)

    price_panel = pd.DataFrame(rows)
    print(f"  Price panel: {len(price_panel)} tickers with valid price data")

    print("Fetching fundamentals from Yahoo Finance...")
    fundamentals = fetch_fundamentals(tickers)

    panel = price_panel.merge(fundamentals, on="ticker", how="left")
    panel["month"] = pd.to_datetime(panel["month"])

    print(f"  Live panel ready: {len(panel)} rows, month={live_month.date()}")
    return panel


def fetch_index_monthly() -> pd.DataFrame:
    """
    Fetch S&P 500 index monthly data from yfinance.
    Returns DataFrame with columns: month, close, sma_10m, regime_on, next_ret.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download("^GSPC", period="15y", auto_adjust=True, progress=False)
    daily = raw["Close"].dropna()
    daily.index = pd.to_datetime(daily.index)

    monthly = daily.resample("ME").last().rename("close").to_frame()
    monthly["sma_10m"] = monthly["close"].rolling(10, min_periods=10).mean()
    monthly["regime_on"] = monthly["close"] > monthly["sma_10m"]
    monthly["next_ret"] = monthly["close"].pct_change().shift(-1)
    monthly.index.name = "month"
    monthly = monthly.reset_index()
    monthly["month"] = pd.to_datetime(monthly["month"])
    return monthly


def fetch_index_ma_signals() -> pd.DataFrame:
    """
    Compute 11 MA deviation signals for the S&P 500 index at each month-end.
    Returns DataFrame with columns: month, ma_dev_3, ..., ma_dev_1000.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download("^GSPC", period="15y", auto_adjust=True, progress=False)
    daily = raw["Close"].dropna()
    daily.index = pd.to_datetime(daily.index)

    rows = []
    month_ends = daily.resample("ME").last().index
    for me in month_ends:
        series = daily[daily.index <= me]
        ma_devs = _compute_ma_deviations(series, MA_WINDOWS)
        rows.append({"month": me, **ma_devs})

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month"])
    return df
