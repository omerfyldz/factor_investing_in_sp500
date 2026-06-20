"""
Signal generation for live paper trading.

All factor-scoring logic is self-contained here — no heavy project_core import
needed, so paper_trading/ only requires pandas, numpy, statsmodels, yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import MA_COLS, MA_WINDOWS, TREND_MIN_OBS


# ---------------------------------------------------------------------------
# Inlined from project_core.py (lines 628-641) — no change to logic
# ---------------------------------------------------------------------------
def winsorized_zscore_by_month(df: pd.DataFrame, source: str, out_col: str) -> pd.Series:
    """Cross-sectional winsorised z-score per month (1st/99th percentile clip)."""
    def _transform(s: pd.Series) -> pd.Series:
        x = s.replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 25:
            return pd.Series(np.nan, index=s.index)
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        clipped = x.clip(lo, hi)
        sigma = clipped.std(ddof=0)
        if not np.isfinite(sigma) or sigma == 0:
            return pd.Series(np.nan, index=s.index)
        return (clipped - clipped.mean()) / sigma

    return df.groupby("month", group_keys=False)[source].apply(_transform).rename(out_col)

# ---------------------------------------------------------------------------
# Trend factor (expanding window, no look-ahead) — live month only
# ---------------------------------------------------------------------------

def _fit_ols_hac(y: pd.Series, x: pd.DataFrame, hac_lags: int = 3):
    X = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def compute_expanding_trend_live(
    live_month: pd.Timestamp,
    hist_index_monthly: pd.DataFrame,
    hist_index_ma: pd.DataFrame,
    live_stock_ma: pd.DataFrame,
    min_obs: int = TREND_MIN_OBS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run the expanding-window index trend regression using all historical
    index data strictly before live_month, then apply the fitted coefficients
    to each stock's current-month MA deviations.

    Parameters
    ----------
    live_month        : Current signal month (pd.Timestamp, month-end)
    hist_index_monthly: DataFrame with columns [month, next_ret, ...]
    hist_index_ma     : DataFrame with columns [month, ma_dev_3, ..., ma_dev_1000]
    live_stock_ma     : DataFrame with columns [ticker, month, ma_dev_3, ..., ma_dev_1000]
                        for the current month only.

    Returns
    -------
    DataFrame with columns [ticker, trend_expanding_raw]
    """
    # Merge index returns with MA signals (training data = all months before live_month)
    train_data = (
        hist_index_monthly[["month", "next_ret"]]
        .merge(hist_index_ma[["month", *MA_COLS]], on="month", how="inner")
        .loc[lambda df: df["month"] < live_month]
        .sort_values("month")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["next_ret", *MA_COLS])
        .reset_index(drop=True)
    )

    tickers = live_stock_ma["ticker"].tolist()
    nan_result = pd.DataFrame({"ticker": tickers, "trend_expanding_raw": np.nan})

    if len(train_data) < min_obs:
        print(f"  Trend: only {len(train_data)} training months < {min_obs} required. Using NaN.")
        return nan_result

    y = train_data["next_ret"]
    x = train_data[MA_COLS]

    full_fit = _fit_ols_hac(y, x)
    pvals = full_fit.pvalues.drop(labels=["const"], errors="ignore")
    selected = [col for col in MA_COLS if pvals.get(col, np.nan) <= alpha]

    if selected:
        refit = _fit_ols_hac(y, x[selected])
        params = refit.params.reindex(["const", *MA_COLS]).fillna(0.0)
    else:
        const_only = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
        refit = sm.OLS(y, const_only, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        params = pd.Series(0.0, index=["const", *MA_COLS])
        params["const"] = float(refit.params.get("const", 0.0))

    print(f"  Trend regression: {len(train_data)} training months, {len(selected)} significant MAs selected")

    # Apply coefficients to stocks
    x_stock = live_stock_ma[MA_COLS].to_numpy(dtype=float)
    b = params.reindex(MA_COLS).fillna(0.0).to_numpy(dtype=float)
    intercept = float(params.get("const", 0.0))
    valid = np.isfinite(x_stock).all(axis=1)
    trend_vals = np.full(len(live_stock_ma), np.nan)
    trend_vals[valid] = intercept + (x_stock[valid] @ b)

    return pd.DataFrame({"ticker": tickers, "trend_expanding_raw": trend_vals})


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_live_panel(
    fetched_data: pd.DataFrame,
    hist_index_monthly: pd.DataFrame,
    hist_index_ma: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a current-month factor panel from yfinance-fetched data.

    Parameters
    ----------
    fetched_data      : Output of fetcher.build_live_panel() —
                        one row per ticker, current month.
    hist_index_monthly: Historical S&P 500 monthly data (from processed files or yfinance).
    hist_index_ma     : Historical index MA deviation signals.

    Returns
    -------
    DataFrame with one row per eligible ticker, including z-scored factors:
    roe_z, pe_z, momentum_z, trend_expanding_z, eligible, score columns.
    """
    df = fetched_data.copy()
    live_month = pd.to_datetime(df["month"].iloc[0])

    # --- Raw factor columns ---
    df["pe_clean"] = df["pe_ratio"].where(
        df["pe_ratio"].notna() & df["pe_ratio"].gt(0)
    )
    df["pe_factor"] = -df["pe_clean"]
    df["roe_factor"] = df["roe"]
    df["momentum_factor"] = df["momentum_12m"]

    # --- Trend factor (expanding window) ---
    stock_ma_live = df[["ticker", "month", *MA_COLS]].copy()
    trend_df = compute_expanding_trend_live(
        live_month, hist_index_monthly, hist_index_ma, stock_ma_live
    )
    df = df.merge(trend_df, on="ticker", how="left")
    df["trend_expanding_factor"] = df["trend_expanding_raw"]

    # --- Cross-sectional z-scores (winsorised at 1%/99% per project_core) ---
    df["roe_z"] = winsorized_zscore_by_month(df, "roe_factor", "roe_z")
    df["pe_z"] = winsorized_zscore_by_month(df, "pe_factor", "pe_z")
    df["momentum_z"] = winsorized_zscore_by_month(df, "momentum_factor", "momentum_z")
    df["trend_expanding_z"] = winsorized_zscore_by_month(
        df, "trend_expanding_factor", "trend_expanding_z"
    )

    # --- Eligibility (simplified: price > $5, at least one factor available) ---
    factor_available = (
        df["roe_z"].notna()
        | df["pe_z"].notna()
        | df["momentum_z"].notna()
        | df["trend_expanding_z"].notna()
    )
    df["eligible"] = df["close"].gt(5) & factor_available

    return df


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------

Z_COLS = ["roe_z", "pe_z", "momentum_z", "trend_expanding_z"]


def compute_composite_score(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight composite of the four z-scored factors."""
    scores = panel[Z_COLS].mean(axis=1, skipna=True)
    valid_count = panel[Z_COLS].notna().sum(axis=1)
    scores[valid_count < 3] = np.nan  # require ≥3 of 4 factors
    return scores


def get_target_tickers(live_panel: pd.DataFrame, top_n: int) -> list[str]:
    """
    Score eligible stocks and return the top-N tickers by composite z-score.

    Parameters
    ----------
    live_panel : Output of build_live_panel().
    top_n      : Number of positions to select.

    Returns
    -------
    List of ticker strings, length ≤ top_n.
    """
    df = live_panel.copy()
    df["score"] = compute_composite_score(df)

    eligible = df[df["eligible"] & df["score"].notna()].copy()
    ranked = eligible.sort_values("score", ascending=False)
    selected = ranked.head(top_n)

    tickers = selected["ticker"].tolist()
    print(f"  Top-{top_n} selected: {tickers}")
    return tickers


def print_factor_summary(live_panel: pd.DataFrame) -> None:
    """Print a quick sanity-check table of factor coverage and top scores."""
    df = live_panel.copy()
    df["score"] = compute_composite_score(df)

    coverage = {col: df[col].notna().sum() for col in Z_COLS}
    print("\n  Factor coverage (non-NaN tickers):")
    for col, n in coverage.items():
        print(f"    {col}: {n}/{len(df)}")

    top10 = df[df["score"].notna()].sort_values("score", ascending=False).head(10)
    print("\n  Top-10 composite scores:")
    for _, row in top10.iterrows():
        print(
            f"    {row['ticker']:<6}  score={row['score']:+.3f}"
            f"  roe_z={row['roe_z']:+.2f}  pe_z={row['pe_z']:+.2f}"
            f"  mom_z={row['momentum_z']:+.2f}  trend_z={row['trend_expanding_z']:+.2f}"
        )
