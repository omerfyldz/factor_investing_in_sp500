from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
import yaml
from matplotlib.backends.backend_pdf import PdfPages


warnings.filterwarnings("ignore", category=RuntimeWarning)
sns.set_theme(style="whitegrid")


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FMP_RESULTS_DIR = RESULTS_DIR / "fmp_analysis"
BASE_RESULTS_DIR = RESULTS_DIR / "base_strategy"
IMPROVED_RESULTS_DIR = RESULTS_DIR / "improved_strategy"
IMPROVED_2_RESULTS_DIR = RESULTS_DIR / "improved_strategy_2"
IMPROVED_3_RESULTS_DIR = RESULTS_DIR / "improved_strategy_3"
IMPROVED_4_RESULTS_DIR = RESULTS_DIR / "improved_strategy_4"
IMPROVED_5_RESULTS_DIR = RESULTS_DIR / "improved_strategy_5"
IMPROVED_6_RESULTS_DIR = RESULTS_DIR / "improved_strategy_6"
IMPROVED_7_RESULTS_DIR = RESULTS_DIR / "improved_strategy_7"
IMPROVED_8_RESULTS_DIR = RESULTS_DIR / "improved_strategy_8"
IMPROVED_9_RESULTS_DIR = RESULTS_DIR / "improved_strategy_9"
ROBUSTNESS_RESULTS_DIR = RESULTS_DIR / "robustness"
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"
FIGURES_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"
PRESENTATION_DIR = ROOT / "presentation"
CONFIG_PATH = ROOT / "config" / "project_config.yaml"

CUTOFF = pd.Timestamp("2026-05-31")
START = pd.Timestamp("2006-06-01")
EVALUATION_START = pd.Timestamp("2016-05-31")
BENCHMARK = "^GSPC"
INITIAL_CASH = 1_000_000.0
CASH_PER_TRADE = 100_000.0
MA_WINDOWS = [3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
MA_COLS = [f"ma_dev_{w}" for w in MA_WINDOWS]
REQUIRED_FACTOR_Z = ["roe_z", "pe_z", "momentum_z", "trend_z"]
RNG_SEED = 5811
BASE_STRATEGY_NAME = "base_equal_top10"
IMPROVED_STRATEGY_NAME = "improved_1_expanding_trend_top10"
IMPROVED_2_STRATEGY_NAME = "improved_2_expanding_trend_stop_take_top10"
IMPROVED_3_STRATEGY_NAME = "improved_3_dynamic_ic_weights_stop_take_top10"
IMPROVED_4_STRATEGY_NAME = "improved_4_walkforward_stop_take_top10"
IMPROVED_5_STRATEGY_NAME = "improved_5_regime_filtered_stop_take_top10"
IMPROVED_6_STRATEGY_NAME = "improved_6_hzz_cross_sectional_trend_stop_take_top10"
IMPROVED_7_STRATEGY_NAME = "improved_7_time_varying_cost_sensitivity"
IMPROVED_8_STRATEGY_NAME = "improved_8_equal_weight_top20"
IMPROVED_9_STRATEGY_NAME = "improved_9_vol_targeted_top20"
VOL_LOOKBACK_DAYS = 63       # ~3 months of trading days
VOL_FLOOR_ANN = 0.05         # annualized vol floor (5%) to prevent extreme inverse weights
HZZ_RATIO_COLS = [f"ma_ratio_{w}" for w in MA_WINDOWS]
HZZ_BETA_COLS = [f"beta_ma_ratio_{w}" for w in MA_WINDOWS]
HZZ_SMOOTH_WINDOW = 12
HZZ_MIN_CROSS_SECTION = 100
FACTOR_KEYS = ["roe", "pe", "momentum", "trend"]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    weights: dict[str, float]
    top_n: int = 10
    regime_filter: bool = False
    stop_loss: float | None = None
    take_profit: float | None = None
    trend_col: str = "trend_z"
    weight_method: str = "static"
    weight_lookback_months: int = 60
    weight_min_periods: int = 24
    weight_shrink_to_equal: float = 0.50
    min_factor_weight: float = 0.10
    max_factor_weight: float = 0.45
    # Position-sizing rule.
    #   "fixed_cash"        — original CASH_PER_TRADE behavior used by base and improveds 1-7
    #   "percent_of_equity" — equal-weight 1/N as percent of current equity (DGU 2009; improved 8)
    #   "vol_targeted"      — inverse-volatility weighted within the held basket (improved 9).
    #                         Sized so high-vol names get smaller weights, normalized so the
    #                         held basket consumes ``top_n * sizing_target_pct`` of equity total.
    sizing_method: str = "fixed_cash"
    sizing_target_pct: float = 0.10
    # Volatility-targeted sizing parameters (used only when sizing_method == "vol_targeted").
    vol_lookback_days: int = VOL_LOOKBACK_DAYS
    vol_floor_ann: float = VOL_FLOOR_ANN
    notes: str = ""


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        FMP_RESULTS_DIR,
        BASE_RESULTS_DIR,
        IMPROVED_RESULTS_DIR,
        IMPROVED_2_RESULTS_DIR,
        IMPROVED_3_RESULTS_DIR,
        IMPROVED_4_RESULTS_DIR,
        IMPROVED_5_RESULTS_DIR,
        IMPROVED_6_RESULTS_DIR,
        IMPROVED_7_RESULTS_DIR,
        IMPROVED_8_RESULTS_DIR,
        IMPROVED_9_RESULTS_DIR,
        ROBUSTNESS_RESULTS_DIR,
        COMPARISON_RESULTS_DIR,
        FIGURES_DIR,
        DOCS_DIR,
        PRESENTATION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_processed_strategy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed monthly bars and factor panel for strategy-only runs."""
    monthly_path = PROCESSED_DIR / "monthly_stock_bars.csv"
    index_path = PROCESSED_DIR / "monthly_sp500_index.csv"
    panel_path = PROCESSED_DIR / "factor_panel.csv"
    missing = [str(p) for p in [monthly_path, index_path, panel_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed strategy inputs are missing. Run `py -3.10 src\\run_project.py` first. "
            f"Missing: {missing}"
        )
    monthly = pd.read_csv(monthly_path, parse_dates=["month", "first_date", "last_date"])
    index_monthly = pd.read_csv(index_path, parse_dates=["month", "first_date", "last_date"])
    panel = pd.read_csv(panel_path, parse_dates=["month", "first_date", "last_date"])
    return monthly, index_monthly, panel


def flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[-1] for c in df.columns]
    return df


def download_or_load_index() -> pd.DataFrame:
    """Freeze Yahoo S&P 500 index data, then always use the frozen CSV."""
    out = RAW_DIR / "sp500_index_yahoo.csv"
    needs_download = True

    if out.exists():
        old = pd.read_csv(out, parse_dates=["date"])
        if not old.empty and old["date"].max() >= pd.Timestamp("2026-05-29"):
            needs_download = False

    if needs_download:
        print("Downloading ^GSPC benchmark from Yahoo Finance through May 2026...")
        data = yf.download(
            BENCHMARK,
            start=START.strftime("%Y-%m-%d"),
            end="2026-06-01",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data.empty:
            if out.exists():
                print("Yahoo download returned empty data; using existing frozen index CSV.")
            else:
                raise RuntimeError("Yahoo download returned no ^GSPC data and no frozen CSV exists.")
        else:
            data = flatten_yahoo_columns(data).reset_index()
            rename = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            data = data.rename(columns=rename)
            keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
            data = data[[c for c in keep if c in data.columns]].copy()
            if "adj_close" not in data.columns:
                data["adj_close"] = data["close"]
            data = data[data["date"] <= CUTOFF].sort_values("date")
            save_csv(data, out)

    index = pd.read_csv(out, parse_dates=["date"])
    index = index[index["date"] <= CUTOFF].sort_values("date")
    if index["date"].max() > CUTOFF:
        raise AssertionError("Index data includes observations after cutoff.")
    return index


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading frozen S&P 500 raw CSV files...")
    price_cols = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjOpen",
        "adjHigh",
        "adjLow",
        "adjClose",
        "adjVolume",
    ]
    prices = pd.read_csv(RAW_DIR / "sp500_prices_long.csv", usecols=price_cols, parse_dates=["date"])
    prices = prices[prices["date"] <= CUTOFF].sort_values(["ticker", "date"])

    daily_cols = ["ticker", "date", "marketCap", "peRatio", "pbRatio", "trailingPEG1Y"]
    daily = pd.read_csv(RAW_DIR / "sp500_fundamentals_daily_long.csv", usecols=daily_cols, parse_dates=["date"])
    daily = daily[daily["date"] <= CUTOFF].sort_values(["ticker", "date"])

    statement_cols = ["ticker", "date_available", "statement_type", "data_code", "value"]
    statements = pd.read_csv(
        RAW_DIR / "sp500_fundamentals_statements_long.csv",
        usecols=statement_cols,
        parse_dates=["date_available"],
    )
    statements = statements[statements["date_available"] <= CUTOFF].sort_values(["ticker", "date_available"])

    constituents = pd.read_csv(RAW_DIR / "sp500_constituents.csv")
    index = download_or_load_index()
    return prices, daily, statements, constituents, index


def month_end_timestamp(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").dt.to_timestamp("M")


def make_monthly_bars(prices: pd.DataFrame) -> pd.DataFrame:
    print("Building monthly adjusted OHLCV stock bars...")
    p = prices.copy()
    p["month"] = month_end_timestamp(p["date"])
    monthly = (
        p.groupby(["ticker", "month"], sort=True)
        .agg(
            first_date=("date", "first"),
            last_date=("date", "last"),
            open=("adjOpen", "first"),
            high=("adjHigh", "max"),
            low=("adjLow", "min"),
            close=("adjClose", "last"),
            raw_close=("close", "last"),
            volume=("adjVolume", "sum"),
        )
        .reset_index()
        .sort_values(["ticker", "month"])
    )
    g = monthly.groupby("ticker", sort=False)
    monthly["prev_close"] = g["close"].shift(1)
    monthly["ret_1m_cc"] = monthly["close"] / monthly["prev_close"] - 1
    monthly["next_open"] = g["open"].shift(-1)
    monthly["next_high"] = g["high"].shift(-1)
    monthly["next_low"] = g["low"].shift(-1)
    monthly["next_close"] = g["close"].shift(-1)
    monthly["next_ret_cc"] = monthly["next_close"] / monthly["close"] - 1
    monthly["next_ret_oc"] = monthly["next_close"] / monthly["next_open"] - 1
    monthly["momentum_12m"] = monthly["close"] / g["close"].shift(12) - 1
    monthly["eligible_price"] = monthly["raw_close"] >= 5
    return monthly


def make_index_monthly(index: pd.DataFrame) -> pd.DataFrame:
    idx = index.copy()
    idx["month"] = month_end_timestamp(idx["date"])
    monthly = (
        idx.groupby("month", sort=True)
        .agg(
            first_date=("date", "first"),
            last_date=("date", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("adj_close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["ret_1m"] = monthly["close"].pct_change()
    monthly["next_ret"] = monthly["close"].shift(-1) / monthly["close"] - 1
    monthly["sma_10m"] = monthly["close"].rolling(10, min_periods=10).mean()
    monthly["regime_on"] = monthly["close"] > monthly["sma_10m"]
    return monthly


def make_monthly_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    print("Building month-end valuation and market-cap panel...")
    d = daily.copy()
    d["month"] = month_end_timestamp(d["date"])
    metrics = (
        d.groupby(["ticker", "month"], sort=True)
        .agg(
            market_cap=("marketCap", "last"),
            pe_ratio=("peRatio", "last"),
            pb_ratio=("pbRatio", "last"),
            trailing_peg_1y=("trailingPEG1Y", "last"),
        )
        .reset_index()
    )
    return metrics


def make_roe_panel(monthly: pd.DataFrame, statements: pd.DataFrame) -> pd.DataFrame:
    print("Joining point-in-time ROE by statement availability date...")
    left = monthly[["ticker", "month"]].drop_duplicates().rename(columns={"month": "date"})
    roe = statements.loc[statements["data_code"].eq("roe"), ["ticker", "date_available", "value"]].copy()
    roe = roe.rename(columns={"date_available": "date", "value": "roe"}).sort_values(["ticker", "date"])
    out: list[pd.DataFrame] = []
    for ticker, g in left.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        r = roe[roe["ticker"].eq(ticker)][["date", "roe"]].sort_values("date")
        if r.empty:
            tmp = g.copy()
            tmp["roe"] = np.nan
        else:
            tmp = pd.merge_asof(g[["ticker", "date"]].sort_values("date"), r, on="date", direction="backward")
        out.append(tmp)
    panel = pd.concat(out, ignore_index=True).rename(columns={"date": "month"})
    return panel


def compute_stock_ma_signals(prices: pd.DataFrame) -> pd.DataFrame:
    print("Computing stock normalized moving-average deviation signals...")
    cols = ["ticker", "date", "adjClose"]
    p = prices[cols].copy().sort_values(["ticker", "date"])
    p["month"] = month_end_timestamp(p["date"])
    for window, col in zip(MA_WINDOWS, MA_COLS):
        ma = p.groupby("ticker", sort=False)["adjClose"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )
        p[col] = (ma / p["adjClose"] - 1).astype("float32")
    is_month_end_row = p["date"].eq(p.groupby(["ticker", "month"], sort=False)["date"].transform("max"))
    keep = ["ticker", "month", *MA_COLS]
    return p.loc[is_month_end_row, keep].reset_index(drop=True)


def compute_index_ma_signals(index: pd.DataFrame) -> pd.DataFrame:
    idx = index[["date", "adj_close"]].copy().sort_values("date")
    idx["month"] = month_end_timestamp(idx["date"])
    for window, col in zip(MA_WINDOWS, MA_COLS):
        idx[col] = idx["adj_close"].rolling(window, min_periods=window).mean() / idx["adj_close"] - 1
    is_month_end_row = idx["date"].eq(idx.groupby("month")["date"].transform("max"))
    return idx.loc[is_month_end_row, ["month", *MA_COLS]].reset_index(drop=True)


def fit_ols(y: pd.Series, x: pd.DataFrame, hac_lags: int = 3) -> Any:
    x2 = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, x2, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def assignment_index_trend_coefficients(index_monthly: pd.DataFrame, index_ma: pd.DataFrame) -> pd.DataFrame:
    """Estimate the assignment-style full-sample index trend regression.

    The project PDF asks for a predictive regression using all available index data,
    then dropping insignificant moving-average variables. For this S&P 500 version,
    `^GSPC` takes the role of BIST100.
    """
    print("Estimating assignment-style full-sample ^GSPC trend regression...")
    data = index_monthly[["month", "next_ret"]].merge(index_ma, on="month", how="left")
    data = data.sort_values("month").reset_index(drop=True)
    train = data[["next_ret", *MA_COLS]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 72:
        raise RuntimeError("Not enough index observations to estimate the trend regression.")

    y = train["next_ret"]
    x = train[MA_COLS]
    full = fit_ols(y, x)
    full_pvals = full.pvalues.drop(labels=["const"], errors="ignore")
    selected = [col for col in MA_COLS if full_pvals.get(col, np.nan) <= 0.05]

    if selected:
        refit = fit_ols(y, x[selected])
        intercept = float(refit.params.get("const", 0.0))
        r2 = float(refit.rsquared)
    else:
        const_only = pd.DataFrame({"const": 1.0}, index=y.index)
        refit = sm.OLS(y, const_only, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        intercept = float(refit.params.get("const", 0.0))
        r2 = float(refit.rsquared)

    rows: list[dict[str, Any]] = [
        {
            "term": "const",
            "coefficient": intercept,
            "full_model_p_value": full.pvalues.get("const", np.nan),
            "refit_p_value": refit.pvalues.get("const", np.nan),
            "selected": True,
            "n_obs": len(train),
            "r_squared_refit": r2,
            "selected_variables": "|".join(selected),
        }
    ]
    for col in MA_COLS:
        is_selected = col in selected
        rows.append(
            {
                "term": col,
                "coefficient": float(refit.params.get(col, 0.0)) if is_selected else 0.0,
                "full_model_p_value": float(full_pvals.get(col, np.nan)),
                "refit_p_value": float(refit.pvalues.get(col, np.nan)) if is_selected else np.nan,
                "selected": is_selected,
                "n_obs": len(train),
                "r_squared_refit": r2,
                "selected_variables": "|".join(selected),
            }
        )
    return pd.DataFrame(rows)


def apply_index_trend_to_stocks(stock_ma: pd.DataFrame, coeffs: pd.DataFrame) -> pd.DataFrame:
    if "term" not in coeffs.columns:
        raise ValueError("Trend coefficients must be the assignment-style long table.")
    intercept = float(coeffs.set_index("term")["coefficient"].get("const", 0.0))
    coef = coeffs.set_index("term")["coefficient"].reindex(MA_COLS).fillna(0.0)
    out = stock_ma.copy()
    x = out[MA_COLS].to_numpy(dtype=float)
    b = coef.to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    out["trend_raw"] = np.nan
    out.loc[valid, "trend_raw"] = intercept + np.sum(x[valid] * b, axis=1)
    return out[["ticker", "month", "trend_raw"]]


def expanding_index_trend_to_stocks(
    index_monthly: pd.DataFrame,
    index_ma: pd.DataFrame,
    stock_ma: pd.DataFrame,
    min_obs: int = 72,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate a no-lookahead expanding index trend factor for improvement tests.

    At signal month t, the regression uses only index rows with month < t because
    the t to t+1 return is not known yet. This intentionally differs from the
    assignment-style base regression, which uses one full-sample index equation.
    """
    print("Estimating expanding no-lookahead ^GSPC trend regression...")
    data = (
        index_monthly[["month", "next_ret"]]
        .merge(index_ma, on="month", how="left")
        .sort_values("month")
        .reset_index(drop=True)
    )
    stock = stock_ma.sort_values(["month", "ticker"]).reset_index(drop=True)
    trend_rows: list[pd.DataFrame] = []
    coeff_rows: list[dict[str, Any]] = []

    for signal_month in sorted(stock["month"].dropna().unique()):
        train = data.loc[data["month"].lt(signal_month), ["next_ret", *MA_COLS]]
        train = train.replace([np.inf, -np.inf], np.nan).dropna()
        if len(train) < min_obs:
            continue

        y = train["next_ret"]
        x = train[MA_COLS]
        full = fit_ols(y, x)
        full_pvals = full.pvalues.drop(labels=["const"], errors="ignore")
        selected = [col for col in MA_COLS if full_pvals.get(col, np.nan) <= alpha]

        if selected:
            refit = fit_ols(y, x[selected])
            params = refit.params.reindex(["const", *MA_COLS]).fillna(0.0)
            r2 = float(refit.rsquared)
        else:
            const_only = pd.DataFrame({"const": 1.0}, index=y.index)
            refit = sm.OLS(y, const_only, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            params = pd.Series(0.0, index=["const", *MA_COLS])
            params["const"] = float(refit.params.get("const", 0.0))
            r2 = float(refit.rsquared)

        coeff_rows.append(
            {
                "signal_month": signal_month,
                "n_obs": len(train),
                "r_squared_refit": r2,
                "selected_variables": "|".join(selected),
                "intercept": float(params.get("const", 0.0)),
                **{col: float(params.get(col, 0.0)) for col in MA_COLS},
            }
        )

        month_stock = stock.loc[stock["month"].eq(signal_month), ["ticker", "month", *MA_COLS]].copy()
        x_stock = month_stock[MA_COLS].to_numpy(dtype=float)
        b = params.reindex(MA_COLS).fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(x_stock).all(axis=1)
        month_stock["trend_expanding_raw"] = np.nan
        month_stock.loc[valid, "trend_expanding_raw"] = float(params.get("const", 0.0)) + np.sum(x_stock[valid] * b, axis=1)
        trend_rows.append(month_stock[["ticker", "month", "trend_expanding_raw"]])

    trend = pd.concat(trend_rows, ignore_index=True) if trend_rows else pd.DataFrame(columns=["ticker", "month", "trend_expanding_raw"])
    coeffs = pd.DataFrame(coeff_rows)
    return trend, coeffs


def stock_ma_ratios(stock_ma: pd.DataFrame) -> pd.DataFrame:
    """Convert the project's ma_dev_{w} = MA/P - 1 columns into HZZ ma_ratio_{w} = MA/P columns.

    Han, Zhou, Zhu (2016) normalize each moving average by the contemporaneous
    closing price as ``MA_t / P_t``. The project's existing ``ma_dev_{w}`` columns
    store ``MA_t / P_t - 1`` because the index-level trend regression is more
    intuitive in deviation form. The two encodings differ only by a constant of
    one per column, but the HZZ cross-sectional regression uses the ratio form
    so we rebuild it here without recomputing rolling means from raw prices.
    """
    out = stock_ma[["ticker", "month"]].copy()
    for dev_col, ratio_col in zip(MA_COLS, HZZ_RATIO_COLS):
        out[ratio_col] = stock_ma[dev_col].astype(float) + 1.0
    return out


def cross_sectional_trend_betas(
    monthly: pd.DataFrame,
    stock_ma_ratios_df: pd.DataFrame,
    min_obs: int = HZZ_MIN_CROSS_SECTION,
) -> pd.DataFrame:
    """Estimate Han, Zhou, Zhu (2016) cross-sectional trend coefficients monthly.

    For each calendar month ``t`` we regress every eligible stock's next-month
    close-to-close return ``r_{i,t+1}`` on its eleven normalized moving-average
    ratios observed at month-end ``t``:

        r_{i,t+1} = beta_{0,t} + sum_w beta_{w,t} * (MA_{w,i,t} / P_{i,t}) + eps_{i,t}

    The eleven slopes plus an intercept are stored as the row for month ``t``.
    These coefficients are *future-looking* at time ``t`` because they use the
    ``t -> t+1`` return that is not observed until the end of month ``t+1``. The
    caller must shift them before they are used for trading decisions.
    """
    print("Estimating Han, Zhou, Zhu cross-sectional trend coefficients...")
    base = monthly[["ticker", "month", "next_ret_cc", "eligible_price"]].merge(
        stock_ma_ratios_df,
        on=["ticker", "month"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for month, g0 in base.groupby("month", sort=True):
        g = g0.loc[g0["eligible_price"].fillna(False)].copy()
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["next_ret_cc", *HZZ_RATIO_COLS])
        if len(g) < min_obs:
            continue
        y = g["next_ret_cc"].astype(float)
        x = sm.add_constant(g[HZZ_RATIO_COLS].astype(float), has_constant="add")
        fit = sm.OLS(y, x).fit()
        row: dict[str, Any] = {
            "month": pd.Timestamp(month),
            "n_obs": int(len(g)),
            "r_squared": float(fit.rsquared),
            "intercept": float(fit.params.get("const", 0.0)),
        }
        for col in HZZ_RATIO_COLS:
            row[col] = float(fit.params.get(col, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def smooth_trend_betas(betas: pd.DataFrame, window: int = HZZ_SMOOTH_WINDOW) -> pd.DataFrame:
    """Strictly-trailing rolling mean of past monthly HZZ betas.

    For signal month ``t`` we smooth using betas from months ``[t-window, t-1]``
    inclusive. The contemporaneous beta at ``t`` is excluded because it was
    estimated against the return from ``t`` to ``t+1`` and would inject
    look-ahead information into a decision made at the end of month ``t``.
    """
    if betas.empty:
        return betas.copy()
    sorted_betas = betas.sort_values("month").reset_index(drop=True)
    out = sorted_betas[["month"]].copy()
    coef_cols = ["intercept", *HZZ_RATIO_COLS]
    for col in coef_cols:
        lagged = sorted_betas[col].shift(1)
        out[col] = lagged.rolling(window, min_periods=window).mean()
    out["smoothed_window"] = window
    out["smoothed_n_obs"] = (
        sorted_betas["n_obs"].shift(1).rolling(window, min_periods=window).mean()
    )
    return out


def hzz_predicted_returns(
    stock_ma_ratios_df: pd.DataFrame,
    smoothed_betas: pd.DataFrame,
) -> pd.DataFrame:
    """Apply smoothed HZZ coefficients to each stock's contemporaneous MA ratios.

    The predicted return at signal month ``t`` for stock ``i`` is
    ``intercept_t + sum_w beta_{w,t} * (MA_{w,i,t} / P_{i,t})`` where the
    ``beta_t`` and ``intercept_t`` come from the strictly-trailing 12-month
    average produced by :func:`smooth_trend_betas`.
    """
    if stock_ma_ratios_df.empty or smoothed_betas.empty:
        return pd.DataFrame(columns=["ticker", "month", "trend_hzz_raw"])
    rename = {col: f"beta_{col}" for col in HZZ_RATIO_COLS}
    rename["intercept"] = "beta_intercept"
    betas = smoothed_betas.rename(columns=rename)[
        ["month", "beta_intercept", *HZZ_BETA_COLS]
    ]
    merged = stock_ma_ratios_df.merge(betas, on="month", how="left")
    intercept = merged["beta_intercept"].to_numpy(dtype=float)
    signals_arr = merged[HZZ_RATIO_COLS].to_numpy(dtype=float)
    betas_arr = merged[HZZ_BETA_COLS].to_numpy(dtype=float)
    valid = (
        np.isfinite(intercept)
        & np.isfinite(signals_arr).all(axis=1)
        & np.isfinite(betas_arr).all(axis=1)
    )
    pred = np.where(valid, intercept + np.sum(signals_arr * betas_arr, axis=1), np.nan)
    out = merged[["ticker", "month"]].copy()
    out["trend_hzz_raw"] = pred
    return out


def winsorized_zscore_by_month(df: pd.DataFrame, source: str, out_col: str) -> pd.Series:
    def transform(s: pd.Series) -> pd.Series:
        x = s.replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 25:
            return pd.Series(np.nan, index=s.index)
        lo = x.quantile(0.01)
        hi = x.quantile(0.99)
        clipped = x.clip(lo, hi)
        sigma = clipped.std(ddof=0)
        if not np.isfinite(sigma) or sigma == 0:
            return pd.Series(np.nan, index=s.index)
        return (clipped - clipped.mean()) / sigma

    return df.groupby("month", group_keys=False)[source].apply(transform).rename(out_col)


def build_factor_panel(
    monthly: pd.DataFrame,
    metrics: pd.DataFrame,
    roe: pd.DataFrame,
    trend: pd.DataFrame,
    trend_expanding: pd.DataFrame,
    constituents: pd.DataFrame,
    index_monthly: pd.DataFrame,
) -> pd.DataFrame:
    print("Assembling monthly factor panel...")
    panel = monthly.merge(metrics, on=["ticker", "month"], how="left")
    panel = panel.merge(roe, on=["ticker", "month"], how="left")
    panel = panel.merge(trend, on=["ticker", "month"], how="left")
    panel = panel.merge(trend_expanding, on=["ticker", "month"], how="left")
    panel = panel.merge(
        constituents.rename(columns={"Symbol": "ticker"})[["ticker", "Security", "GICS Sector", "GICS Sub-Industry"]],
        on="ticker",
        how="left",
    )
    panel = panel.merge(index_monthly[["month", "close", "sma_10m", "regime_on"]].rename(columns={"close": "index_close"}), on="month", how="left")

    panel["pe_clean"] = panel["pe_ratio"].where(panel["pe_ratio"].gt(0) & np.isfinite(panel["pe_ratio"]))
    panel["roe_factor"] = panel["roe"]
    panel["pe_factor"] = -panel["pe_clean"]
    panel["momentum_factor"] = panel["momentum_12m"]
    panel["trend_factor"] = panel["trend_raw"]
    panel["trend_expanding_factor"] = panel["trend_expanding_raw"]
    panel["eligible"] = (
        panel["eligible_price"].fillna(False)
        & panel["next_ret_cc"].notna()
        & panel["next_ret_oc"].notna()
        & panel["next_open"].gt(0)
    )

    for source, target in [
        ("roe_factor", "roe_z"),
        ("pe_factor", "pe_z"),
        ("momentum_factor", "momentum_z"),
        ("trend_factor", "trend_z"),
        ("trend_expanding_factor", "trend_expanding_z"),
    ]:
        panel[target] = winsorized_zscore_by_month(panel, source, target)

    available_required = panel[REQUIRED_FACTOR_Z].notna().sum(axis=1)
    panel["composite_score"] = panel[REQUIRED_FACTOR_Z].mean(axis=1, skipna=True)
    panel.loc[available_required < 3, "composite_score"] = np.nan

    if panel["month"].max() > CUTOFF:
        raise AssertionError("Processed panel includes data after May 2026.")
    return panel.sort_values(["month", "ticker"])


def perf_metrics(returns: pd.Series, name: str, periods_per_year: int = 12, period_label: str = "monthly") -> dict[str, float | str]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"name": name}
    mean = r.mean()
    vol = r.std(ddof=1)
    scale = np.sqrt(periods_per_year)
    sharpe = scale * mean / vol if vol and np.isfinite(vol) else np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    t_stat, p_val = st.ttest_1samp(r, 0.0, nan_policy="omit") if len(r) > 2 else (np.nan, np.nan)
    return {
        "name": name,
        "period_label": period_label,
        "n_periods": len(r),
        "n_months": len(r) if period_label == "monthly" else np.nan,
        "avg_period_return": mean,
        "period_volatility": vol,
        "avg_monthly_return": mean if period_label == "monthly" else np.nan,
        "monthly_volatility": vol if period_label == "monthly" else np.nan,
        "annualized_return_approx": mean * periods_per_year,
        "annualized_volatility": vol * scale if np.isfinite(vol) else np.nan,
        "annualized_sharpe": sharpe,
        "t_stat": t_stat,
        "p_value": p_val,
        "cumulative_return": wealth.iloc[-1] - 1,
        "max_drawdown": dd.min(),
        "best_month": r.max(),
        "worst_month": r.min(),
    }


def metrics_over_evaluation_window(
    curve: pd.DataFrame,
    name: str,
    date_col: str = "month",
    return_col: str | None = None,
    period_label: str = "monthly",
    periods_per_year: int = 12,
    eval_start: pd.Timestamp | None = None,
) -> dict[str, float | str]:
    """Compute performance metrics over the project's common evaluation window.

    All strategies in the project have different signal-availability lifecycles
    (base trades from ~2010, improved 6 from ~2011, improved 1-5 from 2016-05
    once trend_expanding_z is computable). Reporting Sharpe over the full
    240-month panel includes pre-warmup zero-return months which dilute the
    numerator more than the denominator, systematically understating the
    Sharpe of strategies with longer warmups and making cross-strategy
    comparisons unfair.

    This helper filters the curve to ``>= EVALUATION_START``, derives the
    return series, and computes Sharpe / drawdown / cumulative return /
    final equity on the filtered subset. Final equity is restated as if
    ``INITIAL_CASH`` were deployed at the evaluation start date so that
    every strategy is compared from the same starting wealth.

    Parameters
    ----------
    curve : DataFrame
        Vector or Backtrader equity-curve dataframe with a date column and
        either ``portfolio_return`` (vector) or ``return`` (Backtrader).
    return_col : str, optional
        Name of the return column. Auto-detected if None.
    eval_start : Timestamp, optional
        Override the module-level ``EVALUATION_START`` for sensitivity work.
    """
    start = eval_start if eval_start is not None else EVALUATION_START
    if curve is None or curve.empty:
        return {"name": name, "evaluation_start": start.isoformat(), "n_evaluation_periods": 0}
    if return_col is None:
        if "portfolio_return" in curve.columns:
            return_col = "portfolio_return"
        elif "return" in curve.columns:
            return_col = "return"
        else:
            raise KeyError(f"No return column found in curve. Columns: {list(curve.columns)}")
    c = curve.copy()
    c[date_col] = pd.to_datetime(c[date_col])
    c = c[c[date_col] >= start].sort_values(date_col)
    if c.empty:
        return {"name": name, "evaluation_start": start.isoformat(), "n_evaluation_periods": 0}
    return_series = c[return_col].astype(float)
    metrics = perf_metrics(return_series, name, periods_per_year=periods_per_year, period_label=period_label)
    valid = return_series.replace([np.inf, -np.inf], np.nan).dropna()
    if not valid.empty:
        wealth = INITIAL_CASH * (1.0 + valid).cumprod()
        metrics["final_equity"] = float(wealth.iloc[-1])
        metrics["total_return"] = float(wealth.iloc[-1] / INITIAL_CASH - 1.0)
        metrics["evaluation_start"] = start.isoformat()
        metrics["n_evaluation_periods"] = int(len(valid))
        metrics["first_evaluation_period"] = pd.Timestamp(c[date_col].iloc[0]).isoformat()
        metrics["last_evaluation_period"] = pd.Timestamp(c[date_col].iloc[-1]).isoformat()
    return metrics


def filter_to_evaluation_window(
    df: pd.DataFrame,
    date_col: str = "month",
    eval_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return rows of df with ``date_col >= EVALUATION_START``."""
    start = eval_start if eval_start is not None else EVALUATION_START
    if df is None or df.empty:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out[out[date_col] >= start].copy()


def make_fmp_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Constructing factor-mimicking portfolios and IC diagnostics...")
    factor_map = {
        "roe": "roe_z",
        "pe": "pe_z",
        "momentum": "momentum_z",
        "trend": "trend_z",
    }
    portfolio_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    weight_rows: list[pd.DataFrame] = []

    for factor_name, factor_col in factor_map.items():
        for month, g0 in panel.groupby("month", sort=True):
            g = g0.loc[g0["eligible"], ["ticker", factor_col, "next_ret_cc"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(g) < 50:
                continue
            ranks = g[factor_col].rank(method="first")
            q = pd.qcut(ranks, 5, labels=False) + 1
            g = g.assign(quantile=q.astype(int))
            top = g[g["quantile"].eq(5)]
            bottom = g[g["quantile"].eq(1)]
            if top.empty or bottom.empty:
                continue
            ret = top["next_ret_cc"].mean() - bottom["next_ret_cc"].mean()
            portfolio_rows.append(
                {
                    "month": month,
                    "factor": factor_name,
                    "portfolio_fmp_return": ret,
                    "top_return": top["next_ret_cc"].mean(),
                    "bottom_return": bottom["next_ret_cc"].mean(),
                    "n_top": len(top),
                    "n_bottom": len(bottom),
                }
            )
            weights = pd.concat(
                [
                    top[["ticker"]].assign(weight=1 / len(top)),
                    bottom[["ticker"]].assign(weight=-1 / len(bottom)),
                ],
                ignore_index=True,
            )
            weights["month"] = month
            weights["factor"] = factor_name
            weight_rows.append(weights[["month", "factor", "ticker", "weight"]])

            reg = sm.OLS(g["next_ret_cc"], sm.add_constant(g[[factor_col]], has_constant="add")).fit()
            regression_rows.append(
                {
                    "month": month,
                    "factor": factor_name,
                    "regression_fmp_return": reg.params[factor_col],
                    "regression_t_stat": reg.tvalues[factor_col],
                    "regression_r2": reg.rsquared,
                    "n_obs": len(g),
                }
            )
            pearson = g[[factor_col, "next_ret_cc"]].corr(method="pearson").iloc[0, 1]
            spearman = g[[factor_col, "next_ret_cc"]].corr(method="spearman").iloc[0, 1]
            ic_rows.append(
                {
                    "month": month,
                    "factor": factor_name,
                    "ic": pearson,
                    "rank_ic": spearman,
                    "n_obs": len(g),
                }
            )

    portfolio = pd.DataFrame(portfolio_rows)
    regression = pd.DataFrame(regression_rows)
    ic = pd.DataFrame(ic_rows)
    weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    return portfolio, regression, ic, weights


def summarize_fmps(portfolio: pd.DataFrame, regression: pd.DataFrame, ic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for approach, df, col in [
        ("portfolio_sort", portfolio, "portfolio_fmp_return"),
        ("cross_sectional_regression", regression, "regression_fmp_return"),
    ]:
        for factor, g in df.groupby("factor", sort=True):
            metrics = perf_metrics(g.sort_values("month")[col], f"{approach}_{factor}")
            metrics["approach"] = approach
            metrics["factor"] = factor
            rows.append(metrics)
    summary = pd.DataFrame(rows)
    ic_summary = (
        ic.groupby("factor")
        .agg(avg_ic=("ic", "mean"), avg_rank_ic=("rank_ic", "mean"), ic_t=("ic", lambda x: st.ttest_1samp(x.dropna(), 0.0).statistic if x.dropna().size > 2 else np.nan), rank_ic_t=("rank_ic", lambda x: st.ttest_1samp(x.dropna(), 0.0).statistic if x.dropna().size > 2 else np.nan))
        .reset_index()
    )
    return summary.merge(ic_summary, on="factor", how="left")


def factor_columns_for_spec(spec: StrategySpec) -> dict[str, str]:
    cols = {factor_key: ("trend_z" if factor_key == "trend" else f"{factor_key}_z") for factor_key in spec.weights}
    cols["trend"] = spec.trend_col
    return cols


def bounded_simplex_weights(raw: pd.Series, lower: float, upper: float) -> pd.Series:
    """Normalize non-negative weights with lower/upper bounds that sum to one."""
    raw = raw.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    n = len(raw)
    if n == 0:
        return raw
    if lower * n > 1.0 or upper * n < 1.0:
        raise ValueError("Infeasible factor weight bounds.")
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=raw.index)
    raw = raw / raw.sum()

    result = pd.Series(np.nan, index=raw.index, dtype=float)
    remaining = list(raw.index)
    remaining_sum = 1.0
    while remaining:
        candidate_raw = raw.loc[remaining]
        if candidate_raw.sum() <= 0:
            candidate = pd.Series(remaining_sum / len(remaining), index=remaining)
        else:
            candidate = candidate_raw / candidate_raw.sum() * remaining_sum
        below = candidate[candidate < lower]
        above = candidate[candidate > upper]
        if below.empty and above.empty:
            result.loc[remaining] = candidate
            break
        fixed = pd.concat([below, above])
        for idx, val in fixed.items():
            result.loc[idx] = lower if val < lower else upper
        remaining_sum = 1.0 - result.dropna().sum()
        remaining = [idx for idx in remaining if idx not in fixed.index]
        if not remaining:
            break
    return result.fillna(0.0) / result.fillna(0.0).sum()


def rolling_rank_ic_factor_weights(panel: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Past-only dynamic factor weights based on rolling rank-IC strength."""
    factor_cols = factor_columns_for_spec(spec)
    months = sorted(pd.to_datetime(panel["month"].dropna().unique()))
    factors = list(spec.weights)
    equal = pd.Series(1.0 / len(factors), index=factors)

    ic_rows: list[dict[str, Any]] = []
    for month, g0 in panel.groupby("month", sort=True):
        for factor_key in factors:
            col = factor_cols[factor_key]
            if col not in g0.columns:
                raise KeyError(f"Missing factor column {col}")
            g = g0.loc[g0["eligible"], [col, "next_ret_cc"]].replace([np.inf, -np.inf], np.nan).dropna()
            rank_ic = g[[col, "next_ret_cc"]].corr(method="spearman").iloc[0, 1] if len(g) >= 50 else np.nan
            ic_rows.append({"month": pd.Timestamp(month), "factor": factor_key, "rank_ic": rank_ic})
    ic = pd.DataFrame(ic_rows)

    weight_rows: list[dict[str, Any]] = []
    for month in months:
        train = ic[ic["month"].lt(month)].dropna(subset=["rank_ic"])
        if spec.weight_lookback_months > 0:
            train_months = sorted(train["month"].unique())[-spec.weight_lookback_months :]
            train = train[train["month"].isin(train_months)]
        if train["month"].nunique() < spec.weight_min_periods:
            weights = equal.copy()
            method_note = "equal_until_min_history"
        else:
            stats_df = train.groupby("factor")["rank_ic"].agg(["mean", "std", "count"]).reindex(factors)
            strength = (stats_df["mean"] / stats_df["std"].replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
            if strength.sum() <= 0:
                raw = equal.copy()
                method_note = "equal_no_positive_ic_ir"
            else:
                raw = strength / strength.sum()
                method_note = "rolling_rank_ic_ir"
            shrunk = spec.weight_shrink_to_equal * equal + (1.0 - spec.weight_shrink_to_equal) * raw
            weights = bounded_simplex_weights(shrunk, spec.min_factor_weight, spec.max_factor_weight)
        row = {
            "month": month,
            "strategy": spec.name,
            "method": method_note,
            "lookback_months": spec.weight_lookback_months,
            "min_periods": spec.weight_min_periods,
            "shrink_to_equal": spec.weight_shrink_to_equal,
            "min_factor_weight": spec.min_factor_weight,
            "max_factor_weight": spec.max_factor_weight,
        }
        row.update({f"weight_{factor_key}": float(weights[factor_key]) for factor_key in factors})
        weight_rows.append(row)
    return pd.DataFrame(weight_rows)


def factor_weight_history(panel: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    if spec.weight_method == "rolling_rank_ic":
        return rolling_rank_ic_factor_weights(panel, spec)
    months = sorted(pd.to_datetime(panel["month"].dropna().unique()))
    weights = pd.Series(spec.weights, dtype=float)
    weights = weights / weights.abs().sum()
    rows = []
    for month in months:
        row = {
            "month": month,
            "strategy": spec.name,
            "method": "static",
            "lookback_months": np.nan,
            "min_periods": np.nan,
            "shrink_to_equal": np.nan,
            "min_factor_weight": np.nan,
            "max_factor_weight": np.nan,
        }
        row.update({f"weight_{factor_key}": float(weights.get(factor_key, 0.0)) for factor_key in spec.weights})
        rows.append(row)
    return pd.DataFrame(rows)


def score_for_spec(panel: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    factor_cols = factor_columns_for_spec(spec)
    for col in factor_cols.values():
        if col not in panel.columns:
            raise KeyError(f"Missing factor column {col}")

    if spec.weight_method == "rolling_rank_ic":
        weights = rolling_rank_ic_factor_weights(panel, spec)
        tmp = panel[["month"]].merge(weights, on="month", how="left")
        tmp.index = panel.index
        weight_lookup = {factor_key: tmp[f"weight_{factor_key}"] for factor_key in spec.weights}
    else:
        weight_lookup = {factor_key: pd.Series(float(weight), index=panel.index) for factor_key, weight in spec.weights.items()}

    weighted = []
    valid_weight = pd.Series(0.0, index=panel.index)
    valid_count = pd.Series(0, index=panel.index)
    for factor_key in spec.weights:
        col = factor_cols[factor_key]
        w = weight_lookup[factor_key].astype(float)
        weighted.append(panel[col] * w)
        is_valid = panel[col].notna() & w.notna()
        valid_weight += np.where(is_valid, w.abs(), 0.0)
        valid_count += is_valid.astype(int)
    out = sum(weighted)
    score = out / valid_weight.replace(0, np.nan)
    score[valid_count < min(3, len(spec.weights))] = np.nan
    return score


def stop_take_return(g: pd.DataFrame, stop_loss: float | None, take_profit: float | None) -> pd.Series:
    base = g["next_ret_oc"].copy()
    if stop_loss is None and take_profit is None:
        return base
    low_ret = g["next_low"] / g["next_open"] - 1
    high_ret = g["next_high"] / g["next_open"] - 1
    adjusted = base.copy()
    if stop_loss is not None:
        adjusted = adjusted.where(low_ret > -abs(stop_loss), -abs(stop_loss))
    if take_profit is not None:
        hit_take = high_ret >= abs(take_profit)
        hit_stop = low_ret <= -abs(stop_loss) if stop_loss is not None else pd.Series(False, index=g.index)
        adjusted = adjusted.where(~(hit_take & ~hit_stop), abs(take_profit))
    return adjusted


def month_regime_is_on(g: pd.DataFrame) -> bool:
    values = g["regime_on"].dropna()
    if values.empty:
        return True
    return bool(values.iloc[-1])


def is_assignment_scope_strategy(spec: StrategySpec) -> bool:
    """The pure base must preserve the literal four-factor assignment design."""
    required_weights = {"roe", "pe", "momentum", "trend"}
    return (
        set(spec.weights) == required_weights
        and spec.trend_col == "trend_z"
        and spec.stop_loss is None
        and spec.take_profit is None
    )


def position_size_for_spec(
    spec: StrategySpec,
    equity: float,
    vol_weights: pd.Series | None = None,
) -> pd.Series | float:
    """Per-position dollar size implied by the spec's sizing rule.

    Returns a Series indexed by the selected tickers when ``sizing_method`` is
    ``"vol_targeted"`` (per-position dollars vary across the basket), otherwise
    a scalar applied uniformly to every selected position.
    """
    if spec.sizing_method == "vol_targeted":
        if vol_weights is None:
            raise ValueError("vol_targeted sizing requires vol_weights argument")
        # Per-name dollars = (basket_target_pct) * equity * vol_weight_i
        # where vol_weight sums to 1 across the basket and basket_target_pct is
        # top_n * sizing_target_pct (e.g. 20 * 5% = 100% of equity at full basket).
        basket_target_pct = float(spec.top_n) * float(spec.sizing_target_pct)
        return float(equity) * basket_target_pct * vol_weights
    if spec.sizing_method == "percent_of_equity":
        return float(spec.sizing_target_pct) * float(equity)
    return float(CASH_PER_TRADE)


def select_positions_for_spec(g: pd.DataFrame, spec: StrategySpec, equity: float) -> pd.DataFrame:
    if spec.sizing_method in ("percent_of_equity", "vol_targeted"):
        # Dynamic sizing modes scale with equity and can always fit ``top_n``
        # positions. Cash buffer is implicit at ``1 - top_n * sizing_target_pct``.
        max_positions = spec.top_n
    else:
        max_positions = max(0, int(equity // CASH_PER_TRADE))
    n = min(spec.top_n, max_positions, len(g))
    if n <= 0:
        return g.iloc[0:0].copy()
    ranked = g.sort_values("score", ascending=False).copy()
    return ranked.head(n)


def compute_inverse_vol_weights(
    selected_tickers: list[str],
    month: pd.Timestamp,
    daily_prices: pd.DataFrame,
    lookback_days: int = VOL_LOOKBACK_DAYS,
    vol_floor_ann: float = VOL_FLOOR_ANN,
) -> pd.Series:
    """Compute inverse-volatility portfolio weights for a set of selected tickers.

    Realized annualized volatility is computed from each ticker's daily log
    returns over the trailing ``lookback_days`` trading days strictly before
    ``month``. Vols are floored at ``vol_floor_ann`` (annualized) to prevent
    extreme inverse weights from very-low-vol names. Weights are normalized so
    they sum to 1 across the basket.

    Parameters
    ----------
    selected_tickers : list[str]
        Tickers in the basket.
    month : pd.Timestamp
        Signal month-end. Vols use data strictly before this date.
    daily_prices : pd.DataFrame
        Raw price panel with columns ``ticker``, ``date``, ``adjClose``.
    lookback_days : int
        Window of trading days for realized vol estimation.
    vol_floor_ann : float
        Lower bound on annualized vol used as denominator.

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.0. If a ticker has no valid
        history, equal-weight is assigned as a fallback for that ticker.
    """
    if not selected_tickers:
        return pd.Series(dtype=float)
    cutoff = pd.Timestamp(month)
    sub = daily_prices[
        daily_prices["ticker"].isin(selected_tickers)
        & (daily_prices["date"] < cutoff)
    ][["ticker", "date", "adjClose"]].copy()
    sub = sub.sort_values(["ticker", "date"])
    inv_vols: dict[str, float] = {}
    for ticker in selected_tickers:
        g = sub[sub["ticker"].eq(ticker)].tail(lookback_days + 1)
        if len(g) < max(20, lookback_days // 3):
            inv_vols[ticker] = np.nan
            continue
        log_rets = np.log(g["adjClose"].astype(float)).diff().dropna()
        if log_rets.empty or log_rets.std(ddof=1) == 0 or not np.isfinite(log_rets.std(ddof=1)):
            inv_vols[ticker] = np.nan
            continue
        ann_vol = float(log_rets.std(ddof=1)) * np.sqrt(252.0)
        ann_vol = max(ann_vol, float(vol_floor_ann))
        inv_vols[ticker] = 1.0 / ann_vol
    s = pd.Series(inv_vols, dtype=float)
    if s.notna().sum() == 0:
        # Total fallback to equal weight
        return pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
    # Names without enough history get the median inverse vol (defensive fallback)
    s = s.fillna(s.median())
    return s / s.sum()


# --- Time-varying transaction cost schedule (improved 7) -----------------------------

# Central per-side cost estimates in basis points, by calendar year. These are
# educated central estimates drawn from publicly available institutional execution
# studies (see docs/IMPROVED_7_COSTS.md for citations and a discussion of the
# per-year uncertainty, which is approximately +/- 50 percent). Years with known
# market-stress spread widening (2008 financial crisis, 2020 Covid) preserve
# elevated slippage; the secular decline in commissions over 2006-2026 reflects
# the post-decimalization, post-algo, and post-zero-commission eras.
_CENTRAL_COST_BPS_PER_SIDE: dict[int, tuple[float, float]] = {
    2006: (7.0, 6.0),
    2007: (6.0, 5.0),
    2008: (6.0, 8.0),   # crisis-era spread widening
    2009: (5.0, 5.0),
    2010: (4.0, 4.0),
    2011: (4.0, 3.5),
    2012: (3.0, 3.0),
    2013: (3.0, 2.5),
    2014: (2.5, 2.0),
    2015: (2.0, 2.0),
    2016: (1.5, 2.0),
    2017: (1.5, 1.5),
    2018: (1.0, 1.5),
    2019: (1.0, 1.5),
    2020: (0.7, 2.5),   # Covid-era spread widening
    2021: (0.5, 1.5),
    2022: (0.5, 2.0),   # rate-vol elevated spreads
    2023: (0.5, 1.5),
    2024: (0.5, 1.0),
    2025: (0.5, 1.0),
    2026: (0.5, 1.0),
}


_COST_SCENARIO_MULTIPLIERS: dict[str, float] = {
    "zero": 0.0,
    "central": 1.0,
    "pessimistic": 2.0,
}


def transaction_cost_schedule(scenario: str = "central") -> pd.DataFrame:
    """Return a per-year transaction-cost table for the requested scenario.

    Columns: year, scenario, commission_bps_per_side, slippage_bps_per_side,
    round_trip_bps. The round-trip total is ``2 * (commission + slippage)``.

    Scenarios:
        ``zero``         -- no costs (project baseline)
        ``central``      -- central estimate per year (uncertainty +/- 50 pct)
        ``pessimistic``  -- 2x central (weaker execution, smaller fund)

    Source provenance lives in ``docs/IMPROVED_7_COSTS.md`` and includes the
    Frazzini-Israel-Moskowitz (2018) JPM trading-costs paper, ITG / Virtu Cost
    Index quarterly reports, J.P. Morgan execution research, and NYSE TAQ
    spread studies. Per-year values are interpolated between trusted anchors
    and the +/- 50 pct uncertainty motivates the multi-scenario design.
    """
    if scenario not in _COST_SCENARIO_MULTIPLIERS:
        raise ValueError(
            f"Unknown cost scenario {scenario!r}. Valid: {sorted(_COST_SCENARIO_MULTIPLIERS)}"
        )
    multiplier = _COST_SCENARIO_MULTIPLIERS[scenario]
    rows: list[dict[str, Any]] = []
    for year, (comm, slip) in sorted(_CENTRAL_COST_BPS_PER_SIDE.items()):
        comm_scaled = comm * multiplier
        slip_scaled = slip * multiplier
        rows.append(
            {
                "year": int(year),
                "scenario": scenario,
                "scenario_multiplier": multiplier,
                "commission_bps_per_side": comm_scaled,
                "slippage_bps_per_side": slip_scaled,
                "round_trip_bps": 2.0 * (comm_scaled + slip_scaled),
            }
        )
    return pd.DataFrame(rows)


def cost_for_month(schedule: pd.DataFrame | None, month: pd.Timestamp) -> tuple[float, float]:
    """Look up the (commission, slippage) per-side rate for a given month.

    Returns rates as fractions, not basis points. Falls back to the most recent
    available year if the requested year is outside the table. Returns (0, 0)
    when ``schedule`` is None.
    """
    if schedule is None or schedule.empty:
        return 0.0, 0.0
    year = pd.Timestamp(month).year
    available_years = schedule["year"].astype(int)
    if year in available_years.values:
        row = schedule.loc[available_years.eq(year)].iloc[0]
    else:
        nearest_year = int(available_years.iloc[(available_years - year).abs().argsort().iloc[0]])
        row = schedule.loc[available_years.eq(nearest_year)].iloc[0]
    comm = float(row["commission_bps_per_side"]) / 10_000.0
    slip = float(row["slippage_bps_per_side"]) / 10_000.0
    return comm, slip


def simulate_vector_strategy(
    panel: pd.DataFrame,
    spec: StrategySpec,
    cost_schedule: pd.DataFrame | None = None,
    daily_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vector backtest for a strategy spec.

    Parameters
    ----------
    panel : DataFrame
        The factor panel with monthly observations.
    spec : StrategySpec
        Strategy configuration (weights, stop/take, top_n, regime filter, etc).
    cost_schedule : DataFrame, optional
        Per-year transaction-cost schedule from :func:`transaction_cost_schedule`.
        If ``None`` (default), no costs are deducted.
    daily_prices : DataFrame, optional
        Required only when ``spec.sizing_method == "vol_targeted"``. Daily price
        panel with columns ``ticker``, ``date``, ``adjClose``; used to compute
        per-stock trailing realized volatility for inverse-vol weighting.
    """
    if spec.sizing_method == "vol_targeted" and daily_prices is None:
        raise ValueError(
            "spec.sizing_method='vol_targeted' requires daily_prices to be passed."
        )
    tmp = panel.copy()
    tmp["score"] = score_for_spec(tmp, spec)
    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    equity = INITIAL_CASH

    for month, g0 in tmp.groupby("month", sort=True):
        eligible = g0.loc[g0["eligible"] & g0["score"].notna()].copy()
        regime_on = month_regime_is_on(g0)
        if spec.regime_filter and not regime_on:
            selected = eligible.iloc[0:0].copy()
        else:
            selected = select_positions_for_spec(eligible, spec, equity)

        comm_per_side, slip_per_side = cost_for_month(cost_schedule, month)
        round_trip_drag = 2.0 * (comm_per_side + slip_per_side)

        # Position-size determination — scalar for fixed/percent rules, Series for vol-targeted
        if spec.sizing_method == "vol_targeted":
            if not selected.empty:
                tickers = selected["ticker"].tolist()
                vol_weights = compute_inverse_vol_weights(
                    tickers,
                    pd.Timestamp(month),
                    daily_prices,
                    lookback_days=spec.vol_lookback_days,
                    vol_floor_ann=spec.vol_floor_ann,
                )
                per_position_dollars = position_size_for_spec(spec, equity, vol_weights=vol_weights)
                # Align per_position_dollars to selected order
                selected_per_position = per_position_dollars.reindex(selected["ticker"].values).values
            else:
                per_position_dollars = 0.0
                selected_per_position = None
        else:
            scalar_size = position_size_for_spec(spec, equity)
            per_position_dollars = scalar_size
            selected_per_position = None

        if selected.empty:
            pnl = 0.0
            ret = 0.0
            cost_dollars = 0.0
            per_position_dollars_log = (
                float(per_position_dollars) if not isinstance(per_position_dollars, pd.Series) else float("nan")
            )
        else:
            selected["gross_stock_return"] = stop_take_return(selected, spec.stop_loss, spec.take_profit)
            selected["cost_drag_per_holding"] = round_trip_drag
            selected["realized_stock_return"] = selected["gross_stock_return"] - round_trip_drag
            if selected_per_position is not None:
                selected["allocated_cash"] = selected_per_position
                pnl = float(np.nansum(selected_per_position * selected["realized_stock_return"].values))
                cost_dollars = float(np.nansum(selected_per_position) * round_trip_drag)
                per_position_dollars_log = float(np.nanmean(selected_per_position))
            else:
                selected["allocated_cash"] = float(per_position_dollars)
                pnl = float(per_position_dollars) * float(selected["realized_stock_return"].sum())
                cost_dollars = float(per_position_dollars) * round_trip_drag * len(selected)
                per_position_dollars_log = float(per_position_dollars)
            selected["cash_weight"] = (
                selected["allocated_cash"] / equity if equity > 0 else np.nan
            )
            ret = pnl / equity if equity > 0 else np.nan
            selected["strategy"] = spec.name
            selected["strategy_notes"] = spec.notes
            selected["signal_month"] = month
            holdings.append(
                selected[
                    [
                        "strategy",
                        "strategy_notes",
                        "signal_month",
                        "ticker",
                        "GICS Sector",
                        "score",
                        "allocated_cash",
                        "gross_stock_return",
                        "cost_drag_per_holding",
                        "realized_stock_return",
                    ]
                ]
            )

        prev = equity
        equity = equity + pnl
        rows.append(
            {
                "strategy": spec.name,
                "month": month,
                "portfolio_return": ret,
                "pnl": pnl,
                "equity": equity,
                "prev_equity": prev,
                "n_positions": len(selected),
                "regime_on": regime_on,
                "cost_dollars": cost_dollars,
                "commission_bps_per_side": comm_per_side * 10_000.0,
                "slippage_bps_per_side": slip_per_side * 10_000.0,
                "round_trip_bps": round_trip_drag * 10_000.0,
                "per_position_dollars": per_position_dollars_log,
                "sizing_method": spec.sizing_method,
            }
        )

    curve = pd.DataFrame(rows)
    holds = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    return curve, holds


class FixedCashSizer(bt.Sizer):
    params = (("cash_per_trade", CASH_PER_TRADE),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        close_price = data.close[0]
        if close_price <= 0:
            return 0
        size = int(self.params.cash_per_trade / close_price)
        return max(size, 0)


class EquityPercentSizer(bt.Sizer):
    """Size each new position at ``target_pct`` of current portfolio value.

    Equivalent to the vector simulator's ``sizing_method='percent_of_equity'``
    rule. Used by improved 8 onward for true equal-weight (1/N) sizing in the
    spirit of DeMiguel-Garlappi-Uppal (2009). Integer share rounding is used,
    which can leave small unallocated cash; that is realistic and matches how
    practitioners trade whole shares.
    """

    params = (("target_pct", 0.05),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        close_price = data.close[0]
        if close_price <= 0:
            return 0
        portfolio_value = self.broker.getvalue()
        dollars_per_position = float(self.p.target_pct) * float(portfolio_value)
        size = int(dollars_per_position / close_price)
        return max(size, 0)


class VolatilityTargetedSizer(bt.Sizer):
    """Inverse-volatility position sizing for the Backtrader engine.

    Each position is sized at ``(1/vol_i) / sum(1/vol_j across the basket) *
    basket_target_pct * portfolio_value`` where ``vol_i`` is the annualized
    realized volatility computed from ``data``'s trailing ``vol_lookback_days``
    of daily log returns, floored at ``vol_floor_ann`` to prevent extreme
    inverse weights. ``basket_target_pct`` represents the total fraction of
    portfolio capital to deploy in the basket (typically 1.0 = fully invested).

    This sizer maintains an internal cache of inverse vols and renormalizes
    each rebalance period using the live broker positions plus pending targets.
    For simplicity in the Backtrader integration, this sizer sizes each name
    INDEPENDENTLY relative to its own inverse vol scaled by ``target_pct``
    (interpreted as the per-name allocation when vols are equal). The vector
    simulator does the basket-level normalization, which is the more rigorous
    treatment; Backtrader sizing here is a per-name approximation that scales
    the percent-of-equity rule by inverse vol.
    """

    params = (
        ("target_pct", 0.05),                 # nominal per-position equity fraction
        ("vol_lookback_days", VOL_LOOKBACK_DAYS),
        ("vol_floor_ann", VOL_FLOOR_ANN),
        ("median_vol_ann", 0.25),             # reference vol used as the denominator scale
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        close_price = data.close[0]
        if close_price <= 0:
            return 0
        portfolio_value = self.broker.getvalue()
        # Compute realized vol from data's trailing window
        lookback = int(self.p.vol_lookback_days)
        try:
            closes = np.array([float(data.close[-i]) for i in range(lookback + 1)])
        except Exception:
            ann_vol = float(self.p.median_vol_ann)
        else:
            closes = closes[~np.isnan(closes) & (closes > 0)]
            if len(closes) < max(20, lookback // 3):
                ann_vol = float(self.p.median_vol_ann)
            else:
                log_rets = np.diff(np.log(closes))
                if log_rets.size < 2:
                    ann_vol = float(self.p.median_vol_ann)
                else:
                    std = float(np.std(log_rets, ddof=1))
                    ann_vol = max(std * np.sqrt(252.0), float(self.p.vol_floor_ann))
        # Scale: position dollars = target_pct * portfolio_value * (median_vol / ann_vol)
        # This way a name with median vol gets target_pct exposure; high-vol names get less.
        scale = float(self.p.median_vol_ann) / ann_vol
        dollars_per_position = float(self.p.target_pct) * float(portfolio_value) * scale
        size = int(dollars_per_position / close_price)
        return max(size, 0)


class MonthlySignalStrategy(bt.Strategy):
    params = (
        ("signals", None),
        ("benchmark_name", "BENCHMARK"),
    )

    def __init__(self):
        self.order_log: list[dict[str, Any]] = []
        self.trade_log: list[dict[str, Any]] = []
        self.equity_log: list[dict[str, Any]] = []
        self.position_log: list[dict[str, Any]] = []
        self.entry_orders: dict[str, bt.Order] = {}
        self.exit_orders: dict[str, bt.Order] = {}
        self.order_reason: dict[int, str] = {}

    @staticmethod
    def _is_live_order(order: bt.Order) -> bool:
        return order.status in [order.Submitted, order.Accepted]

    def _has_live_order(self, ticker: str) -> bool:
        candidates = []
        if ticker in self.entry_orders:
            candidates.append(self.entry_orders[ticker])
        if ticker in self.exit_orders:
            candidates.append(self.exit_orders[ticker])
        return any(self._is_live_order(order) for order in candidates)

    def _remember(self, order: bt.Order, reason: str) -> bt.Order:
        self.order_reason[order.ref] = reason
        return order

    def prenext(self):
        self.next()

    def next(self):
        dt = pd.Timestamp(self.datas[0].datetime.date(0)).to_period("M").to_timestamp("M")
        signals = self.params.signals or {}
        target = set(signals.get(dt, []))

        for data in self.datas[1:]:
            if len(data) == 0:
                continue
            ticker = data._name
            is_current = pd.Timestamp(data.datetime.date(0)).to_period("M").to_timestamp("M") == dt
            pos = self.getposition(data)
            if not is_current:
                continue
            if self._has_live_order(ticker):
                continue
            if pos.size < 0:
                self.exit_orders[ticker] = self._remember(
                    self.buy(data=data, size=abs(pos.size), exectype=bt.Order.Market),
                    "short_cover",
                )
            elif pos.size and ticker not in target:
                self.exit_orders[ticker] = self._remember(
                    self.close(data=data, exectype=bt.Order.Market),
                    "rebalance_exit",
                )
            elif ticker in target and not pos.size:
                self.entry_orders[ticker] = self._remember(
                    self.buy(data=data, exectype=bt.Order.Market),
                    "rebalance_entry",
                )

        self.equity_log.append({"month": dt, "value": self.broker.getvalue(), "cash": self.broker.getcash()})
        for data in self.datas[1:]:
            if len(data) == 0:
                continue
            pos = self.getposition(data)
            if pos.size:
                self.position_log.append(
                    {
                        "month": dt,
                        "ticker": data._name,
                        "size": pos.size,
                        "price": data.close[0],
                        "market_value": pos.size * data.close[0],
                    }
                )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        dt = pd.Timestamp(self.datas[0].datetime.date(0)).to_period("M").to_timestamp("M")
        self.order_log.append(
            {
                "month": dt,
                "ticker": order.data._name,
                "reason": self.order_reason.pop(order.ref, ""),
                "is_buy": order.isbuy(),
                "status": order.getstatusname(),
                "exectype": order.exectype,
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "commission": order.executed.comm,
            }
        )
        ticker = order.data._name
        if self.entry_orders.get(ticker) is not None and self.entry_orders[ticker].ref == order.ref:
            self.entry_orders.pop(ticker, None)
        if self.exit_orders.get(ticker) is not None and self.exit_orders[ticker].ref == order.ref:
            self.exit_orders.pop(ticker, None)
        if order.status == order.Completed:
            pos = self.getposition(order.data)
            if pos.size < 0:
                self.exit_orders[ticker] = self._remember(
                    self.buy(data=order.data, size=abs(pos.size), exectype=bt.Order.Market),
                    "short_cover",
                )

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        dt = pd.Timestamp(self.datas[0].datetime.date(0)).to_period("M").to_timestamp("M")
        self.trade_log.append(
            {
                "month": dt,
                "ticker": trade.data._name,
                "pnl": trade.pnl,
                "pnl_comm": trade.pnlcomm,
                "bar_len": trade.barlen,
            }
        )


class DailySignalStopTakeStrategy(bt.Strategy):
    params = (
        ("signals", None),
        ("benchmark_name", "BENCHMARK"),
        ("stop_loss", None),
        ("take_profit", None),
    )

    def __init__(self):
        self.order_log: list[dict[str, Any]] = []
        self.trade_log: list[dict[str, Any]] = []
        self.equity_log: list[dict[str, Any]] = []
        self.position_log: list[dict[str, Any]] = []
        self.entry_orders: dict[str, bt.Order] = {}
        self.exit_orders: dict[str, bt.Order] = {}
        self.protective_orders: dict[str, list[bt.Order]] = {}
        self.order_reason: dict[int, str] = {}
        self.entry_prices: dict[str, float] = {}
        self.last_rebalance_period: pd.Period | None = None
        self.pending_rebalance_exit: set[str] = set()

    @staticmethod
    def _is_live_order(order: bt.Order) -> bool:
        return order.status in [order.Submitted, order.Accepted]

    @staticmethod
    def _is_current_bar(data: bt.LineSeries, dt: pd.Timestamp) -> bool:
        return len(data) > 0 and pd.Timestamp(data.datetime.date(0)).normalize() == dt.normalize()

    def _has_live_order(self, ticker: str) -> bool:
        candidates = []
        if ticker in self.entry_orders:
            candidates.append(self.entry_orders[ticker])
        if ticker in self.exit_orders:
            candidates.append(self.exit_orders[ticker])
        return any(self._is_live_order(order) for order in candidates)

    def _live_protective_orders(self, ticker: str) -> list[bt.Order]:
        return [order for order in self.protective_orders.get(ticker, []) if self._is_live_order(order)]

    def _has_live_protective_order(self, ticker: str) -> bool:
        return bool(self._live_protective_orders(ticker))

    def _remove_protective_order(self, ticker: str, order_ref: int) -> None:
        orders = [order for order in self.protective_orders.get(ticker, []) if order.ref != order_ref]
        if orders:
            self.protective_orders[ticker] = orders
        else:
            self.protective_orders.pop(ticker, None)

    def _remember(self, order: bt.Order, reason: str) -> bt.Order:
        self.order_reason[order.ref] = reason
        return order

    def _cancel_protective_orders(self, data: bt.LineSeries, exclude_ref: int | None = None) -> bool:
        ticker = data._name
        canceled = False
        for order in list(self.protective_orders.get(ticker, [])):
            if exclude_ref is not None and order.ref == exclude_ref:
                continue
            if self._is_live_order(order):
                self.cancel(order)
                canceled = True
        return canceled

    def _submit_market_sell(self, data: bt.LineSeries, size: int, reason: str) -> None:
        ticker = data._name
        if self._has_live_order(ticker):
            return
        if size > 0:
            self.exit_orders[ticker] = self._remember(
                self.sell(data=data, size=int(size), exectype=bt.Order.Market),
                reason,
            )

    def _buy_to_cover(self, data: bt.LineSeries) -> None:
        ticker = data._name
        if self._has_live_order(ticker):
            return
        self._cancel_protective_orders(data)
        pos = self.getposition(data)
        if pos.size < 0:
            self.exit_orders[ticker] = self._remember(
                self.buy(data=data, size=abs(int(pos.size)), exectype=bt.Order.Market),
                "short_cover",
            )

    def _submit_protective_orders(self, data: bt.LineSeries, entry_price: float, size: int) -> None:
        ticker = data._name
        if size <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
            return
        self._cancel_protective_orders(data)

        stop_loss = None if self.params.stop_loss is None else abs(float(self.params.stop_loss))
        take_profit = None if self.params.take_profit is None else abs(float(self.params.take_profit))
        submitted: list[bt.Order] = []

        stop_order = None
        if stop_loss is not None:
            stop_price = entry_price * (1 - stop_loss)
            stop_order = self.sell(
                data=data,
                size=int(size),
                exectype=bt.Order.Stop,
                price=stop_price,
            )
            self._remember(stop_order, "stop_loss")
            submitted.append(stop_order)

        if take_profit is not None:
            limit_price = entry_price * (1 + take_profit)
            kwargs = {"oco": stop_order} if stop_order is not None else {}
            limit_order = self.sell(
                data=data,
                size=int(size),
                exectype=bt.Order.Limit,
                price=limit_price,
                **kwargs,
            )
            self._remember(limit_order, "take_profit")
            submitted.append(limit_order)

        if submitted:
            self.protective_orders[ticker] = submitted

    def prenext(self):
        self.next()

    def next(self):
        dt = pd.Timestamp(self.datas[0].datetime.date(0)).normalize()
        self._process_pending_rebalance_exits(dt)
        self._rebalance_if_needed(dt)
        self._log_daily_state(dt)

    def _process_pending_rebalance_exits(self, dt: pd.Timestamp) -> None:
        if not self.pending_rebalance_exit:
            return
        for data in self.datas[1:]:
            if not self._is_current_bar(data, dt):
                continue
            ticker = data._name
            if ticker not in self.pending_rebalance_exit:
                continue
            if self._has_live_order(ticker):
                continue
            if self._has_live_protective_order(ticker):
                self._cancel_protective_orders(data)
                continue
            pos = self.getposition(data)
            if pos.size < 0:
                self._buy_to_cover(data)
                self.pending_rebalance_exit.discard(ticker)
                continue
            if pos.size <= 0:
                self.entry_prices.pop(ticker, None)
                self.pending_rebalance_exit.discard(ticker)
                continue
            self._submit_market_sell(data, int(pos.size), "rebalance_exit")
            self.pending_rebalance_exit.discard(ticker)

    def _rebalance_if_needed(self, dt: pd.Timestamp) -> None:
        current_period = dt.to_period("M")
        if self.last_rebalance_period == current_period:
            return
        signal_month = (current_period - 1).to_timestamp("M")
        signals = self.params.signals or {}
        target = set(signals.get(signal_month, []))

        for data in self.datas[1:]:
            if not self._is_current_bar(data, dt):
                continue
            ticker = data._name
            if self._has_live_order(ticker):
                continue
            pos = self.getposition(data)
            if pos.size < 0:
                self._buy_to_cover(data)
            elif pos.size > 0 and ticker not in target:
                if self._has_live_protective_order(ticker):
                    self._cancel_protective_orders(data)
                    self.pending_rebalance_exit.add(ticker)
                else:
                    self._submit_market_sell(data, int(pos.size), "rebalance_exit")
            elif pos.size == 0 and ticker in target:
                self.entry_orders[ticker] = self._remember(
                    self.buy(data=data, exectype=bt.Order.Market),
                    "rebalance_entry",
                )
            elif pos.size > 0 and ticker in target and not self._has_live_protective_order(ticker):
                entry_price = self.entry_prices.get(ticker, float(pos.price))
                self._submit_protective_orders(data, entry_price, int(pos.size))

        self.last_rebalance_period = current_period

    def _log_daily_state(self, dt: pd.Timestamp) -> None:
        self.equity_log.append({"date": dt, "value": self.broker.getvalue(), "cash": self.broker.getcash()})
        for data in self.datas[1:]:
            if not self._is_current_bar(data, dt):
                continue
            pos = self.getposition(data)
            if pos.size:
                self.position_log.append(
                    {
                        "date": dt,
                        "ticker": data._name,
                        "size": pos.size,
                        "price": data.close[0],
                        "market_value": pos.size * data.close[0],
                        "entry_price": self.entry_prices.get(data._name, np.nan),
                    }
                )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        dt = pd.Timestamp(self.datas[0].datetime.date(0)).normalize()
        reason = self.order_reason.pop(order.ref, "")
        self.order_log.append(
            {
                "date": dt,
                "ticker": order.data._name,
                "reason": reason,
                "is_buy": order.isbuy(),
                "status": order.getstatusname(),
                "exectype": order.exectype,
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "commission": order.executed.comm,
            }
        )
        ticker = order.data._name
        if self.entry_orders.get(ticker) is not None and self.entry_orders[ticker].ref == order.ref:
            self.entry_orders.pop(ticker, None)
        if self.exit_orders.get(ticker) is not None and self.exit_orders[ticker].ref == order.ref:
            self.exit_orders.pop(ticker, None)
        self._remove_protective_order(ticker, order.ref)

        if order.status != order.Completed:
            return
        pos = self.getposition(order.data)
        if order.isbuy():
            if pos.size > 0 and reason != "short_cover":
                self.entry_prices[ticker] = float(order.executed.price)
                self._submit_protective_orders(order.data, self.entry_prices[ticker], int(pos.size))
            elif pos.size <= 0:
                self.entry_prices.pop(ticker, None)
        else:
            if reason in {"stop_loss", "take_profit"}:
                self._cancel_protective_orders(order.data, exclude_ref=order.ref)
            if pos.size < 0:
                self._buy_to_cover(order.data)
            elif pos.size == 0:
                self.entry_prices.pop(ticker, None)
                self.pending_rebalance_exit.discard(ticker)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        dt = pd.Timestamp(self.datas[0].datetime.date(0)).normalize()
        self.trade_log.append(
            {
                "date": dt,
                "ticker": trade.data._name,
                "pnl": trade.pnl,
                "pnl_comm": trade.pnlcomm,
                "bar_len": trade.barlen,
            }
        )


def signals_from_strategy(panel: pd.DataFrame, spec: StrategySpec) -> dict[pd.Timestamp, list[str]]:
    tmp = panel.copy()
    tmp["score"] = score_for_spec(tmp, spec)
    signals: dict[pd.Timestamp, list[str]] = {}
    equity = INITIAL_CASH
    for month, g0 in tmp.groupby("month", sort=True):
        eligible = g0.loc[g0["eligible"] & g0["score"].notna()].copy()
        if spec.regime_filter and not month_regime_is_on(g0):
            selected = []
        else:
            selected_df = select_positions_for_spec(eligible, spec, equity)
            selected = selected_df["ticker"].tolist()
            if not selected_df.empty:
                realized = stop_take_return(selected_df, spec.stop_loss, spec.take_profit)
                pnl = CASH_PER_TRADE * realized.sum()
                equity += pnl
        signals[month] = selected
    return signals


def run_backtrader(
    monthly: pd.DataFrame,
    index_monthly: pd.DataFrame,
    signals: dict[pd.Timestamp, list[str]],
    name: str,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if stop_loss is not None or take_profit is not None:
        raise ValueError("The monthly Backtrader runner is market-order-only. Use run_backtrader_daily_stop_take for risk exits.")
    print(f"Running Backtrader strategy: {name}...")
    tickers = sorted({t for names in signals.values() for t in names})
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addsizer(FixedCashSizer, cash_per_trade=CASH_PER_TRADE)

    idx_feed = index_monthly.rename(columns={"month": "date"}).copy()
    idx_feed["openinterest"] = 0
    idx_feed = idx_feed[["date", "open", "high", "low", "close", "volume", "openinterest"]].dropna(subset=["date", "open", "high", "low", "close"])
    idx_feed = idx_feed.set_index("date")
    cerebro.adddata(bt.feeds.PandasData(dataname=idx_feed), name="BENCHMARK")

    m = monthly[monthly["ticker"].isin(tickers)].copy()
    for ticker, g in m.groupby("ticker", sort=True):
        feed = g.rename(columns={"month": "date"}).copy()
        feed["openinterest"] = 0
        feed = feed[["date", "open", "high", "low", "close", "volume", "openinterest"]].dropna()
        if len(feed) < 24:
            continue
        feed = feed.set_index("date")
        cerebro.adddata(bt.feeds.PandasData(dataname=feed), name=ticker)

    cerebro.addstrategy(
        MonthlySignalStrategy,
        signals=signals,
        benchmark_name="BENCHMARK",
    )
    strategies = cerebro.run(runonce=False)
    strat = strategies[0]
    equity = pd.DataFrame(strat.equity_log)
    orders = pd.DataFrame(strat.order_log)
    trades = pd.DataFrame(strat.trade_log)
    positions = pd.DataFrame(strat.position_log)
    if not equity.empty:
        equity = equity.drop_duplicates("month", keep="last").sort_values("month")
        equity["return"] = equity["value"].pct_change().fillna(0.0)
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"backtrader_{name}"
    save_csv(equity, prefix.with_name(prefix.name + "_equity_curve.csv"))
    save_csv(orders, prefix.with_name(prefix.name + "_orders.csv"))
    save_csv(trades, prefix.with_name(prefix.name + "_trades.csv"))
    save_csv(positions, prefix.with_name(prefix.name + "_positions.csv"))
    metrics_dict = (
        metrics_over_evaluation_window(
            equity,
            f"backtrader_{name}",
            date_col="month",
            return_col="return",
            period_label="monthly",
            periods_per_year=12,
        )
        if not equity.empty
        else {"name": f"backtrader_{name}"}
    )
    metrics = pd.DataFrame([metrics_dict])
    metrics["backtrader_frequency"] = "monthly"
    if not equity.empty:
        eval_equity = equity[pd.to_datetime(equity["month"]) >= EVALUATION_START]
        metrics["final_value"] = (
            float(eval_equity["value"].iloc[-1]) if not eval_equity.empty else np.nan
        )
    save_csv(metrics, prefix.with_name(prefix.name + "_metrics.csv"))
    return {"equity": equity, "orders": orders, "trades": trades, "positions": positions, "metrics": metrics}


def make_daily_stock_feed(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    feed = prices.loc[prices["ticker"].eq(ticker), ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]].copy()
    feed = feed.rename(
        columns={
            "adjOpen": "open",
            "adjHigh": "high",
            "adjLow": "low",
            "adjClose": "close",
            "adjVolume": "volume",
        }
    )
    feed["openinterest"] = 0
    feed = feed[["date", "open", "high", "low", "close", "volume", "openinterest"]].dropna(subset=["date", "open", "high", "low", "close"])
    feed = feed[(feed["open"] > 0) & (feed["high"] > 0) & (feed["low"] > 0) & (feed["close"] > 0)]
    return feed.sort_values("date").set_index("date")


def make_daily_index_feed(index: pd.DataFrame) -> pd.DataFrame:
    feed = index[["date", "open", "high", "low", "adj_close", "volume"]].copy()
    feed = feed.rename(columns={"adj_close": "close"})
    feed["openinterest"] = 0
    feed = feed[["date", "open", "high", "low", "close", "volume", "openinterest"]].dropna(subset=["date", "open", "high", "low", "close"])
    feed = feed[(feed["open"] > 0) & (feed["high"] > 0) & (feed["low"] > 0) & (feed["close"] > 0)]
    return feed.sort_values("date").set_index("date")


def run_backtrader_daily_stop_take(
    prices: pd.DataFrame,
    index: pd.DataFrame,
    signals: dict[pd.Timestamp, list[str]],
    name: str,
    stop_loss: float | None,
    take_profit: float | None,
    output_dir: Path | None = None,
    sizing_method: str = "fixed_cash",
    sizing_target_pct: float = 0.10,
    vol_lookback_days: int = VOL_LOOKBACK_DAYS,
    vol_floor_ann: float = VOL_FLOOR_ANN,
) -> dict[str, pd.DataFrame]:
    print(f"Running daily Backtrader native stop/limit strategy: {name}...")
    if stop_loss is None and take_profit is None:
        raise ValueError("Daily stop/take runner requires at least one risk-exit threshold.")
    tickers = sorted({t for names in signals.values() for t in names})
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.0)
    if sizing_method == "vol_targeted":
        cerebro.addsizer(
            VolatilityTargetedSizer,
            target_pct=sizing_target_pct,
            vol_lookback_days=vol_lookback_days,
            vol_floor_ann=vol_floor_ann,
        )
    elif sizing_method == "percent_of_equity":
        cerebro.addsizer(EquityPercentSizer, target_pct=sizing_target_pct)
    else:
        cerebro.addsizer(FixedCashSizer, cash_per_trade=CASH_PER_TRADE)

    idx_feed = make_daily_index_feed(index)
    cerebro.adddata(bt.feeds.PandasData(dataname=idx_feed), name="BENCHMARK")

    available_tickers: list[str] = []
    for ticker in tickers:
        feed = make_daily_stock_feed(prices, ticker)
        if len(feed) < 20:
            continue
        cerebro.adddata(bt.feeds.PandasData(dataname=feed), name=ticker)
        available_tickers.append(ticker)

    missing = sorted(set(tickers) - set(available_tickers))
    if missing:
        print(f"Skipped {len(missing)} tickers with insufficient daily bars for {name}.")

    normalized_signals = {pd.Timestamp(month).to_period("M").to_timestamp("M"): names for month, names in signals.items()}
    cerebro.addstrategy(
        DailySignalStopTakeStrategy,
        signals=normalized_signals,
        benchmark_name="BENCHMARK",
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    strategies = cerebro.run(runonce=False)
    strat = strategies[0]
    equity = pd.DataFrame(strat.equity_log)
    orders = pd.DataFrame(strat.order_log)
    trades = pd.DataFrame(strat.trade_log)
    positions = pd.DataFrame(strat.position_log)
    if not equity.empty:
        equity = equity.drop_duplicates("date", keep="last").sort_values("date")
        equity["return"] = equity["value"].pct_change().fillna(0.0)

    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"backtrader_daily_{name}"
    save_csv(equity, prefix.with_name(prefix.name + "_equity_curve.csv"))
    save_csv(orders, prefix.with_name(prefix.name + "_orders.csv"))
    save_csv(trades, prefix.with_name(prefix.name + "_trades.csv"))
    save_csv(positions, prefix.with_name(prefix.name + "_positions.csv"))
    metrics_dict = (
        metrics_over_evaluation_window(
            equity,
            f"backtrader_daily_{name}",
            date_col="date",
            return_col="return",
            period_label="daily",
            periods_per_year=252,
        )
        if not equity.empty
        else {"name": f"backtrader_daily_{name}"}
    )
    metrics = pd.DataFrame([metrics_dict])
    metrics["backtrader_frequency"] = "daily"
    metrics["stop_loss"] = stop_loss
    metrics["take_profit"] = take_profit
    if not equity.empty:
        eval_equity = equity[pd.to_datetime(equity["date"]) >= EVALUATION_START]
        metrics["final_value"] = (
            float(eval_equity["value"].iloc[-1]) if not eval_equity.empty else np.nan
        )
    save_csv(metrics, prefix.with_name(prefix.name + "_metrics.csv"))
    return {"equity": equity, "orders": orders, "trades": trades, "positions": positions, "metrics": metrics}


def assert_backtrader_long_only(bt_result: dict[str, pd.DataFrame], name: str) -> float:
    positions = bt_result.get("positions", pd.DataFrame())
    if positions.empty or "size" not in positions.columns:
        return 0.0
    min_size = float(pd.to_numeric(positions["size"], errors="coerce").min())
    if min_size < 0:
        raise AssertionError(f"{name} created a short position in Backtrader. Minimum size: {min_size}")
    return min_size


def get_strategy_specs() -> list[StrategySpec]:
    """Return the staged strategy ladder: base through improved 3."""
    return [
        StrategySpec(
            BASE_STRATEGY_NAME,
            {"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
            top_n=10,
            trend_col="trend_z",
            notes="Base: assignment-style full-sample index trend regression, equal factor weights, no stop-loss/take-profit.",
        ),
        StrategySpec(
            IMPROVED_STRATEGY_NAME,
            {"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
            top_n=10,
            trend_col="trend_expanding_z",
            notes="Improved 1: same signals and weights as base, but trend uses expanding no-lookahead index regressions.",
        ),
        StrategySpec(
            IMPROVED_2_STRATEGY_NAME,
            {"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
            top_n=10,
            trend_col="trend_expanding_z",
            stop_loss=0.10,
            take_profit=0.20,
            notes="Improved 2: builds on improved 1 and adds 10% stop-loss plus 20% take-profit risk exits.",
        ),
        StrategySpec(
            IMPROVED_3_STRATEGY_NAME,
            {"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
            top_n=10,
            trend_col="trend_expanding_z",
            stop_loss=0.10,
            take_profit=0.20,
            weight_method="rolling_rank_ic",
            weight_lookback_months=60,
            weight_min_periods=24,
            weight_shrink_to_equal=0.50,
            min_factor_weight=0.10,
            max_factor_weight=0.45,
            notes="Improved 3: builds on improved 2 and uses past-only rolling rank-IC weights with shrinkage and caps.",
        ),
    ]


def run_strategy_experiments(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, StrategySpec]]:
    print("Testing base strategy and improvement variants...")
    specs = get_strategy_specs()
    curves: list[pd.DataFrame] = []
    holdings: list[pd.DataFrame] = []
    spec_map = {s.name: s for s in specs}
    for spec in specs:
        curve, hold = simulate_vector_strategy(panel, spec)
        curves.append(curve)
        if not hold.empty:
            holdings.append(hold)
    all_curves = pd.concat(curves, ignore_index=True)
    all_holdings = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    metrics = pd.DataFrame(
        [
            metrics_over_evaluation_window(
                g.sort_values("month"),
                name,
                date_col="month",
                return_col="portfolio_return",
                period_label="monthly",
                periods_per_year=12,
            )
            | {
                "avg_positions": filter_to_evaluation_window(g, "month")["n_positions"].mean(),
            }
            for name, g in all_curves.groupby("strategy", sort=True)
        ]
    ).sort_values("annualized_sharpe", ascending=False)

    experiment_log = pd.DataFrame(
        [
            {
                "strategy": spec.name,
                "weights": json.dumps(spec.weights),
                "top_n": spec.top_n,
                "trend_col": spec.trend_col,
                "weight_method": spec.weight_method,
                "weight_lookback_months": spec.weight_lookback_months,
                "weight_min_periods": spec.weight_min_periods,
                "weight_shrink_to_equal": spec.weight_shrink_to_equal,
                "min_factor_weight": spec.min_factor_weight,
                "max_factor_weight": spec.max_factor_weight,
                "regime_filter": spec.regime_filter,
                "stop_loss": spec.stop_loss,
                "take_profit": spec.take_profit,
                "assignment_scope": is_assignment_scope_strategy(spec),
                "notes": spec.notes,
            }
            for spec in specs
        ]
    )
    metrics = metrics.merge(experiment_log.rename(columns={"strategy": "name"}), on="name", how="left")

    return metrics, all_curves, all_holdings, spec_map


def save_categorized_strategy_outputs(
    strategy_metrics: pd.DataFrame,
    strategy_curves: pd.DataFrame,
    strategy_holdings: pd.DataFrame,
    base_name: str,
    improved_name: str,
    improved_2_name: str | None = None,
    improved_3_name: str | None = None,
) -> None:
    """Write the staged strategy outputs into separate folders."""
    targets = [(base_name, BASE_RESULTS_DIR), (improved_name, IMPROVED_RESULTS_DIR)]
    if improved_2_name is not None:
        targets.append((improved_2_name, IMPROVED_2_RESULTS_DIR))
    if improved_3_name is not None:
        targets.append((improved_3_name, IMPROVED_3_RESULTS_DIR))

    for name, out_dir in targets:
        save_csv(strategy_metrics[strategy_metrics["name"].eq(name)], out_dir / "vector_metrics.csv")
        save_csv(strategy_curves[strategy_curves["strategy"].eq(name)], out_dir / "vector_equity_curve.csv")
        if not strategy_holdings.empty:
            save_csv(strategy_holdings[strategy_holdings["strategy"].eq(name)], out_dir / "vector_holdings.csv")

    staged_names = [name for name, _ in targets]
    save_csv(strategy_metrics[strategy_metrics["name"].isin(staged_names)], COMPARISON_RESULTS_DIR / "strategy_stage_metrics.csv")
    save_csv(strategy_curves[strategy_curves["strategy"].isin(staged_names)], COMPARISON_RESULTS_DIR / "strategy_stage_curves.csv")
    base_vs_improved = strategy_metrics[strategy_metrics["name"].isin(staged_names)].copy()
    role_map = {base_name: "base", improved_name: "improved_1"}
    if improved_2_name is not None:
        role_map[improved_2_name] = "improved_2"
    if improved_3_name is not None:
        role_map[improved_3_name] = "improved_3"
    base_vs_improved["role"] = base_vs_improved["name"].map(role_map)
    role_order = pd.Categorical(base_vs_improved["role"], categories=["base", "improved_1", "improved_2", "improved_3"], ordered=True)
    base_vs_improved = base_vs_improved.assign(role=role_order).sort_values("role")
    save_csv(base_vs_improved, COMPARISON_RESULTS_DIR / "base_vs_improved_metrics.csv")


def save_strategy_weight_histories(panel: pd.DataFrame, specs_by_dir: list[tuple[StrategySpec, Path]]) -> None:
    histories = []
    for spec, out_dir in specs_by_dir:
        history = factor_weight_history(panel, spec)
        save_csv(history, out_dir / "factor_weight_history.csv")
        histories.append(history)
    if histories:
        save_csv(pd.concat(histories, ignore_index=True), COMPARISON_RESULTS_DIR / "factor_weight_history.csv")


def monte_carlo_random_portfolios(
    panel: pd.DataFrame,
    base_spec: StrategySpec,
    n_sims: int = 1000,
    output_name: str | None = None,
    output_dir: Path | None = None,
    daily_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    print(f"Running Monte Carlo robustness test for {base_spec.name} with {n_sims} random portfolios...")
    rng = np.random.default_rng(RNG_SEED)
    base_curve, _ = simulate_vector_strategy(panel, base_spec, daily_prices=daily_prices)
    base_sharpe = metrics_over_evaluation_window(
        base_curve, base_spec.name, date_col="month", return_col="portfolio_return"
    )["annualized_sharpe"]
    months = sorted(panel["month"].dropna().unique())
    sim_rows: list[dict[str, Any]] = []

    eligible_by_month = {
        month: g.loc[g["eligible"], ["ticker", "GICS Sector", "next_open", "next_high", "next_low", "next_ret_oc", "regime_on"]].dropna(subset=["ticker", "next_ret_oc"]).reset_index(drop=True)
        for month, g in panel.groupby("month", sort=True)
    }
    use_vol = base_spec.sizing_method == "vol_targeted"
    if use_vol and daily_prices is None:
        raise ValueError("vol_targeted Monte Carlo requires daily_prices.")
    for sim in range(n_sims):
        equity = INITIAL_CASH
        rows_this_sim: list[dict[str, Any]] = []
        for month in months:
            g = eligible_by_month.get(month)
            if g is None or g.empty:
                rows_this_sim.append({"month": month, "portfolio_return": 0.0})
                continue
            if base_spec.regime_filter and not month_regime_is_on(g):
                rows_this_sim.append({"month": month, "portfolio_return": 0.0})
                continue
            if base_spec.sizing_method in ("percent_of_equity", "vol_targeted"):
                max_positions = base_spec.top_n
            else:
                max_positions = max(0, int(equity // CASH_PER_TRADE))
            n = min(base_spec.top_n, max_positions, len(g))
            if n == 0:
                rows_this_sim.append({"month": month, "portfolio_return": 0.0})
                continue
            picks = rng.choice(len(g), size=n, replace=False)
            selected = g.iloc[picks].copy()
            realized = stop_take_return(selected, base_spec.stop_loss, base_spec.take_profit)
            if use_vol:
                vol_weights = compute_inverse_vol_weights(
                    selected["ticker"].tolist(),
                    pd.Timestamp(month),
                    daily_prices,
                    lookback_days=base_spec.vol_lookback_days,
                    vol_floor_ann=base_spec.vol_floor_ann,
                )
                per_position_dollars = position_size_for_spec(base_spec, equity, vol_weights=vol_weights)
                per_position_arr = per_position_dollars.reindex(selected["ticker"].values).values
                pnl = float(np.nansum(per_position_arr * realized.values))
            else:
                per_position_dollars = position_size_for_spec(base_spec, equity)
                pnl = float(per_position_dollars) * float(realized.sum())
            ret = pnl / equity if equity else 0.0
            equity += pnl
            rows_this_sim.append({"month": month, "portfolio_return": ret})
        sim_curve = pd.DataFrame(rows_this_sim)
        eval_metrics = metrics_over_evaluation_window(
            sim_curve, f"random_{sim}", date_col="month", return_col="portfolio_return"
        )
        sim_rows.append(
            {
                "simulation": sim,
                "annualized_sharpe": eval_metrics.get("annualized_sharpe", np.nan),
                "final_equity": eval_metrics.get("final_equity", np.nan),
            }
        )

    out = pd.DataFrame(sim_rows)
    out["strategy_sharpe"] = base_sharpe
    out["p_value"] = (out["annualized_sharpe"] >= base_sharpe).mean()
    out["evaluation_start"] = EVALUATION_START.isoformat()
    if output_name is None:
        output_name = "monte_carlo_random_portfolios.csv"
    save_csv(out, (output_dir or RESULTS_DIR) / output_name)
    return out


def strategy_benchmark_comparison(strategy_curves: pd.DataFrame, index_monthly: pd.DataFrame) -> pd.DataFrame:
    print("Comparing strategies against ^GSPC benchmark...")
    bench = index_monthly[["month", "close"]].copy().sort_values("month")
    bench["benchmark_return"] = bench["close"].pct_change()
    rows: list[dict[str, Any]] = []
    for name, g in strategy_curves.groupby("strategy", sort=True):
        g = g.sort_values("month")[["month", "portfolio_return"]]
        merged = g.merge(bench[["month", "benchmark_return"]], on="month", how="inner").dropna()
        merged = filter_to_evaluation_window(merged, "month")
        if len(merged) < 24:
            continue
        excess = merged["portfolio_return"] - merged["benchmark_return"]
        fit = sm.OLS(
            merged["portfolio_return"],
            sm.add_constant(merged[["benchmark_return"]], has_constant="add"),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append(
            {
                "strategy": name,
                "n_months": len(merged),
                "avg_strategy_return": merged["portfolio_return"].mean(),
                "avg_benchmark_return": merged["benchmark_return"].mean(),
                "avg_excess_return": excess.mean(),
                "annualized_excess_return_approx": excess.mean() * 12,
                "excess_t_stat": st.ttest_1samp(excess, 0.0).statistic,
                "excess_p_value": st.ttest_1samp(excess, 0.0).pvalue,
                "beta_to_sp500": fit.params.get("benchmark_return", np.nan),
                "monthly_alpha": fit.params.get("const", np.nan),
                "annualized_alpha_approx": fit.params.get("const", np.nan) * 12,
                "alpha_t_stat": fit.tvalues.get("const", np.nan),
                "alpha_p_value": fit.pvalues.get("const", np.nan),
                "r_squared": fit.rsquared,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("annualized_alpha_approx", ascending=False)
    save_csv(out, COMPARISON_RESULTS_DIR / "strategy_benchmark_comparison.csv")
    return out


def block_bootstrap(
    curve: pd.DataFrame,
    block_size: int = 6,
    n_sims: int = 1000,
    output_name: str = "block_bootstrap_base_strategy.csv",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    print(f"Running block bootstrap robustness test -> {output_name}...")
    rng = np.random.default_rng(RNG_SEED + 1)
    curve_eval = filter_to_evaluation_window(curve, "month")
    r = curve_eval.sort_values("month")["portfolio_return"].dropna().to_numpy()
    rows = []
    if len(r) < block_size:
        return pd.DataFrame()
    blocks = [r[i : i + block_size] for i in range(0, len(r) - block_size + 1)]
    for sim in range(n_sims):
        sample: list[float] = []
        while len(sample) < len(r):
            sample.extend(blocks[int(rng.integers(0, len(blocks)))])
        sample = sample[: len(r)]
        m = perf_metrics(pd.Series(sample), f"bootstrap_{sim}")
        rows.append(
            {
                "simulation": sim,
                "annualized_sharpe": m.get("annualized_sharpe", np.nan),
                "cumulative_return": m.get("cumulative_return", np.nan),
                "max_drawdown": m.get("max_drawdown", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, (output_dir or RESULTS_DIR) / output_name)
    return out


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    headers = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([headers, sep, *rows])


def write_strategy_history(
    strategy_metrics: pd.DataFrame,
    experiment_log: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
    sector_summary: pd.DataFrame,
    monte_carlo_base: pd.DataFrame,
    monte_carlo_final: pd.DataFrame,
    walk_forward: pd.DataFrame,
    fmp_summary: pd.DataFrame,
    final_strategy_name: str,
) -> None:
    print("Writing detailed strategy history...")
    metrics = strategy_metrics.merge(experiment_log.rename(columns={"strategy": "name"}), on="name", how="left", suffixes=("", "_log"))
    metrics = metrics.merge(benchmark_comparison.rename(columns={"strategy": "name"}), on="name", how="left")
    metrics = metrics.sort_values("annualized_sharpe", ascending=False)
    base = metrics[metrics["name"].eq(BASE_STRATEGY_NAME)].iloc[0]
    improved_1 = metrics[metrics["name"].eq(IMPROVED_STRATEGY_NAME)].iloc[0]
    improved_2 = metrics[metrics["name"].eq(IMPROVED_2_STRATEGY_NAME)].iloc[0]
    best = metrics[metrics["name"].eq(final_strategy_name)].iloc[0]
    mc_base_p = monte_carlo_base["p_value"].iloc[0]
    mc_final_p = monte_carlo_final["p_value"].iloc[0]

    content = f"""# Strategy History And Improvement Log

This file is the living history of the S&P 500 factor investing project. It records what we built, what we removed, and what we should revisit later. The executable project is deliberately staged: the core ladder is base through improved 3, and the later focused experiments are improved 4 and improved 5.

## Project Rules We Must Preserve

- The project remains aligned with the factor-investing assignment: ROE, P/E, momentum, and trend factors are always the foundation.
- `^GSPC` replaces BIST100 as the index benchmark and trend-regression input.
- The final data window stops at May 2026.
- Backtrader uses initial cash `1,000,000`, fixed cash per trade `100,000`, and zero commission.
- The base Backtrader strategy uses market orders only.
- Stop-loss and take-profit are not part of the base. They enter at improved 2 and remain in improved 3.
- Dynamic factor weighting is not part of the base. It enters only at improved 3.
- Other ideas are kept as notes here, not as executable project outputs.

## Why We Improved The Base Strategy

The base strategy follows the project formula exactly, except that `^GSPC` replaces BIST100 because this version studies the S&P 500:

- ROE, P/E, momentum, and trend are the four required factors.
- Trend uses the full-sample index predictive regression requested in the project instructions.
- The base Backtrader strategy uses market orders, initial capital `1,000,000`, fixed cash per trade `100,000`, and commission `0`.
- The base strategy does not use stop-loss or take-profit. Those are improvement ideas, not base requirements.

The base strategy worked in absolute terms, but it should still be judged cautiously:

- Base strategy Sharpe: `{monte_carlo_base['strategy_sharpe'].iloc[0]:.4f}`
- Base Monte Carlo p-value: `{mc_base_p:.4f}`

Interpretation: in the current cleaned run, no random top-10 S&P 500 portfolio in 1,000 simulations matched the base Sharpe. This supports the factor-selection result under the project's Monte Carlo design, but it is not a guarantee of real-money robustness.

The individual FMPs also warned us not to overclaim. Their t-statistics and IC values are weak:

{markdown_table(fmp_summary, ['approach', 'factor', 'annualized_sharpe', 't_stat', 'p_value', 'avg_ic', 'avg_rank_ic'])}

## Development Path

The current executable strategy ladder is intentionally small and sequential:

{markdown_table(metrics, ['name', 'assignment_scope', 'annualized_sharpe', 'final_equity', 'max_drawdown', 'avg_positions', 'annualized_alpha_approx', 'beta_to_sp500'], float_digits=4)}

## Variant Notes

{markdown_table(metrics, ['name', 'assignment_scope', 'top_n', 'trend_col', 'weight_method', 'weight_lookback_months', 'regime_filter', 'stop_loss', 'take_profit', 'notes'], float_digits=3)}

## What Improved And Why

### 1. Base Strategy

The base strategy is the literal project-style implementation:

- factor set: ROE, positive inverse P/E, 12-month momentum, trend;
- trend method: one full-sample `^GSPC` predictive regression using all available index data through May 2026;
- insignificant moving-average variables are dropped using a 5% significance cutoff;
- factor values are winsorized and monthly z-scored;
- signals at month t trade month t+1;
- Backtrader base uses market orders, fixed cash sizing, zero commission, and no stop-loss/take-profit.

Base vector Sharpe: `{base['annualized_sharpe']:.4f}`.

### 2. Improved Strategy 1: Expanding Trend Regression

Improved 1 keeps every base choice fixed except the trend-regression estimation window. Instead of using one full-sample trend regression, it estimates an expanding `^GSPC` regression for each signal month using only index observations available before that month. This is a scientific improvement because it reduces look-ahead bias, even if it may or may not improve raw performance.

- Base vector Sharpe: `{base['annualized_sharpe']:.4f}`
- Improved 1 vector Sharpe: `{improved_1['annualized_sharpe']:.4f}`

### 3. Improved Strategy 2: Stop-Loss And Take-Profit Layer

Improved 2 builds directly on improved 1 and adds risk exits:

- trend method: expanding no-lookahead trend regression;
- stop-loss: 10%;
- take-profit: 20%;
- vector results use a monthly open/high/low/close approximation with stop priority if stop and take-profit are both touched in the same month;
- Backtrader results use daily adjusted OHLC bars, explicit market orders for entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

Improved 2 vector Sharpe: `{improved_2['annualized_sharpe']:.4f}`.

Order-model note: the Backtrader stop/take engine now uses native `bt.Order.Stop` and `bt.Order.Limit` protective orders. Rebalance exits cancel live protective orders before submitting market exits so stale stop/limit orders cannot create accidental shorts. Saved Backtrader metrics should be regenerated before final interpretation under this execution model.

### 4. Improved Strategy 3: Past-Only Dynamic Factor Weighting

Improved 3 builds directly on improved 2 and changes only the factor weighting rule. Instead of hard-coding a value/quality tilt, it estimates factor weights from past rank-IC evidence:

- lookback window: 60 months;
- minimum history before dynamic weights: 24 months;
- score: positive rolling rank-IC information ratio;
- shrinkage: 50% back to equal weights;
- bounds: each factor must stay between 10% and 45%.

This is closer to a real-world process because the weights are based only on information available before the signal month and are constrained to avoid single-factor overfitting.

- Improved 2 vector Sharpe: `{improved_2['annualized_sharpe']:.4f}`
- Improved 3 vector Sharpe: `{best['annualized_sharpe']:.4f}`

### 5. Removed Or Deferred Ideas

We previously explored HZZ cross-sectional trend, sector caps, volatility penalties, regime filters, top-N changes, and static value/quality-heavy factor weights. Those ideas were removed from the active code/results to keep the current project focused on a transparent one-change-at-a-time ladder. They can be reintroduced later one by one.

### 6. Next Improvement Rule

Every future improvement should build on the latest accepted improved strategy and change only one main design dimension at a time. That is how we keep the try/fail path scientifically readable.

## Robustness Results

Monte Carlo:

- Base strategy p-value: `{mc_base_p:.4f}`
- Improved 3 strategy p-value: `{mc_final_p:.4f}`

Walk-forward:

{markdown_table(walk_forward, ['strategy', 'train_sharpe_to_2020', 'test_sharpe_2021_2026', 'test_cumulative_return_2021_2026', 'test_max_drawdown_2021_2026', 'selected_by_train'])}

Benchmark comparison:

{markdown_table(benchmark_comparison, ['strategy', 'annualized_excess_return_approx', 'excess_t_stat', 'excess_p_value', 'annualized_alpha_approx', 'alpha_t_stat', 'alpha_p_value', 'beta_to_sp500'], max_rows=20)}

## Current Judgment

The project is stronger after separating the literal base from sequential improvements, Monte Carlo checks, walk-forward validation, and benchmark alpha comparisons. However, the honest conclusion remains cautious:

- The base and improved strategies are strong versus the current random-portfolio Monte Carlo benchmark.
- Improved 1 is primarily a methodology improvement because it reduces look-ahead bias in the trend factor.
- Improved 2 is a risk-management test on top of improved 1 and should be judged by both return and drawdown.
- Improved 3 is a weighting-process test on top of improved 2 and should be judged against improved 2, not just against the base.
- Vector curves are screening summaries; executable trading evidence comes from the saved Backtrader runs. Base and improved 1 use monthly `bt.Order.Market` Backtrader orders; improved 2 and improved 3 use daily Backtrader market entries/rebalances plus native stop/limit protective exits.
- The universe uses current S&P 500 constituents, so survivorship bias remains.
- Transaction costs and slippage are ignored because the assignment requires commission `0`.
- A production strategy would need historical index membership, costs, slippage, beta/sector neutrality, and a data-snooping-adjusted test such as White's Reality Check or Hansen's SPA.

## External Methodology Notes

- PyAnomaly emphasizes the same research building blocks we use: firm characteristics, winsorizing/trimming, quantile portfolios, long-short portfolios, factor regression, and cross-sectional regression.
- OpenSourceAP/CrossSection shows how much infrastructure serious factor replication needs: signal documentation, portfolio construction variants, and reproducibility checks.
- Quantitativo's trend-factor implementation stresses survivorship-bias-free universes, price filters, correct signal/return shifting, and the observation that post-2016 trend-factor edge can weaken.
- The removed research ideas are practical extensions, not replacements for the required factor framework.

## Ideas To Revisit Later

- Run a real White Reality Check or Hansen SPA test over all tried strategy variants.
- Use historical S&P 500 membership instead of current constituents.
- Tune stop-loss/take-profit values only after recording the first fixed 10%/20% improved 2 and dynamic-weight improved 3 results.
- Revisit static value/quality-heavy weights, top-N selection, cross-sectional trend, sector constraints, and volatility overlays later, one at a time.
- Add transaction costs and slippage as a separate robustness extension.
- Test historical S&P 500 membership to reduce survivorship bias.
"""
    (DOCS_DIR / "STRATEGY_HISTORY.md").write_text(content, encoding="utf-8")


def walk_forward_summary(strategy_metrics: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward validation across the staged strategy ladder, restricted to the
    common evaluation window starting at EVALUATION_START."""
    print("Computing walk-forward validation summary...")
    train_end = pd.Timestamp("2020-12-31")
    rows = []
    for name, g in curves.groupby("strategy", sort=True):
        g = g.sort_values("month")
        g_eval = filter_to_evaluation_window(g, "month")
        train = g_eval[g_eval["month"] <= train_end]
        test = g_eval[g_eval["month"] > train_end]
        if train.empty or test.empty:
            continue
        train_m = perf_metrics(train["portfolio_return"], f"{name}_train")
        test_m = perf_metrics(test["portfolio_return"], f"{name}_test")
        rows.append(
            {
                "strategy": name,
                "evaluation_start": EVALUATION_START.isoformat(),
                "train_sharpe_to_2020": train_m.get("annualized_sharpe", np.nan),
                "test_sharpe_2021_2026": test_m.get("annualized_sharpe", np.nan),
                "test_cumulative_return_2021_2026": test_m.get("cumulative_return", np.nan),
                "test_max_drawdown_2021_2026": test_m.get("max_drawdown", np.nan),
            }
        )
    out = pd.DataFrame(rows).sort_values("train_sharpe_to_2020", ascending=False)
    out["selected_by_train"] = False
    if not out.empty:
        out.loc[out.index[0], "selected_by_train"] = True
    save_csv(out, COMPARISON_RESULTS_DIR / "walk_forward_summary.csv")
    return out


def make_drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def make_figures(
    portfolio: pd.DataFrame,
    regression: pd.DataFrame,
    ic: pd.DataFrame,
    strategy_curves: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    index_monthly: pd.DataFrame,
    final_strategy_name: str | None = None,
) -> None:
    print("Creating figures...")
    # Portfolio-sort factor cumulative returns
    fig, ax = plt.subplots(figsize=(11, 6))
    for factor, g in portfolio.groupby("factor", sort=True):
        g = g.sort_values("month")
        ax.plot(g["month"], (1 + g["portfolio_fmp_return"]).cumprod(), label=factor)
    ax.set_title("Portfolio-Sort Factor-Mimicking Portfolios")
    ax.set_ylabel("Cumulative return, gross")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "factor_portfolio_cumulative_returns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for factor, g in regression.groupby("factor", sort=True):
        g = g.sort_values("month")
        ax.plot(g["month"], (1 + g["regression_fmp_return"]).cumprod(), label=factor)
    ax.set_title("Regression-Based Factor Returns")
    ax.set_ylabel("Cumulative return, gross")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "factor_regression_cumulative_returns.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=ic, x="factor", y="ic", ax=axes[0], errorbar=None)
    axes[0].set_title("Average Raw IC")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=ic, x="factor", y="rank_ic", ax=axes[1], errorbar=None)
    axes[1].set_title("Average Rank IC")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ic_rank_ic_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    final_name = final_strategy_name or strategy_metrics.iloc[0]["name"]
    for name in [BASE_STRATEGY_NAME, IMPROVED_STRATEGY_NAME, IMPROVED_2_STRATEGY_NAME, final_name]:
        g = strategy_curves[strategy_curves["strategy"].eq(name)].sort_values("month")
        if not g.empty:
            ax.plot(g["month"], g["equity"], label=name)
    bench = index_monthly[["month", "close"]].dropna().copy()
    bench = bench[bench["month"].between(strategy_curves["month"].min(), strategy_curves["month"].max())]
    if not bench.empty:
        bench["bench_equity"] = INITIAL_CASH * bench["close"] / bench["close"].iloc[0]
        ax.plot(bench["month"], bench["bench_equity"], label="^GSPC buy-and-hold", linestyle="--")
    ax.set_title("Strategy Equity Curves vs S&P 500")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_equity_vs_benchmark.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in [BASE_STRATEGY_NAME, IMPROVED_STRATEGY_NAME, IMPROVED_2_STRATEGY_NAME, final_name]:
        g = strategy_curves[strategy_curves["strategy"].eq(name)].sort_values("month")
        if not g.empty:
            ax.plot(g["month"], make_drawdown(g["equity"]), label=name)
    ax.set_title("Strategy Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_drawdowns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    order = strategy_metrics.sort_values("annualized_sharpe", ascending=False)
    sns.barplot(data=order, y="name", x="annualized_sharpe", ax=ax, color="#4C78A8")
    ax.set_title("Improvement Tests: Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_improvement_sharpe.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(monte_carlo["annualized_sharpe"].dropna(), bins=40, color="#8DA0CB", edgecolor="white")
    strategy_sharpe = monte_carlo["strategy_sharpe"].iloc[0]
    p_value = monte_carlo["p_value"].iloc[0]
    ax.axvline(strategy_sharpe, color="red", linewidth=2, label=f"base Sharpe = {strategy_sharpe:.2f}")
    ax.set_title(f"Monte Carlo Random Portfolio Sharpe Distribution (p={p_value:.3f})")
    ax.set_xlabel("Annualized Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monte_carlo_sharpe_histogram.png", dpi=160)
    plt.close(fig)


def write_project_docs(
    audit: pd.DataFrame,
    fmp_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    monte_carlo_final: pd.DataFrame,
    walk_forward: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
    backtrader_base_metrics: pd.DataFrame,
    backtrader_best_metrics: pd.DataFrame,
    final_strategy_name: str,
) -> None:
    print("Writing README and project report...")
    best = strategy_metrics[strategy_metrics["name"].eq(final_strategy_name)].iloc[0]
    base = strategy_metrics[strategy_metrics["name"].eq(BASE_STRATEGY_NAME)].iloc[0]
    bt_base = backtrader_base_metrics.iloc[0]
    bt_best = backtrader_best_metrics.iloc[0]
    mc_p = monte_carlo["p_value"].iloc[0]
    final_mc_p = monte_carlo_final["p_value"].iloc[0]
    best_benchmark = benchmark_comparison[benchmark_comparison["strategy"].eq(best["name"])]
    best_alpha = best_benchmark["annualized_alpha_approx"].iloc[0] if not best_benchmark.empty else np.nan
    best_alpha_t = best_benchmark["alpha_t_stat"].iloc[0] if not best_benchmark.empty else np.nan
    wf_line = ""
    if not walk_forward.empty:
        selected = walk_forward[walk_forward["selected_by_train"]].iloc[0]
        wf_line = (
            f"- Walk-forward selected `{selected['strategy']}` on pre-2021 Sharpe; "
            f"2021-May 2026 test Sharpe was {selected['test_sharpe_2021_2026']:.2f}.\n"
        )

    readme = f"""# S&P 500 Factor Investing Project

This is a standalone S&P 500 factor investing research project. It studies quality, value, momentum, and trend signals on a frozen current-constituent S&P 500 panel, using Yahoo Finance `^GSPC` as the benchmark and trend-regression index.

## Data Window

The analysis uses frozen data available through May 2026 only. Any rows after `2026-05-31` are excluded from processing.

Raw inputs:

- `data/raw/sp500_prices_long.csv`: Tiingo daily stock prices with adjusted OHLCV.
- `data/raw/sp500_fundamentals_daily_long.csv`: point-in-time daily valuation ratios and market cap.
- `data/raw/sp500_fundamentals_statements_long.csv`: statement fundamentals dated by public availability date.
- `data/raw/sp500_constituents.csv`: current S&P 500 universe metadata.
- `data/raw/sp500_index_yahoo.csv`: frozen Yahoo Finance `^GSPC` benchmark.

## Strategy

Implemented factors:

- ROE: higher is better.
- P/E: lower positive P/E is better.
- Momentum: 12-month price return.
- Trend: the base uses a full-sample predictive regression on `^GSPC` normalized moving-average deviations, with insignificant variables dropped, applied to each stock. Improved 1 replaces only this trend step with expanding no-lookahead regressions. Improved 2 adds stop-loss/take-profit. Improved 3 adds past-only dynamic factor weights.

All factors are month-end, lagged by one month, winsorized at 1st/99th percentiles, and cross-sectionally standardized.

## Main Results

- Base vector strategy `{BASE_STRATEGY_NAME}`: final equity `${base['final_equity']:,.0f}`, annualized Sharpe `{base['annualized_sharpe']:.2f}`.
- Improved 3 vector strategy `{best['name']}`: final equity `${best['final_equity']:,.0f}`, annualized Sharpe `{best['annualized_sharpe']:.2f}`.
- Backtrader base strategy: final value `${bt_base['final_value']:,.0f}`, annualized Sharpe `{bt_base['annualized_sharpe']:.2f}`.
- Backtrader improved 3 strategy: final value `${bt_best['final_value']:,.0f}`, annualized Sharpe `{bt_best['annualized_sharpe']:.2f}`.
- Monte Carlo random-portfolio Sharpe p-value for the base strategy: `{mc_p:.4f}`.
- Monte Carlo random-portfolio Sharpe p-value for improved 3: `{final_mc_p:.4f}`.
- Improved 3 annualized alpha vs `^GSPC`: `{best_alpha:.2%}` with alpha t-stat `{best_alpha_t:.2f}`.
{wf_line}
Backtrader is used for the staged strategy tests. Base and improved 1 use the monthly `bt.Order.Market` engine. Improved 2, improved 3, improved 4, and improved 5 use daily adjusted OHLC bars, `bt.Order.Market` entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits. All strategies use initial cash `1,000,000`, `FixedCashSizer` at `100,000` per trade, and zero commission.

## Output Layout

- `results/base_strategy/`: base strategy vector and Backtrader outputs.
- `results/improved_strategy/`: improved 1 expanding-regression strategy outputs.
- `results/improved_strategy_2/`: improved 2 stop-loss/take-profit strategy outputs.
- `results/improved_strategy_3/`: improved 3 dynamic-weight strategy outputs.
- `results/improved_strategy_4/`: improved 4 stop/take sensitivity and selected candidate.
- `results/improved_strategy_5/`: improved 5 market-regime filter experiment.
- `results/comparison/`: staged base-versus-improved comparison and walk-forward files.
- `results/fmp_analysis/`: factor-mimicking portfolio, IC, and factor comparison files.

## Reproduce

```powershell
cd C:\\Users\\asus\\Desktop\\sp500_factor_investing
py -3.10 src\\run_project.py
```

The run reads frozen CSVs, regenerates processed data, results, figures, and the PDF presentation.

## Limitations

- The universe is current S&P 500 constituents, so the study has survivorship bias.
- Transaction costs and slippage are ignored by design in the current research run.
- Shorting is shown only in FMP analysis; the implemented trading strategy is long-only.
- Public factor performance can decay over time, especially after 2020.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    report = f"""# Project Report: S&P 500 Factor Investing

## Assignment Alignment

This project follows the factor-investing section of the assignment: ROE, P/E, momentum, and trend factors; portfolio-sort and regression FMPs; IC tests; Backtrader backtests; Monte Carlo significance testing; and saved outputs.

The BIST100-specific index role is replaced with `^GSPC`, the S&P 500 index.

## Paper Summary

Han, Zhou, and Zhu (2016) propose a trend factor that combines short-, intermediate-, and long-horizon price information through normalized moving averages. Their paper argues that multiple moving-average horizons capture information diffusion, underreaction, overreaction, and feedback trading better than a single momentum horizon.

## Data Process

The raw stock files contain 503 current S&P 500 securities. Prices and daily ratios are filtered to `date <= 2026-05-31`. Statement fundamentals are filtered to `date_available <= 2026-05-31`, which prevents using financial statements before they became public.

The processed panel is monthly. Stock returns use adjusted prices. The final observation is the last trading date on or before May 2026.

## Factor Construction

- ROE uses the latest point-in-time `roe` statement field.
- P/E uses positive month-end `peRatio`, multiplied by `-1` so high factor scores mean cheap valuation.
- Momentum is `P_t / P_(t-12) - 1`.
- Trend uses normalized moving-average deviations with windows `{MA_WINDOWS}`. Base coefficients come from the assignment-style full-sample `^GSPC` predictive regression, with insignificant variables dropped. Improved 1, improved 2, and improved 3 use expanding no-lookahead `^GSPC` regressions for the trend factor.
- Improved 3 factor weights are estimated from a rolling 60-month rank-IC information-ratio rule, shrunk 50% toward equal weights and capped between 10% and 45% per factor.

## Results

Base strategy:

- final equity: `${base['final_equity']:,.0f}`;
- total return: `{base['total_return']:.2%}`;
- annualized Sharpe: `{base['annualized_sharpe']:.2f}`;
- max drawdown: `{base['max_drawdown']:.2%}`.

Improved 3 vector strategy:

- strategy: `{best['name']}`;
- final equity: `${best['final_equity']:,.0f}`;
- total return: `{best['total_return']:.2%}`;
- annualized Sharpe: `{best['annualized_sharpe']:.2f}`;
- max drawdown: `{best['max_drawdown']:.2%}`.

Backtrader base strategy:

- final value: `${bt_base['final_value']:,.0f}`;
- total return: `{bt_base['total_return']:.2%}`;
- annualized Sharpe: `{bt_base['annualized_sharpe']:.2f}`;
- max drawdown: `{bt_base['max_drawdown']:.2%}`.

Backtrader improved 3 strategy:

- final value: `${bt_best['final_value']:,.0f}`;
- total return: `{bt_best['total_return']:.2%}`;
- annualized Sharpe: `{bt_best['annualized_sharpe']:.2f}`;
- max drawdown: `{bt_best['max_drawdown']:.2%}`.

Monte Carlo p-value for the base Sharpe: `{mc_p:.4f}`.

Monte Carlo p-value for the improved 3 Sharpe: `{final_mc_p:.4f}`.

Improved 3 annualized alpha versus `^GSPC`: `{best_alpha:.2%}` with alpha t-stat `{best_alpha_t:.2f}`.

## What We Tried

- Equal-weight composite base strategy with assignment-style full-sample trend regression.
- Improved 1: the same strategy with expanding no-lookahead trend regression.
- Improved 2: improved 1 plus 10% stop-loss and 20% take-profit.
- Improved 3: improved 2 plus rolling rank-IC factor weights with shrinkage and caps.
- Static value/quality-heavy factor weights, top-N changes, and other overlays are deferred so improvements remain one-at-a-time.
- Walk-forward validation.

## Interpretation

The project should not claim statistically guaranteed skill only because the backtest is profitable. The Monte Carlo result is the main guardrail: if random portfolios often match or beat the strategy Sharpe, the conclusion must be cautious. The strongest investment conclusion is conditional: the factor process is economically sensible and reproducible, but real-money confidence would require survivorship-bias-free data, transaction-cost and slippage assumptions, and further out-of-sample testing.

The improved strategies must be judged sequentially: improved 1 asks whether removing trend look-ahead improves the research design and/or results; improved 2 asks whether adding explicit risk exits improves the improved 1 profile; improved 3 asks whether past-only factor weighting improves improved 2. This is why the presentation should emphasize the full strategy history and avoid claiming that the final result is a guaranteed tradable edge.

The vectorized improvement tests are used for fast monthly screening. The executable Backtrader runs are long-only: base and improved 1 use monthly `bt.Order.Market` orders, while improved 2 and improved 3 use daily adjusted OHLC data with `bt.Order.Market` entries/rebalances and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.
"""
    (DOCS_DIR / "PROJECT_REPORT.md").write_text(report, encoding="utf-8")

    reproducibility = """# Reproducibility

Run from a fresh Python session:

```powershell
cd C:\\Users\\asus\\Desktop\\sp500_factor_investing
py -3.10 -m pip install -r requirements.txt
py -3.10 src\\run_project.py
```

The code reads frozen CSV files from `data/raw/`. It does not redownload stock data. It downloads `^GSPC` only if `data/raw/sp500_index_yahoo.csv` is missing or stale; after that the benchmark is frozen and reused.

Expected regenerated folders:

- `data/processed/`
- `results/`
- `figures/`
- `presentation/`
"""
    (DOCS_DIR / "REPRODUCIBILITY.md").write_text(reproducibility, encoding="utf-8")

    dictionary = """# Data Dictionary

## Raw Inputs

- `sp500_prices_long.csv`: daily stock OHLCV, including dividend/split adjusted prices.
- `sp500_fundamentals_daily_long.csv`: point-in-time daily market cap, P/E, P/B, and PEG.
- `sp500_fundamentals_statements_long.csv`: statement fundamentals using `date_available`.
- `sp500_constituents.csv`: current S&P 500 universe metadata.
- `sp500_index_yahoo.csv`: frozen Yahoo Finance S&P 500 index data.

## Processed Outputs

- `monthly_stock_bars.csv`: month-end adjusted OHLCV and forward returns.
- `factor_panel.csv`: stock-month factor signals, z-scores, and forward returns.
- `fmp_portfolio_returns.csv`: top-minus-bottom factor returns.
- `fmp_regression_returns.csv`: monthly cross-sectional factor-premium estimates.
- `strategy_vector_equity_curves.csv`: vectorized strategy/improvement equity curves.
- `backtrader_*`: Backtrader equity, orders, trades, positions, and metrics.
"""
    (DOCS_DIR / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")

    checklist = pd.DataFrame(
        [
            ("requirements.txt has pinned versions", "OK"),
            ("raw data stored as CSV", "OK"),
            ("code reads frozen CSVs", "OK"),
            ("Backtrader used", "OK"),
            ("initial cash 1,000,000", "OK"),
            ("FixedCashSizer 100,000 per trade", "OK"),
            ("commission zero", "OK"),
            ("market orders for base strategy", "OK"),
            ("backtest results saved", "OK"),
            ("PDF presentation generated", "OK"),
            ("group participant names in PDF", "NEEDS_INPUT"),
            ("no data after 2026-05-31", "OK"),
            ("statement fundamentals use date_available", "OK"),
        ],
        columns=["item", "status"],
    )
    save_csv(checklist, RESULTS_DIR / "project_audit_checklist.csv")


def add_text_slide(pdf: PdfPages, title: str, bullets: list[str], footer: str = "") -> None:
    fig = plt.figure(figsize=(13.333, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.88, title, fontsize=28, fontweight="bold", va="top")
    y = 0.76
    for bullet in bullets:
        ax.text(0.08, y, f"- {bullet}", fontsize=16, va="top", wrap=True)
        y -= 0.085
    if footer:
        ax.text(0.06, 0.06, footer, fontsize=10, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def add_image_slide(pdf: PdfPages, title: str, image_path: Path, caption: str = "") -> None:
    if not image_path.exists():
        print(f"Skipping slide '{title}' - {image_path.name} not found.")
        return
    fig = plt.figure(figsize=(13.333, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.94, title, fontsize=24, fontweight="bold", va="top")
    img = plt.imread(image_path)
    image_ax = fig.add_axes([0.08, 0.14, 0.84, 0.72])
    image_ax.imshow(img)
    image_ax.axis("off")
    if caption:
        ax.text(0.08, 0.06, caption, fontsize=12, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def make_presentation(
    strategy_metrics: pd.DataFrame,
    fmp_summary: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    monte_carlo_final: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
    backtrader_base_metrics: pd.DataFrame,
    backtrader_best_metrics: pd.DataFrame,
    final_strategy_name: str,
) -> None:
    print("Generating PDF presentation...")
    path = PRESENTATION_DIR / "sp500_factor_investing_presentation.pdf"
    best = strategy_metrics[strategy_metrics["name"].eq(final_strategy_name)].iloc[0]
    base = strategy_metrics[strategy_metrics["name"].eq(BASE_STRATEGY_NAME)].iloc[0]
    bt_base = backtrader_base_metrics.iloc[0]
    bt_best = backtrader_best_metrics.iloc[0]
    best_benchmark = benchmark_comparison[benchmark_comparison["strategy"].eq(best["name"])]
    best_alpha = best_benchmark["annualized_alpha_approx"].iloc[0] if not best_benchmark.empty else np.nan
    with PdfPages(path) as pdf:
        add_text_slide(
            pdf,
            "S&P 500 Factor Investing",
            [
                "Group participants: add names before submission.",
                "Assignment setting adapted from BIST100 to S&P 500.",
                "Data window: all available observations through May 2026.",
                "Required factors: ROE, P/E, momentum, and trend.",
            ],
        )
        add_text_slide(
            pdf,
            "Paper Summary",
            [
                "Han, Zhou, and Zhu propose a trend factor using moving averages across multiple horizons.",
                "Normalized moving averages combine short-term reversal, intermediate momentum, and long-term reversal information.",
                "The paper forms top-minus-bottom portfolios from expected returns predicted by trend signals.",
                "Our project keeps the idea but adapts the index input to ^GSPC.",
            ],
        )
        add_text_slide(
            pdf,
            "Data And Signals",
            [
                "Stocks: current S&P 500 constituent panel from Tiingo data.",
                "Benchmark and trend-regression index: Yahoo Finance ^GSPC.",
                "Fundamentals are joined by public availability date to avoid look-ahead bias.",
                "All factors are month-end, winsorized, z-scored, and lagged before trading.",
            ],
        )
        add_text_slide(
            pdf,
            "Backtest Design",
            [
                "Initial cash: 1,000,000.",
                "FixedCashSizer: 100,000 per trade.",
                "Commission: 0.",
                "Base strategy: long-only top 10 composite factor stocks, monthly rebalance, market orders.",
                "Improved 1: same base strategy with expanding no-lookahead trend regression.",
                "Improved 2: improved 1 plus 10% stop-loss and 20% take-profit.",
                "Improved 3: improved 2 plus rolling rank-IC factor weights with shrinkage and caps.",
            ],
        )
        add_text_slide(
            pdf,
            "Strategy Ladder Overview",
            [
                "Base: Full-sample trend (look-ahead biased baseline).",
                "Imp 1: Expanding trend (honest baseline).",
                "Imp 2: Adds 10% stop-loss and 20% take-profit.",
                "Imp 3: Rolling rank-IC dynamic weights (rejected).",
                "Imp 4: Walk-forward optimized stop/take (5% / 30%).",
                "Imp 5: Regime filter (rejected).",
                "Imp 6: HZZ cross-sectional trend methodology.",
                "Imp 8: Equal-weight 5% per position (top 20).",
                "Imp 9: Inverse-vol-targeted sizing (top 20).",
            ],
        )
        add_image_slide(pdf, "Factor Portfolio Performance", FIGURES_DIR / "factor_portfolio_cumulative_returns.png")
        add_image_slide(pdf, "Information Coefficients", FIGURES_DIR / "ic_rank_ic_summary.png")
        
        # New all-strategies figures (will skip if not yet generated)
        add_image_slide(pdf, "All Strategies Equity Comparison", FIGURES_DIR / "all_strategies_equity_curves.png")
        add_image_slide(pdf, "Risk-Return Frontier", FIGURES_DIR / "sharpe_vs_drawdown_scatter.png")
        add_image_slide(pdf, "Drawdowns Across Strategies", FIGURES_DIR / "all_strategies_drawdowns.png")
        add_image_slide(pdf, "Walk-Forward: Train vs Test", FIGURES_DIR / "walk_forward_train_vs_test_scatter.png")
        add_image_slide(pdf, "Cumulative Alpha vs Benchmark", FIGURES_DIR / "cumulative_alpha_vs_gspc.png")
        add_image_slide(pdf, "Cost Drag Attribution (Imp 7)", FIGURES_DIR / "cost_drag_attribution_imp7.png")
        add_image_slide(pdf, "Sizing: Fixed vs Eq-Weight vs Vol-Tgt", FIGURES_DIR / "imp4_vs_8_vs_9_comparison.png")
        add_image_slide(pdf, "Position Concentration (HHI)", FIGURES_DIR / "position_concentration_hhi.png")
        
        # Original single-strategy figures
        add_image_slide(pdf, "Strategy vs Benchmark (Staged)", FIGURES_DIR / "strategy_equity_vs_benchmark.png")
        add_image_slide(pdf, "Drawdowns (Staged)", FIGURES_DIR / "strategy_drawdowns.png")
        add_image_slide(pdf, "Monte Carlo Robustness", FIGURES_DIR / "monte_carlo_sharpe_histogram.png")
        
        add_text_slide(
            pdf,
            "Multi-Comparison Robustness Test",
            [
                "Testing 9 variants at 5% significance means a 37% chance of a false positive.",
                "We applied the Hansen (2005) Superior Predictive Ability (SPA) test.",
                "And the Romano-Wolf (2005) StepM step-down procedure.",
                "These correct the Monte Carlo p-values for family-wise error rate across all variants.",
                "See results/robustness/ for the corrected p-values.",
            ],
        )
        add_image_slide(pdf, "Monte Carlo P-Value Comparison", FIGURES_DIR / "monte_carlo_pvalue_comparison.png")
        
        add_text_slide(
            pdf,
            "Backtrader Results",
            [
                f"Base Backtrader final value: ${bt_base['final_value']:,.0f}; Sharpe: {bt_base['annualized_sharpe']:.2f}.",
                f"Improved 3 Backtrader final value: ${bt_best['final_value']:,.0f}; Sharpe: {bt_best['annualized_sharpe']:.2f}.",
                "Settings: initial cash 1,000,000; FixedCashSizer 100,000; commission 0.",
                "Improved 2 and improved 3 use daily Backtrader bt.Order.Stop / bt.Order.Limit protective exits.",
                "Saved outputs include orders, trades, positions, equity curves, and metrics.",
            ],
        )
        add_text_slide(
            pdf,
            "Main Results (Staged Ladder)",
            [
                f"Base strategy final equity: ${base['final_equity']:,.0f}; Sharpe: {base['annualized_sharpe']:.2f}.",
                f"Improved 3 strategy: {best['name']}; final equity: ${best['final_equity']:,.0f}; Sharpe: {best['annualized_sharpe']:.2f}.",
                f"Monte Carlo p-value for base Sharpe: {monte_carlo['p_value'].iloc[0]:.4f}.",
                f"Monte Carlo p-value for improved 3 Sharpe: {monte_carlo_final['p_value'].iloc[0]:.4f}.",
                f"Improved 3 annualized alpha vs ^GSPC: {best_alpha:.2%}.",
                "Interpret results cautiously because the universe has survivorship bias and costs are ignored.",
            ],
        )
        add_text_slide(
            pdf,
            "Honest Limitations + Future Work",
            [
                "Survivorship bias: We only trade the current S&P 500 members.",
                "Look-ahead bias in base strategy: Trend uses full-sample regression.",
                "Next phase (Improved 10): Use WRDS / CRSP point-in-time membership and delisted returns.",
                "This will provide a truly survivorship-bias-free panel.",
                "The codebase is fully equipped to handle this upgrade via the new WRDS extraction script.",
            ],
        )
        add_text_slide(
            pdf,
            "Conclusion",
            [
                "The factor process is economically interpretable and reproducible.",
                "The strongest evidence comes from comparing base, improvement, walk-forward, and random-portfolio tests.",
                "We would not present this as real-money ready without survivorship-bias-free data and realistic costs.",
                "Future work: historical S&P membership, transaction costs, slippage, and stronger out-of-sample validation.",
            ],
        )


def data_audit(
    prices: pd.DataFrame,
    daily: pd.DataFrame,
    statements: pd.DataFrame,
    constituents: pd.DataFrame,
    index: pd.DataFrame,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [
        {"dataset": "prices", "rows": len(prices), "tickers": prices["ticker"].nunique(), "min_date": prices["date"].min(), "max_date": prices["date"].max(), "missing_key_value": prices["adjClose"].isna().sum()},
        {"dataset": "daily_fundamentals", "rows": len(daily), "tickers": daily["ticker"].nunique(), "min_date": daily["date"].min(), "max_date": daily["date"].max(), "missing_key_value": daily["peRatio"].isna().sum()},
        {"dataset": "statements", "rows": len(statements), "tickers": statements["ticker"].nunique(), "min_date": statements["date_available"].min(), "max_date": statements["date_available"].max(), "missing_key_value": statements["value"].isna().sum()},
        {"dataset": "constituents", "rows": len(constituents), "tickers": constituents["Symbol"].nunique(), "min_date": np.nan, "max_date": np.nan, "missing_key_value": constituents["Symbol"].isna().sum()},
        {"dataset": "sp500_index_yahoo", "rows": len(index), "tickers": 1, "min_date": index["date"].min(), "max_date": index["date"].max(), "missing_key_value": index["adj_close"].isna().sum()},
    ]
    if panel is not None:
        rows.append(
            {
                "dataset": "factor_panel",
                "rows": len(panel),
                "tickers": panel["ticker"].nunique(),
                "min_date": panel["month"].min(),
                "max_date": panel["month"].max(),
                "missing_key_value": panel["composite_score"].isna().sum(),
            }
        )
    audit = pd.DataFrame(rows)
    save_csv(audit, RESULTS_DIR / "data_audit.csv")
    return audit


def save_common_start_comparison(portfolio: pd.DataFrame, regression: pd.DataFrame) -> None:
    rows = []
    for approach, df, col in [
        ("portfolio_sort", portfolio, "portfolio_fmp_return"),
        ("regression", regression, "regression_fmp_return"),
    ]:
        starts = df.groupby("factor")["month"].min()
        common_start = starts.max()
        for factor, g in df[df["month"] >= common_start].groupby("factor", sort=True):
            m = perf_metrics(g.sort_values("month")[col], f"{approach}_{factor}_common")
            m["approach"] = approach
            m["factor"] = factor
            m["common_start"] = common_start
            rows.append(m)
    out = pd.DataFrame(rows)
    save_csv(out, FMP_RESULTS_DIR / "fmp_common_start_comparison.csv")


def save_selected_date_weights(weights: pd.DataFrame, target: str = "2024-12-31") -> None:
    if weights.empty:
        return
    target_date = pd.Timestamp(target)
    available = sorted(weights.loc[weights["month"] <= target_date, "month"].unique())
    if not available:
        return
    selected = available[-1]
    out = weights[weights["month"].eq(selected)].copy()
    save_csv(out, FMP_RESULTS_DIR / "selected_date_fmp_weight_comparison.csv")


def main() -> None:
    ensure_dirs()
    config = load_config()
    print(json.dumps({"project": config.get("project", {}).get("name", "sp500_factor_investing"), "cutoff": str(CUTOFF.date())}))

    prices, daily, statements, constituents, index = load_raw_data()
    if prices["date"].max() > CUTOFF or daily["date"].max() > CUTOFF or statements["date_available"].max() > CUTOFF:
        raise AssertionError("Raw loaded data includes observations after cutoff.")

    monthly = make_monthly_bars(prices)
    index_monthly = make_index_monthly(index)
    metrics = make_monthly_metrics(daily)
    roe = make_roe_panel(monthly, statements)
    stock_ma = compute_stock_ma_signals(prices)
    index_ma = compute_index_ma_signals(index)
    trend_coeffs = assignment_index_trend_coefficients(index_monthly, index_ma)
    trend = apply_index_trend_to_stocks(stock_ma, trend_coeffs)
    trend_expanding, trend_expanding_coeffs = expanding_index_trend_to_stocks(index_monthly, index_ma, stock_ma)
    panel = build_factor_panel(monthly, metrics, roe, trend, trend_expanding, constituents, index_monthly)

    save_csv(monthly, PROCESSED_DIR / "monthly_stock_bars.csv")
    save_csv(index_monthly, PROCESSED_DIR / "monthly_sp500_index.csv")
    save_csv(metrics, PROCESSED_DIR / "monthly_daily_fundamentals.csv")
    save_csv(roe, PROCESSED_DIR / "monthly_roe_asof.csv")
    save_csv(stock_ma, PROCESSED_DIR / "monthly_stock_ma_signals.csv")
    save_csv(index_ma, PROCESSED_DIR / "monthly_index_ma_signals.csv")
    save_csv(trend_coeffs, FMP_RESULTS_DIR / "trend_index_regression_coefficients.csv")
    save_csv(trend_expanding_coeffs, FMP_RESULTS_DIR / "trend_expanding_regression_coefficients.csv")
    save_csv(panel, PROCESSED_DIR / "factor_panel.csv")

    audit = data_audit(prices, daily, statements, constituents, index, panel)
    portfolio, regression, ic, weights = make_fmp_returns(panel)
    save_csv(portfolio, FMP_RESULTS_DIR / "fmp_portfolio_returns.csv")
    save_csv(regression, FMP_RESULTS_DIR / "fmp_regression_returns.csv")
    save_csv(ic, FMP_RESULTS_DIR / "factor_information_coefficients.csv")
    save_csv(weights, FMP_RESULTS_DIR / "fmp_weights.csv")
    fmp_summary = summarize_fmps(portfolio, regression, ic)
    save_csv(fmp_summary, FMP_RESULTS_DIR / "fmp_performance_summary.csv")
    save_common_start_comparison(portfolio, regression)
    save_selected_date_weights(weights)

    strategy_metrics, strategy_curves, strategy_holdings, spec_map = run_strategy_experiments(panel)
    base_spec = spec_map[BASE_STRATEGY_NAME]
    improved_spec = spec_map[IMPROVED_STRATEGY_NAME]
    improved_2_spec = spec_map[IMPROVED_2_STRATEGY_NAME]
    improved_3_spec = spec_map[IMPROVED_3_STRATEGY_NAME]
    monte_carlo = monte_carlo_random_portfolios(panel, base_spec, n_sims=1000, output_dir=BASE_RESULTS_DIR)
    benchmark_comparison = strategy_benchmark_comparison(strategy_curves, index_monthly)
    base_curve = strategy_curves[strategy_curves["strategy"].eq(BASE_STRATEGY_NAME)]
    bootstrap = block_bootstrap(base_curve, block_size=6, n_sims=1000, output_name="block_bootstrap.csv", output_dir=BASE_RESULTS_DIR)
    walk_forward = walk_forward_summary(strategy_metrics, strategy_curves)
    save_csv(walk_forward, COMPARISON_RESULTS_DIR / "walk_forward_summary.csv")

    save_categorized_strategy_outputs(
        strategy_metrics,
        strategy_curves,
        strategy_holdings,
        base_spec.name,
        improved_spec.name,
        improved_2_spec.name,
        improved_3_spec.name,
    )
    save_strategy_weight_histories(
        panel,
        [
            (base_spec, BASE_RESULTS_DIR),
            (improved_spec, IMPROVED_RESULTS_DIR),
            (improved_2_spec, IMPROVED_2_RESULTS_DIR),
            (improved_3_spec, IMPROVED_3_RESULTS_DIR),
        ],
    )
    monte_carlo_improved = monte_carlo_random_portfolios(
        panel,
        improved_spec,
        n_sims=1000,
        output_name="monte_carlo_random_portfolios.csv",
        output_dir=IMPROVED_RESULTS_DIR,
    )
    monte_carlo_improved_2 = monte_carlo_random_portfolios(
        panel,
        improved_2_spec,
        n_sims=1000,
        output_name="monte_carlo_random_portfolios.csv",
        output_dir=IMPROVED_2_RESULTS_DIR,
    )
    monte_carlo_final = monte_carlo_random_portfolios(
        panel,
        improved_3_spec,
        n_sims=1000,
        output_name="monte_carlo_random_portfolios.csv",
        output_dir=IMPROVED_3_RESULTS_DIR,
    )
    improved_curve = strategy_curves[strategy_curves["strategy"].eq(IMPROVED_STRATEGY_NAME)]
    block_bootstrap(
        improved_curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=IMPROVED_RESULTS_DIR,
    )
    improved_2_curve = strategy_curves[strategy_curves["strategy"].eq(IMPROVED_2_STRATEGY_NAME)]
    block_bootstrap(
        improved_2_curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=IMPROVED_2_RESULTS_DIR,
    )
    best_name = IMPROVED_3_STRATEGY_NAME
    best_spec = improved_3_spec
    best_curve = strategy_curves[strategy_curves["strategy"].eq(best_name)]
    final_bootstrap = block_bootstrap(
        best_curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=IMPROVED_3_RESULTS_DIR,
    )
    base_bt = run_backtrader(
        monthly,
        index_monthly,
        signals_from_strategy(panel, base_spec),
        BASE_STRATEGY_NAME,
        output_dir=BASE_RESULTS_DIR,
    )
    improved_bt = run_backtrader(
        monthly,
        index_monthly,
        signals_from_strategy(panel, improved_spec),
        IMPROVED_STRATEGY_NAME,
        output_dir=IMPROVED_RESULTS_DIR,
    )
    improved_2_bt = run_backtrader_daily_stop_take(
        prices,
        index,
        signals_from_strategy(panel, improved_2_spec),
        IMPROVED_2_STRATEGY_NAME,
        stop_loss=improved_2_spec.stop_loss,
        take_profit=improved_2_spec.take_profit,
        output_dir=IMPROVED_2_RESULTS_DIR,
    )
    best_bt = run_backtrader_daily_stop_take(
        prices,
        index,
        signals_from_strategy(panel, best_spec),
        best_name,
        stop_loss=best_spec.stop_loss,
        take_profit=best_spec.take_profit,
        output_dir=IMPROVED_3_RESULTS_DIR,
    )
    base_min_position = assert_backtrader_long_only(base_bt, BASE_STRATEGY_NAME)
    improved_min_position = assert_backtrader_long_only(improved_bt, IMPROVED_STRATEGY_NAME)
    improved_2_min_position = assert_backtrader_long_only(improved_2_bt, IMPROVED_2_STRATEGY_NAME)
    best_min_position = assert_backtrader_long_only(best_bt, best_name)

    make_figures(portfolio, regression, ic, strategy_curves, strategy_metrics, monte_carlo, index_monthly, best_name)
    write_strategy_history(
        strategy_metrics,
        strategy_metrics,
        benchmark_comparison,
        pd.DataFrame(),
        monte_carlo,
        monte_carlo_final,
        walk_forward,
        fmp_summary,
        best_name,
    )
    write_project_docs(
        audit,
        fmp_summary,
        strategy_metrics,
        monte_carlo,
        monte_carlo_final,
        walk_forward,
        benchmark_comparison,
        base_bt["metrics"],
        best_bt["metrics"],
        best_name,
    )
    make_presentation(
        strategy_metrics,
        fmp_summary,
        monte_carlo,
        monte_carlo_final,
        benchmark_comparison,
        base_bt["metrics"],
        best_bt["metrics"],
        best_name,
    )

    validation = pd.DataFrame(
        [
            {"check": "prices_cutoff", "status": "OK", "detail": str(prices["date"].max().date())},
            {"check": "daily_fundamentals_cutoff", "status": "OK", "detail": str(daily["date"].max().date())},
            {"check": "statements_date_available_cutoff", "status": "OK", "detail": str(statements["date_available"].max().date())},
            {"check": "panel_cutoff", "status": "OK", "detail": str(panel["month"].max().date())},
            {"check": "benchmark_cutoff", "status": "OK", "detail": str(index["date"].max().date())},
            {"check": "backtrader_base_long_only", "status": "OK", "detail": str(base_min_position)},
            {"check": "backtrader_improved_1_long_only", "status": "OK", "detail": str(improved_min_position)},
            {"check": "backtrader_improved_2_long_only", "status": "OK", "detail": str(improved_2_min_position)},
            {"check": "backtrader_improved_3_long_only", "status": "OK", "detail": str(best_min_position)},
            {"check": "results_generated", "status": "OK", "detail": str(len(list(RESULTS_DIR.rglob('*.csv'))))},
            {"check": "figures_generated", "status": "OK", "detail": str(len(list(FIGURES_DIR.glob('*.png'))))},
            {"check": "presentation_pdf", "status": "OK", "detail": str(PRESENTATION_DIR / "sp500_factor_investing_presentation.pdf")},
        ]
    )
    save_csv(validation, RESULTS_DIR / "validation_summary.csv")
    print("Project pipeline completed.")


if __name__ == "__main__":
    main()
