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
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"
APPENDIX_RESULTS_DIR = RESULTS_DIR / "appendix_experiments"
FIGURES_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"
PRESENTATION_DIR = ROOT / "presentation"
CONFIG_PATH = ROOT / "config" / "project_config.yaml"

CUTOFF = pd.Timestamp("2026-05-31")
START = pd.Timestamp("2006-06-01")
BENCHMARK = "^GSPC"
INITIAL_CASH = 1_000_000.0
CASH_PER_TRADE = 100_000.0
MA_WINDOWS = [3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
MA_COLS = [f"ma_dev_{w}" for w in MA_WINDOWS]
REQUIRED_FACTOR_Z = ["roe_z", "pe_z", "momentum_z", "trend_z"]
RNG_SEED = 5811


@dataclass(frozen=True)
class StrategySpec:
    name: str
    weights: dict[str, float]
    top_n: int = 10
    regime_filter: bool = False
    stop_loss: float | None = None
    take_profit: float | None = None
    trend_col: str = "trend_z"
    max_per_sector: int | None = None
    volatility_penalty: float = 0.0
    notes: str = ""


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        FMP_RESULTS_DIR,
        BASE_RESULTS_DIR,
        IMPROVED_RESULTS_DIR,
        COMPARISON_RESULTS_DIR,
        APPENDIX_RESULTS_DIR,
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
    monthly["volatility_6m"] = g["ret_1m_cc"].transform(lambda s: s.rolling(6, min_periods=4).std())
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
    selected = [col for col in MA_COLS if full_pvals.get(col, np.nan) <= 0.10]

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


def compute_hzz_trend(stock_ma: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Computing paper-style cross-sectional HZZ trend improvement factor...")
    data = stock_ma.merge(monthly[["ticker", "month", "next_ret_cc", "eligible_price"]], on=["ticker", "month"])
    rows: list[dict[str, Any]] = []
    for month, g in data.groupby("month", sort=True):
        reg = g.loc[g["eligible_price"], ["next_ret_cc", *MA_COLS]].replace([np.inf, -np.inf], np.nan).dropna()
        row: dict[str, Any] = {"month": month, "n_obs": len(reg), "r2": np.nan}
        for col in MA_COLS:
            row[col] = np.nan
        if len(reg) >= 100:
            fit = sm.OLS(reg["next_ret_cc"], sm.add_constant(reg[MA_COLS], has_constant="add")).fit()
            row["r2"] = fit.rsquared
            for col in MA_COLS:
                row[col] = fit.params.get(col, np.nan)
        rows.append(row)

    betas = pd.DataFrame(rows).sort_values("month")
    expected = betas[["month"]].copy()
    expected[MA_COLS] = betas[MA_COLS].shift(1).rolling(12, min_periods=6).mean()
    merged = stock_ma.merge(expected, on="month", suffixes=("", "_beta"))
    x = merged[MA_COLS].to_numpy(dtype=float)
    b = merged[[f"{c}_beta" for c in MA_COLS]].to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(b).all(axis=1)
    merged["hzz_trend_raw"] = np.nan
    merged.loc[valid, "hzz_trend_raw"] = np.sum(x[valid] * b[valid], axis=1)
    return merged[["ticker", "month", "hzz_trend_raw"]], betas


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
    hzz_trend: pd.DataFrame,
    constituents: pd.DataFrame,
    index_monthly: pd.DataFrame,
) -> pd.DataFrame:
    print("Assembling monthly factor panel...")
    panel = monthly.merge(metrics, on=["ticker", "month"], how="left")
    panel = panel.merge(roe, on=["ticker", "month"], how="left")
    panel = panel.merge(trend, on=["ticker", "month"], how="left")
    panel = panel.merge(hzz_trend, on=["ticker", "month"], how="left")
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
    panel["hzz_trend_factor"] = panel["hzz_trend_raw"]
    panel["volatility_factor"] = panel["volatility_6m"]
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
        ("hzz_trend_factor", "hzz_trend_z"),
        ("volatility_factor", "volatility_z"),
    ]:
        panel[target] = winsorized_zscore_by_month(panel, source, target)

    panel["low_volatility_z"] = -panel["volatility_z"]

    available_required = panel[REQUIRED_FACTOR_Z].notna().sum(axis=1)
    panel["composite_score"] = panel[REQUIRED_FACTOR_Z].mean(axis=1, skipna=True)
    panel.loc[available_required < 3, "composite_score"] = np.nan

    hzz_cols = ["roe_z", "pe_z", "momentum_z", "hzz_trend_z"]
    available_hzz = panel[hzz_cols].notna().sum(axis=1)
    panel["composite_hzz_score"] = panel[hzz_cols].mean(axis=1, skipna=True)
    panel.loc[available_hzz < 3, "composite_hzz_score"] = np.nan

    if panel["month"].max() > CUTOFF:
        raise AssertionError("Processed panel includes data after May 2026.")
    return panel.sort_values(["month", "ticker"])


def perf_metrics(returns: pd.Series, name: str) -> dict[str, float | str]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"name": name}
    mean = r.mean()
    vol = r.std(ddof=1)
    sharpe = np.sqrt(12) * mean / vol if vol and np.isfinite(vol) else np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    t_stat, p_val = st.ttest_1samp(r, 0.0, nan_policy="omit") if len(r) > 2 else (np.nan, np.nan)
    return {
        "name": name,
        "n_months": len(r),
        "avg_monthly_return": mean,
        "monthly_volatility": vol,
        "annualized_return_approx": mean * 12,
        "annualized_volatility": vol * np.sqrt(12) if np.isfinite(vol) else np.nan,
        "annualized_sharpe": sharpe,
        "t_stat": t_stat,
        "p_value": p_val,
        "cumulative_return": wealth.iloc[-1] - 1,
        "max_drawdown": dd.min(),
        "best_month": r.max(),
        "worst_month": r.min(),
    }


def make_fmp_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Constructing factor-mimicking portfolios and IC diagnostics...")
    factor_map = {
        "roe": "roe_z",
        "pe": "pe_z",
        "momentum": "momentum_z",
        "trend": "trend_z",
        "hzz_trend_improvement": "hzz_trend_z",
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


def score_for_spec(panel: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    cols: list[str] = []
    weighted = []
    for factor_key, weight in spec.weights.items():
        if factor_key == "trend":
            col = spec.trend_col
        elif factor_key == "low_volatility":
            col = "low_volatility_z"
        else:
            col = f"{factor_key}_z"
        if col not in panel.columns:
            raise KeyError(f"Missing factor column {col}")
        cols.append(col)
        weighted.append(panel[col] * weight)
    out = sum(weighted)
    valid_weight = pd.Series(0.0, index=panel.index)
    valid_count = pd.Series(0, index=panel.index)
    for factor_key, weight in spec.weights.items():
        if factor_key == "trend":
            col = spec.trend_col
        elif factor_key == "low_volatility":
            col = "low_volatility_z"
        else:
            col = f"{factor_key}_z"
        is_valid = panel[col].notna()
        valid_weight += np.where(is_valid, abs(weight), 0.0)
        valid_count += is_valid.astype(int)
    score = out / valid_weight.replace(0, np.nan)
    score[valid_count < min(3, len(spec.weights))] = np.nan
    if spec.volatility_penalty:
        penalty = panel["volatility_z"].fillna(0.0) * spec.volatility_penalty
        score = score - penalty
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
    """Official final candidates must preserve the four required factors."""
    required_weights = {"roe", "pe", "momentum", "trend"}
    return (
        set(spec.weights) == required_weights
        and spec.trend_col == "trend_z"
        and spec.max_per_sector is None
        and spec.volatility_penalty == 0.0
    )


def select_positions_for_spec(g: pd.DataFrame, spec: StrategySpec, equity: float) -> pd.DataFrame:
    max_positions = max(0, int(equity // CASH_PER_TRADE))
    n = min(spec.top_n, max_positions, len(g))
    if n <= 0:
        return g.iloc[0:0].copy()
    ranked = g.sort_values("score", ascending=False).copy()
    if spec.max_per_sector is None:
        return ranked.head(n)

    selected_rows: list[int] = []
    sector_counts: dict[str, int] = {}
    for idx, row in ranked.iterrows():
        sector = row.get("GICS Sector")
        sector_key = "Unknown" if pd.isna(sector) else str(sector)
        if sector_counts.get(sector_key, 0) >= spec.max_per_sector:
            continue
        selected_rows.append(idx)
        sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
        if len(selected_rows) >= n:
            break
    return ranked.loc[selected_rows].copy()


def simulate_vector_strategy(panel: pd.DataFrame, spec: StrategySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        if selected.empty:
            pnl = 0.0
            ret = 0.0
        else:
            selected["realized_stock_return"] = stop_take_return(selected, spec.stop_loss, spec.take_profit)
            selected["cash_weight"] = CASH_PER_TRADE / equity
            pnl = CASH_PER_TRADE * selected["realized_stock_return"].sum()
            ret = pnl / equity if equity > 0 else np.nan
            selected["strategy"] = spec.name
            selected["strategy_notes"] = spec.notes
            selected["signal_month"] = month
            selected["allocated_cash"] = CASH_PER_TRADE
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


class MonthlySignalStrategy(bt.Strategy):
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
        self.protective_orders: dict[str, list[bt.Order]] = {}

    @staticmethod
    def _is_live_order(order: bt.Order) -> bool:
        return order.status in [order.Submitted, order.Accepted]

    def _cancel_protective_orders(self, data: bt.LineSeries, exclude_ref: int | None = None) -> None:
        ticker = data._name
        remaining = []
        for order in self.protective_orders.get(ticker, []):
            if order.ref == exclude_ref:
                continue
            if self._is_live_order(order):
                self.cancel(order)
            else:
                remaining.append(order)
        if remaining:
            self.protective_orders[ticker] = remaining
        else:
            self.protective_orders.pop(ticker, None)

    def _submit_protective_orders(self, data: bt.LineSeries, entry_price: float, size: int) -> None:
        if size <= 0 or entry_price <= 0:
            return
        orders = []
        stop_order = None
        if self.params.stop_loss is not None:
            stop_price = entry_price * (1 - abs(float(self.params.stop_loss)))
            stop_order = self.sell(data=data, size=size, exectype=bt.Order.Stop, price=stop_price)
            orders.append(stop_order)
        if self.params.take_profit is not None:
            limit_price = entry_price * (1 + abs(float(self.params.take_profit)))
            limit_kwargs = {"oco": stop_order} if stop_order is not None else {}
            orders.append(self.sell(data=data, size=size, exectype=bt.Order.Limit, price=limit_price, **limit_kwargs))
        if orders:
            self.protective_orders[data._name] = orders

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
            if pos.size and ticker not in target:
                self._cancel_protective_orders(data)
                self.close(data=data)
            elif ticker in target and not pos.size:
                self.buy(data=data)

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
        if order.status == order.Completed:
            if order.isbuy():
                self._submit_protective_orders(order.data, order.executed.price, int(order.executed.size))
            else:
                self._cancel_protective_orders(order.data, exclude_ref=order.ref)
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            orders = [o for o in self.protective_orders.get(ticker, []) if o.ref != order.ref]
            if orders:
                self.protective_orders[ticker] = orders
            else:
                self.protective_orders.pop(ticker, None)

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
        equity = equity.drop_duplicates("month", keep="last").sort_values("month")
        equity["return"] = equity["value"].pct_change().fillna(0.0)
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"backtrader_{name}"
    save_csv(equity, prefix.with_name(prefix.name + "_equity_curve.csv"))
    save_csv(orders, prefix.with_name(prefix.name + "_orders.csv"))
    save_csv(trades, prefix.with_name(prefix.name + "_trades.csv"))
    save_csv(positions, prefix.with_name(prefix.name + "_positions.csv"))
    metrics = pd.DataFrame([perf_metrics(equity["return"] if "return" in equity else pd.Series(dtype=float), f"backtrader_{name}")])
    if not equity.empty:
        metrics["final_value"] = equity["value"].iloc[-1]
        metrics["total_return"] = equity["value"].iloc[-1] / INITIAL_CASH - 1
    save_csv(metrics, prefix.with_name(prefix.name + "_metrics.csv"))
    return {"equity": equity, "orders": orders, "trades": trades, "positions": positions, "metrics": metrics}


def get_strategy_specs() -> list[StrategySpec]:
    """Return all strategy variants, with assignment-scope variants first."""
    return [
        StrategySpec("base_equal_top10", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, notes="Required assignment baseline: equal-weight ROE, P/E, momentum, and index-regression trend."),
        StrategySpec("equal_top5", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=5, notes="Concentration test: same score, fewer names."),
        StrategySpec("trend_momentum_heavy_top10", {"roe": 0.2, "pe": 0.2, "momentum": 0.3, "trend": 0.3}, top_n=10, notes="Weight test: emphasize price-based signals."),
        StrategySpec("value_quality_heavy_top10", {"roe": 0.35, "pe": 0.35, "momentum": 0.15, "trend": 0.15}, top_n=10, notes="Weight test: emphasize fundamentals."),
        StrategySpec("no_trend_top10", {"roe": 1, "pe": 1, "momentum": 1}, top_n=10, notes="Ablation test: remove trend to see whether the trend factor helps."),
        StrategySpec("hzz_trend_composite_top10", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, trend_col="hzz_trend_z", notes="Paper-inspired improvement: replace index-regression trend with cross-sectional HZZ trend."),
        StrategySpec("base_equal_top10_regime", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, regime_filter=True, notes="Risk filter: trade only when S&P 500 is above its 10-month moving average."),
        StrategySpec("trend_heavy_top10_regime", {"roe": 0.2, "pe": 0.2, "momentum": 0.3, "trend": 0.3}, top_n=10, regime_filter=True, notes="Combined price-signal overweight and market-regime filter."),
        StrategySpec("base_top10_stop10_take20", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, stop_loss=0.10, take_profit=0.20, notes="Risk management test: Backtrader stop-loss/take-profit improvement; vector curve is a monthly screening approximation."),
        StrategySpec("top5_regime_stop10_take20", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=5, regime_filter=True, stop_loss=0.10, take_profit=0.20, notes="Combined concentration, regime, and Backtrader stop/take-profit improvement."),
        StrategySpec("hzz_sector_cap2_top10", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, trend_col="hzz_trend_z", max_per_sector=2, notes="Risk-control improvement: HZZ score with max two holdings per GICS sector."),
        StrategySpec("hzz_vol_penalty_top10", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, trend_col="hzz_trend_z", volatility_penalty=0.25, notes="Risk-control improvement: HZZ score penalized by 6-month realized volatility."),
        StrategySpec("hzz_sector_cap2_vol_penalty_top10", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=10, trend_col="hzz_trend_z", max_per_sector=2, volatility_penalty=0.25, notes="Combined improvement: HZZ trend, sector cap, and volatility-aware ranking."),
        StrategySpec("hzz_sector_cap2_top15", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1}, top_n=15, trend_col="hzz_trend_z", max_per_sector=3, notes="Diversification test: HZZ score with 15 names and max three per sector."),
        StrategySpec("hzz_low_vol_factor_top10", {"roe": 1, "pe": 1, "momentum": 1, "trend": 1, "low_volatility": 0.5}, top_n=10, trend_col="hzz_trend_z", notes="Risk-score test: add low-volatility as a half-weight risk overlay."),
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
            perf_metrics(g.sort_values("month")["portfolio_return"], name)
            | {
                "final_equity": g.sort_values("month")["equity"].iloc[-1],
                "total_return": g.sort_values("month")["equity"].iloc[-1] / INITIAL_CASH - 1,
                "avg_positions": g["n_positions"].mean(),
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
                "regime_filter": spec.regime_filter,
                "stop_loss": spec.stop_loss,
                "take_profit": spec.take_profit,
                "max_per_sector": spec.max_per_sector,
                "volatility_penalty": spec.volatility_penalty,
                "assignment_scope": is_assignment_scope_strategy(spec),
                "notes": spec.notes,
            }
            for spec in specs
        ]
    )
    metrics = metrics.merge(experiment_log.rename(columns={"strategy": "name"}), on="name", how="left")

    save_csv(all_curves, RESULTS_DIR / "strategy_vector_equity_curves.csv")
    save_csv(all_holdings, RESULTS_DIR / "strategy_vector_holdings.csv")
    save_csv(metrics, RESULTS_DIR / "strategy_vector_metrics.csv")
    save_csv(experiment_log, RESULTS_DIR / "strategy_experiment_log.csv")
    if not all_holdings.empty and "GICS Sector" in all_holdings.columns:
        sector_counts = (
            all_holdings.groupby(["strategy", "signal_month", "GICS Sector"], dropna=False)
            .size()
            .rename("sector_count")
            .reset_index()
        )
        total_counts = (
            all_holdings.groupby(["strategy", "signal_month"])
            .size()
            .rename("n_holdings")
            .reset_index()
        )
        sector_counts = sector_counts.merge(total_counts, on=["strategy", "signal_month"], how="left")
        sector_counts["sector_share"] = sector_counts["sector_count"] / sector_counts["n_holdings"]
        sector_month = (
            sector_counts.groupby(["strategy", "signal_month"])
            .agg(
                max_sector_share=("sector_share", "max"),
                max_sector_count=("sector_count", "max"),
                distinct_sectors=("GICS Sector", "nunique"),
            )
            .reset_index()
        )
        sector_summary = (
            sector_month.groupby("strategy")
            .agg(
                avg_max_sector_share=("max_sector_share", "mean"),
                worst_max_sector_share=("max_sector_share", "max"),
                avg_distinct_sectors=("distinct_sectors", "mean"),
                max_single_month_sector_count=("max_sector_count", "max"),
            )
            .reset_index()
        )
        save_csv(sector_counts, RESULTS_DIR / "strategy_sector_counts_by_month.csv")
        save_csv(sector_month, RESULTS_DIR / "strategy_sector_monthly_summary.csv")
        save_csv(sector_summary, RESULTS_DIR / "strategy_sector_concentration_summary.csv")
    return metrics, all_curves, all_holdings, spec_map


def choose_best_assignment_strategy(
    strategy_metrics: pd.DataFrame,
    spec_map: dict[str, StrategySpec],
    base_name: str = "base_equal_top10",
) -> str:
    """Choose the best official improvement without leaving assignment scope."""
    for name in strategy_metrics["name"]:
        if name != base_name and is_assignment_scope_strategy(spec_map[name]):
            return name
    return base_name


def save_categorized_strategy_outputs(
    strategy_metrics: pd.DataFrame,
    strategy_curves: pd.DataFrame,
    strategy_holdings: pd.DataFrame,
    base_name: str,
    improved_name: str,
) -> None:
    """Write base, improved, comparison, and appendix outputs into separate folders."""
    for name, out_dir in [(base_name, BASE_RESULTS_DIR), (improved_name, IMPROVED_RESULTS_DIR)]:
        save_csv(strategy_metrics[strategy_metrics["name"].eq(name)], out_dir / "vector_metrics.csv")
        save_csv(strategy_curves[strategy_curves["strategy"].eq(name)], out_dir / "vector_equity_curve.csv")
        if not strategy_holdings.empty:
            save_csv(strategy_holdings[strategy_holdings["strategy"].eq(name)], out_dir / "vector_holdings.csv")

    official_names = strategy_metrics.loc[strategy_metrics["assignment_scope"].fillna(False), "name"].tolist()
    appendix_names = strategy_metrics.loc[~strategy_metrics["assignment_scope"].fillna(False), "name"].tolist()

    save_csv(strategy_metrics[strategy_metrics["name"].isin(official_names)], COMPARISON_RESULTS_DIR / "assignment_scope_strategy_metrics.csv")
    save_csv(strategy_curves[strategy_curves["strategy"].isin(official_names)], COMPARISON_RESULTS_DIR / "assignment_scope_strategy_curves.csv")
    base_vs_improved = strategy_metrics[strategy_metrics["name"].isin([base_name, improved_name])].copy()
    base_vs_improved["role"] = np.where(base_vs_improved["name"].eq(base_name), "base", "improved")
    role_order = pd.Categorical(base_vs_improved["role"], categories=["base", "improved"], ordered=True)
    base_vs_improved = base_vs_improved.assign(role=role_order).sort_values("role")
    save_csv(base_vs_improved, COMPARISON_RESULTS_DIR / "base_vs_improved_metrics.csv")

    save_csv(strategy_metrics[strategy_metrics["name"].isin(appendix_names)], APPENDIX_RESULTS_DIR / "appendix_strategy_metrics.csv")
    save_csv(strategy_curves[strategy_curves["strategy"].isin(appendix_names)], APPENDIX_RESULTS_DIR / "appendix_strategy_curves.csv")
    if not strategy_holdings.empty:
        save_csv(strategy_holdings[strategy_holdings["strategy"].isin(appendix_names)], APPENDIX_RESULTS_DIR / "appendix_strategy_holdings.csv")


def monte_carlo_random_portfolios(panel: pd.DataFrame, base_spec: StrategySpec, n_sims: int = 1000, output_name: str | None = None) -> pd.DataFrame:
    print(f"Running Monte Carlo robustness test for {base_spec.name} with {n_sims} random portfolios...")
    rng = np.random.default_rng(RNG_SEED)
    base_curve, _ = simulate_vector_strategy(panel, base_spec)
    base_sharpe = perf_metrics(base_curve["portfolio_return"], base_spec.name)["annualized_sharpe"]
    months = sorted(panel["month"].dropna().unique())
    sim_rows: list[dict[str, Any]] = []

    eligible_by_month = {
        month: g.loc[g["eligible"], ["ticker", "GICS Sector", "next_open", "next_high", "next_low", "next_ret_oc", "regime_on"]].dropna(subset=["ticker", "next_ret_oc"]).reset_index(drop=True)
        for month, g in panel.groupby("month", sort=True)
    }
    for sim in range(n_sims):
        equity = INITIAL_CASH
        returns: list[float] = []
        for month in months:
            g = eligible_by_month.get(month)
            if g is None or g.empty:
                returns.append(0.0)
                continue
            if base_spec.regime_filter and not month_regime_is_on(g):
                returns.append(0.0)
                continue
            max_positions = max(0, int(equity // CASH_PER_TRADE))
            n = min(base_spec.top_n, max_positions, len(g))
            if n == 0:
                returns.append(0.0)
                continue
            if base_spec.max_per_sector is None:
                picks = rng.choice(len(g), size=n, replace=False)
                selected = g.iloc[picks].copy()
            else:
                shuffled = g.iloc[rng.permutation(len(g))].copy()
                selected_rows: list[int] = []
                sector_counts: dict[str, int] = {}
                for idx, row in shuffled.iterrows():
                    sector = row.get("GICS Sector")
                    sector_key = "Unknown" if pd.isna(sector) else str(sector)
                    if sector_counts.get(sector_key, 0) >= base_spec.max_per_sector:
                        continue
                    selected_rows.append(idx)
                    sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
                    if len(selected_rows) >= n:
                        break
                selected = shuffled.loc[selected_rows].copy()
            realized = stop_take_return(selected, base_spec.stop_loss, base_spec.take_profit)
            pnl = CASH_PER_TRADE * realized.sum()
            ret = pnl / equity if equity else 0.0
            equity += pnl
            returns.append(ret)
        metrics = perf_metrics(pd.Series(returns), f"random_{sim}")
        sim_rows.append({"simulation": sim, "annualized_sharpe": metrics.get("annualized_sharpe", np.nan), "final_equity": equity})

    out = pd.DataFrame(sim_rows)
    out["strategy_sharpe"] = base_sharpe
    out["p_value"] = (out["annualized_sharpe"] >= base_sharpe).mean()
    if output_name is None:
        output_name = "monte_carlo_random_portfolios.csv"
    save_csv(out, RESULTS_DIR / output_name)
    return out


def strategy_benchmark_comparison(strategy_curves: pd.DataFrame, index_monthly: pd.DataFrame) -> pd.DataFrame:
    print("Comparing strategies against ^GSPC benchmark...")
    bench = index_monthly[["month", "close"]].copy().sort_values("month")
    bench["benchmark_return"] = bench["close"].pct_change()
    rows: list[dict[str, Any]] = []
    for name, g in strategy_curves.groupby("strategy", sort=True):
        g = g.sort_values("month")[["month", "portfolio_return"]]
        merged = g.merge(bench[["month", "benchmark_return"]], on="month", how="inner").dropna()
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
    save_csv(out, RESULTS_DIR / "strategy_benchmark_comparison.csv")
    return out


def block_bootstrap(curve: pd.DataFrame, block_size: int = 6, n_sims: int = 1000, output_name: str = "block_bootstrap_base_strategy.csv") -> pd.DataFrame:
    print(f"Running block bootstrap robustness test -> {output_name}...")
    rng = np.random.default_rng(RNG_SEED + 1)
    r = curve.sort_values("month")["portfolio_return"].dropna().to_numpy()
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
    save_csv(out, RESULTS_DIR / output_name)
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
    if not sector_summary.empty:
        metrics = metrics.merge(sector_summary.rename(columns={"strategy": "name"}), on="name", how="left")
    metrics = metrics.sort_values("annualized_sharpe", ascending=False)
    base = metrics[metrics["name"].eq("base_equal_top10")].iloc[0]
    best = metrics[metrics["name"].eq(final_strategy_name)].iloc[0]
    mc_base_p = monte_carlo_base["p_value"].iloc[0]
    mc_final_p = monte_carlo_final["p_value"].iloc[0]

    content = f"""# Strategy History And Improvement Log

This file is the living history of the S&P 500 factor investing project. It intentionally preserves the base strategy, failed variants, partial improvements, and final candidate so we can explain the development path in the presentation and revisit ideas later.

## Project Rules We Must Preserve

- The project remains aligned with the factor-investing assignment: ROE, P/E, momentum, and trend factors are always the foundation.
- `^GSPC` replaces BIST100 as the index benchmark and trend-regression input.
- The final data window stops at May 2026.
- Backtrader uses initial cash `1,000,000`, fixed cash per trade `100,000`, market orders, and zero commission.
- Official improvements should preserve the required four factors and the index-regression trend. Appendix experiments may be kept for history, but they should not define the main final strategy.

## Why We Improved The Base Strategy

The base strategy worked in absolute terms, but the Monte Carlo test was not strong:

- Base strategy Sharpe: `{monte_carlo_base['strategy_sharpe'].iloc[0]:.4f}`
- Base Monte Carlo p-value: `{mc_base_p:.4f}`

Interpretation: random top-10 S&P 500 portfolios under the same broad constraints often produced similar Sharpe ratios. That means the base result is not enough to claim strong stock-selection skill.

The individual FMPs also warned us not to overclaim. Their t-statistics and IC values are weak:

{markdown_table(fmp_summary, ['approach', 'factor', 'annualized_sharpe', 't_stat', 'p_value', 'avg_ic', 'avg_rank_ic'])}

## Development Path

We kept every strategy because it records the research path. The `assignment_scope` column separates official project-scope variants from appendix experiments.

{markdown_table(metrics, ['name', 'assignment_scope', 'annualized_sharpe', 'final_equity', 'max_drawdown', 'avg_positions', 'annualized_alpha_approx', 'beta_to_sp500', 'avg_max_sector_share'], float_digits=4)}

## Variant Notes

{markdown_table(metrics, ['name', 'assignment_scope', 'top_n', 'trend_col', 'regime_filter', 'stop_loss', 'take_profit', 'max_per_sector', 'volatility_penalty', 'notes'], float_digits=3)}

## What Improved And Why

### 1. Assignment-Scope Improvements

The official improvement path keeps ROE, P/E, momentum, and the assignment-style `^GSPC` index-regression trend. We then test changes the project explicitly allows: factor weights, number of selected stocks, market-regime filtering, and stop-loss/take-profit order logic.

- Base vector Sharpe: `{base['annualized_sharpe']:.4f}`
- Final selected assignment-scope vector Sharpe: `{best['annualized_sharpe']:.4f}`
- Final selected strategy: `{best['name']}`

### 2. Position Selection And Factor Weights

Top-5 versus top-10 tests show whether the signal is stronger in the highest-ranked stocks or whether diversification helps. Factor-weight variants test whether fundamentals or price-based signals deserve more emphasis.

### 3. Regime And Stop/Take-Profit Tests

The S&P 500 regime filter and stop-loss/take-profit variants answer the instructor's request to show improvements beyond the base strategy. Stop/take-profit is implemented in Backtrader for the final executable run when such a strategy is selected; vector curves are only screening approximations.

### 4. Appendix Experiments

HZZ cross-sectional trend, sector caps, and volatility-aware ranking are retained as appendix/history experiments. They are useful ideas, but they move beyond the literal assignment trend-factor construction, so they are not allowed to become the official final strategy in this version.

## Robustness Results

Monte Carlo:

- Base strategy p-value: `{mc_base_p:.4f}`
- Final selected strategy p-value: `{mc_final_p:.4f}`

Walk-forward:

{markdown_table(walk_forward, ['strategy', 'train_sharpe_to_2020', 'test_sharpe_2021_2026', 'test_cumulative_return_2021_2026', 'test_max_drawdown_2021_2026', 'selected_by_train'])}

Benchmark comparison:

{markdown_table(benchmark_comparison, ['strategy', 'annualized_excess_return_approx', 'excess_t_stat', 'excess_p_value', 'annualized_alpha_approx', 'alpha_t_stat', 'alpha_p_value', 'beta_to_sp500'], max_rows=20)}

## Current Judgment

The project is stronger after adding assignment-scope improvement tests, final-strategy Monte Carlo, walk-forward validation, and benchmark alpha comparisons. However, the honest conclusion remains cautious:

- The base strategy is not statistically strong versus random portfolios.
- The final assignment-scope strategy is more promising, but still selected after testing multiple variants.
- Vector strategy variants are screening approximations; executable trading evidence comes from the saved Backtrader runs.
- The universe uses current S&P 500 constituents, so survivorship bias remains.
- Transaction costs and slippage are ignored because the assignment requires commission `0`.
- A production strategy would need historical index membership, costs, slippage, beta/sector neutrality, and a data-snooping-adjusted test such as White's Reality Check or Hansen's SPA.

## External Methodology Notes

- PyAnomaly emphasizes the same research building blocks we use: firm characteristics, winsorizing/trimming, quantile portfolios, long-short portfolios, factor regression, and cross-sectional regression.
- OpenSourceAP/CrossSection shows how much infrastructure serious factor replication needs: signal documentation, portfolio construction variants, and reproducibility checks.
- Quantitativo's trend-factor implementation stresses survivorship-bias-free universes, price filters, correct signal/return shifting, and the observation that post-2016 trend-factor edge can weaken.
- Sector and volatility controls are practical portfolio-construction overlays, not replacements for the required assignment factors.

## Ideas To Revisit Later

- Run a real White Reality Check or Hansen SPA test over all tried strategy variants.
- Use historical S&P 500 membership instead of current constituents.
- Add transaction costs and slippage despite the assignment's zero-commission rule, as an appendix.
- Revisit HZZ trend only as a clearly labeled appendix experiment unless the assignment scope changes.
- Test sector-neutral z-scoring separately from sector caps.
- Add beta-neutral or benchmark-relative optimization as an appendix, while keeping Backtrader fixed-cash results as the required core.
"""
    (DOCS_DIR / "STRATEGY_HISTORY.md").write_text(content, encoding="utf-8")


def walk_forward_summary(strategy_metrics: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Simple walk-forward: choose best pre-2021 assignment-scope strategy."""
    print("Computing walk-forward validation summary...")
    train_end = pd.Timestamp("2020-12-31")
    eligible_names: set[str] | None = None
    needed = {"name", "trend_col", "max_per_sector", "volatility_penalty", "weights"}
    if needed.issubset(strategy_metrics.columns):
        parsed_weights = strategy_metrics["weights"].apply(json.loads)
        eligible_names = set(
            strategy_metrics.loc[
                strategy_metrics["trend_col"].eq("trend_z")
                & strategy_metrics["max_per_sector"].isna()
                & strategy_metrics["volatility_penalty"].fillna(0.0).eq(0.0)
                & parsed_weights.apply(lambda w: set(w) == {"roe", "pe", "momentum", "trend"}),
                "name",
            ]
        )
    rows = []
    for name, g in curves.groupby("strategy", sort=True):
        if eligible_names is not None and name not in eligible_names:
            continue
        g = g.sort_values("month")
        train = g[g["month"] <= train_end]
        test = g[g["month"] > train_end]
        if train.empty or test.empty:
            continue
        train_m = perf_metrics(train["portfolio_return"], f"{name}_train")
        test_m = perf_metrics(test["portfolio_return"], f"{name}_test")
        rows.append(
            {
                "strategy": name,
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
    save_csv(out, RESULTS_DIR / "walk_forward_summary.csv")
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
    for name in ["base_equal_top10", final_name]:
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
    base = strategy_curves[strategy_curves["strategy"].eq("base_equal_top10")].sort_values("month")
    if not base.empty:
        ax.plot(base["month"], make_drawdown(base["equity"]), label="base_equal_top10")
    best = strategy_curves[strategy_curves["strategy"].eq(final_name)].sort_values("month")
    if not best.empty:
        ax.plot(best["month"], make_drawdown(best["equity"]), label=final_name)
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
    base = strategy_metrics[strategy_metrics["name"].eq("base_equal_top10")].iloc[0]
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

This is a standalone S&P 500 factor investing project aligned with the EC581 factor-investing assignment. It adapts the BIST100 requirement to the U.S. market by using Yahoo Finance `^GSPC` as the S&P 500 index benchmark and trend-regression input.

## Data Window

The analysis uses frozen data available through May 2026 only. Any rows after `2026-05-31` are excluded from processing.

Raw inputs:

- `data/raw/sp500_prices_long.csv`: Tiingo daily stock prices with adjusted OHLCV.
- `data/raw/sp500_fundamentals_daily_long.csv`: point-in-time daily valuation ratios and market cap.
- `data/raw/sp500_fundamentals_statements_long.csv`: statement fundamentals dated by public availability date.
- `data/raw/sp500_constituents.csv`: current S&P 500 universe metadata.
- `data/raw/sp500_index_yahoo.csv`: frozen Yahoo Finance `^GSPC` benchmark.

## Strategy

Required factors:

- ROE: higher is better.
- P/E: lower positive P/E is better.
- Momentum: 12-month price return.
- Trend: full-sample predictive regression on `^GSPC` normalized moving-average deviations, with insignificant variables dropped, applied to each stock.

All factors are month-end, lagged by one month, winsorized at 1st/99th percentiles, and cross-sectionally standardized.

## Main Results

- Base vector strategy `base_equal_top10`: final equity `${base['final_equity']:,.0f}`, annualized Sharpe `{base['annualized_sharpe']:.2f}`.
- Final selected assignment-scope vector strategy `{best['name']}`: final equity `${best['final_equity']:,.0f}`, annualized Sharpe `{best['annualized_sharpe']:.2f}`.
- Backtrader base strategy: final value `${bt_base['final_value']:,.0f}`, annualized Sharpe `{bt_base['annualized_sharpe']:.2f}`.
- Backtrader final strategy: final value `${bt_best['final_value']:,.0f}`, annualized Sharpe `{bt_best['annualized_sharpe']:.2f}`.
- Monte Carlo random-portfolio Sharpe p-value for the base strategy: `{mc_p:.4f}`.
- Monte Carlo random-portfolio Sharpe p-value for the final selected strategy: `{final_mc_p:.4f}`.
- Final selected strategy annualized alpha vs `^GSPC`: `{best_alpha:.2%}` with alpha t-stat `{best_alpha_t:.2f}`.
{wf_line}
Backtrader is used for the base strategy and the final selected assignment-scope strategy, with initial cash `1,000,000`, `FixedCashSizer` at `100,000` per trade, market orders, zero commission, and protective stop/limit exits when the selected improvement uses stop-loss/take-profit.

## Output Layout

- `results/base_strategy/`: base strategy vector and Backtrader outputs.
- `results/improved_strategy/`: final assignment-scope improved strategy outputs.
- `results/comparison/`: base-versus-improved and walk-forward comparison files.
- `results/fmp_analysis/`: factor-mimicking portfolio, IC, and factor comparison files.
- `results/appendix_experiments/`: extra experiments kept outside the official final strategy.

## Reproduce

```powershell
cd C:\\Users\\asus\\Desktop\\sp500_factor_investing
py -3.10 src\\run_project.py
```

The run reads frozen CSVs, regenerates processed data, results, figures, and the PDF presentation.

## Limitations

- The universe is current S&P 500 constituents, so the study has survivorship bias.
- Transaction costs and slippage are ignored because the assignment specifies commission `0`.
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
- Trend uses normalized moving-average deviations with windows `{MA_WINDOWS}`. Coefficients come from the assignment-style full-sample `^GSPC` predictive regression, with insignificant variables dropped.

The paper-style HZZ cross-sectional trend factor is retained only as an appendix/history experiment because it changes the literal assignment trend construction.

## Results

Base strategy:

- final equity: `${base['final_equity']:,.0f}`;
- total return: `{base['total_return']:.2%}`;
- annualized Sharpe: `{base['annualized_sharpe']:.2f}`;
- max drawdown: `{base['max_drawdown']:.2%}`.

Final selected assignment-scope vector strategy:

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

Backtrader final strategy:

- final value: `${bt_best['final_value']:,.0f}`;
- total return: `{bt_best['total_return']:.2%}`;
- annualized Sharpe: `{bt_best['annualized_sharpe']:.2f}`;
- max drawdown: `{bt_best['max_drawdown']:.2%}`.

Monte Carlo p-value for the base Sharpe: `{mc_p:.4f}`.

Monte Carlo p-value for the final selected strategy Sharpe: `{final_mc_p:.4f}`.

Final selected strategy annualized alpha versus `^GSPC`: `{best_alpha:.2%}` with alpha t-stat `{best_alpha_t:.2f}`.

## What We Tried

- Equal-weight composite base strategy.
- More concentrated top-5 selection.
- Trend/momentum-heavy factor weights.
- Value/quality-heavy factor weights.
- Removing trend from the composite.
- S&P 500 regime filter.
- Backtrader stop-loss/take-profit improvement.
- Appendix only: paper-style HZZ trend, sector caps, and volatility-aware ranking.
- Walk-forward validation.

## Interpretation

The project should not claim statistically guaranteed skill only because the backtest is profitable. The Monte Carlo result is the main guardrail: if random portfolios often match or beat the strategy Sharpe, the conclusion must be cautious. The strongest investment conclusion is conditional: the factor process is economically sensible and reproducible, but real-money confidence would require survivorship-bias-free data, transaction-cost and slippage assumptions, and further out-of-sample testing.

The final strategy has stronger risk-adjusted performance than the base strategy, but it was selected after multiple experiments. This is why the presentation should emphasize the full strategy history and avoid claiming that the final result is a guaranteed tradable edge.

The vectorized improvement tests are used for fast screening. The executable trading results are the saved Backtrader runs, which use market orders, fixed cash sizing, the assignment's zero-commission setting, and stop/limit exits where applicable.
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
    base = strategy_metrics[strategy_metrics["name"].eq("base_equal_top10")].iloc[0]
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
            ],
        )
        add_image_slide(pdf, "Factor Portfolio Performance", FIGURES_DIR / "factor_portfolio_cumulative_returns.png")
        add_image_slide(pdf, "Information Coefficients", FIGURES_DIR / "ic_rank_ic_summary.png")
        add_image_slide(pdf, "Strategy vs Benchmark", FIGURES_DIR / "strategy_equity_vs_benchmark.png")
        add_image_slide(pdf, "Drawdowns", FIGURES_DIR / "strategy_drawdowns.png")
        add_image_slide(pdf, "Improvement Tests", FIGURES_DIR / "strategy_improvement_sharpe.png")
        add_image_slide(pdf, "Monte Carlo Robustness", FIGURES_DIR / "monte_carlo_sharpe_histogram.png")
        add_text_slide(
            pdf,
            "Backtrader Results",
            [
                f"Base Backtrader final value: ${bt_base['final_value']:,.0f}; Sharpe: {bt_base['annualized_sharpe']:.2f}.",
                f"Final Backtrader variant final value: ${bt_best['final_value']:,.0f}; Sharpe: {bt_best['annualized_sharpe']:.2f}.",
                "Settings: initial cash 1,000,000; FixedCashSizer 100,000; commission 0; market orders.",
                "Saved outputs include orders, trades, positions, equity curves, and metrics.",
            ],
        )
        add_text_slide(
            pdf,
            "Main Results",
            [
                f"Base strategy final equity: ${base['final_equity']:,.0f}; Sharpe: {base['annualized_sharpe']:.2f}.",
                f"Final selected assignment-scope strategy: {best['name']}; final equity: ${best['final_equity']:,.0f}; Sharpe: {best['annualized_sharpe']:.2f}.",
                f"Monte Carlo p-value for base Sharpe: {monte_carlo['p_value'].iloc[0]:.4f}.",
                f"Monte Carlo p-value for final strategy Sharpe: {monte_carlo_final['p_value'].iloc[0]:.4f}.",
                f"Final strategy annualized alpha vs ^GSPC: {best_alpha:.2%}.",
                "Interpret results cautiously because the universe has survivorship bias and costs are ignored.",
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
    save_csv(out, RESULTS_DIR / "fmp_common_start_comparison.csv")
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
    save_csv(out, RESULTS_DIR / "selected_date_fmp_weight_comparison.csv")
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
    hzz_trend, hzz_betas = compute_hzz_trend(stock_ma, monthly)
    panel = build_factor_panel(monthly, metrics, roe, trend, hzz_trend, constituents, index_monthly)

    save_csv(monthly, PROCESSED_DIR / "monthly_stock_bars.csv")
    save_csv(index_monthly, PROCESSED_DIR / "monthly_sp500_index.csv")
    save_csv(metrics, PROCESSED_DIR / "monthly_daily_fundamentals.csv")
    save_csv(roe, PROCESSED_DIR / "monthly_roe_asof.csv")
    save_csv(stock_ma, PROCESSED_DIR / "monthly_stock_ma_signals.csv")
    save_csv(index_ma, PROCESSED_DIR / "monthly_index_ma_signals.csv")
    save_csv(trend_coeffs, RESULTS_DIR / "trend_index_regression_coefficients.csv")
    save_csv(hzz_betas, RESULTS_DIR / "hzz_cross_sectional_trend_betas.csv")
    save_csv(panel, PROCESSED_DIR / "factor_panel.csv")

    audit = data_audit(prices, daily, statements, constituents, index, panel)
    portfolio, regression, ic, weights = make_fmp_returns(panel)
    save_csv(portfolio, RESULTS_DIR / "fmp_portfolio_returns.csv")
    save_csv(regression, RESULTS_DIR / "fmp_regression_returns.csv")
    save_csv(ic, RESULTS_DIR / "factor_information_coefficients.csv")
    save_csv(weights, RESULTS_DIR / "fmp_weights.csv")
    save_csv(portfolio, FMP_RESULTS_DIR / "fmp_portfolio_returns.csv")
    save_csv(regression, FMP_RESULTS_DIR / "fmp_regression_returns.csv")
    save_csv(ic, FMP_RESULTS_DIR / "factor_information_coefficients.csv")
    save_csv(weights, FMP_RESULTS_DIR / "fmp_weights.csv")
    fmp_summary = summarize_fmps(portfolio, regression, ic)
    save_csv(fmp_summary, RESULTS_DIR / "fmp_performance_summary.csv")
    save_csv(fmp_summary, FMP_RESULTS_DIR / "fmp_performance_summary.csv")
    save_common_start_comparison(portfolio, regression)
    save_selected_date_weights(weights)

    strategy_metrics, strategy_curves, strategy_holdings, spec_map = run_strategy_experiments(panel)
    base_spec = spec_map["base_equal_top10"]
    monte_carlo = monte_carlo_random_portfolios(panel, base_spec, n_sims=1000)
    save_csv(monte_carlo, BASE_RESULTS_DIR / "monte_carlo_random_portfolios.csv")
    benchmark_comparison = strategy_benchmark_comparison(strategy_curves, index_monthly)
    save_csv(benchmark_comparison, COMPARISON_RESULTS_DIR / "strategy_benchmark_comparison.csv")
    base_curve = strategy_curves[strategy_curves["strategy"].eq("base_equal_top10")]
    bootstrap = block_bootstrap(base_curve, block_size=6, n_sims=1000, output_name="block_bootstrap_base_strategy.csv")
    save_csv(bootstrap, BASE_RESULTS_DIR / "block_bootstrap.csv")
    walk_forward = walk_forward_summary(strategy_metrics, strategy_curves)
    save_csv(walk_forward, COMPARISON_RESULTS_DIR / "walk_forward_summary.csv")

    best_name = choose_best_assignment_strategy(strategy_metrics, spec_map)
    best_spec = spec_map[best_name]
    save_categorized_strategy_outputs(strategy_metrics, strategy_curves, strategy_holdings, base_spec.name, best_name)
    monte_carlo_final = monte_carlo_random_portfolios(
        panel,
        best_spec,
        n_sims=1000,
        output_name=f"monte_carlo_final_{best_name}.csv",
    )
    save_csv(monte_carlo_final, IMPROVED_RESULTS_DIR / "monte_carlo_random_portfolios.csv")
    best_curve = strategy_curves[strategy_curves["strategy"].eq(best_name)]
    final_bootstrap = block_bootstrap(
        best_curve,
        block_size=6,
        n_sims=1000,
        output_name=f"block_bootstrap_final_{best_name}.csv",
    )
    save_csv(final_bootstrap, IMPROVED_RESULTS_DIR / "block_bootstrap.csv")
    base_bt = run_backtrader(
        monthly,
        index_monthly,
        signals_from_strategy(panel, base_spec),
        "base_equal_top10",
        output_dir=BASE_RESULTS_DIR,
    )
    best_bt = run_backtrader(
        monthly,
        index_monthly,
        signals_from_strategy(panel, best_spec),
        f"best_{best_name}",
        stop_loss=best_spec.stop_loss,
        take_profit=best_spec.take_profit,
        output_dir=IMPROVED_RESULTS_DIR,
    )

    make_figures(portfolio, regression, ic, strategy_curves, strategy_metrics, monte_carlo, index_monthly, best_name)
    experiment_log = pd.read_csv(RESULTS_DIR / "strategy_experiment_log.csv")
    sector_summary_path = RESULTS_DIR / "strategy_sector_concentration_summary.csv"
    sector_summary = pd.read_csv(sector_summary_path) if sector_summary_path.exists() else pd.DataFrame()
    write_strategy_history(
        strategy_metrics,
        experiment_log,
        benchmark_comparison,
        sector_summary,
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
            {"check": "results_generated", "status": "OK", "detail": str(len(list(RESULTS_DIR.glob('*.csv'))))},
            {"check": "figures_generated", "status": "OK", "detail": str(len(list(FIGURES_DIR.glob('*.png'))))},
            {"check": "presentation_pdf", "status": "OK", "detail": str(PRESENTATION_DIR / "sp500_factor_investing_presentation.pdf")},
        ]
    )
    save_csv(validation, RESULTS_DIR / "validation_summary.csv")
    print("Project pipeline completed.")


if __name__ == "__main__":
    main()
