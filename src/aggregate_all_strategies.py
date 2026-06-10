"""Cross-strategy aggregator.

Reads per-strategy ``vector_equity_curve.csv`` files from the staged ladder
(base + improved 1-3) and from the focused improveds (4, 5, 6, 8), then
produces unified comparison tables covering ALL strategies on the common
evaluation window:

- ``results/comparison/all_strategies_walk_forward.csv`` -- train (<= 2020-12)
  vs test (2021+) Sharpe per strategy
- ``results/comparison/all_strategies_benchmark.csv`` -- alpha / beta vs ^GSPC
  per strategy
- ``results/comparison/all_strategies_return_correlation.csv`` -- pairwise
  correlation of monthly returns across all strategies
- ``results/comparison/all_strategies_monte_carlo.csv`` -- per-strategy Monte
  Carlo p-value aggregated from the existing
  ``results/<strategy>/monte_carlo_random_portfolios.csv`` files
- ``results/comparison/all_strategies_metrics.csv`` -- per-strategy Sharpe,
  max DD, final equity, total return on the evaluation window

This script must be run AFTER:
1. ``run_project.py`` (produces base + improved 1-3 outputs)
2. ``run_improved_4_stop_take_sensitivity.py``
3. ``run_improved_5_regime_filter.py``
4. ``run_improved_6_hzz_trend.py``
5. ``run_improved_7_costs.py`` (not aggregated -- it's a cost-sensitivity layer)
6. ``run_improved_8_top_n_sizing.py``

Improved 7 is NOT included in the cross-strategy aggregation because it is a
cost-sensitivity ANALYSIS of improveds 4 and 6, not a stand-alone strategy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import project_core as core


# Mapping: strategy label -> per-strategy results folder
STRATEGY_FOLDERS: dict[str, Path] = {
    "base": core.BASE_RESULTS_DIR,
    "improved_1": core.IMPROVED_RESULTS_DIR,
    "improved_2": core.IMPROVED_2_RESULTS_DIR,
    "improved_3": core.IMPROVED_3_RESULTS_DIR,
    "improved_4": core.IMPROVED_4_RESULTS_DIR,
    "improved_5": core.IMPROVED_5_RESULTS_DIR,
    "improved_6": core.IMPROVED_6_RESULTS_DIR,
    "improved_8": core.IMPROVED_8_RESULTS_DIR,
}

# Mapping: strategy label -> the per-strategy run NAME the run scripts wrote
STRATEGY_NAMES: dict[str, str] = {
    "base": core.BASE_STRATEGY_NAME,
    "improved_1": core.IMPROVED_STRATEGY_NAME,
    "improved_2": core.IMPROVED_2_STRATEGY_NAME,
    "improved_3": core.IMPROVED_3_STRATEGY_NAME,
    "improved_4": core.IMPROVED_4_STRATEGY_NAME,
    "improved_5": core.IMPROVED_5_STRATEGY_NAME,
    "improved_6": core.IMPROVED_6_STRATEGY_NAME,
    "improved_8": core.IMPROVED_8_STRATEGY_NAME,
}


def load_strategy_curve(label: str) -> pd.DataFrame | None:
    """Load a strategy's vector equity curve. Returns None if not found."""
    folder = STRATEGY_FOLDERS[label]
    path = folder / "vector_equity_curve.csv"
    if not path.exists():
        print(f"WARNING: {path} not found; skipping {label}")
        return None
    df = pd.read_csv(path, parse_dates=["month"])
    # Ensure strategy column matches expected label (curves from focused
    # scripts often have the long strategy_name; we standardize to short label
    # for the aggregated tables).
    df["strategy"] = label
    return df


def build_aggregated_curves() -> pd.DataFrame:
    """Concatenate per-strategy curves into one tall dataframe."""
    parts = []
    for label in STRATEGY_FOLDERS:
        df = load_strategy_curve(label)
        if df is not None:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def aggregate_monte_carlo_pvalues() -> pd.DataFrame:
    """Read each strategy's monte_carlo_random_portfolios.csv and aggregate p-values."""
    rows = []
    for label, folder in STRATEGY_FOLDERS.items():
        path = folder / "monte_carlo_random_portfolios.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = {
            "strategy": label,
            "strategy_name": STRATEGY_NAMES[label],
            "strategy_sharpe": float(df["strategy_sharpe"].iloc[0]),
            "p_value": float(df["p_value"].iloc[0]),
            "n_sims": int(len(df)),
            "random_max_sharpe": float(df["annualized_sharpe"].max()),
            "random_median_sharpe": float(df["annualized_sharpe"].median()),
            "random_min_sharpe": float(df["annualized_sharpe"].min()),
        }
        if "evaluation_start" in df.columns:
            row["evaluation_start"] = df["evaluation_start"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def build_return_correlation_matrix(all_curves: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-strategy monthly returns to a wide matrix and compute correlation."""
    eval_curves = core.filter_to_evaluation_window(all_curves, "month")
    if eval_curves.empty:
        return pd.DataFrame()
    wide = eval_curves.pivot_table(
        index="month", columns="strategy", values="portfolio_return", aggfunc="mean"
    )
    # Drop strategies with all-NaN returns in the eval window (e.g., if a script never ran)
    wide = wide.dropna(axis=1, how="all")
    return wide.corr()


def build_per_strategy_metrics(all_curves: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy Sharpe / DD / final equity / total return on eval window."""
    rows = []
    for label, g in all_curves.groupby("strategy", sort=True):
        g = g.sort_values("month")
        metrics = core.metrics_over_evaluation_window(
            g, label, date_col="month", return_col="portfolio_return"
        )
        eval_g = core.filter_to_evaluation_window(g, "month")
        rows.append(
            {
                "strategy": label,
                "strategy_name": STRATEGY_NAMES.get(label, label),
                "annualized_sharpe": metrics.get("annualized_sharpe", np.nan),
                "annualized_return_approx": metrics.get("annualized_return_approx", np.nan),
                "annualized_volatility": metrics.get("annualized_volatility", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                "cumulative_return": metrics.get("cumulative_return", np.nan),
                "final_equity": metrics.get("final_equity", np.nan),
                "total_return": metrics.get("total_return", np.nan),
                "n_evaluation_periods": metrics.get("n_evaluation_periods", np.nan),
                "avg_positions": float(eval_g["n_positions"].mean()) if not eval_g.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("annualized_sharpe", ascending=False)


def main() -> None:
    core.ensure_dirs()
    print("Aggregating cross-strategy comparison tables...")

    all_curves = build_aggregated_curves()
    if all_curves.empty:
        print("No strategy curves found. Run the focused scripts first.")
        return

    # Load index_monthly for benchmark comparison
    _, index_monthly, _ = core.load_processed_strategy_inputs()

    # Per-strategy metrics on the eval window
    metrics_df = build_per_strategy_metrics(all_curves)
    core.save_csv(metrics_df, core.COMPARISON_RESULTS_DIR / "all_strategies_metrics.csv")

    # Walk-forward summary across all strategies
    wf = core.walk_forward_summary(metrics_df, all_curves)
    # walk_forward_summary auto-saves to comparison dir; we also save a clearly-named copy
    core.save_csv(wf, core.COMPARISON_RESULTS_DIR / "all_strategies_walk_forward.csv")

    # Benchmark comparison (alpha / beta vs ^GSPC) across all strategies
    bench = core.strategy_benchmark_comparison(all_curves, index_monthly)
    core.save_csv(bench, core.COMPARISON_RESULTS_DIR / "all_strategies_benchmark.csv")

    # Cross-strategy return correlation matrix
    correlation = build_return_correlation_matrix(all_curves)
    core.save_csv(
        correlation.reset_index().rename(columns={"index": "strategy"}),
        core.COMPARISON_RESULTS_DIR / "all_strategies_return_correlation.csv",
    )

    # Aggregated Monte Carlo p-values
    mc = aggregate_monte_carlo_pvalues()
    core.save_csv(mc, core.COMPARISON_RESULTS_DIR / "all_strategies_monte_carlo.csv")

    print("Aggregator complete.")
    print()
    print("=== Per-strategy metrics (eval window) ===")
    print(
        metrics_df[
            ["strategy", "annualized_sharpe", "max_drawdown", "final_equity", "avg_positions"]
        ].to_string(index=False)
    )
    print()
    print("=== Walk-forward (eval window split at 2020-12) ===")
    if not wf.empty:
        keep = ["strategy", "train_sharpe_to_2020", "test_sharpe_2021_2026", "selected_by_train"]
        keep_present = [c for c in keep if c in wf.columns]
        print(wf[keep_present].to_string(index=False))
    print()
    print("=== Monte Carlo p-values ===")
    if not mc.empty:
        print(mc[["strategy", "strategy_sharpe", "p_value", "n_sims"]].to_string(index=False))


if __name__ == "__main__":
    main()
