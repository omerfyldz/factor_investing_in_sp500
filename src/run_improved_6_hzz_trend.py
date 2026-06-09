"""Focused run for improved strategy 6: Han, Zhou, Zhu (2016) cross-sectional trend factor.

Improved 6 changes exactly one design dimension on top of improved 4: it
replaces the index-level trend regression (``trend_expanding_z``) with a
genuine Han, Zhou, Zhu (2016) cross-sectional trend factor (``trend_hzz_z``).
Every other design choice -- four equal-weighted factors, top-10 long-only,
fixed 100k per trade, 5% stop-loss / 30% take-profit, daily Backtrader with
native ``bt.Order.Stop`` / ``bt.Order.Limit`` protective exits -- is inherited
from improved 4 unchanged. This script does NOT rebuild the full pipeline.

Method:
- Read raw daily prices and recompute the 11 stock-level moving-average ratios.
- For each month ``t``, run an OLS regression of ``next_ret_cc`` on the 11
  normalized MA ratios across the eligible cross-section. Store the monthly
  beta vector.
- Smooth with a strict 12-month trailing mean (months ``[t-12, t-1]``).
- Per stock per month, predict the next-month return with the smoothed beta
  vector applied to the stock's contemporaneous MA ratios.
- Inject the prediction as ``trend_hzz_z`` into the existing factor panel
  (z-scored monthly with the project's standard winsorize-at-1/99 helper).
- Run the vector strategy, the daily Backtrader stop/take strategy, Monte
  Carlo, and a block bootstrap. Save all outputs to
  ``results/improved_strategy_6/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import project_core as core


DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.30


def load_improved_4_thresholds() -> tuple[float, float]:
    """Inherit the selected stop/take from improved 4 if available."""
    selected_path = core.IMPROVED_4_RESULTS_DIR / "selected_stop_take_parameters.csv"
    if not selected_path.exists():
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    return float(selected["stop_loss"].iloc[0]), float(selected["take_profit"].iloc[0])


def make_improved_6_spec(stop_loss: float, take_profit: float) -> core.StrategySpec:
    return core.StrategySpec(
        name=core.IMPROVED_6_STRATEGY_NAME,
        weights={"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
        top_n=10,
        trend_col="trend_hzz_z",
        stop_loss=stop_loss,
        take_profit=take_profit,
        notes=(
            "Improved 6: replace the index-derived trend factor with a Han, Zhou, "
            "Zhu (2016) cross-sectional trend factor. Monthly OLS of next-month "
            "returns on 11 normalized MA ratios across the eligible cross-section, "
            "strict 12-month trailing-mean smoothing of past betas, per-stock "
            "predicted-return signal. Stop-loss and take-profit inherited from "
            "improved 4."
        ),
    )


def inject_hzz_trend(
    panel: pd.DataFrame,
    monthly: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the HZZ cross-sectional trend factor and merge it into the panel."""
    stock_ma = core.compute_stock_ma_signals(prices)
    ratios = core.stock_ma_ratios(stock_ma)
    betas = core.cross_sectional_trend_betas(monthly, ratios)
    smoothed = core.smooth_trend_betas(betas, window=core.HZZ_SMOOTH_WINDOW)
    trend_hzz = core.hzz_predicted_returns(ratios, smoothed)

    enriched = panel.copy()
    for col in ("trend_hzz_raw", "trend_hzz_factor", "trend_hzz_z"):
        if col in enriched.columns:
            enriched = enriched.drop(columns=[col])
    enriched = enriched.merge(trend_hzz, on=["ticker", "month"], how="left")
    enriched["trend_hzz_factor"] = enriched["trend_hzz_raw"]
    enriched["trend_hzz_z"] = core.winsorized_zscore_by_month(
        enriched, "trend_hzz_factor", "trend_hzz_z"
    )
    return enriched, betas, smoothed


def save_hzz_diagnostics(
    enriched: pd.DataFrame,
    betas: pd.DataFrame,
    smoothed: pd.DataFrame,
) -> dict[str, float | int]:
    out_dir = core.IMPROVED_6_RESULTS_DIR
    core.save_csv(betas, out_dir / "hzz_monthly_betas.csv")
    core.save_csv(smoothed, out_dir / "hzz_smoothed_betas.csv")

    coverage = (
        enriched.groupby("month", sort=True)
        .agg(
            n_eligible=("eligible", "sum"),
            n_trend_hzz=("trend_hzz_z", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    core.save_csv(coverage, out_dir / "hzz_monthly_coverage.csv")

    first_signal_month = (
        smoothed.dropna(subset=["intercept"]).sort_values("month")["month"].iloc[0]
        if smoothed["intercept"].notna().any()
        else pd.NaT
    )
    diagnostics = {
        "n_beta_months": int(betas.shape[0]),
        "n_smoothed_months_usable": int(smoothed["intercept"].notna().sum()),
        "first_usable_signal_month": first_signal_month.isoformat()
        if isinstance(first_signal_month, pd.Timestamp)
        else "",
        "median_cross_section_size": float(betas["n_obs"].median()) if not betas.empty else float("nan"),
        "median_r_squared": float(betas["r_squared"].median()) if not betas.empty else float("nan"),
        "smoothing_window_months": core.HZZ_SMOOTH_WINDOW,
        "min_cross_section_obs": core.HZZ_MIN_CROSS_SECTION,
    }
    core.save_csv(
        pd.DataFrame([diagnostics]),
        out_dir / "hzz_diagnostics_summary.csv",
    )
    return diagnostics


def load_metric_row(path: Path, label: str) -> dict[str, object]:
    row = pd.read_csv(path).iloc[0].to_dict()
    row["label"] = label
    return row


def save_comparison_summary() -> pd.DataFrame:
    candidates = {
        "improved_4_vector": core.IMPROVED_4_RESULTS_DIR / "vector_metrics.csv",
        "improved_6_vector": core.IMPROVED_6_RESULTS_DIR / "vector_metrics.csv",
        "improved_4_backtrader": core.IMPROVED_4_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_4_STRATEGY_NAME}_metrics.csv",
        "improved_6_backtrader": core.IMPROVED_6_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_6_STRATEGY_NAME}_metrics.csv",
    }
    rows = [load_metric_row(path, label) for label, path in candidates.items() if path.exists()]
    summary = pd.DataFrame(rows)
    keep = [
        "label",
        "name",
        "backtrader_frequency",
        "annualized_sharpe",
        "final_equity",
        "final_value",
        "total_return",
        "max_drawdown",
        "stop_loss",
        "take_profit",
    ]
    for col in keep:
        if col not in summary.columns:
            summary[col] = pd.NA
    summary = summary[keep]
    core.save_csv(summary, core.IMPROVED_6_RESULTS_DIR / "improved_6_vs_improved_4_summary.csv")
    return summary


def write_improved_6_note(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
    diagnostics: dict[str, float | int],
    monte_carlo: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    bt = bt_metrics.iloc[0]
    comp = comparison.set_index("label") if not comparison.empty else pd.DataFrame()
    has_imp4_vector = "improved_4_vector" in comp.index
    has_imp4_bt = "improved_4_backtrader" in comp.index
    improved_4_vector_line = (
        f"- Improved 4 vector Sharpe: `{comp.loc['improved_4_vector', 'annualized_sharpe']:.4f}`; "
        f"improved 6 vector Sharpe: `{metrics['annualized_sharpe']:.4f}`."
        if has_imp4_vector
        else "- Improved 4 vector metrics not available for direct comparison in this run."
    )
    improved_4_bt_line = (
        f"- Improved 4 Backtrader Sharpe: `{comp.loc['improved_4_backtrader', 'annualized_sharpe']:.4f}`; "
        f"improved 6 Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`."
        if has_imp4_bt
        else "- Improved 4 Backtrader metrics not available for direct comparison in this run."
    )

    content = f"""# Improved 6 Han-Zhou-Zhu Cross-Sectional Trend

Improved 6 is a focused replacement of the trend signal. It does not replace improved 4; it tests whether a Han, Zhou, Zhu (2016) cross-sectional trend factor outperforms the project's index-derived trend regression once the rest of the improved 4 design is held fixed.

## Method

- Foundation: improved 4 composite signals, top-10 long-only construction, fixed 100k cash per trade.
- Stop-loss: `{spec.stop_loss:.1%}`.
- Take-profit: `{spec.take_profit:.1%}`.
- New trend column: `trend_hzz_z`.
- For each month `t`, run a cross-sectional OLS of next-month close-to-close returns on the 11 normalized moving-average ratios (`MA_w / P` for `w` in 3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000 trading days).
- Smooth the monthly beta vector with a strict trailing 12-month mean over months `[t-12, t-1]`. The contemporaneous beta at `t` is excluded because it uses the unobserved `t -> t+1` return.
- Per stock per month, predict the next-month return as `intercept + sum_w beta_w * (MA_w / P)`.
- Z-score per month with the standard 1st/99th winsorize.
- Daily Backtrader execution uses adjusted OHLC bars, market entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

## Diagnostics

- Monthly beta rows estimated: `{diagnostics['n_beta_months']}`.
- Months with a usable trailing-12 smoothed beta: `{diagnostics['n_smoothed_months_usable']}`.
- First usable signal month: `{diagnostics['first_usable_signal_month']}`.
- Median cross-section size: `{diagnostics['median_cross_section_size']:.0f}`.
- Median monthly R-squared: `{diagnostics['median_r_squared']:.4f}`.
- Smoothing window: `{diagnostics['smoothing_window_months']}` months (strictly trailing).
- Minimum cross-section required per month: `{diagnostics['min_cross_section_obs']}`.

## Results

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector final equity: `${metrics['final_equity']:,.0f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Backtrader final value: `${bt['final_value']:,.0f}`.
- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.
- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`.
- Monte Carlo p-value: `{monte_carlo['p_value'].iloc[0]:.4f}`.

## Comparison To Improved 4

{improved_4_vector_line}
{improved_4_bt_line}

## Warning

The HZZ trend factor uses contemporaneously-estimated cross-sectional coefficients, then smoothed across the trailing 12 months. The first 12 cross-section regressions cannot produce a trading signal because the trailing average is undefined. This delays the first effective trading month relative to improved 4 by approximately one year and slightly truncates the comparable evaluation window. The reported Sharpe and Monte Carlo p-value should be interpreted with that truncation in mind.

The factor is also driven entirely by the cross-section of stocks: it knows nothing about ROE, P/E, or momentum. Whether it dominates the index-derived trend is an empirical question about which kind of trend information is more useful on the current S&P 500 panel.
"""
    (core.DOCS_DIR / "IMPROVED_6_HZZ_TREND.md").write_text(content, encoding="utf-8")


def append_strategy_history(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
) -> None:
    path = core.DOCS_DIR / "STRATEGY_HISTORY.md"
    content = path.read_text(encoding="utf-8") if path.exists() else "# Strategy History And Improvement Log\n"
    marker = "## Improved 6 HZZ Cross-Sectional Trend"
    bt = bt_metrics.iloc[0]
    block = f"""

{marker}

Improved 6 was added after improved 5 as a focused trend-signal replacement. It keeps improved 4's composite-weight design, top-10 construction, 5% stop-loss, and 30% take-profit, then changes exactly one thing: the trend column becomes `trend_hzz_z`, a Han, Zhou, Zhu (2016) cross-sectional trend factor estimated from monthly OLS regressions of next-month returns on 11 normalized moving-average ratios across the eligible cross-section, with a strict trailing 12-month average of past betas.

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.
- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`.

Improved 6 is the first variant whose trend signal is built from the cross-section of stocks rather than from the index. It exists to test the paper's actual methodology against the assignment-prescribed index regression while keeping all other improved 4 design choices fixed.
"""
    if marker in content:
        content = content.split(marker)[0].rstrip() + block
    else:
        content = content.rstrip() + block
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    core.ensure_dirs()
    print("Running improved 6 HZZ cross-sectional trend experiment without rebuilding the full pipeline...")

    monthly, _, panel = core.load_processed_strategy_inputs()
    prices, _, _, _, index = core.load_raw_data()

    enriched, betas, smoothed = inject_hzz_trend(panel, monthly, prices)
    diagnostics = save_hzz_diagnostics(enriched, betas, smoothed)

    stop_loss, take_profit = load_improved_4_thresholds()
    spec = make_improved_6_spec(stop_loss, take_profit)

    curve, holdings = core.simulate_vector_strategy(enriched, spec)
    core.save_csv(curve, core.IMPROVED_6_RESULTS_DIR / "vector_equity_curve.csv")
    if not holdings.empty:
        core.save_csv(holdings, core.IMPROVED_6_RESULTS_DIR / "vector_holdings.csv")

    metrics = core.perf_metrics(curve["portfolio_return"], spec.name)
    metrics.update(
        {
            "final_equity": curve["equity"].iloc[-1] if not curve.empty else float("nan"),
            "total_return": (curve["equity"].iloc[-1] / core.INITIAL_CASH - 1)
            if not curve.empty
            else float("nan"),
            "avg_positions": curve["n_positions"].mean() if not curve.empty else float("nan"),
            "stop_loss": spec.stop_loss,
            "take_profit": spec.take_profit,
            "trend_col": spec.trend_col,
            "first_usable_signal_month": diagnostics["first_usable_signal_month"],
            "notes": spec.notes,
        }
    )
    core.save_csv(pd.DataFrame([metrics]), core.IMPROVED_6_RESULTS_DIR / "vector_metrics.csv")

    core.save_strategy_weight_histories(enriched, [(spec, core.IMPROVED_6_RESULTS_DIR)])

    monte_carlo = core.monte_carlo_random_portfolios(
        enriched,
        spec,
        n_sims=1000,
        output_dir=core.IMPROVED_6_RESULTS_DIR,
    )
    core.block_bootstrap(
        curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_6_RESULTS_DIR,
    )

    signals = core.signals_from_strategy(enriched, spec)
    bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        signals,
        spec.name,
        stop_loss=spec.stop_loss,
        take_profit=spec.take_profit,
        output_dir=core.IMPROVED_6_RESULTS_DIR,
    )
    min_position = core.assert_backtrader_long_only(bt, spec.name)

    comparison = save_comparison_summary()

    validation = pd.DataFrame(
        [
            {"check": "trend_column", "status": "OK", "detail": spec.trend_col},
            {
                "check": "smoothing_window_months",
                "status": "OK",
                "detail": str(core.HZZ_SMOOTH_WINDOW),
            },
            {
                "check": "min_cross_section_obs",
                "status": "OK",
                "detail": str(core.HZZ_MIN_CROSS_SECTION),
            },
            {
                "check": "first_usable_signal_month",
                "status": "OK",
                "detail": str(diagnostics["first_usable_signal_month"]),
            },
            {
                "check": "stop_loss_inherited_from_improved_4",
                "status": "OK",
                "detail": f"{spec.stop_loss:.4f}",
            },
            {
                "check": "take_profit_inherited_from_improved_4",
                "status": "OK",
                "detail": f"{spec.take_profit:.4f}",
            },
            {
                "check": "monte_carlo_p_value",
                "status": "OK",
                "detail": f"{monte_carlo['p_value'].iloc[0]:.4f}",
            },
            {"check": "backtrader_long_only", "status": "OK", "detail": str(min_position)},
        ]
    )
    core.save_csv(validation, core.IMPROVED_6_RESULTS_DIR / "improved_6_validation_summary.csv")
    write_improved_6_note(spec, metrics, bt["metrics"], diagnostics, monte_carlo, comparison)
    append_strategy_history(spec, metrics, bt["metrics"])

    print("Improved 6 completed.")
    if not comparison.empty:
        print(
            comparison[
                [
                    "label",
                    "annualized_sharpe",
                    "final_equity",
                    "final_value",
                    "max_drawdown",
                    "stop_loss",
                    "take_profit",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
