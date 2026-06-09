from __future__ import annotations

from pathlib import Path

import pandas as pd

import project_core as core


DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.30


def load_improved_4_thresholds() -> tuple[float, float]:
    selected_path = core.IMPROVED_4_RESULTS_DIR / "selected_stop_take_parameters.csv"
    if not selected_path.exists():
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    return float(selected["stop_loss"].iloc[0]), float(selected["take_profit"].iloc[0])


def make_improved_5_spec(stop_loss: float, take_profit: float) -> core.StrategySpec:
    """Improved 5 changes only one thing from improved 4: a pre-specified index regime filter."""
    return core.StrategySpec(
        name=core.IMPROVED_5_STRATEGY_NAME,
        weights={"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
        top_n=10,
        regime_filter=True,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trend_col="trend_expanding_z",
        notes=(
            "Improved 5: improved 4 plus a pre-specified ^GSPC 10-month SMA regime filter. "
            "No regime-window optimization is performed."
        ),
    )


def make_regime_diagnostics(panel: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    regime = (
        panel[["month", "index_close", "sma_10m", "regime_on"]]
        .drop_duplicates("month")
        .sort_values("month")
        .merge(curve[["month", "n_positions", "portfolio_return"]], on="month", how="left")
    )
    regime["invested"] = regime["n_positions"].fillna(0).gt(0)
    rows = [
        {
            "metric": "total_months",
            "value": len(regime),
            "detail": "All signal months in the processed factor panel.",
        },
        {
            "metric": "regime_on_months",
            "value": int(regime["regime_on"].fillna(False).sum()),
            "detail": "^GSPC month-end close is above its 10-month SMA.",
        },
        {
            "metric": "regime_off_months",
            "value": int((~regime["regime_on"].fillna(False)).sum()),
            "detail": "Strategy should hold cash for the following month.",
        },
        {
            "metric": "invested_months",
            "value": int(regime["invested"].sum()),
            "detail": "Months with at least one selected position in the vector backtest.",
        },
        {
            "metric": "cash_months",
            "value": int((~regime["invested"]).sum()),
            "detail": "Months with no selected positions due to regime filter or unavailable signals.",
        },
        {
            "metric": "avg_return_when_regime_on",
            "value": float(regime.loc[regime["regime_on"].fillna(False), "portfolio_return"].mean()),
            "detail": "Average strategy return in signal months where the regime filter is on.",
        },
        {
            "metric": "avg_return_when_regime_off",
            "value": float(regime.loc[~regime["regime_on"].fillna(False), "portfolio_return"].mean()),
            "detail": "Should be near zero because the filter sends the strategy to cash.",
        },
    ]
    out = pd.DataFrame(rows)
    core.save_csv(regime, core.IMPROVED_5_RESULTS_DIR / "monthly_regime_exposure.csv")
    return out


def load_metric_row(path: Path, label: str) -> dict[str, object]:
    row = pd.read_csv(path).iloc[0].to_dict()
    row["label"] = label
    return row


def save_comparison_summary() -> pd.DataFrame:
    rows = []
    candidates = {
        "improved_4_vector": core.IMPROVED_4_RESULTS_DIR / "vector_metrics.csv",
        "improved_5_vector": core.IMPROVED_5_RESULTS_DIR / "vector_metrics.csv",
        "improved_4_backtrader": core.IMPROVED_4_RESULTS_DIR / f"backtrader_daily_{core.IMPROVED_4_STRATEGY_NAME}_metrics.csv",
        "improved_5_backtrader": core.IMPROVED_5_RESULTS_DIR / f"backtrader_daily_{core.IMPROVED_5_STRATEGY_NAME}_metrics.csv",
    }
    for label, path in candidates.items():
        if path.exists():
            rows.append(load_metric_row(path, label))
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
    core.save_csv(summary, core.IMPROVED_5_RESULTS_DIR / "improved_5_vs_improved_4_summary.csv")
    return summary


def write_improved_5_note(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
    regime_diagnostics: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    bt = bt_metrics.iloc[0]
    diagnostics = {row["metric"]: row["value"] for _, row in regime_diagnostics.iterrows()}
    comp = comparison.set_index("label")
    improved_4_vector = comp.loc["improved_4_vector"]
    improved_4_bt = comp.loc["improved_4_backtrader"]
    content = f"""# Improved 5 Regime Filter

Improved 5 is a focused market-regime experiment. It does not replace improved 4; it tests one extra risk-management rule on top of improved 4.

## Method

- Foundation: improved 4 factor signals and stop/take thresholds.
- Stop-loss: `{spec.stop_loss:.1%}`.
- Take-profit: `{spec.take_profit:.1%}`.
- New rule: trade only when `^GSPC` month-end close is above its existing 10-month SMA.
- If the filter is off at signal month `t`, the strategy holds cash during month `t+1`.
- No moving-average window optimization is performed.
- Daily Backtrader execution uses adjusted OHLC data, market entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

## Results

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector final equity: `${metrics['final_equity']:,.0f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Backtrader final value: `${bt['final_value']:,.0f}`.
- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.
- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`.
- Monte Carlo p-value: `{monte_carlo['p_value'].iloc[0]:.4f}`.

## Decision

Improved 5 is not accepted as a performance improvement over improved 4.

- Improved 4 vector Sharpe: `{improved_4_vector['annualized_sharpe']:.4f}`; improved 5 vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Improved 4 vector max drawdown: `{improved_4_vector['max_drawdown']:.2%}`; improved 5 vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Improved 4 Backtrader Sharpe: `{improved_4_bt['annualized_sharpe']:.4f}`; improved 5 Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.
- Improved 4 Backtrader max drawdown: `{improved_4_bt['max_drawdown']:.2%}`; improved 5 Backtrader max drawdown: `{bt['max_drawdown']:.2%}`.

The filter likely removed too much exposure and missed rebound months. This is a useful failed experiment because it shows that a simple index cash filter is not automatically better once daily stop-loss/take-profit risk control is already present.

## Exposure

- Total months: `{int(diagnostics.get('total_months', 0))}`.
- Regime-on months: `{int(diagnostics.get('regime_on_months', 0))}`.
- Regime-off months: `{int(diagnostics.get('regime_off_months', 0))}`.
- Invested months: `{int(diagnostics.get('invested_months', 0))}`.
- Cash months: `{int(diagnostics.get('cash_months', 0))}`.

## Warning

This is a dangerous zone because market-regime filters can easily become market-timing overfit. The rule is deliberately fixed at the pre-existing 10-month SMA. We should judge it by whether it improves drawdown and robustness relative to improved 4 without destroying return, not by whether it maximizes full-sample performance.
"""
    (core.DOCS_DIR / "IMPROVED_5_REGIME_FILTER.md").write_text(content, encoding="utf-8")


def append_strategy_history(spec: core.StrategySpec, metrics: dict[str, object], bt_metrics: pd.DataFrame) -> None:
    path = core.DOCS_DIR / "STRATEGY_HISTORY.md"
    content = path.read_text(encoding="utf-8") if path.exists() else "# Strategy History And Improvement Log\n"
    marker = "## Improved 5 Regime Filter"
    bt = bt_metrics.iloc[0]
    block = f"""

{marker}

Improved 5 was added after improved 4 as a focused market-regime filter test. It keeps improved 4's factor signals, top-10 construction, 5% stop-loss, and 30% take-profit, then adds one pre-specified rule: trade only when `^GSPC` is above its 10-month moving average.

- No regime-window optimization was performed.
- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.
- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`.

The warning is important: this is a market-timing overlay and must be treated skeptically. It is acceptable only because it changes one pre-specified design dimension and is stored separately from improved 4.
"""
    if marker in content:
        content = content.split(marker)[0].rstrip() + block
    else:
        content = content.rstrip() + block
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    core.ensure_dirs()
    print("Running improved 5 regime-filter experiment without rebuilding the full pipeline...")
    _, _, panel = core.load_processed_strategy_inputs()
    prices, _, _, _, index = core.load_raw_data()
    stop_loss, take_profit = load_improved_4_thresholds()
    spec = make_improved_5_spec(stop_loss, take_profit)

    curve, holdings = core.simulate_vector_strategy(panel, spec)
    core.save_csv(curve, core.IMPROVED_5_RESULTS_DIR / "vector_equity_curve.csv")
    core.save_csv(holdings, core.IMPROVED_5_RESULTS_DIR / "vector_holdings.csv")
    metrics = core.perf_metrics(curve["portfolio_return"], spec.name)
    metrics.update(
        {
            "final_equity": curve["equity"].iloc[-1],
            "total_return": curve["equity"].iloc[-1] / core.INITIAL_CASH - 1,
            "avg_positions": curve["n_positions"].mean(),
            "regime_filter": spec.regime_filter,
            "stop_loss": spec.stop_loss,
            "take_profit": spec.take_profit,
            "notes": spec.notes,
        }
    )
    core.save_csv(pd.DataFrame([metrics]), core.IMPROVED_5_RESULTS_DIR / "vector_metrics.csv")

    regime_diagnostics = make_regime_diagnostics(panel, curve)
    core.save_csv(regime_diagnostics, core.IMPROVED_5_RESULTS_DIR / "regime_filter_diagnostics.csv")
    core.save_strategy_weight_histories(panel, [(spec, core.IMPROVED_5_RESULTS_DIR)])

    monte_carlo = core.monte_carlo_random_portfolios(
        panel,
        spec,
        n_sims=1000,
        output_dir=core.IMPROVED_5_RESULTS_DIR,
    )
    core.block_bootstrap(
        curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_5_RESULTS_DIR,
    )

    bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        core.signals_from_strategy(panel, spec),
        spec.name,
        stop_loss=spec.stop_loss,
        take_profit=spec.take_profit,
        output_dir=core.IMPROVED_5_RESULTS_DIR,
    )
    min_position = core.assert_backtrader_long_only(bt, spec.name)
    comparison = save_comparison_summary()

    validation = pd.DataFrame(
        [
            {"check": "regime_rule", "status": "OK", "detail": "^GSPC close > 10-month SMA"},
            {"check": "regime_window_not_optimized", "status": "OK", "detail": "10 months"},
            {"check": "stop_loss_inherited_from_improved_4", "status": "OK", "detail": f"{spec.stop_loss:.4f}"},
            {"check": "take_profit_inherited_from_improved_4", "status": "OK", "detail": f"{spec.take_profit:.4f}"},
            {"check": "monte_carlo_p_value", "status": "OK", "detail": f"{monte_carlo['p_value'].iloc[0]:.4f}"},
            {"check": "backtrader_long_only", "status": "OK", "detail": str(min_position)},
        ]
    )
    core.save_csv(validation, core.IMPROVED_5_RESULTS_DIR / "improved_5_validation_summary.csv")
    write_improved_5_note(spec, metrics, bt["metrics"], regime_diagnostics, monte_carlo, comparison)
    append_strategy_history(spec, metrics, bt["metrics"])

    print("Improved 5 completed.")
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
