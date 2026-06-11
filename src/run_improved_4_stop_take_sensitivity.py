from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import project_core as core


TRAIN_END = pd.Timestamp("2020-12-31")
STOP_LOSSES = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
TAKE_PROFITS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def pct_label(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", "p").replace("p0", "")


def make_stop_take_spec(stop_loss: float, take_profit: float, name: str | None = None) -> core.StrategySpec:
    """Improved 4 varies only stop/take values around the improved 2 design."""
    label = f"stop{pct_label(stop_loss)}_take{pct_label(take_profit)}"
    return core.StrategySpec(
        name=name or f"improved_4_{label}_top10",
        weights={"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
        top_n=10,
        trend_col="trend_expanding_z",
        stop_loss=stop_loss,
        take_profit=take_profit,
        notes=(
            "Improved 4 candidate: improved 2 static equal-weight signals with alternative "
            f"{stop_loss:.1%} stop-loss and {take_profit:.1%} take-profit."
        ),
    )


def curve_metrics(curve: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, float | str]:
    sample = curve.loc[mask].sort_values("month")
    if sample.empty:
        return {"name": name}
    # Always evaluate over the common evaluation window so train/test/full comparisons
    # are consistent with the rest of the project.
    metrics = core.metrics_over_evaluation_window(
        sample, name, date_col="month", return_col="portfolio_return"
    )
    eval_sample = core.filter_to_evaluation_window(sample, "month")
    if not eval_sample.empty:
        metrics["avg_positions"] = float(eval_sample["n_positions"].mean())
    return metrics


def evaluate_grid(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    rows: list[dict[str, float | str | bool]] = []
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for stop_loss in STOP_LOSSES:
        for take_profit in TAKE_PROFITS:
            if take_profit < stop_loss:
                continue
            spec = make_stop_take_spec(stop_loss, take_profit)
            curve, holdings = core.simulate_vector_strategy(panel, spec)
            outputs[spec.name] = (curve, holdings)
            train = curve["month"] <= TRAIN_END
            test = curve["month"] > TRAIN_END
            full_m = curve_metrics(curve, spec.name, pd.Series(True, index=curve.index))
            train_m = curve_metrics(curve, f"{spec.name}_train", train)
            test_m = curve_metrics(curve, f"{spec.name}_test", test)
            rows.append(
                {
                    "strategy": spec.name,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "train_end": TRAIN_END.date().isoformat(),
                    "full_sharpe": full_m.get("annualized_sharpe", np.nan),
                    "full_final_equity": full_m.get("final_equity", np.nan),
                    "full_max_drawdown": full_m.get("max_drawdown", np.nan),
                    "train_sharpe": train_m.get("annualized_sharpe", np.nan),
                    "train_total_return": train_m.get("total_return", np.nan),
                    "train_max_drawdown": train_m.get("max_drawdown", np.nan),
                    "test_sharpe": test_m.get("annualized_sharpe", np.nan),
                    "test_total_return": test_m.get("total_return", np.nan),
                    "test_max_drawdown": test_m.get("max_drawdown", np.nan),
                    "test_avg_positions": test_m.get("avg_positions", np.nan),
                    "is_improved_2_original": np.isclose(stop_loss, 0.10) and np.isclose(take_profit, 0.20),
                }
            )
    grid = pd.DataFrame(rows)
    grid["stop_rank_index"] = grid["stop_loss"].map({v: i for i, v in enumerate(STOP_LOSSES)})
    grid["take_rank_index"] = grid["take_profit"].map({v: i for i, v in enumerate(TAKE_PROFITS)})
    return add_stability_columns(grid), outputs


def add_stability_columns(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in grid.iterrows():
        neighbors = grid[
            (grid["stop_rank_index"].sub(row["stop_rank_index"]).abs() <= 1)
            & (grid["take_rank_index"].sub(row["take_rank_index"]).abs() <= 1)
            & ~((grid["stop_loss"].eq(row["stop_loss"])) & (grid["take_profit"].eq(row["take_profit"])))
        ]
        rows.append(
            {
                "strategy": row["strategy"],
                "neighbor_train_sharpe_median": neighbors["train_sharpe"].median(),
                "neighbor_test_sharpe_median": neighbors["test_sharpe"].median(),
                "neighbor_count": len(neighbors),
            }
        )
    stability = pd.DataFrame(rows)
    out = grid.merge(stability, on="strategy", how="left")
    out["train_stability_gap"] = out["train_sharpe"] - out["neighbor_train_sharpe_median"]
    out["robust_train_score"] = (
        out["train_sharpe"]
        + out["train_max_drawdown"].fillna(0.0)
        - out["train_stability_gap"].abs().fillna(0.0)
    )
    return out.sort_values(["robust_train_score", "train_sharpe"], ascending=False)


def select_candidate(grid: pd.DataFrame) -> pd.Series:
    eligible = grid[
        grid["train_sharpe"].notna()
        & grid["train_total_return"].gt(0)
        & grid["train_max_drawdown"].gt(-0.25)
        & grid["neighbor_count"].ge(3)
    ].copy()
    if eligible.empty:
        eligible = grid.copy()
    return eligible.sort_values(["robust_train_score", "train_sharpe"], ascending=False).iloc[0]


def save_grid_artifacts(grid: pd.DataFrame, selected: pd.Series) -> None:
    out_dir = core.IMPROVED_4_RESULTS_DIR
    core.save_csv(grid, out_dir / "stop_take_sensitivity_grid.csv")
    core.save_csv(
        grid.sort_values(["robust_train_score", "train_sharpe"], ascending=False).head(10),
        out_dir / "stop_take_top10_by_train_score.csv",
    )
    core.save_csv(
        grid.sort_values(["test_sharpe", "test_max_drawdown"], ascending=False).head(10),
        out_dir / "stop_take_top10_by_test_report_only.csv",
    )
    core.save_csv(pd.DataFrame([selected]), out_dir / "selected_stop_take_parameters.csv")


def make_heatmap(grid: pd.DataFrame) -> None:
    # Figure creation moved to src/make_presentation_figures.py; this runner
    # only persists the sensitivity-grid CSV, which the figure script reads.
    return


def write_improved_4_note(grid: pd.DataFrame, selected: pd.Series, bt_metrics: pd.DataFrame) -> None:
    original = grid.loc[grid["is_improved_2_original"]].iloc[0]
    bt = bt_metrics.iloc[0]
    lines = [
        "# Improved 4 Stop/Take Sensitivity",
        "",
        "Improved 4 is a focused risk-exit parameter experiment. It does not replace the base, improved 1, improved 2, or improved 3 results.",
        "",
        "## Method",
        "",
        "- Foundation: improved 2 static equal-weight factor signals.",
        "- Changed variable: stop-loss and take-profit thresholds only.",
        f"- Training window used for selection: observations through `{TRAIN_END.date()}`.",
        "- Test-period metrics are reported after selection and are not used to pick the winner.",
        "- Selection score: training Sharpe, penalized for training drawdown and isolated parameter peaks.",
        "- Intrabar warning: monthly vector results use OHLC approximations; executable evidence comes from Backtrader.",
        "- Execution check: the selected candidate is run with daily Backtrader adjusted OHLC data, market entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.",
        "",
        "## Selected Candidate",
        "",
        f"- Stop-loss: `{selected['stop_loss']:.1%}`",
        f"- Take-profit: `{selected['take_profit']:.1%}`",
        f"- Train Sharpe: `{selected['train_sharpe']:.4f}`",
        f"- Train max drawdown: `{selected['train_max_drawdown']:.2%}`",
        f"- Test Sharpe: `{selected['test_sharpe']:.4f}`",
        f"- Test max drawdown: `{selected['test_max_drawdown']:.2%}`",
        f"- Backtrader final value: `${bt['final_value']:,.0f}`",
        f"- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`",
        f"- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`",
        "",
        "## Comparison To Improved 2 Original 10%/20%",
        "",
        f"- Original train Sharpe: `{original['train_sharpe']:.4f}`",
        f"- Original train max drawdown: `{original['train_max_drawdown']:.2%}`",
        f"- Original test Sharpe: `{original['test_sharpe']:.4f}`",
        f"- Original test max drawdown: `{original['test_max_drawdown']:.2%}`",
        "",
        "## Warning",
        "",
        "This is still a parameter search. The selected pair should be described as a candidate from a constrained walk-forward-style sensitivity test, not as a proven optimal rule. The more important result is whether nearby parameter pairs behave similarly.",
    ]
    (core.DOCS_DIR / "IMPROVED_4_STOP_TAKE_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_strategy_history(selected: pd.Series) -> None:
    path = core.DOCS_DIR / "STRATEGY_HISTORY.md"
    content = path.read_text(encoding="utf-8") if path.exists() else "# Strategy History And Improvement Log\n"
    marker = "## Improved 4 Stop/Take Sensitivity"
    block = f"""

{marker}

Improved 4 was added after the fixed 10%/20% improved 2 result. It is a focused stop-loss/take-profit sensitivity test built on improved 2's static equal-weight signal design.

- The grid is intentionally small.
- The selected pair is chosen from training data through `{TRAIN_END.date()}`.
- Test-period results are reported after selection and are not used to choose the pair.
- The selected daily Backtrader execution result is saved in `results/improved_strategy_4/`.
- Selected stop-loss: `{selected['stop_loss']:.1%}`.
- Selected take-profit: `{selected['take_profit']:.1%}`.
- Test Sharpe after selection: `{selected['test_sharpe']:.4f}`.

The warning is important: improved 4 is a robustness/sensitivity experiment, not proof that the selected parameters are permanently optimal.
"""
    if marker in content:
        content = content.split(marker)[0].rstrip() + block
    else:
        content = content.rstrip() + block
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    core.ensure_dirs()
    print("Running improved 4 stop/take sensitivity without rebuilding the full pipeline...")
    _, _, panel = core.load_processed_strategy_inputs()
    prices, _, _, _, index = core.load_raw_data()

    grid, vector_outputs = evaluate_grid(panel)
    selected = select_candidate(grid)
    save_grid_artifacts(grid, selected)
    make_heatmap(grid)

    selected_spec = make_stop_take_spec(
        float(selected["stop_loss"]),
        float(selected["take_profit"]),
        name=core.IMPROVED_4_STRATEGY_NAME,
    )
    curve, holdings = core.simulate_vector_strategy(panel, selected_spec)
    core.save_csv(curve, core.IMPROVED_4_RESULTS_DIR / "vector_equity_curve.csv")
    core.save_csv(holdings, core.IMPROVED_4_RESULTS_DIR / "vector_holdings.csv")
    metrics = core.metrics_over_evaluation_window(
        curve, selected_spec.name, date_col="month", return_col="portfolio_return"
    )
    eval_curve = core.filter_to_evaluation_window(curve, "month")
    metrics.update(
        {
            "avg_positions": float(eval_curve["n_positions"].mean()) if not eval_curve.empty else float("nan"),
            "stop_loss": selected_spec.stop_loss,
            "take_profit": selected_spec.take_profit,
            "selection_rule": "train_sharpe_minus_drawdown_and_isolated_peak_penalty",
            "train_end": TRAIN_END.date().isoformat(),
            "notes": selected_spec.notes,
        }
    )
    core.save_csv(pd.DataFrame([metrics]), core.IMPROVED_4_RESULTS_DIR / "vector_metrics.csv")
    core.save_csv(
        pd.concat(
            [
                df.assign(source_strategy=name)
                for name, (df, _) in vector_outputs.items()
            ],
            ignore_index=True,
        ),
        core.IMPROVED_4_RESULTS_DIR / "all_grid_vector_equity_curves.csv",
    )

    monte_carlo = core.monte_carlo_random_portfolios(
        panel,
        selected_spec,
        n_sims=1000,
        output_dir=core.IMPROVED_4_RESULTS_DIR,
    )
    core.block_bootstrap(
        curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_4_RESULTS_DIR,
    )

    bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        core.signals_from_strategy(panel, selected_spec),
        selected_spec.name,
        stop_loss=selected_spec.stop_loss,
        take_profit=selected_spec.take_profit,
        output_dir=core.IMPROVED_4_RESULTS_DIR,
    )
    min_position = core.assert_backtrader_long_only(bt, selected_spec.name)

    validation = pd.DataFrame(
        [
            {"check": "selected_stop_loss", "status": "OK", "detail": f"{selected_spec.stop_loss:.4f}"},
            {"check": "selected_take_profit", "status": "OK", "detail": f"{selected_spec.take_profit:.4f}"},
            {"check": "selection_uses_training_only", "status": "OK", "detail": str(TRAIN_END.date())},
            {"check": "monte_carlo_p_value", "status": "OK", "detail": f"{monte_carlo['p_value'].iloc[0]:.4f}"},
            {"check": "backtrader_long_only", "status": "OK", "detail": str(min_position)},
        ]
    )
    core.save_csv(validation, core.IMPROVED_4_RESULTS_DIR / "improved_4_validation_summary.csv")
    write_improved_4_note(grid, selected, bt["metrics"])
    append_strategy_history(selected)
    print("Improved 4 completed.")
    print(
        pd.DataFrame(
            [
                {
                    "stop_loss": selected_spec.stop_loss,
                    "take_profit": selected_spec.take_profit,
                    "vector_sharpe": metrics.get("annualized_sharpe"),
                    "monte_carlo_p_value": monte_carlo["p_value"].iloc[0],
                    "backtrader_final_value": bt["metrics"]["final_value"].iloc[0],
                    "backtrader_sharpe": bt["metrics"]["annualized_sharpe"].iloc[0],
                }
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
