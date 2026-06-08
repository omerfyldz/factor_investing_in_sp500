from __future__ import annotations

import pandas as pd

import project_core as core


def main() -> None:
    """Run only the assignment-required base Backtrader strategy."""
    core.ensure_dirs()
    monthly, index_monthly, panel = core.load_processed_strategy_inputs()
    base_spec = next(s for s in core.get_strategy_specs() if s.name == "base_equal_top10")

    curve, holdings = core.simulate_vector_strategy(panel, base_spec)
    core.save_csv(curve, core.BASE_RESULTS_DIR / "vector_equity_curve.csv")
    core.save_csv(holdings, core.BASE_RESULTS_DIR / "vector_holdings.csv")
    metrics = core.perf_metrics(curve["portfolio_return"], base_spec.name)
    metrics.update(
        {
            "final_equity": curve["equity"].iloc[-1],
            "total_return": curve["equity"].iloc[-1] / core.INITIAL_CASH - 1,
            "avg_positions": curve["n_positions"].mean(),
            "assignment_scope": core.is_assignment_scope_strategy(base_spec),
            "top_n": base_spec.top_n,
            "trend_col": base_spec.trend_col,
            "regime_filter": base_spec.regime_filter,
            "stop_loss": base_spec.stop_loss,
            "take_profit": base_spec.take_profit,
            "notes": base_spec.notes,
        }
    )
    core.save_csv(
        pd.DataFrame([metrics]),
        core.BASE_RESULTS_DIR / "vector_metrics.csv",
    )
    monte_carlo = core.monte_carlo_random_portfolios(panel, base_spec, n_sims=1000)
    core.save_csv(monte_carlo, core.BASE_RESULTS_DIR / "monte_carlo_random_portfolios.csv")
    bootstrap = core.block_bootstrap(curve, block_size=6, n_sims=1000, output_name="block_bootstrap_base_strategy.csv")
    core.save_csv(bootstrap, core.BASE_RESULTS_DIR / "block_bootstrap.csv")
    core.run_backtrader(
        monthly,
        index_monthly,
        core.signals_from_strategy(panel, base_spec),
        "base_equal_top10",
        output_dir=core.BASE_RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
