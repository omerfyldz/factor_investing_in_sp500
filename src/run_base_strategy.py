from __future__ import annotations

import pandas as pd

import project_core as core


def main() -> None:
    """Run only the pure base strategy from processed CSV inputs."""
    core.ensure_dirs()
    monthly, index_monthly, panel = core.load_processed_strategy_inputs()
    base_spec = next(s for s in core.get_strategy_specs() if s.name == core.BASE_STRATEGY_NAME)

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
            "weight_method": base_spec.weight_method,
            "weight_lookback_months": base_spec.weight_lookback_months,
            "weight_min_periods": base_spec.weight_min_periods,
            "weight_shrink_to_equal": base_spec.weight_shrink_to_equal,
            "min_factor_weight": base_spec.min_factor_weight,
            "max_factor_weight": base_spec.max_factor_weight,
            "regime_filter": base_spec.regime_filter,
            "stop_loss": base_spec.stop_loss,
            "take_profit": base_spec.take_profit,
            "notes": base_spec.notes,
        }
    )
    core.save_strategy_weight_histories(panel, [(base_spec, core.BASE_RESULTS_DIR)])
    core.save_csv(
        pd.DataFrame([metrics]),
        core.BASE_RESULTS_DIR / "vector_metrics.csv",
    )
    core.monte_carlo_random_portfolios(panel, base_spec, n_sims=1000, output_dir=core.BASE_RESULTS_DIR)
    core.block_bootstrap(curve, block_size=6, n_sims=1000, output_name="block_bootstrap.csv", output_dir=core.BASE_RESULTS_DIR)
    core.run_backtrader(
        monthly,
        index_monthly,
        core.signals_from_strategy(panel, base_spec),
        core.BASE_STRATEGY_NAME,
        output_dir=core.BASE_RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
