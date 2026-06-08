from __future__ import annotations

import project_core as core


def main() -> None:
    """Run the official assignment-scope improved strategy."""
    core.ensure_dirs()
    monthly, index_monthly, panel = core.load_processed_strategy_inputs()
    strategy_metrics, strategy_curves, strategy_holdings, spec_map = core.run_strategy_experiments(panel)
    best_name = core.choose_best_assignment_strategy(strategy_metrics, spec_map)
    best_spec = spec_map[best_name]
    core.save_categorized_strategy_outputs(
        strategy_metrics,
        strategy_curves,
        strategy_holdings,
        "base_equal_top10",
        best_name,
    )
    monte_carlo = core.monte_carlo_random_portfolios(
        panel,
        best_spec,
        n_sims=1000,
        output_name=f"monte_carlo_final_{best_name}.csv",
    )
    core.save_csv(monte_carlo, core.IMPROVED_RESULTS_DIR / "monte_carlo_random_portfolios.csv")
    curve = strategy_curves[strategy_curves["strategy"].eq(best_name)]
    bootstrap = core.block_bootstrap(
        curve,
        block_size=6,
        n_sims=1000,
        output_name=f"block_bootstrap_final_{best_name}.csv",
    )
    core.save_csv(bootstrap, core.IMPROVED_RESULTS_DIR / "block_bootstrap.csv")
    core.run_backtrader(
        monthly,
        index_monthly,
        core.signals_from_strategy(panel, best_spec),
        f"best_{best_name}",
        stop_loss=best_spec.stop_loss,
        take_profit=best_spec.take_profit,
        output_dir=core.IMPROVED_RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
