from __future__ import annotations

import project_core as core


def main() -> None:
    """Run improved strategy stages from processed CSV inputs."""
    core.ensure_dirs()
    monthly, index_monthly, panel = core.load_processed_strategy_inputs()
    prices, _, _, _, index = core.load_raw_data()
    strategy_metrics, strategy_curves, strategy_holdings, spec_map = core.run_strategy_experiments(panel)
    base_spec = spec_map[core.BASE_STRATEGY_NAME]
    improved_spec = spec_map[core.IMPROVED_STRATEGY_NAME]
    improved_2_spec = spec_map[core.IMPROVED_2_STRATEGY_NAME]
    improved_3_spec = spec_map[core.IMPROVED_3_STRATEGY_NAME]
    core.save_categorized_strategy_outputs(
        strategy_metrics,
        strategy_curves,
        strategy_holdings,
        base_spec.name,
        improved_spec.name,
        improved_2_spec.name,
        improved_3_spec.name,
    )
    core.save_strategy_weight_histories(
        panel,
        [
            (base_spec, core.BASE_RESULTS_DIR),
            (improved_spec, core.IMPROVED_RESULTS_DIR),
            (improved_2_spec, core.IMPROVED_2_RESULTS_DIR),
            (improved_3_spec, core.IMPROVED_3_RESULTS_DIR),
        ],
    )
    core.monte_carlo_random_portfolios(
        panel,
        improved_spec,
        n_sims=1000,
        output_dir=core.IMPROVED_RESULTS_DIR,
    )
    improved_curve = strategy_curves[strategy_curves["strategy"].eq(improved_spec.name)]
    core.block_bootstrap(
        improved_curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_RESULTS_DIR,
    )
    core.monte_carlo_random_portfolios(
        panel,
        improved_2_spec,
        n_sims=1000,
        output_dir=core.IMPROVED_2_RESULTS_DIR,
    )
    improved_2_curve = strategy_curves[strategy_curves["strategy"].eq(improved_2_spec.name)]
    core.block_bootstrap(
        improved_2_curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_2_RESULTS_DIR,
    )
    core.monte_carlo_random_portfolios(
        panel,
        improved_3_spec,
        n_sims=1000,
        output_dir=core.IMPROVED_3_RESULTS_DIR,
    )
    improved_3_curve = strategy_curves[strategy_curves["strategy"].eq(improved_3_spec.name)]
    core.block_bootstrap(
        improved_3_curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_3_RESULTS_DIR,
    )
    improved_bt = core.run_backtrader(
        monthly,
        index_monthly,
        core.signals_from_strategy(panel, improved_spec),
        improved_spec.name,
        output_dir=core.IMPROVED_RESULTS_DIR,
    )
    improved_2_bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        core.signals_from_strategy(panel, improved_2_spec),
        improved_2_spec.name,
        stop_loss=improved_2_spec.stop_loss,
        take_profit=improved_2_spec.take_profit,
        output_dir=core.IMPROVED_2_RESULTS_DIR,
    )
    improved_3_bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        core.signals_from_strategy(panel, improved_3_spec),
        improved_3_spec.name,
        stop_loss=improved_3_spec.stop_loss,
        take_profit=improved_3_spec.take_profit,
        output_dir=core.IMPROVED_3_RESULTS_DIR,
    )
    core.assert_backtrader_long_only(improved_bt, improved_spec.name)
    core.assert_backtrader_long_only(improved_2_bt, improved_2_spec.name)
    core.assert_backtrader_long_only(improved_3_bt, improved_3_spec.name)


if __name__ == "__main__":
    main()
