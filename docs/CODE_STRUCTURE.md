# Code And Output Structure

## Python Scripts

- `src/project_core.py`: shared data loading, factor construction, FMP analysis, staged strategy logic, Backtrader classes, robustness tests, reporting, and presentation generation.
- `src/run_project.py`: full reproducible pipeline. Run this for final regeneration.
- `src/run_base_strategy.py`: reruns only the pure base strategy from processed CSVs.
- `src/run_improved_strategy.py`: reruns improved 1, improved 2, and improved 3 from processed CSVs; it also reads raw daily prices/index data for the improved 2 and improved 3 daily Backtrader stop/take checks.
- `src/run_improved_4_stop_take_sensitivity.py`: reruns only the improved 4 stop/take sensitivity experiment from processed factor data plus raw daily prices/index data.
- `src/run_improved_5_regime_filter.py`: reruns only the improved 5 market-regime filter experiment from processed factor data plus raw daily prices/index data.
- `src/compare_strategies.py`: rebuilds the staged base-versus-improved comparison table.

## Active Result Folders

- `results/base_strategy/`: base strategy vector results, Backtrader orders/trades/positions/equity, Monte Carlo, and bootstrap.
- `results/improved_strategy/`: improved 1 results. This is the base strategy with the trend factor changed from full-sample regression to expanding no-lookahead regression.
- `results/improved_strategy_2/`: improved 2 results. This builds on improved 1 and adds stop-loss/take-profit risk exits, including daily Backtrader stop/take outputs prefixed with `backtrader_daily_`.
- `results/improved_strategy_3/`: improved 3 results. This builds on improved 2 and adds rolling rank-IC factor weights with shrinkage and caps, including daily Backtrader stop/take outputs prefixed with `backtrader_daily_`.
- `results/improved_strategy_4/`: improved 4 focused stop/take sensitivity outputs. This keeps improved 2's static equal-weight signals and varies only stop-loss/take-profit thresholds using a training-only selection rule.
- `results/improved_strategy_5/`: improved 5 focused market-regime filter outputs. This keeps improved 4's selected stop/take thresholds and adds a pre-specified `^GSPC` 10-month SMA cash filter.
- `results/comparison/`: staged comparison, benchmark comparison, and walk-forward summary.
- `results/fmp_analysis/`: portfolio-sort FMPs, regression FMPs, IC/rank IC, common-start comparison, selected-date weights, and trend-regression coefficients.

## Staging Rule

The active project story is intentionally sequential:

1. Base: literal four-factor assignment-style strategy.
2. Improved 1: change only trend-regression estimation to expanding/no-lookahead.
3. Improved 2: build on improved 1 and add stop-loss/take-profit.
4. Improved 3: build on improved 2 and add past-only dynamic factor weighting.
5. Improved 4: branch from improved 2 and test stop/take threshold sensitivity without overwriting prior stages.
6. Improved 5: branch from improved 4 and test one pre-specified index regime filter without optimizing the regime window.

Deferred ideas are documented in `docs/STRATEGY_HISTORY.md` and should be reintroduced one at a time only after the current staged results are recorded.
