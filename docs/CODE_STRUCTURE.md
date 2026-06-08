# Code And Output Structure

## Python Scripts

- `src/project_core.py`: shared data loading, factor construction, FMP analysis, strategy logic, Backtrader classes, robustness tests, reporting, and presentation generation.
- `src/run_project.py`: full reproducible pipeline. Run this for final regeneration.
- `src/run_base_strategy.py`: reruns only the assignment-required base strategy from processed CSVs.
- `src/run_improved_strategy.py`: reruns only the selected assignment-scope improved strategy from processed CSVs.
- `src/compare_strategies.py`: rebuilds the clean base-versus-improved comparison table.

## Result Folders

- `results/base_strategy/`: base strategy vector results, Backtrader orders/trades/positions/equity, Monte Carlo, and bootstrap.
- `results/improved_strategy/`: final assignment-scope improved strategy results.
- `results/comparison/`: base-versus-improved, assignment-scope strategy comparison, benchmark comparison, and walk-forward summary.
- `results/fmp_analysis/`: portfolio-sort FMPs, regression FMPs, IC/rank IC, common-start comparison, and selected-date weights.
- `results/appendix_experiments/`: HZZ trend, sector-cap, and volatility-overlay experiments kept outside the official final strategy.

The official project story should use `base_strategy`, `improved_strategy`, `comparison`, and `fmp_analysis`. Appendix experiments can be mentioned as additional ideas, but they should not be presented as the final assignment strategy.
