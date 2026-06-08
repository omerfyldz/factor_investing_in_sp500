# S&P 500 Factor Investing Research

This repository contains a reproducible S&P 500 factor investing study through May 2026. The project builds stock-level factor signals, constructs factor-mimicking portfolios, runs long-only Backtrader strategies, and compares a base strategy against an improved factor-weighted variant.

## Research Design

The study uses four core factors:

- **ROE**: higher profitability is preferred.
- **P/E**: lower positive valuation is preferred.
- **Momentum**: trailing 12-month stock return.
- **Trend**: predicted return from a full-sample S&P 500 index trend regression using normalized moving-average deviations.

Signals are formed monthly, outliers are clipped at the 1st and 99th percentiles, and factors are cross-sectionally standardized with z-scores.

## Strategy Summary

The base strategy is `base_equal_top10`:

- equal-weight composite of ROE, P/E, momentum, and trend;
- top 10 stocks selected monthly;
- long-only;
- initial capital `1,000,000`;
- fixed cash per trade `100,000`;
- commission `0`;
- Backtrader market orders.

The selected improved strategy is `value_quality_heavy_top10`:

- same four core factors;
- higher weight on ROE and P/E;
- same long-only monthly trading framework.

## Key Results

| Role | Strategy | Backtrader Final Value | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Base | `base_equal_top10` | `4,944,058` | `1.13` | `-13.57%` |
| Improved | `value_quality_heavy_top10` | `4,816,929` | `1.16` | `-12.69%` |

The improved strategy has lower total return than the base strategy, but better risk-adjusted performance: higher Sharpe and lower maximum drawdown.

## Monte Carlo Robustness

The Monte Carlo test compares each strategy's Sharpe ratio against 1,000 random portfolios using the same broad universe, rebalance schedule, number of positions, and fixed cash sizing.

| Strategy | Strategy Sharpe | Random Mean Sharpe | Random 95% Sharpe | Random Max Sharpe | Empirical p-value |
|---|---:|---:|---:|---:|---:|
| Base | `1.0173` | `0.6470` | `0.7934` | `0.9495` | `< 0.001` |
| Improved | `1.0553` | `0.6470` | `0.7934` | `0.9495` | `< 0.001` |

No random simulation matched or exceeded either strategy's Sharpe ratio.

## Repository Structure

- `src/project_core.py`: data processing, factor construction, FMP analysis, Backtrader strategies, robustness tests, reports, and figures.
- `src/run_project.py`: full project pipeline.
- `src/run_base_strategy.py`: reruns only the base strategy from processed CSVs.
- `src/run_improved_strategy.py`: reruns only the improved strategy from processed CSVs.
- `src/compare_strategies.py`: rebuilds the base-versus-improved comparison table.
- `data/processed/`: processed monthly stock and factor panels.
- `results/base_strategy/`: base strategy vector and Backtrader outputs.
- `results/improved_strategy/`: improved strategy vector and Backtrader outputs.
- `results/comparison/`: comparison tables, benchmark comparison, and walk-forward summary.
- `results/fmp_analysis/`: factor-mimicking portfolio returns, IC/rank IC, and factor comparison files.
- `figures/`: generated charts.
- `presentation/`: generated PDF summary.

## Reproducibility

Install dependencies:

```powershell
py -3.10 -m pip install -r requirements.txt
```

Run the full pipeline:

```powershell
py -3.10 src\run_project.py
```

Large vendor raw files are not included in this GitHub repository because they exceed normal GitHub file-size limits. To rerun the raw-data pipeline from scratch, place the following local CSVs under `data/raw/`:

- `sp500_prices_long.csv`
- `sp500_fundamentals_daily_long.csv`
- `sp500_fundamentals_statements_long.csv`

The repository includes processed data, saved results, figures, and small raw reference files needed to inspect the research outputs.

## Limitations

- The universe uses current S&P 500 constituents, so survivorship bias remains.
- Transaction costs and slippage are ignored.
- Backtests use historical adjusted prices and point-in-time fields where available, but real trading implementation would require stricter execution modeling.
- The strategy was selected after testing multiple variants, so the results should be interpreted as research evidence rather than a production-ready trading system.
