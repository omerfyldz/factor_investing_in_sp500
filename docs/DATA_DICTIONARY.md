# Data Dictionary

## Raw Inputs (`data/raw/`)

| File | Source | Description |
|---|---|---|
| `sp500_prices_long.csv` | Tiingo | Daily stock OHLCV including dividend / split adjusted columns (`adjOpen`, `adjHigh`, `adjLow`, `adjClose`, `adjVolume`). 503 current S&P 500 tickers, 2006-06 to 2026-06. |
| `sp500_fundamentals_daily_long.csv` | Tiingo | Daily point-in-time `marketCap`, `peRatio`, `pbRatio`, `trailingPEG1Y`. |
| `sp500_fundamentals_statements_long.csv` | Tiingo | Statement fundamentals dated by `date_available` (the day the filing actually became public — no look-ahead). |
| `sp500_constituents.csv` | Wikipedia snapshot | Current ticker, security name, GICS sector / sub-industry, date added, headquarters location, CIK. |
| `sp500_index_yahoo.csv` | Yahoo `^GSPC` | Frozen index daily OHLCV. Used as benchmark and as input to the index-level trend regression. |

Large CSVs are tracked via Git LFS.

## Processed Outputs (`data/processed/`)

| File | Description |
|---|---|
| `monthly_stock_bars.csv` | Month-end adjusted OHLCV per ticker + 12-month momentum + forward-return columns (`next_open`, `next_high`, `next_low`, `next_close`, `next_ret_cc`, `next_ret_oc`) |
| `monthly_sp500_index.csv` | Month-end ^GSPC OHLCV + 10-month SMA + `regime_on` flag |
| `monthly_roe_asof.csv` | Per-ticker per-month ROE via `pd.merge_asof` backward on `date_available` (point-in-time) |
| `monthly_daily_fundamentals.csv` | Month-end P/E, P/B, market cap from the daily fundamentals file |
| `monthly_stock_ma_signals.csv` | Per-ticker per-month deviations from 11 moving-average windows (`ma_dev_3`, `ma_dev_5`, …, `ma_dev_1000`) |
| `monthly_index_ma_signals.csv` | Same 11 MA deviations on the index |
| `factor_panel.csv` | Stock-month panel with all factor z-scores (`roe_z`, `pe_z`, `momentum_z`, `trend_z`, `trend_expanding_z`), `composite_score`, `eligible` flag, sector, regime status |

## Results Outputs (`results/`)

Per strategy folder (`base_strategy`, `improved_strategy`, `improved_strategy_2/3/4/5/6/8`):

| File | Description |
|---|---|
| `vector_equity_curve.csv` | Monthly equity curve, return, n_positions, cost_dollars, sizing fields |
| `vector_holdings.csv` | Per-month per-stock holdings with realized returns |
| `vector_metrics.csv` | Annualized Sharpe, max DD, final equity, total return, all over the evaluation window |
| `backtrader_<name>_equity_curve.csv` | Daily Backtrader broker portfolio value |
| `backtrader_<name>_orders.csv` | Every order with `exectype` (0=Market, 2=Limit, 3=Stop), status, fill price |
| `backtrader_<name>_trades.csv` | Closed-trade PnL |
| `backtrader_<name>_positions.csv` | Per-day open-position record |
| `backtrader_<name>_metrics.csv` | Backtrader Sharpe, max DD, final value over the evaluation window |
| `monte_carlo_random_portfolios.csv` | 1000 random-portfolio Sharpes + strategy Sharpe + p-value |
| `block_bootstrap.csv` | 1000 bootstrap-resampled Sharpe / cumulative-return / max-DD samples |
| `factor_weight_history.csv` | Per-month factor weights (constant for static specs, time-varying for improved 3) |
| `improved_<n>_validation_summary.csv` | Per-strategy validation checks (long-only, parameters used, p-value, etc.) |

Improved 7-specific outputs in `results/improved_strategy_7/`:

| File | Description |
|---|---|
| `cost_schedule.csv` | Per-year per-scenario cost rates (zero / central / pessimistic) in basis points |
| `vector_results_grid.csv` | Per (strategy, scenario) Sharpe / DD / final equity / total cost paid |
| `vector_equity_curves.csv` | Per (strategy, scenario) monthly curves |
| `yearly_cost_drag.csv` | Per-year cost-drag attribution |

## Comparison Outputs (`results/comparison/`)

| File | Description |
|---|---|
| `strategy_stage_metrics.csv` | Combined metrics for base + improveds 1-3 (staged ladder) |
| `strategy_stage_curves.csv` | Combined monthly curves for the staged ladder |
| `base_vs_improved_metrics.csv` | Same as stage_metrics with `role` labels |
| `walk_forward_summary.csv` | Train / test Sharpe per strategy (staged ladder currently) |
| `strategy_benchmark_comparison.csv` | Alpha / beta vs ^GSPC per strategy (staged ladder currently) |
| `factor_weight_history.csv` | Cross-strategy factor-weight history |

## Validation (`results/`)

| File | Description |
|---|---|
| `data_audit.csv` | Row counts, ticker counts, date ranges, missing-value counts per raw / processed dataset |
| `validation_summary.csv` | Cutoff verification, Backtrader long-only check, output file counts |

## FMP Analysis (`results/fmp_analysis/`)

| File | Description |
|---|---|
| `fmp_portfolio_returns.csv` | Per-month Q5-Q1 portfolio-sort return per factor |
| `fmp_regression_returns.csv` | Per-month cross-sectional regression coefficient per factor |
| `fmp_information_coefficients.csv` | Per-month Pearson and Spearman IC per factor |
| `fmp_performance_summary.csv` | Annualized Sharpe, t-stat, IC per (approach, factor) |
| `fmp_factor_weights.csv` | Per-month constituent weights of the FMPs |
