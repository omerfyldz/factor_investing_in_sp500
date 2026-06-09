# Data Dictionary

## Raw Inputs

- `sp500_prices_long.csv`: daily stock OHLCV, including dividend/split adjusted prices.
- `sp500_fundamentals_daily_long.csv`: point-in-time daily market cap, P/E, P/B, and PEG.
- `sp500_fundamentals_statements_long.csv`: statement fundamentals using `date_available`.
- `sp500_constituents.csv`: current S&P 500 universe metadata.
- `sp500_index_yahoo.csv`: frozen Yahoo Finance S&P 500 index data.

## Processed Outputs

- `monthly_stock_bars.csv`: month-end adjusted OHLCV and forward returns.
- `factor_panel.csv`: stock-month factor signals, z-scores, and forward returns.
- `fmp_portfolio_returns.csv`: top-minus-bottom factor returns.
- `fmp_regression_returns.csv`: monthly cross-sectional factor-premium estimates.
- `strategy_vector_equity_curves.csv`: vectorized strategy/improvement equity curves.
- `backtrader_*`: monthly Backtrader equity, orders, trades, positions, and metrics for base/improved 1.
- `backtrader_daily_*`: daily Backtrader equity, orders, trades, positions, and metrics for improved 2/improved 3/improved 4/improved 5 stop-loss/take-profit execution checks.
- `results/improved_strategy_4/stop_take_sensitivity_grid.csv`: improved 4 stop-loss/take-profit grid with full, training, test, and stability metrics.
- `results/improved_strategy_4/selected_stop_take_parameters.csv`: selected improved 4 threshold pair and the training/test diagnostics used to audit it.
- `results/improved_strategy_5/monthly_regime_exposure.csv`: month-level regime state, invested/cash status, and strategy return for improved 5.
- `results/improved_strategy_5/regime_filter_diagnostics.csv`: compact regime exposure diagnostics for improved 5.
