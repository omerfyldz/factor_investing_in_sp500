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
- `backtrader_*`: Backtrader equity, orders, trades, positions, and metrics.
