# S&P 500 Factor Investing Project

This is a standalone S&P 500 factor investing research project. It studies quality, value, momentum, and trend signals on a frozen current-constituent S&P 500 panel, using Yahoo Finance `^GSPC` as the benchmark and trend-regression index.

## Data Window

The analysis uses frozen data available through May 2026 only. Any rows after `2026-05-31` are excluded from processing.

Raw inputs:

- `data/raw/sp500_prices_long.csv`: Tiingo daily stock prices with adjusted OHLCV.
- `data/raw/sp500_fundamentals_daily_long.csv`: point-in-time daily valuation ratios and market cap.
- `data/raw/sp500_fundamentals_statements_long.csv`: statement fundamentals dated by public availability date.
- `data/raw/sp500_constituents.csv`: current S&P 500 universe metadata.
- `data/raw/sp500_index_yahoo.csv`: frozen Yahoo Finance `^GSPC` benchmark.

## Strategy

Implemented factors:

- ROE: higher is better.
- P/E: lower positive P/E is better.
- Momentum: 12-month price return.
- Trend: the base uses a full-sample predictive regression on `^GSPC` normalized moving-average deviations, with insignificant variables dropped, applied to each stock. Improved 1 replaces only this trend step with expanding no-lookahead regressions. Improved 2 adds stop-loss/take-profit. Improved 3 adds past-only dynamic factor weights.

All factors are month-end, lagged by one month, winsorized at 1st/99th percentiles, and cross-sectionally standardized.

## Main Results

- Base vector strategy `base_equal_top10`: final equity `$2,032,039`, annualized Sharpe `1.20`.
- Improved 3 vector strategy `improved_3_dynamic_ic_weights_stop_take_top10`: final equity `$2,635,221`, annualized Sharpe `0.97`.
- Backtrader base strategy: final value `$5,778,291`, annualized Sharpe `1.36`.
- Backtrader improved 3 strategy: final value `$2,680,533`, annualized Sharpe `1.21`.
- Monte Carlo random-portfolio Sharpe p-value for the base strategy: `0.0130`.
- Monte Carlo random-portfolio Sharpe p-value for improved 3: `0.1570`.
- Improved 3 annualized alpha vs `^GSPC`: `11.16%` with alpha t-stat `3.11`.
- Walk-forward selected `improved_2_expanding_trend_stop_take_top10` on pre-2021 Sharpe; 2021-May 2026 test Sharpe was 0.96.

Backtrader is used for the staged strategy tests. Base and improved 1 use the monthly `bt.Order.Market` engine. Improved 2, improved 3, improved 4, and improved 5 use daily adjusted OHLC bars, `bt.Order.Market` entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits. All strategies use initial cash `1,000,000`, `FixedCashSizer` at `100,000` per trade, and zero commission.

## Output Layout

- `results/base_strategy/`: base strategy vector and Backtrader outputs.
- `results/improved_strategy/`: improved 1 expanding-regression strategy outputs.
- `results/improved_strategy_2/`: improved 2 stop-loss/take-profit strategy outputs.
- `results/improved_strategy_3/`: improved 3 dynamic-weight strategy outputs.
- `results/improved_strategy_4/`: improved 4 stop/take sensitivity and selected candidate.
- `results/improved_strategy_5/`: improved 5 market-regime filter experiment.
- `results/comparison/`: staged base-versus-improved comparison and walk-forward files.
- `results/fmp_analysis/`: factor-mimicking portfolio, IC, and factor comparison files.

## Reproduce

```powershell
cd C:\Users\asus\Desktop\sp500_factor_investing
py -3.10 src\run_project.py
```

The run reads frozen CSVs, regenerates processed data, results, figures, and the PDF presentation.

## Limitations

- The universe is current S&P 500 constituents, so the study has survivorship bias.
- Transaction costs and slippage are ignored by design in the current research run.
- Shorting is shown only in FMP analysis; the implemented trading strategy is long-only.
- Public factor performance can decay over time, especially after 2020.
