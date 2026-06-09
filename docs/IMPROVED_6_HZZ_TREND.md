# Improved 6 Han-Zhou-Zhu Cross-Sectional Trend

Improved 6 is a focused replacement of the trend signal. It does not replace improved 4; it tests whether a Han, Zhou, Zhu (2016) cross-sectional trend factor outperforms the project's index-derived trend regression once the rest of the improved 4 design is held fixed.

## Method

- Foundation: improved 4 composite signals, top-10 long-only construction, fixed 100k cash per trade.
- Stop-loss: `5.0%`.
- Take-profit: `30.0%`.
- New trend column: `trend_hzz_z`.
- For each month `t`, run a cross-sectional OLS of next-month close-to-close returns on the 11 normalized moving-average ratios (`MA_w / P` for `w` in 3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000 trading days).
- Smooth the monthly beta vector with a strict trailing 12-month mean over months `[t-12, t-1]`. The contemporaneous beta at `t` is excluded because it uses the unobserved `t -> t+1` return.
- Per stock per month, predict the next-month return as `intercept + sum_w beta_w * (MA_w / P)`.
- Z-score per month with the standard 1st/99th winsorize.
- Daily Backtrader execution uses adjusted OHLC bars, market entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

## Diagnostics

- Monthly beta rows estimated: `192`.
- Months with a usable trailing-12 smoothed beta: `180`.
- First usable signal month: `2011-05-31T00:00:00`.
- Median cross-section size: `450`.
- Median monthly R-squared: `0.1465`.
- Smoothing window: `12` months (strictly trailing).
- Minimum cross-section required per month: `100`.

## Results

- Vector Sharpe: `0.8136`.
- Vector final equity: `$3,053,582`.
- Vector max drawdown: `-11.15%`.
- Backtrader final value: `$3,161,381`.
- Backtrader Sharpe: `0.8395`.
- Backtrader max drawdown: `-11.32%`.
- Monte Carlo p-value: `0.0390`.

## Comparison To Improved 4

- Improved 4 vector Sharpe: `0.7968`; improved 6 vector Sharpe: `0.8136`.
- Improved 4 Backtrader Sharpe: `0.9852`; improved 6 Backtrader Sharpe: `0.8395`.

## Warning

The HZZ trend factor uses contemporaneously-estimated cross-sectional coefficients, then smoothed across the trailing 12 months. The first 12 cross-section regressions cannot produce a trading signal because the trailing average is undefined. This delays the first effective trading month relative to improved 4 by approximately one year and slightly truncates the comparable evaluation window. The reported Sharpe and Monte Carlo p-value should be interpreted with that truncation in mind.

The factor is also driven entirely by the cross-section of stocks: it knows nothing about ROE, P/E, or momentum. Whether it dominates the index-derived trend is an empirical question about which kind of trend information is more useful on the current S&P 500 panel.
