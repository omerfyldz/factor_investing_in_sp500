# Implemented Project Design

## Assignment Mapping

This project implements the factor-investing assignment on the S&P 500 instead of BIST. The S&P 500 index ticker `^GSPC` replaces the BIST100 index as benchmark, market-regime input, and trend-regression input.

The required assignment elements are implemented:

- ROE, P/E, momentum, and trend factors.
- Factor-mimicking portfolios by portfolio-sort and cross-sectional-regression methods.
- IC/rank IC, t-tests, performance metrics, common-start comparisons, and selected-date FMP weights.
- Backtrader backtests with `1,000,000` initial cash, `100,000` fixed cash per trade, market orders, and zero commission.
- Monte Carlo Sharpe p-value against random portfolios with the same monthly selection constraints.
- Saved result files, figures, and PDF presentation.

## Data Process

The project reads frozen CSV files from `data/raw/`. The supplied stock-level files contain observations into early June 2026, but the pipeline filters every analysis input to observations on or before `2026-05-31`.

Statement fundamentals use `date_available`, not fiscal period end dates, so the factor panel only uses information available to the market at the signal date.

The final processed panel is monthly. Signals at month `t` are used for month `t+1` returns/trades.

## Factors

- `roe_z`: latest point-in-time ROE, winsorized and z-scored cross-sectionally.
- `pe_z`: positive month-end P/E multiplied by `-1`, so higher scores mean cheaper stocks.
- `momentum_z`: 12-month adjusted-price return.
- `trend_z`: expanding `^GSPC` predictive-regression coefficients applied to each stock's normalized moving-average deviations.
- `hzz_trend_z`: paper-style cross-sectional trend improvement inspired by Han, Zhou, and Zhu.

The base composite score is the equal-weight average of ROE, P/E, momentum, and assignment-style trend, requiring at least three valid factors.

## Strategy Tests

The base strategy is a monthly long-only top-10 composite strategy. Improvements tested:

- top-5 concentration;
- trend/momentum-heavy weights;
- value/quality-heavy weights;
- no-trend composite;
- paper-style HZZ trend composite;
- S&P 500 regime filter;
- monthly stop-loss/take-profit approximation;
- walk-forward validation.

## Main Files

- `src/run_project.py`: full reproducible pipeline.
- `requirements.txt`: pinned runtime.
- `data/processed/factor_panel.csv`: final factor panel.
- `results/`: saved FMP, IC, strategy, Backtrader, Monte Carlo, and validation outputs.
- `figures/`: saved plots.
- `presentation/sp500_factor_investing_presentation.pdf`: final presentation.
