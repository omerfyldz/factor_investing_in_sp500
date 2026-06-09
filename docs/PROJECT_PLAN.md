# Implemented Project Design

## Assignment Mapping

This project implements the factor-investing assignment on the S&P 500 instead of BIST. The S&P 500 index ticker `^GSPC` replaces the BIST100 index as benchmark and trend-regression input. Other requirements remain aligned with the assignment: frozen CSV data, month-end factors, lagged signals, Backtrader, saved results, and reproducibility.

## Data Process

The project reads frozen CSV files from `data/raw/`. The supplied stock-level files contain observations into early June 2026, but the pipeline filters every analysis input to observations on or before `2026-05-31`.

Statement fundamentals use `date_available`, not fiscal period end dates, so the factor panel only uses information available to the market at the signal date.

The final processed panel is monthly. Signals at month `t` are used for month `t+1` returns/trades.

## Factors

- `roe_z`: latest point-in-time ROE, winsorized and z-scored cross-sectionally.
- `pe_z`: positive month-end P/E multiplied by `-1`, so higher scores mean cheaper stocks.
- `momentum_z`: 12-month adjusted-price return.
- `trend_z`: base assignment-style trend factor. It uses one full-sample `^GSPC` predictive regression with insignificant moving-average variables dropped.
- `trend_expanding_z`: improved trend factor. For each signal month, it estimates the `^GSPC` predictive regression using only earlier index observations.

The base composite score is the equal-weight average of ROE, P/E, momentum, and `trend_z`, requiring at least three valid factors.

## Strategy Ladder

The active strategy set contains a core ladder plus focused follow-up experiments:

1. `base_equal_top10`: equal-weight ROE/P/E/momentum/full-sample-trend composite, top 10, no stop-loss/take-profit.
2. `improved_1_expanding_trend_top10`: same as base, but trend uses expanding no-lookahead regression.
3. `improved_2_expanding_trend_stop_take_top10`: same as improved 1, plus 10% stop-loss and 20% take-profit.
4. `improved_3_dynamic_ic_weights_stop_take_top10`: same as improved 2, plus rolling rank-IC factor weights with 50% shrinkage to equal weights and 10%-45% factor caps.
5. `improved_4_walkforward_stop_take_top10`: focused stop/take sensitivity branch from improved 2. It keeps the same static equal-weight signals and changes only stop-loss/take-profit thresholds selected from training data through 2020.
6. `improved_5_regime_filtered_stop_take_top10`: focused regime-filter branch from improved 4. It keeps the same signals and selected stop/take thresholds, then trades only when `^GSPC` is above its 10-month SMA.

This design makes the improvement path readable: each stage changes one main idea.

Backtrader execution is separated by strategy stage. Base and improved 1 use explicit monthly `bt.Order.Market` orders. Improved 2 and improved 3 use daily adjusted OHLC bars, explicit `bt.Order.Market` entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.
Improved 4 uses the same daily Backtrader native stop/limit execution for the selected threshold pair.
Improved 5 uses the same daily Backtrader native stop/limit execution as improved 4, but its monthly signal list is empty when the regime filter is off.
For risk-managed strategies, rebalance exits cancel live protective orders before submitting market exits. This prevents stale stop/limit orders from firing after a position has already been closed.

## Main Files

- `src/run_project.py`: full reproducible pipeline.
- `requirements.txt`: pinned runtime.
- `data/processed/factor_panel.csv`: final factor panel after the next full run.
- `results/`: saved FMP, IC, staged strategy, Backtrader, Monte Carlo, and validation outputs.
- `figures/`: saved plots.
- `presentation/sp500_factor_investing_presentation.pdf`: final presentation.
