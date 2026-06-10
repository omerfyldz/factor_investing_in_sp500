# Implemented Project Design

## Assignment Mapping

This project implements the factor-investing assignment on the S&P 500 instead of BIST100. The S&P 500 index ticker `^GSPC` replaces the BIST100 index as benchmark and trend-regression input. Other requirements remain aligned: frozen CSV data, month-end factors, lagged signals, Backtrader execution simulation, saved results, and reproducibility.

The project extends the assignment with six additional dimensions: a paper-faithful HZZ cross-sectional trend factor (improved 6), realistic time-varying transaction-cost analysis (improved 7), equal-weight 1/N sizing with top-N expansion (improved 8), inverse-volatility targeted sizing (improved 9), the common-evaluation-window methodology fix, and structured robustness testing (Monte Carlo + block bootstrap + walk-forward + cost sensitivity + multi-comparison correction via Hansen SPA and Romano-Wolf StepM).

## Data Process

The project reads frozen CSV files from `data/raw/`. The supplied stock-level files contain observations into early June 2026, but the pipeline filters every analysis input to observations on or before `2026-05-31`.

Statement fundamentals use `date_available`, not fiscal period end dates, so the factor panel only uses information available to the market at the signal date.

The final processed panel is monthly. Signals at month `t` are used for month `t+1` returns/trades.

## Factors

- `roe_z` — latest point-in-time ROE, winsorized and z-scored cross-sectionally.
- `pe_z` — positive month-end P/E multiplied by `-1`, so higher scores mean cheaper stocks.
- `momentum_z` — 12-month adjusted-price return.
- `trend_z` — base assignment-style trend factor. One full-sample `^GSPC` predictive regression with insignificant moving-average variables dropped.
- `trend_expanding_z` — no-lookahead trend factor. For each signal month, estimates the `^GSPC` predictive regression using only earlier index observations.
- `trend_hzz_z` — Han-Zhou-Zhu cross-sectional trend factor. For each signal month, runs a cross-sectional OLS of next-month stock returns on stock-level 11 normalized MA ratios, then smooths the coefficient vector across the strict trailing 12 months. Per-stock predicted-return signal.

The base composite score is the equal-weight average of ROE, P/E, momentum, and a trend column (chosen by the strategy spec). The trading version of the composite requires all four factors to be non-NaN; this is more conservative than the `composite_score` column in the panel itself, which implements a documented 3-of-4 skipna rule but is not used by `score_for_spec`.

## Common Evaluation Window

`EVALUATION_START = 2016-05-31` is the date at which all eight strategies have their signals computable. All Sharpe / drawdown / cumulative return / Monte Carlo p-value / walk-forward / benchmark alpha metrics are computed over `>= EVALUATION_START`. This is the standard academic convention for cross-strategy comparison and is implemented via `metrics_over_evaluation_window` and `filter_to_evaluation_window` helpers in `project_core.py`.

Before this fix, the project reported Sharpes computed over all 240 months of the panel, which mathematically penalized strategies with longer warmups (more pre-warmup zero-return idle months in the denominator). The fix removes this asymmetry.

## Strategy Ladder

| # | Name | Foundation | One change | Notes |
|---|---|---|---|---|
| 0 | `base_equal_top10` | — | Assignment-prescribed full-sample trend | Look-ahead biased; reported because the rubric requires it |
| 1 | `improved_1_expanding_trend_top10` | base | Trend → expanding no-lookahead regression | Honest baseline |
| 2 | `improved_2_expanding_trend_stop_take_top10` | improved 1 | Adds 10% stop / 20% take | Risk-management |
| 3 | `improved_3_dynamic_ic_weights_stop_take_top10` | improved 2 | Adds rolling rank-IC dynamic weights | Rejected — no improvement |
| 4 | `improved_4_walkforward_stop_take_top10` | improved 2 | Walk-forward selected 5% / 30% stops | **Best risk-managed**; train ≤ 2020-12 |
| 5 | `improved_5_regime_filtered_stop_take_top10` | improved 4 | Adds ^GSPC 10-month SMA regime filter | Rejected — hurt performance |
| 6 | `improved_6_hzz_cross_sectional_trend_stop_take_top10` | improved 4 | Trend → HZZ cross-sectional | Paper-faithful; lower per-trade alpha than improved 4 |
| 7 | `improved_7_time_varying_cost_sensitivity` | improved 4 + improved 6 | Adds year-keyed transaction-cost schedule | Cost-sensitivity study, not a new strategy |
| 8 | `improved_8_equal_weight_top20` | improved 4 | top-N 10→20, sizing → 5%-of-equity | Wealth-maximizer; lower Sharpe than improved 4 |
| 9 | `improved_9_vol_targeted_top20` | improved 8 | sizing → inverse-vol weighted (top-20) | Risk-budget sizing; low-vol names get more capital |

## Execution Model

Initial capital: `$1,000,000`. Commission: 0 in baseline backtests (transaction costs added as sensitivity layer in improved 7). Monthly rebalance.

| Variant | Backtrader engine | Sizer | Stop/take exits |
|---|---|---|---|
| base, improved 1 | `MonthlySignalStrategy` | `FixedCashSizer` ($100k) | None |
| improveds 2, 3, 4, 5, 6 | `DailySignalStopTakeStrategy` | `FixedCashSizer` ($100k) | Native `bt.Order.Stop` + `bt.Order.Limit` (OCO-linked) |
| improved 7 | Vector-only cost sensitivity (uses existing improved 4 / 6 signals) | n/a | Inherited |
| improved 8 | `DailySignalStopTakeStrategy` | `EquityPercentSizer` (5% per position) | Native `bt.Order.Stop` + `bt.Order.Limit` (OCO-linked) |
| improved 9 | `DailySignalStopTakeStrategy` | `VolatilityTargetedSizer` (5% nominal, scaled by median_vol/stock_vol) | Native `bt.Order.Stop` + `bt.Order.Limit` (OCO-linked) |

For risk-managed strategies, rebalance exits cancel live protective orders before submitting market exits. This prevents stale stop/limit orders from firing after a position has already been closed.

## Robustness Testing

- **Monte Carlo**: 1,000 random portfolios per strategy, sampled from the same eligible universe with matching top-N, sizing, stop/take, and regime-filter rules. p-value computed over the common evaluation window.
- **Block bootstrap**: 1,000 resamples per strategy, 6-month blocks, applied to the strategy's monthly return series in the eval window.
- **Walk-forward**: train (eval window through 2020-12) vs test (2021+). Computed for all 9 strategies via `aggregate_all_strategies.py`.
- **Time-varying transaction-cost sensitivity** (improved 7): zero / central (per-year estimates from Frazzini-Israel-Moskowitz 2018, ITG/Virtu, JPM, NYSE TAQ) / pessimistic (2× central) scenarios.
- **Multi-comparison correction** (`src/run_multi_comparison_test.py`): Hansen (2005) SPA test and Romano-Wolf (2005) StepM step-down procedure correct for the multi-comparison problem (9 strategies tested → family-wise error rate inflation). Uses `arch.bootstrap` stationary block bootstrap with 10,000 reps. Results in `results/robustness/`. See `docs/MULTI_COMPARISON_TEST.md`.

## Main Files

- `src/run_project.py` — full reproducible pipeline (base + improved 1-3).
- `src/run_improved_4_stop_take_sensitivity.py` — focused improved 4 grid + selected candidate.
- `src/run_improved_5_regime_filter.py` — focused improved 5.
- `src/run_improved_6_hzz_trend.py` — focused improved 6 HZZ.
- `src/run_improved_7_costs.py` — focused improved 7 cost sensitivity.
- `src/run_improved_8_top_n_sizing.py` — focused improved 8 equal-weight top-20.
- `src/run_improved_9_vol_targeted.py` — focused improved 9 volatility-targeted sizing.
- `src/aggregate_all_strategies.py` — unified cross-strategy summary tables (metrics, walk-forward, MC, benchmark alpha) for all 9 strategies.
- `src/run_multi_comparison_test.py` — Hansen SPA + Romano-Wolf StepM multi-comparison robustness test.
- `src/make_presentation_figures.py` — builds 15+ presentation figures and summary tables from all 9 strategy outputs.
- `requirements.txt` — pinned runtime.
- `data/processed/factor_panel.csv` — final factor panel.
- `results/` — saved FMP, IC, staged strategy, focused improveds 4-9, Backtrader, Monte Carlo, validation, comparison, and robustness outputs.
- `figures/` — saved plots.
- `presentation/sp500_factor_investing_presentation.pdf` — generated deck.
- `docs/SIZING_AND_MARGIN.md` — explains sizer mechanics, margin rejections, and long-only money growth.
- `docs/MULTI_COMPARISON_TEST.md` — methodology and results of Hansen SPA + Romano-Wolf multi-comparison correction (auto-generated).
