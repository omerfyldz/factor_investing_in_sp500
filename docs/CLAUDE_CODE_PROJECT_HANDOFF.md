# Claude Code Project Handoff

Last updated: 2026-06-09

Project root:

```text
C:\Users\asus\Desktop\sp500_factor_investing
```

This file is the detailed working handoff for continuing the S&P 500 factor investing project in Claude Code. It is intentionally explicit and repetitive because the next agent should be able to recover the project state, reasoning, latest results, and remaining risks without relying on chat history.

## 1. Current State In One Paragraph

The project is a standalone S&P 500 factor investing research project. It uses a frozen current-constituent S&P 500 stock panel, frozen Yahoo Finance `^GSPC` index data, monthly quality/value/momentum/trend factors, factor-mimicking portfolio analysis, Backtrader trading strategies, Monte Carlo random-portfolio tests, block bootstrap, walk-forward diagnostics, figures, and a generated PDF presentation. The project has a sequential strategy ladder: base, improved 1, improved 2, improved 3, improved 4, and improved 5. The most recent critical change was replacing the old manual stop/take threshold exit approximation with native Backtrader orders: `bt.Order.Market` for entries and rebalances, `bt.Order.Stop` for stop-loss, and `bt.Order.Limit` for take-profit. The full project pipeline and focused improved 4/5 scripts were rerun successfully after this order-model change.

## 2. Important Collaboration Rules

- Do not commit or push automatically. The user explicitly asked that commits happen only when commanded.
- Keep the GitHub-facing `README.md` independent. It should not describe the work as a course submission or rely on course wording.
- Internal docs may still discuss assignment alignment because the research design intentionally follows a required factor-investing structure.
- Preserve the staged improvement history. Do not delete older strategies just because a later strategy looks better or worse.
- Future improvements should be one-change-at-a-time so the try/fail path stays scientifically readable.
- Do not rerun the full pipeline casually. It is slow. On the latest run, `src\run_project.py` took about 56 minutes, improved 4 took about 36 minutes, and improved 5 took about 21 minutes.

## 3. Latest Commands Run

These commands were run from the project root:

```powershell
py -3.10 src\run_project.py
py -3.10 src\run_improved_4_stop_take_sensitivity.py
py -3.10 src\run_improved_5_regime_filter.py
```

All three completed successfully.

Before the run, these checks also passed:

```powershell
py -3.10 -m py_compile src\run_improved_4_stop_take_sensitivity.py src\run_base_strategy.py src\project_core.py src\compare_strategies.py src\run_improved_strategy.py src\run_improved_5_regime_filter.py src\run_project.py src\__init__.py
py -3.10 -c "import sys; sys.path.insert(0, 'src'); import project_core; print('project_core import ok')"
git diff --check
```

`git diff --check` only produced normal Windows LF/CRLF warnings, not whitespace errors.

## 4. Data Scope And Cutoff

All analysis is filtered to observations available on or before `2026-05-31`.

Latest validation summary:

```text
prices_cutoff:                     OK, 2026-05-29
daily_fundamentals_cutoff:         OK, 2026-05-29
statements_date_available_cutoff:  OK, 2026-05-29
panel_cutoff:                      OK, 2026-05-31
benchmark_cutoff:                  OK, 2026-05-29
presentation_pdf:                  OK
```

Raw data audit:

```text
prices:                 2,292,658 rows, 503 tickers, 2006-06-05 to 2026-05-29
daily_fundamentals:     2,295,967 rows, 499 tickers, 2006-06-05 to 2026-05-29
statements:             3,458,978 rows, 499 tickers, 2005-04-29 to 2026-05-29
constituents:             503 rows,   503 tickers
sp500_index_yahoo:          5,030 rows, 2006-06-01 to 2026-05-29
factor_panel:             109,492 rows, 503 tickers, 2006-06-30 to 2026-05-31
```

Raw input files:

```text
data/raw/sp500_prices_long.csv
data/raw/sp500_fundamentals_daily_long.csv
data/raw/sp500_fundamentals_statements_long.csv
data/raw/sp500_constituents.csv
data/raw/sp500_index_yahoo.csv
```

Large raw CSVs are tracked with Git LFS.

## 5. Factor Construction

The project constructs four main factors:

```text
roe_z        Higher ROE is better.
pe_z         Lower positive P/E is better, implemented by multiplying positive P/E by -1 before z-scoring.
momentum_z   12-month adjusted-price momentum, P_t / P_t-12 - 1.
trend_z      Base trend factor from full-sample ^GSPC moving-average predictive regression.
```

The improved trend factor is:

```text
trend_expanding_z
```

`trend_expanding_z` is more conservative because it estimates the trend regression using only past index data available before each signal month.

Preprocessing:

- Use month-end stock observations.
- Lag signals before returns/trades.
- Winsorize outliers at 1st/99th percentiles.
- Cross-sectionally standardize factors with z-scores.
- Composite scores require at least three valid factor values.

Base trend regression:

- Uses `^GSPC` as the analog index.
- Uses normalized moving-average deviations.
- Runs one full-sample predictive regression.
- Drops insignificant variables at the 5% cutoff.
- Applies retained coefficients to each stock's moving-average deviations.

Improved trend regression:

- Uses expanding regressions.
- Reduces look-ahead bias.
- Should be treated as a methodology improvement, not automatically a performance improvement.

## 6. Strategy Ladder

The current active strategy ladder is:

```text
Base:
  base_equal_top10
  Full-sample trend_z
  Equal factor weights
  Top 10 stocks
  No stop-loss/take-profit
  Monthly Backtrader bt.Order.Market orders

Improved 1:
  improved_1_expanding_trend_top10
  Same as base, but trend_expanding_z replaces trend_z
  No stop-loss/take-profit
  Monthly Backtrader bt.Order.Market orders

Improved 2:
  improved_2_expanding_trend_stop_take_top10
  Builds on improved 1
  Adds 10% stop-loss and 20% take-profit
  Daily Backtrader bt.Order.Market entries/rebalances
  Daily Backtrader bt.Order.Stop and bt.Order.Limit protective exits

Improved 3:
  improved_3_dynamic_ic_weights_stop_take_top10
  Builds on improved 2
  Adds rolling rank-IC dynamic factor weights
  60-month lookback, 24-month minimum history
  50% shrinkage to equal weights
  Factor weights capped between 10% and 45%
  Daily Backtrader native stop/limit protective exits

Improved 4:
  improved_4_walkforward_stop_take_top10
  Branches from improved 2, not improved 3
  Keeps static equal factor weights
  Tests stop/take grid using training data through 2020
  Selects 5% stop-loss and 30% take-profit
  Daily Backtrader native stop/limit protective exits

Improved 5:
  improved_5_regime_filtered_stop_take_top10
  Branches from improved 4
  Keeps 5% stop-loss and 30% take-profit
  Adds fixed ^GSPC 10-month SMA regime filter
  Holds cash when regime filter is off
  Rejected because it worsened performance and Monte Carlo robustness
```

The current accepted practical risk-managed variant is improved 4. Improved 5 is a documented failed experiment.

## 7. Backtrader Execution Model

This is the most important recent change.

Base and improved 1:

```python
self.buy(..., exectype=bt.Order.Market)
self.close(..., exectype=bt.Order.Market)
```

Improved 2, 3, 4, and 5:

```python
self.buy(..., exectype=bt.Order.Market)
self.sell(..., exectype=bt.Order.Market)  # rebalance exits
self.sell(..., exectype=bt.Order.Stop, price=stop_price)
self.sell(..., exectype=bt.Order.Limit, price=limit_price, oco=stop_order)
```

Native order design:

- A market entry is submitted when a stock enters the top-N signal list.
- After the market buy completes, protective stop and limit sell orders are submitted.
- Stop-loss is a native `bt.Order.Stop`.
- Take-profit is a native `bt.Order.Limit`.
- When both stop and limit are present, the limit order is linked OCO to the stop order.
- If a protective exit fills, sibling protective orders are canceled.
- If the strategy rebalances out of a position, live protective orders are canceled before a market exit is submitted.
- This avoids stale stop/limit orders creating accidental short positions after the position is already closed.
- Long-only validation is run after Backtrader output generation.

Backtrader order `exectype` audit from the latest run:

```text
0 = Market
2 = Limit
3 = Stop
```

Latest order logs show:

```text
base:        exectype 0 only
improved 1:  exectype 0 only
improved 2:  exectype 0, 2, 3
improved 3:  exectype 0, 2, 3
improved 4:  exectype 0, 2, 3
improved 5:  exectype 0, 2, 3
```

Long-only validation:

```text
base min position size:        35
improved 1 min position size:  28
improved 2 min position size:  28
improved 3 min position size:  30
improved 4 min position size:  28
improved 5 min position size:  28
```

No negative positions were found.

## 8. Important Backtrader Caveats

Some entry orders have `Margin` status. This is not currently treated as a project bug.

Reason:

- Initial capital is `1,000,000`.
- Fixed cash per trade is `100,000`.
- Top-N is 10.
- A nominal 10 positions times `100,000` equals all capital.
- Integer sizing and next-bar execution prices can push requested trade value slightly above available cash.
- Backtrader then rejects some buy orders as `Margin`.

This is acceptable under the current fixed-cash design, but it means the realized number of holdings can be slightly below the target. Do not "fix" it by changing fixed cash sizing unless the user explicitly approves, because the user currently wants to keep fixed `100,000`.

## 9. Latest Vector Strategy Results

From `results/comparison/strategy_stage_metrics.csv`:

```text
base_equal_top10
  final equity:       5,319,947
  total return:       431.99%
  annualized Sharpe:  1.1317
  max drawdown:      -10.80%
  Monte Carlo p:      0.0000

improved_1_expanding_trend_top10
  final equity:       3,510,333
  total return:       251.03%
  annualized Sharpe:  0.8184
  max drawdown:      -15.78%
  Monte Carlo p:      0.0210

improved_2_expanding_trend_stop_take_top10
  final equity:       2,721,443
  total return:       172.14%
  annualized Sharpe:  0.7310
  max drawdown:      -13.03%
  Monte Carlo p:      0.0610

improved_3_dynamic_ic_weights_stop_take_top10
  final equity:       2,635,221
  total return:       163.52%
  annualized Sharpe:  0.6788
  max drawdown:      -14.13%
  Monte Carlo p:      0.1710

improved_4_walkforward_stop_take_top10
  final equity:       2,751,551
  total return:       175.16%
  annualized Sharpe:  0.7968
  max drawdown:       -7.36%
  Monte Carlo p:      0.0560

improved_5_regime_filtered_stop_take_top10
  final equity:       2,148,104
  total return:       114.81%
  annualized Sharpe:  0.6525
  max drawdown:      -11.19%
  Monte Carlo p:      0.6030
```

Interpretation:

- The base strategy has the strongest vector result, but it uses full-sample trend coefficients and is less conservative.
- Improved 1 is methodologically cleaner but weaker.
- Improved 2 adds risk exits, but lowers vector Sharpe versus improved 1.
- Improved 3 dynamic IC weights do not improve improved 2.
- Improved 4 is the best risk-managed branch so far because it improves drawdown materially and has better Sharpe than improved 2/3.
- Improved 5 fails. The regime filter removed too much exposure and worsened Monte Carlo p-value.

## 10. Latest Backtrader Results

From latest regenerated Backtrader metrics:

```text
base
  final value:        5,778,291
  annualized Sharpe:  1.2023
  max drawdown:      -10.42%
  exectype:           Market only

improved 1
  final value:        3,768,578
  annualized Sharpe:  0.8403
  max drawdown:      -17.17%
  exectype:           Market only

improved 2
  final value:        2,891,391
  annualized Sharpe:  0.8860
  max drawdown:      -13.54%
  exectype:           Market, Limit, Stop

improved 3
  final value:        2,680,533
  annualized Sharpe:  0.8526
  max drawdown:      -15.51%
  exectype:           Market, Limit, Stop

improved 4
  final value:        2,816,266
  annualized Sharpe:  0.9852
  max drawdown:       -7.69%
  exectype:           Market, Limit, Stop

improved 5
  final value:        2,109,769
  annualized Sharpe:  0.7493
  max drawdown:      -10.64%
  exectype:           Market, Limit, Stop
```

Important: improved 4 and improved 5 Backtrader values changed after switching from manual threshold checks to native Backtrader stop/limit orders. That change is expected and should be described honestly.

Old threshold-exit values are no longer the current executable evidence.

## 11. Latest Improved 4 Details

Improved 4 was run with:

```text
selected stop-loss:    5%
selected take-profit:  30%
selection data:        training observations through 2020-12-31
selection logic:       training Sharpe penalized for training drawdown and isolated parameter peaks
```

Latest summary:

```text
vector Sharpe:              0.7968
vector final equity:        2,751,551
vector max drawdown:       -7.36%
Monte Carlo p-value:        0.0560
Backtrader final value:     2,816,266
Backtrader Sharpe:          0.9852
Backtrader max drawdown:   -7.69%
long-only validation:       OK
```

Interpretation:

- Improved 4 is the best current risk-managed branch.
- It does not beat the base on raw performance.
- It does materially reduce drawdown versus base and improved 1/2/3.
- Its Monte Carlo p-value is borderline, not decisive.
- It should be described as promising but not real-money ready.

## 12. Latest Improved 5 Details

Improved 5 was run with:

```text
foundation:           improved 4
stop-loss:            5%
take-profit:          30%
regime filter:        trade only when ^GSPC month-end close > 10-month SMA
regime optimization:  none
```

Latest comparison:

```text
improved_4_vector
  Sharpe:        0.7968
  final equity:  2,751,551
  max drawdown: -7.36%

improved_5_vector
  Sharpe:        0.6525
  final equity:  2,148,104
  max drawdown: -11.19%

improved_4_backtrader
  Sharpe:        0.9852
  final value:   2,816,266
  max drawdown: -7.69%

improved_5_backtrader
  Sharpe:        0.7493
  final value:   2,109,769
  max drawdown: -10.64%

improved 5 Monte Carlo p-value: 0.6030
```

Conclusion:

- Improved 5 is rejected.
- The regime filter likely removed exposure during rebound or recovery periods.
- It worsened Sharpe, final value, drawdown, and Monte Carlo robustness versus improved 4.
- Keep it as a failed experiment in the history, but do not build the next accepted strategy on improved 5 unless the user explicitly asks.

## 13. Factor-Mimicking Portfolio Results

From `results/fmp_analysis/fmp_performance_summary.csv`:

Portfolio-sort FMPs:

```text
momentum
  Sharpe:        0.0792
  avg IC:        0.0209
  avg rank IC:   0.0042

P/E
  Sharpe:        0.2051
  avg IC:       -0.0084
  avg rank IC:   0.0118

ROE
  Sharpe:       -0.2064
  avg IC:       -0.0016
  avg rank IC:   0.0038

trend
  Sharpe:        0.3602
  avg IC:        0.0191
  avg rank IC:   0.0224
  rank IC t:     2.1792
```

Cross-sectional regression FMPs:

```text
momentum
  Sharpe:        0.1987

P/E
  Sharpe:       -0.2403

ROE
  Sharpe:       -0.2055

trend
  Sharpe:        0.3899
```

Interpretation:

- Trend is the most consistently positive factor in the FMP analysis.
- Rank IC for trend is the strongest among the four factors.
- Momentum is weak but not useless.
- ROE and P/E are not strongly positive in this dataset.
- This helps explain why dynamic factor weighting is noisy and did not clearly improve results.

## 14. Walk-Forward Results

From `results/comparison/walk_forward_summary.csv`:

```text
base_equal_top10
  train Sharpe to 2020:       1.1072
  test Sharpe 2021-2026:      1.3003
  test cumulative return:     46.46%
  test max drawdown:          -3.61%
  selected by train:          True

improved_1_expanding_trend_top10
  train Sharpe to 2020:       0.5704
  test Sharpe 2021-2026:      1.3359
  test cumulative return:     103.12%
  test max drawdown:          -5.24%

improved_2_expanding_trend_stop_take_top10
  train Sharpe to 2020:       0.6269
  test Sharpe 2021-2026:      0.9588
  test cumulative return:     58.54%
  test max drawdown:          -5.75%

improved_3_dynamic_ic_weights_stop_take_top10
  train Sharpe to 2020:       0.5412
  test Sharpe 2021-2026:      0.9842
  test cumulative return:     57.43%
  test max drawdown:          -10.47%
```

Important nuance:

- Walk-forward on the first four staged strategies selected the base by pre-2021 Sharpe.
- Improved 1 had strong post-2020 test Sharpe even though it was not selected by train Sharpe.
- Improved 4 is a separate focused stop/take branch and is not included in that original four-strategy walk-forward table.

## 15. Monte Carlo Interpretation

Monte Carlo p-value definition:

```text
p-value = fraction of random portfolios whose Sharpe >= strategy Sharpe
```

The random portfolios use:

- the same universe,
- the same rebalance schedule,
- the same number of positions,
- the same fixed-cash sizing logic,
- the same stop/take assumptions where relevant.

Latest p-values:

```text
base:        0.0000
improved 1:  0.0210
improved 2:  0.0610
improved 3:  0.1710
improved 4:  0.0560
improved 5:  0.6030
```

Interpretation:

- Base is strongly better than random portfolios under this Monte Carlo design.
- Improved 1 is also fairly strong.
- Improved 2 and improved 4 are borderline.
- Improved 3 is weaker.
- Improved 5 is bad from a robustness standpoint.
- A high p-value does not prove the strategy is bad; it means the selected strategy is not clearly distinguishable from random selection under the simulated benchmark.
- Monte Carlo is still imperfect because the universe has survivorship bias and no transaction costs/slippage.

## 16. Generated Outputs

Important result folders:

```text
results/base_strategy
results/improved_strategy
results/improved_strategy_2
results/improved_strategy_3
results/improved_strategy_4
results/improved_strategy_5
results/comparison
results/fmp_analysis
```

Important processed data:

```text
data/processed/factor_panel.csv
data/processed/monthly_stock_bars.csv
data/processed/monthly_sp500_index.csv
data/processed/monthly_roe_asof.csv
```

Important figures:

```text
figures/factor_portfolio_cumulative_returns.png
figures/factor_regression_cumulative_returns.png
figures/ic_rank_ic_summary.png
figures/improved4_stop_take_sensitivity_heatmap.png
figures/monte_carlo_sharpe_histogram.png
figures/strategy_drawdowns.png
figures/strategy_equity_vs_benchmark.png
figures/strategy_improvement_sharpe.png
```

Presentation:

```text
presentation/sp500_factor_investing_presentation.pdf
```

## 17. Source Code Map

Primary file:

```text
src/project_core.py
```

Important parts of `project_core.py`:

```text
Global constants
  INITIAL_CASH = 1_000_000.0
  CASH_PER_TRADE = 100_000.0
  CUTOFF_DATE = 2026-05-31

Data functions
  load_raw_data()
  build_monthly_stock_bars()
  build_monthly_ratio_panel()
  build_monthly_roe_asof()
  build_index_panel()

Trend functions
  index_ma_features()
  estimate_full_sample_trend_regression()
  estimate_expanding_trend()
  compute_stock_trend_scores()

Factor panel
  assemble_factor_panel()
  winsorize_by_month()
  zscore_by_month()

FMP analysis
  construct_fmp_portfolio_sort()
  construct_fmp_regression()
  factor_information_coefficients()

Vector strategies
  score_for_spec()
  select_positions_for_spec()
  simulate_vector_strategy()
  stop_take_return()

Backtrader
  FixedCashSizer
  MonthlySignalStrategy
  DailySignalStopTakeStrategy
  run_backtrader()
  run_backtrader_daily_stop_take()
  assert_backtrader_long_only()

Robustness
  monte_carlo_random_portfolios()
  block_bootstrap()
  walk_forward_summary()

Reporting
  make_figures()
  write_strategy_history()
  write_project_docs()
  make_presentation()
  main()
```

Entry scripts:

```text
src/run_project.py
  Full reproducible pipeline. Slow.

src/run_base_strategy.py
  Focused base rerun from processed data.

src/run_improved_strategy.py
  Focused improved 1, 2, and 3 rerun.

src/run_improved_4_stop_take_sensitivity.py
  Focused improved 4 stop/take grid and selected candidate.

src/run_improved_5_regime_filter.py
  Focused improved 5 regime filter.

src/compare_strategies.py
  Rebuilds comparison tables.
```

## 18. Reproducibility Commands

Install dependencies:

```powershell
py -3.10 -m pip install -r requirements.txt
```

Full run:

```powershell
py -3.10 src\run_project.py
```

Focused runs:

```powershell
py -3.10 src\run_base_strategy.py
py -3.10 src\run_improved_strategy.py
py -3.10 src\run_improved_4_stop_take_sensitivity.py
py -3.10 src\run_improved_5_regime_filter.py
```

Recommended after code changes:

```powershell
py -3.10 -m py_compile src\run_improved_4_stop_take_sensitivity.py src\run_base_strategy.py src\project_core.py src\compare_strategies.py src\run_improved_strategy.py src\run_improved_5_regime_filter.py src\run_project.py src\__init__.py
py -3.10 -c "import sys; sys.path.insert(0, 'src'); import project_core; print('project_core import ok')"
```

## 19. Current Git Status Expectations

After the latest run and handoff work, many files are expected to be modified:

- source code for native order changes,
- docs,
- README,
- presentation PDF,
- Backtrader outputs for improved 2, 3, 4, and 5,
- validation summaries.

Do not treat this as accidental churn. The output changes are expected because native stop/limit orders produce different executable results than the older manual threshold exit approximation.

Do not commit until the user explicitly asks.

## 20. Known Limitations

The project is not live-trading ready.

Main limitations:

- Current S&P 500 constituents create survivorship bias.
- No transaction costs.
- No slippage.
- No borrow constraints or shorting in the trading strategy.
- Daily OHLC data cannot know exact intraday path.
- Backtrader stop/limit orders are more realistic than manual threshold checks, but still bar-based.
- Fundamentals vendor definitions may affect ROE and P/E quality.
- Some Backtrader entry orders get `Margin` status because fixed cash sizing is exactly `100,000` with `1,000,000` starting capital and top 10 names.
- Improved 4 involves parameter selection, so it needs cautious interpretation.
- Monte Carlo tests are useful but not a full data-snooping correction.
- White Reality Check or Hansen SPA is not implemented.

## 21. What Not To Change Accidentally

Do not change these unless the user explicitly asks:

- `INITIAL_CASH = 1_000_000.0`
- `CASH_PER_TRADE = 100_000.0`
- `commission=0.0`
- Base uses full-sample `trend_z`
- Improved 1 uses `trend_expanding_z`
- Improved 2 uses 10% stop and 20% take
- Improved 4 selected pair is 5% stop and 30% take
- Improved 5 remains a failed documented experiment
- GitHub README should stay independent in tone
- Raw data cutoff should stay `2026-05-31`

## 22. Recommended Next Improvements

The next improvement should build from improved 4, because improved 4 is currently the best accepted risk-managed branch and improved 5 was rejected.

Recommended next sequence:

```text
Improved 6:
  Top-N sensitivity on top of improved 4.
  Test top 15 and top 20 while keeping fixed cash per trade at 100,000.
  Important: with 1,000,000 initial cash and 100,000 per trade, top 20 cannot hold 20 names at inception.
  Therefore top 20 means "select up to top 20 subject to available equity and fixed-cash sizing", not force 20 holdings.
  This may improve diversification after equity grows, but early portfolio will still be cash constrained.

Improved 7:
  Add transaction-cost and slippage robustness.
  This should be a diagnostic layer, not a replacement for zero-commission baseline.
  Keep baseline results unchanged.

Improved 8:
  Sector concentration diagnostics or sector caps.
  First measure sector exposure; only then add caps.
  Avoid changing factor logic and sector rules in the same step.

Improved 9:
  Historical S&P 500 membership if data can be sourced.
  This directly attacks survivorship bias.

Improved 10:
  White Reality Check or Hansen SPA over all tried variants.
  This addresses data snooping from trying multiple improvements.
```

Suggested immediate next step:

```text
Create improved_6_topn_sensitivity_from_improved_4.
Keep:
  trend_expanding_z
  equal factor weights
  5% stop-loss
  30% take-profit
  Backtrader native stop/limit orders
  fixed 100,000 per trade
  commission 0

Change only:
  top_n values, for example 10, 15, 20

Decision rule:
  Use training data through 2020 for selection if selecting a parameter.
  Report test-period metrics separately.
  Keep improved 4 as benchmark.
```

## 23. How To Explain The Current Conclusion

Use a cautious conclusion:

```text
The project shows that a four-factor S&P 500 strategy can produce strong backtested results, especially under the base full-sample trend design. However, the more conservative no-lookahead and risk-managed variants are weaker. Improved 4 is the best practical risk-managed version so far because it lowers drawdown meaningfully and keeps reasonable Sharpe, but its Monte Carlo p-value is borderline. Improved 5 failed, showing that adding a blunt market-regime filter can remove too much useful exposure. We would not treat the strategy as real-money ready without historical membership, transaction costs, slippage, stronger data-snooping corrections, and additional out-of-sample testing.
```

Do not oversell improved 4:

- It is promising.
- It is the best risk-managed variant so far.
- It is not proof of a tradable edge.
- It should not replace the need for better data and robustness.

## 24. Last Critical Technical Change

The last technical issue was whether stop-loss/take-profit should be manual threshold logic or native Backtrader orders.

Final decision:

```text
Use native Backtrader orders.

Entries and rebalances:
  bt.Order.Market

Stop-loss:
  bt.Order.Stop

Take-profit:
  bt.Order.Limit
```

Why:

- It matches the intended Backtrader order model.
- It is more defensible than manually checking OHLC thresholds and then issuing market exits.
- It keeps the base strategy simple and market-order-only.
- It lets improvements use realistic order types without confusing the base.

Implementation nuance:

- Protective orders are submitted after market entry completion because the actual entry price is needed.
- Rebalance exits cancel protective orders before market selling.
- OCO is used between stop and limit where both exist.
- Long-only validation passed after the latest run.

## 25. Archive Note

The user asked to compress the project after this handoff. The archive should be created outside the project folder, not inside it, and should exclude `.git`, `__pycache__`, and `.pytest_cache`. It should include source, docs, data, results, figures, notebooks if present, requirements, README, and presentation.

