# S&P 500 Factor Investing

A reproducible four-factor research project on the S&P 500. The pipeline builds quality, value, momentum, and trend signals from a frozen current-constituent panel, runs them through a staged ladder of long-only top-10 strategies, validates each variant with Monte Carlo and walk-forward tests, and executes the final candidates in Backtrader with native protective orders.

The project is deliberately layered: each improvement changes exactly one design dimension on top of the previous accepted variant so the try/fail path stays scientifically readable.

---

## Research Context

The trend factor follows Han, Zhou, and Zhu (2016, *Journal of Financial Economics* — "A Trend Factor: Any Economic Gains from Using Information over Investment Horizons?"), which combines short-, intermediate-, and long-term price information through normalized moving averages across 11 horizons (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000 trading days).

The project implements two different trend procedures and reports both:

- **Index-level trend regression** (`trend_z`, `trend_expanding_z`) — a single time-series regression of `^GSPC` next-month return on `^GSPC`'s own normalized MA deviations, with insignificant variables dropped at the 5% level. The base uses one full-sample fit; improvement 1 uses an expanding no-lookahead fit. Coefficients are then applied to every stock's MA signals. This is the project's literal trend factor used in improvements 1–5.
- **Cross-sectional HZZ trend factor** (`trend_hzz_z`) — the paper's actual methodology. Each month a cross-sectional OLS regresses every eligible stock's next-month return on its 11 normalized MA ratios; the resulting monthly coefficient vector is smoothed with a strict 12-month trailing mean; per-stock predicted returns combine the smoothed coefficients with the stock's contemporaneous MA ratios. Used in improvement 6.

Both signals coexist in the panel so any future strategy can opt into either trend formulation.

---

## Data

Frozen through **2026-05-31**. Any rows after that cutoff are dropped at load time.

| File | Source | Description |
|---|---|---|
| `data/raw/sp500_prices_long.csv` | Tiingo | Daily OHLCV with split/dividend-adjusted columns for 503 current S&P 500 tickers |
| `data/raw/sp500_fundamentals_daily_long.csv` | Tiingo | Daily point-in-time market cap, P/E, P/B, trailing PEG |
| `data/raw/sp500_fundamentals_statements_long.csv` | Tiingo | Statement fundamentals dated by `date_available` (the day the filing actually became public), avoiding look-ahead from filing dates |
| `data/raw/sp500_constituents.csv` | Wikipedia snapshot | Current ticker → security / GICS sector / date added |
| `data/raw/sp500_index_yahoo.csv` | Yahoo Finance `^GSPC` | Frozen benchmark daily OHLCV |

Large raw CSVs are tracked with Git LFS.

Validation summary at the latest run:

```text
prices_cutoff:                    OK, 2026-05-29
daily_fundamentals_cutoff:        OK, 2026-05-29
statements_date_available_cutoff: OK, 2026-05-29
panel_cutoff:                     OK, 2026-05-31
benchmark_cutoff:                 OK, 2026-05-29
prices:               2,292,658 rows, 503 tickers, 2006-06-05 to 2026-05-29
daily_fundamentals:   2,295,967 rows, 499 tickers, 2006-06-05 to 2026-05-29
statements:           3,458,978 rows, 499 tickers, 2005-04-29 to 2026-05-29
factor_panel:           109,492 rows, 503 tickers, 2006-06-30 to 2026-05-31
```

---

## Factor Construction

| Factor | Formula | Z-score column |
|---|---|---|
| Quality (ROE) | Latest available point-in-time ROE from statement filings | `roe_z` |
| Value (P/E inverted) | `-1 × pe_ratio` for positive P/E only (so higher = cheaper) | `pe_z` |
| Momentum | `P_t / P_{t-12} − 1` | `momentum_z` |
| Trend (assignment) | Index-level full-sample predictive regression coefficients applied to each stock's MA deviations | `trend_z` |
| Trend (no-lookahead) | Same as above but coefficients from expanding `^GSPC` regressions using only past data | `trend_expanding_z` |
| Trend (HZZ cross-sectional) | Monthly cross-sectional regression of next-month returns on 11 normalized MA ratios; 12-month strict-trailing β smoothing; per-stock predicted return | `trend_hzz_z` |

All factors are computed at month-end, lagged one month before use, winsorized at the 1st/99th percentiles within each month, and standardized cross-sectionally. The composite score is the mean of available z-scores; a stock must have at least three of the four factors valid to receive a composite score.

---

## Strategy Ladder

Each improvement changes exactly one dimension on top of the previous accepted variant. Bold rows are the honest finalists; base has full-sample look-ahead and is reported because the assignment prescribes it.

| Strategy | Trend signal | Stop / Take | Weights | Regime filter | Notes |
|---|---|---|---|---|---|
| `base_equal_top10` | Full-sample index trend (`trend_z`) | none | equal | off | Assignment baseline; monthly Backtrader market orders |
| `improved_1_expanding_trend_top10` | Expanding no-lookahead index trend (`trend_expanding_z`) | none | equal | off | Removes look-ahead from the trend regression |
| `improved_2_expanding_trend_stop_take_top10` | `trend_expanding_z` | 10% / 20% | equal | off | Adds risk exits; daily Backtrader with native stop/limit orders |
| `improved_3_dynamic_ic_weights_stop_take_top10` | `trend_expanding_z` | 10% / 20% | rolling rank-IC (60m lookback, 50% shrinkage, 10–45% caps) | off | Dynamic weights; rejected as no improvement |
| **`improved_4_walkforward_stop_take_top10`** | `trend_expanding_z` | **5% / 30%** (walk-forward selected on train ≤ 2020) | equal | off | Best risk-managed branch; lowest drawdown of the ladder |
| `improved_5_regime_filtered_stop_take_top10` | `trend_expanding_z` | 5% / 30% | equal | **on** (`^GSPC` > 10-month SMA) | Rejected; cash filter destroyed too much exposure |
| **`improved_6_hzz_cross_sectional_trend_stop_take_top10`** | **HZZ cross-sectional (`trend_hzz_z`)** | 5% / 30% | equal | off | Implements the paper's actual trend methodology; best terminal wealth and lowest Monte Carlo p among honest variants |

---

## Execution Model

| Component | Setting |
|---|---|
| Initial cash | `$1,000,000` |
| Sizing | `FixedCashSizer`, `$100,000` per trade |
| Top-N target | 10 stocks (realized average 4–8 because integer sizing + next-bar pricing produces some `Margin` rejections; the run-time portfolio holds *up to* top 10) |
| Commission | `0.0` |
| Slippage | none in the current run (cost layer is on the future-work list) |
| Rebalance | monthly at month-end signal → next-month-open execution |
| Entries / rebalance exits | `bt.Order.Market` |
| Stop-loss | `bt.Order.Stop` (improvements 2–6) |
| Take-profit | `bt.Order.Limit`, linked OCO to the stop order (improvements 2–6) |
| Engine | `MonthlySignalStrategy` (base, improved 1) or `DailySignalStopTakeStrategy` (improved 2–6) |

Protective stop/limit orders are submitted only after the market entry completes, because the realized fill price is required to set them. Rebalance exits cancel any live protective orders first, so stale stop/limit orders cannot accidentally create short positions. A long-only assertion runs after every Backtrader output.

---

## Results

### Full comparison

| Strategy | Vec Sharpe | BT Sharpe | Vec Max DD | BT Max DD | Vec Final Eq | BT Final | MC p | Avg Pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | +1.13 | +1.20 | -10.8% | -10.4% | $5.32M | $5.78M | 0.000 | 8.0 |
| improved_1 | +0.82 | +0.84 | -15.8% | -17.2% | $3.51M | $3.77M | 0.021 | 4.3 |
| improved_2 | +0.73 | +0.89 | -13.0% | -13.5% | $2.72M | $2.89M | 0.061 | 4.3 |
| improved_3 | +0.68 | +0.85 | -14.1% | -15.5% | $2.64M | $2.68M | 0.171 | 4.3 |
| **improved_4** | +0.80 | **+0.99** | **-7.4%** | **-7.7%** | $2.75M | $2.82M | 0.056 | 4.3 |
| improved_5 | +0.65 | +0.75 | -11.2% | -10.6% | $2.15M | $2.11M | 0.603 | 3.5 |
| **improved_6** | **+0.81** | +0.84 | -11.2% | -11.3% | **$3.05M** | **$3.16M** | **0.039** | **7.5** |

Bold cells = best in column among honest variants (base excluded because of full-sample trend look-ahead).

### Two honest finalists

| Optimize for | Winner |
|---|---|
| Lowest drawdown | improved 4 (-7.7% Backtrader) |
| Highest terminal wealth | improved 6 ($3.16M Backtrader) |
| Highest Backtrader Sharpe | improved 4 (0.99) |
| Highest vector Sharpe | improved 6 (0.81) |
| Best Monte Carlo p-value | improved 6 (0.039) |
| Most diversified (positions held) | improved 6 (7.5) |
| Paper-faithful methodology | improved 6 (true HZZ cross-sectional trend) |
| Assignment-faithful methodology | improved 4 (composite around index trend) |

Improved 4 is the cautious investor's choice. Improved 6 is the academically faithful choice. They share the same risk-exit infrastructure and differ only in the trend signal.

### HZZ trend factor diagnostics (improved 6)

- 192 monthly cross-section regressions estimated (one per month, 2007-01 to 2026-05 with full eligibility)
- 180 usable smoothed-β months after the 12-month strict-trailing warm-up
- First tradable signal month: **2011-05-31** (improved 6 has ~15 trading years vs ~19 for other variants)
- Median cross-section size: **450 stocks** per monthly regression
- Median monthly R²: **0.147** — the single-factor cross-section regression explains ~15% of next-month return variance; unusually high for a cross-sectional return regression

### Factor-mimicking portfolio summary

| Approach | Factor | Annualized Sharpe | t-stat | Avg Rank IC |
|---|---|---:|---:|---:|
| portfolio_sort | momentum | 0.08 | 0.34 | 0.004 |
| portfolio_sort | pe | 0.21 | 0.92 | 0.012 |
| portfolio_sort | roe | -0.21 | -0.92 | 0.004 |
| portfolio_sort | trend | **0.36** | 1.44 | **0.022** |
| cross_sectional_regression | momentum | 0.20 | 0.86 | 0.004 |
| cross_sectional_regression | pe | -0.24 | -1.07 | 0.012 |
| cross_sectional_regression | roe | -0.21 | -0.92 | 0.004 |
| cross_sectional_regression | trend | **0.39** | 1.56 | **0.022** |

Trend is the only single factor with a positive Sharpe in both implementations and the strongest rank IC.

### Walk-forward (train ≤ 2020-12-31, test = 2021-01 to 2026-05)

| Strategy | Train Sharpe | Test Sharpe | Test Return | Test Max DD |
|---|---:|---:|---:|---:|
| base_equal_top10 | 1.11 | 1.30 | 46.5% | -3.6% |
| improved_1_expanding_trend_top10 | 0.57 | 1.34 | 103.1% | -5.2% |
| improved_2_expanding_trend_stop_take_top10 | 0.63 | 0.96 | 58.5% | -5.8% |
| improved_3_dynamic_ic_weights_stop_take_top10 | 0.54 | 0.98 | 57.4% | -10.5% |

The base wins the pre-2021 selection but improved 1 has the strongest post-2020 test Sharpe and return — a clear example of how a single train/test split is too narrow to call a winner.

### Benchmark comparison vs `^GSPC`

| Strategy | Annualized Alpha | Alpha t-stat | Beta | Excess Return p-value |
|---|---:|---:|---:|---:|
| base_equal_top10 | 9.1% | 4.70 | -0.04 | 0.71 |
| improved_1_expanding_trend_top10 | 6.9% | 3.54 | -0.03 | 0.37 |
| improved_2_expanding_trend_stop_take_top10 | 5.6% | 3.30 | -0.03 | 0.21 |
| improved_3_dynamic_ic_weights_stop_take_top10 | 5.4% | 2.94 | -0.03 | 0.20 |

All variants produce statistically significant alpha vs the index but with low to mildly negative beta, consistent with the factor tilt actively rotating into and out of names the index passively holds.

---

## Robustness

- **Monte Carlo random portfolios:** 1,000 simulations per strategy, sampling random eligible portfolios with matching size, sizing, and risk-exit rules. The p-value is the fraction of random portfolios with Sharpe ≥ strategy Sharpe.
- **Block bootstrap:** 6-month blocks, 1,000 resamples of the monthly return series, used for per-strategy Sharpe / drawdown distribution.
- **Walk-forward selection:** train through 2020-12-31, report test results after selection. Used for improved 4's stop/take grid; the test-period metrics are reported after the fact and are not used to choose the winner.
- **Long-only assertion:** every Backtrader run is checked for negative positions at the end and aborts if found.
- **Order exectype audit:** every Backtrader run records the distinct `bt.Order.exectype` values used. Base and improved 1 use Market only; improvements 2–6 use Market + Stop + Limit.

---

## Reproduce

Install dependencies once:

```powershell
py -3.10 -m pip install -r requirements.txt
```

Full pipeline (slow; ~56 minutes on the reference machine):

```powershell
py -3.10 src\run_project.py
```

Focused per-strategy runs (each reads existing processed CSVs and is much faster):

```powershell
py -3.10 src\run_base_strategy.py
py -3.10 src\run_improved_strategy.py
py -3.10 src\run_improved_4_stop_take_sensitivity.py
py -3.10 src\run_improved_5_regime_filter.py
py -3.10 src\run_improved_6_hzz_trend.py
```

After code changes:

```powershell
py -3.10 -m py_compile src\project_core.py src\run_project.py src\run_improved_4_stop_take_sensitivity.py src\run_improved_5_regime_filter.py src\run_improved_6_hzz_trend.py
py -3.10 -c "import sys; sys.path.insert(0, 'src'); import project_core; print('project_core import ok')"
```

---

## Source Layout

```text
src/
  project_core.py                       — all data loading, factor construction, vector
                                          strategies, Backtrader strategies, robustness
                                          tests, figures, documentation, presentation
  run_project.py                        — full reproducible pipeline
  run_base_strategy.py                  — focused base rerun
  run_improved_strategy.py              — focused improved 1, 2, 3 rerun
  run_improved_4_stop_take_sensitivity.py
                                        — focused stop/take grid + selected candidate
  run_improved_5_regime_filter.py       — focused regime filter experiment
  run_improved_6_hzz_trend.py           — focused HZZ cross-sectional trend experiment
  compare_strategies.py                 — rebuilds comparison tables
```

Key parts of `project_core.py`:

```text
Data:                load_raw_data, make_monthly_bars, make_monthly_metrics, make_roe_panel
Trend (index):       assignment_index_trend_coefficients, expanding_index_trend_to_stocks
Trend (HZZ):         stock_ma_ratios, cross_sectional_trend_betas, smooth_trend_betas,
                     hzz_predicted_returns
Factor panel:        build_factor_panel, winsorized_zscore_by_month
FMP analysis:        make_fmp_returns, summarize_fmps
Vector strategies:   score_for_spec, select_positions_for_spec, simulate_vector_strategy
Backtrader:          FixedCashSizer, MonthlySignalStrategy, DailySignalStopTakeStrategy,
                     run_backtrader, run_backtrader_daily_stop_take,
                     assert_backtrader_long_only
Robustness:          monte_carlo_random_portfolios, block_bootstrap, walk_forward_summary
Reporting:           make_figures, write_strategy_history, make_presentation, main
```

---

## Output Layout

```text
results/
  base_strategy/                — base vector and Backtrader outputs
  improved_strategy/            — improved 1
  improved_strategy_2/          — improved 2 (10% / 20% stops)
  improved_strategy_3/          — improved 3 (dynamic IC weights)
  improved_strategy_4/          — improved 4 (walk-forward 5% / 30% stops)
  improved_strategy_5/          — improved 5 (regime filter; rejected)
  improved_strategy_6/          — improved 6 (HZZ cross-sectional trend)
  comparison/                   — strategy_stage_metrics, walk-forward, benchmark comparison
  fmp_analysis/                 — factor-mimicking portfolio summary, IC tables
data/processed/                 — monthly bars, factor panel, monthly index, ROE as-of
figures/                        — equity curves, drawdowns, factor portfolio returns,
                                  Monte Carlo histogram, stop/take heatmap, IC summary
presentation/                   — generated PDF deck
docs/                           — STRATEGY_HISTORY, PROJECT_REPORT, IMPROVED_4/5/6 notes,
                                  CODE_STRUCTURE, DATA_DICTIONARY, PROJECT_PLAN,
                                  CLAUDE_CODE_PROJECT_HANDOFF
```

---

## Limitations

The project is honest about what it cannot claim.

- **Survivorship bias.** The universe is the 503 *current* S&P 500 tickers. Companies that were S&P 500 members but failed or were removed (Lehman, Bear Stearns, WaMu, GM, Yahoo, GE pre-2018, Sears, RadioShack, and many others) are entirely absent. Every Sharpe in the table is biased upward and every drawdown biased downward as a result. Late-inclusion backfill is the second symptom: companies that recently joined the S&P 500 (Palantir 2024, Coinbase 2025, etc.) appear in the panel from their IPO dates because they are in today's roster, so the strategy can "pick" them before they were actually in the index.
- **No transaction costs or slippage.** Monthly rebalancing of a top-10 portfolio with ~20–30% turnover at 0 bps overstates net Sharpe by an estimated 0.1–0.3 versus a realistic 5–10 bps round-trip cost world.
- **Long-only and no leverage.** Short positions exist only in the factor-mimicking-portfolio diagnostics, not in the executed strategies.
- **Daily OHLC stop/limit fills.** Backtrader uses bar-based stop/limit triggers, which approximate but do not exactly replicate intraday execution.
- **Fixed-cash sizing interacts with top-N.** With $1M initial capital and $100K per trade, 10 positions consume all capital at inception, and integer rounding plus next-bar pricing produces some `Margin` rejections. Realized average holdings are 4–8, not 10. This is acceptable under the current sizing rule but should not be silently labeled "top 10."
- **Base has look-ahead.** The base trend regression is fit on the full 2006–2026 sample and then applied to 2007–2026 signals. The 0.31 Sharpe drop from base to improved 1 *is* the cost of that look-ahead. The base is reported because the assignment prescribes that exact procedure; it is not the honest performance baseline.
- **Improved 4 selection is borderline.** Train/test split is ~14 vs ~5.5 years; the test window is too short to draw a strong out-of-sample conclusion. Monte Carlo p = 0.056 is suggestive, not decisive.
- **Multiple-comparisons problem.** We tried seven strategy variants and report the best. A formal Hansen SPA or Romano-Wolf step-down test is on the future-work list and has not been run.
- **HZZ trend factor warm-up.** The 12-month strict-trailing β smoothing makes improved 6's first tradable month 2011-05, so improved 6 is evaluated over ~15 years vs ~19 for the other variants.

---

## Future Work

In priority order.

1. **Transaction cost layer** — re-run improved 4 and improved 6 with 5 / 10 / 20 bps commission and ±5 bps slippage, as a standalone diagnostic. Keep the zero-cost baseline as the assignment-faithful result.
2. **Historical S&P 500 membership** — switch from current-only constituents to point-in-time membership using a survivorship-bias-free historical ledger (e.g., the open `fja05680/sp500` repository). This alone removes the late-inclusion backfill bias for free.
3. **Delisted ticker price coverage** — add prices for the ~600 dead tickers the current panel is missing (Stooq bulk US daily data covers most of them for free, with Sharadar SF1/SEP as a paid upgrade for fundamentals on dead tickers).
4. **Hansen SPA / Romano-Wolf multi-comparison correction** across all seven variants using the `arch.bootstrap` package — directly addresses the data-snooping problem from running the ladder.
5. **HZZ paper-faithful long-short portfolio** — Q5 long / Q1 short equal-weighted on the cross-sectional predicted return, for direct comparison with the paper's headline.
6. **Factor orthogonalization** — regress trend on momentum and use the residual to remove double-counting in the composite score.
7. **Top-N sensitivity** — vary the target portfolio size around 10 while keeping the fixed-cash sizing rule, and report how realized vs target holdings interact.

---

## References

- Han, Y., Zhou, G., & Zhu, Y. (2016). *A Trend Factor: Any Economic Gains from Using Information over Investment Horizons?* Journal of Financial Economics, 122(2), 352–375.
- Chen, A. Y., & Zimmermann, T. (2020). *Open Source Cross-Sectional Asset Pricing.* OpenSourceAP/CrossSection.
- Quantitativo (2024). *Coding Trend Factor.* Reference implementation of HZZ.
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability.* Journal of Business & Economic Statistics, 23(4), 365–380.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica, 68(5), 1097–1126.
