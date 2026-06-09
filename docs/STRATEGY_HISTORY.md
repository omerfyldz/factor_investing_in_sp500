# Strategy History And Improvement Log

This file is the living history of the S&P 500 factor investing project. It records what we built, what we removed, and what we should revisit later. The executable project is deliberately staged: the core ladder is base through improved 3, and the later focused experiments are improved 4 and improved 5.

## Project Rules We Must Preserve

- The project remains aligned with the factor-investing assignment: ROE, P/E, momentum, and trend factors are always the foundation.
- `^GSPC` replaces BIST100 as the index benchmark and trend-regression input.
- The final data window stops at May 2026.
- Backtrader uses initial cash `1,000,000`, fixed cash per trade `100,000`, and zero commission.
- The base Backtrader strategy uses market orders only.
- Stop-loss and take-profit are not part of the base. They enter at improved 2 and remain in improved 3.
- Dynamic factor weighting is not part of the base. It enters only at improved 3.
- Other ideas are kept as notes here, not as executable project outputs.

## Why We Improved The Base Strategy

The base strategy follows the project formula exactly, except that `^GSPC` replaces BIST100 because this version studies the S&P 500:

- ROE, P/E, momentum, and trend are the four required factors.
- Trend uses the full-sample index predictive regression requested in the project instructions.
- The base Backtrader strategy uses market orders, initial capital `1,000,000`, fixed cash per trade `100,000`, and commission `0`.
- The base strategy does not use stop-loss or take-profit. Those are improvement ideas, not base requirements.

The base strategy worked in absolute terms, but it should still be judged cautiously:

- Base strategy Sharpe: `1.1969`
- Base Monte Carlo p-value: `0.0130`

Interpretation: in the current cleaned run, no random top-10 S&P 500 portfolio in 1,000 simulations matched the base Sharpe. This supports the factor-selection result under the project's Monte Carlo design, but it is not a guarantee of real-money robustness.

The individual FMPs also warned us not to overclaim. Their t-statistics and IC values are weak:

| approach | factor | annualized_sharpe | t_stat | p_value | avg_ic | avg_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| portfolio_sort | momentum | 0.0792 | 0.3444 | 0.7309 | 0.0209 | 0.0042 |
| portfolio_sort | pe | 0.2051 | 0.9153 | 0.3610 | -0.0084 | 0.0118 |
| portfolio_sort | roe | -0.2064 | -0.9213 | 0.3578 | -0.0016 | 0.0038 |
| portfolio_sort | trend | 0.3602 | 1.4409 | 0.1513 | 0.0191 | 0.0224 |
| cross_sectional_regression | momentum | 0.1987 | 0.8644 | 0.3883 | 0.0209 | 0.0042 |
| cross_sectional_regression | pe | -0.2403 | -1.0723 | 0.2847 | -0.0084 | 0.0118 |
| cross_sectional_regression | roe | -0.2055 | -0.9173 | 0.3599 | -0.0016 | 0.0038 |
| cross_sectional_regression | trend | 0.3899 | 1.5597 | 0.1205 | 0.0191 | 0.0224 |

## Development Path

The current executable strategy ladder is intentionally small and sequential:

| name | assignment_scope | annualized_sharpe | final_equity | max_drawdown | avg_positions | annualized_alpha_approx | beta_to_sp500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_equal_top10 | True | 1.1969 | 2032039.2891 | -0.0726 | 9.9174 | 0.0791 | -0.0482 |
| improved_1_expanding_trend_top10 | False | 1.1834 | 3510333.0263 | -0.1578 | 8.4298 | 0.1416 | -0.0733 |
| improved_2_expanding_trend_stop_take_top10 | False | 1.0507 | 2721442.6689 | -0.1303 | 8.4298 | 0.1152 | -0.0750 |
| improved_3_dynamic_ic_weights_stop_take_top10 | False | 0.9727 | 2635221.3785 | -0.1413 | 8.4298 | 0.1116 | -0.0690 |

## Variant Notes

| name | assignment_scope | top_n | trend_col | weight_method | weight_lookback_months | regime_filter | stop_loss | take_profit | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_equal_top10 | True | 10 | trend_z | static | 60 | False |  |  | Base: assignment-style full-sample index trend regression, equal factor weights, no stop-loss/take-profit. |
| improved_1_expanding_trend_top10 | False | 10 | trend_expanding_z | static | 60 | False |  |  | Improved 1: same signals and weights as base, but trend uses expanding no-lookahead index regressions. |
| improved_2_expanding_trend_stop_take_top10 | False | 10 | trend_expanding_z | static | 60 | False | 0.100 | 0.200 | Improved 2: builds on improved 1 and adds 10% stop-loss plus 20% take-profit risk exits. |
| improved_3_dynamic_ic_weights_stop_take_top10 | False | 10 | trend_expanding_z | rolling_rank_ic | 60 | False | 0.100 | 0.200 | Improved 3: builds on improved 2 and uses past-only rolling rank-IC weights with shrinkage and caps. |

## What Improved And Why

### 1. Base Strategy

The base strategy is the literal project-style implementation:

- factor set: ROE, positive inverse P/E, 12-month momentum, trend;
- trend method: one full-sample `^GSPC` predictive regression using all available index data through May 2026;
- insignificant moving-average variables are dropped using a 5% significance cutoff;
- factor values are winsorized and monthly z-scored;
- signals at month t trade month t+1;
- Backtrader base uses market orders, fixed cash sizing, zero commission, and no stop-loss/take-profit.

Base vector Sharpe: `1.1969`.

### 2. Improved Strategy 1: Expanding Trend Regression

Improved 1 keeps every base choice fixed except the trend-regression estimation window. Instead of using one full-sample trend regression, it estimates an expanding `^GSPC` regression for each signal month using only index observations available before that month. This is a scientific improvement because it reduces look-ahead bias, even if it may or may not improve raw performance.

- Base vector Sharpe: `1.1969`
- Improved 1 vector Sharpe: `1.1834`

### 3. Improved Strategy 2: Stop-Loss And Take-Profit Layer

Improved 2 builds directly on improved 1 and adds risk exits:

- trend method: expanding no-lookahead trend regression;
- stop-loss: 10%;
- take-profit: 20%;
- vector results use a monthly open/high/low/close approximation with stop priority if stop and take-profit are both touched in the same month;
- Backtrader results use daily adjusted OHLC bars, explicit market orders for entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

Improved 2 vector Sharpe: `1.0507`.

Order-model note: the Backtrader stop/take engine now uses native `bt.Order.Stop` and `bt.Order.Limit` protective orders. Rebalance exits cancel live protective orders before submitting market exits so stale stop/limit orders cannot create accidental shorts. Saved Backtrader metrics should be regenerated before final interpretation under this execution model.

### 4. Improved Strategy 3: Past-Only Dynamic Factor Weighting

Improved 3 builds directly on improved 2 and changes only the factor weighting rule. Instead of hard-coding a value/quality tilt, it estimates factor weights from past rank-IC evidence:

- lookback window: 60 months;
- minimum history before dynamic weights: 24 months;
- score: positive rolling rank-IC information ratio;
- shrinkage: 50% back to equal weights;
- bounds: each factor must stay between 10% and 45%.

This is closer to a real-world process because the weights are based only on information available before the signal month and are constrained to avoid single-factor overfitting.

- Improved 2 vector Sharpe: `1.0507`
- Improved 3 vector Sharpe: `0.9727`

### 5. Removed Or Deferred Ideas

We previously explored HZZ cross-sectional trend, sector caps, volatility penalties, regime filters, top-N changes, and static value/quality-heavy factor weights. Those ideas were removed from the active code/results to keep the current project focused on a transparent one-change-at-a-time ladder. They can be reintroduced later one by one.

### 6. Next Improvement Rule

Every future improvement should build on the latest accepted improved strategy and change only one main design dimension at a time. That is how we keep the try/fail path scientifically readable.

## Robustness Results

Monte Carlo:

- Base strategy p-value: `0.0130`
- Improved 3 strategy p-value: `0.1570`

Walk-forward:

| strategy | train_sharpe_to_2020 | test_sharpe_2021_2026 | test_cumulative_return_2021_2026 | test_max_drawdown_2021_2026 | selected_by_train |
| --- | --- | --- | --- | --- | --- |
| improved_2_expanding_trend_stop_take_top10 | 1.1421 | 0.9588 | 0.5854 | -0.0575 | True |
| base_equal_top10 | 1.0954 | 1.3003 | 0.4646 | -0.0361 | False |
| improved_1_expanding_trend_top10 | 1.0324 | 1.3359 | 1.0312 | -0.0524 | False |
| improved_3_dynamic_ic_weights_stop_take_top10 | 0.9766 | 0.9842 | 0.5743 | -0.1047 | False |

Benchmark comparison:

| strategy | annualized_excess_return_approx | excess_t_stat | excess_p_value | annualized_alpha_approx | alpha_t_stat | alpha_p_value | beta_to_sp500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_1_expanding_trend_top10 | -0.0101 | -0.1616 | 0.8719 | 0.1416 | 3.9605 | 0.0001 | -0.0733 |
| improved_2_expanding_trend_stop_take_top10 | -0.0367 | -0.6081 | 0.5443 | 0.1152 | 3.6526 | 0.0003 | -0.0750 |
| improved_3_dynamic_ic_weights_stop_take_top10 | -0.0395 | -0.6468 | 0.5190 | 0.1116 | 3.1078 | 0.0019 | -0.0690 |
| base_equal_top10 | -0.0690 | -1.2803 | 0.2029 | 0.0791 | 4.4267 | 0.0000 | -0.0482 |

## Current Judgment

The project is stronger after separating the literal base from sequential improvements, Monte Carlo checks, walk-forward validation, and benchmark alpha comparisons. However, the honest conclusion remains cautious:

- The base and improved strategies are strong versus the current random-portfolio Monte Carlo benchmark.
- Improved 1 is primarily a methodology improvement because it reduces look-ahead bias in the trend factor.
- Improved 2 is a risk-management test on top of improved 1 and should be judged by both return and drawdown.
- Improved 3 is a weighting-process test on top of improved 2 and should be judged against improved 2, not just against the base.
- Vector curves are screening summaries; executable trading evidence comes from the saved Backtrader runs. Base and improved 1 use monthly `bt.Order.Market` Backtrader orders; improved 2 and improved 3 use daily Backtrader market entries/rebalances plus native stop/limit protective exits.
- The universe uses current S&P 500 constituents, so survivorship bias remains.
- Transaction costs and slippage are ignored because the assignment requires commission `0`.
- A production strategy would need historical index membership, costs, slippage, beta/sector neutrality, and a data-snooping-adjusted test such as White's Reality Check or Hansen's SPA.

## External Methodology Notes

- PyAnomaly emphasizes the same research building blocks we use: firm characteristics, winsorizing/trimming, quantile portfolios, long-short portfolios, factor regression, and cross-sectional regression.
- OpenSourceAP/CrossSection shows how much infrastructure serious factor replication needs: signal documentation, portfolio construction variants, and reproducibility checks.
- Quantitativo's trend-factor implementation stresses survivorship-bias-free universes, price filters, correct signal/return shifting, and the observation that post-2016 trend-factor edge can weaken.
- The removed research ideas are practical extensions, not replacements for the required factor framework.

## Ideas To Revisit Later

- Run a real White Reality Check or Hansen SPA test over all tried strategy variants.
- Use historical S&P 500 membership instead of current constituents.
- Tune stop-loss/take-profit values only after recording the first fixed 10%/20% improved 2 and dynamic-weight improved 3 results.
- Revisit static value/quality-heavy weights, top-N selection, cross-sectional trend, sector constraints, and volatility overlays later, one at a time.
- Add transaction costs and slippage as a separate robustness extension.
- Test historical S&P 500 membership to reduce survivorship bias.

## Improved 4 Stop/Take Sensitivity

Improved 4 was added after the fixed 10%/20% improved 2 result. It is a focused stop-loss/take-profit sensitivity test built on improved 2's static equal-weight signal design.

- The grid is intentionally small.
- The selected pair is chosen from training data through `2020-12-31`.
- Test-period results are reported after selection and are not used to choose the pair.
- The selected daily Backtrader execution result is saved in `results/improved_strategy_4/`.
- Selected stop-loss: `5.0%`.
- Selected take-profit: `30.0%`.
- Test Sharpe after selection: `1.0221`.

The warning is important: improved 4 is a robustness/sensitivity experiment, not proof that the selected parameters are permanently optimal.

## Improved 5 Regime Filter

Improved 5 was added after improved 4 as a focused market-regime filter test. It keeps improved 4's factor signals, top-10 construction, 5% stop-loss, and 30% take-profit, then adds one pre-specified rule: trade only when `^GSPC` is above its 10-month moving average.

- No regime-window optimization was performed.
- Vector Sharpe: `0.9336`.
- Vector max drawdown: `-11.19%`.
- Backtrader Sharpe: `1.0609`.
- Backtrader max drawdown: `-10.64%`.

The warning is important: this is a market-timing overlay and must be treated skeptically. It is acceptable only because it changes one pre-specified design dimension and is stored separately from improved 4.

## Improved 6 HZZ Cross-Sectional Trend

Improved 6 was added after improved 5 as a focused trend-signal replacement. It keeps improved 4's composite-weight design, top-10 construction, 5% stop-loss, and 30% take-profit, then changes exactly one thing: the trend column becomes `trend_hzz_z`, a Han, Zhou, Zhu (2016) cross-sectional trend factor estimated from monthly OLS regressions of next-month returns on 11 normalized moving-average ratios across the eligible cross-section, with a strict trailing 12-month average of past betas.

- Vector Sharpe: `1.0235`.
- Vector max drawdown: `-8.04%`.
- Backtrader Sharpe: `1.0159`.
- Backtrader max drawdown: `-11.15%`.

Improved 6 is the first variant whose trend signal is built from the cross-section of stocks rather than from the index. It exists to test the paper's actual methodology against the assignment-prescribed index regression while keeping all other improved 4 design choices fixed.

## Improved 7 Time-Varying Cost Sensitivity

Improved 7 is not a new strategy. It re-runs improved 4 and improved 6 under a
time-varying transaction-cost schedule (per-year commission + slippage, central
estimates drawn from Frazzini-Israel-Moskowitz 2018, ITG / Virtu Cost Index,
J.P. Morgan execution research, and NYSE TAQ literature). Three scenarios are
reported: zero (baseline), central (typical institutional execution), and
pessimistic (2x central).

- Improved 4 vector Sharpe -- zero: `1.1504`,
  central: `1.1182`,
  pessimistic: `1.0861`.
- Improved 6 vector Sharpe -- zero: `1.0235`,
  central: `0.9855`,
  pessimistic: `0.9477`.

Both finalists survive central-case costs with positive Sharpe. Improved 6
pays more friction because it holds more concurrent positions; improved 4's
lower turnover is a structural cost-robustness advantage. See
`docs/IMPROVED_7_COSTS.md` for the full schedule, sources, and methodology.

## Improved 8 Equal-Weight Top 20

Improved 8 changes two mechanically-coupled design dimensions on top of improved 4:
top-N moves from 10 to 20, and position sizing moves from fixed `$100,000`
per trade to equal-weight `5.00%` of current portfolio equity per position.
The two changes are treated as one improvement because fixed-dollar sizing is
mechanically incompatible with meaningful top-N expansion (`$1M` of capital
cannot fund `20 x $100k = $2M`). Justification draws on DeMiguel-Garlappi-Uppal
(2009) for the 1/N choice and Plyakha-Uppal-Vilkov (2014) for the equal-weight
concentrated portfolio rationale; the industry analog is Invesco's RSP equal-
weight S&P 500 ETF. Foundation is improved 4 (the cost-robust winner from
improved 7).

- Vector Sharpe: `1.0333`.
- Vector max drawdown: `-10.94%`.
- Backtrader Sharpe: `1.1098`.
- Backtrader max drawdown: `-11.90%`.

See `docs/IMPROVED_8_TOP_N_SIZING.md` for full methodology and references.

