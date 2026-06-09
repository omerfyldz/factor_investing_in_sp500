# S&P 500 Factor Investing Research

This repository is a reproducible S&P 500 factor investing research project through May 2026. It builds stock-level factor signals, constructs factor-mimicking portfolios, evaluates information coefficients, runs long-only Backtrader strategies, and records a staged improvement history instead of hiding failed experiments.

The final research conclusion is deliberately cautious: the factor process is economically sensible and reproducible, and the best risk-managed variant is improved 4, but the results are not enough to claim a production-ready trading edge without survivorship-bias-free constituents, transaction costs, slippage, and stronger data-snooping controls.

## What This Project Does

- Uses a current S&P 500 stock universe with frozen daily adjusted prices and fundamentals.
- Uses Yahoo Finance `^GSPC` as the S&P 500 benchmark and index trend-regression input.
- Builds four required factor families: ROE, P/E, 12-month momentum, and trend.
- Forms monthly cross-sectional factor z-scores after outlier treatment.
- Constructs factor-mimicking portfolios by portfolio sorting and cross-sectional regression.
- Tests raw IC and rank IC.
- Runs long-only Backtrader strategies with fixed cash sizing.
- Saves all processed data, results, diagnostics, figures, and a PDF presentation.
- Keeps a strategy history that records both improvements and failed ideas.

## Data Window

All analysis is capped at `2026-05-31`.

Because May 31, 2026 was not necessarily a trading day for every data source, the last observed trading date in the frozen stock and benchmark files is `2026-05-29`. The processed monthly factor panel extends to the May 2026 month end.

Validation summary:

| Check | Status | Detail |
|---|---:|---|
| Stock price cutoff | OK | `2026-05-29` |
| Daily fundamentals cutoff | OK | `2026-05-29` |
| Statement fundamentals cutoff | OK | `2026-05-29` by `date_available` |
| Benchmark cutoff | OK | `2026-05-29` |
| Factor panel cutoff | OK | `2026-05-31` |

## Raw Data

Raw files live under `data/raw/`.

| File | Purpose |
|---|---|
| `sp500_prices_long.csv` | Daily stock OHLCV, including adjusted OHLCV. |
| `sp500_fundamentals_daily_long.csv` | Daily valuation ratios and market cap. |
| `sp500_fundamentals_statements_long.csv` | Statement fundamentals using `date_available`. |
| `sp500_constituents.csv` | Current S&P 500 universe metadata. |
| `sp500_index_yahoo.csv` | Frozen Yahoo Finance `^GSPC` benchmark. |

The three large raw vendor CSVs are tracked with Git LFS because they are larger than GitHub's normal 100 MB file limit.

## Factor Construction

Signals are created monthly. A signal observed at month `t` is used for the following month, so the strategy does not trade on same-period returns.

### ROE

ROE is taken from statement fundamentals using `date_available`. This is important because it avoids using a financial statement before it became public.

Higher ROE is better.

### P/E

P/E uses positive month-end `peRatio`. The factor is multiplied by `-1`, so lower valuation receives a higher factor score.

Lower positive P/E is better.

### Momentum

Momentum is:

```text
P_t / P_(t-12) - 1
```

The project uses a 12-month adjusted-price return. No extra skip month is used in the current implementation.

### Trend

Trend follows the paper-inspired moving-average deviation idea:

1. Compute normalized moving-average deviations for the S&P 500 index.
2. Run predictive regressions on index returns.
3. Drop statistically insignificant moving-average variables.
4. Apply retained index-regression coefficients to each stock's moving-average deviations.
5. Use the predicted return as the stock's trend factor.

Two versions are used:

| Version | Use |
|---|---|
| `trend_z` | Base strategy. Uses one full-sample index regression with insignificant variables dropped. |
| `trend_expanding_z` | Improved strategies. Uses expanding historical regressions to reduce look-ahead bias. |

### Preprocessing

For each month:

- discard impossible or invalid observations;
- winsorize factor values at 1st and 99th cross-sectional percentiles;
- convert factors to z-scores;
- require at least three valid factors for composite scoring.

## Factor-Mimicking Portfolio Analysis

The project builds factor-mimicking portfolios in two ways:

1. Portfolio sort: top quintile minus bottom quintile.
2. Cross-sectional regression: monthly factor-risk-premium estimate.

The project also saves:

- cumulative FMP returns;
- IC and rank IC;
- average monthly return and t-tests;
- Sharpe, volatility, and drawdown statistics;
- common-start comparisons;
- selected-date FMP weights.

Main files:

| File | Purpose |
|---|---|
| `results/fmp_analysis/fmp_portfolio_returns.csv` | Portfolio-sort FMP returns. |
| `results/fmp_analysis/fmp_regression_returns.csv` | Cross-sectional regression FMP returns. |
| `results/fmp_analysis/factor_information_coefficients.csv` | IC and rank IC. |
| `results/fmp_analysis/fmp_performance_summary.csv` | FMP performance table. |
| `results/fmp_analysis/trend_index_regression_coefficients.csv` | Base trend regression coefficients. |
| `results/fmp_analysis/trend_expanding_regression_coefficients.csv` | Expanding trend regression coefficients. |

## Trading Strategy Design

The implemented trading strategies are long-only. Each rebalance ranks eligible S&P 500 stocks by composite factor score and buys the highest-ranked stocks subject to fixed cash sizing.

Core trading assumptions:

| Assumption | Value |
|---|---:|
| Initial capital | `1,000,000` |
| Fixed cash per trade | `100,000` |
| Commission | `0` |
| Base order type | Market order |
| Base rebalance frequency | Monthly |
| Shorting | Not used in trading strategy |

The top-N rule is implemented in `select_positions_for_spec()`: rank by composite score and select the highest `top_n` names, constrained by available equity and fixed `100,000` cash-per-trade sizing.

## Strategy Ladder

The main research value of this repository is the staged improvement path. Each stage changes one main design dimension.

| Stage | Strategy | Main Change | Accepted As Improvement? |
|---|---|---|---|
| Base | `base_equal_top10` | Full-sample trend regression, equal factor weights, top 10 stocks, no stop/take. | Baseline |
| Improved 1 | `improved_1_expanding_trend_top10` | Replaces full-sample trend with expanding no-lookahead trend regression. | Methodology improvement, but weaker performance |
| Improved 2 | `improved_2_expanding_trend_stop_take_top10` | Adds 10% stop-loss and 20% take-profit. | Improves some Backtrader risk metrics versus improved 1 |
| Improved 3 | `improved_3_dynamic_ic_weights_stop_take_top10` | Adds rolling rank-IC factor weights with shrinkage and caps. | Not accepted as stronger than improved 2 |
| Improved 4 | `improved_4_walkforward_stop_take_top10` | Tests stop/take sensitivity and selects 5% stop-loss, 30% take-profit using training data through 2020. | Current best risk-managed variant |
| Improved 5 | `improved_5_regime_filtered_stop_take_top10` | Adds a fixed `^GSPC` 10-month SMA market-regime cash filter on top of improved 4. | Failed experiment |

## Strategy Results

### Vector Monthly Results

| Strategy | Final Equity | Sharpe | Max Drawdown | Monte Carlo p-value |
|---|---:|---:|---:|---:|
| Base | `$5,319,947` | `1.1317` | `-10.80%` | `0.0000` |
| Improved 1 | `$3,510,333` | `0.8184` | `-15.78%` | `0.0210` |
| Improved 2 | `$2,721,443` | `0.7310` | `-13.03%` | `0.0610` |
| Improved 3 | `$2,635,221` | `0.6788` | `-14.13%` | `0.1710` |
| Improved 4 | `$2,751,551` | `0.7968` | `-7.36%` | `0.0560` |
| Improved 5 | `$2,148,104` | `0.6525` | `-11.19%` | `0.6030` |

### Backtrader Results

| Strategy | Engine | Final Value | Sharpe | Max Drawdown |
|---|---|---:|---:|---:|
| Base | Monthly market orders | `$5,778,291` | `1.2023` | `-10.42%` |
| Improved 1 | Monthly market orders | `$3,768,578` | `0.8403` | `-17.17%` |
| Improved 2 | Daily stop/take checks | `$2,885,335` | `0.8811` | `-13.97%` |
| Improved 3 | Daily stop/take checks | `$2,643,354` | `0.8202` | `-15.58%` |
| Improved 4 | Daily stop/take checks | `$2,983,357` | `1.0392` | `-7.62%` |
| Improved 5 | Daily stop/take checks | `$2,323,043` | `0.8593` | `-11.57%` |

## What Improved And What Did Not

### Base

The base is the strongest full-sample performer. It uses full-sample trend coefficients, which follows the literal assignment-style design but is less conservative from a live-trading perspective.

### Improved 1

Improved 1 uses expanding trend regressions. This is more scientifically conservative because each signal month uses only prior index observations. It performs worse than the base, but it is methodologically cleaner.

### Improved 2

Improved 2 adds 10% stop-loss and 20% take-profit. It improves Backtrader Sharpe relative to improved 1 and reduces drawdown relative to improved 1, but it does not beat the base.

### Improved 3

Improved 3 adds rolling rank-IC factor weights. It was designed to be more adaptive, but it underperformed improved 2. The likely reason is that rolling IC estimates are noisy, so dynamic weighting can chase unstable factor leadership.

### Improved 4

Improved 4 is the best accepted risk-managed variant. It keeps improved 2's static equal-weight signal design but tests a small stop/take grid. The selected pair is:

```text
Stop-loss: 5%
Take-profit: 30%
```

Selection used training data through `2020-12-31`, with a penalty for isolated parameter peaks. The result is not treated as a proven optimum, but nearby stop/take settings also looked reasonable, which makes it less suspicious than a single lucky parameter pair.

### Improved 5

Improved 5 tested a fixed market-regime filter: trade only when `^GSPC` is above its 10-month SMA. This failed. It reduced Sharpe, final value, and drawdown quality versus improved 4, and its Monte Carlo p-value worsened to `0.6030`.

The likely explanation is that the index filter removed too much exposure and missed rebound months. Since improved 4 already has daily stop/take risk control, an additional blunt market-timing overlay was too defensive.

## Robustness Tests

The project includes several robustness checks:

- Monte Carlo random portfolios with the same universe, rebalance schedule, number of positions, and stop/take assumptions.
- Block bootstrap of monthly returns.
- Walk-forward summary using pre-2021 and 2021-May 2026 periods.
- Backtrader long-only position checks.
- Separate decision summaries for improvement experiments.

Important robustness interpretation:

- A low Monte Carlo p-value supports that the strategy outperformed random selection under the same test design.
- A high Monte Carlo p-value means random portfolios often matched or beat the strategy.
- Improved 5 is rejected partly because its Monte Carlo p-value rose to `0.6030`.
- Improved 4 is promising, but its p-value of `0.0560` is still borderline. It should be described cautiously.

## Output Files

### Processed Data

| File | Purpose |
|---|---|
| `data/processed/factor_panel.csv` | Final stock-month panel with factors, z-scores, returns, and regime state. |
| `data/processed/monthly_stock_bars.csv` | Monthly adjusted OHLCV and forward returns. |
| `data/processed/monthly_sp500_index.csv` | Monthly `^GSPC` benchmark and regime filter fields. |
| `data/processed/monthly_roe_asof.csv` | Point-in-time ROE panel. |

### Strategy Results

| Folder | Contents |
|---|---|
| `results/base_strategy/` | Base vector, Backtrader, Monte Carlo, bootstrap. |
| `results/improved_strategy/` | Improved 1 outputs. |
| `results/improved_strategy_2/` | Improved 2 outputs. |
| `results/improved_strategy_3/` | Improved 3 outputs. |
| `results/improved_strategy_4/` | Improved 4 stop/take sensitivity and selected candidate. |
| `results/improved_strategy_5/` | Improved 5 regime filter results and rejection decision. |
| `results/comparison/` | Stage comparison, benchmark comparison, walk-forward summary. |
| `results/fmp_analysis/` | Factor portfolio and IC outputs. |

### Figures

| File | Purpose |
|---|---|
| `figures/factor_portfolio_cumulative_returns.png` | Portfolio-sort FMP cumulative returns. |
| `figures/factor_regression_cumulative_returns.png` | Regression FMP cumulative returns. |
| `figures/ic_rank_ic_summary.png` | IC and rank IC summary. |
| `figures/strategy_equity_vs_benchmark.png` | Strategy equity curves versus benchmark. |
| `figures/strategy_drawdowns.png` | Strategy drawdowns. |
| `figures/strategy_improvement_sharpe.png` | Strategy Sharpe comparison. |
| `figures/monte_carlo_sharpe_histogram.png` | Monte Carlo Sharpe distribution. |
| `figures/improved4_stop_take_sensitivity_heatmap.png` | Improved 4 stop/take grid heatmap. |

### Documentation

| File | Purpose |
|---|---|
| `docs/PROJECT_REPORT.md` | Concise report. |
| `docs/STRATEGY_HISTORY.md` | Full improvement log and failed experiments. |
| `docs/IMPROVED_4_STOP_TAKE_SENSITIVITY.md` | Improved 4 details. |
| `docs/IMPROVED_5_REGIME_FILTER.md` | Improved 5 details and rejection decision. |
| `docs/REPRODUCIBILITY.md` | Fresh-run instructions. |
| `docs/DATA_DICTIONARY.md` | Data and output dictionary. |
| `presentation/sp500_factor_investing_presentation.pdf` | Generated presentation. |

## Reproducibility

This repository uses Python 3.10.

Install dependencies:

```powershell
py -3.10 -m pip install -r requirements.txt
```

Run the full project:

```powershell
py -3.10 src\run_project.py
```

Run focused stages:

```powershell
py -3.10 src\run_base_strategy.py
py -3.10 src\run_improved_strategy.py
py -3.10 src\run_improved_4_stop_take_sensitivity.py
py -3.10 src\run_improved_5_regime_filter.py
```

The full run regenerates processed data, FMP analysis, strategy outputs, figures, docs, and the PDF presentation. The focused improved 4 and improved 5 scripts do not rebuild the full project; they use existing processed factor data and raw daily prices.

## Important Implementation Notes

- Base and improved 1 use monthly Backtrader market orders.
- Improved 2, 3, 4, and 5 use daily Backtrader adjusted OHLC stop/take threshold checks and market exits.
- If both stop-loss and take-profit are touched within the same daily bar, stop-loss takes priority.
- The monthly vector stop/take calculations are approximations based on OHLC bars.
- The Backtrader runners include long-only validation to catch accidental short-position bugs.
- Large raw vendor CSVs are tracked with Git LFS.

## Limitations

- The universe uses current S&P 500 constituents, so survivorship bias remains.
- Commission is set to zero and slippage is not modeled.
- The strategy is long-only, while FMP analysis includes long-short analytical portfolios.
- Raw fundamentals coverage and vendor definitions may affect ROE and P/E quality.
- Stop/take tests use OHLC path assumptions; intraday order sequencing is not fully known.
- Improved 4 involves parameter selection, so it should be interpreted as a constrained sensitivity result, not a permanently optimal setting.
- No White Reality Check or Hansen SPA test is implemented yet.

## Future Work

The next improvements should be added one at a time and recorded in `docs/STRATEGY_HISTORY.md`.

Priority ideas:

1. Test top 20 instead of top 10 while preserving `100,000` fixed cash sizing.
2. Add realistic transaction costs and slippage as a separate robustness layer.
3. Replace current-constituent universe with historical S&P 500 membership.
4. Add sector exposure constraints after measuring current sector concentration.
5. Add volatility-based position risk diagnostics without changing fixed cash sizing.
6. Run White Reality Check or Hansen SPA over the full strategy family.
7. Compare equal-weight factor scores with rank-based composite scoring.
8. Add a real intraday or next-open stop/take simulation if intraday data becomes available.

Current best practical version:

```text
Improved 4: top 10, expanding trend, equal factor weights, 5% stop-loss, 30% take-profit.
```

But the current best scientific conclusion remains cautious: improved 4 is the best risk-managed backtest variant in this repository, not proof of a live trading edge.
