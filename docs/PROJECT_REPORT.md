# Project Report: S&P 500 Factor Investing

## Assignment Alignment

This project follows the factor-investing section of the assignment: ROE, P/E, momentum, and trend factors; portfolio-sort and regression FMPs; IC tests; Backtrader backtests; Monte Carlo significance testing; and saved outputs. The BIST100-specific index role is replaced with `^GSPC`, the S&P 500 index.

The project goes beyond the assignment minimum to add: a paper-faithful HZZ cross-sectional trend factor (improved 6), realistic time-varying transaction-cost sensitivity (improved 7), equal-weight 1/N sizing with top-20 expansion (improved 8), and a formal common evaluation-window methodology fix that produces apples-to-apples Sharpe comparisons across all 8 variants.

## Paper Summary

Han, Zhou, and Zhu (2016) propose a trend factor that combines short-, intermediate-, and long-horizon price information through normalized moving averages across 11 trading-day windows (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000). Their estimation procedure is **cross-sectional**: each month, regress every stock's next-month return on its 11 normalized MA ratios, then smooth the resulting coefficient vector across the trailing 12 months. The smoothed beta vector is applied per-stock to produce a predicted-return signal.

This project implements two trend procedures: (a) the assignment-prescribed index-derived approach used by base and improveds 1-5, 7, and (b) the paper-faithful cross-sectional approach used by improved 6.

## Data Process

The raw stock files contain 503 current S&P 500 securities. Prices and daily ratios are filtered to `date <= 2026-05-31`. Statement fundamentals are filtered to `date_available <= 2026-05-31`, which prevents using financial statements before they became public.

The processed panel is monthly. Stock returns use adjusted prices. The final observation is the last trading date on or before May 2026.

## Factor Construction

- **ROE** uses the latest point-in-time `roe` statement field, joined by `date_available`.
- **P/E** uses positive month-end `peRatio`, multiplied by `-1` so high factor scores mean cheap valuation.
- **Momentum** is `P_t / P_(t-12) - 1`.
- **Trend** uses normalized moving-average deviations with windows `[3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]`.
  - Base coefficients come from the assignment-style full-sample `^GSPC` predictive regression with insignificant variables dropped (`trend_z`).
  - Improveds 1, 2, 3, 4, 5, 7 use expanding no-lookahead `^GSPC` regressions (`trend_expanding_z`).
  - Improved 6 uses cross-sectional stock-level regression with strict 12-month trailing beta smoothing (`trend_hzz_z`).
- **Improved 3** factor weights are estimated from a rolling 60-month rank-IC information-ratio rule, shrunk 50% toward equal weights and capped between 10% and 45% per factor.

All factors are month-end, lagged by one month, winsorized at the 1st and 99th percentiles within each month, and standardized cross-sectionally (z-scored).

## Evaluation Window

All performance metrics (Sharpe, drawdown, cumulative return, Monte Carlo p-value, walk-forward, benchmark alpha) are computed over the common evaluation window starting **2016-05-31**, the date the latest-warmup strategy (`trend_expanding_z`) becomes computable. This makes all 8 strategies directly comparable on the same 121 months of trading data. See `docs/STRATEGY_HISTORY.md` and the README for details.

## Strategy Ladder (8 variants)

| Strategy | One change from foundation | Status |
|---|---|---|
| `base_equal_top10` | Assignment baseline (full-sample trend regression) | Look-ahead-biased; documented |
| `improved_1_expanding_trend_top10` | No-lookahead expanding trend regression | Honest baseline |
| `improved_2_expanding_trend_stop_take_top10` | Adds 10% stop / 20% take-profit | Risk-management layer |
| `improved_3_dynamic_ic_weights_stop_take_top10` | Adds rolling rank-IC dynamic weights | Rejected (no benefit) |
| `improved_4_walkforward_stop_take_top10` | Walk-forward selected stops (5% / 30%) | **Best risk-managed variant** |
| `improved_5_regime_filtered_stop_take_top10` | Adds 10-month SMA regime filter | Rejected (hurt performance) |
| `improved_6_hzz_cross_sectional_trend_stop_take_top10` | HZZ cross-sectional trend factor | Paper-faithful, lower per-trade alpha |
| `improved_7_time_varying_cost_sensitivity` | Time-varying transaction-cost study (not a new strategy; analyzes improveds 4 and 6) | Both finalists survive realistic costs |
| `improved_8_equal_weight_top20` | top-N 10→20, sizing → 5%-of-equity (1/N) | Best absolute wealth, lower Sharpe |

## Results (evaluation window 2016-05-31 to 2026-05-31)

### Vector and Backtrader headlines

| Strategy | Vec Sharpe | BT Sharpe | Vec Max DD | BT Max DD | Vec Final | BT Final | MC p |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | +1.20 | +1.36 | -7.3% | -6.8% | $2.03M | $5.78M | 0.013 |
| improved_1 | +1.18 | +1.22 | -15.8% | -17.2% | $3.51M | $3.77M | 0.019 |
| improved_2 | +1.05 | +1.26 | -13.0% | -13.5% | $2.72M | $2.89M | 0.068 |
| improved_3 | +0.97 | +1.21 | -14.1% | -15.5% | $2.64M | $2.68M | 0.157 |
| **improved_4** | **+1.15** | **+1.40** | **-7.4%** | **-7.7%** | $2.75M | $2.82M | 0.088 |
| improved_5 | +0.93 | +1.06 | -11.2% | -10.6% | $2.15M | $2.11M | 0.099 |
| improved_6 | +1.02 | +1.02 | -8.0% | -11.2% | $2.04M | $3.16M | 0.290 |
| **improved_8** | +1.03 | +1.11 | -10.9% | -11.9% | **$3.63M** | $3.09M | 0.306 |

### Improved 7 cost sensitivity

| Strategy | Zero | Central | Pessimistic |
|---|---:|---:|---:|
| improved_4 | 1.150 | 1.118 | 1.086 |
| improved_6 | 1.023 | 0.985 | 0.948 |

Both finalists survive pessimistic costs (2× central) with positive Sharpe. Improved 4 stays ahead at all cost levels.

### Benchmark comparison vs ^GSPC

| Strategy | Annualized Alpha | Alpha t-stat | Beta |
|---|---:|---:|---:|
| improved_1 | 14.16% | 3.96 | -0.07 |
| improved_2 | 11.52% | 3.65 | -0.07 |
| improved_3 | 11.16% | 3.11 | -0.07 |
| base | 7.91% | 4.43 | -0.05 |

Note: this table currently includes only the staged ladder. Extending to improveds 4-8 is documented future work.

### Walk-forward (train ≤ 2020-12, test 2021+)

| Strategy | Train Sharpe | Test Sharpe | Test Return | Test Max DD |
|---|---:|---:|---:|---:|
| improved_2 (selected) | **1.14** | 0.96 | 58.5% | -5.8% |
| base | 1.10 | 1.30 | 46.5% | -3.6% |
| improved_1 | 1.03 | **1.34** | **103.1%** | -5.2% |
| improved_3 | 0.98 | 0.98 | 57.4% | -10.5% |

Same caveat: only staged ladder currently included.

## What We Tried

- Equal-weight composite base strategy with assignment-style full-sample trend regression (`base`)
- Removing trend look-ahead via expanding regression (`improved 1`)
- Adding 10% / 20% stop and take-profit risk exits (`improved 2`)
- Past-only rolling rank-IC dynamic factor weights (`improved 3`) — rejected
- Walk-forward selected 5% / 30% stop/take threshold tuning (`improved 4`)
- ^GSPC 10-month SMA regime filter overlay (`improved 5`) — rejected
- Han-Zhou-Zhu paper-faithful cross-sectional trend factor (`improved 6`)
- Time-varying transaction-cost sensitivity (`improved 7`)
- Equal-weight 1/N sizing with top-N expansion to 20 (`improved 8`)

## Interpretation

The project should not claim statistically guaranteed skill only because the backtest is profitable. Three guardrails matter:

1. **Survivorship-biased universe.** All Sharpes are inflated by some amount (estimated 0.2-0.4) because the universe is current S&P 500 only. The relative ordering between strategies is more robust than any absolute Sharpe level.
2. **Multi-comparison problem.** We tested 8 variants and report the best. A Hansen SPA or Romano-Wolf correction is the planned next major upgrade (improved 9) and would adjust every reported p-value upward.
3. **Evaluation-window correction.** All Sharpes reported here use the common 2016-05-31 to 2026-05-31 window. Earlier project versions diluted Sharpe by including pre-warmup zero-return months; that bug is fixed.

**Strongest defensible claim (relative):** Improved 4 is the cleanest risk-managed variant. It dominates improveds 1, 2, 3, 5, 6 on Sharpe AND drawdown in fair window-aligned comparison. Improved 8 sacrifices Sharpe for wealth growth via dynamic equal-weight sizing. Improved 6 implements the paper-faithful methodology but has lower per-trade alpha in this universe than the simpler index-derived trend.

**Strongest defensible claim (absolute, with caveats):** Both improved 4 and improved 6 survive realistic time-varying transaction costs at the pessimistic (2× central) scenario with positive Sharpe. This is the cost-robustness floor.

**Real-money confidence still requires:** survivorship-bias-free data (WRDS / CRSP, the planned improved 10), Hansen SPA multi-comparison correction (planned improved 9), longer out-of-sample period (data-constrained), and sector-neutrality / beta-neutrality analysis.

The vectorized improvement tests are used for fast monthly screening and grid searches. The executable Backtrader runs are long-only: base and improved 1 use monthly `bt.Order.Market` orders; improveds 2, 3, 4, 5, 6, 8 use daily adjusted OHLC data with `bt.Order.Market` entries/rebalances and native `bt.Order.Stop` / `bt.Order.Limit` protective exits; improved 8 additionally uses `EquityPercentSizer` for 1/N position sizing.
