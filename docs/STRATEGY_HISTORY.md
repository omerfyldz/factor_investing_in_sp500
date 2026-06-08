# Strategy History And Improvement Log

This file is the living history of the S&P 500 factor investing project. It intentionally preserves the base strategy, failed variants, partial improvements, and final candidate so we can explain the development path in the presentation and revisit ideas later.

## Project Rules We Must Preserve

- The project remains aligned with the factor-investing assignment: ROE, P/E, momentum, and trend factors are always the foundation.
- `^GSPC` replaces BIST100 as the index benchmark and trend-regression input.
- The final data window stops at May 2026.
- Backtrader uses initial cash `1,000,000`, fixed cash per trade `100,000`, market orders, and zero commission.
- Official improvements should preserve the required four factors and the index-regression trend. Appendix experiments may be kept for history, but they should not define the main final strategy.

## Why We Improved The Base Strategy

The base strategy worked in absolute terms, but the Monte Carlo test was not strong:

- Base strategy Sharpe: `1.0173`
- Base Monte Carlo p-value: `0.0000`

Interpretation: random top-10 S&P 500 portfolios under the same broad constraints often produced similar Sharpe ratios. That means the base result is not enough to claim strong stock-selection skill.

The individual FMPs also warned us not to overclaim. Their t-statistics and IC values are weak:

| approach | factor | annualized_sharpe | t_stat | p_value | avg_ic | avg_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| portfolio_sort | hzz_trend_improvement | 0.0734 | 0.2889 | 0.7730 | -0.0033 | -0.0004 |
| portfolio_sort | momentum | 0.0792 | 0.3444 | 0.7309 | 0.0209 | 0.0042 |
| portfolio_sort | pe | 0.2051 | 0.9153 | 0.3610 | -0.0084 | 0.0118 |
| portfolio_sort | roe | -0.2064 | -0.9213 | 0.3578 | -0.0016 | 0.0038 |
| portfolio_sort | trend | 0.0663 | 0.2654 | 0.7910 | -0.0017 | 0.0017 |
| cross_sectional_regression | hzz_trend_improvement | 0.0289 | 0.1136 | 0.9097 | -0.0033 | -0.0004 |
| cross_sectional_regression | momentum | 0.1987 | 0.8644 | 0.3883 | 0.0209 | 0.0042 |
| cross_sectional_regression | pe | -0.2403 | -1.0723 | 0.2847 | -0.0084 | 0.0118 |
| cross_sectional_regression | roe | -0.2055 | -0.9173 | 0.3599 | -0.0016 | 0.0038 |
| cross_sectional_regression | trend | -0.0177 | -0.0706 | 0.9438 | -0.0017 | 0.0017 |

## Development Path

We kept every strategy because it records the research path. The `assignment_scope` column separates official project-scope variants from appendix experiments.

| name | assignment_scope | annualized_sharpe | final_equity | max_drawdown | avg_positions | annualized_alpha_approx | beta_to_sp500 | avg_max_sector_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| value_quality_heavy_top10 | True | 1.0553 | 4058753.3865 | -0.1341 | 7.9958 | 0.0752 | -0.0227 | 0.3340 |
| hzz_sector_cap2_top15 | False | 1.0522 | 5759815.4046 | -0.1541 | 11.0917 | 0.0948 | -0.0283 | 0.2132 |
| base_equal_top10 | True | 1.0173 | 4585175.5438 | -0.1545 | 7.9958 | 0.0833 | -0.0355 | 0.3487 |
| hzz_sector_cap2_top10 | False | 0.9863 | 4320308.8236 | -0.1405 | 7.7250 | 0.0799 | -0.0317 | 0.2007 |
| hzz_low_vol_factor_top10 | False | 0.9747 | 4168087.3547 | -0.1437 | 7.7333 | 0.0780 | -0.0319 | 0.3536 |
| equal_top5 | True | 0.9680 | 2906694.8810 | -0.0918 | 4.0000 | 0.0559 | -0.0059 | 0.4417 |
| trend_momentum_heavy_top10 | True | 0.9577 | 4695284.2093 | -0.1756 | 7.9958 | 0.0838 | -0.0233 | 0.3706 |
| hzz_trend_composite_top10 | False | 0.9413 | 4484498.2020 | -0.1653 | 7.7250 | 0.0810 | -0.0191 | 0.3568 |
| trend_heavy_top10_regime | True | 0.9186 | 3549961.3014 | -0.1245 | 6.6250 | 0.0628 | 0.0353 | 0.3717 |
| hzz_sector_cap2_vol_penalty_top10 | False | 0.9156 | 3587687.6455 | -0.1397 | 7.7375 | 0.0693 | -0.0235 | 0.2004 |
| base_top10_stop10_take20 | True | 0.9050 | 3486751.0170 | -0.1591 | 7.9958 | 0.0680 | -0.0250 | 0.3487 |
| base_equal_top10_regime | True | 0.8968 | 3290488.3253 | -0.1327 | 6.6208 | 0.0591 | 0.0322 | 0.3417 |
| hzz_vol_penalty_top10 | False | 0.8850 | 3609066.2783 | -0.1672 | 7.7250 | 0.0702 | -0.0273 | 0.3490 |
| no_trend_top10 | False | 0.8486 | 5330140.9060 | -0.3719 | 9.3292 | 0.0839 | 0.0591 | 0.3461 |
| top5_regime_stop10_take20 | True | 0.6947 | 1863192.4954 | -0.1015 | 3.3125 | 0.0301 | 0.0219 | 0.4390 |

## Variant Notes

| name | assignment_scope | top_n | trend_col | regime_filter | stop_loss | take_profit | max_per_sector | volatility_penalty | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| value_quality_heavy_top10 | True | 10 | trend_z | False |  |  |  | 0.000 | Weight test: emphasize fundamentals. |
| hzz_sector_cap2_top15 | False | 15 | hzz_trend_z | False |  |  | 3.000 | 0.000 | Diversification test: HZZ score with 15 names and max three per sector. |
| base_equal_top10 | True | 10 | trend_z | False |  |  |  | 0.000 | Required assignment baseline: equal-weight ROE, P/E, momentum, and index-regression trend. |
| hzz_sector_cap2_top10 | False | 10 | hzz_trend_z | False |  |  | 2.000 | 0.000 | Risk-control improvement: HZZ score with max two holdings per GICS sector. |
| hzz_low_vol_factor_top10 | False | 10 | hzz_trend_z | False |  |  |  | 0.000 | Risk-score test: add low-volatility as a half-weight risk overlay. |
| equal_top5 | True | 5 | trend_z | False |  |  |  | 0.000 | Concentration test: same score, fewer names. |
| trend_momentum_heavy_top10 | True | 10 | trend_z | False |  |  |  | 0.000 | Weight test: emphasize price-based signals. |
| hzz_trend_composite_top10 | False | 10 | hzz_trend_z | False |  |  |  | 0.000 | Paper-inspired improvement: replace index-regression trend with cross-sectional HZZ trend. |
| trend_heavy_top10_regime | True | 10 | trend_z | True |  |  |  | 0.000 | Combined price-signal overweight and market-regime filter. |
| hzz_sector_cap2_vol_penalty_top10 | False | 10 | hzz_trend_z | False |  |  | 2.000 | 0.250 | Combined improvement: HZZ trend, sector cap, and volatility-aware ranking. |
| base_top10_stop10_take20 | True | 10 | trend_z | False | 0.100 | 0.200 |  | 0.000 | Risk management test: Backtrader stop-loss/take-profit improvement; vector curve is a monthly screening approximation. |
| base_equal_top10_regime | True | 10 | trend_z | True |  |  |  | 0.000 | Risk filter: trade only when S&P 500 is above its 10-month moving average. |
| hzz_vol_penalty_top10 | False | 10 | hzz_trend_z | False |  |  |  | 0.250 | Risk-control improvement: HZZ score penalized by 6-month realized volatility. |
| no_trend_top10 | False | 10 | trend_z | False |  |  |  | 0.000 | Ablation test: remove trend to see whether the trend factor helps. |
| top5_regime_stop10_take20 | True | 5 | trend_z | True | 0.100 | 0.200 |  | 0.000 | Combined concentration, regime, and Backtrader stop/take-profit improvement. |

## What Improved And Why

### 1. Assignment-Scope Improvements

The official improvement path keeps ROE, P/E, momentum, and the assignment-style `^GSPC` index-regression trend. We then test changes the project explicitly allows: factor weights, number of selected stocks, market-regime filtering, and stop-loss/take-profit order logic.

- Base vector Sharpe: `1.0173`
- Final selected assignment-scope vector Sharpe: `1.0553`
- Final selected strategy: `value_quality_heavy_top10`

### 2. Position Selection And Factor Weights

Top-5 versus top-10 tests show whether the signal is stronger in the highest-ranked stocks or whether diversification helps. Factor-weight variants test whether fundamentals or price-based signals deserve more emphasis.

### 3. Regime And Stop/Take-Profit Tests

The S&P 500 regime filter and stop-loss/take-profit variants answer the instructor's request to show improvements beyond the base strategy. Stop/take-profit is implemented in Backtrader for the final executable run when such a strategy is selected; vector curves are only screening approximations.

### 4. Appendix Experiments

HZZ cross-sectional trend, sector caps, and volatility-aware ranking are retained as appendix/history experiments. They are useful ideas, but they move beyond the literal assignment trend-factor construction, so they are not allowed to become the official final strategy in this version.

## Robustness Results

Monte Carlo:

- Base strategy p-value: `0.0000`
- Final selected strategy p-value: `0.0000`

Walk-forward:

| strategy | train_sharpe_to_2020 | test_sharpe_2021_2026 | test_cumulative_return_2021_2026 | test_max_drawdown_2021_2026 | selected_by_train |
| --- | --- | --- | --- | --- | --- |
| value_quality_heavy_top10 | 1.0677 | 1.0538 | 0.3397 | -0.0639 | True |
| base_equal_top10 | 0.9787 | 1.1673 | 0.4854 | -0.0620 | False |
| equal_top5 | 0.9275 | 1.0769 | 0.3651 | -0.0728 | False |
| base_top10_stop10_take20 | 0.9257 | 0.8420 | 0.3286 | -0.0521 | False |
| trend_momentum_heavy_top10 | 0.8956 | 1.2091 | 0.5323 | -0.0652 | False |
| trend_heavy_top10_regime | 0.8139 | 1.2084 | 0.5589 | -0.0540 | False |
| base_equal_top10_regime | 0.8115 | 1.1218 | 0.5037 | -0.0499 | False |
| top5_regime_stop10_take20 | 0.6696 | 0.7539 | 0.2127 | -0.0538 | False |

Benchmark comparison:

| strategy | annualized_excess_return_approx | excess_t_stat | excess_p_value | annualized_alpha_approx | alpha_t_stat | alpha_p_value | beta_to_sp500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hzz_sector_cap2_top15 | -0.0099 | -0.2459 | 0.8060 | 0.0948 | 4.9047 | 0.0000 | -0.0283 |
| no_trend_top10 | -0.0120 | -0.2987 | 0.7654 | 0.0839 | 2.9622 | 0.0031 | 0.0591 |
| trend_momentum_heavy_top10 | -0.0205 | -0.5119 | 0.6092 | 0.0838 | 4.0112 | 0.0001 | -0.0233 |
| base_equal_top10 | -0.0222 | -0.5597 | 0.5762 | 0.0833 | 4.3305 | 0.0000 | -0.0355 |
| hzz_trend_composite_top10 | -0.0229 | -0.5759 | 0.5653 | 0.0810 | 4.4952 | 0.0000 | -0.0191 |
| hzz_sector_cap2_top10 | -0.0252 | -0.6395 | 0.5231 | 0.0799 | 4.5878 | 0.0000 | -0.0317 |
| hzz_low_vol_factor_top10 | -0.0271 | -0.6894 | 0.4913 | 0.0780 | 4.5821 | 0.0000 | -0.0319 |
| value_quality_heavy_top10 | -0.0290 | -0.7558 | 0.4505 | 0.0752 | 4.6094 | 0.0000 | -0.0227 |
| hzz_vol_penalty_top10 | -0.0344 | -0.8780 | 0.3808 | 0.0702 | 3.9986 | 0.0001 | -0.0273 |
| hzz_sector_cap2_vol_penalty_top10 | -0.0350 | -0.9012 | 0.3684 | 0.0693 | 4.2353 | 0.0000 | -0.0235 |
| base_top10_stop10_take20 | -0.0364 | -0.9403 | 0.3480 | 0.0680 | 3.7390 | 0.0002 | -0.0250 |
| trend_heavy_top10_regime | -0.0355 | -0.9636 | 0.3362 | 0.0628 | 3.8168 | 0.0001 | 0.0353 |
| base_equal_top10_regime | -0.0395 | -1.0776 | 0.2823 | 0.0591 | 3.6885 | 0.0002 | 0.0322 |
| equal_top5 | -0.0466 | -1.2643 | 0.2073 | 0.0559 | 4.0240 | 0.0001 | -0.0059 |
| top5_regime_stop10_take20 | -0.0695 | -1.9759 | 0.0493 | 0.0301 | 3.0055 | 0.0027 | 0.0219 |

## Current Judgment

The project is stronger after adding assignment-scope improvement tests, final-strategy Monte Carlo, walk-forward validation, and benchmark alpha comparisons. However, the honest conclusion remains cautious:

- The base strategy is not statistically strong versus random portfolios.
- The final assignment-scope strategy is more promising, but still selected after testing multiple variants.
- Vector strategy variants are screening approximations; executable trading evidence comes from the saved Backtrader runs.
- The universe uses current S&P 500 constituents, so survivorship bias remains.
- Transaction costs and slippage are ignored because the assignment requires commission `0`.
- A production strategy would need historical index membership, costs, slippage, beta/sector neutrality, and a data-snooping-adjusted test such as White's Reality Check or Hansen's SPA.

## External Methodology Notes

- PyAnomaly emphasizes the same research building blocks we use: firm characteristics, winsorizing/trimming, quantile portfolios, long-short portfolios, factor regression, and cross-sectional regression.
- OpenSourceAP/CrossSection shows how much infrastructure serious factor replication needs: signal documentation, portfolio construction variants, and reproducibility checks.
- Quantitativo's trend-factor implementation stresses survivorship-bias-free universes, price filters, correct signal/return shifting, and the observation that post-2016 trend-factor edge can weaken.
- Sector and volatility controls are practical portfolio-construction overlays, not replacements for the required assignment factors.

## Ideas To Revisit Later

- Run a real White Reality Check or Hansen SPA test over all tried strategy variants.
- Use historical S&P 500 membership instead of current constituents.
- Add transaction costs and slippage despite the assignment's zero-commission rule, as an appendix.
- Revisit HZZ trend only as a clearly labeled appendix experiment unless the assignment scope changes.
- Test sector-neutral z-scoring separately from sector caps.
- Add beta-neutral or benchmark-relative optimization as an appendix, while keeping Backtrader fixed-cash results as the required core.
