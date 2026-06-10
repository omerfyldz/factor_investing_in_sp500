# Multi-Comparison Robustness Test — Hansen SPA + Romano-Wolf StepM

This is **not a strategy.** It is a statistical robustness layer that
addresses the multi-comparison problem: we tested 9 strategy variants
(base + improved 1-6, 8, 9) and report the best, so any single Monte Carlo
p-value is biased downward by the search-space size.

## Why this matters

With 9 strategies tested at the 5% significance level, the probability that
at least ONE strategy beats random by chance is roughly
`1 - (1 - 0.05)^9 = 37%`. Without correction, the project's per-strategy
Monte Carlo p-values are not trustworthy as standalone significance claims.

Two complementary tests close this gap.

## Hansen (2005) SPA test

Hansen's Superior Predictive Ability test asks: **is the single best strategy
significantly better than the benchmark after accounting for the full search
space?** It uses a stationary block bootstrap of the loss differences between
each model and the benchmark, then computes a multi-comparison-corrected
p-value.

**Inputs**: 9 strategy monthly returns × 121 months from
2016-05-31 to 2026-05-31
(the common evaluation window). Benchmark is ^GSPC monthly returns over the
same window. Loss function = `-excess_return` (lower loss = better strategy).
Bootstrap: 10,000 reps, 6-month blocks.

**Result**:

| Statistic | Value |
|---|---:|
| Consistent p-value | `0.6737` |
| Lower p-value | `0.5404` |
| Upper p-value | `0.7379` |

**Interpretation**: a small consistent p-value (< 0.05) rejects the null that
no strategy outperforms after multi-comparison correction. The lower and upper
p-values are bounds under different studentization assumptions; the consistent
version is the standard headline.

## Romano-Wolf StepM (2005) step-down procedure

Hansen's SPA gives a single yes/no verdict for the best strategy. Romano-Wolf
StepM goes further: it identifies WHICH individual strategies survive the
multi-comparison correction, controlling the family-wise error rate at the
5% level.

**Result**: 0 of 8 strategies survive Romano-Wolf at
the 5% level.

| Strategy | Annualized Excess Return | Superior after StepM? |
|---|---:|:---:|
| improved_8 | -0.42% | no |
| improved_1 | -1.01% | no |
| improved_4 | -3.64% | no |
| improved_2 | -3.67% | no |
| improved_3 | -3.95% | no |
| improved_5 | -6.17% | no |
| improved_6 | -6.78% | no |
| base | -6.90% | no |

## Interpretation

- **Strategies marked YES survive multi-comparison correction.** Their
  outperformance of ^GSPC is statistically significant even after accounting
  for the fact that we tested 9 variants.
- **Strategies marked no do NOT necessarily lack edge** — they just don't
  meet the strict family-wise error control bar. Their per-strategy Monte
  Carlo p-values (uncorrected) may still be informative.
- **Hansen SPA's consistent p-value** is the answer to "is there at least
  ONE good strategy in this batch?" If it's small, the project's best
  strategy is genuinely better than random; if large, the best is
  indistinguishable from luck given 9 tries.

## Caveats

- **Block size** of 6 months is the project default. Larger blocks
  (12, 24) capture more serial correlation; smaller (3) are more powerful but
  may under-represent the true autocorrelation structure of monthly factor
  returns.
- **Survivorship-biased universe.** Both the strategies and the benchmark
  draw from the current S&P 500. The test cannot correct for the universe
  bias — only for the multi-comparison bias.
- **Excess returns vs ^GSPC** assumes the benchmark is the natural "no skill"
  comparison. A more demanding test would use a factor-based benchmark
  (Fama-French 5, Carhart 4); planned for future work.
- **10,000 bootstrap reps** is sufficient for stable p-value
  estimates to 3 decimal places. Increasing further gives diminishing returns.

## References

- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of
  Business & Economic Statistics*, 23(4), 365-380.
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized
  Data Snooping. *Econometrica*, 73(4), 1237-1282.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*,
  68(5), 1097-1126. — The original "data snooping" framework that SPA
  refines.
- Implemented via the ``arch.bootstrap`` package (Kevin Sheppard).
