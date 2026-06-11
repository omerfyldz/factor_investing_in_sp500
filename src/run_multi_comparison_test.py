"""Multi-comparison robustness test — Hansen SPA and Romano-Wolf StepM.

This is NOT a strategy. It is a statistical robustness layer (parallel to
improved 7's cost-sensitivity analysis) that addresses the multi-comparison
problem: we tested 8 strategy variants and report the best, so any raw
Monte Carlo p-value is biased downward. Hansen's (2005) SPA test and the
Romano-Wolf (2005) StepM procedure correct for this by controlling the
family-wise error rate across the full search space of strategies tried.

Both tests are implemented in the ``arch.bootstrap`` package and use a
stationary block bootstrap of monthly excess returns. The benchmark is
^GSPC (the natural "no-skill" benchmark in factor investing).

Outputs:
- ``results/robustness/hansen_spa_results.csv`` -- SPA test statistic and
  p-values (consistent / lower / upper) under the null that no strategy
  significantly outperforms the benchmark
- ``results/robustness/romano_wolf_stepm_results.csv`` -- per-strategy
  adjusted significance markers from the StepM step-down procedure
- ``results/robustness/strategy_excess_returns.csv`` -- the input return
  matrix used by both tests
- ``docs/MULTI_COMPARISON_TEST.md`` -- methodology and interpretation

Run AFTER all 8 strategies have completed and ``aggregate_all_strategies.py``
has run. Runtime: ~5 minutes (bootstrap is the bottleneck).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import project_core as core

try:
    from arch.bootstrap import SPA, StepM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "arch package required: `py -3.10 -m pip install arch`. "
        f"Original error: {exc}"
    ) from exc


STRATEGY_FOLDERS: dict[str, Path] = {
    "base": core.BASE_RESULTS_DIR,
    "improved_1": core.IMPROVED_RESULTS_DIR,
    "improved_2": core.IMPROVED_2_RESULTS_DIR,
    "improved_3": core.IMPROVED_3_RESULTS_DIR,
    "improved_4": core.IMPROVED_4_RESULTS_DIR,
    "improved_5": core.IMPROVED_5_RESULTS_DIR,
    "improved_6": core.IMPROVED_6_RESULTS_DIR,
    "improved_8": core.IMPROVED_8_RESULTS_DIR,
}

BOOTSTRAP_REPS = 10_000
BLOCK_SIZE = 6  # months
SIGNIFICANCE_LEVEL = 0.05


def load_strategy_returns() -> pd.DataFrame:
    """Build a wide returns matrix (rows = months, columns = strategies)."""
    series_dict: dict[str, pd.Series] = {}
    for label, folder in STRATEGY_FOLDERS.items():
        path = folder / "vector_equity_curve.csv"
        if not path.exists():
            print(f"WARNING: {path} not found; skipping {label}")
            continue
        df = pd.read_csv(path, parse_dates=["month"])
        df = df.sort_values("month")
        s = df.set_index("month")["portfolio_return"].astype(float)
        series_dict[label] = s
    if not series_dict:
        raise RuntimeError("No strategy curves found. Run focused scripts first.")
    wide = pd.DataFrame(series_dict)
    # Restrict to eval window
    wide = wide[wide.index >= core.EVALUATION_START]
    # Drop months where any strategy is NaN to align the bootstrap samples
    wide = wide.dropna(how="any")
    return wide


def load_benchmark_returns(months: pd.DatetimeIndex) -> pd.Series:
    """Compute month-over-month ^GSPC returns aligned to the strategy months."""
    _, index_monthly, _ = core.load_processed_strategy_inputs()
    idx = index_monthly[["month", "close"]].sort_values("month").copy()
    idx["benchmark_return"] = idx["close"].pct_change()
    idx = idx.set_index("month")
    return idx["benchmark_return"].reindex(months).fillna(0.0)


def run_hansen_spa(
    strategy_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> pd.DataFrame:
    """Run Hansen (2005) SPA test.

    Tests the null: the benchmark is not inferior to ANY strategy after
    accounting for the full search space. A small p-value rejects the null
    and indicates at least one strategy genuinely outperforms.

    Loss function: negative excess return (lower loss = better). SPA tests
    whether the BEST model has loss significantly LOWER than the benchmark's
    loss.
    """
    print(f"Running Hansen SPA test with {BOOTSTRAP_REPS} bootstrap reps...")
    # Loss for each model = -excess_return; lower is better.
    # arch's SPA takes (benchmark_losses, model_losses) arrays.
    benchmark_loss = -benchmark_returns.values  # shape (T,)
    model_losses = -strategy_returns.values     # shape (T, K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spa = SPA(benchmark_loss, model_losses, reps=BOOTSTRAP_REPS, block_size=BLOCK_SIZE)
        spa.compute()
    pvals = spa.pvalues  # consistent, lower, upper
    return pd.DataFrame(
        [
            {
                "test": "Hansen_SPA",
                "n_models": int(strategy_returns.shape[1]),
                "n_periods": int(strategy_returns.shape[0]),
                "bootstrap_reps": BOOTSTRAP_REPS,
                "block_size": BLOCK_SIZE,
                "p_value_consistent": float(pvals["consistent"]),
                "p_value_lower": float(pvals["lower"]),
                "p_value_upper": float(pvals["upper"]),
                "interpretation": (
                    "Small p_value_consistent (< 0.05) rejects the null that no "
                    "strategy outperforms the benchmark after multi-comparison "
                    "correction. lower/upper are bounds under different "
                    "studentization assumptions."
                ),
            }
        ]
    )


def run_romano_wolf_stepm(
    strategy_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> pd.DataFrame:
    """Run Romano-Wolf StepM (2005) step-down procedure.

    Identifies WHICH individual strategies significantly outperform the
    benchmark while controlling the family-wise error rate. Returns one row
    per strategy with a `superior_to_benchmark` flag.
    """
    print(f"Running Romano-Wolf StepM with {BOOTSTRAP_REPS} bootstrap reps...")
    benchmark_loss = -benchmark_returns.values
    model_losses = -strategy_returns.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stepm = StepM(
            benchmark_loss,
            model_losses,
            size=SIGNIFICANCE_LEVEL,
            reps=BOOTSTRAP_REPS,
            block_size=BLOCK_SIZE,
        )
        stepm.compute()
    # arch's StepM exposes .superior_models (column indices of significant ones)
    superior_indices = set(stepm.superior_models)
    rows = []
    for i, strategy_label in enumerate(strategy_returns.columns):
        mean_excess = float(
            (strategy_returns[strategy_label] - benchmark_returns).mean()
        )
        annualized_excess = mean_excess * 12.0
        rows.append(
            {
                "strategy": strategy_label,
                "mean_monthly_excess_return": mean_excess,
                "annualized_excess_return_approx": annualized_excess,
                "superior_to_benchmark_after_stepm": strategy_label in superior_indices
                or i in superior_indices,
                "significance_level": SIGNIFICANCE_LEVEL,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "block_size": BLOCK_SIZE,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "annualized_excess_return_approx", ascending=False
    )


def write_doc(
    spa_results: pd.DataFrame,
    stepm_results: pd.DataFrame,
    strategy_returns: pd.DataFrame,
) -> None:
    spa_row = spa_results.iloc[0]
    significant_strategies = stepm_results[
        stepm_results["superior_to_benchmark_after_stepm"]
    ]["strategy"].tolist()
    n_significant = len(significant_strategies)
    n_total = int(strategy_returns.shape[1])

    stepm_table_rows = [
        "| Strategy | Annualized Excess Return | Superior after StepM? |",
        "|---|---:|:---:|",
    ]
    for _, row in stepm_results.iterrows():
        flag = "YES" if row["superior_to_benchmark_after_stepm"] else "no"
        stepm_table_rows.append(
            f"| {row['strategy']} | {row['annualized_excess_return_approx']:+.2%} | {flag} |"
        )
    stepm_table = "\n".join(stepm_table_rows)

    content = f"""# Multi-Comparison Robustness Test — Hansen SPA + Romano-Wolf StepM

This is **not a strategy.** It is a statistical robustness layer that
addresses the multi-comparison problem: we tested 8 strategy variants
(base + improved 1-6, 8) and report the best, so any single Monte Carlo
p-value is biased downward by the search-space size.

## Why this matters

With 8 strategies tested at the 5% significance level, the probability that
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

**Inputs**: 8 strategy monthly returns × {strategy_returns.shape[0]} months from
{strategy_returns.index.min().date()} to {strategy_returns.index.max().date()}
(the common evaluation window). Benchmark is ^GSPC monthly returns over the
same window. Loss function = `-excess_return` (lower loss = better strategy).
Bootstrap: {BOOTSTRAP_REPS:,} reps, {BLOCK_SIZE}-month blocks.

**Result**:

| Statistic | Value |
|---|---:|
| Consistent p-value | `{spa_row['p_value_consistent']:.4f}` |
| Lower p-value | `{spa_row['p_value_lower']:.4f}` |
| Upper p-value | `{spa_row['p_value_upper']:.4f}` |

**Interpretation**: a small consistent p-value (< 0.05) rejects the null that
no strategy outperforms after multi-comparison correction. The lower and upper
p-values are bounds under different studentization assumptions; the consistent
version is the standard headline.

## Romano-Wolf StepM (2005) step-down procedure

Hansen's SPA gives a single yes/no verdict for the best strategy. Romano-Wolf
StepM goes further: it identifies WHICH individual strategies survive the
multi-comparison correction, controlling the family-wise error rate at the
{SIGNIFICANCE_LEVEL:.0%} level.

**Result**: {n_significant} of {n_total} strategies survive Romano-Wolf at
the {SIGNIFICANCE_LEVEL:.0%} level.

{stepm_table}

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

- **Block size** of {BLOCK_SIZE} months is the project default. Larger blocks
  (12, 24) capture more serial correlation; smaller (3) are more powerful but
  may under-represent the true autocorrelation structure of monthly factor
  returns.
- **Survivorship-biased universe.** Both the strategies and the benchmark
  draw from the current S&P 500. The test cannot correct for the universe
  bias — only for the multi-comparison bias.
- **Excess returns vs ^GSPC** assumes the benchmark is the natural "no skill"
  comparison. A more demanding test would use a factor-based benchmark
  (Fama-French 5, Carhart 4); planned for future work.
- **{BOOTSTRAP_REPS:,} bootstrap reps** is sufficient for stable p-value
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
"""
    (core.DOCS_DIR / "MULTI_COMPARISON_TEST.md").write_text(content, encoding="utf-8")


def main() -> None:
    core.ensure_dirs()
    print("Running multi-comparison robustness test (Hansen SPA + Romano-Wolf StepM)...")

    strategy_returns = load_strategy_returns()
    if strategy_returns.empty:
        print("No aligned strategy returns. Aborting.")
        return
    print(
        f"Loaded {strategy_returns.shape[1]} strategies, "
        f"{strategy_returns.shape[0]} eval months "
        f"({strategy_returns.index.min().date()} to {strategy_returns.index.max().date()})"
    )

    benchmark_returns = load_benchmark_returns(strategy_returns.index)

    # Save the input matrix for reproducibility
    excess_df = strategy_returns.subtract(benchmark_returns, axis=0)
    excess_df.index.name = "month"
    core.save_csv(
        excess_df.reset_index(),
        core.ROBUSTNESS_RESULTS_DIR / "strategy_excess_returns.csv",
        index=False,
    )

    spa_results = run_hansen_spa(strategy_returns, benchmark_returns)
    core.save_csv(spa_results, core.ROBUSTNESS_RESULTS_DIR / "hansen_spa_results.csv")

    stepm_results = run_romano_wolf_stepm(strategy_returns, benchmark_returns)
    core.save_csv(
        stepm_results, core.ROBUSTNESS_RESULTS_DIR / "romano_wolf_stepm_results.csv"
    )

    write_doc(spa_results, stepm_results, strategy_returns)

    print("Multi-comparison test complete.")
    print()
    print("=== Hansen SPA ===")
    print(spa_results.to_string(index=False))
    print()
    print("=== Romano-Wolf StepM ===")
    print(stepm_results.to_string(index=False))


if __name__ == "__main__":
    main()
