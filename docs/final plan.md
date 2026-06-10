# Project Finalization Plan — Improveds 9 + 10, Aggregator, Figures, Doc Sync

## Context

The S&P 500 factor investing project is in finalization. Before the final end-to-end pipeline run, we need to land a coherent set of changes that close the remaining gaps identified in `docs/PROJECT_REVIEW_AND_FUTURE_WORK.md`. Specifically:

- Add **improved 9** = volatility-targeted top-20 (inverse-vol weights), built on improved 4's risk-managed foundation. This addresses the "no volatility-targeted sizing" gap (industry standard at AQR/Two Sigma) and provides a sister variant to improved 8's equal-weight 1/N for the wealth-vs-Sharpe trade-off discussion.
- Add **multi-comparison test as a robustness analysis** (NOT named "improved 10"). Hansen SPA + Romano-Wolf StepM via `arch.bootstrap` applied across all 9 strategy variants. Outputs land in `results/robustness/` to make clear this is a statistical correction, not a new strategy. Critical for any statistical-significance claim because we tested 9 variants and report the best.
- Build a **separate aggregator script** (`aggregate_all_strategies.py`) that produces unified walk-forward and benchmark-comparison tables covering all 9 strategies (currently restricted to the staged ladder of base + improved 1-3).
- Build a **separate figure script** (`make_presentation_figures.py`) producing 15+ presentation-grade figures and tables across all strategies (currently only 11 figures, none of which include improveds 4-9 on a single chart).
- Update **docs** (README, PROJECT_REPORT, PROJECT_PLAN, CODE_STRUCTURE, DATA_DICTIONARY) to reflect improveds 9 + 10 and the new aggregator/figures.
- Add a new doc **SIZING_AND_MARGIN.md** explaining the long-only money-growth mechanics, Margin rejection semantics, and how each sizer handles cash through time.

After these changes, the user runs:
1. `py -3.10 src\run_project.py` (full pipeline — base + improveds 1-3 with regenerated tables/figures and presentation)
2. `py -3.10 src\run_improved_4_stop_take_sensitivity.py`
3. `py -3.10 src\run_improved_5_regime_filter.py`
4. `py -3.10 src\run_improved_6_hzz_trend.py`
5. `py -3.10 src\run_improved_7_costs.py`
6. `py -3.10 src\run_improved_8_top_n_sizing.py`
7. `py -3.10 src\run_improved_9_vol_targeted.py` (NEW)
8. `py -3.10 src\aggregate_all_strategies.py` (NEW)
9. `py -3.10 src\run_multi_comparison_test.py` (NEW — robustness analysis, runs last because it needs all 9 strategies' curves)
10. `py -3.10 src\make_presentation_figures.py` (NEW)

The intended outcome is a project where every claim is statistically corrected for multi-comparison, every strategy is compared on the same walk-forward and benchmark grid, the presentation deck has comprehensive figures across all variants, and the docs accurately reflect the 10-strategy state.

---

## Implementation

### 1. `src/project_core.py` additions

**Constants** (insert near existing IMPROVED_8_* block):

```python
IMPROVED_9_RESULTS_DIR = RESULTS_DIR / "improved_strategy_9"
ROBUSTNESS_RESULTS_DIR = RESULTS_DIR / "robustness"
IMPROVED_9_STRATEGY_NAME = "improved_9_vol_targeted_top20"
VOL_LOOKBACK_DAYS = 63    # ~3 months of trading days
VOL_FLOOR = 0.05          # annualized vol floor (prevents division by tiny vols)
```

Add both `IMPROVED_9_RESULTS_DIR` and `ROBUSTNESS_RESULTS_DIR` to `ensure_dirs` list.

**StrategySpec field additions** (existing dataclass at lines 72-95):

Add two fields with defaults that preserve backwards compatibility:
- `vol_lookback_days: int = 63`
- `vol_floor: float = 0.05`

Keep `sizing_method` as the discriminator. Accept new value `"vol_targeted"` alongside existing `"fixed_cash"` and `"percent_of_equity"`.

**New Backtrader sizer `VolatilityTargetedSizer`** (insert after `EquityPercentSizer` around line 1338):

```python
class VolatilityTargetedSizer(bt.Sizer):
    """Inverse-volatility weighted sizing: each position sized so that
    expected risk contribution is equal across the held names.

    Position dollars = (1/vol_i) / sum(1/vol_j across held names) * portfolio_value
    where vol_i is annualized realized volatility over the last `vol_lookback_days`
    trading days, floored at `vol_floor` to prevent extreme concentration.
    """
    params = (
        ("vol_lookback_days", VOL_LOOKBACK_DAYS),
        ("vol_floor", VOL_FLOOR),
        ("target_n", 20),
    )
    def _getsizing(self, comminfo, cash, data, isbuy):
        # Compute realized vol from data.close history; flo and invert
        # Normalize against currently-held + targeted basket. See full impl in code.
        ...
```

**Vector engine extension** in `simulate_vector_strategy` (around line 1249):

Add a new branch when `spec.sizing_method == "vol_targeted"`:
```python
if spec.sizing_method == "vol_targeted":
    # Compute per-stock realized vol from the panel's price history
    selected = compute_inverse_vol_weights(selected, panel, spec.vol_lookback_days, spec.vol_floor)
    per_position_dollars = selected["vol_weight"] * equity
    # ...
```

Add helper `compute_inverse_vol_weights(selected_df, panel, lookback, floor)` near `position_size_for_spec`. Uses past 63 trading days of `next_ret_cc` (or computes from `prev_close → close` returns in monthly_stock_bars).

**Update `position_size_for_spec`** to handle the three sizing methods:
```python
def position_size_for_spec(spec, equity, per_stock_vol_weights=None):
    if spec.sizing_method == "vol_targeted":
        return per_stock_vol_weights * equity
    if spec.sizing_method == "percent_of_equity":
        return float(spec.sizing_target_pct) * float(equity)
    return CASH_PER_TRADE
```

**Update `select_positions_for_spec`** to treat `"vol_targeted"` like `"percent_of_equity"` for capacity (always supports target_n).

**Update Backtrader runner `run_backtrader_daily_stop_take`** to accept `sizing_method="vol_targeted"` and instantiate `VolatilityTargetedSizer`.

**Update `monte_carlo_random_portfolios`** to handle the new sizing rule (random portfolios sampled with the same vol-weighting).

---

### 2. `src/run_improved_9_vol_targeted.py` (NEW)

Mirrors `src/run_improved_8_top_n_sizing.py` structure. Key differences:

- Spec: `top_n=20`, `sizing_method="vol_targeted"`, `vol_lookback_days=63`, `stop_loss=0.05`, `take_profit=0.30`, `trend_col="trend_expanding_z"`, foundation = improved 4
- Saves vector + Backtrader + MC + bootstrap to `results/improved_strategy_9/`
- Writes `docs/IMPROVED_9_VOL_TARGETED.md` with full justification (cite AQR vol-targeting research, Carhart 1997, the "risk parity" literature)
- Appends section to `docs/STRATEGY_HISTORY.md`
- Compares against improved 4 (sister cost-robust variant) and improved 8 (sister 1/N variant)

Runtime: ~25 min (similar to improved 8).

---

### 3. `src/aggregate_all_strategies.py` (NEW)

Standalone script that produces unified comparison tables across all 9 strategies. Run AFTER all focused scripts.

```python
import project_core as pc
def main():
    # Read base + improved 1-3 curves from staged ladder
    staged = pd.read_csv("results/comparison/strategy_stage_curves.csv", parse_dates=["month"])
    # Read improveds 4, 5, 6, 8, 9 from per-strategy folders
    extras = []
    for d in [IMPROVED_4_RESULTS_DIR, IMPROVED_5_RESULTS_DIR, IMPROVED_6_RESULTS_DIR,
              IMPROVED_8_RESULTS_DIR, IMPROVED_9_RESULTS_DIR]:
        path = d / "vector_equity_curve.csv"
        if path.exists():
            curve = pd.read_csv(path, parse_dates=["month"])
            # Curves from focused scripts may already have "strategy" column; if not, add it
            if "strategy" not in curve.columns:
                curve["strategy"] = d.name
            extras.append(curve)
    all_curves = pd.concat([staged, *extras], ignore_index=True)
    # Call existing functions on full curve set
    wf = pc.walk_forward_summary(strategy_metrics=None, curves=all_curves)
    bench = pc.strategy_benchmark_comparison(all_curves, index_monthly)
    # Save with `all_strategies_` prefix to avoid clobbering staged-ladder versions
    pc.save_csv(wf, pc.COMPARISON_RESULTS_DIR / "all_strategies_walk_forward.csv")
    pc.save_csv(bench, pc.COMPARISON_RESULTS_DIR / "all_strategies_benchmark.csv")
    # Also build cross-strategy correlation matrix on monthly returns
    correlation = pivot_returns(all_curves).corr()
    pc.save_csv(correlation, pc.COMPARISON_RESULTS_DIR / "all_strategies_return_correlation.csv")
    # Per-strategy MC p-value table aggregated from each results/.../monte_carlo_random_portfolios.csv
    mc_table = aggregate_mc_pvalues()
    pc.save_csv(mc_table, pc.COMPARISON_RESULTS_DIR / "all_strategies_monte_carlo.csv")
```

Runtime: ~30 seconds.

---

### 4. `src/run_multi_comparison_test.py` (NEW — robustness analysis, NOT an "improved 10" strategy)

Applies Hansen SPA and Romano-Wolf StepM tests across all 9 strategies. Treated as a robustness layer (like improved 7), not a new strategy in the ladder. Requires all focused scripts to have run first.

```python
from arch.bootstrap import SPA, StepM
import project_core as pc

def main():
    # Load monthly returns from each strategy's vector_equity_curve.csv
    # Restrict to eval window
    returns_matrix = build_returns_matrix(strategies=ALL_9_STRATEGIES)
    # Hansen SPA: tests if the BEST strategy outperforms all others by chance
    # Loss function: -portfolio_return (lower = better strategy)
    spa = SPA(benchmark=returns_matrix["base"], models=returns_matrix.drop(columns=["base"]),
              reps=10000, block_size=6)
    spa.compute()
    # Romano-Wolf StepM: identifies WHICH strategies are individually significant
    stepm = StepM(benchmark=returns_matrix["base"], models=returns_matrix.drop(columns=["base"]),
                  reps=10000, block_size=6)
    stepm.compute()
    # Save adjusted p-values per strategy
    pc.save_csv(spa_results, ROBUSTNESS_RESULTS_DIR / "hansen_spa_results.csv")
    pc.save_csv(stepm_results, ROBUSTNESS_RESULTS_DIR / "romano_wolf_stepm_results.csv")
    write_doc(...)
```

Writes `docs/MULTI_COMPARISON_TEST.md` with methodology and corrected per-strategy p-values. Does NOT append to `STRATEGY_HISTORY.md` because it isn't a strategy — instead, adds a "Robustness" section to that doc or lets the README document it. Runtime: ~5 min (just stats on existing curves).

---

### 5. `src/make_presentation_figures.py` (NEW)

Builds 15+ presentation-grade figures and tables. Run after aggregator.

Figures to generate (saved to `figures/`):

1. `all_strategies_equity_curves.png` — all 9 strategy equity curves on one chart, with ^GSPC overlay
2. `all_strategies_drawdowns.png` — drawdowns over time, all 9 strategies
3. `sharpe_vs_drawdown_scatter.png` — risk-return scatter (annualized Sharpe × max DD), bubbled by final equity
4. `monthly_returns_heatmap_imp4.png` — calendar heatmap (year × month) for improved 4
5. `monthly_returns_heatmap_imp8.png` — same for improved 8
6. `rolling_12m_sharpe.png` — rolling 12-month Sharpe for top 4 strategies
7. `annual_returns_bar_chart.png` — calendar-year returns per strategy
8. `per_sector_contribution_imp4.png` — stacked bar of return contribution by GICS sector
9. `avg_position_count_over_time.png` — n_positions monthly for each strategy
10. `composite_score_distribution.png` — boxplot of composite z per month
11. `factor_weight_evolution_imp3.png` — improved 3's dynamic weights over time
12. `cost_drag_attribution_imp7.png` — stacked bar of cost drag per year per scenario (improved 7)
13. `walk_forward_train_vs_test_scatter.png` — train Sharpe × test Sharpe for all 9
14. `cumulative_alpha_vs_gspc.png` — running cumulative alpha for top strategies
15. `position_concentration_hhi.png` — Herfindahl-Hirschman index over time

Tables saved to `results/comparison/` (in addition to aggregator outputs):

- `annual_returns_table.csv` — year × strategy matrix
- `best_worst_months_per_strategy.csv` — top 10 / bottom 10
- `hit_rate_per_strategy.csv` — % positive months
- `tail_risk_metrics.csv` — Sortino, Calmar, Omega
- `cross_strategy_correlation.csv` — already produced by aggregator
- `avg_days_held_per_stock.csv` — turnover characterization

Runtime: ~2-3 min.

---

### 6. `make_presentation` updates in `project_core.py`

Replace the hardcoded slide sequence (lines 2870-2951) with a more structured deck that includes:

- New slide: "Strategy Ladder Overview" — 1-page table of all 9 strategy variants (base + improved 1-9)
- New slide: "Multi-Comparison Robustness Test" — Hansen SPA + Romano-Wolf results across all 9 strategies
- New slide: "All Strategies Equity Comparison" — the new equity-curves figure
- New slide: "Risk-Return Frontier" — Sharpe-vs-drawdown scatter
- New slide: "Cost Sensitivity" — improved 7 cost-drag figure
- New slide: "Volatility-Targeted vs Equal-Weight (Improved 8 vs 9)" — sister-strategy comparison
- New slide: "Honest Limitations + Future Work" — survivorship bias, WRDS plan

Total presentation grows from ~12 slides to ~20.

---

### 7. Documentation updates

**README.md** — add subsection for improved 9 in Section 6 (Strategy Ladder). Add a new "Multi-Comparison Robustness Test" subsection under Section 10 (Robustness). Update Section 9 (Results) to include all 9 strategy variants plus the multi-comparison corrected p-value column.

**docs/PROJECT_REPORT.md** — extend results table to include improved 9. Add multi-comparison-corrected p-value column. Add a dedicated "Multi-Comparison Robustness" section.

**docs/PROJECT_PLAN.md** — extend Strategy Ladder table to 9 strategy variants (base + improved 1-9). Add the multi-comparison test as a robustness layer note (parallel to improved 7), explicitly NOT in the strategy ladder.

**docs/CODE_STRUCTURE.md** — add `run_improved_9_vol_targeted.py`, `run_multi_comparison_test.py`, `aggregate_all_strategies.py`, `make_presentation_figures.py`. Add `results/robustness/` to result folders list.

**docs/DATA_DICTIONARY.md** — add `results/improved_strategy_9/` and `results/robustness/` file lists.

**docs/REPRODUCIBILITY.md** — update run order to include improved 9 + aggregator + multi-comparison test + figure builder.

**NEW: docs/SIZING_AND_MARGIN.md** — answers the user's Q3. Explains:
- How `FixedCashSizer` causes Margin rejections (when realized cost > free cash due to integer rounding)
- That "Margin rejections" mean some positions don't get filled → realized portfolio < target top-N
- How the long-only strategy holds excess cash idle (no money-market rate modeled, earns 0)
- How `EquityPercentSizer` solves this (1/N scales dynamically with equity)
- How `VolatilityTargetedSizer` further refines (inverse-vol weights normalized within the basket)
- Worked numerical example: $1M → $2M growth, what happens to position size under each sizer
- Why the strategy "can't buy more" in fixed-cash mode even when equity grows (top_n is a hard cap)

---

## Critical files to be modified

- `src/project_core.py` — add IMPROVED_9_*, ROBUSTNESS_RESULTS_DIR, `VolatilityTargetedSizer`, `compute_inverse_vol_weights`, extend `position_size_for_spec`, `select_positions_for_spec`, `simulate_vector_strategy`, `run_backtrader_daily_stop_take`, `monte_carlo_random_portfolios` for `"vol_targeted"` sizing; update `make_presentation` slide sequence
- `docs/PROJECT_REPORT.md`, `docs/PROJECT_PLAN.md`, `docs/CODE_STRUCTURE.md`, `docs/DATA_DICTIONARY.md`, `docs/REPRODUCIBILITY.md` — sync to 9-strategy + multi-comparison-robustness state
- `README.md` — extend strategy ladder (9 variants), results, robustness sections; add multi-comparison test as a Section 10 subsection

## New files to be created

- `src/run_improved_9_vol_targeted.py`
- `src/run_multi_comparison_test.py` (NOT named improved 10)
- `src/aggregate_all_strategies.py`
- `src/make_presentation_figures.py`
- `docs/IMPROVED_9_VOL_TARGETED.md`
- `docs/MULTI_COMPARISON_TEST.md` (NOT IMPROVED_10_*.md)
- `docs/SIZING_AND_MARGIN.md`

---

## Existing functions reused (do not reinvent)

- `pc.metrics_over_evaluation_window`, `pc.filter_to_evaluation_window` — eval-window metrics
- `pc.walk_forward_summary`, `pc.strategy_benchmark_comparison` — already generic over `curves`, just need the full set
- `pc.monte_carlo_random_portfolios`, `pc.block_bootstrap` — reused by improved 9
- `pc.run_backtrader_daily_stop_take` — extended to accept `"vol_targeted"` sizing
- `pc.compute_stock_ma_signals`, `pc.cross_sectional_trend_betas`, `pc.smooth_trend_betas`, `pc.hzz_predicted_returns` — unchanged
- `arch.bootstrap.SPA`, `arch.bootstrap.StepM` — for the multi-comparison robustness test
- `pc.make_drawdown` (line 2479) — used in new presentation figures
- `pc.perf_metrics` — for tail-risk metric variants (Sortino requires downside-only std)

---

## Verification

After implementation:

1. **Compile check** (under 30 seconds):
   ```powershell
   py -3.10 -m py_compile src\project_core.py src\run_improved_9_vol_targeted.py ^
       src\run_multi_comparison_test.py src\aggregate_all_strategies.py ^
       src\make_presentation_figures.py
   py -3.10 -c "import sys; sys.path.insert(0, 'src'); import project_core; print('OK')"
   ```

2. **Verify VolatilityTargetedSizer math** on synthetic data (no full pipeline needed):
   ```powershell
   py -3.10 -c "
   import sys; sys.path.insert(0, 'src')
   import project_core as pc
   import numpy as np
   import pandas as pd
   # Synthetic test of compute_inverse_vol_weights
   # Expected: high-vol stock gets smaller weight; sum of weights = 1
   ...
   "
   ```

3. **Run the full sequence** (user does this):
   ```powershell
   py -3.10 src\run_project.py
   py -3.10 src\run_improved_4_stop_take_sensitivity.py
   py -3.10 src\run_improved_5_regime_filter.py
   py -3.10 src\run_improved_6_hzz_trend.py
   py -3.10 src\run_improved_7_costs.py
   py -3.10 src\run_improved_8_top_n_sizing.py
   py -3.10 src\run_improved_9_vol_targeted.py
   py -3.10 src\aggregate_all_strategies.py
   py -3.10 src\run_multi_comparison_test.py
   py -3.10 src\make_presentation_figures.py
   ```

4. **Sanity-check the outputs**:
   - `results/comparison/all_strategies_walk_forward.csv` has 9 rows
   - `results/comparison/all_strategies_benchmark.csv` has 9 rows
   - `results/comparison/all_strategies_return_correlation.csv` is a 9×9 matrix
   - `results/robustness/hansen_spa_results.csv` has consensus p-value
   - `results/robustness/romano_wolf_stepm_results.csv` has per-strategy adjusted p-values
   - `figures/` contains 15+ new PNG files
   - `presentation/sp500_factor_investing_presentation.pdf` has the expanded slide deck

5. **Long-only assertion** runs after each Backtrader output (existing safety check); no negative positions allowed.

6. **MC p-value reconciliation**: improved 9's raw MC p-value vs Romano-Wolf-corrected p-value should differ (corrected ≥ raw); document the magnitude.

---

## Estimated work

- Code implementation: ~5-6 hours
- User-side pipeline reruns: ~2.5-3 hours (sequential, mostly waiting)
- Final commit + push: ~5 minutes

Total elapsed time before final state: ~8-9 hours including the user's pipeline runs.
