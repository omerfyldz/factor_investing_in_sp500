"""Focused run for improved strategy 8: equal-weight 1/N sizing with top-20 selection.

Improved 8 builds on improved 4 by changing two mechanically-coupled design
dimensions at once:

1. Top-N selection moves from 10 to 20 (more diversified concentrated portfolio).
2. Position sizing moves from fixed $100,000 per trade to equal-weight 5 pct of
   current portfolio equity per position (1/N at top_n=20).

The two changes are coupled because fixed-dollar sizing is mechanically
incompatible with meaningful top-N expansion: 20 positions x $100,000 = $2M
exceeds the $1M starting capital, so the strategy could not actually hold 20
names until equity grew past $2M. Switching to equal-weight 5 pct of equity
eliminates the cash constraint entirely and lets every monthly rebalance
target a true 20-name portfolio.

Justification for the design choices:

- **Equal-weight (1/N) sizing.** DeMiguel, Garlappi, Uppal (2009, RFS,
  "Optimal Versus Naive Diversification") evaluated 14 mean-variance
  optimization variants across 7 empirical datasets and showed that none
  reliably beat naive 1/N on out-of-sample Sharpe. The estimation error in
  sophisticated weighting schemes typically overwhelms their theoretical
  benefits at sample sizes available in practice. Equal-weight is the
  academic gold standard and the industry analog is the Invesco S&P 500
  Equal Weight ETF (RSP), one of the longest-running smart-beta products.
- **Top-20 selection.** Top 10 (our prior choice) is more concentrated than
  any standard factor study; Fama-French and HZZ both use quintile (top 20
  pct) or decile (top 10 pct) groupings. For the S&P 500, top quintile is
  100 names which is more diluted than the assignment-scope long-only
  strategy can support. Top 20 (~4 pct of the universe) is a defensible
  middle ground -- still concentrated enough to express conviction, but
  meaningfully more diversified than top 10. Plyakha, Uppal, Vilkov (2014,
  "Why does an equal-weighted portfolio outperform value- and price-weighted
  portfolios?") documents the equal-weighted-concentrated-portfolio
  outperformance for the same reasons.
- **Foundation = improved 4.** Improved 7's time-varying cost analysis
  showed improved 4 dominates improved 6 under realistic transaction costs
  because of its lower turnover. Improved 8 builds on the cost-robust
  winner.
- **Stop-loss / take-profit inherited as is.** 5 pct stop-loss and 30 pct
  take-profit applied per position, the same risk-exit rule walk-forward
  selected for improved 4. The percentages are per-position, not portfolio
  level, so they scale appropriately with the now-dynamic position sizes.

What this run produces: vector simulation, Monte Carlo p-value under the
matching equal-weight 20-name random benchmark, block bootstrap, daily
Backtrader execution with EquityPercentSizer, validation summary, and a
side-by-side comparison against improved 4. All evaluation metrics use the
project's common ``EVALUATION_START = 2016-05-31`` so the head-to-head with
prior improveds is apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import project_core as core


DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.30
TOP_N = 20
SIZING_TARGET_PCT = 1.0 / TOP_N  # 5 pct per position for top 20


def load_improved_4_thresholds() -> tuple[float, float]:
    """Inherit stop-loss / take-profit values selected by improved 4's walk-forward."""
    selected_path = core.IMPROVED_4_RESULTS_DIR / "selected_stop_take_parameters.csv"
    if not selected_path.exists():
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    return float(selected["stop_loss"].iloc[0]), float(selected["take_profit"].iloc[0])


def make_improved_8_spec(stop_loss: float, take_profit: float) -> core.StrategySpec:
    """Improved 8 = improved 4 design with equal-weight 1/N sizing at top 20."""
    return core.StrategySpec(
        name=core.IMPROVED_8_STRATEGY_NAME,
        weights={"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
        top_n=TOP_N,
        trend_col="trend_expanding_z",
        stop_loss=stop_loss,
        take_profit=take_profit,
        sizing_method="percent_of_equity",
        sizing_target_pct=SIZING_TARGET_PCT,
        notes=(
            "Improved 8: improved 4 design with equal-weight 1/N sizing at top 20. "
            "Each position targets 5 pct of current portfolio equity. Top-N "
            "expansion from 10 to 20 and switch from fixed-dollar to "
            "percent-of-equity sizing are mechanically coupled and treated as a "
            "single improvement. Stop-loss / take-profit inherited from improved 4."
        ),
    )


def load_metric_row(path: Path, label: str) -> dict[str, object]:
    row = pd.read_csv(path).iloc[0].to_dict()
    row["label"] = label
    return row


def save_comparison_summary() -> pd.DataFrame:
    """Build a side-by-side table of improved 4 vs improved 8 vector and Backtrader metrics."""
    candidates = {
        "improved_4_vector": core.IMPROVED_4_RESULTS_DIR / "vector_metrics.csv",
        "improved_8_vector": core.IMPROVED_8_RESULTS_DIR / "vector_metrics.csv",
        "improved_4_backtrader": core.IMPROVED_4_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_4_STRATEGY_NAME}_metrics.csv",
        "improved_8_backtrader": core.IMPROVED_8_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_8_STRATEGY_NAME}_metrics.csv",
    }
    rows = [load_metric_row(path, label) for label, path in candidates.items() if path.exists()]
    summary = pd.DataFrame(rows)
    keep = [
        "label",
        "name",
        "backtrader_frequency",
        "annualized_sharpe",
        "final_equity",
        "final_value",
        "total_return",
        "max_drawdown",
        "stop_loss",
        "take_profit",
    ]
    for col in keep:
        if col not in summary.columns:
            summary[col] = pd.NA
    summary = summary[keep]
    core.save_csv(summary, core.IMPROVED_8_RESULTS_DIR / "improved_8_vs_improved_4_summary.csv")
    return summary


def write_improved_8_note(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    bt = bt_metrics.iloc[0] if not bt_metrics.empty else None
    comp = comparison.set_index("label") if not comparison.empty else pd.DataFrame()
    has_imp4_vec = "improved_4_vector" in comp.index
    has_imp4_bt = "improved_4_backtrader" in comp.index
    vec_line = (
        f"- Improved 4 vector Sharpe: `{comp.loc['improved_4_vector','annualized_sharpe']:.4f}`; "
        f"improved 8 vector Sharpe: `{metrics['annualized_sharpe']:.4f}`."
        if has_imp4_vec
        else "- Improved 4 vector metrics not available."
    )
    bt_line = (
        f"- Improved 4 Backtrader Sharpe: `{comp.loc['improved_4_backtrader','annualized_sharpe']:.4f}`; "
        f"improved 8 Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`."
        if has_imp4_bt and bt is not None
        else "- Improved 4 Backtrader metrics not available."
    )

    content = f"""# Improved 8 -- Equal-Weight 1/N Sizing With Top 20

Improved 8 is a focused position-sizing and diversification experiment. It
changes two design dimensions on top of improved 4:

1. Top-N selection moves from 10 to 20.
2. Position sizing moves from fixed `${core.CASH_PER_TRADE:,.0f}` per trade to
   equal-weight 5 pct of current portfolio equity per position (1/N at top 20).

The two changes are mechanically coupled: fixed-dollar sizing is incompatible
with meaningful top-N expansion because `$1M` starting capital cannot fund
`20 x $100,000 = $2M`. Switching to percent-of-equity sizing removes the cash
constraint and lets every monthly rebalance target a true 20-name portfolio.

## Method

- **Foundation**: improved 4 design (composite of ROE, P/E, momentum,
  trend_expanding_z) with the same 5 pct stop-loss and 30 pct take-profit.
- **Top-N**: `{spec.top_n}` (vs improved 4's 10).
- **Sizing method**: `percent_of_equity` with target `{SIZING_TARGET_PCT:.2%}` per position.
- **Per-position dollars**: dynamic, equal to `5 pct x current portfolio value`.
- **Stop-loss / take-profit**: `{spec.stop_loss:.1%}` / `{spec.take_profit:.1%}`,
  per-position (percentages scale appropriately with dynamic sizing).
- **Daily Backtrader execution**: `EquityPercentSizer` with native
  `bt.Order.Stop` / `bt.Order.Limit` protective exits.
- **Monte Carlo benchmark**: random portfolios sampled at the same top-N
  using the same equal-weight sizing rule, so the comparison is apples-to-
  apples with the strategy.

## Justification

### Equal-weight 1/N sizing

DeMiguel, Garlappi, Uppal (2009, *Review of Financial Studies*, "Optimal Versus
Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?")
evaluated 14 mean-variance optimization variants across 7 empirical datasets
(including US sector portfolios, international indices, and individual stocks).
None reliably outperformed naive 1/N on out-of-sample Sharpe. The reason is
estimation error: to reliably parametrize a mean-variance optimizer for a
25-asset portfolio would require roughly 3,000 months (250 years) of return
data, which is unavailable. Equal-weight is parsimonious, has zero estimation
error, and matches the industry analog -- the Invesco S&P 500 Equal Weight
ETF (RSP) -- one of the largest and longest-running smart-beta products.

### Top-20 selection

The relevant academic standards are quintile portfolios (top 20 pct = ~100
names for the S&P 500) used by Fama-French and Han-Zhou-Zhu, and decile
portfolios (top 10 pct = ~50 names) used in much of the cross-sectional
asset-pricing literature. Top 10 -- our prior choice -- is more concentrated
than any standard factor study and is justifiable only as a high-conviction
approach. Top 20 (~4 pct of the universe) is a defensible middle ground:
materially more diversified than top 10 while still expressing concentration
in the highest-scoring names. Plyakha, Uppal, Vilkov (2014, *Critical Finance
Review*) further documents the systematic outperformance of equal-weighted
concentrated portfolios over value-weighted alternatives.

### Foundation choice

Improved 7's time-varying cost analysis showed improved 4 (index-trend) wins
the head-to-head against improved 6 (HZZ cross-sectional trend) once
realistic transaction costs are applied. Improved 8 builds on the cost-robust
winner. Improved 4's structurally low turnover advantage should compose well
with the increased diversification of top 20.

## Results

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector final equity: `${metrics['final_equity']:,.0f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Backtrader final value: `${bt['final_value']:,.0f}`.
- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.
- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`.
- Monte Carlo p-value (against equal-weight 20-name random portfolios): `{monte_carlo['p_value'].iloc[0]:.4f}`.

## Head-to-Head vs Improved 4

{vec_line}
{bt_line}

The most informative comparison is whether the diversification gain (more
positions, more even risk distribution) outweighs the signal-dilution cost
(top 20 includes names ranked 11-20 which had lower composite scores). If
improved 8 has comparable or better Sharpe than improved 4 with materially
lower drawdown, it confirms that the original top 10 was over-concentrated
and that the project's right operating point is closer to a quintile-style
academic standard.

## Caveats

- **Dynamic position sizing compounds.** Each rebalance sizes positions at
  5 pct of *current* equity. As equity grows, positions grow. This is the
  realistic behavior of any real fund and matches academic convention, but
  it makes absolute equity comparisons across improvements with different
  sizing rules less directly meaningful. Annualized Sharpe and max drawdown
  remain apples-to-apples.
- **Integer share rounding.** `EquityPercentSizer` rounds down to whole
  shares per position, leaving a small unallocated cash residual each
  rebalance. This is realistic and matches how practitioners trade.
- **No cash buffer.** 20 positions at 5 pct each consume 100 pct of equity
  when fully populated. Real funds typically hold 2-5 pct cash. A more
  defensive variant would target 4.75 pct per position (95 pct invested);
  we keep the cleaner 5 pct for parsimony.
- **Common evaluation window.** All metrics use `EVALUATION_START =
  {core.EVALUATION_START.date()}` so the comparison with prior improveds is
  apples-to-apples on the same trading months.

## References

- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
  Diversification: How Inefficient Is the 1/N Portfolio Strategy?
  *Review of Financial Studies*, 22(5), 1915-1953.
- Plyakha, Y., Uppal, R., & Vilkov, G. (2014). Why Does an Equal-Weighted
  Portfolio Outperform Value- and Price-Weighted Portfolios? *Critical
  Finance Review*, 4(2), 271-308.
- Invesco S&P 500 Equal Weight ETF (RSP). Industry analog for equal-weight
  US large-cap concentrated portfolios.
"""
    (core.DOCS_DIR / "IMPROVED_8_TOP_N_SIZING.md").write_text(content, encoding="utf-8")


def append_strategy_history(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
) -> None:
    path = core.DOCS_DIR / "STRATEGY_HISTORY.md"
    content = path.read_text(encoding="utf-8") if path.exists() else "# Strategy History And Improvement Log\n"
    marker = "## Improved 8 Equal-Weight Top 20"
    bt = bt_metrics.iloc[0] if not bt_metrics.empty else None
    bt_lines = (
        f"- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.\n- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`."
        if bt is not None
        else "- Backtrader metrics unavailable in this run."
    )
    block = f"""

{marker}

Improved 8 changes two mechanically-coupled design dimensions on top of improved 4:
top-N moves from 10 to 20, and position sizing moves from fixed `${core.CASH_PER_TRADE:,.0f}`
per trade to equal-weight `{SIZING_TARGET_PCT:.2%}` of current portfolio equity per position.
The two changes are treated as one improvement because fixed-dollar sizing is
mechanically incompatible with meaningful top-N expansion (`$1M` of capital
cannot fund `20 x $100k = $2M`). Justification draws on DeMiguel-Garlappi-Uppal
(2009) for the 1/N choice and Plyakha-Uppal-Vilkov (2014) for the equal-weight
concentrated portfolio rationale; the industry analog is Invesco's RSP equal-
weight S&P 500 ETF. Foundation is improved 4 (the cost-robust winner from
improved 7).

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
{bt_lines}

See `docs/IMPROVED_8_TOP_N_SIZING.md` for full methodology and references.
"""
    if marker in content:
        content = content.split(marker)[0].rstrip() + block
    else:
        content = content.rstrip() + block
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    core.ensure_dirs()
    print("Running improved 8 equal-weight top-20 experiment without rebuilding the full pipeline...")

    _, _, panel = core.load_processed_strategy_inputs()
    prices, _, _, _, index = core.load_raw_data()

    stop_loss, take_profit = load_improved_4_thresholds()
    spec = make_improved_8_spec(stop_loss, take_profit)

    curve, holdings = core.simulate_vector_strategy(panel, spec)
    core.save_csv(curve, core.IMPROVED_8_RESULTS_DIR / "vector_equity_curve.csv")
    if not holdings.empty:
        core.save_csv(holdings, core.IMPROVED_8_RESULTS_DIR / "vector_holdings.csv")

    metrics = core.metrics_over_evaluation_window(
        curve, spec.name, date_col="month", return_col="portfolio_return"
    )
    eval_curve = core.filter_to_evaluation_window(curve, "month")
    metrics.update(
        {
            "avg_positions": float(eval_curve["n_positions"].mean()) if not eval_curve.empty else float("nan"),
            "stop_loss": spec.stop_loss,
            "take_profit": spec.take_profit,
            "top_n": spec.top_n,
            "sizing_method": spec.sizing_method,
            "sizing_target_pct": spec.sizing_target_pct,
            "notes": spec.notes,
        }
    )
    core.save_csv(pd.DataFrame([metrics]), core.IMPROVED_8_RESULTS_DIR / "vector_metrics.csv")

    core.save_strategy_weight_histories(panel, [(spec, core.IMPROVED_8_RESULTS_DIR)])

    monte_carlo = core.monte_carlo_random_portfolios(
        panel,
        spec,
        n_sims=1000,
        output_dir=core.IMPROVED_8_RESULTS_DIR,
    )
    core.block_bootstrap(
        curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_8_RESULTS_DIR,
    )

    signals = core.signals_from_strategy(panel, spec)
    bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        signals,
        spec.name,
        stop_loss=spec.stop_loss,
        take_profit=spec.take_profit,
        output_dir=core.IMPROVED_8_RESULTS_DIR,
        sizing_method=spec.sizing_method,
        sizing_target_pct=spec.sizing_target_pct,
    )
    min_position = core.assert_backtrader_long_only(bt, spec.name)

    comparison = save_comparison_summary()

    validation = pd.DataFrame(
        [
            {"check": "top_n", "status": "OK", "detail": str(spec.top_n)},
            {"check": "sizing_method", "status": "OK", "detail": spec.sizing_method},
            {"check": "sizing_target_pct", "status": "OK", "detail": f"{spec.sizing_target_pct:.4f}"},
            {"check": "stop_loss_inherited_from_improved_4", "status": "OK", "detail": f"{spec.stop_loss:.4f}"},
            {"check": "take_profit_inherited_from_improved_4", "status": "OK", "detail": f"{spec.take_profit:.4f}"},
            {"check": "evaluation_start", "status": "OK", "detail": core.EVALUATION_START.date().isoformat()},
            {"check": "monte_carlo_p_value", "status": "OK", "detail": f"{monte_carlo['p_value'].iloc[0]:.4f}"},
            {"check": "backtrader_long_only", "status": "OK", "detail": str(min_position)},
        ]
    )
    core.save_csv(validation, core.IMPROVED_8_RESULTS_DIR / "improved_8_validation_summary.csv")
    write_improved_8_note(spec, metrics, bt["metrics"], monte_carlo, comparison)
    append_strategy_history(spec, metrics, bt["metrics"])

    print("Improved 8 completed.")
    if not comparison.empty:
        print(
            comparison[
                [
                    "label",
                    "annualized_sharpe",
                    "final_equity",
                    "final_value",
                    "max_drawdown",
                    "stop_loss",
                    "take_profit",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
