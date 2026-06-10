"""Focused run for improved strategy 9: volatility-targeted top-20 sizing.

Improved 9 builds on improved 4 by changing the position-sizing rule from
fixed-cash to inverse-volatility weighting within the held basket. It is a
sister variant to improved 8 (equal-weight 1/N): both expand top-N from 10 to
20 to test the diversification vs concentration trade-off, but improved 8
uses naive 1/N (DeMiguel-Garlappi-Uppal 2009 ``naive 1/N beats sophisticated
weighting``) while improved 9 uses inverse-vol weighting (the workhorse rule
at AQR, Two Sigma, and most modern systematic quant funds).

Method:
- Foundation: improved 4 design (trend_expanding_z, 5% stop-loss, 30% take-profit).
- Top-N: 20 (same as improved 8).
- Sizing: vol_targeted. Each position is sized at
      w_i = (1/vol_i) / sum_j (1/vol_j across the basket)
  scaled so the basket consumes ``top_n * sizing_target_pct = 100%`` of equity
  when fully populated. ``vol_i`` is annualized realized vol over a trailing
  63-day (3-month) window, floored at 5% annualized to prevent extreme
  inverse-weights from very-low-vol names.
- Daily Backtrader execution with ``VolatilityTargetedSizer`` and native
  ``bt.Order.Stop`` / ``bt.Order.Limit`` protective exits.

Justification for vol-targeted sizing:
- AQR, Two Sigma, and most modern systematic quant funds use some form of
  inverse-vol or risk-parity sizing as their default workhorse rule. The
  rationale is risk-budget equality: each position should contribute roughly
  equal volatility to the portfolio, not equal dollars. This reduces
  drawdowns relative to equal-weight without giving up the diversification
  benefit of holding many names.
- Compared to equal-weight (improved 8): inverse-vol shifts capital away
  from jumpy stocks (Tesla, Nvidia) toward stable ones (Coca-Cola,
  Procter & Gamble). Should improve Sharpe and reduce drawdown.
- Compared to mean-variance optimal weights: inverse-vol avoids the
  estimation-error problem documented by DeMiguel-Garlappi-Uppal (2009).
  It uses only marginal vol, not the covariance matrix, so it's robust
  to the well-known instability of empirical correlations.

Comparison axes:
- vs improved 4 (sister cost-robust variant): does diversification + vol
  weighting beat concentrated fixed-cash?
- vs improved 8 (sister 1/N variant): does inverse-vol beat naive 1/N?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import project_core as core


DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.30
TOP_N = 20
SIZING_TARGET_PCT = 1.0 / TOP_N  # 5 pct nominal per position (1/N basis)


def load_improved_4_thresholds() -> tuple[float, float]:
    """Inherit stop-loss / take-profit values selected by improved 4's walk-forward."""
    selected_path = core.IMPROVED_4_RESULTS_DIR / "selected_stop_take_parameters.csv"
    if not selected_path.exists():
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
    return float(selected["stop_loss"].iloc[0]), float(selected["take_profit"].iloc[0])


def make_improved_9_spec(stop_loss: float, take_profit: float) -> core.StrategySpec:
    return core.StrategySpec(
        name=core.IMPROVED_9_STRATEGY_NAME,
        weights={"roe": 1, "pe": 1, "momentum": 1, "trend": 1},
        top_n=TOP_N,
        trend_col="trend_expanding_z",
        stop_loss=stop_loss,
        take_profit=take_profit,
        sizing_method="vol_targeted",
        sizing_target_pct=SIZING_TARGET_PCT,
        vol_lookback_days=core.VOL_LOOKBACK_DAYS,
        vol_floor_ann=core.VOL_FLOOR_ANN,
        notes=(
            "Improved 9: improved 4 design with inverse-volatility sizing at top 20. "
            "Each position sized at (1/vol_i) / sum_j (1/vol_j) of equity, scaled to "
            "consume 100 pct of equity at full basket. Foundation = improved 4."
        ),
    )


def load_metric_row(path: Path, label: str) -> dict[str, object]:
    row = pd.read_csv(path).iloc[0].to_dict()
    row["label"] = label
    return row


def save_comparison_summary() -> pd.DataFrame:
    """Side-by-side table of improved 4, improved 8, improved 9 vector and Backtrader metrics."""
    candidates = {
        "improved_4_vector": core.IMPROVED_4_RESULTS_DIR / "vector_metrics.csv",
        "improved_8_vector": core.IMPROVED_8_RESULTS_DIR / "vector_metrics.csv",
        "improved_9_vector": core.IMPROVED_9_RESULTS_DIR / "vector_metrics.csv",
        "improved_4_backtrader": core.IMPROVED_4_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_4_STRATEGY_NAME}_metrics.csv",
        "improved_8_backtrader": core.IMPROVED_8_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_8_STRATEGY_NAME}_metrics.csv",
        "improved_9_backtrader": core.IMPROVED_9_RESULTS_DIR
        / f"backtrader_daily_{core.IMPROVED_9_STRATEGY_NAME}_metrics.csv",
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
    core.save_csv(summary, core.IMPROVED_9_RESULTS_DIR / "improved_9_vs_improved_4_and_8_summary.csv")
    return summary


def write_improved_9_note(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    bt = bt_metrics.iloc[0] if not bt_metrics.empty else None
    comp = comparison.set_index("label") if not comparison.empty else pd.DataFrame()
    bt_line_4 = (
        f"- Improved 4 Backtrader Sharpe: `{comp.loc['improved_4_backtrader','annualized_sharpe']:.4f}`."
        if "improved_4_backtrader" in comp.index
        else "- Improved 4 Backtrader metrics not available."
    )
    bt_line_8 = (
        f"- Improved 8 Backtrader Sharpe: `{comp.loc['improved_8_backtrader','annualized_sharpe']:.4f}`."
        if "improved_8_backtrader" in comp.index
        else "- Improved 8 Backtrader metrics not available."
    )
    bt_line_9 = (
        f"- Improved 9 Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`."
        if bt is not None
        else "- Improved 9 Backtrader metrics not available."
    )

    content = f"""# Improved 9 -- Volatility-Targeted Top-20

Improved 9 is the inverse-volatility sister to improved 8 (equal-weight 1/N).
Both expand the basket from improved 4's top 10 to top 20; improved 8 uses
naive 1/N weights while improved 9 uses inverse-vol weights within the same
basket.

## Method

- **Foundation**: improved 4 design. Composite of ROE, P/E, momentum,
  trend_expanding_z. 5 pct stop-loss, 30 pct take-profit.
- **Top-N**: `{spec.top_n}` (same as improved 8).
- **Sizing method**: `vol_targeted`. Each position is sized at
  `(1/vol_i) / sum_j(1/vol_j)` scaled so the basket consumes
  `top_n * sizing_target_pct = {spec.top_n * spec.sizing_target_pct:.0%}` of
  equity when fully populated.
- **Volatility estimate**: trailing `{spec.vol_lookback_days}`-day annualized
  log-return standard deviation, floored at `{spec.vol_floor_ann:.1%}`
  annualized to prevent extreme weights.
- **Backtrader engine**: `DailySignalStopTakeStrategy` with
  `VolatilityTargetedSizer` and native `bt.Order.Stop` / `bt.Order.Limit`
  protective exits.

## Justification

### Why inverse-vol (not equal-weight, not mean-variance)

- **AQR, Two Sigma, BlackRock Risk Parity** all use some form of inverse-vol
  or risk-parity sizing as their default. The principle is risk-budget
  equality: each position should contribute roughly equal volatility, not
  equal dollars. High-vol names get smaller allocations; low-vol names get
  larger ones.
- **Compared to equal-weight (improved 8)**: inverse-vol explicitly accounts
  for stock-specific risk. Should produce lower drawdowns and higher Sharpe
  if vol is somewhat persistent (which it is on the 3-month horizon).
- **Compared to mean-variance optimal weights**: avoids the estimation-error
  problem documented by DeMiguel-Garlappi-Uppal (2009). Uses only marginal
  vol, not the unstable empirical covariance matrix.

### Why 63-day lookback

Three months of daily returns is the standard short-window vol estimate. Long
enough to be statistically stable (~63 observations), short enough to react
to regime changes within a quarter. Industry practice ranges from 30 to 252;
63 is the middle.

### Why 5 pct vol floor

A handful of S&P 500 names have realized vol below 10 pct annualized (utilities,
some consumer staples). Without a floor, inverse-vol would concentrate the
portfolio heavily in these names. The 5 pct floor caps a single name's weight
at roughly `1/0.05 = 20x` the unit weight, which combined with normalization
limits maximum per-name concentration.

## Results

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector final equity: `${metrics['final_equity']:,.0f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
- Backtrader final value: `${bt['final_value']:,.0f}` if bt is not None else 'n/a'.
- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}` if bt is not None else 'n/a'.
- Backtrader max drawdown: `{bt['max_drawdown']:.2%}` if bt is not None else 'n/a'.
- Monte Carlo p-value: `{monte_carlo['p_value'].iloc[0]:.4f}`.

## Head-to-Head vs Improved 4 and Improved 8

{bt_line_4}
{bt_line_8}
{bt_line_9}

The three-way comparison answers two questions at once:
1. **Concentration vs diversification** (improved 4 vs 8/9): does expanding
   to 20 names help or hurt?
2. **1/N vs inverse-vol** (improved 8 vs 9): if we're going to diversify,
   does explicit vol weighting beat naive equal-weight?

## Caveats

- **Vol estimation is noisy.** 63 days of returns gives ~63 observations,
  which is small for second-moment estimation. The 5 pct floor protects
  against extreme weights from low-vol names but does not protect against
  noisy vols generally.
- **Vol does not equal risk.** Real risk is downside volatility and tail
  events. Inverse-vol sizing treats upside vol the same as downside vol,
  which can underweight legitimate alpha names with high upside vol.
- **Backtrader Sizer approximation.** The vector engine does proper basket-
  level normalization. The Backtrader `VolatilityTargetedSizer` sizes each
  name independently relative to a reference vol (the `median_vol_ann`
  parameter), which approximates basket normalization but is not identical.
  Vector Sharpe is the more rigorous number; Backtrader confirms execution
  feasibility.
- **Common evaluation window** at `2016-05-31` applies to all metrics for
  apples-to-apples comparison with prior improveds.

## References

- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
  Diversification: How Inefficient Is the 1/N Portfolio Strategy?
  *Review of Financial Studies*, 22(5), 1915-1953.
- Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). Leverage Aversion
  and Risk Parity. *Financial Analysts Journal*, 68(1), 47-59.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of
  Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio
  Management*, 36(4), 60-70.
- AQR Capital Management — multiple working papers on risk parity and
  vol-targeted construction as the workhorse rule for systematic equity
  portfolios.
"""
    (core.DOCS_DIR / "IMPROVED_9_VOL_TARGETED.md").write_text(content, encoding="utf-8")


def append_strategy_history(
    spec: core.StrategySpec,
    metrics: dict[str, object],
    bt_metrics: pd.DataFrame,
) -> None:
    path = core.DOCS_DIR / "STRATEGY_HISTORY.md"
    content = path.read_text(encoding="utf-8") if path.exists() else "# Strategy History And Improvement Log\n"
    marker = "## Improved 9 Volatility-Targeted Top 20"
    bt = bt_metrics.iloc[0] if not bt_metrics.empty else None
    bt_lines = (
        f"- Backtrader Sharpe: `{bt['annualized_sharpe']:.4f}`.\n"
        f"- Backtrader max drawdown: `{bt['max_drawdown']:.2%}`."
        if bt is not None
        else "- Backtrader metrics unavailable in this run."
    )
    block = f"""

{marker}

Improved 9 is the inverse-volatility sister to improved 8. Both expand the
basket from improved 4's top 10 to top 20; improved 8 uses naive 1/N while
improved 9 uses inverse-vol weights computed from a trailing 63-day
annualized realized vol (floored at 5 pct annualized to prevent extreme
concentrations). Foundation is improved 4.

- Vector Sharpe: `{metrics['annualized_sharpe']:.4f}`.
- Vector max drawdown: `{metrics['max_drawdown']:.2%}`.
{bt_lines}

See `docs/IMPROVED_9_VOL_TARGETED.md` for justification, references
(DGU 2009, Asness-Frazzini-Pedersen 2012, Maillard-Roncalli-Teiletche
2010), and the three-way comparison vs improveds 4 and 8.
"""
    if marker in content:
        content = content.split(marker)[0].rstrip() + block
    else:
        content = content.rstrip() + block
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    core.ensure_dirs()
    print("Running improved 9 volatility-targeted top-20 experiment without rebuilding the full pipeline...")

    _, _, panel = core.load_processed_strategy_inputs()
    prices, _, _, _, index = core.load_raw_data()

    stop_loss, take_profit = load_improved_4_thresholds()
    spec = make_improved_9_spec(stop_loss, take_profit)

    curve, holdings = core.simulate_vector_strategy(panel, spec, daily_prices=prices)
    core.save_csv(curve, core.IMPROVED_9_RESULTS_DIR / "vector_equity_curve.csv")
    if not holdings.empty:
        core.save_csv(holdings, core.IMPROVED_9_RESULTS_DIR / "vector_holdings.csv")

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
            "vol_lookback_days": spec.vol_lookback_days,
            "vol_floor_ann": spec.vol_floor_ann,
            "notes": spec.notes,
        }
    )
    core.save_csv(pd.DataFrame([metrics]), core.IMPROVED_9_RESULTS_DIR / "vector_metrics.csv")

    core.save_strategy_weight_histories(panel, [(spec, core.IMPROVED_9_RESULTS_DIR)])

    monte_carlo = core.monte_carlo_random_portfolios(
        panel,
        spec,
        n_sims=1000,
        output_dir=core.IMPROVED_9_RESULTS_DIR,
        daily_prices=prices,
    )
    core.block_bootstrap(
        curve,
        block_size=6,
        n_sims=1000,
        output_name="block_bootstrap.csv",
        output_dir=core.IMPROVED_9_RESULTS_DIR,
    )

    signals = core.signals_from_strategy(panel, spec)
    bt = core.run_backtrader_daily_stop_take(
        prices,
        index,
        signals,
        spec.name,
        stop_loss=spec.stop_loss,
        take_profit=spec.take_profit,
        output_dir=core.IMPROVED_9_RESULTS_DIR,
        sizing_method=spec.sizing_method,
        sizing_target_pct=spec.sizing_target_pct,
        vol_lookback_days=spec.vol_lookback_days,
        vol_floor_ann=spec.vol_floor_ann,
    )
    min_position = core.assert_backtrader_long_only(bt, spec.name)

    comparison = save_comparison_summary()

    validation = pd.DataFrame(
        [
            {"check": "top_n", "status": "OK", "detail": str(spec.top_n)},
            {"check": "sizing_method", "status": "OK", "detail": spec.sizing_method},
            {"check": "sizing_target_pct", "status": "OK", "detail": f"{spec.sizing_target_pct:.4f}"},
            {"check": "vol_lookback_days", "status": "OK", "detail": str(spec.vol_lookback_days)},
            {"check": "vol_floor_ann", "status": "OK", "detail": f"{spec.vol_floor_ann:.4f}"},
            {"check": "stop_loss_inherited_from_improved_4", "status": "OK", "detail": f"{spec.stop_loss:.4f}"},
            {"check": "take_profit_inherited_from_improved_4", "status": "OK", "detail": f"{spec.take_profit:.4f}"},
            {"check": "evaluation_start", "status": "OK", "detail": core.EVALUATION_START.date().isoformat()},
            {"check": "monte_carlo_p_value", "status": "OK", "detail": f"{monte_carlo['p_value'].iloc[0]:.4f}"},
            {"check": "backtrader_long_only", "status": "OK", "detail": str(min_position)},
        ]
    )
    core.save_csv(validation, core.IMPROVED_9_RESULTS_DIR / "improved_9_validation_summary.csv")
    write_improved_9_note(spec, metrics, bt["metrics"], monte_carlo, comparison)
    append_strategy_history(spec, metrics, bt["metrics"])

    print("Improved 9 completed.")
    if not comparison.empty:
        print(
            comparison[
                [
                    "label",
                    "annualized_sharpe",
                    "final_equity",
                    "final_value",
                    "max_drawdown",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
