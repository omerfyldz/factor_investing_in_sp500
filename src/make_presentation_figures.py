"""Presentation figures builder — all 9 strategies.

Generates 15+ publication-grade figures and several summary tables that were
missing from the original ``make_figures`` output (which only covered the
staged ladder base + improved 1-3).

Run AFTER:
1. ``run_project.py``
2. ``run_improved_4_stop_take_sensitivity.py``
3. ``run_improved_5_regime_filter.py``
4. ``run_improved_6_hzz_trend.py``
5. ``run_improved_7_costs.py``
6. ``run_improved_8_top_n_sizing.py``
7. ``run_improved_9_vol_targeted.py``
8. ``aggregate_all_strategies.py``  (produces all_strategies_*.csv)

Figures saved to ``figures/``.  Tables saved to ``results/comparison/``.
Runtime: ~2-3 min.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

import project_core as core

# ---------------------------------------------------------------------------
# Colour palette — one colour per strategy, consistent across all charts
# ---------------------------------------------------------------------------
STRATEGY_ORDER = [
    "base",
    "improved_1",
    "improved_2",
    "improved_3",
    "improved_4",
    "improved_5",
    "improved_6",
    "improved_8",
    "improved_9",
]

STRATEGY_LABELS = {
    "base": "Base (full-sample trend)",
    "improved_1": "Imp 1 (expanding trend)",
    "improved_2": "Imp 2 (+stops 10/20%)",
    "improved_3": "Imp 3 (+dyn IC weights)",
    "improved_4": "Imp 4 (walk-fwd 5/30%)",
    "improved_5": "Imp 5 (regime filter) ✗",
    "improved_6": "Imp 6 (HZZ trend)",
    "improved_8": "Imp 8 (eq-wt top-20)",
    "improved_9": "Imp 9 (vol-tgt top-20)",
}

PALETTE = sns.color_palette("tab10", n_colors=len(STRATEGY_ORDER))
COLOR_MAP = dict(zip(STRATEGY_ORDER, PALETTE))

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150})


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all_curves() -> pd.DataFrame:
    """Load all 9 strategy equity curves into one long DataFrame."""
    folder_map = {
        "base": core.BASE_RESULTS_DIR,
        "improved_1": core.IMPROVED_RESULTS_DIR,
        "improved_2": core.IMPROVED_2_RESULTS_DIR,
        "improved_3": core.IMPROVED_3_RESULTS_DIR,
        "improved_4": core.IMPROVED_4_RESULTS_DIR,
        "improved_5": core.IMPROVED_5_RESULTS_DIR,
        "improved_6": core.IMPROVED_6_RESULTS_DIR,
        "improved_8": core.IMPROVED_8_RESULTS_DIR,
        "improved_9": core.IMPROVED_9_RESULTS_DIR,
    }
    parts = []
    for label, folder in folder_map.items():
        path = folder / "vector_equity_curve.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found for {label} — skipping")
            continue
        df = pd.read_csv(path, parse_dates=["month"])
        df["strategy"] = label
        parts.append(df)
    if not parts:
        raise RuntimeError("No strategy curves found. Run focused scripts first.")
    return pd.concat(parts, ignore_index=True).sort_values(["strategy", "month"])


def load_index_monthly() -> pd.DataFrame:
    _, idx, _ = core.load_processed_strategy_inputs()
    return idx


def load_all_metrics() -> pd.DataFrame:
    p = core.COMPARISON_RESULTS_DIR / "all_strategies_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    print("  WARNING: all_strategies_metrics.csv not found — run aggregate_all_strategies.py first")
    return pd.DataFrame()


def load_walk_forward() -> pd.DataFrame:
    p = core.COMPARISON_RESULTS_DIR / "all_strategies_walk_forward.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_monte_carlo_agg() -> pd.DataFrame:
    p = core.COMPARISON_RESULTS_DIR / "all_strategies_monte_carlo.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_improved7_cost_curves() -> pd.DataFrame:
    p = core.IMPROVED_7_RESULTS_DIR / "cost_sensitivity_vector_curves.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_improved3_weights() -> pd.DataFrame:
    p = core.IMPROVED_3_RESULTS_DIR / "strategy_weight_history.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper: build a rebased equity series starting at INITIAL_CASH on eval start
# ---------------------------------------------------------------------------

def rebase_eval(curve_g: pd.DataFrame, return_col: str = "portfolio_return") -> pd.DataFrame:
    """Return eval-window rows with equity rebased to INITIAL_CASH at eval start."""
    g = core.filter_to_evaluation_window(curve_g, "month").sort_values("month")
    if g.empty:
        return g
    r = g[return_col].astype(float).fillna(0.0)
    g = g.copy()
    g["equity_rebased"] = core.INITIAL_CASH * (1 + r).cumprod()
    return g


# ---------------------------------------------------------------------------
# Figure 1 — All strategies equity curves on one chart
# ---------------------------------------------------------------------------

def fig_all_equity_curves(all_curves: pd.DataFrame, index_monthly: pd.DataFrame) -> None:
    print("  Fig 1: all-strategies equity curves")
    fig, ax = plt.subplots(figsize=(13, 7))

    # Benchmark ^GSPC rebased
    idx_eval = core.filter_to_evaluation_window(index_monthly, "month").sort_values("month")
    if not idx_eval.empty:
        br = idx_eval["ret_1m"].astype(float).fillna(0.0)
        bm_equity = core.INITIAL_CASH * (1 + br).cumprod()
        ax.plot(idx_eval["month"], bm_equity / 1e6, color="black", linewidth=2,
                linestyle="--", label="^GSPC (benchmark)", zorder=5)

    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        if g.empty:
            continue
        g = rebase_eval(g)
        if g.empty:
            continue
        ax.plot(g["month"], g["equity_rebased"] / 1e6,
                color=COLOR_MAP[label], linewidth=1.6,
                label=STRATEGY_LABELS.get(label, label))

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax.set_title("All 9 Strategies — Equity Curves (eval window, rebased to $1M)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Portfolio Value")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "all_strategies_equity_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — All strategies drawdown curves
# ---------------------------------------------------------------------------

def fig_all_drawdowns(all_curves: pd.DataFrame) -> None:
    print("  Fig 2: all-strategies drawdown curves")
    fig, ax = plt.subplots(figsize=(13, 6))

    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        if g.empty:
            continue
        g = rebase_eval(g)
        if g.empty:
            continue
        dd = core.make_drawdown(g["equity_rebased"])
        ax.plot(g["month"], dd * 100, color=COLOR_MAP[label], linewidth=1.4,
                label=STRATEGY_LABELS.get(label, label))

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("All 9 Strategies — Drawdown from Peak (eval window)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "all_strategies_drawdowns.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Sharpe vs Max-Drawdown scatter, bubble = final equity
# ---------------------------------------------------------------------------

def fig_sharpe_vs_drawdown(metrics: pd.DataFrame) -> None:
    print("  Fig 3: Sharpe vs max-drawdown scatter")
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in metrics.iterrows():
        label = row.get("strategy", "")
        sharpe = row.get("annualized_sharpe", np.nan)
        dd = row.get("max_drawdown", np.nan)
        eq = row.get("final_equity", 1e6)
        if np.isnan(sharpe) or np.isnan(dd):
            continue
        color = COLOR_MAP.get(label, "grey")
        size = max(50, min(600, (eq / 1e6) * 60))
        ax.scatter(abs(dd) * 100, sharpe, s=size, color=color,
                   alpha=0.8, edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(STRATEGY_LABELS.get(label, label), (abs(dd) * 100, sharpe),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax.set_xlabel("Max Drawdown (absolute, %)", fontsize=11)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
    ax.set_title("Risk-Return Frontier — Sharpe vs Max Drawdown\n(bubble size ∝ final equity)", fontsize=12)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Sharpe = 1.0")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "sharpe_vs_drawdown_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Rolling 12-month Sharpe for top strategies
# ---------------------------------------------------------------------------

def fig_rolling_sharpe(all_curves: pd.DataFrame) -> None:
    print("  Fig 4: rolling 12-month Sharpe")
    top_labels = ["base", "improved_4", "improved_6", "improved_8", "improved_9"]
    fig, ax = plt.subplots(figsize=(13, 6))

    for label in top_labels:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if len(g) < 14:
            continue
        r = g["portfolio_return"].astype(float)
        roll_sharpe = r.rolling(12).apply(
            lambda x: np.sqrt(12) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
            raw=True,
        )
        ax.plot(g["month"], roll_sharpe, color=COLOR_MAP.get(label, "grey"),
                linewidth=1.5, label=STRATEGY_LABELS.get(label, label))

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Rolling 12-Month Sharpe Ratio — Top Strategies", fontsize=13)
    ax.set_xlabel("Month (end of rolling window)")
    ax.set_ylabel("Rolling Sharpe (annualized)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "rolling_12m_sharpe.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Calendar-year returns bar chart
# ---------------------------------------------------------------------------

def fig_annual_returns(all_curves: pd.DataFrame) -> pd.DataFrame:
    print("  Fig 5: annual returns bar chart + table")
    eval_curves = core.filter_to_evaluation_window(all_curves, "month")
    eval_curves = eval_curves.copy()
    eval_curves["year"] = pd.to_datetime(eval_curves["month"]).dt.year

    annual = (
        eval_curves.groupby(["strategy", "year"])["portfolio_return"]
        .apply(lambda r: (1 + r.astype(float)).prod() - 1)
        .reset_index()
        .rename(columns={"portfolio_return": "annual_return"})
    )
    # Save table
    pivot = annual.pivot(index="year", columns="strategy", values="annual_return")
    # reorder columns
    present = [s for s in STRATEGY_ORDER if s in pivot.columns]
    pivot = pivot[present]
    core.save_csv(pivot.reset_index().rename(columns={"index": "year"}),
                  core.COMPARISON_RESULTS_DIR / "annual_returns_table.csv")

    # Plot top strategies only to keep chart readable
    top = ["base", "improved_4", "improved_6", "improved_8", "improved_9"]
    top_present = [s for s in top if s in annual["strategy"].unique()]
    annual_top = annual[annual["strategy"].isin(top_present)]
    years = sorted(annual["year"].unique())

    x = np.arange(len(years))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, label in enumerate(top_present):
        g = annual_top[annual_top["strategy"] == label].set_index("year").reindex(years)
        ax.bar(x + i * width, g["annual_return"] * 100, width,
               color=COLOR_MAP.get(label, "grey"), alpha=0.85,
               label=STRATEGY_LABELS.get(label, label))

    ax.set_xticks(x + width * (len(top_present) - 1) / 2)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Calendar-Year Returns — Top Strategies (eval window)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Return (%)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "annual_returns_bar_chart.png", dpi=150)
    plt.close(fig)
    return pivot


# ---------------------------------------------------------------------------
# Figure 6 — Monthly returns calendar heatmap for improved 4 and improved 8
# ---------------------------------------------------------------------------

def fig_monthly_heatmap(all_curves: pd.DataFrame, strategy: str) -> None:
    print(f"  Fig 6: monthly heatmap for {strategy}")
    g = core.filter_to_evaluation_window(
        all_curves[all_curves["strategy"] == strategy], "month"
    ).sort_values("month")
    if g.empty:
        print(f"    No data for {strategy}")
        return
    g = g.copy()
    g["year"] = pd.to_datetime(g["month"]).dt.year
    g["month_num"] = pd.to_datetime(g["month"]).dt.month
    pivot = g.pivot(index="year", columns="month_num", values="portfolio_return") * 100
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Monthly Return (%)", "shrink": 0.6})
    ax.set_title(f"Monthly Returns Heatmap — {STRATEGY_LABELS.get(strategy, strategy)}", fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    fname = f"monthly_returns_heatmap_{strategy}.png"
    fig.savefig(core.FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Average position count over time
# ---------------------------------------------------------------------------

def fig_avg_position_count(all_curves: pd.DataFrame) -> None:
    print("  Fig 7: average position count over time")
    if "n_positions" not in all_curves.columns:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if g.empty or "n_positions" not in g.columns:
            continue
        roll = g["n_positions"].astype(float).rolling(6, min_periods=1).mean()
        ax.plot(g["month"], roll, color=COLOR_MAP.get(label, "grey"),
                linewidth=1.3, label=STRATEGY_LABELS.get(label, label))
    ax.set_title("Average Position Count Over Time (6-month rolling, eval window)", fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Holdings")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "avg_position_count_over_time.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — Walk-forward: train Sharpe vs test Sharpe scatter
# ---------------------------------------------------------------------------

def fig_walk_forward_scatter(wf: pd.DataFrame) -> None:
    print("  Fig 8: walk-forward train vs test scatter")
    if wf.empty:
        print("    No walk-forward data available")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in wf.iterrows():
        label = str(row.get("strategy", ""))
        ts = row.get("train_sharpe_to_2020", np.nan)
        test_s = row.get("test_sharpe_2021_2026", np.nan)
        if np.isnan(ts) or np.isnan(test_s):
            continue
        color = COLOR_MAP.get(label, "grey")
        ax.scatter(ts, test_s, s=120, color=color, edgecolors="white",
                   linewidths=0.8, zorder=3)
        ax.annotate(STRATEGY_LABELS.get(label, label), (ts, test_s),
                    textcoords="offset points", xytext=(5, 4), fontsize=7.5)

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]) - 0.05,
        max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="train = test")
    ax.set_xlabel("Train Sharpe (2016-05 → 2020-12)", fontsize=11)
    ax.set_ylabel("Test Sharpe (2021-01 → 2026-05)", fontsize=11)
    ax.set_title("Walk-Forward: In-Sample vs Out-of-Sample Sharpe\n(all strategies)", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "walk_forward_train_vs_test_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — Cumulative alpha vs ^GSPC for top strategies
# ---------------------------------------------------------------------------

def fig_cumulative_alpha(all_curves: pd.DataFrame, index_monthly: pd.DataFrame) -> None:
    print("  Fig 9: cumulative alpha vs ^GSPC")
    idx = core.filter_to_evaluation_window(index_monthly, "month").sort_values("month")
    if idx.empty:
        return
    bm_r = idx.set_index("month")["ret_1m"].astype(float).fillna(0.0)

    top_labels = ["base", "improved_4", "improved_6", "improved_8", "improved_9"]
    fig, ax = plt.subplots(figsize=(13, 6))
    for label in top_labels:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if g.empty:
            continue
        g = g.set_index("month")
        r = g["portfolio_return"].astype(float)
        aligned = r.subtract(bm_r, fill_value=0.0)
        cum_alpha = (1 + aligned).cumprod() - 1
        ax.plot(cum_alpha.index, cum_alpha * 100, color=COLOR_MAP.get(label, "grey"),
                linewidth=1.5, label=STRATEGY_LABELS.get(label, label))

    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Cumulative Excess Return vs ^GSPC (eval window)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Alpha (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "cumulative_alpha_vs_gspc.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10 — Position concentration HHI over time (improved 4 and improved 9)
# ---------------------------------------------------------------------------

def fig_hhi_concentration(all_curves: pd.DataFrame) -> None:
    """Use n_positions as a proxy: HHI(equal) = 1/n_positions."""
    print("  Fig 10: position concentration HHI")
    labels = ["improved_4", "improved_8", "improved_9"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for label in labels:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if g.empty or "n_positions" not in g.columns:
            continue
        n = g["n_positions"].astype(float).replace(0, np.nan)
        hhi = 1.0 / n  # equal-weight HHI = 1/N
        ax.plot(g["month"], hhi, color=COLOR_MAP.get(label, "grey"),
                linewidth=1.4, label=STRATEGY_LABELS.get(label, label))

    ax.set_title("Position Concentration — Herfindahl-Hirschman Index (equal-weight proxy)\n"
                 "Higher = more concentrated", fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("HHI (equal-weight = 1/N)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "position_concentration_hhi.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11 — Cost drag attribution for improved 7
# ---------------------------------------------------------------------------

def fig_cost_drag(cost_curves: pd.DataFrame) -> None:
    print("  Fig 11: cost drag attribution (improved 7)")
    if cost_curves.empty:
        print("    Improved 7 cost curves not found — skipping")
        return

    cost_curves = cost_curves.copy()
    cost_curves["year"] = pd.to_datetime(cost_curves["month"]).dt.year

    needed = ["strategy_label", "cost_scenario", "portfolio_return", "year"]
    if not all(c in cost_curves.columns for c in needed):
        print(f"    Missing columns in cost curves: {set(needed) - set(cost_curves.columns)}")
        return

    zero = cost_curves[cost_curves["cost_scenario"] == "zero"]
    central = cost_curves[cost_curves["cost_scenario"] == "central"]
    if zero.empty or central.empty:
        return

    strategies = cost_curves["strategy_label"].unique()
    fig, axes = plt.subplots(1, len(strategies), figsize=(7 * len(strategies), 5), sharey=False)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        z = zero[zero["strategy_label"] == strat].groupby("year")["portfolio_return"].apply(
            lambda r: (1 + r.astype(float)).prod() - 1
        )
        c = central[central["strategy_label"] == strat].groupby("year")["portfolio_return"].apply(
            lambda r: (1 + r.astype(float)).prod() - 1
        )
        drag = (z - c).dropna() * 100
        years = drag.index
        ax.bar(years, drag, color="steelblue", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Cost Drag by Year\n({strat})", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("Return drag (pct points)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.2f}pp"))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Improved 7 — Annual Return Cost Drag (zero vs central scenario)", fontsize=13)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "cost_drag_attribution_imp7.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12 — Factor weight evolution for improved 3
# ---------------------------------------------------------------------------

def fig_factor_weights(imp3_weights: pd.DataFrame) -> None:
    print("  Fig 12: factor weight evolution (improved 3)")
    if imp3_weights.empty:
        print("    Improved 3 weight history not found — skipping")
        return

    factor_cols = [c for c in ["roe", "pe", "momentum", "trend"] if c in imp3_weights.columns]
    if not factor_cols:
        return
    g = core.filter_to_evaluation_window(imp3_weights, "month").sort_values("month")
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    for fc in factor_cols:
        ax.plot(g["month"], g[fc], linewidth=1.5, label=fc.upper())
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Equal (25%)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0, 0.6)
    ax.set_title("Improved 3 — Dynamic Factor Weight Evolution (eval window)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Factor Weight")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "factor_weight_evolution_imp3.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13 — Improved 4 vs 8 vs 9 head-to-head equity curves
# ---------------------------------------------------------------------------

def fig_imp4_vs_8_vs_9(all_curves: pd.DataFrame) -> None:
    print("  Fig 13: improved 4 vs 8 vs 9 head-to-head")
    labels = ["improved_4", "improved_8", "improved_9"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label in labels:
        g = all_curves[all_curves["strategy"] == label]
        g = rebase_eval(g)
        if g.empty:
            continue
        color = COLOR_MAP.get(label, "grey")
        ax1.plot(g["month"], g["equity_rebased"] / 1e6, color=color, linewidth=2.0,
                 label=STRATEGY_LABELS.get(label, label))
        dd = core.make_drawdown(g["equity_rebased"])
        ax2.plot(g["month"], dd * 100, color=color, linewidth=2.0,
                 label=STRATEGY_LABELS.get(label, label))

    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax1.set_title("Equity Curves", fontsize=12)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=9)

    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Drawdown", fontsize=12)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)

    fig.suptitle("Sizing Comparison: Fixed ($100K) vs Equal-Weight (1/N) vs Vol-Targeted\n"
                 "Improved 4 vs Improved 8 vs Improved 9 (eval window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "imp4_vs_8_vs_9_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14 — Monte Carlo p-value comparison bar chart
# ---------------------------------------------------------------------------

def fig_mc_pvalues(mc: pd.DataFrame) -> None:
    print("  Fig 14: Monte Carlo p-value comparison")
    if mc.empty:
        print("    No MC p-value data available")
        return

    mc = mc.copy()
    present = [s for s in STRATEGY_ORDER if s in mc["strategy"].values]
    mc_ordered = mc.set_index("strategy").reindex(present).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [COLOR_MAP.get(s, "grey") for s in mc_ordered["strategy"]]
    bars = ax.bar(mc_ordered["strategy"], mc_ordered["p_value"], color=colors, alpha=0.85)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="p = 0.05 threshold")
    ax.axhline(0.10, color="orange", linestyle=":", linewidth=1.0, label="p = 0.10 threshold")
    ax.set_xticklabels(
        [STRATEGY_LABELS.get(s, s) for s in mc_ordered["strategy"]],
        rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("Monte Carlo p-value")
    ax.set_title("Monte Carlo p-value by Strategy\n(fraction of random portfolios ≥ strategy Sharpe)", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "monte_carlo_pvalue_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 15 — Return correlation heatmap
# ---------------------------------------------------------------------------

def fig_return_correlation(all_curves: pd.DataFrame) -> None:
    print("  Fig 15: cross-strategy return correlation")
    eval_curves = core.filter_to_evaluation_window(all_curves, "month")
    if eval_curves.empty:
        return

    wide = eval_curves.pivot_table(
        index="month", columns="strategy", values="portfolio_return", aggfunc="mean"
    )
    present = [s for s in STRATEGY_ORDER if s in wide.columns]
    wide = wide[present].dropna(how="all")
    if wide.shape[1] < 2:
        return
    corr = wide.corr()

    labels_map = {s: STRATEGY_LABELS.get(s, s) for s in corr.columns}
    corr_display = corr.rename(index=labels_map, columns=labels_map)

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)
    sns.heatmap(corr_display, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                mask=mask, ax=ax,
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.7})
    ax.set_title("Cross-Strategy Monthly Return Correlation (eval window)", fontsize=13)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "cross_strategy_correlation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def make_hit_rate_table(all_curves: pd.DataFrame) -> pd.DataFrame:
    print("  Table: hit rate (% positive months)")
    rows = []
    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month")
        if g.empty:
            continue
        r = g["portfolio_return"].astype(float).dropna()
        rows.append({
            "strategy": label,
            "strategy_label": STRATEGY_LABELS.get(label, label),
            "n_months": len(r),
            "hit_rate_pct": (r > 0).mean() * 100,
            "avg_up_month_pct": r[r > 0].mean() * 100 if (r > 0).any() else np.nan,
            "avg_down_month_pct": r[r < 0].mean() * 100 if (r < 0).any() else np.nan,
            "best_month_pct": r.max() * 100,
            "worst_month_pct": r.min() * 100,
        })
    df = pd.DataFrame(rows)
    core.save_csv(df, core.COMPARISON_RESULTS_DIR / "hit_rate_per_strategy.csv")
    return df


def make_tail_risk_table(all_curves: pd.DataFrame) -> pd.DataFrame:
    print("  Table: tail risk metrics (Sortino, Calmar)")
    rows = []
    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        g = rebase_eval(g)
        if g.empty:
            continue
        r = g["portfolio_return"].astype(float).dropna()
        if len(r) < 12:
            continue
        ann_return = r.mean() * 12
        downside = r[r < 0]
        sortino_denom = downside.std(ddof=1) * np.sqrt(12) if len(downside) > 1 else np.nan
        sortino = ann_return / sortino_denom if sortino_denom and np.isfinite(sortino_denom) else np.nan
        eq = g["equity_rebased"]
        dd = core.make_drawdown(eq)
        max_dd = abs(dd.min())
        calmar = ann_return / max_dd if max_dd > 0 else np.nan
        rows.append({
            "strategy": label,
            "strategy_label": STRATEGY_LABELS.get(label, label),
            "annualized_return_approx": ann_return,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": dd.min(),
        })
    df = pd.DataFrame(rows)
    core.save_csv(df, core.COMPARISON_RESULTS_DIR / "tail_risk_metrics.csv")
    return df


def make_best_worst_months_table(all_curves: pd.DataFrame) -> pd.DataFrame:
    print("  Table: best/worst months per strategy")
    rows = []
    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if g.empty:
            continue
        r = g.set_index("month")["portfolio_return"].astype(float).dropna()
        for m, v in r.nlargest(5).items():
            rows.append({"strategy": label, "type": "best", "month": m, "return_pct": v * 100})
        for m, v in r.nsmallest(5).items():
            rows.append({"strategy": label, "type": "worst", "month": m, "return_pct": v * 100})
    df = pd.DataFrame(rows)
    core.save_csv(df, core.COMPARISON_RESULTS_DIR / "best_worst_months_per_strategy.csv")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    core.ensure_dirs()
    print("Building presentation figures and tables for all 9 strategies...")

    # Load data
    all_curves = load_all_curves()
    index_monthly = load_index_monthly()
    metrics = load_all_metrics()
    wf = load_walk_forward()
    mc = load_monte_carlo_agg()
    cost_curves = load_improved7_cost_curves()
    imp3_weights = load_improved3_weights()

    print(f"\nLoaded {all_curves['strategy'].nunique()} strategies, "
          f"{len(all_curves)} curve rows\n")

    # --- Figures ---
    fig_all_equity_curves(all_curves, index_monthly)
    fig_all_drawdowns(all_curves)
    fig_sharpe_vs_drawdown(metrics)
    fig_rolling_sharpe(all_curves)
    annual_table = fig_annual_returns(all_curves)
    fig_monthly_heatmap(all_curves, "improved_4")
    fig_monthly_heatmap(all_curves, "improved_8")
    if "improved_9" in all_curves["strategy"].unique():
        fig_monthly_heatmap(all_curves, "improved_9")
    fig_avg_position_count(all_curves)
    fig_walk_forward_scatter(wf)
    fig_cumulative_alpha(all_curves, index_monthly)
    fig_hhi_concentration(all_curves)
    fig_cost_drag(cost_curves)
    fig_factor_weights(imp3_weights)
    fig_imp4_vs_8_vs_9(all_curves)
    fig_mc_pvalues(mc)
    fig_return_correlation(all_curves)

    # --- Tables ---
    hit_rate = make_hit_rate_table(all_curves)
    tail_risk = make_tail_risk_table(all_curves)
    best_worst = make_best_worst_months_table(all_curves)

    # Print summary
    print("\n=== Hit Rate Table ===")
    if not hit_rate.empty:
        print(hit_rate[["strategy", "hit_rate_pct", "best_month_pct", "worst_month_pct"]].to_string(index=False))

    print("\n=== Tail Risk Table ===")
    if not tail_risk.empty:
        print(tail_risk[["strategy", "sortino_ratio", "calmar_ratio", "max_drawdown"]].to_string(index=False))

    print("\n\nPresentation figures complete.")
    figures = sorted(core.FIGURES_DIR.glob("*.png"))
    print(f"Total figures in figures/: {len(figures)}")
    for f in figures:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
