"""Presentation figures builder — all 8 strategies.

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
7. ``aggregate_all_strategies.py``  (produces all_strategies_*.csv)

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
}

PALETTE = sns.color_palette("tab10", n_colors=len(STRATEGY_ORDER))
COLOR_MAP = dict(zip(STRATEGY_ORDER, PALETTE))

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    # Use DejaVu (bundled with matplotlib) — has Unicode coverage (★ ✗ ∝ ≤ ≥ →)
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all_curves() -> pd.DataFrame:
    """Load all 8 strategy equity curves into one long DataFrame."""
    folder_map = {
        "base": core.BASE_RESULTS_DIR,
        "improved_1": core.IMPROVED_RESULTS_DIR,
        "improved_2": core.IMPROVED_2_RESULTS_DIR,
        "improved_3": core.IMPROVED_3_RESULTS_DIR,
        "improved_4": core.IMPROVED_4_RESULTS_DIR,
        "improved_5": core.IMPROVED_5_RESULTS_DIR,
        "improved_6": core.IMPROVED_6_RESULTS_DIR,
        "improved_8": core.IMPROVED_8_RESULTS_DIR,
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
    p = core.IMPROVED_3_RESULTS_DIR / "factor_weight_history.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_fmp_portfolio_returns() -> pd.DataFrame:
    p = core.FMP_RESULTS_DIR / "fmp_portfolio_returns.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_fmp_regression_returns() -> pd.DataFrame:
    p = core.FMP_RESULTS_DIR / "fmp_regression_returns.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_factor_ic() -> pd.DataFrame:
    p = core.FMP_RESULTS_DIR / "factor_information_coefficients.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_improved4_grid() -> pd.DataFrame:
    p = core.IMPROVED_4_RESULTS_DIR / "stop_take_sensitivity_grid.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_improved7_vector_curves() -> pd.DataFrame:
    p = core.IMPROVED_7_RESULTS_DIR / "vector_equity_curves.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_improved7_yearly_drag() -> pd.DataFrame:
    p = core.IMPROVED_7_RESULTS_DIR / "yearly_cost_drag.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_improved7_cost_schedule() -> pd.DataFrame:
    p = core.IMPROVED_7_RESULTS_DIR / "cost_schedule.csv"
    if p.exists():
        return pd.read_csv(p)
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
    ax.set_title("All 8 Strategies — Equity Curves (eval window, rebased to $1M)", fontsize=13)
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
    ax.set_title("All 8 Strategies — Drawdown from Peak (eval window)", fontsize=13)
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
    top_labels = ["base", "improved_4", "improved_6", "improved_8"]
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
    top = ["base", "improved_4", "improved_6", "improved_8"]
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

    top_labels = ["base", "improved_4", "improved_6", "improved_8"]
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
# Figure 10 — Position concentration HHI over time (improved 4 and improved 8)
# ---------------------------------------------------------------------------

def fig_hhi_concentration(all_curves: pd.DataFrame) -> None:
    """Use n_positions as a proxy: HHI(equal) = 1/n_positions."""
    print("  Fig 10: position concentration HHI")
    labels = ["improved_4", "improved_8"]
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

    # Columns in factor_weight_history.csv are prefixed weight_*; allow both naming styles.
    factor_cols = [c for c in ["weight_roe", "weight_pe", "weight_momentum", "weight_trend"]
                   if c in imp3_weights.columns]
    if not factor_cols:
        factor_cols = [c for c in ["roe", "pe", "momentum", "trend"] if c in imp3_weights.columns]
    if not factor_cols:
        return
    g = core.filter_to_evaluation_window(imp3_weights, "month").sort_values("month")
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    for fc in factor_cols:
        label = fc.replace("weight_", "").upper()
        ax.plot(g["month"], g[fc], linewidth=1.5, label=label)
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
# Figure 13 — Improved 4 vs 8 head-to-head equity curves
# ---------------------------------------------------------------------------

def fig_imp4_vs_8(all_curves: pd.DataFrame) -> None:
    print("  Fig 13: improved 4 vs 8 head-to-head")
    labels = ["improved_4", "improved_8"]
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

    fig.suptitle("Sizing Comparison: Fixed ($100K) vs Equal-Weight (1/N)\n"
                 "Improved 4 vs Improved 8 (eval window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "imp4_vs_8_comparison.png", dpi=150)
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
# Migrated figures — formerly produced by project_core / run_improved_4 / run_improved_7
# ---------------------------------------------------------------------------

def fig_fmp_portfolio_cumulative_returns(portfolio: pd.DataFrame) -> None:
    print("  Fig (migrated): FMP portfolio-sort cumulative returns")
    if portfolio.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for factor, g in portfolio.groupby("factor", sort=True):
        g = g.sort_values("month")
        ax.plot(g["month"], (1 + g["portfolio_fmp_return"]).cumprod(), label=factor)
    ax.set_title("Portfolio-Sort Factor-Mimicking Portfolios")
    ax.set_ylabel("Cumulative return, gross")
    ax.legend()
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "factor_portfolio_cumulative_returns.png", dpi=160)
    plt.close(fig)


def fig_fmp_regression_cumulative_returns(regression: pd.DataFrame) -> None:
    print("  Fig (migrated): FMP regression cumulative returns")
    if regression.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for factor, g in regression.groupby("factor", sort=True):
        g = g.sort_values("month")
        ax.plot(g["month"], (1 + g["regression_fmp_return"]).cumprod(), label=factor)
    ax.set_title("Regression-Based Factor Returns")
    ax.set_ylabel("Cumulative return, gross")
    ax.legend()
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "factor_regression_cumulative_returns.png", dpi=160)
    plt.close(fig)


def fig_ic_rank_ic_summary(ic: pd.DataFrame) -> None:
    print("  Fig (migrated): IC / Rank-IC summary")
    if ic.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=ic, x="factor", y="ic", ax=axes[0], errorbar=None)
    axes[0].set_title("Average Raw IC")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=ic, x="factor", y="rank_ic", ax=axes[1], errorbar=None)
    axes[1].set_title("Average Rank IC")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "ic_rank_ic_summary.png", dpi=160)
    plt.close(fig)


def fig_strategy_improvement_sharpe(metrics: pd.DataFrame) -> None:
    print("  Fig (migrated): staged-ladder annualized Sharpe bar")
    if metrics.empty:
        return
    df = metrics.copy()
    df["display_name"] = df["strategy"].map(STRATEGY_LABELS).fillna(df.get("strategy_name", df["strategy"]))
    order = df.sort_values("annualized_sharpe", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=order, y="display_name", x="annualized_sharpe", ax=ax, color="#4C78A8")
    ax.set_title("All Strategies: Annualized Sharpe (eval window)")
    ax.set_xlabel("Annualized Sharpe")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "strategy_improvement_sharpe.png", dpi=160)
    plt.close(fig)


def fig_improved4_stop_take_heatmap(grid: pd.DataFrame) -> None:
    print("  Fig (migrated): improved 4 stop/take sensitivity heatmap")
    if grid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "train_sharpe", "Train Sharpe"),
        (axes[1], "test_sharpe", "Test Sharpe Reported After Selection"),
    ]:
        pivot = grid.pivot(index="stop_loss", columns="take_profit", values=metric).sort_index(ascending=True)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": metric})
        ax.set_title(title)
        ax.set_xlabel("Take-profit")
        ax.set_ylabel("Stop-loss")
    fig.savefig(core.FIGURES_DIR / "improved4_stop_take_sensitivity_heatmap.png", dpi=160)
    plt.close(fig)


def fig_improved7_cost_scenarios_equity(curves: pd.DataFrame) -> None:
    print("  Fig (migrated): improved 7 cost-scenarios equity")
    if curves.empty:
        return
    scenarios = sorted(curves["cost_scenario"].dropna().unique().tolist())
    strategies = [s for s in ["improved_4", "improved_6"] if s in curves["strategy_label"].unique()]
    if not strategies:
        return
    fig, axes = plt.subplots(1, len(strategies), figsize=(7 * len(strategies), 5),
                              constrained_layout=True, sharey=False, squeeze=False)
    for ax, strategy_label in zip(axes[0], strategies):
        for scenario in scenarios:
            sub = curves[(curves["strategy_label"].eq(strategy_label))
                         & (curves["cost_scenario"].eq(scenario))].sort_values("month")
            if sub.empty:
                continue
            ax.plot(pd.to_datetime(sub["month"]), sub["equity"], label=scenario)
        ax.set_title(f"{strategy_label} equity under cost scenarios")
        ax.set_xlabel("Month")
        ax.set_ylabel("Equity ($)")
        ax.legend(title="Scenario")
        ax.grid(True, alpha=0.3)
    fig.savefig(core.FIGURES_DIR / "improved7_cost_scenarios_equity.png", dpi=160)
    plt.close(fig)


def fig_improved7_cost_drag_by_year(yearly_drag: pd.DataFrame) -> None:
    print("  Fig (migrated): improved 7 cost-drag by year")
    if yearly_drag.empty:
        return
    strategies = [s for s in ["improved_4", "improved_6"] if s in yearly_drag["strategy_label"].unique()]
    if not strategies:
        return
    fig, axes = plt.subplots(1, len(strategies), figsize=(7 * len(strategies), 5),
                              constrained_layout=True, sharey=True, squeeze=False)
    for ax, strategy_label in zip(axes[0], strategies):
        sub = yearly_drag[(yearly_drag["strategy_label"].eq(strategy_label))
                          & (yearly_drag["cost_scenario"].eq("central"))].sort_values("year")
        if sub.empty:
            continue
        ax.bar(sub["year"], sub["cost_drag_pct_of_avg_equity"] * 100, color="steelblue")
        ax.set_title(f"{strategy_label} per-year cost drag (central scenario)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cost drag, % of avg equity")
        ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(core.FIGURES_DIR / "improved7_cost_drag_by_year.png", dpi=160)
    plt.close(fig)


def fig_improved7_cost_schedule(schedule: pd.DataFrame) -> None:
    print("  Fig (migrated): improved 7 cost schedule")
    if schedule.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for scenario in sorted(schedule["scenario"].dropna().unique().tolist()):
        sub = schedule[schedule["scenario"].eq(scenario)].sort_values("year")
        ax.plot(sub["year"], sub["round_trip_bps"], marker="o", label=scenario)
    ax.set_title("Time-varying transaction-cost schedule (round-trip basis points)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Round-trip cost (bps)")
    ax.legend(title="Scenario")
    ax.grid(True, alpha=0.3)
    fig.savefig(core.FIGURES_DIR / "improved7_cost_schedule.png", dpi=160)
    plt.close(fig)


# ===========================================================================
# COMPOSITE FIGURES — the new deck-ready set
# ===========================================================================

# Helpers shared across composites -----------------------------------------------

SELECTED_STRATEGY = "improved_4"        # risk-adjusted winner (best Sharpe)
WEALTH_WINNER = "improved_8"            # wealth-maximization winner (highest CumRet / final equity)
CO_WINNERS = {"improved_4": "#D62728",  # red
              "improved_8": "#FFB000"}  # gold
REJECTED_STRATEGIES = {"improved_3", "improved_5"}


def _significance_stars(p_value: float) -> str:
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _bench_monthly_returns(index_monthly: pd.DataFrame) -> pd.Series:
    idx = core.filter_to_evaluation_window(index_monthly, "month").sort_values("month")
    return idx.set_index("month")["ret_1m"].astype(float).fillna(0.0)


def _bootstrap_sharpe_ci(strategy: str) -> tuple[float, float, np.ndarray]:
    folder_map = {
        "base": core.BASE_RESULTS_DIR,
        "improved_1": core.IMPROVED_RESULTS_DIR,
        "improved_2": core.IMPROVED_2_RESULTS_DIR,
        "improved_3": core.IMPROVED_3_RESULTS_DIR,
        "improved_4": core.IMPROVED_4_RESULTS_DIR,
        "improved_5": core.IMPROVED_5_RESULTS_DIR,
        "improved_6": core.IMPROVED_6_RESULTS_DIR,
        "improved_8": core.IMPROVED_8_RESULTS_DIR,
    }
    folder = folder_map.get(strategy)
    if folder is None:
        return np.nan, np.nan, np.array([])
    p = folder / "block_bootstrap.csv"
    if not p.exists():
        return np.nan, np.nan, np.array([])
    df = pd.read_csv(p)
    if "annualized_sharpe" not in df.columns:
        return np.nan, np.nan, np.array([])
    s = df["annualized_sharpe"].astype(float).dropna().values
    if len(s) == 0:
        return np.nan, np.nan, s
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975)), s


# ---------------------------------------------------------------------------
# Fig 1 — Factor Analysis Panel (2x2)
# ---------------------------------------------------------------------------

def fig_factor_analysis_panel(
    portfolio: pd.DataFrame,
    regression: pd.DataFrame,
    ic: pd.DataFrame,
    fmp_summary: pd.DataFrame,
) -> None:
    print("  Composite 1: factor analysis panel (2x2)")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax_pf, ax_rg, ax_ic, ax_sh = axes.flatten()

    if not portfolio.empty:
        for factor, g in portfolio.groupby("factor", sort=True):
            g = g.sort_values("month")
            ax_pf.plot(g["month"], (1 + g["portfolio_fmp_return"]).cumprod(), label=factor, linewidth=1.6)
        ax_pf.set_title("(a) Portfolio-Sort FMP — Cumulative Return", fontsize=11)
        ax_pf.set_ylabel("Cumulative return")
        ax_pf.legend(fontsize=9)
        ax_pf.grid(True, alpha=0.3)

    if not regression.empty:
        for factor, g in regression.groupby("factor", sort=True):
            g = g.sort_values("month")
            ax_rg.plot(g["month"], (1 + g["regression_fmp_return"]).cumprod(), label=factor, linewidth=1.6)
        ax_rg.set_title("(b) Regression FMP — Cumulative Return", fontsize=11)
        ax_rg.set_ylabel("Cumulative return")
        ax_rg.legend(fontsize=9)
        ax_rg.grid(True, alpha=0.3)

    if not ic.empty:
        ic_mean = ic.groupby("factor")[["ic", "rank_ic"]].mean().reset_index()
        ic_mean = ic_mean.sort_values("rank_ic", ascending=False)
        x = np.arange(len(ic_mean))
        width = 0.38
        ax_ic.bar(x - width / 2, ic_mean["ic"], width, label="Raw IC", color="#4C78A8")
        ax_ic.bar(x + width / 2, ic_mean["rank_ic"], width, label="Rank IC", color="#F58518")
        ax_ic.set_xticks(x)
        ax_ic.set_xticklabels(ic_mean["factor"])
        ax_ic.axhline(0, color="black", linewidth=0.6)
        ax_ic.set_title("(c) Average IC and Rank IC", fontsize=11)
        ax_ic.set_ylabel("IC")
        ax_ic.legend(fontsize=9)
        ax_ic.grid(True, axis="y", alpha=0.3)

    if not fmp_summary.empty:
        sh = fmp_summary[["factor", "approach", "annualized_sharpe", "t_stat", "p_value"]].copy()
        sh = sh.dropna(subset=["annualized_sharpe"])
        if not sh.empty:
            pivot_s = sh.pivot(index="factor", columns="approach", values="annualized_sharpe")
            pivot_p = sh.pivot(index="factor", columns="approach", values="p_value")
            pivot_s = pivot_s.reindex(sorted(pivot_s.index, key=lambda f: -pivot_s.loc[f].mean()))
            factors = list(pivot_s.index)
            approaches = list(pivot_s.columns)
            x = np.arange(len(factors))
            width = 0.35
            colors = {"portfolio_sort": "#54A24B", "cross_sectional_regression": "#B279A2"}
            for i, app in enumerate(approaches):
                vals = pivot_s[app].values
                ps = pivot_p[app].values
                offset = (i - (len(approaches) - 1) / 2) * width
                bars = ax_sh.bar(x + offset, vals, width,
                                 label=app.replace("_", " "),
                                 color=colors.get(app, None))
                for rect, p in zip(bars, ps):
                    star = _significance_stars(p)
                    if star:
                        y = rect.get_height()
                        ax_sh.text(rect.get_x() + rect.get_width() / 2,
                                   y + 0.01 if y >= 0 else y - 0.04,
                                   star, ha="center", fontsize=10, color="black")
            ax_sh.set_xticks(x)
            ax_sh.set_xticklabels(factors)
            ax_sh.axhline(0, color="black", linewidth=0.6)
            ax_sh.set_title("(d) Factor Risk Premia — Annualized Sharpe (★ = signif.)", fontsize=11)
            ax_sh.set_ylabel("Annualized Sharpe")
            ax_sh.legend(fontsize=8)
            ax_sh.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Factor Analysis — FMPs, IC, and Risk Premia",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "01_factor_analysis_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — FMP weights at picked date (top-20 names per factor)
# ---------------------------------------------------------------------------

def fig_fmp_picked_date_weights(picked: pd.DataFrame) -> None:
    print("  Composite 2: FMP weights at picked date")
    if picked.empty:
        return
    picked = picked.copy()
    picked["month"] = pd.to_datetime(picked["month"])
    date = picked["month"].max()
    snap = picked[picked["month"].eq(date)]
    factors = sorted(snap["factor"].unique())
    fig, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 7),
                              constrained_layout=True, sharex=False)
    if len(factors) == 1:
        axes = [axes]
    for ax, factor in zip(axes, factors):
        g = (snap[snap["factor"].eq(factor)]
             .sort_values("weight", ascending=False)
             .head(20))
        ax.barh(g["ticker"][::-1], g["weight"][::-1] * 100, color="#4C78A8")
        ax.set_title(f"{factor}\n(top 20 names, equal-weighted)", fontsize=10)
        ax.set_xlabel("Weight (%)")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"FMP Long-Leg Composition at {date.date()}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "02_fmp_picked_date_weights.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Strategy Ladder (equity + drawdown stacked)
# ---------------------------------------------------------------------------

def fig_strategy_ladder(all_curves: pd.DataFrame, index_monthly: pd.DataFrame) -> None:
    print("  Composite 3: strategy ladder (equity + drawdown)")
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(14, 11),
                                         sharex=True,
                                         gridspec_kw={"height_ratios": [1.4, 1]},
                                         constrained_layout=True)

    idx_eval = core.filter_to_evaluation_window(index_monthly, "month").sort_values("month")
    if not idx_eval.empty:
        br = idx_eval["ret_1m"].astype(float).fillna(0.0)
        bm_equity = core.INITIAL_CASH * (1 + br).cumprod()
        ax_eq.plot(idx_eval["month"], bm_equity / 1e6, color="black", linewidth=2.4,
                   linestyle="--", label="^GSPC (benchmark)", zorder=10)

    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        if g.empty:
            continue
        g = rebase_eval(g)
        if g.empty:
            continue
        if label in CO_WINNERS:
            suffix = " ★ risk-adj." if label == SELECTED_STRATEGY else " ★ wealth-max."
            color, lw, alpha, z = CO_WINNERS[label], 3.0, 1.0, 9
            display_label = STRATEGY_LABELS.get(label, label) + suffix
        elif label in REJECTED_STRATEGIES:
            color, lw, alpha, z = "lightgrey", 1.0, 0.7, 1
            display_label = STRATEGY_LABELS.get(label, label)
        else:
            color, lw, alpha, z = COLOR_MAP[label], 1.4, 0.85, 5
            display_label = STRATEGY_LABELS.get(label, label)
        ax_eq.plot(g["month"], g["equity_rebased"] / 1e6,
                   color=color, linewidth=lw, alpha=alpha,
                   label=display_label, zorder=z)
        dd = core.make_drawdown(g["equity_rebased"]) * 100
        ax_dd.plot(g["month"], dd, color=color, linewidth=lw, alpha=alpha, zorder=z,
                   label=display_label)

    ax_eq.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax_eq.set_ylabel("Portfolio Value")
    ax_eq.set_title("Equity Curves — Co-winners: Imp 4 (risk-adj, red) & Imp 8 (wealth-max, gold); rejected greyed",
                    fontsize=12)
    ax_eq.legend(fontsize=8, ncol=2, loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    ax_dd.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.axhline(0, color="black", linewidth=0.6)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Month")
    ax_dd.set_title("Drawdown from Peak", fontsize=12)
    ax_dd.grid(True, alpha=0.3)

    fig.suptitle("Strategy Ladder — Equity & Drawdown (eval window, rebased to $1M)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(core.FIGURES_DIR / "03_strategy_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — Risk-Return Scoreboard (Sharpe vs MaxDD, ^GSPC ref point)
# ---------------------------------------------------------------------------

def fig_risk_return_scoreboard(metrics: pd.DataFrame, index_monthly: pd.DataFrame) -> None:
    print("  Composite 4: risk-return scoreboard")
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8))

    # ^GSPC reference point
    br = _bench_monthly_returns(index_monthly)
    if len(br) > 1:
        bench_eq = core.INITIAL_CASH * (1 + br).cumprod()
        bench_sharpe = np.sqrt(12) * br.mean() / br.std(ddof=1) if br.std(ddof=1) > 0 else np.nan
        bench_dd = core.make_drawdown(bench_eq).min()
        if np.isfinite(bench_sharpe):
            ax.scatter(abs(bench_dd) * 100, bench_sharpe, s=350, color="black",
                       marker="X", zorder=10, edgecolors="white", linewidths=1.5,
                       label="^GSPC benchmark")
            ax.annotate("^GSPC", (abs(bench_dd) * 100, bench_sharpe),
                        textcoords="offset points", xytext=(8, 6), fontsize=9, fontweight="bold")

    for _, row in metrics.iterrows():
        label = row.get("strategy", "")
        sharpe = row.get("annualized_sharpe", np.nan)
        dd = row.get("max_drawdown", np.nan)
        eq = row.get("final_equity", 1e6)
        if not (np.isfinite(sharpe) and np.isfinite(dd)):
            continue
        color = COLOR_MAP.get(label, "grey")
        size = max(80, min(800, (eq / 1e6) * 70))
        if label in CO_WINNERS:
            edge = CO_WINNERS[label]
            edge_w = 3.0
        else:
            edge = "white"
            edge_w = 0.8
        ax.scatter(abs(dd) * 100, sharpe, s=size, color=color,
                   alpha=0.85, edgecolors=edge, linewidths=edge_w, zorder=5)
        ax.annotate(STRATEGY_LABELS.get(label, label), (abs(dd) * 100, sharpe),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(10.0, color="grey", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.text(0.5, 1.02, "Sharpe = 1.0", fontsize=8, color="grey", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Max Drawdown (absolute, %)", fontsize=11)
    ax.set_ylabel("Annualized Sharpe", fontsize=11)
    ax.set_title("Risk-Return Frontier — Sharpe vs Max Drawdown\n(bubble ∝ final equity; red edge = Imp 4 risk-adj winner, gold edge = Imp 8 wealth-max winner)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "04_risk_return_scoreboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 — Walk-Forward Train vs Test scatter
# ---------------------------------------------------------------------------

def fig_walk_forward_polished(wf: pd.DataFrame) -> None:
    print("  Composite 5: walk-forward train vs test")
    if wf.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 8))
    lim_lo, lim_hi = -0.3, 1.7
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="grey",
            linestyle="--", linewidth=1, label="Train = Test (fair)")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)

    for _, r in wf.iterrows():
        label = r["strategy"]
        x = r["train_sharpe_to_2020"]
        y = r["test_sharpe_2021_2026"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        is_sel = bool(r.get("selected_by_train", False))
        color = COLOR_MAP.get(label, "grey")
        size = 300 if is_sel else 130
        marker = "*" if is_sel else "o"
        ax.scatter(x, y, s=size, marker=marker, color=color,
                   edgecolors="red" if is_sel else "white",
                   linewidths=2 if is_sel else 0.8, zorder=5)
        ax.annotate(STRATEGY_LABELS.get(label, label), (x, y),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Train Sharpe (≤ 2020)", fontsize=11)
    ax.set_ylabel("Test Sharpe (2021 – May 2026)", fontsize=11)
    ax.set_title("Walk-Forward Validation — Train vs Test Sharpe\n(★ = strategy selected by train; red edge = selected)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "05_walk_forward_train_vs_test.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 — Selected (imp4) Deep-Dive 2x2
# ---------------------------------------------------------------------------

def fig_imp4_deep_dive(all_curves: pd.DataFrame, index_monthly: pd.DataFrame) -> None:
    print("  Composite 6: improved 4 deep-dive")
    g = all_curves[all_curves["strategy"] == SELECTED_STRATEGY]
    if g.empty:
        return
    g = core.filter_to_evaluation_window(g, "month").sort_values("month").copy()
    r = g["portfolio_return"].astype(float).fillna(0.0)
    g["equity_rebased"] = core.INITIAL_CASH * (1 + r).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax_rs, ax_hm, ax_al, ax_ar = axes.flatten()

    # (a) Rolling 12m Sharpe with ^GSPC
    roll = r.rolling(12).apply(
        lambda x: np.sqrt(12) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
        raw=True,
    )
    ax_rs.plot(g["month"], roll, color="#D62728", linewidth=2, label="Imp 4")
    br = _bench_monthly_returns(index_monthly)
    if len(br) >= 14:
        br_roll = br.rolling(12).apply(
            lambda x: np.sqrt(12) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
            raw=True,
        )
        ax_rs.plot(br_roll.index, br_roll.values, color="black", linewidth=1.4,
                   linestyle="--", label="^GSPC")
    ax_rs.axhline(0, color="black", linewidth=0.4)
    ax_rs.axhline(1, color="grey", linewidth=0.6, linestyle=":", alpha=0.7)
    ax_rs.set_title("(a) Rolling 12-Month Sharpe", fontsize=11)
    ax_rs.set_ylabel("Annualized Sharpe")
    ax_rs.legend(fontsize=9)
    ax_rs.grid(True, alpha=0.3)

    # (b) Monthly returns heatmap (year × month)
    heat = g.copy()
    heat["year"] = heat["month"].dt.year
    heat["mon"] = heat["month"].dt.month
    pivot = heat.pivot_table(index="year", columns="mon", values="portfolio_return", aggfunc="sum") * 100
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                ax=ax_hm, cbar_kws={"label": "Monthly return (%)"}, linewidths=0.4)
    ax_hm.set_title("(b) Monthly Returns Heatmap (%)", fontsize=11)
    ax_hm.set_ylabel("Year")
    ax_hm.set_xlabel("")

    # (c) Cumulative alpha vs ^GSPC
    bm = _bench_monthly_returns(index_monthly).reindex(g["month"].values).fillna(0)
    excess = r.values - bm.values
    cumalpha = pd.Series(excess, index=g["month"].values).cumsum() * 100
    ax_al.plot(cumalpha.index, cumalpha.values, color="#D62728", linewidth=2)
    ax_al.fill_between(cumalpha.index, cumalpha.values, 0,
                       where=(cumalpha.values >= 0), color="#D62728", alpha=0.2)
    ax_al.fill_between(cumalpha.index, cumalpha.values, 0,
                       where=(cumalpha.values < 0), color="grey", alpha=0.2)
    ax_al.axhline(0, color="black", linewidth=0.6)
    ax_al.set_title("(c) Cumulative Excess Return vs ^GSPC", fontsize=11)
    ax_al.set_ylabel("Cumulative excess (%)")
    ax_al.grid(True, alpha=0.3)

    # (d) Annual returns
    g_year = g.copy()
    g_year["year"] = g_year["month"].dt.year
    annual = g_year.groupby("year")["portfolio_return"].apply(lambda x: (1 + x).prod() - 1) * 100
    bench_df = pd.DataFrame({
        "month": pd.to_datetime(g["month"].values),
        "ret": bm.values,
    })
    bench_df["year"] = bench_df["month"].dt.year
    bm_year = bench_df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1) * 100
    bm_year = bm_year.reindex(annual.index).fillna(0.0)
    x_pos = np.arange(len(annual))
    width = 0.4
    ax_ar.bar(x_pos - width / 2, annual.values, width, label="Imp 4", color="#D62728")
    ax_ar.bar(x_pos + width / 2, bm_year.values, width, label="^GSPC", color="grey", alpha=0.7)
    ax_ar.set_xticks(x_pos)
    ax_ar.set_xticklabels(annual.index, rotation=0)
    ax_ar.axhline(0, color="black", linewidth=0.4)
    ax_ar.set_title("(d) Annual Returns vs Benchmark", fontsize=11)
    ax_ar.set_ylabel("Annual return (%)")
    ax_ar.legend(fontsize=9)
    ax_ar.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Selected Strategy: Improved 4 — Walk-Forward Stop/Take (5% / 30%)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "06_imp4_deep_dive.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7 — Robustness Verdict Forest (annualized excess return ± bootstrap CI)
# ---------------------------------------------------------------------------

def fig_robustness_verdict_forest(all_curves: pd.DataFrame,
                                    index_monthly: pd.DataFrame,
                                    stepm: pd.DataFrame) -> None:
    print("  Composite 7: robustness verdict forest")
    bm = _bench_monthly_returns(index_monthly)
    rows = []
    for label in STRATEGY_ORDER:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if g.empty:
            continue
        r = g.set_index("month")["portfolio_return"].astype(float).fillna(0.0)
        common = r.index.intersection(bm.index)
        excess = (r.reindex(common) - bm.reindex(common)).values
        if len(excess) < 12:
            continue
        rng = np.random.default_rng(seed=hash(label) & 0xFFFFFFFF)
        block_size = 6
        n = len(excess)
        ann = excess.mean() * 12
        boot = []
        for _ in range(1000):
            sample = []
            while len(sample) < n:
                start = rng.integers(0, max(1, n - block_size))
                sample.extend(excess[start:start + block_size])
            boot.append(np.mean(sample[:n]) * 12)
        boot = np.asarray(boot)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"strategy": label, "ann_excess": ann, "ci_lo": lo, "ci_hi": hi})

    if not rows:
        return
    df = pd.DataFrame(rows).set_index("strategy")
    if not stepm.empty and "superior_to_benchmark_after_stepm" in stepm.columns:
        verdict = stepm.set_index("strategy")["superior_to_benchmark_after_stepm"]
        df["stepm_pass"] = df.index.map(verdict).fillna(False)
    else:
        df["stepm_pass"] = False

    df = df.sort_values("ann_excess")
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (label, r) in enumerate(df.iterrows()):
        color = "#54A24B" if r["stepm_pass"] else "#D62728"
        ax.plot([r["ci_lo"] * 100, r["ci_hi"] * 100], [i, i], color=color, linewidth=3, alpha=0.6)
        ax.scatter(r["ann_excess"] * 100, i, color=color, s=120, zorder=5,
                   edgecolors="white", linewidths=1)
        ax.text(r["ci_hi"] * 100 + 0.3, i,
                f"{r['ann_excess']*100:+.2f}%  [{r['ci_lo']*100:+.1f}, {r['ci_hi']*100:+.1f}]",
                fontsize=8, va="center")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="^GSPC parity")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([STRATEGY_LABELS.get(l, l) for l in df.index])
    ax.set_xlabel("Annualized Excess Return vs ^GSPC (%)  ·  95% block-bootstrap CI")
    ax.set_title("Robustness Verdict — Excess Return vs ^GSPC (Romano-Wolf StepM colored)\n"
                 "Green = passes StepM (none here), Red = does not reject the null",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "07_robustness_verdict_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 8 — Block Bootstrap Sharpe Distributions (small multiples)
# ---------------------------------------------------------------------------

def fig_bootstrap_violins(metrics: pd.DataFrame, index_monthly: pd.DataFrame) -> None:
    print("  Composite 8: bootstrap Sharpe violins")
    if metrics.empty:
        return
    bm = _bench_monthly_returns(index_monthly)
    bench_sharpe = np.sqrt(12) * bm.mean() / bm.std(ddof=1) if bm.std(ddof=1) > 0 else np.nan
    strategies = [s for s in STRATEGY_ORDER if s in metrics["strategy"].values]
    n = len(strategies)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                              constrained_layout=True, sharex=False)
    axes = np.atleast_2d(axes).flatten()
    for ax, label in zip(axes, strategies):
        lo, hi, samples = _bootstrap_sharpe_ci(label)
        if len(samples) == 0:
            ax.set_visible(False)
            continue
        realised = metrics.loc[metrics["strategy"] == label, "annualized_sharpe"].iloc[0]
        ax.violinplot(samples, vert=False, showmeans=False, showmedians=True, widths=0.7)
        ax.axvline(realised, color="red", linewidth=2, label=f"Realized = {realised:.2f}")
        if np.isfinite(bench_sharpe):
            ax.axvline(bench_sharpe, color="black", linestyle="--", linewidth=1.2,
                       label=f"^GSPC = {bench_sharpe:.2f}")
        ax.set_title(STRATEGY_LABELS.get(label, label), fontsize=10)
        ax.set_xlabel("Bootstrap Sharpe")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Block-Bootstrap Sharpe Distributions (1000 sims, 6-mo blocks)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "08_bootstrap_sharpe_violins.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 9 — Cost Sensitivity (2 panels — equity scenarios + drag by year)
# ---------------------------------------------------------------------------

def fig_cost_sensitivity_panel(curves: pd.DataFrame, drag: pd.DataFrame) -> None:
    print("  Composite 9: cost sensitivity panel")
    if curves.empty and drag.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ax_eq, ax_dr = axes

    if not curves.empty:
        scenarios = sorted(curves["cost_scenario"].dropna().unique())
        sub = curves[curves["strategy_label"].eq("improved_4")]
        scen_colors = {"zero": "#54A24B", "central": "#4C78A8", "pessimistic": "#D62728"}
        for scenario in scenarios:
            g = sub[sub["cost_scenario"].eq(scenario)].sort_values("month")
            if g.empty:
                continue
            ax_eq.plot(pd.to_datetime(g["month"]), g["equity"] / 1e6,
                       label=f"{scenario}", linewidth=1.8,
                       color=scen_colors.get(scenario))
        ax_eq.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
        ax_eq.set_title("(a) Imp 4 Equity Under Cost Scenarios", fontsize=11)
        ax_eq.set_xlabel("Month")
        ax_eq.set_ylabel("Equity")
        ax_eq.legend(title="Scenario", fontsize=9)
        ax_eq.grid(True, alpha=0.3)

    if not drag.empty:
        sub = drag[(drag["strategy_label"].eq("improved_4"))
                   & (drag["cost_scenario"].eq("central"))].sort_values("year")
        if not sub.empty:
            ax_dr.bar(sub["year"], sub["cost_drag_pct_of_avg_equity"] * 100,
                      color="#4C78A8")
            ax_dr.set_title("(b) Imp 4 Per-Year Cost Drag (central scenario)", fontsize=11)
            ax_dr.set_xlabel("Year")
            ax_dr.set_ylabel("Cost drag, % of avg equity")
            ax_dr.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Transaction-Cost Sensitivity (Improved 7 overlay on Improved 4)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "09_cost_sensitivity_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 10 — Research Diagnostics (imp4 stop/take + HZZ smoothed-β heatmap)
# ---------------------------------------------------------------------------

def _load_hzz_smoothed_betas() -> pd.DataFrame:
    p = core.IMPROVED_6_RESULTS_DIR / "hzz_smoothed_betas.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def fig_research_diagnostics(grid: pd.DataFrame) -> None:
    print("  Composite 10: research diagnostics (imp 4 stop/take sensitivity)")
    if grid.empty:
        return
    fig, ax_grid = plt.subplots(figsize=(10, 6), constrained_layout=True)
    pivot = grid.pivot(index="stop_loss", columns="take_profit", values="train_sharpe").sort_index()
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis",
                ax=ax_grid, cbar_kws={"label": "Train Sharpe"})
    ax_grid.set_title("Imp 4 Stop/Take Sensitivity — Train Sharpe across grid",
                      fontsize=12, fontweight="bold")
    ax_grid.set_xlabel("Take-profit")
    ax_grid.set_ylabel("Stop-loss")
    fig.savefig(core.FIGURES_DIR / "10_research_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loaders + helpers for Fig 11 and T6 (imp4 vs imp6)
# ---------------------------------------------------------------------------

def _load_trades(strategy_label: str) -> pd.DataFrame:
    folder_files = {
        "improved_4": (core.IMPROVED_4_RESULTS_DIR,
                         "backtrader_daily_improved_4_walkforward_stop_take_top10_trades.csv"),
        "improved_6": (core.IMPROVED_6_RESULTS_DIR,
                         "backtrader_daily_improved_6_hzz_cross_sectional_trend_stop_take_top10_trades.csv"),
    }
    if strategy_label not in folder_files:
        return pd.DataFrame()
    folder, fname = folder_files[strategy_label]
    p = folder / fname
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def _load_hzz_coverage() -> pd.DataFrame:
    p = core.IMPROVED_6_RESULTS_DIR / "hzz_monthly_coverage.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def _strategy_hit_rate(all_curves: pd.DataFrame, label: str) -> float:
    g = all_curves[all_curves["strategy"] == label]
    g = core.filter_to_evaluation_window(g, "month")
    if g.empty:
        return np.nan
    r = g["portfolio_return"].astype(float).dropna()
    return (r > 0).mean() * 100


# ---------------------------------------------------------------------------
# Fig 11 — Imp 4 vs Imp 6 Head-to-Head (2x3)
# ---------------------------------------------------------------------------

def fig_imp4_vs_imp6_panel(all_curves: pd.DataFrame, index_monthly: pd.DataFrame,
                             metrics: pd.DataFrame) -> None:
    print("  Composite 11: imp 4 vs imp 6 head-to-head")
    trades4 = _load_trades("improved_4")
    trades6 = _load_trades("improved_6")
    cov = _load_hzz_coverage()
    hzz = _load_hzz_smoothed_betas()

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])  # equity overlay
    ax_b = fig.add_subplot(gs[0, 1])  # edge accumulation
    ax_c = fig.add_subplot(gs[0, 2])  # HZZ coverage
    ax_d = fig.add_subplot(gs[1, 0])  # trade pnl distribution
    ax_e = fig.add_subplot(gs[1, 1])  # HZZ β heatmap
    ax_f = fig.add_subplot(gs[1, 2])  # metric chips

    # ----- (a) Equity curves overlay -----
    idx_eval = core.filter_to_evaluation_window(index_monthly, "month").sort_values("month")
    if not idx_eval.empty:
        br = idx_eval["ret_1m"].astype(float).fillna(0.0)
        bm_eq = core.INITIAL_CASH * (1 + br).cumprod()
        ax_a.plot(idx_eval["month"], bm_eq / 1e6, color="black", linewidth=1.6,
                  linestyle="--", label="^GSPC", zorder=4)
    g4 = rebase_eval(all_curves[all_curves["strategy"] == "improved_4"])
    g6 = rebase_eval(all_curves[all_curves["strategy"] == "improved_6"])
    if not g4.empty:
        ax_a.plot(g4["month"], g4["equity_rebased"] / 1e6, color="#D62728",
                  linewidth=2.2, label="Imp 4 (expanding trend)", zorder=6)
    if not g6.empty:
        ax_a.plot(g6["month"], g6["equity_rebased"] / 1e6, color="#1F77B4",
                  linewidth=2.2, label="Imp 6 (HZZ trend)", zorder=5)
    ax_a.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax_a.set_title("(a) Equity Curves — Imp 4 vs Imp 6", fontsize=11)
    ax_a.set_ylabel("Equity (rebased to $1M)")
    ax_a.legend(fontsize=9, loc="upper left")
    ax_a.grid(True, alpha=0.3)

    # ----- (b) Edge accumulation: cumulative (imp4 - imp6) -----
    r4 = all_curves[all_curves["strategy"] == "improved_4"][["month", "portfolio_return"]].copy()
    r6 = all_curves[all_curves["strategy"] == "improved_6"][["month", "portfolio_return"]].copy()
    if not r4.empty and not r6.empty:
        merged = r4.merge(r6, on="month", suffixes=("_4", "_6"))
        merged = core.filter_to_evaluation_window(merged, "month").sort_values("month")
        merged["diff"] = merged["portfolio_return_4"].astype(float) - merged["portfolio_return_6"].astype(float)
        merged["cum_diff"] = merged["diff"].cumsum() * 100
        ax_b.plot(merged["month"], merged["cum_diff"], color="black", linewidth=1.6)
        ax_b.fill_between(merged["month"], merged["cum_diff"], 0,
                          where=(merged["cum_diff"] >= 0), color="#D62728", alpha=0.3, label="Imp 4 ahead")
        ax_b.fill_between(merged["month"], merged["cum_diff"], 0,
                          where=(merged["cum_diff"] < 0), color="#1F77B4", alpha=0.3, label="Imp 6 ahead")
        ax_b.axhline(0, color="black", linewidth=0.6)
        ax_b.set_title("(b) Edge Accumulation — cumulative (Imp 4 − Imp 6) %", fontsize=11)
        ax_b.set_ylabel("Cumulative excess (%)")
        ax_b.legend(fontsize=9, loc="upper left")
        ax_b.grid(True, alpha=0.3)

    # ----- (c) HZZ signal coverage -----
    if not cov.empty:
        cov = cov.sort_values("month").copy()
        cov["pct_covered"] = np.where(cov["n_eligible"] > 0,
                                       100 * cov["n_trend_hzz"] / cov["n_eligible"], 0)
        ax_c.plot(cov["month"], cov["pct_covered"], color="#1F77B4", linewidth=1.8,
                  label="% stocks with HZZ signal")
        eval_start = pd.Timestamp("2016-05-31")
        ax_c.axvspan(cov["month"].min(), eval_start, color="grey", alpha=0.18,
                     label="pre-eval window")
        ax_c.axvline(eval_start, color="black", linestyle=":", linewidth=1)
        ax_c.set_title("(c) HZZ Signal Coverage Over Time", fontsize=11)
        ax_c.set_ylabel("% of eligible stocks with HZZ signal")
        ax_c.set_ylim(-3, 105)
        ax_c.legend(fontsize=9, loc="lower right")
        ax_c.grid(True, alpha=0.3)
        # Annotation: explain the smoking gun
        first_signal = cov.loc[cov["n_trend_hzz"] > 0, "month"].min()
        if pd.notna(first_signal):
            ax_c.annotate(f"First HZZ signal\n{first_signal.date()}",
                          xy=(first_signal, 0), xytext=(20, 30),
                          textcoords="offset points", fontsize=8,
                          arrowprops=dict(arrowstyle="->", color="#1F77B4"))

    # ----- (d) Trade PnL distributions -----
    if not trades4.empty or not trades6.empty:
        bins = np.linspace(-15000, 15000, 50)
        if not trades4.empty:
            t4_eval = trades4[trades4["date"] >= pd.Timestamp("2016-05-31")]
            ax_d.hist(t4_eval["pnl_comm"], bins=bins, color="#D62728", alpha=0.55,
                      label=f"Imp 4 (n={len(t4_eval)})", edgecolor="white", linewidth=0.3)
        if not trades6.empty:
            t6_eval = trades6[trades6["date"] >= pd.Timestamp("2016-05-31")]
            ax_d.hist(t6_eval["pnl_comm"], bins=bins, color="#1F77B4", alpha=0.55,
                      label=f"Imp 6 (n={len(t6_eval)})", edgecolor="white", linewidth=0.3)
        ax_d.axvline(0, color="black", linewidth=0.6)
        ax_d.set_title("(d) Trade PnL Distribution (eval window)", fontsize=11)
        ax_d.set_xlabel("Per-trade PnL ($, after commission)")
        ax_d.set_ylabel("Number of trades")
        ax_d.legend(fontsize=9)
        ax_d.grid(True, axis="y", alpha=0.3)

    # ----- (e) HZZ smoothed-β heatmap -----
    if not hzz.empty:
        ma_cols = [c for c in hzz.columns if c.startswith("ma_ratio_")]
        if ma_cols:
            tmp = hzz.set_index("month")[ma_cols].dropna(how="all")
            tmp.columns = [c.replace("ma_ratio_", "") for c in ma_cols]
            sns.heatmap(tmp.T, cmap="RdBu_r", center=0, ax=ax_e,
                        cbar_kws={"label": "Smoothed β"})
            ax_e.set_title("(e) HZZ Smoothed β by MA Lag (drives Imp 6 signal)", fontsize=11)
            ax_e.set_ylabel("MA lag (days)")
            ax_e.set_xlabel("Month")
            ticks = ax_e.get_xticks()
            keep_idx = np.linspace(0, len(ticks) - 1, num=min(8, len(ticks))).astype(int)
            try:
                labels = [t.get_text() for t in ax_e.get_xticklabels()]
                ax_e.set_xticks([ticks[i] for i in keep_idx])
                ax_e.set_xticklabels([labels[i][:7] if i < len(labels) else "" for i in keep_idx],
                                       rotation=45)
            except Exception:
                pass

    # ----- (f) Metric chips bar -----
    if not metrics.empty:
        m = metrics.set_index("strategy")
        def get(label, col):
            try:
                return float(m.loc[label, col])
            except Exception:
                return np.nan
        chips = []
        # (label, imp4 value, imp6 value, fmt, higher_is_better)
        sharpe4, sharpe6 = get("improved_4", "annualized_sharpe"), get("improved_6", "annualized_sharpe")
        retann4, retann6 = get("improved_4", "annualized_return_approx") * 100, get("improved_6", "annualized_return_approx") * 100
        dd4, dd6 = abs(get("improved_4", "max_drawdown")) * 100, abs(get("improved_6", "max_drawdown")) * 100
        n4, n6 = len(trades4) if not trades4.empty else 0, len(trades6) if not trades6.empty else 0
        bl4 = trades4["bar_len"].mean() if not trades4.empty else np.nan
        bl6 = trades6["bar_len"].mean() if not trades6.empty else np.nan
        hr4 = _strategy_hit_rate(all_curves, "improved_4")
        hr6 = _strategy_hit_rate(all_curves, "improved_6")
        chips = [
            ("Sharpe",         sharpe4, sharpe6, "{:.2f}",  True),
            ("Ann. Ret. (%)",  retann4, retann6, "{:.1f}",  True),
            ("Max DD (%)",     dd4, dd6, "{:.1f}",          False),
            ("# Trades",       n4, n6, "{:.0f}",            False),  # fewer = less churn
            ("Avg hold (days)", bl4, bl6, "{:.0f}",         True),
            ("Hit-rate (%)",   hr4, hr6, "{:.1f}",          True),
        ]
        labels = [c[0] for c in chips]
        v4 = np.array([c[1] for c in chips], dtype=float)
        v6 = np.array([c[2] for c in chips], dtype=float)
        higher_better = [c[4] for c in chips]
        y = np.arange(len(chips))
        width = 0.4
        ax_f.barh(y - width / 2, v4, width, color="#D62728", label="Imp 4")
        ax_f.barh(y + width / 2, v6, width, color="#1F77B4", label="Imp 6")
        ax_f.set_yticks(y)
        ax_f.set_yticklabels(labels)
        ax_f.invert_yaxis()
        for i, (val4, val6, fmt, hb) in enumerate(zip(v4, v6, [c[3] for c in chips], higher_better)):
            if not (np.isfinite(val4) and np.isfinite(val6)):
                continue
            # Mark winner
            imp4_wins = (val4 > val6) if hb else (val4 < val6)
            star_4 = "★" if imp4_wins else ""
            star_6 = "" if imp4_wins else "★"
            ax_f.text(val4, i - width / 2, f"  {fmt.format(val4)}{star_4}",
                      va="center", fontsize=9, color="black")
            ax_f.text(val6, i + width / 2, f"  {fmt.format(val6)}{star_6}",
                      va="center", fontsize=9, color="black")
        ax_f.set_title("(f) Head-to-Head Metric Chips  (★ = winner)", fontsize=11)
        ax_f.legend(fontsize=9, loc="lower right")
        ax_f.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Imp 4 (expanding-^GSPC trend) vs Imp 6 (HZZ paper-faithful trend) — "
                 "Why Imp 4 Wins",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "11_imp4_vs_imp6_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 12 — Improvement Waterfall
# ---------------------------------------------------------------------------

def fig_improvement_waterfall(metrics: pd.DataFrame) -> None:
    print("  Composite 12: improvement waterfall (kept ladder + rejected branches)")
    if metrics.empty:
        return
    kept = ["base", "improved_1", "improved_2", "improved_4", "improved_8"]
    rejected = ["improved_3", "improved_5", "improved_6"]
    m = metrics.set_index("strategy")

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Kept ladder bars (green-ish, left-to-right narrative; co-winners highlighted)
    x_kept = np.arange(len(kept))
    sh_kept = [m.loc[k, "annualized_sharpe"] if k in m.index else np.nan for k in kept]
    bar_colors = ["#4C9A2A"] * len(kept)
    bar_edges = ["white"] * len(kept)
    bar_widths = [1.2] * len(kept)
    for i, k in enumerate(kept):
        if k in CO_WINNERS:
            bar_colors[i] = CO_WINNERS[k]
            bar_edges[i] = "black"
            bar_widths[i] = 2.4
    bars = ax.bar(x_kept, sh_kept, width=0.55, color=bar_colors, alpha=0.9,
                   edgecolor=bar_edges, linewidth=bar_widths,
                   label="Kept on the improvement path")
    # Crown markers above co-winners
    for i, k in enumerate(kept):
        if k in CO_WINNERS and np.isfinite(sh_kept[i]):
            crown = "♛ risk-adj.\nwinner" if k == SELECTED_STRATEGY else "♛ wealth-max.\nwinner"
            ax.text(i, sh_kept[i] + 0.18, crown, ha="center", fontsize=8,
                    fontweight="bold", color=CO_WINNERS[k])
    # Δ-arrow annotations between kept bars
    for i in range(1, len(kept)):
        prev, curr = sh_kept[i - 1], sh_kept[i]
        if np.isfinite(prev) and np.isfinite(curr):
            delta = curr - prev
            color = "#2E7D32" if delta >= 0 else "#C62828"
            sign = "+" if delta >= 0 else ""
            ax.annotate(f"Δ {sign}{delta:.2f}",
                        xy=(i - 0.5, max(prev, curr) + 0.05),
                        ha="center", fontsize=9, color=color, fontweight="bold")
            ax.annotate("", xy=(i, curr), xytext=(i - 1, prev),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.7))
    # Value on top of each kept bar
    for bar, v in zip(bars, sh_kept):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    # Rejected branch bars (right side, grey)
    x_rej = np.arange(len(rejected)) + len(kept) + 1.5
    sh_rej = [m.loc[k, "annualized_sharpe"] if k in m.index else np.nan for k in rejected]
    rej_bars = ax.bar(x_rej, sh_rej, width=0.55, color="#9E9E9E", alpha=0.7,
                       edgecolor="white", linewidth=1.2, label="Rejected branches")
    for bar, k, v in zip(rej_bars, rejected, sh_rej):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", fontsize=10)
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.10,
                    "✗", ha="center", fontsize=14, color="#C62828", fontweight="bold")

    # ^GSPC reference line
    bench_sharpe = 0.92  # close to computed value; if we want exact use index_monthly
    ax.axhline(bench_sharpe, color="black", linestyle="--", linewidth=1,
               alpha=0.7, label=f"^GSPC Sharpe ≈ {bench_sharpe:.2f}")

    # X-axis
    all_x = np.concatenate([x_kept, x_rej])
    all_labels = [STRATEGY_LABELS.get(k, k) for k in kept] + [STRATEGY_LABELS.get(k, k) for k in rejected]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Annualized Sharpe (eval window)")
    ax.set_title("Improvement Waterfall — Co-winners: Imp 4 (risk-adj, red) & Imp 8 (wealth-max, gold)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max([s for s in sh_kept + sh_rej if np.isfinite(s)] + [bench_sharpe]) + 0.45)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(core.FIGURES_DIR / "12_improvement_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 13 — Imp 5 Regime Ribbon
# ---------------------------------------------------------------------------

def _load_imp5_regime() -> pd.DataFrame:
    p = core.IMPROVED_5_RESULTS_DIR / "monthly_regime_exposure.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def fig_imp5_regime_ribbon() -> None:
    print("  Composite 13: imp 5 regime ribbon (why regime filter was rejected)")
    df = _load_imp5_regime()
    if df.empty:
        return
    df = df.sort_values("month").copy()
    df_eval = core.filter_to_evaluation_window(df, "month")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6),
                                          sharex=True,
                                          gridspec_kw={"height_ratios": [4, 1]},
                                          constrained_layout=True)

    # Top: ^GSPC index price + 10-month SMA
    ax_top.plot(df["month"], df["index_close"], color="black", linewidth=1.6,
                label="^GSPC monthly close")
    if "sma_10m" in df.columns:
        ax_top.plot(df["month"], df["sma_10m"], color="#1F77B4", linewidth=1.4,
                    linestyle="--", label="10-month SMA")
    eval_start = pd.Timestamp("2016-05-31")
    ax_top.axvline(eval_start, color="grey", linestyle=":", linewidth=1)
    ax_top.text(eval_start, ax_top.get_ylim()[1] * 0.97, " eval start",
                fontsize=8, va="top", color="grey")
    ax_top.set_title("(top) ^GSPC price + 10-month SMA",
                     fontsize=11)
    ax_top.set_ylabel("Index level")
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.grid(True, alpha=0.3)

    # Bottom: regime ribbon
    months = df["month"].values
    regime = df["regime_on"].astype(bool).values
    # Build segments
    for i in range(len(months)):
        color = "#4C9A2A" if regime[i] else "#D62728"
        ax_bot.axvspan(months[i] - pd.Timedelta(days=15),
                       months[i] + pd.Timedelta(days=15),
                       color=color, alpha=0.85)
    ax_bot.set_yticks([])
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xlabel("Month")
    ax_bot.set_title("(bottom) Regime ribbon — green = IN market, red = OUT", fontsize=10)
    # Annotation: months OUT in eval window
    if not df_eval.empty:
        out_eval = (~df_eval["regime_on"].astype(bool)).sum()
        total_eval = len(df_eval)
        annotation = (f"In the evaluation window, Imp 5 sat OUT for {out_eval} of "
                      f"{total_eval} months ({100*out_eval/total_eval:.1f} %). "
                      f"Most of these were 2020-03 → 2020-09 — missing the rebound.")
        ax_top.text(0.5, -0.20, annotation, transform=ax_top.transAxes,
                    ha="center", fontsize=9, style="italic", color="#555555")

    fig.suptitle("Imp 5 — Regime Filter Exposure (why Imp 5 was rejected)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "13_imp5_regime_ribbon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 14 — Trend Regression Survivors
# ---------------------------------------------------------------------------

def _load_trend_regression_coeffs() -> pd.DataFrame:
    p = core.FMP_RESULTS_DIR / "trend_index_regression_coefficients.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def fig_trend_regression_survivors() -> None:
    print("  Composite 14: trend regression survivors")
    df = _load_trend_regression_coeffs()
    if df.empty:
        return
    df = df[df["term"].str.startswith("ma_dev_", na=False)].copy()
    if df.empty:
        return
    df["lag"] = df["term"].str.replace("ma_dev_", "", regex=False).astype(int)
    df = df.sort_values("lag")

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    colors = ["#4C9A2A" if bool(s) else "#BDBDBD" for s in df["selected"]]
    y = np.arange(len(df))
    bars = ax.barh(y, df["coefficient"], color=colors, edgecolor="white", linewidth=0.6)
    for i, (_, r) in enumerate(df.iterrows()):
        marker = "✓ kept" if bool(r["selected"]) else "  dropped"
        ax.text(0.001 if r["coefficient"] >= 0 else -0.001, i,
                f"  p={r['full_model_p_value']:.3f}  {marker}",
                va="center", fontsize=9,
                color="#2E7D32" if bool(r["selected"]) else "#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"ma_dev_{lag}" for lag in df["lag"]])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Regression coefficient on ^GSPC predictive regression")
    r2 = df["r_squared_refit"].iloc[0] if "r_squared_refit" in df.columns else np.nan
    ax.set_title(
        f"Trend Predictive Regression on ^GSPC — Variable Selection (R² of refit = {r2*100:.2f} %)",
        fontsize=12, fontweight="bold")
    fig.text(0.5, -0.01,
              "Assignment requirement: keep only MA-deviation terms with p < 0.05 in the full model.  "
              "Only ma_dev_5 survives.",
              ha="center", fontsize=9, style="italic", color="#555555")
    fig.savefig(core.FIGURES_DIR / "14_trend_regression_survivors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 15 — Imp 4 vs Imp 8 Head-to-Head (Co-Winners)
# ---------------------------------------------------------------------------

def _load_imp8_holdings_or_metrics():
    """Imp 8 doesn't share imp 4's trades schema (it uses equity-pct sizing).
    We use the vector_holdings file to compute # trades / avg holding length."""
    p4 = core.IMPROVED_4_RESULTS_DIR / "backtrader_daily_improved_4_walkforward_stop_take_top10_trades.csv"
    p8 = core.IMPROVED_8_RESULTS_DIR / "backtrader_daily_improved_8_equal_weight_top20_trades.csv"
    t4 = pd.read_csv(p4, parse_dates=["date"]) if p4.exists() else pd.DataFrame()
    t8 = pd.read_csv(p8, parse_dates=["date"]) if p8.exists() else pd.DataFrame()
    return t4, t8


def fig_imp4_vs_imp8_panel(all_curves: pd.DataFrame, index_monthly: pd.DataFrame,
                             metrics: pd.DataFrame) -> None:
    print("  Composite 15: imp 4 vs imp 8 head-to-head (co-winners)")
    trades4, trades8 = _load_imp8_holdings_or_metrics()

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])  # equity overlay
    ax_b = fig.add_subplot(gs[0, 1])  # edge accumulation
    ax_c = fig.add_subplot(gs[0, 2])  # rolling 12m Sharpe
    ax_d = fig.add_subplot(gs[1, 0])  # annual returns
    ax_e = fig.add_subplot(gs[1, 1])  # position count over time
    ax_f = fig.add_subplot(gs[1, 2])  # metric chips

    c4, c8 = CO_WINNERS["improved_4"], CO_WINNERS["improved_8"]

    # ----- (a) Equity curves overlay -----
    idx_eval = core.filter_to_evaluation_window(index_monthly, "month").sort_values("month")
    if not idx_eval.empty:
        br = idx_eval["ret_1m"].astype(float).fillna(0.0)
        bm_eq = core.INITIAL_CASH * (1 + br).cumprod()
        ax_a.plot(idx_eval["month"], bm_eq / 1e6, color="black", linewidth=1.6,
                  linestyle="--", label="^GSPC", zorder=4)
    g4 = rebase_eval(all_curves[all_curves["strategy"] == "improved_4"])
    g8 = rebase_eval(all_curves[all_curves["strategy"] == "improved_8"])
    if not g4.empty:
        ax_a.plot(g4["month"], g4["equity_rebased"] / 1e6, color=c4,
                  linewidth=2.4, label="Imp 4 (risk-adj)", zorder=6)
    if not g8.empty:
        ax_a.plot(g8["month"], g8["equity_rebased"] / 1e6, color=c8,
                  linewidth=2.4, label="Imp 8 (wealth-max)", zorder=6)
    ax_a.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax_a.set_title("(a) Equity Curves — Imp 4 vs Imp 8", fontsize=11)
    ax_a.set_ylabel("Equity (rebased to $1M)")
    ax_a.legend(fontsize=9, loc="upper left")
    ax_a.grid(True, alpha=0.3)

    # ----- (b) Edge accumulation: cumulative (imp8 - imp4) -----
    r4 = all_curves[all_curves["strategy"] == "improved_4"][["month", "portfolio_return"]]
    r8 = all_curves[all_curves["strategy"] == "improved_8"][["month", "portfolio_return"]]
    if not r4.empty and not r8.empty:
        merged = r4.merge(r8, on="month", suffixes=("_4", "_8"))
        merged = core.filter_to_evaluation_window(merged, "month").sort_values("month")
        merged["diff"] = (merged["portfolio_return_8"].astype(float)
                          - merged["portfolio_return_4"].astype(float))
        merged["cum_diff"] = merged["diff"].cumsum() * 100
        ax_b.plot(merged["month"], merged["cum_diff"], color="black", linewidth=1.6)
        ax_b.fill_between(merged["month"], merged["cum_diff"], 0,
                          where=(merged["cum_diff"] >= 0), color=c8, alpha=0.35,
                          label="Imp 8 ahead (return)")
        ax_b.fill_between(merged["month"], merged["cum_diff"], 0,
                          where=(merged["cum_diff"] < 0), color=c4, alpha=0.35,
                          label="Imp 4 ahead (return)")
        ax_b.axhline(0, color="black", linewidth=0.6)
        ax_b.set_title("(b) Edge Accumulation — cumulative (Imp 8 − Imp 4) %", fontsize=11)
        ax_b.set_ylabel("Cumulative excess (%)")
        ax_b.legend(fontsize=9, loc="upper left")
        ax_b.grid(True, alpha=0.3)

    # ----- (c) Rolling 12m Sharpe (both + ^GSPC) -----
    bm = _bench_monthly_returns(index_monthly)
    for label, color in [("improved_4", c4), ("improved_8", c8)]:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if len(g) >= 14:
            r = g["portfolio_return"].astype(float)
            roll = r.rolling(12).apply(
                lambda x: np.sqrt(12) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
                raw=True,
            )
            ax_c.plot(g["month"], roll, color=color, linewidth=1.8,
                      label=STRATEGY_LABELS.get(label, label))
    if len(bm) >= 14:
        br_roll = bm.rolling(12).apply(
            lambda x: np.sqrt(12) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
            raw=True,
        )
        ax_c.plot(br_roll.index, br_roll.values, color="black", linewidth=1.2,
                  linestyle="--", label="^GSPC")
    ax_c.axhline(0, color="black", linewidth=0.4)
    ax_c.axhline(1, color="grey", linewidth=0.5, linestyle=":", alpha=0.7)
    ax_c.set_title("(c) Rolling 12-Month Sharpe", fontsize=11)
    ax_c.set_ylabel("Annualized Sharpe")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # ----- (d) Annual returns -----
    annual_dfs = {}
    for label in ["improved_4", "improved_8"]:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month").copy()
        if g.empty:
            continue
        g["year"] = g["month"].dt.year
        annual_dfs[label] = (g.groupby("year")["portfolio_return"]
                              .apply(lambda x: (1 + x).prod() - 1) * 100)
    if annual_dfs:
        years = sorted(set().union(*[set(s.index) for s in annual_dfs.values()]))
        x = np.arange(len(years))
        width = 0.28
        bench_eval = pd.DataFrame({"month": pd.to_datetime(bm.index), "ret": bm.values})
        bench_eval["year"] = bench_eval["month"].dt.year
        bm_year = (bench_eval.groupby("year")["ret"]
                   .apply(lambda x: (1 + x).prod() - 1) * 100).reindex(years).fillna(0)
        ax_d.bar(x - width, [annual_dfs.get("improved_4", pd.Series()).get(y, 0) for y in years],
                  width, label="Imp 4", color=c4)
        ax_d.bar(x, [annual_dfs.get("improved_8", pd.Series()).get(y, 0) for y in years],
                  width, label="Imp 8", color=c8)
        ax_d.bar(x + width, bm_year.values, width, label="^GSPC", color="grey", alpha=0.6)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(years, rotation=0, fontsize=9)
        ax_d.axhline(0, color="black", linewidth=0.4)
        ax_d.set_title("(d) Annual Returns vs Benchmark", fontsize=11)
        ax_d.set_ylabel("Annual return (%)")
        ax_d.legend(fontsize=9)
        ax_d.grid(True, axis="y", alpha=0.3)

    # ----- (e) Position count over time (top-10 vs top-20) -----
    for label, color in [("improved_4", c4), ("improved_8", c8)]:
        g = all_curves[all_curves["strategy"] == label]
        g = core.filter_to_evaluation_window(g, "month").sort_values("month")
        if "n_positions" in g.columns and not g.empty:
            ax_e.plot(g["month"], g["n_positions"], color=color, linewidth=1.8,
                      label=STRATEGY_LABELS.get(label, label))
    ax_e.set_title("(e) # Positions Held Over Time (top-10 vs top-20)", fontsize=11)
    ax_e.set_ylabel("# positions")
    ax_e.legend(fontsize=9)
    ax_e.grid(True, alpha=0.3)

    # ----- (f) Metric chips -----
    if not metrics.empty:
        m = metrics.set_index("strategy")
        def get(label, col, default=np.nan):
            try:
                return float(m.loc[label, col])
            except Exception:
                return default
        sharpe4, sharpe8 = get("improved_4", "annualized_sharpe"), get("improved_8", "annualized_sharpe")
        ret4, ret8 = get("improved_4", "annualized_return_approx") * 100, get("improved_8", "annualized_return_approx") * 100
        dd4, dd8 = abs(get("improved_4", "max_drawdown")) * 100, abs(get("improved_8", "max_drawdown")) * 100
        cr4, cr8 = get("improved_4", "cumulative_return") * 100, get("improved_8", "cumulative_return") * 100
        fe4, fe8 = get("improved_4", "final_equity") / 1e6, get("improved_8", "final_equity") / 1e6
        n4, n8 = len(trades4) if not trades4.empty else 0, len(trades8) if not trades8.empty else 0
        hr4, hr8 = _strategy_hit_rate(all_curves, "improved_4"), _strategy_hit_rate(all_curves, "improved_8")

        chips = [
            ("Sharpe",         sharpe4, sharpe8, "{:.2f}",   True),
            ("Ann. Ret. (%)",  ret4, ret8,       "{:.1f}",   True),
            ("Max DD (%)",     dd4, dd8,         "{:.1f}",   False),
            ("Cum. Ret. (%)",  cr4, cr8,         "{:.0f}",   True),
            ("Final $ (M)",    fe4, fe8,         "{:.2f}",   True),
            ("Hit-rate (%)",   hr4, hr8,         "{:.1f}",   True),
        ]
        labels = [c[0] for c in chips]
        v4 = np.array([c[1] for c in chips], dtype=float)
        v8 = np.array([c[2] for c in chips], dtype=float)
        higher_better = [c[4] for c in chips]
        y = np.arange(len(chips))
        width = 0.4
        ax_f.barh(y - width / 2, v4, width, color=c4, label="Imp 4 (risk-adj)")
        ax_f.barh(y + width / 2, v8, width, color=c8, label="Imp 8 (wealth-max)")
        ax_f.set_yticks(y)
        ax_f.set_yticklabels(labels)
        ax_f.invert_yaxis()
        for i, (val4, val8, fmt, hb) in enumerate(zip(v4, v8, [c[3] for c in chips], higher_better)):
            if not (np.isfinite(val4) and np.isfinite(val8)):
                continue
            imp4_wins = (val4 > val8) if hb else (val4 < val8)
            star_4 = " ★" if imp4_wins else ""
            star_8 = "" if imp4_wins else " ★"
            ax_f.text(val4, i - width / 2, f"  {fmt.format(val4)}{star_4}",
                      va="center", fontsize=9, color="black")
            ax_f.text(val8, i + width / 2, f"  {fmt.format(val8)}{star_8}",
                      va="center", fontsize=9, color="black")
        ax_f.set_title("(f) Head-to-Head Metric Chips  (★ = winner per metric)", fontsize=11)
        ax_f.legend(fontsize=9, loc="lower right")
        ax_f.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Co-Winners: Imp 4 (top-10, walk-forward stops) vs Imp 8 (top-20, 1/N sizing) — "
                 "Risk-Adjusted vs Wealth-Maximization",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(core.FIGURES_DIR / "15_imp4_vs_imp8_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# TABLE-PNG RENDERERS — slide-ready tables saved as figures
# ===========================================================================

def _render_table_png(df: pd.DataFrame, path: Path, title: str,
                       caption: str = "", col_colors: dict | None = None,
                       row_colors: list | None = None,
                       cell_color_fn=None,
                       fontsize: int = 10, scale: tuple = (1, 1.5)) -> None:
    # Width-aware sizing: estimate column widths from content
    char_w = 0.12  # rough inches per character at fontsize 10
    col_widths = []
    for col in df.columns:
        max_len = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) else 0)
        col_widths.append(max(1.0, (max_len + 2) * char_w))
    total_w = sum(col_widths)
    fig_w = max(8.0, min(22, total_w + 1.5))
    fig_h = max(2.5, 0.55 * (len(df) + 2) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values.astype(str),
                    colLabels=df.columns.tolist(),
                    loc="center",
                    cellLoc="center",
                    colWidths=[w / total_w for w in col_widths])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    # NOTE: do not call tbl.scale() — it stretches cells without reflowing text,
    # causing content to overflow column boundaries.  Row height is controlled
    # via fig_h above instead.
    _ = scale  # unused, retained for API compat

    # Header styling
    for c in range(len(df.columns)):
        cell = tbl[0, c]
        cell.set_facecolor("#1F3A68")
        cell.set_text_props(color="white", fontweight="bold")

    # Cell coloring callback
    if cell_color_fn is not None:
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                color = cell_color_fn(df.iloc[r, c], col, r)
                if color is not None:
                    tbl[r + 1, c].set_facecolor(color)

    # Zebra rows by default unless overridden
    if cell_color_fn is None:
        for r in range(len(df)):
            for c in range(len(df.columns)):
                tbl[r + 1, c].set_facecolor("#F7F7F7" if r % 2 == 0 else "white")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    if caption:
        fig.text(0.5, 0.02, caption, ha="center", fontsize=8, style="italic", color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _color_by_sharpe(val: str) -> str | None:
    try:
        f = float(val)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    if f >= 1.2:
        return "#A8E6A1"
    if f >= 1.0:
        return "#D4F1D4"
    if f >= 0.8:
        return "#FFF4C2"
    return "#F5C2C2"


def _color_by_dd(val: str) -> str | None:
    try:
        f = float(val.rstrip("%"))
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    # f is the absolute drawdown in percent
    if f <= 8:
        return "#A8E6A1"
    if f <= 12:
        return "#FFF4C2"
    return "#F5C2C2"


def table_universe_summary(path: Path = None) -> None:
    print("  Table T1: universe summary")
    if path is None:
        path = core.FIGURES_DIR / "T1_universe_summary.png"
    audit_p = core.RESULTS_DIR / "data_audit.csv"
    val_p = core.RESULTS_DIR / "validation_summary.csv"
    rows: list[dict] = []
    if audit_p.exists():
        a = pd.read_csv(audit_p)
        rows.append({"Item": "Universe size (current S&P 500 constituents)",
                     "Value": str(int(a["n_tickers"].iloc[0])) if "n_tickers" in a.columns else "—"})
    rows.append({"Item": "Benchmark", "Value": "^GSPC (Yahoo Finance, frozen)"})
    rows.append({"Item": "Panel frequency", "Value": "Monthly (factors); Daily (Backtrader)"})
    rows.append({"Item": "Data window", "Value": "2006-06 → 2026-05 (frozen)"})
    rows.append({"Item": "Evaluation window", "Value": "2016-05-31 onward (common to all strategies)"})
    rows.append({"Item": "Initial capital", "Value": "$1,000,000"})
    rows.append({"Item": "Sizing (base ladder)", "Value": "FixedCashSizer = $100,000 per position"})
    rows.append({"Item": "Commission", "Value": "0 (sensitivity layer in Imp 7)"})
    rows.append({"Item": "Factors", "Value": "ROE, P/E, 12m Momentum, Trend (assignment + HZZ paper)"})
    rows.append({"Item": "Strategies tested", "Value": "8 (base + improved 1–6, 8) + 1 robustness overlay (improved 7 costs)"})
    df = pd.DataFrame(rows)
    _render_table_png(df, path,
                       title="T1 — Data, Universe & Setup",
                       caption="All figures use the frozen panel and the common evaluation window.",
                       scale=(1, 1.6))


def table_strategy_scoreboard(metrics: pd.DataFrame, mc: pd.DataFrame, stepm: pd.DataFrame,
                                path: Path = None) -> None:
    print("  Table T2: strategy scoreboard")
    if path is None:
        path = core.FIGURES_DIR / "T2_strategy_scoreboard.png"
    if metrics.empty:
        return
    df = metrics.copy().set_index("strategy")
    mc_p = mc.set_index("strategy")["p_value"] if not mc.empty and "p_value" in mc.columns else pd.Series(dtype=float)
    if not stepm.empty and "superior_to_benchmark_after_stepm" in stepm.columns:
        sw = stepm.set_index("strategy")["superior_to_benchmark_after_stepm"]
    else:
        sw = pd.Series(dtype=bool)

    rows = []
    for label in STRATEGY_ORDER:
        if label not in df.index:
            continue
        r = df.loc[label]
        rows.append({
            "Strategy": STRATEGY_LABELS.get(label, label),
            "Sharpe": f"{r['annualized_sharpe']:.2f}",
            "Ann. Ret.": f"{r['annualized_return_approx']*100:.2f}%",
            "Max DD": f"{abs(r['max_drawdown'])*100:.1f}%",
            "Cum. Ret.": f"{r['cumulative_return']*100:.1f}%",
            "Final $": f"${r['final_equity']/1e6:.2f}M",
            "MC p (raw)": f"{mc_p.get(label, np.nan):.3f}" if label in mc_p.index else "—",
            "Beats ^GSPC?": "Yes" if sw.get(label, False) else "No",
        })
    out = pd.DataFrame(rows)

    def cell_color(val, col, _row):
        if col == "Sharpe":
            return _color_by_sharpe(val)
        if col == "Max DD":
            return _color_by_dd(val)
        if col == "Beats ^GSPC?":
            return "#A8E6A1" if val == "Yes" else "#F5C2C2"
        return None

    _render_table_png(out, path,
                       title="T2 — 8-Strategy Scoreboard (eval window)",
                       caption="Sharpe colored green ≥ 1.2, yellow ≥ 1.0, red < 0.8.  "
                               "MaxDD colored green ≤ 8 %, yellow ≤ 12 %, red > 12 %.  "
                               "‘Beats ^GSPC?’ uses Romano-Wolf StepM after multi-comparison correction.",
                       cell_color_fn=cell_color,
                       fontsize=10, scale=(1, 1.5))


def table_factor_premia(fmp_summary: pd.DataFrame, path: Path = None) -> None:
    print("  Table T3: factor risk premia")
    if path is None:
        path = core.FIGURES_DIR / "T3_factor_risk_premia.png"
    if fmp_summary.empty:
        return
    df = fmp_summary.copy()
    df["approach_short"] = df["approach"].map({
        "portfolio_sort": "Portfolio",
        "cross_sectional_regression": "Regression",
    }).fillna(df["approach"])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Factor": r["factor"].upper(),
            "Approach": r["approach_short"],
            "Ann. Sharpe": f"{r['annualized_sharpe']:.3f}",
            "Ann. Return": f"{r['annualized_return_approx']*100:.2f}%",
            "t-stat": f"{r['t_stat']:.2f}",
            "p-value": f"{r['p_value']:.3f}",
            "Sig.": _significance_stars(r["p_value"]),
            "Avg IC": f"{r['avg_ic']:.3f}",
            "Rank IC": f"{r.get('avg_rank_ic', np.nan):.3f}" if "avg_rank_ic" in df.columns else "—",
        })
    out = pd.DataFrame(rows).sort_values(["Factor", "Approach"]).reset_index(drop=True)

    def cell_color(val, col, _row):
        if col == "Sig.":
            return "#A8E6A1" if val.strip() else None
        if col == "t-stat":
            try:
                if abs(float(val)) >= 1.96:
                    return "#D4F1D4"
            except Exception:
                pass
            return None
        return None

    _render_table_png(out, path,
                       title="T3 — Factor Risk Premia (portfolio-sort vs cross-sectional regression)",
                       caption="*** p<0.01, ** p<0.05, * p<0.10.  "
                               "Sharpe and t-stat over each factor's available months.",
                       cell_color_fn=cell_color,
                       fontsize=9, scale=(1, 1.5))


def table_walk_forward(wf: pd.DataFrame, path: Path = None) -> None:
    print("  Table T4: walk-forward train->test")
    if path is None:
        path = core.FIGURES_DIR / "T4_walk_forward.png"
    if wf.empty:
        return
    df = wf.copy()
    df["Train Sharpe (≤2020)"] = df["train_sharpe_to_2020"].map(lambda x: f"{x:.2f}")
    df["Test Sharpe (2021+)"] = df["test_sharpe_2021_2026"].map(lambda x: f"{x:.2f}")
    df["Test CumRet"] = df["test_cumulative_return_2021_2026"].map(lambda x: f"{x*100:.1f}%")
    df["Test MaxDD"] = df["test_max_drawdown_2021_2026"].map(lambda x: f"{abs(x)*100:.1f}%")
    df["Selected by Train?"] = df["selected_by_train"].map({True: "★ Yes", False: ""})
    df["Strategy"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    out = df[["Strategy", "Train Sharpe (≤2020)", "Test Sharpe (2021+)",
              "Test CumRet", "Test MaxDD", "Selected by Train?"]]
    out = out.sort_values("Train Sharpe (≤2020)", ascending=False).reset_index(drop=True)

    def cell_color(val, col, _row):
        if col == "Selected by Train?" and "Yes" in val:
            return "#A8E6A1"
        if col in ("Train Sharpe (≤2020)", "Test Sharpe (2021+)"):
            return _color_by_sharpe(val)
        return None

    _render_table_png(out, path,
                       title="T4 — Walk-Forward Train → Test (train ≤ 2020-12, test 2021 → May 2026)",
                       caption="Strategy is selected on train-only Sharpe; test Sharpe is reported, "
                               "not optimized over.",
                       cell_color_fn=cell_color,
                       fontsize=10, scale=(1, 1.5))


def table_multi_comparison(stepm: pd.DataFrame, spa: pd.DataFrame, path: Path = None) -> None:
    print("  Table T5: multi-comparison robustness")
    if path is None:
        path = core.FIGURES_DIR / "T5_multi_comparison_robustness.png"
    if stepm.empty and spa.empty:
        return
    rows = []
    if not spa.empty:
        s = spa.iloc[0]
        rows.append({"Test": "Hansen SPA (overall)", "Detail": "Any of the 8 beat ^GSPC by chance?",
                     "Statistic": f"p_consistent = {s.get('p_value_consistent', np.nan):.4f}",
                     "Reject H0?": "Yes" if s.get("p_value_consistent", 1) < 0.05 else "No"})
    if not stepm.empty:
        for _, r in stepm.sort_values("annualized_excess_return_approx", ascending=False).iterrows():
            rows.append({
                "Test": f"Romano-Wolf StepM — {STRATEGY_LABELS.get(r['strategy'], r['strategy'])}",
                "Detail": "Per-strategy step-down test vs ^GSPC",
                "Statistic": f"Ann. excess = {r['annualized_excess_return_approx']*100:+.2f}%",
                "Reject H0?": "Yes" if r["superior_to_benchmark_after_stepm"] else "No",
            })
    out = pd.DataFrame(rows)

    def cell_color(val, col, _row):
        if col == "Reject H0?":
            return "#A8E6A1" if val == "Yes" else "#F5C2C2"
        return None

    _render_table_png(out, path,
                       title="T5 — Multi-Comparison Robustness (Hansen SPA + Romano-Wolf StepM)",
                       caption="10,000 stationary block-bootstrap reps, 6-month blocks.  "
                               "Both tests correct for the 8 strategies we tried before reporting the best.",
                       cell_color_fn=cell_color,
                       fontsize=9, scale=(1, 1.5))


def table_imp4_vs_imp6(metrics: pd.DataFrame, all_curves: pd.DataFrame,
                         stepm: pd.DataFrame, mc: pd.DataFrame,
                         path: Path = None) -> None:
    print("  Table T6: imp 4 vs imp 6 head-to-head")
    if path is None:
        path = core.FIGURES_DIR / "T6_imp4_vs_imp6.png"
    if metrics.empty:
        return
    m = metrics.set_index("strategy")
    if "improved_4" not in m.index or "improved_6" not in m.index:
        return
    trades4 = _load_trades("improved_4")
    trades6 = _load_trades("improved_6")
    cov = _load_hzz_coverage()
    sw = stepm.set_index("strategy")["superior_to_benchmark_after_stepm"] if not stepm.empty else pd.Series(dtype=bool)
    mc_p = mc.set_index("strategy")["p_value"] if not mc.empty else pd.Series(dtype=float)

    def fmt(v, spec="{:.2f}"):
        try:
            return spec.format(float(v))
        except Exception:
            return "—"

    def fmt_signal_start(label, trades_df):
        if trades_df.empty:
            return "—"
        return str(trades_df["date"].min().date())

    def winner(v4, v6, higher_better=True):
        try:
            v4, v6 = float(v4), float(v6)
            if not (np.isfinite(v4) and np.isfinite(v6)):
                return ""
            if v4 == v6:
                return "tie"
            if higher_better:
                return "Imp 4" if v4 > v6 else "Imp 6"
            return "Imp 4" if v4 < v6 else "Imp 6"
        except Exception:
            return ""

    rows = []
    sharpe4, sharpe6 = m.loc["improved_4", "annualized_sharpe"], m.loc["improved_6", "annualized_sharpe"]
    rows.append({"Metric": "Annualized Sharpe", "Imp 4": fmt(sharpe4),
                 "Imp 6": fmt(sharpe6),
                 "Δ (4−6)": fmt(sharpe4 - sharpe6, "{:+.2f}"),
                 "Winner": winner(sharpe4, sharpe6)})
    ret4, ret6 = m.loc["improved_4", "annualized_return_approx"] * 100, m.loc["improved_6", "annualized_return_approx"] * 100
    rows.append({"Metric": "Annualized Return", "Imp 4": fmt(ret4, "{:.2f}%"),
                 "Imp 6": fmt(ret6, "{:.2f}%"),
                 "Δ (4−6)": fmt(ret4 - ret6, "{:+.2f} pp"),
                 "Winner": winner(ret4, ret6)})
    dd4, dd6 = abs(m.loc["improved_4", "max_drawdown"]) * 100, abs(m.loc["improved_6", "max_drawdown"]) * 100
    rows.append({"Metric": "Max Drawdown",       "Imp 4": fmt(dd4, "{:.1f}%"),
                 "Imp 6": fmt(dd6, "{:.1f}%"),
                 "Δ (4−6)": fmt(dd4 - dd6, "{:+.1f} pp"),
                 "Winner": winner(dd4, dd6, higher_better=False)})
    cr4, cr6 = m.loc["improved_4", "cumulative_return"] * 100, m.loc["improved_6", "cumulative_return"] * 100
    rows.append({"Metric": "Cumulative Return",  "Imp 4": fmt(cr4, "{:.1f}%"),
                 "Imp 6": fmt(cr6, "{:.1f}%"),
                 "Δ (4−6)": fmt(cr4 - cr6, "{:+.1f} pp"),
                 "Winner": winner(cr4, cr6)})
    fe4, fe6 = m.loc["improved_4", "final_equity"] / 1e6, m.loc["improved_6", "final_equity"] / 1e6
    rows.append({"Metric": "Final Equity",       "Imp 4": fmt(fe4, "${:.2f}M"),
                 "Imp 6": fmt(fe6, "${:.2f}M"),
                 "Δ (4−6)": fmt(fe4 - fe6, "{:+.2f}"),
                 "Winner": winner(fe4, fe6)})
    n4, n6 = len(trades4), len(trades6)
    rows.append({"Metric": "# Trades (full)",    "Imp 4": f"{n4}",
                 "Imp 6": f"{n6}",
                 "Δ (4−6)": f"{n4 - n6:+}",
                 "Winner": winner(n4, n6, higher_better=False)})
    bl4 = trades4["bar_len"].mean() if not trades4.empty else np.nan
    bl6 = trades6["bar_len"].mean() if not trades6.empty else np.nan
    rows.append({"Metric": "Avg hold (days)",    "Imp 4": fmt(bl4, "{:.1f}"),
                 "Imp 6": fmt(bl6, "{:.1f}"),
                 "Δ (4−6)": fmt(bl4 - bl6, "{:+.1f}"),
                 "Winner": winner(bl4, bl6)})
    hr4 = _strategy_hit_rate(all_curves, "improved_4")
    hr6 = _strategy_hit_rate(all_curves, "improved_6")
    rows.append({"Metric": "Hit-rate (%)",       "Imp 4": fmt(hr4, "{:.1f}%"),
                 "Imp 6": fmt(hr6, "{:.1f}%"),
                 "Δ (4−6)": fmt(hr4 - hr6, "{:+.1f} pp"),
                 "Winner": winner(hr4, hr6)})
    rows.append({"Metric": "Signal start date",
                 "Imp 4": fmt_signal_start("improved_4", trades4),
                 "Imp 6": fmt_signal_start("improved_6", trades6),
                 "Δ (4−6)": "earlier = more data",
                 "Winner": ""})
    if not cov.empty:
        eval_cov = cov[cov["month"] >= pd.Timestamp("2016-05-31")]
        avg_cov = (eval_cov["n_trend_hzz"] / eval_cov["n_eligible"].clip(lower=1)).mean() * 100
        rows.append({"Metric": "HZZ signal coverage (avg)",
                     "Imp 4": "100 % (uses expanding ^GSPC reg.)",
                     "Imp 6": fmt(avg_cov, "{:.0f}%"),
                     "Δ (4−6)": fmt(100 - avg_cov, "{:+.0f} pp"),
                     "Winner": "Imp 4"})
    p4 = mc_p.get("improved_4", np.nan) if "improved_4" in mc_p.index else np.nan
    p6 = mc_p.get("improved_6", np.nan) if "improved_6" in mc_p.index else np.nan
    rows.append({"Metric": "MC p-value (raw)",
                 "Imp 4": fmt(p4, "{:.3f}"),
                 "Imp 6": fmt(p6, "{:.3f}"),
                 "Δ (4−6)": fmt(p4 - p6, "{:+.3f}"),
                 "Winner": winner(p4, p6, higher_better=False)})
    rw4 = "Yes" if sw.get("improved_4", False) else "No"
    rw6 = "Yes" if sw.get("improved_6", False) else "No"
    rows.append({"Metric": "Romano-Wolf vs ^GSPC", "Imp 4": rw4, "Imp 6": rw6,
                 "Δ (4−6)": "", "Winner": "tie" if rw4 == rw6 else ("Imp 4" if rw4 == "Yes" else "Imp 6")})

    out = pd.DataFrame(rows)

    def cell_color(val, col, _row):
        if col == "Winner":
            if val == "Imp 4":
                return "#FDD0CC"   # imp 4 = red
            if val == "Imp 6":
                return "#CCDDF0"   # imp 6 = blue
            if val == "tie":
                return "#EEEEEE"
        return None

    _render_table_png(out, path,
                       title="T6 — Imp 4 vs Imp 6 (Why Imp 4 Wins)",
                       caption="Imp 4 uses the expanding-^GSPC predictive trend; "
                               "Imp 6 uses the Han-Zhou-Zhu cross-sectional MA regression. "
                               "Both share top-10 + stop/take execution. "
                               "Imp 4 wins on signal coverage and lower churn — "
                               "neither beats ^GSPC after Romano-Wolf correction.",
                       cell_color_fn=cell_color,
                       fontsize=9, scale=(1, 1.5))


def table_imp4_vs_imp8(metrics: pd.DataFrame, all_curves: pd.DataFrame,
                         stepm: pd.DataFrame, mc: pd.DataFrame,
                         path: Path = None) -> None:
    print("  Table T7: imp 4 vs imp 8 co-winners")
    if path is None:
        path = core.FIGURES_DIR / "T7_imp4_vs_imp8.png"
    if metrics.empty:
        return
    m = metrics.set_index("strategy")
    if "improved_4" not in m.index or "improved_8" not in m.index:
        return
    trades4, trades8 = _load_imp8_holdings_or_metrics()
    sw = stepm.set_index("strategy")["superior_to_benchmark_after_stepm"] if not stepm.empty else pd.Series(dtype=bool)
    mc_p = mc.set_index("strategy")["p_value"] if not mc.empty else pd.Series(dtype=float)

    def fmt(v, spec="{:.2f}"):
        try:
            return spec.format(float(v))
        except Exception:
            return "—"

    def winner(v4, v8, higher_better=True):
        try:
            v4, v8 = float(v4), float(v8)
            if not (np.isfinite(v4) and np.isfinite(v8)):
                return ""
            if v4 == v8:
                return "tie"
            if higher_better:
                return "Imp 4" if v4 > v8 else "Imp 8"
            return "Imp 4" if v4 < v8 else "Imp 8"
        except Exception:
            return ""

    rows = []
    rows.append({"Metric": "Annualized Sharpe", "Imp 4": fmt(m.loc["improved_4", "annualized_sharpe"]),
                 "Imp 8": fmt(m.loc["improved_8", "annualized_sharpe"]),
                 "Δ (4−8)": fmt(m.loc["improved_4", "annualized_sharpe"] - m.loc["improved_8", "annualized_sharpe"], "{:+.2f}"),
                 "Winner": winner(m.loc["improved_4", "annualized_sharpe"], m.loc["improved_8", "annualized_sharpe"])})
    rows.append({"Metric": "Annualized Return",
                 "Imp 4": fmt(m.loc["improved_4", "annualized_return_approx"] * 100, "{:.2f}%"),
                 "Imp 8": fmt(m.loc["improved_8", "annualized_return_approx"] * 100, "{:.2f}%"),
                 "Δ (4−8)": fmt((m.loc["improved_4", "annualized_return_approx"] - m.loc["improved_8", "annualized_return_approx"]) * 100, "{:+.2f} pp"),
                 "Winner": winner(m.loc["improved_4", "annualized_return_approx"], m.loc["improved_8", "annualized_return_approx"])})
    rows.append({"Metric": "Max Drawdown",
                 "Imp 4": fmt(abs(m.loc["improved_4", "max_drawdown"]) * 100, "{:.1f}%"),
                 "Imp 8": fmt(abs(m.loc["improved_8", "max_drawdown"]) * 100, "{:.1f}%"),
                 "Δ (4−8)": fmt((abs(m.loc["improved_4", "max_drawdown"]) - abs(m.loc["improved_8", "max_drawdown"])) * 100, "{:+.1f} pp"),
                 "Winner": winner(abs(m.loc["improved_4", "max_drawdown"]), abs(m.loc["improved_8", "max_drawdown"]), higher_better=False)})
    rows.append({"Metric": "Cumulative Return",
                 "Imp 4": fmt(m.loc["improved_4", "cumulative_return"] * 100, "{:.1f}%"),
                 "Imp 8": fmt(m.loc["improved_8", "cumulative_return"] * 100, "{:.1f}%"),
                 "Δ (4−8)": fmt((m.loc["improved_4", "cumulative_return"] - m.loc["improved_8", "cumulative_return"]) * 100, "{:+.1f} pp"),
                 "Winner": winner(m.loc["improved_4", "cumulative_return"], m.loc["improved_8", "cumulative_return"])})
    rows.append({"Metric": "Final Equity",
                 "Imp 4": fmt(m.loc["improved_4", "final_equity"] / 1e6, "${:.2f}M"),
                 "Imp 8": fmt(m.loc["improved_8", "final_equity"] / 1e6, "${:.2f}M"),
                 "Δ (4−8)": fmt((m.loc["improved_4", "final_equity"] - m.loc["improved_8", "final_equity"]) / 1e6, "{:+.2f} M"),
                 "Winner": winner(m.loc["improved_4", "final_equity"], m.loc["improved_8", "final_equity"])})
    avg_pos4 = m.loc["improved_4", "avg_positions"] if "avg_positions" in m.columns else np.nan
    avg_pos8 = m.loc["improved_8", "avg_positions"] if "avg_positions" in m.columns else np.nan
    rows.append({"Metric": "Avg # positions", "Imp 4": fmt(avg_pos4, "{:.1f}"),
                 "Imp 8": fmt(avg_pos8, "{:.1f}"),
                 "Δ (4−8)": fmt(avg_pos4 - avg_pos8, "{:+.1f}"),
                 "Winner": "—"})
    n4 = len(trades4) if not trades4.empty else 0
    n8 = len(trades8) if not trades8.empty else 0
    rows.append({"Metric": "# Trades (full)", "Imp 4": f"{n4}", "Imp 8": f"{n8}",
                 "Δ (4−8)": f"{n4 - n8:+}",
                 "Winner": winner(n4, n8, higher_better=False)})
    hr4 = _strategy_hit_rate(all_curves, "improved_4")
    hr8 = _strategy_hit_rate(all_curves, "improved_8")
    rows.append({"Metric": "Hit-rate (%)", "Imp 4": fmt(hr4, "{:.1f}%"),
                 "Imp 8": fmt(hr8, "{:.1f}%"),
                 "Δ (4−8)": fmt(hr4 - hr8, "{:+.1f} pp"),
                 "Winner": winner(hr4, hr8)})
    rows.append({"Metric": "Sizing rule",
                 "Imp 4": "FixedCash $100k / position (top-10)",
                 "Imp 8": "Equity %  5 % per position (top-20)",
                 "Δ (4−8)": "different by design",
                 "Winner": "—"})
    p4 = mc_p.get("improved_4", np.nan) if "improved_4" in mc_p.index else np.nan
    p8 = mc_p.get("improved_8", np.nan) if "improved_8" in mc_p.index else np.nan
    rows.append({"Metric": "MC p-value (raw)",
                 "Imp 4": fmt(p4, "{:.3f}"),
                 "Imp 8": fmt(p8, "{:.3f}"),
                 "Δ (4−8)": fmt(p4 - p8, "{:+.3f}"),
                 "Winner": winner(p4, p8, higher_better=False)})
    rw4 = "Yes" if sw.get("improved_4", False) else "No"
    rw8 = "Yes" if sw.get("improved_8", False) else "No"
    rows.append({"Metric": "Romano-Wolf vs ^GSPC", "Imp 4": rw4, "Imp 8": rw8,
                 "Δ (4−8)": "", "Winner": "tie" if rw4 == rw8 else ("Imp 4" if rw4 == "Yes" else "Imp 8")})

    out = pd.DataFrame(rows)

    def cell_color(val, col, _row):
        if col == "Winner":
            if val == "Imp 4":
                return "#FDD0CC"   # red tint
            if val == "Imp 8":
                return "#FFE9B0"   # gold tint
            if val == "tie":
                return "#EEEEEE"
        return None

    _render_table_png(out, path,
                       title="T7 — Imp 4 vs Imp 8: Co-Winners (Risk-Adjusted vs Wealth-Max)",
                       caption="Imp 4 wins on Sharpe and MaxDD (better risk-adjusted profile).  "
                               "Imp 8 wins on absolute return and final equity (more capital growth).  "
                               "Both fail Romano-Wolf vs ^GSPC after multi-comparison correction — "
                               "this is the honest finding.",
                       cell_color_fn=cell_color,
                       fontsize=9, scale=(1, 1.5))


# ---------------------------------------------------------------------------
# Loader for FMP summary + picked-date + Hansen / StepM
# ---------------------------------------------------------------------------

def load_fmp_summary() -> pd.DataFrame:
    p = core.FMP_RESULTS_DIR / "fmp_performance_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_picked_date_weights() -> pd.DataFrame:
    p = core.FMP_RESULTS_DIR / "selected_date_fmp_weight_comparison.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["month"])
    return pd.DataFrame()


def load_hansen_spa() -> pd.DataFrame:
    p = core.RESULTS_DIR / "robustness" / "hansen_spa_results.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_stepm() -> pd.DataFrame:
    p = core.RESULTS_DIR / "robustness" / "romano_wolf_stepm_results.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    core.ensure_dirs()
    print("Building presentation figures and tables (composite deck)...")

    # Load data
    all_curves = load_all_curves()
    index_monthly = load_index_monthly()
    metrics = load_all_metrics()
    wf = load_walk_forward()
    mc = load_monte_carlo_agg()
    fmp_portfolio = load_fmp_portfolio_returns()
    fmp_regression = load_fmp_regression_returns()
    factor_ic = load_factor_ic()
    fmp_summary = load_fmp_summary()
    picked = load_picked_date_weights()
    imp4_grid = load_improved4_grid()
    imp7_vector_curves = load_improved7_vector_curves()
    imp7_yearly_drag = load_improved7_yearly_drag()
    spa = load_hansen_spa()
    stepm = load_stepm()

    print(f"\nLoaded {all_curves['strategy'].nunique()} strategies, "
          f"{len(all_curves)} curve rows\n")

    imp3_weights = load_improved3_weights()

    # --- Composite figures (1-10) ---
    fig_factor_analysis_panel(fmp_portfolio, fmp_regression, factor_ic, fmp_summary)
    fig_fmp_picked_date_weights(picked)
    fig_strategy_ladder(all_curves, index_monthly)
    fig_risk_return_scoreboard(metrics, index_monthly)
    fig_walk_forward_polished(wf)
    fig_imp4_deep_dive(all_curves, index_monthly)
    fig_robustness_verdict_forest(all_curves, index_monthly, stepm)
    fig_bootstrap_violins(metrics, index_monthly)
    fig_cost_sensitivity_panel(imp7_vector_curves, imp7_yearly_drag)
    fig_research_diagnostics(imp4_grid)

    # --- Why-it-wins figures (11-15) + imp 3 weight evolution ---
    fig_imp4_vs_imp6_panel(all_curves, index_monthly, metrics)
    fig_improvement_waterfall(metrics)
    fig_imp5_regime_ribbon()
    fig_trend_regression_survivors()
    fig_imp4_vs_imp8_panel(all_curves, index_monthly, metrics)
    fig_factor_weights(imp3_weights)  # now produces figures/factor_weight_evolution_imp3.png (loader fix)

    # --- Tables (T1-T7) ---
    table_universe_summary()
    table_strategy_scoreboard(metrics, mc, stepm)
    table_factor_premia(fmp_summary)
    table_walk_forward(wf)
    table_multi_comparison(stepm, spa)
    table_imp4_vs_imp6(metrics, all_curves, stepm, mc)
    table_imp4_vs_imp8(metrics, all_curves, stepm, mc)

    # --- Comparison-results tables (CSV side-effects retained) ---
    make_hit_rate_table(all_curves)
    make_tail_risk_table(all_curves)
    make_best_worst_months_table(all_curves)

    print("\n\nDeck-ready outputs:")
    figures = sorted(core.FIGURES_DIR.glob("*.png"))
    print(f"Total figures in figures/: {len(figures)}")
    for f in figures:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
