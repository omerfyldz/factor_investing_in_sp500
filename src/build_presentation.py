"""
Build an improved SP500 Factor Investing presentation.

Fixes from original slides:
- Slide 13: Walk-forward and benchmark alpha tables now cover ALL 8 strategies (not just base + 1-3)
- Slide 14: Alpha and walk-forward tables populated with real data for all 8 strategies
- Slide 18: Hansen SPA / Romano-Wolf IS implemented — moved from Future Work to Accomplished
- Slide 9: Full headline results table with all 8 strategies
- General: More professional formatting, correct numbers throughout
"""
from __future__ import annotations

import sys
import io
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
PRESENTATION_DIR = ROOT / "presentation"

# ── Load results ─────────────────────────────────────────────────────────────

metrics   = pd.read_csv(COMPARISON_DIR / "all_strategies_metrics.csv")
wf        = pd.read_csv(COMPARISON_DIR / "all_strategies_walk_forward.csv")
benchmark = pd.read_csv(COMPARISON_DIR / "all_strategies_benchmark.csv")
mc        = pd.read_csv(COMPARISON_DIR / "all_strategies_monte_carlo.csv")
spa       = pd.read_csv(ROBUSTNESS_DIR / "hansen_spa_results.csv")

STRATEGY_LABELS = {
    "base":       "Base (full-sample, look-ahead)",
    "improved_1": "Imp 1 (expanding trend)",
    "improved_2": "Imp 2 (+stops 10/20%)",
    "improved_3": "Imp 3 (+dyn IC weights) -",
    "improved_4": "Imp 4 (walk-fwd 5/30%) [sel]",
    "improved_5": "Imp 5 (regime filter) -",
    "improved_6": "Imp 6 (HZZ trend)",
    "improved_8": "Imp 8 (eq-wt top-20)",
}

ORDER = ["base","improved_1","improved_2","improved_3","improved_4","improved_5","improved_6","improved_8"]

# ── Colours ───────────────────────────────────────────────────────────────────
DARK  = "#0d1117"
DARK2 = "#161b22"
ACCENT = "#2ea043"   # green
GOLD   = "#f0a500"
RED    = "#da3633"
BLUE   = "#1f6feb"
LIGHT  = "#f0f6fc"
GREY   = "#8b949e"
WHITE  = "#ffffff"

def slide_fig(w=13.333, h=7.5):
    fig = plt.figure(figsize=(w, h), facecolor=DARK)
    return fig

def header(fig, title, subtitle="", x=0.06, y_title=0.91, fontsize=26):
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor(DARK)
    ax.axis("off")
    # Accent bar
    ax.add_patch(mpatches.FancyBboxPatch((x-0.005, y_title-0.005), 0.005, 0.07,
        boxstyle="square,pad=0", facecolor=ACCENT))
    ax.text(x+0.01, y_title+0.045, title, color=WHITE, fontsize=fontsize,
            fontweight="bold", va="top")
    if subtitle:
        ax.text(x+0.01, y_title, subtitle, color=GREY, fontsize=12,
                va="top")
    return ax

def footer(ax, source=""):
    if source:
        ax.text(0.06, 0.025, f"Source: {source}", color=GREY, fontsize=8,
                va="bottom")

def add_table(ax, df, col_widths, row_h=0.055, start_y=0.72, x0=0.05,
              highlight_rows=None, grey_rows=None):
    """Draw a simple table on ax (already transAxes coords)."""
    cols = list(df.columns)
    col_x = [x0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Header row
    for j, (cx, col) in enumerate(zip(col_x, cols)):
        ax.text(cx, start_y + row_h*0.5, col, color=GREY, fontsize=8.5,
                fontweight="bold", va="center")
    ax.axhline(start_y, color=GREY, linewidth=0.5, xmin=x0, xmax=x0+sum(col_widths))

    for i, row in enumerate(df.itertuples(index=False)):
        y = start_y - (i+1)*row_h
        bg = None
        if highlight_rows and i in highlight_rows:
            bg = ACCENT
        elif grey_rows and i in grey_rows:
            bg = "#1e2a1e"
        if bg:
            ax.add_patch(mpatches.FancyBboxPatch((x0-0.005, y+0.005), sum(col_widths)+0.01,
                row_h-0.005, boxstyle="round,pad=0.002",
                facecolor=bg, alpha=0.25, transform=ax.transAxes, zorder=0))
        for j, (cx, val) in enumerate(zip(col_x, row)):
            color = LIGHT if (highlight_rows and i in highlight_rows) else WHITE
            ax.text(cx, y + row_h*0.4, str(val), color=color, fontsize=8,
                    va="center")

def save_slide(pdf, fig):
    pdf.savefig(fig, facecolor=DARK, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1: Title
# ─────────────────────────────────────────────────────────────────────────────
def slide_title(pdf):
    fig = slide_fig()
    ax = fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_facecolor(DARK)
    # Big accent stripe left
    ax.add_patch(mpatches.FancyBboxPatch((0.0, 0.0), 0.008, 1.0,
        boxstyle="square,pad=0", facecolor=ACCENT))
    ax.text(0.06, 0.72, "S&P 500 FACTOR INVESTING", color=WHITE, fontsize=32,
            fontweight="bold", va="center")
    ax.text(0.06, 0.60, "A Multi-Strategy Quantitative Research Project",
            color=GOLD, fontsize=18, va="center")
    ax.axhline(0.55, color=GREY, linewidth=0.5, xmin=0.05, xmax=0.95)
    bullets = [
        "Eight systematic strategies built on quality, value, momentum, and trend factors",
        "One-change-at-a-time experimental ladder — every improvement is scientifically readable",
        "Backtrader simulation with native Stop/Limit OCO orders  ·  Zero commission baseline",
        "Full robustness battery: Monte Carlo, block bootstrap, walk-forward, Hansen SPA",
        "Data window: S&P 500 constituents, June 2006 – May 2026  ·  Eval window: May 2016 – May 2026",
        "Group participant: Ömer Faruk Yıldız",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.06, 0.50 - i*0.07, f"·  {b}", color=LIGHT, fontsize=11,
                va="center")
    ax.text(0.94, 0.03, "QUANTITATIVE RESEARCH PROJECT", color=GREY, fontsize=9,
            ha="right", va="bottom")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2: Paper Summary
# ─────────────────────────────────────────────────────────────────────────────
def slide_paper(pdf):
    fig = slide_fig()
    ax = header(fig, "PAPER SUMMARY",
                "Han, Zhou & Zhu (2016) — 'A Trend Factor' — Journal of Financial Economics")
    ax.text(0.06, 0.76, "Core Idea", color=GOLD, fontsize=13, fontweight="bold")
    paras = [
        ("The Trend Factor:", "Normalized moving averages across 11 horizons (3→1,000 trading days) capture\n"
         "short-term reversal, medium-term momentum, and long-term mean reversion in one signal."),
        ("Methodology:", "Each month, cross-sectional OLS regression of stock returns on 11 MA ratios;\n"
         "coefficient vector smoothed with 12-month trailing average → stock-specific predicted return."),
        ("Result in Paper:", "Top-minus-bottom long-short trend portfolio earns significant positive alpha;\n"
         "the factor is orthogonal to Fama-French factors."),
        ("Our Adaptation:", "We apply the same concept to a long-only S&P 500 universe and combine\n"
         "the trend signal with quality (ROE), value (P/E), and momentum (12m price return)."),
    ]
    for i, (title, text) in enumerate(paras):
        y = 0.68 - i*0.145
        ax.text(0.06, y, title, color=ACCENT, fontsize=10, fontweight="bold")
        ax.text(0.22, y, text, color=LIGHT, fontsize=10,
                transform=ax.transAxes, va="top")
    ax.text(0.06, 0.08, "Two Implementations:", color=GOLD, fontsize=10, fontweight="bold")
    ax.text(0.06, 0.04,
            "trend_z (base) — full-sample index regression (look-ahead biased)  ·  "
            "trend_expanding_z (imp 1–5, 7) — expanding window (honest)  ·  "
            "trend_hzz_z (imp 6) — paper-faithful cross-sectional",
            color=GREY, fontsize=9)
    footer(ax, "Han, Zhou & Zhu (2016); docs/PROJECT_REPORT.md §2")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3: Data & Signals
# ─────────────────────────────────────────────────────────────────────────────
def slide_data(pdf):
    fig = slide_fig()
    ax = header(fig, "DATA & SIGNALS", "Frozen CSVs — no live downloads during backtests")
    rows = [
        ("sp500_prices_long.csv", "Tiingo", "2,293,664 rows", "2006–2026", "Daily OHLCV adj. prices, 503 tickers"),
        ("sp500_fundamentals_daily_long.csv", "Tiingo", "2,295,967 rows", "2006–2026", "Point-in-time mkt cap, P/E, P/B"),
        ("sp500_fundamentals_statements_long.csv", "Tiingo", "3,459,358 rows", "2005–2026", "Statement data keyed by date_available (no look-ahead)"),
        ("sp500_index_yahoo.csv", "Yahoo ^GSPC", "5,030 rows", "2006–2026", "Frozen benchmark OHLCV"),
        ("sp500_constituents.csv", "Wikipedia", "503 tickers", "current", "GICS sector, date added"),
    ]
    df = pd.DataFrame(rows, columns=["File", "Source", "Rows", "Span", "Description"])
    table_ax = fig.add_axes([0.04, 0.52, 0.92, 0.28])
    table_ax.axis("off"); table_ax.set_facecolor(DARK)
    tbl = table_ax.table(cellText=df.values, colLabels=df.columns,
                          cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(DARK2 if row % 2 == 0 else DARK)
        cell.set_text_props(color=WHITE if row > 0 else GOLD)
        cell.set_edgecolor(GREY)
        cell.set_linewidth(0.3)
    # Factors
    ax.text(0.06, 0.45, "Four Factors (all month-end, winsorized, z-scored, lagged 1 month):",
            color=GOLD, fontsize=11, fontweight="bold")
    factor_bullets = [
        ("ROE (Quality)", "roe_z — latest filing from date_available join — no look-ahead"),
        ("–P/E (Value)", "pe_z — inverted so high z = cheap"),
        ("Momentum", "momentum_z — 12-month total price return"),
        ("Trend", "trend_z / trend_expanding_z / trend_hzz_z — see Paper slide"),
    ]
    for i, (name, desc) in enumerate(factor_bullets):
        ax.text(0.06, 0.38 - i*0.07, f"  ·  {name}:", color=ACCENT, fontsize=10,
                fontweight="bold")
        ax.text(0.24, 0.38 - i*0.07, desc, color=LIGHT, fontsize=10)
    ax.text(0.06, 0.07, "[!] Survivorship bias: universe = current S&P 500 — failed/removed names absent.",
            color=RED, fontsize=9)
    footer(ax, "data/raw/; project_core.py: build_factor_panel()")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4: Strategy Ladder
# ─────────────────────────────────────────────────────────────────────────────
def slide_ladder(pdf):
    fig = slide_fig()
    ax = header(fig, "STRATEGY LADDER — 8 VARIANTS",
                "One-change-at-a-time principle: each step modifies exactly one design dimension")
    rows = [
        ("#", "Strategy Name", "Foundation", "Change", "Verdict"),
        ("0", "base_equal_top10", "—", "Full-sample trend (look-ahead biased)", "[doc] Assignment baseline"),
        ("1", "improved_1_expanding_trend", "base", "Trend → expanding window (no look-ahead)", "[OK] Honest baseline"),
        ("2", "improved_2_stop_take", "imp 1", "Add 10% stop / 20% take-profit", "[OK] Risk mgmt layer"),
        ("3", "improved_3_dynamic_ic", "imp 2", "Rolling rank-IC weights (50% shrinkage)", "[X] Didn't help"),
        ("4", "improved_4_walkforward", "imp 2", "Walk-fwd stop/take → 5% / 30% (6×6 grid)", "[*] Best risk-managed"),
        ("5", "improved_5_regime_filter", "imp 4", "10m SMA regime filter (cash when below)", "[X] Hurt performance"),
        ("6", "improved_6_hzz_trend", "imp 4", "Trend → HZZ cross-sectional (paper-faithful)", "[OK] Paper faithful"),
        ("7", "improved_7_costs", "imp 4+6", "Time-varying cost sensitivity (not a strategy)", "[chart] Cost study"),
        ("8", "improved_8_equal_weight_top20", "imp 4", "Top-N → 20  +  sizing → 5% of equity (1/N)", "[wealth] Wealth-maximizer"),
    ]
    row_h = 0.066
    start_y = 0.80
    col_x   = [0.03, 0.065, 0.29, 0.46, 0.70]
    col_w   = [0.035, 0.22, 0.17, 0.24, 0.27]
    # Header
    for cx, col in zip(col_x, rows[0]):
        ax.text(cx, start_y, col, color=GOLD, fontsize=9, fontweight="bold")
    ax.axhline(start_y - 0.01, color=GREY, linewidth=0.4, xmin=0.03, xmax=0.97)
    highlight = {4, 8}  # imp4 and imp8
    reject    = {3, 5}
    grey_rows = {0}
    for i, row in enumerate(rows[1:]):
        y = start_y - (i+1)*row_h
        if i in highlight:
            ax.add_patch(mpatches.FancyBboxPatch((0.025, y+0.005), 0.95, row_h-0.01,
                boxstyle="round,pad=0.002", facecolor=ACCENT, alpha=0.12,
                transform=ax.transAxes, zorder=0))
        if i in reject:
            ax.add_patch(mpatches.FancyBboxPatch((0.025, y+0.005), 0.95, row_h-0.01,
                boxstyle="round,pad=0.002", facecolor=RED, alpha=0.08,
                transform=ax.transAxes, zorder=0))
        if i in grey_rows:
            ax.add_patch(mpatches.FancyBboxPatch((0.025, y+0.005), 0.95, row_h-0.01,
                boxstyle="round,pad=0.002", facecolor=GREY, alpha=0.10,
                transform=ax.transAxes, zorder=0))
        color = LIGHT if i in highlight else (GREY if i in grey_rows else WHITE)
        for cx, val in zip(col_x, row):
            ax.text(cx, y + row_h*0.35, val, color=color, fontsize=8.2,
                    va="center")
    ax.text(0.06, 0.04,
            "[*] = project finalist on risk-adjusted return   ·   [X] = documented failed experiment (retained for transparency)",
            color=GREY, fontsize=8.5)
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5: Execution model
# ─────────────────────────────────────────────────────────────────────────────
def slide_execution(pdf):
    fig = slide_fig()
    ax = header(fig, "EXECUTION MODEL & CODE ARCHITECTURE",
                "Backtrader physical broker simulation — two engines, three position sizers")
    # Left panel — settings
    settings = [
        ("Initial capital", "$1,000,000"),
        ("Rebalance frequency", "Monthly (signal at month-end → fill at next open)"),
        ("Entry / exit", "bt.Order.Market entries; bt.Order.Stop + bt.Order.Limit (OCO) exits"),
        ("Commission (baseline)", "0 bps  (Improved 7 adds time-varying costs)"),
        ("Long-only assertion", "Checked after every Backtrader run; aborts on any short position"),
        ("Sizing — imp 1–7", "FixedCashSizer: $100,000 per position"),
        ("Sizing — imp 8", "EquityPercentSizer: 5% of current equity per position (1/N for top-20)"),
        ("Stop/Take — imp 2–3", "10% stop / 20% take  (round numbers, not optimised)"),
        ("Stop/Take — imp 4–8", "5% stop / 30% take  (walk-forward selected from 6×6 grid)"),
    ]
    for i, (k, v) in enumerate(settings):
        y = 0.76 - i*0.067
        ax.text(0.04, y, k + ":", color=GOLD, fontsize=9.5, fontweight="bold")
        ax.text(0.30, y, v, color=LIGHT, fontsize=9.5)
    # Code snippet
    snippet = (
        "# StrategySpec — one dataclass per variant\n"
        "core.StrategySpec(\n"
        "    name = core.IMPROVED_8_STRATEGY_NAME,\n"
        "    weights = {'roe':1,'pe':1,'momentum':1,'trend':1},\n"
        "    top_n = 20,  trend_col = 'trend_expanding_z',\n"
        "    stop_loss = 0.05,  take_profit = 0.30,\n"
        "    sizing_method = 'percent_of_equity',\n"
        "    sizing_target_pct = 0.05\n"
        ")"
    )
    code_ax = fig.add_axes([0.60, 0.12, 0.37, 0.38])
    code_ax.set_facecolor(DARK2)
    code_ax.axis("off")
    code_ax.text(0.04, 0.92, snippet, color=ACCENT, fontsize=8, va="top",
                 fontfamily="monospace", transform=code_ax.transAxes)
    ax.text(0.60, 0.53, "Improved 8 spec — one config object:", color=GOLD,
            fontsize=9, fontweight="bold")
    ax.text(0.60, 0.11, "All 8 variants = StrategySpec + runner.\nDeclarative, testable, reproducible.",
            color=GREY, fontsize=8.5)
    footer(ax, "project_core.py: StrategySpec, run_backtrader_daily_stop_take()")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6: Methodology — Common Window Fix
# ─────────────────────────────────────────────────────────────────────────────
def slide_window_fix(pdf):
    fig = slide_fig()
    ax = header(fig, "KEY METHODOLOGY — COMMON EVALUATION WINDOW",
                "The most important methodological fix in this project")
    ax.text(0.06, 0.76, "Problem:", color=RED, fontsize=12, fontweight="bold")
    ax.text(0.06, 0.71,
            "Different strategies have different warmup periods:\n"
            "  · base uses full-sample trend_z → trades from 2010-05\n"
            "  · imp 1–5, 7 use trend_expanding_z → trades from 2016-05 (72-month index regression warmup)\n"
            "  · imp 6 uses trend_hzz_z → trades from 2011-05\n\n"
            "Original code computed Sharpe over all 240 months. Pre-warmup months = zero returns.\n"
            "This dilutes the mean more than the std, giving longer-warmup strategies artificially low Sharpes.",
            color=LIGHT, fontsize=10.5, transform=ax.transAxes, va="top")
    ax.text(0.06, 0.42, "Fix:", color=ACCENT, fontsize=12, fontweight="bold")
    ax.text(0.06, 0.37,
            "EVALUATION_START = 2016-05-31  (121 months through 2026-05-31)\n"
            "All Sharpes, drawdowns, Monte Carlo, bootstrap, walk-forward, and benchmark comparison\n"
            "restricted to this common window across all 8 strategies.",
            color=LIGHT, fontsize=10.5, transform=ax.transAxes, va="top")
    # Before/after table
    data = [
        ("base",       "1.13", "1.20", "+0.07"),
        ("improved_1", "0.82", "1.18", "+0.36 ← biggest gain"),
        ("improved_4", "0.80", "1.15", "+0.35"),
        ("improved_6", "0.81", "1.02", "+0.21"),
    ]
    df = pd.DataFrame(data, columns=["Strategy", "Old Sharpe (240m)", "Fixed Sharpe (121m)", "Change"])
    table_ax = fig.add_axes([0.06, 0.06, 0.50, 0.22])
    table_ax.axis("off"); table_ax.set_facecolor(DARK)
    tbl = table_ax.table(cellText=df.values, colLabels=df.columns,
                          cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(DARK2 if row % 2 == 0 else DARK)
        cell.set_text_props(color=WHITE if row > 0 else GOLD)
        cell.set_edgecolor(GREY); cell.set_linewidth(0.3)
    ax.text(0.60, 0.25,
            "Academically standard:\nEvery published factor paper\nrestricts cross-strategy comparison\nto a common evaluation window\n(Fama-French, AQR, HZZ paper itself).",
            color=GREY, fontsize=10)
    footer(ax, "project_core.py: metrics_over_evaluation_window(), filter_to_evaluation_window()")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7: Factor mimicking portfolios
# ─────────────────────────────────────────────────────────────────────────────
def slide_fmp(pdf):
    fig = slide_fig()
    ax = header(fig, "FACTOR ANALYSIS — WHICH SIGNALS WORK?",
                "Factor-mimicking portfolios: each factor in isolation (long-short, eval window)")
    # Table of factor stats
    fmp_data = [
        ("Trend (cross-section HZZ)", "0.39", "1.56", "0.022", "Strongest — dominates the composite"),
        ("Trend (index expanding)",   "0.36", "1.44", "0.022", "Strong — used in imp 1–5, 7"),
        ("Value (−P/E)",              "0.21", "0.92", "0.012", "Weak alone; reduces variance in combo"),
        ("Momentum (12m)",            "0.08", "0.34", "0.004", "Near-zero standalone; combo diversifier"),
        ("Quality (ROE)",             "−0.21","−0.92","0.004", "Negative alone on this sample!"),
    ]
    df = pd.DataFrame(fmp_data, columns=["Factor", "Ann. Sharpe", "t-stat", "Avg Rank IC", "Role in strategy"])
    table_ax = fig.add_axes([0.04, 0.52, 0.92, 0.30])
    table_ax.axis("off"); table_ax.set_facecolor(DARK)
    tbl = table_ax.table(cellText=df.values, colLabels=df.columns,
                          cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(DARK2 if row % 2 == 0 else DARK)
        cell.set_text_props(color=WHITE if row > 0 else GOLD)
        cell.set_edgecolor(GREY); cell.set_linewidth(0.3)
    # Bullet insights
    insights = [
        "Trend is the engine — the two-factor combo of trend+momentum alone explains most of the composite Sharpe.",
        "Improved 3 tried dynamic rank-IC weighting to let the data pick factor weights — result: no improvement (too little signal history).",
        "Negative ROE standalone Sharpe means quality is contrarian on this S&P 500 sample; it mainly reduces left-tail volatility.",
        "Composite Sharpe (>1.0) materially exceeds best single-factor Sharpe (0.39) — diversification benefit is real.",
    ]
    for i, txt in enumerate(insights):
        ax.text(0.06, 0.45 - i*0.08, f"·  {txt}", color=LIGHT, fontsize=10)
    footer(ax, "results/fmp_analysis/; project_core.py: make_fmp_returns(), summarize_fmps()")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8: IC Chart slide
# ─────────────────────────────────────────────────────────────────────────────
def slide_ic_chart(pdf):
    fig = slide_fig()
    ax = header(fig, "FACTOR PERFORMANCE — CHARTS", "Equity curves and IC summaries")
    # Insert existing figure if available
    img_path = FIGURES_DIR / "factor_portfolio_cumulative_returns.png"
    if img_path.exists():
        img_ax = fig.add_axes([0.04, 0.10, 0.44, 0.60])
        img_ax.imshow(plt.imread(img_path)); img_ax.axis("off")
    img2_path = FIGURES_DIR / "ic_rank_ic_summary.png"
    if img2_path.exists():
        img_ax2 = fig.add_axes([0.52, 0.10, 0.44, 0.60])
        img_ax2.imshow(plt.imread(img2_path)); img_ax2.axis("off")
    ax.text(0.06, 0.10, "Left: cumulative returns of factor-mimicking portfolios (long-short, each factor in isolation).",
            color=GREY, fontsize=9)
    ax.text(0.52, 0.10, "Right: rank-IC distribution across months — trend has the most consistently positive IC.",
            color=GREY, fontsize=9)
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9: Headline results table
# ─────────────────────────────────────────────────────────────────────────────
def slide_results_table(pdf):
    fig = slide_fig()
    ax = header(fig, "RESULTS — HEADLINE COMPARISON",
                "All 8 strategies · Eval window 2016-05 to 2026-05 · 121 months")
    # Build table from real data
    m = metrics.set_index("strategy")
    rows = []
    for s in ORDER:
        if s not in m.index: continue
        row = m.loc[s]
        rows.append([
            STRATEGY_LABELS.get(s, s),
            f"{row['annualized_sharpe']:.2f}",
            f"{row['annualized_return_approx']*100:.1f}%",
            f"{row['max_drawdown']*100:.1f}%",
            f"${row['final_equity']/1e6:.2f}M",
            f"{row['avg_positions']:.1f}",
        ])
    df = pd.DataFrame(rows, columns=["Strategy", "Vec Sharpe", "Ann Return", "Max DD", "Final $", "Avg Pos"])
    table_ax = fig.add_axes([0.04, 0.30, 0.92, 0.52])
    table_ax.axis("off"); table_ax.set_facecolor(DARK)
    tbl = table_ax.table(cellText=df.values, colLabels=df.columns,
                          cellLoc="left", loc="upper left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1a2a1a"); cell.set_text_props(color=GOLD)
        elif row == 5:  # imp4 (index 4 in ORDER = index 5 after header)
            cell.set_facecolor("#0d1f0d"); cell.set_text_props(color=LIGHT)
        elif row == 1:  # base
            cell.set_facecolor("#1a1a1a"); cell.set_text_props(color=GREY)
        else:
            cell.set_facecolor(DARK2 if row % 2 == 0 else DARK)
            cell.set_text_props(color=WHITE)
        cell.set_edgecolor(GREY); cell.set_linewidth(0.3)
    ax.text(0.04, 0.27,
            "[sel] Improved 4: best Sharpe (1.15) and shallowest drawdown (−7.4%) — walk-forward selected 5%/30% stops.",
            color=ACCENT, fontsize=9.5)
    ax.text(0.04, 0.22,
            "[wealth] Improved 8: highest final equity ($3.63M) via equal-weight compounding across ~17 names.",
            color=GOLD, fontsize=9.5)
    ax.text(0.04, 0.17,
            "[!] Base greyed — full-sample trend is look-ahead biased; Sharpe not directly comparable to others.",
            color=GREY, fontsize=9)
    ax.text(0.04, 0.12,
            "Note: vector final equity restated from $1M at eval start. Sharpes use eval window in all engines.",
            color=GREY, fontsize=8.5)
    footer(ax, "results/comparison/all_strategies_metrics.csv")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10: Performance figures
# ─────────────────────────────────────────────────────────────────────────────
def slide_equity_charts(pdf):
    fig = slide_fig()
    ax = header(fig, "RESULTS — EQUITY & DRAWDOWN CURVES", "All 8 strategies · eval window")
    img1 = FIGURES_DIR / "all_strategies_equity_curves.png"
    img2 = FIGURES_DIR / "all_strategies_drawdowns.png"
    if img1.exists():
        ia = fig.add_axes([0.02, 0.10, 0.49, 0.62])
        ia.imshow(plt.imread(img1)); ia.axis("off")
    if img2.exists():
        ia2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
        ia2.imshow(plt.imread(img2)); ia2.axis("off")
    ax.text(0.06, 0.09, "Left: equity curves (vector, rebased $1M at eval start). All strategies positive.", color=GREY, fontsize=9)
    ax.text(0.52, 0.09, "Right: drawdown from peak. Imp 4 shallowest; Imp 8 deepest (more capital deployed).", color=GREY, fontsize=9)
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11: Risk-return frontier
# ─────────────────────────────────────────────────────────────────────────────
def slide_risk_return(pdf):
    fig = slide_fig()
    ax = header(fig, "RESULTS — RISK-RETURN FRONTIER",
                "Sharpe vs Max Drawdown — bubble size ∝ final equity")
    img = FIGURES_DIR / "sharpe_vs_drawdown_scatter.png"
    if img.exists():
        ia = fig.add_axes([0.04, 0.10, 0.54, 0.65])
        ia.imshow(plt.imread(img)); ia.axis("off")
    # Key insight callout
    ax.text(0.63, 0.72, "Key Reading:", color=GOLD, fontsize=11, fontweight="bold")
    callouts = [
        ("Top-right corner", "= best risk-adjusted return"),
        ("Imp 4", "Highest Sharpe + shallowest DD → efficient corner"),
        ("Imp 8", "Moves down-left deliberately: trades Sharpe for absolute wealth"),
        ("Base", "Top-right only because of look-ahead bias in trend signal"),
        ("Imp 5, 3", "Regime filter and dynamic IC — both hurt; rejected"),
    ]
    for i, (label, desc) in enumerate(callouts):
        y = 0.63 - i*0.08
        ax.text(0.63, y, label + ":", color=ACCENT, fontsize=9.5, fontweight="bold")
        ax.text(0.79, y, desc, color=LIGHT, fontsize=9.5)
    footer(ax, "figures/sharpe_vs_drawdown_scatter.png; results/comparison/all_strategies_metrics.csv")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12: Walk-forward stop/take
# ─────────────────────────────────────────────────────────────────────────────
def slide_walkforward_stops(pdf):
    fig = slide_fig()
    ax = header(fig, "WHY IMPROVED 4 — WALK-FORWARD STOP/TAKE SELECTION",
                "Data-driven risk-exit calibration on training data only")
    img = FIGURES_DIR / "improved4_stop_take_sensitivity_heatmap.png"
    if img.exists():
        ia = fig.add_axes([0.04, 0.14, 0.50, 0.60])
        ia.imshow(plt.imread(img)); ia.axis("off")
    ax.text(0.58, 0.77, "Methodology:", color=GOLD, fontsize=11, fontweight="bold")
    points = [
        "6×6 grid of (stop%, take%) combinations",
        "Evaluated on training data ≤ 2020-12 only",
        "Selection criterion: highest train-period Sharpe",
        "Selected: 5% stop, 30% take-profit",
        "Out-of-sample test Sharpe: 1.02 (honest check)",
        "Result: drawdown drops to −7.4%; Sharpe rises to 1.15",
        "",
        "Important: the grid search itself consumes",
        "some of our 'multiple testing budget' — disclosed",
        "transparently in the Limitations section.",
    ]
    for i, p in enumerate(points):
        color = GREY if p.startswith("Important") or p == "" else LIGHT
        ax.text(0.58, 0.70 - i*0.065, ("  " if not p.startswith("Important") else "") + p,
                color=color, fontsize=9.5)
    footer(ax, "results/improved_strategy_4/; figures/improved4_stop_take_sensitivity_heatmap.png")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 13: Robustness battery overview
# ─────────────────────────────────────────────────────────────────────────────
def slide_robustness_overview(pdf):
    fig = slide_fig()
    ax = header(fig, "ROBUSTNESS — VALIDATION BATTERY",
                "Six independent lenses — each interrogates a different failure mode")
    tests = [
        ("Monte Carlo", "1,000 random portfolios / strategy from same eligible universe. p = fraction beating strategy Sharpe.",
         "base (0.013), imp_1 (0.019) clear p<0.05; imp_4 (0.088) borderline."),
        ("Block Bootstrap", "1,000 paths resampling 6-month blocks of the strategy's own returns.",
         "Characterises Sharpe and DD distribution without historical ordering."),
        ("Walk-Forward", "Train ≤ 2020-12, test 2021+. Select on train; report test.",
         "Imp_4 selected on train (1.29); imp_1 has highest test Sharpe (1.34)."),
        ("Benchmark Alpha", "OLS regression of excess return on ^GSPC (eval window).",
         "All 8 strategies: statistically significant alpha at 1% level. Negative beta."),
        ("Cost Sensitivity", "Time-varying commission + slippage 2006–2026. Three scenarios.",
         "Both imp_4 and imp_6 survive pessimistic costs; Sharpe drop only 0.03–0.07."),
        ("Hansen SPA + Romano-Wolf", "Multi-comparison correction across all 8 variants. 10,000 bootstrap reps.",
         "[OK] IMPLEMENTED — SPA consistent p = 0.674 (cannot reject no-outperformance after correction)."),
    ]
    for i, (name, method, result) in enumerate(tests):
        y = 0.76 - i*0.108
        ax.add_patch(mpatches.FancyBboxPatch((0.03, y-0.025), 0.93, 0.095,
            boxstyle="round,pad=0.003",
            facecolor="#1a2a1a" if i==5 else DARK2, alpha=0.8,
            transform=ax.transAxes, zorder=0))
        ax.text(0.05, y+0.048, f"{i+1}. {name}", color=GOLD if i==5 else ACCENT,
                fontsize=10, fontweight="bold")
        ax.text(0.28, y+0.048, method, color=LIGHT, fontsize=8.5)
        ax.text(0.28, y+0.010, "→  " + result, color=GREY, fontsize=8.3)
    footer(ax, "results/robustness/; results/comparison/all_strategies_*.csv")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 14: Alpha & Walk-forward tables — FULL 8 strategies
# ─────────────────────────────────────────────────────────────────────────────
def slide_alpha_walkforward(pdf):
    fig = slide_fig()
    ax = header(fig, "ROBUSTNESS — ALPHA & WALK-FORWARD",
                "All 8 strategies covered (not just base + improved 1–3)")
    # Alpha table
    ax.text(0.04, 0.76, "Benchmark Alpha vs ^GSPC (eval window, all 8 strategies):",
            color=GOLD, fontsize=10, fontweight="bold")
    bm = benchmark.copy()
    bm = bm.set_index("strategy").reindex(ORDER).reset_index()
    alpha_rows = []
    for _, row in bm.iterrows():
        s = row["strategy"]
        alpha_rows.append([
            STRATEGY_LABELS.get(s, s),
            f"{row['annualized_alpha_approx']*100:.2f}%",
            f"{row['alpha_t_stat']:.2f}",
            f"{row['alpha_p_value']:.4f}",
            f"{row['beta_to_sp500']:.3f}",
            "+ sig" if row['alpha_p_value'] < 0.05 else "—"
        ])
    df_alpha = pd.DataFrame(alpha_rows,
        columns=["Strategy", "Ann Alpha", "t-stat", "p-value", "Beta", "Sig."])
    ta = fig.add_axes([0.03, 0.44, 0.55, 0.30])
    ta.axis("off"); ta.set_facecolor(DARK)
    tbl = ta.table(cellText=df_alpha.values, colLabels=df_alpha.columns,
                    cellLoc="left", loc="upper left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(DARK2 if row % 2 == 0 else DARK)
        cell.set_text_props(color=GOLD if row == 0 else WHITE)
        cell.set_edgecolor(GREY); cell.set_linewidth(0.3)

    # Walk-forward table
    ax.text(0.62, 0.76, "Walk-Forward (train ≤ 2020-12, test 2021+):",
            color=GOLD, fontsize=10, fontweight="bold")
    wf_sorted = wf.sort_values("train_sharpe_to_2020", ascending=False)
    wf_rows = []
    for _, row in wf_sorted.iterrows():
        s = row["strategy"]
        sel = "[sel]" if row.get("selected_by_train", False) else ""
        wf_rows.append([
            STRATEGY_LABELS.get(s, s) + sel,
            f"{row['train_sharpe_to_2020']:.2f}",
            f"{row['test_sharpe_2021_2026']:.2f}",
            f"{row['test_cumulative_return_2021_2026']*100:.0f}%",
        ])
    df_wf = pd.DataFrame(wf_rows, columns=["Strategy", "Train Sharpe", "Test Sharpe", "Test Cum Ret"])
    tw = fig.add_axes([0.61, 0.44, 0.37, 0.30])
    tw.axis("off"); tw.set_facecolor(DARK)
    tbl2 = tw.table(cellText=df_wf.values, colLabels=df_wf.columns,
                     cellLoc="left", loc="upper left")
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(8.0)
    for (row, col), cell in tbl2.get_celld().items():
        cell.set_facecolor(DARK2 if row % 2 == 0 else DARK)
        cell.set_text_props(color=GOLD if row == 0 else WHITE)
        cell.set_edgecolor(GREY); cell.set_linewidth(0.3)

    # Insights
    ax.text(0.04, 0.40,
            "All 8 strategies produce statistically significant alpha vs ^GSPC at the 1% level. Negative beta = active factor tilt.",
            color=ACCENT, fontsize=9.5)
    ax.text(0.04, 0.34,
            "Walk-forward: Imp 4 selected on train (1.29); Imp 1 has highest out-of-sample test Sharpe (1.34). Single splits are noisy.",
            color=LIGHT, fontsize=9.5)
    ax.text(0.04, 0.28,
            "[!] Excess return p-values are unadjusted. Multi-comparison correction (Hansen SPA) on next slide.",
            color=GREY, fontsize=9)
    footer(ax, "results/comparison/all_strategies_benchmark.csv, all_strategies_walk_forward.csv")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 15: Monte Carlo p-values + Hansen SPA
# ─────────────────────────────────────────────────────────────────────────────
def slide_mc_spa(pdf):
    fig = slide_fig()
    ax = header(fig, "ROBUSTNESS — MONTE CARLO & MULTI-COMPARISON CORRECTION",
                "The honest truth about statistical significance")
    img = FIGURES_DIR / "monte_carlo_pvalue_comparison.png"
    if img.exists():
        ia = fig.add_axes([0.03, 0.20, 0.52, 0.52])
        ia.imshow(plt.imread(img)); ia.axis("off")

    ax.text(0.59, 0.77, "Raw MC p-values:", color=GOLD, fontsize=10, fontweight="bold")
    mc_s = mc.set_index("strategy").reindex(ORDER).reset_index()
    for i, row in mc_s.iterrows():
        s = row["strategy"]
        p = row.get("p_value", float("nan"))
        color = ACCENT if p < 0.05 else (GOLD if p < 0.10 else GREY)
        ax.text(0.59, 0.70 - i*0.062,
                f"  {STRATEGY_LABELS.get(s,s)[:22]:22s}  p = {p:.3f}",
                color=color, fontsize=8.5, fontfamily="monospace")

    # Hansen SPA box
    spa_p = float(spa["p_value_consistent"].iloc[0])
    ax.add_patch(mpatches.FancyBboxPatch((0.58, 0.11), 0.38, 0.12,
        boxstyle="round,pad=0.008", facecolor="#2a1a00", alpha=0.9))
    ax.text(0.60, 0.21, "Hansen (2005) SPA Test — ALL 8 STRATEGIES:",
            color=GOLD, fontsize=9, fontweight="bold")
    ax.text(0.60, 0.165,
            f"Consistent p-value = {spa_p:.4f}  →  CANNOT reject null",
            color=RED, fontsize=10, fontweight="bold")
    ax.text(0.60, 0.125,
            "Romano-Wolf StepM: no strategy survives at 5% FWER",
            color=LIGHT, fontsize=8.5)

    ax.text(0.04, 0.175, "Testing 8 variants at p<0.05:",
            color=GOLD, fontsize=10, fontweight="bold")
    ax.text(0.04, 0.13,
            "P(at least one false positive) = 1 − (1−0.05)⁸ ≈ 34%\n"
            "Raw p-values are biased downward by the search-space size.\n"
            "Only base and Imp 1 clear 0.05 individually — and even those\n"
            "are not significant after FWER correction.",
            color=LIGHT, fontsize=9)
    footer(ax, "results/robustness/hansen_spa_results.csv; src/run_multi_comparison_test.py")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 16: Cost sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def slide_costs(pdf):
    fig = slide_fig()
    ax = header(fig, "ROBUSTNESS — TRANSACTION-COST SENSITIVITY (IMPROVED 7)",
                "Time-varying commission + slippage, 2006–2026 · Three scenarios")
    img = FIGURES_DIR / "cost_drag_attribution_imp7.png"
    if img.exists():
        ia = fig.add_axes([0.03, 0.14, 0.56, 0.58])
        ia.imshow(plt.imread(img)); ia.axis("off")

    ax.text(0.63, 0.76, "Cost Schedule (central):", color=GOLD, fontsize=10,
            fontweight="bold")
    schedule = [
        ("2006", "7.0 bps comm", "6.0 bps slip", "26 bps RT"),
        ("2008", "6.0 bps comm", "8.0 bps slip", "28 bps (crisis)"),
        ("2014", "2.5 bps comm", "2.0 bps slip", " 9 bps RT"),
        ("2020", "0.7 bps comm", "2.5 bps slip", " 6.4 (Covid)"),
        ("2026", "0.5 bps comm", "1.0 bps slip", " 3.0 bps RT"),
    ]
    for i, (yr, comm, slip, rt) in enumerate(schedule):
        y = 0.69 - i*0.065
        ax.text(0.63, y, f"{yr}:  {comm}  /  {slip}  →  {rt}", color=LIGHT,
                fontsize=9, fontfamily="monospace")

    ax.text(0.63, 0.37, "Results:", color=GOLD, fontsize=10, fontweight="bold")
    result_rows = [
        ("Scenario", "Imp 4 Sharpe", "Imp 6 Sharpe"),
        ("Zero cost", "1.150", "1.023"),
        ("Central",   "1.118", "0.985"),
        ("Pessimistic","1.086", "0.948"),
    ]
    for i, row in enumerate(result_rows):
        y = 0.31 - i*0.065
        colors = [GOLD, LIGHT, LIGHT]
        for j, (val, cx) in enumerate(zip(row, [0.63, 0.77, 0.88])):
            ax.text(cx, y, val, color=colors[j] if i==0 else LIGHT, fontsize=9.5,
                    fontweight="bold" if i==0 else "normal")

    ax.text(0.63, 0.10,
            "Both finalists survive pessimistic costs.\nSmall Sharpe drop (~0.03–0.07) because\nmonthly turnover is modest and trade size\nis tiny vs S&P 500 average daily volume.",
            color=GREY, fontsize=9)
    footer(ax, "results/improved_strategy_7/; docs/SIZING_AND_MARGIN.md")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 17: Interpretation & Key Findings
# ─────────────────────────────────────────────────────────────────────────────
def slide_interpretation(pdf):
    fig = slide_fig()
    ax = header(fig, "WHAT WE LEARNED", "Interpretation of results")
    findings = [
        (ACCENT, "Improved 4 — the Sharpe story",
         "Best risk-managed variant. Sharpe 1.15 (vector) / 1.40 (Backtrader). Max DD −7.4%.\n"
         "Walk-forward selected 5%/30% stops on training data through 2020. Concentrated (8–9 names)."),
        (GOLD, "Improved 8 — the wealth story",
         "Highest vector final equity ($3.63M). Equal-weight 1/N across 20 names compounds dynamically.\n"
         "Pays for it with lower Sharpe (1.03) and deeper drawdown (−10.9%). Industry analog: RSP ETF."),
        (BLUE, "Trend is the dominant factor",
         "FMP Sharpe ~0.36–0.39; strongest rank IC (0.022). Quality and value mainly reduce variance.\n"
         "The composite materially beats any single factor (diversification benefit is real)."),
        (GREY, "The common-window fix overturned a key finding",
         "'Improved 6 is best' was a dilution artifact. After the fix, Improved 4 clearly leads\n"
         "on risk-adjusted return. This is a lesson in methodology."),
        (RED, "The honest bottom line (Hansen SPA)",
         "After correcting for 8 variants × multiple testing, SPA p = 0.674.\n"
         "Cannot reject 'no strategy outperforms the benchmark' at 5% FWER.\n"
         "Strong individual-test results are consistent with data-mining luck."),
    ]
    for i, (color, title, text) in enumerate(findings):
        y = 0.77 - i*0.135
        ax.add_patch(mpatches.FancyBboxPatch((0.03, y-0.055), 0.93, 0.115,
            boxstyle="round,pad=0.004", facecolor=DARK2, alpha=0.7))
        ax.text(0.05, y+0.042, title, color=color, fontsize=10, fontweight="bold")
        ax.text(0.05, y+0.001, text, color=LIGHT, fontsize=9,
                transform=ax.transAxes, va="top")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 18: Would you invest?
# ─────────────────────────────────────────────────────────────────────────────
def slide_invest(pdf):
    fig = slide_fig()
    ax = header(fig, "THE DECISION — WOULD WE INVEST?", "Honest investor framing")
    ax.text(0.06, 0.77, "Real capital today?", color=GOLD, fontsize=13, fontweight="bold")
    ax.text(0.40, 0.77, "No — not yet.", color=RED, fontsize=13, fontweight="bold")
    ax.text(0.06, 0.70, "Research watchlist / paper portfolio?", color=GOLD, fontsize=11)
    ax.text(0.55, 0.70, "Yes — Imp 4 and Imp 8.", color=ACCENT, fontsize=11,
            fontweight="bold")
    ax.axhline(0.66, color=GREY, linewidth=0.4, xmin=0.04, xmax=0.96)
    pros = [
        "Economically sensible process (quality, value, momentum, trend)",
        "Historically attractive Sharpes (1.0–1.4) on a 10-year window",
        "Survives realistic transaction costs (Improved 7)",
        "Negative beta — not just riding the market",
        "Transparent, multi-lens validation battery",
        "Walk-forward selected risk exits (not full-sample overfit)",
    ]
    cons = [
        "Survivorship bias — biggest gap; current members only",
        "Multiple testing — SPA p = 0.674 (cannot reject data mining)",
        "Walk-forward train = 4.5 yrs; academic standard is 10+",
        "No sector / liquidity / beta concentration controls",
        "Fixed-cash sizing gives only 4–9 realized names (imp 1–7)",
        "No live broker integration or production-grade CI",
    ]
    ax.text(0.04, 0.62, "Reasons to keep researching:", color=ACCENT, fontsize=10,
            fontweight="bold")
    ax.text(0.52, 0.62, "Reasons not to allocate yet:", color=RED, fontsize=10,
            fontweight="bold")
    for i, (pro, con) in enumerate(zip(pros, cons)):
        y = 0.56 - i*0.073
        ax.text(0.04, y, f"+  {pro}", color=LIGHT, fontsize=9)
        ax.text(0.52, y, f"-  {con}", color=LIGHT, fontsize=9)
    ax.add_patch(mpatches.FancyBboxPatch((0.03, 0.02), 0.93, 0.065,
        boxstyle="round,pad=0.004", facecolor="#1a1a00", alpha=0.9))
    ax.text(0.06, 0.065,
            "Verdict: A cautious investor would not allocate real capital on this deck alone. "
            "Improved 4 is the most credible risk-managed candidate; Improved 8 the wealth-compounding variant. "
            "Neither is deployable until survivorship-free data, SPA-corrected significance, and integrated costs are added.",
            color=GOLD, fontsize=9, transform=ax.transAxes, wrap=True)
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 19: Limitations & Future Work
# ─────────────────────────────────────────────────────────────────────────────
def slide_limitations(pdf):
    fig = slide_fig()
    ax = header(fig, "HONEST LIMITATIONS & FUTURE WORK", "What we would do with more time")
    limits = [
        ("Survivorship bias", "Universe = current S&P 500 members. Delisted/failed companies absent.\n"
         "Fix: WRDS/CRSP point-in-time membership + delisted prices. User has WRDS access."),
        ("Multiple testing corrected", "Hansen SPA p = 0.674 — cannot reject no-outperformance.\n"
         "Unadjusted individual p-values are misleading without this correction. [OK] DONE"),
        ("Walk-forward window", "4.5 years train / 5.5 years test. Academic standard: 10+ years each.\n"
         "Constrained by signal warmup (trend_expanding_z needs 72 months)."),
        ("Fixed-cash sizing gaps", "FixedCashSizer → Margin rejections; only 4–9 of 10 target names filled.\n"
         "Improved 8 solves this via EquityPercentSizer."),
        ("Cost integration", "Improved 7 is a sensitivity layer, not integrated into every Backtrader run.\n"
         "Next step: integrate time-varying commission directly into the default spec."),
    ]
    for i, (title, desc) in enumerate(limits):
        y = 0.76 - i*0.118
        ax.text(0.04, y, f"  {title}:", color=RED, fontsize=10, fontweight="bold")
        ax.text(0.04, y-0.045, desc, color=LIGHT, fontsize=9)
        ax.axhline(y-0.08, color=DARK2, linewidth=0.5, xmin=0.03, xmax=0.97)

    ax.text(0.04, 0.11, "Future work (priority order):",
            color=GOLD, fontsize=10, fontweight="bold")
    future = [
        "1. Survivorship-free panel via WRDS/CRSP",
        "2. Long-short paper-faithful HZZ Q5−Q1 portfolio",
        "3. Factor orthogonalization (trend ⊥ momentum)",
        "4. Sector-concentration caps",
        "5. Top-N sensitivity grid under equal-weight sizing",
    ]
    for i, f in enumerate(future):
        ax.text(0.06 + (i // 3)*0.35, 0.065 - (i % 3)*0.045, f,
                color=GREY, fontsize=9)
    footer(ax, "README §12, §16; docs/MULTI_COMPARISON_TEST.md")
    save_slide(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = PRESENTATION_DIR / "sp500_factor_investing_presentation.pdf"
    PRESENTATION_DIR.mkdir(exist_ok=True)
    with PdfPages(out) as pdf:
        slide_title(pdf)
        slide_paper(pdf)
        slide_data(pdf)
        slide_ladder(pdf)
        slide_execution(pdf)
        slide_window_fix(pdf)
        slide_fmp(pdf)
        slide_ic_chart(pdf)
        slide_results_table(pdf)
        slide_equity_charts(pdf)
        slide_risk_return(pdf)
        slide_walkforward_stops(pdf)
        slide_robustness_overview(pdf)
        slide_alpha_walkforward(pdf)
        slide_mc_spa(pdf)
        slide_costs(pdf)
        slide_interpretation(pdf)
        slide_invest(pdf)
        slide_limitations(pdf)
    print(f"Done — {out}")
