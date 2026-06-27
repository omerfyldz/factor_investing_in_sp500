"""Run the full backtest: strategy vs SPY buy-and-hold vs 60/40, save outputs.

Usage (from the trend_following/ directory):
    python -m backtest.run_backtest                 # uses cached data if present
    python -m backtest.run_backtest --force-download # refetch from yfinance
    python -m backtest.run_backtest --cost-bps 10    # stress the cost assumption

Outputs (committed by the Actions run):
    results/metrics.md          markdown comparison table + inception report
    results/nav.csv             NAV curves (strategy + benchmarks)
    results/equity_curve.png    log-scale equity curves (if matplotlib present)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest.data import inception_report, load_or_download
from backtest.engine import align_common, benchmark_60_40, benchmark_spy, run_backtest
from backtest.metrics import format_table, metrics
from core import MA_MONTHS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--ma-months", type=int, default=MA_MONTHS)
    ap.add_argument("--start", default="2005-01-01")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading prices...")
    prices = load_or_download(start=args.start, force=args.force_download)
    incep = inception_report(prices)
    print(incep.to_string(index=False))

    print(f"\nRunning strategy (MA={args.ma_months}mo, cost={args.cost_bps}bps)...")
    strat = run_backtest(prices, ma_months=args.ma_months, cost_bps=args.cost_bps)
    spy = benchmark_spy(prices)
    p6040 = benchmark_60_40(prices)

    strat, spy, p6040 = align_common(strat, spy, p6040)

    named = {
        "GTAA Trend": metrics(strat["monthly_returns"], strat.get("turnover")),
        "SPY Buy&Hold": metrics(spy["monthly_returns"]),
        "60/40": metrics(p6040["monthly_returns"]),
    }
    table = format_table(named)
    print("\n" + table)

    # --- Save NAV curves ---
    nav = pd.DataFrame({
        "GTAA_Trend": strat["nav"],
        "SPY_BuyHold": spy["nav"],
        "Portfolio_60_40": p6040["nav"],
    })
    nav.to_csv(RESULTS / "nav.csv")

    # --- Save markdown report ---
    md = [
        "# Trend-Following (Faber GTAA) — Backtest Results",
        "",
        f"Universe: {', '.join(strat['weights'].columns)}",
        f"Lookback: {args.ma_months}-month SMA · Cost: {args.cost_bps} bps/side · "
        "Monthly rebalance · Look-ahead guarded (weights shifted 1 month).",
        "",
        "## Performance",
        "",
        table,
        "",
        "## ETF inception (binds the backtest start)",
        "",
        "| Ticker | First date |",
        "|---|---|",
    ]
    for _, row in incep.iterrows():
        md.append(f"| {row['ticker']} | {row['first_date']} |")
    (RESULTS / "metrics.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved results to {RESULTS}/")

    _maybe_plot(nav)


def _maybe_plot(nav: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping equity-curve plot.")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in nav.columns:
        ax.plot(nav.index, nav[col], label=col.replace("_", " "))
    ax.set_yscale("log")
    ax.set_title("Growth of $1 — Trend Following vs Benchmarks (log scale)")
    ax.set_ylabel("NAV (log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "equity_curve.png", dpi=120)
    print(f"Saved {RESULTS}/equity_curve.png")


if __name__ == "__main__":
    main()
