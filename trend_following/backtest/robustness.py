"""Robustness sweep — show the result is not a knife-edge on the '10' in 10-month.

Re-runs the strategy for several SMA lookbacks (6/8/10/12 months) and several
cost assumptions, printing CAGR / Sharpe / MaxDD for each. If the strategy only
"works" at exactly 10 months, that is a red flag for overfitting; a good trend
rule degrades gracefully across nearby lookbacks.

Usage (from trend_following/):
    python -m backtest.robustness
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.data import load_or_download
from backtest.engine import align_common, benchmark_spy, run_backtest
from backtest.metrics import metrics

RESULTS = Path(__file__).resolve().parent.parent / "results"
LOOKBACKS = [6, 8, 10, 12]
COSTS = [0.0, 5.0, 10.0]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    prices = load_or_download()

    rows = []
    for ma in LOOKBACKS:
        for cost in COSTS:
            strat = run_backtest(prices, ma_months=ma, cost_bps=cost)
            spy = benchmark_spy(prices)
            strat, _ = align_common(strat, spy)
            m = metrics(strat["monthly_returns"], strat.get("turnover"))
            rows.append({
                "MA_months": ma,
                "cost_bps": cost,
                "CAGR": round(m["CAGR"], 4),
                "Sharpe": round(m["Sharpe"], 2),
                "MaxDD": round(m["MaxDD"], 4),
                "Calmar": round(m["Calmar"], 2),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "robustness.csv", index=False)

    md = ["# Robustness sweep (SMA lookback × cost)", "",
          "| MA (months) | Cost (bps) | CAGR | Sharpe | MaxDD | Calmar |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['MA_months']} | {r['cost_bps']:.0f} | {r['CAGR']:+.2%} | "
                  f"{r['Sharpe']:.2f} | {r['MaxDD']:+.2%} | {r['Calmar']:.2f} |")
    (RESULTS / "robustness.md").write_text("\n".join(md) + "\n")
    print(f"\nSaved {RESULTS}/robustness.md")


if __name__ == "__main__":
    main()
