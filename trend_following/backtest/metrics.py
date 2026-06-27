"""Performance metrics for a monthly-return series. All annualised with 12.

Sharpe/Sortino are reported with rf=0 (total-return convention) and the SAME
convention is applied to every strategy and benchmark, so cross-comparisons are
apples-to-apples. The strategy already holds BIL (T-bills) as its defensive
leg, so its returns include the cash yield rather than excluding it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 12


def max_drawdown(nav: pd.Series) -> float:
    """Most negative peak-to-trough drop of the NAV curve (e.g. -0.15)."""
    return float((nav / nav.cummax() - 1.0).min())


def metrics(monthly_returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = monthly_returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    nav = (1.0 + r).cumprod()
    cagr = float(nav.iloc[-1] ** (PERIODS / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(PERIODS))
    ann_ret = float(r.mean() * PERIODS)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    downside = r[r < 0]
    dd_std = float(downside.std(ddof=1) * np.sqrt(PERIODS)) if len(downside) > 1 else np.nan
    sortino = float(ann_ret / dd_std) if dd_std and dd_std > 0 else np.nan
    mdd = max_drawdown(nav)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    out = {
        "months": n,
        "start": r.index[0].date().isoformat(),
        "end": r.index[-1].date().isoformat(),
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": mdd,
        "Calmar": calmar,
        "PctPosMonths": float((r > 0).mean()),
        "BestMonth": float(r.max()),
        "WorstMonth": float(r.min()),
    }
    if turnover is not None and len(turnover.dropna()):
        out["AvgAnnTurnover"] = float(turnover.reindex(r.index).fillna(0).mean() * PERIODS)
    return out


def format_table(named_metrics: dict[str, dict]) -> str:
    """Render {name: metrics_dict} as an aligned markdown table."""
    rows = [
        ("CAGR", "CAGR", "{:+.2%}"),
        ("Ann. Vol", "AnnVol", "{:.2%}"),
        ("Sharpe", "Sharpe", "{:.2f}"),
        ("Sortino", "Sortino", "{:.2f}"),
        ("Max Drawdown", "MaxDD", "{:+.2%}"),
        ("Calmar", "Calmar", "{:.2f}"),
        ("% Positive Months", "PctPosMonths", "{:.1%}"),
        ("Worst Month", "WorstMonth", "{:+.2%}"),
        ("Avg Ann. Turnover", "AvgAnnTurnover", "{:.0%}"),
    ]
    names = list(named_metrics)
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [header, sep]
    for label, key, fmt in rows:
        cells = []
        for nm in names:
            v = named_metrics[nm].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) and not _isnan(v) else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # period footer
    any_m = next(iter(named_metrics.values()))
    lines.append("")
    lines.append(f"_Period: {any_m.get('start','?')} → {any_m.get('end','?')} "
                 f"({any_m.get('months','?')} months). Sharpe/Sortino at rf=0._")
    return "\n".join(lines)


def _isnan(v) -> bool:
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False
