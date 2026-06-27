# Trend Following on ETFs — Faber GTAA (v1)

A multi-asset **trend-following** strategy that holds an ETF only while it is in
an uptrend and rotates to T-bills otherwise. It deliberately trades a small set
of liquid ETFs rather than individual stocks, which **eliminates survivorship
bias** (ETFs are not delisted out of a backtest the way failed stocks are).

> Status: **v1 = canonical Faber GTAA**, validated by backtest before live paper
> trading. Volatility-targeted sizing (Moskowitz–Ooi–Pedersen) is deferred to v2.

## The rule (Faber 2007)

At each **month-end**, for each risk ETF:

```
if  month_end_close > 10-month simple moving average   ->  hold the ETF
else                                                    ->  hold T-bills (BIL)
```

Capital is split **equally** across the 9 risk sleeves (1/9 each). Any sleeve
that is below its 10-month SMA parks its 1/9 in **BIL** until the trend turns
back up. Rebalanced monthly; signals act the **following** month (no look-ahead).

The 10-month SMA is Faber's canonical, pre-registered parameter (≈ the 200-day
MA). Using it as-is — rather than searching for the "best" lookback — is what
keeps the rule honest and hard to overfit. We still publish a 6/8/10/12-month
robustness sweep to show the result is not a knife-edge.

## Universe (one clean representative per asset class)

| ETF | Asset class | | ETF | Asset class |
|---|---|---|---|---|
| SPY | US equity | | HYG | US high-yield credit |
| EFA | Intl developed equity | | GLD | Gold |
| EEM | Emerging-market equity | | DBC | Broad commodities |
| TLT | US long treasury | | VNQ | US real estate (REIT) |
| IEF | US intermediate treasury | | **BIL** | **T-bills (defensive leg)** |

QQQ / XLK / XLE / XLF were excluded as redundant with SPY; **USO was excluded
for contango/roll drag** and replaced with DBC. **BIL** is the cash leg in both
backtest and live (same ticker, inception May 2007). Binding inception is
HYG/BIL (2007), so the backtest cleanly spans the 2008 GFC, 2020 COVID crash and
2022 bond bear — the three best stress tests for trend following.

## Why trend following

- **Crisis alpha:** when equities crash, the rule is already in T-bills, so the
  strategy's drawdowns are far shallower than buy-and-hold (the whole point).
- **No survivorship bias / no fundamental data:** price-only signals on ETFs.
- **Robust across regimes:** documented over a century of data (AQR).

## Layout

```
trend_following/
├── core.py                 # shared signal/weight logic (single source of truth)
├── backtest/
│   ├── data.py             # yfinance download -> month-end adj closes (cached)
│   ├── engine.py           # monthly backtest, .shift(1) look-ahead guard, costs
│   ├── metrics.py          # CAGR, Sharpe, Sortino, MaxDD, Calmar, turnover
│   ├── run_backtest.py     # strategy vs SPY buy&hold vs 60/40 -> results/
│   └── robustness.py       # 6/8/10/12-month × cost sweep
├── live/                   # (added after the backtest gate) Alpaca paper trader
├── tests/
│   ├── test_core.py        # weight invariants
│   └── test_lookahead.py   # proves the engine cannot peek at same-month returns
├── results/                # metrics.md, nav.csv, equity_curve.png (from Actions)
└── requirements.txt
```

## Running

The dev sandbox blocks market-data hosts, so the backtest runs in **GitHub
Actions** (`Trend Following Backtest` workflow, manual dispatch). It downloads
prices, runs the unit tests, the backtest and the robustness sweep, and commits
the outputs into `results/`. Locally (with network) you can run:

```bash
cd trend_following
python tests/test_core.py && python tests/test_lookahead.py
python -m backtest.run_backtest --force-download
python -m backtest.robustness
```

## References

- Mebane T. Faber, *A Quantitative Approach to Tactical Asset Allocation*, J. of
  Wealth Management (2007). SSRN 962461.
- Moskowitz, Ooi, Pedersen, *Time Series Momentum*, J. Financial Economics (2012).
- Hurst, Ooi, Pedersen (AQR), *A Century of Evidence on Trend-Following Investing*.
- Antonacci, *Dual Momentum Investing* (2014).
