# Project Review, Honest Critique, and Future Work

This document is a comprehensive review of the S&P 500 factor investing project answering 11 questions raised during the finalization phase. It is intentionally critical — it lists 50+ real problems, explains every nuance of the current results, and lays out a concrete roadmap for further improvements.

This is the document to read alongside the README before declaring the project "done."

---

## 1. Why Improved 4's "Better" Stop/Take Reduced Money and Raised P-value

**Numbers reminder:**

| Strategy | BT Sharpe | Max DD | Final $ | MC p |
|---|---:|---:|---:|---:|
| Improved 2 (10/20 stops) | 1.26 | -13.5% | $2.89M | 0.068 |
| Improved 4 (5/30 stops) | **1.40** | **-7.7%** | $2.82M | 0.088 |

**Why Sharpe went UP but money went DOWN:**

Improved 4's 5% stop-loss is much tighter than improved 2's 10%. It cuts losing positions faster, which:

- Dramatically reduces drawdown (-7.7% vs -13.5%) — half the pain
- Reduces volatility → higher Sharpe (1.40 vs 1.26)
- Triggers more often on noise — cuts some positions that would have recovered
- Reduces total return slightly ($2.82M vs $2.89M) — gave up some upside

This is the classical **risk control trade-off** in factor investing. You buy lower drawdown by paying with some absolute return. The Sharpe rises because volatility falls faster than mean return. This is a deliberate design choice for risk-managed strategies.

**Why p-value went UP (worse) despite higher Sharpe:**

When stop/take changes from 10/20 to 5/30, that change benefits **both the strategy AND random portfolios** in the Monte Carlo benchmark. Random selection with 5/30 stops also produces less volatile, more diversified outcomes. So:

- Strategy Sharpe: 1.26 → 1.40 (+0.14)
- Random max Sharpe: ~0.95 → ~1.0 (also up)
- Spread between strategy and random: roughly unchanged
- p-value: slightly worse because the random benchmark also shifted

**The purpose of improved 4 was risk reduction, not return maximization or significance maximization.** It traded a tiny bit of return and a tiny bit of p-value for a massive drawdown improvement. If you can stomach -13% drawdown vs -7%, improved 2 has roughly equivalent economic edge. If you care about drawdown, improved 4 wins by a wide margin.

---

## 2. Why Didn't Improved 8 Build on Improved 6? Variable N? Volatility Sizing?

### Why on improved 4 instead of improved 6

Improved 7's cost analysis showed improved 4 dominates improved 6 under realistic friction. Improved 8 was built on the cost-robust winner. **The user is right that we could have done both** — built two variants (8a on improved 4, 8b on improved 6). One was picked for parsimony. Building 8b is reasonable future work.

### Volatility-targeted sizing — real industry practice

| Sizing rule | Formula | Used by |
|---|---|---|
| Equal-weight (our improved 8) | `w_i = 1/N` | RSP ETF, DGU 2009 |
| Inverse volatility | `w_i ∝ 1/σ_i` | AQR, Two Sigma, most modern quant |
| Risk parity (ERC) | each position contributes equal risk | Bridgewater All Weather |
| Maximum diversification | maximize diversification ratio | Some smart-beta ETFs |
| Conviction-weighted | `w_i ∝ composite_score_i` | Discretionary funds |
| Mean-variance optimal | Markowitz | Mostly textbook — DGU 2009 shows naive 1/N wins out of sample |

Volatility-targeted sizing reduces concentration in jumpy stocks and increases it in stable ones. Should improve risk-adjusted returns. Recommended as **improved 9 or 10**.

### Variable N (range 5-15 instead of fixed 20)

This is a "conditional N" or "threshold-based selection" approach.

- **Industry**: long-short equity hedge funds routinely vary names held based on signal strength
- **Academic**: signal-strength-weighted portfolios (Hou-Xue-Zhang 2015 q-factor model)

How it works: pick all stocks with composite z-score > 1.0 (or some threshold), no fixed N. Some months hold 5 names (only 5 strong signals), some months hold 30 (broad signal). Adapts to market conditions.

Trade-off: less predictable portfolio characteristics, harder to budget capital, but better signal-to-noise ratio per position.

### Other improved 8 enhancements to consider

| Idea | Description | Effort |
|---|---|---|
| Volatility-targeted sizing | `w_i ∝ 1/realized_vol_i` | Easy |
| Threshold-based N (5-15 range) | Hold all stocks with z > X | Easy |
| Conviction-weighted | Position size ∝ composite score | Easy |
| Sector caps | Max 30% in any GICS sector | Medium |
| Single-name cap | Max 10% in any position | Easy |
| Risk parity (ERC) | Equal risk contribution | Medium |
| Hierarchical risk parity | Lopez de Prado 2016 | Hard |
| Time-based exits | Max holding period (e.g., 6 months) | Easy |
| Trailing stops | Stop tightens as price rises | Medium |
| ATR-based stops | Stop = entry - n*ATR | Medium |
| Volatility-budgeted portfolio | Target total portfolio vol | Medium |
| Beta-neutral hedge | Pair with short index futures | Hard |

---

## 3. Harsh Project Criticism — 50+ Real Problems

### DATA QUALITY (1-10)

1. **Severe survivorship bias** — 503 current S&P 500 constituents only, no failed companies
2. **Late-inclusion backfill bias** — Palantir, Coinbase, GE Vernova in panel from IPO, can be "picked" before joining index
3. **Vendor-specific fundamentals** — Tiingo ROE/P/E may differ from CRSP/Compustat standards
4. **Only 20 years of data** — limits regime variety (one financial crisis, one Covid spike, mostly bull market)
5. **No total-return benchmark** — ^GSPC is price index; should compare to total-return ^SPXTR
6. **No risk-free rate** in Sharpe computation (implicitly assumes rf=0)
7. **No currency model** — assumes USD throughout, ignores FX-listed ADRs
8. **Statements `date_available` accuracy** — Tiingo's timestamp is approximate; real availability could differ by days
9. **Daily close prices used as "fill prices"** — real fills are intraday, not at close
10. **No corporate-action verification** — splits, mergers, spinoffs rely on Tiingo's adjustments

### METHODOLOGY (11-20)

11. **Multi-comparison problem** — tested 8 variants, reported best, no Hansen SPA correction (biggest statistical flaw)
12. **Walk-forward train period too short** — 4.5 years (academic standard: 10+)
13. **Walk-forward test period too short** — 5.5 years
14. **`score_for_spec` requires all 4 factors** despite documented "3-of-4 rule" in composite_score
15. **Improved 4's parameters selected on training data** — overfit risk despite penalty for isolated peaks
16. **No factor orthogonalization** — momentum and trend are correlated (both price-based)
17. **No sector neutralization** — strategies may concentrate in single GICS sectors by accident
18. **No industry concentration limits** — same as above at sub-industry level
19. **Look-ahead in base by design** — assignment-prescribed, but should not be reported as honest baseline
20. **HZZ applied in different universe than where validated** — paper uses CRSP 1926-2014 with delisted, we use S&P 500 current

### EXECUTION (21-30)

21. **Fixed $100K sizing creates Margin rejections** — improveds 1-7 hold 4-9 names, not target 10
22. **Strategies labeled "top 10" but realized average is 4-9** — misleading naming
23. **Backtrader daily bar stop/limit fills** — intraday execution may differ significantly
24. **No bid-ask spread modeling** per stock
25. **No market impact for trade sizes** — even though our trades are small relative to ADV
26. **Position sizing ignores stock-specific volatility** — high-vol Tesla treated same as low-vol Coca-Cola
27. **No conviction-weighted sizing tested** — top-1 stock with z=3 sized same as top-10 with z=0.5
28. **EquityPercentSizer rounds to integer shares** — acceptable but documented
29. **No cash buffer** — 100% invested when fully populated
30. **No earnings-announcement avoidance** — trades into / out of earnings unpredictably

### ROBUSTNESS / STATISTICAL (31-40)

31. **No Hansen SPA / Romano-Wolf** — multi-comparison correction absent
32. **Monte Carlo null is weak** — random portfolios from same eligibility, not true zero-edge
33. **Block bootstrap uses single 6-month block size** — should test 3, 6, 12 months
34. **No truly held-out out-of-sample period** — 2021-2026 has been examined many times in development
35. **Bootstrap may not capture serial correlation** — IID block resampling assumption
36. **No Bayesian alternative** to frequentist p-values
37. **Eval window 2016-05 skips 2008 crisis** — no real stress test
38. **Improved 5 (regime filter) untested in 2008 era** — exactly where it should have helped
39. **No confidence intervals on Sharpe estimates** — point estimates only
40. **No tail-risk metrics** (Sortino, CVaR, Calmar)

### PRESENTATION / DOCS (41-50)

41. **Improved 7 not in headline tables** (it's an analysis, not a strategy — but should be referenced)
42. **Walk-forward only includes 4 strategies** (`base`, improved 1-3) — improveds 4-8 missing
43. **Benchmark comparison only includes 4 strategies** — same gap
44. **Vector vs Backtrader equity differences not fully explained per strategy**
45. **STRATEGY_HISTORY.md may have outdated sections** — auto-regenerated but verification needed
46. **PROJECT_REPORT.md was outdated** — has now been synced (see commit 81c329b)
47. **No equity curve comparison plot for all 8 strategies on one chart**
48. **No drawdown comparison plot for all 8 strategies**
49. **No Sharpe-vs-drawdown scatter plot**
50. **No improved 7 cost-drag chart in main figures rotation**

### Bonus issues beyond 50

- `project_core.py` is 2000+ lines (should be modularized)
- No unit tests anywhere
- Few defensive assertions in helper functions
- Magic numbers throughout (RNG_SEED=5811, min_obs=72, etc.)
- Cost schedule hardcoded as Python dict — should be CSV-driven
- No CI/CD or automated test runs
- Git LFS used for data but not enforced via pre-commit

---

## 4. MD/TXT File Audit Status

Five docs were stale and have been updated in commit `81c329b`:

| Doc | Was | Now |
|---|---|---|
| `docs/PROJECT_REPORT.md` | Only base + improved 1-3 results, pre-eval-window numbers | Full 8-strategy results, eval-window methodology, cost sensitivity, three guardrails |
| `docs/PROJECT_PLAN.md` | Strategy ladder stopped at improved 5 | All 8 variants documented, both Backtrader engines + both sizers, robustness testing section |
| `docs/CODE_STRUCTURE.md` | Missing improveds 6/7/8 scripts and folders | All 8 scripts and 9 result folders, dual-engine and dual-sizer notes |
| `docs/DATA_DICTIONARY.md` | Minimal | Every file documented with columns and purpose |
| `docs/REPRODUCIBILITY.md` | Just full pipeline | Focused-script commands for improveds 4-8, reproducibility guarantees |

`docs/STRATEGY_HISTORY.md` is auto-generated by `make_presentation` and should be current. `docs/CLAUDE_CODE_PROJECT_HANDOFF.md` is internal working notes. `data/raw/DATA_DESCRIPTION.md` and `data/raw/README.md` describe the raw data which hasn't changed.

---

## 5. Is `data_audit.csv` Correct?

**Yes, it's correct and current:**

```text
prices              2,292,658 rows  503 tickers  2006-06 to 2026-05  0 missing
daily_fundamentals  2,295,967 rows  499 tickers  2006-06 to 2026-05  6,517 missing
statements          3,458,978 rows  499 tickers  2005-04 to 2026-05  2 missing
constituents        503 rows        503 tickers
sp500_index_yahoo   5,030 rows                   2006-06 to 2026-05  0 missing
factor_panel        109,492 rows    503 tickers  2006-06 to 2026-05  9,236 missing
```

Three minor notes (not errors):

- Daily fundamentals has **499 tickers, not 503** — 4 stocks have no fundamentals in Tiingo (likely recent IPOs)
- 6,517 missing values in daily fundamentals = ~0.3% of rows, normal for vendor data
- 9,236 missing composite scores in factor_panel = ~8.4% of rows, mostly early-period pre-warmup months

`validation_summary.csv` is also correct — confirms all cutoffs, Backtrader long-only validation passed (no negative positions), and 116 result files + 11 figures + 1 presentation PDF were generated. **However**, validation_summary.csv only lists long-only checks for base + improveds 1-3. The actual improveds 4-8 long-only checks are in their per-strategy `improved_<n>_validation_summary.csv` files, which do exist and pass.

---

## 6. Why Doesn't Improved 7 Appear in Headline Tables?

**Because improved 7 isn't a strategy — it's a cost-sensitivity ANALYSIS of improveds 4 and 6.** It has no unique trading rule, no unique portfolio, no unique returns. Including it in a strategy comparison would be misleading.

What improved 7 produces:

- Three cost scenarios applied to improved 4 → three (improved_4, scenario) results
- Three cost scenarios applied to improved 6 → three (improved_6, scenario) results
- Total: 6 cells in the grid, none of which is a stand-alone strategy

Where improved 7 IS referenced in the README:

- Section 6 (Strategy Ladder) — listed as the cost-sensitivity layer
- Section 10.4 (Robustness) — its full results table
- Section 11 mentions improved 7 as the foundation justification for improved 8

This is the correct treatment. A clarifying note in Section 9 saying "improved 7 is a cost-sensitivity layer, not a stand-alone strategy" would help avoid confusion.

---

## 7. Why Don't Walk-Forward / Benchmark / MC Tables Include Every Strategy?

**Real code limitation, not a methodology choice:**

- `walk_forward_summary` reads from `strategy_curves` which is built by `run_strategy_experiments` — and that function only runs the staged ladder (base + improved 1-3), NOT improveds 4-8 (which are run via focused scripts)
- `strategy_benchmark_comparison` has the same dependency

**Why this matters:** improveds 4, 6, 8 are the most interesting variants but they're missing from walk-forward and benchmark alpha tables.

**Fix (future work):** Modify `run_strategy_experiments` (or write a new aggregator) to read the per-strategy curves from `results/improved_strategy_4/vector_equity_curve.csv` etc. and include them in `walk_forward_summary` and `strategy_benchmark_comparison`. This is a ~1-day code change, not a methodology change.

This is one of the **most impactful improvements** for the project's presentation — getting walk-forward and benchmark alpha for improveds 4-8 would let us claim much more rigorously which strategy wins on out-of-sample.

---

## 8. Do We Really Need Vector AND Backtrader?

**No — keep both. This is industry-standard quant fund architecture:**

- **Research engine** (vector simulator) — fast, simple, for screening / grid search / Monte Carlo
- **Production engine** (Backtrader) — realistic execution, for final validation

| Use case | Vector | Backtrader |
|---|---|---|
| Single strategy run | seconds | 10-20 min |
| 6×6 stop/take grid (improved 4) | 36 × seconds = 2 min | 36 × 10 min = 6 hours |
| Monte Carlo 1000 sims | 5 min | weeks |
| Cost sensitivity 6 scenarios × 2 strategies (improved 7) | 1 min | 2 hours |
| Realistic execution check | approximation | high fidelity |
| Order-type-specific behavior (stops, limits, OCO) | simplified | exact |

**Without vector**: improved 4's grid search, improved 7's cost sensitivity, and every Monte Carlo would be infeasible. Vector adds maybe 200 lines of code for an order-of-magnitude speedup on the most expensive operations.

**Without Backtrader**: no realistic order-execution validation. The Backtrader Sharpe being close to vector Sharpe (within 0.1-0.3 in most cases) is empirical evidence that the vector simulator captures the main mechanics; Backtrader confirms with high fidelity.

**Dual-engine is the right design.** Two Sigma, AQR, Bridgewater all separate research and production engines for the same reason.

---

## 9. Additional Data / Signals / Sizing / Risk Management

### Free additional data sources

| Source | What it gives | API |
|---|---|---|
| **SEC EDGAR (Form 4)** | Insider trading signals | Free REST |
| **FRED** | Macro indicators (VIX, yield curve, unemployment) | Free key-based REST |
| **Yahoo Finance** | VIX, sector ETFs, international indices | yfinance |
| **Nasdaq.com** | Short interest data (~biweekly) | Free CSV |
| **Finnhub** (free tier) | News sentiment, earnings calendar | 60 req/min free |
| **Alpha Vantage** (free tier) | Some fundamentals + technicals | 25 calls/day free |
| **Quandl/Nasdaq Data Link** (free tier) | Some commodity / FX / rate data | Free with registration |

### Additional signals/indicators worth testing

| Category | Signal | Implementation effort |
|---|---|---|
| Quality refinements | Gross profitability (gross_profit / assets) | Easy — already have inputs |
| Quality refinements | Accruals quality (Sloan 1996) | Medium |
| Value refinements | P/B, EV/EBITDA, FCF yield | Easy with Tiingo |
| Value refinements | Shiller CAPE (sector-relative) | Medium |
| Momentum refinements | 6-month or 3-month momentum | Easy |
| Momentum refinements | Risk-adjusted momentum (12m / 12m-vol) | Easy |
| Low-volatility | Realized 12-month volatility, inverted | Easy |
| Investment factor | Asset growth (negative) | Easy with Compustat |
| Profitability | Gross profits / total assets | Easy |
| Technical | RSI, MACD, Bollinger Band position | Easy with `ta` library |
| Sentiment | News tone via Finnhub | Medium |
| Macro overlay | VIX-conditioned position scaling | Easy |
| Insider signals | Net insider buys/sells | Medium (EDGAR scraping) |
| Short interest | Days-to-cover ratio (contrarian) | Easy with Nasdaq data |

### Position sizing alternatives (improved 9 candidates)

| Rule | Formula | Industry use |
|---|---|---|
| Volatility-targeted | `w_i ∝ 1/σ_i` | AQR, Two Sigma, most modern quant |
| Equal risk contribution (ERC / risk parity) | Each position contributes equal volatility | Bridgewater All Weather |
| Conviction-weighted | `w_i ∝ z_i / sum(z_j)` | Long-short hedge funds |
| Hierarchical Risk Parity (HRP) | Cluster-then-weight (Lopez de Prado 2016) | Modern quant research |
| Volatility-budgeted | Target total portfolio vol of X% annualized | Pension funds |
| Maximum diversification | Maximize (weighted avg vol) / (portfolio vol) | TOBAM |

### Stop-loss / take-profit alternatives

| Rule | Description | Industry use |
|---|---|---|
| ATR-based stops | `stop = entry - n × ATR(14)` | CTAs, trend followers |
| Trailing stops | Stop tightens as price rises (locks profit) | Retail and discretionary |
| Time-based exits | Max holding period (e.g., 90 days) | Some equity factor funds |
| Earnings-avoidance exits | Exit position 1-2 days before earnings | Risk-management overlay |
| Volatility-targeted stops | `stop = entry - 1.5 × realized_vol_t` | Adaptive stops |
| Drawdown-conditional exits | Tighten stops after portfolio drawdown | Portfolio insurance |
| Chandelier exit | `exit = max(high_n_days) - n × ATR` | Trend-following CTAs |

---

## 10. More Figures and Tables for Presentation

**Current figures (11):** factor portfolio + regression cumulative returns, IC summary, improved 4 stop/take heatmap, improved 7 cost figures (3), MC histogram, strategy drawdowns, strategy equity vs benchmark, strategy improvement Sharpe.

**Missing figures to add (would significantly strengthen presentation):**

| Figure | Purpose |
|---|---|
| All-8-strategies equity curves on one chart | One-glance comparison |
| All-8-strategies drawdown curves on one chart | Risk-profile comparison |
| Sharpe-vs-drawdown scatter (all 8 + RSP + SP500) | Optimization-frontier view |
| Monthly returns heatmap per strategy (year × month) | Stability over time |
| Rolling 12-month Sharpe per strategy | Regime sensitivity |
| Per-sector contribution stacked bar | Sector concentration |
| Average position count over time | Strategy capacity / liquidity |
| Composite score distribution per month (boxplot) | Signal quality over time |
| HZZ beta time series (per MA window) | Improved 6 diagnostic |
| Factor weight evolution over time (improved 3) | Dynamic weighting visualization |
| Cost-drag attribution stacked bar (improved 7) | Year-by-year cost breakdown |
| Walk-forward train vs test scatter for all 8 | OOS performance vs IS |
| Cumulative alpha vs ^GSPC per strategy | Active management visualization |
| Calendar-year returns bar chart | Year-by-year wins/losses |
| Position concentration (HHI) over time | Diversification metric |

**Missing tables:**

| Table | Purpose |
|---|---|
| Annual returns table (year × strategy) | Year-by-year comparison |
| Best/worst 10 months per strategy | Tail behavior |
| Per-sector contribution per strategy | Source of alpha |
| Top 20 contributing stocks per strategy | Idiosyncratic vs systematic |
| Hit rate (% positive months) per strategy | Consistency metric |
| Sortino / Calmar / Omega ratios | Alternative risk metrics |
| Cross-strategy return correlation matrix | Diversification within ladder |
| Average days held per stock | Turnover characterization |
| Average gross / net exposure | Cash drag analysis |

All of these could be built in a single `make_presentation_figures.py` script. Estimated effort: **~2-3 hours of work**. This is the next-highest-priority project improvement after the Hansen SPA test, because it directly strengthens the presentation deliverable.

---

## 11. What Else to Check / Ask About

### Academic concerns

| Question | Why it matters |
|---|---|
| **Is the strategy correlated with Fama-French 5-factor model?** | If our "alpha" is just loading on market / size / value / profitability / investment, it's not unique alpha |
| **What's the strategy's loading on the momentum factor specifically (Carhart 4)?** | Our trend factor + 12-month momentum might just be a momentum bet in disguise |
| **What's the half-life of the signal?** (1-month, 3-month, 6-month forward returns) | Tells us whether the signal is short-term reversal or long-term predictive |
| **Is the signal stable across market regimes?** (high-vol vs low-vol, bull vs bear) | Strategies that only work in one regime are fragile |
| **What's the calendar effect?** (month-end pressure, January effect, tax-loss harvesting) | These can produce fake alpha |
| **Are returns negatively skewed?** (tail risk hidden by Sharpe) | Stop-loss strategies typically are |

### Industry concerns

| Question | Why it matters |
|---|---|
| **What's the strategy's capacity?** (How much AUM before performance degrades) | Real fund question: matters for actual deployment |
| **Maximum participation rate per name?** | Don't be > 10% of any stock's average daily volume |
| **Average daily turnover?** | Tells us realistic trading cost exposure |
| **What's the longest underwater period?** (time to recovery from drawdown) | Investor psychology: even good strategies are uninvestable if drawdowns last 3+ years |
| **What's the worst single-day loss?** | Tail risk |
| **What's the strategy's beta to common risk factors (size, value, momentum)?** | Are we generating alpha or just selling cheap risk premia? |
| **Can the strategy handle quarterly investor redemptions?** | Liquidity constraint |
| **What's the gross vs net exposure?** | Are we 100% invested, 80%, 120%? Risk implication |

### Risk management questions

| Question | Why it matters |
|---|---|
| **Single-stock cap?** | Don't be > 5% in any single name (we sometimes exceed via integer rounding) |
| **Sector cap?** | Don't be > 25-30% in any GICS sector |
| **Country / currency cap?** | N/A here (US-only), but relevant if expanded |
| **Concentration HHI?** | Single number for portfolio diversification |
| **What happens in a market crash?** | Backtest 2008 if data permits; we currently skip it via the eval window |

### Things to run / build next

1. **Hansen SPA / Romano-Wolf** — biggest statistical-rigor improvement (1 day)
2. **All-strategies-on-one-chart equity / drawdown plots** — biggest presentation improvement (2 hours)
3. **Walk-forward + benchmark alpha extended to all 8 strategies** — biggest results-completeness improvement (1 day)
4. **Volatility-targeted improved 9** — biggest "real-world realism" sizing improvement (1 day)
5. **Sector exposure analysis** — biggest unaddressed risk concern (4 hours)
6. **Annual returns table + monthly heatmap** — biggest "shows stability over time" addition (2 hours)
7. **Fama-French 5-factor regression of strategy returns** — biggest academic alpha-attribution improvement (4 hours)
8. **Carhart momentum factor loading check** — biggest "is this real or just momentum" check (2 hours)

### Recommended 2-3 week roadmap

**Week 1 — Statistical rigor and presentation:**

- Hansen SPA correction (item 1)
- Extend walk-forward + benchmark to all 8 strategies (item 3)
- All-strategies-on-one-chart figures (item 2)
- Annual returns table + monthly heatmap (item 6)

**Week 2 — Sophistication:**

- Volatility-targeted improved 9 (item 4)
- Sector exposure analysis (item 5)
- Fama-French 5-factor + Carhart momentum loading regressions (items 7-8)

**Optional Week 3 — The big one:**

- WRDS / CRSP survivorship-bias-free panel (improved 10) → re-run everything

These are all defensible additions that match what an academic reviewer or quant fund interviewer would actually ask for. None requires data we don't have access to.
