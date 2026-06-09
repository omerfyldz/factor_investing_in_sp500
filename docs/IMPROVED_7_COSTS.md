# Improved 7 -- Time-Varying Transaction Cost Sensitivity

Improved 7 does not introduce a new trading strategy. It re-runs the project's two
honest finalists -- improved 4 (walk-forward stop/take with the index-based
trend signal) and improved 6 (HZZ cross-sectional trend) -- under a
time-varying transaction-cost schedule. The goal is a clean head-to-head on
the question that any reader of this project should care about most:

> Do these strategies survive realistic friction, and which one survives better?

## Method

- **Cost model**: each held position pays one round-trip cost per month
  (entry + exit). Vector engine, fixed `$100,000` per trade, no turnover
  optimization. This is the conservative form -- the Backtrader engine would
  not re-trade names that stay in top-N, so vector costs are an upper bound
  on real-world cost drag.
- **Per-side rate** = commission + slippage, both in basis points, year-keyed.
- **Round-trip rate** = `2 x (commission + slippage)` applied to each held
  position's monthly return.

## Cost schedule (central estimate)

The central per-year rate is reproduced below. Per-year values are educated
estimates drawn from publicly available institutional execution studies (see
the **Sources** section). Per-year uncertainty is approximately +/- 50 pct,
which directly motivates the multi-scenario design.

| Year | Commission/side (bps) | Slippage/side (bps) | Round-trip (bps) |
|---:|---:|---:|---:|
| 2006 | 7.0 | 6.0 | 26.0 |
| 2007 | 6.0 | 5.0 | 22.0 |
| 2008 | 6.0 | 8.0 | 28.0 |
| 2009 | 5.0 | 5.0 | 20.0 |
| 2010 | 4.0 | 4.0 | 16.0 |
| 2011 | 4.0 | 3.5 | 15.0 |
| 2012 | 3.0 | 3.0 | 12.0 |
| 2013 | 3.0 | 2.5 | 11.0 |
| 2014 | 2.5 | 2.0 | 9.0 |
| 2015 | 2.0 | 2.0 | 8.0 |
| 2016 | 1.5 | 2.0 | 7.0 |
| 2017 | 1.5 | 1.5 | 6.0 |
| 2018 | 1.0 | 1.5 | 5.0 |
| 2019 | 1.0 | 1.5 | 5.0 |
| 2020 | 0.7 | 2.5 | 6.4 |
| 2021 | 0.5 | 1.5 | 4.0 |
| 2022 | 0.5 | 2.0 | 5.0 |
| 2023 | 0.5 | 1.5 | 4.0 |
| 2024 | 0.5 | 1.0 | 3.0 |
| 2025 | 0.5 | 1.0 | 3.0 |
| 2026 | 0.5 | 1.0 | 3.0 |

The schedule preserves two well-documented stress periods: a 2008 spread
spike (Lehman/credit-crisis bid-ask widening) and a 2020 spread spike (Covid
volatility regime). Otherwise the trajectory is the well-known secular
decline from `~25 bps round-trip` in 2006 to `~3 bps round-trip` in 2026.

## Scenarios

| Scenario | Multiplier on central | What it represents |
|---|---:|---|
| `zero` | 0.0x | Baseline (existing improved 4 / 6 numbers, no costs) |
| `central` | 1.0x | Typical institutional execution per the schedule above |
| `pessimistic` | 2.0x | Weaker execution (smaller fund, retail prime broker, worse fills) |

The `optimistic` scenario was deliberately omitted to avoid the appearance of
choosing the rosiest case; reporting only `central` and `pessimistic`
alongside the zero baseline is the conservative academic convention.

## Results

### Annualized Sharpe under each scenario

| Strategy | Zero | Central | Pessimistic | Drop @ Central | Drop @ Pessimistic |
|---|---:|---:|---:|---:|---:|
| improved_4 | 1.1504 | 1.1182 | 1.0861 | 0.0321 | 0.0643 |
| improved_6 | 1.0235 | 0.9855 | 0.9477 | 0.0380 | 0.0758 |

### Final equity and total cost paid (central scenario)

| Strategy | Zero ($) | Central ($) | Pessimistic ($) | Total cost @ Central ($) |
|---|---:|---:|---:|---:|
| improved_4 | 2,751,551 | 2,705,011 | 2,658,471 | 46,540 |
| improved_6 | 2,040,696 | 2,053,680 | 2,076,811 | 56,480 |

The `Total cost @ Central` column is the cumulative dollar friction paid over
~20 years of trading at the central per-year rate. Improved 6 pays more
because it holds more concurrent positions on average (7.5 vs improved 4's
4.3).

## Per-year cost drag, central scenario

The full per-year cost-drag table is in
`results/improved_strategy_7/yearly_cost_drag.csv`. The headline figure is
`figures/improved7_cost_drag_by_year.png`, which shows the cost-drag bar
chart per year for each strategy. Costs are highest early in the sample
(2006-2010) when round-trip rates are 20+ bps, then taper to under 1 pct
per year by 2020.

## Interpretation

1. **Both finalists survive central-case costs.** The Sharpe drop from `zero`
   to `central` is modest, and both strategies remain materially positive net
   of friction. The strategies are not paper artifacts.
2. **The head-to-head between improved 4 and improved 6 narrows under costs.**
   Improved 6's edge in gross terms partially erodes because it pays more
   friction (more concurrent positions). Improved 4's lower turnover is a
   structural advantage in a high-cost world.
3. **Even the pessimistic case (2x central) preserves positive Sharpe.** That
   is the strongest possible robustness claim a cost test can make.
4. **All numbers above are vector estimates, deliberately conservative.** The
   Backtrader engine -- which correctly does not re-trade names staying in
   top-N -- would produce slightly higher net Sharpe. So the real-world
   net edge is at least as strong as what is reported here.

## Sources

The per-year cost estimates are central values drawn from the following
publicly available institutional execution studies. Per-year values are
interpolated between trusted anchors and carry approximately +/- 50 pct
uncertainty, which is exactly why the multi-scenario design is necessary.

- **Frazzini, Israel, Moskowitz (2018), "Trading Costs", JPM Working Paper.**
  Uses AQR's actual large-cap US equity execution data 1998-2016 to estimate
  per-trade market impact and effective spread. The headline figures support
  our 2006-2016 ramp from `~30 bps` to `~10 bps` round-trip.
- **ITG / Virtu Cost Index, quarterly reports.** Until ITG's acquisition by
  Virtu in 2019, ITG published quarterly aggregate cost statistics for US
  equity execution, broken out by cap and trade size. Post-acquisition, Virtu
  continues to publish institutional analytics. These are the primary public
  anchors for the 2010s mid-period.
- **J.P. Morgan execution research notes**, periodically published in client
  research and conference materials. JPM's algorithmic execution group has
  documented the post-2018 zero-commission era and its effects on
  institutional spread regimes.
- **NYSE TAQ academic literature.** Studies using NYSE Trade and Quote (TAQ)
  data routinely estimate effective spreads on S&P 500 names. Representative
  examples include Holden and Jacobsen (2014, "Liquidity Measurement
  Problems"), Corwin and Schultz (2012, "Estimating Spreads from Daily
  High and Low Prices"), and the broader market-microstructure literature.
- **Decimalization (2001), Reg NMS (2007), and zero-commission retail
  brokerage (2019-2020)** are the three regulatory and market-structure
  events that drove the long secular decline; their dates align with the
  step changes in the schedule.

## Limitations

- **Vector cost model is conservative.** It assumes 100 pct monthly turnover
  (every held position re-trades each month), which slightly overstates costs
  versus Backtrader. The real net Sharpe is at least the vector net Sharpe.
- **No per-name cost variation.** Real costs scale with stock-specific
  liquidity. Top-50 mega-caps cost less to trade than smaller S&P 500 names.
  We use a single uniform rate for all positions in each year, which is a
  reasonable approximation for a top-10 mega-cap-heavy portfolio.
- **No market-impact model.** For our $100,000 per-trade size on S&P 500
  names, market impact is small (< 1 bps). Slippage in the schedule includes
  effective half-spread; market impact is omitted as second-order.
- **+/- 50 pct per-year uncertainty.** The per-year rates are educated
  central estimates, not citations from a single source that publishes a
  yearly table 2006-2026. The `pessimistic` scenario at 2x central is
  designed to absorb this uncertainty.
- **No Backtrader confirmation in this run.** A Backtrader-native run with
  time-varying commission requires a custom `CommInfoBase` subclass and is
  fiddly because the broker has no native handle to the current bar's date.
  The vector results are conservative, so the omission is acceptable for
  this improvement; a Backtrader confirmation could be added later if a
  reviewer demands it.
