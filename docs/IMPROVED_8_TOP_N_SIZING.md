# Improved 8 -- Equal-Weight 1/N Sizing With Top 20

Improved 8 is a focused position-sizing and diversification experiment. It
changes two design dimensions on top of improved 4:

1. Top-N selection moves from 10 to 20.
2. Position sizing moves from fixed `$100,000` per trade to
   equal-weight 5 pct of current portfolio equity per position (1/N at top 20).

The two changes are mechanically coupled: fixed-dollar sizing is incompatible
with meaningful top-N expansion because `$1M` starting capital cannot fund
`20 x $100,000 = $2M`. Switching to percent-of-equity sizing removes the cash
constraint and lets every monthly rebalance target a true 20-name portfolio.

## Method

- **Foundation**: improved 4 design (composite of ROE, P/E, momentum,
  trend_expanding_z) with the same 5 pct stop-loss and 30 pct take-profit.
- **Top-N**: `20` (vs improved 4's 10).
- **Sizing method**: `percent_of_equity` with target `5.00%` per position.
- **Per-position dollars**: dynamic, equal to `5 pct x current portfolio value`.
- **Stop-loss / take-profit**: `5.0%` / `30.0%`,
  per-position (percentages scale appropriately with dynamic sizing).
- **Daily Backtrader execution**: `EquityPercentSizer` with native
  `bt.Order.Stop` / `bt.Order.Limit` protective exits.
- **Monte Carlo benchmark**: random portfolios sampled at the same top-N
  using the same equal-weight sizing rule, so the comparison is apples-to-
  apples with the strategy.

## Justification

### Equal-weight 1/N sizing

DeMiguel, Garlappi, Uppal (2009, *Review of Financial Studies*, "Optimal Versus
Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?")
evaluated 14 mean-variance optimization variants across 7 empirical datasets
(including US sector portfolios, international indices, and individual stocks).
None reliably outperformed naive 1/N on out-of-sample Sharpe. The reason is
estimation error: to reliably parametrize a mean-variance optimizer for a
25-asset portfolio would require roughly 3,000 months (250 years) of return
data, which is unavailable. Equal-weight is parsimonious, has zero estimation
error, and matches the industry analog -- the Invesco S&P 500 Equal Weight
ETF (RSP) -- one of the largest and longest-running smart-beta products.

### Top-20 selection

The relevant academic standards are quintile portfolios (top 20 pct = ~100
names for the S&P 500) used by Fama-French and Han-Zhou-Zhu, and decile
portfolios (top 10 pct = ~50 names) used in much of the cross-sectional
asset-pricing literature. Top 10 -- our prior choice -- is more concentrated
than any standard factor study and is justifiable only as a high-conviction
approach. Top 20 (~4 pct of the universe) is a defensible middle ground:
materially more diversified than top 10 while still expressing concentration
in the highest-scoring names. Plyakha, Uppal, Vilkov (2014, *Critical Finance
Review*) further documents the systematic outperformance of equal-weighted
concentrated portfolios over value-weighted alternatives.

### Foundation choice

Improved 7's time-varying cost analysis showed improved 4 (index-trend) wins
the head-to-head against improved 6 (HZZ cross-sectional trend) once
realistic transaction costs are applied. Improved 8 builds on the cost-robust
winner. Improved 4's structurally low turnover advantage should compose well
with the increased diversification of top 20.

## Results

- Vector Sharpe: `1.0333`.
- Vector final equity: `$3,633,331`.
- Vector max drawdown: `-10.94%`.
- Backtrader final value: `$3,088,557`.
- Backtrader Sharpe: `1.1098`.
- Backtrader max drawdown: `-11.90%`.
- Monte Carlo p-value (against equal-weight 20-name random portfolios): `0.3060`.

## Head-to-Head vs Improved 4

- Improved 4 vector Sharpe: `1.1504`; improved 8 vector Sharpe: `1.0333`.
- Improved 4 Backtrader Sharpe: `1.3961`; improved 8 Backtrader Sharpe: `1.1098`.

The most informative comparison is whether the diversification gain (more
positions, more even risk distribution) outweighs the signal-dilution cost
(top 20 includes names ranked 11-20 which had lower composite scores). If
improved 8 has comparable or better Sharpe than improved 4 with materially
lower drawdown, it confirms that the original top 10 was over-concentrated
and that the project's right operating point is closer to a quintile-style
academic standard.

## Caveats

- **Dynamic position sizing compounds.** Each rebalance sizes positions at
  5 pct of *current* equity. As equity grows, positions grow. This is the
  realistic behavior of any real fund and matches academic convention, but
  it makes absolute equity comparisons across improvements with different
  sizing rules less directly meaningful. Annualized Sharpe and max drawdown
  remain apples-to-apples.
- **Integer share rounding.** `EquityPercentSizer` rounds down to whole
  shares per position, leaving a small unallocated cash residual each
  rebalance. This is realistic and matches how practitioners trade.
- **No cash buffer.** 20 positions at 5 pct each consume 100 pct of equity
  when fully populated. Real funds typically hold 2-5 pct cash. A more
  defensive variant would target 4.75 pct per position (95 pct invested);
  we keep the cleaner 5 pct for parsimony.
- **Common evaluation window.** All metrics use `EVALUATION_START =
  2016-05-31` so the comparison with prior improveds is
  apples-to-apples on the same trading months.

## References

- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
  Diversification: How Inefficient Is the 1/N Portfolio Strategy?
  *Review of Financial Studies*, 22(5), 1915-1953.
- Plyakha, Y., Uppal, R., & Vilkov, G. (2014). Why Does an Equal-Weighted
  Portfolio Outperform Value- and Price-Weighted Portfolios? *Critical
  Finance Review*, 4(2), 271-308.
- Invesco S&P 500 Equal Weight ETF (RSP). Industry analog for equal-weight
  US large-cap concentrated portfolios.
