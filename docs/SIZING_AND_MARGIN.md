# Sizing, Margin Rejections, and Long-Only Money Mechanics

This document explains how position sizing works in the strategy, what "Margin rejections" mean in Backtrader, and how the portfolio's money grows (and stays long-only) over time. It answers three related questions:

1. Why does Backtrader say some orders are rejected as `Margin`?
2. In a long-only strategy, how does the portfolio buy stocks as its wealth grows?
3. How does each sizer handle cash through time?

---

## 1. The Three Sizers Used in This Project

| Sizer | Used by | Formula | Key property |
|---|---|---|---|
| `FixedCashSizer` | Base, improved 1–7 | Shares = floor($100,000 / price) | Dollar amount per position is always $100K regardless of portfolio size |
| `EquityPercentSizer` | Improved 8 | Shares = floor(target_pct × portfolio_value / price) | Position size scales with portfolio equity |
| `VolatilityTargetedSizer` | Improved 9 | Shares = floor(target_pct × (median_vol / stock_vol) × portfolio_value / price) | Like EquityPercent but high-vol stocks get less capital |

---

## 2. Why Margin Rejections Happen (FixedCashSizer)

### Setup
- Initial capital: **$1,000,000**
- Fixed cash per trade: **$100,000**
- Target top-N: **10 positions**
- Nominal total: 10 × $100,000 = $1,000,000 = exactly all capital

### The Problem
When the strategy tries to open 10 positions simultaneously, Backtrader sizes each one as:

```
shares = floor($100,000 / current_price)
actual_cost = shares × current_price  (< $100,000 due to floor)
```

Due to integer rounding, each fill typically costs slightly less than $100,000. But Backtrader executes orders sequentially and checks cash before each fill. If the price has moved slightly since the order was submitted (next-bar execution), the actual cost can exceed the available cash for later positions.

**Result:** The last 1–2 buy orders get rejected with `Margin` status. The realized portfolio holds 8–9 names instead of 10, and some cash sits idle.

### Is This a Bug?
**No.** It is honest behavior. A realistic broker would also reject orders that exceed available cash. The project documents this as a known limitation:

> *"Strategies labeled 'top 10' but realized average is 4–9 due to FixedCashSizer margin constraints"* — `PROJECT_REVIEW_AND_FUTURE_WORK.md`, issue #22

The fix would be to reduce `CASH_PER_TRADE` from $100,000 to, say, $90,000 (leaving a buffer), but the project preserves the original $100,000 figure to match the assignment specification.

### What Happens to the Rejected Cash?
The undeployed cash sits **idle in the broker account earning zero return** (no money-market rate is modeled). This is a conservative assumption — it slightly understates portfolio performance relative to a frictionless ideal.

---

## 3. How a Long-Only Strategy Buys Stocks as Equity Grows

### Under FixedCashSizer (Base, Improved 1–7)

The sizer always bids exactly $100,000 per position, regardless of portfolio size. This means:

- **At $1M**: holds up to 10 names at $100K each
- **At $2M**: still holds up to 10 names at $100K each — the extra $1M sits as cash
- **At $3M**: same — 10 names at $100K, $2M idle

The strategy **cannot buy "more" even when the portfolio doubles**. The fixed-cash rule is mechanically frozen at 10% of the original capital. This is why the base strategy's final equity ($5.3M) has a lower Sharpe than you might expect — the growing cash drag reduces total return relative to a fully-invested strategy.

#### Worked Example: $1M → $2M under FixedCashSizer

| State | Portfolio value | Deployed | Idle cash | Positions |
|---|---|---|---|---|
| Inception | $1,000,000 | $1,000,000 | $0 | 10 × $100K |
| After good year (+50%) | $1,500,000 | $1,000,000 | $500,000 | 10 × $100K |
| After another good year | $2,000,000 | $1,000,000 | $1,000,000 | 10 × $100K |

The position size in dollar terms stays frozen; only the number of shares changes at each rebalance (because the price has moved). The growing cash pile earns 0%.

---

### Under EquityPercentSizer (Improved 8)

Position size = 5% of current portfolio value. As the portfolio grows, each position grows proportionally.

- **At $1M**: 20 positions at $50K each = $1M fully deployed
- **At $2M**: 20 positions at $100K each = $2M fully deployed
- **At $3M**: 20 positions at $150K each = $3M fully deployed

The portfolio stays **fully invested** (modulo integer rounding). No growing idle cash. This is why improved 8 is a "wealth-maximizer" — it compounds more aggressively than the fixed-cash alternatives.

#### Worked Example: $1M → $2M under EquityPercentSizer

| State | Portfolio value | Per-position allocation | Positions |
|---|---|---|---|
| Inception | $1,000,000 | $50,000 | 20 × 5% |
| After good year | $1,500,000 | $75,000 | 20 × 5% |
| After another good year | $2,000,000 | $100,000 | 20 × 5% |

---

### Under VolatilityTargetedSizer (Improved 9)

Position size = 5% × (median_vol / stock_vol) × current portfolio value.

- High-vol stock (e.g., realized vol = 50% ann.): gets 5% × (25% / 50%) = 2.5% of equity
- Low-vol stock (e.g., realized vol = 10% ann.): gets 5% × (25% / 10%) = 12.5% of equity (capped by the basket normalization)
- Median-vol stock: gets the nominal 5%

Like EquityPercentSizer, the allocations scale with portfolio value. Unlike EquityPercentSizer, each name gets a different fraction of capital — low-volatility names get more, high-volatility names get less. The **theoretical justification** is risk-budget equality: each position contributes equal expected volatility to the portfolio, not equal dollars.

The **vector engine** does exact basket normalization:
```
w_i = (1/vol_i) / sum_j(1/vol_j) for j in the held basket
dollars_i = w_i × portfolio_value
```

The **Backtrader VolatilityTargetedSizer** approximates this per-name:
```
dollars_i ≈ target_pct × (median_vol / vol_i) × portfolio_value
```

The approximation is slightly different from the exact normalization (it does not guarantee the full basket sums to exactly portfolio_value), but it captures the same inverse-vol spirit and is sufficient for execution confirmation.

---

## 4. Summary Comparison

| Property | FixedCashSizer | EquityPercentSizer | VolatilityTargetedSizer |
|---|---|---|---|
| Position $$ | Fixed at $100K | 1/N × equity | (1/vol_i) / Σ(1/vol_j) × equity |
| Scales with equity growth | ❌ No | ✅ Yes | ✅ Yes |
| Cash drag as equity grows | ❌ Growing idle cash | ✅ Minimal (rounding only) | ✅ Minimal (rounding only) |
| All names treated equally | ✅ Equal dollars | ✅ Equal dollars | ❌ High-vol names get less |
| Margin rejection risk | ✅ Medium (top-N at capacity) | ✅ Low | ✅ Low |
| Vector ↔ Backtrader agreement | Good | Good | Approximate |

---

## 5. The Long-Only Guarantee

All strategies are strictly long-only. This is enforced by:

1. **Vector engine**: only `buy` actions, no short positions. `selected` is always the top-scoring subset — never a negative-weight position.
2. **Backtrader**: `FixedCashSizer`, `EquityPercentSizer`, and `VolatilityTargetedSizer` all return `max(size, 0)` — a non-negative number of shares. They never return negative sizes.
3. **Post-run assertion**: `assert_backtrader_long_only(bt, strategy_name)` checks that every position in the Backtrader position log is non-negative. This is run automatically at the end of every focused script and raises an `AssertionError` if any negative position is found.

The assertion has passed for all strategies: base, improved 1–9.
