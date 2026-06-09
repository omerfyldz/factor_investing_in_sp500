# Improved 5 Regime Filter

Improved 5 is a focused market-regime experiment. It does not replace improved 4; it tests one extra risk-management rule on top of improved 4.

## Method

- Foundation: improved 4 factor signals and stop/take thresholds.
- Stop-loss: `5.0%`.
- Take-profit: `30.0%`.
- New rule: trade only when `^GSPC` month-end close is above its existing 10-month SMA.
- If the filter is off at signal month `t`, the strategy holds cash during month `t+1`.
- No moving-average window optimization is performed.
- Daily Backtrader execution uses adjusted OHLC data, market entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

## Results

- Vector Sharpe: `0.9336`.
- Vector final equity: `$2,148,104`.
- Vector max drawdown: `-11.19%`.
- Backtrader final value: `$2,109,769`.
- Backtrader Sharpe: `1.0609`.
- Backtrader max drawdown: `-10.64%`.
- Monte Carlo p-value: `0.0990`.

## Decision

Improved 5 is not accepted as a performance improvement over improved 4.

- Improved 4 vector Sharpe: `1.1504`; improved 5 vector Sharpe: `0.9336`.
- Improved 4 vector max drawdown: `-7.36%`; improved 5 vector max drawdown: `-11.19%`.
- Improved 4 Backtrader Sharpe: `1.3961`; improved 5 Backtrader Sharpe: `1.0609`.
- Improved 4 Backtrader max drawdown: `-7.69%`; improved 5 Backtrader max drawdown: `-10.64%`.

The filter likely removed too much exposure and missed rebound months. This is a useful failed experiment because it shows that a simple index cash filter is not automatically better once daily stop-loss/take-profit risk control is already present.

## Exposure

- Total months: `240`.
- Regime-on months: `179`.
- Regime-off months: `61`.
- Invested months: `85`.
- Cash months: `155`.

## Warning

This is a dangerous zone because market-regime filters can easily become market-timing overfit. The rule is deliberately fixed at the pre-existing 10-month SMA. We should judge it by whether it improves drawdown and robustness relative to improved 4 without destroying return, not by whether it maximizes full-sample performance.
