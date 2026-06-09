# Improved 4 Stop/Take Sensitivity

Improved 4 is a focused risk-exit parameter experiment. It does not replace the base, improved 1, improved 2, or improved 3 results.

## Method

- Foundation: improved 2 static equal-weight factor signals.
- Changed variable: stop-loss and take-profit thresholds only.
- Training window used for selection: observations through `2020-12-31`.
- Test-period metrics are reported after selection and are not used to pick the winner.
- Selection score: training Sharpe, penalized for training drawdown and isolated parameter peaks.
- Intrabar warning: monthly vector results use OHLC approximations; executable evidence comes from Backtrader.
- Execution check: the selected candidate is run with daily Backtrader adjusted OHLC data, market entries/rebalances, and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.

## Selected Candidate

- Stop-loss: `5.0%`
- Take-profit: `30.0%`
- Train Sharpe: `0.6991`
- Train max drawdown: `-7.36%`
- Test Sharpe: `1.0221`
- Test max drawdown: `-6.02%`
- Backtrader final value: `$2,816,266`
- Backtrader Sharpe: `0.9852`
- Backtrader max drawdown: `-7.69%`

## Comparison To Improved 2 Original 10%/20%

- Original train Sharpe: `0.6269`
- Original train max drawdown: `-13.03%`
- Original test Sharpe: `0.9588`
- Original test max drawdown: `-5.75%`

## Warning

This is still a parameter search. The selected pair should be described as a candidate from a constrained walk-forward-style sensitivity test, not as a proven optimal rule. The more important result is whether nearby parameter pairs behave similarly.
