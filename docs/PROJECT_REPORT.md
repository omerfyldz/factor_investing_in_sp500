# Project Report: S&P 500 Factor Investing

## Assignment Alignment

This project follows the factor-investing section of the assignment: ROE, P/E, momentum, and trend factors; portfolio-sort and regression FMPs; IC tests; Backtrader backtests; Monte Carlo significance testing; and saved outputs.

The BIST100-specific index role is replaced with `^GSPC`, the S&P 500 index.

## Paper Summary

Han, Zhou, and Zhu (2016) propose a trend factor that combines short-, intermediate-, and long-horizon price information through normalized moving averages. Their paper argues that multiple moving-average horizons capture information diffusion, underreaction, overreaction, and feedback trading better than a single momentum horizon.

## Data Process

The raw stock files contain 503 current S&P 500 securities. Prices and daily ratios are filtered to `date <= 2026-05-31`. Statement fundamentals are filtered to `date_available <= 2026-05-31`, which prevents using financial statements before they became public.

The processed panel is monthly. Stock returns use adjusted prices. The final observation is the last trading date on or before May 2026.

## Factor Construction

- ROE uses the latest point-in-time `roe` statement field.
- P/E uses positive month-end `peRatio`, multiplied by `-1` so high factor scores mean cheap valuation.
- Momentum is `P_t / P_(t-12) - 1`.
- Trend uses normalized moving-average deviations with windows `[3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]`. Base coefficients come from the assignment-style full-sample `^GSPC` predictive regression, with insignificant variables dropped. Improved 1, improved 2, and improved 3 use expanding no-lookahead `^GSPC` regressions for the trend factor.
- Improved 3 factor weights are estimated from a rolling 60-month rank-IC information-ratio rule, shrunk 50% toward equal weights and capped between 10% and 45% per factor.

## Results

Base strategy:

- final equity: `$2,032,039`;
- total return: `103.20%`;
- annualized Sharpe: `1.20`;
- max drawdown: `-7.26%`.

Improved 3 vector strategy:

- strategy: `improved_3_dynamic_ic_weights_stop_take_top10`;
- final equity: `$2,635,221`;
- total return: `163.52%`;
- annualized Sharpe: `0.97`;
- max drawdown: `-14.13%`.

Backtrader base strategy:

- final value: `$5,778,291`;
- total return: `123.20%`;
- annualized Sharpe: `1.36`;
- max drawdown: `-6.80%`.

Backtrader improved 3 strategy:

- final value: `$2,680,533`;
- total return: `168.05%`;
- annualized Sharpe: `1.21`;
- max drawdown: `-15.51%`.

Monte Carlo p-value for the base Sharpe: `0.0130`.

Monte Carlo p-value for the improved 3 Sharpe: `0.1570`.

Improved 3 annualized alpha versus `^GSPC`: `11.16%` with alpha t-stat `3.11`.

## What We Tried

- Equal-weight composite base strategy with assignment-style full-sample trend regression.
- Improved 1: the same strategy with expanding no-lookahead trend regression.
- Improved 2: improved 1 plus 10% stop-loss and 20% take-profit.
- Improved 3: improved 2 plus rolling rank-IC factor weights with shrinkage and caps.
- Static value/quality-heavy factor weights, top-N changes, and other overlays are deferred so improvements remain one-at-a-time.
- Walk-forward validation.

## Interpretation

The project should not claim statistically guaranteed skill only because the backtest is profitable. The Monte Carlo result is the main guardrail: if random portfolios often match or beat the strategy Sharpe, the conclusion must be cautious. The strongest investment conclusion is conditional: the factor process is economically sensible and reproducible, but real-money confidence would require survivorship-bias-free data, transaction-cost and slippage assumptions, and further out-of-sample testing.

The improved strategies must be judged sequentially: improved 1 asks whether removing trend look-ahead improves the research design and/or results; improved 2 asks whether adding explicit risk exits improves the improved 1 profile; improved 3 asks whether past-only factor weighting improves improved 2. This is why the presentation should emphasize the full strategy history and avoid claiming that the final result is a guaranteed tradable edge.

The vectorized improvement tests are used for fast monthly screening. The executable Backtrader runs are long-only: base and improved 1 use monthly `bt.Order.Market` orders, while improved 2 and improved 3 use daily adjusted OHLC data with `bt.Order.Market` entries/rebalances and native `bt.Order.Stop` / `bt.Order.Limit` protective exits.
