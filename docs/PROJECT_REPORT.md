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
- Trend uses normalized moving-average deviations with windows `[3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]`. Coefficients come from the assignment-style full-sample `^GSPC` predictive regression, with insignificant variables dropped.

The paper-style HZZ cross-sectional trend factor is retained only as an appendix/history experiment because it changes the literal assignment trend construction.

## Results

Base strategy:

- final equity: `$4,585,176`;
- total return: `358.52%`;
- annualized Sharpe: `1.02`;
- max drawdown: `-15.45%`.

Final selected assignment-scope vector strategy:

- strategy: `value_quality_heavy_top10`;
- final equity: `$4,058,753`;
- total return: `305.88%`;
- annualized Sharpe: `1.06`;
- max drawdown: `-13.41%`.

Backtrader base strategy:

- final value: `$4,944,058`;
- total return: `394.41%`;
- annualized Sharpe: `1.13`;
- max drawdown: `-13.57%`.

Backtrader final strategy:

- final value: `$4,816,929`;
- total return: `381.69%`;
- annualized Sharpe: `1.16`;
- max drawdown: `-12.69%`.

Monte Carlo p-value for the base Sharpe: `0.0000`.

Monte Carlo p-value for the final selected strategy Sharpe: `0.0000`.

Final selected strategy annualized alpha versus `^GSPC`: `7.52%` with alpha t-stat `4.61`.

## What We Tried

- Equal-weight composite base strategy.
- More concentrated top-5 selection.
- Trend/momentum-heavy factor weights.
- Value/quality-heavy factor weights.
- Removing trend from the composite.
- S&P 500 regime filter.
- Backtrader stop-loss/take-profit improvement.
- Appendix only: paper-style HZZ trend, sector caps, and volatility-aware ranking.
- Walk-forward validation.

## Interpretation

The project should not claim statistically guaranteed skill only because the backtest is profitable. The Monte Carlo result is the main guardrail: if random portfolios often match or beat the strategy Sharpe, the conclusion must be cautious. The strongest investment conclusion is conditional: the factor process is economically sensible and reproducible, but real-money confidence would require survivorship-bias-free data, transaction-cost and slippage assumptions, and further out-of-sample testing.

The final strategy has stronger risk-adjusted performance than the base strategy, but it was selected after multiple experiments. This is why the presentation should emphasize the full strategy history and avoid claiming that the final result is a guaranteed tradable edge.

The vectorized improvement tests are used for fast screening. The executable trading results are the saved Backtrader runs, which use market orders, fixed cash sizing, the assignment's zero-commission setting, and stop/limit exits where applicable.
