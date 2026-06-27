"""Performance report for the trend-following paper account.

Usage (from trend_following/):  python -m live.report
Prints account equity, current positions and weights, and SPY over the same
period since the first rebalance for a rough benchmark.
"""
from __future__ import annotations

import warnings
from datetime import date

import yfinance as yf

from live.config import ALPACA_KEY, ALPACA_SECRET
from live.state import load_state


def _spy_return_since(since: str) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spy = yf.download("SPY", start=since, auto_adjust=True, progress=False)["Close"]
        return None if spy.empty else float(spy.iloc[-1] / spy.iloc[0] - 1)
    except Exception:
        return None


def main() -> None:
    state = load_state()
    last = state.get("last_rebalance")
    print("=" * 56)
    print(f"  Trend Following — Report  ({date.today().isoformat()})")
    print(f"  Last rebalance: {last or 'never'}")
    print("=" * 56)

    tw = state.get("target_weights", {})
    if tw:
        print("\n  Target weights (last rebalance):")
        for k, v in sorted(tw.items(), key=lambda x: -x[1]):
            if v > 1e-9:
                print(f"    {k:<5} {v:6.2%}")

    if not ALPACA_KEY or not ALPACA_SECRET:
        print("\n  No Alpaca credentials — state-file view only.")
        return

    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
        acct = client.get_account()
        equity = float(acct.equity)
        pnl = equity - 1_000_000
        print(f"\n  Equity:     ${equity:,.2f}")
        print(f"  P&L vs $1M: ${pnl:+,.2f}  ({pnl / 1_000_000:+.2%})")

        positions = client.get_all_positions()
        if positions:
            print(f"\n  Open positions ({len(positions)}):")
            for p in sorted(positions, key=lambda x: -float(x.market_value)):
                print(f"    {p.symbol:<5} ${float(p.market_value):>11,.0f}  "
                      f"({float(p.market_value) / equity:5.1%})  "
                      f"P&L ${float(p.unrealized_pl):+,.0f}")
        else:
            print("\n  No open positions.")

        if last:
            spy = _spy_return_since(last)
            if spy is not None:
                print(f"\n  Since {last}:  strategy {pnl / 1_000_000:+.2%}  |  SPY {spy:+.2%}")
    except Exception as exc:
        print(f"  Alpaca error: {exc}")


if __name__ == "__main__":
    main()
