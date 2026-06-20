"""
Performance report for both paper trading strategies.

Usage:
    python report.py

Prints current holdings, account equity, P&L since first trade,
and comparison to SPY over the same period.
"""
from __future__ import annotations

import warnings
from datetime import date

import yfinance as yf

from config import STRATEGIES
from state import load_state


def _fetch_spy_return(since: str) -> float | None:
    """Return SPY total return from `since` (ISO date string) to today."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spy = yf.download("SPY", start=since, auto_adjust=True, progress=False)["Close"]
        if spy.empty:
            return None
        return float(spy.iloc[-1] / spy.iloc[0] - 1)
    except Exception:
        return None


def print_alpaca_report(strategy_name: str, cfg: dict, start_date: str | None) -> None:
    """Connect to Alpaca and print equity + positions."""
    if not cfg["alpaca_key"] or not cfg["alpaca_secret"]:
        print(f"  [{strategy_name}] No Alpaca credentials — showing state file only.")
        return

    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(cfg["alpaca_key"], cfg["alpaca_secret"], paper=True)
        account = client.get_account()
        equity = float(account.equity)
        pnl = equity - 1_000_000  # Alpaca paper accounts start at $100K by default;
        # adjust initial_cash if you configured a different amount in Alpaca settings.

        print(f"\n  Strategy: {strategy_name}")
        print(f"    Equity:       ${equity:,.2f}")
        print(f"    P&L vs $1M:   ${pnl:+,.2f}  ({pnl / 1_000_000:+.2%})")

        positions = client.get_all_positions()
        if positions:
            print(f"    Open positions ({len(positions)}):")
            for p in sorted(positions, key=lambda x: x.symbol):
                unrealized = float(p.unrealized_pl) if p.unrealized_pl else 0.0
                print(
                    f"      {p.symbol:<6}  qty={p.qty:>5}  "
                    f"cost=${float(p.cost_basis):,.2f}  "
                    f"mkt=${float(p.market_value):,.2f}  "
                    f"P&L=${unrealized:+,.2f}"
                )
        else:
            print("    No open positions.")

        if start_date:
            spy_ret = _fetch_spy_return(start_date)
            if spy_ret is not None:
                strat_ret = pnl / 1_000_000
                print(f"\n    Since {start_date}:")
                print(f"      Strategy return: {strat_ret:+.2%}")
                print(f"      SPY return:      {spy_ret:+.2%}")
                print(f"      Excess:          {strat_ret - spy_ret:+.2%}")

    except Exception as exc:
        print(f"  [{strategy_name}] Alpaca error: {exc}")


def main() -> None:
    state = load_state()
    last_rebalance = state.get("last_rebalance")
    today = date.today().isoformat()

    print("=" * 60)
    print(f"  Paper Trading Performance Report — {today}")
    print(f"  Last rebalance: {last_rebalance or 'never'}")
    print("=" * 60)

    for strategy_name, cfg in STRATEGIES.items():
        holdings = state.get(strategy_name, {}).get("holdings", [])
        print(f"\n[{strategy_name}]  State-file holdings: {holdings}")
        print_alpaca_report(strategy_name, cfg, start_date=last_rebalance)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
