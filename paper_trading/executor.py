"""
Alpaca paper trading execution layer.

Handles rebalancing: close positions that left the target list and
open new ones with bracket stop-loss / take-profit orders.
"""
from __future__ import annotations

import time

import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)


def make_client(api_key: str, secret_key: str) -> TradingClient:
    """Create an Alpaca paper trading client."""
    return TradingClient(api_key, secret_key, paper=True)


def get_current_holdings(client: TradingClient) -> set[str]:
    """Return the set of tickers currently held in the paper account."""
    positions = client.get_all_positions()
    return {p.symbol for p in positions}


def get_position_qty(client: TradingClient, symbol: str) -> float:
    """Return the number of shares held for a symbol (0 if not held)."""
    positions = client.get_all_positions()
    for p in positions:
        if p.symbol == symbol:
            return float(p.qty)
    return 0.0


def cancel_open_orders_for(client: TradingClient, symbol: str) -> None:
    """Cancel all open orders for a specific symbol."""
    orders = client.get_orders()
    for order in orders:
        if order.symbol == symbol:
            try:
                client.cancel_order_by_id(str(order.id))
            except Exception:
                pass


def get_latest_price(ticker: str) -> float:
    """Fetch the latest market price via yfinance (used for qty calculation)."""
    info = yf.Ticker(ticker).fast_info
    price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
    if price is None or price <= 0:
        raise ValueError(f"Could not fetch a valid price for {ticker}")
    return float(price)


def _compute_qty(ticker: str, sizing: str, cfg: dict, account_equity: float) -> int:
    """Determine number of shares to buy based on sizing rule."""
    price = get_latest_price(ticker)
    if sizing == "fixed_cash":
        qty = int(cfg["cash_per_trade"] / price)
    else:  # pct_equity
        target_value = account_equity * cfg["equity_pct"]
        qty = int(target_value / price)
    return max(qty, 1)


def close_position(client: TradingClient, ticker: str, dry_run: bool = False) -> None:
    """Cancel any open orders and close the position at market."""
    print(f"    EXIT {ticker}")
    if dry_run:
        return
    cancel_open_orders_for(client, ticker)
    qty = get_position_qty(client, ticker)
    if qty > 0:
        order = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        client.submit_order(order)
        time.sleep(0.3)


def open_bracket_position(
    client: TradingClient,
    ticker: str,
    qty: int,
    entry_price: float,
    stop_pct: float,
    take_pct: float,
    dry_run: bool = False,
) -> None:
    """Open a new long position with bracket stop-loss and take-profit orders."""
    stop_price = round(entry_price * (1 - stop_pct), 2)
    take_price = round(entry_price * (1 + take_pct), 2)
    print(
        f"    ENTER {ticker}  qty={qty}  entry≈${entry_price:.2f}"
        f"  stop=${stop_price:.2f} ({stop_pct:.0%})  take=${take_price:.2f} ({take_pct:.0%})"
    )
    if dry_run:
        return
    order = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_price),
        stop_loss=StopLossRequest(stop_price=stop_price),
    )
    client.submit_order(order)
    time.sleep(0.3)


def rebalance(
    client: TradingClient,
    current_holdings: set[str],
    target_tickers: list[str],
    cfg: dict,
    dry_run: bool = False,
) -> list[str]:
    """
    Rebalance the Alpaca paper account to match the target ticker list.

    1. Close positions in current_holdings that are NOT in target.
    2. Open new positions for tickers in target that are NOT currently held.
    3. Tickers in both sets are held as-is (bracket orders remain active).

    Returns the final list of tickers held after rebalancing.
    """
    target_set = set(target_tickers)
    exits = current_holdings - target_set
    entries = target_set - current_holdings
    holds = current_holdings & target_set

    print(f"  Current holdings ({len(current_holdings)}): {sorted(current_holdings)}")
    print(f"  Target tickers  ({len(target_tickers)}): {sorted(target_set)}")
    print(f"  Exits: {sorted(exits)}")
    print(f"  Entries: {sorted(entries)}")
    print(f"  Holds (no action): {sorted(holds)}")

    # --- Close exits ---
    for ticker in sorted(exits):
        close_position(client, ticker, dry_run=dry_run)

    # --- Get account equity for percent-of-equity sizing ---
    account_equity = 1_000_000.0  # fallback
    if not dry_run:
        try:
            account = client.get_account()
            account_equity = float(account.equity)
        except Exception:
            pass

    # --- Open entries ---
    for ticker in sorted(entries):
        try:
            price = get_latest_price(ticker)
            qty = _compute_qty(ticker, cfg["sizing"], cfg, account_equity)
            open_bracket_position(
                client,
                ticker,
                qty=qty,
                entry_price=price,
                stop_pct=cfg["stop_pct"],
                take_pct=cfg["take_pct"],
                dry_run=dry_run,
            )
        except Exception as exc:
            print(f"    WARNING: could not enter {ticker}: {exc}")

    return list(target_set)
