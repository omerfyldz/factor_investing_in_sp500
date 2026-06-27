"""Alpaca paper execution: rebalance the account to target dollar weights.

Uses NOTIONAL (dollar-denominated) market orders, so we can target weights
directly without share-count math. Sells/reductions run before buys so freed
cash is available for purchases. Trend exits are signal-based (a sleeve drops
out when it falls below its 10-month SMA) — there are deliberately NO bracket
or stop orders, which would fight the strategy.
"""
from __future__ import annotations

import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def make_client(api_key: str, secret_key: str) -> TradingClient:
    return TradingClient(api_key, secret_key, paper=True)


def get_equity_and_positions(client: TradingClient) -> tuple[float, dict[str, float]]:
    """Return (account equity, {symbol: current market value})."""
    equity = float(client.get_account().equity)
    positions = {p.symbol: float(p.market_value) for p in client.get_all_positions()}
    return equity, positions


def _submit_notional(client: TradingClient, symbol: str, notional: float, side: OrderSide) -> None:
    client.submit_order(MarketOrderRequest(
        symbol=symbol,
        notional=round(notional, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    ))


def build_plan(equity: float, positions: dict[str, float], target_weights: dict[str, float]) -> list[dict]:
    """Compute per-symbol current/target dollars and the delta to trade."""
    targets = {sym: w * equity for sym, w in target_weights.items()}
    plan = []
    for sym in sorted(set(targets) | set(positions)):
        cur = positions.get(sym, 0.0)
        tgt = targets.get(sym, 0.0)
        plan.append({"symbol": sym, "current": cur, "target": tgt, "delta": tgt - cur})
    return plan


def rebalance(
    client: TradingClient,
    target_weights: dict[str, float],
    min_trade_usd: float = 20.0,
    dry_run: bool = False,
) -> dict:
    """Rebalance the account to ``target_weights``. Returns a summary dict."""
    equity, positions = get_equity_and_positions(client)
    plan = build_plan(equity, positions, target_weights)

    print(f"  Account equity: ${equity:,.2f}")
    for row in plan:
        print(f"    {row['symbol']:<5} cur ${row['current']:>11,.0f} -> "
              f"tgt ${row['target']:>11,.0f}  (Δ ${row['delta']:>+11,.0f})")

    # --- Sells / reductions first (frees buying power) ---
    for row in plan:
        if row["delta"] < -min_trade_usd:
            sym, cur, tgt = row["symbol"], row["current"], row["target"]
            if tgt <= min_trade_usd:
                print(f"  CLOSE {sym} (${cur:,.0f} -> $0)")
                if not dry_run:
                    try:
                        client.close_position(sym)
                    except Exception as exc:
                        print(f"    WARN close {sym}: {exc}")
                    time.sleep(0.3)
            else:
                print(f"  SELL  {sym} ${-row['delta']:,.0f}")
                if not dry_run:
                    try:
                        _submit_notional(client, sym, -row["delta"], OrderSide.SELL)
                    except Exception as exc:
                        print(f"    WARN sell {sym}: {exc}")
                    time.sleep(0.3)

    # --- Buys / additions next ---
    for row in plan:
        if row["delta"] > min_trade_usd:
            sym = row["symbol"]
            print(f"  BUY   {sym} ${row['delta']:,.0f}")
            if not dry_run:
                try:
                    _submit_notional(client, sym, row["delta"], OrderSide.BUY)
                except Exception as exc:
                    print(f"    WARN buy {sym}: {exc}")
                time.sleep(0.3)

    return {"equity": equity, "plan": plan}
