#!/usr/bin/env python3
"""Tiny CLI wrapper around the official alpaca-py SDK for the PAPER account.

Usage:
    python3 alpaca_cli.py account      # connect + show account summary
    python3 alpaca_cli.py positions    # list open positions
    python3 alpaca_cli.py orders       # list recent orders
    python3 alpaca_cli.py clock        # market open/close status
"""
import argparse
import os
import sys
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def load_env() -> None:
    """Minimal .env loader (no extra dependency)."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        sys.exit(
            "No .env found. Copy .env.example to .env and add your paper API keys."
        )
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_client() -> TradingClient:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
    if not key or not secret or "your_paper" in key:
        sys.exit("ALPACA_API_KEY_ID / ALPACA_API_SECRET not set in .env")
    return TradingClient(key, secret, paper=paper)


def cmd_account(client: TradingClient) -> None:
    a = client.get_account()
    print(f"Account:      {a.account_number}  ({a.status})")
    print(f"Mode:         {'PAPER' if client._sandbox else 'LIVE'}")
    print(f"Currency:     {a.currency}")
    print(f"Cash:         {a.cash}")
    print(f"Buying power: {a.buying_power}")
    print(f"Portfolio:    {a.portfolio_value}")
    print(f"Blocked:      trading={a.trading_blocked}  account={a.account_blocked}")


def cmd_positions(client: TradingClient) -> None:
    positions = client.get_all_positions()
    if not positions:
        print("No open positions.")
        return
    for p in positions:
        print(f"{p.symbol:<8} qty={p.qty:<10} mkt_value={p.market_value:<14} "
              f"unrealized_pl={p.unrealized_pl}")


def cmd_orders(client: TradingClient) -> None:
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=20)
    orders = client.get_orders(filter=req)
    if not orders:
        print("No orders.")
        return
    for o in orders:
        print(f"{o.submitted_at}  {o.side.value:<4} {o.qty} {o.symbol:<8} "
              f"{o.type.value:<8} {o.status.value}")


def cmd_clock(client: TradingClient) -> None:
    c = client.get_clock()
    print(f"Market is {'OPEN' if c.is_open else 'CLOSED'}")
    print(f"Now:        {c.timestamp}")
    print(f"Next open:  {c.next_open}")
    print(f"Next close: {c.next_close}")


COMMANDS = {
    "account": cmd_account,
    "positions": cmd_positions,
    "orders": cmd_orders,
    "clock": cmd_clock,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca paper-account CLI")
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args()

    load_env()
    client = get_client()
    COMMANDS[args.command](client)


if __name__ == "__main__":
    main()
