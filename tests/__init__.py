"""tests package — synthetic, network-free test fixtures + builders.

NO test in this package may hit the network or submit an order. Everything is driven by
synthetic `Bar` data and fake/injected clients. The helpers below build the exact inputs
the agents expect, bound to the real contract in `agentic_trader.types` / `.config`.

Conventions
-----------
- Bar.start is tz-aware America/New_York (per the Snapshot contract). `bars_session()`
  lays minute bars out from 09:30 NY by default.
- Numeric Alpaca broker fields are STRINGS (API_MAP §4). `FakePosition` / `FakeAccount`
  / `FakeOrder` mirror that so the Risk/Position agents exercise their float() parsing.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agentic_trader.config import UNIVERSE
from agentic_trader.types import (
    Bar,
    Indicators,
    PositionState,
    PositionStatus,
    Snapshot,
)

NY = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Synthetic bar builders
# --------------------------------------------------------------------------- #
def ny(year: int = 2024, month: int = 6, day: int = 3,
       hour: int = 9, minute: int = 30, second: int = 0) -> _dt.datetime:
    """A tz-aware America/New_York datetime (default 09:30 of a regular trading day)."""
    return _dt.datetime(year, month, day, hour, minute, second, tzinfo=NY)


def make_bar(symbol: str, start: _dt.datetime, *, open_: float, high: float,
             low: float, close: float, volume: float,
             trade_count: Optional[float] = None, vwap: Optional[float] = None) -> Bar:
    """Construct one Bar with explicit OHLCV (keyword-only to avoid arg-order mistakes)."""
    return Bar(
        symbol=symbol, start=start, open=open_, high=high, low=low,
        close=close, volume=volume, trade_count=trade_count, vwap=vwap,
    )


def bars_session(symbol: str, closes: List[float], volumes: List[float], *,
                 start: Optional[_dt.datetime] = None,
                 highs: Optional[List[float]] = None,
                 lows: Optional[List[float]] = None,
                 opens: Optional[List[float]] = None) -> List[Bar]:
    """Build a list of consecutive 1-minute bars from parallel close/volume series.

    Bars start at `start` (default 09:30 NY) and step +1 minute each. When highs/lows/opens
    are not supplied they default to close (a flat bar) so VWAP math is exactly the close.
    """
    assert len(closes) == len(volumes), "closes and volumes must be the same length"
    base = start or ny()
    bars: List[Bar] = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        h = highs[i] if highs is not None else c
        lo = lows[i] if lows is not None else c
        o = opens[i] if opens is not None else c
        bars.append(make_bar(symbol, base + _dt.timedelta(minutes=i),
                              open_=o, high=h, low=lo, close=c, volume=v))
    return bars


def flat_then_spike(symbol: str, *, n_flat: int = 20, flat_close: float = 100.0,
                    flat_vol: float = 1000.0, decision_close: float = 99.0,
                    decision_vol: float = 2000.0,
                    start: Optional[_dt.datetime] = None) -> List[Bar]:
    """`n_flat` identical flat bars followed by ONE decision bar (price drop + vol spike).

    With n_flat=20 this yields 21 bars total => n_bars == MIN_BARS_FOR_ENTRY (eligible),
    the rolling-20 window is exactly the flat bars, and the decision bar sits below VWAP
    with an elevated volume_ratio — the canonical entry-boundary fixture.
    """
    closes = [flat_close] * n_flat + [decision_close]
    vols = [flat_vol] * n_flat + [decision_vol]
    return bars_session(symbol, closes, vols, start=start)


# --------------------------------------------------------------------------- #
# Snapshot helpers
# --------------------------------------------------------------------------- #
def make_snapshot(symbol: str, bars: List[Bar], indicators: Indicators, *,
                  position_state: Optional[PositionState] = None,
                  asof: Optional[_dt.datetime] = None) -> Snapshot:
    """Assemble a Snapshot from bars + indicators (asof defaults to the last bar start)."""
    return Snapshot(
        symbol=symbol,
        asof=asof or (bars[-1].start if bars else ny()),
        indicators=indicators,
        position_state=position_state or PositionState(symbol=symbol),
        bars=tuple(bars),
    )


def long_position(symbol: str, *, qty: float = 1.0, avg_entry_price: float = 100.0,
                  entry_time: Optional[_dt.datetime] = None,
                  unrealized_plpc: Optional[float] = None) -> PositionState:
    """A LONG PositionState for exit tests."""
    return PositionState(
        symbol=symbol,
        status=PositionStatus.LONG,
        qty=qty,
        avg_entry_price=avg_entry_price,
        entry_time=entry_time,
        unrealized_plpc=unrealized_plpc,
    )


# --------------------------------------------------------------------------- #
# Fake Alpaca broker objects (numeric fields are STRINGS, like the real SDK)
# --------------------------------------------------------------------------- #
@dataclass
class FakePosition:
    """Mimics an alpaca Position: numeric fields are strings (API_MAP §4)."""

    symbol: str
    qty: str = "1"
    avg_entry_price: str = "100"
    current_price: str = "100"
    market_value: str = "100"
    unrealized_pl: str = "0"
    unrealized_plpc: str = "0"


@dataclass
class FakeOrder:
    """Mimics an open alpaca Order (only `symbol` is needed by the Risk guard)."""

    symbol: str
    id: str = "ord-0"


@dataclass
class FakeAccount:
    """Mimics an alpaca TradeAccount sanity surface."""

    account_blocked: bool = False
    trading_blocked: bool = False
    trade_suspended_by_user: bool = False
