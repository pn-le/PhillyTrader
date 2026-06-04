"""test_orchestrator.py — cooldown seeding + per-symbol cooldown timestamp resolution.

No network, no orders. Regression coverage for the per-symbol cooldown (A5) bugs:
  - Finding #3/#8 (HIGH): the cooldown store is SEEDED from broker order/position history
    at process start, so the 10-min entry cap survives restarts and single-shot `run`
    invocations (it is NOT silently empty).
  - Finding #7 (HIGH): a just-approved entry records its cooldown using the SYMBOL'S OWN
    decision-bar timestamp (or the live UTC clock when the snapshot has no real bars), NOT
    the shared snapshots[0].asof — so a data gap on the first universe symbol can't make the
    cooldown look expired next cycle.

Together with test_risk.py's cooldown-window tests, these prove a freshly-restarted cycle
still rejects a same-symbol entry within 10 minutes of the last broker BUY.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from agentic_trader.agents.risk_agent import RiskAgent
from agentic_trader.config import RiskLimits, Settings, StrategyParams
from agentic_trader.orchestrator import _entry_cooldown_ts, seed_cooldowns
from agentic_trader.types import (
    Indicators,
    Intent,
    PositionState,
    PositionStatus,
    Proposal,
    Side,
    Snapshot,
    Verdict,
)

from . import bars_session, ny

LIMITS = RiskLimits()
PARAMS = StrategyParams()


# --------------------------------------------------------------------------- #
# Fake broker objects for seeding from order history
# --------------------------------------------------------------------------- #
@dataclass
class _FakeClosedBuy:
    symbol: str
    filled_at: _dt.datetime
    side: Any = None  # _entry_times_from_orders does not filter on side (the request does)


class _FakeTrading:
    """Returns a fixed list of CLOSED filled BUY orders from get_orders(filter=...)."""

    def __init__(self, orders: List[_FakeClosedBuy]) -> None:
        self._orders = orders

    def get_orders(self, filter=None):  # noqa: A002 — mirror the alpaca kw name
        return list(self._orders)


# --------------------------------------------------------------------------- #
# Finding #3 / #8: cooldowns seeded from broker history
# --------------------------------------------------------------------------- #
def test_seed_cooldowns_from_recent_broker_buys():
    """seed_cooldowns rebuilds the cooldown map from recent filled BUY orders."""
    t0 = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=3)
    clients = {"trading": _FakeTrading([_FakeClosedBuy("SPY", t0), _FakeClosedBuy("AAPL", t0)])}
    seeded = seed_cooldowns(clients, position_states={})
    assert seeded.get("SPY") == t0
    assert seeded.get("AAPL") == t0


def test_seeded_cooldown_blocks_restart_reentry_within_window(monkeypatch):
    """A freshly-restarted cycle (empty in-memory map) seeded from a 3-min-old broker BUY
    must STILL reject a same-symbol entry — the 10-min cap survives the restart."""
    t_recent = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=3)  # < 10 min
    clients = {"trading": _FakeTrading([_FakeClosedBuy("SPY", t_recent)])}
    cooldowns = seed_cooldowns(clients, position_states={})

    # Force a paper env so the paper guard does not short-circuit the cooldown check.
    monkeypatch.setattr("agentic_trader.agents.risk_agent.is_paper_env", lambda *_a, **_k: True)
    agent = RiskAgent(LIMITS, settings=Settings())
    buy = Proposal(symbol="SPY", side=Side.BUY, intent=Intent.ENTRY,
                   reason="entry", notional=100.0)
    dec = agent.review(buy, None, [], [], cooldowns, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "cooldown"


def test_seed_cooldowns_uses_open_position_entry_time():
    """An open position whose BUY order has aged out is still covered via its entry_time."""
    et = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=2)
    clients = {"trading": _FakeTrading([])}  # no recent orders
    pstates = {"MSFT": PositionState(symbol="MSFT", status=PositionStatus.LONG, qty=1.0,
                                     entry_time=et)}
    seeded = seed_cooldowns(clients, position_states=pstates)
    assert seeded.get("MSFT") == et


def test_seed_cooldowns_normalizes_naive_entry_time():
    """A tz-naive seed timestamp is normalized to UTC (RiskAgent compares against UTC now)."""
    naive = _dt.datetime(2024, 6, 3, 9, 45, 0)
    clients = {"trading": _FakeTrading([_FakeClosedBuy("SPY", naive)])}
    seeded = seed_cooldowns(clients, position_states={})
    assert seeded["SPY"].tzinfo is not None


# --------------------------------------------------------------------------- #
# Finding #7: cooldown timestamp uses the symbol's OWN snapshot, never the 09:30
# session-open fallback of a bars-less snapshot.
# --------------------------------------------------------------------------- #
def _snapshot(symbol: str, asof: _dt.datetime, *, with_bars: bool) -> Snapshot:
    bars = bars_session(symbol, [100.0] * 3, [1000.0] * 3, start=asof) if with_bars else []
    ind = Indicators(symbol=symbol, session_vwap=100.0, last_price=100.0,
                     dist_from_vwap=0.0, current_volume=1000.0, rolling20_avg_vol=1000.0,
                     volume_ratio=1.0, n_bars=len(bars), valid=bool(bars))
    return Snapshot(symbol=symbol, asof=asof, indicators=ind,
                    position_state=PositionState(symbol=symbol), bars=tuple(bars))


def test_entry_cooldown_ts_uses_symbol_snapshot_when_bars_present():
    """A snapshot WITH real bars contributes its own decision-bar asof as the cooldown ts."""
    asof = ny(hour=14, minute=0)
    snap = _snapshot("AAPL", asof, with_bars=True)
    ts = _entry_cooldown_ts(snap, now=None)
    assert ts == asof


def test_entry_cooldown_ts_ignores_barless_session_open_fallback():
    """A bars-less snapshot's asof is the 09:30 session-open FALLBACK — it must NOT be used
    (that would look ~hours stale next cycle and bypass the cooldown). Use the live clock."""
    session_open = ny(hour=9, minute=30)   # the market_data fallback when a symbol has no bars
    now = _dt.datetime.now(_dt.timezone.utc)
    snap = _snapshot("SPY", session_open, with_bars=False)
    ts = _entry_cooldown_ts(snap, now=now)
    assert ts == now            # used the live clock, NOT the 09:30 fallback
    assert ts != session_open


def test_entry_cooldown_ts_falls_back_to_now_when_no_snapshot():
    """No snapshot at all -> the live UTC clock (never None / never a stale value)."""
    now = _dt.datetime(2024, 6, 3, 18, 0, 0, tzinfo=_dt.timezone.utc)
    ts = _entry_cooldown_ts(None, now=now)
    assert ts == now
