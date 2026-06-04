"""test_risk.py — RiskAgent VETO authority (STRATEGY_RULES §8 + INV-3/4).

No network, no orders. Builds Proposals + fake broker objects (string numeric fields) and
asserts every cap:
  - max_open_positions (>= 4 open => reject)
  - max_total_exposure (> $500 => reject or REDUCE to remaining room)
  - per-symbol cooldown (within 10 min => reject)
  - duplicate open order (A17 => reject, entries AND exits)
  - REDUCE path (partial room => adjusted_notional == room)
  - EXIT/SELL always approved
  - paper guard (ALPACA_PAPER != "true" => reject)
  - HOLD proposal => reject("no_action")
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional

import pytest

from agentic_trader.agents.risk_agent import RiskAgent
from agentic_trader.config import RiskLimits, Settings, StrategyParams
from agentic_trader.types import (
    Intent,
    Proposal,
    RiskDecision,
    Side,
    Verdict,
)

from . import FakeAccount, FakeOrder, FakePosition

LIMITS = RiskLimits()  # max_open_positions=4, max_total_exposure=500, cooldown=10min
PARAMS = StrategyParams()
ACCOUNT = FakeAccount()


def _buy(symbol: str = "SPY", notional: float = 100.0) -> Proposal:
    return Proposal(symbol=symbol, side=Side.BUY, intent=Intent.ENTRY,
                    reason="test entry", notional=notional)


def _sell(symbol: str = "SPY", qty: float = 1.0) -> Proposal:
    return Proposal(symbol=symbol, side=Side.SELL, intent=Intent.EXIT,
                    reason="test exit", qty=qty)


def _hold(symbol: str = "SPY") -> Proposal:
    return Proposal(symbol=symbol, side=Side.HOLD, intent=Intent.NONE, reason="hold")


def _agent(monkeypatch) -> RiskAgent:
    """A RiskAgent forced into a paper env (so the paper guard does not short-circuit)."""
    monkeypatch.setattr("agentic_trader.agents.risk_agent.is_paper_env", lambda *_a, **_k: True)
    return RiskAgent(LIMITS, settings=Settings())


# --------------------------------------------------------------------------- #
# Paper guard (INV-3)
# --------------------------------------------------------------------------- #
def test_reject_when_not_paper_env(monkeypatch):
    monkeypatch.setattr("agentic_trader.agents.risk_agent.is_paper_env", lambda *_a, **_k: False)
    agent = RiskAgent(LIMITS, settings=Settings())
    dec = agent.review(_buy(), ACCOUNT, [], [], {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "not_a_paper_account"
    assert not dec.approved


# --------------------------------------------------------------------------- #
# HOLD never executes
# --------------------------------------------------------------------------- #
def test_hold_proposal_rejected(monkeypatch):
    agent = _agent(monkeypatch)
    dec = agent.review(_hold(), ACCOUNT, [], [], {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "no_action"


# --------------------------------------------------------------------------- #
# EXIT / SELL always approved
# --------------------------------------------------------------------------- #
def test_exit_always_approved(monkeypatch):
    agent = _agent(monkeypatch)
    # Even at full positions / max exposure, an exit is risk-reducing and approved.
    positions = [FakePosition(f"S{i}", market_value="125") for i in range(4)]
    dec = agent.review(_sell("SPY", qty=3.0), ACCOUNT, positions, [], {}, PARAMS)
    assert dec.verdict is Verdict.APPROVE
    assert dec.approved
    assert dec.reason == "exit_approved"


# --------------------------------------------------------------------------- #
# max_open_positions
# --------------------------------------------------------------------------- #
def test_reject_when_max_open_positions(monkeypatch):
    agent = _agent(monkeypatch)
    positions = [FakePosition(f"S{i}", market_value="1") for i in range(4)]  # 4 == cap
    dec = agent.review(_buy("AAPL"), ACCOUNT, positions, [], {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "max_open_positions"


def test_pending_approved_buys_count_toward_open_slots(monkeypatch):
    """Three open + one already-approved this-cycle buy == 4 -> next entry rejected (A14)."""
    agent = _agent(monkeypatch)
    positions = [FakePosition(f"S{i}", market_value="1") for i in range(3)]
    prior = RiskDecision(_buy("MSFT"), Verdict.APPROVE, "entry_approved")
    dec = agent.review(_buy("AAPL"), ACCOUNT, positions, [], {}, PARAMS, approved_buys=[prior])
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "max_open_positions"


# --------------------------------------------------------------------------- #
# Duplicate open order guard (A17)
# --------------------------------------------------------------------------- #
def test_reject_duplicate_open_order_entry(monkeypatch):
    agent = _agent(monkeypatch)
    open_orders = [FakeOrder("AAPL")]
    dec = agent.review(_buy("AAPL"), ACCOUNT, [], open_orders, {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "open_order_pending"


def test_reject_duplicate_open_order_exit(monkeypatch):
    """The duplicate guard applies to exits too (never stack a second order)."""
    agent = _agent(monkeypatch)
    open_orders = [FakeOrder("SPY")]
    dec = agent.review(_sell("SPY", qty=1.0), ACCOUNT, [], open_orders, {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "open_order_pending"


# --------------------------------------------------------------------------- #
# Per-symbol cooldown
# --------------------------------------------------------------------------- #
def test_reject_within_cooldown(monkeypatch):
    agent = _agent(monkeypatch)
    recent = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=5)  # < 10 min
    dec = agent.review(_buy("SPY"), ACCOUNT, [], [], {"SPY": recent}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "cooldown"


def test_allowed_after_cooldown(monkeypatch):
    agent = _agent(monkeypatch)
    old = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=11)  # > 10 min
    dec = agent.review(_buy("SPY"), ACCOUNT, [], [], {"SPY": old}, PARAMS)
    assert dec.verdict is Verdict.APPROVE
    assert dec.reason == "entry_approved"


# --------------------------------------------------------------------------- #
# Exposure cap + REDUCE path (A13)
# --------------------------------------------------------------------------- #
def test_reject_when_exposure_room_zero(monkeypatch):
    """Open MV already at the $500 cap => no room => reject."""
    agent = _agent(monkeypatch)
    positions = [FakePosition("X", market_value="500")]
    dec = agent.review(_buy("SPY"), ACCOUNT, positions, [], {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "max_total_exposure"


def test_reduce_to_remaining_room(monkeypatch):
    """Open MV 450 + a $100 buy would breach $500 => REDUCE to the $50 room."""
    agent = _agent(monkeypatch)
    positions = [FakePosition("X", market_value="450")]
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, positions, [], {}, PARAMS)
    assert dec.verdict is Verdict.REDUCE
    assert dec.adjusted_notional == pytest.approx(50.0)
    assert dec.effective_notional == pytest.approx(50.0)
    assert dec.approved  # REDUCE is still an approval


def test_reject_when_room_below_min_notional(monkeypatch):
    """Room < MIN_NOTIONAL ($1) => reject rather than submit a sub-minimum order."""
    agent = _agent(monkeypatch)
    positions = [FakePosition("X", market_value="499.5")]  # room 0.5 < 1.0
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, positions, [], {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "exposure_room_below_min"


def test_approve_within_exposure(monkeypatch):
    """Open MV 100 + a $100 buy = 200 <= 500 => full approval at requested notional."""
    agent = _agent(monkeypatch)
    positions = [FakePosition("X", market_value="100")]
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, positions, [], {}, PARAMS)
    assert dec.verdict is Verdict.APPROVE
    assert dec.effective_notional == pytest.approx(100.0)


def test_approved_buys_consume_exposure(monkeypatch):
    """A prior approved $400 buy + open $50 leaves $50 room => REDUCE the next $100 buy."""
    agent = _agent(monkeypatch)
    positions = [FakePosition("X", market_value="50")]
    prior = RiskDecision(_buy("MSFT", notional=400.0), Verdict.APPROVE, "entry_approved")
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, positions, [], {}, PARAMS,
                       approved_buys=[prior])
    assert dec.verdict is Verdict.REDUCE
    assert dec.adjusted_notional == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# Account sanity + a logged-reason invariant
# --------------------------------------------------------------------------- #
def test_reject_blocked_account(monkeypatch):
    agent = _agent(monkeypatch)
    blocked = FakeAccount(trading_blocked=True)
    dec = agent.review(_buy("SPY"), blocked, [], [], {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "trading_blocked"


def test_every_decision_has_a_reason(monkeypatch):
    """INV-4: every RiskDecision carries a non-empty reason (approve or reject)."""
    agent = _agent(monkeypatch)
    for prop, pos, orders, cds in [
        (_buy("SPY"), [], [], {}),
        (_sell("SPY"), [], [], {}),
        (_hold("SPY"), [], [], {}),
        (_buy("SPY"), [FakePosition(f"S{i}") for i in range(4)], [], {}),
    ]:
        dec = agent.review(prop, ACCOUNT, pos, orders, cds, PARAMS)
        assert isinstance(dec.reason, str) and dec.reason


# --------------------------------------------------------------------------- #
# Finding #6 (HIGH): pending OPEN BUY orders from PRIOR cycles must count toward
# BOTH the exposure cap and the position-slot count (cross-cycle breach guard).
# A pending BUY is committed capital / an occupied slot even before it fills.
# --------------------------------------------------------------------------- #
@dataclass
class FakeBuyOrder:
    """An OPEN market BUY order at the broker (side + committed notional, string fields)."""
    symbol: str
    side: str = "buy"          # mirrors alpaca OrderSide.value
    notional: str = "100"
    qty: Optional[str] = None
    id: str = "ord-buy"


def test_pending_open_buy_order_notional_counts_toward_exposure(monkeypatch):
    """4 unfilled $100 BUY orders from prior cycles (other symbols) = $400 committed;
    a new $100 entry on a DIFFERENT symbol would breach $500 -> REDUCE to the $100 room."""
    agent = _agent(monkeypatch)
    open_buys = [FakeBuyOrder(s) for s in ("QQQ", "IWM", "MSFT", "NVDA")]  # $400 committed
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, [], open_buys, {}, PARAMS)
    # 4 pending BUY orders also occupy 4 slots == max_open_positions -> rejected on COUNT first.
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "max_open_positions"


def test_pending_open_buy_order_counts_toward_exposure_room(monkeypatch):
    """Three $150 pending BUYs = $450 committed (3 slots < 4); a $100 SPY entry would
    breach $500 -> REDUCE to the $50 remaining room (not approved at full notional)."""
    agent = _agent(monkeypatch)
    open_buys = [FakeBuyOrder(s, notional="150") for s in ("QQQ", "IWM", "MSFT")]  # $450
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, [], open_buys, {}, PARAMS)
    assert dec.verdict is Verdict.REDUCE
    assert dec.adjusted_notional == pytest.approx(50.0)


def test_pending_open_buy_order_counts_toward_slot_count(monkeypatch):
    """One open position + three pending BUY orders (other symbols) == 4 slots ->
    a new entry is rejected on the position-count cap even though nothing has filled."""
    agent = _agent(monkeypatch)
    positions = [FakePosition("AMZN", market_value="50")]
    open_buys = [FakeBuyOrder(s, notional="50") for s in ("QQQ", "IWM", "MSFT")]
    dec = agent.review(_buy("SPY", notional=50.0), ACCOUNT, positions, open_buys, {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "max_open_positions"


def test_own_symbol_pending_buy_not_double_counted(monkeypatch):
    """A pending BUY on the SAME symbol as the proposal is blocked by the A17 duplicate
    guard FIRST (open_order_pending), not silently double-counted into exposure."""
    agent = _agent(monkeypatch)
    open_buys = [FakeBuyOrder("SPY", notional="100")]
    dec = agent.review(_buy("SPY", notional=100.0), ACCOUNT, [], open_buys, {}, PARAMS)
    assert dec.verdict is Verdict.REJECT
    assert dec.reason == "open_order_pending"
