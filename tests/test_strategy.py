"""test_strategy.py — StrategyAgent decision logic (STRATEGY_RULES §3/§4/§5).

PURE decision tests (no network, no orders):
  - ENTRY fires EXACTLY at the boundary (dist <= -0.5% AND volume_ratio >= 1.2x) and is
    suppressed when either condition is just inside it.
  - Each EXIT trigger (vwap_revert, max_hold > 15min, stop_loss -0.5%, eod_flatten) fires
    INDEPENDENTLY.
  - The ML gate blocks an otherwise-valid entry when p_win < ml_threshold, and passes when
    p_win >= ml_threshold; a None scorer is passthrough (p_win == 1.0).
"""

from __future__ import annotations

import datetime as _dt

import pytest

from agentic_trader.agents.strategy_agent import StrategyAgent
from agentic_trader.config import StrategyParams
from agentic_trader.data.market_data import compute_indicators
from agentic_trader.types import (
    ExitStatus,
    Indicators,
    Intent,
    PositionState,
    Side,
)

from . import (
    bars_session,
    flat_then_spike,
    long_position,
    make_snapshot,
    ny,
)

PARAMS = StrategyParams()  # entry_dist=0.005, vol_mult=1.2, max_hold=15, vwap_exit_band=0.001, stop_loss=0.005


# --------------------------------------------------------------------------- #
# Helpers: a flat snapshot engineered to a precise (dist_from_vwap, volume_ratio).
# --------------------------------------------------------------------------- #
def _entry_snapshot(symbol: str, *, dist: float, vol_ratio: float,
                    valid: bool = True) -> "tuple":
    """Build a FLAT snapshot whose indicators carry an exact dist & volume_ratio.

    Bars are synthesized so build_features() (called inside the strategy on entry) has the
    >= K+1 bars it needs and consistent rolling/last values. We override the Indicators
    explicitly to hit precise boundary values without fighting float rounding.
    """
    bars = flat_then_spike(symbol, n_flat=20, flat_close=100.0, decision_close=100.0,
                           flat_vol=1000.0, decision_vol=1000.0)
    last_price = bars[-1].close
    session_vwap = last_price / (1.0 + dist)  # so (last-vwap)/vwap == dist exactly
    rolling = 1000.0
    current_vol = vol_ratio * rolling
    ind = Indicators(
        symbol=symbol,
        session_vwap=session_vwap,
        last_price=last_price,
        dist_from_vwap=dist,
        current_volume=current_vol,
        rolling20_avg_vol=rolling,
        volume_ratio=vol_ratio,
        n_bars=len(bars),
        valid=valid,
        invalid_reason=None if valid else "forced_invalid",
    )
    snap = make_snapshot(symbol, bars, ind, position_state=PositionState(symbol=symbol))
    return snap


class _ConstScorer:
    """A fake EntryScorer that returns a constant p_win (no model needed)."""

    def __init__(self, p: float):
        self._p = p

    @property
    def is_passthrough(self) -> bool:
        return False

    def p_win(self, feature_vector) -> float:  # noqa: ARG002 - constant
        return self._p


# --------------------------------------------------------------------------- #
# ENTRY boundary
# --------------------------------------------------------------------------- #
def test_entry_fires_exactly_at_boundary():
    """dist == -entry_dist AND volume_ratio == vol_mult => ENTRY (predicate is <= / >=)."""
    agent = StrategyAgent(PARAMS)
    snap = _entry_snapshot("SPY", dist=-PARAMS.entry_dist, vol_ratio=PARAMS.vol_mult)
    prop = agent.propose(snap)
    assert prop.side is Side.BUY
    assert prop.intent is Intent.ENTRY
    assert prop.is_entry
    assert prop.notional == PARAMS.notional
    assert prop.features is not None
    assert prop.ml_p_win == pytest.approx(1.0)  # no scorer -> passthrough


def test_entry_blocked_just_inside_dist_boundary():
    """dist just ABOVE -entry_dist (not far enough below VWAP) => HOLD even with volume."""
    agent = StrategyAgent(PARAMS)
    snap = _entry_snapshot("SPY", dist=-PARAMS.entry_dist + 1e-6, vol_ratio=PARAMS.vol_mult)
    prop = agent.propose(snap)
    assert prop.side is Side.HOLD
    assert prop.intent is Intent.NONE


def test_entry_blocked_just_inside_volume_boundary():
    """volume_ratio just BELOW vol_mult => HOLD even when far enough below VWAP."""
    agent = StrategyAgent(PARAMS)
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=PARAMS.vol_mult - 1e-6)
    prop = agent.propose(snap)
    assert prop.side is Side.HOLD
    assert prop.intent is Intent.NONE


def test_entry_blocked_when_indicators_invalid():
    """indicators.valid == False (e.g. < 21 bars) => never enter, regardless of signals."""
    agent = StrategyAgent(PARAMS)
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=2.0, valid=False)
    prop = agent.propose(snap)
    assert prop.side is Side.HOLD


def test_entry_realistic_fixture_passes_rules_end_to_end():
    """A real flat_then_spike fixture (computed indicators) clears the default predicate."""
    agent = StrategyAgent(PARAMS)
    # Drop 1% below the flat 100 with a 2x volume spike -> both conditions satisfied.
    bars = flat_then_spike("AAPL", flat_close=100.0, decision_close=99.0,
                           flat_vol=1000.0, decision_vol=2000.0)
    ind = compute_indicators(bars, PARAMS)
    assert ind.valid and ind.dist_from_vwap < -PARAMS.entry_dist and ind.volume_ratio >= PARAMS.vol_mult
    snap = make_snapshot("AAPL", bars, ind, position_state=PositionState(symbol="AAPL"))
    prop = agent.propose(snap)
    assert prop.is_entry


# --------------------------------------------------------------------------- #
# EXIT triggers — each must fire INDEPENDENTLY
# --------------------------------------------------------------------------- #
def _long_exit_snapshot(symbol: str, *, dist, holding_min, plpc, asof_eod=False):
    """Build a LONG snapshot for exit testing with controllable dist/holding/plpc."""
    bars = bars_session(symbol, [100.0] * 5, [1000.0] * 5)
    asof = ny(hour=15, minute=56) if asof_eod else ny(hour=10, minute=0)
    entry_time = asof - _dt.timedelta(minutes=holding_min) if holding_min is not None else None
    ind = Indicators(
        symbol=symbol, session_vwap=100.0, last_price=100.0,
        dist_from_vwap=dist, current_volume=1000.0, rolling20_avg_vol=1000.0,
        volume_ratio=1.0, n_bars=len(bars), valid=True,
    )
    pos = long_position(symbol, qty=2.0, avg_entry_price=100.0,
                        entry_time=entry_time, unrealized_plpc=plpc)
    return make_snapshot(symbol, bars, ind, position_state=pos, asof=asof)


def test_exit_vwap_revert_fires():
    """dist_from_vwap >= -vwap_exit_band (price back near VWAP) => VWAP_REVERT exit."""
    agent = StrategyAgent(PARAMS)
    # dist at exactly -band is the boundary (>=), holding short, no loss -> only vwap fires.
    snap = _long_exit_snapshot("SPY", dist=-PARAMS.vwap_exit_band, holding_min=1, plpc=0.0)
    prop = agent.propose(snap)
    assert prop.side is Side.SELL
    assert prop.intent is Intent.EXIT
    assert prop.qty == 2.0  # full position
    assert prop.signal_values["exit_status"] == ExitStatus.VWAP_REVERT.value


def test_exit_max_hold_fires_independently():
    """held > 15 min with price still BELOW vwap and no stop-loss => MAX_HOLD exit."""
    agent = StrategyAgent(PARAMS)
    # dist well below the exit band so vwap_revert does NOT fire; plpc 0 so no stop.
    snap = _long_exit_snapshot("SPY", dist=-0.02, holding_min=16, plpc=0.0)
    prop = agent.propose(snap)
    assert prop.is_exit
    assert prop.signal_values["exit_status"] == ExitStatus.MAX_HOLD.value


def test_exit_max_hold_boundary_not_at_15():
    """max_hold is STRICTLY greater-than: exactly 15 min does NOT trigger max_hold."""
    agent = StrategyAgent(PARAMS)
    snap = _long_exit_snapshot("SPY", dist=-0.02, holding_min=15, plpc=0.0)
    prop = agent.propose(snap)
    # No exit trigger at all (still below vwap, not over hold, no loss) -> HOLD.
    assert prop.side is Side.HOLD


def test_exit_stop_loss_fires_independently():
    """unrealized_plpc <= -stop_loss with price below vwap and short hold => STOP_LOSS."""
    agent = StrategyAgent(PARAMS)
    snap = _long_exit_snapshot("SPY", dist=-0.02, holding_min=1, plpc=-PARAMS.stop_loss)
    prop = agent.propose(snap)
    assert prop.is_exit
    assert prop.signal_values["exit_status"] == ExitStatus.STOP_LOSS.value


def test_exit_eod_flatten_overrides_all():
    """On/after 15:55 NY, EOD_FLATTEN fires regardless of other conditions."""
    agent = StrategyAgent(PARAMS)
    # Far below vwap, short hold, no loss -> nothing else would fire, but EOD does.
    snap = _long_exit_snapshot("SPY", dist=-0.02, holding_min=1, plpc=0.0, asof_eod=True)
    prop = agent.propose(snap)
    assert prop.is_exit
    assert prop.signal_values["exit_status"] == ExitStatus.EOD_FLATTEN.value


def test_long_holds_when_no_exit_trigger():
    """A long position with no trigger => HOLD (not an erroneous SELL)."""
    agent = StrategyAgent(PARAMS)
    snap = _long_exit_snapshot("SPY", dist=-0.02, holding_min=5, plpc=-0.001)
    prop = agent.propose(snap)
    assert prop.side is Side.HOLD
    assert prop.intent is Intent.NONE


# --------------------------------------------------------------------------- #
# Finding #4 (HIGH): a tz-NAIVE entry_time must not crash (and silently abort) the
# WHOLE exit evaluation. holding_minutes is computed defensively; EOD flatten and the
# price-based exits still fire even when the holding time can't be derived.
# --------------------------------------------------------------------------- #
def _long_exit_snapshot_naive_entry(symbol: str, *, dist, plpc, asof_eod=False):
    """Like _long_exit_snapshot but with a tz-NAIVE entry_time (the regression input)."""
    bars = bars_session(symbol, [100.0] * 5, [1000.0] * 5)
    asof = ny(hour=15, minute=56) if asof_eod else ny(hour=10, minute=0)
    # Deliberately tz-NAIVE entry_time (e.g. from a mocked/proxied/future SDK position).
    naive_entry = _dt.datetime(2024, 6, 3, 9, 45, 0)
    assert naive_entry.tzinfo is None
    ind = Indicators(
        symbol=symbol, session_vwap=100.0, last_price=100.0,
        dist_from_vwap=dist, current_volume=1000.0, rolling20_avg_vol=1000.0,
        volume_ratio=1.0, n_bars=len(bars), valid=True,
    )
    pos = long_position(symbol, qty=2.0, avg_entry_price=100.0,
                        entry_time=naive_entry, unrealized_plpc=plpc)
    return make_snapshot(symbol, bars, ind, position_state=pos, asof=asof)


def test_exit_eod_flatten_fires_with_naive_entry_time():
    """A tz-naive entry_time must NOT abort the exit branch: EOD flatten still fires.

    Regression for the same-bar TypeError: previously _holding_minutes(asof, naive) raised
    and (being computed at the TOP of _propose_exit) suppressed EOD flatten too, leaving the
    position to be carried overnight. It must now produce the EOD_FLATTEN exit cleanly."""
    agent = StrategyAgent(PARAMS)
    snap = _long_exit_snapshot_naive_entry("SPY", dist=-0.02, plpc=0.0, asof_eod=True)
    prop = agent.propose(snap)  # must not raise
    assert prop.is_exit
    assert prop.signal_values["exit_status"] == ExitStatus.EOD_FLATTEN.value


def test_exit_vwap_revert_fires_with_naive_entry_time():
    """A tz-naive entry_time must not block a price-based exit either (vwap_revert)."""
    agent = StrategyAgent(PARAMS)
    snap = _long_exit_snapshot_naive_entry("SPY", dist=-PARAMS.vwap_exit_band, plpc=0.0)
    prop = agent.propose(snap)  # must not raise
    assert prop.is_exit
    assert prop.signal_values["exit_status"] == ExitStatus.VWAP_REVERT.value


def test_no_entry_in_eod_window():
    """Even a perfect entry signal is suppressed in the EOD window (>= 15:55)."""
    agent = StrategyAgent(PARAMS)
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=2.0)
    eod_snap = make_snapshot("SPY", list(snap.bars), snap.indicators,
                             position_state=PositionState(symbol="SPY"),
                             asof=ny(hour=15, minute=55))
    prop = agent.propose(eod_snap)
    assert prop.side is Side.HOLD


# --------------------------------------------------------------------------- #
# ML gate (§5)
# --------------------------------------------------------------------------- #
def test_ml_gate_blocks_entry_below_threshold():
    """rules pass but p_win < ml_threshold => HOLD (learned gate vetoes)."""
    params = StrategyParams(ml_threshold=0.6)
    agent = StrategyAgent(params)
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=2.0)
    prop = agent.propose(snap, ml_scorer=_ConstScorer(0.4))
    assert prop.side is Side.HOLD
    assert prop.intent is Intent.NONE
    assert prop.ml_p_win == pytest.approx(0.4)


def test_ml_gate_allows_entry_at_threshold():
    """p_win == ml_threshold passes the gate (>=)."""
    params = StrategyParams(ml_threshold=0.6)
    agent = StrategyAgent(params)
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=2.0)
    prop = agent.propose(snap, ml_scorer=_ConstScorer(0.6))
    assert prop.is_entry
    assert prop.ml_p_win == pytest.approx(0.6)


def test_ml_none_scorer_is_passthrough():
    """No scorer (None) -> p_win 1.0 -> rules-only entry at default threshold 0.0."""
    agent = StrategyAgent(StrategyParams())
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=2.0)
    prop = agent.propose(snap, ml_scorer=None)
    assert prop.is_entry
    assert prop.ml_p_win == pytest.approx(1.0)


def test_strategy_is_deterministic():
    """Identical input -> identical proposal side/intent/notional (no LLM, no randomness)."""
    agent = StrategyAgent(PARAMS)
    snap = _entry_snapshot("SPY", dist=-0.01, vol_ratio=2.0)
    a = agent.propose(snap)
    b = agent.propose(snap)
    assert (a.side, a.intent, a.notional) == (b.side, b.intent, b.notional)
