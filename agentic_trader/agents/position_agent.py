"""position_agent.py — Agent ⑥ Position Analysis (INTERFACE_SPEC §6).

Read-only P&L / exposure reporting + exit classification for open positions.
This agent NEVER proposes or submits orders — it only summarizes reconciled broker
state for logging, the orchestrator's position_summary event, and offline ML labels.

It is DETERMINISTIC and pure-ish: given the same positions, params, snapshots, and `now`
it returns the same PositionSummary list every time. The only impurity is reading the
clock when `now` is not supplied (used solely as the holding-time / EOD reference).

Sources & conventions (per STRATEGY_RULES & API_MAP):
  - Alpaca Position numeric fields are STRINGS (API_MAP §4) -> parsed with float().
  - holding_minutes uses `entry_time` resolved per A6: prefer the reconciled entry time
    carried on snapshots[symbol].position_state.entry_time (the orchestrator sets this from
    the local trade log, else the broker open time); else fall back to a timestamp found on
    the raw Alpaca position object; else None (holding_minutes unknown -> max_hold cannot
    fire, but the other exit predicates still apply).
  - exit_status uses the SAME §4 predicates as the Strategy agent, with the SAME priority:
    eod_flatten (on/after 15:55 NY, A9) overrides everything; otherwise
    vwap_revert > max_hold > stop_loss; else HOLD.
  - dist_from_vwap for the vwap_revert test comes from snapshots[symbol].indicators (the
    decision-bar value). If no snapshot is available it is treated as unknown and the
    vwap_revert branch is skipped (A4: no VWAP -> no vwap_revert), matching the strategy.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..config import EOD_FLATTEN_TIME, SESSION_TZ, StrategyParams
from ..logging_util import JsonlLogger
from ..types import ExitStatus, PositionSummary, Snapshot


# Parse "HH:MM:SS" (EOD_FLATTEN_TIME) once into a time for the A9 comparison.
def _parse_eod_time(text: str) -> _dt.time:
    h, m, s = (int(x) for x in text.split(":"))
    return _dt.time(hour=h, minute=m, second=s)


_EOD_TIME = _parse_eod_time(EOD_FLATTEN_TIME)


def _to_float(value: Any) -> Optional[float]:
    """Coerce an Alpaca string/None numeric field to float; None if missing/unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def total_exposure(positions: List[Any]) -> float:
    """Sum of float(position.market_value) over open positions (helper for risk/logging).

    Tolerates missing/unparseable market_value (treated as 0.0) so a malformed position
    never breaks exposure accounting. NOT a substitute for the Risk agent's own check.
    """
    total = 0.0
    for p in positions or []:
        mv = _to_float(getattr(p, "market_value", None))
        if mv is not None:
            total += mv
    return total


def _resolve_entry_time(symbol: str, position: Any,
                        snapshots: Optional[Dict[str, Snapshot]]) -> Optional[_dt.datetime]:
    """Resolve entry_time per A6: snapshot's reconciled value -> raw position attr -> None.

    The orchestrator builds snapshots[symbol].position_state.entry_time from the local
    trade log (preferred) or the broker open time. If no snapshot is supplied we look for
    a timestamp on the raw Alpaca position object (created_at / etc.); none of those are
    guaranteed by the SDK, so this is best-effort and returns None when unavailable.
    """
    if snapshots is not None:
        snap = snapshots.get(symbol)
        if snap is not None and snap.position_state is not None:
            et = snap.position_state.entry_time
            if et is not None:
                return et
    # Best-effort fall back to a timestamp on the raw broker object.
    for attr in ("entry_time", "created_at", "filled_at", "submitted_at"):
        candidate = getattr(position, attr, None)
        if isinstance(candidate, _dt.datetime):
            return candidate
    return None


def _reference_time(symbol: str, snapshots: Optional[Dict[str, Snapshot]],
                    now: Optional[_dt.datetime]) -> _dt.datetime:
    """The 'now' used for holding_minutes and the EOD check, in America/New_York.

    Matches STRATEGY_RULES §4: holding_minutes = (decision_bar.start - entry_time). So when
    a snapshot exists we use its `asof` (the decision-bar start); otherwise we use the
    supplied `now`, else the current wall clock. Always returned tz-aware in NY time.
    """
    tz = ZoneInfo(SESSION_TZ)
    if snapshots is not None:
        snap = snapshots.get(symbol)
        if snap is not None and snap.asof is not None:
            return snap.asof.astimezone(tz)
    ref = now if now is not None else _dt.datetime.now(_dt.timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=_dt.timezone.utc)
    return ref.astimezone(tz)


def _holding_minutes(entry_time: Optional[_dt.datetime],
                     ref_now: _dt.datetime) -> Optional[float]:
    """Minutes held = (ref_now - entry_time)/60. None if entry_time unknown."""
    if entry_time is None:
        return None
    et = entry_time
    if et.tzinfo is None:
        et = et.replace(tzinfo=_dt.timezone.utc)
    delta = ref_now - et.astimezone(ref_now.tzinfo)
    return delta.total_seconds() / 60.0


def _classify_exit(dist_from_vwap: Optional[float], holding_minutes: Optional[float],
                   unrealized_plpc: Optional[float], ref_now: _dt.datetime,
                   params: StrategyParams) -> ExitStatus:
    """Classify the exit trigger using the §4 predicates and priority.

    Priority: eod_flatten (A9, on/after 15:55 NY) > vwap_revert > max_hold > stop_loss > HOLD.
    Each branch is skipped when its required input is unknown (mirrors the strategy: A4 means
    an undefined VWAP cannot fire vwap_revert; an unknown holding time cannot fire max_hold).
    """
    # A9: end-of-day force-exit overrides everything.
    if ref_now.timetz().replace(tzinfo=None) >= _EOD_TIME:
        return ExitStatus.EOD_FLATTEN

    # vwap_revert: price back within vwap_exit_band of VWAP (dist >= -band).
    if dist_from_vwap is not None and dist_from_vwap >= -params.vwap_exit_band:
        return ExitStatus.VWAP_REVERT

    # max_hold: held strictly longer than max_hold minutes.
    if holding_minutes is not None and holding_minutes > params.max_hold:
        return ExitStatus.MAX_HOLD

    # stop_loss: unrealized loss reached the cap.
    if unrealized_plpc is not None and unrealized_plpc <= -params.stop_loss:
        return ExitStatus.STOP_LOSS

    return ExitStatus.HOLD


class PositionAgent:
    """Summarize reconciled open positions: per-position P&L + exit classification.

    Stateless aside from injected params/settings/logger. `summarize()` is the only public
    entrypoint and never mutates broker state or proposes orders (read-only by contract).
    """

    def __init__(self, params: StrategyParams, settings: Optional[Any] = None,
                 logger: Optional[JsonlLogger] = None) -> None:
        self.params = params
        self.settings = settings
        self.logger = logger

    def summarize(self, positions: List[Any], params: Optional[StrategyParams] = None,
                  snapshots: Optional[Dict[str, Snapshot]] = None,
                  now: Optional[_dt.datetime] = None) -> List[PositionSummary]:
        """Build one PositionSummary per open Alpaca position. Deterministic, read-only.

        For each position: parse the string numeric fields to float, resolve entry_time
        (A6) and holding_minutes, classify exit_status via the §4 predicates, and set
        would_exit = (exit_status != HOLD). Order mirrors the input `positions` order.
        `params` overrides self.params when given. Errors on a single position are logged
        and that position is skipped (never raises out of a logging/summary path).
        """
        active = params if params is not None else self.params
        summaries: List[PositionSummary] = []

        for pos in positions or []:
            try:
                symbol = getattr(pos, "symbol", None)
                if symbol is None:
                    continue

                qty = _to_float(getattr(pos, "qty", None)) or 0.0
                avg_entry = _to_float(getattr(pos, "avg_entry_price", None))
                current_price = _to_float(getattr(pos, "current_price", None))
                market_value = _to_float(getattr(pos, "market_value", None))
                unrealized_pl = _to_float(getattr(pos, "unrealized_pl", None))
                unrealized_plpc = _to_float(getattr(pos, "unrealized_plpc", None))

                ref_now = _reference_time(symbol, snapshots, now)
                entry_time = _resolve_entry_time(symbol, pos, snapshots)
                holding_min = _holding_minutes(entry_time, ref_now)

                # dist_from_vwap (for vwap_revert) comes from the decision-bar indicators.
                dist_from_vwap: Optional[float] = None
                if snapshots is not None:
                    snap = snapshots.get(symbol)
                    if snap is not None and snap.indicators is not None:
                        dist_from_vwap = snap.indicators.dist_from_vwap

                exit_status = _classify_exit(
                    dist_from_vwap=dist_from_vwap,
                    holding_minutes=holding_min,
                    unrealized_plpc=unrealized_plpc,
                    ref_now=ref_now,
                    params=active,
                )

                summaries.append(PositionSummary(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=avg_entry,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pl=unrealized_pl,
                    unrealized_plpc=unrealized_plpc,
                    holding_minutes=holding_min,
                    exit_status=exit_status,
                    would_exit=exit_status != ExitStatus.HOLD,
                ))
            except Exception as exc:  # never let one bad position break the summary
                if self.logger is not None:
                    self.logger.log_error(
                        "PositionAgent.summarize",
                        f"failed to summarize position {getattr(pos, 'symbol', '?')}",
                        exc,
                    )
                continue

        return summaries


__all__ = ["PositionAgent", "total_exposure"]
