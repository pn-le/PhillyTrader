"""orchestrator.py — one full agentic trade cycle (STRATEGY_RULES §10).

`run_cycle` performs ONE deterministic pass over the universe and wires the six agents
together behind the hard safety/risk invariants:

    RECONCILE broker state (INV-6: positions + open orders)        ->
    MarketDataAgent.get_snapshots  (log data_snapshot)             ->
    StrategyAgent.propose          (exits first, then entries;     ->
                                    log strategy_proposal + ml_score)
    RiskAgent.review               (threads approved_buys; INV-4;  ->
                                    log risk_decision)
    ExecutionAgent.execute(armed)  (INV-1/2/3; log order_request/  ->
                                    order_response)
    PositionAgent.summarize        (log position_summary)

Every step is wrapped so a single failure logs an error (log_error) and is collected in
`CycleReport.errors` rather than crashing the cycle. The per-symbol cooldown store
(symbol -> last ENTRY timestamp) is updated after each submitted/intended entry so the
Risk agent's cooldown cap (A5) holds across cycles within a process.

PAPER ONLY. Deterministic. No LLM in the trade loop.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agents.execution_agent import ExecutionAgent
from .agents.position_agent import PositionAgent, total_exposure
from .agents.risk_agent import RiskAgent
from .agents.strategy_agent import StrategyAgent
from .config import (
    RiskLimits,
    Settings,
    StrategyParams,
)
from .data.market_data import MarketDataAgent
from .logging_util import JsonlLogger
from .types import (
    OrderResult,
    PositionState,
    PositionStatus,
    Proposal,
    RiskDecision,
    Snapshot,
)


@dataclass
class CycleReport:
    """Structured result of one full trade cycle (returned by run_cycle)."""

    asof: _dt.datetime
    snapshots: List[Snapshot] = field(default_factory=list)
    proposals: List[Proposal] = field(default_factory=list)
    decisions: List[RiskDecision] = field(default_factory=list)
    orders: List[OrderResult] = field(default_factory=list)
    summaries: List[Any] = field(default_factory=list)  # List[PositionSummary]
    exposure: float = 0.0
    armed: bool = False
    errors: List[str] = field(default_factory=list)


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce an Alpaca string/numeric field to float (Alpaca numerics are strings)."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _entry_cooldown_ts(
    snap: Optional[Snapshot], now: Optional[_dt.datetime]
) -> _dt.datetime:
    """Resolve the cooldown timestamp to store for a just-approved entry (A5).

    Prefer the symbol's OWN decision-bar `asof` (real wall-clock time of the bar we decided
    on) — but only when the snapshot actually has bars. A bars-less snapshot's `asof` is the
    market_data 09:30-NY session-open FALLBACK, which would make the cooldown look stale on
    the next cycle and let the symbol be re-entered immediately. In that case (or when no
    snapshot exists) use the live UTC clock so the stored timestamp is always a truthful,
    monotonic 'just entered now'. Always tz-aware.
    """
    if snap is not None and getattr(snap, "bars", None) and isinstance(snap.asof, _dt.datetime):
        return snap.asof
    if isinstance(now, _dt.datetime):
        return now if now.tzinfo is not None else now.replace(tzinfo=_dt.timezone.utc)
    return _dt.datetime.now(_dt.timezone.utc)


def _position_entry_time(pos: Any) -> Optional[_dt.datetime]:
    """Best-effort entry timestamp for a reconciled broker position.

    Alpaca's Position model does NOT carry an open time, so this returns None unless a
    future SDK adds one. The orchestrator separately enriches entry_time from the open
    BUY order's fill time when available (see _entry_times_from_orders).
    """
    for attr in ("entry_time", "filled_at", "created_at", "submitted_at"):
        candidate = getattr(pos, attr, None)
        if isinstance(candidate, _dt.datetime):
            return candidate
    return None


def reconcile(
    clients: Dict[str, Any],
    settings: Settings,
    logger: Optional[JsonlLogger] = None,
) -> Tuple[Any, List[Any], List[Any], Dict[str, PositionState]]:
    """RECONCILE broker truth (INV-6): account, positions, open orders, position_states.

    Returns (account, positions, open_orders, position_states) where `position_states`
    maps every UNIVERSE symbol -> PositionState (LONG with qty/avg/entry_time if held,
    else FLAT). Entry time is enriched from each symbol's most recent filled/submitted
    order when reconstructable, else falls back to a timestamp on the position object.
    Never raises: a broker error degrades to empty lists / FLAT states and is logged.
    """
    trading = clients.get("trading")
    account: Any = None
    positions: List[Any] = []
    open_orders: List[Any] = []

    try:
        account = trading.get_account()
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.log_error("orchestrator.reconcile.account", "get_account failed", exc)

    try:
        positions = list(trading.get_all_positions() or [])
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.log_error("orchestrator.reconcile.positions", "get_all_positions failed", exc)
        positions = []

    try:
        open_orders = _fetch_open_orders(trading)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.log_error("orchestrator.reconcile.orders", "get_orders failed", exc)
        open_orders = []

    # Enrich entry times from order history (best-effort; never required).
    entry_times = _entry_times_from_orders(trading, logger)

    position_states: Dict[str, PositionState] = {}
    for sym in settings.universe:
        position_states[sym] = PositionState(symbol=sym)  # default FLAT

    for pos in positions:
        sym = getattr(pos, "symbol", None)
        if sym is None:
            continue
        qty = _to_float(getattr(pos, "qty", None), 0.0) or 0.0
        if qty <= 0.0:
            continue  # short/empty positions are out of scope (long-only strategy)
        entry_time = entry_times.get(sym) or _position_entry_time(pos)
        position_states[sym] = PositionState(
            symbol=sym,
            status=PositionStatus.LONG,
            qty=qty,
            avg_entry_price=_to_float(getattr(pos, "avg_entry_price", None)),
            entry_time=entry_time,
            market_value=_to_float(getattr(pos, "market_value", None)),
            unrealized_pl=_to_float(getattr(pos, "unrealized_pl", None)),
            unrealized_plpc=_to_float(getattr(pos, "unrealized_plpc", None)),
            current_price=_to_float(getattr(pos, "current_price", None)),
        )

    return account, positions, open_orders, position_states


def seed_cooldowns(
    clients: Dict[str, Any],
    position_states: Dict[str, PositionState],
    logger: Optional[JsonlLogger] = None,
) -> Dict[str, _dt.datetime]:
    """Rebuild the per-symbol cooldown store from BROKER truth (A5), so the 10-min entry
    cap survives process restarts and single-shot `run` invocations.

    Docstrings across the system claim cooldowns are 'derived from the persisted trade log'
    — but an in-memory map created empty each process start silently bypasses the cap: a
    symbol entered moments ago has no cooldown record after a restart / a fresh `run`, so a
    brand-new entry is wrongly approved. We reconstruct it from the data reconcile already
    has at hand: the most recent filled BUY order timestamp per symbol (the actual entry
    time) plus each currently-open position's entry_time. The most recent (max) timestamp
    per symbol wins. All timestamps are tz-aware (RiskAgent compares against UTC now).
    """
    trading = clients.get("trading")
    seeded: Dict[str, _dt.datetime] = {}

    # Most-recent filled BUY per symbol (these ARE the last entries).
    try:
        entry_times = _entry_times_from_orders(trading, logger)
    except Exception as exc:  # noqa: BLE001 — never let seeding break the cycle
        if logger is not None:
            logger.log_error("orchestrator.seed_cooldowns.orders", "entry-time seeding failed", exc)
        entry_times = {}
    for sym, ts in entry_times.items():
        if isinstance(ts, _dt.datetime):
            seeded[sym] = ts if ts.tzinfo is not None else ts.replace(tzinfo=_dt.timezone.utc)

    # Open positions' reconciled entry_time (covers a held entry whose BUY order has aged out
    # of the recent-orders window). Keep the most recent timestamp per symbol.
    for sym, pstate in (position_states or {}).items():
        et = getattr(pstate, "entry_time", None)
        if isinstance(et, _dt.datetime):
            et = et if et.tzinfo is not None else et.replace(tzinfo=_dt.timezone.utc)
            prior = seeded.get(sym)
            if prior is None or et > prior:
                seeded[sym] = et

    return seeded


def _fetch_open_orders(trading: Any) -> List[Any]:
    """Fetch all currently OPEN orders at the broker (for the A17 duplicate guard)."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    return list(trading.get_orders(filter=req) or [])


def _entry_times_from_orders(
    trading: Any, logger: Optional[JsonlLogger]
) -> Dict[str, _dt.datetime]:
    """Map symbol -> entry timestamp from recent CLOSED/filled BUY orders (best-effort).

    Used to set PositionState.entry_time for held positions (A6) so the §4 max_hold
    predicate can fire. Pulls a bounded window of recent orders; a failure degrades to an
    empty map (entry_time stays None -> max_hold simply cannot fire that cycle).
    """
    out: Dict[str, _dt.datetime] = {}
    try:
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            side=OrderSide.BUY,
            limit=100,
            direction="desc",
        )
        orders = list(trading.get_orders(filter=req) or [])
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.log_error("orchestrator._entry_times_from_orders", "get_orders(closed) failed", exc)
        return out

    # orders are newest-first; keep the most recent filled BUY per symbol.
    for o in orders:
        sym = getattr(o, "symbol", None)
        if sym is None or sym in out:
            continue
        ts = getattr(o, "filled_at", None) or getattr(o, "submitted_at", None)
        if isinstance(ts, _dt.datetime):
            out[sym] = ts
    return out


def run_cycle(
    clients: Dict[str, Any],
    params: StrategyParams,
    limits: RiskLimits,
    scorer: "Any | None",
    *,
    armed: bool,
    settings: Optional[Settings] = None,
    logger: Optional[JsonlLogger] = None,
    cooldowns: Optional[Dict[str, _dt.datetime]] = None,
    now: Optional[_dt.datetime] = None,
) -> CycleReport:
    """Run ONE full cycle and return a CycleReport (STRATEGY_RULES §10).

    `clients` is the dict from config.make_clients() ({"trading","data","paper"}).
    `scorer` is an EntryScorer (or None/passthrough). `armed` flips live submission
    (still gated by INV-1/2/3 inside Execution). `cooldowns` (symbol -> last ENTRY ts) is
    consulted by Risk and UPDATED in place after each entry order so the per-symbol
    cooldown cap (A5) survives across cycles within a process. Every step is try-wrapped;
    failures are logged and collected in `CycleReport.errors`.
    """
    settings = settings or Settings()
    logger = logger or JsonlLogger()
    cooldowns = cooldowns if cooldowns is not None else {}
    errors: List[str] = []

    def _err(where: str, msg: str, exc: Optional[BaseException] = None) -> None:
        errors.append(f"{where}: {msg}")
        try:
            logger.log_error(where, msg, exc)
        except Exception:  # logging must never break the cycle
            pass

    # 1) RECONCILE broker truth (INV-6).
    try:
        account, positions, open_orders, position_states = reconcile(clients, settings, logger)
    except Exception as exc:  # noqa: BLE001 — reconcile is already defensive, this is belt+braces
        _err("orchestrator.run_cycle.reconcile", "reconcile failed", exc)
        account, positions, open_orders, position_states = None, [], [], {
            s: PositionState(symbol=s) for s in settings.universe
        }

    # Seed the cooldown store from BROKER truth when it is empty (A5). This covers a fresh
    # process: a single-shot `run` (cmd_run passes no cooldowns) and the FIRST cycle after a
    # loop restart both start with an empty map, which would otherwise let a symbol entered
    # moments-ago (pre-restart) be re-entered immediately, breaching the 10-min cap. Once
    # seeded, the in-memory map persists across cycles within the process, so we only seed
    # when empty (never clobbering a fresher in-memory timestamp with a lagging broker one).
    if not cooldowns:
        try:
            cooldowns.update(seed_cooldowns(clients, position_states, logger))
        except Exception as exc:  # noqa: BLE001
            _err("orchestrator.run_cycle.seed_cooldowns", "cooldown seeding failed", exc)

    # 2) MarketDataAgent snapshots (log data_snapshot per symbol).
    md_agent = MarketDataAgent(clients.get("data"), settings, params, logger)
    try:
        snapshots = md_agent.get_snapshots(
            universe=list(settings.universe),
            position_states=position_states,
            now=now,
        )
    except Exception as exc:  # noqa: BLE001
        _err("orchestrator.run_cycle.snapshots", "get_snapshots failed", exc)
        snapshots = []

    for snap in snapshots:
        try:
            logger.log_snapshot(snap)
        except Exception as exc:  # noqa: BLE001
            _err("orchestrator.run_cycle.log_snapshot", f"log_snapshot failed for {snap.symbol}", exc)

    asof = snapshots[0].asof if snapshots else (now or _dt.datetime.now(_dt.timezone.utc))
    snap_by_symbol = {s.symbol: s for s in snapshots}

    # 3) StrategyAgent proposals — EXITS first, then ENTRIES, in UNIVERSE order.
    strat = StrategyAgent(params, settings, logger)
    proposals: List[Proposal] = []

    exit_snaps = [s for s in snapshots if s.position_state.is_long]
    entry_snaps = [s for s in snapshots if s.position_state.is_flat]

    for snap in exit_snaps + entry_snaps:
        try:
            prop = strat.propose(snap, params, scorer)
            proposals.append(prop)
            logger.log_proposal(prop)
        except Exception as exc:  # noqa: BLE001
            _err("orchestrator.run_cycle.propose", f"propose failed for {snap.symbol}", exc)

    # 4) RiskAgent review — exits first, then entries; thread approved_buys (A14).
    risk = RiskAgent(limits, settings, logger)
    exec_agent = ExecutionAgent(clients.get("trading"), settings, logger=logger)

    exit_props = [p for p in proposals if p.is_exit]
    entry_props = [p for p in proposals if p.is_entry]
    other_props = [p for p in proposals if not (p.is_exit or p.is_entry)]

    decisions: List[RiskDecision] = []
    orders: List[OrderResult] = []
    approved_buys: List[RiskDecision] = []

    # Exits + sells first (risk-reducing, free up slots/exposure).
    for prop in exit_props:
        decision = _review_and_execute(
            prop, risk, exec_agent, account, positions, open_orders,
            cooldowns, params, limits, approved_buys=None, armed=armed,
            decisions=decisions, orders=orders, on_error=_err,
        )

    # Entries next, threading prior approvals so one slot/dollar serves one proposal.
    for prop in entry_props:
        decision = _review_and_execute(
            prop, risk, exec_agent, account, positions, open_orders,
            cooldowns, params, limits, approved_buys=approved_buys, armed=armed,
            decisions=decisions, orders=orders, on_error=_err,
        )
        if decision is not None and decision.approved:
            approved_buys.append(decision)
            # Update the cooldown store so the next cycle honours the per-symbol window (A5).
            # Use THIS symbol's own decision-bar timestamp — NOT the shared snapshots[0].asof,
            # which is SPY's asof and falls back to 09:30 NY when SPY has a data gap (that would
            # make every entry this cycle look ~hours old and bypass the 10-min cooldown next
            # cycle). When the symbol lacks real bars, fall back to the live UTC clock so the
            # stored timestamp stays monotonic and the cooldown can never be spuriously expired.
            cooldowns[prop.symbol] = _entry_cooldown_ts(snap_by_symbol.get(prop.symbol), now)

    # HOLD / other proposals: log a no-action risk decision for the audit trail.
    for prop in other_props:
        try:
            decision = risk.review(prop, account, positions, open_orders, cooldowns, params, limits, approved_buys)
            decisions.append(decision)
        except Exception as exc:  # noqa: BLE001
            _err("orchestrator.run_cycle.review_hold", f"risk review failed for {prop.symbol}", exc)

    # 5) PositionAgent summary (log position_summary + exposure).
    pos_agent = PositionAgent(params, settings, logger)
    try:
        summaries = pos_agent.summarize(positions, params, snap_by_symbol, now=now)
    except Exception as exc:  # noqa: BLE001
        _err("orchestrator.run_cycle.summarize", "position summary failed", exc)
        summaries = []

    try:
        exposure = total_exposure(positions)
    except Exception:  # noqa: BLE001
        exposure = 0.0

    try:
        logger.log_position_summary(summaries, exposure=exposure)
    except Exception as exc:  # noqa: BLE001
        _err("orchestrator.run_cycle.log_position_summary", "log_position_summary failed", exc)

    return CycleReport(
        asof=asof,
        snapshots=snapshots,
        proposals=proposals,
        decisions=decisions,
        orders=orders,
        summaries=summaries,
        exposure=exposure,
        armed=bool(armed),
        errors=errors,
    )


def _review_and_execute(
    proposal: Proposal,
    risk: RiskAgent,
    exec_agent: ExecutionAgent,
    account: Any,
    positions: List[Any],
    open_orders: List[Any],
    cooldowns: Dict[str, _dt.datetime],
    params: StrategyParams,
    limits: RiskLimits,
    *,
    approved_buys: Optional[List[RiskDecision]],
    armed: bool,
    decisions: List[RiskDecision],
    orders: List[OrderResult],
    on_error,
) -> Optional[RiskDecision]:
    """Run Risk review then Execution for a single actionable proposal (INV-4/INV-1/2/3).

    Risk logs its own decision; Execution logs order_request/order_response. Only an
    approved (APPROVE/REDUCE) decision reaches Execution; rejected proposals are recorded
    but never executed. Returns the RiskDecision (or None on a review failure).
    """
    try:
        decision = risk.review(
            proposal, account, positions, open_orders, cooldowns, params, limits, approved_buys
        )
    except Exception as exc:  # noqa: BLE001
        on_error("orchestrator._review_and_execute.review", f"risk review failed for {proposal.symbol}", exc)
        return None

    decisions.append(decision)

    if not decision.approved:
        return decision  # rejected -> never reaches Execution (INV-4)

    try:
        result = exec_agent.execute(decision, armed=armed)
        orders.append(result)
    except Exception as exc:  # noqa: BLE001
        on_error("orchestrator._review_and_execute.execute", f"execution failed for {proposal.symbol}", exc)

    return decision


__all__ = ["CycleReport", "run_cycle", "reconcile", "seed_cooldowns"]
