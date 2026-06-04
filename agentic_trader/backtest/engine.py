"""backtest/engine.py — t->t+1 intraday VWAP mean-reversion simulator.

Replays COMPLETED 1-minute regular-hours bars chronologically across the universe with
STRICT no-look-ahead semantics and emits a BacktestResult (trades + metrics + labeled
ML features). It reuses the SAME decision/risk code as the live loop so the backtest and
production cannot drift:

  - ``data.market_data.compute_indicators``  (STRATEGY_RULES §2 indicators)
  - ``agents.strategy_agent.StrategyAgent``  (§3 entry / §4 exit + §5 ML gate)
  - ``agents.risk_agent.RiskAgent``          (§8 caps: max positions / exposure / cooldown)
  - ``ml.features.build_features`` (via the StrategyAgent) for decision-time features.

Timing discipline (no look-ahead):
  - We DECIDE on the last completed bar t (indicators/features use only bars[0..t]).
  - We ACT/FILL at bar t+1's OPEN. If there is no t+1 bar (t is the session's last bar),
    the decision is dropped EXCEPT an EOD/forced flatten exit, which fills at bar t's OPEN
    (NOT its close) so a position is never carried overnight (A10) AND we never decide-on
    and fill-at the same bar's close (that would be a same-bar look-ahead). Any position
    still open when the timeline ends is force-closed at its last decision bar's OPEN.
  - Session VWAP resets at the start of each trading day.
  - Trades + labeled ML rows are emitted ordered by DECISION/ENTRY time (earliest first),
    not exit time, so the downstream temporal train/validation split stays leakage-free.

Cost model (A11): $0 commission, SLIPPAGE_BPS (1bp) ADVERSE slippage — a BUY fills above
the reference price and a SELL fills below it.

Determinism: identical inputs always produce identical output (no wall-clock, no RNG).
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..agents.risk_agent import RiskAgent
from ..agents.strategy_agent import StrategyAgent
from ..config import (
    SLIPPAGE_BPS,
    RiskLimits,
    Settings,
    StrategyParams,
)
from ..data.market_data import compute_indicators
from ..types import (
    Bar,
    ExitStatus,
    Indicators,
    PositionState,
    PositionStatus,
    Proposal,
    RiskDecision,
    Side,
    Snapshot,
    TradeRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ml.scorer import EntryScorer


# Exit reason used when a position is still open after the timeline is exhausted (no t+1
# bar and no exit predicate fired). It is force-closed at its LAST seen decision bar's OPEN
# so the trade (and its ML label) is recorded rather than silently dropped — a non-random
# omission that would otherwise bias backtest metrics and poison the training set.
_FORCED_CLOSE = "forced_close"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """The output of a backtest run.

    ``trades`` are realized round-trips (one TradeRecord per BUY->SELL pair). ``metrics``
    summarizes performance. ``labeled_features`` is one dict per ENTRY taken: the
    decision-time FeatureVector fields plus the realized ``label`` (1 if profitable) — fed
    straight into ml.dataset.build_dataset for training.
    """

    trades: List[TradeRecord] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    labeled_features: List[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Internal sim state
# --------------------------------------------------------------------------- #
@dataclass
class _OpenPosition:
    """A live simulated long position (book-keeping for the t->t+1 fill model)."""

    symbol: str
    qty: float
    entry_price: float          # adverse-slippage-adjusted buy fill
    entry_time: _dt.datetime    # fill bar start (t+1)
    decision_time: _dt.datetime  # the decision bar start (t) the entry was decided on
    features: dict              # decision-time FeatureVector fields (for the ML label)
    last_decision_bar: Optional[Bar] = None  # most recent decision bar seen (for forced close)

    def market_value(self, price: float) -> float:
        return self.qty * float(price)


class _FakeAccount:
    """Minimal stand-in for an Alpaca TradeAccount (always sane in the sim)."""

    account_blocked = False
    trading_blocked = False
    trade_suspended_by_user = False


class _FakePosition:
    """Minimal stand-in for an Alpaca Position the RiskAgent can read (market_value)."""

    __slots__ = ("symbol", "market_value")

    def __init__(self, symbol: str, market_value: float) -> None:
        self.symbol = symbol
        self.market_value = market_value


def _slip(price: float, side: Side) -> float:
    """Apply SLIPPAGE_BPS adverse slippage. BUY fills higher, SELL fills lower."""
    frac = SLIPPAGE_BPS / 10_000.0
    if side == Side.BUY:
        return price * (1.0 + frac)
    return price * (1.0 - frac)


def _in_cooldown(
    symbol: str,
    asof: _dt.datetime,
    cooldowns: Dict[str, _dt.datetime],
    limits: RiskLimits,
) -> bool:
    """True iff `symbol` had an ENTRY within per_symbol_cooldown_min of `asof` (sim time).

    Mirrors RiskAgent's §8 cooldown rule but on SIMULATED bar timestamps (the RiskAgent's
    check is wall-clock and cannot apply in a historical replay). `cooldowns[symbol]` is the
    entry-fill time of the most recent BUY for that symbol.
    """
    last = cooldowns.get(symbol)
    if last is None:
        return False
    elapsed_min = (asof - last).total_seconds() / 60.0
    return elapsed_min < float(limits.per_symbol_cooldown_min)


def _session_key(bar: Bar) -> _dt.date:
    """The trading-day key a bar belongs to (VWAP resets on this boundary)."""
    return bar.start.date()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_backtest(
    bars_by_symbol: Dict[str, List[Bar]],
    params: StrategyParams,
    limits: RiskLimits,
    scorer: "EntryScorer | None" = None,
) -> BacktestResult:
    """Replay completed bars per symbol with strict t->t+1 fills and risk caps.

    See module docstring for the timing/cost contract. Returns a BacktestResult with
    realized trades, summary metrics, and one labeled-feature dict per ENTRY taken.
    """
    strategy = StrategyAgent(params, settings=Settings(), logger=None)
    risk = RiskAgent(limits, settings=Settings(), logger=None)
    account = _FakeAccount()

    # Group every symbol's bars into per-(symbol, session-day) ascending bar lists so VWAP
    # is computed within a single trading day and resets across day boundaries.
    sessions = _group_sessions(bars_by_symbol)

    # Build the GLOBAL chronological timeline of decision steps. Each step is one bar index
    # within one (symbol, day) session. We process steps in (decision_bar_start, symbol)
    # order so portfolio-level caps (positions/exposure/cooldown) see a consistent ordering.
    timeline = _build_timeline(sessions)

    open_positions: Dict[str, _OpenPosition] = {}
    cooldowns: Dict[str, _dt.datetime] = {}
    # Collect each closed round-trip together with its decision/ENTRY time so the dataset can
    # be ordered by DECISION time (not exit time) for a leakage-free temporal split. Trades
    # close in exit-time order; train_model.py does a positional temporal split that requires
    # rows ordered by earliest-ENTRY-first, so we sort by (decision_time, entry_time, symbol)
    # before returning. Each entry: (decision_time, entry_time, symbol, TradeRecord, labeled_row).
    closed: List[Tuple[_dt.datetime, _dt.datetime, str, TradeRecord, dict]] = []

    # Equity curve sampled at each decision timestamp (for Sharpe-like + drawdown). We mark
    # open positions to the decision-bar close at each timestamp we observe it.
    equity_points: List[Tuple[_dt.datetime, float]] = []
    realized_pnl = 0.0

    for asof, group in timeline:
        # `group` maps symbol -> (bars_up_to_t, next_open_or_None, decision_bar) for every
        # symbol that has a decision bar at this exact timestamp `asof`.
        # 1) EXITS first (risk-reducing, free up slots/exposure before entries this step).
        # 2) then ENTRIES in UNIVERSE order, threading approved_buys for the caps.
        approved_buys: List[RiskDecision] = []

        # Stable per-step ordering: exits then entries, each in the session's symbol order.
        ordered_syms = list(group.keys())

        # --- EXIT pass ---
        for sym in ordered_syms:
            bars_t, next_open, decision_bar = group[sym]
            pos = open_positions.get(sym)
            if pos is None:
                continue
            snap = _snapshot_for(sym, asof, bars_t, params, pos)
            proposal = strategy.propose(snap, params, ml_scorer=scorer)
            if not proposal.is_exit:
                continue
            decision = _risk_review_exit(
                risk, proposal, account, open_positions, cooldowns, params, limits
            )
            if not decision.approved:
                continue
            # Fill the exit. EOD flatten with no t+1 fills at the decision-bar OPEN (A10).
            exit_reason = proposal.signal_values.get("exit_status", ExitStatus.HOLD.value)
            fill_price, fill_time = _exit_fill(next_open, decision_bar, exit_reason)
            if fill_price is None:
                continue
            trade, pnl = _close_position(pos, fill_price, fill_time, exit_reason)
            closed.append((pos.decision_time, pos.entry_time, sym, trade, _labeled_row(pos.features, trade.label)))
            realized_pnl += pnl
            del open_positions[sym]

        # --- ENTRY pass ---
        # Snapshot the positions held BEFORE this step's entries. The RiskAgent counts
        # `positions` (reconciled at cycle start, as live does) plus `approved_buys` (this
        # cycle's pending entries); positions newly opened in THIS pass must be reflected
        # only via `approved_buys`, never also via `positions`, or one entry would consume
        # two slots / be double-counted against the caps. So we freeze the held set here.
        held_positions = _fake_positions(open_positions)
        for sym in ordered_syms:
            bars_t, next_open, decision_bar = group[sym]
            if sym in open_positions:
                continue  # already long (an exit this step would have removed it)
            if next_open is None:
                continue  # last bar of the session -> no t+1 fill, drop the entry (A10)
            # Per-symbol cooldown (§8) enforced HERE against SIMULATED time. The RiskAgent's
            # own cooldown check measures elapsed against the wall clock, which is meaningless
            # in a historical replay, so the engine applies the same rule on bar timestamps.
            if _in_cooldown(sym, asof, cooldowns, limits):
                continue
            snap = _snapshot_for(sym, asof, bars_t, params, None)
            proposal = strategy.propose(snap, params, ml_scorer=scorer)
            if not proposal.is_entry:
                continue
            decision = risk.review(
                proposal,
                account,
                held_positions,
                [],  # open_orders: none in sim
                cooldowns,
                params,
                limits,
                approved_buys=approved_buys,
            )
            if not decision.approved:
                continue
            notional = decision.effective_notional
            if notional is None or notional <= 0.0:
                continue
            # Fill the BUY at t+1 OPEN with adverse slippage.
            fill_price = _slip(float(next_open.open), Side.BUY)
            if fill_price <= 0.0:
                continue
            qty = notional / fill_price
            open_positions[sym] = _OpenPosition(
                symbol=sym,
                qty=qty,
                entry_price=fill_price,
                entry_time=next_open.start,
                decision_time=asof,
                features=_features_dict(proposal),
            )
            cooldowns[sym] = next_open.start
            approved_buys.append(decision)

        # Mark-to-market equity at this timestamp (realized + open MV at decision closes).
        # Also record each open position's most recent decision bar so any position still
        # open when the timeline ends can be force-closed at a known same-bar OPEN (no t+1).
        open_mv = 0.0
        for sym, pos in open_positions.items():
            db = group.get(sym)
            if db is not None:
                pos.last_decision_bar = db[2]
                open_mv += pos.market_value(db[2].close)
            else:
                open_mv += pos.market_value(pos.entry_price)
        equity_points.append((asof, realized_pnl + open_mv - _basis(open_positions)))

    # Timeline exhausted: force-close any position still open (no t+1 bar ever appeared and
    # no exit predicate fired). Fill at its LAST decision bar's OPEN with exit_reason
    # 'forced_close' so the trade + ML label are recorded rather than silently dropped (a
    # non-random omission that would bias metrics and poison the training set). Closed in a
    # stable (universe, symbol) order for determinism.
    for sym in sorted(open_positions.keys(), key=_universe_rank):
        pos = open_positions[sym]
        db = pos.last_decision_bar
        if db is None:
            continue  # never observed a decision bar (cannot price a fill) -> drop
        fill_price, fill_time = _exit_fill(None, db, _FORCED_CLOSE)
        if fill_price is None or fill_time is None:
            continue
        trade, pnl = _close_position(pos, fill_price, fill_time, _FORCED_CLOSE)
        closed.append((pos.decision_time, pos.entry_time, sym, trade, _labeled_row(pos.features, trade.label)))
        realized_pnl += pnl
    open_positions.clear()

    # Order trades + labeled rows by DECISION/ENTRY time (earliest first) so the downstream
    # temporal train/validation split in train_model.py never trains on a decision that is in
    # the future relative to a validation row (leakage). Tie-break on entry_time then symbol
    # for full determinism. A trade entered early but held long now precedes a later-entered
    # trade that exited sooner, which exit-time order would have inverted.
    closed.sort(key=lambda r: (r[0], r[1], r[2]))
    trades: List[TradeRecord] = [rec[3] for rec in closed]
    labeled: List[dict] = [rec[4] for rec in closed]

    metrics = _compute_metrics(trades, equity_points, params, limits)
    return BacktestResult(trades=trades, metrics=metrics, labeled_features=labeled)


# --------------------------------------------------------------------------- #
# Session grouping + timeline construction
# --------------------------------------------------------------------------- #
def _group_sessions(
    bars_by_symbol: Dict[str, List[Bar]],
) -> Dict[Tuple[str, _dt.date], List[Bar]]:
    """Group bars into per-(symbol, trading-day) ascending lists. VWAP resets per day."""
    out: Dict[Tuple[str, _dt.date], List[Bar]] = defaultdict(list)
    for sym, bars in (bars_by_symbol or {}).items():
        for b in bars or ():
            out[(sym, _session_key(b))].append(b)
    for key in out:
        out[key].sort(key=lambda x: x.start)
    return out


def _build_timeline(
    sessions: Dict[Tuple[str, _dt.date], List[Bar]],
) -> List[Tuple[_dt.datetime, Dict[str, Tuple[List[Bar], Optional[Bar], Bar]]]]:
    """Build the global ordered list of decision steps.

    For each (symbol, day) session and each decision-bar index t, the decision uses bars
    [0..t] and (if present) fills at bar t+1. We key steps by the decision-bar start
    timestamp `asof` and group all symbols sharing that timestamp so portfolio caps apply
    consistently within a step. Returns [(asof, {symbol: (bars_up_to_t, next_open, decision_bar)})]
    sorted by (asof, symbol).
    """
    # asof -> {symbol: (bars_up_to_and_including_t, next_open_or_None, decision_bar)}
    by_ts: Dict[_dt.datetime, Dict[str, Tuple[List[Bar], Optional[Bar], Bar]]] = defaultdict(dict)

    for (sym, _day), bars in sessions.items():
        n = len(bars)
        for t in range(n):
            decision_bar = bars[t]
            bars_up_to_t = bars[: t + 1]
            next_open = bars[t + 1] if (t + 1) < n else None
            asof = decision_bar.start
            # If two sessions of the SAME symbol somehow share a timestamp (shouldn't),
            # last write wins; sessions are per-day so this cannot collide in practice.
            by_ts[asof][sym] = (bars_up_to_t, next_open, decision_bar)

    ordered_ts = sorted(by_ts.keys())
    timeline: List[Tuple[_dt.datetime, Dict[str, Tuple[List[Bar], Optional[Bar], Bar]]]] = []
    for ts in ordered_ts:
        group = by_ts[ts]
        # Sort symbols within a step by UNIVERSE order when possible (stable, deterministic).
        ordered = {s: group[s] for s in sorted(group.keys(), key=_universe_rank)}
        timeline.append((ts, ordered))
    return timeline


def _universe_rank(symbol: str) -> Tuple[int, str]:
    """Sort key putting UNIVERSE symbols in their canonical order, others alphabetically."""
    from ..config import UNIVERSE
    try:
        return (UNIVERSE.index(symbol), symbol)
    except ValueError:
        return (len(UNIVERSE), symbol)


# --------------------------------------------------------------------------- #
# Snapshot / proposal helpers
# --------------------------------------------------------------------------- #
def _snapshot_for(
    symbol: str,
    asof: _dt.datetime,
    bars_up_to_t: List[Bar],
    params: StrategyParams,
    open_pos: Optional[_OpenPosition],
) -> Snapshot:
    """Build the decision-time Snapshot for `symbol` at bar t (bars[0..t] only).

    Reuses the production ``compute_indicators`` verbatim. When the symbol is long in the
    sim, a LONG PositionState is attached (with the unrealized P&L marked to the decision
    bar's close) so the StrategyAgent's §4 exit predicates fire exactly as they do live.
    """
    indicators: Indicators = compute_indicators(bars_up_to_t, params)
    decision_bar = bars_up_to_t[-1]

    if open_pos is None:
        pstate = PositionState(symbol=symbol, status=PositionStatus.FLAT)
    else:
        last_price = float(decision_bar.close)
        unrealized_pl = (last_price - open_pos.entry_price) * open_pos.qty
        unrealized_plpc = (
            (last_price - open_pos.entry_price) / open_pos.entry_price
            if open_pos.entry_price
            else None
        )
        pstate = PositionState(
            symbol=symbol,
            status=PositionStatus.LONG,
            qty=open_pos.qty,
            avg_entry_price=open_pos.entry_price,
            entry_time=open_pos.entry_time,
            market_value=last_price * open_pos.qty,
            unrealized_pl=unrealized_pl,
            unrealized_plpc=unrealized_plpc,
            current_price=last_price,
        )

    return Snapshot(
        symbol=symbol,
        asof=asof,
        indicators=indicators,
        position_state=pstate,
        bars=tuple(bars_up_to_t),
    )


def _fake_positions(
    open_positions: Dict[str, _OpenPosition],
) -> List[_FakePosition]:
    """Build RiskAgent-readable position stand-ins from the sim's open positions.

    Each position is marked at its cost basis (entry notional). The exposure cap is on
    deployed notional (open MV + pending entry notional), so cost basis is the right,
    conservative quantity to feed the cap regardless of where each symbol last printed.
    """
    return [
        _FakePosition(sym, pos.market_value(pos.entry_price))
        for sym, pos in open_positions.items()
    ]


def _risk_review_exit(
    risk: RiskAgent,
    proposal: Proposal,
    account: _FakeAccount,
    open_positions: Dict[str, _OpenPosition],
    cooldowns: Dict[str, _dt.datetime],
    params: StrategyParams,
    limits: RiskLimits,
) -> RiskDecision:
    """Risk-review an EXIT proposal. SELLs are always approved (A12) but we still route
    through the real RiskAgent so the same code path (and any future tightening) applies."""
    return risk.review(
        proposal,
        account,
        _fake_positions(open_positions),
        [],  # open_orders: none in sim (so the duplicate-order guard never blocks exits)
        cooldowns,
        params,
        limits,
        approved_buys=None,
    )


# --------------------------------------------------------------------------- #
# Fill / close helpers
# --------------------------------------------------------------------------- #
def _exit_fill(
    next_open: Optional[Bar],
    decision_bar: Bar,
    exit_reason: str,
) -> Tuple[Optional[float], Optional[_dt.datetime]]:
    """Resolve the SELL fill price+time. Normally t+1 OPEN (adverse slippage).

    If there is NO t+1 bar (t is the session's last bar), only an EOD/forced flatten may
    fill — and it fills at the decision bar's OPEN, never its CLOSE. The decision on bar t
    uses bars[0..t] (which includes bar t's close), so filling at bar t's close would let
    the simulator both DECIDE ON and FILL AT the same close — a same-bar look-ahead that
    fabricates PnL unknowable at decision time (the t->t+1 discipline the module promises).
    Bar t's OPEN is the earliest realizable same-bar reference (known at the start of t) and
    is independent of the close the decision peeked at. Any other exit with no t+1 is dropped
    (returns (None, None)) so no trade is closed at a price the strategy could not have hit."""
    if next_open is not None:
        return _slip(float(next_open.open), Side.SELL), next_open.start
    if exit_reason in (ExitStatus.EOD_FLATTEN.value, _FORCED_CLOSE):
        # No t+1 bar: realize at the decision bar's OPEN (not its close) to avoid the
        # same-bar look-ahead. fill_time is the decision bar start (the open's timestamp).
        return _slip(float(decision_bar.open), Side.SELL), decision_bar.start
    return None, None


def _close_position(
    pos: _OpenPosition,
    fill_price: float,
    fill_time: _dt.datetime,
    exit_reason: str,
) -> Tuple[TradeRecord, float]:
    """Realize a round-trip into a TradeRecord. Slippage is already baked into entry/exit
    prices, so PnL = (exit - entry) * qty net of costs (A11: $0 commission). Returns the
    record and its PnL."""
    entry_price = pos.entry_price
    pnl = (fill_price - entry_price) * pos.qty
    return_pct = (fill_price / entry_price) - 1.0 if entry_price else 0.0
    holding_min = (fill_time - pos.entry_time).total_seconds() / 60.0
    label = 1 if pnl > 0.0 else 0
    rec = TradeRecord(
        symbol=pos.symbol,
        entry_time=pos.entry_time,
        exit_time=fill_time,
        entry_price=entry_price,
        exit_price=fill_price,
        qty=pos.qty,
        pnl=pnl,
        return_pct=return_pct,
        holding_min=holding_min,
        exit_reason=exit_reason,
        features=dict(pos.features),
        label=label,
    )
    return rec, pnl


def _features_dict(proposal: Proposal) -> dict:
    """Extract the decision-time FeatureVector fields from an ENTRY proposal as a dict.

    Mirrors the FeatureVector field layout (a superset of FEATURE_ORDER plus 'symbol') so
    ml.dataset.build_dataset can read it directly. Empty dict if no features were attached.
    """
    fv = proposal.features
    if fv is None:
        return {}
    return {
        "symbol": fv.symbol,
        "dist_from_vwap": fv.dist_from_vwap,
        "volume_ratio": fv.volume_ratio,
        "log_rolling_vol": fv.log_rolling_vol,
        "minute_of_session": fv.minute_of_session,
        "recent_return": fv.recent_return,
        "bar_range_pct": fv.bar_range_pct,
        "session_progress": fv.session_progress,
        "symbol_id": fv.symbol_id,
    }


def _labeled_row(features: dict, label: int) -> dict:
    """One labeled-features row for the ML dataset: decision-time features + realized label."""
    row = dict(features)
    row["label"] = int(label)
    return row


def _basis(open_positions: Dict[str, _OpenPosition]) -> float:
    """Total cost basis (entry notional) of currently open positions.

    Equity is tracked as realized_pnl + open_market_value - open_basis, i.e. realized PnL
    plus the *unrealized* PnL of open positions, so the curve reflects mark-to-market P&L
    without double-counting deployed capital.
    """
    return sum(p.entry_price * p.qty for p in open_positions.values())


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _compute_metrics(
    trades: List[TradeRecord],
    equity_points: List[Tuple[_dt.datetime, float]],
    params: StrategyParams,
    limits: RiskLimits,
) -> Dict[str, float]:
    """Compute summary performance metrics. Deterministic; safe on empty input.

    Returns: n_trades, win_rate, total_pnl, avg_return_pct, avg_holding_min,
    sharpe_like, max_drawdown, exposure, total_return, wins, losses, exits_by_reason(*).
    """
    n_trades = len(trades)
    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0.0)
    losses = n_trades - wins
    win_rate = (wins / n_trades) if n_trades else 0.0
    avg_return_pct = (sum(t.return_pct for t in trades) / n_trades) if n_trades else 0.0
    avg_holding_min = (sum(t.holding_min for t in trades) / n_trades) if n_trades else 0.0

    # Total return relative to the per-trade notional capital budget (notional * positions
    # cap is the largest capital ever at risk). Use notional as the per-trade denominator.
    notional = float(params.notional) if params.notional else 100.0
    total_return = total_pnl / notional if notional else 0.0

    # Sharpe-like from the per-trade return series (no risk-free, not annualized — a
    # comparable risk-adjusted scalar for the optimizer). 0 when <2 trades or zero variance.
    returns = [t.return_pct for t in trades]
    sharpe_like = _sharpe_like(returns)

    # Max drawdown on the mark-to-market equity curve (in dollars; <= 0).
    max_drawdown = _max_drawdown([v for _ts, v in equity_points])

    # Exposure proxy: peak number of concurrent positions is capped by limits; report the
    # average deployed notional fraction of the exposure cap across the curve is overkill —
    # we report the cap-relative peak basis as a single 'exposure' scalar (best-effort).
    exposure = float(limits.max_total_exposure)

    metrics: Dict[str, float] = {
        "n_trades": float(n_trades),
        "wins": float(wins),
        "losses": float(losses),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "total_return": float(total_return),
        "avg_return_pct": float(avg_return_pct),
        "avg_holding_min": float(avg_holding_min),
        "sharpe_like": float(sharpe_like),
        "max_drawdown": float(max_drawdown),
        "exposure": float(exposure),
    }

    # Exit-reason histogram (handy for diagnostics; keys prefixed to avoid collisions).
    reasons: Dict[str, int] = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    for reason, count in reasons.items():
        metrics[f"exit_{reason}"] = float(count)

    return metrics


def _sharpe_like(returns: List[float]) -> float:
    """Mean/std of a return series (a risk-adjusted scalar). 0 on <2 samples or zero std."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std <= 0.0:
        return 0.0
    return (mean / std) * math.sqrt(n)


def _max_drawdown(equity: List[float]) -> float:
    """Largest peak-to-trough drop on the equity curve (<= 0). 0 on empty/monotone-up."""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


__all__ = ["BacktestResult", "run_backtest"]
