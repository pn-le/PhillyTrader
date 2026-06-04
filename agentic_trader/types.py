"""types.py — the SHARED INTER-AGENT CONTRACT.

Every module in agentic_trader binds to these frozen dataclasses and enums.
Getting these exactly right prevents integration drift across parallel implementers.

Design rules:
  - All dataclasses are frozen (immutable) so they are safe to pass between agents.
  - Numbers are plain Python floats/ints (Alpaca returns strings; callers parse to float
    BEFORE constructing these types — see API_MAP.md §4).
  - Timestamps are timezone-aware datetimes. Bar.start is America/New_York (the minute the
    bar covers); event/log timestamps are emitted as UTC ISO strings by logging_util.
  - No I/O, no Alpaca imports here. Pure data. This keeps the contract import-light.
  - Enums use UPPERCASE names; their .value strings match the JSONL log vocabulary
    (lowercase for sides to mirror Alpaca's OrderSide values).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Enums  (string-valued so they serialize cleanly into JSONL logs)
# --------------------------------------------------------------------------- #
class Side(str, Enum):
    """Order side. Values mirror alpaca.trading.enums.OrderSide."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"  # used by Proposal when no action is taken (logged, never executed)


class Intent(str, Enum):
    """What a proposal is trying to do in strategy terms."""

    ENTRY = "entry"
    EXIT = "exit"
    NONE = "none"  # paired with Side.HOLD


class Verdict(str, Enum):
    """Risk agent's disposition of a proposal. Risk has VETO authority (INV-4)."""

    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"  # a.k.a. VETO


class PositionStatus(str, Enum):
    """High-level lifecycle classification used by the Position agent."""

    FLAT = "flat"
    LONG = "long"


class ExitStatus(str, Enum):
    """Why an open position should/should not exit, per STRATEGY_RULES §4.

    Priority when multiple fire (logging order): vwap_revert > max_hold > stop_loss.
    eod_flatten (A9) overrides all of the above. HOLD means no exit this cycle.
    """

    HOLD = "hold"
    VWAP_REVERT = "vwap_revert"
    MAX_HOLD = "max_hold"
    STOP_LOSS = "stop_loss"
    EOD_FLATTEN = "eod_flatten"


# --------------------------------------------------------------------------- #
# Market-data primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bar:
    """One completed 1-minute regular-hours bar (STRATEGY_RULES §1).

    `start` is the minute the bar covers, in America/New_York (e.g. 09:31).
    A bar is only ever constructed when complete (now_utc >= start_utc + 60s).
    """

    symbol: str
    start: _dt.datetime  # tz-aware, America/New_York
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: Optional[float] = None
    vwap: Optional[float] = None  # Alpaca's per-bar vwap; NOT the session vwap


@dataclass(frozen=True)
class Indicators:
    """Per-symbol indicators computed from completed session bars (STRATEGY_RULES §2).

    `valid` is False when the symbol cannot be evaluated for ENTRY this cycle
    (n < 21, degenerate volume, etc.); `invalid_reason` explains why. Exits may still
    be evaluated even when valid is False (exits never need the rolling window).
    """

    symbol: str
    session_vwap: Optional[float]
    last_price: Optional[float]
    dist_from_vwap: Optional[float]  # (last_price - session_vwap)/session_vwap; <0 = below
    current_volume: Optional[float]
    rolling20_avg_vol: Optional[float]  # mean of 20 bars STRICTLY preceding decision bar
    volume_ratio: Optional[float]  # current_volume / rolling20_avg_vol
    n_bars: int  # number of completed session bars up to & including the decision bar
    valid: bool = False  # True iff this symbol is eligible for ENTRY evaluation
    invalid_reason: Optional[str] = None


@dataclass(frozen=True)
class PositionState:
    """Reconciled position for one symbol (STRATEGY_RULES §7).

    FLAT: status == PositionStatus.FLAT and qty == 0.0.
    LONG: status == PositionStatus.LONG with qty/avg_entry_price/entry_time populated.
    """

    symbol: str
    status: PositionStatus = PositionStatus.FLAT
    qty: float = 0.0
    avg_entry_price: Optional[float] = None
    entry_time: Optional[_dt.datetime] = None  # tz-aware; source per A6
    market_value: Optional[float] = None
    unrealized_pl: Optional[float] = None
    unrealized_plpc: Optional[float] = None
    current_price: Optional[float] = None

    @property
    def is_flat(self) -> bool:
        return self.status == PositionStatus.FLAT or self.qty == 0.0

    @property
    def is_long(self) -> bool:
        return self.status == PositionStatus.LONG and self.qty > 0.0


@dataclass(frozen=True)
class Snapshot:
    """The complete per-symbol view handed from Market Data → Strategy.

    `asof` is the decision-bar start (America/New_York) — the minute t we decide on.
    `bars` is the completed session-bar window used to compute `indicators` (for ML
    features and audit). It MUST contain only completed bars with start <= asof.
    """

    symbol: str
    asof: _dt.datetime  # decision-bar start, tz-aware America/New_York
    indicators: Indicators
    position_state: PositionState
    bars: Tuple[Bar, ...] = ()  # completed session bars [0..n-1], ascending by start


# --------------------------------------------------------------------------- #
# ML feature vector
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureVector:
    """Decision-time feature vector for the ML entry scorer (STRATEGY_RULES §5.1).

    ALL fields are known at minute t — absolutely NO future data. `symbol_id` is the
    integer index of the symbol in UNIVERSE (A8). `to_row()` and FEATURE_ORDER live in
    ml/features.py so live + training produce identical layouts.
    """

    symbol: str
    dist_from_vwap: float
    volume_ratio: float
    log_rolling_vol: float  # log1p(rolling20_avg_vol)
    minute_of_session: int  # minutes since 09:30 of decision bar, 0..389
    recent_return: float  # close[t]/close[t-5] - 1.0  (K=RECENT_RETURN_K)
    bar_range_pct: float  # (high-low)/close on decision bar
    session_progress: float  # minute_of_session / 389.0, in [0,1]
    symbol_id: int  # UNIVERSE.index(symbol), 0..9 (categorical)


# --------------------------------------------------------------------------- #
# Strategy → Risk → Execution contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proposal:
    """A strategy proposal (STRATEGY_RULES §3/§4; ARCHITECTURE §7).

    ENTRY proposals carry `notional` (USD, default 100.0) and a populated `features` /
    `ml_p_win`. EXIT proposals carry `qty` (full position) and leave notional None.
    HOLD proposals (side=HOLD, intent=NONE) are logged but never reach Risk/Execution.

    `signal_values` is a free-form dict of the indicator values that drove the decision
    (for logging/audit), e.g. {"dist_from_vwap": -0.0061, "volume_ratio": 1.43}.
    """

    symbol: str
    side: Side
    intent: Intent
    reason: str
    notional: Optional[float] = None  # set for ENTRY (BUY); None for EXIT
    qty: Optional[float] = None  # set for EXIT (full SELL); None for ENTRY
    signal_values: Dict[str, Any] = field(default_factory=dict)
    ml_p_win: Optional[float] = None  # populated for ENTRY after the ML gate
    features: Optional[FeatureVector] = None  # decision-time features for ENTRY
    asof: Optional[_dt.datetime] = None  # decision-bar start for traceability

    @property
    def is_entry(self) -> bool:
        return self.intent == Intent.ENTRY and self.side == Side.BUY

    @property
    def is_exit(self) -> bool:
        return self.intent == Intent.EXIT and self.side == Side.SELL


@dataclass(frozen=True)
class RiskDecision:
    """Risk agent's verdict on a single proposal (STRATEGY_RULES §8; INV-4).

    `adjusted_notional` is set only for Verdict.REDUCE (the reduced BUY notional).
    For APPROVE it mirrors proposal.notional; for REJECT it is None. `reason` is
    ALWAYS populated (mandatory for non-approvals, informative otherwise).
    """

    proposal: Proposal
    verdict: Verdict
    reason: str
    adjusted_notional: Optional[float] = None

    @property
    def approved(self) -> bool:
        return self.verdict in (Verdict.APPROVE, Verdict.REDUCE)

    @property
    def effective_notional(self) -> Optional[float]:
        """The notional Execution should actually use (reduced if REDUCE)."""
        if self.verdict == Verdict.REDUCE:
            return self.adjusted_notional
        if self.verdict == Verdict.APPROVE:
            return self.proposal.notional
        return None


@dataclass(frozen=True)
class OrderResult:
    """Outcome of an execution attempt (ARCHITECTURE §7; INV-1/2/3).

    `dry_run=True` means NOTHING was submitted to the broker (the default mode). In that
    case `id`/`status` describe the intended order (status typically "dry_run") and
    `submitted_at` is the local decision time. When dry_run is False, fields mirror the
    Alpaca Order response (id/status/submitted_at from the broker).
    """

    symbol: str
    side: Side
    status: str  # broker OrderStatus value, or "dry_run" / "refused" / "error"
    dry_run: bool
    submitted_at: _dt.datetime  # tz-aware UTC
    id: Optional[str] = None  # broker order id (None for dry-run/refused/error)
    notional: Optional[float] = None
    qty: Optional[float] = None
    filled_avg_price: Optional[float] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Position analysis + ML labels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PositionSummary:
    """Per-position P&L / exposure snapshot + exit classification (Position agent).

    `exit_status` classifies whether (and why) this position should exit this cycle, so
    the orchestrator can log/act consistently. `would_exit` is a convenience flag.
    """

    symbol: str
    qty: float
    avg_entry_price: Optional[float]
    current_price: Optional[float]
    market_value: Optional[float]
    unrealized_pl: Optional[float]
    unrealized_plpc: Optional[float]
    holding_minutes: Optional[float]
    exit_status: ExitStatus = ExitStatus.HOLD
    would_exit: bool = False


@dataclass(frozen=True)
class TradeRecord:
    """A closed (or simulated) round-trip trade — the ML training row source.

    Costs are explicit (A11): $0 commission, 1bp adverse slippage. `label` is 1 iff
    `pnl > 0` (realized_pnl_after_costs > 0), else 0. `features` is the decision-time
    feature dict (FeatureVector.__dict__-like) captured at ENTRY; never future data.
    """

    symbol: str
    entry_time: _dt.datetime  # tz-aware
    exit_time: _dt.datetime  # tz-aware
    entry_price: float  # adverse-slippage-adjusted buy fill
    exit_price: float  # adverse-slippage-adjusted sell fill
    qty: float
    pnl: float  # realized_pnl_after_costs
    return_pct: float  # (exit_price/entry_price) - 1.0
    holding_min: float
    exit_reason: str  # ExitStatus value that closed the trade
    features: Dict[str, Any] = field(default_factory=dict)  # decision-time features
    label: int = 0  # 1 if pnl > 0 else 0


# Public re-exports so `from agentic_trader.types import *` is well-defined.
__all__ = [
    "Side",
    "Intent",
    "Verdict",
    "PositionStatus",
    "ExitStatus",
    "Bar",
    "Indicators",
    "PositionState",
    "Snapshot",
    "FeatureVector",
    "Proposal",
    "RiskDecision",
    "OrderResult",
    "PositionSummary",
    "TradeRecord",
]
