"""strategy_agent.py — Agent ② Strategy (deterministic VWAP mean-reversion).

PURE decision logic ONLY. The StrategyAgent turns a per-symbol Snapshot into a single
Proposal by applying STRATEGY_RULES.md §3 (ENTRY), §4 (EXIT), and §5 (the ML entry gate)
EXACTLY. It does NO risk checking, NO order placement, and makes NO Alpaca calls — those
belong to the Risk and Execution agents. Decisioning is deterministic: identical inputs
always produce an identical Proposal (INV: no LLM in the trade loop).

Contract (INTERFACE_SPEC §3):
    StrategyAgent(params, settings=None, logger=None)
    StrategyAgent.propose(snapshot, params=None, ml_scorer=None) -> Proposal

Decision order, per STRATEGY_RULES §10:
  1. If LONG: evaluate EXIT predicates (§4). eod_flatten (>= 15:55 NY) overrides all
     others; otherwise priority vwap_revert > max_hold > stop_loss. On any trigger ->
     SELL/EXIT for the FULL position qty with the triggering reason.
  2. Else if FLAT: in the EOD window (>= 15:55 NY) no new entries -> HOLD. Otherwise,
     if indicators.valid and the §3 entry predicate passes, build the decision-time
     FeatureVector and run the ML gate (§5): p_win = ml_scorer.p_win(features) (1.0 if
     no scorer / passthrough); require p_win >= params.ml_threshold. On pass ->
     BUY/ENTRY for params.notional (carrying ml_p_win + features). On fail -> HOLD.
  3. Else -> HOLD.

NO look-ahead: everything is a function of snapshot.indicators / snapshot.bars, which the
Market Data agent guarantees contain only completed bars with start <= asof.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..config import EOD_FLATTEN_TIME, Settings, StrategyParams
from ..ml.features import build_features
from ..types import (
    ExitStatus,
    Indicators,
    Intent,
    PositionState,
    Proposal,
    Side,
    Snapshot,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard import dependency
    from ..ml.scorer import EntryScorer


def _parse_eod_time(raw: str = EOD_FLATTEN_TIME) -> _dt.time:
    """Parse the EOD flatten cutoff ('15:55:00') into a datetime.time. Pure."""
    parts = [int(p) for p in raw.split(":")]
    while len(parts) < 3:
        parts.append(0)
    h, m, s = parts[0], parts[1], parts[2]
    return _dt.time(hour=h, minute=m, second=s)


# Cutoff computed once at import (a constant); comparison stays time-only / tz-naive-safe.
_EOD_CUTOFF = _parse_eod_time()


def _is_eod_window(asof: Optional[_dt.datetime]) -> bool:
    """True iff the decision-bar start is at/after 15:55 NY (A9 end-of-day window).

    `asof` is the decision-bar start in America/New_York (per the Snapshot contract). We
    compare its wall-clock time-of-day against EOD_FLATTEN_TIME. If `asof` is missing we
    conservatively report False (treat as in-session) so we never spuriously block exits/
    entries on a malformed snapshot — the higher layers still gate on validity.
    """
    if asof is None:
        return False
    return asof.timetz().replace(tzinfo=None) >= _EOD_CUTOFF


def _holding_minutes(asof: Optional[_dt.datetime], entry_time: Optional[_dt.datetime]) -> Optional[float]:
    """Minutes held = (decision_bar.start - entry_time) in minutes (STRATEGY_RULES §4).

    Defensive against tz-naive inputs: `asof` is tz-aware per the Snapshot contract, but
    `entry_time` is reconstructed from broker order timestamps / position attrs and could be
    tz-naive (a mocked/proxied/future SDK or a manually-seeded position). A raw subtraction
    of an aware and a naive datetime raises TypeError, which — since this is computed at the
    TOP of _propose_exit — would abort the ENTIRE exit evaluation (vwap_revert / stop_loss /
    EOD flatten included), silently leaving the position unmanaged and possibly carried past
    15:55. So we normalize naive datetimes to UTC (mirroring PositionAgent._holding_minutes)
    and NEVER raise: on any failure we return None, which simply means max_hold cannot fire —
    the price-based and EOD exits still evaluate normally."""
    if asof is None or entry_time is None:
        return None
    try:
        a = asof if asof.tzinfo is not None else asof.replace(tzinfo=_dt.timezone.utc)
        e = entry_time if entry_time.tzinfo is not None else entry_time.replace(tzinfo=_dt.timezone.utc)
        return (a - e).total_seconds() / 60.0
    except (TypeError, ValueError, AttributeError):
        return None


def _signal_values(ind: Indicators, pos: PositionState, **extra: Any) -> Dict[str, Any]:
    """Assemble the free-form audit dict of indicator values that drove the decision."""
    sv: Dict[str, Any] = {
        "session_vwap": ind.session_vwap,
        "last_price": ind.last_price,
        "dist_from_vwap": ind.dist_from_vwap,
        "current_volume": ind.current_volume,
        "rolling20_avg_vol": ind.rolling20_avg_vol,
        "volume_ratio": ind.volume_ratio,
        "n_bars": ind.n_bars,
    }
    if pos.is_long:
        sv["qty"] = pos.qty
        sv["avg_entry_price"] = pos.avg_entry_price
        sv["unrealized_plpc"] = pos.unrealized_plpc
    sv.update(extra)
    return sv


class StrategyAgent:
    """Deterministic intraday VWAP mean-reversion strategy (proposals only).

    Stateless apart from its default params/settings/logger. `propose()` is a pure
    function of its inputs (and the optional ml_scorer), so the agent can be reused
    across symbols and cycles without accumulating state.
    """

    def __init__(
        self,
        params: StrategyParams,
        settings: Optional[Settings] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.params = params
        self.settings = settings or Settings()
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def propose(
        self,
        snapshot: Snapshot,
        params: Optional[StrategyParams] = None,
        ml_scorer: "EntryScorer | None" = None,
    ) -> Proposal:
        """Produce one Proposal for `snapshot` (STRATEGY_RULES §3/§4/§5).

        Exits are evaluated before entries. Uses `params` if supplied, else the agent's
        configured params. `ml_scorer` (if given) gates ENTRIES only; a None/passthrough
        scorer yields p_win == 1.0 (rule-only). Never mutates inputs.
        """
        p = params if params is not None else self.params
        ind = snapshot.indicators
        pos = snapshot.position_state
        symbol = snapshot.symbol
        asof = snapshot.asof

        # 1) EXIT first (only relevant if we are long).
        if pos.is_long:
            return self._propose_exit(snapshot, p)

        # 2) ENTRY (only relevant if we are flat).
        if pos.is_flat:
            return self._propose_entry(snapshot, p, ml_scorer)

        # 3) Fallback (should not occur — status is FLAT or LONG).
        return Proposal(
            symbol=symbol,
            side=Side.HOLD,
            intent=Intent.NONE,
            reason="indeterminate position state; holding",
            signal_values=_signal_values(ind, pos),
            asof=asof,
        )

    # ------------------------------------------------------------------ #
    # EXIT logic (STRATEGY_RULES §4 + A9)
    # ------------------------------------------------------------------ #
    def _propose_exit(self, snapshot: Snapshot, p: StrategyParams) -> Proposal:
        """Classify the open position's exit per §4. eod_flatten overrides all others;
        otherwise priority vwap_revert > max_hold > stop_loss. HOLD if nothing fires."""
        ind = snapshot.indicators
        pos = snapshot.position_state
        symbol = snapshot.symbol
        asof = snapshot.asof

        dist = ind.dist_from_vwap
        last_price = ind.last_price
        holding_min = _holding_minutes(asof, pos.entry_time)

        # Prefer the broker-reported unrealized_plpc; else compute from last_price.
        unrealized_plpc = pos.unrealized_plpc
        if unrealized_plpc is None and last_price is not None and pos.avg_entry_price:
            unrealized_plpc = (last_price - pos.avg_entry_price) / pos.avg_entry_price

        exit_status = ExitStatus.HOLD
        reason = ""

        # A9: end-of-day flatten overrides everything (no overnight positions).
        if _is_eod_window(asof):
            exit_status = ExitStatus.EOD_FLATTEN
            reason = (
                f"EOD flatten: decision bar {self._fmt_time(asof)} NY is at/after "
                f"{EOD_FLATTEN_TIME}; force-exit {symbol} to avoid overnight exposure."
            )
        # vwap_revert (highest non-EOD priority): price back within band of VWAP.
        # Requires a defined dist_from_vwap (A4: undefined VWAP -> this exit cannot fire).
        elif dist is not None and dist >= -p.vwap_exit_band:
            exit_status = ExitStatus.VWAP_REVERT
            reason = (
                f"VWAP revert: dist_from_vwap {dist:+.4f} >= -{p.vwap_exit_band:.4f} "
                f"(price returned to within {p.vwap_exit_band * 100:.2f}% of VWAP)."
            )
        # max_hold: held strictly longer than the cap.
        elif holding_min is not None and holding_min > p.max_hold:
            exit_status = ExitStatus.MAX_HOLD
            reason = (
                f"Max hold: held {holding_min:.1f} min > max_hold {p.max_hold:.0f} min."
            )
        # stop_loss: unrealized loss reached the cap.
        elif unrealized_plpc is not None and unrealized_plpc <= -p.stop_loss:
            exit_status = ExitStatus.STOP_LOSS
            reason = (
                f"Stop loss: unrealized_plpc {unrealized_plpc:+.4f} <= -{p.stop_loss:.4f} "
                f"({-unrealized_plpc * 100:.2f}% loss)."
            )

        sv = _signal_values(
            ind,
            pos,
            holding_minutes=holding_min,
            unrealized_plpc=unrealized_plpc,
            exit_status=exit_status.value,
        )

        if exit_status == ExitStatus.HOLD:
            return Proposal(
                symbol=symbol,
                side=Side.HOLD,
                intent=Intent.NONE,
                reason=(
                    f"Hold {symbol}: long {pos.qty:g} sh, no exit trigger "
                    f"(dist_from_vwap={self._fmt(dist)}, holding={self._fmt(holding_min)} min, "
                    f"unrealized_plpc={self._fmt(unrealized_plpc)})."
                ),
                signal_values=sv,
                asof=asof,
            )

        return Proposal(
            symbol=symbol,
            side=Side.SELL,
            intent=Intent.EXIT,
            reason=reason,
            qty=pos.qty,
            signal_values=sv,
            asof=asof,
        )

    # ------------------------------------------------------------------ #
    # ENTRY logic (STRATEGY_RULES §3 + §5 ML gate)
    # ------------------------------------------------------------------ #
    def _propose_entry(
        self,
        snapshot: Snapshot,
        p: StrategyParams,
        ml_scorer: "EntryScorer | None",
    ) -> Proposal:
        """Apply §3 entry predicate then the §5 ML gate. HOLD on any failure."""
        ind = snapshot.indicators
        pos = snapshot.position_state
        symbol = snapshot.symbol
        asof = snapshot.asof
        sv = _signal_values(ind, pos)

        def hold(reason: str) -> Proposal:
            return Proposal(
                symbol=symbol,
                side=Side.HOLD,
                intent=Intent.NONE,
                reason=reason,
                signal_values=sv,
                asof=asof,
            )

        # A9: no NEW entries on/after 15:55 NY.
        if _is_eod_window(asof):
            return hold(
                f"No entry {symbol}: decision bar {self._fmt_time(asof)} NY is in the "
                f"EOD window (>= {EOD_FLATTEN_TIME}); new entries are disabled."
            )

        # Min-bars / degenerate-data guard (A1/A4): indicators.valid gates entry.
        if not ind.valid:
            return hold(
                f"No entry {symbol}: indicators not valid for entry "
                f"({ind.invalid_reason or 'insufficient data'}; n_bars={ind.n_bars})."
            )

        dist = ind.dist_from_vwap
        vol_ratio = ind.volume_ratio
        if dist is None or vol_ratio is None:
            return hold(
                f"No entry {symbol}: indicators incomplete "
                f"(dist_from_vwap={self._fmt(dist)}, volume_ratio={self._fmt(vol_ratio)})."
            )

        # §3 entry predicate (BOTH conditions required).
        cond_below_vwap = dist <= -p.entry_dist
        cond_volume = vol_ratio >= p.vol_mult
        if not (cond_below_vwap and cond_volume):
            return hold(
                f"No entry {symbol}: rule predicate not met "
                f"(dist_from_vwap {dist:+.4f} vs <= -{p.entry_dist:.4f} -> "
                f"{'pass' if cond_below_vwap else 'fail'}; "
                f"volume_ratio {vol_ratio:.2f} vs >= {p.vol_mult:.2f} -> "
                f"{'pass' if cond_volume else 'fail'})."
            )

        # §5 ML gate. Build decision-time features; passthrough scorer -> p_win = 1.0.
        try:
            features = build_features(snapshot, p)
        except ValueError as exc:
            # Defensive: indicators.valid is True but features still cannot be formed.
            return hold(f"No entry {symbol}: cannot build features ({exc}).")

        if ml_scorer is None:
            p_win = 1.0
        else:
            p_win = float(ml_scorer.p_win(features))

        gate_pass = p_win >= p.ml_threshold
        self._log_ml_score(symbol, p_win, p.ml_threshold, gate_pass)

        if not gate_pass:
            return Proposal(
                symbol=symbol,
                side=Side.HOLD,
                intent=Intent.NONE,
                reason=(
                    f"ML gate blocked {symbol}: p_win {p_win:.4f} < ml_threshold "
                    f"{p.ml_threshold:.4f} (rules passed, learned gate vetoed entry)."
                ),
                signal_values=sv,
                ml_p_win=p_win,
                features=features,
                asof=asof,
            )

        # ENTRY approved by strategy: propose a $notional MARKET BUY.
        return Proposal(
            symbol=symbol,
            side=Side.BUY,
            intent=Intent.ENTRY,
            reason=(
                f"Entry {symbol}: dist_from_vwap {dist:+.4f} <= -{p.entry_dist:.4f} "
                f"({-dist * 100:.2f}% below VWAP) and volume_ratio {vol_ratio:.2f} "
                f">= {p.vol_mult:.2f}; ML p_win {p_win:.4f} >= {p.ml_threshold:.4f}. "
                f"BUY ${p.notional:.0f} notional (MARKET)."
            ),
            notional=p.notional,
            signal_values=sv,
            ml_p_win=p_win,
            features=features,
            asof=asof,
        )

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _log_ml_score(self, symbol: str, p_win: float, ml_threshold: float, gate_pass: bool) -> None:
        """Emit an ml_score event if a logger is attached. Never raises into decisioning."""
        if self.logger is None:
            return
        try:
            self.logger.log_ml_score(
                symbol,
                p_win,
                ml_threshold,
                "pass" if gate_pass else "fail",
            )
        except Exception:  # pragma: no cover - logging must never break the trade loop
            pass

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        """Format an optional float for human-readable reasons ('n/a' when None)."""
        return "n/a" if value is None else f"{value:.4f}"

    @staticmethod
    def _fmt_time(asof: Optional[_dt.datetime]) -> str:
        return "n/a" if asof is None else asof.strftime("%H:%M:%S")


__all__ = ["StrategyAgent"]
