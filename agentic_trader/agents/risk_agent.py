"""agents/risk_agent.py — Agent ④ Risk (VETO authority).

The Risk agent is the single chokepoint between Strategy and Execution (INV-4). Strategy
proposes; Risk approves / reduces / vetoes — and ONLY approved (possibly reduced) orders
ever reach Execution. Every non-approval carries a clear, logged reason.

This module is PURE policy + deterministic accounting: it places NO orders, performs NO
network I/O, and never mutates the broker objects it inspects. It simply returns a
`RiskDecision` per `Proposal` per STRATEGY_RULES §8 and INTERFACE_SPEC §4.

Caps enforced (against RECONCILED broker state + this-cycle prior approvals):
  - is_paper_env: refuse everything if ALPACA_PAPER != "true" (INV-3 precondition).
  - max_open_positions: count current open positions + pending this-cycle entries.
  - max_total_exposure ($500): open market value + approved-buy notional + this proposal.
  - per-symbol cooldown (10 min): at most one new ENTRY per symbol per window.
  - duplicate-order guard (A17): reject if the broker already shows an OPEN order for the
    symbol (applies to BOTH entries and exits — never stack a second order).
  - account sanity: refuse if the account is blocked / trading is blocked / suspended.

EXIT (SELL) proposals are risk-reducing and are APPROVED (A12) unless an open order for
that symbol already exists. Where an ENTRY would breach the exposure cap but partially
fits, the notional is REDUCED to the remaining room (A13) instead of rejected; if the
room is below the Alpaca minimum notional, it is rejected.

Broker objects (`account`, `positions`, `open_orders`) are raw Alpaca Pydantic models
whose numeric fields are STRINGS (API_MAP §4); we parse defensively with float().
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from ..config import MIN_NOTIONAL, RiskLimits, Settings, StrategyParams, is_paper_env
from ..logging_util import JsonlLogger
from ..types import Proposal, RiskDecision, Verdict


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce an Alpaca string/numeric field to float, tolerating None/garbage.

    Alpaca returns numeric fields as strings (API_MAP §4); this never raises so a single
    malformed field cannot crash a risk review.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _symbol_of(obj: Any) -> Optional[str]:
    """Best-effort symbol extraction from a raw Alpaca position/order object."""
    return getattr(obj, "symbol", None)


def _open_order_symbols(open_orders: List[Any]) -> set:
    """Set of symbols that currently have an OPEN order at the broker (A17 guard)."""
    out: set = set()
    for o in open_orders or ():
        sym = _symbol_of(o)
        if sym:
            out.add(sym)
    return out


def _order_is_buy(order: Any) -> bool:
    """True iff a raw Alpaca order is a BUY (its committed notional adds to exposure)."""
    side = getattr(order, "side", None)
    val = str(getattr(side, "value", side) or "").lower()
    return val == "buy"


def _open_buy_orders(open_orders: List[Any]) -> List[Any]:
    """The subset of reconciled OPEN orders that are BUYs (pending NEW entries)."""
    return [o for o in (open_orders or ()) if _order_is_buy(o)]


def _open_order_notional(order: Any) -> float:
    """Committed USD notional of a single OPEN BUY order (best-effort, conservative).

    Prefers the order's `notional` (how this system sizes market entries). Falls back to
    qty * a per-share price (limit_price, else filled_avg_price). If neither is derivable
    the order still consumes a position SLOT (counted separately); its notional contribution
    is then 0.0 rather than a fabricated number.
    """
    notional = _to_float(getattr(order, "notional", None), 0.0)
    if notional > 0.0:
        return notional
    qty = _to_float(getattr(order, "qty", None), 0.0)
    if qty > 0.0:
        price = _to_float(getattr(order, "limit_price", None), 0.0) or _to_float(
            getattr(order, "filled_avg_price", None), 0.0
        )
        if price > 0.0:
            return qty * price
    return 0.0


def _pending_open_buy_notional(open_orders: List[Any], exclude_symbol: Optional[str] = None) -> float:
    """Sum committed notional of OPEN BUY orders (optionally excluding one symbol)."""
    total = 0.0
    for o in _open_buy_orders(open_orders):
        if exclude_symbol is not None and _symbol_of(o) == exclude_symbol:
            continue
        total += _open_order_notional(o)
    return total


def _total_open_market_value(positions: List[Any]) -> float:
    """Sum of float(position.market_value) over reconciled open positions."""
    return sum(_to_float(getattr(p, "market_value", 0.0)) for p in (positions or ()))


def _pending_buy_notional(approved_buys: Optional[List[RiskDecision]]) -> float:
    """Sum the effective notional of buys already approved earlier THIS cycle (A14).

    Uses each decision's `effective_notional` (the reduced amount for REDUCE, the
    proposal notional for APPROVE), so a single dollar of room serves one proposal only.
    """
    total = 0.0
    for d in approved_buys or ():
        if d is None or not d.approved:
            continue
        eff = d.effective_notional
        if eff is not None:
            total += float(eff)
    return total


class RiskAgent:
    """Reviews strategy proposals and enforces hard risk caps with VETO authority.

    Deterministic and side-effect-free except for optional JSONL logging of each
    decision. The orchestrator threads `approved_buys` (this-cycle prior approvals) so
    concurrent proposals cannot both consume one open slot or one dollar of exposure.
    """

    def __init__(
        self,
        limits: RiskLimits,
        settings: Optional[Settings] = None,
        logger: Optional[JsonlLogger] = None,
    ) -> None:
        self.limits = limits
        self.settings = settings or Settings()
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def review(
        self,
        proposal: Proposal,
        account: Any,
        positions: List[Any],
        open_orders: List[Any],
        cooldowns: Dict[str, _dt.datetime],
        params: StrategyParams,
        limits: Optional[RiskLimits] = None,
        approved_buys: Optional[List[RiskDecision]] = None,
    ) -> RiskDecision:
        """Return a RiskDecision (reason ALWAYS set) per STRATEGY_RULES §8 + INV-4.

        Order of checks (first failing check wins):
          1. ALPACA_PAPER != "true"            -> REJECT("not_a_paper_account")  [INV-3]
          2. HOLD proposal                     -> REJECT("no_action")            (never executed)
          3. account/trading blocked           -> REJECT(<sanity reason>)
          4. open order already exists (A17)    -> REJECT("open_order_pending")
          5. SELL/EXIT                          -> APPROVE (risk-reducing, A12)
          6. BUY: max_open_positions            -> REJECT("max_open_positions")
          7. BUY: per-symbol cooldown           -> REJECT("cooldown")
          8. BUY: exposure cap                  -> REDUCE / REJECT  (A13)
          9. otherwise                          -> APPROVE

        `cooldowns[symbol]` is the timestamp of the last ENTRY for that symbol (derived
        from the persisted trade log, A5); a missing/None entry means no cooldown applies.
        """
        limits = limits or self.limits
        decision = self._review(proposal, account, positions, open_orders, cooldowns, limits, approved_buys)
        if self.logger is not None:
            try:
                self.logger.log_risk(decision)
            except Exception:  # logging must never break the trade loop (INV-5 best-effort)
                pass
        return decision

    # ------------------------------------------------------------------ #
    # Internal policy (no logging side effects)
    # ------------------------------------------------------------------ #
    def _review(
        self,
        proposal: Proposal,
        account: Any,
        positions: List[Any],
        open_orders: List[Any],
        cooldowns: Dict[str, _dt.datetime],
        limits: RiskLimits,
        approved_buys: Optional[List[RiskDecision]],
    ) -> RiskDecision:
        # 1. Paper-account precondition (INV-3). Risk refuses to approve anything in a
        #    non-paper environment — Execution enforces this too, but Risk is the gate.
        if not is_paper_env(self.settings.env_path):
            return RiskDecision(proposal, Verdict.REJECT, "not_a_paper_account")

        # 2. HOLD proposals are logged upstream but must never be executed.
        if proposal.side.value == "hold" or proposal.intent.value == "none":
            return RiskDecision(proposal, Verdict.REJECT, "no_action")

        # 3. Basic account sanity — refuse to act on a blocked/suspended account.
        sanity_reason = self._account_sanity(account)
        if sanity_reason is not None:
            return RiskDecision(proposal, Verdict.REJECT, sanity_reason)

        # 4. Duplicate-order guard (A17): never stack a second order on a symbol that
        #    already has an OPEN order at the broker (applies to entries AND exits).
        if proposal.symbol in _open_order_symbols(open_orders):
            return RiskDecision(proposal, Verdict.REJECT, "open_order_pending")

        # 5. EXIT / SELL proposals are risk-reducing — always approved (A12).
        if proposal.is_exit or proposal.side.value == "sell":
            return RiskDecision(proposal, Verdict.APPROVE, "exit_approved")

        # ---- BUY / ENTRY proposals only below ---- #
        return self._review_entry(proposal, positions, open_orders, cooldowns, limits, approved_buys)

    def _review_entry(
        self,
        proposal: Proposal,
        positions: List[Any],
        open_orders: List[Any],
        cooldowns: Dict[str, _dt.datetime],
        limits: RiskLimits,
        approved_buys: Optional[List[RiskDecision]],
    ) -> RiskDecision:
        # Pending NEW entries already committed at the broker but NOT YET filled (so they are
        # neither a Position nor in this-cycle approved_buys) must still count toward BOTH
        # caps — they are committed capital / occupied slots. Without this, a market BUY from
        # cycle N that is still 'new/pending_new' in cycle N+1 is invisible to the exposure
        # and position-count math, letting cycle N+1 over-commit on OTHER symbols and breach
        # the $500 / 4-position caps cross-cycle. We exclude this proposal's OWN symbol from
        # the notional sum to avoid double-counting it against the A17 duplicate guard (which
        # already blocks a second order on the same symbol).
        open_buys = _open_buy_orders(open_orders)
        pending_order_count = len(open_buys)
        pending_order_notional = _pending_open_buy_notional(open_orders, exclude_symbol=proposal.symbol)

        # 6. Max open positions — count current open positions plus pending this-cycle
        #    entries (A14) plus pending OPEN BUY orders from prior cycles (each occupies a slot).
        pending_entry_count = sum(1 for d in (approved_buys or ()) if d is not None and d.approved)
        open_count = len(positions or ()) + pending_entry_count + pending_order_count
        if open_count >= limits.max_open_positions:
            return RiskDecision(proposal, Verdict.REJECT, "max_open_positions")

        # 7. Per-symbol cooldown — at most one new ENTRY per symbol per window (A5).
        cooldown_reason = self._cooldown_violation(proposal.symbol, cooldowns, limits)
        if cooldown_reason is not None:
            return RiskDecision(proposal, Verdict.REJECT, cooldown_reason)

        # 8. Exposure cap (A13). open MV + already-approved buy notional + pending OPEN BUY
        #    order notional + this proposal must stay <= max_total_exposure. If it breaches,
        #    compute the remaining room; REDUCE to the room when room >= MIN_NOTIONAL, else REJECT.
        requested = float(proposal.notional) if proposal.notional is not None else 0.0
        if requested <= 0.0:
            return RiskDecision(proposal, Verdict.REJECT, "invalid_notional")

        open_mv = _total_open_market_value(positions)
        pending = _pending_buy_notional(approved_buys) + pending_order_notional
        projected = open_mv + pending + requested
        cap = float(limits.max_total_exposure)

        if projected > cap:
            room = cap - open_mv - pending
            if room <= 0.0:
                return RiskDecision(proposal, Verdict.REJECT, "max_total_exposure")
            if room < MIN_NOTIONAL:
                return RiskDecision(proposal, Verdict.REJECT, "exposure_room_below_min")
            return RiskDecision(
                proposal,
                Verdict.REDUCE,
                f"reduced_to_exposure_room:{room:.2f}",
                adjusted_notional=round(room, 2),
            )

        # 9. Clear to enter at the full requested notional.
        return RiskDecision(proposal, Verdict.APPROVE, "entry_approved")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _account_sanity(account: Any) -> Optional[str]:
        """Return a reject reason if the account/trading is blocked, else None.

        Tolerates a missing `account` (None) — treated as sane so risk reviews can run
        in tests/backtests that don't pass a TradeAccount.
        """
        if account is None:
            return None
        if bool(getattr(account, "account_blocked", False)):
            return "account_blocked"
        if bool(getattr(account, "trading_blocked", False)):
            return "trading_blocked"
        if bool(getattr(account, "trade_suspended_by_user", False)):
            return "trade_suspended"
        return None

    @staticmethod
    def _cooldown_violation(
        symbol: str,
        cooldowns: Dict[str, _dt.datetime],
        limits: RiskLimits,
    ) -> Optional[str]:
        """Return "cooldown" if the symbol's last entry is within the window, else None.

        `cooldowns[symbol]` is the timestamp (tz-aware) of the most recent submitted BUY
        for that symbol. A missing/None entry means no prior entry -> no cooldown. The
        comparison uses an aware UTC `now`; if the stored timestamp is naive it is treated
        as UTC so the subtraction never raises.
        """
        last = (cooldowns or {}).get(symbol)
        if last is None:
            return None
        now = _dt.datetime.now(_dt.timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=_dt.timezone.utc)
        elapsed_min = (now - last).total_seconds() / 60.0
        if elapsed_min < float(limits.per_symbol_cooldown_min):
            return "cooldown"
        return None


__all__ = ["RiskAgent"]
