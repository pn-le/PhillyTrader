"""execution_agent.py — Agent ⑤ Execution (the HARD SAFETY GATE).

INV-1/2/3 live here. This is the ONLY component that ever submits an order to the
broker, and it does so only when EVERY safety precondition holds:

    is_paper_env()          # ALPACA_PAPER == "true"            (INV-3)
  AND trading_client paper  # TradingClient(paper=True)         (INV-1)
  AND risk_decision.approved# Verdict.APPROVE or REDUCE         (INV-4)
  AND armed                 # explicit --arm / live_paper=True  (INV-2)

In every other case it is a DRY-RUN: it logs the intended order and submits NOTHING,
returning OrderResult(dry_run=True). If the env is not a paper account it REFUSES with a
clear logged error. There is NO real-money mode (INV-3).

This module contains NO strategy logic. It only translates an already-risk-approved
RiskDecision into a simple MARKET order (notional for entries, qty for full-position
exits), enforces the gate, dedupes against existing open orders, and logs both the
order_request and order_response events (INV-5). NO bracket / OCO / stop / limit orders.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

from ..config import Settings, is_paper_env
from ..logging_util import JsonlLogger
from ..types import OrderResult, RiskDecision, Side


def _utc_now() -> _dt.datetime:
    """Current UTC time (tz-aware). Computed at call time only."""
    return _dt.datetime.now(_dt.timezone.utc)


_PAPER_HOST = "paper-api.alpaca.markets"


def _is_paper_client(trading_client: Any) -> bool:
    """Verify the TradingClient targets the PAPER endpoint (INV-1, fail-closed).

    alpaca-py 0.43.4 does NOT expose a public `paper` flag; the authoritative signal is
    the routed base URL (`_base_url == BaseURL.TRADING_PAPER ==
    'https://paper-api.alpaca.markets'`). We treat a base URL pointing at the paper host
    as paper. As a forward-compatible fallback we also honour an explicit `_paper`/`paper`
    attribute if a future SDK adds one. Defaults to FALSE so an unexpected client shape
    can never accidentally satisfy the safety gate (no real-money submissions, ever).
    """
    base = getattr(trading_client, "_base_url", None)
    base_str = str(getattr(base, "value", base) or "").lower()
    if base_str:
        return _PAPER_HOST in base_str
    for attr in ("_paper", "paper"):
        if hasattr(trading_client, attr):
            try:
                return bool(getattr(trading_client, attr))
            except Exception:
                return False
    return False


class ExecutionAgent:
    """Translates a risk-approved decision into a paper MARKET order behind the gate.

    Construction is side-effect free (no time work, no broker calls). `settings` carries
    `env_path` so the env check uses the same .env the clients were built from.
    """

    def __init__(
        self,
        trading_client: Any,
        settings: Settings,
        *,
        logger: Optional[JsonlLogger] = None,
    ) -> None:
        self.trading_client = trading_client
        self.settings = settings
        self.logger = logger or JsonlLogger()

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def execute(self, risk_decision: RiskDecision, *, armed: bool) -> OrderResult:
        """Submit a simple MARKET order IFF the full safety gate holds; else DRY-RUN.

        Order shape (A11): BUY -> notional=effective_notional; SELL -> qty=proposal.qty;
        time_in_force=DAY. Before any live submission, re-check the broker for a pending
        OPEN order on the symbol and skip (dry-run) if one exists (A17). Catches SDK
        errors and returns OrderResult(status="error", error=...). Logs order_request
        (with mode) then order_response (INV-5).
        """
        proposal = risk_decision.proposal
        symbol = proposal.symbol
        side = proposal.side

        # Intended order economics: BUY uses effective notional; SELL uses full qty.
        notional = risk_decision.effective_notional if side == Side.BUY else None
        qty = proposal.qty if side == Side.SELL else None

        env_is_paper = is_paper_env(self.settings.env_path)
        client_is_paper = _is_paper_client(self.trading_client)
        approved = risk_decision.approved
        # The full safety gate (INV-1/2/3/4).
        should_submit = env_is_paper and client_is_paper and approved and bool(armed)
        mode = "live_paper" if should_submit else "dry_run"

        # Always log the intended order BEFORE acting (INV-5).
        self.logger.log_order_request(
            {
                "symbol": symbol,
                "side": side.value,
                "intent": proposal.intent.value,
                "notional": notional,
                "qty": qty,
                "time_in_force": "day",
                "verdict": risk_decision.verdict.value,
                "armed": bool(armed),
                "env_is_paper": env_is_paper,
                "client_is_paper": client_is_paper,
                "approved": approved,
            },
            mode=mode,
        )

        # INV-3: never submit against a non-paper account — refuse with a clear error.
        if not env_is_paper:
            msg = "ALPACA_PAPER != 'true' — refusing to submit any order (no real-money mode)."
            self.logger.log_error("execution.execute", msg, symbol=symbol, side=side.value)
            return self._result(
                symbol, side, status="refused", dry_run=True,
                notional=notional, qty=qty, error=msg,
            )

        # Any other failed precondition -> DRY-RUN (submit NOTHING).
        if not should_submit:
            reason = self._dry_run_reason(client_is_paper, approved, armed)
            result = self._result(
                symbol, side, status="dry_run", dry_run=True,
                notional=notional, qty=qty, error=None,
            )
            self.logger.log_order_response(result, dry_run_reason=reason)
            return result

        # --- ARMED + APPROVED + PAPER: this is a real paper submission -------- #

        # A17: skip if the broker already shows a pending OPEN order for this symbol.
        if self._has_open_order(symbol):
            result = self._result(
                symbol, side, status="skipped_open_order", dry_run=True,
                notional=notional, qty=qty, error=None,
            )
            self.logger.log_order_response(result, skip_reason="open_order_pending")
            return result

        try:
            order = self._submit(symbol, side, notional, qty)
        except Exception as exc:  # SDK / network / validation errors
            msg = f"submit_order failed: {exc}"
            self.logger.log_error("execution.execute", msg, exc=exc, symbol=symbol, side=side.value)
            result = self._result(
                symbol, side, status="error", dry_run=False,
                notional=notional, qty=qty, error=msg,
            )
            self.logger.log_order_response(result)
            return result

        result = self._from_order(order, symbol, side, notional, qty)
        self.logger.log_order_response(result)
        return result

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #
    def _submit(self, symbol: str, side: Side, notional: Optional[float], qty: Optional[float]):
        """Build a simple MarketOrderRequest (TIF=DAY) and submit it. Paper only.

        Imports alpaca lazily so this module stays importable without alpaca installed.
        Passes notional XOR qty (never both) per API_MAP §2. Does NOT set `type`
        (MarketOrderRequest auto-sets OrderType.MARKET). No bracket/OCO/stop/limit.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_side = OrderSide.BUY if side == Side.BUY else OrderSide.SELL
        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "side": order_side,
            "time_in_force": TimeInForce.DAY,
        }
        if side == Side.BUY:
            kwargs["notional"] = float(notional)
        else:
            kwargs["qty"] = float(qty)

        order_req = MarketOrderRequest(**kwargs)
        return self.trading_client.submit_order(order_data=order_req)

    def _has_open_order(self, symbol: str) -> bool:
        """Return True iff the broker shows a pending OPEN order for `symbol` (A17).

        Fails OPEN-SAFE: if the open-order check itself errors, we do NOT block the
        submission on that account (we log and proceed), since the gate has already
        confirmed paper+armed+approved. The reconcile loop catches duplicates next cycle.
        """
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            open_orders = self.trading_client.get_orders(filter=req)
        except Exception as exc:
            self.logger.log_error(
                "execution._has_open_order",
                f"open-order check failed for {symbol}: {exc}",
                exc=exc, symbol=symbol,
            )
            return False
        for o in open_orders or []:
            if getattr(o, "symbol", None) == symbol:
                return True
        return False

    @staticmethod
    def _dry_run_reason(client_is_paper: bool, approved: bool, armed: bool) -> str:
        """Human-readable reason a submission was downgraded to DRY-RUN."""
        if not approved:
            return "not_risk_approved"
        if not armed:
            return "not_armed"
        if not client_is_paper:
            return "client_not_paper"
        return "dry_run"

    @staticmethod
    def _result(
        symbol: str,
        side: Side,
        *,
        status: str,
        dry_run: bool,
        notional: Optional[float],
        qty: Optional[float],
        error: Optional[str],
        order_id: Optional[str] = None,
        filled_avg_price: Optional[float] = None,
    ) -> OrderResult:
        """Construct an OrderResult with a call-time UTC submitted_at."""
        return OrderResult(
            symbol=symbol,
            side=side,
            status=status,
            dry_run=dry_run,
            submitted_at=_utc_now(),
            id=order_id,
            notional=notional,
            qty=qty,
            filled_avg_price=filled_avg_price,
            error=error,
        )

    def _from_order(
        self,
        order: Any,
        symbol: str,
        side: Side,
        notional: Optional[float],
        qty: Optional[float],
    ) -> OrderResult:
        """Map a broker Order (alpaca Pydantic model) into an OrderResult.

        Numeric fields come back as strings (API_MAP §4) — parse defensively. The broker
        `submitted_at`/`status`/`id` win over our intended values when present.
        """
        status = self._coerce_status(getattr(order, "status", None))
        order_id = getattr(order, "id", None)
        submitted = getattr(order, "submitted_at", None)
        if not isinstance(submitted, _dt.datetime):
            submitted = _utc_now()

        return OrderResult(
            symbol=getattr(order, "symbol", None) or symbol,
            side=side,
            status=status,
            dry_run=False,
            submitted_at=submitted,
            id=str(order_id) if order_id is not None else None,
            notional=self._coerce_float(getattr(order, "notional", None), notional),
            qty=self._coerce_float(getattr(order, "qty", None), qty),
            filled_avg_price=self._coerce_float(getattr(order, "filled_avg_price", None), None),
            error=None,
        )

    @staticmethod
    def _coerce_status(status: Any) -> str:
        """Coerce an Alpaca OrderStatus enum / value into its string value."""
        if status is None:
            return "submitted"
        value = getattr(status, "value", status)
        return str(value)

    @staticmethod
    def _coerce_float(value: Any, fallback: Optional[float]) -> Optional[float]:
        """Parse a (possibly string) numeric broker field to float, else fallback."""
        if value is None or value == "":
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback


__all__ = ["ExecutionAgent"]
