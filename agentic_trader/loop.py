"""loop.py — the live, market-hours-aware run loop.

`run_loop` drives `orchestrator.run_cycle` repeatedly:

  - Builds the clients/params/limits/scorer (sensible defaults) ONCE.
  - Each iteration RECONCILES via run_cycle (INV-6) and runs one full pass.
  - Is MARKET-HOURS AWARE: it asks the Alpaca clock; when the market is CLOSED it sleeps
    until `next_open` (capped per nap so SIGINT stays responsive) instead of trading.
  - Aligns cycles to ~bar close: it sleeps to the top of the next minute (+ a small guard)
    so each cycle decides on freshly completed 1-minute bars (no look-ahead).
  - GRACEFUL SIGINT: a Ctrl-C requests stop; the loop finishes the current cycle, logs a
    shutdown event, and returns cleanly.
  - `armed` is threaded straight through to run_cycle — the loop NEVER assumes armed.
  - `once=True` runs exactly one cycle (after reconciling) and returns.

PAPER ONLY. Deterministic trade logic. No real-money mode.
"""

from __future__ import annotations

import datetime as _dt
import signal
import time
from typing import Any, Dict, Optional

from .config import (
    RiskLimits,
    Settings,
    StrategyParams,
    load_strategy_params,
    make_clients,
)
from .logging_util import JsonlLogger
from .orchestrator import CycleReport, run_cycle

# Small guard (seconds) added after the minute boundary so the just-closed bar is fully
# available from the data feed before we decide on it.
_BAR_CLOSE_GUARD_SEC = 2.0
# Max seconds to sleep in a single nap while the market is closed; keeps SIGINT snappy.
_MAX_CLOSED_NAP_SEC = 30.0


class _StopFlag:
    """Cooperative stop flag toggled by SIGINT/SIGTERM (graceful shutdown)."""

    def __init__(self) -> None:
        self.stop = False

    def request(self, *_args: Any) -> None:
        self.stop = True


def run_loop(
    *,
    armed: bool,
    interval: Optional[float] = None,
    once: bool = False,
    params: Optional[StrategyParams] = None,
    limits: Optional[RiskLimits] = None,
    scorer: "Any | None" = None,
    settings: Optional[Settings] = None,
    clients: Optional[Dict[str, Any]] = None,
    logger: Optional[JsonlLogger] = None,
) -> None:
    """Run the live loop until the market closes for the day or SIGINT (or one cycle).

    Defaults: params = load_strategy_params(), limits = RiskLimits(), scorer =
    EntryScorer.load() (lazily imported), settings = Settings(), clients =
    make_clients(). `interval` defaults to Settings.loop_interval_sec. `once=True` runs a
    single reconcile+cycle and returns. `armed` is passed verbatim to run_cycle (still
    gated by INV-1/2/3 in Execution).
    """
    settings = settings or Settings()
    params = params if params is not None else load_strategy_params()
    limits = limits if limits is not None else RiskLimits()
    interval = float(interval) if interval is not None else float(settings.loop_interval_sec)
    logger = logger or JsonlLogger(echo=True)

    if scorer is None:
        # Lazy import keeps loop.py importable even if the ml stack is partially built.
        from .ml.scorer import EntryScorer

        scorer = EntryScorer.load()

    if clients is None:
        clients = make_clients(settings.env_path)

    # Cooldown store persists ACROSS cycles within this process (A5).
    cooldowns: Dict[str, _dt.datetime] = {}

    stop = _StopFlag()
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, stop.request)
        signal.signal(signal.SIGTERM, stop.request)
    except (ValueError, OSError):
        # signal handlers can only be installed in the main thread; tolerate otherwise.
        pass

    logger.log_event(
        "loop_start",
        {"armed": bool(armed), "interval": interval, "once": bool(once),
         "scorer_passthrough": getattr(scorer, "is_passthrough", None)},
    )

    try:
        if once:
            _run_one(clients, params, limits, scorer, armed, settings, logger, cooldowns)
            return

        while not stop.stop:
            # Market-hours gate: if closed, sleep toward next_open (interruptibly).
            clock = _get_clock(clients, logger)
            if clock is not None and not bool(getattr(clock, "is_open", False)):
                if _sleep_until_open(clock, stop, logger):
                    continue  # woke up; re-check the clock
                break  # stop requested while waiting

            _run_one(clients, params, limits, scorer, armed, settings, logger, cooldowns)

            if stop.stop:
                break

            # If the clock told us the session has ended for the day, stop after this cycle.
            if clock is not None and not bool(getattr(clock, "is_open", False)):
                break

            _sleep_to_next_cycle(interval, stop)
    finally:
        try:
            logger.log_event("loop_stop", {"reason": "sigint" if stop.stop else "complete", "armed": bool(armed)})
        except Exception:
            pass
        # Restore prior signal handlers (best-effort).
        try:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
        except (ValueError, OSError, TypeError):
            pass


def _run_one(
    clients: Dict[str, Any],
    params: StrategyParams,
    limits: RiskLimits,
    scorer: Any,
    armed: bool,
    settings: Settings,
    logger: JsonlLogger,
    cooldowns: Dict[str, _dt.datetime],
) -> CycleReport:
    """Run a single cycle, never letting an exception escape the loop."""
    try:
        return run_cycle(
            clients, params, limits, scorer,
            armed=armed, settings=settings, logger=logger, cooldowns=cooldowns,
        )
    except Exception as exc:  # noqa: BLE001 — the loop must survive a cycle failure
        logger.log_error("loop._run_one", "run_cycle raised", exc)
        return CycleReport(asof=_dt.datetime.now(_dt.timezone.utc), armed=bool(armed),
                           errors=[f"loop._run_one: {exc}"])


def _get_clock(clients: Dict[str, Any], logger: JsonlLogger) -> Optional[Any]:
    """Fetch the Alpaca market clock; return None (treated as 'unknown -> trade') on error."""
    trading = clients.get("trading")
    if trading is None:
        return None
    try:
        return trading.get_clock()
    except Exception as exc:  # noqa: BLE001
        logger.log_error("loop._get_clock", "get_clock failed", exc)
        return None


def _sleep_until_open(clock: Any, stop: _StopFlag, logger: JsonlLogger) -> bool:
    """Sleep (interruptibly) toward clock.next_open. Returns True if it should re-check.

    Naps in <= _MAX_CLOSED_NAP_SEC chunks so a SIGINT during a long overnight wait is
    honoured promptly. Returns False if a stop was requested while waiting.
    """
    next_open = getattr(clock, "next_open", None)
    now = getattr(clock, "timestamp", None) or _dt.datetime.now(_dt.timezone.utc)
    wait_sec = _MAX_CLOSED_NAP_SEC
    if isinstance(next_open, _dt.datetime) and isinstance(now, _dt.datetime):
        try:
            wait_sec = max(1.0, (next_open - now).total_seconds())
        except Exception:
            wait_sec = _MAX_CLOSED_NAP_SEC

    logger.log_event("market_closed",
                     {"next_open": next_open, "sleep_sec": min(wait_sec, _MAX_CLOSED_NAP_SEC)})

    napped = min(wait_sec, _MAX_CLOSED_NAP_SEC)
    return _interruptible_sleep(napped, stop)


def _sleep_to_next_cycle(interval: float, stop: _StopFlag) -> None:
    """Sleep until the next cycle boundary (interruptibly), aligned to bar close.

    When the interval is ~one minute (the default), align to the top of the next minute
    plus a small guard so the just-completed bar is available. Otherwise sleep `interval`.
    """
    if 50.0 <= interval <= 70.0:
        now = _dt.datetime.now(_dt.timezone.utc)
        nxt = (now.replace(second=0, microsecond=0) + _dt.timedelta(minutes=1))
        target = nxt + _dt.timedelta(seconds=_BAR_CLOSE_GUARD_SEC)
        sleep_sec = max(0.0, (target - now).total_seconds())
    else:
        sleep_sec = max(0.0, interval)
    _interruptible_sleep(sleep_sec, stop)


def _interruptible_sleep(seconds: float, stop: _StopFlag) -> bool:
    """Sleep up to `seconds`, waking early on a stop request. Returns False if stopped."""
    deadline = time.monotonic() + max(0.0, seconds)
    while not stop.stop:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 1.0))
    return False


__all__ = ["run_loop"]
