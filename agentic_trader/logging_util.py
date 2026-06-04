"""logging_util.py — append-only JSONL event logger (INV-5).

Writes one JSON object per line to /Users/pnle/Desktop/alpaca-cli/logs/<date>.jsonl with a
UTC ISO-8601 `ts` and a `type`. Errors also tee to logs/errors-<date>.jsonl.

IMPORTANT: timestamps and the log file path are computed ONLY at call time, never at
import time, so the module is import-safe and a long-running process rolls over at the
UTC day boundary. The logs directory is created lazily on first write.

Every required event type (ARCHITECTURE §5) has a convenience method:
  data_snapshot, strategy_proposal, ml_score, risk_decision,
  order_request, order_response, position_summary, error.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import config


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing 'Z'. Computed at CALL time only."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_date_str() -> str:
    """Current UTC date as YYYY-MM-DD. Computed at CALL time only."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _jsonable(obj: Any) -> Any:
    """Recursively coerce dataclasses/enums/datetimes/Paths into JSON-safe values.

    Falls back to str() for anything exotic so logging NEVER raises and never drops an
    event. This is the serialization contract every agent's payloads pass through.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, (_dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)


class JsonlLogger:
    """Append-only JSONL logger. One file per UTC day; errors tee to an errors file.

    Construction does NO time work and only resolves (does not create) the directory.
    Files/dirs are created lazily on the first write. Pass `echo=True` to also print each
    event to stdout (useful for dry-run runs and the CLI).
    """

    REQUIRED_TYPES = (
        "data_snapshot", "strategy_proposal", "ml_score", "risk_decision",
        "order_request", "order_response", "position_summary", "error",
    )

    def __init__(
        self,
        logs_dir: os.PathLike | str | None = None,
        *,
        prefix: str = "trader",
        echo: bool = False,
    ) -> None:
        self.logs_dir = Path(logs_dir) if logs_dir is not None else config.LOGS_DIR
        self.prefix = prefix
        self.echo = echo

    # --- path helpers (call-time only) ------------------------------------- #
    def _main_path(self) -> Path:
        return self.logs_dir / f"{self.prefix}-{_utc_date_str()}.jsonl"

    def _error_path(self) -> Path:
        return self.logs_dir / f"errors-{_utc_date_str()}.jsonl"

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_jsonable(record), separators=(",", ":"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(line)

    # --- generic entrypoint ------------------------------------------------ #
    def log_event(self, kind: str, payload: Optional[Dict[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
        """Append one event: {"ts": <utc iso>, "type": kind, ...payload, ...fields}.

        Returns the written record (also handy for tests). `payload` and `**fields` are
        merged (fields win on key collision). Never raises on serialization.
        """
        record: Dict[str, Any] = {"ts": _utc_now_iso(), "type": kind}
        if payload:
            record.update(payload)
        if fields:
            record.update(fields)
        self._append(self._main_path(), record)
        return record

    # --- convenience methods (the required event vocabulary) -------------- #
    def log_snapshot(self, snapshot: Any, **extra: Any) -> Dict[str, Any]:
        """type=data_snapshot — a per-symbol indicator/data snapshot."""
        return self.log_event("data_snapshot", {"snapshot": snapshot}, **extra)

    def log_proposal(self, proposal: Any, **extra: Any) -> Dict[str, Any]:
        """type=strategy_proposal — a strategy Proposal."""
        return self.log_event("strategy_proposal", {"proposal": proposal}, **extra)

    def log_ml_score(self, symbol: str, p_win: float, ml_threshold: float, gate: str, **extra: Any) -> Dict[str, Any]:
        """type=ml_score — the entry scorer's p_win + gate pass/fail."""
        return self.log_event(
            "ml_score",
            {"symbol": symbol, "p_win": p_win, "ml_threshold": ml_threshold, "gate": gate},
            **extra,
        )

    def log_risk(self, decision: Any, **extra: Any) -> Dict[str, Any]:
        """type=risk_decision — the Risk verdict (+ reason for non-approvals)."""
        return self.log_event("risk_decision", {"decision": decision}, **extra)

    def log_order_request(self, order: Any, *, mode: str, **extra: Any) -> Dict[str, Any]:
        """type=order_request — the intended order. mode is 'dry_run' or 'live_paper'."""
        return self.log_event("order_request", {"order": order, "mode": mode}, **extra)

    def log_order_response(self, result: Any, **extra: Any) -> Dict[str, Any]:
        """type=order_response — the OrderResult (broker response or dry-run record)."""
        return self.log_event("order_response", {"result": result}, **extra)

    def log_position_summary(self, summaries: Iterable[Any], exposure: Optional[float] = None, **extra: Any) -> Dict[str, Any]:
        """type=position_summary — open positions + total exposure."""
        return self.log_event(
            "position_summary",
            {"open": list(summaries), "exposure": exposure},
            **extra,
        )

    def log_error(self, where: str, msg: str, exc: Optional[BaseException] = None, **extra: Any) -> Dict[str, Any]:
        """type=error — write to BOTH the main log and the errors tee file.

        Captures a traceback string if `exc` is provided. Never raises.
        """
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else None
        record: Dict[str, Any] = {"ts": _utc_now_iso(), "type": "error", "where": where, "msg": msg}
        if tb:
            record["traceback"] = tb
        if extra:
            record.update(extra)
        self._append(self._main_path(), record)
        self._append(self._error_path(), record)
        if self.echo:
            pass  # already printed by _append
        return record


__all__ = ["JsonlLogger"]
