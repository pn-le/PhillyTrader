"""market_data.py — Agent ① Market Data (STRATEGY_RULES §1/§2).

Fetches today's COMPLETED regular-hours 1-minute bars per symbol from Alpaca's
StockHistoricalDataClient (IEX free feed), computes the per-symbol Indicators, and
builds a Snapshot handed to the Strategy agent.

`compute_indicators(bars, params)` is a PURE function (no I/O, no Alpaca) so the
backtester can reuse it verbatim. It implements STRATEGY_RULES §2 EXACTLY:
  - session_vwap = sum(typical_price*volume)/sum(volume) over ALL completed session
    bars INCLUDING the decision bar (A2).
  - rolling20_avg_vol = mean of the 20 bars STRICTLY preceding the decision bar
    (indices [n-21 : n-1]); EXCLUDES the current bar (A1).
  - valid=True iff n >= MIN_BARS_FOR_ENTRY (21) and the VWAP / rolling denominators
    are positive; otherwise valid=False with an invalid_reason (never crashes).

NO LOOK-AHEAD: indicators use only the bars passed in (completed, ascending). Bars
whose minute is not yet complete (start_utc + 60s > now_utc) are dropped before the
Snapshot is built.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from ..config import (
    MIN_BARS_FOR_ENTRY,
    ROLLING_VOL_WINDOW,
    SESSION_CLOSE,
    SESSION_OPEN,
    Settings,
    StrategyParams,
)
from ..logging_util import JsonlLogger
from ..types import Bar, Indicators, PositionState, Snapshot


def _parse_hms(hms: str) -> _dt.time:
    """Parse 'HH:MM:SS' into a datetime.time."""
    h, m, s = (int(x) for x in hms.split(":"))
    return _dt.time(h, m, s)


_OPEN_T = _parse_hms(SESSION_OPEN)   # 09:30:00
_CLOSE_T = _parse_hms(SESSION_CLOSE)  # 16:00:00 (exclusive for bar START)


# --------------------------------------------------------------------------- #
# PURE indicator computation (reused by the backtester)
# --------------------------------------------------------------------------- #
def compute_indicators(bars: List[Bar], params: StrategyParams) -> Indicators:
    """Compute per-symbol Indicators from COMPLETED session bars (STRATEGY_RULES §2).

    PURE: no I/O. `bars` must be completed regular-hours bars for ONE symbol, ascending
    by start. The decision bar is bars[-1]. Returns valid=True only when the symbol is
    eligible for ENTRY evaluation (n >= 21 and positive denominators); otherwise returns
    an Indicators with the fields it CAN compute and valid=False + invalid_reason. Never
    raises on empty/short/degenerate input.
    """
    n = len(bars)
    symbol = bars[-1].symbol if n else ""

    if n == 0:
        return Indicators(
            symbol=symbol,
            session_vwap=None,
            last_price=None,
            dist_from_vwap=None,
            current_volume=None,
            rolling20_avg_vol=None,
            volume_ratio=None,
            n_bars=0,
            valid=False,
            invalid_reason="no_bars",
        )

    decision = bars[-1]
    last_price = float(decision.close)
    current_volume = float(decision.volume)

    # Session VWAP over ALL completed bars incl. decision bar (A2).
    num = 0.0
    den = 0.0
    for b in bars:
        typical = (float(b.high) + float(b.low) + float(b.close)) / 3.0
        vol = float(b.volume)
        num += typical * vol
        den += vol

    session_vwap: Optional[float] = (num / den) if den > 0 else None
    dist_from_vwap: Optional[float] = (
        (last_price - session_vwap) / session_vwap
        if session_vwap is not None and session_vwap != 0
        else None
    )

    # Rolling 20-bar avg volume — EXCLUDES the current bar: bars[n-21 : n-1] (A1).
    window = bars[max(0, n - (ROLLING_VOL_WINDOW + 1)): n - 1]
    rolling20_avg_vol: Optional[float] = None
    if window:
        rolling20_avg_vol = sum(float(b.volume) for b in window) / len(window)

    volume_ratio: Optional[float] = (
        current_volume / rolling20_avg_vol
        if rolling20_avg_vol is not None and rolling20_avg_vol > 0
        else None
    )

    # Eligibility for ENTRY evaluation (A1/A4).
    valid = True
    invalid_reason: Optional[str] = None
    if n < MIN_BARS_FOR_ENTRY:
        valid = False
        invalid_reason = f"insufficient_bars(n={n}<{MIN_BARS_FOR_ENTRY})"
    elif den <= 0:
        valid = False
        invalid_reason = "degenerate_vwap_volume"
    elif rolling20_avg_vol is None or rolling20_avg_vol <= 0:
        valid = False
        invalid_reason = "degenerate_rolling_volume"

    return Indicators(
        symbol=symbol,
        session_vwap=session_vwap,
        last_price=last_price,
        dist_from_vwap=dist_from_vwap,
        current_volume=current_volume,
        rolling20_avg_vol=rolling20_avg_vol,
        volume_ratio=volume_ratio,
        n_bars=n,
        valid=valid,
        invalid_reason=invalid_reason,
    )


# --------------------------------------------------------------------------- #
# Market Data Agent
# --------------------------------------------------------------------------- #
class MarketDataAgent:
    """Fetch + window today's completed regular-hours 1-min bars, build Snapshots.

    Uses the shared StockHistoricalDataClient (IEX). The agent owns NO mutable trading
    state; it only reads bars and computes indicators. `position_state` is supplied by
    the caller (reconciled from the broker) or fetched is left to the orchestrator.
    """

    def __init__(
        self,
        data_client,
        settings: Settings,
        params: StrategyParams,
        logger: Optional[JsonlLogger] = None,
    ) -> None:
        self.data_client = data_client
        self.settings = settings
        self.params = params
        self.logger = logger
        self._tz = ZoneInfo(settings.timezone)

    # --- public API -------------------------------------------------------- #
    def get_snapshot(
        self,
        symbol: str,
        position_state: Optional[PositionState] = None,
        now: Optional[_dt.datetime] = None,
    ) -> Snapshot:
        """Build a Snapshot for one symbol for the session containing `now`.

        Fetches 1-min IEX bars for today's session (America/New_York), drops any bar that
        is not yet complete (start_utc + 60s > now_utc) and any bar outside regular hours,
        computes indicators on the resulting window, and returns a Snapshot whose `asof`
        is the decision-bar start. `position_state` defaults to FLAT. Never raises on a
        data error — logs it and returns an empty/invalid Snapshot instead.
        """
        now_utc = self._now_utc(now)
        pstate = position_state or PositionState(symbol=symbol)

        try:
            bars = self._fetch_session_bars(symbol, now_utc)
        except Exception as exc:  # noqa: BLE001 — never crash the cycle on a data error
            if self.logger is not None:
                self.logger.log_error("market_data.get_snapshot", f"fetch failed for {symbol}", exc, symbol=symbol)
            bars = []

        indicators = compute_indicators(bars, self.params)
        asof = bars[-1].start if bars else self._session_open_ny(now_utc)
        return Snapshot(
            symbol=symbol,
            asof=asof,
            indicators=indicators,
            position_state=pstate,
            bars=tuple(bars),
        )

    def get_snapshots(
        self,
        universe: Optional[List[str]] = None,
        position_states: Optional[Dict[str, PositionState]] = None,
        now: Optional[_dt.datetime] = None,
    ) -> List[Snapshot]:
        """Build a Snapshot for each symbol in `universe` (defaults Settings.universe).

        Order is preserved. `position_states` maps symbol -> reconciled PositionState
        (missing symbols default FLAT). Each symbol is fetched independently so one bad
        symbol does not block the rest.
        """
        syms = list(universe) if universe is not None else list(self.settings.universe)
        pstates = position_states or {}
        return [self.get_snapshot(s, pstates.get(s), now) for s in syms]

    # --- internals --------------------------------------------------------- #
    def _now_utc(self, now: Optional[_dt.datetime]) -> _dt.datetime:
        """Return a tz-aware UTC 'now'. Defaults to the real clock."""
        if now is None:
            return _dt.datetime.now(_dt.timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=_dt.timezone.utc)
        return now.astimezone(_dt.timezone.utc)

    def _session_open_ny(self, now_utc: _dt.datetime) -> _dt.datetime:
        """09:30 NY on the session date implied by `now_utc` (used as fallback asof)."""
        now_ny = now_utc.astimezone(self._tz)
        return _dt.datetime.combine(now_ny.date(), _OPEN_T, tzinfo=self._tz)

    def _fetch_session_bars(self, symbol: str, now_utc: _dt.datetime) -> List[Bar]:
        """Fetch + filter today's completed regular-hours 1-min bars for `symbol`.

        Returns Bar objects with `start` in America/New_York, ascending, completed-only,
        regular-hours-only, for the single session date implied by `now_utc` (NY date).
        """
        # Lazy alpaca imports keep the contract layer import-light.
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        now_ny = now_utc.astimezone(self._tz)
        session_date = now_ny.date()

        # Request the full session window in UTC; over-fetch a touch on each side and
        # then filter precisely, so DST / feed-edge effects can't truncate the session.
        start_ny = _dt.datetime.combine(session_date, _OPEN_T, tzinfo=self._tz)
        end_ny = _dt.datetime.combine(session_date, _CLOSE_T, tzinfo=self._tz)
        start_utc = start_ny.astimezone(_dt.timezone.utc)
        end_utc = end_ny.astimezone(_dt.timezone.utc)

        feed = self._resolve_feed(DataFeed)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_utc,
            end=end_utc,
            feed=feed,
        )
        barset = self.data_client.get_stock_bars(req)
        raw = barset.data.get(symbol, []) if hasattr(barset, "data") else []

        bars: List[Bar] = []
        for b in raw:
            ts_utc = self._to_utc(b.timestamp)
            # Completed-only: a 1-min bar is complete once now >= start + 60s (A16).
            if ts_utc + _dt.timedelta(seconds=60) > now_utc:
                continue
            start_local = ts_utc.astimezone(self._tz)
            # Same session date + regular hours [09:30, 16:00) on bar START (A3/§1).
            if start_local.date() != session_date:
                continue
            t = start_local.time()
            if t < _OPEN_T or t >= _CLOSE_T:
                continue
            bars.append(
                Bar(
                    symbol=symbol,
                    start=start_local,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                    trade_count=(float(b.trade_count) if getattr(b, "trade_count", None) is not None else None),
                    vwap=(float(b.vwap) if getattr(b, "vwap", None) is not None else None),
                )
            )

        bars.sort(key=lambda x: x.start)
        return bars

    def _resolve_feed(self, data_feed_enum):
        """Map the Settings.feed string to a DataFeed enum (defaults IEX)."""
        name = (self.settings.feed or "iex").lower()
        for member in data_feed_enum:
            if member.value == name:
                return member
        return data_feed_enum.IEX

    @staticmethod
    def _to_utc(ts: _dt.datetime) -> _dt.datetime:
        """Coerce a bar timestamp to tz-aware UTC."""
        if ts.tzinfo is None:
            return ts.replace(tzinfo=_dt.timezone.utc)
        return ts.astimezone(_dt.timezone.utc)


__all__ = ["MarketDataAgent", "compute_indicators"]
