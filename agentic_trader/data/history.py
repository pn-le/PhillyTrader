"""history.py — historical 1-minute bar backfill + on-disk cache.

Pulls historical 1-minute IEX bars from Alpaca per symbol, filters to regular hours
(America/New_York), and caches one file per symbol under
/Users/pnle/Desktop/alpaca-cli/data/cache/ in the ENV_REPORT format (Parquet via
pyarrow; CSV fallback if pyarrow is absent). `load_cached_bars` reads a clean, sorted,
de-duplicated per-symbol DataFrame back.

These functions feed the backtester/optimizer (offline). They are I/O wrappers around
the same regular-hours windowing that MarketDataAgent applies live, so cached bars and
live bars are filtered identically.

Pagination: alpaca-py's StockHistoricalDataClient handles page tokens internally for a
single get_stock_bars call, but we still chunk very large requests by month to keep
memory bounded and to make partial-failure recovery cheap.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from ..config import (
    DATASET_FORMAT,
    SESSION_CLOSE,
    SESSION_OPEN,
    Settings,
)
from ..logging_util import JsonlLogger

# Canonical cached-bar column layout (INTERFACE_SPEC §2 load_cached_bars contract).
_COLUMNS = ["symbol", "start", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def _parse_hms(hms: str) -> _dt.time:
    h, m, s = (int(x) for x in hms.split(":"))
    return _dt.time(h, m, s)


_OPEN_T = _parse_hms(SESSION_OPEN)   # 09:30:00
_CLOSE_T = _parse_hms(SESSION_CLOSE)  # 16:00:00 (exclusive for bar START)


# --------------------------------------------------------------------------- #
# Backend detection (Parquet preferred per ENV_REPORT; CSV fallback)
# --------------------------------------------------------------------------- #
def _use_parquet() -> bool:
    """True iff pyarrow is importable AND config asks for parquet."""
    if str(DATASET_FORMAT).lower() != "parquet":
        return False
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _cache_path(symbol: str, settings: Settings) -> Path:
    """Absolute cache file path for one symbol (extension matches the active backend)."""
    ext = "parquet" if _use_parquet() else "csv"
    cache_dir = Path(settings.data_cache_dir)
    return cache_dir / f"{symbol}_1min.{ext}"


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def _to_utc(ts: _dt.datetime) -> _dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=_dt.timezone.utc)
    return ts.astimezone(_dt.timezone.utc)


def _month_chunks(start_utc: _dt.datetime, end_utc: _dt.datetime) -> List[tuple]:
    """Split [start, end] into <=~31-day UTC chunks for bounded paginated fetches."""
    chunks: List[tuple] = []
    cur = start_utc
    step = _dt.timedelta(days=31)
    while cur < end_utc:
        nxt = min(cur + step, end_utc)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


# --------------------------------------------------------------------------- #
# Backfill
# --------------------------------------------------------------------------- #
def backfill(
    symbols: List[str],
    start: _dt.datetime,
    end: _dt.datetime,
    *,
    data_client=None,
    settings: Optional[Settings] = None,
    logger: Optional[JsonlLogger] = None,
) -> Dict[str, Path]:
    """Pull 1-min IEX bars for each symbol over [start, end] and cache one file per symbol.

    Regular-hours filtered (America/New_York), sorted ascending, de-duplicated on `start`.
    Returns {symbol: cache_path}. Symbols that fail (or return no bars) are logged and
    omitted from the returned mapping; the rest still succeed. Writes Parquet if pyarrow
    is available, else CSV (config.DATASET_FORMAT).
    """
    import pandas as pd

    settings = settings or Settings()
    tz = ZoneInfo(settings.timezone)
    Path(settings.data_cache_dir).mkdir(parents=True, exist_ok=True)

    if data_client is None:
        from ..config import make_data_client
        data_client = make_data_client(settings.env_path)

    start_utc = _to_utc(start)
    end_utc = _to_utc(end)

    out: Dict[str, Path] = {}
    for symbol in symbols:
        try:
            rows = _fetch_symbol(data_client, symbol, start_utc, end_utc, settings, tz)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.log_error("history.backfill", f"fetch failed for {symbol}", exc, symbol=symbol)
            continue

        if not rows:
            if logger is not None:
                logger.log_event("backfill", {"symbol": symbol, "n_bars": 0, "note": "no_bars"})
            continue

        df = pd.DataFrame(rows, columns=_COLUMNS)
        df = df.drop_duplicates(subset=["start"]).sort_values("start").reset_index(drop=True)

        path = _cache_path(symbol, settings)
        _write_cache(df, path)
        out[symbol] = path
        if logger is not None:
            logger.log_event("backfill", {"symbol": symbol, "n_bars": int(len(df)), "path": str(path)})

    return out


def _fetch_symbol(
    data_client,
    symbol: str,
    start_utc: _dt.datetime,
    end_utc: _dt.datetime,
    settings: Settings,
    tz: ZoneInfo,
) -> List[dict]:
    """Fetch + regular-hours-filter all 1-min bars for one symbol over [start, end].

    Returns a list of row dicts (cache schema). Chunked by month to bound memory and
    keep pagination cheap. Bars are kept iff their START (in NY) falls in [09:30, 16:00).
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    feed = _resolve_feed(DataFeed, settings.feed)
    rows: List[dict] = []

    for c_start, c_end in _month_chunks(start_utc, end_utc):
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=c_start,
            end=c_end,
            feed=feed,
        )
        barset = data_client.get_stock_bars(req)
        raw = barset.data.get(symbol, []) if hasattr(barset, "data") else []
        for b in raw:
            ts_utc = _to_utc(b.timestamp)
            start_local = ts_utc.astimezone(tz)
            t = start_local.time()
            if t < _OPEN_T or t >= _CLOSE_T:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "start": start_local,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "trade_count": (float(b.trade_count) if getattr(b, "trade_count", None) is not None else None),
                    "vwap": (float(b.vwap) if getattr(b, "vwap", None) is not None else None),
                }
            )
    return rows


def _resolve_feed(data_feed_enum, feed_name: Optional[str]):
    name = (feed_name or "iex").lower()
    for member in data_feed_enum:
        if member.value == name:
            return member
    return data_feed_enum.IEX


# --------------------------------------------------------------------------- #
# Cache write / read
# --------------------------------------------------------------------------- #
def _write_cache(df, path: Path) -> None:
    """Persist a per-symbol bar DataFrame to `path` (Parquet or CSV per backend).

    `start` is stored tz-aware (America/New_York). Parquet preserves tz natively; CSV
    stores ISO-8601 strings (with offset) that load_cached_bars re-parses to tz-aware.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if _use_parquet():
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_cached_bars(
    symbol: str,
    start: _dt.datetime,
    end: _dt.datetime,
    settings: Optional[Settings] = None,
) -> "pandas.DataFrame":  # noqa: F821 — pandas typed lazily
    """Load cached bars for `symbol`, filtered to [start, end] (inclusive of start).

    Returns a clean DataFrame with columns
    ['symbol','start','open','high','low','close','volume','trade_count','vwap'],
    tz-aware 'start' in America/New_York, regular-hours-only, sorted ascending, de-duped.
    Returns an EMPTY DataFrame (with the right columns) if no cache file exists.
    """
    import pandas as pd

    settings = settings or Settings()
    tz = ZoneInfo(settings.timezone)
    path = _cache_path(symbol, settings)

    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)

    # Normalise the timestamp column to tz-aware America/New_York.
    s = pd.to_datetime(df["start"], utc=True)
    df["start"] = s.dt.tz_convert(tz)

    # Regular-hours guard (defensive; cache is already filtered).
    local_t = df["start"].dt.time
    df = df[(local_t >= _OPEN_T) & (local_t < _CLOSE_T)]

    # Range filter [start, end].
    lo = pd.Timestamp(_to_utc(start)).tz_convert(tz)
    hi = pd.Timestamp(_to_utc(end)).tz_convert(tz)
    df = df[(df["start"] >= lo) & (df["start"] <= hi)]

    # Ensure every expected column exists, then order/clean.
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_COLUMNS].drop_duplicates(subset=["start"]).sort_values("start").reset_index(drop=True)
    return df


__all__ = ["backfill", "load_cached_bars"]
