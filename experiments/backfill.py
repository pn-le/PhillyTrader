"""backfill.py — IEX 1-minute history backfill for the research universe.

Pulls as much free-tier IEX 1-minute history as Alpaca will give for the universe,
filters to regular trading hours (America/New_York, [09:30, 16:00) on bar START),
sorts, de-duplicates, and caches one parquet file per symbol under
/Users/pnle/Desktop/alpaca-cli/experiments/cache.

Pure research / read-only market-data fetch. Places ZERO orders.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.backfill
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

CACHE_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/cache")
RESULTS_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results")
ENV_PATH = Path("/Users/pnle/Desktop/alpaca-cli/.env")

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]

NY = ZoneInfo("America/New_York")
OPEN_T = _dt.time(9, 30, 0)
CLOSE_T = _dt.time(16, 0, 0)  # exclusive on bar START

START = _dt.datetime(2025, 9, 1, tzinfo=_dt.timezone.utc)
END = _dt.datetime(2026, 6, 4, tzinfo=_dt.timezone.utc)

COLUMNS = ["symbol", "start", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def load_env(path: Path = ENV_PATH) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def make_data_client():
    from alpaca.data.historical import StockHistoricalDataClient

    env = load_env()
    return StockHistoricalDataClient(env["ALPACA_API_KEY_ID"], env["ALPACA_API_SECRET"])


def _month_chunks(start_utc, end_utc, days=28):
    chunks = []
    cur = start_utc
    step = _dt.timedelta(days=days)
    while cur < end_utc:
        nxt = min(cur + step, end_utc)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def fetch_symbol(client, symbol, start_utc, end_utc):
    """Fetch + RTH-filter all 1-min IEX bars for one symbol. Returns list of row dicts."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    rows = []
    for c_start, c_end in _month_chunks(start_utc, end_utc):
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=c_start,
            end=c_end,
            feed=DataFeed.IEX,
        )
        barset = client.get_stock_bars(req)
        raw = barset.data.get(symbol, []) if hasattr(barset, "data") else []
        for b in raw:
            ts = b.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            local = ts.astimezone(NY)
            t = local.time()
            if t < OPEN_T or t >= CLOSE_T:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "start": local,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "trade_count": float(b.trade_count) if getattr(b, "trade_count", None) is not None else None,
                    "vwap": float(b.vwap) if getattr(b, "vwap", None) is not None else None,
                }
            )
    return rows


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = make_data_client()

    coverage = {}
    for sym in UNIVERSE:
        rows = fetch_symbol(client, sym, START, END)
        if not rows:
            coverage[sym] = {"n_bars": 0, "first": None, "last": None, "n_days": 0}
            print(f"{sym}: 0 bars")
            continue
        df = pd.DataFrame(rows, columns=COLUMNS)
        # tz-aware NY timestamps, sorted, deduped on start
        df["start"] = pd.to_datetime(df["start"], utc=True).dt.tz_convert(NY)
        df = df.drop_duplicates(subset=["start"]).sort_values("start").reset_index(drop=True)
        path = CACHE_DIR / f"{sym}_1min.parquet"
        df.to_parquet(path, index=False)
        first = df["start"].iloc[0]
        last = df["start"].iloc[-1]
        n_days = df["start"].dt.date.nunique()
        coverage[sym] = {
            "n_bars": int(len(df)),
            "first": str(first),
            "last": str(last),
            "n_days": int(n_days),
        }
        print(f"{sym}: {len(df)} bars, {n_days} days, {first.date()} -> {last.date()}")

    (RESULTS_DIR / "backfill_coverage.json").write_text(json.dumps(coverage, indent=2))
    print("\nCoverage written to", RESULTS_DIR / "backfill_coverage.json")
    return coverage


if __name__ == "__main__":
    main()
