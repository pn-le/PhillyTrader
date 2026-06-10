"""fetch_daily.py — fetch & cache ~10y DAILY split+dividend-adjusted bars.

Fetches TimeFrame.Day bars with adjustment='all' (Adjustment.ALL) for the ~100-name
large-cap universe + SPY over 2015-01-01 .. 2026-06-03 and caches one parquet per symbol
under experiments/factors/cache/. Daily bars are FULL QUALITY on the Alpaca free tier
(the IEX intraday-volume defect does not apply to daily bars). Records actual coverage
(earliest date, #names with full history, total rows) to results/coverage.json.

Pure research / read-only market-data fetch. Places ZERO orders.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.fetch_daily
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/factors/cache")
RESULTS_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/factors/results")
ENV_PATH = Path("/Users/pnle/Desktop/alpaca-cli/.env")

# ~100 liquid large-caps (today's membership — see SURVIVORSHIP caveat in factor_harness).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "LLY", "AVGO", "JPM",
    "V", "XOM", "UNH", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "MRK",
    "CVX", "KO", "PEP", "ADBE", "WMT", "BAC", "CRM", "ACN", "MCD", "NFLX",
    "TMO", "LIN", "ABT", "CSCO", "AMD", "PM", "DHR", "TXN", "QCOM", "VZ",
    "CMCSA", "WFC", "PFE", "IBM", "GE", "CAT", "NKE", "AMGN", "NOW", "UNP",
    "HON", "COP", "LOW", "SPGI", "INTU", "MS", "BA", "AXP", "BKNG", "GS",
    "T", "BLK", "DE", "SYK", "ELV", "MDT", "TJX", "ADP", "VRTX", "GILD",
    "LRCX", "C", "MMC", "PLD", "REGN", "MO", "SO", "ZTS", "BSX", "MU",
    "PANW", "SBUX", "ADI", "CI", "SCHW", "DUK", "CB", "BMY", "ETN", "ORCL",
    "INTC", "NEE", "WM", "ITW", "EOG", "APH", "MAR", "PYPL",
]
BENCHMARK = "SPY"

# We ATTEMPT 2015-01-01, but the Alpaca free tier only serves daily bars back to a rolling
# floor (~2020-07 as of 2026-06; probed empirically). The server clips silently; we record
# the ACTUAL earliest bar in coverage.json. Starting the request at 2015 is harmless.
START = _dt.datetime(2015, 1, 1, tzinfo=_dt.timezone.utc)
END = _dt.datetime(2026, 6, 4, tzinfo=_dt.timezone.utc)  # exclusive-ish; covers through 2026-06-03
ATTEMPTED_START_LABEL = "2015-01-01 (free-tier floor ~2020-07 applies)"

COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


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


def _year_chunks(start_utc, end_utc, days=365):
    chunks = []
    cur = start_utc
    step = _dt.timedelta(days=days)
    while cur < end_utc:
        nxt = min(cur + step, end_utc)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def fetch_symbol(client, symbol, start_utc, end_utc):
    """Fetch all DAILY split+dividend-adjusted bars for one symbol. Returns list of row dicts.

    Uses Adjustment.ALL so close/open/high/low/volume are split+dividend adjusted — the
    correct basis for multi-year return series. `date` is the calendar date of the session
    (Alpaca daily bar timestamps are midnight-of-session UTC; we keep the date only).
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    rows = []
    for c_start, c_end in _year_chunks(start_utc, end_utc):
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=c_start,
            end=c_end,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        barset = client.get_stock_bars(req)
        raw = barset.data.get(symbol, []) if hasattr(barset, "data") else []
        for b in raw:
            ts = b.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            d = ts.date()
            rows.append(
                {
                    "symbol": symbol,
                    "date": d,
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


def _build_coverage(per_symbol: dict, total_rows: int) -> dict:
    """Derive coverage from per-symbol {n_rows, first, last}.

    The Alpaca free tier serves daily bars from a common rolling floor (empirically
    ~2020-07-27 as of the fetch date). A name has FULL HISTORY if it starts on/before the
    MODAL (most common) first-bar date AND ends at the MODAL last-bar date — i.e. it spans
    the full common window with no stale tail (a name like MMC that stops mid-history is
    flagged as partial so it is dropped from the panel).
    """
    import collections

    firsts = [v["first"] for v in per_symbol.values() if v.get("first")]
    lasts = [v["last"] for v in per_symbol.values() if v.get("last")]
    earliest = min(firsts) if firsts else None
    latest = max(lasts) if lasts else None

    universe_firsts = [
        per_symbol[s]["first"] for s in UNIVERSE
        if per_symbol.get(s, {}).get("first")
    ]
    universe_lasts = [
        per_symbol[s]["last"] for s in UNIVERSE
        if per_symbol.get(s, {}).get("last")
    ]
    modal_first = collections.Counter(universe_firsts).most_common(1)[0][0] if universe_firsts else None
    modal_last = collections.Counter(universe_lasts).most_common(1)[0][0] if universe_lasts else None

    full_names = []
    if modal_first and modal_last:
        mf = pd.Timestamp(modal_first)
        ml = pd.Timestamp(modal_last)
        for s in UNIVERSE:
            v = per_symbol.get(s, {})
            if v.get("first") and v.get("last"):
                # starts at/before the common floor AND runs to the common latest date
                if pd.Timestamp(v["first"]) <= mf and pd.Timestamp(v["last"]) >= ml:
                    full_names.append(s)
    full_names = sorted(full_names)
    partial = sorted(s for s in UNIVERSE if s not in full_names)

    return {
        "attempted_start": ATTEMPTED_START_LABEL,
        "attempted_end": "2026-06-03",
        "free_tier_note": (
            "Alpaca free tier serves daily bars only from a rolling floor (~2020-07-27 here); "
            "2015-2020 was unavailable. Window covers the 2022 bear + 2023-26 bull + 2025 "
            "tariff selloff regimes, but NOT 2018Q4 or the Feb-Mar 2020 COVID crash."
        ),
        "earliest_bar_date_any_symbol": earliest,
        "latest_bar_date_any_symbol": latest,
        "modal_first_date": modal_first,
        "modal_last_date": modal_last,
        "n_symbols_universe": len(UNIVERSE),
        "n_symbols_with_data": sum(1 for s in UNIVERSE if per_symbol.get(s, {}).get("n_rows", 0) > 0),
        "n_names_full_history": len(full_names),
        "names_full_history": full_names,
        "names_partial_or_missing": partial,
        "total_rows": total_rows,
        "benchmark": BENCHMARK,
        "benchmark_coverage": per_symbol.get(BENCHMARK),
        "per_symbol": per_symbol,
    }


def rebuild_coverage_from_cache():
    """Re-derive coverage.json from the existing parquet cache WITHOUT re-fetching."""
    per_symbol = {}
    total_rows = 0
    for sym in UNIVERSE + [BENCHMARK]:
        path = CACHE_DIR / f"{sym}_daily.parquet"
        if not path.exists():
            per_symbol[sym] = {"n_rows": 0, "first": None, "last": None}
            continue
        df = pd.read_parquet(path)
        if df.empty:
            per_symbol[sym] = {"n_rows": 0, "first": None, "last": None}
            continue
        d = pd.to_datetime(df["date"])
        per_symbol[sym] = {
            "n_rows": int(len(df)),
            "first": str(d.min().date()),
            "last": str(d.max().date()),
        }
        total_rows += int(len(df))
    coverage = _build_coverage(per_symbol, total_rows)
    (RESULTS_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2))
    return coverage


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = make_data_client()

    all_syms = UNIVERSE + [BENCHMARK]
    per_symbol = {}
    total_rows = 0
    for sym in all_syms:
        try:
            rows = fetch_symbol(client, sym, START, END)
        except Exception as e:  # network/symbol issues — record and continue
            per_symbol[sym] = {"n_rows": 0, "first": None, "last": None, "error": str(e)[:200]}
            print(f"{sym}: ERROR {e}", flush=True)
            continue
        if not rows:
            per_symbol[sym] = {"n_rows": 0, "first": None, "last": None}
            print(f"{sym}: 0 bars", flush=True)
            continue
        df = pd.DataFrame(rows, columns=COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        path = CACHE_DIR / f"{sym}_daily.parquet"
        df.to_parquet(path, index=False)
        first = df["date"].iloc[0].date()
        last = df["date"].iloc[-1].date()
        per_symbol[sym] = {"n_rows": int(len(df)), "first": str(first), "last": str(last)}
        total_rows += int(len(df))
        print(f"{sym}: {len(df)} rows, {first} -> {last}", flush=True)

    coverage = _build_coverage(per_symbol, total_rows)
    (RESULTS_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(
        f"\nCoverage: earliest={coverage['earliest_bar_date_any_symbol']} "
        f"latest={coverage['latest_bar_date_any_symbol']} "
        f"full_history_names={coverage['n_names_full_history']}/{len(UNIVERSE)} total_rows={total_rows}",
        flush=True,
    )
    print("Written to", RESULTS_DIR / "coverage.json", flush=True)
    return coverage


if __name__ == "__main__":
    main()
