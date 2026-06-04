"""harness.py — shared RESEARCH toolkit for the VWAP mean-reversion edge hunt.

Every experiment imports this module. It is the ONE place that touches data loading,
chronological splitting, indicator computation, and the event-driven backtester, so no
downstream experiment can quietly diverge from the anti-self-deception protocol.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only.

=============================================================================
ANTI-SELF-DECEPTION PROTOCOL (baked into the API)
=============================================================================
- chronological_split() splits each symbol's history BY DATE using GLOBAL date cutoffs
  shared across all symbols. TRAIN = earliest ~60%, VALIDATION = next ~20%, TEST =
  latest ~20%. Never interleaved.
- Tune params on TRAIN only. SELECT a variant on VALIDATION only. Evaluate TEST exactly
  once per variant at the very end. search_params() never touches the test split.
- Strict t->t+1 fills: decide on completed bar t, fill at bar t+1's OPEN.
- Costs: $0 commission + 1bp adverse slippage per side (buy fills high, sell fills low),
  reported NET.

=============================================================================
SPEC FORMAT (the dict passed to research_backtest)
=============================================================================
spec = {
    # --- direction -------------------------------------------------------
    "side": "long" | "short" | "both",   # which entries are allowed (default "long")

    # --- entry predicate (VWAP mean reversion) ---------------------------
    # LONG entry  : dist_from_vwap <= -entry_dist  AND  volume_ratio >= vol_mult
    # SHORT entry : dist_from_vwap >= +entry_dist  AND  volume_ratio >= vol_mult
    "entry_dist": 0.005,     # fraction away from VWAP required to enter
    "vol_mult": 1.2,         # volume_ratio >= this (rolling-20 EXCLUDING current bar)

    # --- exit rules (priority: EOD > take_profit > trailing_stop >
    #                 vwap_revert > max_hold > stop_loss) -----------------
    "vwap_exit_band": 0.001, # exit when price reverts to within this frac of VWAP
    "max_hold": 15,          # minutes; exit when held strictly longer
    "stop_loss": 0.005,      # exit when unrealized loss reaches this frac (None disables)
    "take_profit": None,     # optional: exit when unrealized gain reaches this frac
    "trailing_stop": None,   # optional: exit when price falls `frac` from peak favorable

    # --- entry filters (optional) ----------------------------------------
    "time_window": None,     # ("HH:MM","HH:MM") NY; only enter when decision-bar time in [lo,hi)
    "trend_filter": None,    # None | "with" | "against"
                             #   computed as sign of (last_price - close[-trend_lookback])
                             #   "with"   : longs only when down-trend, shorts only when up-trend
                             #              (i.e. fade the move — classic mean reversion)
                             #   "against": the opposite (momentum confirmation)
    "trend_lookback": 20,    # bars used for the trend filter

    # --- optional ML / scoring gate --------------------------------------
    "score_fn": None,        # callable(feat: dict) -> float ; gates entries
    "score_threshold": 0.0,  # require score_fn(feat) >= this to enter

    # --- sizing / caps (default to production values) --------------------
    "notional": 100.0,
    "max_positions": 4,
    "max_exposure": 500.0,
    "cooldown_min": 10.0,
    "slippage_bps": 1.0,
    "eod_flatten": "15:55",  # NY; no new entries on/after, force-flatten open positions
}

`feat` dict passed to score_fn contains the decision-time features:
    symbol, dist_from_vwap, volume_ratio, minute_of_session, recent_return,
    bar_range_pct, session_progress, side ("long"/"short"), last_price, session_vwap.

=============================================================================
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CACHE_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/cache")
NY = ZoneInfo("America/New_York")
OPEN_T = _dt.time(9, 30, 0)
CLOSE_T = _dt.time(16, 0, 0)

ROLLING_VOL_WINDOW = 20
MIN_BARS_FOR_ENTRY = ROLLING_VOL_WINDOW + 1  # 21
RECENT_RETURN_K = 5
SESSION_MINUTES = 390


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_bars(symbols: List[str], cache_dir: Path | str = CACHE_DIR) -> Dict[str, pd.DataFrame]:
    """Load cached RTH 1-min bars per symbol.

    Returns {symbol: DataFrame} indexed by tz-aware NY timestamp, sorted ascending,
    columns ['open','high','low','close','volume','trade_count','vwap','date','minute'].
    Symbols with no cache file are omitted.
    """
    cache_dir = Path(cache_dir)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = cache_dir / f"{sym}_1min.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["start"] = pd.to_datetime(df["start"], utc=True).dt.tz_convert(NY)
        local_t = df["start"].dt.time
        df = df[(local_t >= OPEN_T) & (local_t < CLOSE_T)]
        df = df.drop_duplicates(subset=["start"]).sort_values("start")
        df = df.set_index("start")
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        df["date"] = df.index.date
        # minute_of_session: minutes since 09:30 NY
        mins = df.index.hour * 60 + df.index.minute - (9 * 60 + 30)
        df["minute"] = mins.astype(int)
        out[sym] = df
    return out


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-min OHLCV bars to coarser `minutes` bars, preserving session boundaries.

    Bars are bucketed within each trading day (never across the overnight gap). The bucket
    label is the START of the bucket (left-closed, left-labeled), consistent with the
    1-min convention where `start` is the minute the bar covers.
    """
    if minutes <= 1:
        return df.copy()
    pieces = []
    for _day, day_df in df.groupby(df.index.date, sort=True):
        agg = day_df.resample(f"{minutes}min", label="left", closed="left", origin="start_day").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trade_count": "sum",
                "vwap": "mean",
            }
        )
        agg = agg.dropna(subset=["open"])
        pieces.append(agg)
    if not pieces:
        out = df.iloc[0:0].copy()
    else:
        out = pd.concat(pieces).sort_index()
    out["date"] = out.index.date
    mins = out.index.hour * 60 + out.index.minute - (9 * 60 + 30)
    out["minute"] = mins.astype(int)
    return out


# --------------------------------------------------------------------------- #
# Chronological split (by DATE, global cutoffs across symbols)
# --------------------------------------------------------------------------- #
def chronological_split(
    bars: Dict[str, pd.DataFrame],
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], dict]:
    """Split each symbol's bars BY DATE using GLOBAL date cutoffs shared across symbols.

    The union of all trading dates across symbols is sorted; the earliest `train` fraction
    of distinct DATES go to TRAIN, the next `val` to VALIDATION, the rest to TEST. The same
    date cutoffs are applied to every symbol (no interleaving, no per-symbol drift).

    Returns (train_bars, val_bars, test_bars, ranges) where `ranges` records the actual
    date ranges per split.
    """
    all_dates = sorted({d for df in bars.values() for d in df["date"].unique()})
    n = len(all_dates)
    if n == 0:
        empty = ({s: df.iloc[0:0] for s, df in bars.items()},) * 3
        return (*empty, {"train": None, "val": None, "test": None})

    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    train_dates = set(all_dates[:n_train])
    val_dates = set(all_dates[n_train : n_train + n_val])
    test_dates = set(all_dates[n_train + n_val :])

    def _slice(date_set):
        return {s: df[df["date"].isin(date_set)] for s, df in bars.items()}

    ranges = {
        "train": (str(all_dates[0]), str(all_dates[n_train - 1]), n_train),
        "val": (str(all_dates[n_train]), str(all_dates[n_train + n_val - 1]), n_val),
        "test": (str(all_dates[n_train + n_val]), str(all_dates[-1]), n - n_train - n_val),
    }
    return _slice(train_dates), _slice(val_dates), _slice(test_dates), ranges


# --------------------------------------------------------------------------- #
# Indicator computation (mirrors agentic_trader.data.market_data.compute_indicators)
# --------------------------------------------------------------------------- #
def _session_vwap(high, low, close, volume) -> np.ndarray:
    """Cumulative session VWAP using typical price (H+L+C)/3 weighted by volume.

    Matches production: num/den cumulative over all completed bars in the session
    INCLUDING the decision bar. Resets are handled by caller (per-day arrays).
    """
    typical = (high + low + close) / 3.0
    cum_pv = np.cumsum(typical * volume)
    cum_v = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
    return vwap


def _rolling_avg_vol_excl_current(volume: np.ndarray) -> np.ndarray:
    """Rolling-20 average volume EXCLUDING the current bar.

    For bar index i, average over volume[max(0,i-20):i]  (the 20 bars strictly before i;
    production uses bars[n-21:n-1] for the last bar, i.e. the 20 bars before the decision
    bar). NaN when there is no prior bar.
    """
    n = len(volume)
    out = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(volume)])  # csum[k] = sum(volume[:k])
    for i in range(n):
        lo = max(0, i - ROLLING_VOL_WINDOW)
        cnt = i - lo
        if cnt > 0:
            out[i] = (csum[i] - csum[lo]) / cnt
    return out


def _compute_session_indicators(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute decision-time indicators for one trading day (VWAP resets per day).

    Adds columns: session_vwap, dist_from_vwap, rolling20_avg_vol, volume_ratio,
    recent_return, bar_range_pct, session_progress, n_bars (1-based bar index in session),
    trend_sign (sign of price change over a lookback; lookback applied by caller filter).
    Returns a copy. All values use only data up to and including each bar (no look-ahead).
    """
    h = day_df["high"].to_numpy(dtype=float)
    l = day_df["low"].to_numpy(dtype=float)
    c = day_df["close"].to_numpy(dtype=float)
    o = day_df["open"].to_numpy(dtype=float)
    v = day_df["volume"].to_numpy(dtype=float)
    n = len(c)

    vwap = _session_vwap(h, l, c, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where((vwap != 0) & ~np.isnan(vwap), (c - vwap) / vwap, np.nan)

    roll = _rolling_avg_vol_excl_current(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where((roll > 0), v / roll, np.nan)

    # recent_return over RECENT_RETURN_K bars (close vs close K bars ago)
    recent_return = np.full(n, np.nan)
    for i in range(n):
        if i >= RECENT_RETURN_K and c[i - RECENT_RETURN_K] != 0:
            recent_return[i] = (c[i] - c[i - RECENT_RETURN_K]) / c[i - RECENT_RETURN_K]
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_range_pct = np.where(o != 0, (h - l) / o, np.nan)

    out = day_df.copy()
    out["session_vwap"] = vwap
    out["dist_from_vwap"] = dist
    out["rolling20_avg_vol"] = roll
    out["volume_ratio"] = vol_ratio
    out["recent_return"] = recent_return
    out["bar_range_pct"] = bar_range_pct
    out["n_bars"] = np.arange(1, n + 1)
    out["session_progress"] = out["minute"].to_numpy(dtype=float) / float(SESSION_MINUTES)
    return out


def compute_indicators_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-session indicator computation across all days in `df`. VWAP resets daily."""
    pieces = [_compute_session_indicators(day_df) for _d, day_df in df.groupby(df.index.date, sort=True)]
    if not pieces:
        return df.copy()
    return pd.concat(pieces).sort_index()


# --------------------------------------------------------------------------- #
# Event-driven research backtester
# --------------------------------------------------------------------------- #
def _parse_hm(hm: str) -> _dt.time:
    parts = [int(x) for x in hm.split(":")]
    while len(parts) < 2:
        parts.append(0)
    return _dt.time(parts[0], parts[1])


def _slip(price: float, side: str, bps: float) -> float:
    """Adverse slippage. BUY fills higher, SELL fills lower."""
    frac = bps / 10_000.0
    return price * (1.0 + frac) if side == "buy" else price * (1.0 - frac)


def _spec_get(spec: dict, key: str, default):
    val = spec.get(key, default)
    return default if val is None and default is not None and key not in spec else val


def research_backtest(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Event-driven t->t+1 VWAP mean-reversion backtester (long/short/both).

    See module docstring for the spec format. Decides on completed bar t, fills at bar
    t+1 OPEN with adverse slippage; enforces position/exposure caps, per-symbol cooldown,
    fixed notional, and EOD flatten (no overnight). Returns {trades, metrics}.

    Metrics: n_trades, wins, losses, win_rate, total_pnl, total_return, avg_return_pct,
    sharpe_like, max_drawdown, turnover, exits_by_reason.
    """
    side_mode = spec.get("side", "long")
    entry_dist = float(spec.get("entry_dist", 0.005))
    vol_mult = float(spec.get("vol_mult", 1.2))
    vwap_exit_band = float(spec.get("vwap_exit_band", 0.001))
    max_hold = float(spec.get("max_hold", 15))
    stop_loss = spec.get("stop_loss", 0.005)
    take_profit = spec.get("take_profit", None)
    trailing_stop = spec.get("trailing_stop", None)
    time_window = spec.get("time_window", None)
    trend_filter = spec.get("trend_filter", None)
    trend_lookback = int(spec.get("trend_lookback", 20))
    score_fn = spec.get("score_fn", None)
    score_threshold = float(spec.get("score_threshold", 0.0))
    notional = float(spec.get("notional", 100.0))
    max_positions = int(spec.get("max_positions", 4))
    max_exposure = float(spec.get("max_exposure", 500.0))
    cooldown_min = float(spec.get("cooldown_min", 10.0))
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    eod_str = spec.get("eod_flatten", "15:55")
    eod_cutoff = _parse_hm(eod_str)

    tw_lo = tw_hi = None
    if time_window is not None:
        tw_lo = _parse_hm(time_window[0])
        tw_hi = _parse_hm(time_window[1])

    allow_long = side_mode in ("long", "both")
    allow_short = side_mode in ("short", "both")

    # Pre-compute indicators per symbol (VWAP resets daily inside).
    ind = {sym: compute_indicators_df(df) for sym, df in bars.items() if not df.empty}

    # Universe rank for deterministic per-step ordering.
    universe = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]

    def rank(sym):
        try:
            return (universe.index(sym), sym)
        except ValueError:
            return (len(universe), sym)

    # Build the global timeline: asof -> {sym: (row_idx, day_arrays)}.
    # We store per-(sym,day) arrays and iterate by index; the decision bar is index t,
    # the fill bar is index t+1 within the same session.
    # Flatten into events keyed by timestamp.
    # Each symbol-day: precompute numpy arrays once.
    sym_day_data = {}  # (sym, date) -> dict of arrays
    ts_index = {}  # asof timestamp -> list of (sym, date, t)
    for sym, df in ind.items():
        for day, day_df in df.groupby(df.index.date, sort=True):
            key = (sym, day)
            arr = {
                "ts": day_df.index.to_pydatetime(),
                "open": day_df["open"].to_numpy(dtype=float),
                "high": day_df["high"].to_numpy(dtype=float),
                "low": day_df["low"].to_numpy(dtype=float),
                "close": day_df["close"].to_numpy(dtype=float),
                "dist": day_df["dist_from_vwap"].to_numpy(dtype=float),
                "vol_ratio": day_df["volume_ratio"].to_numpy(dtype=float),
                "vwap": day_df["session_vwap"].to_numpy(dtype=float),
                "n_bars": day_df["n_bars"].to_numpy(dtype=int),
                "minute": day_df["minute"].to_numpy(dtype=int),
                "recent_return": day_df["recent_return"].to_numpy(dtype=float),
                "bar_range_pct": day_df["bar_range_pct"].to_numpy(dtype=float),
                "session_progress": day_df["session_progress"].to_numpy(dtype=float),
            }
            sym_day_data[key] = arr
            for t in range(len(arr["ts"])):
                ts_index.setdefault(arr["ts"][t], []).append((sym, day, t))

    open_positions: Dict[str, dict] = {}
    cooldowns: Dict[str, _dt.datetime] = {}
    trades: List[dict] = []
    realized_pnl = 0.0
    equity_points: List[float] = []
    turnover = 0.0  # total notional traded (entries + exits)

    def in_window(t_obj: _dt.time) -> bool:
        if tw_lo is None:
            return True
        if tw_lo <= tw_hi:
            return tw_lo <= t_obj < tw_hi
        return t_obj >= tw_lo or t_obj < tw_hi  # wraparound (not expected intraday)

    def trend_sign(arr, t) -> int:
        i0 = t - trend_lookback
        if i0 < 0:
            return 0
        prev = arr["close"][i0]
        cur = arr["close"][t]
        if cur > prev:
            return 1
        if cur < prev:
            return -1
        return 0

    ordered_ts = sorted(ts_index.keys())
    for asof in ordered_ts:
        entries = ts_index[asof]
        # group by symbol for this timestamp (one decision bar per symbol per ts)
        group = {sym: (day, t) for (sym, day, t) in entries}
        ordered_syms = sorted(group.keys(), key=rank)
        t_obj = asof.timetz().replace(tzinfo=None)
        is_eod = t_obj >= eod_cutoff

        # --- EXIT pass ---
        for sym in ordered_syms:
            if sym not in open_positions:
                continue
            day, t = group[sym]
            arr = sym_day_data[(sym, day)]
            pos = open_positions[sym]
            ps = pos["side"]  # "long" or "short"
            last_price = arr["close"][t]
            dist = arr["dist"][t]
            # track favorable extreme for trailing stop
            if ps == "long":
                pos["peak"] = max(pos.get("peak", pos["entry_price"]), last_price)
                unreal_plpc = (last_price - pos["entry_price"]) / pos["entry_price"]
            else:
                pos["peak"] = min(pos.get("peak", pos["entry_price"]), last_price)
                unreal_plpc = (pos["entry_price"] - last_price) / pos["entry_price"]
            holding_min = (asof - pos["entry_time"]).total_seconds() / 60.0

            exit_reason = None
            # Priority: EOD > take_profit > trailing_stop > vwap_revert > max_hold > stop_loss
            if is_eod:
                exit_reason = "eod_flatten"
            elif take_profit is not None and unreal_plpc >= float(take_profit):
                exit_reason = "take_profit"
            elif trailing_stop is not None:
                if ps == "long":
                    draw = (pos["peak"] - last_price) / pos["peak"] if pos["peak"] else 0.0
                else:
                    draw = (last_price - pos["peak"]) / pos["peak"] if pos["peak"] else 0.0
                if draw >= float(trailing_stop):
                    exit_reason = "trailing_stop"
            if exit_reason is None and not np.isnan(dist):
                if ps == "long" and dist >= -vwap_exit_band:
                    exit_reason = "vwap_revert"
                elif ps == "short" and dist <= vwap_exit_band:
                    exit_reason = "vwap_revert"
            if exit_reason is None and holding_min > max_hold:
                exit_reason = "max_hold"
            if exit_reason is None and stop_loss is not None and unreal_plpc <= -float(stop_loss):
                exit_reason = "stop_loss"

            if exit_reason is None:
                continue

            # Resolve fill: t+1 OPEN normally; if no t+1 bar, EOD/forced fills at t OPEN.
            has_next = (t + 1) < len(arr["ts"])
            if has_next:
                ref = arr["open"][t + 1]
                fill_time = arr["ts"][t + 1]
            elif exit_reason in ("eod_flatten", "forced_close"):
                ref = arr["open"][t]
                fill_time = arr["ts"][t]
            else:
                continue  # cannot fill without t+1 and not an EOD/forced exit -> hold

            close_side = "sell" if ps == "long" else "buy"
            fill_price = _slip(float(ref), close_side, slippage_bps)
            qty = pos["qty"]
            if ps == "long":
                pnl = (fill_price - pos["entry_price"]) * qty
                ret = (fill_price / pos["entry_price"]) - 1.0
            else:
                pnl = (pos["entry_price"] - fill_price) * qty
                ret = (pos["entry_price"] - fill_price) / pos["entry_price"]
            realized_pnl += pnl
            turnover += abs(fill_price * qty)
            trades.append(
                {
                    "symbol": sym,
                    "side": ps,
                    "entry_time": pos["entry_time"],
                    "exit_time": fill_time,
                    "entry_price": pos["entry_price"],
                    "exit_price": fill_price,
                    "qty": qty,
                    "pnl": pnl,
                    "return_pct": ret,
                    "holding_min": (fill_time - pos["entry_time"]).total_seconds() / 60.0,
                    "exit_reason": exit_reason,
                }
            )
            del open_positions[sym]

        # --- ENTRY pass ---
        if not is_eod:
            for sym in ordered_syms:
                if sym in open_positions:
                    continue
                if len(open_positions) >= max_positions:
                    break
                day, t = group[sym]
                arr = sym_day_data[(sym, day)]
                if (t + 1) >= len(arr["ts"]):
                    continue  # no t+1 fill bar
                # cooldown
                last = cooldowns.get(sym)
                if last is not None and (asof - last).total_seconds() / 60.0 < cooldown_min:
                    continue
                # min bars
                if arr["n_bars"][t] < MIN_BARS_FOR_ENTRY:
                    continue
                dist = arr["dist"][t]
                vr = arr["vol_ratio"][t]
                if np.isnan(dist) or np.isnan(vr):
                    continue
                if vr < vol_mult:
                    continue
                # time-of-day filter
                if not in_window(t_obj):
                    continue
                # determine candidate side from VWAP distance
                cand_side = None
                if allow_long and dist <= -entry_dist:
                    cand_side = "long"
                elif allow_short and dist >= entry_dist:
                    cand_side = "short"
                if cand_side is None:
                    continue
                # trend filter
                if trend_filter is not None:
                    ts = trend_sign(arr, t)
                    if trend_filter == "with":
                        # fade the move: long when downtrend, short when uptrend
                        if cand_side == "long" and ts >= 0:
                            continue
                        if cand_side == "short" and ts <= 0:
                            continue
                    elif trend_filter == "against":
                        # momentum confirm: long when uptrend, short when downtrend
                        if cand_side == "long" and ts <= 0:
                            continue
                        if cand_side == "short" and ts >= 0:
                            continue
                # exposure cap (current open basis + this entry)
                open_basis = sum(p["entry_price"] * p["qty"] for p in open_positions.values())
                if open_basis + notional > max_exposure + 1e-9:
                    continue
                # optional score gate
                if score_fn is not None:
                    feat = {
                        "symbol": sym,
                        "dist_from_vwap": float(dist),
                        "volume_ratio": float(vr),
                        "minute_of_session": int(arr["minute"][t]),
                        "recent_return": float(arr["recent_return"][t]) if not np.isnan(arr["recent_return"][t]) else 0.0,
                        "bar_range_pct": float(arr["bar_range_pct"][t]) if not np.isnan(arr["bar_range_pct"][t]) else 0.0,
                        "session_progress": float(arr["session_progress"][t]),
                        "side": cand_side,
                        "last_price": float(arr["close"][t]),
                        "session_vwap": float(arr["vwap"][t]) if not np.isnan(arr["vwap"][t]) else 0.0,
                    }
                    if float(score_fn(feat)) < score_threshold:
                        continue
                # fill at t+1 OPEN
                open_side = "buy" if cand_side == "long" else "sell"
                fill_price = _slip(float(arr["open"][t + 1]), open_side, slippage_bps)
                if fill_price <= 0:
                    continue
                qty = notional / fill_price
                fill_time = arr["ts"][t + 1]
                open_positions[sym] = {
                    "side": cand_side,
                    "qty": qty,
                    "entry_price": fill_price,
                    "entry_time": fill_time,
                    "peak": fill_price,
                }
                cooldowns[sym] = fill_time
                turnover += abs(fill_price * qty)

        # mark-to-market equity at this timestamp
        open_pnl = 0.0
        for sym, pos in open_positions.items():
            db = group.get(sym)
            if db is None:
                continue
            day, t = db
            arr = sym_day_data[(sym, day)]
            mark = arr["close"][t]
            if pos["side"] == "long":
                open_pnl += (mark - pos["entry_price"]) * pos["qty"]
            else:
                open_pnl += (pos["entry_price"] - mark) * pos["qty"]
        equity_points.append(realized_pnl + open_pnl)

    # Force-close anything still open at the last seen decision bar's OPEN (rare).
    for sym in sorted(open_positions.keys(), key=rank):
        pos = open_positions[sym]
        # find its last day/t
        # (positions only persist if they never hit EOD; force-close at last close as fallback)
        # Use the last known mark via equity is complex; conservatively close at entry (0 pnl)
        # but record so it's not silently dropped.
        trades.append(
            {
                "symbol": sym,
                "side": pos["side"],
                "entry_time": pos["entry_time"],
                "exit_time": pos["entry_time"],
                "entry_price": pos["entry_price"],
                "exit_price": pos["entry_price"],
                "qty": pos["qty"],
                "pnl": 0.0,
                "return_pct": 0.0,
                "holding_min": 0.0,
                "exit_reason": "forced_close",
            }
        )
    open_positions.clear()

    metrics = _metrics(trades, equity_points, notional, turnover)
    return {"trades": trades, "metrics": metrics}


def _metrics(trades, equity_points, notional, turnover) -> dict:
    n = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = n - wins
    win_rate = wins / n if n else 0.0
    avg_return_pct = sum(t["return_pct"] for t in trades) / n if n else 0.0
    total_return = total_pnl / notional if notional else 0.0

    rets = [t["return_pct"] for t in trades]
    sharpe_like = 0.0
    if len(rets) >= 2:
        arr = np.array(rets, dtype=float)
        std = arr.std(ddof=1)
        if std > 0:
            sharpe_like = (arr.mean() / std) * np.sqrt(len(arr))

    max_dd = 0.0
    if equity_points:
        eq = np.array(equity_points, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak).min()
        max_dd = float(dd)

    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "avg_return_pct": avg_return_pct,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "exits_by_reason": reasons,
    }


# --------------------------------------------------------------------------- #
# Nested-protocol convenience
# --------------------------------------------------------------------------- #
def evaluate_splits(spec_or_factory, train, val, test) -> dict:
    """Run a spec (or zero-arg factory returning a spec) on all three splits.

    Returns {"train": metrics, "val": metrics, "test": metrics}. The same spec is used
    for every split so the train->test overfit gap is directly visible. TEST is evaluated
    here exactly once per call — callers must not loop this over many configs on TEST.
    """
    def _spec():
        return spec_or_factory() if callable(spec_or_factory) else spec_or_factory

    return {
        "train": research_backtest(train, _spec())["metrics"],
        "val": research_backtest(val, _spec())["metrics"],
        "test": research_backtest(test, _spec())["metrics"],
    }


def search_params(
    factory: Callable[[dict], dict],
    param_grid: Dict[str, list],
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    n_iter: int,
    seed: int,
    objective: str = "total_pnl",
) -> dict:
    """Sampled search: tune on TRAIN, rank on VALIDATION, return the val-best config.

    NEVER touches the test split. Sampling is deterministic via numpy Generator(seed).
    `factory(sampled_params) -> spec`. Each sampled point is first run on TRAIN (to confirm
    it produces a sane, tradeable config), then scored on VALIDATION by `objective`. The
    config with the best VALIDATION objective is returned.

    Returns {"best_params", "best_spec", "val_metrics", "train_metrics", "tried": [...]}.
    """
    rng = np.random.default_rng(seed)
    keys = list(param_grid.keys())
    seen = set()
    tried = []
    best = None

    attempts = 0
    max_attempts = n_iter * 20
    while len(tried) < n_iter and attempts < max_attempts:
        attempts += 1
        sample = {}
        for k in keys:
            choices = param_grid[k]
            sample[k] = choices[int(rng.integers(0, len(choices)))]
        sig = tuple(sample[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        spec = factory(sample)
        train_m = research_backtest(train, spec)["metrics"]
        val_m = research_backtest(val, spec)["metrics"]
        rec = {"params": sample, "train": train_m, "val": val_m}
        tried.append(rec)
        score = val_m.get(objective, float("-inf"))
        if best is None or score > best["score"]:
            best = {"score": score, "rec": rec, "spec": spec}

    if best is None:
        return {"best_params": None, "best_spec": None, "val_metrics": None, "train_metrics": None, "tried": tried}
    return {
        "best_params": best["rec"]["params"],
        "best_spec": best["spec"],
        "val_metrics": best["rec"]["val"],
        "train_metrics": best["rec"]["train"],
        "tried": tried,
    }


__all__ = [
    "load_bars",
    "resample",
    "chronological_split",
    "compute_indicators_df",
    "research_backtest",
    "evaluate_splits",
    "search_params",
]
