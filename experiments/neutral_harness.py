"""neutral_harness.py — WALK-FORWARD + BETA-DECOMPOSITION + MARKET-NEUTRAL toolkit.

This module EXTENDS (never modifies) experiments/harness.py. It imports the validated
primitives directly so no backtest logic can quietly diverge from the production-identical
engine:

    from experiments.harness import (
        load_bars, resample, chronological_split, compute_indicators_df,
        research_backtest, _slip, _metrics, _parse_hm, _session_vwap,
        ROLLING_VOL_WINDOW, MIN_BARS_FOR_ENTRY, SESSION_MINUTES, NY,
    )

The prior edge hunt (results/SUMMARY.md) found 0/9 real edges: every "winning" variant
was long-beta riding ONE homogeneous bull TEST window. This harness exists to kill that
failure mode by (a) scoring across MULTIPLE sequential out-of-sample folds spanning
DIFFERENT regimes, and (b) regressing every variant's daily returns on SPY to prove any
surviving PnL is alpha, not beta.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only.

=============================================================================
COST & FILL MODEL (identical to harness.research_backtest, applied to EVERY leg)
=============================================================================
- Strict t->t+1 fills: decide on completed bar t, fill at bar t+1 OPEN.
- $0 commission + `slippage_bps` (default 1bp) adverse slippage per side, via harness._slip
  (BUY fills high, SELL fills low). The SPY hedge leg and cross-sectional short legs pay
  the SAME 1bp/side on every rebalance.
- No overnight: EOD flatten at `eod_flatten` (default 15:55 NY). Completed bars only.
- Same caps as production: notional / max_positions / max_exposure / cooldown.

=============================================================================
SPEC FORMAT for research_backtest_neutral(bars, spec)
=============================================================================
spec = {
    "mode": "long_short" | "spy_hedge" | "cross_sectional",

    # ---- shared mean-reversion entry/exit knobs (passed through to harness) ----
    # All the same keys harness.research_backtest understands are honored where the
    # mode delegates to it (notably 'spy_hedge'): entry_dist, vol_mult, vwap_exit_band,
    # max_hold, stop_loss, take_profit, trailing_stop, time_window, trend_filter,
    # trend_lookback, score_fn, score_threshold, notional, max_positions, max_exposure,
    # cooldown_min, slippage_bps, eod_flatten.

    "entry_dist": 0.005,
    "vol_mult": 1.2,
    "vwap_exit_band": 0.001,
    "max_hold": 15,
    "stop_loss": 0.005,
    "notional": 100.0,
    "max_positions": 4,
    "max_exposure": 500.0,
    "cooldown_min": 10.0,
    "slippage_bps": 1.0,
    "eod_flatten": "15:55",

    # ---- mode='long_short' specific ----
    # Long names that are >= entry_dist BELOW vwap, short names >= entry_dist ABOVE vwap,
    # each sized to `notional`. The book self-balances toward dollar-neutral because the
    # long and short legs draw from the same symmetric VWAP-distance predicate. To
    # reproduce harness long-only behavior set "enable_short": False (SANITY hook).
    "enable_long": True,
    "enable_short": True,

    # ---- mode='cross_sectional' specific ----
    # Each bar, rank the universe by dist_from_vwap; go LONG the bottom `xs_quantile`
    # fraction (most below vwap) and SHORT the top `xs_quantile` (most above vwap),
    # dollar-neutral (equal $ per leg). Held until next rebalance / EOD.
    "xs_quantile": 0.3,
    "xs_book": 500.0,        # total long $ == total short $ each bar (dollar-neutral)
    "xs_min_names": 2,       # require >= this many names per side to trade a bar

    # ---- mode='spy_hedge' specific ----
    # Run the existing LONG mean-reversion (delegated to harness.research_backtest), then
    # each bar short SPY notional == current net long market value so net beta ~0. The
    # hedge is rebalanced only when net long exposure CHANGES (avoids churn). Hedge fills
    # at t+1 OPEN with the same 1bp/side slippage; hedge PnL+costs are tracked.
    "hedge_symbol": "SPY",
    "hedge_rebalance_eps": 1.0,   # $ change in net-long below which we don't rebalance
}

CAPITAL BASE for daily returns: `return_base` (default = max_exposure, i.e. the $500
notional book). daily_returns divides each day's net PnL by this base.

=============================================================================
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- reuse the validated production-identical primitives (do NOT reimplement) ---
from experiments.harness import (  # noqa: F401  (re-exported for downstream convenience)
    load_bars,
    resample,
    chronological_split,
    compute_indicators_df,
    research_backtest,
    _slip,
    _metrics,
    _parse_hm,
    _session_vwap,
    ROLLING_VOL_WINDOW,
    MIN_BARS_FOR_ENTRY,
    SESSION_MINUTES,
    NY,
)

UNIVERSE_RANK = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
TRADING_YEAR = 252


# --------------------------------------------------------------------------- #
# 1) Walk-forward folds
# --------------------------------------------------------------------------- #
def walk_forward_folds(
    dates: List, n_folds: int = 4, train_frac: float = 0.5
) -> List[dict]:
    """Build `n_folds` SEQUENTIAL expanding-train / fixed-test walk-forward folds.

    `dates` is the sorted list of distinct trading dates spanning the full history. The
    OOS test windows are NON-OVERLAPPING and tile the back of history, so each fold's
    test is a different, later, out-of-sample slice. For each fold k (0-indexed), the test
    window is the k-th tile of the post-warmup region; TRAIN is everything strictly before
    that test window (expanding/anchored). This guarantees test data is never seen during
    that fold's tuning.

    Layout (n_folds=4, train_frac=0.5):
        | -------- initial train (50%) -------- | test0 | test1 | test2 | test3 |
        train for foldk = all dates before testk's start.

    Returns a list of dicts:
        {"fold": k,
         "train": (train_start, train_end, n_train_days),
         "test":  (test_start,  test_end,  n_test_days)}
    Dates are stringified (YYYY-MM-DD) for reporting; slicing uses date objects via
    slice_by_date_range().
    """
    ds = sorted(set(dates))
    n = len(ds)
    if n < n_folds + 2:
        raise ValueError(f"need at least {n_folds + 2} dates, got {n}")
    # Region reserved for OOS testing = the tail (1 - train_frac) of history.
    n_initial_train = max(1, int(round(n * train_frac)))
    n_initial_train = min(n_initial_train, n - n_folds)  # leave >= n_folds test days
    test_region = ds[n_initial_train:]
    m = len(test_region)
    # Tile the test region into n_folds contiguous chunks (last absorbs remainder).
    base = m // n_folds
    folds = []
    cursor = 0
    for k in range(n_folds):
        size = base + (1 if k >= n_folds - (m - base * n_folds) else 0)
        test_slice = test_region[cursor : cursor + size]
        cursor += size
        if not test_slice:
            continue
        test_start = test_slice[0]
        test_end = test_slice[-1]
        # TRAIN = all dates strictly before test_start (anchored/expanding).
        train_slice = [d for d in ds if d < test_start]
        folds.append(
            {
                "fold": k,
                "train": (str(train_slice[0]), str(train_slice[-1]), len(train_slice)),
                "test": (str(test_start), str(test_end), len(test_slice)),
                "_train_dates": set(train_slice),
                "_test_dates": set(test_slice),
            }
        )
    return folds


def slice_by_date_range(bars: Dict[str, pd.DataFrame], date_set) -> Dict[str, pd.DataFrame]:
    """Slice every symbol's bars to the given set of date objects (or stringy dates)."""
    norm = set()
    for d in date_set:
        if isinstance(d, str):
            norm.add(_dt.date.fromisoformat(d))
        else:
            norm.add(d)
    return {s: df[df["date"].isin(norm)] for s, df in bars.items()}


def all_trading_dates(bars: Dict[str, pd.DataFrame]) -> List:
    return sorted({d for df in bars.values() for d in df["date"].unique()})


# --------------------------------------------------------------------------- #
# 2) Daily returns
# --------------------------------------------------------------------------- #
def daily_returns(trades: List[dict], return_base: float) -> pd.Series:
    """Per-day strategy return series on the `return_base` capital base.

    Each closed trade contributes its realized $PnL to the day it CLOSED (exit_time's NY
    date). day_return = sum(pnl that day) / return_base. Indexed by trading date
    (datetime.date), sorted ascending. Days with no closed trade are omitted here; callers
    that need calendar alignment vs SPY should reindex to the union of trading dates and
    fill 0.0 (align_daily_returns does this).
    """
    if return_base <= 0:
        raise ValueError("return_base must be > 0")
    pnl_by_day: Dict[_dt.date, float] = {}
    for t in trades:
        et = t["exit_time"]
        d = et.date() if hasattr(et, "date") else pd.Timestamp(et).date()
        pnl_by_day[d] = pnl_by_day.get(d, 0.0) + float(t["pnl"])
    if not pnl_by_day:
        return pd.Series(dtype=float)
    s = pd.Series(pnl_by_day) / float(return_base)
    s.index = pd.Index([pd.Timestamp(d) for d in s.index])
    return s.sort_index()


def daily_returns_from_equity(equity_by_day: Dict, return_base: float) -> pd.Series:
    """Per-day return series from a {date: net_pnl_that_day} mapping (for hedge/xs modes)."""
    if return_base <= 0:
        raise ValueError("return_base must be > 0")
    if not equity_by_day:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): v / float(return_base) for d, v in equity_by_day.items()})
    return s.sort_index()


def align_daily_returns(strat: pd.Series, spy: pd.Series) -> Tuple[np.ndarray, np.ndarray, List]:
    """Align two daily-return series on the UNION of their dates (missing -> 0.0 PnL day).

    The strategy is flat (0 return) on a trading day with no closed position, so a 0.0 fill
    is the correct economic value, not a dropped observation. Returns (strat_arr, spy_arr,
    dates) over the union of dates that EITHER series covers, intersected with SPY's domain
    (we can only regress on days SPY has a return for).
    """
    if spy.empty:
        return np.array([]), np.array([]), []
    idx = spy.index  # regress only on days we have a market return
    s = strat.reindex(idx).fillna(0.0)
    return s.to_numpy(dtype=float), spy.reindex(idx).to_numpy(dtype=float), list(idx)


# --------------------------------------------------------------------------- #
# 3) Beta decomposition (OLS, numpy only — no statsmodels dependency)
# --------------------------------------------------------------------------- #
def beta_decompose(strat_daily: pd.Series, spy_daily: pd.Series) -> dict:
    """OLS regression of strategy daily returns on SPY daily returns.

        strat_t = alpha + beta * spy_t + eps_t

    Returns:
        {alpha_per_day, alpha_annual, beta, alpha_tstat, beta_tstat, r2, n_days,
         strat_mean_daily, strat_ann_return}
    alpha_annual = alpha_per_day * 252 (arithmetic). alpha_tstat uses the OLS standard
    error of the intercept (HC0 not applied; daily equity-curve returns, classical SE).
    A 'market-neutral' claim requires |beta| < ~0.15 AND alpha_tstat >= 2.
    """
    y, x, dates = align_daily_returns(strat_daily, spy_daily)
    n = len(y)
    out = {
        "alpha_per_day": 0.0,
        "alpha_annual": 0.0,
        "beta": 0.0,
        "alpha_tstat": 0.0,
        "beta_tstat": 0.0,
        "r2": 0.0,
        "n_days": int(n),
        "strat_mean_daily": float(np.mean(y)) if n else 0.0,
        "strat_ann_return": float(np.mean(y) * TRADING_YEAR) if n else 0.0,
    }
    if n < 3:
        return out
    # Design matrix [1, x]; OLS via normal equations.
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return out
    beta_hat = XtX_inv @ (X.T @ y)  # [alpha, beta]
    resid = y - X @ beta_hat
    dof = n - 2
    if dof <= 0:
        return out
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    alpha = float(beta_hat[0])
    beta = float(beta_hat[1])
    out.update(
        {
            "alpha_per_day": alpha,
            "alpha_annual": alpha * TRADING_YEAR,
            "beta": beta,
            "alpha_tstat": alpha / se_alpha if se_alpha > 0 else 0.0,
            "beta_tstat": beta / se_beta if se_beta > 0 else 0.0,
            "r2": r2,
            "n_days": int(n),
            "strat_mean_daily": float(y.mean()),
            "strat_ann_return": float(y.mean() * TRADING_YEAR),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Internal: per-day net PnL aggregation helper for hedge/xs modes
# --------------------------------------------------------------------------- #
def _day_pnl_from_trades(trades: List[dict]) -> Dict[_dt.date, float]:
    out: Dict[_dt.date, float] = {}
    for t in trades:
        et = t["exit_time"]
        d = et.date() if hasattr(et, "date") else pd.Timestamp(et).date()
        out[d] = out.get(d, 0.0) + float(t["pnl"])
    return out


# --------------------------------------------------------------------------- #
# 4) research_backtest_neutral — three market-neutral modes
# --------------------------------------------------------------------------- #
def research_backtest_neutral(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Dispatch to the requested neutral mode. Returns {trades, daily, metrics, mode, ...}.

    `daily` is a pd.Series of per-day returns on the capital base. All modes use strict
    t->t+1 fills, 1bp/side on every leg, EOD flatten, and the production caps.
    """
    mode = spec.get("mode", "long_short")
    if mode == "long_short":
        return _bt_long_short(bars, spec)
    if mode == "spy_hedge":
        return _bt_spy_hedge(bars, spec)
    if mode == "cross_sectional":
        return _bt_cross_sectional(bars, spec)
    raise ValueError(f"unknown mode {mode!r}")


# ---- mode: long_short --------------------------------------------------------
def _bt_long_short(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Symmetric VWAP mean-reversion: long names below vwap, short names above vwap.

    Delegates to harness.research_backtest with side derived from enable_long/enable_short.
    Because the long and short predicates are mirror images (dist <= -entry_dist vs
    dist >= +entry_dist) and each fill is sized to the same `notional`, the book trends
    toward dollar-neutral. enable_short=False reproduces harness long-only EXACTLY (SANITY).
    """
    enable_long = bool(spec.get("enable_long", True))
    enable_short = bool(spec.get("enable_short", True))
    if enable_long and enable_short:
        side = "both"
    elif enable_long:
        side = "long"
    elif enable_short:
        side = "short"
    else:
        raise ValueError("long_short needs at least one of enable_long/enable_short")

    sub = dict(spec)
    sub.pop("mode", None)
    sub["side"] = side
    res = research_backtest(bars, sub)
    base = float(spec.get("return_base", spec.get("max_exposure", 500.0)))
    daily = daily_returns(res["trades"], base)
    return {
        "mode": "long_short",
        "side": side,
        "trades": res["trades"],
        "metrics": res["metrics"],
        "daily": daily,
        "return_base": base,
    }


# ---- mode: spy_hedge ---------------------------------------------------------
def _bt_spy_hedge(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Long VWAP mean-reversion (delegated) + per-bar SPY short to net beta ~0.

    Step 1: run the existing LONG strategy via harness.research_backtest (side forced
            'long'); collect its trades -> long-leg PnL by day.
    Step 2: reconstruct the strategy's NET LONG market value bar-by-bar from its trade
            ledger, then short SPY notional equal to that net long value each bar. The
            hedge is rebalanced only when net long changes by more than hedge_rebalance_eps
            (avoids churn). Each hedge adjustment fills at t+1 SPY OPEN with 1bp/side
            adverse slippage. Hedge PnL and costs are tracked and added to daily PnL.

    The hedge holds short SPY shares H_t. PnL of the hedge over a bar = -H_t * dSPY_open.
    Adjusting H pays slippage on the |delta shares| traded at that bar's t+1 open.
    """
    hedge_sym = spec.get("hedge_symbol", "SPY")
    if hedge_sym not in bars or bars[hedge_sym].empty:
        raise ValueError(f"spy_hedge requires {hedge_sym} bars in `bars`")
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    rebalance_eps = float(spec.get("hedge_rebalance_eps", 1.0))
    eod_cutoff = _parse_hm(spec.get("eod_flatten", "15:55"))
    base = float(spec.get("return_base", spec.get("max_exposure", 500.0)))

    # --- Step 1: long leg (production-identical) ---
    sub = dict(spec)
    sub.pop("mode", None)
    sub["side"] = "long"
    long_res = research_backtest(bars, sub)
    long_trades = long_res["trades"]

    # --- Build a per-bar timeline of NET LONG market value of the strategy book. ---
    # We mark each open long position at its CURRENT bar close (mark-to-market), same as
    # harness's equity loop. Reconstruct open intervals from the trade ledger.
    # Each trade has entry_time (a fill at some bar's open) and exit_time.
    # Net long value at bar t = sum over positions open at t of qty * close_t.
    # We approximate "open at bar t" as entry_time <= t < exit_time (position is on the
    # book and marked from its entry fill bar up to but not including its exit fill bar).
    spy_ind = compute_indicators_df(bars[hedge_sym])
    spy_open = spy_ind["open"]
    spy_ts = list(spy_ind.index)
    spy_date = spy_ind["date"]

    # Index symbol closes for marking long positions.
    sym_close: Dict[str, pd.Series] = {}
    for sym, df in bars.items():
        if df.empty:
            continue
        sym_close[sym] = df["close"]

    # For each bar timestamp present in SPY, compute net long market value.
    # Pre-sort trades by entry for interval logic.
    intervals = []
    for tr in long_trades:
        intervals.append(
            {
                "symbol": tr["symbol"],
                "qty": float(tr["qty"]),
                "entry_time": pd.Timestamp(tr["entry_time"]),
                "exit_time": pd.Timestamp(tr["exit_time"]),
            }
        )

    # Net long value per SPY bar timestamp.
    spy_index = pd.DatetimeIndex(spy_ts)
    net_long = pd.Series(0.0, index=spy_index)
    for iv in intervals:
        sym = iv["symbol"]
        cl = sym_close.get(sym)
        if cl is None:
            continue
        mask = (spy_index >= iv["entry_time"]) & (spy_index < iv["exit_time"])
        if not mask.any():
            continue
        # mark the position at this symbol's close on those bars (reindex to spy bars).
        marks = cl.reindex(spy_index[mask]).ffill().fillna(0.0)
        net_long.loc[spy_index[mask]] += iv["qty"] * marks.to_numpy(dtype=float)

    # --- Step 2: run the hedge sized to net_long, rebalanced t->t+1, with slippage. ---
    # Walk SPY bars in order, per day (flatten hedge at EOD, never overnight).
    hedge_pnl_by_day: Dict[_dt.date, float] = {}
    hedge_cost_total = 0.0
    hedge_rebalances = 0

    spy_open_arr = spy_open.to_numpy(dtype=float)
    net_long_arr = net_long.to_numpy(dtype=float)
    dates_arr = list(spy_date.to_numpy())
    n = len(spy_ts)

    # group bar indices by day to enforce no-overnight hedge.
    from itertools import groupby

    cur_shares = 0.0  # short SPY shares currently held (>=0 means short that many)
    for day, day_iter in groupby(range(n), key=lambda i: dates_arr[i]):
        day_idxs = list(day_iter)
        cur_shares = 0.0  # start flat each day
        day_pnl = 0.0
        for pos, t in enumerate(day_idxs):
            ts = spy_ts[t]
            t_obj = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
            is_eod = t_obj >= eod_cutoff
            has_next = (pos + 1) < len(day_idxs)
            # 1) Mark hedge PnL over the NEXT bar move (held t -> t+1 at open prices),
            #    consistent with t->t+1 fills (we hold cur_shares short into next bar).
            if has_next:
                nxt = day_idxs[pos + 1]
                d_open = spy_open_arr[nxt] - spy_open_arr[t]
                day_pnl += -cur_shares * d_open  # short SPY: gain when SPY open falls
            # 2) Decide target hedge for next bar = net long value at THIS bar / SPY price.
            #    Flatten at EOD (no overnight).
            if is_eod or not has_next:
                target_shares = 0.0
            else:
                target_notional = net_long_arr[t]
                fill_ref = spy_open_arr[day_idxs[pos + 1]]  # rebalance fills at t+1 open
                target_shares = target_notional / fill_ref if fill_ref > 0 else 0.0
            # 3) Rebalance if change in notional terms exceeds eps.
            delta_shares = target_shares - cur_shares
            fill_ref = spy_open_arr[day_idxs[pos + 1]] if has_next else spy_open_arr[t]
            delta_notional = abs(delta_shares) * fill_ref
            if delta_notional > rebalance_eps:
                # slippage on the traded shares: increasing short = SELL, covering = BUY.
                trade_side = "sell" if delta_shares > 0 else "buy"
                fill_price = _slip(float(fill_ref), trade_side, slippage_bps)
                # cost = adverse slippage vs mid = |delta_shares| * |fill_price - fill_ref|
                hedge_cost_total += abs(delta_shares) * abs(fill_price - fill_ref)
                day_pnl -= abs(delta_shares) * abs(fill_price - fill_ref)
                cur_shares = target_shares
                hedge_rebalances += 1
        hedge_pnl_by_day[day if not hasattr(day, "date") else day] = (
            hedge_pnl_by_day.get(day, 0.0) + day_pnl
        )

    # --- Combine long-leg + hedge daily PnL ---
    long_pnl_by_day = _day_pnl_from_trades(long_trades)
    all_days = sorted(set(long_pnl_by_day) | set(hedge_pnl_by_day))
    combined: Dict[_dt.date, float] = {}
    for d in all_days:
        d if not hasattr(d, "date") else d  # numpy date or date
        combined[d] = long_pnl_by_day.get(d, 0.0) + hedge_pnl_by_day.get(d, 0.0)

    daily = daily_returns_from_equity(combined, base)

    # Metrics: reuse harness._metrics on long trades, then patch in hedge-adjusted totals.
    long_metrics = long_res["metrics"]
    total_long_pnl = long_metrics["total_pnl"]
    total_hedge_pnl = sum(hedge_pnl_by_day.values())
    net_total_pnl = total_long_pnl + total_hedge_pnl
    metrics = dict(long_metrics)
    metrics.update(
        {
            "long_pnl": total_long_pnl,
            "hedge_pnl": total_hedge_pnl,
            "hedge_cost_total": hedge_cost_total,
            "hedge_rebalances": hedge_rebalances,
            "net_total_pnl": net_total_pnl,
            "total_return": net_total_pnl / base if base else 0.0,
        }
    )
    return {
        "mode": "spy_hedge",
        "trades": long_trades,
        "metrics": metrics,
        "daily": daily,
        "return_base": base,
        "hedge_pnl_by_day": {str(k): v for k, v in hedge_pnl_by_day.items()},
    }


# ---- mode: cross_sectional ---------------------------------------------------
def _bt_cross_sectional(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Each bar, rank universe by dist_from_vwap; long bottom quantile, short top quantile.

    Dollar-neutral: equal $ split across the long basket and (separately) the short basket,
    with total long $ == total short $ == xs_book/2 each side. Positions are opened at the
    NEXT bar's open (t->t+1) with 1bp/side slippage, marked to close, and rolled to the new
    target basket each bar; EOD flatten. We track per-bar PnL as the held basket's mark move
    plus rebalance slippage costs. Returns synthetic 'trades' = per-bar leg fills for
    accounting + a per-day return series.

    This is a portfolio (not a per-symbol position machine), so it does NOT reuse
    harness.research_backtest; instead it reuses compute_indicators_df + _slip + _metrics so
    the indicator math and cost model stay identical to production.
    """
    xs_quantile = float(spec.get("xs_quantile", 0.3))
    xs_book = float(spec.get("xs_book", spec.get("max_exposure", 500.0)))
    xs_min_names = int(spec.get("xs_min_names", 2))
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    eod_cutoff = _parse_hm(spec.get("eod_flatten", "15:55"))
    base = float(spec.get("return_base", xs_book))
    side_book = xs_book / 2.0  # $ on each side (long basket, short basket)

    # indicators per symbol
    ind = {sym: compute_indicators_df(df) for sym, df in bars.items() if not df.empty}
    syms = [s for s in UNIVERSE_RANK if s in ind] + [s for s in ind if s not in UNIVERSE_RANK]

    # Build a global timeline of bar timestamps -> {sym: (dist, open, close, n_bars)}.
    # Precompute per-symbol dicts keyed by timestamp for fast lookup.
    by_ts: Dict[pd.Timestamp, Dict[str, dict]] = {}
    for sym in syms:
        df = ind[sym]
        dist = df["dist_from_vwap"].to_numpy(dtype=float)
        op = df["open"].to_numpy(dtype=float)
        cl = df["close"].to_numpy(dtype=float)
        nb = df["n_bars"].to_numpy(dtype=int)
        ts = list(df.index)
        dd = df["date"].to_numpy()
        for i, t in enumerate(ts):
            by_ts.setdefault(t, {})[sym] = {
                "dist": dist[i],
                "open": op[i],
                "close": cl[i],
                "n_bars": nb[i],
                "date": dd[i],
                "idx": i,
            }
    # we need next-bar open per (sym, ts): build per-sym index->open and ts->idx map.
    {sym: ind[sym]["open"].to_numpy(dtype=float) for sym in syms}
    sym_ts_list = {sym: list(ind[sym].index) for sym in syms}
    sym_ts_pos = {sym: {t: i for i, t in enumerate(sym_ts_list[sym])} for sym in syms}

    ordered_ts = sorted(by_ts.keys())

    # held basket from PREVIOUS decision: {sym: signed_shares} (long>0, short<0),
    # entered at the bar following the decision; we mark its move over each bar.
    held: Dict[str, float] = {}
    pnl_by_day: Dict[_dt.date, float] = {}
    cost_total = 0.0
    trades: List[dict] = []  # synthetic per-rebalance fills for accounting
    rebalances = 0

    # group timestamps by day to reset/flatten at EOD
    from itertools import groupby

    def _day_of(t):
        return by_ts[t][next(iter(by_ts[t]))]["date"]

    for day, day_ts_iter in groupby(ordered_ts, key=_day_of):
        day_ts = list(day_ts_iter)
        held = {}  # flat at open each day
        day_pnl = 0.0
        for pos, t in enumerate(day_ts):
            row = by_ts[t]
            t_obj = t.timetz().replace(tzinfo=None) if hasattr(t, "timetz") else t.time()
            is_eod = t_obj >= eod_cutoff
            has_next = (pos + 1) < len(day_ts)

            # 1) mark currently-held basket over t -> t+1 (close-to-close of the held names).
            if has_next and held:
                nxt = day_ts[pos + 1]
                nxt_row = by_ts[nxt]
                for sym, sh in held.items():
                    if sym in row and sym in nxt_row:
                        d_close = nxt_row[sym]["close"] - row[sym]["close"]
                        day_pnl += sh * d_close

            # 2) decide the NEW target basket from THIS bar's ranks (fill next bar open).
            target: Dict[str, float] = {}
            if has_next and not is_eod:
                cand = []
                for sym, d in row.items():
                    if (
                        not np.isnan(d["dist"])
                        and d["n_bars"] >= MIN_BARS_FOR_ENTRY
                        and sym in sym_ts_pos
                        and t in sym_ts_pos[sym]
                    ):
                        cand.append((sym, d["dist"]))
                if len(cand) >= xs_min_names * 2:
                    cand.sort(key=lambda kv: kv[1])  # most-below vwap first
                    k = max(1, int(round(len(cand) * xs_quantile)))
                    longs = [s for s, _ in cand[:k]]
                    shorts = [s for s, _ in cand[-k:]]
                    if len(longs) >= xs_min_names and len(shorts) >= xs_min_names and not (set(longs) & set(shorts)):
                        per_long = side_book / len(longs)
                        per_short = side_book / len(shorts)
                        for s in longs:
                            nxt = day_ts[pos + 1]
                            o = by_ts[nxt][s]["open"] if s in by_ts.get(nxt, {}) else None
                            if o and o > 0:
                                target[s] = per_long / o
                        for s in shorts:
                            nxt = day_ts[pos + 1]
                            o = by_ts[nxt][s]["open"] if s in by_ts.get(nxt, {}) else None
                            if o and o > 0:
                                target[s] = -(per_short / o)

            # 3) rebalance held -> target at t+1 open, pay slippage on |delta shares|.
            if has_next:
                nxt = day_ts[pos + 1]
                names = set(held) | set(target)
                for sym in names:
                    cur = held.get(sym, 0.0)
                    tgt = target.get(sym, 0.0)
                    delta = tgt - cur
                    if abs(delta) < 1e-12:
                        continue
                    o = by_ts[nxt].get(sym, {}).get("open")
                    if not o or o <= 0:
                        continue
                    trade_side = "buy" if delta > 0 else "sell"
                    fill_price = _slip(float(o), trade_side, slippage_bps)
                    cost = abs(delta) * abs(fill_price - o)
                    cost_total += cost
                    day_pnl -= cost
                    rebalances += 1
                    trades.append(
                        {
                            "symbol": sym,
                            "side": "long" if delta > 0 else "short",
                            "entry_time": nxt,
                            "exit_time": nxt,
                            "entry_price": fill_price,
                            "exit_price": fill_price,
                            "qty": abs(delta),
                            "pnl": -cost,  # accounting placeholder; real PnL is in day_pnl marks
                            "return_pct": 0.0,
                            "holding_min": 0.0,
                            "exit_reason": "xs_rebalance",
                        }
                    )
                held = target
            else:
                held = {}  # last bar of day: flat
        pnl_by_day[day] = pnl_by_day.get(day, 0.0) + day_pnl

    daily = daily_returns_from_equity(pnl_by_day, base)
    total_pnl = sum(pnl_by_day.values())
    metrics = {
        "n_rebalance_fills": len(trades),
        "rebalances": rebalances,
        "total_pnl": total_pnl,
        "total_return": total_pnl / base if base else 0.0,
        "cost_total": cost_total,
        "n_days": len(pnl_by_day),
        "mean_daily_return": float(daily.mean()) if len(daily) else 0.0,
    }
    return {
        "mode": "cross_sectional",
        "trades": trades,
        "metrics": metrics,
        "daily": daily,
        "return_base": base,
    }


# --------------------------------------------------------------------------- #
# 5) Walk-forward evaluation with concatenated-OOS beta decomposition
# --------------------------------------------------------------------------- #
def evaluate_walkforward(
    spec_or_factory,
    bars: Dict[str, pd.DataFrame],
    spy_bars: Dict[str, pd.DataFrame],
    n_folds: int = 4,
    train_frac: float = 0.5,
    spy_symbol: str = "SPY",
) -> dict:
    """Run a neutral spec across `n_folds` sequential OOS folds; aggregate + beta-decompose.

    `spec_or_factory`:
        - a static spec dict (same config scored on every fold's TEST), OR
        - a callable(train_bars, fold_info) -> spec (tune on that fold's TRAIN ONLY, then
          the returned spec is scored on the untouched TEST). The factory must never look
          at test data.

    For each fold we run research_backtest_neutral on the TEST slice and collect that
    fold's per-day returns + metrics. We then CONCATENATE all OOS days across folds and run
    beta_decompose against SPY's daily returns over the same union of OOS days — this is the
    headline neutrality test (alpha t-stat, beta, R^2 on the full OOS path).

    Returns:
        {
          "n_folds", "folds": [per-fold dict],
          "folds_positive", "oos_mean_return", "oos_sharpe", "oos_n_trades",
          "oos_total_return", "oos_n_days",
          "beta_decompose": {... on concatenated OOS days ...},
          "fold_betas": [per-fold beta_decompose],
        }
    """
    dates = all_trading_dates(bars)
    folds = walk_forward_folds(dates, n_folds=n_folds, train_frac=train_frac)

    # SPY daily returns (close-to-close, RTH) for the full history — used for beta regress.
    spy_daily_full = spy_close_to_close_returns(spy_bars[spy_symbol]) if spy_symbol in spy_bars else pd.Series(dtype=float)

    fold_records = []
    oos_daily_pieces = []
    fold_betas = []
    total_trades = 0
    positive = 0
    for f in folds:
        test_bars = slice_by_date_range(bars, f["_test_dates"])
        if callable(spec_or_factory):
            train_bars = slice_by_date_range(bars, f["_train_dates"])
            spec = spec_or_factory(train_bars, f)
        else:
            spec = dict(spec_or_factory)
        res = research_backtest_neutral(test_bars, spec)
        daily = res["daily"]
        n_tr = len(res["trades"])
        total_trades += n_tr
        tot_ret = float(daily.sum()) if len(daily) else 0.0
        if tot_ret > 0:
            positive += 1
        oos_daily_pieces.append(daily)
        # per-fold beta on that fold's OOS days ONLY (restrict SPY to the strategy's
        # actual trading-day index so the regression is not diluted by non-traded days).
        spy_fold = spy_daily_full.reindex(daily.index).dropna() if len(daily) else spy_daily_full.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append(
            {
                "fold": f["fold"],
                "train": f["train"],
                "test": f["test"],
                "n_trades": n_tr,
                "total_return": tot_ret,
                "mean_daily_return": float(daily.mean()) if len(daily) else 0.0,
                "n_days": int(len(daily)),
                "metrics": res["metrics"],
            }
        )

    oos_daily = pd.concat(oos_daily_pieces).sort_index() if oos_daily_pieces else pd.Series(dtype=float)
    # de-dup any overlapping dates (shouldn't happen — test windows are disjoint).
    oos_daily = oos_daily[~oos_daily.index.duplicated(keep="first")]

    oos_mean = float(oos_daily.mean()) if len(oos_daily) else 0.0
    oos_std = float(oos_daily.std(ddof=1)) if len(oos_daily) > 1 else 0.0
    oos_sharpe = (oos_mean / oos_std * np.sqrt(TRADING_YEAR)) if oos_std > 0 else 0.0
    # Concatenated-OOS beta: regress ONLY on the OOS trading days (restrict SPY to the
    # OOS index) so the headline neutrality test is not diluted by non-OOS calendar days.
    spy_oos = spy_daily_full.reindex(oos_daily.index).dropna() if len(oos_daily) else spy_daily_full.iloc[0:0]
    beta_full = beta_decompose(oos_daily, spy_oos)

    return {
        "n_folds": len(folds),
        "folds": fold_records,
        "folds_positive": positive,
        "oos_mean_return": oos_mean,
        "oos_sharpe": oos_sharpe,
        "oos_n_trades": total_trades,
        "oos_total_return": float(oos_daily.sum()) if len(oos_daily) else 0.0,
        "oos_n_days": int(len(oos_daily)),
        "beta_decompose": beta_full,
        "fold_betas": fold_betas,
    }


def spy_close_to_close_returns(spy_df: pd.DataFrame) -> pd.Series:
    """Daily SPY return from RTH 1-min bars: (last close of day / prev day's last close) - 1.

    Used as the market factor in beta_decompose. Indexed by trading date (Timestamp).
    """
    if spy_df is None or spy_df.empty:
        return pd.Series(dtype=float)
    last_close = spy_df.groupby(spy_df["date"])["close"].last()
    last_close.index = pd.Index([pd.Timestamp(d) for d in last_close.index])
    last_close = last_close.sort_index()
    ret = last_close.pct_change().dropna()
    return ret


__all__ = [
    "walk_forward_folds",
    "slice_by_date_range",
    "all_trading_dates",
    "daily_returns",
    "daily_returns_from_equity",
    "align_daily_returns",
    "beta_decompose",
    "research_backtest_neutral",
    "evaluate_walkforward",
    "spy_close_to_close_returns",
]
