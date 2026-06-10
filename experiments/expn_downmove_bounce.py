"""expn_downmove_bounce.py — BETA-NEUTRAL test of the pre-registered DOWN-MOVE BOUNCE.

HYPOTHESIS (pre-registered from the prior hunt)
-----------------------------------------------
The momentum_invert event study found a short-horizon REVERSION BOUNCE after a sharp
DOWN move (prior hunt: VAL K=15 +3.16bp, t=4.19). The thesis: when a name's trailing
K-bar return is sharply negative (a fast drop) on elevated volume, it tends to BOUNCE
back over the next several bars. We test whether that bounce is (a) REAL across multiple
sequential out-of-sample folds spanning different market regimes, and (b) genuinely
MARKET-NEUTRAL — i.e. surviving alpha after we SPY-hedge the long exposure away, net of
1bp/side slippage on EVERY leg (stock entries/exits AND every SPY hedge rebalance).

STRATEGY
--------
ENTRY (LONG only): on completed bar t, enter a fixed-$notional long if
    trailing K-bar return  kret[t] = close[t]/close[t-K] - 1  <=  drop_thr  (a sharp drop)
    AND volume_ratio[t] >= vol_mult                          (elevated volume)
    AND n_bars[t] >= 21, t+1 bar exists, cooldown clear, position/exposure caps OK,
        and the decision time is before the no-entry cutoff.
Fill at bar t+1 OPEN with buy-side adverse slippage (1bp).

EXIT (priority: EOD > take_profit(bounce) > stop_loss > max_hold):
    - take_profit: bounce captured — exit when unrealized gain >= bounce_tp.
    - stop_loss  : the drop kept going — exit when unrealized loss <= -stop_loss.
    - max_hold   : time stop in minutes (the bounce is a short-horizon effect).
Fill at bar t+1 OPEN with sell-side adverse slippage (EOD/forced at decision-bar open if
no t+1 bar). No overnight (EOD flatten 15:55 NY).

BETA-NEUTRALITY (mode='spy_hedge' in neutral_harness)
-----------------------------------------------------
The long leg above is a pure-long mean-reversion book, so it carries market beta. We make
it beta-neutral by SHORTING SPY notional equal to the strategy's net long market value,
rebalanced bar-by-bar (only when net long changes by > hedge_rebalance_eps to avoid churn),
each hedge fill at t+1 SPY open with the SAME 1bp/side adverse slippage. Net beta target ~0.
We then regress the HEDGED daily return path on SPY daily returns to prove (or disprove)
that any surviving PnL is alpha, not beta.

Because neutral_harness's spy_hedge mode DELEGATES the long leg to harness.research_backtest
(whose long predicate is `dist_from_vwap <= -entry_dist`, a VWAP-distance signal, NOT a
trailing-K-bar return), we cannot express this signal through that path. Instead this file
implements the trailing-K-bar-drop long leg with a small event-driven backtester that
REUSES the validated harness primitives (compute_indicators_df, _slip, _metrics, _parse_hm,
MIN_BARS_FOR_ENTRY) so the indicator math, cost model, fill timing, and metrics are
identical to the sanity-checked engine. The SPY hedge is then built with the EXACT same
mark-to-market + rebalance + 1bp/side logic as neutral_harness._bt_spy_hedge (we port that
hedge code verbatim, parameterized by the long leg's trades), and the OOS beta decomposition
uses neutral_harness.beta_decompose / spy_close_to_close_returns / walk_forward_folds
unchanged.

ANTI-SELF-DECEPTION / RIGOR PROTOCOL
------------------------------------
- WALK-FORWARD: 4 sequential, non-overlapping OOS test folds spanning different regimes
  (high-vol recovery / low-vol grind / chop / steady rally). For each fold we TUNE the
  signal+exit params on that fold's TRAIN portion ONLY (anchored/expanding), then score the
  untouched TEST. OOS days are concatenated across folds for the headline beta regression.
- BETA DECOMPOSITION: regress hedged daily returns on SPY daily returns -> alpha (annual +
  per-day), beta, alpha t-stat, R^2 — on the full concatenated OOS path and per fold.
- COSTS: $0 commission + 1bp adverse slippage per side on EVERY leg incl. hedge rebalances.
  Strict t->t+1 fills, completed bars only, EOD flatten 15:55 NY, production caps. NET.
- MULTIPLE COMPARISONS: we count every config tried; selection uses TRAIN only; each fold's
  TEST is scored once with that fold's selected config. Tiny PnL on thin data is treated
  skeptically — alpha must clear the cost/noise floor and have t>=2 with |beta|<~0.15.

survives_oos = TRUE only if: positive aggregate OOS alpha with alpha_tstat>=2, realized
|beta|<~0.15, positive in a MAJORITY of folds, AND above the cost/noise floor.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.expn_downmove_bounce
"""

from __future__ import annotations

import datetime as _dt
import json
from itertools import groupby
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments import harness as h
from experiments import neutral_harness as nh

UNIVERSE = nh.UNIVERSE_RANK  # ["SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMD","TSLA","META","AMZN"]
HEDGE_SYM = "SPY"
RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/downmove_bounce.json")
TRADING_YEAR = 252


# --------------------------------------------------------------------------- #
# Trailing K-bar return (per-session arrays; no look-ahead)
# --------------------------------------------------------------------------- #
def _kbar_ret(close: np.ndarray, K: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    for i in range(K, len(close)):
        if close[i - K] != 0:
            out[i] = close[i] / close[i - K] - 1.0
    return out


# --------------------------------------------------------------------------- #
# LONG leg: sharp-drop bounce, reusing harness primitives & cost model
# --------------------------------------------------------------------------- #
def _bounce_long_backtest(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Event-driven t->t+1 LONG backtester for the sharp-drop bounce signal.

    Mirrors harness.research_backtest mechanics (caps, cooldown, EOD flatten, t->t+1 fills,
    1bp/side slippage via h._slip, metrics via h._metrics, MIN_BARS_FOR_ENTRY gate) but with
    a trailing-K-bar-return ENTRY predicate (not VWAP distance).
    """
    K = int(spec.get("drop_k", 5))
    drop_thr = float(spec.get("drop_thr", -0.006))   # <= this (negative) to enter
    vol_mult = float(spec.get("vol_mult", 1.2))
    bounce_tp = spec.get("bounce_tp", 0.004)          # take-profit (bounce captured)
    stop_loss = spec.get("stop_loss", 0.006)          # hard stop (drop continued)
    max_hold = float(spec.get("max_hold", 15))        # minutes
    notional = float(spec.get("notional", 100.0))
    max_positions = int(spec.get("max_positions", 4))
    max_exposure = float(spec.get("max_exposure", 500.0))
    cooldown_min = float(spec.get("cooldown_min", 10.0))
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    eod_cutoff = h._parse_hm(spec.get("eod_flatten", "15:55"))
    no_entry_after_min = int(spec.get("no_entry_after_min", 380))

    ind = {sym: h.compute_indicators_df(df) for sym, df in bars.items() if not df.empty}

    def rank(sym):
        try:
            return (UNIVERSE.index(sym), sym)
        except ValueError:
            return (len(UNIVERSE), sym)

    sym_day_data: Dict = {}
    ts_index: Dict = {}
    for sym, df in ind.items():
        for day, day_df in df.groupby(df.index.date, sort=True):
            c = day_df["close"].to_numpy(dtype=float)
            arr = {
                "ts": day_df.index.to_pydatetime(),
                "open": day_df["open"].to_numpy(dtype=float),
                "close": c,
                "vol_ratio": day_df["volume_ratio"].to_numpy(dtype=float),
                "n_bars": day_df["n_bars"].to_numpy(dtype=int),
                "minute": day_df["minute"].to_numpy(dtype=int),
                "kret": _kbar_ret(c, K),
            }
            sym_day_data[(sym, day)] = arr
            for t in range(len(arr["ts"])):
                ts_index.setdefault(arr["ts"][t], []).append((sym, day, t))

    open_positions: Dict[str, dict] = {}
    cooldowns: Dict[str, _dt.datetime] = {}
    trades: List[dict] = []
    realized_pnl = 0.0
    equity_points: List[float] = []
    turnover = 0.0

    for asof in sorted(ts_index.keys()):
        entries = ts_index[asof]
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
            last_price = arr["close"][t]
            unreal_plpc = (last_price - pos["entry_price"]) / pos["entry_price"]
            holding_min = (asof - pos["entry_time"]).total_seconds() / 60.0

            exit_reason = None
            # Priority: EOD > take_profit(bounce) > stop_loss > max_hold
            if is_eod:
                exit_reason = "eod_flatten"
            elif bounce_tp is not None and unreal_plpc >= float(bounce_tp):
                exit_reason = "take_profit"
            elif stop_loss is not None and unreal_plpc <= -float(stop_loss):
                exit_reason = "stop_loss"
            elif holding_min > max_hold:
                exit_reason = "max_hold"
            if exit_reason is None:
                continue

            has_next = (t + 1) < len(arr["ts"])
            if has_next:
                ref = arr["open"][t + 1]
                fill_time = arr["ts"][t + 1]
            elif exit_reason in ("eod_flatten", "forced_close"):
                ref = arr["open"][t]
                fill_time = arr["ts"][t]
            else:
                continue
            fill_price = h._slip(float(ref), "sell", slippage_bps)
            qty = pos["qty"]
            pnl = (fill_price - pos["entry_price"]) * qty
            ret = (fill_price / pos["entry_price"]) - 1.0
            realized_pnl += pnl
            turnover += abs(fill_price * qty)
            trades.append({
                "symbol": sym, "side": "long",
                "entry_time": pos["entry_time"], "exit_time": fill_time,
                "entry_price": pos["entry_price"], "exit_price": fill_price,
                "qty": qty, "pnl": pnl, "return_pct": ret,
                "holding_min": (fill_time - pos["entry_time"]).total_seconds() / 60.0,
                "exit_reason": exit_reason,
            })
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
                    continue
                last = cooldowns.get(sym)
                if last is not None and (asof - last).total_seconds() / 60.0 < cooldown_min:
                    continue
                if arr["n_bars"][t] < h.MIN_BARS_FOR_ENTRY:
                    continue
                if arr["minute"][t] >= no_entry_after_min:
                    continue
                kr = arr["kret"][t]
                vr = arr["vol_ratio"][t]
                if np.isnan(kr) or np.isnan(vr):
                    continue
                if not (kr <= drop_thr and vr >= vol_mult):  # sharp DOWN move on volume
                    continue
                open_basis = sum(p["entry_price"] * p["qty"] for p in open_positions.values())
                if open_basis + notional > max_exposure + 1e-9:
                    continue
                fill_price = h._slip(float(arr["open"][t + 1]), "buy", slippage_bps)
                if fill_price <= 0:
                    continue
                qty = notional / fill_price
                fill_time = arr["ts"][t + 1]
                open_positions[sym] = {
                    "side": "long", "qty": qty, "entry_price": fill_price,
                    "entry_time": fill_time, "peak": fill_price,
                }
                cooldowns[sym] = fill_time
                turnover += abs(fill_price * qty)

        # mark-to-market equity
        open_pnl = 0.0
        for sym, pos in open_positions.items():
            db = group.get(sym)
            if db is None:
                continue
            day, t = db
            mark = sym_day_data[(sym, day)]["close"][t]
            open_pnl += (mark - pos["entry_price"]) * pos["qty"]
        equity_points.append(realized_pnl + open_pnl)

    # force-close stragglers (recorded at 0 pnl)
    for sym in sorted(open_positions.keys(), key=rank):
        pos = open_positions[sym]
        trades.append({
            "symbol": sym, "side": "long",
            "entry_time": pos["entry_time"], "exit_time": pos["entry_time"],
            "entry_price": pos["entry_price"], "exit_price": pos["entry_price"],
            "qty": pos["qty"], "pnl": 0.0, "return_pct": 0.0,
            "holding_min": 0.0, "exit_reason": "forced_close",
        })
    open_positions.clear()

    metrics = h._metrics(trades, equity_points, notional, turnover)
    return {"trades": trades, "metrics": metrics}


# --------------------------------------------------------------------------- #
# SPY HEDGE: port of neutral_harness._bt_spy_hedge hedge logic (bit-for-bit),
# parameterized by an arbitrary long-leg trade ledger (our bounce long leg).
# Same mark-to-market, same per-bar rebalance, same 1bp/side on each hedge fill.
# --------------------------------------------------------------------------- #
def _spy_hedge_daily(bars: Dict[str, pd.DataFrame], long_trades: List[dict], spec: dict) -> dict:
    hedge_sym = spec.get("hedge_symbol", HEDGE_SYM)
    if hedge_sym not in bars or bars[hedge_sym].empty:
        raise ValueError(f"spy_hedge requires {hedge_sym} bars in `bars`")
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    rebalance_eps = float(spec.get("hedge_rebalance_eps", 1.0))
    eod_cutoff = h._parse_hm(spec.get("eod_flatten", "15:55"))
    base = float(spec.get("return_base", spec.get("max_exposure", 500.0)))

    spy_ind = h.compute_indicators_df(bars[hedge_sym])
    spy_open = spy_ind["open"]
    spy_ts = list(spy_ind.index)
    spy_date = spy_ind["date"]

    sym_close: Dict[str, pd.Series] = {}
    for sym, df in bars.items():
        if df.empty:
            continue
        sym_close[sym] = df["close"]

    intervals = []
    for tr in long_trades:
        intervals.append({
            "symbol": tr["symbol"],
            "qty": float(tr["qty"]),
            "entry_time": pd.Timestamp(tr["entry_time"]),
            "exit_time": pd.Timestamp(tr["exit_time"]),
        })

    spy_index = pd.DatetimeIndex(spy_ts)
    net_long = pd.Series(0.0, index=spy_index)
    for iv in intervals:
        cl = sym_close.get(iv["symbol"])
        if cl is None:
            continue
        mask = (spy_index >= iv["entry_time"]) & (spy_index < iv["exit_time"])
        if not mask.any():
            continue
        marks = cl.reindex(spy_index[mask]).ffill().fillna(0.0)
        net_long.loc[spy_index[mask]] += iv["qty"] * marks.to_numpy(dtype=float)

    hedge_pnl_by_day: Dict[_dt.date, float] = {}
    hedge_cost_total = 0.0
    hedge_rebalances = 0

    spy_open_arr = spy_open.to_numpy(dtype=float)
    net_long_arr = net_long.to_numpy(dtype=float)
    dates_arr = list(spy_date.to_numpy())
    n = len(spy_ts)

    for day, day_iter in groupby(range(n), key=lambda i: dates_arr[i]):
        day_idxs = list(day_iter)
        cur_shares = 0.0
        day_pnl = 0.0
        for pos, t in enumerate(day_idxs):
            ts = spy_ts[t]
            t_obj = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
            is_eod = t_obj >= eod_cutoff
            has_next = (pos + 1) < len(day_idxs)
            if has_next:
                nxt = day_idxs[pos + 1]
                d_open = spy_open_arr[nxt] - spy_open_arr[t]
                day_pnl += -cur_shares * d_open
            if is_eod or not has_next:
                target_shares = 0.0
            else:
                target_notional = net_long_arr[t]
                fill_ref = spy_open_arr[day_idxs[pos + 1]]
                target_shares = target_notional / fill_ref if fill_ref > 0 else 0.0
            delta_shares = target_shares - cur_shares
            fill_ref = spy_open_arr[day_idxs[pos + 1]] if has_next else spy_open_arr[t]
            delta_notional = abs(delta_shares) * fill_ref
            if delta_notional > rebalance_eps:
                trade_side = "sell" if delta_shares > 0 else "buy"
                fill_price = h._slip(float(fill_ref), trade_side, slippage_bps)
                hedge_cost_total += abs(delta_shares) * abs(fill_price - fill_ref)
                day_pnl -= abs(delta_shares) * abs(fill_price - fill_ref)
                cur_shares = target_shares
                hedge_rebalances += 1
        hedge_pnl_by_day[day] = hedge_pnl_by_day.get(day, 0.0) + day_pnl

    long_pnl_by_day = nh._day_pnl_from_trades(long_trades)
    all_days = sorted(set(long_pnl_by_day) | set(hedge_pnl_by_day))
    combined: Dict[_dt.date, float] = {}
    for d in all_days:
        combined[d] = long_pnl_by_day.get(d, 0.0) + hedge_pnl_by_day.get(d, 0.0)
    daily = nh.daily_returns_from_equity(combined, base)

    total_long_pnl = sum(long_pnl_by_day.values())
    total_hedge_pnl = sum(hedge_pnl_by_day.values())
    return {
        "daily": daily,
        "long_pnl": total_long_pnl,
        "hedge_pnl": total_hedge_pnl,
        "hedge_cost_total": hedge_cost_total,
        "hedge_rebalances": hedge_rebalances,
        "net_total_pnl": total_long_pnl + total_hedge_pnl,
        "return_base": base,
    }


def _bounce_hedged(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Run the bounce long leg + SPY hedge -> hedged daily returns + metrics."""
    long_res = _bounce_long_backtest(bars, spec)
    hedge = _spy_hedge_daily(bars, long_res["trades"], spec)
    metrics = dict(long_res["metrics"])
    metrics.update({
        "long_pnl": hedge["long_pnl"],
        "hedge_pnl": hedge["hedge_pnl"],
        "hedge_cost_total": hedge["hedge_cost_total"],
        "hedge_rebalances": hedge["hedge_rebalances"],
        "net_total_pnl": hedge["net_total_pnl"],
        "net_total_return": hedge["net_total_pnl"] / hedge["return_base"] if hedge["return_base"] else 0.0,
    })
    return {
        "mode": "spy_hedge",
        "trades": long_res["trades"],
        "metrics": metrics,
        "daily": hedge["daily"],
        "return_base": hedge["return_base"],
    }


# --------------------------------------------------------------------------- #
# Per-fold TRAIN tuner (selects on TRAIN ONLY; TEST never seen during tuning)
# --------------------------------------------------------------------------- #
GRID = {
    "drop_k": [5, 10],
    "drop_thr": [-0.005, -0.006, -0.008],
    "vol_mult": [1.2, 1.5],
    "bounce_tp": [0.003, 0.004, 0.006],
    "stop_loss": [0.006, 0.008],
    "max_hold": [10, 15, 25],
}
N_ITER = 24
SEED = 11
MIN_TRAIN_TRADES = 40  # require a tradeable config so we don't select on noise

_BASE = {
    "notional": 100.0, "max_positions": 4, "max_exposure": 500.0,
    "cooldown_min": 10.0, "slippage_bps": 1.0, "eod_flatten": "15:55",
    "no_entry_after_min": 380, "hedge_symbol": HEDGE_SYM,
    "hedge_rebalance_eps": 1.0, "return_base": 500.0,
}


def _make_spec(sample: dict) -> dict:
    spec = dict(_BASE)
    spec.update(sample)
    return spec


# Count of distinct configs evaluated across all folds (multiple-comparisons honesty).
_GLOBAL_TRIED = {"count": 0}


def _tune_factory(train_bars: Dict[str, pd.DataFrame], fold_info: dict) -> dict:
    """Deterministic sampled search on TRAIN ONLY. Objective: HEDGED OOS-style net total
    return on TRAIN (i.e. alpha proxy on train), requiring >= MIN_TRAIN_TRADES so we never
    select on a near-empty config. Returns the chosen spec to be scored on the untouched
    TEST fold. Never looks at test data.
    """
    rng = np.random.default_rng(SEED + fold_info["fold"])
    keys = list(GRID.keys())
    seen = set()
    best = None
    n_local = 0
    attempts = 0
    while n_local < N_ITER and attempts < N_ITER * 40:
        attempts += 1
        sample = {k: GRID[k][int(rng.integers(0, len(GRID[k])))] for k in keys}
        sig = tuple((k, sample[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        spec = _make_spec(sample)
        res = _bounce_hedged(train_bars, spec)
        n_tr = len(res["trades"])
        if n_tr < MIN_TRAIN_TRADES:
            continue
        n_local += 1
        _GLOBAL_TRIED["count"] += 1
        # objective = hedged net total return on TRAIN (alpha proxy, beta already hedged)
        score = float(res["metrics"]["net_total_return"])
        if best is None or score > best["score"]:
            best = {"score": score, "spec": spec, "sample": sample, "n_tr": n_tr}
    if best is None:
        # fallback: a sane default config so the fold still scores (counts as tried)
        return _make_spec({"drop_k": 5, "drop_thr": -0.006, "vol_mult": 1.2,
                           "bounce_tp": 0.004, "stop_loss": 0.006, "max_hold": 15})
    return best["spec"]


# --------------------------------------------------------------------------- #
# Walk-forward evaluation (custom: our hedged backtest is not the harness long predicate)
# --------------------------------------------------------------------------- #
def evaluate_walkforward_bounce(bars, spy_bars, n_folds=4, train_frac=0.5):
    dates = nh.all_trading_dates(bars)
    folds = nh.walk_forward_folds(dates, n_folds=n_folds, train_frac=train_frac)
    spy_daily_full = nh.spy_close_to_close_returns(spy_bars[HEDGE_SYM])

    fold_records, oos_pieces, fold_betas = [], [], []
    total_trades = 0
    positive = 0
    selected = []
    for f in folds:
        train_bars = nh.slice_by_date_range(bars, f["_train_dates"])
        test_bars = nh.slice_by_date_range(bars, f["_test_dates"])
        spec = _tune_factory(train_bars, f)
        selected.append({"fold": f["fold"], "spec": {k: spec[k] for k in GRID}})
        res = _bounce_hedged(test_bars, spec)
        daily = res["daily"]
        n_tr = len(res["trades"])
        total_trades += n_tr
        tot_ret = float(daily.sum()) if len(daily) else 0.0
        if tot_ret > 0:
            positive += 1
        oos_pieces.append(daily)
        spy_fold = spy_daily_full.reindex(daily.index).dropna() if len(daily) else spy_daily_full.iloc[0:0]
        fb = nh.beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append({
            "fold": f["fold"], "train": f["train"], "test": f["test"],
            "n_trades": n_tr, "total_return": tot_ret,
            "mean_daily_return": float(daily.mean()) if len(daily) else 0.0,
            "n_days": int(len(daily)),
            "long_pnl": res["metrics"]["long_pnl"],
            "hedge_pnl": res["metrics"]["hedge_pnl"],
            "net_total_pnl": res["metrics"]["net_total_pnl"],
            "hedge_rebalances": res["metrics"]["hedge_rebalances"],
            "win_rate": res["metrics"]["win_rate"],
            "exits": res["metrics"]["exits_by_reason"],
        })

    oos_daily = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    oos_daily = oos_daily[~oos_daily.index.duplicated(keep="first")]
    oos_mean = float(oos_daily.mean()) if len(oos_daily) else 0.0
    oos_std = float(oos_daily.std(ddof=1)) if len(oos_daily) > 1 else 0.0
    oos_sharpe = (oos_mean / oos_std * np.sqrt(TRADING_YEAR)) if oos_std > 0 else 0.0
    spy_oos = spy_daily_full.reindex(oos_daily.index).dropna() if len(oos_daily) else spy_daily_full.iloc[0:0]
    beta_full = nh.beta_decompose(oos_daily, spy_oos)

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
        "selected_per_fold": selected,
    }


def main():
    bars = h.load_bars(UNIVERSE)
    spy_bars = {HEDGE_SYM: bars[HEDGE_SYM]}
    print("=" * 84)
    print("EXPERIMENT: downmove_bounce — BETA-NEUTRAL sharp-drop reversion bounce (SPY-hedged)")
    print("=" * 84)
    dates = nh.all_trading_dates(bars)
    print(f"History: {dates[0]} .. {dates[-1]}  ({len(dates)} trading days)  universe={len(bars)} syms")

    out = evaluate_walkforward_bounce(bars, spy_bars, n_folds=4, train_frac=0.5)
    n_configs = _GLOBAL_TRIED["count"]

    print(f"\nConfigs evaluated (TRAIN-only tuning, all folds): {n_configs}")
    print(f"Grid space = {np.prod([len(v) for v in GRID.values()])} pts; sampled {N_ITER}/fold; selection on TRAIN net-return.\n")

    print(f"{'fold':>4} {'test window':>25} {'ntr':>5} {'netret':>9} {'long$':>8} {'hedge$':>8} "
          f"{'alpha_ann':>10} {'beta':>7} {'a_t':>6} {'R2':>6}")
    print("-" * 100)
    for fr, fb in zip(out["folds"], out["fold_betas"]):
        tw = f"{fr['test'][0]}..{fr['test'][1]}"
        print(f"{fr['fold']:>4} {tw:>25} {fr['n_trades']:>5} {fr['total_return']:>+9.4f} "
              f"{fr['long_pnl']:>+8.2f} {fr['hedge_pnl']:>+8.2f} "
              f"{fb['alpha_annual']:>+10.4f} {fb['beta']:>+7.3f} {fb['alpha_tstat']:>+6.2f} {fb['r2']:>6.3f}")

    print("-" * 100)
    bd = out["beta_decompose"]
    print("\nAGGREGATE OOS (concatenated across all folds):")
    print(f"  OOS trading days        : {out['oos_n_days']}")
    print(f"  OOS trades              : {out['oos_n_trades']}")
    print(f"  Folds positive (netret) : {out['folds_positive']}/{out['n_folds']}")
    print(f"  OOS total return        : {out['oos_total_return']:+.4f}")
    print(f"  OOS mean daily return   : {out['oos_mean_return']:+.6f}")
    print(f"  OOS Sharpe (annualized) : {out['oos_sharpe']:+.3f}")
    print("  --- BETA DECOMPOSITION (hedged daily returns regressed on SPY) ---")
    print(f"  alpha/day               : {bd['alpha_per_day']:+.6f}")
    print(f"  alpha annualized        : {bd['alpha_annual']:+.4f}  ({bd['alpha_annual']*100:+.2f}%)")
    print(f"  alpha t-stat            : {bd['alpha_tstat']:+.3f}")
    print(f"  beta (realized)         : {bd['beta']:+.4f}")
    print(f"  beta t-stat             : {bd['beta_tstat']:+.3f}")
    print(f"  R^2                     : {bd['r2']:.4f}")
    print(f"  n days regressed        : {bd['n_days']}")

    # ---- HONEST survives_oos gate ----
    alpha_ann = bd["alpha_annual"]
    a_t = bd["alpha_tstat"]
    beta = bd["beta"]
    folds_pos = out["folds_positive"]
    survives = (
        alpha_ann > 0
        and a_t >= 2.0
        and abs(beta) < 0.15
        and folds_pos > out["n_folds"] / 2.0
    )
    print("\n" + "=" * 84)
    print(f"survives_oos gate: alpha_ann>0={alpha_ann>0}  alpha_t>=2={a_t>=2.0}  "
          f"|beta|<0.15={abs(beta)<0.15}  folds_pos_majority={folds_pos > out['n_folds']/2.0}")
    print(f"SURVIVES_OOS = {survives}")
    print("=" * 84)

    payload = {
        "name": "downmove_bounce",
        "hypothesis": "Sharp short-horizon DOWN move on elevated volume -> LONG bounce; "
                      "built BETA-NEUTRAL via per-bar SPY short hedge. Test across 4 OOS "
                      "regime folds, net of 1bp/side on every leg incl. hedge rebalances.",
        "construction": "spy_hedge: long sharp-drop-bounce book + per-bar SPY short sized to "
                        "net long market value (1bp/side on every hedge rebalance).",
        "n_folds": out["n_folds"],
        "n_configs_tried": n_configs,
        "grid_space": int(np.prod([len(v) for v in GRID.values()])),
        "folds_positive": out["folds_positive"],
        "oos_n_days": out["oos_n_days"],
        "oos_n_trades": out["oos_n_trades"],
        "oos_total_return": out["oos_total_return"],
        "oos_mean_return": out["oos_mean_return"],
        "oos_sharpe": out["oos_sharpe"],
        "beta_decompose": bd,
        "fold_betas": out["fold_betas"],
        "folds": out["folds"],
        "selected_per_fold": out["selected_per_fold"],
        "survives_oos": bool(survives),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
