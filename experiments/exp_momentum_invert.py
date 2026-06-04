"""exp_momentum_invert.py — INVERT the reversion thesis: trade TREND/MOMENTUM, not reversion.

HYPOTHESIS
----------
The validated production strategy is VWAP MEAN-REVERSION: it buys when price is
`entry_dist` BELOW VWAP and exits when price reverts toward VWAP. That strategy is
net-negative on this thin IEX slice. This experiment inverts the thesis:

    Go LONG when price breaks OUT >= entry_dist ABOVE VWAP on HIGH volume (a momentum
    / breakout signal), and ride the continuation: exit only on a FADE back toward
    VWAP (trailing-style), a hard stop, a take-profit, max-hold, or EOD flatten.

If TREND beats REVERSION here, this breakout-long should be net-positive OOS while the
reversion base is not.

WHY A LOCAL BACKTESTER (and how it stays honest)
------------------------------------------------
`harness.research_backtest` HARD-CODES direction from VWAP distance: LONG only fires
BELOW VWAP and its `vwap_revert` exit is a reversion take-profit. It therefore cannot
express "LONG when ABOVE VWAP, ride the breakout". So this file implements a small
event-driven backtester for the INVERTED entry, but reuses the harness's *validated*
primitives so the simulator logic is identical in spirit to the sanity-checked engine:

  - h.compute_indicators_df  (same session VWAP, dist_from_vwap, volume_ratio, n_bars)
  - h.chronological_split     (same global by-DATE TRAIN/VAL/TEST cutoffs)
  - h._slip                   (same 1bp adverse slippage per side: buy*(1+1e-4), sell*(1-1e-4))
  - h._metrics                (same metric definitions: pnl, win_rate, sharpe_like, max_dd...)
  - strict t->t+1 fills, EOD flatten, per-symbol cooldown, $100 notional, 4-pos / $500 caps

ANTI-SELF-DECEPTION
-------------------
- TRAIN (earliest ~60%) only for tuning, VALIDATION (~20%) only for SELECTION, TEST
  (latest ~20%) evaluated EXACTLY ONCE for the single selected config.
- Splits are by DATE, never interleaved (delegated to h.chronological_split).
- All PnL is NET of costs. We count every config tried (multiple-comparisons honesty).

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.exp_momentum_invert
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments import harness as h

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/momentum_invert.json")


# --------------------------------------------------------------------------- #
# Inverted (momentum / breakout) backtester
# --------------------------------------------------------------------------- #
def momentum_backtest(bars: Dict[str, pd.DataFrame], spec: dict) -> dict:
    """Event-driven t->t+1 backtester for the INVERTED (breakout-long) thesis.

    ENTRY (long only): on completed bar t, enter if
        dist_from_vwap >= +entry_dist   (price broke OUT above VWAP)
        AND volume_ratio >= vol_mult     (high volume)
        AND (optional) recent_return >= mom_min   (require up-momentum confirmation)
        AND n_bars >= 21, time-of-day < no_entry_after, t+1 bar exists, cooldown clear,
        position/exposure caps OK.
    Fill at bar t+1 OPEN, buy-side adverse slippage.

    EXIT (priority): EOD > take_profit > trailing_stop > vwap_fade > max_hold > stop_loss
      - vwap_fade: price falls back to within `fade_band` ABOVE VWAP (momentum exhausted),
        i.e. exit when dist_from_vwap <= fade_band. This is the momentum analogue of the
        reversion strategy's vwap_revert, but it lets the breakout RUN (fade_band can be
        small/negative to ride further, or large to bail fast).
      - take_profit / trailing_stop / max_hold / stop_loss as usual.
    Fill at bar t+1 OPEN, sell-side adverse slippage (EOD/forced at decision-bar open if no t+1).

    Mirrors harness mechanics (caps, cooldown, EOD, t->t+1, slippage, metrics).
    """
    entry_dist = float(spec.get("entry_dist", 0.003))
    vol_mult = float(spec.get("vol_mult", 1.5))
    mom_min = spec.get("mom_min", None)  # require recent_return >= this (None = off)
    fade_band = float(spec.get("fade_band", 0.0))  # exit when dist <= fade_band
    max_hold = float(spec.get("max_hold", 30))
    stop_loss = spec.get("stop_loss", 0.006)
    take_profit = spec.get("take_profit", None)
    trailing_stop = spec.get("trailing_stop", None)
    notional = float(spec.get("notional", 100.0))
    max_positions = int(spec.get("max_positions", 4))
    max_exposure = float(spec.get("max_exposure", 500.0))
    cooldown_min = float(spec.get("cooldown_min", 10.0))
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    eod_cutoff = h._parse_hm(spec.get("eod_flatten", "15:55"))
    no_entry_after_min = int(spec.get("no_entry_after_min", 355))  # minute_of_session

    ind = {sym: h.compute_indicators_df(df) for sym, df in bars.items() if not df.empty}

    def rank(sym):
        try:
            return (UNIVERSE.index(sym), sym)
        except ValueError:
            return (len(UNIVERSE), sym)

    # Precompute per (sym,day) arrays + a global timestamp index (same shape as harness).
    sym_day_data: Dict = {}
    ts_index: Dict = {}
    for sym, df in ind.items():
        for day, day_df in df.groupby(df.index.date, sort=True):
            arr = {
                "ts": day_df.index.to_pydatetime(),
                "open": day_df["open"].to_numpy(dtype=float),
                "close": day_df["close"].to_numpy(dtype=float),
                "dist": day_df["dist_from_vwap"].to_numpy(dtype=float),
                "vol_ratio": day_df["volume_ratio"].to_numpy(dtype=float),
                "n_bars": day_df["n_bars"].to_numpy(dtype=int),
                "minute": day_df["minute"].to_numpy(dtype=int),
                "recent_return": day_df["recent_return"].to_numpy(dtype=float),
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
            dist = arr["dist"][t]
            pos["peak"] = max(pos.get("peak", pos["entry_price"]), last_price)
            unreal_plpc = (last_price - pos["entry_price"]) / pos["entry_price"]
            holding_min = (asof - pos["entry_time"]).total_seconds() / 60.0

            exit_reason = None
            if is_eod:
                exit_reason = "eod_flatten"
            elif take_profit is not None and unreal_plpc >= float(take_profit):
                exit_reason = "take_profit"
            elif trailing_stop is not None:
                draw = (pos["peak"] - last_price) / pos["peak"] if pos["peak"] else 0.0
                if draw >= float(trailing_stop):
                    exit_reason = "trailing_stop"
            if exit_reason is None and not np.isnan(dist) and dist <= fade_band:
                exit_reason = "vwap_fade"
            if exit_reason is None and holding_min > max_hold:
                exit_reason = "max_hold"
            if exit_reason is None and stop_loss is not None and unreal_plpc <= -float(stop_loss):
                exit_reason = "stop_loss"
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
                dist = arr["dist"][t]
                vr = arr["vol_ratio"][t]
                if np.isnan(dist) or np.isnan(vr):
                    continue
                if not (dist >= entry_dist and vr >= vol_mult):  # INVERTED: breakout ABOVE vwap
                    continue
                if mom_min is not None:
                    rr = arr["recent_return"][t]
                    if np.isnan(rr) or rr < float(mom_min):
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

        open_pnl = 0.0
        for sym, pos in open_positions.items():
            db = group.get(sym)
            if db is None:
                continue
            day, t = db
            mark = sym_day_data[(sym, day)]["close"][t]
            open_pnl += (mark - pos["entry_price"]) * pos["qty"]
        equity_points.append(realized_pnl + open_pnl)

    # force-close stragglers (recorded at 0 pnl so nothing is silently dropped)
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
# Search: tune on TRAIN, SELECT on VAL, never touch TEST
# --------------------------------------------------------------------------- #
def _slim(m: dict) -> dict:
    return {
        "n_trades": m["n_trades"], "win_rate": round(m["win_rate"], 4),
        "total_pnl": round(m["total_pnl"], 4), "total_return": round(m["total_return"], 6),
        "sharpe": round(m["sharpe_like"], 4), "max_dd": round(m["max_drawdown"], 4),
        "exits": m["exits_by_reason"],
    }


def search(train, val, grid, n_iter, seed, objective="total_pnl"):
    """Deterministic sampled search. Tune on TRAIN (sanity), rank on VAL. Never touches TEST."""
    rng = np.random.default_rng(seed)
    keys = list(grid.keys())
    seen = set()
    tried = []
    best = None
    attempts = 0
    while len(tried) < n_iter and attempts < n_iter * 40:
        attempts += 1
        sample = {k: grid[k][int(rng.integers(0, len(grid[k])))] for k in keys}
        sig = tuple((k, sample[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        spec = dict(sample)
        tm = momentum_backtest(train, spec)["metrics"]
        vm = momentum_backtest(val, spec)["metrics"]
        # require a tradeable config on TRAIN (>= 20 trades) so we don't select on noise
        if tm["n_trades"] < 20:
            continue
        rec = {"params": sample, "train": _slim(tm), "val": _slim(vm)}
        tried.append(rec)
        score = vm.get(objective, float("-inf"))
        if best is None or score > best["score"]:
            best = {"score": score, "rec": rec, "spec": spec}
    return best, tried


def main():
    bars = h.load_bars(UNIVERSE)
    train, val, test, ranges = h.chronological_split(bars)
    print("=" * 78)
    print("EXPERIMENT: momentum_invert — LONG breakout ABOVE VWAP (trend, not reversion)")
    print("=" * 78)
    print(f"Split dates: TRAIN {ranges['train']}  VAL {ranges['val']}  TEST {ranges['test']}")

    # Grid for the inverted breakout-long. fade_band: small/negative rides further;
    # mom_min: optional up-momentum confirmation. These are the momentum levers.
    grid = {
        "entry_dist": [0.002, 0.003, 0.004, 0.005],
        "vol_mult": [1.2, 1.5, 2.0],
        "fade_band": [-0.002, 0.0, 0.001, 0.002],   # exit when dist falls back to <= this
        "max_hold": [15, 30, 45, 60],
        "stop_loss": [0.004, 0.006, 0.008],
        "take_profit": [None, 0.006, 0.010],
        "trailing_stop": [None, 0.004, 0.006],
        "mom_min": [None, 0.0, 0.001],
    }
    N_ITER = 40
    SEED = 7
    best, tried = search(train, val, grid, n_iter=N_ITER, seed=SEED, objective="total_pnl")
    n_configs = len(tried)
    print(f"\nConfigs evaluated (TRAIN+VAL only): {n_configs}")

    if best is None:
        print("No tradeable config found.")
        return

    sel = best["rec"]["params"]
    print("\nSELECTED config (by VALIDATION total_pnl):")
    for k, v in sel.items():
        print(f"  {k:14s} = {v}")
    print(f"\n  TRAIN: {best['rec']['train']}")
    print(f"  VAL  : {best['rec']['val']}")

    # ---- TEST evaluated EXACTLY ONCE for the single selected config ----
    test_m = momentum_backtest(test, dict(sel))["metrics"]
    test_slim = _slim(test_m)
    print(f"  TEST : {test_slim}")

    tr_pnl = best["rec"]["train"]["total_pnl"]
    te_pnl = test_slim["total_pnl"]
    print("\n" + "-" * 78)
    print(f"Train->Test PnL: {tr_pnl:+.4f} -> {te_pnl:+.4f}   (TEST trades={test_slim['n_trades']})")

    per_trade = (te_pnl / test_slim["n_trades"]) if test_slim["n_trades"] else 0.0
    # HONEST gate: a real edge must (a) be net-positive on TEST with >=20 trades, AND
    # (b) have had genuine forward support on VAL (the selection split) -- not merely be
    # the least-bad of a losing landscape. A break-even/negative VAL means any TEST profit
    # is regime luck, not a selected edge. We require VAL to be clearly positive too.
    val_pnl = best["rec"]["val"]["total_pnl"]
    naive_pass = (te_pnl > 0) and (test_slim["n_trades"] >= 20)
    survives = naive_pass and (val_pnl > 0.5)  # VAL must show real (not break-even) support
    print(f"TEST net pnl/trade: {per_trade:+.5f}  (cost floor ~ a few bp on $100 notional)")
    print(f"VAL total_pnl: {val_pnl:+.4f}  (selection split must show genuine support)")
    print(f"naive gate (TEST+>=20 trades): {naive_pass}")
    print(f"survives_oos (HONEST: naive AND VAL clearly positive): {survives}")
    print("=" * 78)

    out = {
        "name": "momentum_invert",
        "hypothesis": "Data favors TREND not reversion: go LONG on breakout >= entry_dist "
                      "ABOVE VWAP with high volume, ride continuation, exit on fade/stop/TP/time.",
        "split_ranges": ranges,
        "n_configs_tried": n_configs,
        "selected_config": sel,
        "train": best["rec"]["train"],
        "val": best["rec"]["val"],
        "test": test_slim,
        "test_pnl_per_trade": per_trade,
        "naive_gate_pass": bool(naive_pass),
        "survives_oos": bool(survives),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    main()
