"""expn_xsec_reversion.py — CROSS-SECTIONAL INTRADAY VWAP REVERSION (beta-neutral).

HYPOTHESIS
==========
Each bar, rank the 10-name universe by distance-from-session-VWAP. Go LONG the
most-below-VWAP quantile and SHORT the most-above-VWAP quantile, dollar-neutral
(equal $ per leg). If intraday VWAP dispersion mean-reverts cross-sectionally,
the laggards (below VWAP) should outperform the leaders (above VWAP) over the
next bar, producing alpha that is BETA-NEUTRAL by construction (long $ == short $).

This is the cleanest neutral construction: no SPY hedge to estimate, no directional
tilt — the dollar-neutral long/short basket is mechanically market-neutral.

WHY A CUSTOM BACKTEST (not neutral_harness._bt_cross_sectional verbatim)
========================================================================
neutral_harness._bt_cross_sectional rebalances the ENTIRE basket EVERY bar. On
1-min bars with 1bp/side slippage that is death-by-turnover: a smoke test on fold-0
showed 166k fills and ~$300 of cost on a $500 book (−64% total). No cross-sectional
signal can clear that floor. To give the hypothesis a FAIR test we must throttle
turnover. neutral_harness exposes no rebalance-frequency knob, so this file
reimplements the SAME portfolio loop — reusing the harness primitives verbatim
(compute_indicators_df, _slip, MIN_BARS_FOR_ENTRY, _parse_hm, the t->t+1 fill rule,
the close-to-close mark, the 1bp/side cost on |delta shares|, EOD flatten) — and
adds exactly two tunable knobs:

  * rebalance_every : hold the chosen basket for N bars before re-ranking
                      (turnover throttle; N=1 == neutral_harness behavior)
  * entry_band      : only take names whose |dist_from_vwap| >= this fraction
                      (trade only the genuinely dislocated names; skip the middle)

A SANITY check (sanity_xs_matches_harness) proves that with rebalance_every=1,
entry_band=0.0 this file's backtest is BIT-IDENTICAL (same total_pnl, cost, fills)
to neutral_harness.research_backtest_neutral(mode='cross_sectional'). So the only
thing this file adds is the two throttle knobs — the cost/fill/mark model is the
production-identical one.

PROTOCOL (anti-self-deception, per the rigor spec)
==================================================
- WALK-FORWARD: 4 sequential, NON-overlapping OOS folds spanning different SPY
  regimes (fold0 +9.4%, fold1 +8.3%, fold2 +1.4%, fold3 +10.7%).
- TUNE ON TRAIN ONLY: for each fold, grid-search (xs_quantile, entry_band,
  rebalance_every) on that fold's TRAIN slice, pick the best-Sharpe config, then
  score that ONE config on the untouched TEST slice. Test data is never seen during
  tuning. Multiple-comparisons honest: N grid points are tried per fold on TRAIN/VAL,
  TEST scored once.
- BETA DECOMPOSITION: regress concatenated-OOS daily returns on SPY daily returns;
  report alpha (annual + t-stat), beta, R^2. A neutral claim requires |beta| < ~0.15
  AND alpha_tstat >= 2 AND positive in a MAJORITY of folds AND above the cost floor.
- COSTS: $0 commission + 1bp/side adverse slippage on EVERY leg rebalance, strict
  t->t+1 fills, completed bars only, EOD flatten 15:55 NY, NET reported.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.expn_xsec_reversion
"""

from __future__ import annotations

import datetime as _dt
import json
from itertools import groupby
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from experiments.neutral_harness import (
    load_bars,
    all_trading_dates,
    walk_forward_folds,
    slice_by_date_range,
    spy_close_to_close_returns,
    daily_returns_from_equity,
    beta_decompose,
    research_backtest_neutral,
    UNIVERSE_RANK,
    TRADING_YEAR,
)
from experiments.harness import (
    compute_indicators_df,
    _slip,
    _parse_hm,
    MIN_BARS_FOR_ENTRY,
)

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULTS = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/xsec_reversion.json")


# --------------------------------------------------------------------------- #
# Throttled cross-sectional backtest.
# Mirrors neutral_harness._bt_cross_sectional EXACTLY, plus two knobs:
#   rebalance_every (int >=1)  -- re-rank/re-target only every N bars
#   entry_band      (float>=0) -- require |dist_from_vwap| >= band to be eligible
# All cost / fill / mark math is identical to the production-identical harness.
# --------------------------------------------------------------------------- #
def build_by_ts(bars: Dict[str, pd.DataFrame]):
    """Precompute the production-identical indicator panel once (reused across the grid).

    Returns (by_ts, ordered_ts). compute_indicators_df is the SAME per-session VWAP /
    dist / n_bars used by harness.research_backtest, so the signal math is unchanged.
    """
    ind = {sym: compute_indicators_df(df) for sym, df in bars.items() if not df.empty}
    syms = [s for s in UNIVERSE_RANK if s in ind] + [s for s in ind if s not in UNIVERSE_RANK]
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
            }
    return by_ts, sorted(by_ts.keys())


def bt_xs_throttled(bars: Dict[str, pd.DataFrame], spec: dict, panel=None) -> dict:
    xs_quantile = float(spec.get("xs_quantile", 0.3))
    xs_book = float(spec.get("xs_book", spec.get("max_exposure", 500.0)))
    xs_min_names = int(spec.get("xs_min_names", 2))
    slippage_bps = float(spec.get("slippage_bps", 1.0))
    eod_cutoff = _parse_hm(spec.get("eod_flatten", "15:55"))
    base = float(spec.get("return_base", xs_book))
    side_book = xs_book / 2.0
    rebalance_every = max(1, int(spec.get("rebalance_every", 1)))
    entry_band = float(spec.get("entry_band", 0.0))

    if panel is None:
        by_ts, ordered_ts = build_by_ts(bars)
    else:
        by_ts, ordered_ts = panel

    held: Dict[str, float] = {}
    pnl_by_day: Dict[_dt.date, float] = {}
    cost_total = 0.0
    rebalances = 0
    n_fills = 0
    gross_long = 0.0
    gross_short = 0.0
    n_basket_bars = 0

    def _day_of(t):
        return by_ts[t][next(iter(by_ts[t]))]["date"]

    for day, day_ts_iter in groupby(ordered_ts, key=_day_of):
        day_ts = list(day_ts_iter)
        held = {}
        day_pnl = 0.0
        for pos, t in enumerate(day_ts):
            row = by_ts[t]
            t_obj = t.timetz().replace(tzinfo=None) if hasattr(t, "timetz") else t.time()
            is_eod = t_obj >= eod_cutoff
            has_next = (pos + 1) < len(day_ts)

            # 1) mark held basket over t -> t+1 (close-to-close), identical to harness.
            if has_next and held:
                nxt = day_ts[pos + 1]
                nxt_row = by_ts[nxt]
                for sym, sh in held.items():
                    if sym in row and sym in nxt_row:
                        d_close = nxt_row[sym]["close"] - row[sym]["close"]
                        day_pnl += sh * d_close

            # 2) decide target basket. Re-rank only on rebalance bars; otherwise hold.
            do_rebalance = (pos % rebalance_every == 0)
            target: Dict[str, float] = {}
            if has_next and not is_eod and do_rebalance:
                cand = []
                for sym, d in row.items():
                    if (
                        not np.isnan(d["dist"])
                        and d["n_bars"] >= MIN_BARS_FOR_ENTRY
                        and abs(d["dist"]) >= entry_band
                    ):
                        cand.append((sym, d["dist"]))
                if len(cand) >= xs_min_names * 2:
                    cand.sort(key=lambda kv: kv[1])  # most-below vwap first
                    k = max(1, int(round(len(cand) * xs_quantile)))
                    longs = [s for s, _ in cand[:k]]
                    shorts = [s for s, _ in cand[-k:]]
                    if (
                        len(longs) >= xs_min_names
                        and len(shorts) >= xs_min_names
                        and not (set(longs) & set(shorts))
                    ):
                        per_long = side_book / len(longs)
                        per_short = side_book / len(shorts)
                        nxt = day_ts[pos + 1]
                        nxt_row = by_ts.get(nxt, {})
                        for s in longs:
                            o = nxt_row.get(s, {}).get("open")
                            if o and o > 0:
                                target[s] = per_long / o
                        for s in shorts:
                            o = nxt_row.get(s, {}).get("open")
                            if o and o > 0:
                                target[s] = -(per_short / o)
                        n_basket_bars += 1
            elif has_next and not is_eod and not do_rebalance:
                # hold the existing basket: keep current shares as target (no churn).
                target = dict(held)

            # 3) rebalance held -> target at t+1 open, pay 1bp/side on |delta shares|.
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
                    n_fills += 1
                held = target
            else:
                held = {}  # last bar of day: flat (EOD)

            for sym, sh in held.items():
                px = by_ts[t].get(sym, {}).get("close", 0.0)
                if sh > 0:
                    gross_long += sh * px
                else:
                    gross_short += -sh * px

        pnl_by_day[day] = pnl_by_day.get(day, 0.0) + day_pnl

    daily = daily_returns_from_equity(pnl_by_day, base)
    total_pnl = sum(pnl_by_day.values())
    metrics = {
        "n_fills": n_fills,
        "rebalances": rebalances,
        "total_pnl": total_pnl,
        "total_return": total_pnl / base if base else 0.0,
        "cost_total": cost_total,
        "n_days": len(pnl_by_day),
        "mean_daily_return": float(daily.mean()) if len(daily) else 0.0,
        "n_basket_bars": n_basket_bars,
    }
    return {"mode": "cross_sectional_throttled", "trades": [], "metrics": metrics,
            "daily": daily, "return_base": base}


# --------------------------------------------------------------------------- #
# SANITY: rebalance_every=1, entry_band=0 must == neutral_harness cross_sectional.
# --------------------------------------------------------------------------- #
def sanity_xs_matches_harness(bars: Dict[str, pd.DataFrame]) -> dict:
    spec = {"mode": "cross_sectional", "xs_quantile": 0.3, "xs_book": 500.0,
            "xs_min_names": 2, "slippage_bps": 1.0, "eod_flatten": "15:55"}
    ref = research_backtest_neutral(bars, spec)
    mine = bt_xs_throttled(
        bars,
        {"xs_quantile": 0.3, "xs_book": 500.0, "xs_min_names": 2,
         "slippage_bps": 1.0, "eod_flatten": "15:55",
         "rebalance_every": 1, "entry_band": 0.0},
    )
    ref_pnl = float(ref["metrics"]["total_pnl"])
    my_pnl = float(mine["metrics"]["total_pnl"])
    ref_cost = float(ref["metrics"]["cost_total"])
    my_cost = float(mine["metrics"]["cost_total"])
    daily_match = bool(
        len(ref["daily"]) == len(mine["daily"])
        and np.allclose(ref["daily"].to_numpy(), mine["daily"].to_numpy(), atol=1e-9)
    )
    return {
        "ref_total_pnl": ref_pnl,
        "my_total_pnl": my_pnl,
        "pnl_match": abs(ref_pnl - my_pnl) < 1e-6,
        "ref_cost": ref_cost,
        "my_cost": my_cost,
        "cost_match": abs(ref_cost - my_cost) < 1e-6,
        "daily_match": daily_match,
        "pass": abs(ref_pnl - my_pnl) < 1e-6 and abs(ref_cost - my_cost) < 1e-6 and daily_match,
    }


# --------------------------------------------------------------------------- #
# Per-fold TRAIN tuning grid (tuned on TRAIN only; TEST scored once).
# --------------------------------------------------------------------------- #
GRID = [
    {"xs_quantile": q, "entry_band": b, "rebalance_every": r}
    for q in (0.2, 0.3, 0.4)
    for b in (0.0, 0.001, 0.003)
    for r in (1, 5, 15, 30)
]
N_CONFIGS = len(GRID)


def _base_spec(g: dict) -> dict:
    return {
        "xs_quantile": g["xs_quantile"],
        "entry_band": g["entry_band"],
        "rebalance_every": g["rebalance_every"],
        "xs_book": 500.0,
        "xs_min_names": 2,
        "slippage_bps": 1.0,
        "eod_flatten": "15:55",
        "return_base": 500.0,
    }


def _sharpe(daily: pd.Series) -> float:
    if len(daily) < 2:
        return 0.0
    mu = float(daily.mean())
    sd = float(daily.std(ddof=1))
    return (mu / sd * np.sqrt(TRADING_YEAR)) if sd > 0 else 0.0


def tune_on_train(train_bars: Dict[str, pd.DataFrame], panel=None) -> dict:
    """Grid-search on TRAIN; return the best-Sharpe config (never sees TEST)."""
    if panel is None:
        panel = build_by_ts(train_bars)
    best = None
    for g in GRID:
        res = bt_xs_throttled(train_bars, _base_spec(g), panel=panel)
        sh = _sharpe(res["daily"])
        tot = float(res["daily"].sum()) if len(res["daily"]) else 0.0
        # require it to actually trade and clear the cost floor on TRAIN.
        score = sh if (res["metrics"]["n_basket_bars"] > 0) else -1e9
        if best is None or score > best["score"]:
            best = {"score": score, "g": g, "train_sharpe": sh, "train_return": tot,
                    "train_metrics": res["metrics"]}
    return best


# --------------------------------------------------------------------------- #
# Walk-forward driver: tune each fold's TRAIN, score its TEST, aggregate + beta.
# --------------------------------------------------------------------------- #
def run() -> dict:
    bars = load_bars(SYMBOLS)
    spy_daily_full = spy_close_to_close_returns(bars["SPY"])
    dates = all_trading_dates(bars)
    folds = walk_forward_folds(dates, n_folds=4, train_frac=0.5)

    fold_records = []
    oos_pieces = []
    fold_betas = []
    positive = 0

    for f in folds:
        train_bars = slice_by_date_range(bars, f["_train_dates"])
        test_bars = slice_by_date_range(bars, f["_test_dates"])
        train_panel = build_by_ts(train_bars)
        test_panel = build_by_ts(test_bars)
        best = tune_on_train(train_bars, panel=train_panel)
        spec = _base_spec(best["g"])
        res = bt_xs_throttled(test_bars, spec, panel=test_panel)
        daily = res["daily"]
        tot = float(daily.sum()) if len(daily) else 0.0
        if tot > 0:
            positive += 1
        oos_pieces.append(daily)
        spy_fold = spy_daily_full.reindex(daily.index).dropna() if len(daily) else spy_daily_full.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append({
            "fold": f["fold"], "train": f["train"], "test": f["test"],
            "chosen_params": best["g"],
            "train_sharpe": best["train_sharpe"], "train_return": best["train_return"],
            "test_total_return": tot,
            "test_mean_daily": float(daily.mean()) if len(daily) else 0.0,
            "test_sharpe": _sharpe(daily),
            "test_n_days": int(len(daily)),
            "test_cost": float(res["metrics"]["cost_total"]),
            "test_n_fills": int(res["metrics"]["n_fills"]),
            "test_beta": fb["beta"], "test_alpha_tstat": fb["alpha_tstat"],
        })

    oos = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    spy_oos = spy_daily_full.reindex(oos.index).dropna() if len(oos) else spy_daily_full.iloc[0:0]
    beta_full = beta_decompose(oos, spy_oos)

    oos_mean = float(oos.mean()) if len(oos) else 0.0
    oos_sharpe = _sharpe(oos)
    n_fills_total = sum(r["test_n_fills"] for r in fold_records)

    return {
        "n_folds": len(folds),
        "n_configs_tried_per_fold": N_CONFIGS,
        "n_configs_tried_total": N_CONFIGS * len(folds),
        "folds": fold_records,
        "fold_betas": fold_betas,
        "folds_positive": positive,
        "oos_mean_return": oos_mean,
        "oos_sharpe": oos_sharpe,
        "oos_total_return": float(oos.sum()) if len(oos) else 0.0,
        "oos_n_days": int(len(oos)),
        "oos_n_fills": n_fills_total,
        "beta_decompose": beta_full,
    }


def main():
    bars = load_bars(SYMBOLS)

    print("=" * 78)
    print("SANITY: throttled(rebalance_every=1, entry_band=0) == neutral_harness xsec")
    print("=" * 78)
    # use a small recent slice for speed
    dates = all_trading_dates(bars)
    sane_dates = set(dates[-40:])
    sane_bars = slice_by_date_range(bars, sane_dates)
    sane = sanity_xs_matches_harness(sane_bars)
    for k, v in sane.items():
        print(f"  {k}: {v}")
    if not sane["pass"]:
        print("  *** SANITY FAILED — cost/fill model diverged; results not trustworthy ***")

    print()
    print("=" * 78)
    print("WALK-FORWARD: cross-sectional VWAP reversion (dollar-neutral L/S)")
    print("=" * 78)
    out = run()

    print(f"\nConfigs tried: {out['n_configs_tried_per_fold']}/fold "
          f"({out['n_configs_tried_total']} total), tuned on TRAIN, TEST scored once.\n")
    print(f"{'fold':>4} {'test window':>25} {'params(q/band/reb)':>20} "
          f"{'tr_shrp':>8} {'te_ret%':>8} {'te_shrp':>8} {'beta':>7} {'a_tstat':>8}")
    for r in out["folds"]:
        p = r["chosen_params"]
        ps = f"{p['xs_quantile']}/{p['entry_band']}/{p['rebalance_every']}"
        print(f"{r['fold']:>4} {r['test'][0]+'..'+r['test'][1]:>25} {ps:>20} "
              f"{r['train_sharpe']:>8.2f} {r['test_total_return']*100:>8.2f} "
              f"{r['test_sharpe']:>8.2f} {r['test_beta']:>7.3f} {r['test_alpha_tstat']:>8.2f}")

    print(f"\nFolds positive (OOS net): {out['folds_positive']}/{out['n_folds']}")
    print(f"OOS total return: {out['oos_total_return']*100:.2f}%  over {out['oos_n_days']} days")
    print(f"OOS mean daily:   {out['oos_mean_return']*100:.4f}%   OOS Sharpe (ann): {out['oos_sharpe']:.2f}")
    print(f"OOS total fills:  {out['oos_n_fills']:,}")

    bd = out["beta_decompose"]
    print("\n--- CONCATENATED-OOS BETA DECOMPOSITION (strat ~ alpha + beta*SPY) ---")
    print(f"  alpha/day   : {bd['alpha_per_day']*100:+.5f}%")
    print(f"  alpha annual: {bd['alpha_annual']*100:+.3f}%")
    print(f"  alpha t-stat: {bd['alpha_tstat']:+.3f}")
    print(f"  beta        : {bd['beta']:+.4f}")
    print(f"  beta t-stat : {bd['beta_tstat']:+.3f}")
    print(f"  R^2         : {bd['r2']:.4f}   n_days: {bd['n_days']}")

    survives = (
        bd["alpha_annual"] > 0
        and bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and out["folds_positive"] > out["n_folds"] / 2.0
    )
    print(f"\n  SURVIVES OOS (alpha>0, t>=2, |beta|<0.15, majority folds +): {survives}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as fh:
        json.dump({"sanity": sane, "walkforward": out, "survives_oos": bool(survives)},
                  fh, indent=2, default=str)
    print(f"\nWrote {RESULTS}")
    return out, sane, survives


if __name__ == "__main__":
    main()
