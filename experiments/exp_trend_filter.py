"""exp_trend_filter.py — Does a downtrend filter rescue VWAP dip-buying?

HYPOTHESIS
----------
The base long-only VWAP mean-reversion strategy fades dips (buys when price is
`entry_dist` below VWAP). The claim under test: it loses money mostly because it
"catches falling knives" — buying dips that occur DURING strong intraday
downtrends. If we refuse to buy the dip when the short-term trend is steeply
negative, we should remove the worst trades and (hopefully) turn the edge
positive net of costs.

We implement the trend filter THREE ways (all decision-time, no look-ahead),
gated through the harness `score_fn` so we never touch a future bar:

  1. recent_return floor  : reject longs when recent_return (close[t]/close[t-5]-1)
                            is below `rr_floor` (steep recent down-move = knife).
  2. distance cap         : reject longs when price is more than `dist_cap` below
                            VWAP (deep knives fall further).
  3. both                 : apply both gates.

We ALSO let the base params (entry_dist, vol_mult, max_hold, stop_loss,
vwap_exit_band) move, because a trend filter only matters relative to the base
predicate it gates.

PROTOCOL
--------
- Splits are chronological by DATE via harness.chronological_split
  (TRAIN ~60% = 2025-09-02..2026-02-12, VAL ~20% = 2026-02-13..2026-04-09,
   TEST ~20% = 2026-04-10..2026-06-03).
- We run one sampled search (harness.search_params) that tunes on TRAIN and
  RANKS on VALIDATION by total_pnl. The single val-best config is then evaluated
  EXACTLY ONCE on TEST.
- Costs: $0 commission + 1bp adverse slippage per side, NET, baked into the
  harness. Strict t->t+1 open fills.

This file is RESEARCH ONLY. It places ZERO orders.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.exp_trend_filter
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.harness import (
    chronological_split,
    evaluate_splits,
    load_bars,
    search_params,
)

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/trend_filter.json")
SEED = 7
N_ITER = 60  # sampled configs evaluated on TRAIN+VAL (TEST never touched here)


# --------------------------------------------------------------------------- #
# Trend filter as a decision-time score_fn (NO look-ahead — uses only `feat`,
# which the harness builds purely from bars [0..t]).
# --------------------------------------------------------------------------- #
def make_trend_score(rr_floor, dist_cap):
    """Return a score_fn that gates LONG dip-buys away from steep downtrends.

    rr_floor : float | None  -> reject long if recent_return < rr_floor
                                (recent_return = close[t]/close[t-5]-1, a 5-bar slope)
    dist_cap : float | None  -> reject long if dist_from_vwap < -dist_cap
                                (cap how deep below VWAP we will fade)
    Returns 1.0 to allow, -1.0 to reject. score_threshold=0.0 in the spec means
    only score >= 0 passes. Shorts (not used here, side='long') always pass.
    """
    def fn(feat):
        if feat.get("side") != "long":
            return 1.0
        if rr_floor is not None and feat["recent_return"] < rr_floor:
            return -1.0
        if dist_cap is not None and feat["dist_from_vwap"] < -dist_cap:
            return -1.0
        return 1.0

    return fn


def factory(p: dict) -> dict:
    """Build a spec from sampled params. score_fn carries the trend filter."""
    spec = {
        "side": "long",
        "entry_dist": p["entry_dist"],
        "vol_mult": p["vol_mult"],
        "max_hold": p["max_hold"],
        "vwap_exit_band": p["vwap_exit_band"],
        "stop_loss": p["stop_loss"],
        "score_threshold": 0.0,
        "score_fn": make_trend_score(p["rr_floor"], p["dist_cap"]),
    }
    return spec


PARAM_GRID = {
    "entry_dist": [0.004, 0.005, 0.007, 0.010],
    "vol_mult": [1.2, 1.5, 2.0],
    "max_hold": [10, 15, 30, 60],
    "vwap_exit_band": [0.0005, 0.001, 0.002],
    "stop_loss": [0.003, 0.005, 0.008],
    # trend filter knobs (None = that gate disabled)
    "rr_floor": [None, -0.002, -0.003, -0.005, -0.008],
    "dist_cap": [None, 0.006, 0.008, 0.010, 0.013],
}


def _fmt(tag, m):
    return (
        f"{tag:6s} n={m['n_trades']:5d} pnl={m['total_pnl']:+9.3f} "
        f"ret={m['total_return']:+.4f} wr={m['win_rate']:.3f} "
        f"sharpe={m['sharpe_like']:+.2f} dd={m['max_drawdown']:+.2f}"
    )


def main():
    bars = load_bars(SYMBOLS)
    tr, va, te, ranges = chronological_split(bars)
    print("SPLIT DATE RANGES (train / val / test):")
    for k in ("train", "val", "test"):
        lo, hi, n = ranges[k]
        print(f"  {k:6s} {lo} .. {hi}  ({n} trading days)")

    # --- Reference: base strategy with NO trend filter (default params) ------
    base_spec = {
        "side": "long",
        "entry_dist": 0.005,
        "vol_mult": 1.2,
        "max_hold": 15,
        "vwap_exit_band": 0.001,
        "stop_loss": 0.005,
    }
    base_res = evaluate_splits(base_spec, tr, va, te)
    print("\n=== BASELINE (no trend filter, default params) ===")
    for sp in ("train", "val", "test"):
        print("  " + _fmt(sp, base_res[sp]))

    # --- Protocol search: tune on TRAIN, RANK on VAL, never touch TEST -------
    print(f"\n=== SEARCH: {N_ITER} sampled configs, tune TRAIN / select VAL by total_pnl ===")
    search = search_params(
        factory=factory,
        param_grid=PARAM_GRID,
        train=tr,
        val=va,
        n_iter=N_ITER,
        seed=SEED,
        objective="total_pnl",
    )
    best_params = dict(search["best_params"])
    n_tried = len(search["tried"])
    print(f"  configs actually evaluated (deduped): {n_tried}")
    print(f"  VAL-best params: {best_params}")
    print("  " + _fmt("train", search["train_metrics"]))
    print("  " + _fmt("val", search["val_metrics"]))

    # --- ONE-SHOT TEST on the selected config -------------------------------
    best_spec_factory = lambda: factory(best_params)  # noqa: E731 (fresh score_fn each split)
    selected = evaluate_splits(best_spec_factory, tr, va, te)
    print("\n=== SELECTED CONFIG ON ALL SPLITS (TEST evaluated exactly once) ===")
    for sp in ("train", "val", "test"):
        print("  " + _fmt(sp, selected[sp]))

    test_m = selected["test"]
    survives = bool(
        test_m["total_pnl"] > 0
        and test_m["n_trades"] >= 20
        and test_m["total_return"] > 0
    )
    print(f"\nsurvives_oos (net-positive on TEST, >=20 trades): {survives}")

    # --- persist machine-readable result ------------------------------------
    out = {
        "name": "trend_filter",
        "seed": SEED,
        "n_configs_tried": n_tried,
        "split_ranges": ranges,
        "baseline": {sp: base_res[sp] for sp in ("train", "val", "test")},
        "best_params": {k: best_params[k] for k in best_params},
        "selected": {sp: selected[sp] for sp in ("train", "val", "test")},
        "survives_oos": survives,
    }
    # score_fn is not JSON serializable; best_params holds only plain values.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    main()
