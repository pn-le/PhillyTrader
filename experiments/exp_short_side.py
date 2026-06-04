"""exp_short_side.py — Is there an OUT-OF-SAMPLE short-side edge?

HYPOTHESIS
----------
The VWAP mean-reversion strategy is symmetric: SHORTING when price is >= entry_dist
ABOVE VWAP with elevated volume, then covering on revert-to-VWAP / stop / time, should
have edge (paper trading supports shorting). This experiment ISOLATES the short side
(no long entries) and asks honestly whether that edge survives on an untouched TEST split.

PROTOCOL (anti-self-deception — see harness.py)
-----------------------------------------------
- Chronological split by DATE: TRAIN ~60% / VAL ~20% / TEST ~20% (global date cutoffs).
- Tune params on TRAIN only; SELECT the single config on VALIDATION only (search_params,
  which never touches TEST). Evaluate that ONE selected config on TEST exactly once.
- Strict t->t+1 fills, $0 commission + 1bp adverse slippage per side, all metrics NET.
- An "edge" counts ONLY if TEST PnL is clearly positive with >= ~20 trades and a sane
  train->test gap.

WHAT WE SEARCH
--------------
A modest, deliberately-bounded grid over the short-entry geometry and exit discipline:
  entry_dist   : how far ABOVE VWAP price must be to short (fade strength)
  vol_mult     : volume-spike confirmation
  max_hold     : minutes before time-stop
  stop_loss    : adverse-move cap
  take_profit  : optional quick-profit exit (None disables)
  trend_filter : None / "with" (fade the move) / "against" (momentum-confirm the short)
  time_window  : None or a session sub-window (intraday seasonality)
The objective ranks VALIDATION total_pnl. We report n_configs_tried for multiple-comparisons
honesty.

This file is runnable as:
    cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.exp_short_side
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.harness import (
    chronological_split,
    evaluate_splits,
    load_bars,
    research_backtest,
    search_params,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/short_side.json")

# Search space (short side only). Kept modest & robust on purpose.
PARAM_GRID = {
    "entry_dist": [0.004, 0.006, 0.008, 0.010, 0.012],
    "vol_mult": [1.2, 1.5, 2.0],
    "max_hold": [10, 15, 30],
    "stop_loss": [0.004, 0.006, 0.008],
    "take_profit": [None, 0.003, 0.005],
    "trend_filter": [None, "with", "against"],
    "time_window": [None, ("09:30", "11:00"), ("11:00", "15:55")],
}

N_ITER = 60   # sampled configs evaluated on TRAIN+VAL (never TEST)
SEED = 20260604


def make_spec(p: dict) -> dict:
    """Factory: sampled params -> a full short-side spec (production caps/costs)."""
    return {
        "side": "short",
        "entry_dist": p["entry_dist"],
        "vol_mult": p["vol_mult"],
        "vwap_exit_band": 0.001,
        "max_hold": p["max_hold"],
        "stop_loss": p["stop_loss"],
        "take_profit": p["take_profit"],
        "trend_filter": p["trend_filter"],
        "time_window": p["time_window"],
        # production sizing / caps / costs
        "notional": 100.0,
        "max_positions": 4,
        "max_exposure": 500.0,
        "cooldown_min": 10.0,
        "slippage_bps": 1.0,
        "eod_flatten": "15:55",
    }


def _fmt(m: dict) -> str:
    return (
        f"n={m['n_trades']:5d}  pnl={m['total_pnl']:9.3f}  ret={m['total_return']:8.4f}  "
        f"wr={m['win_rate']:.3f}  sharpe={m['sharpe_like']:7.2f}  maxdd={m['max_drawdown']:9.3f}"
    )


def main() -> dict:
    bars = load_bars(UNIVERSE)
    train, val, test, ranges = chronological_split(bars)

    print("=" * 90)
    print("EXPERIMENT: short_side — isolated SHORT VWAP mean-reversion, out-of-sample test")
    print("=" * 90)
    print(f"Symbols loaded : {sorted(bars.keys())}")
    print(f"TRAIN dates    : {ranges['train']}")
    print(f"VAL   dates    : {ranges['val']}")
    print(f"TEST  dates    : {ranges['test']}")
    print(f"Grid size      : {sum(len(v) for v in PARAM_GRID.values())} levels across "
          f"{len(PARAM_GRID)} params; sampling N_ITER={N_ITER} configs (seed={SEED})")
    print("-" * 90)

    # --- Reference: default-param baseline short (NOT selected; context only) -----------
    base_spec = make_spec(
        {"entry_dist": 0.005, "vol_mult": 1.2, "max_hold": 15,
         "stop_loss": 0.005, "take_profit": None, "trend_filter": None, "time_window": None}
    )
    base = evaluate_splits(base_spec, train, val, test)
    print("Reference baseline short (entry_dist=0.005, vol_mult=1.2, defaults):")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {_fmt(base[s])}")
    print("-" * 90)

    # --- TUNE on TRAIN, SELECT on VALIDATION (never touches TEST) ------------------------
    search = search_params(
        factory=make_spec,
        param_grid=PARAM_GRID,
        train=train,
        val=val,
        n_iter=N_ITER,
        seed=SEED,
        objective="total_pnl",
    )
    n_tried = len(search["tried"])
    best_params = search["best_params"]
    best_spec = search["best_spec"]

    print(f"Configs actually evaluated on TRAIN+VAL: {n_tried}")
    print(f"VALIDATION-selected config (best val total_pnl): {best_params}")
    print(f"  selected TRAIN metrics: {_fmt(search['train_metrics'])}")
    print(f"  selected VAL   metrics: {_fmt(search['val_metrics'])}")
    print("-" * 90)

    # --- Evaluate the ONE selected config on ALL THREE splits (TEST exactly once) -------
    final = evaluate_splits(best_spec, train, val, test)
    print("FINAL — single selected config evaluated on all three splits:")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {_fmt(final[s])}  exits={final[s]['exits_by_reason']}")
    print("-" * 90)

    test_m = final["test"]
    survives = bool(test_m["total_pnl"] > 0 and test_m["n_trades"] >= 20)
    print(f"survives_oos (TEST pnl>0 AND n_trades>=20): {survives}")
    print("=" * 90)

    payload = {
        "name": "short_side",
        "hypothesis": "Shorting when price >= entry_dist ABOVE VWAP with elevated volume, "
                      "covering on revert/stop/time, has out-of-sample edge.",
        "split_ranges": ranges,
        "n_configs_tried": n_tried,
        "best_config": best_params,
        "baseline_default": {s: base[s] for s in ("train", "val", "test")},
        "train": final["train"],
        "val": final["val"],
        "test": final["test"],
        "survives_oos": survives,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
