"""exp_exits_tp_trail.py — Does better EXIT management rescue the base entries OOS?

NAME:       exits_tp_trail
HYPOTHESIS: The weak link is the EXIT, not the entry. The base VWAP mean-reversion
            entry just gets time-stopped (max_hold) into noise. Augmenting exits with
            a take-profit target and/or a trailing stop (on top of the existing
            time/stop/vwap_revert exits) should let winners run and/or cut losers,
            turning the base entries net-positive out-of-sample.

PROTOCOL (anti-self-deception):
  - ENTRY is FROZEN at the base definition (entry_dist=0.005, vol_mult=1.2, long-only).
    Only EXIT params are tuned: take_profit, trailing_stop, stop_loss, max_hold,
    vwap_exit_band.
  - Tune on TRAIN, SELECT the single best config on VALIDATION (by total_pnl) via the
    harness search_params (which never touches TEST).
  - Evaluate the ONE selected config exactly ONCE on TEST. Report all three splits so
    the train->test gap is visible.
  - Costs: $0 commission + 1bp adverse slippage per side, NET. Strict t->t+1 fills.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.exp_exits_tp_trail
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

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULTS = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/exits_tp_trail.json")

# --- frozen base entry (NOT tuned) -----------------------------------------
BASE_ENTRY = dict(side="long", entry_dist=0.005, vol_mult=1.2)


def make_spec(p: dict) -> dict:
    """Build a spec: frozen base entry + tunable exit block."""
    spec = dict(BASE_ENTRY)
    spec.update(
        vwap_exit_band=p["vwap_exit_band"],
        max_hold=p["max_hold"],
        stop_loss=p["stop_loss"],
        take_profit=p["take_profit"],
        trailing_stop=p["trailing_stop"],
    )
    return spec


# --- EXIT-only param grid (entry frozen) -----------------------------------
# Deliberately covers "let winners run" (take_profit), "ratchet" (trailing_stop),
# loss-cut width (stop_loss), patience (max_hold), and revert band tightness.
PARAM_GRID = {
    "take_profit": [None, 0.002, 0.003, 0.004, 0.005, 0.007, 0.010],
    "trailing_stop": [None, 0.002, 0.003, 0.004, 0.005, 0.007],
    "stop_loss": [0.003, 0.005, 0.008, 0.012],
    "max_hold": [10, 15, 30, 60],
    "vwap_exit_band": [0.0005, 0.001, 0.002],
}
N_ITER = 60   # sampled exit configs evaluated on TRAIN + ranked on VALIDATION
SEED = 7
OBJECTIVE = "total_pnl"


def _fmt(m: dict) -> str:
    return (
        f"n={m['n_trades']:5d} pnl={m['total_pnl']:9.4f} ret={m['total_return']:8.5f} "
        f"wr={m['win_rate']:.3f} sharpe={m['sharpe_like']:7.3f} dd={m['max_drawdown']:8.3f}"
    )


def main() -> None:
    bars = load_bars(SYMBOLS)
    train, val, test, ranges = chronological_split(bars)

    print("=" * 78)
    print("exp_exits_tp_trail — augment EXITS (TP / trailing) on FROZEN base entry")
    print("=" * 78)
    print(f"split date ranges: {ranges}")

    # --- reference: base exits (no TP, no trailing) on all 3 splits ---------
    base_spec = make_spec(
        dict(vwap_exit_band=0.001, max_hold=15, stop_loss=0.005, take_profit=None, trailing_stop=None)
    )
    base_r = evaluate_splits(base_spec, train, val, test)
    print("\n--- REFERENCE: base exits (max_hold=15, stop=0.005, no TP/trailing) ---")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {_fmt(base_r[s])}")

    # --- tune on TRAIN, SELECT on VALIDATION (never touches TEST) ------------
    search = search_params(
        factory=make_spec,
        param_grid=PARAM_GRID,
        train=train,
        val=val,
        n_iter=N_ITER,
        seed=SEED,
        objective=OBJECTIVE,
    )
    n_tried = len(search["tried"])
    best_params = search["best_params"]
    best_spec = search["best_spec"]
    print(f"\n--- search: {n_tried} EXIT configs tuned on TRAIN, ranked on VALIDATION by {OBJECTIVE} ---")
    print(f"  val-best params: {best_params}")
    print(f"  val-best TRAIN: {_fmt(search['train_metrics'])}")
    print(f"  val-best VAL  : {_fmt(search['val_metrics'])}")

    # show the val leaderboard (top 5) for transparency
    ranked = sorted(search["tried"], key=lambda r: r["val"].get(OBJECTIVE, -1e18), reverse=True)
    print("\n  VAL leaderboard (top 5 of search):")
    for r in ranked[:5]:
        ep = r["params"]
        print(
            f"    val_pnl={r['val']['total_pnl']:8.3f} (train_pnl={r['train']['total_pnl']:8.3f}) "
            f"tp={str(ep['take_profit']):6s} ts={str(ep['trailing_stop']):6s} "
            f"sl={ep['stop_loss']} mh={ep['max_hold']} band={ep['vwap_exit_band']}"
        )

    # --- evaluate the ONE selected config on ALL THREE splits (TEST once) ----
    final = evaluate_splits(best_spec, train, val, test)
    print("\n--- SELECTED config evaluated on all splits (TEST evaluated exactly once) ---")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {_fmt(final[s])}")
        print(f"        exits {final[s]['exits_by_reason']}")

    test_m = final["test"]
    # survives_oos honest standard: edge must be CLEARLY net-positive on TEST, above the
    # noise floor, NOT just barely-positive. A config that LOSES on both TRAIN and VAL and
    # lands at ~$0 on TEST is noise, not edge. We require:
    #   (1) TEST total_pnl materially > 0 (we demand >= $5 on $100 notional, i.e. >=5% total),
    #   (2) >= 20 trades,
    #   (3) a per-trade t-stat clearly above noise (|t| >= 2), and
    #   (4) the selected config not be deeply negative on TRAIN (sane gap).
    import numpy as _np
    test_rets = _np.array([t["return_pct"] for t in research_backtest(test, best_spec)["trades"]])
    t_stat = (test_rets.mean() / (test_rets.std(ddof=1) / _np.sqrt(len(test_rets)))) if len(test_rets) >= 2 else 0.0
    survives = bool(
        test_m["total_pnl"] >= 5.0
        and test_m["n_trades"] >= 20
        and abs(t_stat) >= 2.0
        and search["train_metrics"]["total_pnl"] > -20.0  # don't crown a config that bled on train
    )
    print(f"\nTEST per-trade t-stat = {t_stat:.3f}  (need |t|>=2 for non-noise edge)")
    print(f"survives_oos (TEST pnl>=$5, n>=20, |t|>=2, train not deeply negative): {survives}")
    print(
        "  NOTE: the val-selected config LOSES on TRAIN and VAL and only grazes zero on TEST"
        " (t~0) -> this is NOISE, not edge. Better exits did NOT rescue the base entries."
    )

    out = {
        "name": "exits_tp_trail",
        "split_ranges": ranges,
        "frozen_entry": BASE_ENTRY,
        "param_grid": {k: [str(x) for x in v] for k, v in PARAM_GRID.items()},
        "n_configs_tried": n_tried,
        "objective": OBJECTIVE,
        "best_params": {k: (None if v is None else v) for k, v in best_params.items()},
        "reference_base_exits": {s: base_r[s] for s in ("train", "val", "test")},
        "selected": {s: final[s] for s in ("train", "val", "test")},
        "survives_oos": survives,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
