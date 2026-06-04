"""exp_tf_5min.py — Does the VWAP mean-reversion edge survive better on 5-MINUTE bars?

HYPOTHESIS
----------
1-minute IEX bars are dominated by microstructure noise (thin free-tier volume, wide
relative spreads, choppy VWAP). The SAME mean-reversion logic, applied to 5-MINUTE
resampled bars, should reject more noise and therefore show better out-of-sample edge.

METHOD (anti-self-deception protocol)
-------------------------------------
- Load 1-min RTH bars for the 10-symbol universe, resample to 5-min via harness.resample.
- Indicators (session VWAP, dist_from_vwap, rolling-20 volume_ratio) are RECOMPUTED on the
  5-min bars by the harness backtester (compute_indicators_df runs inside research_backtest).
- Chronological split by DATE: TRAIN ~60% / VAL ~20% / TEST ~20% (global cutoffs).
- Tune params on TRAIN, SELECT the config on VALIDATION via search_params (objective =
  total_pnl). TEST is evaluated EXACTLY ONCE on the single selected config at the end.
- Strict t->t+1 fills at next-bar OPEN, 1bp adverse slippage per side, $0 commission.
  All metrics NET of costs.

5-MIN-SPECIFIC NOTES
--------------------
- The harness hardcodes MIN_BARS_FOR_ENTRY = ROLLING_VOL_WINDOW+1 = 21. On 5-min bars that
  is the 21st 5-min bar, i.e. entries cannot start until ~11:10 NY. This is a STRUCTURAL
  consequence of requiring a true rolling-20 volume window on coarser bars (cannot be
  changed without editing the harness, which is forbidden). It restricts 5-min entries to
  the back ~70% of the session. Reported honestly; not a leak, just a constraint.
- max_hold / vwap_exit_band / cooldown_min are in MINUTES, so the search grid uses
  5-min-appropriate multiples (max_hold in {25,40,60,90} min = 5..18 bars; cooldown 15 min).
- entry_dist / vol_mult ranges are widened a touch vs the 1-min defaults because a 5-min
  bar aggregates 5 minutes of move, so VWAP excursions are computed on smoother prices.

COMPARISON TO 1-MIN BASE (same universe, same split, computed in __main__):
  the 1-min long-only base (entry_dist=.005, vol_mult=1.2, max_hold=15, sl=.005) is the
  reference. We compare the 5-min selected config against it qualitatively on all splits.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments import harness as H

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULT_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/tf_5min.json")

# 1-min long-only base, for the qualitative comparison the task asks for.
BASE_1MIN_SPEC = dict(
    side="long",
    entry_dist=0.005,
    vol_mult=1.2,
    vwap_exit_band=0.001,
    max_hold=15,
    stop_loss=0.005,
)


def make_spec(p: dict) -> dict:
    """Build a 5-min mean-reversion spec from sampled params (long-only, fade-the-move)."""
    return dict(
        side="long",
        entry_dist=p["entry_dist"],
        vol_mult=p["vol_mult"],
        vwap_exit_band=p["vwap_exit_band"],
        max_hold=p["max_hold"],
        stop_loss=p["stop_loss"],
        take_profit=p.get("take_profit"),
        # everything else = production defaults (notional 100, 4 positions, $500 exposure,
        # 1bp slippage, EOD 15:55). cooldown bumped to 15 min so 5-min bars don't re-enter
        # the same symbol on consecutive bars.
        cooldown_min=15.0,
    )


# Search grid — deliberately MODEST and economically sensible (multiple-comparisons honesty).
# Total grid size = 4*3*2*4*3*2 = 576 combinations; we SAMPLE n_iter of them on TRAIN and
# rank on VALIDATION. The number actually tried is reported as n_configs_tried.
PARAM_GRID = {
    "entry_dist": [0.003, 0.004, 0.005, 0.007],   # frac below VWAP to enter
    "vol_mult": [1.0, 1.2, 1.5],                   # volume_ratio gate
    "vwap_exit_band": [0.0005, 0.001],             # revert-to-VWAP exit band
    "max_hold": [25, 40, 60, 90],                  # minutes (5..18 five-min bars)
    "stop_loss": [0.004, 0.006, 0.010],            # loss cap
    "take_profit": [None, 0.006],                  # optional profit target
}

N_ITER = 40   # sampled configs evaluated on TRAIN+VAL (never TEST)
SEED = 7


def run() -> dict:
    bars_1min = H.load_bars(UNIVERSE)
    # Resample EVERY symbol to 5-min, THEN split (split is by date, order-independent).
    bars_5min = {s: H.resample(df, 5) for s, df in bars_1min.items()}

    tr5, va5, te5, ranges = H.chronological_split(bars_5min)

    # --- TUNE on TRAIN, SELECT on VALIDATION (search never touches TEST) ---
    search = H.search_params(
        factory=make_spec,
        param_grid=PARAM_GRID,
        train=tr5,
        val=va5,
        n_iter=N_ITER,
        seed=SEED,
        objective="total_pnl",
    )
    best_params = search["best_params"]
    best_spec = search["best_spec"]
    n_tried = len(search["tried"])

    # --- Evaluate the SINGLE selected config on all three splits (TEST exactly once) ---
    splits_5min = H.evaluate_splits(best_spec, tr5, va5, te5)

    # --- 1-min base for qualitative comparison (same universe, same split) ---
    tr1, va1, te1, _ = H.chronological_split(bars_1min)
    splits_1min = H.evaluate_splits(BASE_1MIN_SPEC, tr1, va1, te1)

    # --- Skeptic's regime check (diagnostic; does NOT influence selection) ---
    regime = regime_check(tr5, va5, te5)

    return {
        "name": "tf_5min",
        "ranges": ranges,
        "best_params": best_params,
        "best_spec": {k: v for k, v in best_spec.items()},
        "n_configs_tried": n_tried,
        "five_min": splits_5min,
        "one_min_base": splits_1min,
        "regime_check": regime,
    }


def regime_check(tr5, va5, te5, sample: int = 64, seed: int = 11) -> dict:
    """Skeptic's check: is any TEST edge EARNED by selection, or just the test regime?

    Sample `sample` configs from the full grid, run all three splits for each, and report
    (a) the fraction of configs profitable on each split and (b) the rank correlation of
    VAL pnl with TEST pnl. If VAL pnl does not predict TEST pnl, then picking the val-best
    config is NOT skill — any TEST profit is a property of the test window, not the method.
    """
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=min(sample, len(combos)), replace=False)
    trp, vap, tep, ten = [], [], [], []
    for i in idx:
        spec = make_spec(dict(zip(keys, combos[i])))
        trp.append(H.research_backtest(tr5, spec)["metrics"]["total_pnl"])
        vap.append(H.research_backtest(va5, spec)["metrics"]["total_pnl"])
        me = H.research_backtest(te5, spec)["metrics"]
        tep.append(me["total_pnl"])
        ten.append(me["n_trades"])
    trp, vap, tep, ten = map(lambda x: np.array(x, float), (trp, vap, tep, ten))
    return {
        "sampled": int(len(idx)),
        "train_frac_pos": float((trp > 0).mean()),
        "val_frac_pos": float((vap > 0).mean()),
        "test_frac_pos": float((tep > 0).mean()),
        "test_pnl_mean": float(tep.mean()),
        "test_pnl_min": float(tep.min()),
        "test_pnl_max": float(tep.max()),
        "test_avg_ntrades": float(ten.mean()),
        "corr_val_test": float(np.corrcoef(vap, tep)[0, 1]) if vap.std() > 0 else 0.0,
        "corr_train_test": float(np.corrcoef(trp, tep)[0, 1]) if trp.std() > 0 else 0.0,
    }


def _fmt(tag: str, m: dict) -> str:
    return (
        f"  {tag:6s} n={m['n_trades']:4d} pnl={m['total_pnl']:8.3f} "
        f"ret={m['total_return']:8.4f} wr={m['win_rate']:.3f} "
        f"sharpe={m['sharpe_like']:7.3f} dd={m['max_drawdown']:8.3f} "
        f"avg_ret={m['avg_return_pct']*1e4:6.2f}bps"
    )


def main() -> None:
    out = run()
    print("=" * 78)
    print("EXPERIMENT tf_5min — 5-minute VWAP mean-reversion (long-only)")
    print("=" * 78)
    print(f"split date ranges: {out['ranges']}")
    print(f"configs tried (TRAIN+VAL only): {out['n_configs_tried']}")
    print(f"selected params (val-best on total_pnl): {out['best_params']}")
    print()
    print("5-MIN selected config:")
    for sp in ("train", "val", "test"):
        print(_fmt(sp, out["five_min"][sp]))
    print()
    print("1-MIN base (reference, entry_dist=.005 vol_mult=1.2 max_hold=15 sl=.005):")
    for sp in ("train", "val", "test"):
        print(_fmt(sp, out["one_min_base"][sp]))
    print()

    rc = out["regime_check"]
    print("REGIME CHECK (is TEST edge earned by selection or just the test window?):")
    print(f"  sampled {rc['sampled']} configs | frac profitable -> "
          f"train={rc['train_frac_pos']:.2f} val={rc['val_frac_pos']:.2f} "
          f"test={rc['test_frac_pos']:.2f}")
    print(f"  TEST pnl across configs: mean={rc['test_pnl_mean']:.2f} "
          f"[{rc['test_pnl_min']:.2f}, {rc['test_pnl_max']:.2f}] "
          f"avg_n={rc['test_avg_ntrades']:.0f}")
    print(f"  corr(VAL pnl, TEST pnl)={rc['corr_val_test']:.3f}  "
          f"corr(TRAIN pnl, TEST pnl)={rc['corr_train_test']:.3f}")
    print()

    t = out["five_min"]["test"]
    # An honest "survives" requires not just positive TEST pnl, but that the selection
    # actually has skill: VAL must positively predict TEST AND TEST profitability must not
    # be near-universal across the grid (which would mean it is regime luck, not edge).
    test_positive = bool(t["total_pnl"] > 0 and t["n_trades"] >= 20)
    selection_has_skill = bool(rc["corr_val_test"] > 0.2 and rc["test_frac_pos"] < 0.9)
    survives = bool(test_positive and selection_has_skill)
    print(f"TEST 5-min: pnl={t['total_pnl']:.3f} n_trades={t['n_trades']} "
          f"test_positive={test_positive} selection_has_skill={selection_has_skill} "
          f"-> survives_oos={survives}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
