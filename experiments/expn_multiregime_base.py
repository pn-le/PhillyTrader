"""expn_multiregime_base.py — CONTROL / baseline for the beta-neutral multi-regime hunt.

ROLE
----
This is the ANCHOR experiment, not a candidate edge. It runs the ORIGINAL long-only
VWAP mean-reversion base strategy — the exact production defaults

    entry_dist=0.005, vol_mult=1.2, max_hold=15, vwap_exit_band=0.001, stop_loss=0.005

— WALK-FORWARD across every sequential out-of-sample fold spanning DIFFERENT market
regimes, then regresses its daily returns on SPY to decompose alpha vs beta.

PURPOSE
-------
The prior single-split hunt (results/SUMMARY.md) found 0/9 real edges: every "winning"
variant was long-beta riding ONE homogeneous bull TEST window. This control quantifies
exactly how much of that "+edge" was a single-regime / beta artifact. We EXPECT:
  - per-fold OOS returns that FLIP SIGN across regimes (a rally fold is green, a
    chop/drawdown fold is red) — i.e. NOT consistent edge,
  - realized beta ~1 (it is a long-only book of large-cap equities/ETFs, so it IS the
    market by construction), and
  - alpha indistinguishable from zero once you subtract that beta.

So `survives_oos` is almost certainly FALSE, and that is the honest, intended result.
It is the yardstick the genuinely-neutral variants must beat.

WHY long_short(enable_short=False) and NOT a tuned spec
-------------------------------------------------------
- A CONTROL must not be tuned: we score the SAME fixed default spec on every fold's
  untouched TEST slice (spec_or_factory is a STATIC dict). n_configs_tried = 1.
- Sanity A in results/neutral_harness_sanity.md proved that the neutral
  long_short mode with enable_short=False is BIT-IDENTICAL to harness long-only. So
  routing through research_backtest_neutral keeps the daily-return / beta plumbing
  identical to the other neutral experiments while reproducing the production long-only
  engine exactly.

PROTOCOL (see neutral_harness.py)
---------------------------------
- WALK-FORWARD: 4 sequential, non-overlapping OOS test folds tiling the back of history;
  TRAIN is everything strictly before each fold's TEST (anchored/expanding). The control
  ignores TRAIN entirely (no tuning) and scores the fixed spec on each TEST.
- Strict t->t+1 fills, completed bars only, no overnight (EOD flatten 15:55 NY).
- Costs: $0 commission + 1bp adverse slippage per side on every leg. Reported NET.
- BETA DECOMPOSITION: concatenate all OOS days across folds, regress daily strat returns
  on SPY daily (close-to-close) returns -> alpha (annual + t-stat), beta, R^2.
- survives_oos = TRUE only if aggregate OOS alpha > 0 with alpha_tstat >= 2 AND
  |beta| < ~0.15 AND positive in a MAJORITY of folds AND above the cost/noise floor.

Runnable as:
    cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.expn_multiregime_base
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.neutral_harness import (
    evaluate_walkforward,
    load_bars,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/multiregime_base.json")

N_FOLDS = 4
TRAIN_FRAC = 0.5
RETURN_BASE = 500.0  # max_exposure book size; daily returns are PnL / this base

# The ORIGINAL long-only base strategy, production defaults, ZERO tuning.
# mode='long_short' with enable_short=False == harness long-only (Sanity A: bit-identical).
BASE_SPEC = {
    "mode": "long_short",
    "enable_long": True,
    "enable_short": False,  # long-only — reproduces the production base strategy exactly
    "side": "long",
    "entry_dist": 0.005,
    "vol_mult": 1.2,
    "vwap_exit_band": 0.001,
    "max_hold": 15,
    "stop_loss": 0.005,
    "take_profit": None,
    "trailing_stop": None,
    "trend_filter": None,
    "time_window": None,
    # production sizing / caps / costs
    "notional": 100.0,
    "max_positions": 4,
    "max_exposure": 500.0,
    "cooldown_min": 10.0,
    "slippage_bps": 1.0,
    "eod_flatten": "15:55",
    "return_base": RETURN_BASE,
}

# Multiple-comparisons honesty: this control tries exactly ONE fixed config (no search).
N_CONFIGS_TRIED = 1


def _fmt_fold(fr: dict, fb: dict) -> str:
    m = fr["metrics"]
    return (
        f"fold {fr['fold']}  TEST {fr['test'][0]}..{fr['test'][1]} ({fr['test'][2]}d)  "
        f"n_tr={fr['n_trades']:5d}  ret={fr['total_return']:+8.4f}  "
        f"mean_daily={fr['mean_daily_return']:+.6f}  "
        f"beta={fb['beta']:+.3f}  alpha_ann={fb['alpha_annual']:+.4f}  "
        f"alpha_t={fb['alpha_tstat']:+.2f}  R2={fb['r2']:.3f}  "
        f"net_pnl={m.get('total_pnl', float('nan')):+.3f}"
    )


def main() -> dict:
    bars = load_bars(UNIVERSE)
    if "SPY" not in bars:
        raise RuntimeError("SPY bars required for beta decomposition / hedge factor")

    print("=" * 100)
    print("CONTROL: multiregime_base — original LONG-ONLY base strategy, walk-forward, beta-decomposed")
    print("=" * 100)
    print(f"Symbols loaded : {sorted(bars.keys())}")
    print("Spec (FIXED, untuned): entry_dist=0.005 vol_mult=1.2 max_hold=15 "
          "vwap_exit_band=0.001 stop_loss=0.005  side=long  (enable_short=False)")
    print(f"Walk-forward   : {N_FOLDS} sequential OOS folds, train_frac={TRAIN_FRAC} "
          f"(anchored/expanding train; TRAIN unused — this is a no-tune control)")
    print(f"Costs          : $0 commission + 1bp/side adverse slippage; t->t+1 fills; "
          f"EOD flatten 15:55 NY; return_base=${RETURN_BASE:.0f}")
    print(f"n_configs_tried: {N_CONFIGS_TRIED} (single fixed config — no search, no selection)")
    print("-" * 100)

    res = evaluate_walkforward(
        BASE_SPEC,
        bars=bars,
        spy_bars=bars,
        n_folds=N_FOLDS,
        train_frac=TRAIN_FRAC,
        spy_symbol="SPY",
    )

    # Per-fold lines (return + beta decomposition) — expect sign flips + beta ~1.
    print("PER-FOLD OOS (each fold's TEST scored once; beta regressed on that fold's OOS days):")
    fb_by_fold = {fb["fold"]: fb for fb in res["fold_betas"]}
    for fr in res["folds"]:
        print("  " + _fmt_fold(fr, fb_by_fold[fr["fold"]]))
    print("-" * 100)

    bd = res["beta_decompose"]
    print("AGGREGATE OOS (all folds concatenated):")
    print(f"  OOS days regressed     : {bd['n_days']}")
    print(f"  OOS total return       : {res['oos_total_return']:+.4f}  "
          f"(on ${RETURN_BASE:.0f} book)")
    print(f"  OOS mean daily return  : {res['oos_mean_return']:+.6f}")
    print(f"  OOS Sharpe (annualized): {res['oos_sharpe']:+.3f}")
    print(f"  OOS trades total       : {res['oos_n_trades']}")
    print(f"  Folds positive (net)   : {res['folds_positive']} / {res['n_folds']}")
    print("  --- beta decomposition (strat_t = alpha + beta*SPY_t + eps) ---")
    print(f"  beta                   : {bd['beta']:+.4f}   (need |beta| < ~0.15 for neutral)")
    print(f"  beta t-stat            : {bd['beta_tstat']:+.3f}")
    print(f"  alpha (per day)        : {bd['alpha_per_day']:+.6e}")
    print(f"  alpha (annualized)     : {bd['alpha_annual']:+.4f}")
    print(f"  alpha t-stat           : {bd['alpha_tstat']:+.3f}   (need >= 2 for real alpha)")
    print(f"  R^2 (var explained)    : {bd['r2']:.4f}   (high => returns ARE the market)")
    print("-" * 100)

    # survives_oos gate (expected FALSE for this control)
    majority = res["folds_positive"] > (res["n_folds"] / 2.0)
    cost_floor_daily = 2.0 / RETURN_BASE  # ~2bp round-trip per $ traded; loose noise floor on the book return
    above_floor = res["oos_mean_return"] > cost_floor_daily
    survives = bool(
        bd["alpha_annual"] > 0
        and bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and majority
        and above_floor
    )
    print("survives_oos gate:")
    print(f"  alpha_annual > 0          : {bd['alpha_annual'] > 0}  ({bd['alpha_annual']:+.4f})")
    print(f"  alpha_tstat >= 2          : {bd['alpha_tstat'] >= 2.0}  ({bd['alpha_tstat']:+.2f})")
    print(f"  |beta| < 0.15             : {abs(bd['beta']) < 0.15}  (|{bd['beta']:.3f}|)")
    print(f"  positive in majority folds: {majority}  ({res['folds_positive']}/{res['n_folds']})")
    print(f"  above cost/noise floor    : {above_floor}  "
          f"(mean_daily {res['oos_mean_return']:+.6f} vs floor {cost_floor_daily:.6f})")
    print(f"  => survives_oos           : {survives}")
    print("=" * 100)
    print("INTERPRETATION (control / anchor): a long-only large-cap book IS the market by")
    print("construction. Expect beta ~1, high R^2, alpha ~0, and per-fold returns that flip")
    print("sign with the regime — confirming the prior 'edges' were directional beta, not")
    print("market-neutral alpha. This is the yardstick the neutral variants must beat.")
    print("=" * 100)

    payload = {
        "name": "multiregime_base",
        "role": "control_baseline",
        "hypothesis": "The original long-only VWAP mean-reversion base strategy has NO "
                      "market-neutral alpha across regimes; its OOS 'edge' is directional "
                      "SPY beta (beta ~1, alpha ~0) and per-fold returns flip sign with the "
                      "regime. Run as the anchor/control to quantify the prior beta artifact.",
        "construction": "long-only (no hedge, no short leg) — DELIBERATELY beta-loaded; "
                        "this is the control, not a neutral candidate",
        "spec": {k: v for k, v in BASE_SPEC.items()},
        "n_configs_tried": N_CONFIGS_TRIED,
        "n_folds": res["n_folds"],
        "folds_positive": res["folds_positive"],
        "oos_total_return": res["oos_total_return"],
        "oos_mean_return": res["oos_mean_return"],
        "oos_sharpe": res["oos_sharpe"],
        "oos_n_trades": res["oos_n_trades"],
        "oos_n_days": res["oos_n_days"],
        "beta_decompose": bd,
        "fold_betas": res["fold_betas"],
        "folds": [
            {
                "fold": fr["fold"],
                "train": fr["train"],
                "test": fr["test"],
                "n_trades": fr["n_trades"],
                "total_return": fr["total_return"],
                "mean_daily_return": fr["mean_daily_return"],
                "n_days": fr["n_days"],
                "net_pnl": fr["metrics"].get("total_pnl"),
                "win_rate": fr["metrics"].get("win_rate"),
            }
            for fr in res["folds"]
        ],
        "survives_oos": survives,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
