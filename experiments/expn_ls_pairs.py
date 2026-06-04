"""expn_ls_pairs.py — DOLLAR-NEUTRAL symmetric long/short VWAP reversion (walk-forward).

HYPOTHESIS
----------
A symmetric long/short VWAP mean-reversion book — go LONG names trading below their
session VWAP, SHORT names trading above it, with the long and short legs drawn from the
SAME mirror-image VWAP-distance predicate — strips out market beta and exposes genuine
intraday cross-name mean-reversion alpha. If true, the realized SPY beta should be near
zero and a positive, t-stat>=2 alpha should persist across MULTIPLE out-of-sample regimes.

PROTOCOL (anti-self-deception — the prior hunt found 0/9 real edges, all long-beta)
-----------------------------------------------------------------------------------
- WALK-FORWARD: 4 sequential expanding-train / disjoint-test folds spanning DIFFERENT
  regimes (high-vol recovery, calm grind, chop, rally). Tune params on each fold's TRAIN
  ONLY; score that fold's untouched TEST; AGGREGATE OOS across all folds.
- Per fold we sample a small param grid (entry_dist / vol_mult / vwap_exit_band / max_hold
  / stop_loss) on TRAIN via neutral-aware selection, pick the TRAIN-best by a
  NEUTRALITY-AWARE objective (penalize long/short trade-count imbalance so we don't select
  a closet-directional book), then run that single spec on TEST.
- BETA DECOMPOSITION is the arbiter: regress concatenated OOS daily returns on SPY daily
  returns -> alpha (annual + t-stat), beta, R^2. Market-neutral REQUIRES realized
  |beta| < ~0.15 AND alpha t-stat >= 2 AND positive in a MAJORITY of folds.
- Costs: $0 commission + 1bp adverse slippage per side on EVERY leg (both long and short).
  Strict t->t+1 fills, completed bars only, EOD flatten 15:55 NY, production caps.
- Multiple-comparisons honesty: TRAIN-only selection, OOS scored once per fold. The total
  config count tried is reported.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only.

Run: cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.expn_ls_pairs
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.harness import load_bars
from experiments.neutral_harness import (
    research_backtest_neutral,
    walk_forward_folds,
    all_trading_dates,
    slice_by_date_range,
    daily_returns,
    beta_decompose,
    spy_close_to_close_returns,
    TRADING_YEAR,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
SPY_SYMBOL = "SPY"
N_FOLDS = 4
TRAIN_FRAC = 0.5
RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/ls_pairs.json")

# Capital base for daily returns: the $500 notional book (max_exposure). Because the book
# is long AND short, gross exposure can reach 2x; we keep the SAME base as every other
# neutral experiment (max_exposure) so cross-experiment numbers are comparable.
RETURN_BASE = 500.0

# Fixed (non-tuned) caps — held constant across folds so the only thing that varies fold to
# fold is the mean-reversion entry/exit shape.
BASE_SPEC = dict(
    mode="long_short",
    enable_long=True,
    enable_short=True,
    notional=100.0,
    max_positions=4,
    max_exposure=500.0,
    cooldown_min=10.0,
    slippage_bps=1.0,
    eod_flatten="15:55",
    return_base=RETURN_BASE,
)

# Small, pre-registered tuning grid (Cartesian product = 3*3*2*2*2 = 72 candidates per
# fold). We deliberately keep it modest so the per-fold TRAIN search is honest and cheap.
PARAM_GRID = {
    "entry_dist": [0.003, 0.005, 0.008],
    "vol_mult": [1.0, 1.2, 1.5],
    "vwap_exit_band": [0.0005, 0.001],
    "max_hold": [10, 20],
    "stop_loss": [0.005, 0.008],
}


def _grid_points(grid: Dict[str, list]) -> List[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]


def _make_spec(params: dict) -> dict:
    spec = dict(BASE_SPEC)
    spec.update(params)
    return spec


def _neutrality_aware_score(res: dict) -> float:
    """TRAIN selection objective: reward net PnL but PENALIZE long/short trade imbalance.

    A symmetric book should fire a roughly equal number of long and short entries. A config
    that "wins" on TRAIN by being lopsided (e.g. 90% long) is closet beta, exactly the trap
    the prior hunt fell into. We therefore score on TRAIN by:

        total_return  -  imbalance_penalty

    where imbalance = |n_long - n_short| / max(1, n_long + n_short) in [0,1], scaled so a
    fully one-sided book is heavily discouraged. We also require a minimum trade count so we
    don't select a degenerate config that barely trades.
    """
    trades = res["trades"]
    n = len(trades)
    if n < 30:  # too few trades on TRAIN to trust — disqualify
        return -1e9
    n_long = sum(1 for t in trades if t["side"] == "long")
    n_short = n - n_long
    imbalance = abs(n_long - n_short) / max(1, n_long + n_short)
    total_return = res["metrics"].get("total_return", 0.0)
    # penalty weight 0.5: a fully lopsided book loses 0.5 (50% return) of score.
    return float(total_return) - 0.5 * imbalance


def tune_on_train(train_bars: Dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """Pick the neutrality-aware TRAIN-best spec. Returns (best_spec, diagnostics).

    Tuning touches TRAIN ONLY. The returned spec is then scored on the untouched TEST.
    """
    best = None
    best_params = None
    best_train_res = None
    for params in _grid_points(PARAM_GRID):
        spec = _make_spec(params)
        res = research_backtest_neutral(train_bars, spec)
        score = _neutrality_aware_score(res)
        if best is None or score > best:
            best = score
            best_params = params
            best_train_res = res
    spec = _make_spec(best_params)
    tr = best_train_res["trades"]
    n_long = sum(1 for t in tr if t["side"] == "long")
    diag = {
        "best_params": best_params,
        "train_score": best,
        "train_n_trades": len(tr),
        "train_n_long": n_long,
        "train_n_short": len(tr) - n_long,
        "train_total_return": best_train_res["metrics"].get("total_return", 0.0),
    }
    return spec, diag


def main() -> dict:
    print("=" * 78)
    print("expn_ls_pairs — DOLLAR-NEUTRAL symmetric long/short VWAP reversion (walk-forward)")
    print("=" * 78)

    bars = load_bars(UNIVERSE)
    spy_bars = {SPY_SYMBOL: bars[SPY_SYMBOL]}
    spy_daily_full = spy_close_to_close_returns(bars[SPY_SYMBOL])

    dates = all_trading_dates(bars)
    folds = walk_forward_folds(dates, n_folds=N_FOLDS, train_frac=TRAIN_FRAC)
    n_candidates = len(_grid_points(PARAM_GRID))
    print(f"Universe: {UNIVERSE}")
    print(f"History: {dates[0]} .. {dates[-1]} ({len(dates)} trading days)")
    print(f"Folds: {len(folds)} | grid candidates per fold: {n_candidates}")
    print(f"Configs tried (selection, TRAIN only): {n_candidates * len(folds)}")
    print()

    fold_records = []
    fold_betas = []
    oos_daily_pieces = []
    total_trades = 0
    positive = 0

    for f in folds:
        train_bars = slice_by_date_range(bars, f["_train_dates"])
        test_bars = slice_by_date_range(bars, f["_test_dates"])

        spec, diag = tune_on_train(train_bars)
        res = research_backtest_neutral(test_bars, spec)
        trades = res["trades"]
        daily = res["daily"]
        n_tr = len(trades)
        total_trades += n_tr
        tot_ret = float(daily.sum()) if len(daily) else 0.0
        if tot_ret > 0:
            positive += 1
        oos_daily_pieces.append(daily)

        n_long = sum(1 for t in trades if t["side"] == "long")
        n_short = n_tr - n_long
        # per-fold beta on that fold's OOS trading days only.
        spy_fold = spy_daily_full.reindex(daily.index).dropna() if len(daily) else spy_daily_full.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})

        rec = {
            "fold": f["fold"],
            "train": f["train"],
            "test": f["test"],
            "tuned_params": diag["best_params"],
            "train_diag": diag,
            "n_trades": n_tr,
            "n_long": n_long,
            "n_short": n_short,
            "ls_imbalance": abs(n_long - n_short) / max(1, n_tr),
            "oos_total_return": tot_ret,
            "oos_mean_daily": float(daily.mean()) if len(daily) else 0.0,
            "oos_n_days": int(len(daily)),
            "net_total_pnl": res["metrics"].get("total_pnl", 0.0),
            "win_rate": res["metrics"].get("win_rate", 0.0),
            "turnover": res["metrics"].get("turnover", 0.0),
        }
        fold_records.append(rec)

        print(f"--- FOLD {f['fold']} ---")
        print(f"  TRAIN {f['train'][0]}..{f['train'][1]} ({f['train'][2]}d)  ->  "
              f"TEST {f['test'][0]}..{f['test'][1]} ({f['test'][2]}d)")
        print(f"  tuned: {diag['best_params']}")
        print(f"  OOS trades={n_tr} (L={n_long} S={n_short}, imbalance={rec['ls_imbalance']:.2f})  "
              f"win_rate={rec['win_rate']:.3f}")
        print(f"  OOS total_return={tot_ret*100:+.2f}%  mean_daily={rec['oos_mean_daily']*1e4:+.2f}bp  "
              f"n_days={rec['oos_n_days']}")
        print(f"  beta={fb['beta']:+.3f}  alpha_ann={fb['alpha_annual']*100:+.2f}%  "
              f"alpha_t={fb['alpha_tstat']:+.2f}  R2={fb['r2']:.3f}")
        print()

    # --- aggregate OOS path + concatenated beta decomposition ---
    oos_daily = pd.concat(oos_daily_pieces).sort_index() if oos_daily_pieces else pd.Series(dtype=float)
    oos_daily = oos_daily[~oos_daily.index.duplicated(keep="first")]
    oos_mean = float(oos_daily.mean()) if len(oos_daily) else 0.0
    oos_std = float(oos_daily.std(ddof=1)) if len(oos_daily) > 1 else 0.0
    oos_sharpe = (oos_mean / oos_std * np.sqrt(TRADING_YEAR)) if oos_std > 0 else 0.0
    spy_oos = spy_daily_full.reindex(oos_daily.index).dropna() if len(oos_daily) else spy_daily_full.iloc[0:0]
    beta_full = beta_decompose(oos_daily, spy_oos)

    # cost/noise floor: per-trade net edge in bp vs the 2bp round-trip cost.
    per_trade_bp = (oos_daily.sum() * RETURN_BASE) / max(1, total_trades) / 100.0 * 1e4 if total_trades else 0.0

    print("=" * 78)
    print("AGGREGATE OUT-OF-SAMPLE (concatenated across all folds)")
    print("=" * 78)
    print(f"  OOS days={len(oos_daily)}  total OOS trades={total_trades}")
    print(f"  folds positive: {positive}/{len(folds)}")
    print(f"  OOS total_return={float(oos_daily.sum())*100:+.2f}%  mean_daily={oos_mean*1e4:+.2f}bp")
    print(f"  OOS Sharpe (ann)={oos_sharpe:+.2f}")
    print(f"  --- BETA DECOMPOSITION (vs SPY, on OOS days) ---")
    print(f"  beta={beta_full['beta']:+.4f}  beta_t={beta_full['beta_tstat']:+.2f}  R2={beta_full['r2']:.4f}")
    print(f"  alpha_per_day={beta_full['alpha_per_day']*1e4:+.3f}bp  alpha_annual={beta_full['alpha_annual']*100:+.2f}%")
    print(f"  alpha_tstat={beta_full['alpha_tstat']:+.3f}")
    print(f"  strat_ann_return={beta_full['strat_ann_return']*100:+.2f}%  n_days={beta_full['n_days']}")
    print()

    # --- VERDICT ---
    survives = (
        beta_full["alpha_annual"] > 0
        and beta_full["alpha_tstat"] >= 2.0
        and abs(beta_full["beta"]) < 0.15
        and positive > (len(folds) / 2.0)
    )
    print("=" * 78)
    print(f"VERDICT: survives_oos = {survives}")
    print("  Criteria: alpha_annual>0 AND alpha_tstat>=2 AND |beta|<0.15 AND folds_positive>majority")
    print(f"  -> alpha_annual={beta_full['alpha_annual']*100:+.2f}% (>0: {beta_full['alpha_annual']>0})")
    print(f"  -> alpha_tstat={beta_full['alpha_tstat']:.2f} (>=2: {beta_full['alpha_tstat']>=2.0})")
    print(f"  -> |beta|={abs(beta_full['beta']):.3f} (<0.15: {abs(beta_full['beta'])<0.15})")
    print(f"  -> folds_positive={positive}/{len(folds)} (majority: {positive>len(folds)/2.0})")
    print("=" * 78)

    out = {
        "name": "ls_pairs",
        "hypothesis": ("Dollar-neutral symmetric long/short VWAP reversion (long below VWAP, "
                       "short above) strips market beta and reveals intraday cross-name "
                       "mean-reversion alpha."),
        "construction": "long/short symmetric VWAP mean-reversion (mode=long_short, side=both)",
        "n_folds": len(folds),
        "n_configs_tried": n_candidates * len(folds),
        "folds_positive": positive,
        "oos_n_trades": total_trades,
        "oos_n_days": int(len(oos_daily)),
        "oos_total_return": float(oos_daily.sum()),
        "oos_mean_return": oos_mean,
        "oos_sharpe": oos_sharpe,
        "per_trade_bp": per_trade_bp,
        "beta": beta_full["beta"],
        "beta_tstat": beta_full["beta_tstat"],
        "alpha_per_day": beta_full["alpha_per_day"],
        "alpha_annual": beta_full["alpha_annual"],
        "alpha_tstat": beta_full["alpha_tstat"],
        "r2": beta_full["r2"],
        "survives_oos": bool(survives),
        "folds": fold_records,
        "fold_betas": fold_betas,
        "beta_decompose": beta_full,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    main()
