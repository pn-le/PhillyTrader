"""fac_combo.py — DAILY cross-sectional MULTI-FACTOR COMBO (mom + reversal + low-vol).

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on cached daily bars.

=============================================================================
HYPOTHESIS
=============================================================================
A diversified equal-ish blend of THREE standardized cross-sectional factors is MORE ROBUST
out-of-sample than any single factor, because the three lean on different, partly-orthogonal
return sources:

  - MOMENTUM (mom)      : 12-month total return skipping the most recent month
                          (close_{d-21}/close_{d-252} - 1). Classic Jegadeesh-Titman
                          medium-term momentum, with the 1-month skip to avoid contaminating
                          it with the short-term reversal effect. Higher => go LONG.
  - SHORT-TERM REVERSAL : negative of the last-~month return (-(close_d/close_{d-21}-1)).
                          Buy recent 1-month losers, short recent 1-month winners. Higher
                          score => bigger recent loser => go LONG.
  - LOW-VOL (lowvol)    : negative of trailing ~63d daily-return volatility. The low-volatility
                          anomaly: low-vol names tend to earn better risk-adjusted returns.
                          Higher score => lower vol => go LONG.

Each component is computed using ONLY information available at the CLOSE of day d (the
factor_fn contract), z-scored CROSS-SECTIONALLY per day (so the three live on a comparable
scale), then blended by weights (w_mom, w_rev, w_lowvol) into one composite score. We RANK
the composite, go dollar-neutral LONG the top quintile / SHORT the bottom quintile, MONTHLY
rebalance (freq=21). Monthly rebalance keeps turnover (hence cost) low — the right cadence
for a momentum/low-vol-dominated blend (reversal alone would want faster cadence but costs
more; the blend deliberately trades slowly).

=============================================================================
SELECTION DISCIPLINE (honest: TRAIN-only tuning, OOS scored once)
=============================================================================
We use the harness FACTORY form: for each walk-forward fold, we tune the blend weights on
that fold's TRAIN window ONLY (maximize TRAIN net Sharpe over a small weight grid), then
score the chosen blend ONCE on the untouched TEST window. The factory never sees test data.

Weight grid (the ONLY knob tuned): the 3 component weights are drawn from a small simplex
grid {0, 0.25, 0.5, 0.75, 1.0} normalized to sum to 1, EXCLUDING the all-zero point and
requiring at least two non-zero components (we are testing the COMBO hypothesis, not letting
the tuner collapse onto a single factor). All other knobs are FIXED a priori:
  - quintile (top_q = bottom_q = 0.2)
  - monthly rebalance (rebalance_freq = 21)
  - lookbacks: mom 252/skip 21, reversal 21, low-vol 63
  - min_names_per_side = 5

Config count is reported honestly. The composite-score sign / ranking is fixed; only the
blend mixture is searched, on TRAIN only.

=============================================================================
EDGE GATE (same brutal standard that found 0 edges intraday)
=============================================================================
survives_oos = TRUE only if, on the CONCATENATED OOS days:
  alpha_tstat >= 2  AND  |beta| < 0.15  AND  positive in a MAJORITY of folds  AND
  it survives the 5bp/side base cost.
Cost sensitivity reported at 2 / 5 / 10 bp. SURVIVORSHIP/LOOK-AHEAD bias is acknowledged
(today's large-caps applied backward inflates results) — marginal edges treated skeptically.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.fac_combo
"""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from experiments.factors.factor_harness import (
    RESULTS_DIR,
    backtest_xsection,
    evaluate_walkforward,
    form_dollar_neutral_portfolio,
    load_daily,
    spy_daily_returns,
)
from experiments.factors.fetch_daily import rebuild_coverage_from_cache

# ----------------------------------------------------------------------------- #
# FIXED knobs (chosen a priori, NOT tuned)
# ----------------------------------------------------------------------------- #
MOM_LOOKBACK = 252      # ~12 months
MOM_SKIP = 21           # skip most recent ~1 month
REV_LOOKBACK = 21       # ~1 month short-term reversal
VOL_LOOKBACK = 63       # ~3 months trailing vol for low-vol
TOP_Q = 0.2
BOTTOM_Q = 0.2
REBALANCE_FREQ = 21     # monthly
MIN_NAMES_PER_SIDE = 5
N_FOLDS = 4
TRAIN_FRAC = 0.4
BASE_COST_BPS = 5.0


# ----------------------------------------------------------------------------- #
# Component factor scores (each: only info up to day d's close; NaN-safe)
# ----------------------------------------------------------------------------- #
def _zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (per-row) z-score: (x - row_mean) / row_std, NaNs kept NaN."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0)
    z = df.sub(mu, axis=0).div(sd.where(sd > 0, np.nan), axis=0)
    return z


def momentum_score(panel) -> pd.DataFrame:
    """12-1 momentum: total return from d-252 to d-21 (skip the most recent month).

    Uses close prices only; the most recent price used is close_{d-21}, so it cannot peek at
    the last month and is comfortably look-ahead-free at day d.
    """
    close = panel.close
    past = close.shift(MOM_SKIP)             # close_{d-21}
    older = close.shift(MOM_LOOKBACK)        # close_{d-252}
    mom = past / older - 1.0
    return mom


def reversal_score(panel) -> pd.DataFrame:
    """Short-term (1-month) reversal: -(close_d / close_{d-21} - 1). Buy recent losers.

    Uses close_d (decided at close of d) and close_{d-21}: look-ahead-free at day d.
    """
    close = panel.close
    r_1m = close / close.shift(REV_LOOKBACK) - 1.0
    return -r_1m


def lowvol_score(panel) -> pd.DataFrame:
    """Low-vol: -trailing 63d std of daily returns (returns up to & including day d)."""
    vol = panel.rets.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK // 2).std()
    return -vol


def combo_factor_fn(weights: tuple[float, float, float]):
    """Return a factor_fn(panel) -> composite z-blended score for a given (w_mom,w_rev,w_lv).

    Each component is cross-sectionally z-scored per day, then blended. A name is scored on a
    day only if ALL three components are non-NaN there (so the blend is on a common basis);
    otherwise NaN -> excluded from that day's ranking. Look-ahead-free by the component defs.
    """
    w_mom, w_rev, w_lv = weights

    def factor_fn(panel) -> pd.DataFrame:
        z_mom = _zscore_xs(momentum_score(panel))
        z_rev = _zscore_xs(reversal_score(panel))
        z_lv = _zscore_xs(lowvol_score(panel))
        composite = w_mom * z_mom + w_rev * z_rev + w_lv * z_lv
        # require all three present (common basis); else drop the name that day
        valid = z_mom.notna() & z_rev.notna() & z_lv.notna()
        return composite.where(valid)

    return factor_fn


# ----------------------------------------------------------------------------- #
# Blend-weight grid (the ONLY tuned knob) — simplex grid, >=2 non-zero comps
# ----------------------------------------------------------------------------- #
def _weight_grid() -> list[tuple[float, float, float]]:
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = []
    for a, b, c in product(levels, repeat=3):
        s = a + b + c
        if s <= 0:
            continue
        nz = (a > 0) + (b > 0) + (c > 0)
        if nz < 2:           # combo hypothesis: require at least two factors active
            continue
        w = (round(a / s, 6), round(b / s, 6), round(c / s, 6))
        if w not in grid:
            grid.append(w)
    return grid


WEIGHT_GRID = _weight_grid()


# ----------------------------------------------------------------------------- #
# TRAIN-only tuning -> a fold factory (scores OOS once on the untouched TEST)
# ----------------------------------------------------------------------------- #
def _train_sharpe(panel, train_idx, weights: tuple[float, float, float]) -> float:
    """Net Sharpe of the combo blend evaluated ON THE TRAIN WINDOW ONLY (in-sample fit).

    Scores the full train panel, forms monthly-rebalanced dollar-neutral weights, backtests
    NET of 5bp on the train dates, returns annualized Sharpe. Used purely to RANK candidate
    blend weights on TRAIN; the winner is then scored ONCE on the held-out TEST by the harness.
    """
    factor_fn = combo_factor_fn(weights)
    scores = factor_fn(panel)            # panel already restricted to <= train_end by caller
    scores_tr = scores.reindex(train_idx)
    w = form_dollar_neutral_portfolio(
        scores_tr, top_q=TOP_Q, bottom_q=BOTTOM_Q, min_names_per_side=MIN_NAMES_PER_SIDE
    )
    bt = backtest_xsection(
        weights_over_time=w,
        returns=panel.rets.reindex(train_idx),
        rebalance_freq=REBALANCE_FREQ,
        cost_bps_per_side=BASE_COST_BPS,
    )
    return float(bt["metrics"]["sharpe"])


def make_factory():
    """Build the harness FACTORY: (train_panel, fold_info) -> tuned factor_fn (TRAIN-only)."""

    def factory(train_panel, fold_info):
        train_idx = fold_info["_train_dates"]
        best_w = None
        best_sharpe = -np.inf
        for w in WEIGHT_GRID:
            s = _train_sharpe(train_panel, train_idx, w)
            if np.isfinite(s) and s > best_sharpe:
                best_sharpe = s
                best_w = w
        if best_w is None:
            best_w = (1 / 3, 1 / 3, 1 / 3)
        factor_fn = combo_factor_fn(best_w)
        factor_fn.chosen_params = {
            "w_mom": best_w[0],
            "w_rev": best_w[1],
            "w_lowvol": best_w[2],
            "train_sharpe": round(best_sharpe, 4),
        }
        return factor_fn

    factory.is_factory = True
    return factory


# ----------------------------------------------------------------------------- #
# Run: walk-forward, tuned-on-TRAIN, scored OOS once, cost sensitivity, decompose
# ----------------------------------------------------------------------------- #
def main():
    cov = rebuild_coverage_from_cache()
    full = cov["names_full_history"]
    panel = load_daily(full)
    spy = spy_daily_returns()

    # Base case: 5bp/side. Factory tunes blend on each fold's TRAIN, scores OOS once.
    res_base = evaluate_walkforward(
        make_factory(),
        panel,
        spy,
        n_folds=N_FOLDS,
        cost_bps=BASE_COST_BPS,
        top_q=TOP_Q,
        bottom_q=BOTTOM_Q,
        rebalance_freq=REBALANCE_FREQ,
        min_names_per_side=MIN_NAMES_PER_SIDE,
        train_frac=TRAIN_FRAC,
    )

    # Cost sensitivity at 0 / 2 / 5 / 10 bp (same TRAIN-tuned blends each fold).
    cost_curve = {}
    for bps in (0.0, 2.0, 5.0, 10.0):
        r = evaluate_walkforward(
            make_factory(),
            panel,
            spy,
            n_folds=N_FOLDS,
            cost_bps=bps,
            top_q=TOP_Q,
            bottom_q=BOTTOM_Q,
            rebalance_freq=REBALANCE_FREQ,
            min_names_per_side=MIN_NAMES_PER_SIDE,
            train_frac=TRAIN_FRAC,
        )
        bd = r["beta_decompose"]
        cost_curve[f"{bps:.0f}bp"] = {
            "oos_annual_return": r["oos_annual_return"],
            "oos_sharpe": r["oos_sharpe"],
            "alpha_annual": bd["alpha_annual"],
            "alpha_tstat": bd["alpha_tstat"],
            "beta": bd["beta"],
            "oos_total_return": r["oos_total_return"],
        }

    bd = res_base["beta_decompose"]
    n_folds = res_base["n_folds"]
    folds_positive = res_base["folds_positive"]

    # Config count: a fresh weight-grid search on each fold's TRAIN, OOS scored ONCE.
    n_configs = len(WEIGHT_GRID)

    # Survival gates
    def survives_at(bps_key: str) -> bool:
        c = cost_curve[bps_key]
        return c["oos_annual_return"] > 0 and c["alpha_tstat"] >= 2.0 and abs(c["beta"]) < 0.15

    survives_2 = survives_at("2bp")
    survives_5 = survives_at("5bp")
    survives_10 = survives_at("10bp")
    survives_levels = ",".join(
        lbl for lbl, ok in (("2bp", survives_2), ("5bp", survives_5), ("10bp", survives_10)) if ok
    ) or "none"

    survives_oos = bool(
        bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and folds_positive > (n_folds / 2.0)
        and survives_5
    )

    out = {
        "name": "combo_mom_rev_lowvol",
        "hypothesis": (
            "An equal-ish blend of cross-sectionally z-scored 12-1 momentum + 1-month "
            "reversal + 63d low-vol, dollar-neutral quintile L/S, monthly rebalance, is more "
            "robust OOS than any single factor. Blend weights tuned on TRAIN only; OOS once."
        ),
        "construction": (
            "Per day d (close): z-score cross-sectionally each of mom(252-skip-21), "
            "rev(-1m), lowvol(-std63); composite = w_mom*z_mom + w_rev*z_rev + w_lv*z_lv "
            "(name scored only if all 3 present). Rank composite, LONG top 20% / SHORT bottom "
            "20% equal-dollar (gross=1, net=0), monthly (21d) rebalance. Blend weights from a "
            f"{n_configs}-point simplex grid (>=2 factors active) selected by TRAIN net Sharpe "
            "per fold; scored OOS once. Costs on turnover, both sides."
        ),
        "n_configs_tried": n_configs,
        "n_folds": n_folds,
        "folds_positive": folds_positive,
        "oos_annual_return": res_base["oos_annual_return"],
        "oos_sharpe": res_base["oos_sharpe"],
        "oos_max_drawdown": res_base["oos_max_drawdown"],
        "oos_total_return": res_base["oos_total_return"],
        "oos_n_days": res_base["oos_n_days"],
        "oos_avg_turnover": res_base["oos_avg_turnover"],
        "cost_bps_base": BASE_COST_BPS,
        "alpha_annual": bd["alpha_annual"],
        "alpha_tstat": bd["alpha_tstat"],
        "beta": bd["beta"],
        "beta_tstat": bd["beta_tstat"],
        "r2": bd["r2"],
        "survives_oos": survives_oos,
        "survives_at_costs": survives_levels,
        "cost_curve": cost_curve,
        "fold_records": res_base["folds"],
        "fold_betas": res_base["fold_betas"],
        "panel": {
            "n_names": len(panel.symbols),
            "n_days": len(panel.dates),
            "start": str(panel.dates.min().date()),
            "end": str(panel.dates.max().date()),
        },
        "caveats": (
            "SURVIVORSHIP/LOOK-AHEAD: today's large-caps applied backward (no point-in-time "
            "membership on free data) inflates results. Data starts 2020-07-27 so it covers "
            "the 2022 bear + 2023-26 bull + 2025 tariff selloff but NOT 2018Q4 or the COVID "
            "crash. SPY benchmark from same IEX daily feed (daily bars full quality). Target "
            "is a small REAL edge (Sharpe ~0.5-1), not a printer."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "combo.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))

    # Console summary
    print("=" * 78)
    print("COMBO (mom + reversal + low-vol) — dollar-neutral, monthly, walk-forward OOS")
    print("=" * 78)
    print(f"Panel: {len(panel.symbols)} names, {len(panel.dates)} days, "
          f"{panel.dates.min().date()} .. {panel.dates.max().date()}")
    print(f"Weight-grid configs (TRAIN-only select): {n_configs}   folds: {n_folds}")
    print(f"OOS n_days={res_base['oos_n_days']}  avg_turnover={res_base['oos_avg_turnover']:.4f}")
    print("-" * 78)
    print("Per-fold (TRAIN-tuned blend, scored OOS once):")
    for fr, fb in zip(res_base["folds"], res_base["fold_betas"]):
        cp = fr["chosen_params"] or {}
        print(f"  fold{fr['fold']} test {fr['test'][0]}..{fr['test'][1]} "
              f"({fr['n_days']}d)  w=({cp.get('w_mom',0):.2f},{cp.get('w_rev',0):.2f},"
              f"{cp.get('w_lowvol',0):.2f})  "
              f"ret={fr['annual_return']*100:6.2f}%  Sh={fr['sharpe']:5.2f}  "
              f"alpha_t={fb['alpha_tstat']:5.2f}  beta={fb['beta']:+.3f}")
    print("-" * 78)
    print(f"OOS annual return : {res_base['oos_annual_return']*100:.2f}%")
    print(f"OOS Sharpe        : {res_base['oos_sharpe']:.3f}")
    print(f"OOS max drawdown  : {res_base['oos_max_drawdown']*100:.2f}%")
    print(f"folds positive    : {folds_positive}/{n_folds}")
    print(f"alpha (annual)    : {bd['alpha_annual']*100:.2f}%")
    print(f"alpha t-stat      : {bd['alpha_tstat']:.3f}")
    print(f"beta              : {bd['beta']:+.4f}  (t={bd['beta_tstat']:.2f}, R^2={bd['r2']:.3f})")
    print("-" * 78)
    print("Cost sensitivity (OOS, concatenated):")
    print(f"  {'bps':>5} {'ann.ret':>9} {'Sharpe':>7} {'alpha_ann':>10} {'alpha_t':>8} {'beta':>7}")
    for k, c in cost_curve.items():
        print(f"  {k:>5} {c['oos_annual_return']*100:8.2f}% {c['oos_sharpe']:7.2f} "
              f"{c['alpha_annual']*100:9.2f}% {c['alpha_tstat']:8.2f} {c['beta']:+7.3f}")
    print("-" * 78)
    print(f"survives_at_costs : {survives_levels}")
    print(f"SURVIVES OOS GATE : {survives_oos}  "
          f"(alpha_t>=2 & |beta|<0.15 & folds_pos>{n_folds/2:.0f} & survives 5bp)")
    print(f"Written: {out_path}")
    return out


if __name__ == "__main__":
    main()
