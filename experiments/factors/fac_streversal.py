"""fac_streversal.py — SHORT-TERM REVERSAL daily cross-sectional factor.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on cached daily bars.

=============================================================================
HYPOTHESIS
=============================================================================
Short-term (1-week) reversal: stocks that have recently UNDERPERFORMED their
cross-section tend to bounce back, and recent OUTPERFORMERS tend to give it back,
over a ~1-day-to-1-week horizon. So the cross-sectional factor score is the
NEGATIVE of the trailing N-day return:

    score_d(s) = - ( close_d(s) / close_{d-N}(s) - 1 )

We go LONG the top quantile (biggest recent LOSERS) and SHORT the bottom quantile
(biggest recent WINNERS), in equal dollars => dollar-neutral by construction, so
any edge is (close to) market-neutral alpha, not beta.

This is the canonical Lehmann (1990) / Lo-MacKinlay (1990) contrarian effect. It is
known to be REAL in gross terms but HIGH TURNOVER — it rebalances against last
week's moves constantly. The whole question this experiment answers is: does ANY of
the gross reversal alpha survive a realistic 5bp/side slippage charge once we honestly
account for turnover? Be brutal about cost; report 2/5/10bp sensitivity.

=============================================================================
WHAT IS TUNED (TRAIN-only) vs SCORED (OOS once)
=============================================================================
Per walk-forward fold, on that fold's TRAIN window ONLY we grid-search:
    lookback   in {3, 5, 10} trading days   (the reversal horizon)
    quantile   in {0.1, 0.2, 0.3}           (top/bottom fraction per side)
    rebalance  in {1, 3, 5} trading days     (how often we trade — turnover knob)
selecting the (lookback, quantile, rebalance) with the best NET-of-5bp TRAIN Sharpe.
That single chosen config is then scored ONCE on the untouched TEST slice. The OOS
days from all folds are concatenated and beta-decomposed vs SPY.

Config count: 3 x 3 x 3 = 27 candidates evaluated per fold's TRAIN (TRAIN-only
selection); each fold's TEST is scored exactly once. We do NOT peek at TEST to pick
knobs.

=============================================================================
HONESTY / BIASES (read before trusting any number)
=============================================================================
- SURVIVORSHIP: the universe is TODAY's ~97 large-caps applied backward. Names that
  were large then but later dropped out are ABSENT; names that grew into large-cap are
  present for their whole window. This INFLATES results. Treat marginal edges skeptically.
- REGIME COVERAGE: free-tier daily history starts ~2020-07-27, so this sample does NOT
  contain 2018 Q4 or the Feb/Mar-2020 COVID crash. It DOES contain the 2022 bear and the
  2023-26 bull. Reversal is historically strongest in high-volatility/crisis windows, so
  missing 2020 likely UNDERSTATES gross reversal alpha while the bull-heavy sample is a
  fair test of "does it survive in calm regimes too."
- HIGH TURNOVER: a 1-day-rebalanced reversal trades ~1.5x notional/day. At 5bp/side that
  is brutal. The verdict hinges almost entirely on cost; this file reports it explicitly.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.fac_streversal
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.factors.factor_harness import (
    RESULTS_DIR,
    Panel,
    backtest_xsection,
    evaluate_walkforward,
    form_dollar_neutral_portfolio,
    load_daily,
    spy_daily_returns,
    walk_forward_folds,
)
from experiments.factors.fetch_daily import rebuild_coverage_from_cache

NAME = "streversal"

# Tunable grids (selected on TRAIN only).
LOOKBACKS = [3, 5, 10]
QUANTILES = [0.1, 0.2, 0.3]
REBALANCES = [1, 3, 5]

N_FOLDS = 4
TRAIN_FRAC = 0.4
BASE_COST_BPS = 5.0
COST_GRID = [2.0, 5.0, 10.0]
MIN_NAMES_PER_SIDE = 3


# --------------------------------------------------------------------------- #
# The factor: score = -trailing `lookback`-day return (buy recent losers).
# --------------------------------------------------------------------------- #
def make_reversal_factor(lookback: int):
    """Return a factor_fn(panel) -> scores (date x symbol) = -trailing lookback-day return.

    score_d(s) = -(close_d / close_{d-lookback} - 1). Uses ONLY closes up to and including
    day d (no look-ahead): close_d is the most recent close, close_{d-lookback} is `lookback`
    trading days earlier. Rows with insufficient lookback are NaN (excluded from ranking).
    """

    def factor_fn(p: Panel) -> pd.DataFrame:
        c = p.close
        trailing_ret = c / c.shift(lookback) - 1.0
        # Negative of trailing return => recent losers score HIGH => go long.
        return -trailing_ret

    return factor_fn


# --------------------------------------------------------------------------- #
# TRAIN-only knob selection -> factory the harness calls per fold.
# --------------------------------------------------------------------------- #
def _train_sharpe(panel: Panel, train_dates, lookback, quantile, rebalance, cost_bps):
    """NET-of-cost Sharpe of the reversal config over the TRAIN window only.

    Scores are computed on the panel up to the last train date (factor needs lookback
    history that predates train_start, which is fine — it is all in-sample TRAIN data), then
    restricted to the train dates, turned into dollar-neutral weights, and backtested NET of
    cost on the train window extended by one day. NO test data is touched.
    """
    factor_fn = make_reversal_factor(lookback)
    train_idx = pd.DatetimeIndex(train_dates)
    upto = panel.dates[panel.dates <= train_idx[-1]]
    scores = factor_fn(_slice_close_only(panel, upto)).reindex(train_idx)
    weights = form_dollar_neutral_portfolio(
        scores, top_q=quantile, bottom_q=quantile, min_names_per_side=MIN_NAMES_PER_SIDE
    )
    # Extend by one day so the last train weight earns its t+1 realization (stays in TRAIN
    # region — the day after the last train date is still < the test window start).
    ext = _extend_one(panel.dates, train_idx)
    rets_window = panel.rets.reindex(ext)
    bt = backtest_xsection(weights, rets_window, rebalance_freq=rebalance, cost_bps_per_side=cost_bps)
    return bt["metrics"]["sharpe"]


def make_factory():
    """Build the factory the harness calls once per fold with (train_panel, fold_info).

    On the fold's TRAIN, grid-search (lookback, quantile, rebalance) for the best NET-of-5bp
    TRAIN Sharpe; attach the chosen config as `.chosen_params`; return the factor_fn (the
    harness scores it ONCE on the untouched TEST and forms weights with the chosen quantile).

    NOTE on quantile/rebalance: the harness's evaluate_walkforward forms weights with the
    top_q/bottom_q/rebalance_freq passed to IT, not per-fold. So we cannot let the harness
    apply a per-fold quantile/rebalance directly. Instead the factory records the chosen
    knobs for reporting, and we run evaluate_walkforward with the MODAL chosen quantile/
    rebalance across folds (computed in run()). The lookback IS applied per-fold (it only
    changes the score, which the factory fully controls). This keeps selection honest:
    quantile/rebalance are still chosen on TRAIN only; we just need a single value to hand the
    harness, so we use the most-frequently-selected one. See run() for the two-pass design.
    """

    def factory(train_panel: Panel, fold_info: dict):
        train_idx = train_panel.dates
        best = None
        for lb in LOOKBACKS:
            for q in QUANTILES:
                for rb in REBALANCES:
                    s = _train_sharpe(train_panel, train_idx, lb, q, rb, BASE_COST_BPS)
                    if best is None or s > best[0]:
                        best = (s, lb, q, rb)
        _, lb, q, rb = best
        fn = make_reversal_factor(lb)
        fn.chosen_params = {"lookback": lb, "quantile": q, "rebalance": rb, "train_sharpe": best[0]}
        fn.is_factory = False
        return fn

    factory.is_factory = True
    return factory


# --------------------------------------------------------------------------- #
# Local panel helpers (avoid importing private harness helpers).
# --------------------------------------------------------------------------- #
def _slice_close_only(panel: Panel, idx: pd.DatetimeIndex) -> Panel:
    idx = panel.dates.intersection(pd.DatetimeIndex(idx))
    return Panel(
        close=panel.close.reindex(idx),
        open=panel.open.reindex(idx),
        high=panel.high.reindex(idx),
        low=panel.low.reindex(idx),
        volume=panel.volume.reindex(idx),
        rets=panel.rets.reindex(idx),
        symbols=panel.symbols,
        dates=idx,
    )


def _extend_one(all_dates: pd.DatetimeIndex, window: pd.DatetimeIndex) -> pd.DatetimeIndex:
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    last = window[-1]
    after = all_dates[all_dates > last]
    if len(after):
        return window.append(pd.DatetimeIndex([after[0]]))
    return window


# --------------------------------------------------------------------------- #
# Honest two-pass driver
# --------------------------------------------------------------------------- #
def run():
    cov = rebuild_coverage_from_cache()
    full = cov["names_full_history"]
    panel = load_daily(full)
    spy = spy_daily_returns()

    folds = walk_forward_folds(list(panel.dates), n_folds=N_FOLDS, train_frac=TRAIN_FRAC)

    # ---- PASS 1: honest TRAIN-only selection of (lookback, quantile, rebalance) per fold.
    # We run the grid search ourselves (the harness's evaluate_walkforward applies ONE global
    # quantile/rebalance, so to honor per-fold selection we choose knobs here, record them, and
    # then drive the harness with the MODAL chosen quantile+rebalance — the lookback is applied
    # truly per-fold via the factory since it only alters the score the factory returns).
    per_fold_choice = []
    for f in folds:
        train_idx = f["_train_dates"]
        best = None
        for lb in LOOKBACKS:
            for q in QUANTILES:
                for rb in REBALANCES:
                    s = _train_sharpe(panel, train_idx, lb, q, rb, BASE_COST_BPS)
                    if best is None or s > best[0]:
                        best = (s, lb, q, rb)
        _, lb, q, rb = best
        per_fold_choice.append({"fold": f["fold"], "lookback": lb, "quantile": q,
                                "rebalance": rb, "train_sharpe": best[0]})

    # Modal quantile & rebalance across folds (single value to hand the harness). Lookback is
    # applied per-fold by the factory; quantile/rebalance use the modal TRAIN choice.
    def _modal(key):
        vals = [c[key] for c in per_fold_choice]
        return max(set(vals), key=vals.count)

    modal_q = _modal("quantile")
    modal_rb = _modal("rebalance")

    # ---- PASS 2: score OOS once via the harness, lookback chosen per-fold on TRAIN.
    factory = make_factory()
    res = evaluate_walkforward(
        factory, panel, spy,
        n_folds=N_FOLDS, cost_bps=BASE_COST_BPS,
        top_q=modal_q, bottom_q=modal_q, rebalance_freq=modal_rb,
        min_names_per_side=MIN_NAMES_PER_SIDE, train_frac=TRAIN_FRAC,
    )

    # ---- Cost sensitivity: re-run OOS at 2/5/10bp (same per-fold TRAIN selection at base 5bp).
    cost_results = {}
    for bps in COST_GRID:
        r = evaluate_walkforward(
            make_factory(), panel, spy,
            n_folds=N_FOLDS, cost_bps=bps,
            top_q=modal_q, bottom_q=modal_q, rebalance_freq=modal_rb,
            min_names_per_side=MIN_NAMES_PER_SIDE, train_frac=TRAIN_FRAC,
        )
        bd = r["beta_decompose"]
        cost_results[str(int(bps))] = {
            "oos_annual_return": r["oos_annual_return"],
            "oos_sharpe": r["oos_sharpe"],
            "oos_max_drawdown": r["oos_max_drawdown"],
            "alpha_annual": bd["alpha_annual"],
            "alpha_tstat": bd["alpha_tstat"],
            "beta": bd["beta"],
            "folds_positive": r["folds_positive"],
        }

    # ---- Gross (0bp) for context: how much reversal alpha exists BEFORE cost.
    gross = evaluate_walkforward(
        make_factory(), panel, spy,
        n_folds=N_FOLDS, cost_bps=0.0,
        top_q=modal_q, bottom_q=modal_q, rebalance_freq=modal_rb,
        min_names_per_side=MIN_NAMES_PER_SIDE, train_frac=TRAIN_FRAC,
    )

    bd = res["beta_decompose"]
    n_configs = N_FOLDS * len(LOOKBACKS) * len(QUANTILES) * len(REBALANCES)

    # ---- Verdict gates (base 5bp).
    survives = (
        bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and res["folds_positive"] > (res["n_folds"] / 2.0)
        and cost_results["5"]["oos_annual_return"] > 0.0
    )

    survives_at = [bps for bps in ("2", "5", "10") if cost_results[bps]["oos_annual_return"] > 0.0]

    out = {
        "name": NAME,
        "hypothesis": (
            "Short-term reversal: cross-sectional score = -(trailing N-day return). Long recent "
            "losers, short recent winners, dollar-neutral. Edge is contrarian mean-reversion; the "
            "open question is whether high turnover lets any gross alpha survive 5bp/side cost."
        ),
        "construction": (
            "Daily cross-sectional dollar-neutral L/S. Rank ~97 large-caps by -trailing-return; "
            "long top quantile (losers), short bottom quantile (winners), equal dollars/side "
            "(gross 1.0, net 0). Lookback tuned per-fold on TRAIN over {3,5,10}d; quantile over "
            "{0.1,0.2,0.3}; rebalance over {1,3,5}d. Modal TRAIN quantile/rebalance handed to the "
            "harness; lookback applied per-fold. OOS scored once; concatenated OOS days "
            "beta-decomposed vs SPY."
        ),
        "n_configs_tried": n_configs,
        "n_folds": res["n_folds"],
        "folds_positive": res["folds_positive"],
        "cost_bps_base": BASE_COST_BPS,
        "modal_quantile": modal_q,
        "modal_rebalance": modal_rb,
        "per_fold_choice": per_fold_choice,
        "oos_annual_return": res["oos_annual_return"],
        "oos_sharpe": res["oos_sharpe"],
        "oos_max_drawdown": res["oos_max_drawdown"],
        "oos_total_return": res["oos_total_return"],
        "oos_n_days": res["oos_n_days"],
        "oos_avg_turnover": res["oos_avg_turnover"],
        "alpha_annual": bd["alpha_annual"],
        "alpha_tstat": bd["alpha_tstat"],
        "beta": bd["beta"],
        "beta_tstat": bd["beta_tstat"],
        "r2": bd["r2"],
        "gross_0bp": {
            "oos_annual_return": gross["oos_annual_return"],
            "oos_sharpe": gross["oos_sharpe"],
            "alpha_annual": gross["beta_decompose"]["alpha_annual"],
            "alpha_tstat": gross["beta_decompose"]["alpha_tstat"],
            "beta": gross["beta_decompose"]["beta"],
            "avg_turnover": gross["oos_avg_turnover"],
        },
        "cost_sensitivity": cost_results,
        "survives_at_costs": survives_at,
        "survives_oos": bool(survives),
        "folds": res["folds"],
        "fold_betas": res["fold_betas"],
        "biases": (
            "Survivorship (today's large-caps backfilled -> inflates). Sample starts ~2020-07-27 "
            "so NO 2018Q4 or 2020 COVID crash (reversal is strongest in crises, so gross alpha is "
            "likely UNDERSTATED here); contains 2022 bear + 2023-26 bull. Costs charged on turnover "
            "both sides; verdict hinges on cost."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{NAME}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    # ---- Console summary.
    print(f"=== fac_{NAME} ===")
    print(f"configs tried (TRAIN-only): {n_configs}  | folds: {res['n_folds']}  "
          f"| modal q={modal_q} rebal={modal_rb}d")
    print("per-fold TRAIN choice:")
    for c in per_fold_choice:
        print(f"  fold {c['fold']}: lookback={c['lookback']}d q={c['quantile']} "
              f"rebal={c['rebalance']}d (train Sharpe {c['train_sharpe']:.2f})")
    print(f"\nGROSS (0bp): ann {gross['oos_annual_return']*100:.2f}%  "
          f"Sharpe {gross['oos_sharpe']:.2f}  alpha {gross['beta_decompose']['alpha_annual']*100:.2f}% "
          f"(t={gross['beta_decompose']['alpha_tstat']:.2f})  "
          f"beta {gross['beta_decompose']['beta']:.3f}  turnover {gross['oos_avg_turnover']:.3f}/day")
    print(f"\nOOS @5bp: ann {res['oos_annual_return']*100:.2f}%  Sharpe {res['oos_sharpe']:.2f}  "
          f"maxDD {res['oos_max_drawdown']*100:.2f}%  n_days {res['oos_n_days']}")
    print(f"  alpha {bd['alpha_annual']*100:.2f}% (t={bd['alpha_tstat']:.2f})  "
          f"beta {bd['beta']:.3f} (t={bd['beta_tstat']:.2f})  R2 {bd['r2']:.3f}  "
          f"folds_positive {res['folds_positive']}/{res['n_folds']}")
    print("\ncost sensitivity (OOS):")
    for bps in ("2", "5", "10"):
        cr = cost_results[bps]
        print(f"  {bps:>2}bp: ann {cr['oos_annual_return']*100:7.2f}%  Sharpe {cr['oos_sharpe']:6.2f}  "
              f"alpha {cr['alpha_annual']*100:7.2f}% (t={cr['alpha_tstat']:5.2f})  "
              f"beta {cr['beta']:.3f}  +folds {cr['folds_positive']}/{res['n_folds']}")
    print(f"\nsurvives at costs: {survives_at or 'NONE'}")
    print(f"SURVIVES_OOS (alpha_t>=2, |beta|<0.15, majority folds +, +@5bp): {survives}")
    print(f"\nwritten: {out_path}")
    return out


if __name__ == "__main__":
    run()
