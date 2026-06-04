"""fac_momentum.py — DAILY cross-sectional 12-1 MOMENTUM factor (dollar-neutral, walk-forward).

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on the cached daily panel.

=============================================================================
HYPOTHESIS
=============================================================================
Classic cross-sectional price MOMENTUM (Jegadeesh-Titman / Carhart UMD): names with the
HIGHEST trailing 12-month return (SKIPPING the most recent 1 month to dodge short-term
reversal) keep outperforming names with the LOWEST, over the next month. We go LONG the top
quantile and SHORT the bottom quantile in equal dollars (market-neutral BY CONSTRUCTION),
rebalanced monthly (low turnover), and ask whether the spread carries a small REAL
beta-neutral alpha that survives OOS folds AND realistic 5bp/side cost.

FACTOR (no look-ahead):
    score_d(s) = close_{d - skip}(s) / close_{d - skip - lookback}(s) - 1
i.e. the cumulative return over the window ending `skip` trading days before day d and
spanning `lookback` trading days. Computed with close.shift(skip) / close.shift(skip+lookback)
- 1, so the last close used is day (d - skip): STRICTLY in the past, no peek at day-d+1
returns. Higher score => more long.

=============================================================================
HONEST METHOD (same brutal standard that found 0 edges intraday)
=============================================================================
- DOLLAR-NEUTRAL L/S by construction (top quantile long +0.5, bottom short -0.5; gross 1).
- WALK-FORWARD: n_folds sequential, NON-overlapping OOS test windows; TRAIN is everything
  strictly before each test window. We TUNE knobs (lookback 6/9/12m, skip 0/1m, quantile
  decile/quintile, rebalance/holding 21/42/63d) on each fold's TRAIN ONLY by maximizing
  TRAIN net Sharpe at the 5bp base cost, then score the chosen config on the untouched TEST
  ONCE. Honest selection: the test set is never consulted during tuning.
- BETA DECOMPOSITION on the CONCATENATED OOS days: regress strat daily ret on SPY ->
  alpha (annualized), beta, alpha t-stat, R^2. Real-edge gate: alpha_tstat >= 2,
  |beta| < 0.15, positive in a MAJORITY of folds, AND survives the 5bp base cost.
- COSTS charged on TURNOVER, both sides; base 5bp/side, sensitivity 2bp & 10bp. Monthly
  rebalance => low turnover, so momentum should bleed far less than a daily reversal.

CAVEAT (baked in): the universe is TODAY's large-caps applied backward over 2020-07..2026-06
(survivorship/look-ahead membership bias INFLATES results) and the free-tier window MISSES
2018Q4 and the 2020 COVID crash — it spans the 2022 bear + 2023-26 bull + 2025 tariff
selloff only. Momentum is also a known crash-prone factor (2009/2020 momentum crashes);
this window largely avoids the worst momentum-crash regimes, so treat any edge skeptically.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.fac_momentum
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.factors.factor_harness import (
    RESULTS_DIR,
    Panel,
    backtest_xsection,
    beta_decompose,
    evaluate_walkforward,
    form_dollar_neutral_portfolio,
    load_daily,
    spy_daily_returns,
    walk_forward_folds,
)
from experiments.factors.fetch_daily import rebuild_coverage_from_cache

NAME = "momentum"
MONTH = 21  # ~21 trading days per month

# Tuning grid (selected on TRAIN only):
#   lookback (months): 6 / 9 / 12   -> trailing-return window length
#   skip (months):     0 / 1        -> skip the most recent month (classic 12-1)
#   quantile:          0.1 / 0.2    -> decile / quintile per side
#   rebalance (days):  21 / 42 / 63 -> monthly / 2-monthly / quarterly (holding period)
LOOKBACK_MONTHS = [6, 9, 12]
SKIP_MONTHS = [0, 1]
QUANTILES = [0.1, 0.2]
REBALANCE_DAYS = [21, 42, 63]

BASE_COST_BPS = 5.0
N_FOLDS = 4
TRAIN_FRAC = 0.4
MIN_NAMES = 3


# --------------------------------------------------------------------------- #
# Factor scoring (no look-ahead): close.shift(skip) / close.shift(skip+lookback) - 1
# --------------------------------------------------------------------------- #
def make_momentum_factor(lookback_days: int, skip_days: int):
    """Return a factor_fn(panel) -> scores DataFrame for a given lookback/skip in trading days.

    score_d(s) = close_{d-skip}/close_{d-skip-lookback} - 1.  The last close used is day
    (d - skip), strictly in the past, so the score at row d never peeks at the d->d+1 return.
    """
    def factor_fn(p: Panel) -> pd.DataFrame:
        c = p.close
        recent = c.shift(skip_days)
        base = c.shift(skip_days + lookback_days)
        score = recent / base - 1.0
        return score

    factor_fn.chosen_params = {"lookback_days": lookback_days, "skip_days": skip_days}
    return factor_fn


# --------------------------------------------------------------------------- #
# TRAIN-only tuner: pick (lookback, skip, quantile, rebalance) maximizing TRAIN net Sharpe.
# --------------------------------------------------------------------------- #
def _train_sharpe(train_panel: Panel, lookback_days, skip_days, q, rebalance, spy=None) -> float:
    """Net (5bp) Sharpe of one config evaluated ON THE TRAIN PANEL ONLY (selection metric).

    Builds scores over the train panel, forms dollar-neutral weights, backtests NET of the
    base cost on the train dates. No test data is ever touched here.
    """
    factor_fn = make_momentum_factor(lookback_days, skip_days)
    scores = factor_fn(train_panel)
    # need at least the lookback to have any scoreable rows
    valid = scores.dropna(how="all")
    if len(valid) < rebalance * 2 + 5:
        return -np.inf
    weights = form_dollar_neutral_portfolio(
        scores, top_q=q, bottom_q=q, min_names_per_side=MIN_NAMES
    )
    bt = backtest_xsection(
        weights_over_time=weights,
        returns=train_panel.rets,
        rebalance_freq=rebalance,
        cost_bps_per_side=BASE_COST_BPS,
    )
    sh = bt["metrics"]["sharpe"]
    return float(sh) if np.isfinite(sh) else -np.inf


def make_factory():
    """A walk-forward FACTORY: (train_panel, fold_info) -> tuned factor_fn (+chosen_params).

    Tunes lookback/skip/quantile/rebalance on the fold's TRAIN ONLY (max TRAIN net Sharpe at
    5bp), stashes the winning quantile & rebalance on the returned factor_fn so the caller can
    read them, and records every config's TRAIN score for the honest config-count audit.
    """
    grid = list(itertools.product(LOOKBACK_MONTHS, SKIP_MONTHS, QUANTILES, REBALANCE_DAYS))

    def factory(train_panel: Panel, fold_info: dict):
        best = None
        best_sh = -np.inf
        for lb_m, sk_m, q, reb in grid:
            sh = _train_sharpe(train_panel, lb_m * MONTH, sk_m * MONTH, q, reb)
            if sh > best_sh:
                best_sh = sh
                best = (lb_m, sk_m, q, reb)
        lb_m, sk_m, q, reb = best
        factor_fn = make_momentum_factor(lb_m * MONTH, sk_m * MONTH)
        factor_fn.chosen_params = {
            "lookback_months": lb_m,
            "skip_months": sk_m,
            "quantile": q,
            "rebalance_days": reb,
            "train_sharpe": round(best_sh, 4),
        }
        # carry tuned construction knobs so evaluate_walkforward can use them per fold
        factor_fn._top_q = q
        factor_fn._bottom_q = q
        factor_fn._rebalance = reb
        return factor_fn

    factory.is_factory = True
    factory.n_configs = len(grid)
    return factory


# --------------------------------------------------------------------------- #
# Walk-forward with PER-FOLD tuned quantile & rebalance.
# evaluate_walkforward uses fixed top_q/rebalance for ALL folds, so to honor per-fold tuning
# we run the walk-forward loop here ourselves (mirroring the harness) and let each fold's
# tuned q/rebalance flow into form_dollar_neutral_portfolio + backtest_xsection.
# --------------------------------------------------------------------------- #
def _slice_panel(panel: Panel, idx: pd.DatetimeIndex) -> Panel:
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


def _extend_one_day(all_dates: pd.DatetimeIndex, window: pd.DatetimeIndex) -> pd.DatetimeIndex:
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    last = window[-1]
    after = all_dates[all_dates > last]
    if len(after):
        return window.append(pd.DatetimeIndex([after[0]]))
    return window


def evaluate_perfold_tuned(panel: Panel, spy: pd.Series, cost_bps: float) -> dict:
    """Walk-forward eval honoring each fold's TRAIN-tuned (q, rebalance) AND (lookback, skip).

    For each fold:
      1) tune the full config on TRAIN only (max TRAIN net-Sharpe at 5bp),
      2) score the chosen lookback/skip on the panel up to test_end (factor needs lookback),
      3) form dollar-neutral weights with the tuned quantile,
      4) backtest NET of `cost_bps` over the TEST slice with the tuned rebalance freq.
    Concatenate OOS days; beta-decompose vs SPY. Returns the same shape as
    evaluate_walkforward plus chosen_params per fold.
    """
    factory = make_factory()
    folds = walk_forward_folds(list(panel.dates), n_folds=N_FOLDS, train_frac=TRAIN_FRAC)

    fold_records, oos_pieces, fold_betas = [], [], []
    positive = 0
    total_to_w = 0.0
    total_days = 0

    for f in folds:
        train_idx, test_idx = f["_train_dates"], f["_test_dates"]
        train_panel = _slice_panel(panel, panel.dates[panel.dates <= train_idx[-1]])
        factor_fn = factory(train_panel, f)
        q = factor_fn._top_q
        reb = factor_fn._rebalance
        chosen = factor_fn.chosen_params

        upto = panel.dates[panel.dates <= test_idx[-1]]
        scores = factor_fn(_slice_panel(panel, upto))
        scores_test = scores.reindex(test_idx)
        weights = form_dollar_neutral_portfolio(
            scores_test, top_q=q, bottom_q=q, min_names_per_side=MIN_NAMES
        )
        rets_window = panel.rets.reindex(_extend_one_day(panel.dates, test_idx))
        bt = backtest_xsection(
            weights_over_time=weights,
            returns=rets_window,
            rebalance_freq=reb,
            cost_bps_per_side=cost_bps,
        )
        daily = bt["daily"]
        if len(daily) and float(daily.iloc[0]) == 0.0 and float(bt["turnover"].iloc[0]) == 0.0:
            daily = daily.iloc[1:]
        m = bt["metrics"]
        tot = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
        if tot > 0:
            positive += 1
        oos_pieces.append(daily)
        total_to_w += m["avg_turnover"] * m["n_days"]
        total_days += m["n_days"]

        spy_fold = spy.reindex(daily.index).dropna() if len(daily) else spy.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append({
            "fold": f["fold"], "train": f["train"], "test": f["test"],
            "n_days": m["n_days"], "total_return": tot, "annual_return": m["annual_return"],
            "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
            "avg_turnover": m["avg_turnover"], "chosen_params": chosen,
        })

    oos = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    n = len(oos)
    mean = float(oos.mean()) if n else 0.0
    std = float(oos.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0.0
    if n:
        eq = (1.0 + oos).cumprod()
        max_dd = float((eq / eq.cummax() - 1.0).min())
    else:
        max_dd = 0.0
    spy_oos = spy.reindex(oos.index).dropna() if n else spy.iloc[0:0]
    beta_full = beta_decompose(oos, spy_oos)

    return {
        "n_folds": len(folds),
        "folds": fold_records,
        "folds_positive": positive,
        "oos_annual_return": mean * 252,
        "oos_sharpe": sharpe,
        "oos_max_drawdown": max_dd,
        "oos_total_return": float((1.0 + oos).prod() - 1.0) if n else 0.0,
        "oos_n_days": int(n),
        "oos_avg_turnover": (total_to_w / total_days) if total_days else 0.0,
        "cost_bps_per_side": cost_bps,
        "beta_decompose": beta_full,
        "fold_betas": fold_betas,
        "n_configs_per_fold": factory.n_configs,
        "_oos_daily": oos,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cov = rebuild_coverage_from_cache()
    full = cov["names_full_history"]
    panel = load_daily(full)
    spy = spy_daily_returns()

    # Base case: per-fold-tuned, 5bp.
    res5 = evaluate_perfold_tuned(panel, spy, cost_bps=BASE_COST_BPS)
    # Cost sensitivity: re-score the SAME per-fold-tuned configs at 2bp and 10bp.
    res2 = evaluate_perfold_tuned(panel, spy, cost_bps=2.0)
    res10 = evaluate_perfold_tuned(panel, spy, cost_bps=10.0)

    bd = res5["beta_decompose"]
    survives_at = []
    for label, r in (("2bp", res2), ("5bp", res5), ("10bp", res10)):
        if r["beta_decompose"]["alpha_annual"] > 0 and r["oos_annual_return"] > 0:
            survives_at.append(label)

    survives_oos = (
        bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and res5["folds_positive"] > res5["n_folds"] / 2.0
        and ("5bp" in survives_at)
    )

    out = {
        "name": NAME,
        "hypothesis": (
            "Cross-sectional 12-1 price momentum: long top-quantile trailing-return names, "
            "short bottom-quantile (dollar-neutral), monthly rebalance. Tests for a small "
            "beta-neutral alpha surviving walk-forward OOS + 5bp/side cost."
        ),
        "construction": (
            "score = close.shift(skip)/close.shift(skip+lookback)-1 (trailing return skipping "
            "most recent month); dollar-neutral L/S top vs bottom quantile, equal-weight each "
            "leg; per-fold TRAIN-tuned lookback/skip/quantile/rebalance (max TRAIN net Sharpe "
            "@5bp), scored OOS once."
        ),
        "data_window": f"{str(panel.dates.min().date())}..{str(panel.dates.max().date())}",
        "n_names": len(panel.symbols),
        "n_trading_days": len(panel.dates),
        "n_configs_per_fold": res5["n_configs_per_fold"],
        "n_configs_tried_total": res5["n_configs_per_fold"] * res5["n_folds"],
        "n_folds": res5["n_folds"],
        "folds_positive": res5["folds_positive"],
        "base_cost_bps": BASE_COST_BPS,
        "base_5bp": {
            "oos_annual_return": res5["oos_annual_return"],
            "oos_sharpe": res5["oos_sharpe"],
            "oos_max_drawdown": res5["oos_max_drawdown"],
            "oos_total_return": res5["oos_total_return"],
            "oos_n_days": res5["oos_n_days"],
            "oos_avg_turnover": res5["oos_avg_turnover"],
            "alpha_annual": bd["alpha_annual"],
            "alpha_tstat": bd["alpha_tstat"],
            "beta": bd["beta"],
            "beta_tstat": bd["beta_tstat"],
            "r2": bd["r2"],
        },
        "cost_sensitivity": {
            "2bp": {
                "oos_annual_return": res2["oos_annual_return"],
                "oos_sharpe": res2["oos_sharpe"],
                "alpha_annual": res2["beta_decompose"]["alpha_annual"],
                "alpha_tstat": res2["beta_decompose"]["alpha_tstat"],
            },
            "5bp": {
                "oos_annual_return": res5["oos_annual_return"],
                "oos_sharpe": res5["oos_sharpe"],
                "alpha_annual": bd["alpha_annual"],
                "alpha_tstat": bd["alpha_tstat"],
            },
            "10bp": {
                "oos_annual_return": res10["oos_annual_return"],
                "oos_sharpe": res10["oos_sharpe"],
                "alpha_annual": res10["beta_decompose"]["alpha_annual"],
                "alpha_tstat": res10["beta_decompose"]["alpha_tstat"],
            },
        },
        "survives_at_costs": ",".join(survives_at) if survives_at else "none",
        "survives_oos": bool(survives_oos),
        "fold_betas": [
            {k: v for k, v in fb.items() if k in ("fold", "alpha_annual", "alpha_tstat", "beta", "n_days")}
            for fb in res5["fold_betas"]
        ],
        "folds": [
            {
                "fold": fr["fold"], "test": fr["test"], "n_days": fr["n_days"],
                "annual_return": fr["annual_return"], "sharpe": fr["sharpe"],
                "max_drawdown": fr["max_drawdown"], "avg_turnover": fr["avg_turnover"],
                "chosen_params": fr["chosen_params"],
            }
            for fr in res5["folds"]
        ],
        "caveats": (
            "Universe = today's large-caps applied backward (survivorship/membership bias "
            "INFLATES results). Free-tier window 2020-07..2026-06 MISSES 2018Q4 and the 2020 "
            "COVID crash; spans 2022 bear + 2023-26 bull + 2025 tariff selloff. Momentum is "
            "crash-prone (2009/2020 momentum crashes) and this window largely dodges those "
            "regimes. Daily OLS t-stats are not regime-robust over ~4y OOS."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{NAME}.json").write_text(json.dumps(out, indent=2, default=str))

    # console summary
    print(f"=== fac_{NAME} ===")
    print(f"window {out['data_window']}  names={out['n_names']}  days={out['n_trading_days']}")
    print(f"configs/fold={out['n_configs_per_fold']}  folds={out['n_folds']}  "
          f"total configs tried={out['n_configs_tried_total']}")
    b = out["base_5bp"]
    print(f"\nBASE 5bp: ann.ret={b['oos_annual_return']*100:.2f}%  Sharpe={b['oos_sharpe']:.2f}  "
          f"maxDD={b['oos_max_drawdown']*100:.2f}%  turnover={b['oos_avg_turnover']:.4f}")
    print(f"          alpha(ann)={b['alpha_annual']*100:.2f}%  alpha_t={b['alpha_tstat']:.2f}  "
          f"beta={b['beta']:.3f}  R2={b['r2']:.3f}")
    print(f"folds positive: {out['folds_positive']}/{out['n_folds']}")
    print("\nper-fold:")
    for fr in out["folds"]:
        cp = fr["chosen_params"]
        print(f"  fold {fr['fold']} {fr['test'][0]}..{fr['test'][1]} ({fr['n_days']}d): "
              f"ann={fr['annual_return']*100:6.2f}% Sh={fr['sharpe']:5.2f} "
              f"to={fr['avg_turnover']:.3f}  picked lb={cp['lookback_months']}m "
              f"skip={cp['skip_months']}m q={cp['quantile']} reb={cp['rebalance_days']}d "
              f"(trainSh={cp['train_sharpe']})")
    print("\ncost sensitivity (alpha ann / alpha t):")
    for lab in ("2bp", "5bp", "10bp"):
        cs = out["cost_sensitivity"][lab]
        print(f"  {lab:>4}: ann.ret={cs['oos_annual_return']*100:6.2f}%  Sh={cs['oos_sharpe']:5.2f}  "
              f"alpha={cs['alpha_annual']*100:6.2f}%  t={cs['alpha_tstat']:.2f}")
    print(f"\nsurvives at costs: {out['survives_at_costs']}")
    print(f"SURVIVES OOS (t>=2, |beta|<0.15, majority folds +, 5bp): {out['survives_oos']}")
    print("\nwritten to", RESULTS_DIR / f"{NAME}.json")
    return out


if __name__ == "__main__":
    main()
