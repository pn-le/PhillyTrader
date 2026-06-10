"""fac_lowvol.py — DAILY cross-sectional LOW-VOLATILITY / LOW-BETA anomaly experiment.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on cached daily bars.

=============================================================================
HYPOTHESIS
=============================================================================
The low-volatility anomaly: stocks with LOW trailing realized volatility tend to deliver
better risk-adjusted (and historically even better raw) returns than HIGH-vol stocks — the
opposite of what CAPM predicts. We exploit it dollar-neutral:

    score(s, d) = - trailing realized vol of s over the last `lookback` days (using returns
                   up to and INCLUDING day d's close — no look-ahead).

Higher score = LOWER vol => the construction goes LONG the lowest-vol quintile and SHORT the
highest-vol quintile, in equal dollars. Rebalanced MONTHLY (vol is slow-moving; daily rebal
would only churn cost). This is market-neutral BY DOLLAR CONSTRUCTION, but the well-known
catch is that low-vol names are typically lower-beta and high-vol names higher-beta, so a
LONG-lowvol / SHORT-highvol book carries a structurally NEGATIVE market beta. We report that
beta honestly: a genuine NEUTRAL alpha must clear |beta| < 0.15, NOT just look good because
it was short beta into a flat/down tape.

=============================================================================
METHOD (honest walk-forward, TRAIN-only selection, OOS scored once)
=============================================================================
- WALK-FORWARD: experiments.factors.walk_forward_folds builds N sequential expanding-train /
  non-overlapping-test folds over 2020-07 .. 2026-06 (covers 2022 bear, 2023-26 bull, 2025
  tariff selloff; NOT 2018Q4 or 2020 COVID — free-tier floor, acknowledged).
- TUNE on TRAIN ONLY: for each fold we grid-search {vol lookback in 20/60/120d} x
  {quantile in 0.1/0.2/0.3} x {rebalance in 21/42d} on that fold's TRAIN slice, pick the
  config with the best TRAIN net Sharpe (5bp), then score that ONE config on the untouched
  TEST slice. We NEVER look at test returns to choose knobs.
- Strict t->t+1 timing and turnover costs are delegated to factor_harness.backtest_xsection
  (the same engine the sanity suite locked). Weights decided at close of day d earn the
  d->d+1 return; cost charged on turnover at each rebalance, both sides.
- AGGREGATE: concatenate all OOS days, run beta_decompose vs SPY (the headline neutrality
  test), report per-fold + aggregate Sharpe / annual return / max DD / alpha / beta, and
  cost sensitivity at 2 / 5 / 10 bp per side.

EDGE GATE (survives_oos): alpha_tstat >= 2 AND |beta| < 0.15 AND positive in a MAJORITY of
folds AND survives the 5bp base cost. Anything less is honestly reported as NO neutral edge.

SURVIVORSHIP CAVEAT: the universe is TODAY's ~97 large-caps applied backward — survivors
only. This INFLATES results; treat any marginal edge skeptically.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.fac_lowvol
"""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from experiments.factors.factor_harness import (
    RESULTS_DIR,
    Panel,
    backtest_xsection,
    beta_decompose,
    form_dollar_neutral_portfolio,
    load_daily,
    spy_daily_returns,
    walk_forward_folds,
    _extend_one_day,
    _slice_panel,
)
from experiments.factors.fetch_daily import rebuild_coverage_from_cache

TRADING_YEAR = 252
NAME = "lowvol"

# Knob grids tuned on TRAIN only.
LOOKBACKS = [20, 60, 120]          # trailing-vol window in trading days
QUANTILES = [0.10, 0.20, 0.30]     # top/bottom fraction per side
REBALANCES = [21, 42]              # ~monthly / ~bi-monthly (vol is slow; never daily)
N_FOLDS = 4
TRAIN_FRAC = 0.40
BASE_COST_BPS = 5.0
COST_GRID = [2.0, 5.0, 10.0]
MIN_NAMES_PER_SIDE = 3


# --------------------------------------------------------------------------- #
# Factor: score = -trailing realized vol (low-vol => high score => long)
# --------------------------------------------------------------------------- #
def lowvol_scores(panel: Panel, lookback: int) -> pd.DataFrame:
    """score(d, s) = -std of s's daily returns over the trailing `lookback` days up to d.

    Uses panel.rets (daily simple close-to-close returns). The rolling std at row d uses
    returns realized OVER days <= d (rets.loc[d] is the d-1->d return), so the score at the
    CLOSE of day d sees only information available by that close — NO look-ahead. min_periods
    requires a (mostly) full window so early rows are NaN and excluded from ranking.
    """
    vol = panel.rets.rolling(window=lookback, min_periods=max(5, lookback // 2)).std()
    return -vol


def _net_sharpe_on_dates(
    panel: Panel,
    spy: pd.Series,
    score_dates: pd.DatetimeIndex,
    lookback: int,
    top_q: float,
    rebalance: int,
    cost_bps: float,
) -> tuple[float, pd.Series, dict]:
    """Backtest the lowvol config over `score_dates` (strict t->t+1, net of cost).

    Returns (net Sharpe, daily-return series indexed by realization date, backtest metrics).
    Mirrors evaluate_walkforward's per-fold mechanics exactly so TRAIN scoring and TEST
    scoring use identical timing/cost logic.
    """
    if len(score_dates) < 3:
        return 0.0, pd.Series(dtype=float), {}
    # Score on the panel up to the END of this window (full lookback available), keep only
    # the window's decision dates.
    upto = panel.dates[panel.dates <= score_dates[-1]]
    sub = _slice_panel(panel, upto)
    scores = lowvol_scores(sub, lookback).reindex(score_dates)
    weights = form_dollar_neutral_portfolio(
        scores, top_q=top_q, bottom_q=top_q, min_names_per_side=MIN_NAMES_PER_SIDE
    )
    rets_window = panel.rets.reindex(_extend_one_day(panel.dates, score_dates))
    bt = backtest_xsection(
        weights_over_time=weights,
        returns=rets_window,
        rebalance_freq=rebalance,
        cost_bps_per_side=cost_bps,
    )
    daily = bt["daily"]
    # Drop leading flat day (no book carried in), same as the harness.
    if len(daily) and float(daily.iloc[0]) == 0.0 and float(bt["turnover"].iloc[0]) == 0.0:
        daily = daily.iloc[1:]
    if len(daily) < 3:
        return 0.0, daily, bt["metrics"]
    mu = float(daily.mean())
    sd = float(daily.std(ddof=1))
    sharpe = (mu / sd * np.sqrt(TRADING_YEAR)) if sd > 0 else 0.0
    return sharpe, daily, bt["metrics"]


def _tune_on_train(panel: Panel, spy: pd.Series, train_dates: pd.DatetimeIndex) -> dict:
    """Grid-search lookback x quantile x rebalance on TRAIN ONLY; pick best TRAIN net Sharpe.

    Returns the chosen config dict. Ties broken by lower turnover (cheaper to run) then
    smaller lookback. NEVER touches test data.
    """
    best = None
    for lookback, top_q, rebalance in product(LOOKBACKS, QUANTILES, REBALANCES):
        # Need the train window to span enough days past the lookback to be meaningful.
        if len(train_dates) <= lookback + 5:
            continue
        sharpe, daily, metrics = _net_sharpe_on_dates(
            panel, spy, train_dates, lookback, top_q, rebalance, BASE_COST_BPS
        )
        turn = metrics.get("avg_turnover", 1.0) if metrics else 1.0
        key = (sharpe, -turn, -lookback)  # maximize sharpe, prefer low turnover, small lb
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "lookback": lookback,
                "top_q": top_q,
                "rebalance": rebalance,
                "train_sharpe": sharpe,
            }
    if best is None:
        # Degenerate (tiny train) — fall back to a sane default.
        best = {"lookback": 60, "top_q": 0.20, "rebalance": 21, "train_sharpe": 0.0, "key": None}
    best.pop("key", None)
    return best


# --------------------------------------------------------------------------- #
# Walk-forward driver (tune on TRAIN, score OOS once, aggregate)
# --------------------------------------------------------------------------- #
def run_walkforward(panel: Panel, spy: pd.Series, cost_bps: float, verbose: bool = False) -> dict:
    folds = walk_forward_folds(list(panel.dates), n_folds=N_FOLDS, train_frac=TRAIN_FRAC)
    fold_records, fold_betas, oos_pieces = [], [], []
    positive = 0
    tot_to_w = 0.0
    tot_days = 0
    n_configs = len(LOOKBACKS) * len(QUANTILES) * len(REBALANCES)

    for f in folds:
        train_idx = f["_train_dates"]
        test_idx = f["_test_dates"]
        chosen = _tune_on_train(panel, spy, train_idx)
        sharpe, daily, metrics = _net_sharpe_on_dates(
            panel, spy, test_idx,
            chosen["lookback"], chosen["top_q"], chosen["rebalance"], cost_bps,
        )
        tot_ret = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
        if tot_ret > 0:
            positive += 1
        oos_pieces.append(daily)
        tot_to_w += metrics.get("avg_turnover", 0.0) * metrics.get("n_days", 0)
        tot_days += metrics.get("n_days", 0)

        spy_fold = spy.reindex(daily.index).dropna() if len(daily) else spy.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append({
            "fold": f["fold"],
            "train": f["train"],
            "test": f["test"],
            "n_days": metrics.get("n_days", 0),
            "total_return": tot_ret,
            "annual_return": metrics.get("annual_return", 0.0),
            "sharpe": sharpe,
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "avg_turnover": metrics.get("avg_turnover", 0.0),
            "chosen_params": chosen,
        })
        if verbose:
            print(f"  fold {f['fold']}: train {f['train'][:2]} test {f['test'][:2]} "
                  f"chosen={chosen} OOS Sharpe={sharpe:.2f} ret={tot_ret*100:.2f}% "
                  f"beta={fb['beta']:.3f} alpha_t={fb['alpha_tstat']:.2f}")

    oos = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    n = len(oos)
    mu = float(oos.mean()) if n else 0.0
    sd = float(oos.std(ddof=1)) if n > 1 else 0.0
    oos_sharpe = (mu / sd * np.sqrt(TRADING_YEAR)) if sd > 0 else 0.0
    oos_annual = mu * TRADING_YEAR
    if n:
        eq = (1.0 + oos).cumprod()
        oos_max_dd = float((eq / eq.cummax() - 1.0).min())
    else:
        oos_max_dd = 0.0
    avg_to = (tot_to_w / tot_days) if tot_days else 0.0
    spy_oos = spy.reindex(oos.index).dropna() if n else spy.iloc[0:0]
    bd = beta_decompose(oos, spy_oos)

    return {
        "n_folds": len(folds),
        "n_configs_tried_per_fold": n_configs,
        "folds": fold_records,
        "fold_betas": fold_betas,
        "folds_positive": positive,
        "oos_annual_return": oos_annual,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": oos_max_dd,
        "oos_total_return": float((1.0 + oos).prod() - 1.0) if n else 0.0,
        "oos_n_days": n,
        "oos_avg_turnover": avg_to,
        "cost_bps_per_side": cost_bps,
        "beta_decompose": bd,
        "oos_daily": oos,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    cov = rebuild_coverage_from_cache()
    full = cov["names_full_history"]
    panel = load_daily(full)
    spy = spy_daily_returns()

    print(f"[lowvol] panel: {len(panel.symbols)} names, {len(panel.dates)} days, "
          f"{panel.dates.min().date()} .. {panel.dates.max().date()}")
    print(f"[lowvol] grid: lookbacks={LOOKBACKS} quantiles={QUANTILES} rebalances={REBALANCES} "
          f"-> {len(LOOKBACKS)*len(QUANTILES)*len(REBALANCES)} configs/fold (TRAIN-only selection)")
    print(f"[lowvol] walk-forward, base cost {BASE_COST_BPS}bp/side:")

    base = run_walkforward(panel, spy, BASE_COST_BPS, verbose=True)

    # Cost sensitivity (re-tune per fold at each cost so selection stays honest at that cost).
    cost_results = {}
    for c in COST_GRID:
        r = run_walkforward(panel, spy, c, verbose=False)
        cost_results[f"{c:.0f}bp"] = {
            "oos_annual_return": r["oos_annual_return"],
            "oos_sharpe": r["oos_sharpe"],
            "oos_total_return": r["oos_total_return"],
            "alpha_annual": r["beta_decompose"]["alpha_annual"],
            "alpha_tstat": r["beta_decompose"]["alpha_tstat"],
            "beta": r["beta_decompose"]["beta"],
            "folds_positive": r["folds_positive"],
        }

    bd = base["beta_decompose"]
    survives_at = [lvl for lvl, v in cost_results.items() if v["oos_annual_return"] > 0]
    majority = base["folds_positive"] >= (base["n_folds"] // 2 + 1)
    survives_5bp = cost_results["5bp"]["oos_annual_return"] > 0
    survives_oos = (
        bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and majority
        and survives_5bp
    )

    print("\n[lowvol] ===== AGGREGATE OOS (concatenated, base 5bp) =====")
    print(f"  n_days={base['oos_n_days']} folds_positive={base['folds_positive']}/{base['n_folds']}")
    print(f"  OOS Sharpe={base['oos_sharpe']:.3f}  annual={base['oos_annual_return']*100:.2f}%  "
          f"maxDD={base['oos_max_drawdown']*100:.2f}%  avg_turnover={base['oos_avg_turnover']:.4f}")
    print(f"  alpha(ann)={bd['alpha_annual']*100:.2f}%  alpha_tstat={bd['alpha_tstat']:.2f}  "
          f"beta={bd['beta']:.3f}  beta_tstat={bd['beta_tstat']:.2f}  R^2={bd['r2']:.3f}")
    print("\n[lowvol] cost sensitivity:")
    for lvl, v in cost_results.items():
        print(f"  {lvl:>5}: annual={v['oos_annual_return']*100:7.2f}%  Sharpe={v['oos_sharpe']:6.3f}  "
              f"alpha_t={v['alpha_tstat']:6.2f}  beta={v['beta']:.3f}  pos={v['folds_positive']}/{base['n_folds']}")
    print(f"\n[lowvol] survives_at_costs={survives_at}  majority_positive={majority}")
    print(f"[lowvol] SURVIVES_OOS = {survives_oos}  "
          f"(alpha_t>=2: {bd['alpha_tstat']>=2}, |beta|<0.15: {abs(bd['beta'])<0.15}, "
          f"majority: {majority}, 5bp+: {survives_5bp})")

    # Persist a JSON summary (results dir, never touches protected files).
    out = {
        "name": NAME,
        "hypothesis": (
            "Low-volatility anomaly: LONG lowest trailing-vol quintile / SHORT highest, "
            "dollar-neutral, monthly rebalance. Score = -trailing realized vol."
        ),
        "construction": (
            "Each rebalance, rank universe by -trailing_vol(lookback); long top quantile, "
            "short bottom, equal dollars per leg (gross=1, net=0). Tuned lookback{20/60/120}"
            " x quantile{0.1/0.2/0.3} x rebalance{21/42d} on each fold's TRAIN net Sharpe (5bp); "
            "scored OOS once. Strict t->t+1, turnover costs both sides."
        ),
        "n_folds": base["n_folds"],
        "n_configs_per_fold": base["n_configs_tried_per_fold"],
        "n_configs_total": base["n_configs_tried_per_fold"] * base["n_folds"],
        "folds_positive": base["folds_positive"],
        "oos_annual_return": base["oos_annual_return"],
        "oos_sharpe": base["oos_sharpe"],
        "oos_max_drawdown": base["oos_max_drawdown"],
        "oos_total_return": base["oos_total_return"],
        "oos_n_days": base["oos_n_days"],
        "oos_avg_turnover": base["oos_avg_turnover"],
        "beta_decompose": {k: v for k, v in bd.items()},
        "fold_records": [{k: v for k, v in fr.items()} for fr in base["folds"]],
        "fold_betas": base["fold_betas"],
        "cost_sensitivity": cost_results,
        "base_cost_bps": BASE_COST_BPS,
        "survives_at_costs": survives_at,
        "survives_oos": survives_oos,
        "survivorship_caveat": (
            "Universe is TODAY's ~97 large-cap survivors applied backward (no point-in-time "
            "membership on free data) -> INFLATES results; treat marginal edges skeptically. "
            "Window omits 2018Q4 and 2020 COVID (free-tier floor ~2020-07)."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{NAME}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[lowvol] wrote {path}")
    return out


if __name__ == "__main__":
    main()
