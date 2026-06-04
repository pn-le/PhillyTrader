"""fac_overnight.py — the CLOSE-TO-OPEN "OVERNIGHT" anomaly (daily, beta-decomposed).

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on the cached daily panel.

=============================================================================
THE EFFECT
=============================================================================
The "overnight anomaly": historically (esp. pre-2010 US equities) almost ALL of the
equity risk premium accrued OVERNIGHT (close -> next open) while the INTRADAY session
(open -> close) earned ~nothing. A naive "buy at close, sell at next open" captures the
overnight leg. The skeptical question this file answers: on our 2020-2026 large-cap
universe, is there REAL beta-neutral alpha in overnight returns, or is it just market
beta captured at a different time of day (lower-beta because you hold fewer hours)?

We test THREE things, all on the SAME adjusted daily bars, all beta-decomposed vs SPY:

  (A) TS_OVERNIGHT  — long the equal-weight universe OVERNIGHT only (close->open),
                      flat intraday. Time-series, NOT neutral by construction; we
                      regress on SPY close-to-close to see how much "alpha" is just
                      reduced beta (the trap).
  (B) TS_INTRADAY   — the mirror: long the universe INTRADAY only (open->close). Control.
  (C) XS_OVERNIGHT  — the proper NEUTRAL construction: each day rank names by their
                      RECENT overnight-return tendency (trailing mean close->open return),
                      go LONG the top quantile / SHORT the bottom quantile in equal
                      dollars, and HOLD that book OVERNIGHT only (earn close_d->open_{d+1}).
                      Dollar-neutral => market-neutral BY CONSTRUCTION. This is the one
                      that can have alpha WITHOUT beta. Walk-forward, tuned on TRAIN only.

=============================================================================
RETURN PLUMBING (the one subtle thing)
=============================================================================
The harness `panel.rets` is CLOSE-TO-CLOSE. The overnight strategy earns a DIFFERENT
return stream, so we build our own per-symbol frames from the panel's open/close
(both split+dividend adjusted, so close_{d-1}->open_d is a clean adjusted overnight ret):

    overnight_ret.loc[t, s] = open[t,s]  / close[t-1,s] - 1     # close_{t-1} -> open_t
    intraday_ret.loc[t, s]  = close[t,s] / open[t,s]   - 1      # open_t      -> close_t

`backtest_xsection` lags weights internally: a weight decided at the CLOSE of day d earns
`returns.loc[d+1]`. So passing `overnight_ret` makes weight_d earn overnight_ret_{d+1} =
open_{d+1}/close_d - 1 == exactly the close_d -> open_{d+1} hold. Correct by construction.

COSTS: a daily overnight strategy ENTERS at each close and EXITS at each open => it is a
fully-daily-rebalanced strategy (rebalance_freq=1). The TS book turns over ~2x/day (close
in, open out); the harness charges turnover on |w_new - w_old_drifted| per side both ways.
We report 2 / 5 / 10 bp per side. The base gate is 5bp/side.

GATES (a real edge clears ALL): alpha_tstat >= 2, |beta| < 0.15, positive in a MAJORITY
of OOS folds, AND survives the 5bp base cost.

SURVIVORSHIP / LOOK-AHEAD: today's large-caps applied backward (see factor_harness header).
Inflates results; marginal edges treated skeptically. Window has NO 2018Q4 / COVID crash
(free-tier floor ~2020-07-27) but spans 2022 bear + 2023-26 bull + 2025 tariff selloff.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.fac_overnight
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.factors.factor_harness import (
    Panel,
    backtest_xsection,
    beta_decompose,
    form_dollar_neutral_portfolio,
    load_daily,
    spy_daily_returns,
    walk_forward_folds,
)
from experiments.factors.fetch_daily import UNIVERSE

RESULTS_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/factors/results")
CACHE = Path("/Users/pnle/Desktop/alpaca-cli/experiments/factors/cache")
TRADING_YEAR = 252
N_FOLDS = 4
TRAIN_FRAC = 0.4


# --------------------------------------------------------------------------- #
# Return-stream construction (overnight / intraday) from the panel's open/close
# --------------------------------------------------------------------------- #
def overnight_returns(panel: Panel) -> pd.DataFrame:
    """close_{t-1} -> open_t per symbol (the 'overnight' hold). First row NaN."""
    return panel.open / panel.close.shift(1) - 1.0


def intraday_returns(panel: Panel) -> pd.DataFrame:
    """open_t -> close_t per symbol (the 'intraday' session). No look-ahead lag needed."""
    return panel.close / panel.open - 1.0


def spy_overnight_intraday() -> tuple[pd.Series, pd.Series, pd.Series]:
    """SPY's own overnight (close->open), intraday (open->close), and close->close returns."""
    df = pd.read_parquet(CACHE / "SPY_daily.parquet").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    on = (df["open"] / df["close"].shift(1) - 1.0).dropna()
    intr = (df["close"] / df["open"] - 1.0).dropna()
    c2c = df["close"].pct_change().dropna()
    return on, intr, c2c


# --------------------------------------------------------------------------- #
# (A)(B) Time-series long-universe legs (NOT neutral; the "beta at a different
#         time of day?" diagnostic). Equal-weight long the whole universe.
# --------------------------------------------------------------------------- #
def _equal_weight_leg(ret_frame: pd.DataFrame) -> pd.Series:
    """Equal-weight long-universe daily return from a per-symbol return frame.

    Each day, average the available names' returns (a fully-invested, long-only,
    equal-weight book). This is the realized leg return BEFORE costs. We charge cost
    separately for the overnight leg (it round-trips daily: in at close, out at open).
    """
    return ret_frame.mean(axis=1, skipna=True).dropna()


def _annualized(daily: pd.Series) -> dict:
    n = len(daily)
    mean = float(daily.mean()) if n else 0.0
    std = float(daily.std(ddof=1)) if n > 1 else 0.0
    eq = (1.0 + daily).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min()) if n else 0.0
    return {
        "ann_return": mean * TRADING_YEAR,
        "sharpe": (mean / std * np.sqrt(TRADING_YEAR)) if std > 0 else 0.0,
        "max_drawdown": mdd,
        "n_days": n,
    }


def run_timeseries_legs(panel: Panel, spy_c2c: pd.Series) -> dict:
    """(A) overnight long-universe and (B) intraday long-universe, beta-decomposed vs SPY c2c.

    A long-only equal-weight book turns over essentially 0 close-to-close (you hold the
    same names), BUT an OVERNIGHT-only book buys every close and sells every open: gross ~2
    one-way turnover/day. We apply a flat round-trip cost = 2 * cost_bps/side on the full
    book each day to the overnight leg (it fully cycles daily). The intraday leg likewise.
    These legs are diagnostics (not the neutral candidate), so we report gross + net@5bp.
    """
    on = _equal_weight_leg(overnight_returns(panel))
    intr = _equal_weight_leg(intraday_returns(panel))

    out = {}
    for name, leg in [("ts_overnight", on), ("ts_intraday", intr)]:
        # cost: full book in and out each day => 2 sides * cost on gross notional 1.0.
        net = {}
        for bps in (0.0, 2.0, 5.0, 10.0):
            cost_daily = 2.0 * bps / 1e4  # round trip on a fully-invested book
            net[str(int(bps))] = _annualized(leg - cost_daily)
        bd = beta_decompose(leg, spy_c2c)  # gross beta decomposition
        out[name] = {
            "gross": _annualized(leg),
            "net_by_cost": net,
            "beta_decompose_gross": bd,
        }
    return out


# --------------------------------------------------------------------------- #
# (C) Cross-sectional OVERNIGHT-tendency factor (the NEUTRAL candidate)
# --------------------------------------------------------------------------- #
def make_overnight_tendency_factor(lookback: int):
    """factor_fn(panel) -> scores: trailing-mean OVERNIGHT return per name (close->open).

    score[d, s] = mean over the last `lookback` overnight returns ENDING at day d's open.
    Higher score => the name has been a stronger overnight performer recently => go LONG it
    (and short the weak-overnight names). This bets the overnight tendency PERSISTS
    cross-sectionally. No look-ahead: overnight_ret.loc[d] = open_d/close_{d-1} uses only
    info known by day d's open, which is <= day d's close (the decision time). The harness
    then holds w_d over the NEXT overnight (close_d -> open_{d+1}).
    """
    def factor_fn(panel: Panel) -> pd.DataFrame:
        on = overnight_returns(panel)
        return on.rolling(lookback, min_periods=max(3, lookback // 2)).mean()

    factor_fn.lookback = lookback
    return factor_fn


def _eval_xs_config(
    panel: Panel,
    test_idx: pd.DatetimeIndex,
    lookback: int,
    top_q: float,
    cost_bps: float,
) -> dict:
    """Score the XS overnight-tendency factor on `test_idx`, holding OVERNIGHT, net of cost.

    Uses the harness primitives: form_dollar_neutral_portfolio + backtest_xsection, but feeds
    backtest_xsection the OVERNIGHT return frame (not close-to-close) so weight_d earns the
    close_d -> open_{d+1} hold. Returns the OOS daily net series for this slice.
    """
    factor_fn = make_overnight_tendency_factor(lookback)
    upto = panel.dates[panel.dates <= test_idx[-1]]
    sub = _slice(panel, upto)
    scores = factor_fn(sub).reindex(test_idx)
    weights = form_dollar_neutral_portfolio(scores, top_q=top_q, bottom_q=top_q, min_names_per_side=3)

    # Overnight return frame extended one day so the last test weight earns its t+1 overnight.
    on_full = overnight_returns(panel)
    ext = _extend_one(panel.dates, test_idx)
    bt = backtest_xsection(
        weights_over_time=weights,
        returns=on_full.reindex(ext),
        rebalance_freq=1,           # overnight = daily re-entry by definition
        cost_bps_per_side=cost_bps,
    )
    daily = bt["daily"]
    # drop the leading flat realization day (no book carried in)
    if len(daily) and float(daily.iloc[0]) == 0.0 and float(bt["turnover"].iloc[0]) == 0.0:
        daily = daily.iloc[1:]
    return {"daily": daily, "metrics": bt["metrics"]}


def run_xsection_walkforward(panel: Panel, spy_c2c: pd.Series, cost_bps: float) -> dict:
    """Walk-forward the XS overnight-tendency factor; TUNE (lookback, top_q) on TRAIN only.

    For each fold: grid-search (lookback x top_q) on the fold's TRAIN slice, pick the best
    TRAIN Sharpe (net at the SAME cost), then score that ONE config on the untouched TEST.
    Concatenate OOS days and beta-decompose vs SPY close-to-close. Returns the full record.

    Returns also `n_configs` = total TRAIN evaluations across folds (honest selection count).
    """
    lookback_grid = [3, 5, 10, 21, 42, 63]
    topq_grid = [0.1, 0.2, 0.3]

    folds = walk_forward_folds(list(panel.dates), n_folds=N_FOLDS, train_frac=TRAIN_FRAC)
    fold_records = []
    fold_betas = []
    oos_pieces = []
    positive = 0
    n_configs = 0
    tot_to_w = 0.0
    tot_days = 0

    for f in folds:
        train_idx = f["_train_dates"]
        test_idx = f["_test_dates"]

        # --- tune on TRAIN only (score each config on the TRAIN slice, pick best Sharpe) ---
        best = None
        for lb in lookback_grid:
            for tq in topq_grid:
                # need lookback history before train start: score on panel up to train end,
                # evaluate only on train dates that have >= a few prior overnight obs.
                tr_eval = train_idx[train_idx > train_idx[min(lb, len(train_idx) - 1)]]
                if len(tr_eval) < 20:
                    tr_eval = train_idx[lb:] if len(train_idx) > lb + 20 else train_idx
                res = _eval_xs_config(panel, tr_eval, lb, tq, cost_bps)
                n_configs += 1
                m = res["metrics"]
                key = m["sharpe"]
                if best is None or key > best["sharpe"]:
                    best = {"lookback": lb, "top_q": tq, "sharpe": m["sharpe"]}

        # --- score the chosen config ONCE on the untouched TEST slice ---
        res = _eval_xs_config(panel, test_idx, best["lookback"], best["top_q"], cost_bps)
        daily = res["daily"]
        m = res["metrics"]
        tot_ret = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
        if tot_ret > 0:
            positive += 1
        oos_pieces.append(daily)
        tot_to_w += m["avg_turnover"] * m["n_days"]
        tot_days += m["n_days"]

        spy_fold = spy_c2c.reindex(daily.index).dropna() if len(daily) else spy_c2c.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append({
            "fold": f["fold"],
            "train": f["train"],
            "test": f["test"],
            "chosen": {"lookback": best["lookback"], "top_q": best["top_q"],
                       "train_sharpe": round(best["sharpe"], 3)},
            "oos_n_days": m["n_days"],
            "oos_total_return": tot_ret,
            "oos_annual_return": m["annual_return"],
            "oos_sharpe": m["sharpe"],
            "oos_max_drawdown": m["max_drawdown"],
            "oos_avg_turnover": m["avg_turnover"],
        })

    oos = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    agg = _annualized(oos)
    spy_oos = spy_c2c.reindex(oos.index).dropna() if len(oos) else spy_c2c.iloc[0:0]
    bd = beta_decompose(oos, spy_oos)
    return {
        "cost_bps_per_side": cost_bps,
        "n_folds": len(folds),
        "n_configs_tried_train": n_configs,
        "folds": fold_records,
        "folds_positive": positive,
        "oos_annual_return": agg["ann_return"],
        "oos_sharpe": agg["sharpe"],
        "oos_max_drawdown": agg["max_drawdown"],
        "oos_n_days": agg["n_days"],
        "oos_avg_turnover": (tot_to_w / tot_days) if tot_days else 0.0,
        "beta_decompose": bd,
        "fold_betas": fold_betas,
        "_oos_daily": oos,  # kept in-memory for cross-cost reporting; stripped before JSON
    }


# --------------------------------------------------------------------------- #
# small local helpers (mirror harness internals; kept private to this file)
# --------------------------------------------------------------------------- #
def _slice(panel: Panel, idx: pd.DatetimeIndex) -> Panel:
    idx = panel.dates.intersection(pd.DatetimeIndex(idx))
    return Panel(
        close=panel.close.reindex(idx), open=panel.open.reindex(idx),
        high=panel.high.reindex(idx), low=panel.low.reindex(idx),
        volume=panel.volume.reindex(idx), rets=panel.rets.reindex(idx),
        symbols=panel.symbols, dates=idx,
    )


def _extend_one(all_dates: pd.DatetimeIndex, window: pd.DatetimeIndex) -> pd.DatetimeIndex:
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    after = all_dates[all_dates > window[-1]]
    return window.append(pd.DatetimeIndex([after[0]])) if len(after) else window


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    # Use only FULL-HISTORY names (coverage.json flags MMC as partial: its stale tail ends
    # 2026-01-13 and would clip the whole panel under min_full_overlap, losing ~5 months and
    # the full-window. Matches the 97-name panel the sanity suite validated.)
    full_names = [s for s in UNIVERSE if s != "MMC"]
    panel = load_daily(full_names)
    spy_c2c = spy_daily_returns()
    spy_on, spy_in, spy_c2c_dbg = spy_overnight_intraday()

    # (A)(B) time-series diagnostic legs
    ts = run_timeseries_legs(panel, spy_c2c)

    # (C) cross-sectional neutral candidate at the 5bp base cost (the gate)
    xs_5 = run_xsection_walkforward(panel, spy_c2c, cost_bps=5.0)
    oos_daily = xs_5.pop("_oos_daily")

    # cost sensitivity for the XS candidate at 2 / 10 bp (re-walk, since cost changes
    # which config TRAIN picks; honest — selection is re-done at each cost).
    xs_2 = run_xsection_walkforward(panel, spy_c2c, cost_bps=2.0); xs_2.pop("_oos_daily")
    xs_10 = run_xsection_walkforward(panel, spy_c2c, cost_bps=10.0); xs_10.pop("_oos_daily")

    bd = xs_5["beta_decompose"]
    survives_oos = bool(
        bd["alpha_tstat"] >= 2.0
        and abs(bd["beta"]) < 0.15
        and xs_5["folds_positive"] > xs_5["n_folds"] / 2.0
        and xs_5["oos_annual_return"] > 0.0
    )
    # which cost levels stay positive (OOS annual return > 0)
    survive_costs = []
    for tag, r in [("2bp", xs_2), ("5bp", xs_5), ("10bp", xs_10)]:
        if r["oos_annual_return"] > 0:
            survive_costs.append(tag)

    universe_overnight_ann = float(
        _equal_weight_leg(overnight_returns(panel)).mean() * TRADING_YEAR
    )

    report = {
        "name": "overnight",
        "universe_n": len(panel.symbols),
        "window": (str(panel.dates[0].date()), str(panel.dates[-1].date()), len(panel.dates)),
        "spy_overnight_ann_pct": round(spy_on.mean() * TRADING_YEAR * 100, 2),
        "spy_intraday_ann_pct": round(spy_in.mean() * TRADING_YEAR * 100, 2),
        "universe_ew_overnight_ann_pct": round(universe_overnight_ann * 100, 2),
        "timeseries_legs": ts,
        "xsection": {"cost_2bp": xs_2, "cost_5bp": xs_5, "cost_10bp": xs_10},
        "verdict": {
            "survives_oos": survives_oos,
            "survives_at_costs": ",".join(survive_costs) if survive_costs else "none",
            "alpha_tstat_5bp": bd["alpha_tstat"],
            "beta_5bp": bd["beta"],
            "oos_sharpe_5bp": xs_5["oos_sharpe"],
            "folds_positive_5bp": xs_5["folds_positive"],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "overnight.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- console summary ----
    print("=" * 78)
    print("OVERNIGHT ANOMALY — daily, beta-decomposed, walk-forward, cost-aware")
    print("=" * 78)
    w = report["window"]
    print(f"universe={report['universe_n']} names  window={w[0]}..{w[1]} ({w[2]}d)")
    print(f"SPY overnight ann={report['spy_overnight_ann_pct']}%  "
          f"SPY intraday ann={report['spy_intraday_ann_pct']}%  "
          f"(classic anomaly: overnight >> intraday)")
    print(f"EW universe overnight-only ann={report['universe_ew_overnight_ann_pct']}%")
    print("-" * 78)
    print("(A/B) TIME-SERIES LONG-UNIVERSE LEGS (NOT neutral; beta-at-different-hour test):")
    for nm in ("ts_overnight", "ts_intraday"):
        g = ts[nm]["gross"]; b = ts[nm]["beta_decompose_gross"]; n5 = ts[nm]["net_by_cost"]["5"]
        print(f"  {nm:13s} GROSS ann={g['ann_return']*100:6.2f}% sh={g['sharpe']:5.2f} | "
              f"alpha_ann={b['alpha_annual']*100:6.2f}% t={b['alpha_tstat']:5.2f} beta={b['beta']:.3f} | "
              f"NET@5bp ann={n5['ann_return']*100:6.2f}% sh={n5['sharpe']:5.2f}")
    print("-" * 78)
    print("(C) CROSS-SECTIONAL OVERNIGHT-TENDENCY L/S (dollar-neutral; the real candidate):")
    for tag, r in [("2bp", xs_2), ("5bp", xs_5), ("10bp", xs_10)]:
        b = r["beta_decompose"]
        print(f"  cost={tag:4s}  OOS ann={r['oos_annual_return']*100:6.2f}% sh={r['oos_sharpe']:5.2f} "
              f"mdd={r['oos_max_drawdown']*100:6.2f}% | alpha_ann={b['alpha_annual']*100:6.2f}% "
              f"t={b['alpha_tstat']:5.2f} beta={b['beta']:+.3f} R2={b['r2']:.3f} | "
              f"folds+={r['folds_positive']}/{r['n_folds']} turn={r['oos_avg_turnover']:.2f}")
    print(f"  TRAIN configs tried (per cost re-walk): {xs_5['n_configs_tried_train']}")
    print("  per-fold (5bp): " + " | ".join(
        f"f{fr['fold']}:{fr['chosen']} oos_sh={fr['oos_sharpe']:.2f} ret={fr['oos_total_return']*100:.1f}%"
        for fr in xs_5["folds"]))
    print("=" * 78)
    v = report["verdict"]
    print(f"VERDICT survives_oos={v['survives_oos']}  survives_at_costs={v['survives_at_costs']}")
    print(f"  (gate: alpha_t>=2 AND |beta|<0.15 AND folds+>half AND survives 5bp)")
    print(f"  alpha_t@5bp={v['alpha_tstat_5bp']:.2f} beta@5bp={v['beta_5bp']:+.3f} "
          f"oos_sharpe@5bp={v['oos_sharpe_5bp']:.2f} folds+={v['folds_positive_5bp']}/{xs_5['n_folds']}")
    print("Written", RESULTS_DIR / "overnight.json")
    return report


if __name__ == "__main__":
    main()
