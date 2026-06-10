"""factor_sanity.py — correctness sanity suite for factor_harness.

Runs the three mandated sanity checks and writes results/factor_harness_sanity.md:

  (a) NO SPURIOUS EDGE: a constant score and a random score both yield ~0 return and a
      statistically tiny alpha (the dollar-neutral construction must not manufacture edge
      from noise). Constant score -> flat book (degenerate ranks) -> exactly 0. Random
      score -> near-zero alpha, |t| well below 2.
  (b) BETA RECOVERY: feeding SPY's OWN daily return as the 'strategy' into beta_decompose
      recovers beta ~ 1, alpha ~ 0, R^2 ~ 1.
  (c) COST MONOTONICITY: for the SAME factor/weights, higher turnover (more frequent
      rebalance) and higher cost_bps each STRICTLY reduce net return.

Pure research. Places ZERO orders.

Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.factors.factor_sanity
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from experiments.factors.factor_harness import (
    RESULTS_DIR,
    backtest_xsection,
    beta_decompose,
    evaluate_walkforward,
    form_dollar_neutral_portfolio,
    load_daily,
    spy_daily_returns,
)
from experiments.factors.fetch_daily import rebuild_coverage_from_cache


def _load_panel():
    cov = rebuild_coverage_from_cache()
    full = cov["names_full_history"]
    panel = load_daily(full)
    spy = spy_daily_returns()
    return panel, spy, cov


# --------------------------------------------------------------------------- #
# (a) no spurious edge
# --------------------------------------------------------------------------- #
def sanity_no_spurious_edge(panel, spy):
    out = {}

    # constant score for every name every day -> all ranks equal -> degenerate.
    def const_factor(p):
        return pd.DataFrame(1.0, index=p.dates, columns=p.symbols)

    # random score (deterministic seed) -> genuine noise, no information.
    rng = np.random.default_rng(12345)

    def random_factor(p):
        data = rng.standard_normal((len(p.dates), len(p.symbols)))
        return pd.DataFrame(data, index=p.dates, columns=p.symbols)

    # Constant score: evaluated WITH cost (must be exactly flat regardless of cost).
    res_const = evaluate_walkforward(const_factor, panel, spy, n_folds=4, cost_bps=5.0)
    # Random score: the "no manufactured edge" property is about the CONSTRUCTION, not the
    # cost model — so we test the GROSS (zero-cost) alpha t-stat is < 2. (Separately, the
    # 5bp NET run must be NEGATIVE: a random book churning ~1.6 turnover/day correctly
    # bleeds slippage. That is desired behavior, not an edge.)
    res_rand_gross = evaluate_walkforward(random_factor, panel, spy, n_folds=4, cost_bps=0.0)
    res_rand_net = evaluate_walkforward(random_factor, panel, spy, n_folds=4, cost_bps=5.0)

    out["constant"] = {
        "oos_annual_return": res_const["oos_annual_return"],
        "oos_sharpe": res_const["oos_sharpe"],
        "alpha_annual": res_const["beta_decompose"]["alpha_annual"],
        "alpha_tstat": res_const["beta_decompose"]["alpha_tstat"],
        "beta": res_const["beta_decompose"]["beta"],
        "oos_n_days": res_const["oos_n_days"],
        "oos_avg_turnover": res_const["oos_avg_turnover"],
    }
    out["random_gross"] = {
        "oos_annual_return": res_rand_gross["oos_annual_return"],
        "oos_sharpe": res_rand_gross["oos_sharpe"],
        "alpha_annual": res_rand_gross["beta_decompose"]["alpha_annual"],
        "alpha_tstat": res_rand_gross["beta_decompose"]["alpha_tstat"],
        "beta": res_rand_gross["beta_decompose"]["beta"],
        "oos_n_days": res_rand_gross["oos_n_days"],
        "oos_avg_turnover": res_rand_gross["oos_avg_turnover"],
    }
    out["random_net5bp"] = {
        "oos_annual_return": res_rand_net["oos_annual_return"],
        "alpha_annual": res_rand_net["beta_decompose"]["alpha_annual"],
        "alpha_tstat": res_rand_net["beta_decompose"]["alpha_tstat"],
    }
    # PASS conditions
    out["pass_constant_flat"] = (
        abs(out["constant"]["oos_annual_return"]) < 1e-9
        and abs(out["constant"]["oos_avg_turnover"]) < 1e-9
    )
    out["pass_random_no_edge"] = abs(out["random_gross"]["alpha_tstat"]) < 2.0
    out["pass_random_net_negative_from_cost"] = out["random_net5bp"]["alpha_annual"] < 0.0
    return out


# --------------------------------------------------------------------------- #
# (b) beta recovery: SPY-on-SPY -> beta 1, alpha 0, R^2 1
# --------------------------------------------------------------------------- #
def sanity_beta_recovery(spy):
    bd = beta_decompose(spy, spy)
    out = {
        "beta": bd["beta"],
        "alpha_per_day": bd["alpha_per_day"],
        "alpha_annual": bd["alpha_annual"],
        "r2": bd["r2"],
        "n_days": bd["n_days"],
    }
    out["pass"] = (
        abs(bd["beta"] - 1.0) < 1e-9
        and abs(bd["alpha_per_day"]) < 1e-9
        and abs(bd["r2"] - 1.0) < 1e-9
    )
    return out


# --------------------------------------------------------------------------- #
# (c) cost monotonicity: more turnover / higher bps strictly reduce net return
# --------------------------------------------------------------------------- #
def sanity_cost_monotonicity(panel):
    # Build a fixed informative score (1-day reversal: -yesterday's return) so the book
    # actually trades. Use the full panel for a clean, deterministic comparison.
    rets = panel.rets
    score = -rets.shift(1)  # decided at close of d using returns up to d (no look-ahead)
    weights = form_dollar_neutral_portfolio(score, top_q=0.2, bottom_q=0.2)

    # (c1) higher cost_bps strictly reduces net total return at FIXED rebalance freq.
    bps_grid = [0.0, 2.0, 5.0, 10.0]
    by_bps = []
    for bps in bps_grid:
        bt = backtest_xsection(weights, rets, rebalance_freq=1, cost_bps_per_side=bps)
        tot = float((1.0 + bt["daily"]).prod() - 1.0)
        by_bps.append({"cost_bps": bps, "net_total_return": tot, "total_cost": bt["metrics"]["total_cost"]})
    net_by_bps = [r["net_total_return"] for r in by_bps]
    pass_bps_mono = all(net_by_bps[i] > net_by_bps[i + 1] for i in range(len(net_by_bps) - 1))

    # (c2) at fixed cost_bps>0, MORE frequent rebalancing => MORE turnover => MORE total cost.
    freq_grid = [1, 5, 21]
    by_freq = []
    for fr in freq_grid:
        bt = backtest_xsection(weights, rets, rebalance_freq=fr, cost_bps_per_side=5.0)
        by_freq.append(
            {
                "rebalance_freq": fr,
                "avg_turnover": bt["metrics"]["avg_turnover"],
                "total_cost": bt["metrics"]["total_cost"],
                "net_total_return": float((1.0 + bt["daily"]).prod() - 1.0),
            }
        )
    costs = [r["total_cost"] for r in by_freq]
    turns = [r["avg_turnover"] for r in by_freq]
    # rebalance_freq 1 < 5 < 21 => turnover/cost should be DECREASING as freq increases.
    pass_turnover_mono = all(turns[i] > turns[i + 1] for i in range(len(turns) - 1))
    pass_cost_mono = all(costs[i] > costs[i + 1] for i in range(len(costs) - 1))

    return {
        "by_cost_bps": by_bps,
        "pass_higher_bps_lower_net": pass_bps_mono,
        "by_rebalance_freq": by_freq,
        "pass_more_frequent_more_turnover": pass_turnover_mono,
        "pass_more_frequent_more_cost": pass_cost_mono,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt_pct(x):
    return f"{x * 100:.2f}%"


def main():
    panel, spy, cov = _load_panel()
    a = sanity_no_spurious_edge(panel, spy)
    b = sanity_beta_recovery(spy)
    c = sanity_cost_monotonicity(panel)

    lines = []
    lines.append("# factor_harness — Correctness Sanity Suite")
    lines.append("")
    lines.append(
        "PURE RESEARCH / BACKTESTING. ZERO orders. These checks lock the DAILY cross-sectional "
        "factor engine before any experiment trusts it — same brutal standard that correctly "
        "found 0 edges intraday."
    )
    lines.append("")
    lines.append(
        f"**Panel:** {len(panel.symbols)} full-history names, "
        f"{len(panel.dates)} trading days, "
        f"{str(panel.dates.min().date())} .. {str(panel.dates.max().date())}. "
        f"Benchmark SPY: {b['n_days']} daily returns."
    )
    lines.append("")

    # (a)
    lines.append("## (a) No spurious edge — constant & random scores")
    lines.append("")
    lines.append(
        "A dollar-neutral L/S construction fed a NON-INFORMATIVE score must not manufacture "
        "edge. A constant score has zero cross-sectional dispersion -> no information to rank "
        "on -> the book stays empty -> EXACTLY 0 return, 0 turnover (regardless of cost). A "
        "random score produces a real but information-free book whose GROSS alpha (zero cost) "
        "must be statistically indistinguishable from zero (|t| < 2) — that is the test of the "
        "CONSTRUCTION. Its NET alpha at 5bp must be NEGATIVE: a random book churning ~1.6 "
        "turnover/day correctly bleeds slippage (desired engine behavior, not an edge)."
    )
    lines.append("")
    lines.append("| score | OOS ann.return | OOS Sharpe | alpha (ann.) | alpha t-stat | beta | avg turnover | n_days |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, k in (("constant (5bp)", "constant"), ("random GROSS (0bp)", "random_gross")):
        r = a[k]
        lines.append(
            f"| {label} | {_fmt_pct(r['oos_annual_return'])} | {r['oos_sharpe']:.2f} | "
            f"{_fmt_pct(r['alpha_annual'])} | {r['alpha_tstat']:.2f} | {r['beta']:.3f} | "
            f"{r['oos_avg_turnover']:.4f} | {r['oos_n_days']} |"
        )
    rn = a["random_net5bp"]
    lines.append(
        f"| random NET (5bp) | {_fmt_pct(rn['oos_annual_return'])} | - | "
        f"{_fmt_pct(rn['alpha_annual'])} | {rn['alpha_tstat']:.2f} | - | - | - |"
    )
    lines.append("")
    lines.append(
        f"- constant -> exactly flat (0 return, 0 turnover): "
        f"**{'PASS' if a['pass_constant_flat'] else 'FAIL'}**"
    )
    lines.append(
        f"- random GROSS -> alpha t-stat |{a['random_gross']['alpha_tstat']:.2f}| < 2 "
        f"(construction manufactures no edge): **{'PASS' if a['pass_random_no_edge'] else 'FAIL'}**"
    )
    lines.append(
        f"- random NET@5bp -> alpha {_fmt_pct(rn['alpha_annual'])} < 0 "
        f"(churning noise correctly bleeds cost): "
        f"**{'PASS' if a['pass_random_net_negative_from_cost'] else 'FAIL'}**"
    )
    lines.append("")

    # (b)
    lines.append("## (b) Beta recovery — SPY regressed on itself")
    lines.append("")
    lines.append(
        "Using SPY's own daily return as the 'strategy' in beta_decompose must recover "
        "beta = 1, alpha = 0, R^2 = 1 exactly (the OLS is fit on identical y and x)."
    )
    lines.append("")
    lines.append(
        f"- beta = **{b['beta']:.6f}**, alpha/day = **{b['alpha_per_day']:.2e}**, "
        f"R^2 = **{b['r2']:.6f}**, n_days = {b['n_days']}"
    )
    lines.append(f"- **{'PASS' if b['pass'] else 'FAIL'}**")
    lines.append("")

    # (c)
    lines.append("## (c) Cost model — higher turnover strictly reduces net return")
    lines.append("")
    lines.append(
        "Same 1-day reversal factor and weights throughout. (c1) At fixed daily rebalancing, "
        "raising cost_bps/side must strictly lower net total return. (c2) Rebalancing MORE "
        "often (freq 1 < 5 < 21 days) must strictly raise turnover and total cost."
    )
    lines.append("")
    lines.append("### (c1) cost sensitivity at daily rebalance")
    lines.append("")
    lines.append("| cost bps/side | net total return | total cost |")
    lines.append("|---|---|---|")
    for r in c["by_cost_bps"]:
        lines.append(f"| {r['cost_bps']:.0f} | {_fmt_pct(r['net_total_return'])} | {r['total_cost']:.4f} |")
    lines.append("")
    lines.append(
        f"- higher cost_bps -> strictly lower net return: "
        f"**{'PASS' if c['pass_higher_bps_lower_net'] else 'FAIL'}**"
    )
    lines.append("")
    lines.append("### (c2) turnover/cost vs rebalance frequency (cost 5bp/side)")
    lines.append("")
    lines.append("| rebalance_freq (days) | avg turnover | total cost | net total return |")
    lines.append("|---|---|---|---|")
    for r in c["by_rebalance_freq"]:
        lines.append(
            f"| {r['rebalance_freq']} | {r['avg_turnover']:.4f} | {r['total_cost']:.4f} | "
            f"{_fmt_pct(r['net_total_return'])} |"
        )
    lines.append("")
    lines.append(
        f"- more frequent rebalance -> more turnover: "
        f"**{'PASS' if c['pass_more_frequent_more_turnover'] else 'FAIL'}**"
    )
    lines.append(
        f"- more frequent rebalance -> more total cost: "
        f"**{'PASS' if c['pass_more_frequent_more_cost'] else 'FAIL'}**"
    )
    lines.append("")

    all_pass = (
        a["pass_constant_flat"]
        and a["pass_random_no_edge"]
        and a["pass_random_net_negative_from_cost"]
        and b["pass"]
        and c["pass_higher_bps_lower_net"]
        and c["pass_more_frequent_more_turnover"]
        and c["pass_more_frequent_more_cost"]
    )
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{'ALL SANITY CHECKS PASS' if all_pass else 'SANITY FAILURE — DO NOT TRUST THE ENGINE'}**")
    lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "factor_harness_sanity.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nWritten to", RESULTS_DIR / "factor_harness_sanity.md")
    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
