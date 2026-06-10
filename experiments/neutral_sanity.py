"""neutral_sanity.py — confirm neutral_harness reuses the validated primitives.

Two registered sanity checks (the task spec):
  (A) A 'long_short' run with the SHORT side DISABLED must reproduce
      harness.research_backtest long-only trades on a shared window (bit-identical ledger).
  (B) beta_decompose must recover beta ~= 1.0, alpha ~= 0 when fed SPY's OWN daily returns
      as the "strategy" returns.

Writes results/neutral_harness_sanity.md.

PURE RESEARCH. Places ZERO orders.
Run:  cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.neutral_sanity
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.harness import load_bars, research_backtest
from experiments.neutral_harness import (
    research_backtest_neutral,
    beta_decompose,
    spy_close_to_close_returns,
    all_trading_dates,
    slice_by_date_range,
)

RESULTS_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results")
UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]


def _trade_signature(trades):
    """Canonical comparable signature of a trade ledger (order-independent on key fields)."""
    rows = []
    for t in trades:
        rows.append(
            (
                t["symbol"],
                t["side"],
                pd.Timestamp(t["entry_time"]).isoformat(),
                pd.Timestamp(t["exit_time"]).isoformat(),
                round(float(t["entry_price"]), 8),
                round(float(t["exit_price"]), 8),
                round(float(t["qty"]), 8),
                round(float(t["pnl"]), 8),
                t["exit_reason"],
            )
        )
    return sorted(rows)


def sanity_a_long_only_equivalence():
    """long_short with enable_short=False == harness long-only on a shared window."""
    bars = load_bars(UNIVERSE)
    # Use a bounded shared window (the most recent ~80 days) for speed; same window both ways.
    dates = all_trading_dates(bars)
    window = set(dates[-80:])
    win_bars = slice_by_date_range(bars, window)

    base_spec = {
        "entry_dist": 0.005,
        "vol_mult": 1.2,
        "vwap_exit_band": 0.001,
        "max_hold": 15,
        "stop_loss": 0.005,
        "notional": 100.0,
        "max_positions": 4,
        "max_exposure": 500.0,
        "cooldown_min": 10.0,
        "slippage_bps": 1.0,
        "eod_flatten": "15:55",
    }

    # harness long-only
    harness_spec = dict(base_spec, side="long")
    harness_res = research_backtest(win_bars, harness_spec)

    # neutral long_short, short disabled
    neutral_spec = dict(base_spec, mode="long_short", enable_long=True, enable_short=False)
    neutral_res = research_backtest_neutral(win_bars, neutral_spec)

    sig_h = _trade_signature(harness_res["trades"])
    sig_n = _trade_signature(neutral_res["trades"])
    identical = sig_h == sig_n
    pnl_h = harness_res["metrics"]["total_pnl"]
    pnl_n = neutral_res["metrics"]["total_pnl"]

    return {
        "window_first": str(dates[-80]),
        "window_last": str(dates[-1]),
        "harness_n_trades": len(harness_res["trades"]),
        "neutral_n_trades": len(neutral_res["trades"]),
        "ledgers_bit_identical": bool(identical),
        "harness_total_pnl": float(pnl_h),
        "neutral_total_pnl": float(pnl_n),
        "pnl_match": bool(abs(pnl_h - pnl_n) < 1e-9),
    }


def sanity_b_beta_recovery():
    """beta_decompose(SPY_returns, SPY_returns) must give beta~1, alpha~0, R2~1."""
    bars = load_bars(["SPY"])
    spy_daily = spy_close_to_close_returns(bars["SPY"])
    bd = beta_decompose(spy_daily, spy_daily)
    return {
        "n_days": bd["n_days"],
        "beta": bd["beta"],
        "alpha_per_day": bd["alpha_per_day"],
        "alpha_annual": bd["alpha_annual"],
        "r2": bd["r2"],
        "beta_recovered_near_1": bool(abs(bd["beta"] - 1.0) < 1e-6),
        "alpha_recovered_near_0": bool(abs(bd["alpha_per_day"]) < 1e-9),
        "r2_near_1": bool(abs(bd["r2"] - 1.0) < 1e-6),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    a = sanity_a_long_only_equivalence()
    b = sanity_b_beta_recovery()

    a_pass = a["ledgers_bit_identical"] and a["pnl_match"]
    b_pass = b["beta_recovered_near_1"] and b["alpha_recovered_near_0"] and b["r2_near_1"]

    md = []
    md.append("# Neutral Harness — Sanity Checks\n")
    md.append("**Purpose:** prove `neutral_harness.py` reuses the validated, "
              "production-identical primitives from `harness.py` (no quiet divergence) "
              "and that `beta_decompose` is a correct OLS.\n")
    md.append("`neutral_harness` imports `load_bars, research_backtest, _slip, _metrics, "
              "_parse_hm, compute_indicators_df, chronological_split` directly from "
              "`harness.py` — it does not reimplement them.\n")

    md.append("## Sanity A — long_short(short disabled) == harness long-only\n")
    md.append(f"- Shared window: **{a['window_first']} .. {a['window_last']}** "
              f"(most recent 80 trading days)\n")
    md.append(f"- harness.research_backtest long-only trades: **{a['harness_n_trades']}**\n")
    md.append(f"- neutral long_short (enable_short=False) trades: **{a['neutral_n_trades']}**\n")
    md.append(f"- Trade ledgers BIT-IDENTICAL: **{a['ledgers_bit_identical']}**\n")
    md.append(f"- harness total_pnl = {a['harness_total_pnl']:.6f} ; "
              f"neutral total_pnl = {a['neutral_total_pnl']:.6f} ; "
              f"match = **{a['pnl_match']}**\n")
    md.append(f"- **RESULT: {'PASS' if a_pass else 'FAIL'}**\n")

    md.append("## Sanity B — beta_decompose recovers beta~1, alpha~0 on SPY-on-SPY\n")
    md.append(f"- n_days regressed: **{b['n_days']}**\n")
    md.append(f"- beta = {b['beta']:.12f}  (target 1.0, |err|<1e-6: {b['beta_recovered_near_1']})\n")
    md.append(f"- alpha_per_day = {b['alpha_per_day']:.3e}  "
              f"(target 0, |a|<1e-9: {b['alpha_recovered_near_0']})\n")
    md.append(f"- alpha_annual = {b['alpha_annual']:.3e}\n")
    md.append(f"- R^2 = {b['r2']:.12f}  (target 1.0, |err|<1e-6: {b['r2_near_1']})\n")
    md.append(f"- **RESULT: {'PASS' if b_pass else 'FAIL'}**\n")

    md.append("## Overall\n")
    md.append(f"- Sanity A: **{'PASS' if a_pass else 'FAIL'}**\n")
    md.append(f"- Sanity B: **{'PASS' if b_pass else 'FAIL'}**\n")

    (RESULTS_DIR / "neutral_harness_sanity.md").write_text("\n".join(md))
    print("A:", a)
    print("B:", b)
    print("A_PASS:", a_pass, "B_PASS:", b_pass)
    print("Wrote", RESULTS_DIR / "neutral_harness_sanity.md")
    return a_pass and b_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
