"""sanity_check.py — cross-check the research backtester vs the production engine.

Runs BOTH the research backtester (experiments.harness.research_backtest) and the
production engine (agentic_trader.backtest.engine.run_backtest) on the SAME bars and the
SAME base long-only params, and compares trade counts + PnL sign/magnitude.

Both engines are fed identical Bar data from the experiments cache (the production engine
is data-source agnostic — it consumes Dict[str, List[Bar]]). This isolates the SIMULATOR
logic, which is what we are validating.

Run: cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.sanity_check
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments import harness as H

RESULTS_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results")

# Base long-only params per the task.
BASE = dict(entry_dist=0.005, vol_mult=1.2, max_hold=15, vwap_exit_band=0.001, stop_loss=0.005)

# Use a bounded shared window so the production engine (pure-python, O(bars^2)-ish on the
# rolling window and per-step grouping) runs in reasonable time. ~3 weeks of all symbols.
SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
WINDOW_START = "2025-09-02"
WINDOW_END = "2025-09-22"


def _df_to_bars(sym, df):
    """Convert a research-cache DataFrame slice to a list of production Bar objects."""
    from agentic_trader.types import Bar

    out = []
    for ts, row in df.iterrows():
        out.append(
            Bar(
                symbol=sym,
                start=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                trade_count=float(row["trade_count"]) if row["trade_count"] is not None else None,
                vwap=float(row["vwap"]) if row["vwap"] is not None else None,
            )
        )
    return out


def main():
    bars = H.load_bars(SYMBOLS)
    # restrict to the shared window
    import pandas as pd

    lo = pd.Timestamp(WINDOW_START, tz=H.NY)
    hi = pd.Timestamp(WINDOW_END, tz=H.NY) + pd.Timedelta(days=1)
    win = {}
    for s, df in bars.items():
        sl = df[(df.index >= lo) & (df.index < hi)]
        if not sl.empty:
            win[s] = sl

    # --- research backtester ---
    spec = dict(side="long", **BASE)
    research = H.research_backtest(win, spec)
    rm = research["metrics"]

    # --- production engine ---
    from agentic_trader.backtest.engine import run_backtest
    from agentic_trader.config import RiskLimits, StrategyParams

    bars_by_symbol = {s: _df_to_bars(s, df) for s, df in win.items()}
    params = StrategyParams(
        entry_dist=BASE["entry_dist"],
        vol_mult=BASE["vol_mult"],
        max_hold=BASE["max_hold"],
        vwap_exit_band=BASE["vwap_exit_band"],
        stop_loss=BASE["stop_loss"],
        ml_threshold=0.0,
        notional=100.0,
    )
    limits = RiskLimits()
    prod = run_backtest(bars_by_symbol, params, limits)
    pm = prod.metrics

    report = {
        "window": {"start": WINDOW_START, "end": WINDOW_END, "symbols": list(win.keys())},
        "research": {
            "n_trades": rm["n_trades"],
            "total_pnl": round(rm["total_pnl"], 4),
            "win_rate": round(rm["win_rate"], 4),
            "exits_by_reason": rm["exits_by_reason"],
        },
        "production": {
            "n_trades": int(pm["n_trades"]),
            "total_pnl": round(pm["total_pnl"], 4),
            "win_rate": round(pm["win_rate"], 4),
            "exits": {k.replace("exit_", ""): int(v) for k, v in pm.items() if k.startswith("exit_")},
        },
    }
    print(json.dumps(report, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "harness_sanity.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
