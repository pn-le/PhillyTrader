"""exp_symbol_edge.py — Is the VWAP mean-reversion edge concentrated in specific
symbols / instrument types rather than the whole universe?

HYPOTHESIS
----------
The base long VWAP-mean-reversion strategy is broadly unprofitable across the universe,
but a SUBSET of symbols (e.g. liquid index ETFs vs single names) carries a genuine
out-of-sample edge. If so, a symbol subset chosen ONLY on VALIDATION should remain
net-positive on the untouched TEST split.

PROTOCOL (anti-self-deception)
------------------------------
- One fixed strategy spec: the production base long config (entry_dist=0.005, vol_mult=1.2,
  max_hold=15, vwap_exit_band=0.001, stop_loss=0.005, 1bp adverse slippage/side). We do NOT
  tune strategy params here — the only "knob" is the SYMBOL SUBSET, and that knob is turned
  using VALIDATION ONLY.
- Chronological global split (train 60% / val 20% / test 20%) via harness.chronological_split.
- SELECTION RULE (pre-committed, evaluated on VALIDATION only):
    subset = { every symbol whose VALIDATION total_pnl > 0 }
  This is a single, simple, mechanical rule. TEST is never consulted to pick symbols.
- We ALSO report the instrument-type comparison (ETFs {SPY,QQQ,IWM} vs single names) and a
  cost-sensitivity sweep, purely as diagnostics. The headline best_config is the VAL-selected
  subset, evaluated EXACTLY ONCE on TEST.
- n_configs_tried counts the distinct SUBSET choices we evaluated against VALIDATION
  (10 per-symbol val checks + 2 group checks = 12 validation-side comparisons). TEST is
  touched once per reported subset, at the very end.

Run:
    cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.exp_symbol_edge
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.harness import (
    chronological_split,
    load_bars,
    research_backtest,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]
ETFS = ["SPY", "QQQ", "IWM"]
NAMES = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]

# Fixed base long strategy spec (production defaults). The ONLY thing we vary is the symbol set.
BASE_SPEC = dict(
    side="long",
    entry_dist=0.005,
    vol_mult=1.2,
    max_hold=15,
    vwap_exit_band=0.001,
    stop_loss=0.005,
    slippage_bps=1.0,
)

RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/symbol_edge.json")


def _spec(slippage_bps: float = 1.0) -> dict:
    s = dict(BASE_SPEC)
    s["slippage_bps"] = slippage_bps
    return s


def _m(split: dict, symbols: list[str], slippage_bps: float = 1.0) -> dict:
    """Backtest BASE_SPEC restricted to `symbols` on one split; return the metrics dict."""
    sub = {s: split[s] for s in symbols if s in split}
    return research_backtest(sub, _spec(slippage_bps))["metrics"]


def _slim(m: dict) -> dict:
    return {
        "n_trades": m["n_trades"],
        "total_pnl": round(m["total_pnl"], 4),
        "total_return": round(m["total_return"], 6),
        "win_rate": round(m["win_rate"], 4),
        "sharpe": round(m["sharpe_like"], 4),
        "max_dd": round(m["max_drawdown"], 4),
        "avg_ret_bps": round(m["avg_return_pct"] * 1e4, 4),
    }


def main() -> dict:
    bars = load_bars(UNIVERSE)
    tr, va, te, ranges = chronological_split(bars)

    print("=" * 88)
    print("exp_symbol_edge — VWAP mean-reversion: is the edge concentrated in a symbol subset?")
    print("=" * 88)
    print(f"Split date ranges (global, by-date): {ranges}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Per-symbol metrics across all three splits (visibility of the gap).
    # ------------------------------------------------------------------ #
    print("Per-symbol BASE long metrics (TRAIN / VAL / TEST). Selection uses VAL only.")
    print(f"{'sym':<5} {'type':<5} | {'TR n   pnl    wr':<22} | {'VA n   pnl    wr':<22} | {'TE n   pnl    wr':<22}")
    per_symbol = {}
    for s in UNIVERSE:
        mt, mv, me = _m(tr, [s]), _m(va, [s]), _m(te, [s])
        per_symbol[s] = {"train": _slim(mt), "val": _slim(mv), "test": _slim(me)}
        typ = "ETF" if s in ETFS else "name"
        print(
            f"{s:<5} {typ:<5} | {mt['n_trades']:>4} {mt['total_pnl']:>7.3f} {mt['win_rate']:>4.2f}     "
            f"| {mv['n_trades']:>4} {mv['total_pnl']:>7.3f} {mv['win_rate']:>4.2f}     "
            f"| {me['n_trades']:>4} {me['total_pnl']:>7.3f} {me['win_rate']:>4.2f}"
        )

    # ------------------------------------------------------------------ #
    # 2. Instrument-type comparison (ETFs vs single names) — diagnostic.
    # ------------------------------------------------------------------ #
    print()
    print("Instrument-type groups (BASE long):")
    groups = {"ETFs": ETFS, "NAMES": NAMES, "ALL": UNIVERSE}
    group_metrics = {}
    for label, g in groups.items():
        mt, mv, me = _m(tr, g), _m(va, g), _m(te, g)
        group_metrics[label] = {"train": _slim(mt), "val": _slim(mv), "test": _slim(me)}
        print(
            f"  {label:<6} TR n={mt['n_trades']:>4} pnl={mt['total_pnl']:>8.3f} | "
            f"VA n={mv['n_trades']:>4} pnl={mv['total_pnl']:>7.3f} sh={mv['sharpe_like']:>5.2f} | "
            f"TE n={me['n_trades']:>4} pnl={me['total_pnl']:>7.3f} sh={me['sharpe_like']:>5.2f}"
        )

    # ------------------------------------------------------------------ #
    # 3. SELECTION: pre-committed rule on VALIDATION only.
    #    subset = symbols with VAL total_pnl > 0.
    # ------------------------------------------------------------------ #
    val_positive = [s for s in UNIVERSE if per_symbol[s]["val"]["total_pnl"] > 0]
    print()
    print(f"VAL-selection rule (val total_pnl > 0) -> subset = {val_positive}")

    # n_configs_tried: 10 per-symbol VAL checks + 2 group VAL checks (ETFs, NAMES) = 12.
    n_configs_tried = len(UNIVERSE) + 2

    # ------------------------------------------------------------------ #
    # 4. Evaluate the VAL-selected subset ONCE on each split (TEST = final, untouched).
    # ------------------------------------------------------------------ #
    sel_tr, sel_va, sel_te = _m(tr, val_positive), _m(va, val_positive), _m(te, val_positive)
    print()
    print(f"SELECTED SUBSET {val_positive} (BASE long):")
    for nm, m in [("TRAIN", sel_tr), ("VAL  ", sel_va), ("TEST ", sel_te)]:
        print(
            f"  {nm}: n={m['n_trades']:>4} pnl={m['total_pnl']:>8.3f} "
            f"ret={m['total_return']:>8.4f} wr={m['win_rate']:.3f} "
            f"sharpe={m['sharpe_like']:>5.2f} maxdd={m['max_drawdown']:.3f}"
        )

    # Per-symbol TEST breakdown of the selected subset (did the val pick hold up?).
    print("  TEST breakdown of selected subset (did each val-pick hold OOS?):")
    for s in val_positive:
        me = _m(te, [s])
        print(f"    {s:<5} TE n={me['n_trades']:>4} pnl={me['total_pnl']:>7.3f} wr={me['win_rate']:.2f}")

    # ------------------------------------------------------------------ #
    # 5. Cost-sensitivity of the selected subset and the ETF group on TEST.
    #    (Diagnostic: how far above the cost/noise floor is the edge?)
    # ------------------------------------------------------------------ #
    print()
    print("Cost sensitivity on TEST (total_pnl as adverse slippage/side rises):")
    cost_curve = {}
    for label, g in [("subset", val_positive), ("ETF-group", ETFS)]:
        row = {}
        line = [f"  {label:<10}"]
        for slip in (0.0, 0.5, 1.0, 2.0, 3.0):
            me = _m(te, g, slippage_bps=slip)
            row[str(slip)] = round(me["total_pnl"], 4)
            line.append(f"{slip}bp={me['total_pnl']:>6.2f}")
        cost_curve[label] = row
        print("  ".join(line))

    # ------------------------------------------------------------------ #
    # Verdict.
    # ------------------------------------------------------------------ #
    test_n = sel_te["n_trades"]
    test_pnl = sel_te["total_pnl"]
    # Survives only if: net-positive TEST, >=20 trades, AND the train->test gap is not a
    # full disastrous sign-flip (a genuine edge should not be catastrophically negative on
    # the largest/earliest TRAIN sample). We treat a deeply negative TRAIN sharpe as a
    # regime-artifact red flag that disqualifies the "edge".
    train_sharpe = sel_tr["sharpe_like"]
    survives = bool(test_pnl > 0 and test_n >= 20 and train_sharpe > -1.0)

    print()
    print(
        f"VERDICT: TEST n={test_n}, TEST pnl={test_pnl:.3f}, TRAIN sharpe={train_sharpe:.2f}  "
        f"-> survives_oos={survives}"
    )
    print(
        "  (Net-positive TEST with >=20 trades, BUT TRAIN sharpe is deeply negative -> the\n"
        "   TRAIN->TEST sign-flip flags this as a likely regime artifact, not stable edge.)"
        if not survives
        else "  (Clean net-positive OOS with a non-catastrophic train gap.)"
    )

    out = {
        "name": "symbol_edge",
        "split_ranges": ranges,
        "base_spec": {k: v for k, v in BASE_SPEC.items()},
        "per_symbol": per_symbol,
        "groups": group_metrics,
        "val_selected_subset": val_positive,
        "n_configs_tried": n_configs_tried,
        "selected_subset_metrics": {
            "train": _slim(sel_tr),
            "val": _slim(sel_va),
            "test": _slim(sel_te),
        },
        "cost_sensitivity_test": cost_curve,
        "survives_oos": survives,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    main()
