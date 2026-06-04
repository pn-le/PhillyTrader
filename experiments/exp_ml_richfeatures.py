"""exp_ml_richfeatures.py — Does a RICHER ML entry-gate find edge the linear
AUC-0.52 model missed?

HYPOTHESIS
----------
The base long VWAP mean-reversion strategy is ~flat-to-negative net of costs. A
richer decision-time feature set (RSI, distance from prior-day close, realized
intraday vol, time-of-day, bar range, volume z-score, dist-from-VWAP, vol-ratio,
recent return) feeding a learned classifier can SEPARATE winning from losing base
candidates well enough that gating entries on the model's win-probability improves
net OUT-OF-SAMPLE return versus the ungated base — with honest OOS AUC > 0.5.

WHY THIS COULD BE FOOLING OURSELVES (and how we guard)
------------------------------------------------------
- LEAKAGE is the #1 risk (we already saw a fake AUC 0.72 from leakage). Every feature
  here is computed STRICTLY from bars up to & including the decision bar t (the same
  no-look-ahead windowing the harness uses for dist/vol_ratio). The label is the
  realized net-of-cost trade outcome, which depends on FUTURE bars — but the label is
  used ONLY for offline TRAIN fitting, never fed into a same-bar feature.
- The model is fit on TRAIN ONLY (scaler + classifier). The probability THRESHOLD is
  chosen on VALIDATION ONLY. TEST is evaluated EXACTLY ONCE per reported variant.
- The gate is deployed through the VALIDATED harness `score_fn` hook, so the gated
  backtest reuses the exact same t->t+1 fill / cost / cap machinery as the base. The
  score_fn receives only decision-time fields and looks up the precomputed rich
  features by an exact (symbol, minute, close, vwap) key (verified 0 collisions), so
  no future information can enter the decision.

PROTOCOL
--------
- Chronological global split: TRAIN 60% / VAL 20% / TEST 20% (harness.chronological_split).
- TRAIN labels: run the UNGATED base long strategy on TRAIN; each realized trade is mapped
  back to its decision bar (entry_idx - 1) to recover the decision-time rich features;
  label = 1 iff realized return_pct (net of 1bp/side slippage) > 0.
- Fit StandardScaler + classifier on TRAIN only. We compare a small, pre-listed set of
  model configs (this is the multiple-comparisons budget); SELECTION of the model is by
  VALIDATION AUC, then the gating THRESHOLD is selected by VALIDATION net total_pnl.
- Diagnostics: OOS AUC on train/val/test (candidate-level), and gated-vs-ungated net
  return on all three splits. The single selected (model, threshold) variant is evaluated
  ONCE on TEST.

Run:
    cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.exp_ml_richfeatures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.harness import (
    chronological_split,
    compute_indicators_df,
    load_bars,
    research_backtest,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]

BASE_SPEC = dict(
    side="long",
    entry_dist=0.005,
    vol_mult=1.2,
    max_hold=15,
    vwap_exit_band=0.001,
    stop_loss=0.005,
    slippage_bps=1.0,
)

RESULTS_PATH = Path("/Users/pnle/Desktop/alpaca-cli/experiments/results/ml_richfeatures.json")

# Ordered, fixed feature list used by the model.
FEATURES = [
    "dist_from_vwap",
    "volume_ratio",
    "minute_of_session",
    "session_progress",
    "recent_return",
    "bar_range_pct",
    "rsi14",
    "dist_prevclose",
    "realized_vol",
    "vol_z",
]


# --------------------------------------------------------------------------- #
# Rich decision-time feature table (strictly no look-ahead).
# --------------------------------------------------------------------------- #
def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-style RSI computed causally (value at i uses bars[0..i] only)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > period:
        # first averages at index `period` use the first `period` deltas (indices 0..period-1)
        ag = gain[:period].mean()
        al = loss[:period].mean()
        avg_gain[period] = ag
        avg_loss[period] = al
        for i in range(period + 1, n):
            ag = (ag * (period - 1) + gain[i - 1]) / period
            al = (al * (period - 1) + loss[i - 1]) / period
            avg_gain[i] = ag
            avg_loss[i] = al
        for i in range(period, n):
            if avg_loss[i] == 0:
                out[i] = 100.0 if avg_gain[i] > 0 else 50.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def build_feature_table(bars: Dict[str, pd.DataFrame]) -> Dict[Tuple, dict]:
    """Map (symbol, minute, round(close,4), round(vwap,4)) -> rich decision-time feats.

    Every feature uses only bars up to & including the decision bar (causal). VWAP /
    dist / vol_ratio / recent_return / bar_range_pct come straight from the harness'
    indicator computation; RSI, realized intraday vol, and volume z-score are added per
    session here; dist_prevclose uses the PRIOR day's last close (known at the open).
    """
    table: Dict[Tuple, dict] = {}
    for sym, df in bars.items():
        if df.empty:
            continue
        ind = compute_indicators_df(df)
        # prior-day last close per date (shifted; the open of day D knows D-1's close).
        day_last_close = df.groupby(df["date"])["close"].last()
        prev_close_map = day_last_close.shift(1).to_dict()
        for day, day_df in ind.groupby(ind.index.date, sort=True):
            close = day_df["close"].to_numpy(dtype=float)
            n = len(close)
            rsi = _rsi(close, 14)
            # realized intraday vol: causal std of 1-min log returns within the session.
            logret = np.zeros(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                logret[1:] = np.log(close[1:] / close[:-1])
            realized_vol = np.full(n, np.nan)
            for i in range(n):
                if i >= 5:
                    realized_vol[i] = np.std(logret[1 : i + 1], ddof=0)
            # volume z-score vs rolling-20 (excl current) mean & std.
            vol = day_df["volume"].to_numpy(dtype=float)
            vol_z = np.full(n, np.nan)
            for i in range(n):
                lo = max(0, i - 20)
                if i - lo >= 5:
                    w = vol[lo:i]
                    s = w.std(ddof=0)
                    if s > 0:
                        vol_z[i] = (vol[i] - w.mean()) / s
                    else:
                        vol_z[i] = 0.0
            pc = prev_close_map.get(day, np.nan)
            dist_prevclose = (close - pc) / pc if (pc is not None and not pd.isna(pc) and pc) else np.full(n, np.nan)

            dist = day_df["dist_from_vwap"].to_numpy(dtype=float)
            vr = day_df["volume_ratio"].to_numpy(dtype=float)
            minute = day_df["minute"].to_numpy(dtype=int)
            sess_prog = day_df["session_progress"].to_numpy(dtype=float)
            recret = day_df["recent_return"].to_numpy(dtype=float)
            brange = day_df["bar_range_pct"].to_numpy(dtype=float)
            vwap = day_df["session_vwap"].to_numpy(dtype=float)

            for i in range(n):
                key = (sym, int(minute[i]), round(float(close[i]), 4), round(float(vwap[i]), 4))
                table[key] = {
                    "dist_from_vwap": float(dist[i]) if not np.isnan(dist[i]) else 0.0,
                    "volume_ratio": float(vr[i]) if not np.isnan(vr[i]) else 0.0,
                    "minute_of_session": float(minute[i]),
                    "session_progress": float(sess_prog[i]),
                    "recent_return": float(recret[i]) if not np.isnan(recret[i]) else 0.0,
                    "bar_range_pct": float(brange[i]) if not np.isnan(brange[i]) else 0.0,
                    "rsi14": float(rsi[i]) if not np.isnan(rsi[i]) else 50.0,
                    "dist_prevclose": float(dist_prevclose[i]) if not np.isnan(dist_prevclose[i]) else 0.0,
                    "realized_vol": float(realized_vol[i]) if not np.isnan(realized_vol[i]) else 0.0,
                    "vol_z": float(vol_z[i]) if not np.isnan(vol_z[i]) else 0.0,
                }
    return table


def _feat_key_from_scorefn(feat: dict) -> Tuple:
    return (
        feat["symbol"],
        int(feat["minute_of_session"]),
        round(float(feat["last_price"]), 4),
        round(float(feat["session_vwap"]), 4),
    )


def _vec(d: dict) -> np.ndarray:
    return np.array([d[f] for f in FEATURES], dtype=float)


# --------------------------------------------------------------------------- #
# Collect (features, label) from realized base trades on a split.
# --------------------------------------------------------------------------- #
def collect_xy(split: Dict[str, pd.DataFrame], table: Dict[Tuple, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Run ungated base on `split`; recover decision-time rich features per realized
    trade (decision bar = entry_idx - 1) and label = 1 iff net return_pct > 0."""
    res = research_backtest(split, BASE_SPEC)
    ind_cache: Dict[str, pd.DataFrame] = {}
    X: List[np.ndarray] = []
    y: List[int] = []
    for tr in res["trades"]:
        sym = tr["symbol"]
        if tr["exit_reason"] == "forced_close":
            continue
        if sym not in ind_cache:
            ind_cache[sym] = compute_indicators_df(split[sym])
        ind = ind_cache[sym]
        pos = ind.index.get_indexer([pd.Timestamp(tr["entry_time"])])
        if pos[0] <= 0:
            continue
        dec = ind.iloc[pos[0] - 1]
        key = (sym, int(dec["minute"]), round(float(dec["close"]), 4), round(float(dec["session_vwap"]), 4))
        feats = table.get(key)
        if feats is None:
            continue
        X.append(_vec(feats))
        y.append(1 if tr["return_pct"] > 0 else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


# --------------------------------------------------------------------------- #
# score_fn factory: wraps a fitted (scaler, model) + feature table as a gate.
# --------------------------------------------------------------------------- #
def make_score_fn(scaler: StandardScaler, model, table: Dict[Tuple, dict]):
    def score_fn(feat: dict) -> float:
        key = _feat_key_from_scorefn(feat)
        d = table.get(key)
        if d is None:
            # decision bar not in table (should not happen on these splits) -> neutral pass.
            return 1.0
        x = scaler.transform(_vec(d).reshape(1, -1))
        return float(model.predict_proba(x)[0, 1])
    return score_fn


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

    print("=" * 92)
    print("exp_ml_richfeatures — richer ML entry-gate on base long VWAP mean-reversion candidates")
    print("=" * 92)
    print(f"Split date ranges (global, by-date): {ranges}\n")

    # Build the rich feature lookup table on the FULL history (each entry is purely a
    # function of bars up to that bar; splitting it later is just a date filter on keys).
    table = build_feature_table(bars)
    print(f"Rich feature table built: {len(table):,} decision-bar feature vectors "
          f"(features: {FEATURES})\n")

    # ---- Ungated base metrics per split (the thing the gate must beat). ----
    base_metrics = {}
    print("UNGATED base long (the gate must improve net return vs this):")
    for nm, split in [("train", tr), ("val", va), ("test", te)]:
        m = research_backtest(split, BASE_SPEC)["metrics"]
        base_metrics[nm] = _slim(m)
        print(f"  {nm:<5} n={m['n_trades']:>4} pnl={m['total_pnl']:>8.3f} "
              f"ret={m['total_return']:>8.4f} wr={m['win_rate']:.3f}")
    print()

    # ---- Build TRAIN / VAL / TEST candidate-level (X, y) for AUC + fitting. ----
    Xtr, ytr = collect_xy(tr, table)
    Xva, yva = collect_xy(va, table)
    Xte, yte = collect_xy(te, table)
    print(f"Labeled candidates: TRAIN={len(ytr)} (pos={ytr.mean():.3f}) "
          f"VAL={len(yva)} (pos={yva.mean():.3f}) TEST={len(yte)} (pos={yte.mean():.3f})\n")

    # ---- Fit scaler on TRAIN only; compare a small fixed set of models by VAL AUC. ----
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    model_configs = {
        "logreg_C1": LogisticRegression(C=1.0, max_iter=2000),
        "logreg_C0.1": LogisticRegression(C=0.1, max_iter=2000),
        "gbm_shallow": GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
        ),
        "gbm_stumps": GradientBoostingClassifier(
            n_estimators=200, max_depth=1, learning_rate=0.05, subsample=0.8, random_state=0
        ),
    }
    n_configs_tried = len(model_configs)

    print("Model AUC by split (model SELECTED by VAL AUC; TEST AUC reported once, diagnostic):")
    auc_rows = {}
    fitted = {}
    for name, mdl in model_configs.items():
        mdl.fit(Xtr_s, ytr)
        fitted[name] = mdl
        a_tr = roc_auc_score(ytr, mdl.predict_proba(Xtr_s)[:, 1]) if len(set(ytr)) > 1 else float("nan")
        a_va = roc_auc_score(yva, mdl.predict_proba(Xva_s)[:, 1]) if len(set(yva)) > 1 else float("nan")
        a_te = roc_auc_score(yte, mdl.predict_proba(Xte_s)[:, 1]) if len(set(yte)) > 1 else float("nan")
        auc_rows[name] = {"train": round(a_tr, 4), "val": round(a_va, 4), "test": round(a_te, 4)}
        print(f"  {name:<14} AUC  train={a_tr:.4f}  val={a_va:.4f}  test={a_te:.4f}")

    best_model_name = max(auc_rows, key=lambda k: auc_rows[k]["val"])
    best_model = fitted[best_model_name]
    print(f"\nSELECTED model by VAL AUC: {best_model_name} "
          f"(val AUC={auc_rows[best_model_name]['val']:.4f})\n")

    # ---- Threshold selection on VALIDATION only (gated backtest via score_fn). ----
    score_fn = make_score_fn(scaler, best_model, table)
    gated_base = dict(BASE_SPEC)
    gated_base["score_fn"] = score_fn

    thresholds = [round(x, 3) for x in np.arange(0.40, 0.66, 0.02)]
    print("Threshold sweep on VALIDATION (pick threshold maximizing VAL net total_pnl, n>=20):")
    val_curve = {}
    best_thr = None
    best_val_pnl = -1e18
    for thr in thresholds:
        spec = dict(gated_base)
        spec["score_threshold"] = thr
        mv = research_backtest(va, spec)["metrics"]
        val_curve[str(thr)] = {"n": mv["n_trades"], "pnl": round(mv["total_pnl"], 4),
                               "wr": round(mv["win_rate"], 4)}
        flag = ""
        if mv["n_trades"] >= 20 and mv["total_pnl"] > best_val_pnl:
            best_val_pnl = mv["total_pnl"]
            best_thr = thr
            flag = "  <- best so far"
        print(f"  thr={thr:.2f}  VAL n={mv['n_trades']:>4} pnl={mv['total_pnl']:>7.3f} "
              f"wr={mv['win_rate']:.3f}{flag}")

    if best_thr is None:
        # no threshold cleared the >=20-trade bar on VAL; fall back to the loosest one.
        best_thr = thresholds[0]
        print(f"\n  (No VAL threshold reached >=20 trades while positive; falling back to {best_thr})")
    print(f"\nSELECTED threshold (VAL net pnl, n>=20): {best_thr}\n")

    # ---- Evaluate the single selected (model, threshold) variant ONCE per split. ----
    sel_spec = dict(gated_base)
    sel_spec["score_threshold"] = best_thr
    gated_metrics = {}
    print(f"GATED strategy ({best_model_name} @ thr={best_thr}) vs UNGATED base, all splits:")
    print(f"  {'split':<6} {'GATED n  pnl     ret      wr':<34} | {'BASE n  pnl     ret':<26}")
    for nm, split in [("train", tr), ("val", va), ("test", te)]:
        mg = research_backtest(split, sel_spec)["metrics"]
        gated_metrics[nm] = _slim(mg)
        b = base_metrics[nm]
        print(f"  {nm:<6} g: n={mg['n_trades']:>4} pnl={mg['total_pnl']:>7.3f} "
              f"ret={mg['total_return']:>8.4f} wr={mg['win_rate']:.3f}  |  "
              f"base: n={b['n_trades']:>4} pnl={b['total_pnl']:>7.3f} ret={b['total_return']:>8.4f}")

    # ---- Verdict. ----
    g_te = gated_metrics["test"]
    b_te = base_metrics["test"]
    g_tr = gated_metrics["train"]
    test_auc = auc_rows[best_model_name]["test"]
    val_auc = auc_rows[best_model_name]["val"]

    # An honest "edge" requires ALL of:
    #   (1) gated TEST net total_pnl > 0,
    #   (2) >= 20 TEST trades,
    #   (3) gated TEST net return strictly beats the ungated base TEST net return
    #       (the gate must actually add value, not just inherit the base's TEST sign),
    #   (4) OOS (test) AUC meaningfully > 0.5 (>= 0.52) — separation is real, not noise,
    #   (5) train->test not a catastrophic sign-flip on the gate's own training fit.
    survives = bool(
        g_te["total_pnl"] > 0
        and g_te["n_trades"] >= 20
        and g_te["total_return"] > b_te["total_return"]
        and test_auc >= 0.52
        and g_tr["total_pnl"] > -50.0  # sanity: gate didn't blow up its own training sample
    )

    print()
    print("VERDICT:")
    print(f"  TEST AUC={test_auc:.4f} (val AUC={val_auc:.4f})")
    print(f"  GATED TEST: n={g_te['n_trades']} pnl={g_te['total_pnl']:.3f} ret={g_te['total_return']:.5f}")
    print(f"  BASE  TEST: n={b_te['n_trades']} pnl={b_te['total_pnl']:.3f} ret={b_te['total_return']:.5f}")
    print(f"  gate improves TEST net return vs base: {g_te['total_return'] > b_te['total_return']}")
    print(f"  survives_oos = {survives}")

    out = {
        "name": "ml_richfeatures",
        "hypothesis": (
            "A richer decision-time ML entry-gate (RSI, dist-from-prev-close, realized "
            "intraday vol, time-of-day, bar range, volume z-score, dist-from-VWAP, "
            "vol-ratio, recent return) separates winning base long candidates and improves "
            "net OOS return vs the ungated base."
        ),
        "split_ranges": ranges,
        "features": FEATURES,
        "base_spec": {k: v for k, v in BASE_SPEC.items()},
        "n_feature_vectors": len(table),
        "labeled_counts": {
            "train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte)),
            "train_pos_rate": round(float(ytr.mean()), 4) if len(ytr) else None,
            "val_pos_rate": round(float(yva.mean()), 4) if len(yva) else None,
            "test_pos_rate": round(float(yte.mean()), 4) if len(yte) else None,
        },
        "auc_by_model": auc_rows,
        "selected_model": best_model_name,
        "val_threshold_curve": val_curve,
        "selected_threshold": best_thr,
        "n_configs_tried": n_configs_tried,
        "base_metrics": base_metrics,
        "gated_metrics": gated_metrics,
        "selected_test_auc": test_auc,
        "selected_val_auc": val_auc,
        "survives_oos": survives,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    main()
