"""backtest/optimize.py — walk-forward, seeded random search over the trainable params.

Searches StrategyParams.TRAINABLE (entry_dist, vol_mult, max_hold, vwap_exit_band,
stop_loss, ml_threshold) for a robust configuration — robust meaning it scores well on
OUT-OF-SAMPLE validation windows, not just on the data it was tuned on.

Method (dependency-free; no optuna):
  1. Split the history's trading days into ``folds`` sequential time blocks. Within each
     fold we use a WALK-FORWARD split: the earlier portion is "train", the later portion is
     "validation". We optimize/score on the held-out validation window of every fold, so a
     param set must generalize forward in time, not memorize one stretch.
  2. Sample ``n_iter`` candidate param sets from the search space using numpy's SEEDED
     Generator (default seed=0) — fully reproducible, NO wall-clock / unseeded entropy.
  3. For each candidate, run_backtest on each fold's validation window, aggregate a
     risk-adjusted objective across folds, and PENALIZE inconsistency (reward the mean of
     per-fold scores minus their dispersion) so a single lucky fold can't win.
  4. Return the best StrategyParams plus a report (leaderboard + per-fold metrics).

Does NOT write files — the caller persists the winner via config.save_best_params.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from ..config import RiskLimits, StrategyParams
from ..types import Bar
from .engine import BacktestResult, run_backtest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ml.scorer import EntryScorer


# --------------------------------------------------------------------------- #
# Search space — (low, high) bounds per trainable param. Sampled uniformly by the
# seeded Generator and rounded to a sensible grid so the search is reproducible AND
# the chosen values are clean. ml_threshold is sampled but stays low/off-able so the
# rule-only system (no model) is never penalized into never trading.
# --------------------------------------------------------------------------- #
SEARCH_SPACE: Dict[str, Tuple[float, float]] = {
    "entry_dist": (0.002, 0.012),       # 0.2% .. 1.2% below VWAP
    "vol_mult": (1.0, 2.5),             # volume_ratio threshold
    "max_hold": (5.0, 30.0),            # minutes
    "vwap_exit_band": (0.0005, 0.004),  # 0.05% .. 0.4% of VWAP
    "stop_loss": (0.003, 0.012),        # 0.3% .. 1.2% loss
    "ml_threshold": (0.0, 0.6),         # learned-gate threshold (0 == disabled)
}

# Rounding grid per param (keeps sampled values clean + the search effectively discrete).
_ROUND: Dict[str, int] = {
    "entry_dist": 4,
    "vol_mult": 2,
    "max_hold": 0,
    "vwap_exit_band": 4,
    "stop_loss": 4,
    "ml_threshold": 2,
}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def optimize(
    history: Dict[str, List[Bar]],
    base_params: StrategyParams,
    limits: RiskLimits,
    n_iter: int = 50,
    folds: int = 3,
    scorer: "EntryScorer | None" = None,
    seed: int = 0,
) -> Tuple[StrategyParams, Dict[str, Any]]:
    """Walk-forward sampled search. Returns (best_params, report).

    `history` maps symbol -> ascending list of completed Bars (multiple trading days).
    `base_params` supplies the fixed sizing (notional) and the starting point; the trainable
    fields are overwritten by each sampled candidate. The objective is the cross-fold mean
    of each fold's risk-adjusted return minus a dispersion penalty (robustness). The first
    candidate evaluated is always `base_params` itself so a tuned config can only be beaten,
    never silently discarded. Deterministic for a given seed.
    """
    rng = np.random.default_rng(int(seed))

    fold_windows = _make_fold_validation_windows(history, folds)
    if not fold_windows:
        # Not enough data to validate on — return base params with an explanatory report.
        return base_params, {
            "status": "insufficient_history",
            "n_folds": 0,
            "n_iter": 0,
            "best_params": base_params.to_dict(),
            "leaderboard": [],
            "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
            "seed": int(seed),
        }

    candidates = _sample_candidates(base_params, n_iter, rng)

    leaderboard: List[Dict[str, Any]] = []
    for cand in candidates:
        per_fold = _evaluate_candidate(cand, fold_windows, limits, scorer)
        objective = _aggregate_objective(per_fold)
        leaderboard.append(
            {
                "params": cand.to_dict(),
                "objective": objective,
                "per_fold": per_fold,
                "total_trades": int(sum(f["n_trades"] for f in per_fold)),
            }
        )

    # Rank by objective desc; tie-break preferring more trades (more reliable signal) then
    # the candidate that appeared earlier (base_params first) for full determinism.
    leaderboard_sorted = sorted(
        enumerate(leaderboard),
        key=lambda iv: (iv[1]["objective"], iv[1]["total_trades"], -iv[0]),
        reverse=True,
    )
    ranked = [row for _i, row in leaderboard_sorted]

    best_row = ranked[0]
    best_params = StrategyParams.from_dict(best_row["params"])
    # Keep the caller's fixed sizing (notional) regardless of what the search carried.
    best_params = replace(best_params, notional=base_params.notional)

    report = {
        "status": "ok",
        "n_folds": len(fold_windows),
        "n_iter": len(candidates),
        "seed": int(seed),
        "objective": best_row["objective"],
        "best_params": best_params.to_dict(),
        "best_per_fold": best_row["per_fold"],
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        # Trimmed leaderboard (top 10) so the report stays human-readable.
        "leaderboard": [
            {
                "params": r["params"],
                "objective": round(r["objective"], 6),
                "total_trades": r["total_trades"],
            }
            for r in ranked[:10]
        ],
        "fold_windows": [
            {"fold": i, "val_days": [d.isoformat() for d in w["val_days"]]}
            for i, w in enumerate(fold_windows)
        ],
    }
    return best_params, report


# --------------------------------------------------------------------------- #
# Fold construction (walk-forward, day-aligned)
# --------------------------------------------------------------------------- #
def _make_fold_validation_windows(
    history: Dict[str, List[Bar]],
    folds: int,
) -> List[Dict[str, Any]]:
    """Split history's trading DAYS into `folds` sequential blocks; each fold's VALIDATION
    window is the later half of its block (walk-forward: tune on the front, validate on the
    held-out tail). Returns a list of {"val_days": [...dates...], "bars": {sym: [Bar,...]}}.

    Day-aligned so VWAP sessions are never cut mid-day. If there are too few distinct days
    (< 2 per fold after splitting), we degrade gracefully: with very little data we make a
    single fold whose validation window is the whole history (better than crashing). If
    there is no data at all we return [] and the caller short-circuits.
    """
    all_days = sorted({b.start.date() for bars in (history or {}).values() for b in (bars or ())})
    if not all_days:
        return []

    folds = max(1, int(folds))
    # If we can't give each fold its own block, collapse to as many folds as we have days.
    folds = min(folds, len(all_days))

    blocks = _split_sequential(all_days, folds)
    windows: List[Dict[str, Any]] = []
    for block in blocks:
        if not block:
            continue
        if len(block) == 1:
            # Single-day block: validate on that day (no separate train slice possible).
            val_days = block
        else:
            # Walk-forward: validation is the later half of the block.
            split = len(block) // 2
            val_days = block[split:]
        val_set = set(val_days)
        bars = {
            sym: [b for b in (bars_list or ()) if b.start.date() in val_set]
            for sym, bars_list in (history or {}).items()
        }
        # Drop symbols with no bars in this window to keep run_backtest input tight.
        bars = {sym: bl for sym, bl in bars.items() if bl}
        if bars:
            windows.append({"val_days": list(val_days), "bars": bars})
    return windows


def _split_sequential(items: List[Any], n: int) -> List[List[Any]]:
    """Split a list into `n` sequential, near-equal contiguous blocks (front-loaded)."""
    n = max(1, n)
    k, m = divmod(len(items), n)
    blocks: List[List[Any]] = []
    idx = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        blocks.append(items[idx: idx + size])
        idx += size
    return blocks


# --------------------------------------------------------------------------- #
# Candidate sampling (seeded) + evaluation
# --------------------------------------------------------------------------- #
def _sample_candidates(
    base_params: StrategyParams,
    n_iter: int,
    rng: "np.random.Generator",
) -> List[StrategyParams]:
    """Draw `n_iter` candidate StrategyParams from SEARCH_SPACE with the seeded Generator.

    The FIRST candidate is always `base_params` (so the incumbent is always evaluated and
    can only be beaten on out-of-sample objective). Remaining candidates are uniform draws
    in each param's bounds, rounded to its grid. Sizing (notional) is held fixed.
    """
    n_iter = max(1, int(n_iter))
    candidates: List[StrategyParams] = [replace(base_params)]
    for _ in range(n_iter - 1):
        overrides: Dict[str, float] = {}
        for name in StrategyParams.TRAINABLE:
            lo, hi = SEARCH_SPACE[name]
            raw = float(rng.uniform(lo, hi))
            overrides[name] = round(raw, _ROUND[name])
        candidates.append(replace(base_params, **overrides))
    return candidates


def _evaluate_candidate(
    cand: StrategyParams,
    fold_windows: List[Dict[str, Any]],
    limits: RiskLimits,
    scorer: "EntryScorer | None",
) -> List[Dict[str, Any]]:
    """Run the candidate on each fold's out-of-sample validation window. Returns one
    per-fold metrics dict (the fields the objective needs + diagnostics)."""
    per_fold: List[Dict[str, Any]] = []
    for w in fold_windows:
        result: BacktestResult = run_backtest(w["bars"], cand, limits, scorer=scorer)
        m = result.metrics
        per_fold.append(
            {
                "n_trades": int(m.get("n_trades", 0.0)),
                "total_pnl": float(m.get("total_pnl", 0.0)),
                "total_return": float(m.get("total_return", 0.0)),
                "win_rate": float(m.get("win_rate", 0.0)),
                "sharpe_like": float(m.get("sharpe_like", 0.0)),
                "max_drawdown": float(m.get("max_drawdown", 0.0)),
            }
        )
    return per_fold


def _fold_score(fold: Dict[str, Any]) -> float:
    """Per-fold risk-adjusted score.

    Combines total return with a Sharpe-like consistency reward and a drawdown penalty:
        score = total_return + 0.5 * sharpe_like + max_drawdown_dollars_normalized
    A fold with NO trades scores 0.0 (neither rewarded nor punished — it simply doesn't
    contribute evidence). max_drawdown is <= 0 so it subtracts; we scale it down so it
    nudges rather than dominates.
    """
    if fold["n_trades"] <= 0:
        return 0.0
    total_return = fold["total_return"]
    sharpe = fold["sharpe_like"]
    # Drawdown is in dollars on a ~$100-notional book; normalize by 100 to a return scale.
    dd_norm = fold["max_drawdown"] / 100.0
    return total_return + 0.5 * sharpe + 0.25 * dd_norm


def _aggregate_objective(per_fold: List[Dict[str, Any]]) -> float:
    """Aggregate per-fold scores into a single ROBUST objective.

    Robustness = reward the mean per-fold score but PENALIZE dispersion across folds, so a
    config that wins one fold and bombs another loses to a steadier config:
        objective = mean(scores) - 0.5 * std(scores)
    Folds with zero trades contribute a 0 score (they neither help nor hurt the mean beyond
    diluting it). If NO fold traded at all, the objective is a small negative so a config
    that never trades ranks below any config that does trade profitably.
    """
    if not per_fold:
        return -1.0
    scores = [_fold_score(f) for f in per_fold]
    traded = any(f["n_trades"] > 0 for f in per_fold)
    if not traded:
        return -1.0  # never-trades: rank below any trading config
    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std()) if arr.size > 1 else 0.0
    return mean - 0.5 * std


__all__ = ["optimize", "SEARCH_SPACE"]
