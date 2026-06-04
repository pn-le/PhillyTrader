"""ml/features.py — decision-time feature construction (STRATEGY_RULES §5.1).

PURE functions, NO I/O. The whole point of this module is that live decisioning and
offline training produce IDENTICAL feature layouts. FEATURE_ORDER is the single source
of truth for column order; build_features() / to_row() / feature_names() all derive from
it. Absolutely NO future-looking fields: every input is known at minute t.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from ..config import RECENT_RETURN_K, SESSION_MINUTES, UNIVERSE, StrategyParams
from ..types import Bar, FeatureVector, Snapshot

# Stable, ordered feature layout. CHANGING THIS ORDER IS A BREAKING CONTRACT CHANGE —
# any trained model is keyed to it. `symbol_id` is the categorical (A8); it is RECORDED here
# for traceability but the linear trainer DROPS it before fitting (train_model._EXCLUDE_FROM_MODEL)
# because there is no one-hot encoding in this pipeline and z-scoring an arbitrary ordinal index
# would inject a spurious magnitude ordering. A future tree model could consume it directly.
FEATURE_ORDER: List[str] = [
    "dist_from_vwap",
    "volume_ratio",
    "log_rolling_vol",
    "minute_of_session",
    "recent_return",
    "bar_range_pct",
    "session_progress",
    "symbol_id",
]


def feature_names() -> List[str]:
    """Return a copy of the canonical ordered feature-name list."""
    return list(FEATURE_ORDER)


def to_row(fv: FeatureVector) -> List[float]:
    """Flatten a FeatureVector into a numeric row in FEATURE_ORDER (no symbol string).

    Mirror of feature_names(): row[i] corresponds to feature_names()[i]. Used by both
    the live scorer and dataset construction so layouts can never drift.
    """
    mapping = {
        "dist_from_vwap": fv.dist_from_vwap,
        "volume_ratio": fv.volume_ratio,
        "log_rolling_vol": fv.log_rolling_vol,
        "minute_of_session": float(fv.minute_of_session),
        "recent_return": fv.recent_return,
        "bar_range_pct": fv.bar_range_pct,
        "session_progress": fv.session_progress,
        "symbol_id": float(fv.symbol_id),
    }
    return [float(mapping[name]) for name in FEATURE_ORDER]


def _minute_of_session(start, session_open_minutes: int = 9 * 60 + 30) -> int:
    """Minutes since 09:30 of the decision-bar start (clamped to 0..SESSION_MINUTES-1)."""
    mins = start.hour * 60 + start.minute - session_open_minutes
    if mins < 0:
        mins = 0
    if mins > SESSION_MINUTES - 1:
        mins = SESSION_MINUTES - 1
    return int(mins)


def build_features(snapshot: Snapshot, params: StrategyParams | None = None) -> FeatureVector:
    """Build the decision-time FeatureVector for a candidate entry. PURE; NO future data.

    Uses snapshot.indicators (dist_from_vwap, volume_ratio, rolling20_avg_vol) and the
    completed bar window snapshot.bars[0..n-1]. recent_return uses K=RECENT_RETURN_K and
    requires n >= K+1 (guaranteed for entry candidates since the entry guard needs n>=21).
    `params` is accepted for forward-compatibility (e.g. configurable K) but K is fixed (A7).

    Raises ValueError if the snapshot lacks the data needed to form a valid feature row
    (callers gate on indicators.valid before calling, so this is a defensive guard).
    """
    ind = snapshot.indicators
    bars: Sequence[Bar] = snapshot.bars
    n = len(bars)

    if n < RECENT_RETURN_K + 1:
        raise ValueError(f"need >= {RECENT_RETURN_K + 1} bars for features, got {n}")
    if ind.dist_from_vwap is None or ind.volume_ratio is None or ind.rolling20_avg_vol is None:
        raise ValueError("indicators incomplete; cannot build features")

    decision = bars[n - 1]
    prior = bars[n - 1 - RECENT_RETURN_K]

    if prior.close == 0 or decision.close == 0:
        raise ValueError("zero close price; cannot build features")

    recent_return = (decision.close / prior.close) - 1.0
    bar_range_pct = (decision.high - decision.low) / decision.close
    minute_of_session = _minute_of_session(decision.start)
    session_progress = minute_of_session / float(SESSION_MINUTES - 1)  # 0..1

    try:
        symbol_id = UNIVERSE.index(snapshot.symbol)
    except ValueError:
        symbol_id = -1  # out-of-universe symbol; recorded as -1 (should not happen)

    rolling = ind.rolling20_avg_vol if ind.rolling20_avg_vol is not None else 0.0
    return FeatureVector(
        symbol=snapshot.symbol,
        dist_from_vwap=float(ind.dist_from_vwap),
        volume_ratio=float(ind.volume_ratio),
        log_rolling_vol=float(math.log1p(max(0.0, rolling))),
        minute_of_session=minute_of_session,
        recent_return=float(recent_return),
        bar_range_pct=float(bar_range_pct),
        session_progress=float(session_progress),
        symbol_id=int(symbol_id),
    )


__all__ = ["FEATURE_ORDER", "feature_names", "to_row", "build_features"]
