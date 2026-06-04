"""ml/dataset.py — build & persist the ML training dataset.

Turns closed (or backtest-simulated) round-trip trades into a numeric design matrix the
trainer can fit on. The column order is ALWAYS ``ml.features.FEATURE_ORDER`` (INV — layout
parity), so the live scorer and the trained model see identical layouts.

  - ``build_dataset(trade_records) -> (X, y, feature_names)``
      X : np.ndarray (n_rows, n_features) in FEATURE_ORDER, from each record's ``features``.
      y : np.ndarray (n_rows,) of int labels — 1 iff the trade was profitable, else 0.
      feature_names : == ml.features.feature_names() (a copy of FEATURE_ORDER).

  - ``save_dataset(X, y, feature_names, path) -> Path``
      Persist as Parquet (pyarrow, per config.DATASET_FORMAT) else CSV. Columns are
      ``feature_names + ['label']``.

  - ``load_dataset(path) -> (X, y, feature_names)``
      Inverse of save_dataset, with FEATURE_ORDER preserved.

Guards: tiny/empty inputs are handled gracefully (empty -> zero-row arrays with the right
shape; never raises on emptiness). Labels are derived defensively: prefer the record's
``label`` field, but fall back to ``pnl > 0`` so a mislabeled record can't poison training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..config import DATASET_FORMAT
from ..types import TradeRecord
from .features import FEATURE_ORDER, feature_names


def _label_for(rec: TradeRecord) -> int:
    """Profitable(1)/not(0). Trust ``pnl > 0`` (A11) as the source of truth; the stored
    ``label`` is a convenience mirror, so prefer pnl to avoid drift if they disagree."""
    try:
        return 1 if float(rec.pnl) > 0.0 else 0
    except (TypeError, ValueError):
        return int(rec.label) if rec.label in (0, 1) else 0


def _row_from_features(features: Dict[str, Any]) -> List[float]:
    """Extract a numeric row in FEATURE_ORDER from a record's decision-time feature dict.

    ``features`` is a FeatureVector-like dict (keys == FeatureVector field names, which are
    a superset of FEATURE_ORDER plus 'symbol'). Missing keys default to 0.0 so a slightly
    malformed record degrades instead of crashing the whole build.
    """
    row: List[float] = []
    for name in FEATURE_ORDER:
        val = features.get(name, 0.0)
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(0.0)
    return row


def build_dataset(
    trade_records: List[TradeRecord],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, feature_names) from closed/simulated trades.

    Rows are emitted in FEATURE_ORDER from each record's ``features`` dict; the label is 1
    iff the trade was profitable net of costs (pnl > 0). Records with empty/missing feature
    dicts are SKIPPED (they carry no decision-time signal to learn from). Returns zero-row
    arrays (shaped ``(0, len(FEATURE_ORDER))``) when there is nothing usable to learn from.
    """
    names = feature_names()
    n_features = len(names)

    rows: List[List[float]] = []
    labels: List[int] = []
    for rec in trade_records or []:
        feats = getattr(rec, "features", None)
        if not feats:  # None or empty dict -> nothing to learn from
            continue
        rows.append(_row_from_features(feats))
        labels.append(_label_for(rec))

    if not rows:
        return (
            np.empty((0, n_features), dtype=float),
            np.empty((0,), dtype=int),
            names,
        )

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y, names


def save_dataset(
    X: Any, y: Any, feature_names: Sequence[str], path: "str | Path"
) -> Path:
    """Persist the dataset to ``path`` as Parquet (pyarrow) else CSV.

    Columns = ``list(feature_names) + ['label']``. The parent directory is created. The
    chosen format follows ``config.DATASET_FORMAT`` but transparently falls back to CSV if
    pyarrow is unavailable, so the call never fails for a missing optional dependency.
    """
    import pandas as pd

    cols = list(feature_names)
    Xarr = np.asarray(X, dtype=float)
    yarr = np.asarray(y, dtype=int).reshape(-1)

    if Xarr.ndim != 2 or Xarr.shape[1] != len(cols):
        # Reshape an empty/degenerate matrix to the expected width so columns line up.
        Xarr = Xarr.reshape(-1, len(cols)) if Xarr.size else np.empty((0, len(cols)), dtype=float)
    if yarr.shape[0] != Xarr.shape[0]:
        yarr = np.zeros((Xarr.shape[0],), dtype=int)

    df = pd.DataFrame(Xarr, columns=cols)
    df["label"] = yarr

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # An explicit .csv path is always honored; otherwise default to Parquet (config) and
    # transparently fall back to CSV if pyarrow is somehow unavailable.
    want_parquet = DATASET_FORMAT == "parquet" and p.suffix != ".csv"
    if want_parquet:
        try:
            import pyarrow  # noqa: F401  (presence check)

            target = p if p.suffix == ".parquet" else p.with_suffix(".parquet")
            df.to_parquet(target, index=False)
            return target
        except Exception:
            pass  # fall through to CSV

    target = p if p.suffix == ".csv" else p.with_suffix(".csv")
    df.to_csv(target, index=False)
    return target


def load_dataset(path: "str | Path") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load (X, y, feature_names) from a dataset written by :func:`save_dataset`.

    Reads Parquet or CSV by file extension (auto-detecting a sibling if the exact suffix
    is absent). The returned feature_names preserve FEATURE_ORDER; ``label`` is split off
    into ``y``. Raises FileNotFoundError if no dataset file can be located.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        for alt in (p.with_suffix(".parquet"), p.with_suffix(".csv")):
            if alt.exists():
                p = alt
                break
        else:
            raise FileNotFoundError(f"no dataset at {path} (.parquet/.csv)")

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "label" not in df.columns:
        raise ValueError(f"dataset {p} missing 'label' column")

    # Preserve FEATURE_ORDER for the feature columns; ignore any extra columns.
    cols = [c for c in FEATURE_ORDER if c in df.columns]
    X = df[cols].to_numpy(dtype=float) if cols else np.empty((len(df), 0), dtype=float)
    y = df["label"].to_numpy(dtype=int).reshape(-1)
    return X, y, list(cols)


__all__ = ["build_dataset", "save_dataset", "load_dataset"]
