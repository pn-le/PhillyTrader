"""ml/train_model.py — fit the ML entry scorer with NO temporal leakage.

Trains the learned ENTRY gate consumed by ``ml.scorer.EntryScorer`` (STRATEGY_RULES §5).
The pipeline is deliberately leakage-proof and deterministic:

  1. Load the labeled dataset via ``ml.dataset.load_dataset`` (rows in FEATURE_ORDER, in the
     time order the backtest emitted them — earliest ENTRY first).
  2. TEMPORAL split (NO shuffling): the earlier block trains, the later block validates, so
     the model is never scored on rows that precede its own training window.
  3. Standardize features using TRAIN-ONLY mean/scale (fit on train, applied to val) and
     persist those params WITH the model so the live scorer re-applies the identical
     transform (``EntryScorer._standardize``).
  4. Fit the classifier: sklearn ``LogisticRegression`` when ``MODEL_BACKEND == 'sklearn'``,
     else a self-contained numpy logistic regression (gradient descent + L2). A fixed seed
     keeps any stochastic step reproducible.
  5. Persist to ``out_dir/ml_scorer.pkl`` as a dict bundle that ``EntryScorer.load`` accepts
     (joblib if available, else a JSON ``logreg_json`` weight dump), and write metrics +
     a short human-readable report to ``out_dir/``.

Tiny / degenerate datasets are handled gracefully: too few rows or a single-class dataset
cannot train a useful gate, so we WARN and persist a passthrough-equivalent artifact (or
clearly skip) rather than fitting a misleading model. The trade loop stays safe regardless
because ``ml_threshold`` defaults to 0.0 (gate disabled) until a real model proves out.

Returns a metrics dict: ``{n_train, n_val, auc, accuracy, precision, recall, val_winrate,
baseline_winrate, n_features, backend, trained, ...}``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import MODEL_BACKEND, MODELS_DIR
from .dataset import load_dataset
from .features import feature_names

# A real (non-degenerate) fit needs at least this many rows AND a validation tail that is
# big enough to mean anything. Below this we warn and persist a passthrough-ish artifact.
_MIN_ROWS_TO_TRAIN = 20
_MIN_VAL_ROWS = 4
_VAL_FRACTION = 0.25  # last 25% of the (time-ordered) rows are the out-of-sample tail
_SEED = 42

# Features EXCLUDED from the fitted model. ``symbol_id`` is a CATEGORICAL (the arbitrary
# UNIVERSE index, A8) — it is NOT a magnitude. Feeding it to a linear model as a single
# z-scored ordinal would inject a spurious monotonic "symbol index" effect (SPY<QQQ<...<AMZN)
# that is pure leakage of the universe ordering, and standardizing a categorical also distorts
# the intercept. There is no one-hot encoder in this pipeline, so the correct minimal fix is
# to DROP symbol_id from the model entirely (small fixed universe; per-symbol signal is weak
# vs. the leakage risk). It is still recorded in the dataset (FEATURE_ORDER) for traceability;
# the model's persisted feature_names simply omit it, and EntryScorer remaps by name at serve.
_EXCLUDE_FROM_MODEL = ("symbol_id",)

# numpy-logreg fallback hyperparameters (only used when sklearn is unavailable).
_LR = 0.1
_EPOCHS = 2000
_L2 = 1e-2


# --------------------------------------------------------------------------- #
# Standardization (fit on TRAIN only — no leakage)
# --------------------------------------------------------------------------- #
def _fit_scaler(X: np.ndarray) -> Tuple[List[float], List[float]]:
    """Return (mean, scale) from TRAIN rows only. Zero-variance columns get scale=1.0."""
    if X.size == 0:
        n = X.shape[1] if X.ndim == 2 else 0
        return [0.0] * n, [1.0] * n
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale <= 0.0, 1.0, scale)  # avoid divide-by-zero on constant columns
    return mean.astype(float).tolist(), scale.astype(float).tolist()


def _apply_scaler(X: np.ndarray, mean: List[float], scale: List[float]) -> np.ndarray:
    if X.size == 0:
        return X
    m = np.asarray(mean, dtype=float)
    s = np.asarray(scale, dtype=float)
    return (X - m) / s


# --------------------------------------------------------------------------- #
# Metrics (dependency-light; AUC + precision/recall computed by hand so the report
# is identical whether or not sklearn.metrics is importable)
# --------------------------------------------------------------------------- #
def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Rank-based ROC AUC (Mann–Whitney). None if only one class present."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks for ties so the statistic is exact.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _precision_recall_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """Return (precision, recall, accuracy) for the positive (profitable) class."""
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    return precision, recall, accuracy


# --------------------------------------------------------------------------- #
# numpy logistic-regression fallback (used only when sklearn is unavailable)
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _fit_numpy_logreg(
    X: np.ndarray, y: np.ndarray, *, lr: float = _LR, epochs: int = _EPOCHS, l2: float = _L2
) -> Tuple[np.ndarray, float]:
    """Batch-gradient-descent logistic regression with L2. Deterministic (zero init)."""
    rng = np.random.default_rng(_SEED)  # seeded even though init is zeros (future-proof)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    yf = y.astype(float)
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        err = p - yf
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    _ = rng  # keep the seeded generator referenced for reproducibility guarantees
    return w, b


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _persist_bundle(bundle: Dict[str, Any], out_dir: Path) -> Tuple[Path, str]:
    """Persist the model bundle to ``out_dir/ml_scorer.pkl``.

    Tries joblib (the verified MODEL_PERSIST backend). If joblib or pickling the sklearn
    estimator fails, fall back to a JSON ``logreg_json`` weight dump that EntryScorer can
    still load (only possible when we have linear coefficients available).
    Returns (path, persist_mode).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "ml_scorer.pkl"

    # Primary: joblib pickle of the full bundle (sklearn estimator OR logreg_json dict).
    try:
        import joblib  # type: ignore

        joblib.dump(bundle, pkl_path)
        return pkl_path, "joblib"
    except Exception:
        pass

    # Fallback: JSON weight dump. Only works if we have linear coef/intercept on hand.
    coef = bundle.get("coef")
    if coef is not None:
        json_payload = {
            "backend": "logreg_json",
            "coef": list(coef),
            "intercept": float(bundle.get("intercept", 0.0)),
            "mean": bundle.get("mean"),
            "scale": bundle.get("scale"),
            "feature_names": bundle.get("feature_names"),
        }
        json_path = out_dir / "ml_scorer.json"
        json_path.write_text(json.dumps(json_payload, indent=2))
        return json_path, "json"

    # Last resort: write the JSON-able subset so something is on disk; scorer will treat a
    # model-less bundle as passthrough.
    json_path = out_dir / "ml_scorer.json"
    json_path.write_text(
        json.dumps(
            {
                "backend": bundle.get("backend", "sklearn"),
                "mean": bundle.get("mean"),
                "scale": bundle.get("scale"),
                "feature_names": bundle.get("feature_names"),
            },
            indent=2,
        )
    )
    return json_path, "json"


def _write_metrics(metrics: Dict[str, Any], out_dir: Path) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def _write_report(metrics: Dict[str, Any], out_dir: Path) -> None:
    """Write a short, human-readable training report next to the metrics."""
    lines: List[str] = []
    lines.append("# ML Entry Scorer — Training Report")
    lines.append("")
    status = "TRAINED" if metrics.get("trained") else "SKIPPED (passthrough)"
    lines.append(f"Status: {status}")
    if metrics.get("warning"):
        lines.append(f"Warning: {metrics['warning']}")
    lines.append("")
    lines.append(f"Backend            : {metrics.get('backend')}")
    lines.append(f"Persist mode       : {metrics.get('persist_mode')}")
    lines.append(f"Model path         : {metrics.get('model_path')}")
    lines.append(f"Features ({metrics.get('n_features')}) : {metrics.get('feature_names')}")
    lines.append("")
    lines.append(f"n_rows total       : {metrics.get('n_rows')}")
    lines.append(f"n_train            : {metrics.get('n_train')}")
    lines.append(f"n_val              : {metrics.get('n_val')}")
    lines.append("")

    def _fmt(v: Any) -> str:
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    lines.append(f"baseline_winrate   : {_fmt(metrics.get('baseline_winrate'))}  (val base rate)")
    lines.append(f"train_winrate      : {_fmt(metrics.get('train_winrate'))}")
    lines.append(f"val_winrate        : {_fmt(metrics.get('val_winrate'))}")
    lines.append("")
    lines.append(f"accuracy (val)     : {_fmt(metrics.get('accuracy'))}")
    lines.append(f"auc (val)          : {_fmt(metrics.get('auc'))}")
    lines.append(f"precision (val)    : {_fmt(metrics.get('precision'))}")
    lines.append(f"recall (val)       : {_fmt(metrics.get('recall'))}")
    lines.append("")
    lines.append("Note: ml_threshold defaults to 0.0 (gate disabled). Raise it only after the")
    lines.append("validation AUC/precision justify gating live entries on this model.")
    lines.append("")
    (out_dir / "training_report.txt").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Passthrough artifact (tiny / single-class datasets)
# --------------------------------------------------------------------------- #
def _persist_passthrough(
    out_dir: Path, names: List[str], n_rows: int, n_train: int, n_val: int, warning: str
) -> Dict[str, Any]:
    """Persist a model-less bundle (EntryScorer loads it as passthrough -> p_win == 1.0)."""
    bundle = {
        "model": None,
        "backend": "passthrough",
        "mean": None,
        "scale": None,
        "feature_names": list(names),
    }
    path, persist_mode = _persist_bundle(bundle, out_dir)
    metrics: Dict[str, Any] = {
        "trained": False,
        "warning": warning,
        "backend": "passthrough",
        "persist_mode": persist_mode,
        "model_path": str(path),
        "n_rows": int(n_rows),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_features": len(names),
        "feature_names": list(names),
        "auc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "val_winrate": None,
        "train_winrate": None,
        "baseline_winrate": None,
        "seed": _SEED,
    }
    _write_metrics(metrics, out_dir)
    _write_report(metrics, out_dir)
    print(f"[train_model] WARNING: {warning} -> persisted passthrough model at {path}")
    return metrics


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def train(dataset_path: "str | Path", out_dir: "str | Path | None" = None) -> Dict[str, Any]:
    """Train the ML entry scorer from a labeled dataset; persist model + metrics.

    See module docstring. ``out_dir`` defaults to ``config.MODELS_DIR``. The artifact at
    ``out_dir/ml_scorer.pkl`` is directly consumable by ``ml.scorer.EntryScorer.load``.
    Never raises on a tiny/degenerate dataset — it warns and persists a passthrough model.
    """
    out = Path(out_dir) if out_dir is not None else Path(MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load (rows already in FEATURE_ORDER, in backtest decision-time order).
    X, y, names = load_dataset(dataset_path)
    if not names:
        names = feature_names()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    # Drop categorical/excluded columns (e.g. symbol_id) BEFORE fitting so the linear model
    # never z-scores an arbitrary ordinal. The persisted bundle records only the kept names,
    # and EntryScorer remaps an incoming row by name to this layout (scorer._row_for_model).
    keep_idx = [i for i, nm in enumerate(names) if nm not in _EXCLUDE_FROM_MODEL]
    if keep_idx and len(keep_idx) < len(names):
        X = X[:, keep_idx]
        names = [names[i] for i in keep_idx]

    n_rows = int(X.shape[0])

    # 2) Tiny-dataset guard.
    if n_rows < _MIN_ROWS_TO_TRAIN:
        n_val_est = max(0, int(round(n_rows * _VAL_FRACTION)))
        return _persist_passthrough(
            out, names, n_rows, n_rows - n_val_est, n_val_est,
            warning=f"only {n_rows} rows (< {_MIN_ROWS_TO_TRAIN}); too few to train a useful gate",
        )

    # 3) TEMPORAL split — NO shuffle. Earlier rows train; later tail validates (no leakage).
    n_val = max(_MIN_VAL_ROWS, int(round(n_rows * _VAL_FRACTION)))
    n_val = min(n_val, n_rows - 1)  # always keep at least 1 training row
    n_train = n_rows - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Single-class TRAIN set -> a classifier would be degenerate. Skip safely.
    if len(np.unique(y_train)) < 2:
        only = int(y_train[0]) if len(y_train) else -1
        return _persist_passthrough(
            out, names, n_rows, n_train, n_val,
            warning=f"training labels are single-class (all={only}); cannot fit a discriminative gate",
        )

    # 4) Standardize using TRAIN-ONLY stats (persist params with the model).
    mean, scale = _fit_scaler(X_train)
    Xs_train = _apply_scaler(X_train, mean, scale)
    Xs_val = _apply_scaler(X_val, mean, scale)

    backend = "sklearn"
    bundle: Dict[str, Any]
    val_scores: np.ndarray

    if MODEL_BACKEND == "sklearn":
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=_SEED,
                class_weight="balanced",  # ENTRY win/loss can be imbalanced
            )
            model.fit(Xs_train, y_train)
            # P(label==1): locate the class-1 column robustly.
            proba = model.predict_proba(Xs_val)
            classes = list(getattr(model, "classes_", [0, 1]))
            idx1 = classes.index(1) if 1 in classes else proba.shape[1] - 1
            val_scores = proba[:, idx1]
            # Persisted as scorer shape (1): bare estimator + separate mean/scale.
            bundle = {
                "model": model,
                "backend": "sklearn",
                "mean": mean,
                "scale": scale,
                "feature_names": list(names),
            }
        except Exception as exc:  # noqa: BLE001 — fall back to numpy logreg on any sklearn issue
            print(f"[train_model] sklearn unavailable/failed ({exc!r}); using numpy logreg fallback")
            backend = "logreg_json"
            w, b = _fit_numpy_logreg(Xs_train, y_train)
            val_scores = _sigmoid(Xs_val @ w + b)
            bundle = {
                "model": None,
                "backend": "logreg_json",
                "coef": w.astype(float).tolist(),
                "intercept": float(b),
                "mean": mean,
                "scale": scale,
                "feature_names": list(names),
            }
    else:
        backend = "logreg_json"
        w, b = _fit_numpy_logreg(Xs_train, y_train)
        val_scores = _sigmoid(Xs_val @ w + b)
        bundle = {
            "model": None,
            "backend": "logreg_json",
            "coef": w.astype(float).tolist(),
            "intercept": float(b),
            "mean": mean,
            "scale": scale,
            "feature_names": list(names),
        }

    # 5) Metrics on the out-of-sample validation tail.
    y_pred = (val_scores >= 0.5).astype(int)
    precision, recall, accuracy = _precision_recall_accuracy(y_val, y_pred)
    auc = _roc_auc(y_val, val_scores)
    baseline_winrate = float(np.mean(y_val)) if len(y_val) else None
    val_winrate = baseline_winrate
    train_winrate = float(np.mean(y_train)) if len(y_train) else None

    # 6) Persist model + metrics + report.
    path, persist_mode = _persist_bundle(bundle, out)

    metrics: Dict[str, Any] = {
        "trained": True,
        "warning": None,
        "backend": backend,
        "persist_mode": persist_mode,
        "model_path": str(path),
        "n_rows": n_rows,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_features": len(names),
        "feature_names": list(names),
        "auc": auc,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "val_winrate": val_winrate,
        "train_winrate": train_winrate,
        "baseline_winrate": baseline_winrate,
        "seed": _SEED,
    }
    _write_metrics(metrics, out)
    _write_report(metrics, out)

    auc_s = f"{auc:.4f}" if auc is not None else "n/a"
    print(
        f"[train_model] trained {backend} on {n_train} rows (val {n_val}): "
        f"acc={accuracy:.4f} auc={auc_s} prec={precision:.4f} rec={recall:.4f} "
        f"baseline={baseline_winrate} -> {path}"
    )
    return metrics


__all__ = ["train"]
