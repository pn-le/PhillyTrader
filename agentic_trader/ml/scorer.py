"""ml/scorer.py — Agent ③ ML entry scorer (learned gate on ENTRIES only).

The scorer is a deterministic function ``p_win = f(features) in [0, 1]`` that gates
candidate BUY proposals (STRATEGY_RULES §5). It NEVER touches exits.

Contract (INTERFACE_SPEC §7):
  - ``EntryScorer.load(path)`` loads a persisted model from ``config.ML_MODEL_PATH`` (or a
    given path). If the file is absent OR loading fails for any reason, it returns a
    PASSTHROUGH scorer whose ``p_win`` is always ``1.0`` so the whole system runs rule-only
    and safely (ml_threshold default = 0.0 means "disabled" even with a model present).
  - ``.p_win(feature_vector)`` returns a calibrated win probability in ``[0, 1]``.
  - ``.is_passthrough`` is True iff no model is loaded.

Layout parity (INV — "Layout parity"): features are flattened with ``ml.features.to_row``
in ``FEATURE_ORDER`` so the live scorer and the training pipeline produce identical rows.
Standardization params (mean/scale) are stored WITH the model and re-applied here so the
live transform matches exactly how the model was trained.

Persistence shapes this loader tolerates (so it integrates with train_model.py regardless
of exactly how it packs the artifact):
  1. A dict "bundle":  {"model": <sklearn estimator>, "mean": [...], "scale": [...],
                         "feature_names": [...], "backend": "sklearn"}.
  2. A bare sklearn estimator / Pipeline (self-contained; scaling already inside it).
  3. A numpy/JSON logistic-regression dump (fallback backend):
        {"backend": "logreg_json", "coef": [...], "intercept": <float>,
         "mean": [...], "scale": [...], "feature_names": [...]}.

If stored ``feature_names`` disagree with the current ``FEATURE_ORDER``, the row is
re-ordered to the model's expected layout (defensive; should never happen in practice).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence

from ..config import ML_MODEL_PATH
from ..types import FeatureVector
from .features import FEATURE_ORDER, feature_names, to_row


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid -> (0, 1)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _clip01(p: float) -> float:
    """Clamp a probability into the closed unit interval [0, 1]."""
    if p != p:  # NaN guard
        return 1.0
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)


class EntryScorer:
    """Learned probability-of-win gate for candidate ENTRIES.

    Construct via :meth:`load` (the public entrypoint). Direct construction is supported
    for tests: pass ``model=None`` for a passthrough scorer, or a loaded artifact plus its
    standardization params for a real model.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        mean: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        model_feature_names: Optional[List[str]] = None,
        backend: str = "sklearn",
        coef: Optional[Sequence[float]] = None,
        intercept: Optional[float] = None,
    ) -> None:
        self._model = model
        self._backend = backend
        self._mean = [float(x) for x in mean] if mean is not None else None
        self._scale = [float(x) for x in scale] if scale is not None else None
        # The feature layout the model was trained on. Default to the canonical order.
        self._model_features = list(model_feature_names) if model_feature_names else feature_names()
        # Pure-python logistic fallback weights (used only when backend == "logreg_json").
        self._coef = [float(c) for c in coef] if coef is not None else None
        self._intercept = float(intercept) if intercept is not None else 0.0

    # ----------------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------------- #
    @classmethod
    def load(cls, path: "str | Path | None" = None) -> "EntryScorer":
        """Load a persisted scorer; fall back to PASSTHROUGH on any problem.

        Tries joblib first (the verified MODEL_PERSIST backend), then a JSON weight dump.
        Any missing file / unreadable artifact / unexpected shape yields a passthrough
        scorer (p_win == 1.0) so the trade loop keeps running rule-only and never crashes
        on a model-loading issue.
        """
        p = Path(path) if path is not None else Path(ML_MODEL_PATH)
        if not p.exists():
            return cls(model=None)

        artifact = cls._read_artifact(p)
        if artifact is None:
            return cls(model=None)

        try:
            return cls._from_artifact(artifact)
        except Exception:
            # Defensive: a malformed-but-loadable artifact must not break trading.
            return cls(model=None)

    @staticmethod
    def _read_artifact(p: Path) -> Any:
        """Read the raw artifact from disk. Returns None if nothing usable loads."""
        # 1) joblib (primary persistence backend per ENV_REPORT).
        try:
            import joblib  # type: ignore

            return joblib.load(p)
        except Exception:
            pass
        # 2) JSON weight dump (fallback backend).
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    @classmethod
    def _from_artifact(cls, artifact: Any) -> "EntryScorer":
        """Build an EntryScorer from a loaded artifact of any tolerated shape."""
        # Shape (1)/(3): a dict bundle.
        if isinstance(artifact, dict):
            backend = str(artifact.get("backend", "sklearn"))
            mean = artifact.get("mean")
            scale = artifact.get("scale")
            feats = artifact.get("feature_names")

            if backend == "logreg_json" or ("coef" in artifact and "model" not in artifact):
                coef = artifact.get("coef")
                intercept = artifact.get("intercept", 0.0)
                # coef may be nested ([[...]] from sklearn) — flatten one level.
                if coef and isinstance(coef[0], (list, tuple)):
                    coef = list(coef[0])
                if coef is None:
                    return cls(model=None)
                return cls(
                    model=None,
                    backend="logreg_json",
                    coef=coef,
                    intercept=float(intercept) if not isinstance(intercept, (list, tuple)) else float(intercept[0]),
                    mean=mean,
                    scale=scale,
                    model_feature_names=feats,
                )

            model = artifact.get("model")
            if model is None or not hasattr(model, "predict_proba"):
                return cls(model=None)
            return cls(
                model=model,
                backend="sklearn",
                mean=mean,
                scale=scale,
                model_feature_names=feats,
            )

        # Shape (2): a bare sklearn estimator / Pipeline.
        if hasattr(artifact, "predict_proba"):
            return cls(model=artifact, backend="sklearn")

        # Unknown shape -> passthrough.
        return cls(model=None)

    # ----------------------------------------------------------------------- #
    # Scoring
    # ----------------------------------------------------------------------- #
    @property
    def is_passthrough(self) -> bool:
        """True iff no usable model is loaded (the system runs rule-only)."""
        if self._backend == "logreg_json":
            return self._coef is None
        return self._model is None

    def p_win(self, feature_vector: FeatureVector) -> float:
        """Return P(profitable) in [0, 1] for a candidate entry.

        Passthrough -> 1.0. Otherwise flatten with ``to_row`` (FEATURE_ORDER parity),
        re-order to the model's expected layout if needed, apply the stored
        standardization, and predict. Any runtime error degrades to 1.0 (fail-open is
        safe here because ml_threshold gating still applies downstream, and a 1.0 only
        means "rules decide").
        """
        if self.is_passthrough:
            return 1.0
        try:
            row = self._row_for_model(feature_vector)
            standardized = self._standardize(row)
            if self._backend == "logreg_json":
                return self._p_win_logreg(standardized)
            return self._p_win_sklearn(standardized)
        except Exception:
            return 1.0

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #
    def _row_for_model(self, fv: FeatureVector) -> List[float]:
        """Flatten the FeatureVector and align to the model's expected feature order.

        ``to_row`` produces values in the canonical FEATURE_ORDER. If the persisted model
        recorded a different order (it shouldn't, but be defensive), remap by name.
        """
        canonical = to_row(fv)  # in FEATURE_ORDER
        if self._model_features == FEATURE_ORDER:
            return canonical
        by_name = dict(zip(FEATURE_ORDER, canonical))
        return [float(by_name.get(name, 0.0)) for name in self._model_features]

    def _standardize(self, row: List[float]) -> List[float]:
        """Apply (x - mean) / scale per feature when standardization params are present.

        sklearn Pipelines that embed their own StandardScaler should NOT also carry
        separate mean/scale — in that case mean/scale are None and we pass the row through
        unchanged (the pipeline scales internally).
        """
        if self._mean is None or self._scale is None:
            return row
        if len(self._mean) != len(row) or len(self._scale) != len(row):
            return row
        out: List[float] = []
        for x, m, s in zip(row, self._mean, self._scale):
            denom = s if s not in (0.0, None) else 1.0
            out.append((x - m) / denom)
        return out

    def _p_win_sklearn(self, row: List[float]) -> float:
        """predict_proba on a single row; take the P(label==1) column robustly."""
        import numpy as np  # local import keeps module import light

        X = np.asarray(row, dtype=float).reshape(1, -1)
        proba = self._model.predict_proba(X)
        arr = np.asarray(proba)
        classes = getattr(self._model, "classes_", None)
        if classes is not None:
            # Find the column whose class label == 1 (profitable).
            idx_list = [i for i, c in enumerate(list(classes)) if int(c) == 1]
            if idx_list:
                return _clip01(float(arr[0, idx_list[0]]))
            # Only one class was seen in training -> degenerate; fall back to last col.
        return _clip01(float(arr[0, -1]))

    def _p_win_logreg(self, row: List[float]) -> float:
        """Pure-python logistic regression: sigmoid(coef . x + intercept)."""
        if self._coef is None:
            return 1.0
        if len(self._coef) != len(row):
            return 1.0
        z = self._intercept + sum(c * x for c, x in zip(self._coef, row))
        return _clip01(_sigmoid(z))


__all__ = ["EntryScorer"]
