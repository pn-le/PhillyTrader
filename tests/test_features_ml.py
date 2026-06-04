"""test_features_ml.py — ML features, EntryScorer, and dataset build/persist.

No network, no orders. Covers:
  - FEATURE_ORDER stability + to_row ordering parity (changing layout is a breaking change).
  - build_features: pure, decision-time only, raises on insufficient data, computes values.
  - EntryScorer.load passthrough -> p_win 1.0 when no model file; is_passthrough True.
  - build_dataset shapes (rows in FEATURE_ORDER; labels from pnl>0); save/load round-trip.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from agentic_trader.config import (
    RECENT_RETURN_K,
    SESSION_MINUTES,
    UNIVERSE,
    StrategyParams,
)
from agentic_trader.data.market_data import compute_indicators
from agentic_trader.ml.dataset import build_dataset, load_dataset, save_dataset
from agentic_trader.ml.features import (
    FEATURE_ORDER,
    build_features,
    feature_names,
    to_row,
)
from agentic_trader.ml.scorer import EntryScorer
from agentic_trader.types import FeatureVector, PositionState, TradeRecord

from . import flat_then_spike, make_snapshot, ny

PARAMS = StrategyParams()


# --------------------------------------------------------------------------- #
# FEATURE_ORDER / to_row parity
# --------------------------------------------------------------------------- #
def test_feature_order_is_canonical_and_stable():
    """FEATURE_ORDER is the exact 8-column layout; feature_names() is a copy of it."""
    assert FEATURE_ORDER == [
        "dist_from_vwap",
        "volume_ratio",
        "log_rolling_vol",
        "minute_of_session",
        "recent_return",
        "bar_range_pct",
        "session_progress",
        "symbol_id",
    ]
    names = feature_names()
    assert names == FEATURE_ORDER
    assert names is not FEATURE_ORDER  # a copy, not the same list object


def test_to_row_matches_feature_order():
    """to_row emits values positionally in FEATURE_ORDER (numeric only, no symbol)."""
    fv = FeatureVector(
        symbol="AAPL", dist_from_vwap=-0.01, volume_ratio=1.5, log_rolling_vol=7.0,
        minute_of_session=30, recent_return=0.002, bar_range_pct=0.004,
        session_progress=0.077, symbol_id=3,
    )
    row = to_row(fv)
    assert len(row) == len(FEATURE_ORDER)
    expected = {
        "dist_from_vwap": -0.01, "volume_ratio": 1.5, "log_rolling_vol": 7.0,
        "minute_of_session": 30.0, "recent_return": 0.002, "bar_range_pct": 0.004,
        "session_progress": 0.077, "symbol_id": 3.0,
    }
    for i, name in enumerate(FEATURE_ORDER):
        assert row[i] == pytest.approx(expected[name])
    assert all(isinstance(x, float) for x in row)


# --------------------------------------------------------------------------- #
# build_features
# --------------------------------------------------------------------------- #
def test_build_features_values_and_purity():
    """build_features computes the documented decision-time fields and is deterministic."""
    bars = flat_then_spike("MSFT", flat_close=100.0, decision_close=99.0,
                           flat_vol=1000.0, decision_vol=2000.0)
    ind = compute_indicators(bars, PARAMS)
    snap = make_snapshot("MSFT", bars, ind, position_state=PositionState(symbol="MSFT"))

    fv = build_features(snap, PARAMS)
    # symbol_id == index in UNIVERSE (A8).
    assert fv.symbol_id == UNIVERSE.index("MSFT")
    # recent_return = close[t]/close[t-K] - 1 ; here close[t]=99, close[t-5]=100.
    assert fv.recent_return == pytest.approx(99.0 / 100.0 - 1.0)
    # dist_from_vwap mirrors the indicator (negative, below VWAP).
    assert fv.dist_from_vwap == pytest.approx(ind.dist_from_vwap)
    assert fv.volume_ratio == pytest.approx(ind.volume_ratio)
    # minute_of_session: decision bar is the 21st minute from 09:30 -> index 20.
    assert fv.minute_of_session == 20
    assert fv.session_progress == pytest.approx(20.0 / (SESSION_MINUTES - 1))

    # Pure / deterministic: same snapshot -> identical feature vector.
    assert build_features(snap, PARAMS) == fv


def test_build_features_raises_on_insufficient_bars():
    """Fewer than RECENT_RETURN_K+1 bars => ValueError (defensive guard)."""
    bars = flat_then_spike("SPY", n_flat=RECENT_RETURN_K - 1)  # K bars total < K+1
    ind = compute_indicators(bars, PARAMS)
    snap = make_snapshot("SPY", bars, ind, position_state=PositionState(symbol="SPY"))
    with pytest.raises(ValueError):
        build_features(snap, PARAMS)


def test_build_features_no_future_data():
    """Appending a FUTURE bar after the decision bar must NOT change features.

    build_features uses bars[n-1] as the decision bar and bars[n-1-K] for recent_return,
    so the feature vector is fully determined by the window passed in. We verify that the
    same decision-bar window produces the same vector whether or not we (incorrectly) had
    appended a later bar to a SEPARATE snapshot — i.e. no look-ahead leakage by construction.
    """
    bars = flat_then_spike("NVDA", flat_close=100.0, decision_close=99.0,
                           flat_vol=1000.0, decision_vol=2000.0)
    ind = compute_indicators(bars, PARAMS)
    snap = make_snapshot("NVDA", bars, ind, position_state=PositionState(symbol="NVDA"))
    fv_before = build_features(snap, PARAMS)

    # A would-be FUTURE spike bar at the NEXT minute — must be irrelevant to the decision
    # made on the prior window. Recomputing features on the ORIGINAL window is unchanged.
    import datetime as _dtmod
    from . import make_bar
    future_bar = make_bar("NVDA", bars[-1].start + _dtmod.timedelta(minutes=1),
                          open_=500.0, high=600.0, low=500.0, close=500.0, volume=99999.0)
    future = bars + [future_bar]
    assert build_features(snap, PARAMS) == fv_before  # original decision is stable

    # The extended window's decision bar is a genuinely LATER minute, proving the two
    # decisions are never conflated (the future bar does not bleed into the earlier one).
    ind2 = compute_indicators(future, PARAMS)
    snap2 = make_snapshot("NVDA", future, ind2, position_state=PositionState(symbol="NVDA"))
    fv2 = build_features(snap2, PARAMS)
    assert fv2.minute_of_session == fv_before.minute_of_session + 1


# --------------------------------------------------------------------------- #
# EntryScorer passthrough
# --------------------------------------------------------------------------- #
def test_scorer_passthrough_when_no_model(tmp_path):
    """No model file => passthrough scorer, p_win == 1.0 for any features."""
    scorer = EntryScorer.load(tmp_path / "does_not_exist.pkl")
    assert scorer.is_passthrough is True
    fv = FeatureVector("AAPL", -0.01, 1.5, 7.0, 30, 0.002, 0.004, 0.077, 3)
    assert scorer.p_win(fv) == 1.0


def test_scorer_default_construct_is_passthrough():
    """A directly-constructed EntryScorer(model=None) is passthrough."""
    scorer = EntryScorer(model=None)
    assert scorer.is_passthrough
    assert scorer.p_win(FeatureVector("SPY", -0.01, 1.5, 7.0, 30, 0.0, 0.0, 0.0, 0)) == 1.0


def test_scorer_logreg_json_roundtrip(tmp_path):
    """A logreg_json artifact loads as a real (non-passthrough) scorer producing [0,1]."""
    import json

    n = len(FEATURE_ORDER)
    artifact = {
        "backend": "logreg_json",
        "coef": [0.0] * n,          # all-zero weights => sigmoid(intercept)
        "intercept": 0.0,           # sigmoid(0) == 0.5
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "feature_names": list(FEATURE_ORDER),
    }
    path = tmp_path / "ml_scorer.pkl"
    path.write_text(json.dumps(artifact))
    scorer = EntryScorer.load(path)
    assert scorer.is_passthrough is False
    p = scorer.p_win(FeatureVector("SPY", -0.01, 1.5, 7.0, 30, 0.0, 0.0, 0.0, 0))
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Dataset build / persist / load
# --------------------------------------------------------------------------- #
def _trade_record(symbol: str, pnl: float, *, dist: float = -0.01) -> TradeRecord:
    fv = FeatureVector(symbol, dist, 1.5, 7.0, 30, 0.002, 0.004, 0.077,
                       UNIVERSE.index(symbol))
    feats = {name: getattr(fv, name) for name in FEATURE_ORDER}
    feats["symbol"] = symbol
    now = ny(hour=10)
    return TradeRecord(
        symbol=symbol, entry_time=now, exit_time=now + _dt.timedelta(minutes=5),
        entry_price=100.0, exit_price=100.0 + pnl, qty=1.0, pnl=pnl,
        return_pct=pnl / 100.0, holding_min=5.0, exit_reason="vwap_revert",
        features=feats, label=1 if pnl > 0 else 0,
    )


def test_build_dataset_shapes_and_labels():
    """X has len(FEATURE_ORDER) columns; y is 1 iff pnl>0; rows == usable records."""
    records = [
        _trade_record("SPY", pnl=1.0),    # label 1
        _trade_record("AAPL", pnl=-0.5),  # label 0
        _trade_record("NVDA", pnl=0.0),   # label 0 (pnl not > 0)
    ]
    X, y, names = build_dataset(records)
    assert names == feature_names()
    assert X.shape == (3, len(FEATURE_ORDER))
    assert y.tolist() == [1, 0, 0]
    # Column 0 is dist_from_vwap == -0.01 for every row.
    assert np.allclose(X[:, FEATURE_ORDER.index("dist_from_vwap")], -0.01)


def test_build_dataset_skips_featureless_records():
    """Records with empty feature dicts carry no signal and are skipped."""
    good = _trade_record("SPY", pnl=1.0)
    empty = TradeRecord("AAPL", ny(), ny(), 100.0, 101.0, 1.0, 1.0, 0.01, 5.0,
                        "vwap_revert", features={}, label=1)
    X, y, _ = build_dataset([good, empty])
    assert X.shape == (1, len(FEATURE_ORDER))
    assert y.shape == (1,)


def test_build_dataset_empty_input():
    """No records => zero-row arrays with the right width (never raises)."""
    X, y, names = build_dataset([])
    assert X.shape == (0, len(FEATURE_ORDER))
    assert y.shape == (0,)
    assert names == feature_names()


def test_dataset_save_load_roundtrip(tmp_path):
    """save_dataset then load_dataset preserves X/y and FEATURE_ORDER columns."""
    records = [_trade_record("SPY", 1.0), _trade_record("AAPL", -1.0)]
    X, y, names = build_dataset(records)
    out = save_dataset(X, y, names, tmp_path / "dataset")
    assert out.exists()

    X2, y2, names2 = load_dataset(out)
    assert names2 == names
    assert np.allclose(X2, X)
    assert y2.tolist() == y.tolist()


# --------------------------------------------------------------------------- #
# Finding #11 (HIGH): the trainer drops symbol_id (a categorical) before fitting,
# so it never z-scores an arbitrary ordinal index. The persisted model omits it and
# the live scorer remaps by name -> p_win is INVARIANT to symbol_id at serve time.
# --------------------------------------------------------------------------- #
def test_trained_model_excludes_symbol_id(tmp_path):
    """train() drops symbol_id from the fitted features and the live scorer is invariant
    to it (changing only symbol_id never changes p_win)."""
    from agentic_trader.ml.train_model import train

    rng = np.random.default_rng(0)
    n = 60
    Xall = rng.normal(size=(n, len(FEATURE_ORDER)))
    Xall[:, FEATURE_ORDER.index("symbol_id")] = rng.integers(0, 10, size=n)  # categorical
    # Label correlates with dist_from_vwap so the model learns a real (non-symbol) signal.
    y = (Xall[:, FEATURE_ORDER.index("dist_from_vwap")] < 0).astype(int)

    ds = save_dataset(Xall, y, feature_names(), tmp_path / "dataset.csv")
    metrics = train(ds, out_dir=tmp_path)
    assert metrics["trained"] is True
    # symbol_id must NOT be among the model's fitted features.
    assert "symbol_id" not in metrics["feature_names"]
    assert metrics["n_features"] == len(FEATURE_ORDER) - 1

    scorer = EntryScorer.load(tmp_path / "ml_scorer.pkl")
    assert scorer.is_passthrough is False
    base = FeatureVector("SPY", -0.02, 1.5, 7.0, 30, 0.002, 0.004, 0.077, 0)
    other = FeatureVector("AMZN", -0.02, 1.5, 7.0, 30, 0.002, 0.004, 0.077, 9)
    # Only symbol_id differs -> p_win must be identical (no spurious ordinal effect).
    assert scorer.p_win(base) == pytest.approx(scorer.p_win(other))
