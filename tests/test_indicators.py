"""test_indicators.py — STRATEGY_RULES §2 indicator correctness (PURE, no network).

Covers `agentic_trader.data.market_data.compute_indicators`:
  - session VWAP on a hand-computed example (typical_price = (h+l+c)/3, vol-weighted).
  - rolling20_avg_vol EXCLUDES the current/decision bar (bars[n-21:n-1]).
  - volume_ratio = current_volume / rolling20_avg_vol.
  - dist_from_vwap sign (negative when last price is below VWAP).
  - valid/invalid gating (n < 21, degenerate volume).
"""

from __future__ import annotations

import math

import pytest

from agentic_trader.config import MIN_BARS_FOR_ENTRY, ROLLING_VOL_WINDOW, StrategyParams
from agentic_trader.data.market_data import compute_indicators

from . import bars_session, flat_then_spike, make_bar, ny

PARAMS = StrategyParams()


def test_session_vwap_hand_computed():
    """VWAP = sum(typical*vol)/sum(vol) over ALL bars incl. the decision bar.

    Three bars with distinct high/low/close so typical_price != close; verifies the
    (h+l+c)/3 typical price is used (not the close) and that it is volume-weighted.
    """
    bars = [
        make_bar("SPY", ny(minute=30), open_=100, high=102, low=98, close=100, volume=100),
        make_bar("SPY", ny(minute=31), open_=100, high=110, low=100, close=105, volume=300),
        make_bar("SPY", ny(minute=32), open_=105, high=106, low=102, close=104, volume=200),
    ]
    # typicals: (102+98+100)/3=100 ; (110+100+105)/3=105 ; (106+102+104)/3=104
    num = 100 * 100 + 105 * 300 + 104 * 200
    den = 100 + 300 + 200
    expected_vwap = num / den

    ind = compute_indicators(bars, PARAMS)
    assert ind.session_vwap == pytest.approx(expected_vwap)
    assert ind.last_price == 104.0  # close of the last (decision) bar
    assert ind.n_bars == 3


def test_dist_from_vwap_sign_negative_below():
    """last_price below VWAP => dist_from_vwap is negative; above => positive."""
    # All flat at 100 except a final dip to 99 -> last price below the ~100 VWAP.
    below = bars_session("SPY", [100, 100, 100, 99], [10, 10, 10, 10])
    ind_below = compute_indicators(below, PARAMS)
    assert ind_below.dist_from_vwap is not None
    assert ind_below.dist_from_vwap < 0

    above = bars_session("SPY", [100, 100, 100, 101], [10, 10, 10, 10])
    ind_above = compute_indicators(above, PARAMS)
    assert ind_above.dist_from_vwap > 0

    # Exact magnitude: last=99, vwap = (100*30 + 99)/40 close-only (flat bars).
    vwap = (100 * 3 + 99) / 4
    assert ind_below.dist_from_vwap == pytest.approx((99 - vwap) / vwap)


def test_rolling20_excludes_current_bar():
    """rolling20_avg_vol must be the mean of the 20 bars STRICTLY before the decision bar.

    21 bars: first 20 have volume 1000 (the rolling window), the decision bar (index 20)
    has volume 9999 which MUST NOT be averaged in. If the current bar leaked into the
    window, the average would be != 1000.
    """
    closes = [100.0] * 20 + [99.0]
    vols = [1000.0] * 20 + [9999.0]
    bars = bars_session("SPY", closes, vols)
    assert len(bars) == MIN_BARS_FOR_ENTRY

    ind = compute_indicators(bars, PARAMS)
    assert ind.rolling20_avg_vol == pytest.approx(1000.0)  # decision bar excluded
    assert ind.current_volume == 9999.0
    assert ind.volume_ratio == pytest.approx(9999.0 / 1000.0)


def test_rolling20_window_is_exactly_20_bars():
    """With > 21 bars the rolling window is still exactly 20 (the 20 before decision)."""
    # 25 bars: vary the early volumes so only the last-20-before-decision matter.
    vols = [50.0] * 4 + [1000.0] * 20 + [7000.0]  # 4 + 20 + 1 = 25 bars
    closes = [100.0] * 24 + [99.0]
    bars = bars_session("SPY", closes, vols)
    assert len(bars) == 25

    ind = compute_indicators(bars, PARAMS)
    # The 20 bars before the decision bar (indices 4..23) all have volume 1000.
    assert ind.rolling20_avg_vol == pytest.approx(1000.0)
    assert ind.n_bars == 25


def test_volume_ratio_value():
    """volume_ratio = current_volume / rolling20_avg_vol with a non-uniform window."""
    # 20 window bars alternating 1000/2000 => mean 1500; decision volume 3000 => ratio 2.0.
    win_vols = [1000.0, 2000.0] * 10  # length 20, mean 1500
    vols = win_vols + [3000.0]
    closes = [100.0] * 20 + [99.0]
    bars = bars_session("SPY", closes, vols)

    ind = compute_indicators(bars, PARAMS)
    assert ind.rolling20_avg_vol == pytest.approx(1500.0)
    assert ind.volume_ratio == pytest.approx(3000.0 / 1500.0)


def test_valid_requires_min_bars():
    """valid is False with fewer than MIN_BARS_FOR_ENTRY bars; True at exactly 21."""
    short = flat_then_spike("SPY", n_flat=ROLLING_VOL_WINDOW - 1)  # 20 bars total
    ind_short = compute_indicators(short, PARAMS)
    assert ind_short.n_bars == 20
    assert ind_short.valid is False
    assert ind_short.invalid_reason is not None
    assert "insufficient" in ind_short.invalid_reason

    exact = flat_then_spike("SPY", n_flat=ROLLING_VOL_WINDOW)  # 21 bars total
    ind_exact = compute_indicators(exact, PARAMS)
    assert ind_exact.n_bars == MIN_BARS_FOR_ENTRY
    assert ind_exact.valid is True
    assert ind_exact.invalid_reason is None


def test_empty_bars_is_invalid_not_crash():
    """compute_indicators must NEVER raise — empty input returns valid=False."""
    ind = compute_indicators([], PARAMS)
    assert ind.valid is False
    assert ind.n_bars == 0
    assert ind.session_vwap is None
    assert ind.invalid_reason == "no_bars"


def test_degenerate_zero_volume_invalid():
    """All-zero rolling volume makes the symbol ineligible (no divide-by-zero)."""
    closes = [100.0] * 20 + [99.0]
    vols = [0.0] * 20 + [500.0]
    bars = bars_session("SPY", closes, vols)
    ind = compute_indicators(bars, PARAMS)
    # VWAP still defined from the decision bar's volume, but rolling window is degenerate.
    assert ind.valid is False
    assert ind.invalid_reason == "degenerate_rolling_volume"
    assert ind.volume_ratio is None


def test_pure_no_mutation_of_input_bars():
    """compute_indicators must not mutate the bars list it is given (determinism)."""
    bars = flat_then_spike("SPY")
    snapshot = list(bars)
    compute_indicators(bars, PARAMS)
    assert bars == snapshot
    # And calling twice yields identical indicators (deterministic).
    a = compute_indicators(bars, PARAMS)
    b = compute_indicators(bars, PARAMS)
    assert a == b


def test_log_rolling_vol_relationship_sanity():
    """Sanity: rolling avg used for volume_ratio is consistent with log1p feature input."""
    bars = flat_then_spike("SPY", flat_vol=1000.0, decision_vol=2000.0)
    ind = compute_indicators(bars, PARAMS)
    assert ind.rolling20_avg_vol == pytest.approx(1000.0)
    # log1p of the rolling vol (what features.build_features will consume) is finite.
    assert math.isfinite(math.log1p(ind.rolling20_avg_vol))
