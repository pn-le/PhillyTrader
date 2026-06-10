"""test_backtest.py — t->t+1 simulator determinism + NO look-ahead.

No network. Drives `agentic_trader.backtest.engine.run_backtest` on synthetic per-symbol
bar series and asserts the two load-bearing invariants:
  - DETERMINISM: same input -> identical trades + metrics.
  - NO LOOK-AHEAD: a price spike that occurs AFTER a decision bar must not change any
    decision made before it (fills happen at t+1 open; the future is unseen at t).

The backtest engine is built by a parallel implementer; if it is not present yet this
module skips cleanly (so the rest of the suite stays green) and runs fully once it lands.
"""

from __future__ import annotations

import copy

import pytest

from agentic_trader.config import RiskLimits, StrategyParams

from . import bars_session, ny

# The engine may not be implemented yet when this test module is collected. Skip the whole
# module (not error) until agentic_trader.backtest.engine exists.
engine = pytest.importorskip(
    "agentic_trader.backtest.engine",
    reason="backtest engine not implemented yet (parallel build)",
)
run_backtest = engine.run_backtest
BacktestResult = engine.BacktestResult

LIMITS = RiskLimits()
PARAMS = StrategyParams()


def _entry_then_revert_series(symbol: str):
    """A session that should trigger ONE entry then a VWAP-revert exit.

    20 flat warm-up bars (price 100, vol 1000) establish VWAP ~100 and the rolling window;
    then a dip to 98.5 with a 2x volume spike (entry candidate, ~1.5% below VWAP), then a
    +2 minute bar for the t+1 fill, then a recovery back to ~100 (vwap_revert exit), plus
    trailing bars so the exit can fill at t+1.
    """
    closes = [100.0] * 20 + [98.5, 98.6, 100.0, 100.1, 100.1, 100.1]
    vols = [1000.0] * 20 + [2500.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0]
    return bars_session(symbol, closes, vols, start=ny(hour=10, minute=0))


def test_backtest_is_deterministic():
    """Identical inputs -> identical trades and metrics across two independent runs."""
    bars_by_symbol = {"SPY": _entry_then_revert_series("SPY")}
    r1 = run_backtest(copy.deepcopy(bars_by_symbol), PARAMS, LIMITS)
    r2 = run_backtest(copy.deepcopy(bars_by_symbol), PARAMS, LIMITS)

    def _trade_key(t):
        return (t.symbol, t.entry_time, t.exit_time, round(t.entry_price, 6),
                round(t.exit_price, 6), round(t.qty, 6), round(t.pnl, 6), t.exit_reason)

    assert [_trade_key(t) for t in r1.trades] == [_trade_key(t) for t in r2.trades]
    assert r1.metrics == r2.metrics


def test_backtest_no_lookahead_future_spike_irrelevant():
    """A spike AFTER the decision window must not change earlier trades/decisions.

    Two scenarios share an identical prefix up to and including every decision the strategy
    could make on the common bars; they differ ONLY in a far-future tail bar. The trades
    decided on the shared prefix must be byte-identical — proving the engine never peeks
    ahead (decision at t uses only bars[0..t]; fills at t+1 open).
    """
    base = _entry_then_revert_series("AAPL")
    # Scenario A: append a calm tail bar.  Scenario B: append a massive future spike.
    calm = base + [base[-1]]
    spike_bar_close = base[-1].close * 5.0
    from . import make_bar
    import datetime as _dt
    spike = base + [make_bar("AAPL", base[-1].start + _dt.timedelta(minutes=1),
                             open_=spike_bar_close, high=spike_bar_close * 1.1,
                             low=spike_bar_close, close=spike_bar_close, volume=99999.0)]

    ra = run_backtest({"AAPL": calm}, PARAMS, LIMITS)
    rb = run_backtest({"AAPL": spike}, PARAMS, LIMITS)

    # The trade that closed on the shared prefix (the revert exit) must be identical: the
    # future tail bar cannot retroactively alter a prior entry/exit price or timing.
    def _closed_before_tail(result, cutoff_start):
        return [t for t in result.trades if t.exit_time < cutoff_start]

    cutoff = base[-1].start  # anything decided strictly before the appended tail bar
    a_closed = _closed_before_tail(ra, cutoff)
    b_closed = _closed_before_tail(rb, cutoff)

    def _key(t):
        return (t.symbol, t.entry_time, t.exit_time, round(t.entry_price, 6),
                round(t.exit_price, 6), round(t.pnl, 6), t.exit_reason)

    assert [_key(t) for t in a_closed] == [_key(t) for t in b_closed]


def test_backtest_result_shape():
    """run_backtest returns a BacktestResult with trades/metrics/labeled_features."""
    result = run_backtest({"SPY": _entry_then_revert_series("SPY")}, PARAMS, LIMITS)
    assert isinstance(result, BacktestResult)
    assert isinstance(result.trades, list)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.labeled_features, list)
    # Every entry taken should have a label (0/1) captured for ML.
    for lf in result.labeled_features:
        assert "label" in lf
        assert lf["label"] in (0, 1)
    # One labeled-feature row per ENTRY round-trip recorded.
    assert len(result.labeled_features) == len(result.trades)


def test_backtest_records_profitable_label_consistency():
    """A recorded TradeRecord's label must equal (pnl > 0)."""
    result = run_backtest({"SPY": _entry_then_revert_series("SPY")}, PARAMS, LIMITS)
    for t in result.trades:
        assert t.label == (1 if t.pnl > 0 else 0)


def test_backtest_no_trades_on_calm_session():
    """A perfectly flat session never triggers the entry predicate (no false entries)."""
    flat = bars_session("QQQ", [100.0] * 30, [1000.0] * 30, start=ny(hour=10))
    result = run_backtest({"QQQ": flat}, PARAMS, LIMITS)
    assert result.trades == []


# --------------------------------------------------------------------------- #
# Finding #1 (CRITICAL): EOD-flatten exit must NOT fill at the decision bar's own
# close (same-bar look-ahead). With no t+1 bar, the EOD flatten fills at the
# decision bar's OPEN, which is knowable at the start of bar t.
# --------------------------------------------------------------------------- #
def _eod_flatten_with_spike_close(symbol: str):
    """A session that enters mid-afternoon and ends on a 15:55 EOD bar with a huge
    close spike but a CALM open. 23 bars from 15:33..15:55 NY:
      - 20 flat warm-up bars (price 100, vol 1000) -> VWAP ~100, rolling window ready.
      - bar 20 (15:53): dip to 98.5 + 2.5x volume -> entry candidate (~1.5% below VWAP).
      - bar 21 (15:54): the t+1 fill bar (open 98.6) -> position opens here.
      - bar 22 (15:55): EOD window. open == 98.7 (calm), close == 500 (fabricated spike).
    The exit decided at 15:55 has NO t+1 bar; it must fill at 98.7 (the OPEN), never 500.
    """
    start = ny(hour=15, minute=33)
    closes = [100.0] * 20 + [98.5, 98.6, 500.0]
    vols = [1000.0] * 20 + [2500.0, 1200.0, 1200.0]
    # opens default to close, but we need the 15:55 OPEN to be calm (98.7) while its CLOSE
    # is the 500 spike, so highs/lows/opens must be explicit on that last bar.
    opens = closes[:-1] + [98.7]
    highs = closes[:-1] + [500.0]
    lows = closes[:-1] + [98.7]
    return bars_session(symbol, closes, vols, start=start, opens=opens, highs=highs, lows=lows)


def test_backtest_eod_flatten_does_not_fill_at_decision_bar_close():
    """An EOD-flatten exit with no t+1 must NOT fill at the decision bar's CLOSE.

    Regression for the same-bar look-ahead: the 15:55 decision bar has a 500 close spike
    but a calm 98.7 open. The exit fill price must be the bar's OPEN (with adverse SELL
    slippage), never anywhere near its 500 close — a price unknowable at decision time.
    """
    bars = _eod_flatten_with_spike_close("SPY")
    result = run_backtest({"SPY": bars}, PARAMS, LIMITS)
    assert len(result.trades) == 1
    t = result.trades[0]
    decision_close = bars[-1].close           # 500.0 (the fabricated spike)
    decision_open = bars[-1].open             # 98.7 (the realizable same-bar reference)
    # The exit MUST be at/near the OPEN, never the close.
    assert t.exit_price < decision_close * 0.5
    assert t.exit_price == pytest.approx(decision_open * (1.0 - 1e-4), rel=1e-6)
    # And the fabricated spike must NOT manufacture a giant winner.
    assert t.pnl < 10.0


# --------------------------------------------------------------------------- #
# Finding #5 (HIGH): a position open at the session's final bar (no t+1, not yet
# EOD) must be force-closed and recorded, never silently dropped.
# --------------------------------------------------------------------------- #
def test_backtest_force_closes_position_at_timeline_end():
    """An entry that fills then runs out of session (no exit predicate, no t+1, pre-EOD)
    must be force-closed at the last decision bar's OPEN and recorded as a trade+label,
    not silently abandoned (which would bias metrics and the ML training set)."""
    # 22 bars from 10:00: 20 flat warm-up, a dip+vol entry candidate at idx 20, one t+1
    # fill bar at idx 21 — then the session ENDS (no further bars, well before 15:55).
    start = ny(hour=10, minute=0)
    closes = [100.0] * 20 + [98.5, 98.6]
    vols = [1000.0] * 20 + [2500.0, 1200.0]
    bars = bars_session("AAPL", closes, vols, start=start)
    result = run_backtest({"AAPL": bars}, PARAMS, LIMITS)
    # The entry fired+filled; it must NOT vanish — exactly one recorded round-trip.
    assert len(result.trades) == 1
    assert len(result.labeled_features) == len(result.trades)
    assert result.trades[0].exit_reason == "forced_close"


# --------------------------------------------------------------------------- #
# Finding #2 / #9 (HIGH/CRITICAL): labeled rows + trades must be ordered by
# DECISION/ENTRY time, not EXIT time, so the temporal ML split stays leakage-free.
# --------------------------------------------------------------------------- #
def test_backtest_rows_ordered_by_entry_not_exit_time():
    """Two symbols: one enters EARLY but holds long (exits late), another enters LATER
    but exits quickly. Exit-time order would invert them; the emitted trades MUST be in
    entry/decision-time order so train_model's positional temporal split is honored."""

    # SPY enters early (~10:21) and holds to max_hold (exits late).
    spy_closes = [100.0] * 20 + [98.5] + [98.5] * 30
    spy_vols = [1000.0] * 20 + [2500.0] + [1000.0] * 30
    spy = bars_session("SPY", spy_closes, spy_vols, start=ny(hour=10, minute=0))

    # AAPL enters LATER (~10:23) but vwap-reverts almost immediately (exits early).
    aapl_closes = [100.0] * 22 + [98.5, 100.0, 100.0, 100.0]
    aapl_vols = [1000.0] * 22 + [2500.0, 1200.0, 1200.0, 1200.0]
    aapl = bars_session("AAPL", aapl_closes, aapl_vols, start=ny(hour=10, minute=0))

    result = run_backtest({"SPY": spy, "AAPL": aapl}, PARAMS, LIMITS)
    # Need both trades present to test ordering.
    assert len(result.trades) >= 2
    entry_times = [t.entry_time for t in result.trades]
    # Entry/decision times must be non-decreasing (earliest decision first).
    assert entry_times == sorted(entry_times)
    # The SPY trade (earlier entry) must precede the AAPL trade even though it exits LATER.
    spy_idx = next(i for i, t in enumerate(result.trades) if t.symbol == "SPY")
    aapl_idx = next(i for i, t in enumerate(result.trades) if t.symbol == "AAPL")
    assert result.trades[spy_idx].entry_time <= result.trades[aapl_idx].entry_time
    assert spy_idx < aapl_idx
    # Cross-check: SPY actually exits AFTER AAPL, so exit-time order WOULD have inverted them.
    assert result.trades[spy_idx].exit_time > result.trades[aapl_idx].exit_time


# --------------------------------------------------------------------------- #
# Finding #10 (HIGH): the ML gate, when active, SHRINKS the labeled candidate set
# (selection bias). This is why dataset generation must run GATE-FREE (scorer=None +
# ml_threshold=0.0) — otherwise each retrain only ever sees trades the prior model
# already approved. We prove the gate biases the set, motivating the gate-free path.
# --------------------------------------------------------------------------- #
class _BlockingScorer:
    """A scorer that vetoes every candidate (p_win == 0.0)."""
    is_passthrough = False

    def p_win(self, _features):
        return 0.0


def test_active_ml_gate_biases_candidate_set_so_dataset_must_be_gate_free():
    """With a non-zero ml_threshold, a blocking scorer suppresses entries that the gate-free
    run takes — so labels generated through the gate are a biased subset. Dataset generation
    must therefore disable the gate (cmd_dataset forces scorer=None + ml_threshold=0.0)."""
    from dataclasses import replace

    bars = {"SPY": _entry_then_revert_series("SPY")}

    # Gate-free (what cmd_dataset now does): label every rule-passing candidate.
    gate_free_params = replace(PARAMS, ml_threshold=0.0)
    gate_free = run_backtest({k: list(v) for k, v in bars.items()}, gate_free_params, LIMITS, scorer=None)

    # Gated (the BUGGY dataset path): a blocking model + threshold drops candidates -> fewer labels.
    gated_params = replace(PARAMS, ml_threshold=0.5)
    gated = run_backtest({k: list(v) for k, v in bars.items()}, gated_params, LIMITS, scorer=_BlockingScorer())

    assert len(gate_free.labeled_features) >= 1
    assert len(gated.labeled_features) < len(gate_free.labeled_features)
