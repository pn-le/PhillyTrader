# VWAP Edge Hunt — Results Summary

**Date:** 2026-06-04 · **Data:** free IEX 1-min bars, 10 symbols, 190 trading days (2025-09-02 .. 2026-06-03)
**Protocol:** chronological TRAIN 60% (→2026-02-12) / VAL 20% (→2026-04-09) / TEST 20% (→2026-06-03). Tune on TRAIN, select on VAL, score TEST **exactly once**. Net of 1bp/side adverse slippage. Research backtester verified **bit-identical** to production engine.

## Verdict: ZERO variants have a real, robust out-of-sample edge.

| Rank | Variant | TEST Sharpe | TEST return | Survives? |
|---|---|---|---|---|
| 1 | momentum_invert | 2.22 | +28.5% | ❌ |
| 2 | wide_param | 2.38 | +15.0% | ❌ |
| 3 | tod_filter | 1.54 | +14.3% | ❌ |
| 4 | trend_filter | 1.84 | +11.6% | ❌ |
| 5 | tf_5min | 1.34 | +6.9% | ❌ |
| 6 | ml_richfeatures | 1.47 | +3.5% | ❌ |
| 7 | symbol_edge | 0.63 | +2.2% | ❌ |
| 8 | short_side | 0.42 | +1.7% | ❌ |
| 9 | exits_tp_trail | 0.03 | +0.3% | ❌ |

(Per-variant detail in `experiments/results/<name>.json`.)

## Why every positive TEST number is fake edge — four recurring red flags
1. **Inverted train→test sign-flip.** 7 of 9 selected configs *lose money* on the 60% TRAIN bulk and only "win" on the latest 38-day TEST slice — a regime artifact, not edge.
2. **TEST positivity is near-universal.** 70–100% of *all* tried configs profit on the TEST window; corr(VAL, TEST) ≈ 0.05–0.34. Validation selection carries no information into TEST.
3. **Beta confound.** SPY +7%/flat/+10.7% across splits; universe +24.5% on TEST. The **market-neutral (side=both) twin** of the best time-of-day config is negative on all three splits — proving the gains were directional beta, not signal.
4. **Below the cost/noise floor.** exits_tp_trail TEST t-stat = 0.029; short_side earns 1.7bp/trade against a 2bp cost with 76% of PnL from one symbol on 5 trades.

ML gate OOS AUC ≈ 0.48–0.53 (coin flip) and underperforms doing nothing.

## One pre-registered hint worth future testing
The momentum_invert event study found a **reversion bounce after DOWN moves** (VAL K=15: +3.16bp, t=4.19) — the *opposite* of breakout momentum. Treat as a hypothesis to test on neutral SIP data, not an edge to trade.

## Recommended next directions (in order)
1. **SIP data, not IEX.** IEX is ~2–3% of true volume; VWAP/volume signals are distorted and PnL is pinned to the noise floor. Re-run the identical protocol on consolidated SIP before concluding the family is dead.
2. **Longer, multi-regime history** so each split spans bull/chop/drawdown (the core defect: TEST was one homogeneous rally).
3. **Beta-neutral by construction** (dollar-neutral long/short or SPY-hedge) so any surviving signal can't be beta in disguise.
4. **Change structure, not parameters** — test the down-move reversion bounce on neutral SIP data.

**Do not paper-trade any of these variants.**
