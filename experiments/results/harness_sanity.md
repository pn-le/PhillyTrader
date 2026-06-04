# Harness Sanity Check — Research Backtester vs Production Engine

**Date:** 2026-06-04
**Purpose:** Validate that `experiments/harness.py:research_backtest` reproduces
`agentic_trader/backtest/engine.py:run_backtest` on the plain long-only base strategy, so
every downstream out-of-sample experiment is built on a backtester that is known-correct.

## Method

Both engines were fed **identical** `Bar` data (the production engine is data-source
agnostic — it consumes `Dict[str, List[Bar]]`), so this isolates the *simulator logic*,
which is what we are validating. Bars come from the experiments IEX cache.

- **Window:** 2025-09-02 .. 2025-09-22 (≈3 weeks, all 10 universe symbols). Bounded so the
  pure-python production engine runs quickly; large enough for a few hundred round-trips.
- **Base params (long-only):** `entry_dist=0.005, vol_mult=1.2, max_hold=15,
  vwap_exit_band=0.001, stop_loss=0.005`, `notional=100`, default risk caps
  (`max_open_positions=4, max_total_exposure=$500, per_symbol_cooldown_min=10`),
  `slippage=1bp` adverse per side, `ml_threshold=0.0` (passthrough — no ML gate).
- Both: strict t→t+1 fills (decide on completed bar t, fill at t+1 open), EOD flatten at
  15:55 NY, no overnight.

## Result — EXACT match (not just "same ballpark")

| Metric | Research backtester | Production engine |
|---|---|---|
| n_trades | 322 | 322 |
| total_pnl (net) | **-4.8430659351** | **-4.8430659351** |
| win_rate | 0.500 | 0.500 |
| exit max_hold | 259 | 259 |
| exit stop_loss | 28 | 28 |
| exit eod_flatten | 19 | 19 |
| exit vwap_revert | 16 | 16 |

**Absolute PnL difference: 0.00e+00** (bit-identical at full float precision).
Trade-count, win-rate, and the full exit-reason histogram all agree exactly.

## What this validates

The exact agreement confirms the research backtester correctly mirrors production on:
- Session VWAP (typical price (H+L+C)/3, volume-weighted, cumulative, **resets daily**).
- `dist_from_vwap = (close - vwap) / vwap`.
- Rolling-20 average volume **excluding the current bar** (`bars[i-20:i]`), and
  `volume_ratio = current_volume / rolling20_avg_vol`.
- Entry predicate: `dist <= -entry_dist` AND `volume_ratio >= vol_mult`, gated on n≥21.
- Exit priority: `eod_flatten > vwap_revert > max_hold > stop_loss` (with the research
  extensions `take_profit`/`trailing_stop` slotting in above `vwap_revert` when enabled).
- t→t+1 fill discipline, 1bp adverse slippage per side, and the risk caps (≤4 concurrent
  positions, ≤$500 exposure, 10-min per-symbol cooldown, $100 notional).

No material divergence found — **no fix required.**

## Notes / caveats

- The base strategy is slightly **negative net of costs** on this thin IEX slice
  (-$4.84 over 322 trades; -$0.015/trade), consistent with the task's warning that IEX
  free data is thin and tiny PnL magnitudes should be treated skeptically. This is the
  expected "no free edge at baseline" starting point for the edge hunt, not a bug.
- Research-only spec extensions (short / both side, `take_profit`, `trailing_stop`,
  `time_window`, `trend_filter`, `score_fn` gate) were exercised on the same window and all
  run cleanly with differentiated, sensible trade counts.
- `search_params` is deterministic (numpy `Generator(seed)`; same seed → same result),
  tunes on TRAIN, ranks on VALIDATION, and **never touches TEST**. A quick run already
  demonstrated the overfit trap the protocol guards against: a config can show +$5.35 on
  VALIDATION while sitting at -$57.28 on TRAIN — pure validation-noise, which is exactly
  why TEST is evaluated only once, at the very end, per variant.

## Reproduce

```bash
cd /Users/pnle/Desktop/alpaca-cli && python3 -m experiments.sanity_check
```
Machine-readable copy: `experiments/results/harness_sanity.json`.
