# Neutral Harness — Sanity Checks

**Purpose:** prove `neutral_harness.py` reuses the validated, production-identical primitives from `harness.py` (no quiet divergence) and that `beta_decompose` is a correct OLS.

`neutral_harness` imports `load_bars, research_backtest, _slip, _metrics, _parse_hm, compute_indicators_df, chronological_split` directly from `harness.py` — it does not reimplement them.

## Sanity A — long_short(short disabled) == harness long-only

- Shared window: **2026-02-09 .. 2026-06-03** (most recent 80 trading days)

- harness.research_backtest long-only trades: **2351**

- neutral long_short (enable_short=False) trades: **2351**

- Trade ledgers BIT-IDENTICAL: **True**

- harness total_pnl = -14.867720 ; neutral total_pnl = -14.867720 ; match = **True**

- **RESULT: PASS**

## Sanity B — beta_decompose recovers beta~1, alpha~0 on SPY-on-SPY

- n_days regressed: **605**

- beta = 1.000000000000  (target 1.0, |err|<1e-6: True)

- alpha_per_day = 0.000e+00  (target 0, |a|<1e-9: True)

- alpha_annual = 0.000e+00

- R^2 = 1.000000000000  (target 1.0, |err|<1e-6: True)

- **RESULT: PASS**

## Overall

- Sanity A: **PASS**

- Sanity B: **PASS**
