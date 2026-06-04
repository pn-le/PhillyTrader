# factor_harness — Correctness Sanity Suite

PURE RESEARCH / BACKTESTING. ZERO orders. These checks lock the DAILY cross-sectional factor engine before any experiment trusts it — same brutal standard that correctly found 0 edges intraday.

**Panel:** 97 full-history names, 1471 trading days, 2020-07-27 .. 2026-06-03. Benchmark SPY: 1471 daily returns.

## (a) No spurious edge — constant & random scores

A dollar-neutral L/S construction fed a NON-INFORMATIVE score must not manufacture edge. A constant score has zero cross-sectional dispersion -> no information to rank on -> the book stays empty -> EXACTLY 0 return, 0 turnover (regardless of cost). A random score produces a real but information-free book whose GROSS alpha (zero cost) must be statistically indistinguishable from zero (|t| < 2) — that is the test of the CONSTRUCTION. Its NET alpha at 5bp must be NEGATIVE: a random book churning ~1.6 turnover/day correctly bleeds slippage (desired engine behavior, not an edge).

| score | OOS ann.return | OOS Sharpe | alpha (ann.) | alpha t-stat | beta | avg turnover | n_days |
|---|---|---|---|---|---|---|---|
| constant (5bp) | 0.00% | 0.00 | 0.00% | 0.00 | 0.000 | 0.0000 | 882 |
| random GROSS (0bp) | -1.64% | -0.39 | -1.83% | -0.81 | 0.009 | 1.5955 | 882 |
| random NET (5bp) | -22.14% | - | -22.25% | -10.27 | - | - | - |

- constant -> exactly flat (0 return, 0 turnover): **PASS**
- random GROSS -> alpha t-stat |-0.81| < 2 (construction manufactures no edge): **PASS**
- random NET@5bp -> alpha -22.25% < 0 (churning noise correctly bleeds cost): **PASS**

## (b) Beta recovery — SPY regressed on itself

Using SPY's own daily return as the 'strategy' in beta_decompose must recover beta = 1, alpha = 0, R^2 = 1 exactly (the OLS is fit on identical y and x).

- beta = **1.000000**, alpha/day = **-2.49e-18**, R^2 = **1.000000**, n_days = 1471
- **PASS**

## (c) Cost model — higher turnover strictly reduces net return

Same 1-day reversal factor and weights throughout. (c1) At fixed daily rebalancing, raising cost_bps/side must strictly lower net total return. (c2) Rebalancing MORE often (freq 1 < 5 < 21 days) must strictly raise turnover and total cost.

### (c1) cost sensitivity at daily rebalance

| cost bps/side | net total return | total cost |
|---|---|---|
| 0 | 5.11% | 0.0000 |
| 2 | -33.65% | 0.4600 |
| 5 | -66.74% | 1.1501 |
| 10 | -89.48% | 2.3001 |

- higher cost_bps -> strictly lower net return: **PASS**

### (c2) turnover/cost vs rebalance frequency (cost 5bp/side)

| rebalance_freq (days) | avg turnover | total cost | net total return |
|---|---|---|---|
| 1 | 1.5636 | 1.1501 | -66.74% |
| 5 | 0.3140 | 0.2309 | -28.14% |
| 21 | 0.0731 | 0.0538 | 17.48% |

- more frequent rebalance -> more turnover: **PASS**
- more frequent rebalance -> more total cost: **PASS**

## Verdict

**ALL SANITY CHECKS PASS**
