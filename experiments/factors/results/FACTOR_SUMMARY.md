# Daily Cross-Sectional Factor Hunt — Honest Verdict

**Date:** 2026-06-04 · **Universe:** ~97 of *today's* large-caps applied backward · **Data:** Alpaca free-tier split/div-adjusted daily bars, **2020-07-27 .. 2026-06-03 (1471 trading days)**. Effective tradeable OOS ≈ 4y (2022-11 → 2026-06) after lookback burn-in.
**Protocol:** dollar-neutral cross-sectional L/S (long top quantile / short bottom, equal dollars per leg → gross 1, net 0 by construction). 4 sequential walk-forward folds (40% expanding train); every knob tuned on each fold's **TRAIN only** by net-of-cost Sharpe; each fold's TEST scored **exactly once**; OOS days concatenated and regressed on SPY close-to-close to split alpha vs beta. Costs charged on turnover, **both sides**, at 2 / 5 / 10 bp (base = **5bp**). $0 commission (Alpaca).

**A real edge must clear ALL four gates:** alpha t-stat ≥ +2 · realized |beta| < 0.15 · positive in the MAJORITY of folds · survives the 5bp base cost.

---

## Ranked table (5bp base case)

| Rank | Factor | alpha t-stat | beta | OOS Sharpe | Ann. return | Max DD | folds + | survives costs | robust | survives ALL gates |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **overnight** (x-sec overnight-tendency L/S) | +1.57 | **-0.0005** | 0.84 | +3.83% | -5.2% | 3/4 | 2/5/10bp ✅ | ❌ (fold-3 only) | ❌ |
| 2 | **momentum** (12-1, monthly) | +1.26 | +0.131* | 0.84 | +13.5% | -21.2% | 3/4 | 2/5/10bp ✅ | ❌ (fold-3 beta) | ❌ |
| 3 | **combo** (mom+rev+lowvol blend) | -0.48 | +0.053 | -0.15 | -1.4% | -29.7% | 1/4 | none | ❌ | ❌ |
| 4 | **streversal** (weekly reversal) | -2.12 | +0.133 | -0.85 | -8.6% | -33.7% | 0/4 | none | ❌ | ❌ |
| 5 | **lowvol** (low-vol L/S) | -1.90 | **-0.596** | -1.64 | -23.8% | -60.1% | 0/4 | none | ❌ | ❌ |

\* momentum's aggregate beta +0.131 is a misleading *average* of wildly sign-flipping per-fold betas: -0.27 / +0.49 / -0.03 / **+0.74** (beta t-stat = +3.63, significantly nonzero). It is **not** beta-neutral in practice.

**0 of 5 factors clear all four gates. 0 of 5 are regime-robust.**

---

## Is there a real, deployable, beta-neutral edge here?

**No.** Not one daily cross-sectional factor in this set is a deployable beta-neutral edge at the honest standard. The two "best" results fail for *different* reasons, and both failure modes are the same traps the prior intraday hunt correctly flagged:

- **momentum** has the prettiest headline (Sharpe 0.84, +13.5%/yr net @5bp, 3/4 folds, low ~0.03/day turnover so cost is irrelevant) — **but its return is leveraged bull-market beta, not alpha.** Strip fold 3 (2025-26 bull, where beta = +0.74) and the other three folds average ~+3.5%/yr with *negative* alpha in two of them. Concatenated alpha t = 1.26 (< 2): you cannot reject zero alpha. Its "dollar-neutral" construction did **not** deliver economic neutrality — the winners-minus-losers basket carries large, time-varying market exposure.

- **overnight** is the honest near-miss and the only genuinely beta-neutral candidate (aggregate beta -0.0005, R²=0.000, |beta|<0.10 in *every* fold — real neutrality, not beta in disguise; low ~0.15/day turnover; positive net at all cost levels). **But its alpha is small and statistically insignificant: t = 1.57 at 5bp (< 2 gate).** It only clears t≥2 at an unrealistically optimistic 2bp (t=2.03), and the alpha is concentrated almost entirely in fold 3 (2025-26 bull tail, alpha +10.8%/yr t=1.83); folds 0-2 are individually noise (t = -0.61, +0.14, +0.74). Not regime-consistent.

- The classic **overnight time-series trap is confirmed explicitly**: "buy at close, sell at next open" on the whole universe shows gross alpha t=2.04 *only* because the overnight hold carries beta 0.31 (vs ~1.0 for the full day) — beta captured at a different hour. It is economically dead: round-tripping the book daily costs ~25%/yr, net @5bp = -13.2%/yr (Sharpe -1.39). The anomaly *shape* is real in the data (SPY overnight +15.1%/yr vs intraday +6.2%/yr) but it is **market beta delivered overnight, not harvestable neutral alpha.**

- **streversal, lowvol, combo** are clean negatives. Weekly reversal has ~zero gross alpha (best robustness-sweep t = +0.25) so cost just turns flat into negative (-2.12 t at 5bp is a turnover artifact, not an inverse edge). Low-vol is a disguised **short-beta book** (-0.60 beta) that got crushed by the AI mega-cap rally and is negative even gross. The combo "diversification helps" thesis is rejected — TRAIN-tuning over-fit reversal, blending destroyed momentum's modest standalone signal, OOS alpha t = -0.48.

### Severe biases — all push toward *over*stating edge, yet none was found
1. **Survivorship / look-ahead membership bias.** Universe = *today's* 97 large-cap survivors applied backward; point-in-time index membership is not available on free data. Delisted/failed names are absent. This inflates every result — so the already-marginal momentum t=1.26 and overnight t=1.57 should be read *even more skeptically* (true edge is likely weaker).
2. **Regime gap.** Free-tier daily floor ≈ 2020-07-27, so the sample **misses 2018Q4 and the Feb/Mar-2020 COVID crash** — exactly the high-vol/crash regimes where momentum famously crashes and reversal/low-vol are historically strongest. The window is 2022-bear + 2023-26-bull + 2025 tariff selloff, dodging the worst momentum-crash windows (optimistic for momentum) and the best reversal windows (pessimistic for reversal). Either way the OOS is not regime-complete.
3. **Multiple comparisons.** 5 factor families, each with per-fold TRAIN grid search (36/27/18/88/72 configs per fold). TEST scored once per fold; no OOS peeking. But finding the *best of 5 families* at t=1.57 is unremarkable — under the null you expect t~1.5+ results from a small search by chance.

---

## Realistic expectation if you tried to deploy the least-bad candidate

The brief sets the honest bar at Sharpe ~0.5-1 for a *real* edge. **Neither candidate clears it on a discounted basis.** Being concrete about what a deployable version would actually look like:

- **Honest expected Sharpe after costs AND after discounting survivorship/regime bias: ~0 to 0.3, not statistically distinguishable from zero.** The 0.84 headline Sharpe on both momentum and overnight is (a) not significant (t < 2), (b) bias-inflated, and (c) concentrated in one bull fold. A bias-discounted, regime-honest estimate of the *neutral* component is essentially zero.
- **Realistic annual return / drawdown:** the overnight L/S net @5bp showed +3.83%/yr with -5.2% max DD on this benign sample — but that is the *in-sample-flattering, survivor-biased* number. On true point-in-time data spanning a crash regime, expect that to compress toward 0-2%/yr with materially deeper drawdowns. Momentum's +13.5%/-21% is mostly recoverable bull beta that would reverse in a momentum crash.

**Dollar outcome (illustrative, NOT a recommendation):** on a $25,000 paper account, an unlevered dollar-neutral overnight book at the optimistic +3.8%/yr ≈ **+$950/yr**, swinging through ~-$1,300 drawdowns — a return indistinguishable from noise and well below the effort/risk. At 2x leverage you double both (~$1,900/yr at ~-$2,600 DD risk) while levering an *insignificant* signal — i.e. you are paying real tail risk for a t=1.57 edge that is probably zero. **Leverage on a non-significant neutral signal is how a flat strategy becomes a losing one.**

---

## Recommendation

**Do NOT deploy any of these to live capital.** None is a real, beta-neutral, regime-consistent edge that survives realistic costs and honest bias discounting. This is fully consistent with the prior intraday VWAP conclusion: change the *structure*, not the parameters or the data vendor first.

**Optional, zero-risk learning step:** the overnight x-sec L/S is the *only* genuinely-neutral construction found, so it is the single defensible thing to **paper-trade as a monitoring experiment** — explicitly to test whether the t=1.57 holds up forward, NOT because it is expected to make money. A paper deployment would look like: each day after close, rank the 97 names by trailing-63d mean overnight return, hold long-top-30% / short-bottom-30% equal dollars overnight only, flatten at the open, $0 net exposure, ~0.15/day turnover, log realized alpha/beta vs SPY weekly. **Treat a forward Sharpe < 0.5 over 6+ months as confirmation it was noise (the likely outcome).** Do not size it as a strategy; size it as a probe.

### Single best next idea
**Get point-in-time universe data and re-run momentum + overnight on a survivorship-free, crash-inclusive history (CRSP/Sharadar-style, or at minimum a Russell-1000 historical membership file), BEFORE testing any new signal.** The single biggest unresolved confound here is not the signal — it is that survivorship + the missing-crash window make even an honest negative result inconclusive at the margin. Two of five factors landed at t=1.3-1.6 *with* the bias tailwind; a clean dataset would tell you definitively whether that collapses to zero (most likely) or firms up. If you must stay on free Alpaca data, the next *structurally different* idea is to move to **crypto (24/7, consolidated free volume, beta-decompose vs BTC)** — it sidesteps both the survivorship problem (smaller, observable universe) and the IEX data defect entirely.

---

**Files:** `experiments/factors/fac_momentum.py`, `fac_overnight.py`, `fac_streversal.py`, `fac_lowvol.py`, `fac_combo.py`; results `experiments/factors/results/{momentum,overnight,streversal,lowvol,combo}.json`; harness `experiments/factors/factor_harness.py` (sanity-verified in `results/factor_harness_sanity.md`).

**Bottom line:** 0 of 5 daily cross-sectional factors clear the honest bar. The momentum headline is bull-market beta in disguise; the overnight signal is genuinely neutral but not statistically significant. There is no deployable beta-neutral edge here — only one neutral probe worth watching on paper, and one clear data upgrade (point-in-time membership) worth doing before believing any of it.
