# Beta-Neutral VWAP-Family — Honest Verdict

**Date:** 2026-06-04 · **Data:** free IEX 1-min bars, 10 symbols (SPY QQQ IWM AAPL MSFT NVDA AMD TSLA META AMZN), backfilled to **606 trading days, 2024-01-02 .. 2026-06-03**.
**Protocol:** WALK-FORWARD, 4 sequential non-overlapping OOS folds tiling 2025-03-20 onward, spanning genuinely distinct SPY regimes — f0 +9.4% @ ~28% vol (high-vol recovery), f1 +8.3% @ ~10% vol (calm grind), f2 +1.4% (chop), f3 +10.7% (rally). Params (where any) tuned on each fold's **TRAIN only**; each fold's TEST scored **exactly once**; OOS days concatenated and regressed on SPY close-to-close to decompose alpha vs beta.
**Costs/fills:** $0 commission + **1bp adverse slippage per side on EVERY leg** (including hedge rebalances), strict t→t+1 fills, completed bars only, EOD flatten 15:55 NY, no overnight, production caps ($100 notional, max 4 positions, $500 book / $500 capital base). All NET. The neutral backtester is verified **bit-identical** to the production engine (`neutral_harness_sanity.md`: long_short with short disabled reproduces `harness.research_backtest` long-only trade-for-trade; SPY-on-SPY regression recovers beta=1.000000, alpha=0, R²=1.000000).

A real edge must clear ALL of: **alpha t-stat ≥ +2**, realized **|beta| < 0.15**, OOS Sharpe > 0, and **positive in the MAJORITY of folds**.

---

## Ranked table

| Rank | Variant | Construction | alpha t-stat | beta | OOS Sharpe | alpha (ann.) | folds + | survives | robust |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **xsec_reversion** | cross-sectional $-neutral L/S basket | **-0.22** | **-0.039** | -0.40 | -1.0% | 2/4 | ❌ | ❌ |
| 2 | **downmove_bounce** | sharp-drop bounce + SPY hedge | -2.79 | **+0.034** | -2.17 | -6.7% | 1/4 | ❌ | ❌ |
| 3 | **spy_hedged** | long VWAP reversion + SPY hedge | -2.96 | **+0.078** | -2.21 | -12.9% | 1/4 | ❌ | ❌ |
| 4 | **ls_pairs** | symmetric L/S VWAP reversion | -2.37 | **-0.071** | -2.32 | -25.4% | 0/4 | ❌ | ❌ |
| — | **multiregime_base** *(CONTROL)* | long-only, NO hedge — beta-loaded | -3.66 | **+0.276** | -2.37 | -32.2% | 0/4 | ❌ | ❌ |

`survives` = passes all four gates. `robust` = positive across the majority of regime folds. **0 of 4 neutral candidates survive. 0 of 4 are robust.**

---

## The control proves the old "edge" was beta, not alpha

`multiregime_base` is the production-default long-only VWAP mean-reversion run **untuned** (one fixed spec, scored once per fold) as the yardstick. It posts the **highest realized beta of the whole set, +0.276 (t=8.66, R²=0.20)** — diluted from ~1.0 only because the book is EOD-flat and intermittently invested (per-fold beta 0.19→0.49, *rising in down regimes*, the classic long-exposure signature). Once you stop letting that beta ride, the picture collapses: alpha = **-32.2%/yr, t=-3.66**, OOS Sharpe -2.37, **0/4 folds positive**. This is the honest anchor — and it confirms the prior hunt's diagnosis exactly: *the only thing that ever made this family look green was positive market beta inside a single homogeneous bull TEST window.* The earlier SUMMARY.md "edges" (+28.5%, +15.0%, etc., all ❌) were that beta in disguise.

The decisive comparison: as you walk **down the beta column** — base +0.276 → ls_pairs -0.071 → spy_hedged +0.078 → downmove_bounce +0.034 → xsec_reversion -0.039 — and the construction actually **achieves neutrality (|beta| < 0.15)**, the alpha does **not** emerge from behind the beta. It stays **negative or zero**. Removing beta did not *expose* a buried edge; it *exposed the absence of one*.

---

## Per-variant truth

- **ls_pairs** (symmetric dollar-neutral L/S): neutrality genuinely achieved (aggregate beta -0.071, |beta|<0.15 PASS; long/short imbalance only 4–5% in f0–2). With beta stripped, the residual is **significantly NEGATIVE**: alpha -25.4%/yr, t=-2.37, Sharpe -2.32, **0/4 folds positive**, win rate 47–49% (sub-coin-flip), per-trade net **-1.99bp** — sitting *exactly* on the -2bp round-trip cost floor. Textbook no-information strategy bleeding slippage.
- **spy_hedged** (long reversion, SPY-hedged to ~0 beta): hedge worked mechanically (beta cut +0.191 unhedged → +0.078 hedged). But the *same tuned long leg unhedged* already has insignificant alpha (t=-0.52) — the long PnL was ~all beta. Hedged: alpha -12.9%/yr, **t=-2.96**, Sharpe -2.21, 1/4 folds positive (the lone winner f3 had alpha_t=+0.13, i.e. zero). The hedge can only subtract: it pays 1bp/side on 3,000+ rebalances/fold with no real alpha to protect.
- **downmove_bounce** (the pre-registered "reversion bounce after a sharp DOWN move," the prior hunt's one surviving hint, VAL K=15 t=4.19): neutrality clean (|beta|=0.034). The directional bounce **is real but tiny** — unhedged long-only win rate is a remarkably consistent 54–56% across all four regimes — yet net PnL/trade is only -1.2/-0.6/+0.9/+0.4 bp. The gross +2–6bp/event bounce from the event study is **almost exactly consumed by the ~2bp round-trip cost**, leaving a coin flip; the hedge then drags it reliably negative (alpha -6.7%/yr, t=-2.79, 1/4 folds). The original t=4.19 was a gross, single-window IEX artifact that does not replicate net of costs across regimes.
- **xsec_reversion** (cross-sectional L/S basket — the cleanest neutral construction, neutral by construction): the most honest near-zero. Beta -0.039 (R²=1.9%, genuinely not beta-in-disguise), but alpha **-1.0%/yr, t=-0.22 — indistinguishable from zero**, Sharpe -0.40, only 2/4 folds positive and those two are +0.02% / +0.09% (noise). TRAIN Sharpes looked encouraging (1.26/0.97/0.85/0.53) and **none** carried into TEST — the canonical overfit/no-signal pattern. Notably, the TRAIN search **unanimously picked the most-throttled config** (rebalance every 30 bars, widest entry band) in all 4 folds — strong evidence there is no per-bar edge that can pay its own transaction costs.

---

## THE SINGLE CLEAR TAKEAWAY

**No.** Not one beta-neutral construction in the VWAP family showed real, regime-consistent out-of-sample alpha. Every variant that genuinely achieved neutrality (|beta| < 0.15) produced alpha that is **negative or statistically zero**, in the **minority** of folds, with per-trade economics pinned **at or below the ~2bp round-trip cost floor**. The neutral experiments are not failures of construction — neutrality was demonstrably achieved (and the engine is bit-verified). They are a clean, honest demonstration that **the intraday VWAP mean-reversion signal carries no alpha on this universe/data once market beta is removed.** This is the opposite of the prior hunt's failure mode (one bull window carrying a long book): here losses/zeros are *uniform across regimes*.

---

## DECISION

**Is $99/mo SIP data justified to sharpen THIS family? NO — not yet, and not for this signal class.**

The honest reasoning: there is **no neutral signal here to sharpen.** SIP would fix a real, known data defect — IEX is only ~2–3% of consolidated volume, so VWAP and volume-ratio inputs are genuinely distorted, and PnL is pinned near a noise floor that better data could move. **But** the failure is not a near-miss being strangled by noise. The structural conclusion (PnL was beta; neutral residual is negative/zero) is **robust across 4 regimes and 5 constructions**, and the cross-sectional variant — which is neutral *by construction*, not by hedge — lands flat at t=-0.22 with no per-bar edge that survives its own costs even on TRAIN. Paying for SIP to "confirm" a signal that is structurally absent on free data is buying a sharper picture of zero. **Spend the $99/mo only after a different signal class shows a green, regime-consistent, neutral OOS pretest on free data — then SIP to confirm/scale it.**

### Best structurally-different next idea

**Move to crypto (24/7), a different microstructure entirely — and there is NO SIP problem there.**
- Crypto spot/perp data is **consolidated and free** at the exchange level (Alpaca crypto, Binance, Coinbase) — the IEX 2–3%-of-volume distortion that crippled VWAP/volume signals here **simply does not exist**. You get true volume for free, removing the single biggest caveat on every result above.
- 24/7 trading removes the EOD-flatten constraint and the overnight-gap problem; mean-reversion and cross-exchange/cross-pair relationships have far more bars per "regime."
- Keep the rigor protocol verbatim: walk-forward over multiple regimes, dollar-neutral L/S by construction, mandatory beta decomposition (vs BTC instead of SPY), 1bp/side on every leg, OOS scored once.

**Second choice (if staying in equities):** abandon the VWAP/volume-reversion signal class and test a **structurally different signal** — e.g. overnight-gap / opening-range behavior, or a cross-sectional **fundamental/event** factor (earnings drift, sector dispersion) that does not depend on intraday tick volume and is therefore not poisoned by IEX. The lesson from 0/9 (prior) + 0/4 (here) is unambiguous: **change the structure, not the parameters, and not the data vendor first.**

**Do not paper-trade any VWAP-family variant.**

---

### Rigor / multiple-comparisons honesty
- Selection counts (TRAIN/VAL only, TEST scored once per fold): ls_pairs 288 (72-pt grid × folds), spy_hedged honest 48 (the JSON's 96 double-counts the re-run unhedged diagnostic), xsec_reversion 144 (36-pt grid × 4 folds), downmove_bounce 96 (216-pt grid, 24 sampled/fold), multiregime_base 1 (fixed control, no tuning).
- No TEST/OOS peeking in any variant. Beta regressions run once on the concatenated OOS days.
- Adversarial stress set was empty (no variant produced a positive OOS candidate worth stress-testing) — itself an honest signal that nothing cleared the bar.
- Files: `expn_ls_pairs.py`, `expn_spy_hedged.py`, `expn_xsec_reversion.py`, `expn_downmove_bounce.py`, `expn_multiregime_base.py` and matching `results/*.json`; neutral engine `neutral_harness.py` (sanity-verified in `results/neutral_harness_sanity.md`).
