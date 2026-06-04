# PhillyTrader

An Alpaca **paper-trading** research project: a deterministic, multi-agent intraday
VWAP mean-reversion system, a trainable parameter-optimizer + ML entry-gate, and a
rigorous out-of-sample **edge-hunt harness** — built to find a real trading edge *or
honestly prove there isn't one.*

**It is paper-only and dry-run by default.** It submits nothing to the broker unless you
explicitly `--arm` it, and it has no real-money mode at all.

> ⚠️ Not financial advice. This is an engineering/research demo against Alpaca's PAPER environment.

## 🔬 Honest status & finding

The system is **built, tested (77 passing tests), and verified end-to-end.** A rigorous
edge hunt then tested **9 strategy variants** under strict chronological train/validation/test
discipline on **190 trading days** of free IEX data (the research backtester is *bit-identical*
to the production engine).

**Result: 0 of 9 variants showed a real, robust out-of-sample edge.** Every superficially
profitable result was long-beta in a single bullish test window, not repeatable alpha (the
market-neutral twin of the best config loses on all three splits). The ML gate's out-of-sample
AUC is ~0.5 (coin flip). The prime suspect is the **free IEX feed** (~2–3% of true volume),
which pins signals to the noise floor. Full writeup → [`experiments/results/SUMMARY.md`](experiments/results/SUMMARY.md).

This is a *successful* outcome for the harness: it refused to manufacture fake edge.

## Repository layout

| Path | What |
|---|---|
| `alpaca_cli.py` | Minimal read-only CLI (`account`/`positions`/`orders`/`clock`) |
| `agentic_trader/` | The system: 5 separated agents (MarketData → Strategy → ML gate → Risk(veto) → Execution → Position), continuous loop, backtest, optimizer, ML training |
| `agentic_trader/*.md` | `STRATEGY_RULES` · `ARCHITECTURE` · `API_MAP` · `INTERFACE_SPEC` · `ENV_REPORT` |
| `experiments/` | Out-of-sample edge-hunt harness + 9 experiments + results |
| `tests/` | 77 pytest tests (indicators, strategy, risk, backtest, ML, orchestrator) |

Setup needs a `.env` (copy `.env.example`) with your Alpaca **paper** keys. The `.env` is gitignored — never commit keys.

---

## The `agentic_trader` system

## What it is

- **Strategy:** intraday session-VWAP mean reversion (deterministic — no LLM in the trade loop).
- **Universe (fixed order):** `SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMD, TSLA, META, AMZN`.
- **Bars:** 1-minute, **regular hours only** (09:30–16:00 America/New_York), **completed bars only**.
- **Timing:** decisions are made on the last *completed* bar (minute `t`); fills happen at
  `t+1` (next bar's open in backtest; immediately in live, since that bar has already closed).
  **No look-ahead** — indicators, features, and predicates use only bars `[0..t]`.

---

## Strategy rules

Per symbol, from the completed session bars:

```
typical_price     = (high + low + close) / 3
session_vwap      = sum(typical_price * volume) / sum(volume)        # over the session so far
last_price        = close of the last completed bar
dist_from_vwap    = (last_price - session_vwap) / session_vwap       # negative = below VWAP
current_volume    = volume of the last completed bar
rolling20_avg_vol = mean volume of the 20 bars IMMEDIATELY PRECEDING the current bar
                    (the current bar is EXCLUDED)
volume_ratio      = current_volume / rolling20_avg_vol
```

**ENTRY** — BUY `$100` notional (MARKET) when **all** are true:
- currently flat in this symbol
- `dist_from_vwap <= -entry_dist`  (default `entry_dist = 0.005` → at least 0.5% below VWAP)
- `volume_ratio   >= vol_mult`     (default `vol_mult   = 1.2`)
- (and the ML gate passes — see below)

**EXIT** — SELL the full position (MARKET) when **any** are true:
- `dist_from_vwap >= -vwap_exit_band`   (default `0.001` → price back within 0.1% of VWAP)
- `holding_minutes > max_hold`          (default `15` minutes)
- `unrealized_plpc <= -stop_loss`       (default `0.005` → 0.5% loss)
- end-of-day flatten on/after 15:55 NY (overrides everything; no overnight positions)

Exit priority when several fire: `eod_flatten` > `vwap_revert` > `max_hold` > `stop_loss`.

**Orders are simple MARKET orders only** — no bracket/OCO/stop/limit orders.

**Risk caps (the Risk agent has VETO authority over every proposal):**
- `max_open_positions = 4`
- `max_total_exposure = $500` (open market value + pending new-entry notional ≤ 500; an entry
  that would breach the cap is *reduced* to the remaining room, or rejected if the room < $1)
- per-symbol cooldown: at most 1 new ENTRY per symbol per 10 minutes
- duplicate-order guard: never stack a second order on a symbol that already has an open order

**Trainable params (what the optimizer searches):**
`entry_dist, vol_mult, max_hold, vwap_exit_band, stop_loss, ml_threshold`.

### ML entry scorer (a learned gate on entries)

At each candidate entry, a decision-time feature vector is built using **only** information
available at minute `t` (no future data):

```
dist_from_vwap, volume_ratio, log1p(rolling20_avg_vol), minute_of_session,
recent_return (close[t]/close[t-5] - 1), bar_range_pct = (high-low)/close,
session_progress in [0,1], symbol_id (index in the fixed universe)
```

The model predicts `p_win ∈ [0,1]`. A live entry requires `(rules pass) AND (p_win >= ml_threshold)`.
**If no model is trained yet, the scorer is a passthrough (`p_win = 1.0`)** so the system runs
rule-only. `ml_threshold` defaults to `0.0` (gate disabled) until a model is trained and you
raise the threshold.

---

## Architecture

See [`agentic_trader/ARCHITECTURE.md`](agentic_trader/ARCHITECTURE.md) for the full design and
[`agentic_trader/INTERFACE_SPEC.md`](agentic_trader/INTERFACE_SPEC.md) for exact module signatures.

```
                          ┌──────────────────────────────────────────────┐
                          │              orchestrator.run_cycle           │
                          │  (one full pass; logs every step to JSONL)    │
                          └──────────────────────────────────────────────┘
                                            │
        RECONCILE broker state (positions + open orders)   ← INV-6 (every cycle)
                                            │
   ┌────────────┐   snapshots   ┌────────────┐  proposal  ┌──────────┐  decision  ┌────────────┐
   │ ① Market   │ ───────────▶  │ ② Strategy │ ─────────▶ │ ④ Risk   │ ─────────▶ │ ⑤ Execution│
   │   Data     │               │  (rules +  │            │ (VETO,   │  APPROVED  │ (safety    │
   │ (bars →    │               │  ③ ML gate)│            │  REDUCE) │   only     │  gate)     │
   │ indicators)│               └────────────┘            └──────────┘            └────────────┘
   └────────────┘                                              │                        │
         │                                                     │                  paper + armed?
         │                                              ⑥ Position Analysis        ├─ yes → submit MARKET order (paper)
         └────────────────────────────────────────────  (P&L + exit class)        └─ no  → DRY-RUN (log only)
```

- **① Market Data** (`data/market_data.py`): fetch completed regular-hours 1-min bars, compute indicators.
- **② Strategy** (`agents/strategy_agent.py`): pure, deterministic ENTRY/EXIT/HOLD proposals.
- **③ ML Scorer** (`ml/scorer.py`): learned `p_win` gate on entries (passthrough until trained).
- **④ Risk** (`agents/risk_agent.py`): the single chokepoint — approves / reduces / vetoes (with a logged reason).
- **⑤ Execution** (`agents/execution_agent.py`): submits a MARKET order ONLY when paper + risk-approved + armed; else dry-run.
- **⑥ Position** (`agents/position_agent.py`): read-only P&L + exit classification for logging/labels.
- **Backtest** (`backtest/engine.py`, `backtest/optimize.py`): strict `t→t+1` simulator + walk-forward search.

---

## Safety model

This is the most important section. The system is built so that the **default is to do nothing
to the broker**, and so that real-money trading is structurally impossible.

1. **Dry-run by default.** Every command runs in DRY-RUN unless you pass `--arm`. In dry-run the
   system computes, logs, and *prints* the intended orders but submits **nothing**.
2. **`--arm` only enables PAPER submission.** Submitting real paper orders requires `--arm`
   (or `Settings.live_paper=True`) **and** a paper account.
3. **No live (real-money) mode.** Clients are always constructed `paper=True`. If `ALPACA_PAPER`
   is not `"true"`, the system refuses to submit any order and logs a clear error.
4. **No order bypasses Risk (INV-4).** Strategy only *proposes*; Risk approves/reduces/vetoes;
   only approved orders reach Execution. Every non-approval logs a reason.
5. **Everything is logged** to `logs/` as JSONL (one event per line, timestamped): data snapshot,
   strategy proposal, ML score, risk decision, order request, order response, position summary, errors.
6. **Reconcile every cycle (INV-6).** State is rebuilt from Alpaca (positions + open orders) each
   cycle rather than trusting only local memory.

Execution submits an order **iff**: the TradingClient is paper (`paper=True`) **AND** the run is
explicitly armed **AND** the order was risk-approved **AND** `ALPACA_PAPER == "true"`.

---

## Setup

Requirements: **Python 3.14**, `alpaca-py`, `numpy`, `pandas` (plus optional `scikit-learn`,
`pyarrow`, `joblib` for the ML path — the code degrades gracefully if they are absent).

```bash
# 1) Install deps (already present in this environment)
python3 -m pip install -r requirements.txt

# 2) Credentials live in .env at the project root (already configured here). Format:
#    ALPACA_API_KEY_ID=PK...
#    ALPACA_API_SECRET=...
#    ALPACA_PAPER=true
#    The key id starts with PK (paper). ALPACA_PAPER MUST be "true" — there is no live mode.

# 3) Sanity-check the account (read-only)
python3 -m agentic_trader account
```

Run anything via either `python3 -m agentic_trader <subcommand>` or `python3 -m agentic_trader.cli <subcommand>`.

---

## Example commands (full workflow)

The intended order is: **backfill → backtest → optimize → train → dry-run → loop --arm.**

```bash
# 0) Inspect state (read-only, no orders)
python3 -m agentic_trader account
python3 -m agentic_trader positions

# 1) BACKFILL — cache historical 1-min bars under data/cache/
python3 -m agentic_trader backfill --start 2024-05-01 --end 2024-05-31
python3 -m agentic_trader backfill --symbols SPY QQQ AAPL --start 2024-05-01 --end 2024-05-31

# 2) BACKTEST — strict t->t+1 simulation over the cached bars
python3 -m agentic_trader backtest --start 2024-05-01 --end 2024-05-31
python3 -m agentic_trader backtest --symbols SPY QQQ --start 2024-05-01 --end 2024-05-31

# 3) OPTIMIZE — walk-forward search over the trainable params; writes best_params.json
python3 -m agentic_trader optimize --n-iter 50 --folds 3 --start 2024-05-01 --end 2024-05-31

# 4) TRAIN — fit the ML entry scorer from a dataset of (decision-time features -> win/loss)
python3 -m agentic_trader train                       # uses the default dataset path
python3 -m agentic_trader train --dataset data/cache/dataset.parquet

# 5) DRY-RUN — one full live cycle that submits NOTHING (default safe mode)
python3 -m agentic_trader dry-run
python3 -m agentic_trader run                          # also dry-run unless --arm is passed

# 6) ARM (PAPER) — actually submit paper orders. Requires a paper account + ALPACA_PAPER=true.
python3 -m agentic_trader run  --arm                   # one armed paper cycle
python3 -m agentic_trader loop --arm --interval 60     # continuous, market-hours aware

# Continuous DRY-RUN loop (watch decisions live without trading):
python3 -m agentic_trader loop --interval 60
```

---

## How retraining works

The ML scorer learns from the system's own realized outcomes. The cycle is:

```
live logs  →  dataset  →  train  →  models/
```

1. **Collect outcomes.** Every cycle logs decision-time features and, as trades close, their
   realized P&L. Backtests also emit labeled decision-time features (one row per ENTRY taken),
   with `label = 1` iff the round-trip was profitable net of costs.
2. **Build the dataset.** Closed/simulated `TradeRecord`s are turned into a design matrix
   (`ml/dataset.build_dataset`) whose columns are exactly `ml.features.FEATURE_ORDER` — the same
   layout the live scorer uses, so training and inference can never drift. It is persisted as
   Parquet (or CSV) via `save_dataset`.
3. **Train.** `agentic_trader train` (→ `ml/train_model.train`) does a **temporal** split (train on
   earlier data, validate on later — no leakage), fits a logistic-regression entry scorer
   (sklearn, with a pure-numpy fallback), and writes `models/ml_scorer.pkl` + `models/metrics.json`.
4. **Use it.** On the next run, `EntryScorer.load()` picks up the model automatically. Until you
   raise `ml_threshold` above `0.0`, the gate stays effectively open; set a threshold (e.g. via
   `best_params.json`) to start requiring `p_win >= ml_threshold` for entries.

If the model file is missing or fails to load, the scorer falls back to passthrough (`p_win = 1.0`)
and the system runs rule-only — it never crashes the trade loop on an ML issue.

---

## Tests

Fully synthetic and network-free — no test hits the network or places an order.

```bash
python3 -m pytest tests/ -q
```

Coverage:
- `test_indicators.py` — session VWAP (hand-computed), rolling-20 **excludes** the current bar,
  `volume_ratio`, `dist_from_vwap` sign, validity gating.
- `test_strategy.py` — entry fires *exactly* at the boundary (and not just inside it); each exit
  trigger (VWAP-revert, >15min, −0.5%, EOD) fires independently; the ML gate blocks/allows entries.
- `test_risk.py` — rejects on >4 positions, >$500 exposure, within-cooldown re-entry, duplicate
  open order; the REDUCE path; exits always approved; paper/blocked guards; every decision has a reason.
- `test_backtest.py` — determinism (same input → same trades) and **no look-ahead** (a future-only
  spike cannot change a past decision; fills occur at `t+1` open).
- `test_features_ml.py` — feature vector is stable + ordered; `EntryScorer` passthrough returns
  `1.0` with no model; dataset build/save/load shapes and labels are correct.
```
