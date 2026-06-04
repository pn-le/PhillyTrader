# INTERFACE_SPEC.md — Single Source of Truth for Module Signatures

> This file is NORMATIVE. Every module built next MUST match the exact public signature
> (name, params, return type) and the one-line behavior contract listed here. All types
> referenced are defined in `agentic_trader/types.py`; all params/limits/settings in
> `agentic_trader/config.py`. Getting these right prevents integration drift across the
> parallel implementers. PAPER ONLY. Deterministic trade loop. ML gates ENTRIES only.

---

## 0. Shared contract (already built — bind to these)

### `agentic_trader/types.py` (frozen dataclasses + enums)
- Enums: `Side(BUY|SELL|HOLD)`, `Intent(ENTRY|EXIT|NONE)`, `Verdict(APPROVE|REDUCE|REJECT)`,
  `PositionStatus(FLAT|LONG)`, `ExitStatus(HOLD|VWAP_REVERT|MAX_HOLD|STOP_LOSS|EOD_FLATTEN)`.
- `Bar(symbol, start, open, high, low, close, volume, trade_count=None, vwap=None)`
- `Indicators(symbol, session_vwap, last_price, dist_from_vwap, current_volume, rolling20_avg_vol, volume_ratio, n_bars, valid=False, invalid_reason=None)`
- `PositionState(symbol, status=FLAT, qty=0.0, avg_entry_price=None, entry_time=None, market_value=None, unrealized_pl=None, unrealized_plpc=None, current_price=None)` — props `.is_flat`, `.is_long`
- `Snapshot(symbol, asof, indicators, position_state, bars=())`
- `FeatureVector(symbol, dist_from_vwap, volume_ratio, log_rolling_vol, minute_of_session, recent_return, bar_range_pct, session_progress, symbol_id)`
- `Proposal(symbol, side, intent, reason, notional=None, qty=None, signal_values={}, ml_p_win=None, features=None, asof=None)` — props `.is_entry`, `.is_exit`
- `RiskDecision(proposal, verdict, reason, adjusted_notional=None)` — props `.approved`, `.effective_notional`
- `OrderResult(symbol, side, status, dry_run, submitted_at, id=None, notional=None, qty=None, filled_avg_price=None, error=None)`
- `PositionSummary(symbol, qty, avg_entry_price, current_price, market_value, unrealized_pl, unrealized_plpc, holding_minutes, exit_status=HOLD, would_exit=False)`
- `TradeRecord(symbol, entry_time, exit_time, entry_price, exit_price, qty, pnl, return_pct, holding_min, exit_reason, features={}, label=0)`

### `agentic_trader/config.py`
- Dataclasses: `StrategyParams(entry_dist=0.005, vol_mult=1.2, max_hold=15.0, vwap_exit_band=0.001, stop_loss=0.005, ml_threshold=0.0, notional=100.0)` (`.to_dict()`, `.from_dict()`, `.TRAINABLE`);
  `RiskLimits(max_open_positions=4, max_total_exposure=500.0, per_symbol_cooldown_min=10.0)`;
  `Settings(universe, timezone, feed='iex', loop_interval_sec=60.0, live_paper=False, logs_dir, models_dir, data_cache_dir, env_path)`.
- Functions: `load_env(path=ENV_PATH) -> dict`, `get_credentials(path=ENV_PATH) -> {api_key, api_secret, paper:bool}`, `is_paper_env(path=ENV_PATH) -> bool`,
  `load_strategy_params(path=BEST_PARAMS_PATH) -> StrategyParams`, `save_best_params(params, path=BEST_PARAMS_PATH) -> Path`, `ensure_dirs(settings=None) -> None`,
  `make_trading_client(path=ENV_PATH) -> TradingClient`, `make_data_client(path=ENV_PATH) -> StockHistoricalDataClient`, `make_clients(path=ENV_PATH) -> {"trading", "data", "paper"}`.
- Constants: `UNIVERSE`, `SESSION_TZ`, `SESSION_OPEN`, `SESSION_CLOSE`, `EOD_FLATTEN_TIME`, `SESSION_MINUTES=390`, `RECENT_RETURN_K=5`, `ROLLING_VOL_WINDOW=20`, `MIN_BARS_FOR_ENTRY=21`, `SLIPPAGE_BPS=1`, `MIN_NOTIONAL=1.0`, `MODEL_BACKEND="sklearn"`, `DATASET_FORMAT="parquet"`, `MODEL_PERSIST="joblib"`, plus all path constants.

### `agentic_trader/logging_util.py`
- `JsonlLogger(logs_dir=None, *, prefix="trader", echo=False)` — timestamps/paths computed at CALL time only; dir created lazily.
  - `.log_event(kind, payload=None, **fields) -> dict`
  - `.log_snapshot(snapshot, **extra)`, `.log_proposal(proposal, **extra)`,
    `.log_ml_score(symbol, p_win, ml_threshold, gate, **extra)`, `.log_risk(decision, **extra)`,
    `.log_order_request(order, *, mode, **extra)`, `.log_order_response(result, **extra)`,
    `.log_position_summary(summaries, exposure=None, **extra)`,
    `.log_error(where, msg, exc=None, **extra)` (tees to errors-<date>.jsonl).

### `agentic_trader/ml/features.py`
- `FEATURE_ORDER: list[str]` — canonical ordered layout (changing it is a breaking change).
- `feature_names() -> list[str]` — copy of FEATURE_ORDER.
- `to_row(fv: FeatureVector) -> list[float]` — flatten in FEATURE_ORDER (numeric only).
- `build_features(snapshot: Snapshot, params: StrategyParams | None = None) -> FeatureVector` — PURE; decision-time only; raises ValueError if data insufficient.

---

## 1. `agentic_trader/data/market_data.py` — Agent ① Market Data

```python
class MarketDataAgent:
    def __init__(self, data_client, settings: Settings, params: StrategyParams,
                 logger: JsonlLogger | None = None) -> None: ...

    def get_snapshot(self, symbol: str, position_state: PositionState | None = None,
                     now: datetime | None = None) -> Snapshot: ...
    # Fetch completed regular-hours 1-min bars for the session containing `now`
    # (America/New_York), drop incomplete bars (start+60s > now_utc), compute indicators,
    # and return a Snapshot whose asof == decision-bar start. position_state defaults FLAT.

    def get_snapshots(self, universe: list[str] | None = None,
                      position_states: dict[str, PositionState] | None = None,
                      now: datetime | None = None) -> list[Snapshot]: ...
    # Snapshot for each symbol in universe (defaults Settings.universe), preserving order.

def compute_indicators(bars: list[Bar], params: StrategyParams) -> Indicators: ...
# PURE. STRATEGY_RULES §2 exactly: session_vwap over ALL completed bars incl. decision bar;
# rolling20_avg_vol = mean of the 20 bars STRICTLY preceding decision bar [n-21:n-1];
# sets valid=True iff n>=MIN_BARS_FOR_ENTRY and denominators>0, else valid=False+invalid_reason.
# NO look-ahead: uses only the bars passed in (already completed, ascending).
```

## 2. `agentic_trader/data/history.py` — historical bar backfill + cache

```python
def backfill(symbols: list[str], start: datetime, end: datetime, *,
             data_client=None, settings: Settings | None = None,
             logger: JsonlLogger | None = None) -> dict[str, Path]: ...
# Pull 1-min IEX bars per symbol for [start,end], regular-hours filtered, write one cache
# file per symbol under data/cache/ (Parquet if pyarrow else CSV). Returns {symbol: path}.

def load_cached_bars(symbol: str, start: datetime, end: datetime,
                     settings: Settings | None = None) -> "pandas.DataFrame": ...
# Load cached bars for symbol, filtered to [start,end]. DataFrame columns:
# ['symbol','start','open','high','low','close','volume','trade_count','vwap'],
# tz-aware 'start' in America/New_York, ascending. Empty DataFrame if no cache.
```

## 3. `agentic_trader/agents/strategy_agent.py` — Agent ② Strategy

```python
class StrategyAgent:
    def __init__(self, params: StrategyParams, settings: Settings | None = None,
                 logger: JsonlLogger | None = None) -> None: ...

    def propose(self, snapshot: Snapshot, params: StrategyParams | None = None,
                ml_scorer: "EntryScorer | None" = None) -> Proposal: ...
    # Apply STRATEGY_RULES §3/§4 to one snapshot. EXIT first if long (vwap_revert>max_hold>
    # stop_loss; eod_flatten on/after 15:55) -> Proposal(side=SELL, intent=EXIT, qty=full).
    # Else if flat and indicators.valid and not in EOD window and entry predicate passes:
    # build features, call ml_scorer.p_win (1.0 if None/untrained), gate p_win>=ml_threshold;
    # on pass -> Proposal(side=BUY, intent=ENTRY, notional=params.notional, ml_p_win, features).
    # Otherwise -> Proposal(side=HOLD, intent=NONE, reason=...). Uses params arg if given else self.
```

## 4. `agentic_trader/agents/risk_agent.py` — Agent ④ Risk (VETO authority)

```python
class RiskAgent:
    def __init__(self, limits: RiskLimits, settings: Settings | None = None,
                 logger: JsonlLogger | None = None) -> None: ...

    def review(self, proposal: Proposal, account, positions: list,
               open_orders: list, cooldowns: dict[str, datetime],
               params: StrategyParams, limits: RiskLimits | None = None,
               approved_buys: list[RiskDecision] | None = None) -> RiskDecision: ...
    # STRATEGY_RULES §8 + INV-4. Returns RiskDecision with a reason ALWAYS set.
    # - ALPACA_PAPER != "true" (is_paper_env) -> REJECT("not_a_paper_account").
    # - HOLD proposals -> REJECT("no_action") (never executed).
    # - SELL/EXIT -> always APPROVE (A12).
    # - BUY: REJECT if open_orders has a pending order for symbol (A17, "open_order_pending");
    #        REJECT("max_open_positions") if len(positions) >= limits.max_open_positions;
    #        REJECT("cooldown") if cooldowns[symbol] within per_symbol_cooldown_min;
    #        exposure = sum(position market_value) + sum(effective_notional of approved_buys)
    #        + proposal.notional; if > max_total_exposure: room = cap - mv - pending;
    #        room<=0 -> REJECT("max_total_exposure"); room<MIN_NOTIONAL -> REJECT("exposure_room_below_min");
    #        else REDUCE(adjusted_notional=room) (A13). Otherwise APPROVE.
    # `account`/`positions`/`open_orders` are raw Alpaca objects (numeric fields are strings).
    # `approved_buys` are this-cycle prior approvals so one slot/dollar serves one proposal (A14).
```

## 5. `agentic_trader/agents/execution_agent.py` — Agent ⑤ Execution (safety gate)

```python
class ExecutionAgent:
    def __init__(self, trading_client, settings: Settings, *,
                 logger: JsonlLogger | None = None) -> None: ...

    def execute(self, risk_decision: RiskDecision, *, armed: bool) -> OrderResult: ...
    # INV-1/2/3. Submit a simple MARKET order IFF: is_paper_env()==True AND trading_client
    # was constructed paper=True AND risk_decision.approved AND armed. Else DRY-RUN: log the
    # intended order, submit NOTHING, return OrderResult(dry_run=True, status="dry_run").
    # ALPACA_PAPER != "true" -> log_error + OrderResult(dry_run=True, status="refused", error=...).
    # BUY uses effective_notional (notional=); SELL uses proposal.qty (qty=); TimeInForce.DAY.
    # Logs order_request (mode) then order_response. NO bracket/OCO/stop/limit.
```

## 6. `agentic_trader/agents/position_agent.py` — Position Analysis

```python
class PositionAgent:
    def __init__(self, params: StrategyParams, settings: Settings | None = None,
                 logger: JsonlLogger | None = None) -> None: ...

    def summarize(self, positions: list, params: StrategyParams | None = None,
                  snapshots: dict[str, Snapshot] | None = None,
                  now: datetime | None = None) -> list[PositionSummary]: ...
    # Build a PositionSummary per open position (parse Alpaca string fields to float),
    # compute holding_minutes from entry_time (snapshots/local log else position open time),
    # and classify exit_status via the SAME §4 predicates (vwap_revert>max_hold>stop_loss,
    # eod_flatten on/after 15:55). would_exit = exit_status != HOLD. Used for logging/labels.

def total_exposure(positions: list) -> float: ...
# Sum of float(position.market_value) over open positions (helper for risk/logging).
```

## 7. `agentic_trader/ml/scorer.py` — Agent ③ ML Scorer (entry gate)

```python
class EntryScorer:
    @classmethod
    def load(cls, path: str | Path | None = None) -> "EntryScorer": ...
    # Load a persisted model (joblib .pkl, else JSON weight dump). If the file is absent or
    # load fails, return a PASSTHROUGH scorer (p_win always 1.0). Default path = config.ML_MODEL_PATH.

    def p_win(self, feature_vector: FeatureVector) -> float: ...
    # Return calibrated win probability in [0,1] for a candidate entry. Passthrough -> 1.0.
    # Uses ml.features.to_row(feature_vector) for layout parity with training.

    @property
    def is_passthrough(self) -> bool: ...
    # True iff no model is loaded (running rule-only).
```

## 8. `agentic_trader/ml/dataset.py` — training dataset build / persistence

```python
def build_dataset(trade_records: list[TradeRecord]
                  ) -> tuple["numpy.ndarray", "numpy.ndarray", list[str]]: ...
# From closed/simulated trades, build X (rows in FEATURE_ORDER from each record.features),
# y (labels, 1 if pnl>0 else 0), and feature_names() (== ml.features.FEATURE_ORDER).

def save_dataset(X, y, feature_names: list[str], path: str | Path) -> Path: ...
# Persist as Parquet (pyarrow) else CSV (per config.DATASET_FORMAT). Columns = feature_names + ['label'].

def load_dataset(path: str | Path) -> tuple["numpy.ndarray", "numpy.ndarray", list[str]]: ...
# Inverse of save_dataset. Returns (X, y, feature_names) with FEATURE_ORDER preserved.
```

## 9. `agentic_trader/ml/train_model.py` — fit the entry scorer (no leakage)

```python
def train(dataset_path: str | Path, out_dir: str | Path | None = None) -> dict: ...
# Load dataset, do a TEMPORAL split (train earlier, validate later — NO leakage), fit a
# LogisticRegression (sklearn; numpy logistic-regression fallback if absent), persist the
# model (joblib .pkl else JSON dump) to out_dir/ml_scorer.pkl and metrics to metrics.json.
# Returns a metrics dict: {n_train, n_val, auc, accuracy, val_winrate, baseline_winrate, ...}.
# out_dir defaults to config.MODELS_DIR.
```

## 10. `agentic_trader/backtest/engine.py` — t→t+1 simulator

```python
@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    metrics: dict            # {n_trades, win_rate, total_pnl, avg_return_pct, max_drawdown, ...}
    labeled_features: list[dict]   # decision-time features + label, one per ENTRY taken

def run_backtest(bars_by_symbol: dict[str, list[Bar]], params: StrategyParams,
                 limits: RiskLimits, scorer: "EntryScorer | None" = None) -> BacktestResult: ...
# Replay completed bars per symbol with STRICT t->t+1 fills (fill at bar t+1 OPEN), completed
# bars only, NO look-ahead. Apply §2 indicators, §3/§4 predicates, §5 ML gate, §8 risk caps
# (incl. cooldown/exposure/max_open across the simulated portfolio). Costs A11: $0 commission,
# SLIPPAGE_BPS adverse. Last-bar entries dropped; eod_flatten fills at close if no t+1 (A10).
# Records a TradeRecord per round-trip (label 1 if pnl>0) and its decision-time features.
```

## 11. `agentic_trader/backtest/optimize.py` — walk-forward param search

```python
def optimize(history: dict[str, list[Bar]], base_params: StrategyParams, limits: RiskLimits,
             n_iter: int = 50, folds: int = 3,
             scorer: "EntryScorer | None" = None) -> tuple[StrategyParams, dict]: ...
# Walk-forward sampled search over StrategyParams.TRAINABLE (entry_dist, vol_mult, max_hold,
# vwap_exit_band, stop_loss, ml_threshold). Split history into `folds` time blocks; for each
# sampled param set, run_backtest per fold's out-of-sample window, aggregate the objective
# (risk-adjusted return). Returns (best_params, report dict with per-fold metrics + leaderboard).
# Does NOT write files; caller persists via config.save_best_params.
```

## 12. `agentic_trader/orchestrator.py` — one full cycle

```python
@dataclass
class CycleReport:
    asof: datetime
    snapshots: list[Snapshot]
    proposals: list[Proposal]
    decisions: list[RiskDecision]
    orders: list[OrderResult]
    summaries: list[PositionSummary]
    exposure: float
    armed: bool
    errors: list[str]

def run_cycle(clients: dict, params: StrategyParams, limits: RiskLimits,
              scorer: "EntryScorer | None", *, armed: bool,
              settings: Settings | None = None,
              logger: JsonlLogger | None = None) -> CycleReport: ...
# One full pass (STRATEGY_RULES §10): RECONCILE broker positions+open orders (INV-6) ->
# MarketDataAgent snapshots (log data_snapshot) -> StrategyAgent.propose (exits first, then
# entries in UNIVERSE order; log strategy_proposal + ml_score) -> RiskAgent.review threading
# approved_buys (log risk_decision) -> ExecutionAgent.execute(armed=armed) (log order_request/
# response) -> PositionAgent.summarize (log position_summary). `clients` is config.make_clients()
# output. Every step is try/wrapped; failures are logged (log_error) and collected in errors.
```

## 13. `agentic_trader/loop.py` — live loop (market-hours aware)

```python
def run_loop(*, armed: bool, interval: float | None = None, once: bool = False,
             params: StrategyParams | None = None, limits: RiskLimits | None = None,
             scorer: "EntryScorer | None" = None,
             settings: Settings | None = None) -> None: ...
# Build clients/params/limits/scorer (defaults: load_strategy_params, RiskLimits(),
# EntryScorer.load()). Each iteration: reconcile + run_cycle(armed=armed). Market-hours aware
# (skip/sleep when the Alpaca clock is closed). `interval` defaults to Settings.loop_interval_sec.
# `once=True` runs a single cycle and returns. Graceful SIGINT (finish current cycle, then stop).
```

## 14. `agentic_trader/cli.py` — argparse entrypoint

```python
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: list[str] | None = None) -> int: ...
# Subcommands (all default to DRY-RUN; submission requires --arm AND a paper account):
#   run        [--arm]                         one cycle via run_cycle
#   loop       [--arm] [--interval N]          continuous run_loop (market-hours aware)
#   dry-run                                    explicit one cycle, never armed (armed=False)
#   backtest   [--start D] [--end D]           run_backtest over cached/backfilled bars
#   optimize   [--n-iter N] [--folds K]        optimize -> save_best_params(best)
#   backfill   [--start D] [--end D] [--symbols ...]   history.backfill
#   train      [--dataset PATH]                ml.train_model.train -> models/
#   positions                                  print reconciled positions + PositionSummary
#   account                                    print TradeAccount summary
# Returns process exit code (0 ok). --arm sets armed=True ONLY; INV-3 still refuses non-paper.
```

---

## Cross-cutting invariants every module must uphold
- **INV-1/2/3 (safety gate):** only ExecutionAgent submits; only when paper-env AND paper-client
  AND risk-approved AND armed. Default DRY-RUN. No real-money mode — refuse if ALPACA_PAPER != "true".
- **INV-4 (risk authority):** no order reaches Execution without a `RiskDecision.approved`. Strategy
  proposes; Risk disposes; every non-approval logs a reason.
- **INV-5 (logging):** every step emits a JSONL event via `JsonlLogger`. Required types:
  data_snapshot, strategy_proposal, ml_score, risk_decision, order_request, order_response,
  position_summary, error.
- **INV-6 (reconcile):** each cycle rebuilds position_state from broker positions + open orders
  before proposing. Local memory is a cache only.
- **No look-ahead:** indicators/features/predicates use only completed bars `[0..t]`. The label is
  the ONLY quantity allowed to use post-entry bars, and only offline.
- **Layout parity:** live + training use `ml.features.FEATURE_ORDER` / `to_row` — never re-derive feature order.
