# STRATEGY_RULES.md — Intraday VWAP Mean-Reversion (PAPER ONLY)

> Machine-readable restatement of the authoritative strategy. This document is the
> single source of truth for implementers. Every formula, window, predicate, cap,
> and timing rule below is normative. Where the prose strategy left a gap, this
> document RESOLVES it with a concrete decision (see ASSUMPTIONS & OPEN QUESTIONS).
>
> SCOPE: PAPER trading only. There is NO real-money mode. Deterministic decisioning
> (no LLM in the trade loop). The ML scorer is a learned, deterministic gate.

---

## 0. CONSTANTS & DEFAULTS

```
UNIVERSE = ["SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMD","TSLA","META","AMZN"]   # 10 symbols, fixed order

# Bars
BAR_TIMEFRAME      = 1 minute
SESSION_TZ         = "America/New_York"
SESSION_OPEN       = 09:30:00   # inclusive (first regular bar opens here)
SESSION_CLOSE      = 16:00:00   # exclusive for bar START (last regular bar starts 15:59)
REGULAR_HOURS_ONLY = true       # discard pre-market and after-hours bars
COMPLETED_ONLY     = true       # never use a bar whose close timestamp is in the future

# Trainable params (the optimizer searches these). Values below are DEFAULTS.
entry_dist      = 0.005   # 0.5% : enter when price is >= this fraction BELOW vwap
vol_mult        = 1.2     # volume_ratio must be >= this
max_hold        = 15      # minutes : force-exit when held longer than this
vwap_exit_band  = 0.001   # 0.1% : exit when price returns to within this of vwap
stop_loss       = 0.005   # 0.5% : exit when unrealized loss reaches this fraction
ml_threshold    = 0.0     # ML gate disabled at 0.0 (passthrough). Range [0.0, 1.0]

# Fixed sizing / risk
NOTIONAL_PER_ENTRY  = 100.0   # USD notional per BUY
MAX_OPEN_POSITIONS  = 4
MAX_TOTAL_EXPOSURE  = 500.0   # USD : open MV + pending-entry notional must stay <= this
COOLDOWN_MINUTES    = 10      # at most 1 new ENTRY per symbol per this window

# ML feature window
RECENT_RETURN_K     = 5       # bars used for recent_return (RESOLVED, see A7)
ROLLING_VOL_WINDOW  = 20      # bars for rolling20_avg_vol (EXCLUDES current bar)
```

---

## 1. DATA MODEL

A **session** is one regular-hours trading day for one symbol. Indicators are computed
from the ordered list of COMPLETED regular-hours 1-minute bars from `SESSION_OPEN` up to
and including the **decision bar** (minute `t`).

```
Bar = {
  symbol:   str,
  start:    datetime  # America/New_York, the minute the bar covers, e.g. 09:31
  open:     float,
  high:     float,
  low:      float,
  close:    float,
  volume:   float,     # shares
  is_complete: bool     # true iff now() >= start + 1 minute
}

SessionBars(symbol) = [ Bar ...]  # filtered to:
    bar.start.date()   == today_in_NY
    SESSION_OPEN <= bar.start.time() < SESSION_CLOSE
    bar.is_complete == true
  sorted ascending by bar.start
```

The **decision bar** is the LAST element of `SessionBars(symbol)` (the most recent
completed bar). All indicators are evaluated on this bar. See Section 6 for timing.

---

## 2. PER-SYMBOL INDICATORS (exact windowing)

Let `B = SessionBars(symbol)`, `n = len(B)`, and let the decision bar be `B[n-1]`.

```
# Typical price per bar
typical_price(b) = (b.high + b.low + b.close) / 3.0

# Session VWAP : cumulative over ALL completed session bars up to & including decision bar
num   = sum( typical_price(b) * b.volume  for b in B )      # i = 0 .. n-1
den   = sum( b.volume                      for b in B )
session_vwap = num / den                                     # REQUIRE den > 0 (see A4)

# Last price
last_price = B[n-1].close

# Distance from VWAP (signed; negative = below VWAP)
dist_from_vwap = (last_price - session_vwap) / session_vwap

# Current bar volume
current_volume = B[n-1].volume

# Rolling 20-bar average volume — EXCLUDES the current (decision) bar.
# Uses the 20 completed bars IMMEDIATELY PRECEDING B[n-1], i.e. indices [n-21 .. n-2].
window = B[ max(0, n-21) : n-1 ]            # length up to 20, never includes B[n-1]
rolling20_avg_vol = mean( b.volume for b in window )         # see A1 for n<21 handling

# Volume ratio
volume_ratio = current_volume / rolling20_avg_vol           # see A1, A4

# Position state (reconciled from broker each cycle; see Section 7)
position_state = flat
              |  long(qty, avg_entry_price, entry_time)
```

**Minimum-bars guard (RESOLVED, A1):** an entry is only EVALUATED when `n >= 21`
(i.e. at least 20 preceding bars exist so `rolling20_avg_vol` is a true 20-bar mean).
Before that, the symbol is skipped for ENTRY this cycle (exits are always evaluated
if a position exists — exits never require the rolling window).

---

## 3. ENTRY PREDICATE

A symbol generates a BUY **proposal** (notional `$100`, MARKET) iff ALL hold:

```
function entry_signal(symbol) -> bool:
    if position_state(symbol) != flat:            return false   # must be flat
    B = SessionBars(symbol); n = len(B)
    if n < 21:                                     return false   # min-bars guard (A1)
    if den_volume(B) <= 0:                         return false   # degenerate VWAP (A4)
    if rolling20_avg_vol <= 0:                      return false   # degenerate ratio (A4)

    cond_below_vwap = (dist_from_vwap <= -entry_dist)     # >= entry_dist BELOW vwap
    cond_volume     = (volume_ratio   >=  vol_mult)
    return cond_below_vwap AND cond_volume
```

If `entry_signal` is true, the candidate then passes through the **ML gate** (Section 5)
before becoming a Risk proposal. Rule-pass AND ml-pass are BOTH required.

---

## 4. EXIT PREDICATE

A held position generates a SELL **proposal** (full qty, MARKET) iff ANY hold:

```
function exit_signal(symbol) -> (bool, reason):
    if position_state(symbol) != long:            return (false, None)
    B = SessionBars(symbol); n = len(B)
    if n < 1:                                       return (false, None)

    holding_minutes  = (decision_bar.start - entry_time).total_seconds() / 60.0
    unrealized_plpc  = (last_price - avg_entry_price) / avg_entry_price

    if dist_from_vwap >= -vwap_exit_band:    return (true, "vwap_revert")   # back to VWAP
    if holding_minutes > max_hold:           return (true, "max_hold")      # strict >
    if unrealized_plpc <= -stop_loss:        return (true, "stop_loss")     # loss cap
    return (false, None)
```

Exit checks are evaluated EVERY cycle for every open position and are NEVER gated by the
ML scorer (ML gates entries only). Exit priority for logging when multiple fire is the
order listed above: `vwap_revert` > `max_hold` > `stop_loss`.

**End-of-day flat (RESOLVED, A9):** on the decision bar starting at `15:55` or later
(`>= 15:55:00 NY`), force-exit any open position regardless of the above (`reason="eod_flatten"`),
so no position is carried overnight. No NEW entries are proposed on/after `15:55`.

---

## 5. ML ENTRY SCORER (learned gate on entries)

The scorer is a deterministic function `p_win = model.predict_proba(features)` in `[0,1]`.

### 5.1 Feature vector (decision-time only — NO future data)

Built at the candidate entry on the decision bar. Every input is known at minute `t`:

```
features(symbol, decision_bar) = {
  dist_from_vwap,                       # signed, from Section 2
  volume_ratio,                         # from Section 2
  log_rolling_vol  = log1p(rolling20_avg_vol),
  minute_of_session= minutes since 09:30 of decision_bar.start   # 0..389
  recent_return    = (B[n-1].close / B[n-1-K].close) - 1.0,  K=RECENT_RETURN_K=5  (A7)
  bar_range_pct    = (B[n-1].high - B[n-1].low) / B[n-1].close,
  session_progress = minute_of_session / 389.0,                  # in [0,1]
  symbol_id        = index of symbol in UNIVERSE (0..9)          # categorical (A8)
}
```

`recent_return` requires `n-1-K >= 0` (i.e. `n >= K+1`). Since the entry min-bars guard
already requires `n >= 21` and `K=5`, this is always satisfied for entry candidates.

### 5.2 Label

```
label = 1 if the realized trade was profitable NET OF COSTS, else 0
```

"Profitable net of costs" is defined in A11: `realized_pnl_after_costs > 0`. The realized
exit comes from the backtest simulator (offline) or the live-logged outcome (online),
recorded when the position closes.

### 5.3 Gate

```
function ml_gate(features) -> bool:
    if no model is trained yet:   p_win = 1.0          # passthrough -> rule-only
    else:                         p_win = model.predict_proba(features)
    return p_win >= ml_threshold
```

With `ml_threshold = 0.0` (default) the gate is effectively disabled even when a model
exists, because any `p_win in [0,1] >= 0.0`. The optimizer raises `ml_threshold` only
after a model is trained and validated.

---

## 6. TIMING DISCIPLINE (t -> t+1, NO LOOK-AHEAD)

```
DECIDE on the last COMPLETED bar = minute t (the decision bar).
ACT/FILL at t+1:
  - BACKTEST: fill at the OPEN of bar t+1 (the bar starting at t+1). If t is the last
              bar of the session, the order is DROPPED (no t+1 to fill against) — except
              eod_flatten exits, which fill at the t+1 open when one exists, else at the
              decision bar close with a logged note (A10).
  - LIVE:     bar t is already closed when we observe it, so submit IMMEDIATELY as a
              MARKET order; the broker fills at the prevailing (>= t+1) price.
NEVER read, index, or aggregate any bar with start > decision_bar.start.
NEVER use a bar whose is_complete == false.
```

All indicators, features, and predicates are functions of bars `[0 .. n-1]` only. The
label (Section 5.2) is the ONLY quantity allowed to depend on future bars, and it is used
ONLY for offline training — never fed back into same-bar decisioning.

---

## 7. STATE RECONCILIATION (every cycle)

```
Each cycle, BEFORE proposing:
  broker_positions   = TradingClient.get_all_positions()      # source of truth
  broker_open_orders = TradingClient.get_orders(status=OPEN)
  rebuild position_state(symbol) for all symbols from broker_positions
     long(qty, avg_entry_price, entry_time)  where:
        qty               = position.qty
        avg_entry_price   = position.avg_entry_price
        entry_time        = from local trade log if present, else position open time (A6)
  Do NOT trust local memory over the broker. Local memory is a cache only.
```

If a position exists at the broker that local memory does not know about, ADOPT it
(treat as `long` with `entry_time` = now if unknown) and log a reconciliation warning.

---

## 8. RISK CAPS (Risk agent has VETO over EVERY proposal)

The Risk agent receives each proposal and returns `APPROVE | REDUCE | VETO` with a
logged reason for any non-approval. Only APPROVED (possibly REDUCED) orders reach Execution.

```
function risk_review(proposal, broker_state) -> decision:
    # 8.1 Paper + arm + paper-account preconditions are checked in EXECUTION (Section 9),
    #     but Risk also refuses to approve anything if ALPACA_PAPER != "true".
    if env.ALPACA_PAPER != "true":          return VETO("not a paper account")

    if proposal.side == SELL:               return APPROVE   # exits are never blocked (A12)

    # ----- BUY proposals only below -----
    open_count = len(broker_positions)
    if open_count >= MAX_OPEN_POSITIONS:    return VETO("max_open_positions")

    # cooldown: at most 1 new ENTRY per symbol per COOLDOWN_MINUTES
    last_entry = last_entry_time(proposal.symbol)            # from trade log (A5)
    if last_entry is not None and (now - last_entry) < COOLDOWN_MINUTES:
                                            return VETO("cooldown")

    # exposure: open MV + notional of pending new entries must stay <= MAX_TOTAL_EXPOSURE
    open_mv        = sum(p.market_value for p in broker_positions)
    pending_notional = sum(o.notional for o in this_cycle_approved_buys)  # incl. this one
    projected      = open_mv + pending_notional + proposal.notional
    if projected > MAX_TOTAL_EXPOSURE:
        room = MAX_TOTAL_EXPOSURE - open_mv - pending_notional
        if room <= 0:                       return VETO("max_total_exposure")
        else:                               return REDUCE(notional=room)   # (A13)

    return APPROVE
```

Notes:
- `MAX_OPEN_POSITIONS` and exposure are evaluated against RECONCILED broker state plus
  any buys already approved earlier in the same cycle (so two simultaneous proposals
  cannot both slip past a single open slot).
- Cooldown is tracked from the persisted trade log so it survives restarts (A5).

---

## 9. HARD SAFETY INVARIANTS (enforced in code)

```
function execute(order):
    # INV-1 + INV-2 + INV-3: submit ONLY when all true; default is DRY-RUN.
    is_paper_client = (TradingClient was constructed with paper=True)
    is_paper_env    = (env.ALPACA_PAPER == "true")
    is_armed        = (config.live_paper == true) OR (CLI flag --arm present)
    is_approved     = (order.risk_decision in {APPROVE, REDUCE})

    if not is_paper_env:
        log_error("ALPACA_PAPER != 'true' — refusing to submit"); return DRY_RUN
    if not (is_paper_client and is_armed and is_approved):
        log("DRY-RUN: would submit", order); return DRY_RUN     # submits NOTHING

    resp = TradingClient.submit_order(MarketOrderRequest(...))   # paper only
    log("order_response", resp); return resp
```

- **INV-1**: submit iff `paper client` AND `risk-approved` AND `armed`.
- **INV-2**: default mode is DRY-RUN; arming requires explicit `--arm` / `live_paper=true`.
- **INV-3**: NO real-money mode exists. `ALPACA_PAPER != "true"` => refuse + clear error.
- **INV-4**: no order bypasses Risk — Strategy proposes, Risk approves/reduces/vetoes
  (reason logged for every rejection), only approved orders reach Execution.
- **INV-5**: every event logged to `/Users/pnle/Desktop/alpaca-cli/logs/*.jsonl`
  (one JSON object per line, UTC-timestamped): data snapshot, strategy proposal,
  risk decision, order request, order response, position summary, errors.
- **INV-6**: each cycle reconciles positions + open orders from Alpaca (Section 7).

Orders are SIMPLE MARKET orders only. NO bracket / OCO / stop / limit orders.

---

## 10. ONE CYCLE (pseudocode)

```
function run_cycle(config):
    broker = reconcile()                          # INV-6, Section 7
    snapshot = {}
    for symbol in UNIVERSE:
        B = SessionBars(symbol)                   # completed regular bars only
        ind = compute_indicators(B)               # Section 2
        snapshot[symbol] = ind
        log("data_snapshot", symbol, ind)         # INV-5

    proposals = []
    # exits first (free up slots / capital before entries)
    for symbol in symbols_with_position(broker):
        ok, reason = exit_signal(symbol)
        if ok: proposals.append(Sell(symbol, qty=full, reason=reason))
    # entries
    if now_NY < 15:55:
        for symbol in UNIVERSE:
            if entry_signal(symbol):
                feats = features(symbol, decision_bar)
                if ml_gate(feats):
                    proposals.append(Buy(symbol, notional=100, p_win=..., features=feats))

    approved = []
    for p in proposals:
        log("strategy_proposal", p)               # INV-5
        d = risk_review(p, broker, approved)      # Section 8, INV-4
        log("risk_decision", p, d)                # INV-5 (reason on non-approve)
        if d in {APPROVE, REDUCE}: approved.append(p.with(d))

    for p in approved:
        log("order_request", p)                   # INV-5
        resp = execute(p, config)                 # Section 9, INV-1/2/3
        log("order_response", resp)               # INV-5

    log("position_summary", reconcile())          # INV-5/6
```

---

## ASSUMPTIONS & OPEN QUESTIONS (each RESOLVED)

Every item below was ambiguous in the prose strategy. Each has a concrete, normative
decision so implementers are never blocked.

- **A1 — rolling20 when fewer than 20 preceding bars exist.**
  Decision: do NOT evaluate ENTRY until `n >= 21` (>= 20 strictly-preceding completed
  bars). `rolling20_avg_vol` is the mean of exactly those 20 bars `[n-21 : n-1]`,
  always excluding the current bar. Exits do not require the window.

- **A2 — does the current bar count in session VWAP?**
  Decision: YES. Session VWAP is cumulative over ALL completed session bars including
  the decision bar `B[n-1]`. (Only `rolling20_avg_vol` excludes the current bar.)

- **A3 — session VWAP daily reset.**
  Decision: VWAP resets at `09:30 NY` each trading day. It is computed only from that
  day's regular-hours completed bars; pre-market/after-hours/prior days never contribute.

- **A4 — division by zero / degenerate denominators.**
  Decision: if `sum(volume) <= 0` (VWAP) or `rolling20_avg_vol <= 0` (ratio), the symbol
  is skipped for ENTRY this cycle and a warning is logged. Exits use only price vs VWAP;
  if VWAP is undefined no exit_signal `vwap_revert` fires (other exit reasons still apply).

- **A5 — cooldown tracking across restarts.**
  Decision: cooldown is derived from the persisted JSONL trade log (last `order_request`
  with side=BUY per symbol), NOT from in-memory state, so it survives process restarts.
  `last_entry_time(symbol)` = max timestamp of submitted BUYs for that symbol.

- **A6 — entry_time source after reconcile.**
  Decision: prefer the local trade log's recorded fill time; if unknown (e.g. position
  adopted from broker), use the Alpaca position's reported open time, else `now()`, and
  log a reconciliation warning. `holding_minutes` uses this `entry_time`.

- **A7 — recent_return lookback k.**
  Decision: `RECENT_RETURN_K = 5` bars: `recent_return = close[t]/close[t-5] - 1`. Fixed
  (not part of the trainable set). Always defined for entry candidates since `n >= 21`.

- **A8 — symbol identifier encoding for ML.**
  Decision: integer `symbol_id = UNIVERSE.index(symbol)` (0..9), treated as a categorical
  feature. Implementations using linear/SVM models must one-hot it; tree/GBM models may
  consume the integer directly. Recorded as the raw integer in feature logs.

- **A9 — end-of-day handling.**
  Decision: no NEW entries proposed on/after `15:55 NY`; any open position is force-exited
  (`eod_flatten`) on the first decision bar at/after `15:55`. No overnight positions.

- **A10 — last-bar order with no t+1 (backtest).**
  Decision: an entry proposed on the final session bar is DROPPED (no fill bar). An
  `eod_flatten` exit on the final bar fills at the decision bar's CLOSE with a logged
  `note="no_t+1_filled_at_close"` so positions are not stranded in the sim.

- **A11 — $100 notional -> shares, and "profitable" net of costs.**
  Decision (live): submit a fractional notional MARKET order (`notional=100`) via Alpaca;
  Alpaca handles fractional shares. Decision (backtest): `shares = 100 / fill_price`
  (fractional allowed). Costs: commission `$0` (Alpaca), slippage = `SLIPPAGE_BPS = 1`
  basis point applied ADVERSELY to each fill (buy fills `*(1+0.0001)`, sell `*(1-0.0001)`).
  `realized_pnl_after_costs = (sell_fill - buy_fill) * shares`. ML `label = 1` iff
  `realized_pnl_after_costs > 0`, else 0. (Costs are explicit so backtest and live agree.)

- **A12 — does Risk ever block an exit?**
  Decision: NO. SELL/exit proposals are always APPROVED by Risk (risk caps only constrain
  new exposure). Exits still go through Risk for logging/audit, but are never vetoed.

- **A13 — exposure cap that partially fits.**
  Decision: REDUCE the BUY notional to the remaining room (`MAX_TOTAL_EXPOSURE - open_mv -
  pending`) when room > 0; VETO when room <= 0. Note Alpaca enforces a minimum notional
  (~$1); if `room < MIN_NOTIONAL` treat as VETO("exposure_room_below_min").

- **A14 — multiple proposals competing for one open slot / capital (same cycle).**
  Decision: process exits first, then entries in UNIVERSE order; each approval decrements
  available slots/room within the cycle, so a single free slot or dollar of room can be
  consumed by only one proposal.

- **A15 — bar source / feed.**
  Decision: use Alpaca market-data 1-minute bars (IEX feed on the free/paper tier),
  `adjustment=raw`, filtered to regular hours in `America/New_York`. Completed-only is
  enforced by dropping any bar whose `start + 1min > now()`.

- **A16 — clock / "now".**
  Decision: all timing uses exchange wall-clock in `America/New_York`; logs are stamped
  in UTC ISO-8601. A bar is complete iff `now_utc >= bar.start_utc + 60s`.

- **A17 — what if the broker shows an OPEN order for a symbol we want to act on.**
  Decision: if a symbol has a pending OPEN order at the broker (from a prior cycle), skip
  proposing a NEW order for that symbol this cycle and log `skip_reason="open_order_pending"`
  (prevents duplicate/stacked orders; reconcile picks it up next cycle).
