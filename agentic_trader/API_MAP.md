# alpaca-py 0.43.4 — Verified API Map

Empirically verified on 2026-06-04 against alpaca-py **0.43.4**, Python **3.14.3**, using the
PAPER account in `/Users/pnle/Desktop/alpaca-cli/.env` (key id starts with `PK`).
Every signature/field below was confirmed via `inspect` + Pydantic `model_fields`, and the
market-data + trading calls were actually executed live (account cash=100000, IEX bars fetched
for AAPL).

Package location: `/Users/pnle/Library/Python/3.14/lib/python/site-packages/alpaca/`

---

## 1. Trading client

```python
from alpaca.trading.client import TradingClient

# All objects are alpaca.trading.models.* (Pydantic models). Fields are accessed as attributes.
client = TradingClient(api_key, secret_key, paper=True)
```

**Constructor (verified):**
```python
TradingClient(
    api_key: str | None = None,
    secret_key: str | None = None,
    oauth_token: str | None = None,
    paper: bool = True,          # default True
    raw_data: bool = False,
    url_override: str | None = None,
) -> None
```

**Methods (verified signatures + EXACT param names):**
```python
client.get_account() -> TradeAccount
client.get_all_positions() -> List[Position]
client.get_open_position(symbol_or_asset_id: UUID | str) -> Position   # NOTE: param is symbol_or_asset_id, not "symbol"
client.get_orders(filter: GetOrdersRequest | None = None) -> List[Order]   # kwarg name is exactly `filter`
client.submit_order(order_data: OrderRequest) -> Order                 # kwarg name is exactly `order_data`
client.close_position(symbol_or_asset_id: UUID | str,
                      close_options: ClosePositionRequest | None = None) -> Order
client.close_all_positions(cancel_orders: bool | None = None) -> List[ClosePositionResponse]
client.get_clock() -> Clock
```

Live-verified: `get_account()`, `get_clock()`, `get_all_positions()` all return real data with the
paper keys. (`submit_order` was built but NOT executed — see snippet in §6.)

---

## 2. Order request objects

```python
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
    OrderRequest, ClosePositionRequest,
)
```

### MarketOrderRequest — CONFIRMED accepts BOTH `notional` AND `qty`, plus `time_in_force`
Fields (Pydantic, all optional unless noted):
```
symbol          : str | None = None         # required in practice
qty             : float | None = None        # mutually exclusive with notional
notional        : float | None = None        # dollar amount; fractional/notional orders
side            : OrderSide | None = None     # required in practice
type            : OrderType                   # AUTO-SET to OrderType.MARKET — do NOT pass it
time_in_force   : TimeInForce                 # required (e.g. TimeInForce.DAY)
order_class     : OrderClass | None = None
extended_hours  : bool | None = None
client_order_id : str | None = None
take_profit     : TakeProfitRequest | None = None
stop_loss       : StopLossRequest | None = None
position_intent : PositionIntent | None = None
```
Verified: `MarketOrderRequest(symbol='AAPL', notional=10, side=OrderSide.BUY,
time_in_force=TimeInForce.DAY)` builds cleanly; `.type` auto-resolves to `OrderType.MARKET`.
Note: pass `qty` XOR `notional`, never both. `qty`/`notional` are coerced to `float`.

### GetOrdersRequest
Fields:
```
status     : QueryOrderStatus | None = None   # OPEN / CLOSED / ALL
limit      : int | None = None
after      : datetime | None = None
until      : datetime | None = None
direction  : Sort | None = None               # alpaca.common.enums.Sort
nested     : bool | None = None
side       : OrderSide | None = None
symbols    : List[str] | None = None
```

`OrderRequest` is the abstract base of `MarketOrderRequest`/`LimitOrderRequest`/etc. — same field
set as MarketOrderRequest above. `submit_order(order_data=...)` accepts any concrete subclass.

---

## 3. Trading enums

```python
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus, OrderType
```

| Enum | Members (name = value) |
|------|------------------------|
| `OrderSide` | `BUY='buy'`, `SELL='sell'` |
| `TimeInForce` | `DAY='day'`, `GTC='gtc'`, `OPG='opg'`, `CLS='cls'`, `IOC='ioc'`, `FOK='fok'` |
| `QueryOrderStatus` | `OPEN='open'`, `CLOSED='closed'`, `ALL='all'` |
| `OrderType` | `MARKET='market'`, `LIMIT='limit'`, `STOP='stop'`, `STOP_LIMIT='stop_limit'`, `TRAILING_STOP='trailing_stop'` |
| `OrderStatus` | `NEW`, `PARTIALLY_FILLED`, `FILLED`, `DONE_FOR_DAY`, `CANCELED`, `EXPIRED`, `REPLACED`, `PENDING_CANCEL`, `PENDING_REPLACE`, `PENDING_REVIEW`, `ACCEPTED`, `PENDING_NEW`, `ACCEPTED_FOR_BIDDING`, `STOPPED`, `REJECTED`, `SUSPENDED`, `CALCULATED`, `HELD` (values are lowercase of name) |

GOTCHA: `QueryOrderStatus` (used in `GetOrdersRequest.status`) is a DIFFERENT enum from
`OrderStatus` (returned on `Order.status`). Don't confuse them.

---

## 4. Returned object field names (attributes)

All numeric fields on TradeAccount/Position/Order are returned as **strings** (parse with
`float(...)` / `Decimal(...)`). Live AAPL bar values verified.

### TradeAccount  (`client.get_account()`)
Key fields (all `str` unless noted):
```
id (UUID), account_number (str), status (AccountStatus enum), currency,
cash, buying_power, regt_buying_power, daytrading_buying_power,
non_marginable_buying_power, options_buying_power,
portfolio_value, equity, last_equity,
long_market_value, short_market_value,
initial_margin, maintenance_margin, last_maintenance_margin, sma,
accrued_fees, pending_transfer_in, pending_transfer_out, multiplier,
pattern_day_trader (bool), trading_blocked (bool), transfers_blocked (bool),
account_blocked (bool), trade_suspended_by_user (bool), shorting_enabled (bool),
daytrade_count (int), created_at (datetime),
options_approved_level (int), options_trading_level (int)
```
Verified live: `cash='100000'`, `buying_power='400000'`, `portfolio_value='100000'`,
`equity='100000'`, `status=AccountStatus.ACTIVE`.

### Position  (`client.get_all_positions()` / `get_open_position()`)
```
asset_id (UUID), symbol (str), exchange (AssetExchange), asset_class (AssetClass),
avg_entry_price (str), qty (str), qty_available (str), side (PositionSide enum),
market_value (str), cost_basis (str),
unrealized_pl (str), unrealized_plpc (str),
unrealized_intraday_pl (str), unrealized_intraday_plpc (str),
current_price (str), lastday_price (str), change_today (str),
asset_marginable (bool)
```

### Order  (returned by `submit_order`, `get_orders`, `close_position`)
```
id (UUID), client_order_id (str),
created_at, updated_at, submitted_at (datetime),
filled_at, expired_at, expires_at, canceled_at, failed_at, replaced_at (datetime|None),
asset_id (UUID|None), symbol (str|None), asset_class,
notional (str|None), qty (str|float|None), filled_qty (str|float|None),
filled_avg_price (str|float|None),
order_class (OrderClass), order_type (OrderType|None), type (OrderType|None),
side (OrderSide|None), time_in_force (TimeInForce),
limit_price (str|float|None), stop_price (str|float|None),
status (OrderStatus), extended_hours (bool),
legs (List[Order]|None), trail_percent, trail_price, hwm,
position_intent (PositionIntent|None)
```
GOTCHA: Order has BOTH `type` and `order_type` (both `OrderType`) — they carry the same value.

### Clock  (`client.get_clock()`)
```
timestamp (datetime), is_open (bool), next_open (datetime), next_close (datetime)
```
Verified live: `is_open=True`.

---

## 5. Market data (historical bars + quotes)

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
```

**Constructor (verified):** keys ARE required.
```python
StockHistoricalDataClient(
    api_key: str | None = None,
    secret_key: str | None = None,
    oauth_token: str | None = None,
    use_basic_auth: bool = False,
    raw_data: bool = False,
    url_override: str | None = None,
    sandbox: bool = False,
) -> None
```
Verified: constructing with NO auth raises `ValueError: You must supply a method of authentication`.
The paper keys (PK...) work for IEX historical bars — confirmed by fetching real AAPL 1-min bars.

**Methods (verified):**
```python
client.get_stock_bars(request_params: StockBarsRequest) -> BarSet
client.get_stock_latest_quote(request_params: StockLatestQuoteRequest) -> Dict[str, Quote]
client.get_stock_latest_trade(request_params: StockLatestTradeRequest) -> Dict[str, Trade]
```
GOTCHA: the single positional arg is named `request_params` (not `filter`/`request`).

### StockBarsRequest fields
```
symbol_or_symbols : str | List[str]   # REQUIRED (no default)
timeframe         : TimeFrame          # REQUIRED (no default)
start             : datetime | None = None
end               : datetime | None = None
limit             : int | None = None
feed              : DataFeed | None = None     # None => server defaults to IEX (free) — verified
adjustment        : Adjustment | None = None
sort              : Sort | None = None
currency          : SupportedCurrencies | None = None
asof              : str | None = None
```

### StockLatestQuoteRequest fields
```
symbol_or_symbols : str | List[str]   # REQUIRED
feed              : DataFeed | None = None
currency          : SupportedCurrencies | None = None
```

### TimeFrame / TimeFrameUnit  (how to express 1-Minute)
```python
TimeFrame(amount: int, unit: TimeFrameUnit)
# 1-minute bars, any equivalent works:
TimeFrame(1, TimeFrameUnit.Minute)   # -> str "1Min"
TimeFrame.Minute                      # classmethod shorthand -> "1Min"
```
`TimeFrameUnit`: `Minute='Min'`, `Hour='Hour'`, `Day='Day'`, `Week='Week'`, `Month='Month'`.
Shorthands: `TimeFrame.Minute` (1Min), `TimeFrame.Hour` (1Hour), `TimeFrame.Day` (1Day).
GOTCHA: the Minute unit VALUE is the abbreviation `'Min'`, not `'Minute'`.

### DataFeed  (IEX is the free default)
```
IEX='iex'  (free, default when feed=None), SIP='sip', DELAYED_SIP='delayed_sip',
OTC='otc', BOATS='boats', OVERNIGHT='overnight'
```

### get_stock_bars return shape — BOTH `.data` and `.df`
`get_stock_bars(...)` returns a `BarSet` exposing two accessors (both verified):

1. **`.data`** — `Dict[str, List[Bar]]` keyed by symbol:
   ```python
   bars = barset.data["AAPL"]      # List[Bar]
   b = bars[0]
   b.timestamp  b.open  b.high  b.low  b.close  b.volume  b.vwap  b.trade_count
   ```
   Per-`Bar` fields (verified): `symbol (str)`, `timestamp (datetime, tz-aware UTC)`,
   `open (float)`, `high (float)`, `low (float)`, `close (float)`, `volume (float)`,
   `trade_count (float|None)`, `vwap (float|None)`.

2. **`.df`** — pandas DataFrame with a **MultiIndex** `(symbol, timestamp)` and columns
   exactly: `['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']`.
   (Note: no `symbol`/`timestamp` columns — they are the index. Select one symbol with
   `barset.df.loc["AAPL"]`.)

---

## 6. Copy-pasteable verified snippets

### Load the existing paper keys from .env (no extra deps)
```python
from pathlib import Path

def load_env(path="/Users/pnle/Desktop/alpaca-cli/.env"):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

_env = load_env()
API_KEY = _env["ALPACA_API_KEY_ID"]      # starts with "PK" (paper)
API_SECRET = _env["ALPACA_API_SECRET"]
PAPER = _env.get("ALPACA_PAPER", "true").lower() == "true"
```

### Fetch recent 1-minute AAPL bars (IEX free feed) — VERIFIED WORKING
```python
import datetime as dt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)  # stay clear of feed delay edge
start = end - dt.timedelta(days=5)

req = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
    start=start,
    end=end,
    feed=DataFeed.IEX,   # or omit entirely; server defaults to IEX on the free plan
    limit=5,
)
barset = data_client.get_stock_bars(req)

for b in barset.data["AAPL"]:
    print(b.timestamp, b.open, b.high, b.low, b.close, b.volume, b.vwap, b.trade_count)

df = barset.df  # MultiIndex (symbol, timestamp); cols open/high/low/close/volume/trade_count/vwap
```

### Submit a $10 notional market BUY — SHOWN, NOT EXECUTED
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

order_req = MarketOrderRequest(
    symbol="AAPL",
    notional=10,                       # dollar amount; use qty=<shares> instead for share-based
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    # do NOT pass type=...; MarketOrderRequest sets OrderType.MARKET automatically
)
# order = trade_client.submit_order(order_data=order_req)   # <-- left commented; not executed
# print(order.id, order.status, order.symbol, order.notional, order.filled_avg_price)
```

### List recent orders / close positions
```python
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

orders = trade_client.get_orders(
    filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=20, symbols=["AAPL"])
)

# trade_client.close_position("AAPL")          # closes the whole AAPL position (returns Order)
# trade_client.close_all_positions(cancel_orders=True)
```

---

## 7. Surprises / gotchas summary
- **`submit_order` kwarg is `order_data=`** and **`get_orders` kwarg is `filter=`** (not `request`).
- **`get_open_position(symbol_or_asset_id)`** — the param is `symbol_or_asset_id`, not `symbol`.
- **`get_stock_bars(request_params=...)`** — positional arg named `request_params`.
- **Account/Position/Order numeric fields are STRINGS** — wrap in `float()`/`Decimal()`.
- **`MarketOrderRequest` sets `type=OrderType.MARKET` automatically** — passing it is redundant; `time_in_force` IS required.
- **`Order` has both `type` and `order_type`** (same value).
- **`TimeFrameUnit.Minute` value is `'Min'`**, not `'Minute'`.
- **`feed=None` defaults to IEX** server-side (free). The paper PK keys authenticate IEX historical bars fine; SIP requires a paid subscription.
- **`BarSet` exposes BOTH `.data` (dict of List[Bar]) and `.df` (pandas, MultiIndex symbol/timestamp)**.
- **`StockHistoricalDataClient` requires auth** — no-arg construction raises `ValueError`.
- **`QueryOrderStatus` (request) != `OrderStatus` (response)** — two distinct enums.
