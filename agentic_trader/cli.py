"""cli.py — argparse entrypoint for the agentic VWAP paper trader.

Subcommands (ALL default to DRY-RUN; a real paper submission requires BOTH `--arm` AND a
paper account — INV-1/2/3):

    account                              print the TradeAccount summary
    positions                            reconciled positions + PositionSummary
    run        [--arm]                   ONE cycle via orchestrator.run_cycle
    dry-run                              ONE cycle, NEVER armed (armed=False)
    loop       [--arm] [--interval N] [--once]   continuous, market-hours-aware run_loop
    backtest   [--symbols ...] [--start D] [--end D]   run_backtest over cached/backfilled bars
    optimize   [--n-iter N] [--folds K]  walk-forward search -> save_best_params(best)
    backfill   [--symbols ...] [--start D] [--end D]   history.backfill into data/cache/
    dataset    [--symbols ...] [--out PATH]   backtest -> labeled ML dataset (parquet/csv)
    train      [--dataset PATH]          ml.train_model.train -> models/

`--arm` ONLY sets armed=True; INV-3 still refuses any submission when ALPACA_PAPER != true.
Run as either `python3 -m agentic_trader.cli ...` or via the console entry `agentic-trader`.
Returns a process exit code (0 = ok).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import config
from .config import (
    RiskLimits,
    Settings,
    ensure_dirs,
    is_paper_env,
    load_strategy_params,
    make_clients,
)
from .logging_util import JsonlLogger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_date(text: str) -> _dt.datetime:
    """Parse a 'YYYY-MM-DD' (or ISO) date into a tz-aware UTC datetime at midnight NY.

    Accepts a bare date or a full ISO timestamp. Bare dates are anchored to 00:00 in the
    session timezone (America/New_York) so backfill/backtest windows align to sessions.
    """
    text = text.strip()
    try:
        if "T" in text or " " in text:
            dt = _dt.datetime.fromisoformat(text)
        else:
            d = _dt.date.fromisoformat(text)
            dt = _dt.datetime.combine(d, _dt.time(0, 0), tzinfo=ZoneInfo(config.SESSION_TZ))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date '{text}' (use YYYY-MM-DD): {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(config.SESSION_TZ))
    return dt.astimezone(_dt.timezone.utc)


def _default_window(days: int = 5) -> tuple[_dt.datetime, _dt.datetime]:
    """A sensible default [start, end] window: the last `days` days up to now (UTC)."""
    end = _dt.datetime.now(_dt.timezone.utc)
    start = end - _dt.timedelta(days=days)
    return start, end


def _load_scorer() -> Any:
    """Load the EntryScorer (passthrough if no model). Lazy import keeps CLI import-light."""
    from .ml.scorer import EntryScorer

    return EntryScorer.load()


def _make_clients_or_exit(settings: Settings, logger: JsonlLogger) -> Optional[Dict[str, Any]]:
    """Build Alpaca clients; print a clear error and return None on failure."""
    try:
        return make_clients(settings.env_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not build Alpaca clients: {exc}", file=sys.stderr)
        logger.log_error("cli.make_clients", "make_clients failed", exc)
        return None


def _resolve_symbols(args: argparse.Namespace, settings: Settings) -> List[str]:
    """Resolve --symbols (if any) to a list; default to the full universe."""
    syms = getattr(args, "symbols", None)
    return list(syms) if syms else list(settings.universe)


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_account(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    clients = _make_clients_or_exit(settings, logger)
    if clients is None:
        return 1
    try:
        a = clients["trading"].get_account()
    except Exception as exc:  # noqa: BLE001
        print(f"error: get_account failed: {exc}", file=sys.stderr)
        return 1
    print(f"Account:      {getattr(a, 'account_number', '?')}  ({getattr(a, 'status', '?')})")
    print(f"Mode:         PAPER (ALPACA_PAPER={'true' if is_paper_env(settings.env_path) else 'NOT true'})")
    print(f"Currency:     {getattr(a, 'currency', '?')}")
    print(f"Cash:         {getattr(a, 'cash', '?')}")
    print(f"Buying power: {getattr(a, 'buying_power', '?')}")
    print(f"Portfolio:    {getattr(a, 'portfolio_value', '?')}")
    print(f"Blocked:      trading={getattr(a, 'trading_blocked', '?')}  account={getattr(a, 'account_blocked', '?')}")
    return 0


def cmd_positions(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    from .agents.position_agent import PositionAgent, total_exposure
    from .orchestrator import reconcile

    clients = _make_clients_or_exit(settings, logger)
    if clients is None:
        return 1

    params = load_strategy_params()
    _account, positions, _open_orders, position_states = reconcile(clients, settings, logger)

    if not positions:
        print("No open positions.")
        return 0

    # Build snapshots so exit classification (vwap_revert/max_hold) is accurate.
    from .data.market_data import MarketDataAgent

    md = MarketDataAgent(clients["data"], settings, params, logger)
    held = [p.symbol for p in positions if getattr(p, "symbol", None)]
    snaps = md.get_snapshots(universe=held, position_states=position_states)
    snap_by_symbol = {s.symbol: s for s in snaps}

    summaries = PositionAgent(params, settings, logger).summarize(positions, params, snap_by_symbol)
    print(f"{'SYMBOL':<8}{'QTY':>10}{'AVG':>10}{'PRICE':>10}{'MKT_VAL':>12}{'UPL':>10}{'HOLD_MIN':>10}  EXIT")
    for s in summaries:
        print(
            f"{s.symbol:<8}"
            f"{_fmt(s.qty):>10}{_fmt(s.avg_entry_price):>10}{_fmt(s.current_price):>10}"
            f"{_fmt(s.market_value):>12}{_fmt(s.unrealized_pl):>10}{_fmt(s.holding_minutes):>10}"
            f"  {s.exit_status.value}{' *' if s.would_exit else ''}"
        )
    print(f"\nTotal exposure: ${total_exposure(positions):.2f}")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    """ONE cycle. Armed only if --arm AND paper env (INV-1/2/3)."""
    from .orchestrator import run_cycle

    armed = bool(getattr(args, "arm", False))
    clients = _make_clients_or_exit(settings, logger)
    if clients is None:
        return 1

    params = load_strategy_params()
    limits = RiskLimits()
    scorer = _load_scorer()

    if armed and not is_paper_env(settings.env_path):
        print("refusing to arm: ALPACA_PAPER != 'true' (no real-money mode).", file=sys.stderr)
        armed = False

    report = run_cycle(clients, params, limits, scorer, armed=armed, settings=settings, logger=logger)
    _print_cycle(report, armed)
    return 0


def cmd_dry_run(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    """Explicit single cycle that is NEVER armed (alias of `run` without --arm)."""
    from .orchestrator import run_cycle

    clients = _make_clients_or_exit(settings, logger)
    if clients is None:
        return 1
    params = load_strategy_params()
    report = run_cycle(clients, params, RiskLimits(), _load_scorer(), armed=False,
                       settings=settings, logger=logger)
    _print_cycle(report, armed=False)
    return 0


def cmd_loop(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    from .loop import run_loop

    armed = bool(getattr(args, "arm", False))
    interval = getattr(args, "interval", None)
    once = bool(getattr(args, "once", False))

    if armed and not is_paper_env(settings.env_path):
        print("refusing to arm: ALPACA_PAPER != 'true' (no real-money mode).", file=sys.stderr)
        armed = False

    clients = _make_clients_or_exit(settings, logger)
    if clients is None:
        return 1

    print(f"starting loop (armed={armed}, interval={interval or settings.loop_interval_sec}s, once={once}). Ctrl-C to stop.")
    run_loop(
        armed=armed, interval=interval, once=once,
        params=load_strategy_params(), limits=RiskLimits(), scorer=_load_scorer(),
        settings=settings, clients=clients, logger=logger,
    )
    return 0


def cmd_backfill(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    from .data import history

    symbols = _resolve_symbols(args, settings)
    start = args.start or _default_window()[0]
    end = args.end or _default_window()[1]

    clients = _make_clients_or_exit(settings, logger)
    if clients is None:
        return 1

    print(f"backfilling {len(symbols)} symbol(s) {start.date()}..{end.date()} (regular hours, 1-min IEX)...")
    out = history.backfill(symbols, start, end, data_client=clients["data"], settings=settings, logger=logger)
    if not out:
        print("no bars cached (check dates / market data access).")
        return 1
    for sym, path in out.items():
        print(f"  {sym:<6} -> {path}")
    return 0


def cmd_backtest(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    try:
        from .backtest.engine import run_backtest
    except ImportError as exc:
        print(f"backtest engine not available: {exc}", file=sys.stderr)
        return 2

    from .data.market_data import compute_indicators  # noqa: F401  (parity import guard)

    symbols = _resolve_symbols(args, settings)
    start = args.start or _default_window(days=10)[0]
    end = args.end or _default_window(days=10)[1]

    bars_by_symbol = _load_bars_for_backtest(symbols, start, end, settings)
    n_bars = sum(len(v) for v in bars_by_symbol.values())
    if n_bars == 0:
        print("no cached bars found — run `backfill` first.", file=sys.stderr)
        return 1

    params = load_strategy_params()
    scorer = _load_scorer()
    print(f"backtesting {len(bars_by_symbol)} symbol(s), {n_bars} bars, params={params.to_dict()}...")
    result = run_backtest(bars_by_symbol, params, RiskLimits(), scorer)

    metrics = getattr(result, "metrics", {}) or {}
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<18} {v}")
    print(f"  trades             {len(getattr(result, 'trades', []) or [])}")
    return 0


def cmd_optimize(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    try:
        from .backtest.optimize import optimize
    except ImportError as exc:
        print(f"optimizer not available: {exc}", file=sys.stderr)
        return 2

    symbols = _resolve_symbols(args, settings)
    start = args.start or _default_window(days=20)[0]
    end = args.end or _default_window(days=20)[1]

    history_bars = _load_bars_for_backtest(symbols, start, end, settings)
    if sum(len(v) for v in history_bars.values()) == 0:
        print("no cached bars found — run `backfill` first.", file=sys.stderr)
        return 1

    base = load_strategy_params()
    scorer = _load_scorer()
    print(f"optimizing over {len(history_bars)} symbol(s): n_iter={args.n_iter}, folds={args.folds}...")
    best, report = optimize(history_bars, base, RiskLimits(), n_iter=args.n_iter, folds=args.folds, scorer=scorer)

    path = config.save_best_params(best)
    print(f"\nbest params: {best.to_dict()}")
    print(f"saved -> {path}")
    leaderboard = (report or {}).get("leaderboard") if isinstance(report, dict) else None
    if leaderboard:
        print("\ntop results:")
        for row in leaderboard[:5]:
            print(f"  {row}")
    return 0


def cmd_dataset(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    """Backtest over cached bars and write a labeled ML dataset for `train`.

    Reuses the SAME run_backtest as `backtest` (so the labeled features come from the exact
    decision/risk path), then ml.dataset.build_dataset/save_dataset to persist rows in
    FEATURE_ORDER + 'label'. Default output is models/dataset.parquet (csv fallback).
    """
    try:
        from .backtest.engine import run_backtest
        from .ml.dataset import build_dataset, save_dataset
    except ImportError as exc:
        print(f"dataset builder not available: {exc}", file=sys.stderr)
        return 2

    symbols = _resolve_symbols(args, settings)
    start = args.start or _default_window(days=20)[0]
    end = args.end or _default_window(days=20)[1]

    bars_by_symbol = _load_bars_for_backtest(symbols, start, end, settings)
    if sum(len(v) for v in bars_by_symbol.values()) == 0:
        print("no cached bars found — run `backfill` first.", file=sys.stderr)
        return 1

    # Dataset generation MUST be GATE-FREE: label every rule-passing candidate, not just the
    # subset the currently-deployed model would approve. If we ran the backtest with the live
    # scorer + a non-zero ml_threshold, candidates the prior model scored below threshold are
    # never taken and never labeled, so each retrain learns only on trades the previous model
    # already liked — a self-reinforcing selection bias that narrows the feature distribution
    # and inflates apparent winrate. So we force the ML gate OFF here: scorer=None (passthrough
    # p_win=1.0) AND ml_threshold=0.0. The optimizer tunes ml_threshold AGAINST this unbiased
    # set afterward; it must never feed back into dataset construction.
    params = replace(load_strategy_params(), ml_threshold=0.0)
    result = run_backtest(bars_by_symbol, params, RiskLimits(), scorer=None)

    X, y, names = build_dataset(getattr(result, "trades", []) or [])
    out_path = getattr(args, "out", None) or (settings.models_dir / "dataset.parquet")
    written = save_dataset(X, y, names, out_path)
    n_rows = int(X.shape[0]) if hasattr(X, "shape") else 0
    print(f"built dataset: {n_rows} labeled rows from {len(getattr(result, 'trades', []) or [])} trades")
    print(f"saved -> {written}")
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings, logger: JsonlLogger) -> int:
    try:
        from .ml.train_model import train
    except ImportError as exc:
        print(f"trainer not available: {exc}", file=sys.stderr)
        return 2

    dataset = getattr(args, "dataset", None)
    if dataset is None:
        print("error: --dataset PATH is required (build one from a backtest first).", file=sys.stderr)
        return 1

    print(f"training entry scorer from {dataset} -> {settings.models_dir} ...")
    try:
        metrics = train(dataset, out_dir=settings.models_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\nTraining metrics:")
    for k, v in (metrics or {}).items():
        print(f"  {k:<18} {v}")
    return 0


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_cycle(report, armed: bool) -> None:
    """Print a concise human summary of a CycleReport (full detail is in the JSONL log)."""
    print(f"\n=== cycle @ {report.asof}  armed={armed}  exposure=${report.exposure:.2f} ===")
    actionable = [p for p in report.proposals if p.is_entry or p.is_exit]
    if actionable:
        print("proposals:")
        for p in actionable:
            tag = "ENTRY" if p.is_entry else "EXIT"
            amt = f"${p.notional:.0f}" if p.notional else f"{p.qty:g} sh"
            print(f"  {tag:<5} {p.symbol:<6} {amt:<10} {p.reason[:80]}")
    else:
        print("proposals: none actionable (all HOLD).")

    if report.orders:
        print("orders:")
        for o in report.orders:
            mode = "DRY-RUN" if o.dry_run else "LIVE-PAPER"
            print(f"  {mode:<10} {o.side.value:<4} {o.symbol:<6} status={o.status}"
                  f"{f' id={o.id}' if o.id else ''}{f' err={o.error}' if o.error else ''}")

    if report.summaries:
        print("positions:")
        for s in report.summaries:
            print(f"  {s.symbol:<6} qty={_fmt(s.qty)} upl={_fmt(s.unrealized_pl)} "
                  f"exit={s.exit_status.value}{' *' if s.would_exit else ''}")

    if report.errors:
        print(f"errors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  ! {e}")


# --------------------------------------------------------------------------- #
# Bar loading for offline commands
# --------------------------------------------------------------------------- #
def _load_bars_for_backtest(symbols: List[str], start: _dt.datetime, end: _dt.datetime,
                            settings: Settings) -> Dict[str, list]:
    """Load cached bars per symbol and convert to lists of typed Bar objects.

    Reads via history.load_cached_bars (regular-hours, ascending, de-duped) and maps each
    DataFrame row to a types.Bar so the backtester sees the same Bar contract as the live
    path. Symbols with no cache are omitted.
    """
    from .data import history
    from .types import Bar

    out: Dict[str, list] = {}
    for sym in symbols:
        df = history.load_cached_bars(sym, start, end, settings)
        if df is None or df.empty:
            continue
        bars = []
        for row in df.itertuples(index=False):
            bars.append(Bar(
                symbol=str(row.symbol),
                start=row.start.to_pydatetime() if hasattr(row.start, "to_pydatetime") else row.start,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                trade_count=(float(row.trade_count) if row.trade_count is not None and row.trade_count == row.trade_count else None),
                vwap=(float(row.vwap) if row.vwap is not None and row.vwap == row.vwap else None),
            ))
        if bars:
            out[sym] = bars
    return out


# --------------------------------------------------------------------------- #
# Parser + dispatch
# --------------------------------------------------------------------------- #
_HANDLERS = {
    "account": cmd_account,
    "positions": cmd_positions,
    "run": cmd_run,
    "dry-run": cmd_dry_run,
    "loop": cmd_loop,
    "backfill": cmd_backfill,
    "backtest": cmd_backtest,
    "optimize": cmd_optimize,
    "dataset": cmd_dataset,
    "train": cmd_train,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Every command defaults to SAFE (DRY-RUN)."""
    parser = argparse.ArgumentParser(
        prog="agentic-trader",
        description="Intraday VWAP mean-reversion PAPER trader (deterministic, dry-run by default).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("account", help="print the paper TradeAccount summary")
    sub.add_parser("positions", help="print reconciled positions + exit classification")

    p_run = sub.add_parser("run", help="run ONE cycle (dry-run unless --arm)")
    p_run.add_argument("--arm", action="store_true", help="permit real PAPER order submission (INV-2)")

    sub.add_parser("dry-run", help="run ONE cycle, never armed")

    p_loop = sub.add_parser("loop", help="continuous market-hours-aware loop")
    p_loop.add_argument("--arm", action="store_true", help="permit real PAPER order submission (INV-2)")
    p_loop.add_argument("--interval", type=float, default=None, help="seconds between cycles (default 60)")
    p_loop.add_argument("--once", action="store_true", help="run a single cycle then exit")

    p_bf = sub.add_parser("backfill", help="cache historical 1-min bars under data/cache/")
    p_bf.add_argument("--symbols", nargs="+", default=None, help="symbols (default: full universe)")
    p_bf.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (default: 5d ago)")
    p_bf.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (default: now)")

    p_bt = sub.add_parser("backtest", help="run a t->t+1 backtest over cached bars")
    p_bt.add_argument("--symbols", nargs="+", default=None, help="symbols (default: full universe)")
    p_bt.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (default: 10d ago)")
    p_bt.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (default: now)")

    p_opt = sub.add_parser("optimize", help="walk-forward param search -> best_params.json")
    p_opt.add_argument("--symbols", nargs="+", default=None, help="symbols (default: full universe)")
    p_opt.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (default: 20d ago)")
    p_opt.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (default: now)")
    p_opt.add_argument("--n-iter", dest="n_iter", type=int, default=50, help="sampled param sets")
    p_opt.add_argument("--folds", type=int, default=3, help="walk-forward time folds")

    p_ds = sub.add_parser("dataset", help="backtest -> labeled ML dataset (parquet/csv)")
    p_ds.add_argument("--symbols", nargs="+", default=None, help="symbols (default: full universe)")
    p_ds.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (default: 20d ago)")
    p_ds.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (default: now)")
    p_ds.add_argument("--out", type=str, default=None, help="output dataset path (default: models/dataset.parquet)")

    p_tr = sub.add_parser("train", help="train the ML entry scorer from a dataset")
    p_tr.add_argument("--dataset", type=str, default=None, help="dataset path (.parquet/.csv)")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse args and dispatch. Returns a process exit code (0 = ok)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    ensure_dirs(settings)
    # echo dry-run/cycle commands to stdout; quiet for plain queries handled by handlers.
    logger = JsonlLogger(echo=False)

    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover — argparse enforces choices
        parser.error(f"unknown command: {args.command}")
        return 2

    try:
        return int(handler(args, settings, logger))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        print(f"error: {exc}", file=sys.stderr)
        logger.log_error(f"cli.{args.command}", "command failed", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
