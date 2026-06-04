"""factor_harness.py — DAILY cross-sectional FACTOR research toolkit.

PURE RESEARCH / BACKTESTING. Places ZERO orders. Read-only on cached market data.

=============================================================================
WHY THIS EXISTS
=============================================================================
The intraday VWAP family is dead (experiments/results/NEUTRAL_SUMMARY.md: 0 beta-neutral
edges on free IEX intraday data — the IEX 2-3%-of-volume defect poisoned every volume/VWAP
signal). This module pivots to DAILY cross-sectional factor strategies on FULL-QUALITY
split+dividend-adjusted free daily bars (daily bars are NOT affected by the IEX defect).

It reuses the proven ideas from neutral_harness.py verbatim in spirit:
  - WALK-FORWARD across multiple sequential OOS folds spanning different regimes.
  - Mandatory BETA DECOMPOSITION (numpy OLS) of strategy daily returns on SPY.
  - Realistic costs charged on TURNOVER, every leg, both sides.
  - DOLLAR-NEUTRAL cross-sectional construction (market-neutral BY CONSTRUCTION):
    each rebalance, rank the universe by the factor, go LONG the top quantile and
    SHORT the bottom quantile in equal dollars.

=============================================================================
THE factor_fn CONTRACT  (the ONE thing every experiment must implement)
=============================================================================
A factor is a callable that maps the price panel to a per-day cross-sectional SCORE:

    factor_fn(panel) -> pd.DataFrame   # index = date, columns = symbols, values = score

  - The value at (date d, symbol s) is the factor score for symbol s computed using ONLY
    information available AT THE CLOSE OF DAY d (no look-ahead). Higher score => the
    construction goes MORE LONG that name (top quantile long, bottom quantile short).
  - NaN means "no score for that name on that day" (e.g. insufficient lookback, or the
    name is not in the tradeable set that day) — it is excluded from ranking that day.
  - The harness handles the t->t+1 timing: a score row dated d decides weights that are
    HELD over the NEXT trading day's close-to-close return (d -> d+1). The factor_fn must
    therefore NOT peek at returns on/after d. Build factors from `panel.close` / returns
    shifted so the last row used is day d's close.

`panel` is the Panel object returned by load_daily() (see below): a tidy collection of
date x symbol frames (close/open/high/low/volume + daily simple returns).

A REVERSAL factor (mean-reversion) sets score = -past_return (buy recent losers).
A MOMENTUM factor sets score = +past_return over a longer lookback skipping the last
month. A LOW-VOL factor sets score = -trailing_vol. Etc. Each lives in its own exp*.py.

=============================================================================
COST / FILL / NEUTRALITY MODEL
=============================================================================
- Strict t->t+1: weights w_d are decided from info up to day d's CLOSE; the realized
  strategy return on day d is w_d . r_{d+1}, where r_{d+1} is the close-to-close simple
  return from day d to day d+1. NO look-ahead (the weight uses no future price).
- DOLLAR-NEUTRAL: sum of long weights = +1, sum of short weights = -1 (gross = 2, net = 0)
  scaled so sum(|w|) = 1 (each side 0.5). Market-neutral by construction.
- COSTS: $0 commission (Alpaca). Per-rebalance turnover cost = cost_bps_per_side/1e4 *
  sum(|w_d - w_{d-1, drifted}|) charged on the day the new weights take effect. Both sides
  pay. Higher turnover (daily reversal) strictly costs more than low turnover (monthly
  momentum) — verified in the sanity suite. Base case 5bp/side; sensitivity 2bp & 10bp.
- BETA DECOMPOSITION: regress strategy daily returns on SPY daily returns -> alpha
  (annualized), beta, alpha t-stat, R^2. Real edge gate: alpha_tstat >= 2, |beta| < 0.15,
  positive in a MAJORITY of folds, survives the 5bp base case.

=============================================================================
SURVIVORSHIP / LOOK-AHEAD BIAS (honest caveat — baked into reporting)
=============================================================================
The universe is TODAY's large-caps applied backward over history. Names that were large
in 2020 but later dropped out (delisted, merged, fell out of the index) are ABSENT, and
names that grew INTO large-cap are present for their whole window. This SURVIVORSHIP bias
INFLATES results (we are implicitly only trading known survivors). We cannot fix
point-in-time membership on free data. Treat any marginal edge skeptically. The target is
a small REAL edge (Sharpe ~0.5-1), not a printer.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/factors/cache")
RESULTS_DIR = Path("/Users/pnle/Desktop/alpaca-cli/experiments/factors/results")
BENCHMARK = "SPY"
TRADING_YEAR = 252


# --------------------------------------------------------------------------- #
# 1) Data loading -> tidy panel
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """A tidy daily panel: aligned date x symbol frames for one universe.

    Every frame shares the SAME DatetimeIndex (sorted ascending, one row per trading date)
    and the SAME columns (the symbols), so positions/scores/returns line up by construction.

    Fields:
        close, open, high, low, volume : DataFrame[date, symbol]  (split+div adjusted)
        rets : DataFrame[date, symbol]  daily SIMPLE close-to-close return r_d =
               close_d / close_{d-1} - 1 (first row NaN). rets.loc[d, s] is realized
               OVER day d (from d-1 close to d close). To get the return realized t->t+1
               of a weight decided at day d, use rets.shift(-1).
        symbols : list[str]
        dates   : DatetimeIndex
    """
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    rets: pd.DataFrame
    symbols: List[str]
    dates: pd.DatetimeIndex


def load_daily(
    symbols: List[str],
    cache_dir: Path | str = CACHE_DIR,
    min_full_overlap: bool = True,
) -> Panel:
    """Load cached daily adjusted bars for `symbols` into an aligned Panel.

    Reads experiments/factors/cache/{SYMBOL}_daily.parquet (written by fetch_daily.py),
    builds per-field date x symbol frames on the UNION of trading dates, and (when
    min_full_overlap=True, the default) restricts to the date range where ALL requested
    symbols have data — so no name silently carries NaNs that would distort cross-sectional
    ranks. Symbols with no cache file are dropped (a warning is not raised; check the
    returned .symbols).

    Daily SIMPLE returns are computed per symbol as close.pct_change().
    """
    cache_dir = Path(cache_dir)
    frames: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = cache_dir / f"{sym}_daily.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
        frames[sym] = df[["open", "high", "low", "close", "volume"]]

    if not frames:
        empty = pd.DataFrame()
        return Panel(empty, empty, empty, empty, empty, empty, [], pd.DatetimeIndex([]))

    syms = sorted(frames.keys())

    def _field(name: str) -> pd.DataFrame:
        cols = {s: frames[s][name] for s in syms}
        out = pd.DataFrame(cols).sort_index()
        return out

    close = _field("close")
    open_ = _field("open")
    high = _field("high")
    low = _field("low")
    volume = _field("volume")

    if min_full_overlap:
        # Restrict to the date span where every symbol has a close (intersection of each
        # symbol's [first, last]). This avoids ragged edges polluting cross-sectional ranks.
        first_common = max(frames[s]["close"].dropna().index.min() for s in syms)
        last_common = min(frames[s]["close"].dropna().index.max() for s in syms)
        mask = (close.index >= first_common) & (close.index <= last_common)
        close, open_, high, low, volume = (df.loc[mask] for df in (close, open_, high, low, volume))

    rets = close.pct_change()

    return Panel(
        close=close,
        open=open_,
        high=high,
        low=low,
        volume=volume,
        rets=rets,
        symbols=syms,
        dates=close.index,
    )


def spy_daily_returns(cache_dir: Path | str = CACHE_DIR, symbol: str = BENCHMARK) -> pd.Series:
    """SPY (benchmark) daily SIMPLE close-to-close returns, indexed by date (Timestamp)."""
    path = Path(cache_dir) / f"{symbol}_daily.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    return df["close"].pct_change().dropna()


# --------------------------------------------------------------------------- #
# 2) Factor utilities
# --------------------------------------------------------------------------- #
def cross_sectional_rank(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-DAY cross-sectional rank of `scores` in [0, 1] (1 = highest score that day).

    NaNs are kept NaN (excluded from ranking). Uses average ranks for ties, normalized by
    the count of non-NaN names that day so the scale is comparable across days with
    different #valid names.
    """
    ranks = scores.rank(axis=1, method="average", na_option="keep")
    counts = scores.notna().sum(axis=1)
    # normalize each row by its valid count (avoid div-by-zero)
    norm = ranks.div(counts.where(counts > 0, np.nan), axis=0)
    return norm


def form_dollar_neutral_portfolio(
    scores: pd.Series | pd.DataFrame,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
    min_names_per_side: int = 3,
) -> pd.Series | pd.DataFrame:
    """Per-day target weights: LONG the top `top_q` fraction, SHORT the bottom `bottom_q`.

    Dollar-neutral by construction: long weights sum to +0.5, short weights sum to -0.5, so
    net = 0 and gross = sum(|w|) = 1.0. Equal dollars WITHIN each leg (equal-weight basket).

    Accepts either a single day's Series (index = symbols) -> Series of weights, or a full
    DataFrame (index = date, columns = symbols) -> DataFrame of weights (applied row-wise).

    A day with fewer than `min_names_per_side` valid names on EITHER side gets all-zero
    weights (flat that day) — we never trade a degenerate cross-section.
    """
    if isinstance(scores, pd.DataFrame):
        out = scores.apply(
            lambda row: form_dollar_neutral_portfolio(
                row, top_q=top_q, bottom_q=bottom_q, min_names_per_side=min_names_per_side
            ),
            axis=1,
        )
        return out.fillna(0.0)

    row = scores.dropna()
    w = pd.Series(0.0, index=scores.index)
    n = len(row)
    if n == 0:
        return w
    # No cross-sectional dispersion => no information to rank on => stay flat (avoid
    # manufacturing a book from arbitrary tie-ordering; this is what makes a constant
    # score yield EXACTLY zero, see factor_sanity (a)).
    if float(row.max() - row.min()) <= 0.0:
        return w
    k_top = max(1, int(np.floor(n * top_q)))
    k_bot = max(1, int(np.floor(n * bottom_q)))
    if k_top < min_names_per_side or k_bot < min_names_per_side:
        return w  # cross-section too small to form a clean neutral book -> flat
    if k_top + k_bot > n:
        # overlapping quantiles -> shrink so longs and shorts are disjoint
        k_top = k_bot = max(1, n // 2)
        if k_top < min_names_per_side:
            return w
    ordered = row.sort_values(ascending=False)
    longs = ordered.index[:k_top]
    shorts = ordered.index[-k_bot:]
    w.loc[longs] = 0.5 / k_top
    w.loc[shorts] = -0.5 / k_bot
    return w


# --------------------------------------------------------------------------- #
# 3) Cross-sectional backtester (strict t->t+1, costs on turnover)
# --------------------------------------------------------------------------- #
def backtest_xsection(
    weights_over_time: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_freq: int = 1,
    cost_bps_per_side: float = 5.0,
) -> dict:
    """Backtest a sequence of daily cross-sectional target weights, NET of turnover cost.

    Parameters
    ----------
    weights_over_time : DataFrame[date, symbol]
        TARGET weights decided AT THE CLOSE OF each date (from the factor scores of that
        date). Dollar-neutral rows (sum 0, gross ~1) are expected but not required.
    returns : DataFrame[date, symbol]
        Daily SIMPLE close-to-close returns r_d (realized OVER day d). Must share columns
        with `weights_over_time` (the harness aligns indices).
    rebalance_freq : int
        Rebalance every `rebalance_freq` trading days. On non-rebalance days the book is
        HELD (drifts with prices); turnover/cost is only paid on rebalance days. freq=1 =
        rebalance daily (high turnover); freq=21 ~ monthly (low turnover).
    cost_bps_per_side : float
        Slippage/spread per side in basis points, charged on the traded notional fraction
        (sum |w_new - w_old_drifted|) at each rebalance. Both sides pay (a 2-sided trade of
        notional X costs 2 * cost_bps * X / 1e4 because both the sell and the buy leg pay).

    TIMING (NO LOOK-AHEAD) — the single most important contract
    -----------------------------------------------------------
    `weights_over_time.loc[d]` are decided at the CLOSE of day d. `returns.loc[t]` is the
    return realized OVER day t (close_{t-1} -> close_t). The strategy return REALIZED on day
    t uses the book carried INTO day t, i.e. the weights last decided at the close of day
    t-1 (then drifted). So:

        strat_ret[t] = (book carried into day t) . returns[t]   -  (cost of trades done at
                                                                     close of t-1, amortized
                                                                     onto the realization day)

    Internally we lag the target/held weights by one day before dotting with returns, so the
    return on date t never uses weights that saw day-t prices. The OUTPUT `daily` is indexed
    by the REALIZATION date t (so it aligns 1:1 with SPY's day-t return for beta_decompose).
    A weight decided on the LAST date in the index earns nothing here (no day after it in the
    index) — callers extend the index by one day to capture it.

    Returns
    -------
    dict with:
        daily : pd.Series  net daily strategy returns (indexed by REALIZATION date)
        gross_daily : pd.Series  gross (pre-cost) daily returns
        equity : pd.Series  cumulative net equity curve (starts at 1.0)
        turnover : pd.Series  per-realization-day one-way turnover fraction
        metrics : dict  annual_return, sharpe, max_drawdown, avg_turnover, total_cost, n_days
    """
    if weights_over_time.empty or returns.empty:
        return _empty_xsection_result()

    # Align columns and dates on the UNION (so we keep every day either side covers).
    cols = [c for c in returns.columns if c in weights_over_time.columns]
    if not cols:
        return _empty_xsection_result()
    idx = weights_over_time.index.union(returns.index).sort_values()
    W = weights_over_time[cols].reindex(idx).fillna(0.0)
    R = returns[cols].reindex(idx)

    n = len(idx)
    if n < 3:
        return _empty_xsection_result()

    cost_frac = cost_bps_per_side / 1e4
    R_arr = R.to_numpy(dtype=float)  # R_arr[t] = return realized OVER day t
    W_arr = W.to_numpy(dtype=float)  # W_arr[t] = target weights decided at CLOSE of day t

    # Single forward pass. `held` = weights on the book carried INTO the current day (decided
    # at a prior close, then drifted with prices). Cost of a rebalance done at the close of
    # day t is attributed to day t+1 (the realization day on which the new book first earns),
    # so that the cost lands on the same day as the PnL of the position it bought.
    held = np.zeros(len(cols))            # book carried INTO day 0 (flat)
    gross_daily = np.zeros(n)
    cost_on_day = np.zeros(n)             # cost_on_day[t] = cost of the rebalance at close of t-1
    turnover = np.zeros(n)                # turnover[t] = one-way traded fraction realized on t

    for t in range(n):
        r_t_safe = np.where(np.isnan(R_arr[t]), 0.0, R_arr[t])
        # (1) realize day t's gross PnL with the book carried INTO day t (no look-ahead).
        gross_daily[t] = float(np.dot(held, r_t_safe))
        # (2) the carried book drifts over day t.
        drifted = held * (1.0 + r_t_safe)
        # (3) at the CLOSE of day t, rebalance to the new target every `rebalance_freq` days;
        #     defer its cost/turnover onto day t+1's realization (where the new book earns).
        if t % rebalance_freq == 0:
            target = W_arr[t]
            to = float(np.nansum(np.abs(target - drifted)))
            if t + 1 < n:
                cost_on_day[t + 1] = cost_frac * to
                turnover[t + 1] = to
            held = target.copy()
        else:
            held = drifted

    net_daily = gross_daily - cost_on_day

    daily = pd.Series(net_daily, index=idx)
    gross = pd.Series(gross_daily, index=idx)
    to_s = pd.Series(turnover, index=idx)
    equity = (1.0 + daily).cumprod()

    metrics = _xsection_metrics(daily, to_s, cost_frac)
    return {
        "daily": daily,
        "gross_daily": gross,
        "equity": equity,
        "turnover": to_s,
        "metrics": metrics,
    }


def _empty_xsection_result() -> dict:
    e = pd.Series(dtype=float)
    return {
        "daily": e,
        "gross_daily": e,
        "equity": e,
        "turnover": e,
        "metrics": {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "total_cost": 0.0,
            "n_days": 0,
        },
    }


def _xsection_metrics(daily: pd.Series, turnover: pd.Series, cost_frac: float) -> dict:
    n = len(daily)
    if n == 0:
        return _empty_xsection_result()["metrics"]
    arr = daily.to_numpy(dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    annual_return = mean * TRADING_YEAR
    sharpe = (mean / std * np.sqrt(TRADING_YEAR)) if std > 0 else 0.0
    eq = (1.0 + daily).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    max_dd = float(dd) if not pd.isna(dd) else 0.0
    avg_to = float(turnover.mean())
    total_cost = float((turnover * cost_frac).sum())
    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_to,
        "total_cost": total_cost,
        "n_days": int(n),
    }


# --------------------------------------------------------------------------- #
# 4) Walk-forward folds
# --------------------------------------------------------------------------- #
def walk_forward_folds(dates: List, n_folds: int = 4, train_frac: float = 0.4) -> List[dict]:
    """Build `n_folds` SEQUENTIAL expanding-train / non-overlapping-test walk-forward folds.

    `dates` is the sorted list of distinct trading dates over the full ~6y history. The OOS
    test windows are NON-OVERLAPPING and tile the back of history so each fold's TEST is a
    different, later, out-of-sample slice spanning a different regime. For fold k, TRAIN is
    everything strictly before that test window (anchored/expanding). Mirrors
    neutral_harness.walk_forward_folds.

    Layout (n_folds=4, train_frac=0.4):
        | ----- initial train (40%) ----- | test0 | test1 | test2 | test3 |
    """
    ds = sorted(set(pd.Timestamp(d) for d in dates))
    n = len(ds)
    if n < n_folds + 2:
        raise ValueError(f"need at least {n_folds + 2} dates, got {n}")
    n_initial_train = max(1, int(round(n * train_frac)))
    n_initial_train = min(n_initial_train, n - n_folds)
    test_region = ds[n_initial_train:]
    m = len(test_region)
    base = m // n_folds
    folds = []
    cursor = 0
    for k in range(n_folds):
        size = base + (1 if k >= n_folds - (m - base * n_folds) else 0)
        test_slice = test_region[cursor : cursor + size]
        cursor += size
        if not test_slice:
            continue
        test_start = test_slice[0]
        test_end = test_slice[-1]
        train_slice = [d for d in ds if d < test_start]
        folds.append(
            {
                "fold": k,
                "train": (str(train_slice[0].date()), str(train_slice[-1].date()), len(train_slice)),
                "test": (str(test_start.date()), str(test_end.date()), len(test_slice)),
                "_train_dates": pd.DatetimeIndex(train_slice),
                "_test_dates": pd.DatetimeIndex(test_slice),
            }
        )
    return folds


# --------------------------------------------------------------------------- #
# 5) Beta decomposition (numpy OLS — mirrors neutral_harness.beta_decompose)
# --------------------------------------------------------------------------- #
def beta_decompose(strat_daily_ret: pd.Series, spy_daily_ret: pd.Series) -> dict:
    """OLS regression of strategy daily returns on SPY daily returns.

        strat_t = alpha + beta * spy_t + eps_t

    Aligns on the intersection of dates (only days BOTH have a return). Returns
    {alpha_annual, beta, alpha_tstat, beta_tstat, r2, n_days, alpha_per_day,
     strat_mean_daily, strat_ann_return}. alpha_annual = alpha_per_day * 252. A
    'market-neutral edge' requires |beta| < 0.15 AND alpha_tstat >= 2. Classical (non-HC)
    OLS standard errors on daily equity-curve returns, same convention as neutral_harness.
    """
    s = strat_daily_ret.dropna()
    m = spy_daily_ret.dropna()
    idx = s.index.intersection(m.index)
    y = s.reindex(idx).to_numpy(dtype=float)
    x = m.reindex(idx).to_numpy(dtype=float)
    n = len(y)
    out = {
        "alpha_per_day": 0.0,
        "alpha_annual": 0.0,
        "beta": 0.0,
        "alpha_tstat": 0.0,
        "beta_tstat": 0.0,
        "r2": 0.0,
        "n_days": int(n),
        "strat_mean_daily": float(np.mean(y)) if n else 0.0,
        "strat_ann_return": float(np.mean(y) * TRADING_YEAR) if n else 0.0,
    }
    if n < 3:
        return out
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return out
    beta_hat = XtX_inv @ (X.T @ y)  # [alpha, beta]
    resid = y - X @ beta_hat
    dof = n - 2
    if dof <= 0:
        return out
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    alpha = float(beta_hat[0])
    beta = float(beta_hat[1])
    out.update(
        {
            "alpha_per_day": alpha,
            "alpha_annual": alpha * TRADING_YEAR,
            "beta": beta,
            "alpha_tstat": alpha / se_alpha if se_alpha > 0 else 0.0,
            "beta_tstat": beta / se_beta if se_beta > 0 else 0.0,
            "r2": r2,
            "n_days": int(n),
            "strat_mean_daily": float(y.mean()),
            "strat_ann_return": float(y.mean() * TRADING_YEAR),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# 6) Walk-forward evaluation of a factor
# --------------------------------------------------------------------------- #
def evaluate_walkforward(
    factor_fn_or_factory,
    panel: Panel,
    spy: pd.Series,
    n_folds: int = 4,
    cost_bps: float = 5.0,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
    rebalance_freq: int = 1,
    min_names_per_side: int = 3,
    train_frac: float = 0.4,
) -> dict:
    """Evaluate a factor across `n_folds` sequential OOS folds; per-fold + aggregate metrics.

    `factor_fn_or_factory`:
        - a factor_fn(panel) -> scores DataFrame (date x symbol), the SAME config scored on
          every fold's TEST, OR
        - a callable(train_panel, fold_info) -> factor_fn  (a FACTORY that may TUNE knobs on
          that fold's TRAIN ONLY, then returns the factor_fn scored on the untouched TEST).
          The factory must never look at test data. Use the factory form to tune
          quantile/lookback/rebalance on TRAIN and score OOS once (honest selection).

    For each fold: score the factor on the TRAIN+TEST window (factors need lookback history
    that predates the test window, so we compute scores on the full panel up to test_end,
    then evaluate ONLY the test dates), form dollar-neutral weights, backtest NET of cost on
    the TEST slice, and collect that fold's daily returns + metrics. We then CONCATENATE all
    OOS days and run beta_decompose vs SPY — the headline neutrality test.

    Returns per-fold records + aggregate OOS (annual_return, sharpe, max_drawdown, turnover)
    + concatenated-OOS beta decomposition + per-fold beta decompositions + folds_positive.
    """
    dates = list(panel.dates)
    folds = walk_forward_folds(dates, n_folds=n_folds, train_frac=train_frac)

    fold_records = []
    oos_daily_pieces = []
    fold_betas = []
    positive = 0
    total_turnover_w = 0.0
    total_days = 0

    for f in folds:
        test_idx = f["_test_dates"]
        train_idx = f["_train_dates"]

        # Resolve the factor for this fold (tune on TRAIN only if a factory is given).
        if _is_factory(factor_fn_or_factory):
            train_panel = _slice_panel(panel, panel.dates[panel.dates <= train_idx[-1]])
            factor_fn = factor_fn_or_factory(train_panel, f)
            chosen = getattr(factor_fn, "chosen_params", None)
        else:
            factor_fn = factor_fn_or_factory
            chosen = None

        # Compute scores on the panel up to the END of this test window, so the factor has
        # all the lookback it needs, then keep ONLY the test-date rows (no future leak: each
        # score row uses only that day's-and-earlier closes by the factor_fn contract).
        upto = panel.dates[panel.dates <= test_idx[-1]]
        sub_panel = _slice_panel(panel, upto)
        scores = factor_fn(sub_panel)
        scores_test = scores.reindex(test_idx)

        # Weights from scores (indexed by DECISION date = close of each test day).
        weights = form_dollar_neutral_portfolio(
            scores_test, top_q=top_q, bottom_q=bottom_q, min_names_per_side=min_names_per_side
        )
        # Returns over the test window EXTENDED by one trading day, so the weight decided on
        # the last test day can still earn its t+1 realization. backtest_xsection lags the
        # weights internally (weight at close of d earns return over d+1), so we pass the
        # NATURAL (un-shifted) close-to-close returns here.
        rets_window = panel.rets.reindex(_extend_one_day(panel.dates, test_idx))
        bt = backtest_xsection(
            weights_over_time=weights,
            returns=rets_window,
            rebalance_freq=rebalance_freq,
            cost_bps_per_side=cost_bps,
        )
        # daily is indexed by REALIZATION date; the first realization is the day AFTER the
        # first weight day. Drop the leading flat day (no book carried in -> 0 return) so OOS
        # days reflect only days the strategy was actually invested.
        daily = bt["daily"]
        if len(daily) and float(daily.iloc[0]) == 0.0 and float(bt["turnover"].iloc[0]) == 0.0:
            daily = daily.iloc[1:]
        m = bt["metrics"]
        tot_ret = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
        if tot_ret > 0:
            positive += 1
        oos_daily_pieces.append(daily)
        total_turnover_w += m["avg_turnover"] * m["n_days"]
        total_days += m["n_days"]

        spy_fold = spy.reindex(daily.index).dropna() if len(daily) else spy.iloc[0:0]
        fb = beta_decompose(daily, spy_fold)
        fold_betas.append({"fold": f["fold"], **fb})
        fold_records.append(
            {
                "fold": f["fold"],
                "train": f["train"],
                "test": f["test"],
                "n_days": m["n_days"],
                "total_return": tot_ret,
                "annual_return": m["annual_return"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "avg_turnover": m["avg_turnover"],
                "chosen_params": chosen,
            }
        )

    oos_daily = pd.concat(oos_daily_pieces).sort_index() if oos_daily_pieces else pd.Series(dtype=float)
    oos_daily = oos_daily[~oos_daily.index.duplicated(keep="first")]

    n = len(oos_daily)
    oos_mean = float(oos_daily.mean()) if n else 0.0
    oos_std = float(oos_daily.std(ddof=1)) if n > 1 else 0.0
    oos_sharpe = (oos_mean / oos_std * np.sqrt(TRADING_YEAR)) if oos_std > 0 else 0.0
    oos_annual = oos_mean * TRADING_YEAR
    if n:
        eq = (1.0 + oos_daily).cumprod()
        oos_max_dd = float((eq / eq.cummax() - 1.0).min())
    else:
        oos_max_dd = 0.0
    avg_turnover = (total_turnover_w / total_days) if total_days else 0.0

    spy_oos = spy.reindex(oos_daily.index).dropna() if n else spy.iloc[0:0]
    beta_full = beta_decompose(oos_daily, spy_oos)

    return {
        "n_folds": len(folds),
        "folds": fold_records,
        "folds_positive": positive,
        "oos_annual_return": oos_annual,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": oos_max_dd,
        "oos_total_return": float((1.0 + oos_daily).prod() - 1.0) if n else 0.0,
        "oos_n_days": int(n),
        "oos_avg_turnover": avg_turnover,
        "cost_bps_per_side": cost_bps,
        "beta_decompose": beta_full,
        "fold_betas": fold_betas,
        "config": {
            "top_q": top_q,
            "bottom_q": bottom_q,
            "rebalance_freq": rebalance_freq,
            "min_names_per_side": min_names_per_side,
            "train_frac": train_frac,
        },
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_factory(obj) -> bool:
    """A factory is a callable that takes (panel, fold_info); a factor_fn takes (panel).

    We mark factories explicitly via a .is_factory attribute (set by experiments). If the
    attribute is absent we assume it is a plain factor_fn.
    """
    return bool(getattr(obj, "is_factory", False))


def _slice_panel(panel: Panel, idx: pd.DatetimeIndex) -> Panel:
    """Return a Panel restricted to the given date index (keeps all symbols)."""
    idx = panel.dates.intersection(pd.DatetimeIndex(idx))
    return Panel(
        close=panel.close.reindex(idx),
        open=panel.open.reindex(idx),
        high=panel.high.reindex(idx),
        low=panel.low.reindex(idx),
        volume=panel.volume.reindex(idx),
        rets=panel.rets.reindex(idx),
        symbols=panel.symbols,
        dates=idx,
    )


def _extend_one_day(all_dates: pd.DatetimeIndex, window: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """window + the single trading date immediately AFTER window's last date (if any).

    Needed so a weight decided on the last test day can still earn the next day's return.
    """
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    last = window[-1]
    after = all_dates[all_dates > last]
    if len(after):
        return window.append(pd.DatetimeIndex([after[0]]))
    return window


__all__ = [
    "Panel",
    "load_daily",
    "spy_daily_returns",
    "cross_sectional_rank",
    "form_dollar_neutral_portfolio",
    "backtest_xsection",
    "walk_forward_folds",
    "beta_decompose",
    "evaluate_walkforward",
]
