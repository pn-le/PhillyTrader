"""config.py — defaults, .env loading, param loading, and Alpaca client helpers.

All paths are ABSOLUTE under /Users/pnle/Desktop/alpaca-cli. Defaults mirror
STRATEGY_RULES.md §0. The .env loader is dependency-free (no python-dotenv). The Alpaca
client helpers construct PAPER clients only (INV-3: there is no real-money mode).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Absolute paths (single source of truth for on-disk layout)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path("/Users/pnle/Desktop/alpaca-cli")
ENV_PATH = PROJECT_ROOT / ".env"
BEST_PARAMS_PATH = PROJECT_ROOT / "best_params.json"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ML_MODEL_PATH = MODELS_DIR / "ml_scorer.pkl"
ML_METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Fixed universe (10 symbols, FIXED ORDER — symbol_id == index, A8)
# --------------------------------------------------------------------------- #
UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN",
]

# Session / bar constants (STRATEGY_RULES §0)
SESSION_TZ = "America/New_York"
SESSION_OPEN = "09:30:00"
SESSION_CLOSE = "16:00:00"
EOD_FLATTEN_TIME = "15:55:00"  # A9: no new entries on/after; force-exit open positions
SESSION_MINUTES = 390  # 09:30..16:00 inclusive of open; minute_of_session range 0..389
RECENT_RETURN_K = 5  # A7
ROLLING_VOL_WINDOW = 20  # excludes current bar
MIN_BARS_FOR_ENTRY = ROLLING_VOL_WINDOW + 1  # A1: n >= 21
SLIPPAGE_BPS = 1  # A11: 1 basis point adverse slippage in backtest
MIN_NOTIONAL = 1.0  # A13: Alpaca minimum notional (~$1)

# ML / dataset backends (per ENV_REPORT.md). Modules fall back to numpy/csv/json if absent.
MODEL_BACKEND = "sklearn"  # LogisticRegression; numpy logreg fallback if sklearn missing
DATASET_FORMAT = "parquet"  # pyarrow; csv fallback if pyarrow missing
MODEL_PERSIST = "joblib"  # joblib.dump/load; json weight dump fallback if joblib missing


# --------------------------------------------------------------------------- #
# Trainable strategy parameters (the optimizer searches these)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyParams:
    """Trainable strategy params + fixed sizing. Defaults = STRATEGY_RULES §0.

    The optimizer searches: entry_dist, vol_mult, max_hold, vwap_exit_band, stop_loss,
    ml_threshold. `notional` is fixed sizing (not searched). All thresholds are fractions
    (0.005 == 0.5%) except max_hold (minutes) and notional (USD).
    """

    entry_dist: float = 0.005       # enter when price >= this fraction BELOW vwap
    vol_mult: float = 1.2           # volume_ratio must be >= this
    max_hold: float = 15.0          # minutes; force-exit when held strictly longer
    vwap_exit_band: float = 0.001   # exit when within this fraction of vwap
    stop_loss: float = 0.005        # exit when unrealized loss reaches this fraction
    ml_threshold: float = 0.0       # ML gate (disabled/passthrough at 0.0); range [0,1]
    notional: float = 100.0         # USD notional per BUY (fixed)

    def to_dict(self) -> Dict[str, float]:
        return {
            "entry_dist": self.entry_dist,
            "vol_mult": self.vol_mult,
            "max_hold": self.max_hold,
            "vwap_exit_band": self.vwap_exit_band,
            "stop_loss": self.stop_loss,
            "ml_threshold": self.ml_threshold,
            "notional": self.notional,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Build from a (possibly partial) dict; unknown keys ignored, missing use defaults."""
        base = cls()
        known = base.to_dict()
        merged = {k: float(d[k]) for k in known if k in d and d[k] is not None}
        return replace(base, **merged)

    # Names the optimizer is allowed to search (notional is fixed sizing, excluded).
    TRAINABLE = ("entry_dist", "vol_mult", "max_hold", "vwap_exit_band", "stop_loss", "ml_threshold")


# --------------------------------------------------------------------------- #
# Risk limits (Risk agent caps)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskLimits:
    """Hard risk caps enforced by the Risk agent (STRATEGY_RULES §8)."""

    max_open_positions: int = 4
    max_total_exposure: float = 500.0      # USD: open MV + pending-entry notional
    per_symbol_cooldown_min: float = 10.0  # at most 1 new ENTRY per symbol per window


# --------------------------------------------------------------------------- #
# Settings (runtime / environment configuration)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Runtime settings. `live_paper` is the in-config arm flag (default False = DRY-RUN).

    `feed` is the Alpaca data feed name ("iex" free default). `loop_interval_sec` is the
    live loop cadence. All paths are absolute and resolved at construction time.
    """

    universe: List[str] = field(default_factory=lambda: list(UNIVERSE))
    timezone: str = SESSION_TZ
    feed: str = "iex"                 # DataFeed.IEX value (free/paper tier)
    loop_interval_sec: float = 60.0   # one cycle per minute by default
    live_paper: bool = False          # ARM flag; False => DRY-RUN (INV-2 default)
    logs_dir: Path = LOGS_DIR
    models_dir: Path = MODELS_DIR
    data_cache_dir: Path = DATA_CACHE_DIR
    env_path: Path = ENV_PATH


# --------------------------------------------------------------------------- #
# .env loading (dependency-free; does NOT modify the .env file)
# --------------------------------------------------------------------------- #
def load_env(path: os.PathLike | str = ENV_PATH) -> Dict[str, str]:
    """Parse a KEY=VALUE .env file into a dict. No python-dotenv dependency.

    Ignores blank lines and # comments; strips surrounding quotes. Returns {} if the
    file is missing (callers should then fall back to os.environ).
    """
    env: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def get_credentials(path: os.PathLike | str = ENV_PATH) -> Dict[str, Any]:
    """Return {api_key, api_secret, paper:bool} sourced from .env then os.environ.

    `paper` is True iff ALPACA_PAPER == "true" (case-insensitive). This is the value
    enforced by INV-3: if it is not True, no order may ever be submitted.
    """
    env = load_env(path)

    def _get(key: str, default: str = "") -> str:
        return env.get(key, os.environ.get(key, default))

    paper_raw = _get("ALPACA_PAPER", "true")
    return {
        "api_key": _get("ALPACA_API_KEY_ID"),
        "api_secret": _get("ALPACA_API_SECRET"),
        "paper": str(paper_raw).strip().lower() == "true",
    }


def is_paper_env(path: os.PathLike | str = ENV_PATH) -> bool:
    """True iff ALPACA_PAPER == 'true'. INV-3 precondition for any submission."""
    return bool(get_credentials(path)["paper"])


# --------------------------------------------------------------------------- #
# Param loading
# --------------------------------------------------------------------------- #
def load_strategy_params(path: os.PathLike | str = BEST_PARAMS_PATH) -> StrategyParams:
    """Load tuned params from best_params.json if present, else return defaults.

    Tolerates partial files (missing keys fall back to defaults) and malformed JSON
    (logs nothing here — pure function — and returns defaults).
    """
    p = Path(path)
    if not p.exists():
        return StrategyParams()
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError):
        return StrategyParams()
    if not isinstance(data, dict):
        return StrategyParams()
    return StrategyParams.from_dict(data)


def save_best_params(params: StrategyParams, path: os.PathLike | str = BEST_PARAMS_PATH) -> Path:
    """Persist params to best_params.json (used by the optimizer). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params.to_dict(), indent=2))
    return p


# --------------------------------------------------------------------------- #
# Directory helpers
# --------------------------------------------------------------------------- #
def ensure_dirs(settings: Optional[Settings] = None) -> None:
    """Create logs/, models/, data/cache/ if missing. Safe to call repeatedly."""
    s = settings or Settings()
    for d in (s.logs_dir, s.models_dir, s.data_cache_dir):
        Path(d).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Alpaca client helpers (PAPER ONLY)
# --------------------------------------------------------------------------- #
def make_trading_client(path: os.PathLike | str = ENV_PATH):
    """Construct a PAPER alpaca TradingClient from .env credentials.

    Always paper=True (INV-3). Imports alpaca lazily so the contract modules stay
    importable without alpaca installed. Returns alpaca.trading.client.TradingClient.
    """
    from alpaca.trading.client import TradingClient  # lazy import

    creds = get_credentials(path)
    return TradingClient(creds["api_key"], creds["api_secret"], paper=True)


def make_data_client(path: os.PathLike | str = ENV_PATH):
    """Construct an alpaca StockHistoricalDataClient from .env credentials.

    Imports alpaca lazily. Returns alpaca.data.historical.StockHistoricalDataClient.
    Paper PK keys authenticate the free IEX historical feed (per API_MAP §5).
    """
    from alpaca.data.historical import StockHistoricalDataClient  # lazy import

    creds = get_credentials(path)
    return StockHistoricalDataClient(creds["api_key"], creds["api_secret"])


def make_clients(path: os.PathLike | str = ENV_PATH) -> Dict[str, Any]:
    """Return {"trading": TradingClient, "data": StockHistoricalDataClient, "paper": bool}.

    Convenience bundle the orchestrator/loop pass around as `clients`.
    """
    return {
        "trading": make_trading_client(path),
        "data": make_data_client(path),
        "paper": is_paper_env(path),
    }


__all__ = [
    "PROJECT_ROOT", "ENV_PATH", "BEST_PARAMS_PATH", "LOGS_DIR", "MODELS_DIR",
    "DATA_CACHE_DIR", "ML_MODEL_PATH", "ML_METRICS_PATH",
    "UNIVERSE", "SESSION_TZ", "SESSION_OPEN", "SESSION_CLOSE", "EOD_FLATTEN_TIME",
    "SESSION_MINUTES", "RECENT_RETURN_K", "ROLLING_VOL_WINDOW", "MIN_BARS_FOR_ENTRY",
    "SLIPPAGE_BPS", "MIN_NOTIONAL", "MODEL_BACKEND", "DATASET_FORMAT", "MODEL_PERSIST",
    "StrategyParams", "RiskLimits", "Settings",
    "load_env", "get_credentials", "is_paper_env",
    "load_strategy_params", "save_best_params", "ensure_dirs",
    "make_trading_client", "make_data_client", "make_clients",
]
