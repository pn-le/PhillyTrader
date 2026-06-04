"""agentic_trader — Intraday VWAP mean-reversion paper trader (deterministic).

This package is the implementation of STRATEGY_RULES.md + ARCHITECTURE.md.
PAPER ONLY. There is NO real-money mode. The ML scorer is a learned, deterministic
gate on ENTRIES only; the trade loop itself contains no LLM.

The single source of truth for inter-module signatures is INTERFACE_SPEC.md.
The shared contract types live in `agentic_trader.types`; configuration in
`agentic_trader.config`; structured logging in `agentic_trader.logging_util`.
"""

__version__ = "0.1.0"
