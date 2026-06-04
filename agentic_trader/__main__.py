"""Package entrypoint so `python3 -m agentic_trader ...` runs the CLI.

`python3 -m agentic_trader.cli ...` works via cli.py's own __main__ block; this module
makes the shorter `python3 -m agentic_trader ...` form work identically.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
