"""Child process: open the data folder at a given moment, then exit (0 = it opened).

Usage: python _opener.py <home> <business> <agent> <start_epoch>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ingatan.scope import Scope, open_context


def main() -> int:
    home, business, agent, start = sys.argv[1:5]
    delay = float(start) - time.time()
    if delay > 0:
        time.sleep(delay)
    try:
        open_context(Scope(business=business, agent=agent, home=Path(home))).close()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
