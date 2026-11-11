"""Child process for the concurrency test: write N agent-tier facts, then exit.

Usage: python _writer.py <home> <business> <agent> <count> <start_epoch>
Both children sleep until `start_epoch` so that their writes overlap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ingatan import facts
from ingatan.scope import Scope, open_context


def main() -> int:
    home, business, agent, count, start = sys.argv[1:6]
    ctx = open_context(Scope(business=business, agent=agent, home=Path(home)))
    try:
        delay = float(start) - time.time()
        if delay > 0:
            time.sleep(delay)
        for n in range(int(count)):
            facts.remember(
                ctx,
                f"{agent} catatan nomor {n}: pelanggan membeli kopi",
                keywords=["catatan", "pelanggan", "kopi"],
                kind="stable",
            )
    finally:
        ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
