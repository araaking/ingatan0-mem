"""Command line entry point: `ingatan` (bare: the MCP server), `ingatan <command>`."""

from __future__ import annotations

import contextlib
import sys

from ingatan.cli.commands import app
from ingatan.errors import IngatanError

__all__ = ["app", "main"]


def _use_utf8() -> None:
    """NFR-17: Indonesian text must not crash or turn into mojibake on a Windows console.

    stdin too, because hooks receive their JSON there.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _use_utf8()
    try:
        app(prog_name="ingatan")
    except IngatanError as exc:
        sys.stderr.write(f"ingatan: {exc.message}\n")
        raise SystemExit(1) from None
