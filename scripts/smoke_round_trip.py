"""Save one fact, find it again, remove it: proof that an installed Ingatan works.

    INGATAN_HOME=/some/throwaway/folder python scripts/smoke_round_trip.py

CI runs this with the Python of the built wheel, in an empty folder, right after
`ingatan setup`. It uses the business and agent that folder is configured for. It refuses to
run unless INGATAN_HOME is set, so it can never write into a real `~/.ingatan`, and it
forgets its one fact before it exits. No embeddings, no network.
"""

from __future__ import annotations

import os
import sys

TEXT = "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler."


def main() -> int:
    if not (os.environ.get("INGATAN_HOME") or "").strip():
        print(
            "Set INGATAN_HOME to a throwaway folder first: this script writes a fact.",
            file=sys.stderr,
        )
        return 2

    from ingatan import facts, search
    from ingatan.scope import open_context, resolve_scope

    with open_context(resolve_scope()) as ctx:
        saved = facts.remember(
            ctx, TEXT, entities=["ongkir Bandung"], keywords=["ongkos kirim", "tarif"]
        )
        if saved.fact_id is None:
            print("remember did not save a fact", file=sys.stderr)
            return 1
        try:
            hits = search.recall(ctx, "ongkir Bandung").facts
            found = any(hit.id == saved.fact_id for hit in hits)
        finally:
            facts.forget(ctx, saved.fact_id)
    if not found:
        print("recall did not return the fact that was just saved", file=sys.stderr)
        return 1
    print("round trip ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
