"""Project-scoped terminology aliases (`aliases` table in `<business>.db`).

Maps local abbreviations or nicknames (e.g. `"LP"` <-> `"landing page"`) so recall queries
expand deterministically with zero latency and zero LLM calls.
"""

from __future__ import annotations

import re

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.errors import ValidationError
from ingatan.redact import redact
from ingatan.scope import Context

__all__ = ["add_alias", "expand_query_with_aliases", "list_aliases", "remove_alias"]


def _clean_phrase(value: str, label: str) -> str:
    cleaned = " ".join(redact(value or "").split()).strip()
    if not cleaned:
        raise ValidationError(f"Alias {label} cannot be empty.")
    return cleaned


def add_alias(ctx: Context, term: str, canonical: str) -> tuple[str, str]:
    """Add an alias mapping `(term, canonical)` in this business database. Idempotent."""
    t = _clean_phrase(term, "term").casefold()
    c = _clean_phrase(canonical, "canonical").casefold()
    if t == c:
        raise ValidationError("Alias term and canonical phrase must be different.")
    now = timeutil.iso(timeutil.now())
    with transaction(ctx.business):
        ctx.business.execute(
            "INSERT OR IGNORE INTO aliases (term, canonical, created_at) VALUES (?, ?, ?)",
            (t, c, now),
        )
    return t, c


def remove_alias(ctx: Context, term: str, canonical: str | None = None) -> int:
    """Remove matching alias row(s). Returns how many rows were removed."""
    t = _clean_phrase(term, "term").casefold()
    with transaction(ctx.business):
        if canonical is not None and canonical.strip():
            c = _clean_phrase(canonical, "canonical").casefold()
            cur = ctx.business.execute(
                "DELETE FROM aliases WHERE (term = ? AND canonical = ?) "
                "OR (term = ? AND canonical = ?)",
                (t, c, c, t),
            )
        else:
            cur = ctx.business.execute(
                "DELETE FROM aliases WHERE term = ? OR canonical = ?", (t, t)
            )
    return cur.rowcount


def list_aliases(ctx: Context) -> list[tuple[str, str]]:
    """All alias mappings `(term, canonical)` for this business, sorted."""
    rows = ctx.business.execute(
        "SELECT term, canonical FROM aliases ORDER BY term ASC, canonical ASC"
    ).fetchall()
    return [(str(r["term"]), str(r["canonical"])) for r in rows]


def expand_query_with_aliases(ctx: Context, query: str) -> list[str]:
    """Return alias expansion phrases matching `query` (bidirectional whole-word/phrase match)."""
    lowered = query.casefold()
    if not lowered.strip():
        return []
    rows = ctx.business.execute("SELECT term, canonical FROM aliases").fetchall()
    if not rows:
        return []
    extra: list[str] = []
    seen: set[str] = set()
    for row in rows:
        term = str(row["term"])
        canonical = str(row["canonical"])
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered) and canonical not in seen:
            seen.add(canonical)
            extra.append(canonical)
        if re.search(rf"(?<!\w){re.escape(canonical)}(?!\w)", lowered) and term not in seen:
            seen.add(term)
            extra.append(term)
    return extra
