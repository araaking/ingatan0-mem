"""Lightweight relation edges between facts (`related_to`, `part_of`) and `pin` curation."""

from __future__ import annotations

from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.facts._util import is_closed, now_iso, resolve_conflicts
from ingatan.scope import Context, find_fact

__all__ = ["RELATIONS", "link_facts", "related_ids", "set_pinned", "unlink_facts"]

RELATIONS = ("related_to", "part_of")


def link_facts(ctx: Context, src_id: str, dst_id: str, *, relation: str = "related_to") -> bool:
    """Link two open, visible facts in `edges` (stored in `business.db`).

    Also resolves any open conflict pair between the two facts since they are explicitly
    complementary. Returns True if a new edge was inserted, False if it already existed.
    """
    if relation not in RELATIONS:
        raise ValidationError(f"Unknown relation '{relation}'. Use one of: {', '.join(RELATIONS)}.")
    if src_id == dst_id:
        raise ValidationError("A fact cannot be linked to itself.")
    src = find_fact(ctx, src_id)
    if src is None:
        raise NotFoundError(f"No fact {src_id}.")
    dst = find_fact(ctx, dst_id)
    if dst is None:
        raise NotFoundError(f"No fact {dst_id}.")
    if is_closed(src[1]):
        raise ValidationError(f"Fact {src_id} is closed and cannot be linked.")
    if is_closed(dst[1]):
        raise ValidationError(f"Fact {dst_id} is closed and cannot be linked.")

    now = now_iso()
    with transaction(ctx.business):
        cur = ctx.business.execute(
            "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, created_at) "
            "VALUES (?, ?, ?, ?)",
            (src_id, dst_id, relation, now),
        )
        resolve_conflicts(ctx, src_id, "link", only_with=[dst_id])
    return cur.rowcount > 0


def unlink_facts(ctx: Context, src_id: str, dst_id: str, *, relation: str | None = None) -> int:
    """Remove edges between `src_id` and `dst_id` (in either direction). Returns count removed."""
    if relation is not None and relation not in RELATIONS:
        raise ValidationError(f"Unknown relation '{relation}'. Use one of: {', '.join(RELATIONS)}.")
    with transaction(ctx.business):
        if relation is not None:
            cur = ctx.business.execute(
                "DELETE FROM edges WHERE relation = ? AND "
                "((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))",
                (relation, src_id, dst_id, dst_id, src_id),
            )
        else:
            cur = ctx.business.execute(
                "DELETE FROM edges WHERE "
                "(src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)",
                (src_id, dst_id, dst_id, src_id),
            )
    return cur.rowcount


def related_ids(ctx: Context, fact_id: str) -> list[tuple[str, str]]:
    """Return `(other_fact_id, relation)` pairs linked to `fact_id`."""
    rows = ctx.business.execute(
        "SELECT CASE WHEN src_id = :id THEN dst_id ELSE src_id END AS other, relation "
        "FROM edges WHERE src_id = :id OR dst_id = :id ORDER BY id ASC",
        {"id": fact_id},
    ).fetchall()
    return [(str(r["other"]), str(r["relation"])) for r in rows]


def set_pinned(ctx: Context, fact_id: str, pinned: bool) -> None:
    """Pin or unpin an open visible fact so it stays prioritized in `brief` and `recall`."""
    found = find_fact(ctx, fact_id)
    if found is None:
        raise NotFoundError(f"No fact {fact_id}.")
    conn, row = found
    if is_closed(row):
        raise ValidationError(f"Fact {fact_id} is closed.")
    with transaction(conn):
        conn.execute("UPDATE facts SET pinned = ? WHERE id = ?", (1 if pinned else 0, fact_id))
