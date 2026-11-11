"""`history`: log entries for one fact id or for every fact linked to an entity."""

from __future__ import annotations

from ingatan import audit
from ingatan.errors import NotFoundError
from ingatan.facts._util import loads_list
from ingatan.scope import Context

__all__ = ["history"]


def history(ctx: Context, id_or_entity: str) -> list[audit.LogEntry]:
    """Oldest first, scope filtered. An id the caller cannot see is `NotFoundError`."""
    key = id_or_entity.strip()
    if key.startswith(("f_", "p_")):
        return _by_id(ctx, key)
    return _by_entity(ctx, key)


def _by_id(ctx: Context, key: str) -> list[audit.LogEntry]:
    if key.startswith("p_") and not _own_proposal(ctx, key):
        raise NotFoundError(f"No history for {key}.")
    found = {e.id: e for e in audit.entries(ctx, fact_id=key)}
    # Approval and rejection entries belong to the proposal or the new fact, but they also
    # changed this fact (closed it, cleared its pending mark). They name it in `affects`.
    for conn in ctx.connections():
        rows = conn.execute(
            f"SELECT id FROM log WHERE {ctx.visible_sql(conn, owner_column='owner')} "
            "AND EXISTS (SELECT 1 FROM json_each(json_extract(change, '$.after.affects')) "
            "WHERE value = :key)",
            {"key": key, **ctx.params},
        ).fetchall()
        for r in rows:
            found.setdefault(r["id"], audit.get(ctx, r["id"]))
    if not found:
        raise NotFoundError(f"No history for {key}.")
    return sorted(found.values(), key=lambda e: (e.ts, e.id))


def _own_proposal(ctx: Context, proposal_id: str) -> bool:
    return any(
        conn.execute(
            "SELECT 1 FROM proposals WHERE id = ? AND agent = ?", (proposal_id, ctx.scope.agent)
        ).fetchone()
        for conn in ctx.connections()
    )


def _by_entity(ctx: Context, name: str) -> list[audit.LogEntry]:
    wanted = name.casefold()
    if not wanted:
        return []
    ids: list[str] = []
    for conn in ctx.connections():
        rows = conn.execute(
            f"SELECT id, entities FROM facts WHERE {ctx.visible_sql(conn)}", ctx.params
        ).fetchall()
        ids.extend(
            r["id"]
            for r in rows
            if wanted in {e.casefold().strip() for e in loads_list(r["entities"])}
        )
    if not ids:
        return []
    return audit.entries(ctx, fact_ids=ids)
