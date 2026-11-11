"""Append-only log of every change: write, read, scrub, diff and undo."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from ingatan import timeutil
from ingatan.errors import NotFoundError
from ingatan.scope import Context

__all__ = ["OPS", "Diff", "LogEntry", "diff", "entries", "get", "scrub", "undo", "write"]

OPS = frozenset(
    {"create", "supersede", "confirm", "forget", "propose", "approve", "reject", "ingest", "undo"}
)


@dataclass(frozen=True)
class LogEntry:
    id: int
    ts: str
    agent: str
    op: str
    fact_id: str | None
    tier: str | None
    owner: str | None
    # {"before": {...} | None, "after": {...} | None}, or None for a content-free entry
    change: dict[str, Any] | None


@dataclass(frozen=True)
class Diff:
    log_id: int
    op: str
    fact_id: str | None
    before: dict[str, Any] | None
    after: dict[str, Any] | None


def write(
    conn: sqlite3.Connection,
    *,
    agent: str,
    op: str,
    fact_id: str | None,
    tier: str | None,
    owner: str | None,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> int:
    """Append one entry and return its id. Must run inside `db.transaction`.

    `before` and `after` both None gives a content-free entry (`change` is NULL).
    """
    if not conn.in_transaction:
        raise RuntimeError("audit.write must be called inside db.transaction(conn)")
    if op not in OPS:
        raise ValueError(f"Unknown log op '{op}'. Known ops: {', '.join(sorted(OPS))}")
    change = (
        None
        if before is None and after is None
        else json.dumps({"before": before, "after": after}, ensure_ascii=False)
    )
    cursor = conn.execute(
        "INSERT INTO log (ts, agent, op, fact_id, tier, owner, change) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timeutil.iso(timeutil.now()), agent, op, fact_id, tier, owner, change),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _entry(row: sqlite3.Row) -> LogEntry:
    raw = row["change"]
    return LogEntry(
        id=row["id"],
        ts=row["ts"],
        agent=row["agent"],
        op=row["op"],
        fact_id=row["fact_id"],
        tier=row["tier"],
        owner=row["owner"],
        change=None if raw is None else json.loads(raw),
    )


def entries(
    ctx: Context,
    *,
    fact_id: str | None = None,
    fact_ids: list[str] | None = None,
    limit: int | None = None,
    newest_first: bool = False,
) -> list[LogEntry]:
    """Log entries the caller may see, from both files, ordered by time then id."""
    wanted: list[str] | None = None
    if fact_id is not None or fact_ids is not None:
        wanted = ([fact_id] if fact_id is not None else []) + list(fact_ids or [])
    direction = "DESC" if newest_first else "ASC"
    found: list[LogEntry] = []
    for conn in ctx.connections():
        where = ctx.visible_sql(conn, owner_column="owner")
        params: dict[str, Any] = dict(ctx.params)
        sql = f"SELECT * FROM log WHERE {where}"
        if wanted is not None:
            sql += " AND fact_id IN (SELECT value FROM json_each(:ids))"
            params["ids"] = json.dumps(wanted)
        sql += f" ORDER BY ts {direction}, id {direction}"
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = limit
        found.extend(_entry(row) for row in conn.execute(sql, params))
    found.sort(key=lambda e: (e.ts, e.id), reverse=newest_first)
    return found[:limit] if limit is not None else found


def scrub(conn: sqlite3.Connection, fact_id: str) -> None:
    """Clear `change` on every entry of this fact. Used only by `forget`."""
    conn.execute("UPDATE log SET change = NULL WHERE fact_id = ?", (fact_id,))


def get(ctx: Context, log_id: int) -> LogEntry:
    """One entry by id. Missing and invisible entries both raise `NotFoundError`."""
    for conn in ctx.connections():
        where = ctx.visible_sql(conn, owner_column="owner")
        row = conn.execute(
            f"SELECT * FROM log WHERE id = :id AND {where}", {"id": log_id, **ctx.params}
        ).fetchone()
        if row is not None:
            return _entry(row)
    raise NotFoundError(f"No log entry {log_id}.")


def diff(ctx: Context, log_id: int) -> Diff:
    """Before and after of one entry. Both are None for a scrubbed or content-free entry."""
    entry = get(ctx, log_id)
    change = entry.change or {}
    return Diff(
        log_id=entry.id,
        op=entry.op,
        fact_id=entry.fact_id,
        before=change.get("before"),
        after=change.get("after"),
    )


def undo(ctx: Context, log_id: int) -> int:
    """Reverse one entry and return the id of the new `undo` entry (OQ-16).

    Handles create, supersede, confirm and approve. Refuses a forget ("a forget cannot be
    undone"), an entry with later changes to the same fact, and every other op.
    """
    from ingatan.audit._undo import undo_entry  # lazy: _undo imports this package

    return undo_entry(ctx, log_id)
