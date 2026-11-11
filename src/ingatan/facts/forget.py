"""`forget`: remove a fact for good, down to the bytes on disk (FR-41 to FR-43, FR-61).

What makes the bytes go away:

- `PRAGMA secure_delete=ON` (set by `db.connect`) zeroes deleted rows and freed pages.
- The FTS5 `secure-delete` option (set in migration 0001) removes deleted terms from the
  index pages instead of leaving tombstones.
- `optimize` rewrites the index segments so no old segment keeps the terms.
- `wal_checkpoint(TRUNCATE)` copies the zeroed pages into the database file and empties the
  write-ahead log, which otherwise still holds the old page images.

No VACUUM is needed: free pages are zeroed on the spot and nothing readable stays in them.
"""

from __future__ import annotations

import json
import sqlite3
import time

from ingatan import audit
from ingatan.db import transaction
from ingatan.errors import IngatanError, NotFoundError, ScopeError
from ingatan.facts._util import now_iso
from ingatan.scope import Context, find_fact

__all__ = ["forget"]

_CHECKPOINT_TRIES = 5


def forget(ctx: Context, fact_id: str, *, by_user: bool = False) -> None:
    """Delete a fact and every trace of its content.

    A `business` or `global` fact needs `by_user=True` (the CLI); an agent gets `ScopeError`.
    Facts that pointed at this one through `superseded_by` stay closed (OQ-17).
    """
    found = find_fact(ctx, fact_id)
    if found is None:
        raise NotFoundError(f"No fact {fact_id}.")
    conn, row = found
    if row["tier"] != "agent" and not by_user:
        raise ScopeError(
            f"{fact_id} is a {row['tier']} fact. Only the user can forget it, with the "
            "ingatan CLI. Nothing was changed."
        )

    dirty: list[sqlite3.Connection] = []
    # Work in the other file first: if something fails, the fact is still there and a
    # second forget can finish the job.
    for other in ctx.connections():
        if other is conn:
            continue
        with transaction(other):
            if _scrub_proposals(other, fact_id):
                dirty.append(other)
    with transaction(ctx.business):
        ctx.business.execute(
            "DELETE FROM conflicts WHERE fact_a = ? OR fact_b = ?", (fact_id, fact_id)
        )
        episodes_gone = _delete_episodes(ctx, row, by_user)
    with transaction(conn):
        _scrub_proposals(conn, fact_id)
        audit.scrub(conn, fact_id)
        conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="forget",
            fact_id=fact_id,
            tier=row["tier"],
            owner=row["agent"],
            before=None,
            after=None,
        )
    if conn not in dirty:
        dirty.append(conn)
    if episodes_gone and ctx.business not in dirty:
        dirty.append(ctx.business)
    for target in dirty:
        _purge(target)


def _delete_episodes(ctx: Context, row: sqlite3.Row, by_user: bool) -> bool:
    """Delete the episode this fact was digested from (its `source.episode`). True if any."""
    try:
        source = json.loads(row["source"])
    except (TypeError, ValueError):
        return False
    episode = source.get("episode") if isinstance(source, dict) else None
    if isinstance(episode, bool) or not isinstance(episode, int):
        return False
    if by_user:
        cursor = ctx.business.execute("DELETE FROM episodes WHERE id = ?", (episode,))
    else:
        cursor = ctx.business.execute(
            "DELETE FROM episodes WHERE id = ? AND agent = ?", (episode, ctx.scope.agent)
        )
    return cursor.rowcount > 0


def _scrub_proposals(conn: sqlite3.Connection, fact_id: str) -> bool:
    """Remove the fact's content from proposals and their log entries. True if any changed."""
    rows = conn.execute(
        "SELECT id, status, supersedes, fact_id, json_extract(payload, '$.origin_fact') AS origin "
        "FROM proposals WHERE fact_id = :id OR origin = :id OR supersedes = :id",
        {"id": fact_id},
    ).fetchall()
    touched = False
    for p in rows:
        if p["fact_id"] == fact_id or p["origin"] == fact_id:
            # This proposal holds a copy of the text (it created the fact, or it was copied
            # from it). Empty the payload, close it if pending and let go of its target.
            conn.execute(
                "UPDATE proposals SET payload = '{}', "
                "status = CASE WHEN status = 'pending' THEN 'rejected' ELSE status END, "
                "decided_at = CASE WHEN status = 'pending' THEN ? ELSE decided_at END "
                "WHERE id = ?",
                (now_iso(), p["id"]),
            )
            audit.scrub(conn, p["id"])
            if p["supersedes"]:
                conn.execute(
                    "UPDATE facts SET pending_proposal = NULL "
                    "WHERE id = ? AND pending_proposal = ?",
                    (p["supersedes"], p["id"]),
                )
            touched = True
        elif p["supersedes"] == fact_id:
            # A pending proposal that wanted to replace this fact keeps no dangling link.
            conn.execute("UPDATE proposals SET supersedes = NULL WHERE id = ?", (p["id"],))
    return touched


def _purge(conn: sqlite3.Connection) -> None:
    """Make deleted bytes unrecoverable in the database file and the WAL file."""
    for table in ("facts_fts", "facts_tri"):
        conn.execute(f"INSERT INTO {table} ({table}) VALUES ('optimize')")
    for attempt in range(_CHECKPOINT_TRIES):
        busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        if not busy:
            return
        time.sleep(0.2 * (attempt + 1))
    raise IngatanError(
        "The fact was deleted, but another process kept the database busy, so the old "
        "bytes may still be in its write-ahead log until that process finishes. "
        "Close the other Ingatan processes and run a forget of any other fact, or restart."
    )
