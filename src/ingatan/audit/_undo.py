"""`audit.undo`: reverse one log entry (OQ-16, FR-39, FR-40).

The code works on plain connections and SQL because `audit` sits below `facts` in the
module order. It reverses exactly what `facts.create`, `facts.change` and `proposals`
put into the log entry's `change` JSON, so it reads those keys by name.

Decisions:

- Undoing a `create` (or an `approve`) *closes* the fact (`valid_to = now`) and does not
  delete the row. A closed fact is never recalled, its id still resolves, `history` by id
  and by entity still shows the whole story (create, undo) and nothing in the database
  disappears except through `forget`. The conflict pairs of the closed fact are resolved
  with resolution `undo`.
- An entry can only be undone while nothing built on it: no later entry may mention the
  same fact or a fact/proposal the entry touched. Later entries that were themselves
  undone do not count, so changes can be unwound newest first.
- One new `undo` entry is written, in the file of the entry it reverses. The old entry is
  never edited.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ingatan import audit, timeutil
from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.scope import Context

FORGET_MESSAGE = "a forget cannot be undone"

_REFUSED = {
    "undo": "an undo cannot be undone. Repeat the original change if you want it back.",
    "propose": (
        "a proposal cannot be undone here. Reject it with `ingatan proposals reject <id>` instead."
    ),
    "reject": ("a rejected proposal cannot be reopened. Ask the agent to propose the fact again."),
    "ingest": "an ingest cannot be undone. Change the file and run `ingatan ingest` again.",
}


def undo_entry(ctx: Context, log_id: int) -> int:
    conn, entry = _locate(ctx, log_id)
    if entry.op == "forget":
        raise ValidationError(FORGET_MESSAGE)
    if entry.op in _REFUSED:
        raise ValidationError(f"Log entry {log_id} is a {entry.op}: {_REFUSED[entry.op]}")
    if entry.op not in _HANDLERS:
        raise ValidationError(f"Log entry {log_id} ({entry.op}) cannot be undone.")
    if entry.change is None or entry.fact_id is None:
        raise ValidationError(
            f"Log entry {log_id} has no content (the fact was forgotten), so it cannot be undone."
        )
    change = entry.change
    before: dict[str, Any] = change.get("before") or {}
    after: dict[str, Any] = change.get("after") or {}

    already = _undone_by(ctx, log_id)
    if already is not None:
        raise ValidationError(f"Log entry {log_id} was already undone by entry {already}.")
    related = _related_ids(entry.op, entry.fact_id, before, after)
    blocking = _later_entries(ctx, conn, entry.id, entry.ts, related)
    if blocking:
        listed = ", ".join(f"{e_id} ({op})" for e_id, op in blocking)
        raise ValidationError(
            f"Log entry {log_id} cannot be undone: later entries changed the same fact: "
            f"{listed}. Undo those first, newest first."
        )

    # Conflict pairs live in the business file, even for a global fact.
    with transaction(ctx.business), transaction(conn):
        undo_before, undo_after = _HANDLERS[entry.op](ctx, conn, entry, before, after)
        undo_after = {**undo_after, "undoes": entry.id, "undone_op": entry.op}
        return audit.write(
            conn,
            agent=ctx.scope.agent,
            op="undo",
            fact_id=entry.fact_id,
            tier=entry.tier,
            owner=entry.owner,
            before=undo_before,
            after=undo_after,
        )


# ---- finding the entry and what depends on it ------------------------------------------


def _locate(ctx: Context, log_id: int) -> tuple[sqlite3.Connection, audit.LogEntry]:
    """The entry and the file it lives in. Invisible entries behave as missing."""
    for conn in ctx.connections():
        where = ctx.visible_sql(conn, owner_column="owner")
        row = conn.execute(
            f"SELECT * FROM log WHERE id = :id AND {where}", {"id": log_id, **ctx.params}
        ).fetchone()
        if row is not None:
            return conn, audit._entry(row)  # pyright: ignore[reportPrivateUsage]
    raise NotFoundError(f"No log entry {log_id}.")


def _undone_by(ctx: Context, log_id: int) -> int | None:
    for conn in ctx.connections():
        row = conn.execute(
            "SELECT id FROM log WHERE op = 'undo' "
            "AND json_extract(change, '$.after.undoes') = ? LIMIT 1",
            (log_id,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
    return None


def _related_ids(op: str, fact_id: str, before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Fact and proposal ids whose later changes would make this undo unsafe."""
    ids = {fact_id}
    if op == "approve":
        ids.add(str(after.get("proposal_id")))
        closed = after.get("closed")
        if isinstance(closed, dict):
            ids.add(str(closed.get("fact_id")))  # pyright: ignore[reportUnknownArgumentType]
    elif op == "supersede" and "pending_proposal" in after:
        ids.add(str(after["pending_proposal"]))
    elif op == "confirm" and before.get("pending_proposal"):
        ids.add(str(before["pending_proposal"]))
    ids.discard("None")
    return ids


def _later_entries(
    ctx: Context, conn: sqlite3.Connection, log_id: int, ts: str, related: set[str]
) -> list[tuple[int, str]]:
    """Later entries that mention a related id and were not themselves undone."""
    ids = json.dumps(sorted(related))
    mentions = (
        "fact_id IN (SELECT value FROM json_each(:ids)) "
        "OR json_extract(change, '$.after.superseded_by') IN (SELECT value FROM json_each(:ids)) "
        "OR json_extract(change, '$.after.proposal_from_fact') "
        "IN (SELECT value FROM json_each(:ids)) "
        "OR json_extract(change, '$.after.proposal_id') IN (SELECT value FROM json_each(:ids)) "
        "OR json_extract(change, '$.after.closed.fact_id') IN (SELECT value FROM json_each(:ids)) "
        "OR EXISTS (SELECT 1 FROM json_each(change, '$.after.affects') "
        "WHERE value IN (SELECT value FROM json_each(:ids)))"
    )
    found: list[tuple[int, str, str | None]] = []
    undone: set[int] = set()
    for other in ctx.connections():
        # Ids order entries inside one file. Across files only the timestamp can.
        later = "id > :id" if other is conn else "ts > :ts"
        for row in other.execute(
            f"SELECT id, op, change FROM log WHERE {later} AND ({mentions})",
            {"id": log_id, "ts": ts, "ids": ids},
        ):
            found.append((row["id"], row["op"], row["change"]))
            if row["op"] == "undo" and row["change"]:
                target = (json.loads(row["change"]).get("after") or {}).get("undoes")
                if isinstance(target, int):
                    undone.add(target)
    return sorted((e_id, op) for e_id, op, _ in found if op != "undo" and e_id not in undone)


# ---- one handler per operation ---------------------------------------------------------

Handler = Any


def _fact(conn: sqlite3.Connection, fact_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    if row is None:
        raise ValidationError(f"Fact {fact_id} no longer exists, so there is nothing to undo.")
    return row


def _resolve_pairs(ctx: Context, fact_id: str, now: str) -> list[int]:
    rows = ctx.business.execute(
        "SELECT id FROM conflicts WHERE resolved_at IS NULL AND (fact_a = :id OR fact_b = :id)",
        {"id": fact_id},
    ).fetchall()
    ids = [int(r["id"]) for r in rows]
    for pair_id in ids:
        ctx.business.execute(
            "UPDATE conflicts SET resolved_at = ?, resolution = 'undo' WHERE id = ?",
            (now, pair_id),
        )
    return ids


def _reopen_pairs(ctx: Context, pair_ids: Any, resolution: str) -> list[int]:
    reopened: list[int] = []
    for pair_id in pair_ids if isinstance(pair_ids, list) else []:
        cursor = ctx.business.execute(
            "UPDATE conflicts SET resolved_at = NULL, resolution = NULL "
            "WHERE id = ? AND resolved_at IS NOT NULL AND resolution = ?",
            (pair_id, resolution),
        )
        if cursor.rowcount:
            reopened.append(int(pair_id))
    return reopened


def _close(conn: sqlite3.Connection, row: sqlite3.Row, what: str, now: str) -> None:
    if row["superseded_by"] is not None or (row["valid_to"] is not None and row["valid_to"] <= now):
        raise ValidationError(
            f"Fact {row['id']} is already closed, so the {what} cannot be undone."
        )
    conn.execute("UPDATE facts SET valid_to = ? WHERE id = ?", (now, row["id"]))


def _undo_create(
    ctx: Context, conn: sqlite3.Connection, entry: audit.LogEntry, before: Any, after: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert entry.fact_id is not None
    now = timeutil.iso(timeutil.now())
    row = _fact(conn, entry.fact_id)
    _close(conn, row, "create", now)
    resolved = _resolve_pairs(ctx, entry.fact_id, now)
    return (
        {"valid_to": row["valid_to"]},
        {"valid_to": now, "resolved_conflicts": resolved},
    )


def _undo_approve(
    ctx: Context, conn: sqlite3.Connection, entry: audit.LogEntry, before: Any, after: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert entry.fact_id is not None
    now = timeutil.iso(timeutil.now())
    proposal_id = after.get("proposal_id")
    proposal = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    if proposal is None or proposal["status"] != "approved":
        raise ValidationError(f"Proposal {proposal_id} is no longer in the approved state.")
    row = _fact(conn, entry.fact_id)

    reopened: dict[str, Any] | None = None
    closed = after.get("closed")
    if isinstance(closed, dict):
        target_id = str(closed["fact_id"])  # pyright: ignore[reportUnknownArgumentType]
        target = conn.execute("SELECT * FROM facts WHERE id = ?", (target_id,)).fetchone()
        if target is not None:
            expected: dict[str, Any] = closed.get("after") or {}  # pyright: ignore[reportUnknownMemberType]
            for key in ("valid_to", "superseded_by"):
                if target[key] != expected.get(key):
                    raise ValidationError(
                        f"Fact {target_id} changed since the approval, so it cannot be reopened."
                    )
            old: dict[str, Any] = closed.get("before") or {}  # pyright: ignore[reportUnknownMemberType]
            conn.execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ?, pending_proposal = ? "
                "WHERE id = ?",
                (
                    old.get("valid_to"),
                    old.get("superseded_by"),
                    old.get("pending_proposal"),
                    target_id,
                ),
            )
            reopened = {"fact_id": target_id, **old}

    _close(conn, row, "approval", now)
    # The proposal is pending again. It keeps its fact_id on purpose: a later forget of
    # the (closed) fact then still finds the proposal that holds a copy of its text.
    conn.execute(
        "UPDATE proposals SET status = 'pending', decided_at = NULL WHERE id = ?", (proposal_id,)
    )
    resolved = _resolve_pairs(ctx, entry.fact_id, now)
    undo_after: dict[str, Any] = {
        "valid_to": now,
        "proposal_id": proposal_id,
        "proposal_status": "pending",
        "resolved_conflicts": resolved,
    }
    if reopened is not None:
        undo_after["reopened"] = reopened
        undo_after["affects"] = [reopened["fact_id"]]
    return (
        {"valid_to": row["valid_to"], "proposal_status": "approved"},
        undo_after,
    )


def _undo_supersede(
    ctx: Context, conn: sqlite3.Connection, entry: audit.LogEntry, before: Any, after: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert entry.fact_id is not None
    row = _fact(conn, entry.fact_id)
    if "pending_proposal" in after:
        return _undo_queued_supersede(conn, row, before, after)
    # Agent-tier flow: the old fact was closed.
    if row["valid_to"] != after.get("valid_to") or row["superseded_by"] != after.get(
        "superseded_by"
    ):
        raise ValidationError(
            f"Fact {row['id']} changed since the supersede, so it cannot be undone."
        )
    conn.execute(
        "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
        (before.get("valid_to"), before.get("superseded_by"), row["id"]),
    )
    reopened = _reopen_pairs(ctx, after.get("resolved_conflicts"), "supersede")
    return (
        {"valid_to": row["valid_to"], "superseded_by": row["superseded_by"]},
        {
            "valid_to": before.get("valid_to"),
            "superseded_by": before.get("superseded_by"),
            "reopened_conflicts": reopened,
        },
    )


def _undo_queued_supersede(
    conn: sqlite3.Connection, row: sqlite3.Row, before: Any, after: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """FR-59 flow: nothing was closed. A proposal was linked, or created, and the fact
    was marked `check first`."""
    proposal_id = after.get("pending_proposal")
    if row["pending_proposal"] != proposal_id:
        raise ValidationError(
            f"Fact {row['id']} changed since the supersede, so it cannot be undone."
        )
    proposal = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    if proposal is not None:
        if proposal["status"] != "pending":
            raise ValidationError(f"Proposal {proposal_id} is already {proposal['status']}.")
        if after.get("proposal_from_fact"):
            # This supersede created the proposal itself (no other log entry mentions it).
            conn.execute("DELETE FROM proposals WHERE id = ?", (proposal_id,))
        else:
            conn.execute("UPDATE proposals SET supersedes = NULL WHERE id = ?", (proposal_id,))
    conn.execute(
        "UPDATE facts SET pending_proposal = ? WHERE id = ?",
        (before.get("pending_proposal"), row["id"]),
    )
    return (
        {"pending_proposal": proposal_id},
        {
            "pending_proposal": before.get("pending_proposal"),
            "proposal_removed": bool(after.get("proposal_from_fact")),
        },
    )


def _undo_confirm(
    ctx: Context, conn: sqlite3.Connection, entry: audit.LogEntry, before: Any, after: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert entry.fact_id is not None
    row = _fact(conn, entry.fact_id)
    if row["review_after"] != after.get("review_after"):
        raise ValidationError(
            f"Fact {row['id']} changed since the confirm, so it cannot be undone."
        )
    conn.execute(
        "UPDATE facts SET review_after = ?, pending_proposal = ? WHERE id = ?",
        (before.get("review_after"), before.get("pending_proposal"), row["id"]),
    )
    reopened = _reopen_pairs(ctx, after.get("resolved_conflicts"), "confirm")
    return (
        {"review_after": row["review_after"], "pending_proposal": row["pending_proposal"]},
        {
            "review_after": before.get("review_after"),
            "pending_proposal": before.get("pending_proposal"),
            "reopened_conflicts": reopened,
        },
    )


_HANDLERS: dict[str, Handler] = {
    "create": _undo_create,
    "approve": _undo_approve,
    "supersede": _undo_supersede,
    "confirm": _undo_confirm,
}
