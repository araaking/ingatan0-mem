"""`supersede` and `confirm`."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from ingatan import audit, timeutil
from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.facts._util import (
    fact_dict,
    is_closed,
    now_iso,
    resolve_conflicts,
)
from ingatan.facts.create import insert_proposal
from ingatan.redact import redact
from ingatan.scope import Context, find_fact

__all__ = ["ConfirmResult", "SupersedeResult", "confirm", "find_own_proposal", "supersede"]

_FACT_ID = re.compile(r"f_[0-9a-f]{12}")


@dataclass(frozen=True)
class SupersedeResult:
    old_id: str
    new_id: str
    closed: bool  # False when the change waits in the proposal queue (FR-59)
    proposal_id: str | None
    message: str


@dataclass(frozen=True)
class ConfirmResult:
    fact_id: str
    review_after: str | None
    resolved_conflicts: int
    message: str


def find_own_proposal(
    ctx: Context, proposal_id: str
) -> tuple[sqlite3.Connection, sqlite3.Row] | None:
    """A proposal made by the calling agent, in either file. Others behave as missing."""
    for conn in ctx.connections():
        row = conn.execute(
            "SELECT * FROM proposals WHERE id = ? AND agent = ?", (proposal_id, ctx.scope.agent)
        ).fetchone()
        if row is not None:
            return conn, row
    return None


def _require_open(row: sqlite3.Row, what: str) -> None:
    if is_closed(row):
        raise ValidationError(f"{what} {row['id']} is already closed (superseded or expired).")


def supersede(ctx: Context, old_id: str, new_id: str) -> SupersedeResult:
    """Replace `old_id` by `new_id` (a fact id, or a pending proposal id)."""
    old = find_fact(ctx, old_id)
    if old is None:
        raise NotFoundError(f"No fact {old_id}.")
    proposal: tuple[sqlite3.Connection, sqlite3.Row] | None = None
    new_fact: tuple[sqlite3.Connection, sqlite3.Row] | None = None
    if new_id.startswith("p_"):
        proposal = find_own_proposal(ctx, new_id)
        if proposal is None:
            raise NotFoundError(f"No pending proposal {new_id}.")
    else:
        new_fact = find_fact(ctx, new_id)
        if new_fact is None:
            raise NotFoundError(f"No fact {new_id}.")
    if old_id == new_id:
        raise ValidationError("A fact cannot supersede itself.")
    old_conn, old_row = old
    _require_open(old_row, "Fact")
    if new_fact is not None:
        _require_open(new_fact[1], "Fact")

    if old_row["tier"] == "agent":
        if new_fact is None:
            raise ValidationError(
                "An agent fact is replaced by another fact, not by a proposal. "
                "Pass the id of the new fact."
            )
        return _close_old(ctx, old_conn, old_row, new_id)
    return _queue_change(ctx, old_conn, old_row, new_id, proposal, new_fact)


def _close_old(
    ctx: Context, conn: sqlite3.Connection, old: sqlite3.Row, new_id: str
) -> SupersedeResult:
    now = now_iso()
    with transaction(conn):
        conn.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
            (now, new_id, old["id"]),
        )
        resolved = resolve_conflicts(ctx, old["id"], "supersede")
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="supersede",
            fact_id=old["id"],
            tier=old["tier"],
            owner=old["agent"],
            before={"valid_to": old["valid_to"], "superseded_by": old["superseded_by"]},
            after={"valid_to": now, "superseded_by": new_id, "resolved_conflicts": resolved},
        )
    return SupersedeResult(
        old["id"], new_id, True, None, f"{old['id']} is closed and now points to {new_id}."
    )


def _queue_change(
    ctx: Context,
    conn: sqlite3.Connection,
    old: sqlite3.Row,
    new_id: str,
    proposal: tuple[sqlite3.Connection, sqlite3.Row] | None,
    new_fact: tuple[sqlite3.Connection, sqlite3.Row] | None,
) -> SupersedeResult:
    """FR-59: a business or global fact is never closed by an agent, only by approval."""
    old_id = old["id"]
    waiting = old["pending_proposal"]
    if (
        waiting
        and conn.execute(
            "SELECT 1 FROM proposals WHERE id = ? AND status = 'pending'", (waiting,)
        ).fetchone()
    ):
        raise ValidationError(
            f"Fact {old_id} already has a change waiting for approval ({waiting}). "
            "The user has to approve or reject it first."
        )
    after: dict[str, Any]
    with transaction(conn):
        if proposal is not None:
            p = proposal[1]
            if p["status"] != "pending":
                raise ValidationError(f"Proposal {p['id']} is already {p['status']}.")
            if p["tier"] != old["tier"]:
                raise ValidationError(
                    f"Proposal {p['id']} is for the {p['tier']} tier but {old_id} is a "
                    f"{old['tier']} fact. Propose it in the same tier."
                )
            if p["supersedes"] not in (None, old_id):
                raise ValidationError(f"Proposal {p['id']} already replaces another fact.")
            proposal_id = p["id"]
            conn.execute("UPDATE proposals SET supersedes = ? WHERE id = ?", (old_id, proposal_id))
            after = {"pending_proposal": proposal_id}
        else:
            assert new_fact is not None
            source_row = new_fact[1]
            if source_row["tier"] != "agent":
                raise ValidationError(
                    f"{old_id} is a {old['tier']} fact. Replace it with one of your own agent "
                    "facts or with a pending proposal id."
                )
            payload = fact_dict(source_row)
            content = {
                "text": payload["text"],
                "entities": payload["entities"],
                "keywords": payload["keywords"],
                "kind": payload["kind"],
                "review_after": None,
                "review_interval_days": payload["review_interval_days"],
                "source": payload["source"],
                "origin_fact": new_id,  # lets forget find this copy of the text
            }
            proposal_id = insert_proposal(
                conn, ctx, tier=old["tier"], payload=content, similar=[], supersedes=old_id
            )
            after = {"pending_proposal": proposal_id, "proposal_from_fact": new_id}
        conn.execute("UPDATE facts SET pending_proposal = ? WHERE id = ?", (proposal_id, old_id))
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="supersede",
            fact_id=old_id,
            tier=old["tier"],
            owner=old["agent"],
            before={"pending_proposal": old["pending_proposal"]},
            after=after,
        )
    return SupersedeResult(
        old_id,
        new_id,
        False,
        proposal_id,
        f"{old_id} is a {old['tier']} fact, so the change waits for the user's approval "
        f"({proposal_id}). Until then it is shown as 'check first'.",
    )


def confirm(ctx: Context, fact_id: str, reason: str) -> ConfirmResult:
    """Say a fact is still true: push `review_after` out and clear its doubt flags."""
    clean_reason = redact(reason or "").strip()
    if not clean_reason:
        raise ValidationError("confirm needs a reason: say how you know the fact is still true.")
    found = find_fact(ctx, fact_id)
    if found is None:
        raise NotFoundError(f"No fact {fact_id}.")
    conn, row = found
    _require_open(row, "Fact")
    named = set(_FACT_ID.findall(clean_reason.lower()))
    new_review = row["review_after"]
    if row["kind"] == "volatile" and row["review_interval_days"]:
        new_review = timeutil.iso(timeutil.now() + timedelta(days=row["review_interval_days"]))
    with transaction(conn):
        conn.execute(
            "UPDATE facts SET review_after = ?, pending_proposal = NULL WHERE id = ?",
            (new_review, fact_id),
        )
        resolved = resolve_conflicts(ctx, fact_id, "confirm", only_with=named)
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="confirm",
            fact_id=fact_id,
            tier=row["tier"],
            owner=row["agent"],
            before={
                "review_after": row["review_after"],
                "pending_proposal": row["pending_proposal"],
            },
            after={
                "review_after": new_review,
                "pending_proposal": None,
                "reason": clean_reason,
                "resolved_conflicts": resolved,
            },
        )
    return ConfirmResult(fact_id, new_review, len(resolved), f"Confirmed {fact_id}. Reason logged.")
