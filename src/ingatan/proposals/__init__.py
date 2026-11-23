"""The proposal queue: business and global facts wait here until the user decides.

`pending`, `approve` and `reject` are for the CLI only. The MCP layer never calls them,
and an agent never sees a pending proposal (OQ-9).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from ingatan import audit
from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.facts import RELATIONS, create_fact, link_facts
from ingatan.facts._util import is_closed, new_id, now_iso, resolve_conflicts
from ingatan.scope import Context, find_fact
from ingatan.search import embed as embed_store
from ingatan.search import similar_facts

__all__ = ["Proposal", "approve", "approve_all", "pending", "pending_count", "reject"]


@dataclass(frozen=True)
class Proposal:
    id: str
    tier: str
    payload: dict[str, Any]
    similar: list[str]
    supersedes: str | None
    agent: str
    created_at: str
    status: str
    decided_at: str | None
    fact_id: str | None

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))


def _proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(
        id=row["id"],
        tier=row["tier"],
        payload=json.loads(row["payload"]),
        similar=[str(i) for i in json.loads(row["similar"])],
        supersedes=row["supersedes"],
        agent=row["agent"],
        created_at=row["created_at"],
        status=row["status"],
        decided_at=row["decided_at"],
        fact_id=row["fact_id"],
    )


def pending(ctx: Context) -> list[Proposal]:
    """Every pending proposal of this business and of the global tier, oldest first."""
    found: list[Proposal] = []
    for conn in ctx.connections():
        rows = conn.execute("SELECT * FROM proposals WHERE status = 'pending'").fetchall()
        found.extend(_proposal(r) for r in rows)
    found.sort(key=lambda p: (p.created_at, p.id))
    return found


def pending_count(ctx: Context) -> int:
    return sum(
        conn.execute("SELECT count(*) FROM proposals WHERE status = 'pending'").fetchone()[0]
        for conn in ctx.connections()
    )


def _load_pending(ctx: Context, proposal_id: str) -> tuple[sqlite3.Connection, sqlite3.Row]:
    for conn in ctx.connections():
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is not None:
            if row["status"] != "pending":
                raise ValidationError(f"Proposal {proposal_id} is already {row['status']}.")
            return conn, row
    raise NotFoundError(f"No proposal {proposal_id}.")


def approve(ctx: Context, proposal_id: str) -> str:
    """Create the proposed fact in its tier (or retire a superseded fact). Returns the fact id.

    The proposing agent stays the fact's `source.agent`. If the proposal replaces a fact,
    that fact is closed and its open conflicts resolved in the same transaction. Similar
    facts are re-evaluated against the current valid store so bulk-queued proposals still
    catch conflicts with each other.
    """
    conn, row = _load_pending(ctx, proposal_id)
    proposal = _proposal(row)
    now = now_iso()

    retire_to = proposal.payload.get("retire_to")
    if isinstance(retire_to, str) and retire_to:
        return _approve_retire(ctx, conn, proposal, retire_to, now)

    # One model call, before any write lock; reused for the conflict check and for storage.
    text = str(proposal.payload.get("text") or "")
    embeddings_on = embed_store.is_enabled(ctx.scope.embeddings)
    vector = (
        embed_store.embed_text(ctx.scope.embeddings, text, is_query=False)
        if embeddings_on
        else None
    )

    # Re-evaluate similar facts against the live store (merging with stored ids),
    # excluding the fact this proposal intentionally supersedes.
    fresh_similar = [
        h.id
        for h in similar_facts(
            ctx,
            text,
            list(proposal.payload.get("entities") or []),
            list(proposal.payload.get("keywords") or []),
            query_vec=vector or [],
        )
    ]
    merged_similar = [
        fid
        for fid in dict.fromkeys([*proposal.similar, *fresh_similar])
        if fid != proposal.supersedes
    ]

    # Links the proposal carries (`remember(related_to=..., part_of=...)`). A target that was
    # closed or forgotten since then is skipped, and the log entry says which.
    wanted_links: list[tuple[str, str]] = []
    skipped_links: list[str] = []
    for item in proposal.payload.get("links") or []:
        target_id, relation = str(item.get("to") or ""), str(item.get("relation") or "")
        found = find_fact(ctx, target_id) if relation in RELATIONS else None
        if found is None or is_closed(found[1]):
            skipped_links.append(target_id)
        else:
            wanted_links.append((target_id, relation))

    with transaction(ctx.business), transaction(conn):
        fields: dict[str, Any] = dict(proposal.payload)
        fields.pop("origin_fact", None)
        fields.pop("links", None)
        fields.update(tier=proposal.tier, agent=proposal.agent, similar=merged_similar)
        source = dict(fields.get("source") or {})
        source["agent"] = proposal.agent
        fields["source"] = source

        extra: dict[str, Any] = {"proposal_id": proposal_id}
        new_fact_id = new_id("f_")
        fields["id"] = new_fact_id
        affects: list[str] = []
        target = proposal.supersedes
        if target is not None:
            old = conn.execute("SELECT * FROM facts WHERE id = ?", (target,)).fetchone()
            if (
                old is not None
                and old["superseded_by"] is None
                and (old["valid_to"] is None or old["valid_to"] > now)
            ):
                before = {
                    "valid_to": old["valid_to"],
                    "superseded_by": old["superseded_by"],
                    "pending_proposal": old["pending_proposal"],
                }
                conn.execute(
                    "UPDATE facts SET valid_to = ?, superseded_by = ?, pending_proposal = NULL "
                    "WHERE id = ?",
                    (now, new_fact_id, target),
                )
                resolved_old = resolve_conflicts(ctx, target, "supersede")
                extra["closed"] = {
                    "fact_id": target,
                    "before": before,
                    "after": {
                        "valid_to": now,
                        "superseded_by": new_fact_id,
                        "pending_proposal": None,
                        "resolved_conflicts": resolved_old,
                    },
                }
                affects.append(target)
            elif old is not None and old["pending_proposal"] == proposal_id:
                conn.execute("UPDATE facts SET pending_proposal = NULL WHERE id = ?", (target,))
        if affects:
            extra["affects"] = affects
        if wanted_links or skipped_links:
            extra["links"] = {
                "created": [[target_id, relation] for target_id, relation in wanted_links],
                "skipped": skipped_links,
            }
        fields["_extra"] = extra
        create_fact(conn, ctx, fields, op="approve")
        for target_id, relation in wanted_links:
            link_facts(ctx, new_fact_id, target_id, relation=relation)
        if embeddings_on:
            embed_store.store_fact_embedding(
                conn, new_fact_id, text, ctx.scope.embeddings, vector=vector or []
            )
        conn.execute(
            "UPDATE proposals SET status = 'approved', decided_at = ?, fact_id = ? WHERE id = ?",
            (now, new_fact_id, proposal_id),
        )
    return new_fact_id


def _approve_retire(
    ctx: Context, conn: sqlite3.Connection, proposal: Proposal, retire_to: str, now: str
) -> str:
    """Approve a retire-only supersede proposal pointing to an already-valid shared fact."""
    target = proposal.supersedes
    with transaction(ctx.business), transaction(conn):
        extra: dict[str, Any] = {"proposal_id": proposal.id, "retire_only": True, "id": retire_to}
        affects: list[str] = []
        if target is not None:
            old = conn.execute("SELECT * FROM facts WHERE id = ?", (target,)).fetchone()
            if (
                old is not None
                and old["superseded_by"] is None
                and (old["valid_to"] is None or old["valid_to"] > now)
            ):
                before = {
                    "valid_to": old["valid_to"],
                    "superseded_by": old["superseded_by"],
                    "pending_proposal": old["pending_proposal"],
                }
                conn.execute(
                    "UPDATE facts SET valid_to = ?, superseded_by = ?, pending_proposal = NULL "
                    "WHERE id = ?",
                    (now, retire_to, target),
                )
                resolved_old = resolve_conflicts(ctx, target, "supersede")
                extra["closed"] = {
                    "fact_id": target,
                    "before": before,
                    "after": {
                        "valid_to": now,
                        "superseded_by": retire_to,
                        "pending_proposal": None,
                        "resolved_conflicts": resolved_old,
                    },
                }
                affects.append(target)
            elif old is not None and old["pending_proposal"] == proposal.id:
                conn.execute("UPDATE facts SET pending_proposal = NULL WHERE id = ?", (target,))
        if affects:
            extra["affects"] = affects
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="approve",
            fact_id=retire_to,
            tier=proposal.tier,
            owner=proposal.agent,
            before=None,
            after=extra,
        )
        conn.execute(
            "UPDATE proposals SET status = 'approved', decided_at = ?, fact_id = ? WHERE id = ?",
            (now, retire_to, proposal.id),
        )
    return retire_to


def approve_all(ctx: Context) -> list[tuple[str, str]]:
    """Approve every pending proposal in creation order. Returns `(proposal_id, fact_id)` pairs."""
    done: list[tuple[str, str]] = []
    for p in pending(ctx):
        done.append((p.id, approve(ctx, p.id)))
    return done


def reject(ctx: Context, proposal_id: str) -> None:
    """Throw a proposal away. The fact it wanted to replace loses its 'check first' mark."""
    conn, row = _load_pending(ctx, proposal_id)
    now = now_iso()
    target = row["supersedes"]
    with transaction(conn):
        conn.execute(
            "UPDATE proposals SET status = 'rejected', decided_at = ? WHERE id = ?",
            (now, proposal_id),
        )
        affects: list[str] = []
        if target is not None:
            cursor = conn.execute(
                "UPDATE facts SET pending_proposal = NULL WHERE id = ? AND pending_proposal = ?",
                (target, proposal_id),
            )
            if cursor.rowcount:
                affects.append(target)
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="reject",
            fact_id=proposal_id,
            tier=row["tier"],
            owner=row["agent"],
            before={"status": "pending", "supersedes": target, "pending_proposal": proposal_id},
            after={"status": "rejected", "pending_proposal": None, "affects": affects},
        )
