"""`remember` and the shared `create_fact` (also used by proposal approval)."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ingatan import audit, timeutil
from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.facts._util import (
    DEFAULT_REVIEW_DAYS,
    clean_list,
    fact_dict,
    new_id,
    now_iso,
    record_conflicts,
)
from ingatan.redact import redact
from ingatan.scope import Context
from ingatan.search import FactHit, similar_facts

__all__ = ["RememberResult", "create_fact", "insert_proposal", "remember"]

TIERS = ("agent", "business", "global")
KINDS = ("stable", "volatile")


@dataclass(frozen=True)
class RememberResult:
    fact_id: str | None
    proposal_id: str | None
    similar: list[dict[str, str]] = field(default_factory=lambda: [])
    message: str = ""


def create_fact(
    conn: sqlite3.Connection, ctx: Context, fields: Mapping[str, Any], *, op: str = "create"
) -> str:
    """Insert a fact, open its conflict pairs and write one log entry. Returns the id.

    Must run inside `transaction(conn)`. `fields` holds already cleaned values:
    text, entities, keywords, tier, agent, kind, source, and optionally review_after,
    review_interval_days, id, `similar` (ids to open conflict pairs with) and `_extra`
    (extra keys for the log entry's `after`, used by approval).
    """
    now = now_iso()
    fact_id = str(fields.get("id") or new_id("f_"))
    kind = str(fields["kind"])
    review_after: str | None = None
    interval: int | None = None
    if kind == "volatile":
        interval = int(fields.get("review_interval_days") or DEFAULT_REVIEW_DAYS)
        review_after = fields.get("review_after") or timeutil.iso(
            timeutil.now() + timedelta(days=interval)
        )
    row: dict[str, Any] = {
        "id": fact_id,
        "text": fields["text"],
        "entities": json.dumps(list(fields.get("entities") or []), ensure_ascii=False),
        "keywords": json.dumps(list(fields.get("keywords") or []), ensure_ascii=False),
        "tier": fields["tier"],
        "agent": fields["agent"],
        "kind": kind,
        "valid_from": now,
        "review_after": review_after,
        "review_interval_days": interval,
        "source": json.dumps(dict(fields["source"]), ensure_ascii=False),
        "created_at": now,
    }
    names = ", ".join(row)
    marks = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO facts ({names}) VALUES ({marks})", row)
    stored = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    after = fact_dict(stored)
    after["conflicts_with"] = record_conflicts(ctx, fact_id, fields.get("similar") or [])
    after.update(fields.get("_extra") or {})
    audit.write(
        conn,
        agent=ctx.scope.agent,
        op=op,
        fact_id=fact_id,
        tier=str(fields["tier"]),
        owner=str(fields["agent"]),
        before=None,
        after=after,
    )
    return fact_id


def insert_proposal(
    conn: sqlite3.Connection,
    ctx: Context,
    *,
    tier: str,
    payload: Mapping[str, Any],
    similar: Sequence[str],
    supersedes: str | None = None,
) -> str:
    """Insert a pending proposal row (no log entry; the caller writes it). In a transaction."""
    proposal_id = new_id("p_")
    conn.execute(
        "INSERT INTO proposals (id, tier, payload, similar, supersedes, agent, created_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
        (
            proposal_id,
            tier,
            json.dumps(dict(payload), ensure_ascii=False),
            json.dumps(list(similar)),
            supersedes,
            ctx.scope.agent,
            now_iso(),
        ),
    )
    return proposal_id


def _source(
    ctx: Context, source: Mapping[str, Any] | str | None, episode: int | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"agent": ctx.scope.agent, "date": now_iso()[:10]}
    extra: Mapping[str, Any]
    if isinstance(source, str):
        extra = {"document": source}
    elif source is not None:
        extra = source
    else:
        extra = {}
    for key in ("session", "document"):
        value = extra.get(key)
        if value:
            out[key] = redact(str(value))
    if episode is not None:
        # Lets `forget` delete the captured turn this fact was digested from (US-9).
        out["episode"] = episode
    return out


def _check_episode(ctx: Context, episode: int | None) -> None:
    if episode is None:
        return
    row = ctx.business.execute(
        "SELECT 1 FROM episodes WHERE id = ? AND agent = ?", (episode, ctx.scope.agent)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"No episode {episode}.")


def _mark_episode(ctx: Context, episode: int | None) -> None:
    # Facts sits below `episodes` in the module order, so this is plain SQL.
    if episode is None:
        return
    with transaction(ctx.business):
        ctx.business.execute(
            "UPDATE episodes SET status = 'processed', processed_at = ? "
            "WHERE id = ? AND agent = ? AND status != 'processed'",
            (now_iso(), episode, ctx.scope.agent),
        )


def _similar_view(hits: Sequence[FactHit]) -> list[dict[str, str]]:
    return [{"id": h.id, "text": h.text, "status": h.status} for h in hits]


def remember(
    ctx: Context,
    text: str,
    *,
    keywords: Sequence[str] | None,
    entities: Sequence[str] | None = None,
    tier: str = "agent",
    kind: str | None = None,
    review_after: str | None = None,
    source: Mapping[str, Any] | str | None = None,
    episode: int | None = None,
) -> RememberResult:
    """Save a fact (tier `agent`) or queue a proposal (tier `business` or `global`)."""
    if tier not in TIERS:
        raise ValidationError(f"Unknown tier '{tier}'. Use agent, business or global.")
    if kind is not None and kind not in KINDS:
        raise ValidationError(f"Unknown kind '{kind}'. Use stable or volatile.")
    clean_text = redact(text or "").strip()
    if not clean_text:
        raise ValidationError("The fact text is empty.")
    keyword_list = clean_list([keywords] if isinstance(keywords, str) else keywords)
    if not keyword_list:
        raise ValidationError(
            "keywords are required. Add synonyms and root words (for example the base form "
            "of each inflected word) so the fact can be found later by a different wording."
        )
    entity_list = clean_list([entities] if isinstance(entities, str) else entities)
    resolved_kind = kind or "volatile"
    explicit_review = timeutil.normalize(review_after) if review_after else None
    interval: int | None = None
    if resolved_kind == "volatile":
        interval = DEFAULT_REVIEW_DAYS
        if explicit_review is not None:
            seconds = (timeutil.parse(explicit_review) - timeutil.now()).total_seconds()
            interval = max(1, math.ceil(seconds / 86400)) if seconds > 0 else DEFAULT_REVIEW_DAYS
    _check_episode(ctx, episode)

    similar = similar_facts(ctx, clean_text, entity_list, keyword_list)
    similar_ids = [h.id for h in similar]
    content: dict[str, Any] = {
        "text": clean_text,
        "entities": entity_list,
        "keywords": keyword_list,
        "kind": resolved_kind,
        "review_after": explicit_review if resolved_kind == "volatile" else None,
        "review_interval_days": interval,
        "source": _source(ctx, source, episode),
    }
    note = ""
    if similar:
        note = (
            f" {len(similar)} similar fact(s) exist. If one is outdated, call "
            "supersede(old, new). If both are true, call confirm(id, reason) with a reason "
            "that names the other fact id."
        )

    if tier == "agent":
        fields = {**content, "tier": tier, "agent": ctx.scope.agent, "similar": similar_ids}
        with transaction(ctx.business):
            fact_id = create_fact(ctx.business, ctx, fields)
            _mark_episode(ctx, episode)
        return RememberResult(fact_id, None, _similar_view(similar), f"Saved as {fact_id}." + note)

    conn = ctx.conn_for_tier(tier)
    with transaction(conn):
        proposal_id = insert_proposal(
            conn, ctx, tier=tier, payload=content, similar=similar_ids, supersedes=None
        )
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="propose",
            fact_id=proposal_id,
            tier=tier,
            owner=ctx.scope.agent,
            before=None,
            after={
                "id": proposal_id,
                "status": "pending",
                "supersedes": None,
                "similar": similar_ids,
                "payload": content,
            },
        )
        if conn is ctx.business:
            _mark_episode(ctx, episode)
    if conn is not ctx.business:
        _mark_episode(ctx, episode)
    return RememberResult(
        None,
        proposal_id,
        _similar_view(similar),
        f"Proposal {proposal_id} for the {tier} tier is waiting for the user's approval. "
        "It is not a fact until approved." + note,
    )
