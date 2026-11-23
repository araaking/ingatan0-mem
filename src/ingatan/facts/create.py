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
from ingatan.db import fact_text_key, normalize_fact_text, transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.facts import backfill
from ingatan.facts._util import (
    DEFAULT_REVIEW_DAYS,
    clean_list,
    fact_dict,
    loads_list,
    new_id,
    now_iso,
    record_conflicts,
)
from ingatan.facts.links import link_facts
from ingatan.redact import redact
from ingatan.scope import Context, find_fact
from ingatan.search import CATEGORIES, FactHit, is_valid_sql, similar_facts
from ingatan.search import embed as embed_store
from ingatan.search.stopwords import extract_auto_keywords

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
    valid_from = str(fields.get("valid_from") or now)
    category = str(fields.get("category") or "fact")
    pinned = 1 if fields.get("pinned") else 0
    row: dict[str, Any] = {
        "id": fact_id,
        "text": fields["text"],
        "entities": json.dumps(list(fields.get("entities") or []), ensure_ascii=False),
        "keywords": json.dumps(list(fields.get("keywords") or []), ensure_ascii=False),
        "tier": fields["tier"],
        "agent": fields["agent"],
        "kind": kind,
        "category": category,
        "pinned": pinned,
        "valid_from": valid_from,
        "review_after": review_after,
        "review_interval_days": interval,
        "source": json.dumps(dict(fields["source"]), ensure_ascii=False),
        "created_at": now,
        "text_key": fact_text_key(str(fields["text"])),
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
    extra: Mapping[str, Any]
    if isinstance(source, str):
        extra = {"document": source}
    elif source is not None:
        extra = source
    else:
        extra = {}
    date_value = now_iso()[:10]
    raw_date = extra.get("date")
    if isinstance(raw_date, str) and raw_date.strip():
        date_value = timeutil.iso(timeutil.parse(raw_date))[:10]
    out: dict[str, Any] = {"agent": ctx.scope.agent, "date": date_value}
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


def _find_duplicate(
    ctx: Context,
    conn: sqlite3.Connection,
    *,
    tier: str,
    category: str,
    clean_text: str,
    entities: list[str],
) -> sqlite3.Row | None:
    """An open fact in the same tier and category that says the same thing, word for word.

    "The same" ignores case, spacing and trailing punctuation, and nothing else. Closeness
    in meaning is never treated as a repeat: "ongkir Rp 15.000" and "ongkir Rp 18.000" are
    closer to each other than two real paraphrases, so merging on a similarity score would
    silently drop a changed value. Such a pair is saved and reported as similar instead.
    """
    norm_new = normalize_fact_text(clean_text)
    ent_set = {e.casefold() for e in entities}
    where = (
        f"text_key = :key AND tier = :tier AND category = :cat AND {is_valid_sql()} "
        f"AND {ctx.visible_sql(conn)}"
    )
    params: dict[str, Any] = {
        "key": fact_text_key(clean_text),
        "tier": tier,
        "cat": category,
        "now": now_iso(),
        **ctx.params,
    }
    if tier == "agent":
        where += " AND agent = :agent"
    # One lookup in the `text_key` index, whatever the size of the table. The comparison
    # below is the real test; the key only narrows it to the rows that can pass.
    rows = conn.execute(f"SELECT * FROM facts WHERE {where}", params).fetchall()
    for row in rows:
        row_ents = {e.casefold() for e in loads_list(str(row["entities"]))}
        if normalize_fact_text(str(row["text"])) == norm_new and row_ents == ent_set:
            return row
    return None


def _refresh_duplicate(
    ctx: Context,
    conn: sqlite3.Connection,
    existing: sqlite3.Row,
    *,
    keywords: list[str],
    entities: list[str],
    pinned: bool,
    episode: int | None,
) -> RememberResult:
    """Refresh review_after and metadata on the fact this text repeats; create nothing."""
    fact_id = str(existing["id"])
    now = now_iso()
    new_review = existing["review_after"]
    if existing["kind"] == "volatile" and existing["review_interval_days"]:
        new_review = timeutil.iso(
            timeutil.now() + timedelta(days=int(existing["review_interval_days"]))
        )
    merged_kw = clean_list([*loads_list(str(existing["keywords"])), *keywords])
    merged_ent = clean_list([*loads_list(str(existing["entities"])), *entities])
    new_pinned = 1 if (pinned or existing["pinned"]) else 0
    with transaction(ctx.business), transaction(conn):
        conn.execute(
            "UPDATE facts SET review_after = ?, last_used = ?, keywords = ?, entities = ?, "
            "pinned = ? WHERE id = ?",
            (
                new_review,
                now,
                json.dumps(merged_kw, ensure_ascii=False),
                json.dumps(merged_ent, ensure_ascii=False),
                new_pinned,
                fact_id,
            ),
        )
        audit.write(
            conn,
            agent=ctx.scope.agent,
            op="confirm",
            fact_id=fact_id,
            tier=str(existing["tier"]),
            owner=str(existing["agent"]),
            before={
                "review_after": existing["review_after"],
                "pending_proposal": existing["pending_proposal"],
            },
            after={
                "review_after": new_review,
                "pending_proposal": existing["pending_proposal"],
                "reason": "dedup: the same text was remembered again",
                "resolved_conflicts": [],
            },
        )
        _mark_episode(ctx, episode)
    return RememberResult(fact_id, None, [], f"Already remembered as {fact_id} (refreshed).")


def remember(
    ctx: Context,
    text: str,
    *,
    keywords: Sequence[str] | None = None,
    entities: Sequence[str] | None = None,
    tier: str = "agent",
    kind: str | None = None,
    category: str = "fact",
    pinned: bool = False,
    review_after: str | None = None,
    source: Mapping[str, Any] | str | None = None,
    episode: int | None = None,
    valid_from: str | None = None,
    check_conflicts: bool = True,
    related_to: Sequence[str] | None = None,
    dedup: bool = True,
) -> RememberResult:
    """Save a fact immediately (write-then-undo), or queue a proposal if approval is required."""
    if tier not in TIERS:
        raise ValidationError(f"Unknown tier '{tier}'. Use agent, business or global.")
    if kind is not None and kind not in KINDS:
        raise ValidationError(f"Unknown kind '{kind}'. Use stable or volatile.")
    if category not in CATEGORIES:
        raise ValidationError(
            f"Unknown category '{category}'. Use one of: {', '.join(CATEGORIES)}."
        )
    clean_text = redact(text or "").strip()
    if not clean_text:
        raise ValidationError("The fact text is empty.")
    entity_list = clean_list([entities] if isinstance(entities, str) else entities)
    keyword_list = clean_list([keywords] if isinstance(keywords, str) else keywords)
    if not keyword_list:
        keyword_list = clean_list(extract_auto_keywords(clean_text, entity_list))
    resolved_kind = kind or "volatile"
    explicit_review = timeutil.normalize(review_after) if review_after else None
    resolved_valid_from = timeutil.normalize(valid_from) if valid_from else None
    interval: int | None = None
    if resolved_kind == "volatile":
        interval = DEFAULT_REVIEW_DAYS
        if explicit_review is not None:
            seconds = (timeutil.parse(explicit_review) - timeutil.now()).total_seconds()
            interval = max(1, math.ceil(seconds / 86400)) if seconds > 0 else DEFAULT_REVIEW_DAYS
    _check_episode(ctx, episode)
    resolved_source = _source(ctx, source, episode)

    link_targets: list[str] = []
    for target_id in dict.fromkeys(related_to or []):
        tid = str(target_id).strip()
        if not tid:
            continue
        if find_fact(ctx, tid) is None:
            raise NotFoundError(f"No related fact {tid}.")
        link_targets.append(tid)

    needs_approval = tier != "agent" and ctx.scope.require_approval
    conn = ctx.conn_for_tier(tier)

    if dedup and not needs_approval and resolved_valid_from is None:
        dup = _find_duplicate(
            ctx,
            conn,
            tier=tier,
            category=category,
            clean_text=clean_text,
            entities=entity_list,
        )
        if dup is not None:
            res = _refresh_duplicate(
                ctx,
                conn,
                dup,
                keywords=keyword_list,
                entities=entity_list,
                pinned=pinned,
                episode=episode,
            )
            for tid in link_targets:
                if tid != res.fact_id and res.fact_id is not None:
                    link_facts(ctx, res.fact_id, tid, relation="related_to")
            return res

    # With `defer`, a fact that is saved directly is saved without a vector: the model is
    # not called here at all, and `backfill` embeds the fact and runs the meaning-based
    # conflict check afterwards. A proposal is not deferred: nothing is saved yet, and the
    # similar facts shown to the approver are worked out now.
    embeddings_on = embed_store.is_enabled(ctx.scope.embeddings)
    deferred = embeddings_on and ctx.scope.embeddings.defer and not needs_approval
    # Otherwise one model call, made before any write lock is taken and only for a fact
    # that will be saved. The vector serves both the conflict check and storage.
    new_vec: list[float] | None = (
        embed_store.embed_text(ctx.scope.embeddings, clean_text, is_query=False)
        if embeddings_on and not deferred
        else None
    )

    similar = (
        similar_facts(
            ctx,
            clean_text,
            entity_list,
            keyword_list,
            linked_ids=link_targets,
            query_vec=new_vec or [],
        )
        if check_conflicts
        else []
    )
    similar_ids = [h.id for h in similar]
    content: dict[str, Any] = {
        "text": clean_text,
        "entities": entity_list,
        "keywords": keyword_list,
        "kind": resolved_kind,
        "category": category,
        "pinned": bool(pinned),
        "review_after": explicit_review if resolved_kind == "volatile" else None,
        "review_interval_days": interval,
        "source": resolved_source,
    }
    if resolved_valid_from is not None:
        content["valid_from"] = resolved_valid_from
    note = ""
    if similar:
        note = (
            f" {len(similar)} similar fact(s) exist. If one is outdated, call "
            "supersede(old, new). If both are true, call confirm(id, reason) with a reason "
            "that names the other fact id."
        )

    if not needs_approval:
        fields = {**content, "tier": tier, "agent": ctx.scope.agent, "similar": similar_ids}
        with transaction(ctx.business), transaction(conn):
            fact_id = create_fact(conn, ctx, fields)
            if new_vec:
                embed_store.store_fact_embedding(
                    conn, fact_id, clean_text, ctx.scope.embeddings, vector=new_vec
                )
            elif deferred:
                backfill.enqueue(conn, fact_id, check_conflicts=check_conflicts)
            for tid in link_targets:
                if tid != fact_id:
                    link_facts(ctx, fact_id, tid, relation="related_to")
            _mark_episode(ctx, episode)
        if deferred:
            backfill.notify(ctx.scope)
            if check_conflicts:
                note += (
                    " The meaning-based check runs in the background: a conflict it finds "
                    "shows as 'possible conflict' in recall."
                )
        return RememberResult(fact_id, None, _similar_view(similar), f"Saved as {fact_id}." + note)

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
