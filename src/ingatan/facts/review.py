"""The open `possible conflict` pairs, and resolving one of them in a single call.

`remember` reports a similar fact to the agent that wrote it. When the agent does nothing,
the pair stays open until a person looks. `open_pairs` lists what is open and `resolve_pair`
applies one of three answers, each through the function an agent would use, so every answer
is in the log and can be undone:

- `keep_both`: `confirm` on the newer fact with a reason that names the older one (that is
  what resolves the pair), then a `related_to` link between the two.
- `keep_newer` / `keep_older`: `supersede` of the other fact.

Forgetting one of the two is not offered here: `forget` cannot be undone and keeps its own
confirmation in the CLI.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ingatan.errors import IngatanError, ValidationError
from ingatan.facts._util import is_closed, now_iso
from ingatan.facts.change import confirm, supersede
from ingatan.facts.links import link_facts
from ingatan.scope import Context, find_fact
from ingatan.search import is_valid_sql

__all__ = ["ACTIONS", "OpenConflict", "ReviewOutcome", "open_pairs", "resolve_pair"]

ACTIONS = ("keep_both", "keep_newer", "keep_older")


@dataclass(frozen=True)
class OpenConflict:
    """An unresolved pair. `newer_id` and `older_id` are `fact_a` and `fact_b` in the order
    of their `valid_from` (a tie goes to the row that was written later)."""

    id: int
    fact_a: str
    text_a: str
    fact_b: str
    text_b: str
    created_at: str
    newer_id: str
    older_id: str


@dataclass(frozen=True)
class ReviewOutcome:
    """What `resolve_pair` did, in one line. `proposal_id` is set when the change it made
    waits in the proposal queue (approval mode) instead of having taken effect."""

    message: str
    proposal_id: str | None = None


def _valid_facts(ctx: Context) -> dict[str, tuple[int, sqlite3.Row]]:
    """Every valid visible fact by id, with the index of the file it lives in."""
    now = now_iso()
    found: dict[str, tuple[int, sqlite3.Row]] = {}
    for index, conn in enumerate(ctx.connections()):
        rows = conn.execute(
            "SELECT id, text, valid_from, created_at, seq FROM facts "
            f"WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)}",
            {"now": now, **ctx.params},
        ).fetchall()
        for row in rows:
            found[str(row["id"])] = (index, row)
    return found


def _age_key(entry: tuple[int, sqlite3.Row]) -> tuple[str, str, int, int]:
    index, row = entry
    # `seq` orders rows inside one file only; the file index keeps the order total.
    return (str(row["valid_from"]), str(row["created_at"]), index, int(row["seq"]))


def open_pairs(ctx: Context) -> list[OpenConflict]:
    """Every unresolved conflict whose both facts are still valid and visible."""
    facts = _valid_facts(ctx)
    rows = ctx.business.execute(
        "SELECT id, fact_a, fact_b, created_at FROM conflicts "
        "WHERE resolved_at IS NULL ORDER BY id ASC"
    ).fetchall()
    out: list[OpenConflict] = []
    for row in rows:
        a, b = str(row["fact_a"]), str(row["fact_b"])
        if a not in facts or b not in facts:
            continue
        older, newer = sorted((a, b), key=lambda fact_id: _age_key(facts[fact_id]))
        out.append(
            OpenConflict(
                id=int(row["id"]),
                fact_a=a,
                text_a=str(facts[a][1]["text"]),
                fact_b=b,
                text_b=str(facts[b][1]["text"]),
                created_at=str(row["created_at"]),
                newer_id=newer,
                older_id=older,
            )
        )
    return out


def _still_open(ctx: Context, pair: OpenConflict) -> bool:
    row = ctx.business.execute(
        "SELECT 1 FROM conflicts WHERE id = ? AND resolved_at IS NULL", (pair.id,)
    ).fetchone()
    if row is None:
        return False
    for fact_id in (pair.newer_id, pair.older_id):
        found = find_fact(ctx, fact_id)
        if found is None or is_closed(found[1]):
            return False
    return True


def resolve_pair(ctx: Context, pair: OpenConflict, action: str) -> ReviewOutcome:
    """Apply one of `ACTIONS` to `pair`.

    A pair that changed since it was listed (resolved, or one fact closed or forgotten) is
    skipped and the outcome says so. In approval mode a `supersede` of a shared fact is
    queued: the outcome then carries the proposal id, and nothing is closed yet.
    """
    if action not in ACTIONS:
        raise ValidationError(f"Unknown action '{action}'. Use one of: {', '.join(ACTIONS)}.")
    if not _still_open(ctx, pair):
        return ReviewOutcome(
            f"Skipped pair #{pair.id}: it was resolved, or one of its facts was closed, "
            "since it was listed."
        )
    if action == "keep_both":
        confirm(
            ctx,
            pair.newer_id,
            f"Reviewed with {pair.older_id}: both are true and stay valid.",
        )
        try:
            link_facts(ctx, pair.newer_id, pair.older_id, relation="related_to")
        except IngatanError as exc:
            # The confirm is in, and it settled the pair. Saying "not changed" would be false.
            return ReviewOutcome(
                f"Kept both: {pair.newer_id} is confirmed and the pair is settled, but not "
                f"linked with {pair.older_id}: {exc.message}"
            )
        return ReviewOutcome(f"Kept both and linked {pair.newer_id} with {pair.older_id}.")
    keep, drop = (
        (pair.newer_id, pair.older_id) if action == "keep_newer" else (pair.older_id, pair.newer_id)
    )
    result = supersede(ctx, drop, keep)
    if result.closed:
        return ReviewOutcome(f"{drop} is superseded by {keep}.")
    return ReviewOutcome(
        f"Replacing {drop} by {keep} waits for approval as {result.proposal_id}.",
        result.proposal_id,
    )
