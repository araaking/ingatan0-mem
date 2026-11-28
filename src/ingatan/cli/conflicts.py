"""The conflict-resolution report (NFR-9, OQ-18), computed from the audit log alone.

- Candidate: a pair recorded at `create` or `approve` time (`after.conflicts_with`).
- Resolved: a candidate whose pair was later closed by a `supersede` that closed one of its
  facts, or by a `confirm` on one fact whose reason names the other fact id.

Entries that were undone do not count, in either direction: a create that was undone
contributes no candidates, a supersede or confirm that was undone resolves nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ingatan import audit, timeutil
from ingatan.audit import LogEntry
from ingatan.db import transaction
from ingatan.facts._util import loads_list, record_conflicts
from ingatan.facts.review import OpenConflict, open_pairs, resolve_pair
from ingatan.scope import Context, find_fact
from ingatan.search import embed as embed_store
from ingatan.search import is_valid_sql, similar_facts

__all__ = [
    "ConflictReport",
    "OpenConflict",
    "PruneResult",
    "ScanResult",
    "compute_report",
    "open_conflicts",
    "prune_conflicts",
    "render_report",
    "scan_conflicts",
]

FAILURE_NOTE = (
    "Fewer than half of the conflict candidates were resolved by the agents. "
    "By the failure line in the PRD (section 9), digest with an API key becomes the "
    "recommended mode. Zero-key mode stays supported."
)


@dataclass(frozen=True)
class ConflictReport:
    candidates: int
    resolved: int

    @property
    def share(self) -> float | None:
        return None if self.candidates == 0 else self.resolved / self.candidates

    @property
    def below_half(self) -> bool:
        return self.candidates > 0 and self.resolved * 2 < self.candidates


def compute_report(log: Iterable[LogEntry]) -> ConflictReport:
    """`log` is a list of entries, oldest first (as `audit.entries` returns them)."""
    ordered = list(log)
    undone: set[int] = set()
    for entry in ordered:
        if entry.op == "undo" and entry.change:
            target = (entry.change.get("after") or {}).get("undoes")
            if isinstance(target, int):
                undone.add(target)

    open_pairs: set[frozenset[str]] = set()
    seen: set[frozenset[str]] = set()
    resolved: set[frozenset[str]] = set()
    for entry in ordered:
        if entry.id in undone or entry.change is None or entry.fact_id is None:
            continue
        after = entry.change.get("after") or {}
        if entry.op in ("create", "approve", "conflict"):
            for other in after.get("conflicts_with") or []:
                pair = frozenset((entry.fact_id, str(other)))
                if len(pair) == 2:
                    seen.add(pair)
                    open_pairs.add(pair)
            closed_info = after.get("closed")
            if isinstance(closed_info, dict) and closed_info.get("fact_id"):
                closed_id = str(closed_info["fact_id"])  # pyright: ignore[reportUnknownArgumentType]
                _close(open_pairs, resolved, lambda pair, fid=closed_id: fid in pair)
        elif entry.op == "supersede" and after.get("superseded_by"):
            # The old fact was closed: every pair it was part of is resolved.
            _close(open_pairs, resolved, lambda pair, fid=entry.fact_id: fid in pair)
        elif entry.op == "confirm":
            reason = str(after.get("reason") or "").lower()
            fid = entry.fact_id
            _close(
                open_pairs,
                resolved,
                lambda pair, fid=fid, reason=reason: (
                    fid in pair and any(other.lower() in reason for other in pair if other != fid)
                ),
            )
    return ConflictReport(candidates=len(seen), resolved=len(resolved))


def _close(
    open_pairs: set[frozenset[str]],
    resolved: set[frozenset[str]],
    matches: Callable[[frozenset[str]], bool],
) -> None:
    for pair in list(open_pairs):
        if matches(pair):
            open_pairs.discard(pair)
            resolved.add(pair)


def render_report(report: ConflictReport) -> str:
    lines = [
        f"Conflict candidates: {report.candidates}",
        f"Resolved by supersede or confirm: {report.resolved}",
    ]
    share = report.share
    if share is None:
        lines.append("Share resolved: n/a (no conflict candidates in the log yet)")
    else:
        lines.append(f"Share resolved: {share * 100:.1f}%")
        if report.below_half:
            lines.append("")
            lines.append(FAILURE_NOTE)
    return "\n".join(lines)


# The open pairs are listed in `facts.review`, next to the code that resolves them. The
# old name stays importable from here.
open_conflicts = open_pairs


@dataclass(frozen=True)
class ScanResult:
    facts_scanned: int
    pairs_found: int
    pairs_recorded: int
    pairs: list[tuple[str, str]]


@dataclass(frozen=True)
class PruneResult:
    pairs_checked: int
    false_positives: int
    pairs_pruned: int
    pairs: list[tuple[int, str, str]]


def _stored_vectors(ctx: Context) -> dict[str, list[float]]:
    """Unpack stored vectors for the active model so scans reuse them without re-embedding."""
    cfg = ctx.scope.embeddings
    if not embed_store.is_enabled(cfg):
        return {}
    model = embed_store.effective_model(cfg)
    if not model:
        return {}
    vectors: dict[str, list[float]] = {}
    for conn in ctx.connections():
        for row in conn.execute(
            "SELECT fact_id, dims, vec FROM fact_embeddings WHERE model = ?", (model,)
        ):
            vec = embed_store.unpack_vector(bytes(row["vec"]), int(row["dims"]))
            if vec:
                vectors[str(row["fact_id"])] = vec
    return vectors


def scan_conflicts(ctx: Context, *, dry_run: bool = False) -> ScanResult:
    """Scan all valid visible facts for unrecorded similar pairs (useful after bulk imports)."""
    now = timeutil.iso(timeutil.now())
    valid_rows: list[tuple[Any, Any]] = []
    for conn in ctx.connections():
        for row in conn.execute(
            f"SELECT * FROM facts WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)} "
            "ORDER BY valid_from ASC, id ASC",
            {"now": now, **ctx.params},
        ).fetchall():
            valid_rows.append((conn, row))
    existing: set[frozenset[str]] = set()
    for r in ctx.business.execute("SELECT fact_a, fact_b FROM conflicts"):
        existing.add(frozenset((str(r["fact_a"]), str(r["fact_b"]))))

    cfg = ctx.scope.embeddings
    sem_enabled = embed_store.is_enabled(cfg)
    entity_free = sem_enabled and embed_store.similarity_profile(cfg).conflict_strict is not None
    stored_vecs = _stored_vectors(ctx)

    discovered: list[tuple[str, str]] = []
    seen_new: set[frozenset[str]] = set()
    recorded = 0
    for conn, row in valid_rows:
        fid = str(row["id"])
        ents = loads_list(str(row["entities"]))
        if not ents and not entity_free:
            continue
        kws = loads_list(str(row["keywords"]))
        hits = similar_facts(
            ctx,
            str(row["text"]),
            ents,
            kws,
            exclude_id=fid,
            query_vec=stored_vecs.get(fid),
            kind=str(row["kind"]),
        )
        new_targets: list[str] = []
        for hit in hits:
            pair = frozenset((fid, hit.id))
            if pair in existing or pair in seen_new:
                continue
            seen_new.add(pair)
            discovered.append((fid, hit.id))
            new_targets.append(hit.id)
        if not dry_run and new_targets:
            with transaction(ctx.business), transaction(conn):
                added = record_conflicts(ctx, fid, new_targets)
                if added:
                    recorded += len(added)
                    audit.write(
                        conn,
                        agent=ctx.scope.agent,
                        op="conflict",
                        fact_id=fid,
                        tier=str(row["tier"]),
                        owner=str(row["agent"]),
                        before={"conflicts_with": []},
                        after={"conflicts_with": added, "found": "conflicts scan"},
                    )
    return ScanResult(
        facts_scanned=len(valid_rows),
        pairs_found=len(discovered),
        pairs_recorded=recorded,
        pairs=discovered,
    )


def prune_conflicts(ctx: Context, *, dry_run: bool = False) -> PruneResult:
    """Re-evaluate open pairs under current `similar_facts` rules and dismiss false positives.

    Unlike `keep_both`, pruning uses `dismiss` so unrelated legacy pairs are closed cleanly
    without creating bogus `related_to` edges. Every closure is logged and undoable.
    """
    pairs = open_pairs(ctx)
    if not pairs:
        return PruneResult(0, 0, 0, [])

    stored_vecs = _stored_vectors(ctx)
    match_cache: dict[str, set[str]] = {}

    def matches_of(fact_id: str) -> set[str]:
        cached = match_cache.get(fact_id)
        if cached is not None:
            return cached
        found = find_fact(ctx, fact_id)
        if found is None:
            match_cache[fact_id] = set()
            return set()
        row = found[1]
        hits = similar_facts(
            ctx,
            str(row["text"]),
            loads_list(str(row["entities"])),
            loads_list(str(row["keywords"])),
            exclude_id=fact_id,
            query_vec=stored_vecs.get(fact_id),
            kind=str(row["kind"]),
            limit=50,
        )
        ids = {h.id for h in hits}
        match_cache[fact_id] = ids
        return ids

    false_pos: list[tuple[int, str, str]] = []
    pruned = 0
    for pair in pairs:
        still_similar = pair.older_id in matches_of(pair.newer_id) or pair.newer_id in matches_of(
            pair.older_id
        )
        if still_similar:
            continue
        false_pos.append((pair.id, pair.newer_id, pair.older_id))
        if not dry_run:
            outcome = resolve_pair(ctx, pair, "dismiss")
            if not outcome.message.startswith("Skipped"):
                pruned += 1
    return PruneResult(
        pairs_checked=len(pairs),
        false_positives=len(false_pos),
        pairs_pruned=pruned,
        pairs=false_pos,
    )
