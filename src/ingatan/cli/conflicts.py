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

from ingatan import timeutil
from ingatan.audit import LogEntry
from ingatan.facts._util import loads_list, record_conflicts
from ingatan.scope import Context
from ingatan.search import is_valid_sql, similar_facts

__all__ = [
    "ConflictReport",
    "OpenConflict",
    "ScanResult",
    "compute_report",
    "open_conflicts",
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
        if entry.op in ("create", "approve"):
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


@dataclass(frozen=True)
class OpenConflict:
    id: int
    fact_a: str
    text_a: str
    fact_b: str
    text_b: str
    created_at: str


@dataclass(frozen=True)
class ScanResult:
    facts_scanned: int
    pairs_found: int
    pairs_recorded: int
    pairs: list[tuple[str, str]]


def _valid_fact_texts(ctx: Context) -> dict[str, str]:
    now = timeutil.iso(timeutil.now())
    texts: dict[str, str] = {}
    for conn in ctx.connections():
        rows = conn.execute(
            f"SELECT id, text FROM facts WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)}",
            {"now": now, **ctx.params},
        ).fetchall()
        for r in rows:
            texts[str(r["id"])] = str(r["text"])
    return texts


def open_conflicts(ctx: Context) -> list[OpenConflict]:
    """Every unresolved conflict whose both facts are still valid and visible."""
    texts = _valid_fact_texts(ctx)
    rows = ctx.business.execute(
        "SELECT id, fact_a, fact_b, created_at FROM conflicts "
        "WHERE resolved_at IS NULL ORDER BY id ASC"
    ).fetchall()
    out: list[OpenConflict] = []
    for r in rows:
        a, b = str(r["fact_a"]), str(r["fact_b"])
        if a in texts and b in texts:
            out.append(OpenConflict(int(r["id"]), a, texts[a], b, texts[b], str(r["created_at"])))
    return out


def scan_conflicts(ctx: Context, *, dry_run: bool = False) -> ScanResult:
    """Scan all valid visible facts for unrecorded similar pairs (useful after bulk imports)."""
    now = timeutil.iso(timeutil.now())
    valid_rows: list[Any] = []
    for conn in ctx.connections():
        valid_rows.extend(
            conn.execute(
                f"SELECT * FROM facts WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)} "
                "ORDER BY valid_from ASC, id ASC",
                {"now": now, **ctx.params},
            ).fetchall()
        )
    existing: set[frozenset[str]] = set()
    for r in ctx.business.execute("SELECT fact_a, fact_b FROM conflicts"):
        existing.add(frozenset((str(r["fact_a"]), str(r["fact_b"]))))

    discovered: list[tuple[str, str]] = []
    seen_new: set[frozenset[str]] = set()
    recorded = 0
    for row in valid_rows:
        fid = str(row["id"])
        ents = loads_list(str(row["entities"]))
        if not ents:
            continue
        kws = loads_list(str(row["keywords"]))
        hits = similar_facts(ctx, str(row["text"]), ents, kws, exclude_id=fid)
        for hit in hits:
            pair = frozenset((fid, hit.id))
            if pair in existing or pair in seen_new:
                continue
            seen_new.add(pair)
            discovered.append((fid, hit.id))
            if not dry_run:
                added = record_conflicts(ctx, fid, [hit.id])
                recorded += len(added)
    return ScanResult(
        facts_scanned=len(valid_rows),
        pairs_found=len(discovered),
        pairs_recorded=recorded,
        pairs=discovered,
    )
