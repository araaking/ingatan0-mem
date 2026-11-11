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

from ingatan.audit import LogEntry

__all__ = ["ConflictReport", "compute_report", "render_report"]

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
