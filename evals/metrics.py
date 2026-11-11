"""The four metrics. Pure functions over `CheckRecord`s, so they test without a memory.

Definitions (also in README.md, from OQ-18):

- recall@5: share of recall checks with an `expect_top5` whose expected facts are all among
  the first five results.
- stale rate: share of recall checks that return a closed fact, or a fact past its review
  date whose status does not say "check first", as if it were current.
- leak rate: share of returned items (recall results and brief fact lines) that come from
  outside the caller's scope.
- tokens per brief: mean and max of `estimate_tokens` over every brief built.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CheckRecord",
    "Metrics",
    "compute",
    "leak_rate",
    "recall_at_5",
    "stale_rate",
    "tokens_per_brief",
]


@dataclass(frozen=True)
class CheckRecord:
    """What one recall or brief check observed."""

    scenario: str
    kind: str  # "recall" or "brief"
    label: str = ""  # the query, or the actor for a brief
    top5_ok: bool | None = None  # None: the check had no expect_top5
    stale: bool = False
    items: int = 0  # results returned
    leaked: int = 0  # of which outside the caller's scope
    tokens: int | None = None  # brief only; None when not measured
    budget: int | None = None  # brief only


@dataclass(frozen=True)
class Metrics:
    recall_at_5: float | None
    recall_checks: int
    stale_rate: float | None
    stale_checks: int
    leak_rate: float | None
    leaked_items: int
    total_items: int
    tokens_mean: float | None
    tokens_max: int | None
    briefs: int


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def recall_at_5(records: list[CheckRecord]) -> float | None:
    scored = [r for r in records if r.kind == "recall" and r.top5_ok is not None]
    return _ratio(sum(1 for r in scored if r.top5_ok), len(scored))


def stale_rate(records: list[CheckRecord]) -> float | None:
    recalls = [r for r in records if r.kind == "recall"]
    return _ratio(sum(1 for r in recalls if r.stale), len(recalls))


def leak_rate(records: list[CheckRecord]) -> float | None:
    return _ratio(sum(r.leaked for r in records), sum(r.items for r in records))


def tokens_per_brief(records: list[CheckRecord]) -> tuple[float, int] | None:
    measured = [r.tokens for r in records if r.kind == "brief" and r.tokens is not None]
    if not measured:
        return None
    return sum(measured) / len(measured), max(measured)


def compute(records: list[CheckRecord]) -> Metrics:
    tokens = tokens_per_brief(records)
    return Metrics(
        recall_at_5=recall_at_5(records),
        recall_checks=sum(1 for r in records if r.kind == "recall" and r.top5_ok is not None),
        stale_rate=stale_rate(records),
        stale_checks=sum(1 for r in records if r.kind == "recall"),
        leak_rate=leak_rate(records),
        leaked_items=sum(r.leaked for r in records),
        total_items=sum(r.items for r in records),
        tokens_mean=tokens[0] if tokens else None,
        tokens_max=tokens[1] if tokens else None,
        briefs=sum(1 for r in records if r.kind == "brief" and r.tokens is not None),
    )
