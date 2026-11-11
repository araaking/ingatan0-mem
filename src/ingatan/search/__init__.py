"""Search: `recall` (facts and document notes), `similar_facts`, and fact status."""

from __future__ import annotations

from ingatan.search.core import (
    FactHit,
    NoteHit,
    RecallResult,
    is_valid_sql,
    recall,
    similar_facts,
    status_of,
)

__all__ = [
    "FactHit",
    "NoteHit",
    "RecallResult",
    "is_valid_sql",
    "recall",
    "similar_facts",
    "status_of",
]
