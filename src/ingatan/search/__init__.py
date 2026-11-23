"""Search: `recall` (facts and document notes), `similar_facts`, and fact status."""

from __future__ import annotations

from ingatan.search.aliases import (
    add_alias,
    expand_query_with_aliases,
    list_aliases,
    remove_alias,
)
from ingatan.search.core import (
    CATEGORIES,
    FactHit,
    NoteHit,
    RecallResult,
    is_valid_sql,
    recall,
    similar_facts,
    status_of,
)
from ingatan.search.embed import (
    ReindexReport,
    last_embed_error,
    probe_embedding,
    reindex,
    unload_local_models,
)

__all__ = [
    "CATEGORIES",
    "FactHit",
    "NoteHit",
    "RecallResult",
    "ReindexReport",
    "add_alias",
    "expand_query_with_aliases",
    "is_valid_sql",
    "last_embed_error",
    "list_aliases",
    "probe_embedding",
    "recall",
    "reindex",
    "remove_alias",
    "similar_facts",
    "status_of",
    "unload_local_models",
]
