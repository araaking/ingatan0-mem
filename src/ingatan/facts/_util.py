"""Small helpers shared by the facts modules (not part of the public API)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.redact import redact
from ingatan.scope import Context

DEFAULT_REVIEW_DAYS = 30


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:12]


def now_iso() -> str:
    return timeutil.iso(timeutil.now())


def clean_list(items: Iterable[str] | None) -> list[str]:
    """Redact, strip and de-duplicate (case-insensitively) a list of short strings."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items or []:
        value = redact(str(item)).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def loads_list(raw: str) -> list[str]:
    value: Any = json.loads(raw)
    return [str(v) for v in value] if isinstance(value, list) else []


def loads_dict(raw: str) -> dict[str, Any]:
    value: Any = json.loads(raw)
    return value if isinstance(value, dict) else {}


def fact_dict(row: Mapping[str, Any] | sqlite3.Row) -> dict[str, Any]:
    """Every column of a fact row (JSON columns decoded) except the internal `seq`."""
    data: Any = row
    keys = data.keys()
    out: dict[str, Any] = {k: data[k] for k in keys if k != "seq"}
    out["entities"] = loads_list(out["entities"])
    out["keywords"] = loads_list(out["keywords"])
    out["source"] = loads_dict(out["source"])
    return out


def fact_fields(row: sqlite3.Row, *names: str) -> dict[str, Any]:
    return {n: row[n] for n in names}


def existing_fact_ids(ctx: Context, ids: Iterable[str]) -> list[str]:
    """The ids that exist as facts in either file (no visibility check)."""
    found: list[str] = []
    for fact_id in dict.fromkeys(ids):
        for conn in ctx.connections():
            if conn.execute("SELECT 1 FROM facts WHERE id = ?", (fact_id,)).fetchone():
                found.append(fact_id)
                break
    return found


def record_conflicts(ctx: Context, new_id_: str, others: Iterable[str]) -> list[str]:
    """Open a `conflicts` pair between `new_id_` and each existing id. Returns those ids.

    Pairs always live in the business file, even when one fact is global.
    """
    ids = [i for i in existing_fact_ids(ctx, others) if i != new_id_]
    if not ids:
        return []
    now = now_iso()
    with transaction(ctx.business):
        ctx.business.executemany(
            "INSERT INTO conflicts (fact_a, fact_b, created_at) VALUES (?, ?, ?)",
            [(new_id_, other, now) for other in ids],
        )
    return ids


def resolve_conflicts(
    ctx: Context, fact_id: str, resolution: str, *, only_with: Iterable[str] | None = None
) -> list[int]:
    """Resolve open pairs of `fact_id` (all, or only those naming one of `only_with`).

    Returns the ids of the pair rows that were resolved.
    """
    wanted = None if only_with is None else set(only_with)
    now = now_iso()
    with transaction(ctx.business):
        rows = ctx.business.execute(
            "SELECT id, fact_a, fact_b FROM conflicts "
            "WHERE resolved_at IS NULL AND (fact_a = :id OR fact_b = :id)",
            {"id": fact_id},
        ).fetchall()
        resolved: list[int] = []
        for r in rows:
            other = r["fact_b"] if r["fact_a"] == fact_id else r["fact_a"]
            if wanted is not None and other not in wanted:
                continue
            ctx.business.execute(
                "UPDATE conflicts SET resolved_at = ?, resolution = ? WHERE id = ?",
                (now, resolution, r["id"]),
            )
            resolved.append(r["id"])
    return resolved


def is_closed(row: sqlite3.Row) -> bool:
    return row["superseded_by"] is not None or (
        row["valid_to"] is not None and row["valid_to"] <= now_iso()
    )
