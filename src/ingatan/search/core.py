"""Recall: BM25 + trigram + entity match + recency, fused by reciprocal rank fusion."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.scope import Context
from ingatan.search import stem_id

__all__ = [
    "FactHit",
    "NoteHit",
    "RecallResult",
    "is_valid_sql",
    "recall",
    "similar_facts",
    "status_of",
]

RRF_K = 60
CANDIDATES_PER_SIGNAL = 50
MAX_QUERY_TOKENS = 40
MIN_TRIGRAM_LENGTH = 3
MAX_EXPANDED_TOKENS = 60
# Weight of the two "expanded" signals (affix-stripped roots of the query words) in the
# fusion, against 1.0 for an exact signal. Two exact signals at the worst rank (50) add up
# to 2/110, and two expanded signals at the best rank to 2*0.25/61, so a fact that matches
# the literal word outranks one that only matches a root, apart from recency and entity bonuses.
EXPANDED_WEIGHT = 0.25

# Column weights for bm25(): (text, entities, keywords) and (heading, text).
_FACT_WEIGHTS = "1.0, 3.0, 2.0"
_NOTE_WEIGHTS = "4.0, 1.0"

# Words that are operators in the FTS5 query language. They are dropped from queries.
_OPERATOR_WORDS = frozenset({"and", "or", "not", "near"})
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class FactHit:
    id: str
    text: str
    tier: str
    source: dict[str, Any]
    date: str
    status: str
    entities: list[str]
    keywords: list[str]
    score: float


@dataclass(frozen=True)
class NoteHit:
    id: int
    path: str
    heading: str
    text: str


@dataclass(frozen=True)
class RecallResult:
    facts: list[FactHit] = field(default_factory=lambda: [])
    notes: list[NoteHit] = field(default_factory=lambda: [])


def is_valid_sql(alias: str = "") -> str:
    """The validity filter. Bind `:now` to `timeutil.iso(timeutil.now())`."""
    p = f"{alias}." if alias else ""
    return (
        f"({p}valid_from <= :now AND ({p}valid_to IS NULL OR {p}valid_to > :now) "
        f"AND {p}superseded_by IS NULL)"
    )


def _now() -> str:
    return timeutil.iso(timeutil.now())


def _tokens(*texts: str) -> list[str]:
    """Distinct word tokens, in order, without FTS operator words, capped in number."""
    seen: set[str] = set()
    found: list[str] = []
    for text in texts:
        for word in _WORD.findall(text):
            key = word.casefold()
            if key in seen or key in _OPERATOR_WORDS:
                continue
            seen.add(key)
            found.append(word)
            if len(found) >= MAX_QUERY_TOKENS:
                return found
    return found


def _expansions(tokens: Sequence[str]) -> list[str]:
    """Extra query terms: roots of the tokens, never a token itself, capped in number."""
    have = {t.casefold() for t in tokens}
    extra: list[str] = []
    for token in tokens:
        for root in stem_id.candidates(token):
            if root not in have:
                have.add(root)
                extra.append(root)
                if len(extra) >= MAX_EXPANDED_TOKENS:
                    return extra
    return extra


def _match(tokens: Sequence[str]) -> str:
    # Every token is quoted, so user text can never be an FTS5 syntax error.
    return " OR ".join(f'"{t}"' for t in tokens)


def _json_list(raw: str) -> list[str]:
    value: Any = json.loads(raw)
    return [str(v) for v in value] if isinstance(value, list) else []


def _json_dict(raw: str) -> dict[str, Any]:
    value: Any = json.loads(raw)
    return value if isinstance(value, dict) else {}


def status_of(ctx: Context, fact_row: Mapping[str, Any] | sqlite3.Row) -> str:
    """`valid`, `check first`, `possible conflict` (both joined with a comma),
    or for a closed fact `superseded` / `expired`."""
    row: Any = fact_row
    now = _now()
    if row["superseded_by"] is not None:
        return "superseded"
    if row["valid_to"] is not None and row["valid_to"] <= now:
        return "expired"
    parts: list[str] = []
    review = row["review_after"]
    overdue = row["kind"] == "volatile" and review is not None and review < now
    if overdue or row["pending_proposal"]:
        parts.append("check first")
    if _in_open_conflict(ctx, row["id"]):
        parts.append("possible conflict")
    return ", ".join(parts) if parts else "valid"


def _in_open_conflict(ctx: Context, fact_id: str) -> bool:
    """An unresolved pair counts only while the other fact is one the caller can see."""
    rows = ctx.business.execute(
        "SELECT CASE WHEN fact_a = :id THEN fact_b ELSE fact_a END AS other FROM conflicts "
        "WHERE resolved_at IS NULL AND (fact_a = :id OR fact_b = :id)",
        {"id": fact_id},
    ).fetchall()
    for r in rows:
        for conn in ctx.connections():
            hit = conn.execute(
                f"SELECT 1 FROM facts WHERE id = :other AND {ctx.visible_sql(conn)}",
                {"other": r["other"], **ctx.params},
            ).fetchone()
            if hit is not None:
                return True
    return False


@dataclass
class _Candidate:
    row: sqlite3.Row
    conn: sqlite3.Connection
    score: float = 0.0


def _fact_matches(
    ctx: Context, conn: sqlite3.Connection, table: str, match: str, closed_ok: bool
) -> list[sqlite3.Row]:
    validity = "f.valid_from <= :now" if closed_ok else is_valid_sql("f")
    sql = (
        f"SELECT f.*, bm25({table}, {_FACT_WEIGHTS}) AS bm FROM {table} "
        f"JOIN facts f ON f.seq = {table}.rowid "
        f"WHERE {table} MATCH :q AND {validity} AND {ctx.visible_sql(conn, alias='f')} "
        "ORDER BY bm LIMIT :lim"
    )
    params: dict[str, Any] = {"q": match, "now": _now(), "lim": CANDIDATES_PER_SIGNAL}
    params.update(ctx.params)
    return conn.execute(sql, params).fetchall()


def _bm25_scores(
    ctx: Context, table: str, match: str, closed_ok: bool, pool: dict[str, _Candidate]
) -> dict[str, float]:
    """BM25 of every candidate of one index, from both files. Fills `pool`."""
    scores: dict[str, float] = {}
    for conn in ctx.connections():
        for row in _fact_matches(ctx, conn, table, match, closed_ok):
            pool.setdefault(row["id"], _Candidate(row, conn))
            scores[row["id"]] = row["bm"]
    return scores


def _entity_count(entities: list[str], query: str) -> int:
    lowered = query.casefold()
    count = 0
    for entity in entities:
        name = entity.casefold().strip()
        if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered):
            count += 1
    return count


def _competition_ranks(values: Mapping[str, float]) -> dict[str, int]:
    """Rank 1 for the highest value; equal values share a rank."""
    ordered = sorted(values.items(), key=lambda kv: -kv[1])
    ranks: dict[str, int] = {}
    previous: float | None = None
    rank = 0
    for position, (key, value) in enumerate(ordered, start=1):
        if value != previous:
            rank = position
            previous = value
        ranks[key] = rank
    return ranks


def _hit(ctx: Context, cand: _Candidate) -> FactHit:
    row = cand.row
    source = _json_dict(row["source"])
    return FactHit(
        id=row["id"],
        text=row["text"],
        tier=row["tier"],
        source=source,
        date=str(source.get("date") or row["valid_from"][:10]),
        status=status_of(ctx, row),
        entities=_json_list(row["entities"]),
        keywords=_json_list(row["keywords"]),
        score=cand.score,
    )


def _touch(hits: list[FactHit], pool: dict[str, _Candidate]) -> None:
    now = _now()
    by_conn: dict[int, tuple[sqlite3.Connection, list[str]]] = {}
    for hit in hits:
        conn = pool[hit.id].conn
        by_conn.setdefault(id(conn), (conn, []))[1].append(hit.id)
    for conn, ids in by_conn.values():
        with transaction(conn):
            conn.executemany(
                "UPDATE facts SET use_count = use_count + 1, last_used = ? WHERE id = ?",
                [(now, i) for i in ids],
            )


def recall(
    ctx: Context,
    query: str,
    *,
    limit: int = 5,
    include_closed: bool = False,
    touch: bool = True,
) -> RecallResult:
    """Facts (both files, visibility applied) and document notes matching `query`."""
    tokens = _tokens(query)
    if not tokens or limit <= 0:
        return RecallResult()
    trigram_tokens = [t for t in tokens if len(t) >= MIN_TRIGRAM_LENGTH]

    pool: dict[str, _Candidate] = {}
    word_bm = _bm25_scores(ctx, "facts_fts", _match(tokens), include_closed, pool)
    tri_bm = (
        _bm25_scores(ctx, "facts_tri", _match(trigram_tokens), include_closed, pool)
        if trigram_tokens
        else {}
    )

    # Roots of the query words, against both indexes, as separate and weaker signals.
    extra = _expansions(tokens)
    exp_word_bm: dict[str, float] = {}
    exp_tri_bm: dict[str, float] = {}
    if extra:
        exp_word_bm = _bm25_scores(ctx, "facts_fts", _match(extra), include_closed, pool)
        exp_tri_bm = _bm25_scores(ctx, "facts_tri", _match(extra), include_closed, pool)

    scores: dict[str, float] = dict.fromkeys(pool, 0.0)
    for bm, weight in (
        (word_bm, 1.0),
        (tri_bm, 1.0),
        (exp_word_bm, EXPANDED_WEIGHT),
        (exp_tri_bm, EXPANDED_WEIGHT),
    ):
        # bm25() is lower for better matches; equal scores share a rank.
        for fact_id, rank in _competition_ranks({i: -v for i, v in bm.items()}).items():
            scores[fact_id] += weight / (RRF_K + rank)

    # A signal adds nothing for a fact it did not match, so an entity match lifts a fact
    # above an otherwise equal one.
    entity_counts: dict[str, float] = {}
    for fact_id, cand in pool.items():
        count = _entity_count(_json_list(cand.row["entities"]), query)
        if count:
            entity_counts[fact_id] = float(count)
    for fact_id, rank in _competition_ranks(entity_counts).items():
        scores[fact_id] += 1.0 / (RRF_K + rank)

    # Recency: newest valid_from first. ISO strings sort as text, so map them to ordinals.
    stamps = sorted({c.row["valid_from"] for c in pool.values()})
    ordinal = {stamp: float(i) for i, stamp in enumerate(stamps)}
    recency = {i: ordinal[c.row["valid_from"]] for i, c in pool.items()}
    for fact_id, rank in _competition_ranks(recency).items():
        scores[fact_id] += 1.0 / (RRF_K + rank)

    # Ties on score fall back to the newest fact, then the id.
    ordered = sorted(pool, key=lambda i: (-round(scores[i], 12), -recency[i], i))[:limit]
    for fact_id in ordered:
        pool[fact_id].score = scores[fact_id]
    hits = [_hit(ctx, pool[i]) for i in ordered]
    if touch and hits:
        _touch(hits, pool)
    return RecallResult(facts=hits, notes=_notes(ctx, tokens, trigram_tokens, extra, limit))


def _notes(
    ctx: Context, tokens: list[str], trigram_tokens: list[str], extra: list[str], limit: int
) -> list[NoteHit]:
    conn = ctx.business
    rows: dict[int, sqlite3.Row] = {}
    scores: dict[int, float] = {}
    signals = (
        ("notes_fts", tokens, 1.0),
        ("notes_tri", trigram_tokens, 1.0),
        ("notes_fts", extra, EXPANDED_WEIGHT),
        ("notes_tri", extra, EXPANDED_WEIGHT),
    )
    for table, toks, weight in signals:
        if not toks:
            continue
        found = conn.execute(
            f"SELECT n.*, bm25({table}, {_NOTE_WEIGHTS}) AS bm FROM {table} "
            f"JOIN notes n ON n.id = {table}.rowid WHERE {table} MATCH :q "
            "ORDER BY bm LIMIT :lim",
            {"q": _match(toks), "lim": CANDIDATES_PER_SIGNAL},
        ).fetchall()
        for rank, row in enumerate(found, start=1):
            rows.setdefault(row["id"], row)
            scores[row["id"]] = scores.get(row["id"], 0.0) + weight / (RRF_K + rank)
    best = sorted(rows, key=lambda i: (-scores[i], i))[:limit]
    return [
        NoteHit(id=i, path=rows[i]["path"], heading=rows[i]["heading"], text=rows[i]["text"])
        for i in best
    ]


def similar_facts(
    ctx: Context,
    text: str,
    entities: Sequence[str],
    keywords: Sequence[str],
    *,
    exclude_id: str | None = None,
    limit: int = 3,
) -> list[FactHit]:
    """The top FTS matches among valid facts that share at least one entity (OQ-7).

    Never touches `use_count`.
    """
    wanted = {e.casefold().strip() for e in entities if e.strip()}
    tokens = _tokens(" ".join(entities), " ".join(keywords), text)
    if not wanted or not tokens or limit <= 0:
        return []
    pool: dict[str, _Candidate] = {}
    scored: list[tuple[float, str]] = []
    for conn in ctx.connections():
        for row in _fact_matches(ctx, conn, "facts_fts", _match(tokens), False):
            if row["id"] == exclude_id:
                continue
            if wanted.isdisjoint(e.casefold().strip() for e in _json_list(row["entities"])):
                continue
            pool[row["id"]] = _Candidate(row, conn)
            scored.append((row["bm"], row["id"]))
    scored.sort()
    return [_hit(ctx, pool[i]) for _, i in scored[:limit]]
