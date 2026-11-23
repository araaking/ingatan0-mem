"""Recall: BM25 + trigram + entity match + recency, fused by reciprocal rank fusion."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.errors import ValidationError
from ingatan.scope import Context
from ingatan.search import embed, stem_id
from ingatan.search.aliases import expand_query_with_aliases
from ingatan.search.stopwords import (
    CONFLICT_NOISE_WORDS,
    STOPWORDS,
    extract_content_tokens,
    extract_identifier_phrases,
    extract_resource_ids,
)

__all__ = [
    "CATEGORIES",
    "FactHit",
    "NoteHit",
    "RecallResult",
    "is_valid_sql",
    "recall",
    "similar_facts",
    "status_of",
]

CATEGORIES = ("fact", "rule", "procedure", "preference")

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
# Freshness of a volatile fact, from its age and not from its rank among the candidates: the
# score is multiplied by 1 + RECENCY_BOOST * (0.5 ** (age_days / half_life) - 0.5). A fact
# written today gains 2%, one a half-life old is unchanged, a very old one loses 2%. That is
# enough to pass a neighbour ranked one place higher (1/61 against 1/62 is 1.6%) and never
# more. Stable facts are not time-sensitive and keep their score. The half-life equals the
# default review interval, so a fact starts to lose ground about when it is due for review.
RECENCY_BOOST = 0.04
RECENCY_HALF_LIFE_DAYS = 30.0
# Usage: a fact that `recall` returned often and recently gains up to USAGE_BOOST. The score
# is multiplied by 1 + USAGE_BOOST * often * recent, where `often` is
# 1 - 0.5 ** (use_count / USAGE_COUNT_SATURATION) and `recent` is
# 0.5 ** (days_since_last_used / USAGE_HALF_LIFE_DAYS). A fact never recalled keeps its score.
#
# What the 1% bound gives. It is smaller than the gap between two neighbouring ranks of one
# signal among the first forty (1/100 against 1/101), so within a single signal usage cannot
# reorder two facts. Fused scores of facts that several signals rank differently can be
# closer than 1%, and there usage decides: that is what it is for, and it is why the bound
# is no larger. The count saturates (five recalls give half the effect, twenty give 94%) and
# the effect halves every 30 days without use, because `recall` itself raises the count:
# without both, what was returned once would be returned more and more.
#
# What keeps a popular wrong value down. A fact named by an unresolved conflict pair gets no
# boost at all: its popularity is exactly what is in doubt. That, and not the bound, is what
# puts a newer fact ahead of the much recalled one it corrects. Where the two were never
# paired (no shared entity, or `check_conflicts=False`), the older one can still lead a
# fact that is only a few days younger: freshness separates two volatile facts by less than
# 1% until they are about twelve days apart, and stable facts not at all.
#
# Measured with `python -m evals` without an embedding model (48 scenarios, 2026-11-23):
# `usage-02-pemakaian-memecah-seri` passes (xfail before), `usage-01-fakta-populer-yang-salah`
# still passes (through the conflict rule), no other scenario changes, recall@5 100.0% over
# 55 checks. Not run with a model.
USAGE_BOOST = 0.01
USAGE_HALF_LIFE_DAYS = 30.0
USAGE_COUNT_SATURATION = 5
# A lexical hit this far below the best hit of its own index and file is a weak tail hit.
LEXICAL_TAIL_RATIO = 0.25
NOTE_TAIL_RATIO = 0.35
# Identifier phrase bonus (e.g. `sync_report` -> `"sync report"` matching `sync_report.py`).
PHRASE_WEIGHT = 1.0
# Project alias glossary expansion weight.
ALIAS_WEIGHT = 0.85
# Semantic embedding similarity weight in RRF (strong enough to surface synonym-only matches).
SEMANTIC_WEIGHT = 1.5
# Gentle tie-breaking boost for pinned facts, rules, and procedures.
ROLE_BOOST = 0.25 / (RRF_K + 1)
# Default maximum number of document excerpts and 1-hop related facts returned alongside hits.
DEFAULT_NOTES_LIMIT = 2
MAX_RELATED = 3
# How many links away a related fact may be. Two reaches the fact behind a neighbour (an
# order, its vendor, the vendor's closing day); more would return what nobody asked about.
MAX_HOPS = 2

# Conflict detection tuning (prevents false positives on generic entities and noisy overlaps).
GENERIC_ENTITY_MIN_FACTS = 10
GENERIC_ENTITY_FREQ_RATIO = 0.35
MIN_CONFLICT_OVERLAP_RATIO = 0.35
GENERIC_ONLY_OVERLAP_RATIO = 0.50
# How many lexically closest facts per file `similar_facts` examines in detail. A store with
# fewer facts sharing the entity than this is examined in full.
SIMILAR_CANDIDATE_LIMIT = 200

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
    category: str = "fact"
    pinned: bool = False


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
    related: list[FactHit] = field(default_factory=lambda: [])


def is_valid_sql(alias: str = "", *, as_of_mode: bool = False) -> str:
    """The validity filter. Bind `:now` to `timeutil.iso(timeutil.now())` (or `as_of`).

    When `as_of_mode` is True (time-travel query), facts that were valid at `:now` are included
    even if they were superseded at a later timestamp.
    """
    p = f"{alias}." if alias else ""
    if as_of_mode:
        return f"({p}valid_from <= :now AND ({p}valid_to IS NULL OR {p}valid_to > :now))"
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
    """An unresolved pair counts only while the other fact is still valid and visible."""
    now = _now()
    rows = ctx.business.execute(
        "SELECT CASE WHEN fact_a = :id THEN fact_b ELSE fact_a END AS other FROM conflicts "
        "WHERE resolved_at IS NULL AND (fact_a = :id OR fact_b = :id)",
        {"id": fact_id},
    ).fetchall()
    for r in rows:
        for conn in ctx.connections():
            hit = conn.execute(
                f"SELECT 1 FROM facts WHERE id = :other AND {is_valid_sql()} "
                f"AND {ctx.visible_sql(conn)}",
                {"other": r["other"], "now": now, **ctx.params},
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
    ctx: Context,
    conn: sqlite3.Connection,
    table: str,
    match: str,
    validity: str,
    time_param: str,
) -> list[sqlite3.Row]:
    sql = (
        f"SELECT f.*, bm25({table}, {_FACT_WEIGHTS}) AS bm FROM {table} "
        f"JOIN facts f ON f.seq = {table}.rowid "
        f"WHERE {table} MATCH :q AND {validity} AND {ctx.visible_sql(conn, alias='f')} "
        "ORDER BY bm LIMIT :lim"
    )
    params: dict[str, Any] = {"q": match, "now": time_param, "lim": CANDIDATES_PER_SIGNAL}
    params.update(ctx.params)
    return conn.execute(sql, params).fetchall()


def _bm25_scores(
    ctx: Context,
    table: str,
    match: str,
    validity: str,
    time_param: str,
    pool: dict[str, _Candidate],
    strength: dict[str, float] | None = None,
) -> dict[str, float]:
    """BM25 of every candidate of one index, from both files. Fills `pool`.

    `strength`, when given, receives each hit's BM25 as a share of the best hit of the same
    index in the same file (1.0 for that best hit). A raw BM25 value depends on the index,
    the column weights and the size of the file, so only this share is comparable.
    """
    scores: dict[str, float] = {}
    for conn in ctx.connections():
        rows = _fact_matches(ctx, conn, table, match, validity, time_param)
        top = max((abs(float(row["bm"])) for row in rows), default=0.0)
        for row in rows:
            pool.setdefault(row["id"], _Candidate(row, conn))
            scores[row["id"]] = row["bm"]
            if strength is not None and top > 0.0:
                share = abs(float(row["bm"])) / top
                strength[row["id"]] = max(strength.get(row["id"], 0.0), share)
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


def _hit(ctx: Context, cand: _Candidate, *, as_of: str | None = None) -> FactHit:
    row = cand.row
    source = _json_dict(row["source"])
    keys = row.keys()
    category = str(row["category"]) if "category" in keys and row["category"] else "fact"
    pinned = bool(row["pinned"]) if "pinned" in keys else False
    status = "valid (as of)" if as_of is not None else status_of(ctx, row)
    return FactHit(
        id=row["id"],
        text=row["text"],
        tier=row["tier"],
        source=source,
        date=str(source.get("date") or row["valid_from"][:10]),
        status=status,
        entities=_json_list(row["entities"]),
        keywords=_json_list(row["keywords"]),
        score=cand.score,
        category=category,
        pinned=pinned,
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


def _valid_fact_count(ctx: Context) -> int:
    now = _now()
    total = 0
    for conn in ctx.connections():
        total += int(
            conn.execute(
                f"SELECT count(*) FROM facts WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)}",
                {"now": now, **ctx.params},
            ).fetchone()[0]
        )
    return total


def _freshness_factor(row: sqlite3.Row, reference: datetime) -> float:
    """Score multiplier between 1 - RECENCY_BOOST/2 and 1 + RECENCY_BOOST/2 (stable: 1)."""
    if row["kind"] != "volatile":
        return 1.0
    try:
        written = timeutil.parse(str(row["valid_from"]))
    except ValidationError:
        return 1.0
    age_days = max(0.0, (reference - written).total_seconds() / 86400.0)
    freshness = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return 1.0 + RECENCY_BOOST * (freshness - 0.5)


def _usage_factor(row: sqlite3.Row, reference: datetime) -> float:
    """Score multiplier between 1 and 1 + USAGE_BOOST (see the constants). Never used: 1."""
    count = int(row["use_count"] or 0)
    last_used = row["last_used"]
    if count <= 0 or not last_used:
        return 1.0
    try:
        used = timeutil.parse(str(last_used))
    except ValidationError:
        return 1.0
    idle_days = max(0.0, (reference - used).total_seconds() / 86400.0)
    often = 1.0 - 0.5 ** (count / USAGE_COUNT_SATURATION)
    recent = 0.5 ** (idle_days / USAGE_HALF_LIFE_DAYS)
    return 1.0 + USAGE_BOOST * often * recent


def _open_conflict_ids(ctx: Context) -> set[str]:
    """Every fact id named by an unresolved conflict pair (pairs live in the business file)."""
    ids: set[str] = set()
    for row in ctx.business.execute(
        "SELECT fact_a, fact_b FROM conflicts WHERE resolved_at IS NULL"
    ):
        ids.add(str(row["fact_a"]))
        ids.add(str(row["fact_b"]))
    return ids


def _row_blob(row: sqlite3.Row) -> str:
    ents = " ".join(_json_list(row["entities"]))
    kws = " ".join(_json_list(row["keywords"]))
    return f"{row['text']} {ents} {kws}".casefold()


def _matched_term_count(blob: str, terms: Sequence[str], extra: Sequence[str]) -> int:
    """Count distinct query tokens matched in `blob` (exact word or trigram/root substring)."""
    words = {w.casefold() for w in _WORD.findall(blob)}
    hits = 0
    for tok in terms:
        low = tok.casefold()
        if low in words or (len(low) >= MIN_TRIGRAM_LENGTH and low in blob):
            hits += 1
            continue
        for root in stem_id.candidates(low):
            if root in words or (len(root) >= MIN_TRIGRAM_LENGTH and root in blob):
                hits += 1
                break
    if hits == 0 and extra:
        for root in extra:
            if root in words or (len(root) >= MIN_TRIGRAM_LENGTH and root in blob):
                hits += 1
                break
    return hits


def _neighbor_ids(ctx: Context, fact_id: str) -> list[str]:
    """The facts linked to `fact_id` in either direction, in the order they were linked."""
    rows = ctx.business.execute(
        "SELECT CASE WHEN src_id = :id THEN dst_id ELSE src_id END AS other "
        "FROM edges WHERE src_id = :id OR dst_id = :id ORDER BY id ASC",
        {"id": fact_id},
    ).fetchall()
    return [str(row["other"]) for row in rows]


def _related_hits(
    ctx: Context,
    hits: Sequence[FactHit],
    validity_sql: str,
    time_param: str,
    *,
    as_of: str | None = None,
    limit: int = MAX_RELATED,
) -> list[FactHit]:
    """Valid visible facts linked to the top `hits`, up to `MAX_HOPS` links away.

    Breadth first: every direct neighbour comes before anything behind it, so a second hop
    is shown only in the room the first hop leaves under `limit`. A fact that is closed, or
    that the caller cannot see, is not shown and is not walked through either: nothing is
    reached by way of a fact one may not read.
    """
    if not hits or limit <= 0:
        return []
    seen = {h.id for h in hits}
    related: list[FactHit] = []
    frontier = [h.id for h in hits]
    for _hop in range(MAX_HOPS):
        reached: list[str] = []
        for fact_id in frontier:
            for other in _neighbor_ids(ctx, fact_id):
                if other in seen:
                    continue
                seen.add(other)
                for conn in ctx.connections():
                    row = conn.execute(
                        f"SELECT f.* FROM facts f WHERE f.id = :id AND {validity_sql} "
                        f"AND {ctx.visible_sql(conn, alias='f')}",
                        {"id": other, "now": time_param, **ctx.params},
                    ).fetchone()
                    if row is not None:
                        related.append(_hit(ctx, _Candidate(row, conn), as_of=as_of))
                        reached.append(other)
                        break
                if len(related) >= limit:
                    return related
        frontier = reached
        if not frontier:
            break
    return related


def recall(
    ctx: Context,
    query: str,
    *,
    limit: int = 5,
    notes_limit: int = DEFAULT_NOTES_LIMIT,
    include_closed: bool = False,
    touch: bool = True,
    as_of: str | None = None,
    category: str | None = None,
    include_related: bool = True,
) -> RecallResult:
    """Facts (both files, visibility applied), document notes, and related facts."""
    if category is not None and category not in CATEGORIES:
        raise ValidationError(
            f"Unknown category '{category}'. Use one of: {', '.join(CATEGORIES)}."
        )
    raw_tokens = _tokens(query)
    if not raw_tokens or limit <= 0:
        return RecallResult()

    resolved_as_of = timeutil.normalize(as_of) if as_of else None
    time_param = resolved_as_of if resolved_as_of is not None else _now()
    if resolved_as_of is not None:
        validity_sql = is_valid_sql("f", as_of_mode=True)
    elif include_closed:
        validity_sql = "f.valid_from <= :now"
    else:
        validity_sql = is_valid_sql("f")

    tokens = [t for t in raw_tokens if t.casefold() not in STOPWORDS] or raw_tokens
    non_noise = [t for t in tokens if t.casefold() not in CONFLICT_NOISE_WORDS]
    trigram_tokens = [t for t in tokens if len(t) >= MIN_TRIGRAM_LENGTH]
    phrases = extract_identifier_phrases(query)

    # Project alias glossary expansion (deterministic, 0 ms).
    alias_phrases = expand_query_with_aliases(ctx, query)
    alias_tokens = _tokens(*alias_phrases) if alias_phrases else []
    alias_tri = [t for t in alias_tokens if len(t) >= MIN_TRIGRAM_LENGTH]

    pool: dict[str, _Candidate] = {}
    lexical_strength: dict[str, float] = {}
    word_bm = _bm25_scores(
        ctx, "facts_fts", _match(tokens), validity_sql, time_param, pool, lexical_strength
    )
    tri_bm = (
        _bm25_scores(
            ctx,
            "facts_tri",
            _match(trigram_tokens),
            validity_sql,
            time_param,
            pool,
            lexical_strength,
        )
        if trigram_tokens
        else {}
    )
    phrase_bm = (
        _bm25_scores(ctx, "facts_fts", _match(phrases), validity_sql, time_param, pool)
        if phrases
        else {}
    )
    alias_word_bm = (
        _bm25_scores(ctx, "facts_fts", _match(alias_tokens), validity_sql, time_param, pool)
        if alias_tokens
        else {}
    )
    alias_tri_bm = (
        _bm25_scores(ctx, "facts_tri", _match(alias_tri), validity_sql, time_param, pool)
        if alias_tri
        else {}
    )

    # Roots of the query words, against both indexes, as separate and weaker signals.
    extra = _expansions(tokens)
    exp_word_bm: dict[str, float] = {}
    exp_tri_bm: dict[str, float] = {}
    if extra:
        exp_word_bm = _bm25_scores(ctx, "facts_fts", _match(extra), validity_sql, time_param, pool)
        exp_tri_bm = _bm25_scores(ctx, "facts_tri", _match(extra), validity_sql, time_param, pool)

    # Optional semantic embedding signal (pluggable, fails gracefully to empty).
    query_vec: list[float] | None = None
    sem_scores: dict[str, float] = {}
    if embed.is_enabled(ctx.scope.embeddings):
        query_vec = embed.embed_text(ctx.scope.embeddings, query, is_query=True)
        if query_vec:
            sem_scores, sem_rows = embed.semantic_fact_scores(
                ctx,
                query,
                validity_sql=validity_sql,
                time_param=time_param,
                query_vec=query_vec,
            )
            for fid, (conn, row) in sem_rows.items():
                pool.setdefault(fid, _Candidate(row, conn))

    scores: dict[str, float] = dict.fromkeys(pool, 0.0)
    for bm, weight in (
        (word_bm, 1.0),
        (tri_bm, 1.0),
        (phrase_bm, PHRASE_WEIGHT),
        (alias_word_bm, ALIAS_WEIGHT),
        (alias_tri_bm, ALIAS_WEIGHT),
        (exp_word_bm, EXPANDED_WEIGHT),
        (exp_tri_bm, EXPANDED_WEIGHT),
    ):
        # bm25() is lower for better matches; equal scores share a rank.
        for fact_id, rank in _competition_ranks({i: -v for i, v in bm.items()}).items():
            scores[fact_id] += weight / (RRF_K + rank)

    if sem_scores:
        for fact_id, rank in _competition_ranks(sem_scores).items():
            scores[fact_id] += SEMANTIC_WEIGHT / (RRF_K + rank)

    # A signal adds nothing for a fact it did not match, so an entity match lifts a fact
    # above an otherwise equal one.
    expanded_query = f"{query} {' '.join(alias_phrases)}" if alias_phrases else query
    entity_counts: dict[str, float] = {}
    for fact_id, cand in pool.items():
        count = _entity_count(_json_list(cand.row["entities"]), expanded_query)
        if count:
            entity_counts[fact_id] = float(count)
    for fact_id, rank in _competition_ranks(entity_counts).items():
        scores[fact_id] += 1.0 / (RRF_K + rank)

    # Filter out noise-only hits on multi-term queries (preserving semantic & alias hits).
    if pool and len(tokens) >= 3 and non_noise:
        kept: dict[str, _Candidate] = {}
        term_hits: dict[str, int] = {}
        for fact_id, cand in pool.items():
            blob = _row_blob(cand.row)
            specific_hits = _matched_term_count(blob, non_noise, extra)
            all_hits = _matched_term_count(blob, tokens, extra)
            term_hits[fact_id] = specific_hits
            if (
                specific_hits >= 1
                or all_hits >= 2
                or fact_id in entity_counts
                or fact_id in phrase_bm
                or fact_id in sem_scores
                or fact_id in alias_word_bm
                or fact_id in alias_tri_bm
            ):
                kept[fact_id] = cand
        pool = kept
        # When the store is non-trivial, drop weak single-term tail hits: those far below
        # the best hit of their own index and file, with nothing else speaking for them.
        if len(pool) > 1 and _valid_fact_count(ctx) >= 8:
            pool = {
                fid: cand
                for fid, cand in pool.items()
                if (
                    term_hits.get(fid, 0) >= 2
                    or fid in entity_counts
                    or fid in phrase_bm
                    or fid in sem_scores
                    or fid in alias_word_bm
                    or fid in alias_tri_bm
                    or lexical_strength.get(fid, 0.0) >= LEXICAL_TAIL_RATIO
                )
            }

    # Optional cognitive category filter.
    if category is not None:
        pool = {
            fid: cand
            for fid, cand in pool.items()
            if ("category" in set(cand.row.keys()) and cand.row["category"] == category)
        }

    # Gentle priority boost for pinned facts, rules, and procedures.
    for fact_id, cand in pool.items():
        keys = cand.row.keys()
        is_pinned = bool(cand.row["pinned"]) if "pinned" in keys else False
        cat = str(cand.row["category"]) if "category" in keys else "fact"
        if is_pinned or cat in ("rule", "procedure"):
            scores[fact_id] += ROLE_BOOST

    # Freshness: a bounded boost from the age of a volatile fact (see RECENCY_BOOST).
    reference = timeutil.parse(time_param)
    for fact_id, cand in pool.items():
        scores[fact_id] *= _freshness_factor(cand.row, reference)

    # Usage: a bounded boost for what was recalled often and recently (see USAGE_BOOST). Not
    # for a look into the past, where today's usage says nothing, and not for a fact whose
    # truth is in doubt.
    if resolved_as_of is None and any(int(c.row["use_count"] or 0) > 0 for c in pool.values()):
        doubted = _open_conflict_ids(ctx)
        for fact_id, cand in pool.items():
            if fact_id not in doubted:
                scores[fact_id] *= _usage_factor(cand.row, reference)

    # ISO strings sort as text, so map them to ordinals for the tie-break below.
    stamps = sorted({c.row["valid_from"] for c in pool.values()})
    ordinal = {stamp: float(i) for i, stamp in enumerate(stamps)}
    recency = {i: ordinal[c.row["valid_from"]] for i, c in pool.items()}

    # Ties on score fall back to the newest fact, then the id.
    ordered = sorted(pool, key=lambda i: (-round(scores[i], 12), -recency[i], i))[:limit]
    for fact_id in ordered:
        pool[fact_id].score = scores[fact_id]
    hits = [_hit(ctx, pool[i], as_of=resolved_as_of) for i in ordered]
    if touch and hits and resolved_as_of is None:
        _touch(hits, pool)
    related = (
        _related_hits(ctx, hits, validity_sql, time_param, as_of=resolved_as_of)
        if include_related and hits
        else []
    )
    effective_notes = min(limit, notes_limit) if notes_limit > 0 else 0
    notes = (
        _notes(
            ctx,
            query,
            tokens,
            non_noise,
            trigram_tokens,
            extra,
            alias_tokens,
            effective_notes,
            query_vec,
        )
        if effective_notes > 0
        else []
    )
    return RecallResult(facts=hits, notes=notes, related=related)


def _notes(
    ctx: Context,
    query: str,
    tokens: list[str],
    non_noise: list[str],
    trigram_tokens: list[str],
    extra: list[str],
    alias_tokens: list[str],
    limit: int,
    query_vec: list[float] | None,
) -> list[NoteHit]:
    conn = ctx.business
    rows: dict[int, sqlite3.Row] = {}
    scores: dict[int, float] = {}
    # Each exact-signal hit's BM25 as a share of the best hit of the same index.
    strength: dict[int, float] = {}
    signals = (
        ("notes_fts", tokens, 1.0),
        ("notes_tri", trigram_tokens, 1.0),
        ("notes_fts", alias_tokens, ALIAS_WEIGHT),
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
        top = max((abs(float(row["bm"])) for row in found), default=0.0)
        for rank, row in enumerate(found, start=1):
            nid = row["id"]
            rows.setdefault(nid, row)
            scores[nid] = scores.get(nid, 0.0) + weight / (RRF_K + rank)
            if weight >= 1.0 and top > 0.0:
                strength[nid] = max(strength.get(nid, 0.0), abs(float(row["bm"])) / top)

    sem_note_ids: set[int] = set()
    if query_vec:
        sem_scores, sem_rows = embed.semantic_note_scores(ctx, query, query_vec=query_vec)
        for nid, row in sem_rows.items():
            rows.setdefault(nid, row)
            sem_note_ids.add(nid)
        for nid_str, rank in _competition_ranks({str(k): v for k, v in sem_scores.items()}).items():
            nid = int(nid_str)
            scores[nid] = scores.get(nid, 0.0) + SEMANTIC_WEIGHT / (RRF_K + rank)

    if not rows:
        return []

    # Drop noise-only notes on multi-term queries and weak tail notes on non-trivial note stores.
    kept_ids = list(rows)
    if len(tokens) >= 3 and non_noise:
        kept_ids = [
            nid
            for nid in kept_ids
            if nid in sem_note_ids
            or _matched_term_count(
                f"{rows[nid]['heading']} {rows[nid]['text']}".casefold(), non_noise, extra
            )
            >= 1
        ]
    note_total = int(conn.execute("SELECT count(*) FROM notes").fetchone()[0])
    if len(kept_ids) > 1 and note_total >= 4 and strength:
        kept_ids = [
            nid
            for nid in kept_ids
            if nid in sem_note_ids or strength.get(nid, 0.0) >= NOTE_TAIL_RATIO
        ]

    best = sorted(kept_ids, key=lambda i: (-scores[i], i))[:limit]
    return [
        NoteHit(id=i, path=rows[i]["path"], heading=rows[i]["heading"], text=rows[i]["text"])
        for i in best
    ]


def _generic_entities(ctx: Context, total: int, now: str) -> set[str]:
    """Entities that represent the scope or appear too widely to signal a conflict.

    `total` is the number of valid visible facts. The counting is done by SQLite, so no
    fact row is read into Python for it.
    """
    generic = {ctx.scope.business.casefold(), ctx.scope.agent.casefold()}
    if total < GENERIC_ENTITY_MIN_FACTS:
        return generic
    counts: dict[str, int] = {}
    for conn in ctx.connections():
        rows = conn.execute(
            "SELECT j.value AS entity, count(DISTINCT f.seq) AS n "
            "FROM facts f, json_each(f.entities) j "
            f"WHERE {is_valid_sql('f')} AND {ctx.visible_sql(conn, alias='f')} "
            "GROUP BY j.value",
            {"now": now, **ctx.params},
        )
        for row in rows:
            name = str(row["entity"]).casefold().strip()
            if name:
                counts[name] = counts.get(name, 0) + int(row["n"])
    cutoff = total * GENERIC_ENTITY_FREQ_RATIO
    generic.update(name for name, freq in counts.items() if freq >= cutoff)
    return generic


def _quoted(text: str) -> str:
    """`text` as one FTS5 string, so the index's own tokenizer splits it.

    A hand-made split can disagree with the tokenizer (it does at a combining accent, where
    `unicode61` keeps one token), and then the phrase would never match what is indexed.
    """
    return '"' + text.replace('"', '""') + '"'


def _shared_entity_rows(
    ctx: Context, wanted: set[str], topic_tokens: set[str], now: str
) -> list[tuple[sqlite3.Connection, sqlite3.Row]]:
    """Valid visible facts whose `entities` column mentions one of `wanted`.

    The word index answers this, so no fact row is read only to be skipped (the lookup still
    gets slower as more facts mention the entity). The match is on word tokens and is
    therefore a superset of "has exactly this entity"; the caller still compares the entity
    names themselves. When more than `SIMILAR_CANDIDATE_LIMIT` facts in a file mention the
    entity, the ones that also share topic words come first.
    """
    named = sorted(name for name in wanted if _WORD.search(name))[:MAX_QUERY_TOKENS]
    # A name with no letter or digit (an emoji, "++") has no token in the word index.
    symbolic = sorted(wanted - set(named))[:MAX_QUERY_TOKENS]
    found: list[tuple[sqlite3.Connection, sqlite3.Row]] = []
    if named:
        phrases = [_quoted(name) for name in named]
        ranking = [
            *phrases,
            *(f'"{token}"' for token in sorted(topic_tokens)[:MAX_QUERY_TOKENS]),
        ]
        # Every row that satisfies the left side also satisfies the right side through the
        # same phrase, so the right side never filters. It only lets topic words raise the
        # rank.
        match = f"(entities : ({' OR '.join(phrases)})) AND ({' OR '.join(ranking)})"
        for conn in ctx.connections():
            rows = conn.execute(
                "SELECT f.* FROM facts_fts JOIN facts f ON f.seq = facts_fts.rowid "
                f"WHERE facts_fts MATCH :q AND {is_valid_sql('f')} "
                f"AND {ctx.visible_sql(conn, alias='f')} "
                f"ORDER BY bm25(facts_fts, {_FACT_WEIGHTS}) LIMIT :lim",
                {"q": match, "now": now, "lim": SIMILAR_CANDIDATE_LIMIT, **ctx.params},
            ).fetchall()
            found.extend((conn, row) for row in rows)
    for name in symbolic:
        # Rare enough to afford the one pass SQLite makes over the entities column.
        for conn in ctx.connections():
            rows = conn.execute(
                "SELECT f.* FROM facts f WHERE EXISTS "
                "(SELECT 1 FROM json_each(f.entities) j WHERE trim(j.value) = :name) "
                f"AND {is_valid_sql('f')} AND {ctx.visible_sql(conn, alias='f')} LIMIT :lim",
                {"name": name, "now": now, "lim": SIMILAR_CANDIDATE_LIMIT, **ctx.params},
            ).fetchall()
            found.extend((conn, row) for row in rows)
    return found


def _linked_neighbors(ctx: Context, fact_id: str | None) -> set[str]:
    if not fact_id:
        return set()
    rows = ctx.business.execute(
        "SELECT CASE WHEN src_id = :id THEN dst_id ELSE src_id END AS other "
        "FROM edges WHERE src_id = :id OR dst_id = :id",
        {"id": fact_id},
    ).fetchall()
    return {str(r["other"]) for r in rows}


def similar_facts(
    ctx: Context,
    text: str,
    entities: Sequence[str],
    keywords: Sequence[str],
    *,
    exclude_id: str | None = None,
    linked_ids: Sequence[str] = (),
    query_vec: Sequence[float] | None = None,
    limit: int = 3,
) -> list[FactHit]:
    """Valid visible facts that share an entity (or strong semantic meaning) and overlap in topic.

    Filters out generic scope/high-frequency entities, boilerplate words, facts already linked
    as complementary via `edges`, and distinct external resource IDs. Never touches `use_count`.
    """
    wanted = {e.casefold().strip() for e in entities if e.strip()}
    cfg = ctx.scope.embeddings
    profile = embed.similarity_profile(cfg)
    sem_enabled = embed.is_enabled(cfg)
    # With no entity to share, only a model family that has an entity-free cut-off can link
    # two facts. Without one there is nothing to look for, and no model call to make.
    if limit <= 0 or (not wanted and not (sem_enabled and profile.conflict_strict is not None)):
        return []

    now = _now()
    total = _valid_fact_count(ctx)
    if total == 0:
        return []

    skip_linked = set(linked_ids) | _linked_neighbors(ctx, exclude_id)
    generic = _generic_entities(ctx, total, now)
    generic_words = frozenset(w.casefold() for g in generic for w in _WORD.findall(g))
    new_tokens = extract_content_tokens(text, " ".join(keywords), exclude=generic_words)
    new_res_ids = extract_resource_ids(text)

    vec: list[float] = []
    if sem_enabled:
        vec = (
            list(query_vec)
            if query_vec is not None
            else embed.embed_text(cfg, text, is_query=False) or []
        )

    # Candidates are looked up, not found by reading every fact row: the facts that mention
    # one of the entities and, for a model family with an entity-free cut-off, the facts
    # that close in meaning.
    candidates: dict[str, tuple[sqlite3.Connection, sqlite3.Row]] = {}
    if wanted:
        for conn, row in _shared_entity_rows(ctx, wanted, new_tokens, now):
            candidates.setdefault(str(row["id"]), (conn, row))
    if vec and profile.conflict_strict is not None:
        _, close_rows = embed.semantic_fact_scores(
            ctx,
            text,
            validity_sql=is_valid_sql("f"),
            time_param=now,
            min_similarity=profile.conflict_strict,
            query_vec=vec,
        )
        for fid, pair in close_rows.items():
            candidates.setdefault(fid, pair)
    # Every candidate is scored itself. The nearest-neighbour list of the whole file is no
    # substitute: it is cut at fifty, and look-alike facts about other subjects would push
    # the one fact that shares the entity out of it.
    sem_map = (
        embed.fact_similarities(ctx, ((conn, fid) for fid, (conn, _row) in candidates.items()), vec)
        if vec and candidates
        else {}
    )

    scored: list[tuple[float, int, int, str, sqlite3.Connection, sqlite3.Row]] = []
    for fid, (conn, row) in candidates.items():
        if fid == exclude_id or fid in skip_linked:
            continue
        row_entities = {e.casefold().strip() for e in _json_list(row["entities"]) if e.strip()}
        shared_ents = wanted & row_entities
        sem_sim = sem_map.get(fid, 0.0)

        # Close enough in meaning to count without a shared entity (never, for a model
        # family whose scores cannot tell two subjects on one template apart).
        close_alone = profile.conflict_strict is not None and sem_sim >= profile.conflict_strict
        if not shared_ents and not close_alone:
            continue

        specific_shared = shared_ents - generic
        shared_ent_words = frozenset(w.casefold() for e in shared_ents for w in _WORD.findall(e))
        cand_tokens = extract_content_tokens(
            str(row["text"]),
            " ".join(_json_list(row["keywords"])),
            exclude=generic_words,
        )
        overlap_all = new_tokens & cand_tokens
        overlap_non_ent = overlap_all - shared_ent_words
        min_len = max(1, min(len(new_tokens), len(cand_tokens)))
        ratio_all = len(overlap_all) / min_len

        # Guard: two facts that name different external resources (two Google Sheets, two
        # URLs) are about two things unless their wording is nearly the same. A similarity
        # score does not override this: such facts are usually written on one template.
        cand_res_ids = extract_resource_ids(str(row["text"]))
        if (
            new_res_ids
            and cand_res_ids
            and new_res_ids.isdisjoint(cand_res_ids)
            and ratio_all < 0.75
        ):
            continue

        new_non_ent = new_tokens - shared_ent_words
        cand_non_ent = cand_tokens - shared_ent_words

        accepted = False
        if close_alone or (specific_shared and sem_sim >= profile.conflict):
            accepted = True
        elif specific_shared:
            if (
                (len(overlap_non_ent) >= 2 and ratio_all >= MIN_CONFLICT_OVERLAP_RATIO)
                or (len(overlap_non_ent) >= 1 and min_len <= 4)
                or (len(new_non_ent) <= 1 and len(cand_non_ent) <= 1 and len(overlap_all) >= 1)
            ):
                accepted = True
        elif shared_ents and len(overlap_non_ent) >= 3 and ratio_all >= GENERIC_ONLY_OVERLAP_RATIO:
            # Only generic/scope entities shared (e.g. business name or frequent entity):
            # require strong lexical overlap on actual topic content.
            accepted = True

        if accepted:
            scored.append(
                (
                    max(ratio_all, sem_sim),
                    len(overlap_non_ent),
                    len(specific_shared),
                    fid,
                    conn,
                    row,
                )
            )

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return [_hit(ctx, _Candidate(row, conn)) for _, _, _, _, conn, row in scored[:limit]]
