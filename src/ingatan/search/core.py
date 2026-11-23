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
from ingatan.search.stopwords import (
    CONFLICT_NOISE_WORDS,
    STOPWORDS,
    extract_content_tokens,
    extract_identifier_phrases,
    extract_resource_ids,
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
# Recency breaks ties and gently favours newer facts without drowning out lexical matches.
RECENCY_WEIGHT = 0.35
# Identifier phrase bonus (e.g. `sync_report` -> `"sync report"` matching `sync_report.py`).
PHRASE_WEIGHT = 1.0
# Default maximum number of document excerpts returned alongside recalled facts.
DEFAULT_NOTES_LIMIT = 2

# Conflict detection tuning (prevents false positives on generic entities and noisy overlaps).
GENERIC_ENTITY_MIN_FACTS = 10
GENERIC_ENTITY_FREQ_RATIO = 0.35
MIN_CONFLICT_OVERLAP_RATIO = 0.35
GENERIC_ONLY_OVERLAP_RATIO = 0.50

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


def recall(
    ctx: Context,
    query: str,
    *,
    limit: int = 5,
    notes_limit: int = DEFAULT_NOTES_LIMIT,
    include_closed: bool = False,
    touch: bool = True,
) -> RecallResult:
    """Facts (both files, visibility applied) and document notes matching `query`."""
    raw_tokens = _tokens(query)
    if not raw_tokens or limit <= 0:
        return RecallResult()
    tokens = [t for t in raw_tokens if t.casefold() not in STOPWORDS] or raw_tokens
    non_noise = [t for t in tokens if t.casefold() not in CONFLICT_NOISE_WORDS]
    trigram_tokens = [t for t in tokens if len(t) >= MIN_TRIGRAM_LENGTH]
    phrases = extract_identifier_phrases(query)

    pool: dict[str, _Candidate] = {}
    word_bm = _bm25_scores(ctx, "facts_fts", _match(tokens), include_closed, pool)
    tri_bm = (
        _bm25_scores(ctx, "facts_tri", _match(trigram_tokens), include_closed, pool)
        if trigram_tokens
        else {}
    )
    phrase_bm = (
        _bm25_scores(ctx, "facts_fts", _match(phrases), include_closed, pool) if phrases else {}
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
        (phrase_bm, PHRASE_WEIGHT),
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

    # Filter out noise-only hits on multi-term queries (prevents long queries from returning
    # unrelated facts that only matched a boilerplate word like "dokumen" or "data").
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
            ):
                kept[fact_id] = cand
        pool = kept
        # When the store is non-trivial and strong hits exist, drop weak single-term tail hits.
        if len(pool) > 1 and _valid_fact_count(ctx) >= 8:
            lex_mag = {
                fid: max(abs(word_bm.get(fid, 0.0)), abs(tri_bm.get(fid, 0.0))) for fid in pool
            }
            top_mag = max(lex_mag.values(), default=0.0)
            if top_mag >= 1.0:
                pool = {
                    fid: cand
                    for fid, cand in pool.items()
                    if (
                        term_hits.get(fid, 0) >= 2
                        or fid in entity_counts
                        or fid in phrase_bm
                        or lex_mag[fid] >= top_mag * 0.25
                    )
                }

    # Recency: newest valid_from first. ISO strings sort as text, so map them to ordinals.
    stamps = sorted({c.row["valid_from"] for c in pool.values()})
    ordinal = {stamp: float(i) for i, stamp in enumerate(stamps)}
    recency = {i: ordinal[c.row["valid_from"]] for i, c in pool.items()}
    for fact_id, rank in _competition_ranks(recency).items():
        scores[fact_id] += RECENCY_WEIGHT / (RRF_K + rank)

    # Ties on score fall back to the newest fact, then the id.
    ordered = sorted(pool, key=lambda i: (-round(scores[i], 12), -recency[i], i))[:limit]
    for fact_id in ordered:
        pool[fact_id].score = scores[fact_id]
    hits = [_hit(ctx, pool[i]) for i in ordered]
    if touch and hits:
        _touch(hits, pool)
    effective_notes = min(limit, notes_limit) if notes_limit > 0 else 0
    notes = (
        _notes(ctx, tokens, non_noise, trigram_tokens, extra, effective_notes)
        if effective_notes > 0
        else []
    )
    return RecallResult(facts=hits, notes=notes)


def _notes(
    ctx: Context,
    tokens: list[str],
    non_noise: list[str],
    trigram_tokens: list[str],
    extra: list[str],
    limit: int,
) -> list[NoteHit]:
    conn = ctx.business
    rows: dict[int, sqlite3.Row] = {}
    scores: dict[int, float] = {}
    best_bm: dict[int, float] = {}
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
            nid = row["id"]
            rows.setdefault(nid, row)
            scores[nid] = scores.get(nid, 0.0) + weight / (RRF_K + rank)
            if weight >= 1.0:
                best_bm[nid] = max(best_bm.get(nid, 0.0), abs(float(row["bm"])))

    if not rows:
        return []

    # Drop noise-only notes on multi-term queries and weak tail notes on non-trivial note stores.
    kept_ids = list(rows)
    if len(tokens) >= 3 and non_noise:
        kept_ids = [
            nid
            for nid in kept_ids
            if _matched_term_count(
                f"{rows[nid]['heading']} {rows[nid]['text']}".casefold(), non_noise, extra
            )
            >= 1
        ]
    note_total = int(conn.execute("SELECT count(*) FROM notes").fetchone()[0])
    if len(kept_ids) > 1 and note_total >= 4:
        top_bm = max((best_bm.get(nid, 0.0) for nid in kept_ids), default=0.0)
        if top_bm >= 1.0:
            kept_ids = [nid for nid in kept_ids if best_bm.get(nid, 0.0) >= top_bm * 0.35]

    best = sorted(kept_ids, key=lambda i: (-scores[i], i))[:limit]
    return [
        NoteHit(id=i, path=rows[i]["path"], heading=rows[i]["heading"], text=rows[i]["text"])
        for i in best
    ]


def _generic_entities(ctx: Context, valid_rows: list[tuple[sqlite3.Connection, sqlite3.Row]]) -> set[str]:
    """Entities that represent the whole scope or appear across too many facts to signal conflict."""
    generic = {ctx.scope.business.casefold(), ctx.scope.agent.casefold()}
    total = len(valid_rows)
    if total < GENERIC_ENTITY_MIN_FACTS:
        return generic
    counts: dict[str, int] = {}
    for _, row in valid_rows:
        for ent in {e.casefold().strip() for e in _json_list(row["entities"]) if e.strip()}:
            counts[ent] = counts.get(ent, 0) + 1
    cutoff = total * GENERIC_ENTITY_FREQ_RATIO
    for ent, freq in counts.items():
        if freq >= cutoff:
            generic.add(ent)
    return generic


def similar_facts(
    ctx: Context,
    text: str,
    entities: Sequence[str],
    keywords: Sequence[str],
    *,
    exclude_id: str | None = None,
    limit: int = 3,
) -> list[FactHit]:
    """Valid visible facts that share an entity AND genuinely overlap in topic content.

    Filters out generic scope/high-frequency entities, boilerplate words, and facts pointing
    to distinct external resource IDs (e.g. different Google Sheets). Never touches `use_count`.
    """
    wanted = {e.casefold().strip() for e in entities if e.strip()}
    if not wanted or limit <= 0:
        return []

    now = _now()
    valid_rows: list[tuple[sqlite3.Connection, sqlite3.Row]] = []
    for conn in ctx.connections():
        rows = conn.execute(
            f"SELECT * FROM facts WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)}",
            {"now": now, **ctx.params},
        ).fetchall()
        valid_rows.extend((conn, r) for r in rows)

    if not valid_rows:
        return []

    generic = _generic_entities(ctx, valid_rows)
    generic_words = frozenset(w.casefold() for g in generic for w in _WORD.findall(g))
    new_tokens = extract_content_tokens(text, " ".join(keywords), exclude=generic_words)
    new_res_ids = extract_resource_ids(text)

    scored: list[tuple[float, int, int, str, sqlite3.Connection, sqlite3.Row]] = []
    for conn, row in valid_rows:
        fid = str(row["id"])
        if fid == exclude_id:
            continue
        row_entities = {e.casefold().strip() for e in _json_list(row["entities"]) if e.strip()}
        shared_ents = wanted & row_entities
        if not shared_ents:
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

        # Guard: distinct external resource IDs (e.g. two different Google Sheets) rarely conflict.
        cand_res_ids = extract_resource_ids(str(row["text"]))
        if new_res_ids and cand_res_ids and new_res_ids.isdisjoint(cand_res_ids) and ratio_all < 0.75:
            continue

        new_non_ent = new_tokens - shared_ent_words
        cand_non_ent = cand_tokens - shared_ent_words

        accepted = False
        if specific_shared:
            if (
                (len(overlap_non_ent) >= 2 and ratio_all >= MIN_CONFLICT_OVERLAP_RATIO)
                or (len(overlap_non_ent) >= 1 and min_len <= 4)
                or (len(new_non_ent) <= 1 and len(cand_non_ent) <= 1 and len(overlap_all) >= 1)
            ):
                accepted = True
        else:
            # Only generic/scope entities shared (e.g. business name or frequent user name):
            # require strong lexical overlap on actual topic content.
            if len(overlap_non_ent) >= 3 and ratio_all >= GENERIC_ONLY_OVERLAP_RATIO:
                accepted = True

        if accepted:
            scored.append(
                (ratio_all, len(overlap_non_ent), len(specific_shared), fid, conn, row)
            )

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return [_hit(ctx, _Candidate(row, conn)) for _, _, _, _, conn, row in scored[:limit]]
