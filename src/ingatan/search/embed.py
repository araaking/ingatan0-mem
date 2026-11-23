"""Optional semantic embeddings stored as L2-normalized float32 BLOBs in SQLite.

Providers:
- `none` (default): no embeddings, pure SQLite FTS5.
- `local`: offline ONNX Runtime + SentencePiece (`ingatan.search.local`). The model files
  are fetched once with `ingatan embeddings pull`, never during a tool call.
- `ollama`: local Ollama `/api/embed` via Python stdlib `urllib`.
- `openai`: OpenAI or any OpenAI-compatible `/embeddings` endpoint via stdlib `urllib`.
- `gemini`: Google Generative Language `embedContent` via stdlib `urllib`.

All vectors live inside `<business>.db` / `global.db` so single-file isolation and byte-level
`forget` (`PRAGMA secure_delete = ON`) remain intact. Runtime embedding calls fail gracefully
(returning `None`) so `recall` and `remember` never break when a model is unavailable.

Cosine scores are not comparable between model families, so every cut-off comes from a
`SimilarityProfile` chosen by model name (`similarity_profile`), never from a bare constant.
"""

from __future__ import annotations

import array
import json
import math
import os
import sqlite3
import struct
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.errors import IngatanError
from ingatan.scope import Context, EmbeddingConfig
from ingatan.search import local

__all__ = [
    "DEFAULT_MODELS",
    "E5_PROFILE",
    "GENERIC_PROFILE",
    "MAX_SEMANTIC_CANDIDATES",
    "REINDEX_BATCH_SIZE",
    "ReindexReport",
    "SimilarityProfile",
    "cosine_similarity",
    "effective_model",
    "embed_batch",
    "embed_text",
    "fact_similarities",
    "is_enabled",
    "last_embed_error",
    "local_runtime_available",
    "pack_vector",
    "probe_embedding",
    "reindex",
    "score_blobs",
    "semantic_fact_scores",
    "semantic_note_scores",
    "similarity_profile",
    "store_fact_embedding",
    "store_note_embedding",
    "unload_local_models",
    "unpack_vector",
]

DEFAULT_MODELS: dict[str, str] = {
    "local": local.DEFAULT_LOCAL_MODEL,
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}

DEFAULT_BASES: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}

DEFAULT_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# Upper bound on semantic candidates returned to RRF (keeps memory & hydration small).
MAX_SEMANTIC_CANDIDATES = 50
# Texts handed to the provider, and rows written, per `reindex` step.
REINDEX_BATCH_SIZE = 32
# Bounded LRU cache of query vectors, kept packed (about 1.5 KiB per 384-dim entry).
_QUERY_CACHE_MAX = 128

_K = TypeVar("_K")


@dataclass(frozen=True)
class SimilarityProfile:
    """Cosine cut-offs for one embedding model family.

    `recall_floor`: a query-to-fact score below this is noise, so a fact that matches a
    query only semantically must reach it to be returned.
    `conflict`: two facts that share a specific entity and score at least this are shown to
    the agent as a possible conflict.
    `conflict_strict`: two facts this close are shown as a possible conflict even without a
    shared entity. None means never: similarity alone is not evidence for this family.

    There is deliberately no "duplicate" cut-off. For every model measured, two statements
    that differ only in a value ("Rp 15.000" against "Rp 18.000") score higher than two true
    paraphrases, so a cosine score can never tell an update from a repeat.
    """

    recall_floor: float
    conflict: float
    conflict_strict: float | None


# For models with no measurement of their own. Suits models whose scores spread over most of
# the 0 to 1 range; override per project with `[embeddings] recall_floor = ...` and friends.
GENERIC_PROFILE = SimilarityProfile(recall_floor=0.35, conflict=0.82, conflict_strict=0.88)
# The E5 family is trained with a very low contrastive temperature, so its scores sit in a
# narrow band near the top: two unrelated Indonesian facts score 0.73 to 0.90.
# Measured with `scripts/calibrate_embeddings.py`; the tables are in docs/EMBEDDINGS.md.
# At 0.83 an off-topic question gets no semantic hit 92% of the time. At 0.90, 97% of
# changed values and 94% of paraphrases are flagged, against 3% of same-topic facts.
# There is no entity-free cut-off: two facts built on one template about different subjects
# ("stok gula ..." and "stok tepung ...") score 0.92 to 0.98, the same as a real update of
# one fact (0.94 to 0.98). Only a shared entity says that two facts are about one thing.
E5_PROFILE = SimilarityProfile(recall_floor=0.83, conflict=0.90, conflict_strict=None)

_state_lock = threading.Lock()
_query_cache: OrderedDict[tuple[int, str, str, str], bytes] = OrderedDict()
_last_error: str = ""


@dataclass(frozen=True)
class ReindexReport:
    model: str
    facts_indexed: int
    facts_skipped: int
    notes_indexed: int
    notes_skipped: int
    errors: int
    last_error: str = ""


def last_embed_error() -> str:
    """Most recent embedding failure message (empty if the last call succeeded)."""
    with _state_lock:
        return _last_error


def _set_last_error(message: str) -> None:
    global _last_error
    with _state_lock:
        _last_error = message


def is_enabled(cfg: EmbeddingConfig) -> bool:
    return cfg.provider != "none"


def local_runtime_available() -> bool:
    """True when the packages of the `vector` extra are importable."""
    return not local.missing_packages()


def _is_e5(model_name: str) -> bool:
    low = model_name.casefold()
    return "e5-" in low or low.endswith(("-e5", "/e5", ":e5")) or low == "e5"


def _apply_model_prefix(model_name: str, text: str, *, is_query: bool) -> str:
    """Apply model-family retrieval prefixes when expected (E5, EmbeddingGemma)."""
    low_model = model_name.casefold()
    if _is_e5(model_name):
        prefix = "query: " if is_query else "passage: "
        return text if text.casefold().startswith(prefix) else prefix + text
    if "embeddinggemma" in low_model:
        prefix = "task: search result | query: " if is_query else "title: none | text: "
        return text if text.casefold().startswith(prefix) else prefix + text
    return text


def effective_model(cfg: EmbeddingConfig) -> str:
    """The model name vectors are stored under. Local names are case-insensitive."""
    if not is_enabled(cfg):
        return ""
    name = cfg.model.strip() or DEFAULT_MODELS.get(cfg.provider, "")
    return name.casefold() if cfg.provider == "local" else name


def similarity_profile(cfg: EmbeddingConfig) -> SimilarityProfile:
    """The cut-offs for the configured model, with any `[embeddings]` overrides applied."""
    base = E5_PROFILE if _is_e5(effective_model(cfg)) else GENERIC_PROFILE
    floor, conflict, strict = base.recall_floor, base.conflict, base.conflict_strict
    if cfg.recall_floor is not None:
        floor = cfg.recall_floor
    if cfg.conflict_similarity is not None:
        conflict = cfg.conflict_similarity
    if cfg.conflict_strict_similarity is not None:
        strict = cfg.conflict_strict_similarity
    return SimilarityProfile(recall_floor=floor, conflict=conflict, conflict_strict=strict)


def _normalize(vec: Sequence[float]) -> list[float]:
    values = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return []
    return [x / norm for x in values]


def pack_vector(vec: Sequence[float]) -> bytes:
    """L2-normalize and pack a vector as little-endian float32 bytes."""
    normed = _normalize(vec)
    if not normed:
        return b""
    return struct.pack(f"<{len(normed)}f", *normed)


def unpack_vector(blob: bytes, dims: int) -> list[float]:
    if not blob or dims <= 0 or len(blob) != dims * 4:
        return []
    return list(struct.unpack(f"<{dims}f", blob))


def cosine_similarity(query_vec: Sequence[float], blob: bytes, dims: int) -> float:
    """Dot product between an L2-normalized `query_vec` and a packed normalized `blob`."""
    if dims <= 0 or len(query_vec) != dims or len(blob) != dims * 4:
        return 0.0
    np = local.load_numpy()
    if np is not None:
        q = np.asarray(query_vec, dtype=np.float32)
        v = np.frombuffer(blob, dtype=np.float32)
        return float(np.dot(q, v))
    arr = array.array("f")
    arr.frombytes(blob)
    sumprod = getattr(math, "sumprod", None)  # Python 3.12 and later
    if sumprod is not None:
        return float(sumprod(query_vec, arr))
    return float(sum(a * b for a, b in zip(query_vec, arr, strict=True)))


def score_blobs(
    query_vec: Sequence[float],
    items: Sequence[tuple[_K, bytes]],
    dims: int,
    *,
    min_similarity: float,
    top_k: int = MAX_SEMANTIC_CANDIDATES,
) -> list[tuple[_K, float]]:
    """Compute top-k cosine similarities over packed normalized float32 BLOBs.

    Uses a single vectorized matrix-vector multiply when NumPy is installed, avoiding
    per-row Python allocation overhead and skipping hydration of non-matching rows.
    """
    if dims <= 0 or len(query_vec) != dims or not items or top_k <= 0:
        return []
    expected_bytes = dims * 4
    valid_keys: list[_K] = []
    valid_blobs: list[bytes] = []
    for key, blob in items:
        raw = bytes(blob)
        if len(raw) == expected_bytes:
            valid_keys.append(key)
            valid_blobs.append(raw)
    if not valid_keys:
        return []

    scored: list[tuple[_K, float]] = []
    np = local.load_numpy()
    if np is not None:
        q = np.asarray(query_vec, dtype=np.float32)
        mat = np.frombuffer(b"".join(valid_blobs), dtype=np.float32).reshape(-1, dims)
        sims = mat @ q
        indices = np.flatnonzero(sims >= min_similarity)
        if indices.size == 0:
            return []
        order = np.argsort(sims[indices])[::-1][:top_k]
        for idx in indices[order].tolist():
            scored.append((valid_keys[idx], float(sims[idx])))
        return scored

    for key, raw in zip(valid_keys, valid_blobs, strict=True):
        sim = cosine_similarity(query_vec, raw, dims)
        if sim >= min_similarity:
            scored.append((key, sim))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def unload_local_models() -> int:
    """Drop the local model from memory and clear the query cache."""
    with _state_lock:
        _query_cache.clear()
    return local.unload()


def _http_post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _embed_local_batch(
    cfg: EmbeddingConfig, model_name: str, texts: Sequence[str], *, is_query: bool
) -> list[list[float] | None]:
    prepared = [_apply_model_prefix(model_name, t, is_query=is_query) for t in texts]
    vectors = local.embed(
        prepared,
        model_name=model_name,
        model_dir=cfg.model_dir,
        threads=cfg.threads,
        idle_unload_seconds=cfg.idle_unload_seconds,
    )
    return [vec if vec else None for vec in vectors]


def _embed_ollama(
    cfg: EmbeddingConfig, model_name: str, text: str, *, is_query: bool
) -> list[float] | None:
    base = (cfg.api_base or DEFAULT_BASES["ollama"]).rstrip("/")
    prepared = _apply_model_prefix(model_name, text, is_query=is_query)
    data = _http_post_json(
        f"{base}/api/embed", {"model": model_name, "input": prepared}, {}, cfg.timeout
    )
    embs = data.get("embeddings")
    if isinstance(embs, list) and embs and isinstance(embs[0], list):
        return _normalize(embs[0])
    single = data.get("embedding")
    if isinstance(single, list) and single:
        return _normalize(single)
    return None


def _embed_openai(cfg: EmbeddingConfig, model_name: str, text: str) -> list[float] | None:
    base = (cfg.api_base or DEFAULT_BASES["openai"]).rstrip("/")
    key_env = cfg.api_key_env or DEFAULT_KEY_ENVS["openai"]
    api_key = (os.environ.get(key_env) or "").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _http_post_json(
        f"{base}/embeddings", {"model": model_name, "input": text}, headers, cfg.timeout
    )
    items = data.get("data")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        emb = items[0].get("embedding")
        if isinstance(emb, list) and emb:
            return _normalize(emb)
    return None


def _embed_gemini(
    cfg: EmbeddingConfig, model_name: str, text: str, *, is_query: bool
) -> list[float] | None:
    base = (cfg.api_base or DEFAULT_BASES["gemini"]).rstrip("/")
    key_env = cfg.api_key_env or DEFAULT_KEY_ENVS["gemini"]
    api_key = (os.environ.get(key_env) or "").strip()
    if not api_key:
        return None
    full_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
    payload: dict[str, Any] = {
        "model": full_model,
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
    }
    # The key travels in a header, not in the URL: URLs end up in logs and error messages.
    data = _http_post_json(
        f"{base}/{full_model}:embedContent", payload, {"x-goog-api-key": api_key}, cfg.timeout
    )
    emb = data.get("embedding")
    if isinstance(emb, dict):
        values = emb.get("values")
        if isinstance(values, list) and values:
            return _normalize(values)
    return None


def _embed_raw(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float] | None:
    if not is_enabled(cfg) or not text.strip():
        return None
    model_name = effective_model(cfg)
    if not model_name:
        return None
    if cfg.provider == "local":
        return _embed_local_batch(cfg, model_name, [text], is_query=is_query)[0]
    if cfg.provider == "ollama":
        return _embed_ollama(cfg, model_name, text, is_query=is_query)
    if cfg.provider == "openai":
        return _embed_openai(cfg, model_name, text)
    if cfg.provider == "gemini":
        return _embed_gemini(cfg, model_name, text, is_query=is_query)
    return None


_DEFAULT_EMBED_RAW = _embed_raw
# What a provider can raise at run time. `IngatanError` covers an unknown local model name.
_EMBED_ERRORS = (
    ImportError,
    OSError,
    ValueError,
    urllib.error.URLError,
    RuntimeError,
    IngatanError,
)


def embed_text(cfg: EmbeddingConfig, text: str, *, is_query: bool = False) -> list[float] | None:
    """Compute an L2-normalized embedding vector, returning `None` on any failure."""
    if not is_enabled(cfg) or not text.strip():
        return None
    model_name = effective_model(cfg)
    cache_key = (id(_embed_raw), cfg.provider, model_name, text)
    if is_query:
        with _state_lock:
            cached = _query_cache.get(cache_key)
            if cached is not None:
                _query_cache.move_to_end(cache_key)
                return unpack_vector(cached, len(cached) // 4)
    try:
        vec = _embed_raw(cfg, text, is_query=is_query)
    except _EMBED_ERRORS as exc:
        _set_last_error(f"{type(exc).__name__}: {exc}")
        return None
    if not vec:
        _set_last_error(f"Provider '{cfg.provider}' returned an empty embedding vector.")
        return None
    _set_last_error("")
    if is_query:
        packed = pack_vector(vec)
        with _state_lock:
            _query_cache[cache_key] = packed
            _query_cache.move_to_end(cache_key)
            while len(_query_cache) > _QUERY_CACHE_MAX:
                _query_cache.popitem(last=False)
    return vec


def probe_embedding(cfg: EmbeddingConfig) -> int:
    """Embed one test sentence and return its vector dimensions, or raise `IngatanError`."""
    if not is_enabled(cfg):
        return 0
    model_name = effective_model(cfg)
    try:
        vec = _embed_raw(cfg, "ingatan embedding health check", is_query=True)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        _set_last_error(msg)
        raise IngatanError(
            f"Embedding probe failed for {cfg.provider} ({model_name}): {msg}"
        ) from exc
    if not vec:
        msg = f"Provider '{cfg.provider}' ({model_name}) returned an empty embedding vector."
        _set_last_error(msg)
        raise IngatanError(msg)
    _set_last_error("")
    return len(vec)


def embed_batch(
    cfg: EmbeddingConfig, texts: Sequence[str], *, is_query: bool = False
) -> list[list[float] | None]:
    """Embed several texts, in order. One model call for the local provider, else one each.

    A text that is empty or fails gets `None`. Never raises.
    """
    if not texts:
        return []
    if not is_enabled(cfg):
        return [None for _ in texts]
    model_name = effective_model(cfg)
    if cfg.provider == "local" and model_name and _embed_raw is _DEFAULT_EMBED_RAW:
        wanted = [(i, t) for i, t in enumerate(texts) if t.strip()]
        out: list[list[float] | None] = [None for _ in texts]
        if not wanted:
            return out
        try:
            vecs = _embed_local_batch(cfg, model_name, [t for _, t in wanted], is_query=is_query)
        except _EMBED_ERRORS as exc:
            _set_last_error(f"{type(exc).__name__}: {exc}")
            return out
        _set_last_error("")
        for (index, _), vec in zip(wanted, vecs, strict=True):
            out[index] = vec
        return out
    return [embed_text(cfg, t, is_query=is_query) for t in texts]


def store_fact_embedding(
    conn: sqlite3.Connection,
    fact_id: str,
    text: str,
    cfg: EmbeddingConfig,
    *,
    vector: Sequence[float] | None = None,
) -> bool:
    """Store (or update) a fact's normalized vector in `fact_embeddings`. Returns True if stored.

    Pass `vector` when calling inside a write transaction: without it the text is embedded
    here, and a model call must never run while the database is locked for writing.

    Nothing is stored, and False is returned, when the fact row is gone: a vector that
    arrives after `forget` must not bring a trace of the fact back.
    """
    if not is_enabled(cfg):
        return False
    vec = list(vector) if vector is not None else embed_text(cfg, text, is_query=False)
    if not vec:
        return False
    blob = pack_vector(vec)
    if not blob:
        return False
    model = effective_model(cfg)
    now = timeutil.iso(timeutil.now())
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO fact_embeddings (fact_id, model, dims, vec, updated_at) "
            "SELECT ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM facts WHERE id = ?) "
            "ON CONFLICT(fact_id) DO UPDATE SET "
            "model = excluded.model, dims = excluded.dims, vec = excluded.vec, "
            "updated_at = excluded.updated_at",
            (fact_id, model, len(vec), blob, now, fact_id),
        )
    return cursor.rowcount > 0


def store_note_embedding(
    conn: sqlite3.Connection,
    note_id: int,
    heading: str,
    text: str,
    cfg: EmbeddingConfig,
    *,
    vector: Sequence[float] | None = None,
) -> bool:
    """Store (or update) a document note's normalized vector in `note_embeddings`.

    Pass `vector` when calling inside a write transaction (see `store_fact_embedding`).
    """
    if not is_enabled(cfg):
        return False
    combined = f"{heading}\n{text}".strip()
    vec = list(vector) if vector is not None else embed_text(cfg, combined, is_query=False)
    if not vec:
        return False
    blob = pack_vector(vec)
    if not blob:
        return False
    model = effective_model(cfg)
    now = timeutil.iso(timeutil.now())
    with transaction(conn):
        conn.execute(
            "INSERT INTO note_embeddings (note_id, model, dims, vec, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(note_id) DO UPDATE SET "
            "model = excluded.model, dims = excluded.dims, vec = excluded.vec, "
            "updated_at = excluded.updated_at",
            (note_id, model, len(vec), blob, now),
        )
    return True


def semantic_fact_scores(
    ctx: Context,
    query: str,
    *,
    validity_sql: str,
    time_param: str,
    min_similarity: float | None = None,
    query_vec: Sequence[float] | None = None,
) -> tuple[dict[str, float], dict[str, tuple[sqlite3.Connection, sqlite3.Row]]]:
    """Cosine similarity scores for visible facts matching the active embedding model.

    `min_similarity` defaults to the model's calibrated `recall_floor`.
    """
    cfg = ctx.scope.embeddings
    if not is_enabled(cfg):
        return {}, {}
    vec = list(query_vec) if query_vec is not None else embed_text(cfg, query, is_query=True)
    if not vec:
        return {}, {}
    floor = similarity_profile(cfg).recall_floor if min_similarity is None else min_similarity
    model = effective_model(cfg)
    dims = len(vec)
    scores: dict[str, float] = {}
    rows_out: dict[str, tuple[sqlite3.Connection, sqlite3.Row]] = {}
    for conn in ctx.connections():
        # Phase 1: lightweight scan of (fact_id, vec) without reading full fact text/JSON.
        sql = (
            "SELECT e.fact_id, e.vec FROM fact_embeddings e "
            "JOIN facts f ON f.id = e.fact_id "
            f"WHERE e.model = :model AND e.dims = :dims AND {validity_sql} "
            f"AND {ctx.visible_sql(conn, alias='f')}"
        )
        params: dict[str, Any] = {
            "model": model,
            "dims": dims,
            "now": time_param,
            **ctx.params,
        }
        raw_items = [(str(r["fact_id"]), bytes(r["vec"])) for r in conn.execute(sql, params)]
        matched = score_blobs(
            vec,
            raw_items,
            dims,
            min_similarity=floor,
            top_k=MAX_SEMANTIC_CANDIDATES,
        )
        if not matched:
            continue
        # Phase 2: hydrate full fact rows only for matched candidates.
        matched_ids = [fid for fid, _ in matched]
        marks = ",".join("?" for _ in matched_ids)
        hydrated = {
            str(r["id"]): r
            for r in conn.execute(f"SELECT * FROM facts WHERE id IN ({marks})", matched_ids)
        }
        for fid, sim in matched:
            row = hydrated.get(fid)
            if row is not None:
                scores[fid] = sim
                rows_out[fid] = (conn, row)
    return scores, rows_out


def fact_similarities(
    ctx: Context,
    facts: Iterable[tuple[sqlite3.Connection, str]],
    query_vec: Sequence[float],
) -> dict[str, float]:
    """Cosine between `query_vec` and the stored vector of each named fact.

    `facts` are `(connection, fact id)` pairs. A fact without a vector of the active model
    is left out. Use this, not the nearest-neighbour list of a whole file, to score known
    candidates: that list is cut at `MAX_SEMANTIC_CANDIDATES`, and look-alike facts about
    other subjects can push the one that matters out of it.
    """
    cfg = ctx.scope.embeddings
    if not is_enabled(cfg) or not query_vec:
        return {}
    model = effective_model(cfg)
    dims = len(query_vec)
    grouped: dict[int, tuple[sqlite3.Connection, list[str]]] = {}
    for conn, fact_id in facts:
        grouped.setdefault(id(conn), (conn, []))[1].append(fact_id)
    scores: dict[str, float] = {}
    for conn, ids in grouped.values():
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" for _ in chunk)
            items = [
                (str(r["fact_id"]), bytes(r["vec"]))
                for r in conn.execute(
                    "SELECT fact_id, vec FROM fact_embeddings "
                    f"WHERE model = ? AND dims = ? AND fact_id IN ({marks})",
                    (model, dims, *chunk),
                )
            ]
            scores.update(
                score_blobs(query_vec, items, dims, min_similarity=-2.0, top_k=len(items))
            )
    return scores


def semantic_note_scores(
    ctx: Context,
    query: str,
    *,
    min_similarity: float | None = None,
    query_vec: Sequence[float] | None = None,
) -> tuple[dict[int, float], dict[int, sqlite3.Row]]:
    """Cosine similarity scores for ingested document notes matching the active model."""
    cfg = ctx.scope.embeddings
    if not is_enabled(cfg):
        return {}, {}
    vec = list(query_vec) if query_vec is not None else embed_text(cfg, query, is_query=True)
    if not vec:
        return {}, {}
    floor = similarity_profile(cfg).recall_floor if min_similarity is None else min_similarity
    model = effective_model(cfg)
    dims = len(vec)
    sql = "SELECT note_id, vec FROM note_embeddings WHERE model = :model AND dims = :dims"
    raw_items = [
        (int(r["note_id"]), bytes(r["vec"]))
        for r in ctx.business.execute(sql, {"model": model, "dims": dims})
    ]
    matched = score_blobs(
        vec,
        raw_items,
        dims,
        min_similarity=floor,
        top_k=MAX_SEMANTIC_CANDIDATES,
    )
    if not matched:
        return {}, {}
    matched_ids = [nid for nid, _ in matched]
    marks = ",".join("?" for _ in matched_ids)
    hydrated = {
        int(r["id"]): r
        for r in ctx.business.execute(f"SELECT * FROM notes WHERE id IN ({marks})", matched_ids)
    }
    scores: dict[int, float] = {}
    rows_out: dict[int, sqlite3.Row] = {}
    for nid, sim in matched:
        row = hydrated.get(nid)
        if row is not None:
            scores[nid] = sim
            rows_out[nid] = row
    return scores, rows_out


def reindex(ctx: Context, *, force: bool = False) -> ReindexReport:
    """Compute embeddings in batches for all valid visible facts and notes, pruning orphans."""
    cfg = ctx.scope.embeddings
    model = effective_model(cfg)
    if not is_enabled(cfg) or not model:
        return ReindexReport("", 0, 0, 0, 0, 0)

    _set_last_error("")
    now = timeutil.iso(timeutil.now())
    facts_indexed = facts_skipped = notes_indexed = notes_skipped = errors = 0
    error_reason = ""
    valid_clause = (
        "(valid_from <= :now AND (valid_to IS NULL OR valid_to > :now) AND superseded_by IS NULL)"
    )

    for conn in ctx.connections():
        with transaction(conn):
            conn.execute("DELETE FROM fact_embeddings WHERE fact_id NOT IN (SELECT id FROM facts)")
        existing_models = {
            str(r["fact_id"]): str(r["model"])
            for r in conn.execute("SELECT fact_id, model FROM fact_embeddings")
        }
        rows = conn.execute(
            f"SELECT id, text FROM facts WHERE {valid_clause} AND {ctx.visible_sql(conn)}",
            {"now": now, **ctx.params},
        ).fetchall()
        pending: list[tuple[str, str]] = []
        for row in rows:
            fid = str(row["id"])
            if not force and existing_models.get(fid) == model:
                facts_skipped += 1
            else:
                pending.append((fid, str(row["text"])))
        for start in range(0, len(pending), REINDEX_BATCH_SIZE):
            chunk = pending[start : start + REINDEX_BATCH_SIZE]
            # The model runs before the write lock is taken, never inside it.
            vecs = embed_batch(cfg, [text for _, text in chunk], is_query=False)
            ts = timeutil.iso(timeutil.now())
            with transaction(conn):
                for (fid, _), vec in zip(chunk, vecs, strict=True):
                    blob = pack_vector(vec) if vec else b""
                    if not blob or not vec:
                        errors += 1
                        if not error_reason:
                            error_reason = last_embed_error()
                        continue
                    conn.execute(
                        "INSERT INTO fact_embeddings (fact_id, model, dims, vec, updated_at) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(fact_id) DO UPDATE SET "
                        "model = excluded.model, dims = excluded.dims, vec = excluded.vec, "
                        "updated_at = excluded.updated_at",
                        (fid, model, len(vec), blob, ts),
                    )
                    facts_indexed += 1

    with transaction(ctx.business):
        ctx.business.execute(
            "DELETE FROM note_embeddings WHERE note_id NOT IN (SELECT id FROM notes)"
        )
    existing_note_models = {
        int(r["note_id"]): str(r["model"])
        for r in ctx.business.execute("SELECT note_id, model FROM note_embeddings")
    }
    pending_notes: list[tuple[int, str]] = []
    for row in ctx.business.execute("SELECT id, heading, text FROM notes").fetchall():
        nid = int(row["id"])
        if not force and existing_note_models.get(nid) == model:
            notes_skipped += 1
        else:
            pending_notes.append((nid, f"{row['heading']}\n{row['text']}".strip()))
    for start in range(0, len(pending_notes), REINDEX_BATCH_SIZE):
        chunk = pending_notes[start : start + REINDEX_BATCH_SIZE]
        vecs = embed_batch(cfg, [text for _, text in chunk], is_query=False)
        ts = timeutil.iso(timeutil.now())
        with transaction(ctx.business):
            for (nid, _), vec in zip(chunk, vecs, strict=True):
                blob = pack_vector(vec) if vec else b""
                if not blob or not vec:
                    errors += 1
                    if not error_reason:
                        error_reason = last_embed_error()
                    continue
                ctx.business.execute(
                    "INSERT INTO note_embeddings (note_id, model, dims, vec, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(note_id) DO UPDATE SET "
                    "model = excluded.model, dims = excluded.dims, vec = excluded.vec, "
                    "updated_at = excluded.updated_at",
                    (nid, model, len(vec), blob, ts),
                )
                notes_indexed += 1

    return ReindexReport(
        model=model,
        facts_indexed=facts_indexed,
        facts_skipped=facts_skipped,
        notes_indexed=notes_indexed,
        notes_skipped=notes_skipped,
        errors=errors,
        last_error=(error_reason or last_embed_error()) if errors else "",
    )
