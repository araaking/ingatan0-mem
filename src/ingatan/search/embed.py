"""Pluggable semantic embeddings stored as L2-normalized float32 BLOBs in SQLite.

Supports:
- `local` (default): offline ONNX multilingual embeddings via `fastembed`
  (`intfloat/multilingual-e5-small`), with bounded thread count and idle auto-unload.
- `ollama`: local Ollama `/api/embed` via Python stdlib `urllib`.
- `openai`: OpenAI or OpenAI-compatible `/embeddings` via stdlib `urllib`.
- `gemini`: Google Generative Language `embedContent` via stdlib `urllib`.
- `none`: disabled, pure SQLite FTS5 minimal-RAM mode.

All vectors live inside `<business>.db` / `global.db` so single-file isolation and byte-level
`forget` (`PRAGMA secure_delete = ON`) remain intact. Runtime embedding calls fail gracefully
(returning `None`) so `recall` and `remember` never break when offline.
"""

from __future__ import annotations

import array
import contextlib
import ctypes
import gc
import importlib.util
import json
import math
import os
import sqlite3
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.scope import Context, EmbeddingConfig

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional extra
    _np = None

__all__ = [
    "DEFAULT_MODELS",
    "MAX_SEMANTIC_CANDIDATES",
    "MIN_SEMANTIC_SIMILARITY",
    "REINDEX_BATCH_SIZE",
    "ReindexReport",
    "cosine_similarity",
    "effective_model",
    "embed_batch",
    "embed_text",
    "is_enabled",
    "local_runtime_available",
    "pack_vector",
    "reindex",
    "score_blobs",
    "semantic_fact_scores",
    "semantic_note_scores",
    "store_fact_embedding",
    "store_note_embedding",
    "unload_local_models",
    "unpack_vector",
]

DEFAULT_MODELS: dict[str, str] = {
    "local": "intfloat/multilingual-e5-small",
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

# Cosine threshold below which a vector match is considered unrelated noise.
MIN_SEMANTIC_SIMILARITY = 0.35
# Upper bound on semantic candidates returned to RRF (keeps memory & hydration small).
MAX_SEMANTIC_CANDIDATES = 50
# Batch size for `reindex` ONNX inference + SQLite writes.
REINDEX_BATCH_SIZE = 32
# Bounded LRU cache for repeated query embeddings (< 200 KiB for 384-d float vectors).
_QUERY_CACHE_MAX = 128

_K = TypeVar("_K")

_local_lock = threading.Lock()
_local_models: dict[tuple[str, int], Any] = {}
_local_last_used: dict[tuple[str, int], float] = {}
_local_timers: dict[tuple[str, int], threading.Timer] = {}
_query_cache: OrderedDict[tuple[int, str, str, str], list[float]] = OrderedDict()


@dataclass(frozen=True)
class ReindexReport:
    model: str
    facts_indexed: int
    facts_skipped: int
    notes_indexed: int
    notes_skipped: int
    errors: int


def is_enabled(cfg: EmbeddingConfig) -> bool:
    return cfg.provider != "none"


def local_runtime_available() -> bool:
    """Return True if the ONNX `fastembed` runtime is importable."""
    return importlib.util.find_spec("fastembed") is not None


def _apply_model_prefix(model_name: str, text: str, *, is_query: bool) -> str:
    """Apply model-family retrieval prefixes when expected (E5, EmbeddingGemma)."""
    low_model = model_name.casefold()
    if "e5-" in low_model:
        prefix = "query: " if is_query else "passage: "
        return text if text.casefold().startswith(prefix) else prefix + text
    if "embeddinggemma" in low_model:
        prefix = "task: search result | query: " if is_query else "title: none | text: "
        return text if text.casefold().startswith(prefix) else prefix + text
    return text


def effective_model(cfg: EmbeddingConfig) -> str:
    if not is_enabled(cfg):
        return ""
    return cfg.model.strip() or DEFAULT_MODELS.get(cfg.provider, "")


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
    if _np is not None:
        q = _np.asarray(query_vec, dtype=_np.float32)
        v = _np.frombuffer(blob, dtype=_np.float32)
        return float(_np.dot(q, v))
    arr = array.array("f")
    arr.frombytes(blob)
    if hasattr(math, "sumprod"):
        return float(math.sumprod(query_vec, arr))
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

    Uses a single vectorized BLAS matrix-vector multiply when NumPy is installed, avoiding
    per-row Python/NumPy allocation overhead and skipping hydration of non-matching rows.
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
    if _np is not None:
        q = _np.asarray(query_vec, dtype=_np.float32)
        mat = _np.frombuffer(b"".join(valid_blobs), dtype=_np.float32).reshape(-1, dims)
        sims = mat @ q
        indices = _np.flatnonzero(sims >= min_similarity)
        if indices.size == 0:
            return []
        if indices.size > top_k:
            sub = sims[indices]
            order = _np.argsort(sub)[::-1][:top_k]
            indices = indices[order]
        else:
            order = _np.argsort(sims[indices])[::-1]
            indices = indices[order]
        for idx in indices.tolist():
            scored.append((valid_keys[idx], float(sims[idx])))
        return scored

    for key, raw in zip(valid_keys, valid_blobs, strict=True):
        sim = cosine_similarity(query_vec, raw, dims)
        if sim >= min_similarity:
            scored.append((key, sim))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def _trim_os_memory() -> None:
    """Release freed C/C++ heap arenas back to the OS (critical for glibc + onnxruntime)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def _unload_if_idle(key: tuple[str, int], idle_seconds: float) -> None:
    dropped = False
    with _local_lock:
        last = _local_last_used.get(key)
        if last is not None and (time.monotonic() - last) >= max(0.0, idle_seconds - 0.05):
            if _local_models.pop(key, None) is not None:
                dropped = True
            _local_last_used.pop(key, None)
            _local_timers.pop(key, None)
    if dropped:
        _trim_os_memory()


def _schedule_unload_locked(key: tuple[str, int], idle_seconds: int) -> None:
    old = _local_timers.pop(key, None)
    if old is not None:
        old.cancel()
    if idle_seconds <= 0:
        return
    timer = threading.Timer(float(idle_seconds), _unload_if_idle, args=(key, float(idle_seconds)))
    timer.daemon = True
    timer.start()
    _local_timers[key] = timer


def unload_local_models() -> int:
    """Unload all cached local ONNX models, clear query cache, and trim OS heap memory."""
    with _local_lock:
        count = len(_local_models)
        for timer in _local_timers.values():
            timer.cancel()
        _local_timers.clear()
        _local_models.clear()
        _local_last_used.clear()
        _query_cache.clear()
    if count > 0:
        _trim_os_memory()
    return count


def _get_local_model(model_name: str, *, threads: int = 2, idle_unload_seconds: int = 300) -> Any:
    key = (model_name, max(1, threads))
    with _local_lock:
        found = _local_models.get(key)
        if found is None:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]

            # Evict stale thread-count instances of the same model before loading.
            for existing_key in [k for k in _local_models if k[0] == model_name]:
                old_timer = _local_timers.pop(existing_key, None)
                if old_timer is not None:
                    old_timer.cancel()
                _local_models.pop(existing_key, None)
                _local_last_used.pop(existing_key, None)
            try:
                found = TextEmbedding(model_name=model_name, threads=key[1])
            except TypeError:
                found = TextEmbedding(model_name=model_name)
            _local_models[key] = found
        _local_last_used[key] = time.monotonic()
        _schedule_unload_locked(key, idle_unload_seconds)
        return found


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


def _embed_local(
    cfg: EmbeddingConfig, model_name: str, text: str, *, is_query: bool
) -> list[float] | None:
    prepared = _apply_model_prefix(model_name, text, is_query=is_query)
    model = _get_local_model(
        model_name,
        threads=cfg.threads,
        idle_unload_seconds=cfg.idle_unload_seconds,
    )
    vec = next(iter(model.embed([prepared])), None)
    if vec is None:
        return None
    return _normalize(vec.tolist() if hasattr(vec, "tolist") else list(vec))


def _embed_local_batch(
    cfg: EmbeddingConfig, model_name: str, texts: Sequence[str], *, is_query: bool
) -> list[list[float] | None]:
    prepared = [_apply_model_prefix(model_name, t, is_query=is_query) for t in texts]
    model = _get_local_model(
        model_name,
        threads=cfg.threads,
        idle_unload_seconds=cfg.idle_unload_seconds,
    )
    out: list[list[float] | None] = []
    for vec in model.embed(prepared):
        normed = _normalize(vec.tolist() if hasattr(vec, "tolist") else list(vec))
        out.append(normed if normed else None)
    return out


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
    url = f"{base}/{full_model}:embedContent?key={api_key}"
    payload: dict[str, Any] = {
        "model": full_model,
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
    }
    data = _http_post_json(url, payload, {}, cfg.timeout)
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
        return _embed_local(cfg, model_name, text, is_query=is_query)
    if cfg.provider == "ollama":
        return _embed_ollama(cfg, model_name, text, is_query=is_query)
    if cfg.provider == "openai":
        return _embed_openai(cfg, model_name, text)
    if cfg.provider == "gemini":
        return _embed_gemini(cfg, model_name, text, is_query=is_query)
    return None


_DEFAULT_EMBED_RAW = _embed_raw


def embed_text(cfg: EmbeddingConfig, text: str, *, is_query: bool = False) -> list[float] | None:
    """Compute an L2-normalized embedding vector, returning `None` on any failure."""
    if not is_enabled(cfg) or not text.strip():
        return None
    model_name = effective_model(cfg)
    cache_key = (id(_embed_raw), cfg.provider, model_name, text)
    if is_query:
        with _local_lock:
            cached = _query_cache.get(cache_key)
            if cached is not None:
                _query_cache.move_to_end(cache_key)
                return list(cached)
    try:
        vec = _embed_raw(cfg, text, is_query=is_query)
    except (
        ImportError,
        OSError,
        ValueError,
        urllib.error.URLError,
        RuntimeError,
    ):
        return None
    if not vec:
        return None
    if is_query:
        with _local_lock:
            _query_cache[cache_key] = list(vec)
            _query_cache.move_to_end(cache_key)
            while len(_query_cache) > _QUERY_CACHE_MAX:
                _query_cache.popitem(last=False)
    return vec


def embed_batch(
    cfg: EmbeddingConfig, texts: Sequence[str], *, is_query: bool = False
) -> list[list[float] | None]:
    """Batch-embed texts efficiently for local ONNX, falling back to per-item calls otherwise."""
    if not texts:
        return []
    if not is_enabled(cfg):
        return [None for _ in texts]
    model_name = effective_model(cfg)
    if cfg.provider == "local" and model_name and _embed_raw is _DEFAULT_EMBED_RAW:
        try:
            return _embed_local_batch(cfg, model_name, texts, is_query=is_query)
        except (ImportError, OSError, ValueError, RuntimeError):
            return [None for _ in texts]
    return [embed_text(cfg, t, is_query=is_query) for t in texts]


def store_fact_embedding(
    conn: sqlite3.Connection,
    fact_id: str,
    text: str,
    cfg: EmbeddingConfig,
    *,
    vector: Sequence[float] | None = None,
) -> bool:
    """Store (or update) a fact's normalized vector in `fact_embeddings`. Returns True if stored."""
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
        conn.execute(
            "INSERT INTO fact_embeddings (fact_id, model, dims, vec, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(fact_id) DO UPDATE SET "
            "model = excluded.model, dims = excluded.dims, vec = excluded.vec, "
            "updated_at = excluded.updated_at",
            (fact_id, model, len(vec), blob, now),
        )
    return True


def store_note_embedding(
    conn: sqlite3.Connection,
    note_id: int,
    heading: str,
    text: str,
    cfg: EmbeddingConfig,
    *,
    vector: Sequence[float] | None = None,
) -> bool:
    """Store (or update) a document note's normalized vector in `note_embeddings`."""
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
    min_similarity: float = MIN_SEMANTIC_SIMILARITY,
    query_vec: Sequence[float] | None = None,
) -> tuple[dict[str, float], dict[str, tuple[sqlite3.Connection, sqlite3.Row]]]:
    """Cosine similarity scores for visible facts matching the active embedding model."""
    cfg = ctx.scope.embeddings
    if not is_enabled(cfg):
        return {}, {}
    vec = list(query_vec) if query_vec is not None else embed_text(cfg, query, is_query=True)
    if not vec:
        return {}, {}
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
            min_similarity=min_similarity,
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


def semantic_note_scores(
    ctx: Context,
    query: str,
    *,
    min_similarity: float = MIN_SEMANTIC_SIMILARITY,
    query_vec: Sequence[float] | None = None,
) -> tuple[dict[int, float], dict[int, sqlite3.Row]]:
    """Cosine similarity scores for ingested document notes matching the active model."""
    cfg = ctx.scope.embeddings
    if not is_enabled(cfg):
        return {}, {}
    vec = list(query_vec) if query_vec is not None else embed_text(cfg, query, is_query=True)
    if not vec:
        return {}, {}
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
        min_similarity=min_similarity,
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

    now = timeutil.iso(timeutil.now())
    facts_indexed = facts_skipped = notes_indexed = notes_skipped = errors = 0
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
            vecs = embed_batch(cfg, [text for _, text in chunk], is_query=False)
            ts = timeutil.iso(timeutil.now())
            with transaction(conn):
                for (fid, _), vec in zip(chunk, vecs, strict=True):
                    blob = pack_vector(vec) if vec else b""
                    if not blob or not vec:
                        errors += 1
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
    )
