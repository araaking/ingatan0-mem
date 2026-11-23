"""Optional pluggable semantic embeddings stored as L2-normalized float32 BLOBs in SQLite.

Supports:
- `none` (default): disabled, zero extra dependencies or RAM.
- `local`: offline ONNX multilingual embeddings via optional `fastembed` (`ingatan[vector]`).
- `ollama`: local Ollama `/api/embed` via Python stdlib `urllib` (zero extra dependencies).
- `openai`: OpenAI or OpenAI-compatible `/embeddings` via stdlib `urllib`.
- `gemini`: Google Generative Language `embedContent` via stdlib `urllib`.

All vectors are stored directly inside `<business>.db` / `global.db` so single-file isolation
and byte-level `forget` (`PRAGMA secure_delete = ON`) remain intact. Every runtime embedding
call fails gracefully (returning `None`) so `recall` and `remember` never break when offline.
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.scope import Context, EmbeddingConfig

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional extra
    _np = None

__all__ = [
    "DEFAULT_MODELS",
    "MIN_SEMANTIC_SIMILARITY",
    "ReindexReport",
    "cosine_similarity",
    "effective_model",
    "embed_text",
    "is_enabled",
    "pack_vector",
    "reindex",
    "semantic_fact_scores",
    "semantic_note_scores",
    "store_fact_embedding",
    "store_note_embedding",
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

_local_lock = threading.Lock()
_local_models: dict[str, Any] = {}


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


def _get_local_model(model_name: str) -> Any:
    with _local_lock:
        found = _local_models.get(model_name)
        if found is None:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]

            found = TextEmbedding(model_name=model_name)
            _local_models[model_name] = found
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


def _embed_local(model_name: str, text: str, *, is_query: bool) -> list[float] | None:
    # E5 family models perform significantly better with "query: " / "passage: " prefixes.
    prepared = text
    if "e5-" in model_name.casefold():
        prefix = "query: " if is_query else "passage: "
        if not text.casefold().startswith(prefix):
            prepared = prefix + text
    model = _get_local_model(model_name)
    vec = next(iter(model.embed([prepared])), None)
    if vec is None:
        return None
    return _normalize(vec.tolist() if hasattr(vec, "tolist") else list(vec))


def _embed_ollama(cfg: EmbeddingConfig, model_name: str, text: str) -> list[float] | None:
    base = (cfg.api_base or DEFAULT_BASES["ollama"]).rstrip("/")
    data = _http_post_json(
        f"{base}/api/embed", {"model": model_name, "input": text}, {}, cfg.timeout
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
        return _embed_local(model_name, text, is_query=is_query)
    if cfg.provider == "ollama":
        return _embed_ollama(cfg, model_name, text)
    if cfg.provider == "openai":
        return _embed_openai(cfg, model_name, text)
    if cfg.provider == "gemini":
        return _embed_gemini(cfg, model_name, text, is_query=is_query)
    return None


def embed_text(cfg: EmbeddingConfig, text: str, *, is_query: bool = False) -> list[float] | None:
    """Compute an L2-normalized embedding vector, returning `None` on any failure."""
    try:
        vec = _embed_raw(cfg, text, is_query=is_query)
        return vec if vec else None
    except (
        ImportError,
        OSError,
        ValueError,
        urllib.error.URLError,
        RuntimeError,
    ):
        return None


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
) -> bool:
    """Store (or update) a document note's normalized vector in `note_embeddings`."""
    if not is_enabled(cfg):
        return False
    combined = f"{heading}\n{text}".strip()
    vec = embed_text(cfg, combined, is_query=False)
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
        sql = (
            "SELECT f.*, e.dims AS _dims, e.vec AS _vec FROM fact_embeddings e "
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
        for row in conn.execute(sql, params):
            sim = cosine_similarity(vec, row["_vec"], int(row["_dims"]))
            if sim >= min_similarity:
                fid = str(row["id"])
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
    scores: dict[int, float] = {}
    rows_out: dict[int, sqlite3.Row] = {}
    sql = (
        "SELECT n.*, e.dims AS _dims, e.vec AS _vec FROM note_embeddings e "
        "JOIN notes n ON n.id = e.note_id "
        "WHERE e.model = :model AND e.dims = :dims"
    )
    for row in ctx.business.execute(sql, {"model": model, "dims": dims}):
        sim = cosine_similarity(vec, row["_vec"], int(row["_dims"]))
        if sim >= min_similarity:
            nid = int(row["id"])
            scores[nid] = sim
            rows_out[nid] = row
    return scores, rows_out


def reindex(ctx: Context, *, force: bool = False) -> ReindexReport:
    """Compute embeddings for all valid visible facts and document notes in this context."""
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
        existing_models = {
            str(r["fact_id"]): str(r["model"])
            for r in conn.execute("SELECT fact_id, model FROM fact_embeddings")
        }
        rows = conn.execute(
            f"SELECT id, text FROM facts WHERE {valid_clause} AND {ctx.visible_sql(conn)}",
            {"now": now, **ctx.params},
        ).fetchall()
        for row in rows:
            fid = str(row["id"])
            if not force and existing_models.get(fid) == model:
                facts_skipped += 1
                continue
            if store_fact_embedding(conn, fid, str(row["text"]), cfg):
                facts_indexed += 1
            else:
                errors += 1

    existing_note_models = {
        int(r["note_id"]): str(r["model"])
        for r in ctx.business.execute("SELECT note_id, model FROM note_embeddings")
    }
    for row in ctx.business.execute("SELECT id, heading, text FROM notes").fetchall():
        nid = int(row["id"])
        if not force and existing_note_models.get(nid) == model:
            notes_skipped += 1
            continue
        if store_note_embedding(ctx.business, nid, str(row["heading"]), str(row["text"]), cfg):
            notes_indexed += 1
        else:
            errors += 1

    return ReindexReport(
        model=model,
        facts_indexed=facts_indexed,
        facts_skipped=facts_skipped,
        notes_indexed=notes_indexed,
        notes_skipped=notes_skipped,
        errors=errors,
    )
