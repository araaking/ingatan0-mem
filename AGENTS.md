# AGENTS.md

Instructions for AI coding agents and contributors working in this repository.

## Project Overview

**Ingatan** is an open-source, local-first shared memory MCP server for AI agents backed by plain SQLite + FTS5 (BM25 + trigram + Indonesian affix-stripped root expansion + RRF). No LLM, embedding model, API key, or network call is required on the server.

Core principles (see [docs/DESIGN.md](docs/DESIGN.md)):
1. **Cheap and free first** — every core feature works offline with zero API keys.
2. **Knows when to doubt** — volatile facts expire into `check first`; similar facts surface as `possible conflict`.
3. **Auditable & reversible** — every mutation is written to an append-only audit log (`ingatan diff`, `ingatan undo`).
4. **Measured** — claims in `README.md` must match [evals/reports/latest.md](evals/reports/latest.md).

## Open-Source & Privacy Hygiene (Strict Rule)

This is a **public open-source repository**. Never introduce personal or private details into code, comments, tests, eval scenarios, docs, or `CHANGELOG.md`:
- Do **not** include real user names, customer names, private business/brand names, internal project names, personal locations/events, or real sheet/document IDs.
- Always use generic placeholders such as `tokoku`, `toko-a`, `toko-b`, `proyekku`, `Budi`, `sync_report.py`, and clearly synthetic IDs.
- Keep all improvements general-purpose and configurable rather than tailored to a single user's setup.
- Never touch the real `~/.ingatan` in tests — tests must always run in isolated temporary `INGATAN_HOME` directories (handled by the autouse fixture in `tests/conftest.py`).

## Architecture & Module Layering

Modules may import **only** from modules earlier in this list (see [CONTRIBUTING.md](CONTRIBUTING.md)):

```
timeutil -> errors -> db -> redact -> scope -> audit -> search -> facts -> proposals -> episodes -> brief -> ingest -> extract -> mcp -> cli -> integrations
```

- Keep `mcp/` and `cli/` thin — business logic belongs in `facts/`, `search/`, `brief/`, `proposals/`, and `scope/` so it is directly unit-testable.
- When a lower module must touch rows owned by a higher module (e.g. `audit/_undo.py` or `facts/create.py` updating `episodes`), use plain SQL instead of upward imports.

## Key Invariants

- **Strict scope isolation**:
  - One SQLite file per business (`<business>.db`) plus `global.db`. Never search across businesses.
  - In single-business MCP mode (`build_server`), no tool schema may expose a `business` or `agent` field.
  - In opt-in multi-business MCP mode (`build_multi_server` via `INGATAN_BUSINESSES` / `allowed_businesses`), every call validates `business` against the explicit allowlist and still runs strictly within that single business DB + `global.db`.
- **Write-then-undo default + opt-in approval queue**:
  - `require_approval` defaults to `False` (`remember` and `supersede` on `business`/`global` apply immediately and are reversible via `ingatan undo`).
  - `require_approval = true` (in `.ingatan.toml`, `~/.ingatan/config.toml`, or `INGATAN_REQUIRE_APPROVAL=1`) enables the manual CLI proposal queue.
  - `forget` on `business` and `global` tiers is always CLI-only (`by_user=True`) because `forget` permanently scrubs bytes and cannot be undone.
- **Byte-level `forget` guarantee**: keep `PRAGMA secure_delete = ON`, FTS5 `secure-delete`, `optimize`, and `PRAGMA wal_checkpoint(TRUNCATE)` intact whenever touching DB schema or deletion paths.
- **Semantic search stays optional**: the default provider is `none`. Nothing may import `numpy`, `onnxruntime` or `sentencepiece` at import time, the server must never download a model (only `ingatan embeddings pull` does, and the two commands that call it, `ingatan embeddings enable` and `ingatan setup`), and an embedding must never be computed inside a write transaction. The shared model process (`[embeddings] shared`, `src/ingatan/search/shared.py`) listens on 127.0.0.1 only, downloads nothing, writes no fact text to disk, and a server that cannot reach it answers from the word index: it never fails a tool call and never loads the model itself as a fallback.
- **No bare similarity constants**: cosine scores are not comparable between embedding models. Cut-offs live in a `SimilarityProfile` (`src/ingatan/search/embed.py`) and are changed only with numbers from `scripts/calibrate_embeddings.py`, recorded in [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md). Two facts are never merged because they are similar.
- **Python 3.10+ compatibility**: every `.py` file must begin with `from __future__ import annotations`.

## Development & Verification Commands

Run all checks before finishing a change:

```bash
uv sync
uv run python -m pytest -q          # full test suite
uv run ruff check .                 # lint
uv run ruff format --check .        # formatting
uv run python -m pyright            # static type check
uv run python -m evals              # offline eval suite (updates evals/reports/latest.md)
```

On Windows machines where `.venv\Scripts\python.exe` is blocked by Application Control, invoke the signed system Python (`py`) with `PYTHONPATH` pointing to `src` and `.venv\Lib\site-packages` (including `win32` paths for MCP stdio tests).

## Documentation & Changelog Maintenance

- Record user-visible changes under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) using Keep a Changelog categories (`Added`, `Changed`, `Fixed`). Keep entries general and implementation-accurate.
- When adding or modifying eval scenarios in `evals/scenarios/` (or anonymized real incidents in `evals/scenarios/incidents/`), run `python -m evals` so [evals/reports/latest.md](evals/reports/latest.md) stays up to date, and sync any changed metrics in [README.md](README.md). If the local embedding model is on your machine, also run `python -m evals --embeddings local` (it rewrites [evals/reports/latest-semantic.md](evals/reports/latest-semantic.md)) and `INGATAN_TEST_MODEL_DIR=<model folder> python -m pytest tests/search/test_real_model.py`; CI has no model and runs neither.
- Memory figures in [docs/RAM.md](docs/RAM.md) come from `scripts/measure_ram.py` and `scripts/measure_embed_ram.py`. Do not edit them by hand.
- For new DB schema changes, add a continuous numbered migration `src/ingatan/db/migrations/NNNN_*.sql` — never edit a shipped migration.
