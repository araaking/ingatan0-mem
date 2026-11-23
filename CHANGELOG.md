# Changelog

All notable changes to **Ingatan** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-23

General usability, search precision, and operational improvements based on real-world multi-project agent usage and a full codebase audit.

### Changed
- **Write-then-undo by default (no mandatory CLI approval)**: `remember` and `supersede` on `business` and `global` tiers now apply immediately (`require_approval = false` by default). Every write remains append-only logged and reversible via `ingatan diff` and `ingatan undo`, while `forget` on shared tiers remains CLI-only. The manual CLI proposal queue is still available via `require_approval = true` in `.ingatan.toml` / `~/.ingatan/config.toml` or `INGATAN_REQUIRE_APPROVAL=1`.
- **Optional `keywords` in `remember` with auto-extraction**: omitting `keywords` no longer fails the call; Ingatan automatically derives clean fallback keywords (stopword-filtered terms plus Indonesian affix-stripped roots from `stem_id`) from `text` and `entities` when none are provided.
- **Tighter, higher-precision conflict detection (`similar_facts`)**:
  - Filters out generic entities (business name, agent name, and high-frequency entities appearing in `>= 35%` of valid facts) and common agent/tooling boilerplate words (`Disimpan oleh asisten`, `Google Sheet`, `MCP`, etc.).
  - Eliminates top-50 FTS5 candidate starvation by evaluating entity matches across all valid visible facts.
  - Requires meaningful content-token overlap and adds a resource-ID guard so facts pointing to distinct external document/sheet IDs or URLs do not false-positive conflict.
  - Skips re-opening pairs already recorded or previously resolved in `conflicts`.
- **More accurate & token-efficient `recall`**:
  - Strips Indonesian & English stopwords from query tokens and drops noise-only / weak tail hits on long queries instead of returning unrelated facts.
  - Rebalances recency weight (`0.35` instead of `1.0`) and adds an identifier phrase signal (e.g. `sync_report` matching `sync_report.py`) so newer logs do not outrank procedures and rules.
  - Reduces default document excerpts to `notes=2` (0 disables excerpts) and trims `NOTE_EXCERPT_CHARS` from `1200` to `400`, with relevance filtering on `notes`.
  - Condenses default fact metadata onto one line and adds `compact=true` mode for minimal token usage.
- **Stable-first cold-start ranking & stricter budget in `brief`**:
  - Prioritizes `kind="stable"` foundational facts ahead of newer `volatile` logs when `use_count == 0`, while preserving `use_count -> last_used` ordering once facts are recalled.
  - Tightens character trimming (`3.5` chars/token target) so Indonesian/mixed briefs stay reliably within `token_budget`.
- **Improved MCP tool ergonomics**: `remember` and `recall` descriptions now open with `"Remember..."` / `"Recall..."` for better tool discovery (`tool_search`) and encourage recording user aliases/nicknames in `keywords`.
- **Docs**: `README.md` and `docs/setup-hermes.md` now recommend `uv tool install --editable` for git checkouts and document write-then-undo, `~/.ingatan/config.toml`, and multi-business setup.

### Added
- **Global home configuration (`~/.ingatan/config.toml`)**: shared defaults (`default_agent`, `require_approval`, `[brief].token_budget`, `allowed_businesses`) inherited by CLI, stdio MCP, and HTTP mode (`TokenStore.lookup`).
- **Opt-in multi-business MCP server mode**: set `INGATAN_BUSINESSES` (comma-separated) or `allowed_businesses = [...]` to serve multiple allowlisted businesses from a single MCP process (7 tools total, per-call `business` validated against the allowlist, still strictly isolated with zero cross-business search). Single-business mode remains unchanged with no `business` parameter in tool schemas.
- **Per-call `check_conflicts=false` on `remember`** for append-only daily logs and periodic snapshots.
- **Preserved original dates on `remember`**: accepts `source.date` (`YYYY-MM-DD`) and optional `valid_from` instead of always overwriting with today's date.
- **CLI upgrades**:
  - `--business` / `-b` flag on all CLI commands (and root `ingatan --business ...`) so you no longer need to `cd` into per-business folders.
  - `ingatan proposals approve --all` and multi-ID support with `[i/n]` progress output.
  - `ingatan conflicts list` and `ingatan conflicts scan [--dry-run]` to inspect or retroactively detect conflicts across existing facts.
  - `ingatan import <file.jsonl|json>` for bulk migration with original dates, auto-keywords, and optional `--scan-conflicts`.
  - `ingatan move <fact_id> --to <business>` to transfer misplaced facts cleanly between business databases.
- **Real-incident eval scenarios**: added 3 anonymized incident scenarios under `evals/scenarios/incidents/` (43 scenarios total, all passing with 100% `recall@5`, 0% stale rate, and 0% leak rate) plus comprehensive regression tests in `tests/test_usage_improvements.py`.

### Fixed
- **Stale/dangling conflicts on closed or expired facts**: `_in_open_conflict` now requires the partner fact to satisfy `is_valid_sql()`, so expired or superseded facts no longer leave active facts stuck in `possible conflict`.
- **`proposals.approve` conflict lifecycle**: approving a proposal now re-evaluates `similar_facts` against the live store (catching conflicts across batch-queued proposals), resolves open conflicts on a superseded target, and updates `ingatan conflicts` audit reporting accordingly.
- **Superseding a shared fact with an already-valid shared fact**: in `require_approval=true` mode, `supersede(old_business, existing_valid_business)` now queues a retire-only proposal (with full `audit.undo` support) instead of failing and forcing duplicate fact creation.
- **Cross-DB transaction atomicity in `confirm`**: wraps `ctx.business` and `conn` transactions together when confirming global facts.
- **Markdown ingest directory filtering**: `ingest_path` now skips hidden and vendor directories (`.git`, `.venv`, `node_modules`, `__pycache__`) when recursing a folder.

---

## [0.1.0] - 2026-11-11

### Added
- Initial Ingatan MVP: local-first SQLite + FTS5 (BM25 + trigram + Indonesian affix-stripped root expansion + RRF) shared memory server over MCP stdio (`brief`, `recall`, `remember`, `supersede`, `confirm`, `forget`, `history`).
- Three-tier isolation (`global`, `business`, `agent`), per-fact `review_after` staleness tracking, secret redaction (`[redacted]`), content-scrubbing `forget` down to SQLite/WAL bytes, append-only audit log with `diff` and `undo`, and Markdown note ingestion.
- Claude Code hooks and Hermes Agent memory-provider plugin for episode capture and brief injection.
- HTTP mode (`ingatan serve --http`) with per-scope SHA-256 hashed bearer tokens (`ingatan token add/list/revoke`) and `SECURITY.md`.
- CLI tooling (`init`, `doctor`, `log`, `diff`, `undo`, `proposals`, `forget`, `ingest`, `conflicts`) and offline Indonesian eval harness (40 pattern scenarios).
