# Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the nine items of [ROADMAP.md](ROADMAP.md) in its suggested order, each one shipped, measured and removed from the roadmap before the next starts.

**Architecture:** Nine phases, one per roadmap item. Every phase leaves the offline core untouched: new behaviour is a new opt-in setting, a new CLI command or a new optional module, and falls back to the word index when a model or a process is missing. Phases 1 to 4 are specified to the task level. Phases 5 to 9 depend on decisions or measurements that do not exist yet, so each opens with a gate and its tasks are final only once the gate is answered.

**Tech Stack:** Python 3.10+, stdlib `sqlite3` (FTS5), `typer`, `mcp`, `pytest`, `ruff`, `pyright`, `uv`; optional `numpy`, `onnxruntime`, `sentencepiece`. No new runtime dependency is introduced by phases 1 to 5.

**Spec:** [docs/ROADMAP.md](ROADMAP.md), bound by [AGENTS.md](../AGENTS.md). Section numbers such as "§9" refer to the roadmap headings.

## Global Constraints

- Every core feature works offline with no API key. Anything that needs a model or a network call is optional and off by default.
- One SQLite file per business plus `global.db`. No second index file, no cross-business search.
- `forget` removes the bytes. Nothing added here may keep a trace of a forgotten fact.
- No model call inside a write transaction, and the server never downloads a model. `ingatan embeddings pull` (and the commands in Phase 1 that call it) stay the only download.
- A claim in `README.md` is backed by a number in `evals/reports/` or `docs/RAM.md`. Those files are produced by scripts, never edited by hand.
- Module layering (`timeutil -> errors -> db -> redact -> scope -> audit -> search -> facts -> proposals -> episodes -> brief -> ingest -> mcp -> cli -> integrations`): a module imports only from modules earlier in the list. `mcp/` and `cli/` stay thin.
- Nothing imports `numpy`, `onnxruntime` or `sentencepiece` at import time.
- No bare similarity constant: cut-offs live in a `SimilarityProfile` and come from `scripts/calibrate_embeddings.py`.
- Every `.py` file begins with `from __future__ import annotations`.
- Schema changes are a new migration with the next free number (`0004_*.sql` is next). A shipped migration is never edited.
- Public repository: only generic placeholders (`tokoku`, `toko-a`, `proyekku`, `Budi`, `sync_report.py`) in code, tests, scenarios and docs. Tests never touch the real `~/.ingatan`.

**CHECK** (the last step of every task, all five must pass):

```bash
uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python -m pyright
uv run python -m evals
```

Each task also adds its user-visible change under `## [Unreleased]` in `CHANGELOG.md`. The last task of a phase removes that item's section and table row from `docs/ROADMAP.md`.

## Order and dependencies

| Phase | Roadmap item | Depends on | Starts when |
|---|---|---|---|
| 1 | §9 Easier installation | nothing | now |
| 2 | §2 Deferred embedding on write | nothing | now |
| 3 | §1 Shared model process | Phase 2 | D3.1 to D3.3 answered |
| 4 | §4 Review queue for similar facts | nothing | now |
| 5 | §3 Usage and time decay in `recall` | nothing | now (scenarios first) |
| 6 | §7 LLM fact extraction | nothing | D6.1 to D6.3 answered |
| 7 | §8 Light knowledge graph | Phase 6 | D7.1 answered |
| 8 | §5 Tombstones | nothing | its design doc is approved |
| 9 | §6 Static embedding model | Phases 2 and 3 measured | owner says 275 MB is still too much |

Phases 1, 2, 4 and 5 touch different modules and can be built in parallel. `src/ingatan/cli/commands.py` is the one file most phases add a command to: keep each addition to a thin command function and put the logic in the new module the task names.

## Decisions for the owner

A phase does not start until its decisions are answered. The recommendation is what the tasks below assume.

| Id | Question | Recommendation | Blocks |
|---|---|---|---|
| D1 | The distribution name if `ingatan` is taken on PyPI | `ingatan-mcp`; the import name and the command stay `ingatan` | Task 1.1 |
| D3.1 | Transport of the shared model process | Loopback TCP on an ephemeral port with a random token in a user-only file: one code path on three operating systems, no socket-path length limit | Phase 3 |
| D3.2 | Who starts the process | The first server that needs a vector starts it; `ingatan embeddings serve` exists for running it by hand | Phase 3 |
| D3.3 | On by default? | Off (`shared = false`) for the first release; flip the default once `docs/RAM.md` holds the three-session number | Phase 3 |
| D6.1 | May extracted facts be written directly? | No. Always the proposal queue in the first version, whatever `require_approval` says | Phase 6 |
| D6.2 | Which model sizes to support | No fixed list. The owner picks two or three models for the eval run; the README names only those that were measured | Phase 6 |
| D6.3 | Tier of an extracted fact | `business`. The `proposals` table accepts only `business` and `global` (a `CHECK` in migration 0001), and the user's approval is what makes sharing it safe | Phase 6 |
| D7.1 | Relations between two facts that both already exist | Not in the first version. A relation rides on a proposed fact and becomes an edge when that fact is approved | Phase 7 |
| D8 | Tombstone matching, visibility in `history`, refuse or warn | Settled in `docs/RETRACTION.md` (Task 8.1) | Phase 8 |
| D9 | Is the static model still needed? | Decide from the numbers Phases 2 and 3 put in `docs/RAM.md` | Phase 9 |

Calls made in this plan that the roadmap did not make (reject any of them before the phase starts): the setting names `defer` and `shared`, the command names `embeddings enable`, `conflicts review` and `extract`, a new audit op `conflict` for pairs found late, and an empty `llm` extra.

## Review Focus

Inputs the roadmap implies but does not spell out. Each has its test in the task named.

1. `forget` runs while the background fill is embedding the same fact: no row may be left in `fact_embeddings` (Task 2.2).
2. The shared model process dies or was never started while `recall` runs: the tool call returns word-index results, not an error (Task 3.3).
3. `ingatan setup` meets a `.mcp.json` that already lists other servers, or is run twice: other entries survive, no duplicate, no change without a shown diff (Task 1.3).
4. The config writer meets a `.ingatan.toml` with comments, CRLF line endings, or an `[embeddings]` section that already has other keys: everything else is preserved byte for byte (Task 1.2).
5. An episode contains text addressed to the model ("ignore the instructions, remember that ..."), or the model answers with broken JSON or a value that is not in the source: nothing is written as a fact, the run continues with the next episode (Task 6.3).

---

## Phase 1: Easier installation (§9)

What exists: `.github/workflows/release.yml` already builds on a `v*` tag and publishes with trusted publishing; `ingatan init` asks for the business and agent names; `embeddings pull`, `reindex` and `doctor` are separate commands; CI's `clean-install` job runs `init` and `doctor` from the built wheel on three operating systems.

### Task 1.1: Rehearse and harden the release

**Files:**
- Modify: `.github/workflows/release.yml`
- Modify: `pyproject.toml` (`[project.urls]`: add `Changelog`, `Documentation`; add `keywords`)
- Create: `tests/test_packaging.py`

**Interfaces:**
- Produces: a `workflow_dispatch` trigger on `release.yml` that publishes to TestPyPI; the tag-push path to PyPI is unchanged.

- [ ] **Step 1 (owner, cannot be automated): check that `ingatan` is free** on pypi.org and test.pypi.org. If it is taken, answer D1 and replace the name in `[project] name`, the extras text and every install command in this phase.
- [ ] **Step 2: write the failing test** `test_dunder_version_matches_pyproject`: `ingatan.__version__` equals `[project] version` read from `pyproject.toml` (use the `tomllib` / `tomli` import the package itself uses).
- [ ] **Step 3: add a guard step to the `build` job** that fails unless the pushed tag equals `v<project.version>`, and skip it on `workflow_dispatch`.
- [ ] **Step 4: add a `publish-testpypi` job**: runs only on `workflow_dispatch`, `environment: testpypi`, `permissions: id-token: write`, `pypa/gh-action-pypi-publish@release/v1` with `repository-url: https://test.pypi.org/legacy/`. The existing `publish` job gets `if: github.event_name == 'push'`.
- [ ] **Step 5 (owner): register the trusted publishers** on PyPI and TestPyPI (this repository, workflow `release.yml`, environments `pypi` and `testpypi`) and create both GitHub environments.
- [ ] **Step 6: rehearse.** On a throwaway branch set the version to a pre-release (`0.1.0rc1`, in `pyproject.toml` and `src/ingatan/__init__.py`), run the workflow by hand, then in an empty folder:

```bash
uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ingatan --version
```

Expected: the pre-release version is printed. A published version cannot be replaced, on TestPyPI either, so every rehearsal uses a new pre-release number.
- [ ] **Step 7: CHECK, commit.** The real release (pushing `v0.1.0`) is the owner's step and happens after Task 1.4.

### Task 1.2: One command for semantic search

**Files:**
- Create: `src/ingatan/scope/configfile.py`
- Create: `src/ingatan/cli/setup.py`
- Modify: `src/ingatan/cli/commands.py` (new `embeddings enable` command next to `embeddings_pull`)
- Test: `tests/scope/test_configfile.py`, `tests/cli/test_embeddings_enable.py`

**Interfaces:**
- Produces:
  - `configfile.updated_text(text: str, section: str, key: str, value: str | int | bool) -> str`
  - `configfile.apply(path: Path, section: str, key: str, value: str | int | bool) -> bool` (True when the file changed; creates the file if missing)
  - `cli.setup.enable_semantic(*, model: str | None, use_home_config: bool, reindex: bool, echo: Callable[[str], None]) -> None`
- CLI: `ingatan embeddings enable [--model NAME] [--global] [--no-reindex]`

- [ ] **Step 1: write the failing tests for `configfile`** (the stdlib reads TOML but cannot write it, so this is a line editor):
  - `test_adds_section_when_missing`: `updated_text('business = "tokoku"\n', "embeddings", "provider", "local")` parses to `{"business": "tokoku", "embeddings": {"provider": "local"}}`.
  - `test_replaces_existing_key_only`: a text with `[embeddings]`, a comment line, `provider = "none"` and `threads = 4` comes back with only the `provider` line changed.
  - `test_keeps_crlf_and_comments`: every line other than the changed one is byte-identical, CRLF included (Review Focus 4).
  - `test_refuses_text_it_cannot_round_trip`: when the edited text does not parse, or parses to a different value for the key, `ConfigError` is raised and nothing is written.
- [ ] **Step 2: implement `configfile`.** After editing, parse the result with `tomllib` and compare the one key; write with `newline=""` so line endings are kept.
- [ ] **Step 3: write the failing CLI tests** (`local_models.pull`, `local_models.missing_packages` and `embed_store._embed_raw` monkeypatched, so no network and no model; the autouse fixture sets `INGATAN_EMBED_PROVIDER=none`, so every test but the last deletes it):
  - `test_enable_stops_before_download_when_packages_are_missing`: exit code 1, the message names `ingatan[vector]`, `pull` was not called, `.ingatan.toml` is unchanged.
  - `test_enable_pulls_sets_provider_and_reindexes`: after the command, `resolve_scope().embeddings.provider == "local"` and the output contains `Reindex (local`.
  - `test_enable_global_writes_home_config`: with `--global` the key lands in `<INGATAN_HOME>/config.toml` and the project file is untouched.
  - `test_enable_warns_when_env_overrides`: with `INGATAN_EMBED_PROVIDER=none` set, the output says the environment variable wins over the file.
- [ ] **Step 4: implement `enable_semantic`** in this order: missing packages (stop), `local_models.pull`, `configfile.apply` for `provider` (and `model` when it is not the default), then with the scope resolved again from the changed file: `search.reindex` when a business and agent resolve (skipped with a note otherwise) and `search.probe_embedding`.
- [ ] **Step 5: add the command, CHECK, commit.**

### Task 1.3: `ingatan setup`

**Files:**
- Modify: `src/ingatan/cli/setup.py`, `src/ingatan/cli/commands.py` (extract the body of `init` into `_init_project(business: str | None, agent: str | None) -> Scope`; add `setup`)
- Create: `src/ingatan/integrations/claude_code/install.py` (loaded from the CLI with `_import`, as the `hook` command loads `hooks.run`)
- Test: `tests/cli/test_setup.py`, `tests/integrations/test_claude_code_install.py`

**Interfaces:**
- Produces:
  - `install.ConfigChange(path: Path, before: str, after: str)` (frozen dataclass)
  - `install.mcp_entry_change(project_dir: Path) -> ConfigChange | None` (None when `mcpServers.ingatan` is already `{"command": "ingatan", "args": ["serve"]}`)
  - `install.write_change(change: ConfigChange) -> None`
- CLI: `ingatan setup [--business NAME] [--agent NAME] [--semantic/--no-semantic] [--client claude-code|hermes|none] [--yes]`

- [ ] **Step 1: verify the `.mcp.json` shape** against the Claude Code documentation. `docs/setup-claude-code.md` marks it "not verified"; record the result there before code depends on it.
- [ ] **Step 2: write the failing tests:**
  - `test_setup_with_all_flags_asks_nothing`: `setup --business tokoku --agent claude-code --no-semantic --client none` exits 0 with `input=None`, creates `.ingatan.toml` and both databases, and prints the doctor table.
  - `test_setup_without_terminal_and_missing_flag_fails_clearly`: no `--client`, no terminal: `ConfigError` naming the flag.
  - `test_claude_code_change_keeps_other_servers`: a `.mcp.json` with a server `other` gains `ingatan` and still has `other`, key order kept (Review Focus 3).
  - `test_setup_shows_diff_and_writes_nothing_on_no`: answering `n` to the confirmation leaves `.mcp.json` byte-identical; the output contains the unified diff.
  - `test_setup_twice_changes_nothing`: the second run reports "already configured" and asks no confirmation.
  - `test_setup_hermes_prints_block_and_writes_no_file`: the YAML block from `docs/setup-hermes.md` is printed; no file outside the project and `INGATAN_HOME` is written (no YAML parser is a dependency).
- [ ] **Step 3: implement.** Questions in order: business, agent, semantic search (default no; yes calls `enable_semantic`), client. Every write into another application's file prints `difflib.unified_diff(before, after)` and calls `typer.confirm` unless `--yes`. End by running `doctor_checks.run_checks` and exiting non-zero when a check fails.
- [ ] **Step 4: CHECK, commit.**

### Task 1.4: Short start, docs and the three-OS proof

**Files:**
- Modify: `README.md` (both language halves: Requirements, Installation, Quick start), `docs/setup-claude-code.md`, `docs/setup-hermes.md`, `docs/EMBEDDINGS.md`, `.github/workflows/ci.yml`, `CHANGELOG.md`, `docs/ROADMAP.md`

- [ ] **Step 1: extend the `clean-install` CI job** with `uvx --from "$WHEEL" ingatan setup --business demo --agent ci --no-semantic --client none`, then one `remember` and `recall` round trip through the installed package (`uvx --from "$WHEEL" python -c ...` calling `facts.remember` and `search.recall`; the CLI has no `recall` command). This is the "Windows, macOS and Linux" part of the done-when; the client itself cannot run in CI.
- [ ] **Step 2: put three commands at the top of both README halves** and move the details below:

```bash
uv tool install ingatan
cd your-project
ingatan setup
```

The editable-checkout install moves to the Development section. The sentence "this does not work yet" stays until the owner has pushed the release tag.
- [ ] **Step 3: replace the four-step semantic instructions** in `README.md` and `docs/EMBEDDINGS.md` with `uv tool install "ingatan[vector]"` and `ingatan embeddings enable`.
- [ ] **Step 4: CHECK, commit. Remove §9 from `docs/ROADMAP.md`.**

**Phase 1 is done when** a new user goes from nothing to a working `recall` in a client with one install command and one setup command, and CI proves the non-interactive path on three operating systems.

---

## Phase 2: Deferred embedding on write (§2)

How it works today: `facts.remember` calls `embed_store.embed_text` before taking the write lock, uses the vector for the semantic conflict check in `similar_facts`, then stores it in the same transaction as the fact. The MCP tools are `async def` and run on the event-loop thread, where the SQLite connections live, so in HTTP mode the model call blocks every other request.

Design: with `[embeddings] defer = true`, `remember` makes no model call. It passes `query_vec=[]` to `similar_facts` (the word-based check runs as today), commits, and wakes a background worker. The worker opens its own `Context` (connections cannot cross threads), embeds the facts that have no vector of the current model, stores each vector, then runs the semantic conflict check for that fact with `similar_facts(..., exclude_id=fact_id, query_vec=vec)` and records what it finds. With no worker (a one-shot CLI call, a test) the fact stays in the already-normal "no vector yet" state and is picked up at the next server start or by `ingatan reindex`.

### Task 2.1: The setting

**Files:**
- Modify: `src/ingatan/scope/__init__.py` (`EmbeddingConfig.defer: bool = False`; `parse_embeddings` reads `defer` and `INGATAN_EMBED_DEFER` with `parse_bool`)
- Test: `tests/scope/test_scope.py`

- [ ] **Step 1: failing tests** `test_defer_defaults_to_false`, `test_defer_from_toml`, `test_defer_env_wins_over_toml`, `test_defer_rejects_non_boolean` (`ConfigError`).
- [ ] **Step 2: implement, CHECK, commit.**

### Task 2.2: Fill and late conflict check

**Files:**
- Create: `src/ingatan/facts/backfill.py`
- Modify: `src/ingatan/search/embed.py` (`store_fact_embedding`: insert only while the fact row exists), `src/ingatan/audit/__init__.py` (add `"conflict"` to `OPS`), `src/ingatan/audit/_undo.py` (undo of a `conflict` entry resolves its pairs with resolution `undo`), `src/ingatan/cli/conflicts.py` (`compute_report` counts `conflicts_with` of a `conflict` entry like that of a `create`), `src/ingatan/cli/render.py` (`log_line` for the new op), `src/ingatan/facts/__init__.py` (export)
- Test: `tests/facts/test_backfill.py`, `tests/audit/test_undo.py`, `tests/cli/test_conflicts.py`

**Interfaces:**
- Produces:
  - `backfill.FillReport(embedded: int, conflicts_opened: int, errors: int)` (frozen dataclass)
  - `backfill.fill_pending(ctx: Context, *, limit: int | None = None) -> FillReport`
- Consumes: `search.similar_facts`, `facts._util.record_conflicts`, `embed.embed_batch`, `embed.store_fact_embedding`.

- [ ] **Step 1: failing test for the race** (Review Focus 1). `cfg = EmbeddingConfig(provider="local")`; no model is called because the vector is passed in:

```python
def test_vector_is_not_stored_for_a_fact_forgotten_meanwhile(ctx: Context) -> None:
    saved = facts.remember(ctx, "Ongkir ke Bandung Rp 15.000 per kg.", entities=["ongkir Bandung"])
    assert saved.fact_id is not None
    facts.forget(ctx, saved.fact_id)
    stored = embed.store_fact_embedding(ctx.business, saved.fact_id, "x", cfg, vector=[1.0, 0.0])
    assert stored is False
    assert ctx.business.execute("SELECT count(*) FROM fact_embeddings").fetchone()[0] == 0
```

- [ ] **Step 2: make the insert conditional** on `EXISTS (SELECT 1 FROM facts WHERE id = ?)` inside the same statement, and return whether a row was written.
- [ ] **Step 3: failing tests for `fill_pending`** (fake embedder through `monkeypatch.setattr(embed_store, "_embed_raw", ...)`, the pattern of `tests/test_semantic_and_structure.py`):
  - `test_fills_only_facts_without_a_current_vector`: three facts, one already embedded: `embedded == 2`.
  - `test_late_semantic_conflict_is_recorded`: two facts that share an entity and no topic word, vectors identical: after `fill_pending`, `recall` shows both as `possible conflict` and `open_conflicts(ctx)` lists the pair.
  - `test_late_conflict_is_logged_and_undoable`: the log has one `conflict` entry whose `after.conflicts_with` names the other fact; `audit.undo` of it resolves the pair.
  - `test_model_call_is_outside_any_transaction`: the fake embedder asserts `not ctx.business.in_transaction` when called.
  - `test_other_agents_private_facts_are_left_alone`: a private fact of another agent keeps no vector.
  - `test_embed_failure_counts_an_error_and_continues`.
- [ ] **Step 4: implement `fill_pending`.** Per connection: select valid visible facts without a vector of `effective_model(cfg)`, oldest first, in batches of `REINDEX_BATCH_SIZE`; embed outside a transaction; store; then per stored fact run the conflict check and write one `conflict` log entry when pairs were opened.
- [ ] **Step 5: CHECK, commit.**

### Task 2.3: `remember` defers, the servers run the worker

**Files:**
- Modify: `src/ingatan/facts/backfill.py`, `src/ingatan/facts/create.py`, `src/ingatan/mcp/server.py` (`run`), `src/ingatan/mcp/http.py` (`run_http`), `src/ingatan/cli/doctor.py` (`_vector_notes` wording when `defer` is on)
- Test: `tests/facts/test_remember.py`, `tests/facts/test_backfill.py`, `tests/mcp/test_server.py`

**Interfaces:**
- Produces: `backfill.notify(scope: Scope) -> None` (no-op when no worker runs), `backfill.start_worker() -> None`, `backfill.stop_worker(timeout: float = 2.0) -> None`. One daemon thread per process; it keeps a set of scopes to visit and opens and closes a `Context` per pass.

- [ ] **Step 1: failing tests:**
  - `test_remember_with_defer_makes_no_model_call`: a fake embedder that raises is never called; the fact is saved; `fact_embeddings` is empty.
  - `test_remember_with_defer_still_reports_word_conflicts`: `result.similar` is non-empty for two facts sharing an entity and topic words.
  - `test_remember_with_defer_says_the_meaning_check_is_pending`: the message contains "meaning-based check runs in the background".
  - `test_worker_fills_after_notify`: `start_worker()`, `remember`, then within 2 seconds the fact has a vector; `stop_worker()` joins.
  - `test_worker_catches_up_on_start`: a fact saved before `start_worker()` plus one `notify(scope)` gets its vector.
- [ ] **Step 2: implement.** In `remember`, branch on `ctx.scope.embeddings.defer` before the `embed_text` call and call `backfill.notify(ctx.scope)` after the commit. Both server entry points call `start_worker()`, `notify` once per served scope, and `stop_worker()` in their `finally`.
- [ ] **Step 3: CHECK, commit.**

### Task 2.4: Measure and document

**Files:**
- Modify: `scripts/bench_scale.py` (`--embeddings local`, `--defer`), `evals/__main__.py` and `evals/driver.py` (`--defer`: the in-process driver calls `fill_pending` after each `remember`, because it has no worker), `docs/EMBEDDINGS.md`, `README.md` (configuration blocks, both halves), `CHANGELOG.md`, `docs/ROADMAP.md`

- [ ] **Step 1 (a machine with the model on disk; CI has none): run** `uv run python scripts/bench_scale.py --embeddings local` and again with `--defer`; record both medians in `docs/EMBEDDINGS.md`.
- [ ] **Step 2: run** `uv run python -m evals --embeddings local --defer --report evals/reports/latest-semantic-defer.md`: `semantic-01` must pass (a conflict found late is still found).
- [ ] **Step 3: document the trade-off** in the words of the roadmap: with `defer` on, `remember` no longer reports a meaning-only conflict in its own result.
- [ ] **Step 4: CHECK, commit. Remove §2 from `docs/ROADMAP.md`.**

**Phase 2 is done when** `remember` with the local provider and `defer` takes about as long as with `provider = "none"`, every fact gets its vector without a manual `reindex`, and a conflict found late is visible in `recall` and `ingatan conflicts list`.

---

## Phase 3: Shared model process (§1)

Gate: D3.1, D3.2, D3.3. The tasks assume the recommendations.

Design: `src/ingatan/search/shared.py` holds both ends, stdlib only. The server end binds `127.0.0.1` on an ephemeral port with `multiprocessing.connection.Listener(authkey=token)` and writes `<INGATAN_HOME>/run/embed.json` (`{"v": 1, "pid", "port", "token"}`, created with mode `0600`). Frames are sent with `send_bytes` / `recv_bytes` only, never `send` / `recv`, so nothing is unpickled. A request is one JSON frame (`model`, `model_dir`, `threads`, `texts`); the answer is one JSON header frame (`ok`, `dims`, `count`, or `error`) and, when `ok`, one frame of little-endian `float32`. The E5 prefixes are applied by the caller, so the process only turns texts into vectors. It loads the model with `local.embed` on the first request, downloads nothing, writes no fact text to disk, and exits after `idle_unload_seconds` without a request.

### Task 3.1: The process

**Files:**
- Create: `src/ingatan/search/shared.py`
- Modify: `src/ingatan/cli/commands.py` (`embeddings serve`, `embeddings stop`)
- Test: `tests/search/test_shared.py`

**Interfaces:**
- Produces:
  - `shared.Unavailable(OSError)`
  - `shared.serve(*, idle_unload_seconds: int) -> None` (blocks until idle or stopped)
  - `shared.status() -> SharedStatus | None` with `SharedStatus(pid: int, port: int)`
  - `shared.stop() -> bool`
  - `shared.request(texts: Sequence[str], *, model_name: str, model_dir: str, threads: int, timeout: float) -> list[list[float]]` (raises `Unavailable`; never starts a process)

- [ ] **Step 1: failing tests** (`local.embed` monkeypatched to a fake; the server runs in a thread of the test):
  - `test_round_trip_returns_vectors_in_order`.
  - `test_wrong_token_is_refused`: a client with another token gets `Unavailable`.
  - `test_binds_loopback_only`: the listener address is `127.0.0.1`.
  - `test_state_file_is_user_only` (POSIX: mode `0o600`) and `test_state_file_holds_no_text`: after a request the file contains none of the request's words.
  - `test_missing_model_file_is_an_error_answer_not_a_download`: the fake raises `FileNotFoundError`; the client gets `Unavailable` carrying "ingatan embeddings pull"; no socket other than loopback was opened.
  - `test_exits_after_idle`: with `idle_unload_seconds=1`, `serve` returns within 3 seconds and removes the state file.
  - `test_second_instance_exits`: a second `serve` in the same home returns at once and the first keeps answering.
- [ ] **Step 2: implement, CHECK, commit.**

### Task 3.2: The setting and the client path

**Files:**
- Modify: `src/ingatan/scope/__init__.py` (`EmbeddingConfig.shared: bool = False`, `INGATAN_EMBED_SHARED`), `src/ingatan/search/shared.py`, `src/ingatan/search/embed.py` (`_embed_local_batch`)
- Test: `tests/scope/test_scope.py`, `tests/search/test_shared.py`

**Interfaces:**
- Produces: `shared.embed(texts, *, model_name, model_dir, threads, idle_unload_seconds, timeout) -> list[list[float]]`: `request`, and on `Unavailable` one attempt to start the process (`subprocess.Popen([sys.executable, "-m", "ingatan.cli", "embeddings", "serve"], ...)`, detached, at most once per 30 seconds per process), a wait of at most `timeout` for the state file, one retry. Still unavailable: raise `Unavailable`.

- [ ] **Step 1: failing tests:**
  - `test_shared_setting_parses` (default false, TOML, environment).
  - `test_embed_uses_the_process_when_shared`: `_embed_local_batch` with `shared=True` calls `shared.embed` and never `local.embed`.
  - `test_spawn_is_rate_limited`: two failing calls in a row start one process.
- [ ] **Step 2: implement, CHECK, commit.**

### Task 3.3: Fallback, doctor, measurement

**Files:**
- Modify: `src/ingatan/cli/doctor.py` (one line for the shared process: running with pid, not running, or unreachable), `scripts/measure_embed_ram.py` (`--sessions N`, `--shared`), `docs/RAM.md` (from the script's output), `docs/EMBEDDINGS.md`, `AGENTS.md` (the semantic-search invariant gains: the shared process is loopback only and downloads nothing), `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md`
- Test: `tests/search/test_shared.py`, `tests/mcp/test_server.py`

- [ ] **Step 1: failing test** (Review Focus 2) `test_recall_answers_from_words_when_the_process_is_unreachable`: `shared=True`, `shared.embed` raises `Unavailable`: `recall` returns the fact found by words, raises nothing, `local.embed` was not called (falling back to an in-process model would silently bring back the 275 MB), and `search.last_embed_error()` names the cause. The same for `remember`.
- [ ] **Step 2: make it pass.** `Unavailable` is an `OSError`, which `embed_text` already turns into "no vector"; the test pins that no other path loads the model.
- [ ] **Step 3: measure.** `uv run python scripts/measure_embed_ram.py --sessions 3` and again with `--shared`; write both totals to `docs/RAM.md` from the script's output.
- [ ] **Step 4: CHECK on all three operating systems in CI, commit. Remove §1 from `docs/ROADMAP.md`.**

**Phase 3 is done when** `scripts/measure_embed_ram.py` shows the total for three sessions close to the total for one, and `docs/RAM.md` records it.

---

## Phase 4: Review queue for similar facts (§4)

What exists: the `conflicts` table, `open_conflicts` and `scan_conflicts` in `cli/conflicts.py`, `facts.confirm` (logged, undoable, resolves the pair when the reason names the other id), `facts.supersede`, `facts.link_facts` (resolves the pair, not logged, reversible with `ingatan unlink`), `facts.forget` (never undoable).

### Task 4.1: Resolve a pair in one call

**Files:**
- Create: `src/ingatan/facts/review.py` (move `OpenConflict` and `open_conflicts` here from `cli/conflicts.py`, which re-imports them; add `newer_id` and `older_id`, by `valid_from`, ties by row order)
- Modify: `src/ingatan/cli/conflicts.py`, `src/ingatan/facts/__init__.py`
- Test: `tests/facts/test_review.py`

**Interfaces:**
- Produces:
  - `review.ACTIONS = ("keep_both", "keep_newer", "keep_older")`
  - `review.open_pairs(ctx: Context) -> list[OpenConflict]`
  - `review.ReviewOutcome(message: str, proposal_id: str | None)` (frozen dataclass)
  - `review.resolve_pair(ctx: Context, pair: OpenConflict, action: str) -> ReviewOutcome`. `facts` sits below `proposals` in the layering, so when `supersede` queues the change (approval mode) the proposal id is returned and Task 4.2 approves it.

- [ ] **Step 1: failing tests:**
  - `test_keep_both_confirms_then_links`: one `confirm` log entry whose reason names the other id, one `related_to` edge, the pair resolved, and `audit.undo` of that entry reopens it.
  - `test_keep_newer_supersedes_the_older`: the older fact is closed with `superseded_by` set to the newer.
  - `test_keep_older_supersedes_the_newer`.
  - `test_approval_mode_returns_the_proposal_id`: with `require_approval=True` on a business fact, `outcome.proposal_id` is set and the old fact is still open.
  - `test_pair_whose_fact_was_closed_meanwhile_is_skipped`: the outcome says so; nothing raises.
  - `test_unknown_action_is_a_validation_error`.
- [ ] **Step 2: implement, CHECK, commit.**

### Task 4.2: `ingatan conflicts review`

**Files:**
- Modify: `src/ingatan/cli/commands.py`
- Create: `src/ingatan/cli/review.py` (rendering one pair and reading one answer)
- Test: `tests/cli/test_review_queue.py`
- Modify: `README.md` (CLI section, both halves), `CHANGELOG.md`, `docs/ROADMAP.md`

**Interfaces:**
- CLI: `ingatan conflicts review [--apply keep_both|keep_newer|keep_older] [--all] [--business NAME] [--agent NAME]`
- Per pair it shows both facts (date, tier, source, full text; "newer" and "older" as labels, no id to type) and asks: `k` keep both and link, `n` newer replaces older, `o` older replaces newer, `f` forget one, `s` skip, `q` quit. A capital `K`, `N` or `O` applies that answer to every remaining pair.

- [ ] **Step 1: failing tests** (driven through `CliRunner` with `input=`):
  - `test_review_resolves_every_pair_without_an_id`: three pairs, input `k\nn\no\n`: `conflicts list` then prints "No open conflicts."
  - `test_capital_answer_applies_to_the_rest`: input `N\n` resolves all three.
  - `test_approval_mode_change_is_applied_at_once`: with `require_approval = true`, answering `n` on a business pair leaves no pending proposal: the command calls `proposals.approve` on the returned id, because the person at the CLI is the approver.
  - `test_forget_asks_which_and_confirms`: `f`, then `older`, then `n` at the confirmation: nothing is deleted. With `y`: the fact is gone and the pair with it.
  - `test_forget_is_never_applied_in_bulk`: there is no `--apply forget`, and no capital `F`.
  - `test_apply_all_needs_no_terminal`: `--apply keep_newer --all` with `input=None` exits 0.
  - `test_review_without_terminal_and_without_apply_fails_clearly`.
- [ ] **Step 2: implement, CHECK, commit. Remove §4 from `docs/ROADMAP.md`.**

**Phase 4 is done when** every open pair can be resolved from the CLI without typing a fact id.

---

## Phase 5: Usage and time decay in `recall` (§3)

Rule from the roadmap: scenarios first; the factor ships only if both new scenarios pass and no existing scenario regresses. If it cannot, Task 5.2 is dropped and §3 is rewritten with what was learned.

### Task 5.1: Scenarios before code

**Files:**
- Modify: `evals/scenarios.py` (category `usage`; recall field `expect_first`: the key that must be result number one), `evals/runner.py` (`_op_recall`), `evals/README.md`
- Create: `evals/scenarios/usage-01-fakta-populer-yang-salah.json`, `evals/scenarios/usage-02-pemakaian-memecah-seri.json`
- Test: `tests/evals/test_scenarios.py`, `tests/evals/test_run.py`

- [ ] **Step 1: failing harness tests**: `expect_first` is validated (a known fact key) and judged (the first hit's id).
- [ ] **Step 2: write `usage-01`**: an old volatile fact is recalled eight times, `advance_days` passes, then a newer fact with the changed value is saved without superseding it; one `recall` must put the newer fact first. It must pass today and after Task 5.2.
- [ ] **Step 3: write `usage-02`**: two stable facts whose texts are built alike, so their word scores tie on the shared query. The older one is recalled five times through a query that matches only it, then `advance_days: 7`; `expect_first` on the shared query is the older one. Mark it `xfail` ("no usage signal in recall yet"): today the tie falls to the newest fact.
- [ ] **Step 4: CHECK** (the report shows one `XFAIL`), **commit.**

### Task 5.2: The bounded factor

**Files:**
- Modify: `src/ingatan/search/core.py`, `evals/scenarios/usage-02-pemakaian-memecah-seri.json` (remove `xfail`), `README.md` (measured numbers), `CHANGELOG.md`, `docs/ROADMAP.md`
- Test: `tests/search/test_ranking.py`

**Interfaces:**
- Produces: `_usage_factor(row: sqlite3.Row, reference: datetime) -> float`, applied next to `_freshness_factor`, skipped for `as_of` recalls. Named constants with the eval numbers in the comment beside them: `USAGE_BOOST` (start at `0.02`), `USAGE_HALF_LIFE_DAYS` (start at `30.0`), `USAGE_COUNT_SATURATION` (start at `5`).

```python
hotness = (1 - 0.5 ** (use_count / USAGE_COUNT_SATURATION)) * 0.5 ** (days_since_last_used / USAGE_HALF_LIFE_DAYS)
factor = 1.0 + USAGE_BOOST * hotness   # a fact never used: exactly 1.0
```

- [ ] **Step 1: failing unit tests:**
  - `test_never_used_fact_is_neutral`: factor `== 1.0`.
  - `test_factor_is_bounded`: for `use_count = 10_000` used today, `factor <= 1.0 + USAGE_BOOST`.
  - `test_usage_boost_stays_below_freshness`: `USAGE_BOOST <= RECENCY_BOOST / 2`, so a new volatile fact still outranks an old popular one of equal relevance.
  - `test_old_popularity_fades`: last used 365 days ago gives a factor below `1.001`.
  - `test_off_topic_fact_is_not_lifted`: a heavily used fact that matches no query term is not returned.
- [ ] **Step 2: implement, then tune** the three constants against `uv run python -m evals` until `usage-01` and `usage-02` pass with no regression. Record the report numbers in the comment.
- [ ] **Step 3: CHECK, sync the README metrics, commit. Remove §3 from `docs/ROADMAP.md`.**

**Phase 5 is done when** the new scenarios are in `evals/scenarios/` and the report shows them passing with the factor on.

---

## Phase 6: LLM fact extraction, optional (§7)

Gate: D6.1, D6.2, D6.3. Findings that shape the design: `proposals.tier` allows only `business` and `global`; `facts.remember` queues a proposal only when `require_approval` is on; `facts.create._source` keeps only `session`, `document` and `episode`; the embedding providers already talk to Ollama and OpenAI-compatible endpoints with stdlib `urllib`, so the `llm` extra needs no package.

### Task 6.1: Configuration and the chat client

**Files:**
- Modify: `src/ingatan/scope/__init__.py` (`LlmConfig(provider="none", model="", api_base="", api_key_env="", timeout=60.0)`, `Scope.llm`, `[llm]` and `INGATAN_LLM_*`; providers `none`, `ollama`, `openai`), `pyproject.toml` (`llm = []`), `AGENTS.md` and `CONTRIBUTING.md` (layering gains `extract` between `ingest` and `mcp`)
- Create: `src/ingatan/extract/__init__.py`, `src/ingatan/extract/llm.py`
- Test: `tests/scope/test_scope.py`, `tests/extract/test_llm.py`

**Interfaces:**
- Produces: `llm.chat(cfg: LlmConfig, system: str, user: str) -> str` (raises `IngatanError`; Ollama `/api/chat`, OpenAI-compatible `/chat/completions`; the key travels in a header).

- [ ] **Step 1: failing tests**: defaults (`provider == "none"`), unknown provider is a `ConfigError`, `chat` with `provider="none"` raises before any socket is opened, request shape per provider (`_http_post_json` faked).
- [ ] **Step 2: implement, CHECK, commit.**

### Task 6.2: Proposals that say where they came from

**Files:**
- Modify: `src/ingatan/facts/create.py` (`remember(..., as_proposal: bool = False)` forces the proposal branch for `business` and `global`; `_source` keeps a redacted `extracted_by`), `src/ingatan/cli/commands.py` (`proposals list` shows `extracted by <provider>/<model>`)
- Test: `tests/facts/test_remember.py`, `tests/cli/test_review_commands.py`

- [ ] **Step 1: failing tests**: `test_as_proposal_queues_even_without_require_approval`, `test_as_proposal_on_agent_tier_is_a_validation_error`, `test_extracted_by_survives_to_the_approved_fact`, `test_proposals_list_marks_extracted`.
- [ ] **Step 2: implement, CHECK, commit.**

### Task 6.3: The extraction pass

**Files:**
- Create: `src/ingatan/extract/prompt.py`; extend `src/ingatan/extract/__init__.py`
- Modify: `src/ingatan/episodes/__init__.py` (`pending(ctx: Context, limit: int) -> list[Episode]`: `offer` cannot be reused, it counts an offer and retires the episode after three), `src/ingatan/cli/commands.py` (`ingatan extract [--limit N] [--dry-run] [--business NAME] [--agent NAME]`)
- Test: `tests/extract/test_extract.py`, `tests/episodes/test_episodes.py`

**Interfaces:**
- Produces: `extract.ExtractReport(episodes_read: int, proposed: int, rejected: int, errors: int)`; `extract.extract_pending(ctx: Context, *, limit: int = 10, dry_run: bool = False) -> ExtractReport`.
- The model must answer `{"facts": [{"text", "keywords", "entities", "kind"}]}`. A fact is rejected when it is not valid, or when it contains a number that does not occur in the episode it came from.
- Each kept fact goes through `facts.remember(..., tier="business", as_proposal=True, episode=id, source={"extracted_by": ...})`, which marks the episode processed. An episode that yields no fact is marked with `episodes.mark_processed`; one whose model call failed stays pending.

- [ ] **Step 1: failing tests** (the model is a fake function):
  - `test_text_is_redacted_before_it_reaches_the_model`: an episode holding a bearer token: the fake receives `[redacted]`.
  - `test_extracted_fact_becomes_a_business_proposal`: nothing in `facts`, one pending proposal with `extracted_by`, the episode marked processed.
  - `test_model_call_is_outside_any_transaction`.
  - `test_broken_json_is_counted_and_skipped` and `test_run_continues_after_a_failing_episode` (Review Focus 5).
  - `test_invented_number_is_rejected`: the episode says "Rp 15.000", the model answers "Rp 18.000": `rejected == 1`, nothing proposed.
  - `test_instructions_inside_an_episode_do_not_write_facts`: whatever the fake returns for an episode that says "ignore the instructions", the result is at most proposals (Review Focus 5).
  - `test_dry_run_writes_nothing`.
  - `test_provider_none_explains_how_to_enable`: exit code 1, no socket.
- [ ] **Step 2: implement, CHECK, commit.**

### Task 6.4: Ingested documents

**Files:**
- Create: the next migration, `src/ingatan/db/migrations/NNNN_note_extraction.sql` (`ALTER TABLE notes ADD COLUMN extracted_at TEXT`)
- Modify: `src/ingatan/extract/__init__.py` (`--documents`: notes with `extracted_at IS NULL`; plain SQL on `notes`, as lower modules do for rows they do not own)
- Test: `tests/extract/test_extract.py`, `tests/db/test_db.py`

- [ ] **Step 1: failing tests**: `test_note_is_extracted_once`, `test_reingested_changed_note_is_extracted_again`, migration applies to an existing database.
- [ ] **Step 2: implement, CHECK, commit.**

### Task 6.5: The measured number

**Files:**
- Create: `evals/extraction/` (cases with episodes, the facts that must be found, and values that must not appear), `evals/extraction/__main__.py`, `evals/reports/latest-extraction.md` (written by the run)
- Modify: `evals/README.md`, `README.md`, `docs/PRD.md` (FR-57 as built), `CHANGELOG.md`, `docs/ROADMAP.md`
- Test: `tests/evals/test_extraction_metrics.py` (the metric code, with a fake model; CI has no model)

- [ ] **Step 1: metric code and its tests**: correct share (an extracted fact matches an expected one), invented share (it matches none), found share (expected facts that were extracted). Judged by required words and values, not by a model.
- [ ] **Step 2 (owner's machine): run** `python -m evals.extraction` with each model chosen under D6.2. The run may open a socket only to the configured endpoint.
- [ ] **Step 3: state the numbers in the README**, with the model names. CHECK, commit. Remove §7 from `docs/ROADMAP.md`.

**Phase 6 is done when** an eval measures how many extracted facts are correct and how many are invented, on scenarios with a known answer, and the README states that number.

---

## Phase 7: Light knowledge graph, optional (§8)

Gate: D7.1, and Phase 6 shipped. What exists: `edges` (business file only, so a relation cannot cross a business), `link_facts`, and `_related_hits` in `search/core.py`, which follows one hop. A gap to close on the way: `remember(related_to=...)` drops its links when the fact becomes a proposal.

### Task 7.1: Relations survive the proposal queue

**Files:** `src/ingatan/facts/create.py` (payload gains `links: [{"to": fact_id, "relation": ...}]`), `src/ingatan/proposals/__init__.py` (`approve` creates each edge whose other end is an open, visible fact; a missing end is skipped and named in the result), `tests/proposals/test_proposals.py`

- [ ] Failing tests: `test_links_are_created_on_approval`, `test_link_to_a_fact_closed_or_forgotten_meanwhile_is_skipped`, `test_rejected_proposal_creates_no_edge`. Implement, CHECK, commit.

### Task 7.2: The extraction pass proposes relations

**Files:** `src/ingatan/extract/prompt.py`, `src/ingatan/extract/__init__.py`, `tests/extract/test_extract.py`

- [ ] The answer format gains `relations: [{"fact": index, "to": existing_fact_id, "relation": "related_to" | "part_of"}]`; the prompt lists the ids and texts of the facts `recall` returns for the episode. Failing tests: `test_relation_to_an_unknown_id_is_dropped`, `test_relation_is_stored_in_the_proposal_payload`. Implement, CHECK, commit.

### Task 7.3: Two hops in `recall`, and the scenario

**Files:** `src/ingatan/search/core.py` (`MAX_HOPS = 2`; `_related_hits` walks breadth-first, one-hop neighbours first, the total still capped by `MAX_RELATED`), `evals/scenarios.py` and `evals/runner.py` (op `link`; recall field `expect_related`), `evals/driver.py`, `evals/scenarios/graph-01-dua-lompatan.json`, `tests/evals/test_run.py`, `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md`

- [ ] Failing tests: `test_second_hop_is_returned_after_first_hops`, `test_related_never_includes_a_closed_or_invisible_fact`, `test_related_is_capped`. The scenario needs a fact two hops away; `tests/evals` runs it with its `link` steps removed and asserts that it then fails. Implement, CHECK, commit. Remove §8 from `docs/ROADMAP.md`.

**Phase 7 is done when** an eval scenario needs a fact two hops away to answer, fails without the relations and passes with them.

---

## Phase 8: Tombstones for wrong values (§5)

The roadmap's own done-when for this item is a design that is written down and agreed before any code. So the phase has one firm task and one that is conditional on it.

### Task 8.1: `docs/RETRACTION.md`

- [ ] **Write the design** and bring it to the owner. It must answer, with the reasoning:
  1. What retracting is: a separate operation on a fact that stays in the file (proposed: `retracted_at` and `retract_reason` columns on `facts`, the fact closed at the same moment, a new logged and undoable op).
  2. How a later `remember` is matched: proposed exact `text_key` plus the same entity set, the test `_find_duplicate` already uses. Anything looser needs a similarity score, and no measured model can tell a repeat from a changed value.
  3. What the caller gets: proposed "not saved: this was recorded before and withdrawn on DATE, because REASON", with a way to record it anyway.
  4. Whether `history` and `recall(include_closed=true)` show a retracted fact, and under which status.
  5. Who may retract: an agent its own facts, the user anything (the rule `forget` follows), and whether it becomes an eighth MCP tool or stays CLI-only.
  6. The proof that `forget` wins: the tombstone is the fact row, so forgetting it removes the tombstone, its bytes and its log content, and the same text can then be saved again.
- [ ] **Owner approves or changes it.** Nothing below starts before that.

### Task 8.2: Build what the design says (outline, final after approval)

- [ ] The next migration; `src/ingatan/facts/retract.py`; the check in `facts.remember` before the duplicate check; undo support in `audit/_undo.py`; the status in `search.status_of` and `facts.history`; `ingatan retract ID --reason TEXT`.
- [ ] Tests: retract then remember the same text is answered, not saved; a different value is saved; undo of a retract reopens the fact; `forget` of a retracted fact leaves no byte (the byte-scan pattern of `tests/facts/test_forget.py`) and the text can be remembered again.
- [ ] Eval: category `retract` and `evals/scenarios/retract-01-nilai-salah-kembali.json`, where an agent reads the same wrong source again.
- [ ] CHECK, commit. Remove §5 from `docs/ROADMAP.md`.

---

## Phase 9: Static embedding model (§6)

Gate D9: start only if the owner, looking at `docs/RAM.md` after Phases 2 and 3, says the memory cost is still too high.

### Task 9.1: Spike (throwaway code, findings in `docs/EMBEDDINGS.md`)

- [ ] Pick one multilingual static model that covers Indonesian. Establish, from its documentation and by running it: the file format of its token vectors, whether it can be tokenized with `sentencepiece` or needs another tokenizer (and what that tokenizer costs in memory: `docs/RAM.md` records 250 MB for one tokenizer library against 35 MB for SentencePiece), whether `numpy` alone can load and pool the vectors, and its resident memory. Stop here and report if any answer makes it larger or heavier than the current int8 model.

### Task 9.2: A second kind of local model

**Files:** `src/ingatan/search/local.py` (`LocalModel.runtime: "onnx" | "static"`; `_load` and `missing_packages` become per-model, so a static model does not ask for `onnxruntime`), `tests/search/test_local_embed.py`, `src/ingatan/cli/doctor.py`

- [ ] Failing tests: `test_static_model_does_not_import_onnxruntime`, `test_static_model_vectors_are_normalized`, `test_unknown_runtime_is_rejected`, `test_pull_verifies_the_static_model_files`. Implement, CHECK, commit.

### Task 9.3: Its own cut-offs and its numbers

**Files:** `src/ingatan/search/embed.py` (a `SimilarityProfile` for the family, selected in `similarity_profile`), `docs/EMBEDDINGS.md`, `docs/RAM.md`, `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md`

- [ ] Run `scripts/calibrate_embeddings.py` for the model; the profile's three values are the script's output, none copied from E5. Run `scripts/measure_embed_ram.py --model NAME` and `python -m evals --embeddings local --model NAME`. It is added as another local model, and becomes the default only if these numbers say so. CHECK, commit. Remove §6 from `docs/ROADMAP.md`.

**Phase 9 is done when** `docs/EMBEDDINGS.md` and `docs/RAM.md` hold its measured numbers next to the current model's.
