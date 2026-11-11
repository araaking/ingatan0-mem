# Contributing to Ingatan

Read [docs/PRD.md](docs/PRD.md) for behaviour, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for
names, signatures and storage, and [docs/PLAN.md](docs/PLAN.md) for the task list. When PRD and
code disagree, PRD wins on behaviour and ARCHITECTURE on names. Say so in your pull request if you
deviate.

Project principles (docs/DESIGN.md): cheap and free first, knows when to doubt, auditable, and
claims follow eval numbers. A change that moves an eval number must update the README only with
numbers copied from `evals/reports/latest.md`.

## Setup

```
uv sync
uv run python -m pytest -q          # tests
uv run ruff check .                 # lint
uv run ruff format --check .        # format (drop --check to rewrite)
uv run python -m pyright            # types
uv run python -m evals              # eval scenarios, rewrites evals/reports/latest.md
```

`python -m pytest` and `python -m pyright` are the forms CI uses. They also work on machines where
the `pytest.exe` or `pyright.exe` launcher in `.venv` is blocked by a security policy. Useful
variations: `-k name`, `-x`, one folder (`tests/search`), and `python -m evals --scenario ID
--verbose` (a filtered eval run does not overwrite the report).

All four checks plus the evals must pass before you open a pull request. CI runs them on Windows,
macOS and Linux for Python 3.10 and 3.13. The code must stay Python 3.10 compatible and start every
file with `from __future__ import annotations` (ruff enforces it). Every test gets an isolated data
folder through `INGATAN_HOME` (autouse fixture in `tests/conftest.py`); never touch the real
`~/.ingatan`.

## The layering rule

A module imports only from modules earlier in this list:

`timeutil`, `errors`, `db`, `redact`, `scope`, `audit`, `search`, `facts`, `proposals`, `episodes`,
`brief`, `ingest`, `mcp`, `cli`, `integrations`

The reason is that each module can then be understood, tested and changed without the ones above
it. When a lower module needs something a higher one owns (for example `audit.undo` has to change
fact rows, `facts` has to mark an episode processed), write plain SQL in the lower module instead
of importing upward. `mcp` and `cli` stay thin: no business logic that a test of `facts`,
`search` or `brief` could not reach.

## Module map

| Module | Owns | Tests |
|---|---|---|
| `src/ingatan/timeutil.py` | `now()`, ISO formatting and parsing, the freezable clock | `tests/timeutil/` |
| `src/ingatan/errors.py` | `IngatanError` and its subclasses (user-facing messages) | `tests/errors/` |
| `src/ingatan/db/` | Connections and pragmas, WAL retry on a fresh file, `transaction`, numbered migrations in `db/migrations/*.sql`, capability check | `tests/db/`, `tests/concurrency/` |
| `src/ingatan/redact/` | Secret patterns and the `[redacted]` marker | `tests/redact/` (corpus files `corpus_secrets.txt`, `corpus_plain.txt`) |
| `src/ingatan/scope/` | Config and environment resolution, tier visibility SQL, `Context`, `find_fact` | `tests/scope/`, `tests/facts/test_isolation.py` |
| `src/ingatan/audit/` | Append-only log, `entries`, `diff`, `undo` (`_undo.py`) | `tests/audit/` |
| `src/ingatan/search/` | BM25, trigram, entity and recency signals, rank fusion, validity, statuses, `stem_id.py` query expansion | `tests/search/` |
| `src/ingatan/facts/` | `remember`, `supersede`, `confirm`, `forget`, `history`, similar facts, conflict pairs | `tests/facts/` |
| `src/ingatan/proposals/` | Proposal queue: pending, approve, reject | `tests/proposals/` |
| `src/ingatan/episodes/` | Capture storage, truncation, offers, processed marks, retention | `tests/episodes/` |
| `src/ingatan/brief/` | Brief builder, "check first" section, token budget | `tests/brief/` |
| `src/ingatan/ingest/` | Markdown ingest into notes | `tests/ingest/` |
| `src/ingatan/mcp/` | MCP stdio server (`MCPServer`) and the seven tool schemas | `tests/mcp/` |
| `src/ingatan/cli/` | `typer` commands (`commands.py`), `doctor.py`, `conflicts.py`, `render.py` | `tests/cli/` |
| `src/ingatan/integrations/claude_code/` | `ingatan hook <event>` handlers | `tests/integrations/test_claude_code_hooks.py` |
| `src/ingatan/integrations/hermes/` | Hermes memory-provider plugin and `plugin.yaml` | `tests/integrations/test_hermes_plugin.py` |
| `evals/` | Harness (`runner.py`, `driver.py`, `metrics.py`, `report.py`, `netguard.py`), scenarios, reports. Not shipped in the wheel | `tests/evals/` |
| `scripts/measure_ram.py` | RAM measurement, appends to `docs/RAM.md` | none (run by hand) |
| `.github/workflows/` | `ci.yml` (checks and clean-install job), `release.yml` (PyPI on a `v*` tag) | the workflows themselves |
| `docs/` | Design, PRD, plan, architecture, setup guides, findings | none |

Write tests next to the behaviour you change, in the folder for that module. Cross-cutting tests
live where the rule lives: `tests/facts/test_write_paths_are_logged.py` guards the audit rule and
`tests/facts/test_isolation.py` the scope rule.

## Add an eval scenario

Scenarios are JSON files in `evals/scenarios/`, written in natural Indonesian. The format, the
steps and the checks are in [evals/README.md](evals/README.md).

1. Copy a scenario of the same category and change it. Name it `<category>-NN-short-topic.json`
   (the `id` equals the file name).
2. Write the facts and queries the way a person would, with the keywords a careful agent would
   write (root words, synonyms). Never copy the query into the keywords to make it pass. If a
   scenario is meant to depend on keywords or on the trigram index, set `"needs"`; `tests/evals`
   checks that it really does.
3. Run `python -m evals --scenario <id> --verbose`, then the whole set.
4. If it fails because of real product behaviour, keep it and add an `"xfail"` reason. Do not weaken
   it. A scenario that fails the leak check always fails the run.
5. Run `python -m evals` without a filter and commit the new `evals/reports/latest.md`, then update
   any README number that changed.

## Add an incident scenario

An incident scenario comes from something that really happened to the project owner or a user. It
must not be invented; none exist yet (see `evals/scenarios/incidents/README.md`).

1. Collect what the README in that folder asks for: what went wrong, the stored facts with their
   tier, business and agent, what changed in between, and the exact query.
2. Remove secrets and customer names first. Redaction does not run on scenario files.
3. Write `evals/scenarios/incidents/<category>-incident-NN-topic.json` with `"source": "incident"`,
   keeping the original wording and mix of languages.
4. A failing incident scenario stays in the set with an `xfail` reason.

The README may only call numbers "from real incidents" once these exist.

## Add a migration

Schema changes are numbered SQL files, applied in order above `PRAGMA user_version`.

1. Add `src/ingatan/db/migrations/NNNN_short_name.sql` with the next number. The numbers must be
   continuous (`0002`, `0003`, ...); `load_migrations` rejects gaps. `LATEST_VERSION` follows from
   the file count.
2. Never edit a migration that has shipped. Add a new one.
3. The file runs inside one `BEGIN IMMEDIATE` transaction on both file kinds (business and global),
   so write it for both. Triggers keep their inner semicolons; `migrate` splits on complete
   statements.
4. If the change touches full-text tables, keep the FTS5 `secure-delete` option for them
   (`forget` relies on it), and check `tests/facts/test_forget.py`.
5. Add a test in `tests/db/` that opens a database at the previous version and checks the upgrade.
6. Update the schema in `docs/ARCHITECTURE.md`. The wheel includes `db/migrations/*.sql`
   automatically (see `pyproject.toml`).

## Pick an issue

Tasks are `T-1` to `T-31` in [docs/PLAN.md](docs/PLAN.md). Each lists what it satisfies (PRD ids),
what it depends on, a size (S under half a day, M about a day, L about two days) and checkboxes
that define done. Open items with no task yet, such as real incidents for the eval set, are listed
in the README "Status" section.

Good starting points: the remaining "not verified" items in `docs/setup-claude-code.md` and
`docs/setup-hermes.md` (running the integration for real and reporting what happened), T-27 (real
incident scenarios, needs the owner), T-30 (the first PyPI release, needs the owner), and T-31
(dogfood notes). Comment on the issue before starting so two people do not do the same task.

## Style

Code, comments and messages in English, comments only where the reason is not obvious. Text files
are UTF-8, JSON is written with `ensure_ascii=False`, and CLI output must not crash on Windows
consoles. Keep commit and pull-request descriptions factual: what changed, what you ran, what you
could not check.
