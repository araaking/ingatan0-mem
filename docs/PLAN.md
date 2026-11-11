# Ingatan: MVP implementation plan

Status: draft. Scope: the MVP in `DESIGN.md`, in two phases (see "Schedule"). Requirement IDs are from `PRD.md`. Each task is written to become one GitHub issue.

Working rules:

- Tests are written before or with the code. A task is not done until its tests run in CI on Windows, macOS and Linux.
- The eval harness (T-3) comes first among the feature tasks. Its scenarios fail until the tools they measure exist. That is intended: every later task is measured against them.
- Sizes: S is under half a day, M is about one day, L is about two days, for one developer working with an AI coding agent.
- Where a task says "Needs OQ-n", the answer is in `PRD.md` section 12. Only OQ-1 and OQ-2 are still open, and T-2 answers them. Provisional answers are tuned from eval results.

## Proposed repository layout

One line per module. Each module has its own tests and imports only from modules listed above it in this table, so it can become its own set of issues.

| Path | Owns |
|---|---|
| `pyproject.toml`, `LICENSE`, `README.md`, `README.id.md`, `CONTRIBUTING.md` | Packaging, Apache-2.0 licence, docs |
| `.github/workflows/` | CI matrix on Windows, macOS, Linux |
| `src/ingatan/db/` | Connections, WAL, FTS5 tables, numbered migrations |
| `src/ingatan/redact/` | Secret detection and the redaction marker |
| `src/ingatan/scope/` | Config reading, tiers, database file paths, scope filtering |
| `src/ingatan/audit/` | Append-only log, diff, undo |
| `src/ingatan/search/` | BM25, trigram, entity and recency signals, rank fusion, validity filter |
| `src/ingatan/facts/` | remember, supersede, confirm, forget, history, similar facts, conflict flags |
| `src/ingatan/proposals/` | Proposal queue storage, approve, reject |
| `src/ingatan/brief/` | Brief builder and the "check first" section |
| `src/ingatan/episodes/` | Capture storage, truncation, digest offer, processed marks |
| `src/ingatan/ingest/` | Markdown ingest |
| `src/ingatan/mcp/` | MCP stdio server and tool schemas. A thin layer over `facts`, `search`, `brief` |
| `src/ingatan/cli/` | `typer` commands: init, doctor, log, diff, undo, proposals, ingest |
| `src/ingatan/integrations/claude_code/` | Claude Code hook scripts and install notes |
| `src/ingatan/integrations/hermes/` | Hermes memory provider plugin |
| `evals/` | Harness, scenario files (Indonesian), metric code, reports. Not shipped in the package |
| `tests/` | Unit and integration tests, one folder per module |
| `docs/` | `DESIGN.md`, `PRD.md`, `PLAN.md`, setup guides |

## Milestone A: foundations and eval (week 1, first days)

### T-1: Repository scaffold, tooling and CI matrix
Create the package, `uv` project, `ruff`, `pyright` and `pytest` config, Apache-2.0 licence, and a CI workflow that runs them on Windows, macOS and Linux. Add an empty test so CI is green from the first commit.
Satisfies: NFR-5, NFR-10, NFR-17 (CI part)
Depends on: none
Size: S
- [ ] CI is green on all three operating systems.
- [ ] `ruff`, `pyright` and `pytest` run in CI and fail the build on error.
- [ ] `LICENSE` is Apache-2.0 and the layout above exists with empty modules.

### T-2: Spike: confirm Hermes and Claude Code hook behaviour
Read Hermes itself and Claude Code docs, then write down what is true: the Hermes memory provider interface, the `exclusive` rule, whether `on_session_end()` can return work, whether a Claude Code hook can hand work back to the agent, and what `SessionStart` can inject. Record the findings in `docs/`.
Satisfies: none directly. Unblocks FR-44 to FR-49. Answers OQ-1, OQ-2
Depends on: none
Size: S
- [ ] A short findings note exists in `docs/`, with each claim marked verified or not.
- [ ] OQ-1 and OQ-2 have an answer or are still marked open with the reason.
- [ ] T-19 and T-20 are adjusted if a finding contradicts `DESIGN.md`. The owner is told.

### T-3: Eval harness and first scenarios
Build the harness: a scenario file format in Indonesian, a runner that drives the memory through a small interface, and the metric code for recall@5, stale rate, leak rate and tokens per brief. Write 6 to 8 first scenarios (stale fact, supersede, leak across agent, leak across business, inflected word, brief size). Run it in CI with no LLM. Ask the owner for the ten real incidents now.
Satisfies: NFR-3, NFR-6, NFR-8
Depends on: T-1
Size: M
- [ ] `evals/` runs from one command and prints the four metrics.
- [ ] The runner makes no network call and CI runs it with no API key.
- [ ] 6 or more scenarios exist and are marked expected-fail until their tools exist.
- [ ] Metric definitions are written down (OQ-18) and the owner has the incident request.

### T-4: Schema, migrations and connection layer
Create the tables from the data model (facts, episodes, log, proposals) with FTS5 and trigram indexes. Use stdlib `sqlite3` in WAL mode and numbered migrations with `PRAGMA user_version`. Check that Indonesian text round-trips.
Satisfies: NFR-2, NFR-11, NFR-17, FR-18 (schema part)
Depends on: T-1
Size: M
- [ ] A new database is created at the latest version. An old one is migrated. An up-to-date one is untouched.
- [ ] FTS5 and the trigram tokenizer are used, and a missing tokenizer gives a clear error.
- [ ] A test stores and reads Indonesian text with non-ASCII characters on all three operating systems.

### T-5: Scope and config resolution
Read business and agent names from a config file or an environment variable. Open the business database and the global database. Apply tier filtering so every query is limited to the caller's global, business and own agent facts. Needs OQ-3 and OQ-4.
Satisfies: FR-16, FR-18, FR-19, FR-20
Depends on: T-4
Size: M
- [ ] Server scope comes from config or environment, with the documented precedence.
- [ ] A test with two agents proves agent A's `agent` facts are invisible to agent B.
- [ ] A test with two businesses proves no query reaches the other file.

### T-6: Audit log core
Implement the append-only log and a write helper that every write path must use. Entries carry timestamp, agent, operation, fact id and the change (before and after).
Satisfies: FR-35, FR-36
Depends on: T-4
Size: M
- [ ] Every write helper call writes exactly one log entry in the same transaction.
- [ ] No public function updates or deletes an entry, except the scrub used by T-11.
- [ ] A test fails if a write path skips the log.

### T-7: Redaction module
Detect API key, password and token patterns and replace them with a marker. Build a corpus of secret-like strings and of ordinary text that should not be redacted. Needs OQ-14.
Satisfies: FR-32 (module), FR-33 (module), FR-34 (module)
Depends on: T-1
Size: S
- [ ] Every string in the secret corpus is redacted.
- [ ] No string in the ordinary-text corpus is changed, including Indonesian sentences.
- [ ] The marker format is documented.

## Milestone B: search, facts and the seven tools (week 1)

### T-8: Search module
Implement FTS5 BM25 over text, entities and keywords. Add the trigram fallback, the entity-match signal and the recency signal, fused by reciprocal rank. Return only valid facts by default, with source, date and status ("check first", "possible conflict"). Update `use_count` and `last_used` (OQ-6).
Satisfies: FR-6, FR-7, FR-19, FR-27, FR-28, FR-29, FR-30, FR-31
Depends on: T-3, T-4, T-5
Size: L
- [ ] Superseded and not-yet-valid facts are not returned by default.
- [ ] An entity match outranks an equal-BM25 non-match.
- [ ] A partial-word query and an inflected-form query (via keywords) both return the fact, in Indonesian scenarios.
- [ ] The first search scenarios in T-3 change from expected-fail to pass.

### T-9: `remember`, similar facts, conflict flag and tier routing
Implement `remember` with optional entities, keywords, tier, kind and review_after, with the defaults. Always save. Return similar valid facts (OQ-7). Flag the new and similar facts as a possible conflict until resolved (OQ-8). Redact before writing. Route `agent` writes to a valid fact immediately. Business and global routing is added by T-12.
Satisfies: FR-8, FR-9, FR-10, FR-11, FR-22, FR-32, FR-35
Depends on: T-5, T-6, T-7, T-8
Size: L
- [ ] A similar fact never blocks the save.
- [ ] Missing kind is stored as volatile and missing review_after becomes 30 days.
- [ ] Missing keywords returns an error that asks for them and saves nothing.
- [ ] A secret in the input is not present in the facts table or the log.
- [ ] A pair flagged at save time shows "possible conflict" in recall.

### T-10: `supersede`, `confirm`, `history`
Implement the three operations. `supersede` closes the old fact and links it. `confirm` requires a reason, extends `review_after` (OQ-10) and logs the reason. `history` returns entries by fact id or entity.
Satisfies: FR-12, FR-13, FR-14, FR-15, FR-35, FR-60, FR-59 (with T-12)
Depends on: T-9
Size: M
- [ ] `confirm` with a reason that names the other fact clears a "possible conflict" pair.
- [ ] `confirm` on a business or global fact is applied directly and logged.
- [ ] After `supersede`, default recall excludes the old fact and history still shows it.
- [ ] `confirm` with an empty reason fails and changes nothing.
- [ ] After `confirm`, the fact is out of "check first".
- [ ] A missing id returns an error and changes nothing.

### T-11: `forget`
Delete content for good: the fact row, its FTS and trigram entries, and its content in earlier log entries. Leave one content-free log entry. Make freed SQLite pages unreadable (for example by secure deletion or a vacuum). Needs OQ-17.
Satisfies: FR-41, FR-42, FR-43, FR-35, FR-61 (MCP part)
Depends on: T-6, T-9
Size: M
- [ ] `forget` through MCP on a business or global fact returns an error and changes nothing.
- [ ] A test stores a unique string, forgets it and scans the database file and its WAL file. The string is not found in either.
- [ ] `recall`, `brief` and `history` return no content for the id. `history` returns the content-free entry.
- [ ] The remaining log entry has id, time, agent and operation only.

### T-12: Proposal queue
Route `remember` with tier `business` or `global` to a pending proposal. Pending proposals are not recalled and not in the brief, and they are counted. Provide approve and reject functions for T-25. Needs OQ-9.
Satisfies: FR-21, FR-23, FR-35, FR-59 (with T-10)
Depends on: T-6, T-9
Size: M
- [ ] `supersede` on a business or global fact queues the new fact and shows the old one as "check first". Approval closes the old fact. Rejection clears the mark.
- [ ] A business-tier `remember` returns a proposal id and creates no valid fact.
- [ ] `recall` and `brief` never show the pending text.
- [ ] Approve and reject functions exist, write log entries and are covered by tests.

### T-13: Brief builder
Build `brief()`: sections in the order global, business, agent, each ordered by use. Put volatile facts past `review_after` in a separate "check first" section. Show the pending proposal count. Keep to about 1,000 tokens (OQ-11).
Satisfies: FR-2, FR-3, FR-4, FR-5
Depends on: T-8, T-9, T-12
Size: M
- [ ] A fact past its review date appears only in "check first".
- [ ] Section order and use ordering are tested.
- [ ] The brief for the eval corpus is about 1,000 tokens and the harness reports the count.
- [ ] The pending count is correct, including zero.

### T-14: MCP stdio server
Expose exactly the seven tools over stdio with the official `mcp` SDK, with `pydantic` schemas and English tool descriptions. No schema has a business or agent field. Connect it to Claude Code on the author's laptop and use it.
Satisfies: FR-1, FR-17, NFR-13, NFR-3
Depends on: T-10, T-11, T-13
Size: M
- [ ] A test lists the tools and finds exactly seven, by name.
- [ ] A test walks every input schema and finds no business or agent field.
- [ ] Claude Code connects and calls `brief`, `remember` and `recall` successfully.
- [ ] The harness from T-3 now runs through this server and the earlier scenarios pass or fail on real behaviour.

### T-15: CLI skeleton and `init`
Add the `typer` entry point and `ingatan init`, which creates the project config, the business database and the global database. It is safe to run twice.
Satisfies: FR-51
Depends on: T-5
Size: S
- [ ] `init` in an empty folder creates the config and both databases.
- [ ] A second `init` changes no file and exits 0.
- [ ] Tests run on all three operating systems.

### T-16: Measure RAM per stdio process
Write a script that starts one server process, measures resident memory idle and after loading the eval corpus, and writes the numbers to `docs/`. Run it on the VPS and on Windows.
Satisfies: NFR-4
Depends on: T-14
Size: S
- [ ] Numbers for the VPS and for Windows are committed, with the command that produced them.
- [ ] OQ-19 is answered by the owner or marked as measure-only.

### T-17: Concurrency test
Start two server processes on the same business file and write from both in a loop. Check that no write is lost and the database is not corrupted. Fix lock handling if needed.
Satisfies: NFR-12
Depends on: T-14
Size: S
- [ ] The test writes at least a few hundred facts from two processes and counts all of them.
- [ ] A database integrity check passes after the run.
- [ ] The test runs in CI on all three operating systems.

## Milestone C: capture, digest and integrations (week 2, except T-22 in weeks 3 to 4)

### T-18: Episodes and capture entry point
Create the shared capture code used by both integrations: write an episode from a user message and a final reply, truncate it (OQ-13), redact it (T-7), and never raise into the caller. Expose an internal entry point the hooks can call.
Satisfies: FR-33, FR-46, NFR-14
Depends on: T-4, T-7, T-2
Size: M
- [ ] An overlong input is stored truncated at the limit.
- [ ] A secret in the input is not in the stored episode.
- [ ] With a locked or bad database the entry point returns without raising and logs the failure.

### T-19: Claude Code hooks (capture and inject)
Write the hook scripts and install notes. Capture records user messages and final replies only. Inject puts the brief into the session at start. Shape depends on T-2 (which hook, how often it fires).
Satisfies: FR-44, FR-47, NFR-14
Depends on: T-2, T-14, T-18
Size: M
- [ ] A real Claude Code session records episodes with no tool output and no file contents.
- [ ] A new session starts with the brief in context without a `brief()` call.
- [ ] A failing hook does not interrupt the session.

### T-20: Hermes plugin (capture and inject)
Write the Hermes memory provider plugin: `sync_turn` for capture, `system_prompt_block()` or `prefetch()` for injection. Verify on the VPS. Shape depends on T-2 and OQ-1.
Satisfies: FR-45, FR-48, NFR-13 (Hermes), NFR-14
Depends on: T-2, T-14, T-18
Size: L
- [ ] A real Hermes session records episodes with no tool output and no file contents.
- [ ] A new session starts with the brief in context.
- [ ] A failing plugin call does not interrupt the session.
- [ ] The Hermes agent calls the seven MCP tools through the same server as Claude Code.

### T-21: Digest offer and processed marks
At session start, with no API key, offer the agent at most three pending episodes with an instruction to call `remember`. Mark an episode processed after it is digested. Needs OQ-12.
Satisfies: FR-49, FR-50
Depends on: T-13, T-18
Size: M
- [ ] With ten pending episodes, the session-start output offers three.
- [ ] A digested episode is not offered in the next session.
- [ ] No network call is made.

### T-22: Markdown ingest
Add `ingatan ingest <path>` for a file or a folder. Redact, record the file path as `source`, log each item, and skip unchanged content on a second run. Needs OQ-15.
Satisfies: FR-34, FR-53, FR-54, FR-55, FR-62, FR-35
Depends on: T-8, T-9, T-18
Size: M
- [ ] Each heading section becomes one document note. `recall` returns it labelled as a document excerpt, and `brief` never shows it.
- [ ] Re-ingesting a changed file replaces that file's notes.
- [ ] Ingesting a folder of Markdown files creates items with the file path as source.
- [ ] A second run on unchanged files creates nothing.
- [ ] A secret in a file does not reach the database.

## Milestone D: remaining CLI (weeks 3 to 4, except T-25 in week 2)

### T-23: CLI `log` and `diff`
List log entries, newest first. Show the before and after of an entry. Needs OQ-16.
Satisfies: FR-37, FR-38
Depends on: T-6, T-15
Size: M
- [ ] `log` lists entries newest first.
- [ ] `diff` on an entry shows before and after text.
- [ ] Output for a forgotten fact contains no content.

### T-24: CLI `undo`
Reverse a log entry, write a new log entry, and refuse for `forget`. Needs OQ-16.
Satisfies: FR-39, FR-40, FR-35
Depends on: T-11, T-23
Size: M
- [ ] Undo of a create, a supersede and a confirm restores the prior state, as tested.
- [ ] Undo writes a new log entry and does not edit the old one.
- [ ] Undo of a `forget` entry fails with the documented message and changes nothing.

### T-25: CLI proposal review
List, approve and reject pending proposals. Forget a business or global fact on the user's command.
Satisfies: FR-24, FR-25, FR-26, FR-35, FR-61 (CLI part)
Depends on: T-12, T-15
Size: S
- [ ] List shows id, tier, text, agent and date.
- [ ] Approve makes the fact valid and recallable, with the writing agent as source.
- [ ] Reject creates no fact. Both actions are logged.
- [ ] The CLI forgets a business or global fact, with the same guarantees as T-11.

### T-26: CLI `doctor`
Check Python version, FTS5, trigram tokenizer, resolved config and that each database opens at the current schema version. Exit non-zero on any failure.
Satisfies: FR-52
Depends on: T-4, T-5, T-15
Size: S
- [ ] A healthy setup prints all checks as pass and exits 0.
- [ ] A test with a missing tokenizer, a broken config or an old schema fails the right check and exits non-zero.

## Milestone E: measure and release (weeks 3 to 4)

### T-27: Complete the eval set
Grow the set to 30 to 50 Indonesian scenarios. About ten come from the author's real incidents. The rest follow those patterns. Include stale-fact, supersede, leak, inflected-word and brief-size cases.
Satisfies: NFR-7, NFR-8
Depends on: T-3, T-14. Needs the author's incidents.
Size: M
- [ ] Between 30 and 50 scenarios exist and run in CI.
- [ ] About ten are marked as from real incidents.
- [ ] The first full numbers for recall@5, stale rate, leak rate and tokens per brief are recorded.

### T-28: Conflict-resolution report
Compute from the audit log the share of conflict candidates the agent resolved, by the method chosen in OQ-18.
Satisfies: NFR-9
Depends on: T-9, T-10, T-14
Size: S
- [ ] A command or script prints the share and the counts it used.
- [ ] A test on a prepared log gives the expected share.

### T-29: Documentation
Write the English README and the Indonesian README with only claims backed by eval numbers. Write CONTRIBUTING (module map, how to run tests and evals, how to pick an issue) and setup notes for Claude Code and Hermes.
Satisfies: NFR-16, NFR-15
Depends on: T-19, T-20, T-27
Size: M
- [ ] A person with a fresh machine can follow the README to a working setup.
- [ ] Every number in the README appears in an eval report.
- [ ] CONTRIBUTING lists each module from the layout with its tests.
- [ ] The docs credit claude-mem for the hook mapping used as a reference.

### T-30: PyPI release and clean-install test
Publish `ingatan` to PyPI. Add a CI job that installs from the built package on each operating system with the one-command method (OQ-20) and runs `init` and `doctor`.
Satisfies: NFR-1, NFR-15
Depends on: T-26, T-29
Size: S
- [ ] The package is on PyPI with an Apache-2.0 licence field.
- [ ] The clean-install job passes on Windows, macOS and Linux.
- [ ] No Docker or database server is installed in that job.

### T-31: Dogfood and decide
Use Ingatan on Hermes and Claude Code for real work. Run the eval, tokens per brief, RAM and the conflict-resolution report. Compare with the targets. If fewer than half of the conflict candidates were resolved, record that API-key digest is now the recommended mode.
Satisfies: none new. Verifies the definition of done.
Depends on: T-27, T-28, T-30
Size: S
- [ ] Measured numbers and the date are written to `docs/`.
- [ ] Each metric is marked as met or not met against its first-guess target.
- [ ] The failure-line decision is written down.

## Definition of done for the MVP

The MVP is done, at the end of weeks 3 to 4, when all of these hold. Targets are first guesses with no data behind them yet.

- [ ] Every MVP requirement in the traceability table has its tasks done and tests passing in CI on Windows, macOS and Linux.
- [ ] Leak rate is 0 on the full eval set.
- [ ] Stale rate is under 5%.
- [ ] recall@5 is over 80%.
- [ ] Tokens per brief are about 1,000 and reported.
- [ ] The share of conflict candidates the agent resolved is reported. If it is under half, the recommendation switch is written down.
- [ ] RAM per stdio process is measured and published.
- [ ] The eval has 30 to 50 Indonesian scenarios, about ten from real incidents, and runs on every commit with no LLM.
- [ ] `ingatan` is on PyPI and a clean install passes on three operating systems.
- [ ] The author has used it on Hermes and Claude Code.

If a metric misses its target, the MVP can still ship. The miss and the reason are written down, and the README states the real number.

## Schedule

The owner agreed on 2026-11-11 to narrow the first two weeks to a build the owner can use every day, and to move the rest to weeks 3 to 4.

| Phase | Tasks | Goal |
|---|---|---|
| Weeks 1 to 2 | T-1 to T-21, T-25 | Daily use on Claude Code and Hermes: the seven tools, scope, proposal review, capture, inject, digest |
| Weeks 3 to 4 | T-22, T-23, T-24, T-26 to T-31 | Markdown ingest, `log`, `diff`, `undo`, full `doctor`, complete eval set, docs, PyPI release, measured dogfooding |

On 2026-11-11 the owner deferred the Hermes plugin (T-20) and the Hermes part of T-2. Hermes uses Ingatan through MCP only until the owner supplies the Hermes version or source. T-20 keeps its place in the plan but does not block the first phase.

The milestone headings below keep their original grouping. This table decides the phase when the two disagree.

## Schedule risk

By the sizes in this plan, weeks 1 to 2 hold about 21 working days of tasks and weeks 3 to 4 about 7. The sizes are estimates and may be too cautious, but the first phase is still about twice what two weeks hold. Expect it to take three to four weeks. The riskiest items are T-8 and T-9 (search and conflict handling are the core), T-20 (Hermes is unchecked) and T-27 (waits for the owner's incidents).

If the first phase must be cut further, cut in this order:

1. T-17 concurrency test, moved to weeks 3 to 4. One agent at a time is enough to start.
2. T-16 RAM measurement on Windows. Keep the VPS number.
3. T-21 digest offer. Capture still stores episodes, so nothing is lost.

Do not cut: the eval harness (T-3), scope (T-5), the audit log (T-6), redaction (T-7), forget (T-11), or the Hermes plugin (T-20). Hermes is the owner's main agent, and the others carry the claims the project makes.

Three things can slip the plan with no coding delay: the ten incidents, access to Hermes source or docs for T-2, and the VPS for T-16 and T-20.

## After the MVP

- Vector search (`ingatan[vector]`, sqlite-vec) as an extra rank in the fusion (FR-56).
- Server-side LLM digest in batches with a cheap model (`ingatan[llm]`, FR-57). Becomes the recommended mode if the failure line is hit.
- HTTP mode for sharing across machines (FR-58).
- LongMemEval subset, with cost reported (NFR-18).
- Code graph integration (`ingatan[graph]`) and a UI.
- Revisit write-then-undo if the proposal queue goes unreviewed.
- Revisit the targets with real data.

## Traceability

Every FR and NFR tagged W1 or W2 in `PRD.md` appears below. Post-MVP requirements (FR-56 to FR-58, NFR-18) have no MVP task.

| Requirement | Tasks |
|---|---|
| FR-1 | T-14 |
| FR-2 | T-13 |
| FR-3 | T-13 |
| FR-4 | T-13 |
| FR-5 | T-13 |
| FR-6 | T-8 |
| FR-7 | T-8 |
| FR-8 | T-9 |
| FR-9 | T-9 |
| FR-10 | T-9 |
| FR-11 | T-9 |
| FR-12 | T-10 |
| FR-13 | T-10 |
| FR-14 | T-10 |
| FR-15 | T-10 |
| FR-16 | T-5 |
| FR-17 | T-14 |
| FR-18 | T-4, T-5 |
| FR-19 | T-5, T-8 |
| FR-20 | T-5 |
| FR-21 | T-12 |
| FR-22 | T-9 |
| FR-23 | T-12 |
| FR-24 | T-25 |
| FR-25 | T-25 |
| FR-26 | T-25 |
| FR-27 | T-8 |
| FR-28 | T-8 |
| FR-29 | T-8 |
| FR-30 | T-8 |
| FR-31 | T-8 |
| FR-32 | T-7, T-9 |
| FR-33 | T-7, T-18 |
| FR-34 | T-7, T-22 |
| FR-35 | T-6, T-9, T-10, T-11, T-12, T-22, T-24, T-25 |
| FR-36 | T-6 |
| FR-37 | T-23 |
| FR-38 | T-23 |
| FR-39 | T-24 |
| FR-40 | T-24 |
| FR-41 | T-11 |
| FR-42 | T-11 |
| FR-43 | T-11 |
| FR-44 | T-19 |
| FR-45 | T-20 |
| FR-46 | T-18 |
| FR-47 | T-19 |
| FR-48 | T-20 |
| FR-49 | T-21 |
| FR-50 | T-21 |
| FR-51 | T-15 |
| FR-52 | T-26 |
| FR-53 | T-22 |
| FR-54 | T-22 |
| FR-55 | T-22 |
| NFR-1 | T-30 |
| NFR-2 | T-4 |
| NFR-3 | T-3, T-14 |
| NFR-4 | T-16 |
| NFR-5 | T-1 |
| NFR-6 | T-3 |
| NFR-7 | T-27 |
| NFR-8 | T-3, T-27 |
| NFR-9 | T-28 |
| NFR-10 | T-1 |
| NFR-11 | T-4 |
| NFR-12 | T-17 |
| NFR-13 | T-14, T-20 |
| NFR-14 | T-18, T-19, T-20 |
| NFR-15 | T-29, T-30 |
| NFR-16 | T-29 |
| NFR-17 | T-1, T-4 |
| FR-59 | T-10, T-12 |
| FR-60 | T-10 |
| FR-61 | T-11, T-25 |
| FR-62 | T-22 |
