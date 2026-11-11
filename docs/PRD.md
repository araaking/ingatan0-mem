# Ingatan: product requirements

Status: draft for the MVP. Source of truth: `DESIGN.md` (27 agreed decisions). This document restates those decisions as testable requirements. Where the design was silent, the gap and the owner's later decision are in section 12.

Tags used on every requirement:

- **W1**: MVP week 1.
- **W2**: MVP week 2. W1 and W2 together are the build the owner uses every day.
- **W3**: weeks 3 to 4. Release work: ingest, the audit CLI, the full eval set, docs and PyPI.
- **Post**: after the MVP.

## 1. Summary

Ingatan is a lightweight shared memory for AI agents. It is a local MCP server over stdio, backed by one SQLite file per business plus one file for facts about the user. It needs no LLM, no embedding model, no API key, no Docker and no Postgres.

Three things set it apart, according to the design:

- Each fact can carry a review date. A fact past its date is shown as doubtful, in a "check first" section, and not as truth.
- The server needs no LLM and no embedding model. The calling agent does the extraction, the keywords and the conflict decisions.
- One SQLite file per business, with scope (which business, which agent) fixed by the server at start-up. An agent picks a tier and never a business name.

Capture across agents and the audit log (read, diff, undo) are supporting features. Other projects already have forms of both.

The MVP is for the author's own use with Hermes Agent and Claude Code. The repository is public from day one. It is built so strangers can install it with one command and contribute to it.

## 2. Problem

The author runs several small businesses and uses several AI agents daily: a general assistant, an SEO agent, a writer agent and coding agents. Four problems follow.

1. **Agents forget between sessions.** The author re-explains the same context every time.
2. **Outdated facts are still used.** When something changes, the old version is not marked outdated, so agents keep acting on it.
3. **Memory must be separate and sometimes shared.** Each business needs its own memory. Each agent needs private notes. Some knowledge is shared across agents, and a little is shared across businesses (facts about the user).
4. **Work context is scattered.** It lives in chats, documents, notes and code.

Constraints that shape the solution:

- The author's VPS has 7.7 GB RAM and is nearly full. Many target users run laptops or small VPSes.
- There is no local LLM and no local embedding model. If an LLM is used at all, it is an API with few calls.
- Stored content is often Indonesian. Exact-word search misses inflected Indonesian forms. This is a known risk.
- Development is on Windows. CI must cover Windows, macOS and Linux.

## 3. Target users

| User | Needs | Served by |
|---|---|---|
| Owner-operator (the author, first) | One memory across several agents and several businesses, without re-explaining, and with outdated facts flagged | All MVP requirements |
| Solo developer or small-team lead on a laptop or small VPS | Install in one command, no database server, low RAM | NFR-1, NFR-2, NFR-4 |
| Indonesian-language user | Search that finds inflected forms | FR-29, FR-30 |
| Agent (Claude Code, Hermes Agent, any MCP client) | A small tool set, a short session-start brief, clear status on each fact | FR-1 to FR-15 |
| Open source contributor | Separate modules, tests, a harness that shows whether a change helped | NFR-5 to NFR-8, NFR-16 |

The long-term goal is enough adoption to qualify for an open source support programme (20 or more external contributors, or 200,000 or more monthly downloads). This is a direction. No MVP requirement depends on it.

## 4. Goals and non-goals

### Goals

- G1. An agent starts a session with a short brief that holds the context it needs.
- G2. A fact that may have changed is shown as doubtful, not as truth.
- G3. Memory is separated by business and by agent. Sharing is explicit and approved by the user.
- G4. Every change is auditable and reversible by a human, except an explicit forget.
- G5. Every feature works with no API key.
- G6. Claims in the README follow eval numbers.

### Non-goals

- No LLM or embedding model on the server in the MVP.
- No cross-business search, by design.
- No code graph. codebase-memory-mcp runs alongside as its own server.
- No promotion or marketing in the MVP. The repository is public and quiet.

## 5. Principles

From `DESIGN.md`:

1. **Cheap and free first.** Every feature must work with no API key. A key may make a feature more accurate, never make it possible.
2. **Knows when to doubt.** A fact that may have changed is shown as doubtful.
3. **Auditable.** Every change is logged. A human can read, diff and undo it.
4. **Measured.** README claims follow eval numbers.

## 6. User stories

| ID | Story | Requirements |
|---|---|---|
| US-1 | As the owner, I want my agent to start a session with a brief of what matters, so that I do not re-explain. | FR-2, FR-3, FR-47, FR-48 |
| US-2 | As the owner, I want a fact I told one agent to be available to my other agents in the same business, once I approve it. | FR-21, FR-23, FR-24 |
| US-3 | As the owner, I want a fact that was true last month to be shown as "check first", so that agents ask me or verify before using it. | FR-5, FR-7 |
| US-4 | As an agent, when I learn that a fact changed, I want to close the old one and link the new one, so that the old one is never used as current. | FR-12 |
| US-5 | As an agent, when I save a fact that resembles an existing one, I want to be shown the existing one, so that I can resolve the overlap. | FR-9, FR-11 |
| US-6 | As the owner, I want my agents to keep notes private to each agent within a business. | FR-19, FR-22 |
| US-7 | As the owner, I want a business's memory never to appear in another business. | FR-18, FR-20 |
| US-8 | As the owner, I want to see what changed, compare it and undo a mistake. | FR-35 to FR-40 |
| US-9 | As the owner, I want something I told an agent by mistake (a password, a private detail) deleted for good. | FR-32, FR-41 to FR-43 |
| US-10 | As the owner, I want what I said and what the agent answered to be captured even if the agent forgets to call `remember()`. | FR-44 to FR-46, FR-49 |
| US-11 | As the owner, I want to load existing Markdown notes into memory. | FR-53 to FR-55 |
| US-12 | As an Indonesian-language user, I want a search for one form of a word to find a fact written with another form. | FR-29, FR-30 |
| US-13 | As a new user, I want to install and set up with one command and a health check. | NFR-1, FR-51, FR-52 |

## 7. Functional requirements

Each requirement has one behaviour and one observable result. "Valid fact" means a fact whose `valid_from` is not in the future, whose `valid_to` is empty or in the future, and which has no `superseded_by`.

### 7.1 MCP tools

| ID | Requirement | Tag |
|---|---|---|
| FR-1 | The stdio MCP server lists exactly seven tools: `brief`, `recall`, `remember`, `supersede`, `confirm`, `forget`, `history`. No other tool is listed. | W1 |
| FR-2 | `brief()` returns facts in this section order: global, business, agent. Within each section facts are ordered by use (see FR-31). | W1 |
| FR-3 | `brief()` output never exceeds the configured token budget. The default budget is 1,000 tokens and the config file can change it. Facts that do not fit stay reachable through `recall`. The harness reports the measured token count of every brief it builds. | W1 |
| FR-4 | `brief()` states the number of pending proposals. | W1 |
| FR-5 | `brief()` lists every `volatile` valid fact whose `review_after` is in the past in a separate "check first" section, and does not list it in the global, business or agent sections. The section carries an instruction: do not act on these facts before checking them, and check only those relevant to the current task. A `stable` fact is never placed there. | W1 |
| FR-6 | `recall(query)` returns only valid facts by default. Each result has id, text, source, date and status. | W1 |
| FR-7 | In `recall` results, a fact past its `review_after` has status "check first". A fact in an unresolved pair (FR-11) has status "possible conflict". | W1 |
| FR-8 | `remember(fact)` always saves the fact. It never refuses a fact because a similar one exists. | W1 |
| FR-9 | `remember(fact)` returns the ids and text of existing valid facts judged similar. | W1 |
| FR-10 | `remember` requires `keywords` (synonyms and root words) and returns an error that asks for them if they are missing. It accepts optional `entities`, `tier`, `kind` and `review_after`. If `kind` is missing it is stored as `volatile`. If `review_after` is missing on a `volatile` fact it is set to 30 days after `valid_from`. | W1 |
| FR-11 | A new fact and each similar fact returned for it by `remember` are shown as "possible conflict" in `recall` and `brief` until one of the pair is closed by `supersede` or the pair is resolved in the way chosen in OQ-8. | W1 |
| FR-12 | `supersede(old, new)` sets the old fact's `valid_to` to now and `superseded_by` to the new fact's id. The old fact is no longer returned by default `recall` and is still returned by `history`. If either id does not exist, it returns an error and changes nothing. | W1 |
| FR-13 | `confirm(id, reason)` moves the fact's `review_after` to a later date and the fact leaves the "check first" section. The extension length is decided in OQ-10. | W1 |
| FR-14 | `confirm` with a missing or empty reason returns an error and changes nothing. A valid call writes the reason to the log. | W1 |
| FR-15 | `history(id or entity)` returns the change entries for that fact id, or for all facts linked to that entity, in time order. | W1 |

### 7.2 Scope and proposal queue

| ID | Requirement | Tag |
|---|---|---|
| FR-16 | At start-up the server reads business name and agent name from a config file in the project folder or from an environment variable. Precedence between the two is decided in OQ-3. | W1 |
| FR-17 | No tool input schema contains a field for business name or agent name. `remember` takes a `tier` and nothing else about scope. | W1 |
| FR-18 | Storage is one SQLite file per business plus one file for the `global` tier. The business file holds the `business` and `agent` tiers. | W1 |
| FR-19 | Facts in the `agent` tier written by agent A are never returned to agent B in the same business by `brief`, `recall` or `history`. | W1 |
| FR-20 | A server started for business X never returns a fact from business Y, and no tool searches across businesses. | W1 |
| FR-21 | `remember` with `tier` set to `business` or `global` creates a pending proposal and returns the proposal id. It does not create a valid fact. | W1 |
| FR-22 | `remember` with `tier` set to `agent` creates a valid fact immediately. | W1 |
| FR-23 | A pending proposal is never returned by `recall`, and is not listed in any `brief` section. `brief` counts it (FR-4). | W1 |
| FR-24 | The CLI approves a pending proposal. The fact becomes valid in the requested tier, keeps the writing agent as `source`, and the approval is logged. | W2 |
| FR-25 | The CLI rejects a pending proposal. No fact is created, and the rejection is logged. | W2 |
| FR-26 | The CLI lists pending proposals with id, tier, text, writing agent and date. | W2 |
| FR-59 | `supersede` on a `business` or `global` fact puts the new fact in the proposal queue and shows the old fact as "check first" at once. The old fact is closed only when the proposal is approved. If it is rejected, the old fact leaves "check first". | W1 |
| FR-60 | `confirm` on a `business` or `global` fact is applied directly and logged. | W1 |
| FR-61 | `forget` through MCP on a `business` or `global` fact returns an error and changes nothing. The CLI forgets such a fact when the user runs it. | W2 |

### 7.3 Search

| ID | Requirement | Tag |
|---|---|---|
| FR-27 | `recall` ranks matches with FTS5 BM25 over `text`, `entities` and `keywords`. | W1 |
| FR-28 | `recall` fuses BM25 rank, entity-match rank and recency rank by reciprocal rank fusion. A fact that matches the query's entity ranks above an equal-BM25 fact that does not. | W1 |
| FR-29 | A query that is a partial word of a stored word returns the fact, through the trigram tokenizer. | W1 |
| FR-30 | A fact stored with agent-supplied `keywords` containing a root word is returned for a query that uses a different inflected form of that word. At least five Indonesian scenarios in the eval set cover this. | W1 |
| FR-31 | When `recall` returns a fact, its `use_count` rises by one and `last_used` is set to now. The exact trigger is confirmed in OQ-6. | W1 |

### 7.4 Redaction of secrets

| ID | Requirement | Tag |
|---|---|---|
| FR-32 | `remember` replaces text that looks like an API key, password or token with a redaction marker before writing. Neither the facts table nor the log contains the original string. | W1 |
| FR-33 | Capture (FR-44, FR-45) applies the same redaction before an episode is written. | W2 |
| FR-34 | Markdown ingest (FR-53) applies the same redaction before anything is written. | W3 |

### 7.5 Audit log, diff and undo

| ID | Requirement | Tag |
|---|---|---|
| FR-35 | Every create, supersede, confirm, forget, proposal approval, proposal rejection and ingest writes one log entry with timestamp, writing agent, operation, fact id and the change. | W1 |
| FR-36 | No MCP tool and no CLI command edits or deletes a log entry. The only change to an existing entry is the content removal in FR-41. | W1 |
| FR-37 | `ingatan log` lists log entries, newest first. | W3 |
| FR-38 | `ingatan diff` shows the before and after of a log entry. What a range or a fact-level diff means is decided in OQ-16. | W3 |
| FR-39 | `ingatan undo <log entry>` reverses that change, restores the previous state and writes a new log entry. Which operations are undoable, and what happens when later changes depend on the entry, is decided in OQ-16. | W3 |
| FR-40 | `ingatan undo` on a `forget` entry fails with a message that a forget cannot be undone, and changes nothing. | W3 |

### 7.6 Forget

| ID | Requirement | Tag |
|---|---|---|
| FR-41 | After `forget(id)`, the fact's text, entities, keywords and source content are not found anywhere in the database file or its FTS index, including earlier log entries about that fact. A byte scan of the file for the original text finds nothing. | W1 |
| FR-42 | `forget(id)` leaves one log entry with fact id, timestamp, writing agent and operation, and no content. | W1 |
| FR-43 | After `forget(id)`, `recall`, `brief` and `history` return no content for that id. `history` returns only the content-free log entry. | W1 |

### 7.7 Capture and digest

| ID | Requirement | Tag |
|---|---|---|
| FR-44 | The Claude Code hook records the user's messages and the agent's final replies as episodes. It records no tool output and no file contents. | W2 |
| FR-45 | The Hermes plugin records each turn (user message and assistant reply) as an episode through `sync_turn`. It records no tool output and no file contents. | W2 |
| FR-46 | An episode longer than the truncation limit is truncated before storage. The limit is decided in OQ-13. | W2 |
| FR-47 | In Claude Code, the brief is injected at session start without the agent calling `brief()`. | W2 |
| FR-48 | In Hermes, the brief is injected through `system_prompt_block()` or `prefetch()`. | W2 |
| FR-49 | With no API key, the session-start output offers the agent at most three pending episodes to digest into facts, with an instruction to call `remember`. How the offer is delivered and what "three" counts is decided in OQ-12. | W2 |
| FR-50 | An episode that has been offered and digested is marked processed and is not offered again. | W2 |

### 7.8 CLI

| ID | Requirement | Tag |
|---|---|---|
| FR-51 | `ingatan init` creates the project config, the business database and the global database if they do not exist. A second run changes nothing. | W1 |
| FR-52 | `ingatan doctor` checks the Python version, FTS5, the trigram tokenizer, the resolved config, and that each database opens at the current schema version. It prints pass or fail per check and exits non-zero if any check fails. | W3 |

The remaining CLI commands (`log`, `diff`, `undo`, proposal review) are FR-24 to FR-26 and FR-37 to FR-40.

### 7.9 Markdown ingest

| ID | Requirement | Tag |
|---|---|---|
| FR-53 | `ingatan ingest <path>` reads one Markdown file, or every `.md` file under a folder. What it produces (episodes, facts, proposals) is decided in OQ-15. | W3 |
| FR-54 | Running `ingatan ingest` twice on unchanged files creates no new items the second time. | W3 |
| FR-55 | Each ingested item records the file path as `source` and is logged. | W3 |
| FR-62 | Ingest stores one document note per heading section. `recall` returns matching notes labelled as document excerpts, separate from facts. Notes never appear in `brief`. Re-ingesting a changed file replaces that file's notes. | W3 |

### 7.10 After the MVP

| ID | Requirement | Tag |
|---|---|---|
| FR-56 | With the `ingatan[vector]` extra installed, `recall` adds a vector rank (sqlite-vec) to the fusion. Without it, `recall` behaves as in FR-27 to FR-29. | Post |
| FR-57 | With the `ingatan[llm]` extra and an API key, a cheap model digests pending episodes in batches. Without a key FR-49 still applies. | Post |
| FR-58 | An HTTP mode lets several machines share one memory. | Post |

## 8. Non-functional requirements

| ID | Requirement | Tag |
|---|---|---|
| NFR-1 | A clean machine with Python 3.10+ installs Ingatan and starts the server with one command. No Docker, Postgres or other service is needed. A clean-install test passes on Windows, macOS and Linux. | W3 |
| NFR-2 | Storage uses only stdlib `sqlite3` with FTS5, in WAL mode. | W1 |
| NFR-3 | With no API key and no network access, every MVP requirement passes its test. The server makes no outbound network call. | W1 |
| NFR-4 | RAM (resident set size) of one stdio server process is measured on the 7.7 GB VPS and on Windows, idle and after the eval corpus is loaded. The numbers are written to the repository. The threshold is decided in OQ-19. | W1 |
| NFR-5 | CI runs lint, type check, tests and eval on Windows, macOS and Linux for every commit. A red job on any of them blocks merge. | W1 |
| NFR-6 | The eval harness runs in CI on every commit with no LLM call. | W1 |
| NFR-7 | The eval set holds 30 to 50 Indonesian scenarios. About 10 come from real incidents supplied by the author. The rest follow those patterns. | W3 |
| NFR-8 | Every eval run reports recall@5, stale rate, leak rate and tokens per brief. | W1 |
| NFR-9 | A report states the share of conflict candidates the agent resolved, computed from the audit log. The method is decided in OQ-18. | W3 |
| NFR-10 | The code base is Python 3.10+ and passes `ruff`, `pyright` and `pytest` in CI. | W1 |
| NFR-11 | Schema changes are numbered SQL migrations tracked with `PRAGMA user_version`. Opening a database at an older version applies the pending migrations, and opening an up-to-date database applies none. | W1 |
| NFR-12 | Two server processes writing to the same business file at the same time lose no write and corrupt nothing. A test proves it. | W1 |
| NFR-13 | The server works with Claude Code over standard MCP stdio with no agent-specific server code. Hermes is connected the same way in W2. | W1 |
| NFR-14 | If a capture hook or plugin fails (database locked, bad input, timeout), the agent session continues and the failure is logged. | W2 |
| NFR-15 | The repository carries an Apache-2.0 licence. The package is published to PyPI as `ingatan`. | W3 |
| NFR-16 | The repository has an English README, an Indonesian README, a CONTRIBUTING guide, and setup notes for Claude Code and Hermes. The docs credit claude-mem for the per-agent hook mapping used as a reference. | W3 |
| NFR-17 | Indonesian text, with accents and non-ASCII punctuation, round-trips through store, search and CLI output unchanged on Windows, macOS and Linux. | W1 |
| NFR-18 | After the API mode exists, a LongMemEval subset (knowledge-update and temporal-reasoning) is run and reported together with cost: LLM calls, RAM and time. | Post |

## 9. Success metrics

All targets below are first guesses. There is no data behind them yet. They will be revisited after the first eval run and after dogfooding.

| Metric | Target | Notes |
|---|---|---|
| Leak rate | 0 | Results from a scope the caller may not see. Any non-zero value is a bug. |
| Stale rate | under 5% | Definition in OQ-18. |
| recall@5 | over 80% | Over the 30 to 50 Indonesian scenarios. |
| Tokens per brief | about 1,000 | From the `brief()` design. The design lists it as a metric without a separate target. |
| Share of conflict candidates the agent resolved | No target. Failure line below. | Measured from real use, see OQ-18. |

Failure line: if agents resolve fewer than half of the conflict candidates, digest with an API key becomes the recommended mode. Zero-key mode stays supported.

Also reported with every accuracy number once the API mode exists: LLM calls, RAM, time (NFR-18).

Long-term direction, not an MVP metric: 20 or more external contributors, or 200,000 or more monthly downloads.

## 10. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R-1 | **The biggest risk.** The zero-LLM approach depends on agents reliably calling `remember()` and resolving the conflicts it returns. If they do not, facts are missing and outdated facts stay valid. | Hooks capture episodes as a safety net (FR-44, FR-45). Episodes are digested at the next session start (FR-49). Unresolved pairs stay visible as "possible conflict" (FR-11). The resolved share is measured (NFR-9). The failure line in section 9 switches the recommended mode to API digest. |
| R-2 | Hermes and Claude Code hooks may not behave as assumed. Hermes `on_session_end()` returns nothing. The Hermes plugin was declared `exclusive` in another project, which suggests one memory provider at a time. A Claude Code `Stop` hook fires every turn. None of this is checked against the agents themselves. | Task T-2 checks them first. Digest at the start of the next session is the default because it does not depend on session-end hooks. |
| R-3 | Indonesian search misses inflected forms. | Agent-supplied keywords, trigram fallback (FR-29, FR-30), Indonesian scenarios as a first-class eval case. |
| R-4 | The proposal queue goes unreviewed, so business and global facts never take effect. | `brief` shows the pending count (FR-4). If the queue goes unreviewed in practice, switch to write-then-undo, as the design says. |
| R-5 | Memory use on the nearly full 7.7 GB VPS. | RAM per stdio process is measured in week 1 (NFR-4). |
| R-6 | A secret is stored because redaction misses its format. | Redaction at every write path (FR-32 to FR-34), with a corpus test. Pattern list in OQ-14. |
| R-7 | `forget` leaves content in free SQLite pages, the FTS index or earlier log entries. | FR-41 requires a byte scan to find nothing. |
| R-8 | Memory leaks across scopes, for example an agent asks for another business or another agent's notes. | Scope is fixed by the server and not a tool parameter (FR-16, FR-17). Leak rate target is 0. |
| R-9 | SQLite builds without FTS5 or the trigram tokenizer on some Python distributions. | `doctor` checks both (FR-52). CI runs on three operating systems. |
| R-10 | Several agents write to one file at once. | WAL mode and a concurrency test (NFR-2, NFR-12). |
| R-11 | The targets in section 9 are guesses and may be wrong. | Stated as guesses. Revisit after first data. |
| R-12 | The plan is larger than the time first set for it. | The first two weeks are narrowed to daily use, and release work moves to weeks 3 to 4. PLAN.md has the schedule, the risk and a cut order. |
| R-13 | The eval set depends on ten real incidents that only the author can supply. | Ask for them in week 1. Fill the rest with pattern-based scenarios. |

## 11. Out of scope for the MVP

- Vector search (`ingatan[vector]`), server-side LLM digest (`ingatan[llm]`), the graph extra (`ingatan[graph]`).
- HTTP mode and sharing across machines. Memory is separate per machine in the MVP.
- Code graph. UI.
- Migration from the author's current memory server. Ingatan starts empty.
- Cross-business search. Not planned at all.
- LongMemEval runs (after the API mode exists).
- Promotion and announcements.

## 12. Decisions on open questions

These were gaps in `DESIGN.md`. The owner decided them on 2026-11-11. Decisions marked "provisional" are implementation defaults to be tuned from eval results. Where a requirement above says "decided in OQ-n", this table is the answer. Two questions are still open because they are facts to check, not choices.

| # | Question | Decision | Status |
|---|---|---|---|
| OQ-1 | Can Ingatan run next to another memory provider in Hermes? | Not known. Check against Hermes itself in T-2. | Open |
| OQ-2 | Can a Claude Code hook hand work back to the agent? | Not known. Check in T-2. Digest at session start does not depend on it. | Open |
| OQ-3 | Config file and precedence | `.ingatan.toml` in the project folder. Environment variables `INGATAN_BUSINESS` and `INGATAN_AGENT` win when both are set. | Provisional |
| OQ-4 | Database location | All database files live in the user data folder (`~/.ingatan/`), one per business plus `global`. Nothing is stored in the project folder, so memory cannot be committed to git by accident. | Provisional |
| OQ-5 | Default `tier` | `agent`, the narrowest. | Provisional |
| OQ-6 | When `use_count` and `last_used` change | When `recall` returns the fact. Appearing in a brief does not count. | Provisional |
| OQ-7 | What "similar" means in `remember` | The top three FTS matches among valid facts that share at least one entity with the new fact. | Provisional |
| OQ-8 | How a "possible conflict" pair ends | Two ways. `supersede` when one replaces the other. `confirm` with a reason that names the other fact when both are true. No eighth tool. | Decided |
| OQ-9 | Changes to `business` and `global` facts | `supersede`: the new fact goes to the proposal queue and the old fact is shown as "check first" at once (FR-59). `confirm`: applied directly and logged (FR-60). `forget`: only through the CLI by the user (FR-61). An agent does not see pending proposals, its own included. | Decided |
| OQ-10 | How far `confirm` moves `review_after` | By the fact's original interval, counted from now. | Provisional |
| OQ-11 | Brief token counting and overflow | Tokens are estimated as characters divided by four. Over budget, the least used facts are dropped first. The "check first" section is kept. | Provisional |
| OQ-12 | Digest offer | Delivered inside the `brief` output. "At most three" counts episodes. | Provisional |
| OQ-13 | Episode truncation and retention | 2,000 characters per message. Digested episodes are kept for 30 days, then deleted. | Provisional |
| OQ-14 | Redaction | A fixed pattern list: common key prefixes, JWTs, and `password=`-style assignments. Marker `[redacted]`. No allow-list in the MVP. | Provisional |
| OQ-15 | What Markdown ingest produces | One document note per heading section. Notes are searchable through `recall` and labelled as document excerpts, not facts (FR-62). They skip the proposal queue because the user ran the command. A changed file replaces its notes. | Decided |
| OQ-16 | Undo and diff | `undo` works on create, supersede, confirm and proposal approval. It refuses if the fact changed after that entry. `diff` shows one log entry. | Provisional |
| OQ-17 | Forgetting a fact that superseded another | The older fact stays closed and its link points to the content-free entry. Backups the user made earlier are outside Ingatan's control, and the docs say so. | Provisional |
| OQ-18 | Metric definitions | recall@5: share of scenarios whose expected fact is in the top five results. Stale rate: share of queries that return a closed or past-review fact as current. Leak rate: share of results from outside the caller's scope. Conflict share: computed from the audit log of real sessions. | Provisional |
| OQ-19 | RAM and latency limits | Measure only. No threshold until the first numbers exist. | Provisional |
| OQ-20 | Install command | `uvx ingatan`. | Provisional |
| OQ-21 | When the full eval set is due | Weeks 3 to 4, with T-27. The first scenarios are due in week 1. | Decided |
