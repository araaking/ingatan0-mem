# Ingatan design

Status: agreed design, no code yet. Last updated 2026-11-11.

Ingatan is a shared memory for AI agents: one SQLite file per business, facts that
know when to be doubted, and no LLM or embedding model required on the server.

## Principles

1. **Cheap and free first.** Every feature must work with no API key. A key may
   make a feature more accurate, never make it possible.
2. **Knows when to doubt.** A fact that may have changed is shown as doubtful, not
   as truth.
3. **Auditable.** Every change is logged. A human can read, diff and undo it.
4. **Measured.** Claims in the README follow eval numbers.

## What sets it apart

Checked against the source of mem0 2.2.1, Hindsight 1.2.1, Graphiti 0.31.2 and
claude-mem 15.16.1 (read only, nothing was run):

| | Ingatan | Others |
|---|---|---|
| Per-fact review date | Yes | Not found in any of the four |
| LLM and embeddings on the server | Not required | Required by default in mem0, Hindsight, Graphiti |
| Storage | One SQLite file per business | Qdrant (mem0), Postgres (Hindsight), Neo4j (Graphiti) |
| Scope fixed by the server | Yes | Caller-supplied filter (`user_id`, `group_id`, `bank_id`) |

Not differentiators: cross-agent capture (claude-mem already has adapters for
many agents) and audit history (mem0 and Hindsight have forms of it).
claude-mem is the closest project on the lightweight side. It has no validity
over time, and its hook capture uses an observer LLM.

## Data model

A **fact** has:

| Field | Purpose |
|---|---|
| `text`, `entities`, `keywords` | Content. The agent supplies entities and keywords (synonyms, root words) |
| `tier` | `global`, `business` or `agent` |
| `kind` | `stable` or `volatile`. Missing means `volatile` |
| `valid_from`, `valid_to`, `superseded_by` | Validity over time. Old facts are never overwritten |
| `review_after` | For `volatile` facts. Default 30 days, settable per fact |
| `source` | Session, document, date and writing agent |
| `use_count`, `last_used` | Ranking signal |

Other tables: `episodes` (raw capture), `log` (append-only change log),
`proposals` (pending writes to a wider tier).

## Scope

- Three tiers: `global` (about the user), `business` (shared by all agents in one
  business), `agent` (private to one agent in one business).
- The business and agent name are fixed when the server starts, from a config
  file in the project folder or an environment variable. An agent picks a tier,
  never a business name.
- One SQLite file per business, plus one for `global`. There is no cross-business
  search, by design.
- Writes to `business` and `global` go to a proposal queue and take effect after
  the user approves them in the CLI. `brief` shows the pending count. If the
  queue goes unreviewed in practice, switch to write-then-undo.

## MCP tools

| Tool | Behaviour |
|---|---|
| `brief()` | Session-start summary, about 1,000 tokens. Order: global, business, agent, each by use. Facts past `review_after` appear in a separate "check first" section |
| `recall(query)` | Valid facts with source, date and status. The agent expands the query itself |
| `remember(fact)` | Always saves. Returns similar facts for the agent to resolve. Unresolved pairs are flagged "possible conflict" |
| `supersede(old, new)` | Closes the old fact and links it to the new one |
| `confirm(id, reason)` | Extends `review_after`. A reason is required and logged |
| `forget(id)` | Permanently deletes the content. A log entry without content remains. Cannot be undone |
| `history(id or entity)` | Change history |

CLI for humans: `init`, `log`, `diff`, `undo`, `doctor`, proposal review.

## Capture and digest

- Capture through hooks is part of the MVP. `remember()` is the fast path, hooks
  are the safety net.
- Hooks record only the user's messages and the agent's final replies, truncated.
  No tool output, no file contents.
- Text that looks like an API key, password or token is replaced with a redaction
  marker, in both `remember()` and capture.
- With no API key, the agent digests pending episodes into facts at the start of
  the next session, at most three per session. With a key, a cheap model digests
  them in batches.

Hook findings so far:

- **Hermes** has a memory provider interface (seen in claude-mem's Hermes
  plugin): `sync_turn(user, assistant)` per turn fits capture,
  `system_prompt_block()` and `prefetch()` fit injection. `on_session_end()`
  returns nothing, so it cannot ask the agent to digest. That plugin is declared
  `kind: exclusive`, which suggests one memory provider at a time. Not yet
  checked against Hermes itself.
- **Claude Code** has a `Stop` hook, but it fires after every turn, not once per
  session. Not yet checked whether it can hand work back to the agent.

Given both, digest at the start of the next session is the default for the MVP.

## Search

FTS5 with BM25, entity match and recency, fused by reciprocal rank. Only valid
facts by default. A trigram tokenizer backs up partial-word matches. Indonesian
text is a first-class eval case, since exact-word search misses inflected forms.
Vector search is an optional extra after the MVP.

## Tech stack

- Python 3.10+, stdlib `sqlite3` with FTS5 in WAL mode, the official `mcp` SDK,
  `pydantic`, `typer`. Numbered SQL migrations with `PRAGMA user_version`.
- Tooling: `uv`, `ruff`, `pytest`, `pyright`, GitHub Actions on Windows, macOS
  and Linux.
- Extras after the MVP: `ingatan[vector]` (sqlite-vec), `ingatan[llm]` (httpx to
  an OpenAI-compatible endpoint), `ingatan[graph]` (graphify).
- Code graph is out of scope. codebase-memory-mcp runs alongside as its own
  server.
- Licence Apache-2.0. Package name `ingatan` (unused on PyPI as of 2026-11-11).
  Code, tool descriptions and main docs in English, with an Indonesian README.

## MVP

For the author's own use on Hermes and Claude Code. Public repo, no promotion.
The schedule was revised on 2026-11-11 after the plan was sized: the first two
weeks are narrowed to a build the author can use every day.

- **Weeks 1 to 2:** schema, the seven MCP tools over stdio, FTS5 and validity,
  scope, `init`, eval harness with the first scenarios, RAM per stdio process
  measured, Claude Code hooks and Hermes plugin (capture and inject), digest at
  session start, proposal review in the CLI. By the plan's own sizes this is
  about 21 working days, so expect three to four weeks.
- **Weeks 3 to 4:** Markdown ingest, `log`, `diff`, `undo`, full `doctor`, the
  complete eval set, docs, PyPI release.
- **Later:** vector search, server-side LLM digest, HTTP mode for sharing across
  machines, code graph, UI.

Detail decisions made after this document (conflict resolution, changes to
shared facts, what ingest produces, and implementation defaults) are in
`PRD.md` section 12. Tasks are in `PLAN.md`.

Memory is separate per machine during the MVP. Start empty, no migration from the
current memory server.

## Evaluation

- 30 to 50 scenarios in Indonesian. About 10 come from real incidents supplied by
  the author, the rest follow those patterns.
- Metrics: recall@5, stale rate, leak rate, tokens per brief, share of conflict
  candidates the agent resolved.
- Targets (first guesses, no data behind them yet): leak rate 0, stale rate under
  5%, recall@5 over 80%.
- Failure line: if agents resolve fewer than half of the conflict candidates,
  digest with an API key becomes the recommended mode. Zero-key stays supported.
- Runs on every commit with no LLM. A LongMemEval subset (knowledge-update,
  temporal-reasoning) follows once the API mode exists. Report accuracy together
  with cost: LLM calls, RAM, time.

## Open items

- Confirm the Hermes provider interface and the `exclusive` rule against Hermes
  itself, and whether a Claude Code hook can hand work back to the agent.
- RAM per stdio process on the 7.7 GB server.
- Ten real incidents for the eval set.
- The references Letta, cognee, supermemory, Memori, MemOS and EverOS were only
  skimmed or not checked.
