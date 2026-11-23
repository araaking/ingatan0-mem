# Roadmap

Status: planned, unless an item says otherwise. This page records what was agreed as worth
building next, why, and the constraints each item has to respect. An item becomes a
changelog entry when it ships and is removed from this page. The task-level plan for all of
it is in [ROADMAP_PLAN.md](ROADMAP_PLAN.md).

Every item is bound by the invariants in [AGENTS.md](../AGENTS.md). The ones that shape
this roadmap most:

- Every core feature works offline with no API key. Anything that needs a model or a
  network call is optional and off by default.
- One SQLite file per business plus `global.db`. No second index file, no cross-business
  search.
- `forget` removes the bytes. Nothing added here may keep a trace of a forgotten fact.
- No model call inside a write transaction, and the server never downloads a model.
- A claim in `README.md` is backed by a number in `evals/reports/` or `docs/RAM.md`.

## Suggested order

| Order | Item | Why here |
|---|---|---|
| 1 | [Easier installation](#9-easier-installation) | Built; only the release to PyPI is left, and that is the owner's to do |
| 2 | [Deferred embedding](#2-deferred-embedding-on-write) | Built; the timing still has to be measured on a machine that has the model |
| 3 | [Shared model process](#1-shared-model-process) | Built; the saving still has to be measured on a machine that has the model |
| 4 | [Review queue for similar facts](#4-review-queue-for-similar-facts) | Mostly a CLI over data that already exists |
| 5 | [Usage and decay in `recall`](#3-usage-and-time-decay-in-recall) | Needs eval scenarios before any code |
| 6 | [LLM fact extraction](#7-llm-fact-extraction-optional) | The largest item; needs design decisions first |
| 7 | [Light knowledge graph](#8-light-knowledge-graph-optional) | Reuses the extraction pass |
| 8 | [Tombstones](#5-tombstones-for-wrong-values) | Needs a design that cannot collide with `forget` |
| 9 | [Static embedding model](#6-static-embedding-model) | Only if 275 MB is still too much after items 2 and 3 |

The numbers in the headings below are the ones the items were first discussed under; the
table above is the order to build them in.

## 1. Shared model process

**Status.** Built: `[embeddings] shared = true` (see the changelog and
[EMBEDDINGS.md](EMBEDDINGS.md)). The two open decisions were settled as loopback TCP with a
token that is proved, never sent, and a process that the first server starts by itself
(`ingatan embeddings serve` runs it by hand). It is off by default.

**What is left.** The done-when is a measurement that needs the model on disk:
`python scripts/measure_embed_ram.py --sessions 3` and the same with `--shared`, with both
totals recorded in [RAM.md](RAM.md). After that, decide whether `shared` becomes the default
for the local provider. This section goes when the numbers are in.

**Done when.** `scripts/measure_embed_ram.py` shows the total for three sessions close to
the total for one, and `docs/RAM.md` records it.

## 2. Deferred embedding on write

**Status.** Built: `[embeddings] defer = true` (see the changelog and
[EMBEDDINGS.md](EMBEDDINGS.md)). Every fact gets its vector without a manual `reindex`, and
a conflict found late is visible in `recall` and `ingatan conflicts`; both are covered by
tests with a stand-in model.

**What is left.** The first part of the done-when is a measurement, and it needs the model
on disk: run `scripts/bench_scale.py` three ways (no embeddings, `--embeddings local`,
`--embeddings local --defer`) and `python -m evals --embeddings local --defer`, and record
the results in EMBEDDINGS.md. This section goes when those numbers are in.

**Done when.** `remember` with the local provider takes about as long as with
`provider = "none"` (`scripts/bench_scale.py`), every fact gets its vector without a manual
`reindex`, and a conflict found late is visible in `recall` and `ingatan conflicts`.

## 3. Usage and time decay in `recall`

**Problem.** `use_count` orders the brief but plays no part in `recall`.

**Idea.** A bounded factor on the fused score from how often and how recently a fact was
used, in the same style as the freshness factor for volatile facts (a few percent, never
enough to lift an off-topic fact). OpenViking scores "hotness" this way.

**Risk.** Facts that are recalled often get recalled more, and a new, correct fact loses
to an old popular one. `recall` itself raises `use_count`, which feeds the loop.

**Constraints.** Scenarios come first: one where the popular fact is the wrong answer, one
where usage rightly breaks a tie. The factor ships only if both pass and no existing
scenario regresses. The bound and the half-life are named constants with the eval numbers
recorded next to them.

**Done when.** The new scenarios are in `evals/scenarios/` and the report shows them
passing with the factor on.

## 4. Review queue for similar facts

**Problem.** A fact that resembles an earlier one is reported to the agent in the result of
`remember`. If the agent does nothing, the pair stays `possible conflict` until someone
notices.

**Idea.** A CLI pass over the open pairs that shows both facts and asks what to do: keep
both and link them, let one supersede the other, or forget one. With a way to apply one
answer to several pairs. claude-mem sends near-duplicates to a review queue in the same
spirit.

**What exists already.** The `conflicts` table, `ingatan conflicts list` and
`ingatan conflicts scan`, linking through `edges`, and `supersede`.

**Constraints.** Every action goes through the existing functions, so it is logged and can
be undone. Forgetting keeps its confirmation.

**Done when.** Every open pair can be resolved from the CLI without typing a fact id.

## 5. Tombstones for wrong values

**Problem.** A value that was recorded and turned out to be wrong can be recorded again by
an agent that reads the same wrong source.

**Idea.** Mark a fact as retracted, so that a later `remember` of the same statement is
answered with "this was recorded before and withdrawn" instead of being saved silently.
Uteke keeps tombstones for wrong values.

**Constraint that decides the design.** `forget` scrubs bytes and leaves no trace, so a
tombstone can never be how a forgotten fact is remembered. Retracting has to be a separate
operation on a fact that stays in the file, and a forgotten fact takes its tombstone with
it.

**Open decisions.** Whether the match is the exact normalized text (the `text_key` index)
or something looser, and whether a retracted fact is still shown by `history`.

**Done when.** The design is written down and agreed before any code; the eval then needs
a scenario where the wrong value comes back.

## 6. Static embedding model

**Idea.** A static token-vector model (the model2vec family) needs no ONNX Runtime and is
several times smaller than the current default. codebase-memory-mcp uses the same kind of
runtime-free vectors.

**Why it is last.** Its quality is below the E5 model that is the default today, and items
1 and 2 may make the memory question go away.

**Constraints.** It gets its own `SimilarityProfile` from
`scripts/calibrate_embeddings.py`; no cut-off is carried over from another model. It is
added as another local model, not as a new default, unless the calibration numbers say it
should be.

**Done when.** `docs/EMBEDDINGS.md` and `docs/RAM.md` hold its measured numbers next to
the current model's.

## 7. LLM fact extraction (optional)

**Problem.** Ingatan remembers only what an agent chooses to pass to `remember`. A long
conversation or a document holds facts nobody wrote down.

**Idea.** An optional extra (`ingatan[llm]`, off by default) that reads episodes and
ingested documents and proposes facts from them. This is the server-side digest already
foreseen as FR-57 in [PRD.md](PRD.md).

**Constraints.**
- The provider is configurable and can be local: an Ollama model or any
  OpenAI-compatible endpoint, the same choice the embedding providers offer. Nothing is
  sent anywhere unless the user configured an endpoint.
- A model can invent things. Extracted facts carry a source that marks them as extracted,
  and by default they go to the proposal queue instead of becoming valid facts.
- It runs as its own command or background step, never inside `remember` and never inside
  a write transaction.
- Redaction applies before any text is sent to a model.

**Open decisions.** Whether extracted facts may ever be written directly (reversible with
`ingatan undo`) or always need approval, and which model sizes are worth supporting.

**Done when.** An eval measures how many extracted facts are correct and how many are
invented, on scenarios with a known answer, and the README states that number.

## 8. Light knowledge graph (optional)

**Idea.** No graph database. The extraction pass of item 7 also proposes relations between
facts, stored in the existing `edges` table (`related_to`, `part_of`), and `recall` follows
them for one or two hops.

**What exists already.** `edges`, `ingatan link` and `unlink`, and `recall` returning facts
one hop away.

**Why not a full graph.** Entity and relation extraction on every write, plus a graph
engine, is what graphiti and cognee do. It needs a model for every `remember` and would
make the offline core depend on one.

**Constraints.** A proposed relation is reviewed like a proposed fact. A relation never
crosses a business file.

**Done when.** An eval scenario needs a fact two hops away to answer, fails without the
relations and passes with them.

## 9. Easier installation

**Status.** Built, except the release itself. `ingatan setup`, `ingatan embeddings enable`,
the three-command start in the README, a rehearsal path to TestPyPI and a tag guard in the
release workflow are in (see the changelog). CI runs `setup` and a remember and recall round
trip from the built wheel on Windows, macOS and Linux.

**What is left, all of it the owner's.**
1. Check that the package name is free on PyPI and on TestPyPI. If it is not, choose a
   distribution name; the import name and the command stay `ingatan`.
2. Register the trusted publishers (this repository, workflow `release.yml`, environments
   `pypi` and `testpypi`) and create both GitHub environments.
3. Rehearse on TestPyPI with a pre-release number (Actions, Release, Run workflow), because
   a published version cannot be replaced.
4. Push the release tag. Then take "does not work yet" out of the README and move the
   install from a checkout to its Development section.

**Done when.** A new user goes from nothing to a working `recall` in a client with one
install command and one setup command, on Windows, macOS and Linux.
