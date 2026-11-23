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
| 6 | [LLM fact extraction](#7-llm-fact-extraction-optional) | Built; the share of correct and invented facts still has to be measured with a real model |
| 8 | [Tombstones](#5-tombstones-for-wrong-values) | The design is written ([RETRACTION.md](RETRACTION.md)) and waits for the owner's answers to its six questions |
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

**Status.** The design is written down in [RETRACTION.md](RETRACTION.md): retracting is a
logged, undoable operation on a fact that stays in the file, the tombstone is the fact row
itself (so `forget` removes it with the fact), and the match is the exact normalized text.
It ends with six questions for the owner. No code exists.

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

**Status.** Built: `ingatan extract`, off by default, switched on under `[llm]` (see the
changelog). The two open decisions were settled like this: an extracted fact is always a
proposal, whatever `require_approval` says, and no model is prescribed (an Ollama model or
any OpenAI-compatible endpoint, named by the user).

**What is left.** The done-when is a number, and it needs a real model:
`python -m evals.extraction` with each model worth naming, the report committed as
`evals/reports/latest-extraction.md`, and the README stating the figures with the model
names. This section goes when they are in. After that, with numbers in hand, decide whether
an extracted fact may ever be written directly.

**Done when.** An eval measures how many extracted facts are correct and how many are
invented, on scenarios with a known answer, and the README states that number.

## 9. Easier installation

**Status.** Built, except the release itself. `ingatan setup`, `ingatan embeddings enable`,
the three-command start in the README, a rehearsal path to TestPyPI and a tag guard in the
release workflow are in (see the changelog). CI is set up to run `setup` and a remember and
recall round trip from the built wheel on Windows, macOS and Linux; that job has not run
yet, because the branch has not been pushed. The same steps pass by hand on macOS.

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
