# Ingatan evals

`uv run python -m evals` runs every scenario under `evals/scenarios/` against the real modules (no MCP server, no network, no LLM, no API key) and prints the four metrics. It writes `evals/reports/latest.md` and exits non-zero when the leak rate is above 0, when a scenario that is not marked `xfail` fails, or when anything tries to open a socket.

## Metric definitions (OQ-18)

All four are computed from what each check observed, judged against the stored data (not against what the memory says about itself).

- **recall@5**: the share of `recall` checks that carry an `expect_top5` whose expected facts are all among the first five results.
- **stale rate**: the share of `recall` checks that return a closed fact (superseded or expired) or a fact past its review date with a status other than "check first", as if it were current. A closed fact shown with `include_closed` and the status `superseded` or `expired` is not stale.
- **leak rate**: the share of returned results, recall results and brief fact lines together, that come from outside the caller's scope: another agent's `agent`-tier fact, another business's fact, a pending proposal, or anything whose origin cannot be found.
- **tokens per brief**: the mean and the maximum of `ingatan.brief.estimate_tokens` over every brief the run builds. Every brief is also checked against its token budget (FR-3). If `ingatan.brief.build` is missing, the figure is reported as `n/a (brief not available)` and the scenarios that need a brief are `SKIP`ped, not failed.

Targets (PRD section 9, first guesses): leak rate 0, stale rate under 5%, recall@5 over 80%, tokens per brief about 1,000.

## Statuses and the exit code

| Status | Meaning | Exit code |
|---|---|---|
| PASS | every check held | 0 |
| FAIL | a check failed and the scenario has no `xfail` | 1 |
| XFAIL | a check failed and the scenario is marked `xfail` (known product gap) | 0 |
| XPASS | a scenario marked `xfail` passed: remove the marker | 0 |
| SKIP | a brief was needed and the brief module is not available, or the scenario needs an embedding model and the run has none | 0 |

A leak makes the run fail whatever the status of its scenario, so an `xfail` can never hide one. Exit code 2 means the scenario files themselves are invalid.

Flags: `--scenario NAME` (repeatable, a filtered run does not overwrite the report unless `--report PATH` is given), `--verbose` (every step and result), `--scenarios-dir DIR`, `--report PATH`, `--embeddings local` (see below).

## Running with an embedding model

The default run uses no embedding model, like the default installation. `uv run python -m evals --embeddings local` runs every scenario again with the local model switched on, and writes `evals/reports/latest-semantic.md` instead. The model must already be on disk (`ingatan embeddings pull`, or `--model-dir DIR`); the run opens no socket and fails like any other if something tries to. `--model NAME` picks another local model.

This is the check that enabling a model does not break anything the word index gets right: stale facts stay marked, nothing leaks, an off-topic question still returns nothing. A scenario with `"needs": "embeddings"` tests something only a model can do; it is `SKIP`ped in the default run, and `tests/evals` verifies that it fails when forced to run without a model. CI runs the default mode only, because it has no model.

`--defer` (with `--embeddings local`) runs every scenario the way `[embeddings] defer = true` does: each fact is saved without a vector and filled in afterwards. A server does that in a background thread; the harness does it right after each `remember`, so the next step sees what a user would see a moment later. It writes `evals/reports/latest-semantic-defer.md`. The scenario that needs a model must still pass: a conflict found late is still found.

## The extraction eval

`uv run python -m evals.extraction` measures the optional fact extraction (`ingatan extract`). It needs the model configured under `[llm]`, so CI does not run it, and it writes `evals/reports/latest-extraction.md`. During the run a connection may go only to the host and port of `[llm] api_base` (or the provider's default); any other attempt is refused and fails the run.

A case (`evals/extraction/cases/<id>.json`) is a few captured exchanges, the facts a careful reader would take from them, and what must not come out:

```json
{
  "id": "ex-01-harga-berubah",
  "title": "short description",
  "episodes": [{"user": "...", "assistant": "..."}],
  "expected": [{"name": "harga baru", "all_of": ["kopi susu"], "values": ["18.000"]}],
  "forbidden": ["20.000", "kata sandi"]
}
```

Each case runs in its own throwaway data folder: its episodes are recorded, extraction reads them, and what reached the proposal queue (after Ingatan's own checks, which is what a user would be asked to approve) is judged without a model:

- **correct**: an extracted fact that contains every word of `all_of` and every value of `values` of one expected fact. Numbers are compared by value, the way `ingatan extract` compares them: "15.000", "15,000" and "15000" are one number, and "1.5" is not "15".
- **wrong**: an extracted fact the case itself shows to be wrong. It contains a `forbidden` word or value, or it holds every word of `all_of` of an expected fact with other values, or the case expects nothing (an empty `expected` says the right answer is nothing).
- **unlisted**: an extracted fact that matches no expected fact and holds nothing forbidden. It may be a true fact the case does not list, or an invention. A list cannot tell those apart, so the count stands on its own and is called neither: read the facts of a case before you call them inventions, and add the true ones to `expected`.
- **found**: an expected fact that at least one extracted fact matches.

The report gives the four as shares, plus how many items Ingatan dropped before the queue (a number that is not in the episode, a malformed item, more than eight per episode) and how many episodes the model could not be read for. `tests/evals/test_extraction_metrics.py` covers the judging and one run with a stand-in model.

## Scenario format

One JSON file per scenario, named `<id>.json`, under `evals/scenarios/` (sub-folders are read too, see `scenarios/incidents/`).

```json
{
  "id": "stale-01-ongkir-lewat-31-hari",
  "title": "short description",
  "category": "stale",
  "source": "pattern",
  "xfail": "optional: why this fails today (real product behaviour)",
  "needs": "keywords",
  "token_budget": 1000,
  "actors": {"asisten": {"business": "tokoku", "agent": "asisten"}},
  "steps": [ ... ]
}
```

- `category`: `stale`, `supersede`, `leak-agent`, `leak-business`, `inflection`, `partial-word`, `brief`, `conflict`, `proposal`, `forget`, `noise` (questions about things the memory does not hold), `semantic` (what only an embedding model can do) `usage` (how often and how recently a fact was recalled, and what that may and may not change in the order of results) or `graph` (facts reached through links, not through the words of the query).
- `source`: `pattern` (written from a known pattern) or `incident` (a real incident from the project owner).
- `needs` (optional): `keywords` means the scenario must depend on the agent-supplied keywords (the query shares no word or fragment with the expected fact's text or entities). `trigram` means it must depend on the trigram index (the query is only a fragment of a word, with no keyword help). `tests/evals` lints these two statically so a scenario cannot pass for the wrong reason. `embeddings` means it needs an embedding model: it is skipped without one, and `tests/evals` runs it without one to prove that it then fails.
- `actors`: named callers, each a business and an agent. The first one is the default. The clock is frozen at 2026-11-11 09:00 UTC at the start of each scenario.
- Facts are referred to by a scenario-local `key`, never by id. A key made by a `business` or `global` remember names a pending proposal until an `approve` step, after which it names the new fact.

Steps run in order. Common fields: `op`, `actor` (default: the first actor), `expect_error` (an error class name, such as `NotFoundError` or `ScopeError`: the call must raise it).

| `op` | Fields |
|---|---|
| `remember` | `key`, `text`, `keywords`, optional `entities`, `tier`, `kind`, `review_in_days` |
| `supersede` | `old`, `new` (keys), optional `as` (a key that names the proposal the call queued) |
| `confirm` | `key`, `reason` (`{other_key}` is replaced by that fact's id) |
| `forget` | `key`, optional `by_user` (the user's CLI forget) |
| `approve`, `reject` | `key` of a proposal (the user's decision) |
| `advance_days` | `days` |
| `link` | `from`, `to` (keys of facts), optional `relation` (`related_to`, the default, or `part_of`) |
| `recall` | `query` or `query_text_of` (a key whose text is used verbatim), optional `limit` (default 10), `include_closed`, and at least one of `expect_top5` (keys that must be in the first five), `expect_absent` (keys that must not be returned at all, compared by id and by text), `expect_status` (key to exact status), `expect_none` (`true`: the query must return nothing at all), `expect_first` (one key: that fact must be result number one), `expect_related` (keys that must be among the related facts returned beside the results, which is where a fact reached through a link appears). Optional `repeat` (1 to 50): the query is run that many times and only the last result is judged, which is how a scenario makes a fact used, since every recall raises the use count of what it returns |
| `brief` | optional `expect_in_section` (section to keys), `expect_not_in` (a list of keys that must be nowhere in the brief, or a section to keys map), `max_tokens`, `expect_pending` (the pending-proposals count the brief states) |

Brief sections are `check first`, `global`, `business` and `agent`.

Every `recall` and `brief` step is also judged for leaks and stale results, whatever its own expectations say. The related facts of a `recall` are judged with its results: they reach the caller all the same.

## Driver

`evals/driver.py` defines `MemoryDriver`, a small protocol (remember, supersede, confirm, forget, approve, reject, recall, link, brief, plus the oracle calls `inspect` and `pending_texts`). A driver that returns related facts beside its recall results leaves them in `last_related` after each `recall`. `InProcessDriver` calls `facts`, `search`, `proposals` and `brief` directly. A driver that goes through the MCP server only has to implement the same protocol. Pass it with `run_all(..., driver_factory)` or `main(argv, driver_factory=...)`.

## Adding a scenario

Write `evals/scenarios/<id>.json`. Write facts, keywords and queries in natural Indonesian, with the keywords a careful agent would write (root words and synonyms), and never copy the query into the keywords to make a scenario pass. If a scenario fails because of real product behaviour, keep it, mark it `xfail` with the reason, and report it. Do not weaken it.
