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
| SKIP | a brief was needed and the brief module is not available | 0 |

A leak makes the run fail whatever the status of its scenario, so an `xfail` can never hide one. Exit code 2 means the scenario files themselves are invalid.

Flags: `--scenario NAME` (repeatable, a filtered run does not overwrite the report unless `--report PATH` is given), `--verbose` (every step and result), `--scenarios-dir DIR`, `--report PATH`.

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

- `category`: `stale`, `supersede`, `leak-agent`, `leak-business`, `inflection`, `partial-word`, `brief`, `conflict`, `proposal` or `forget`.
- `source`: `pattern` (written from a known pattern) or `incident` (a real incident from the project owner).
- `needs` (optional): `keywords` means the scenario must depend on the agent-supplied keywords (the query shares no word or fragment with the expected fact's text or entities). `trigram` means it must depend on the trigram index (the query is only a fragment of a word, with no keyword help). `tests/evals` lints this statically so a scenario cannot pass for the wrong reason.
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
| `recall` | `query` or `query_text_of` (a key whose text is used verbatim), optional `limit` (default 10), `include_closed`, and at least one of `expect_top5` (keys that must be in the first five), `expect_absent` (keys that must not be returned at all, compared by id and by text), `expect_status` (key to exact status) |
| `brief` | optional `expect_in_section` (section to keys), `expect_not_in` (a list of keys that must be nowhere in the brief, or a section to keys map), `max_tokens`, `expect_pending` (the pending-proposals count the brief states) |

Brief sections are `check first`, `global`, `business` and `agent`.

Every `recall` and `brief` step is also judged for leaks and stale results, whatever its own expectations say.

## Driver

`evals/driver.py` defines `MemoryDriver`, a small protocol (remember, supersede, confirm, forget, approve, reject, recall, brief, plus the oracle calls `inspect` and `pending_texts`). `InProcessDriver` calls `facts`, `search`, `proposals` and `brief` directly. A driver that goes through the MCP server only has to implement the same protocol. Pass it with `run_all(..., driver_factory)` or `main(argv, driver_factory=...)`.

## Adding a scenario

Write `evals/scenarios/<id>.json`. Write facts, keywords and queries in natural Indonesian, with the keywords a careful agent would write (root words and synonyms), and never copy the query into the keywords to make a scenario pass. If a scenario fails because of real product behaviour, keep it, mark it `xfail` with the reason, and report it. Do not weaken it.
