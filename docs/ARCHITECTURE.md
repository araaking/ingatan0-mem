# Ingatan: implementation contract

This is the shared contract for everyone implementing `PLAN.md`. `PRD.md` wins on behaviour, this file wins on names, signatures and storage layout. If you must deviate, say so in your final report.

## Ground rules

- Python 3.10+ (`from __future__ import annotations` everywhere). Runtime deps: `mcp` (2.x: `mcp>=2.3,<3`, the `MCPServer` API), `pydantic`, `typer`, and `tomli` only for Python < 3.11. Storage is stdlib `sqlite3` only. No network calls anywhere.
- `src/` layout, package `ingatan`, build backend `hatchling`. Dev deps in a `dev` dependency group: `ruff`, `pyright`, `pytest`.
- Run everything through uv: `uv run python -m pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run python -m pyright`, and `uv run python -m evals`. All five must pass before you report done.
- Tests live in `tests/<module>/test_*.py`. Every test gets an isolated data folder through the `INGATAN_HOME` env var (shared fixture in `tests/conftest.py`). Never touch the real `~/.ingatan`.
- All text files UTF-8. JSON is written with `ensure_ascii=False`. CLI output must not crash on Windows consoles (reconfigure stdout/stderr to UTF-8).
- Code, comments, tool descriptions and messages in English. Comments only where the reason is not obvious.
- A module imports only from modules above it in this order: `timeutil`, `errors`, `db`, `redact`, `scope`, `audit`, `search`, `facts`, `proposals`, `episodes`, `brief`, `ingest`, `mcp`, `cli`, `integrations`. (`proposals` needs `facts` to create the approved fact; `facts.remember` must therefore create proposal rows through a small helper that lives in `facts` or lower. Keep it free of import cycles.)

## Small shared modules

- `ingatan/timeutil.py`: `now() -> datetime` (aware, UTC), `iso(dt) -> str` (`YYYY-MM-DDTHH:MM:SSZ`), `parse(str) -> datetime`. `set_now(dt | None)` freezes the clock for tests and evals. Every timestamp in the code comes from `timeutil.now()`. Timestamps are stored as ISO strings, which compare correctly as text. `review_after` and friends accept a date (`YYYY-MM-DD`) or full ISO string on input and are normalised to the full form.
- `ingatan/errors.py`: `IngatanError(Exception)` with a user-facing message. Subclasses: `ConfigError`, `NotFoundError`, `ValidationError`, `ScopeError`, `SchemaError`. The MCP layer and the CLI turn these into clean error output. Nothing else is caught there.

## Storage layout (OQ-3, OQ-4)

- Data folder: `$INGATAN_HOME` if set, else `~/.ingatan/`.
- Files: `<business>.db` and `global.db`. Business names match `^[a-z0-9][a-z0-9_-]{0,63}$` and may not be `global`. Agent names match the same pattern.
- Config: `.ingatan.toml` found in the current folder or the nearest parent.

```toml
business = "tokoku"
agent = "claude-code"

[brief]
token_budget = 1000
```

- `INGATAN_BUSINESS` and `INGATAN_AGENT` override the file, each on its own. Missing business or agent after both sources is a `ConfigError` that names both ways to set it.

## `ingatan.db`

- `connect(path) -> sqlite3.Connection`: creates the file if needed, `row_factory = sqlite3.Row`, `PRAGMA journal_mode=WAL`, `busy_timeout=10000`, `foreign_keys=ON`, `secure_delete=ON`, `isolation_level=None` (explicit transactions), then runs `migrate`. `busy_timeout` is not honoured by `journal_mode=WAL` on a brand-new file, so `connect` retries that pragma on locked/busy for up to 10 s; several processes can open a fresh data folder at once.
- `transaction(conn)` context manager: `BEGIN IMMEDIATE` … `COMMIT`, rollback on error. Every write goes through it.
- `migrate(conn)`: numbered files `src/ingatan/db/migrations/0001_*.sql`, applied in order above `PRAGMA user_version`, each inside `BEGIN IMMEDIATE` with `user_version` re-checked once the lock is won (a racing process skips what the other applied). A database newer than the code raises `SchemaError`.
- `check_capabilities()` raises `SchemaError` with a clear message when FTS5, the trigram tokenizer, FTS5 `secure-delete` (SQLite 3.42+) or the JSON functions are missing. `LATEST_VERSION` constant.
- `GLOBAL_LOG_OFFSET = 1_000_000_000`: `connect` on `global.db` seeds the `log` AUTOINCREMENT counter to this value (idempotent, under `BEGIN IMMEDIATE`), so a log id names exactly one entry across both files. `audit.get`, `diff` and `undo` rely on it. Business-file ids start at 1.

Both file kinds use the same schema.

```
facts(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,   -- explicit rowid, stable under VACUUM. The FTS rowid
  id TEXT NOT NULL UNIQUE,        -- "f_" + 12 hex
  text TEXT NOT NULL,
  entities TEXT NOT NULL,         -- JSON array of strings
  keywords TEXT NOT NULL,         -- JSON array of strings
  tier TEXT NOT NULL,             -- global | business | agent
  agent TEXT NOT NULL,            -- writing agent. Owner when tier = agent
  kind TEXT NOT NULL,             -- stable | volatile
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  superseded_by TEXT,
  review_after TEXT,              -- NULL for stable
  review_interval_days INTEGER,   -- the original interval, used by confirm
  pending_proposal TEXT,          -- proposal id that wants to supersede this fact (FR-59)
  source TEXT NOT NULL,           -- JSON: {agent, date, session?, document?, episode?}
  use_count INTEGER NOT NULL DEFAULT 0,
  last_used TEXT,
  created_at TEXT NOT NULL
)
facts_fts   -- FTS5(text, entities, keywords), unicode61 remove_diacritics 2, rowid = facts.seq
facts_tri   -- FTS5(text, entities, keywords), tokenize='trigram', rowid = facts.seq
            -- both (and notes_fts, notes_tri) are created with the FTS5 'secure-delete' option on
            -- (migration 0001). `forget` depends on it. Triggers keep them in step with `facts`.
conflicts(id INTEGER PK, fact_a TEXT, fact_b TEXT, created_at TEXT, resolved_at TEXT, resolution TEXT)
log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, agent TEXT NOT NULL, op TEXT NOT NULL,
  fact_id TEXT,                   -- fact or proposal id
  tier TEXT, owner TEXT,          -- copied from the fact so scope filtering works after forget
  change TEXT                     -- JSON {"before": {...}|null, "after": {...}|null}. NULL after forget
)
proposals(
  id TEXT PRIMARY KEY,            -- "p_" + 12 hex
  tier TEXT NOT NULL,             -- business | global
  payload TEXT NOT NULL,          -- JSON of the fact fields to create
  similar TEXT NOT NULL,          -- JSON array of fact ids judged similar at propose time
  supersedes TEXT,                -- fact id this proposal replaces when approved
  agent TEXT NOT NULL, created_at TEXT NOT NULL,
  status TEXT NOT NULL,           -- pending | approved | rejected
  decided_at TEXT, fact_id TEXT
)
episodes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent TEXT NOT NULL, session TEXT,
  user_text TEXT NOT NULL, assistant_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,           -- pending | processed
  offered INTEGER NOT NULL DEFAULT 0, processed_at TEXT
)
notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL, heading TEXT NOT NULL, text TEXT NOT NULL,
  file_hash TEXT NOT NULL, created_at TEXT NOT NULL
)
notes_fts, notes_tri  -- FTS5 over (heading, text), same two tokenizers, rowid = notes.id
```

Log protection lives in the schema: a trigger aborts every `DELETE` on `log`, and aborts every `UPDATE` on `log` unless it only sets `change` to NULL. That is what makes FR-36 true whatever the Python code does.

Where rows live: `global.db` holds `global` facts, their log entries and `global` proposals. `<business>.db` holds everything else, including `conflicts` (a pair may name a global fact by id), `episodes` and `notes`. Fact ids are unique across files.

## `ingatan.scope`

```python
@dataclass(frozen=True)
class Scope: business: str; agent: str; home: Path; token_budget: int = 1000

def resolve_scope(start: Path | None = None, env: Mapping[str, str] | None = None) -> Scope
def data_home(env=None) -> Path
def db_path(home: Path, name: str) -> Path          # name = business name or "global"

class Context:                                        # the one object every module takes
    scope: Scope
    business: sqlite3.Connection
    glob: sqlite3.Connection
    def conn_for_tier(self, tier: str) -> sqlite3.Connection
    def close(self) -> None

def open_context(scope: Scope) -> Context
VISIBLE_SQL                                           # helper(s) producing the WHERE clause below
```

Visibility, applied to facts and to log rows alike: in the business file `tier = 'business' OR (tier = 'agent' AND agent/owner = :agent)`, in the global file `tier = 'global'`. No function takes a business name from a caller. `find_fact(ctx, fact_id) -> (conn, row) | None` looks in both files and applies visibility, so an invisible fact behaves exactly like a missing one.

## `ingatan.redact`

`redact(text: str) -> str`, marker `[redacted]` (constant `MARKER`). Patterns (OQ-14): provider key prefixes (`sk-`, `sk-ant-`, `ghp_`, `gho_`, `github_pat_`, `xoxb-`/`xoxp-`, `AKIA…`, `AIza…`, `glpat-`, `hf_`, Stripe `sk_live_`/`rk_live_`/`pk_live_`), JWTs, `Bearer <token>`, PEM private key blocks, and assignments such as `password=…`, `passwd: …`, `secret=…`, `token=…`, `api_key=…`, `kata sandi: …`, `sandi=…`. For assignments the key name stays and only the value is replaced. Corpus files: `tests/redact/corpus_secrets.txt` and `tests/redact/corpus_plain.txt` (the plain one includes Indonesian sentences that mention the words "password", "token", "kata sandi" without a value).

## `ingatan.audit`

```python
def write(conn, *, agent, op, fact_id, tier, owner, before: dict | None, after: dict | None) -> int
def entries(ctx, *, fact_id=None, fact_ids=None, limit=None, newest_first=False) -> list[LogEntry]   # scope filtered
def scrub(conn, fact_id) -> None            # sets change = NULL on all entries of that fact. Used only by forget
def get(ctx, log_id) -> LogEntry
def diff(ctx, log_id) -> Diff               # before / after of one entry
def undo(ctx, log_id) -> int                # returns the id of the new log entry
```

`write` must be called inside an open `transaction`, and it raises if it is not (`conn.in_transaction`). Ops: `create`, `supersede`, `confirm`, `forget`, `propose`, `approve`, `reject`, `ingest`, `undo`. `undo` handles `create`, `supersede`, `confirm` and `approve`, refuses `forget` with the message `a forget cannot be undone` (and `undo`, `propose`, `reject`, `ingest`, each with its own message), and refuses when a later entry that was not itself undone mentions the same fact (OQ-16). Undoing a `create` or `approve` **closes** the fact (`valid_to = now`) and resolves its conflict pairs with resolution `undo`; it never deletes the row, so the id still resolves and `history` shows create then undo. One new `undo` entry is written in the file of the entry it reverses, with `undoes` and `undone_op` in its `after`. Log ids: see `GLOBAL_LOG_OFFSET` under `db`. Because undo has to change fact rows, its implementation may live where the layering allows (for example `ingatan/audit/undo.py` taking plain connections and SQL), as long as the public name is `audit.undo`.

## `ingatan.search`

```python
def recall(ctx, query: str, *, limit: int = 5, include_closed: bool = False, touch: bool = True) -> RecallResult
# RecallResult.facts: list[FactHit(id, text, tier, source, date, status, entities, keywords, score)]
# RecallResult.notes: list[NoteHit(id, path, heading, text)]     -- document excerpts, never mixed into facts
def similar_facts(ctx, text, entities, keywords, *, exclude_id=None, limit=3) -> list[FactHit]   # OQ-7
def status_of(ctx, fact_row) -> str
def is_valid_sql() -> str                    # shared validity filter
```

- Valid: `valid_from <= now AND (valid_to IS NULL OR valid_to > now) AND superseded_by IS NULL`.
- Status: `check first` when the fact is volatile with `review_after < now`, or has `pending_proposal` set. `possible conflict` when it is in an unresolved `conflicts` pair. Both at once: `check first, possible conflict`. Otherwise `valid`. Closed facts shown with `include_closed`: `superseded` or `expired`.
- Query handling: split the query into word tokens, quote each one for FTS5 and OR them, so user text can never be an FTS syntax error. Tokens under three characters are skipped for the trigram index.
- Signals, fused with reciprocal rank fusion (`k = 60`) over the union of candidates: BM25 on `facts_fts`, trigram match rank, entity match (number of the fact's entities that appear in the query, case-insensitive), recency (`valid_from`, newest first). A signal contributes nothing for a fact it did not match, so an entity match always lifts a fact above an otherwise equal one.
- Query expansion: `search/stem_id.py` (`candidates(word)`) is a dependency-free Indonesian affix stripper that proposes up to 6 plausible roots per query word (`pengiriman` -> `kirim`). It over-generates on purpose and stores nothing. The roots (never the query words themselves, at most 60) run against both indexes as two extra signals with weight `EXPANDED_WEIGHT = 0.25` against 1.0 for the exact ones, so a literal match outranks a root-only match. Notes get the same expansion. Expansion happens at query time only; do not stem at write time.
- `touch=True` raises `use_count` and sets `last_used` for returned facts (OQ-6). `similar_facts` and the brief never touch.
- Runs against both files, visibility applied, results merged before the cut to `limit`.

## `ingatan.facts`

```python
def remember(ctx, text, *, keywords, entities=None, tier="agent", kind=None,
             review_after=None, source=None, episode: int | None = None) -> RememberResult
# RememberResult(fact_id: str | None, proposal_id: str | None, similar: list[{id, text, status}], message: str)
def supersede(ctx, old_id, new_id) -> SupersedeResult
def confirm(ctx, fact_id, reason) -> ConfirmResult
def forget(ctx, fact_id, *, by_user: bool = False) -> None
def history(ctx, id_or_entity) -> list[LogEntry]
def create_fact(conn, ctx, fields, *, op="create") -> str      # shared by remember and proposal approval
```

- `remember`: text is redacted first, and so are keywords and entities. Empty or missing `keywords` raises `ValidationError` asking for synonyms and root words, and saves nothing. `kind` defaults to `volatile`. A volatile fact with no `review_after` gets `valid_from + 30 days`. `review_interval_days` records the interval. `tier="agent"` creates the fact at once, records a `conflicts` pair with each similar fact, and logs `create`. `tier` `business` or `global` creates a pending proposal (`propose` in the log) and no fact. Similar facts are still returned. `episode=<id>` marks that episode processed (it must be one of the caller's own, else `NotFoundError`) and is stored as `source.episode`.
- `supersede(old, new)`:
  - Both ids must exist and be visible, else `NotFoundError` and nothing changes. `new` may be a fact id or a pending proposal id.
  - Old is an `agent` fact and new is a fact: close old (`valid_to = now`, `superseded_by = new`), resolve their conflict pair, log `supersede`.
  - Old is a `business` or `global` fact (FR-59): nothing is closed. If new is a pending proposal of the same tier, set `proposals.supersedes = old`. If new is an agent fact, create a proposal in old's tier with that fact's content and `supersedes = old`. Either way set `old.pending_proposal` so old shows `check first` at once, and log it. The result message tells the agent the change waits for the user's approval.
  - Old is an `agent` fact and new is a proposal: `ValidationError`.
- `confirm`: empty reason raises `ValidationError`. Sets `review_after = now + review_interval_days` (stable facts keep NULL). Resolves every unresolved conflict pair whose other fact id appears in the reason (OQ-8). Works directly on business and global facts (FR-60). Logs the reason.
- `forget`: on a business or global fact with `by_user=False` raises `ScopeError` and changes nothing (FR-61). Otherwise deletes the fact row and its index rows, drops conflict pairs, scrubs earlier log entries, writes one content-free `forget` entry, and makes the bytes unrecoverable in the database file and the WAL file. It also deletes the episode named in the fact's `source.episode` (the captured turn it was digested from; an agent only its own, the user's forget any), and empties proposal payloads and log entries that held a copy of the text. Facts that pointed at it through `superseded_by` stay closed (OQ-17). The byte-scan test in T-11 is the judge of this function.
- `history`: argument starting with `f_` or `p_` is an id, anything else is an entity name (case-insensitive match against `entities`). Scope filtered, oldest first.

## `ingatan.proposals`

```python
def pending(ctx) -> list[Proposal]        # CLI only. Never reachable from MCP
def pending_count(ctx) -> int             # for the brief
def approve(ctx, proposal_id) -> str      # returns the new fact id
def reject(ctx, proposal_id) -> None
```

Approve creates the fact in the proposal's tier with the proposing agent as `source.agent`, records conflict pairs from `similar`, closes `supersedes` if set, clears `pending_proposal`, logs `approve`. Reject clears `pending_proposal` on the fact it wanted to replace and logs `reject`.

## `ingatan.episodes`

```python
TRUNCATE_AT = 2000        # characters per message (OQ-13)
RETENTION_DAYS = 30
MAX_OFFER = 3
MAX_OFFERS_PER_EPISODE = 3

def capture(scope_or_none, user_text, assistant_text, *, session=None) -> bool   # never raises. Returns False on failure
def record(ctx, user_text, assistant_text, *, session=None) -> int               # the raising core used by capture
def offer(ctx, limit=MAX_OFFER) -> list[Episode]     # oldest pending first, own agent only, bumps `offered`
def mark_processed(ctx, episode_id) -> None
def purge(ctx) -> int                                 # deletes processed episodes older than RETENTION_DAYS
```

`capture` resolves the scope itself when given None, redacts, truncates, and on any exception appends a line to `<home>/capture-errors.log` and returns False (NFR-14). An episode is marked processed when the agent passes `episode=<id>` to `remember`, or when it has been offered `MAX_OFFERS_PER_EPISODE` times without that (the agent judged it not worth a fact). Episodes are private to the agent that captured them.

## `ingatan.brief`

`build(ctx, *, offer_episodes: bool = True) -> str` and `estimate_tokens(text) -> int` (`ceil(len / 4)`).

Layout, in this order, empty sections omitted:

```
# Ingatan brief: <business> / <agent>
Pending proposals waiting for the user: N

## Check first
Do not act on these facts before checking them. Check only those relevant to the current task. ...
- [f_xxx] text (review was due 2026-10-01)

## Global
- [f_xxx] text
## Business
## Agent

## Episodes to digest
<instruction: turn what is worth keeping into facts with remember(..., episode=<id>)>
### Episode 12 (2026-11-10)
User: ...
Assistant: ...
```

Facts in each section are ordered by `use_count` desc, then `last_used` desc. A fact with `possible conflict` carries that tag on its line. Over budget: drop the least used facts first, across the three sections. The header, the pending count and the whole "check first" section are kept. If episodes push it over, drop episodes from the end. The final text is never over `scope.token_budget`. Notes never appear.

## `ingatan.ingest`

`ingest_path(ctx, path) -> IngestReport(files: int, created: int, replaced: int, skipped: list[Skipped])`. `skipped` is a list (one entry per file, with the reason: unchanged, not UTF-8, unreadable), not a count; use `len()` for a number. One note per heading section (text before the first heading is a section headed by the file name). Redacted. `path` stored as the resolved absolute path with forward slashes. A file whose hash is unchanged is skipped. A changed file has its notes deleted and rewritten. One `ingest` log entry per note.

## `ingatan.mcp`

`ingatan/mcp/server.py`: `build_server(ctx) -> MCPServer` (`from mcp.server.mcpserver import MCPServer`, the mcp 2.x API, not the 1.x `FastMCP`) and `run()`. `run` opens the context with a short retry on locked/busy and serves stdio. Exactly seven tools: `brief`, `recall`, `remember`, `supersede`, `confirm`, `forget`, `history`. Parameters:

- `brief()`
- `recall(query: str, limit: int = 5, include_closed: bool = False)`
- `remember(text: str, keywords: list[str], entities: list[str] | None, tier: "agent" | "business" | "global" = "agent", kind: "stable" | "volatile" | None, review_after: str | None, episode: int | None)`
- `supersede(old: str, new: str)`
- `confirm(id: str, reason: str)`
- `forget(id: str)`
- `history(id_or_entity: str)`

`keywords` is optional in the JSON schema so that a call without it reaches our code and gets the helpful error (FR-10). Tools return readable text. An `IngatanError` becomes an MCP tool error with its message. Nothing but protocol goes to stdout.

## `ingatan.cli`

`ingatan` with no subcommand starts the MCP server (OQ-20, `uvx ingatan`). Commands: `serve`, `init [--business --agent]`, `doctor`, `log [--limit N] [--fact ID]`, `diff <log_id>`, `undo <log_id>`, `proposals list|approve <id>|reject <id>`, `forget <fact_id>` (the user's forget, any tier, asks for confirmation unless `--yes`), `ingest <path>`, `conflicts` (the NFR-9 report), `hook <event>` (Claude Code hook entry, see `integrations`). Exit code 1 with a one-line message on `IngatanError`.

- Human commands open the context of the configured business and agent: they see `business` and `global` facts and that agent's private entries only. `log`, `diff`, `undo`, `forget` and `conflicts` take `--agent NAME` to open the context as another agent, to audit that agent's private entries. The visibility filter itself is never weakened.
- `hook <event>` always exits 0 and prints nothing of its own. When the hook module is missing or `run` raises, it writes one line to `<data home>/capture-errors.log` if it can (NFR-14). It never exits 2, which would block the agent.

## `evals/`

Not shipped in the wheel. `uv run python -m evals` runs every scenario in `evals/scenarios/*.yaml`-style files (use JSON or TOML so no extra dependency is needed) against the real modules in a temporary `INGATAN_HOME` with a frozen clock, prints recall@5, stale rate, leak rate and tokens per brief, writes `evals/reports/latest.md`, and exits non-zero when leak rate is above 0. Scenario text is Indonesian.
