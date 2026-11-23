# Ingatan: implementation contract

This is the shared contract for everyone implementing `PLAN.md`. `PRD.md` wins on behaviour, this file wins on names, signatures and storage layout. If you must deviate, say so in your final report.

## Ground rules

- Python 3.10+ (`from __future__ import annotations` everywhere). Runtime deps: `mcp` (2.x: `mcp>=2.3,<3`, the `MCPServer` API), `pydantic`, `typer`, and `tomli` only for Python < 3.11. Storage is stdlib `sqlite3` only. The optional extra `ingatan[vector]` adds `numpy`, `onnxruntime` and `sentencepiece` for the local embedding model; nothing imports them unless that provider is in use. No network calls anywhere, with two exceptions the user has to ask for: `ingatan embeddings pull` (downloads the local model once) and a remote embedding provider (`ollama`, `openai`, `gemini`) if one is configured.
- `src/` layout, package `ingatan`, build backend `hatchling`. Dev deps in a `dev` dependency group: `ruff`, `pyright`, `pytest`.
- Run everything through uv: `uv run python -m pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run python -m pyright`, and `uv run python -m evals`. All five must pass before you report done.
- Tests live in `tests/<module>/test_*.py`. Every test gets an isolated data folder through the `INGATAN_HOME` env var (shared fixture in `tests/conftest.py`). Never touch the real `~/.ingatan`.
- All text files UTF-8. JSON is written with `ensure_ascii=False`. CLI output must not crash on Windows consoles (reconfigure stdout/stderr to UTF-8).
- Code, comments, tool descriptions and messages in English. Comments only where the reason is not obvious.
- A module imports only from modules above it in this order: `timeutil`, `errors`, `db`, `redact`, `scope`, `tokens`, `audit`, `search`, `facts`, `proposals`, `episodes`, `brief`, `ingest`, `mcp`, `cli`, `integrations`. (`proposals` needs `facts` to create the approved fact; `facts.remember` must therefore create proposal rows through a small helper that lives in `facts` or lower. Keep it free of import cycles.)

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
- `normalize_fact_text(text)` and `fact_text_key(text)`: the normal form of a fact's text (case, runs of whitespace and trailing punctuation removed, nothing else) and the hash of it stored in `facts.text_key`. `connect` fills the key for rows that predate schema version 3.
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
  category TEXT NOT NULL DEFAULT 'fact', -- fact | rule | procedure | preference
  pinned INTEGER NOT NULL DEFAULT 0,     -- 1 = prioritized in brief and recall
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  superseded_by TEXT,
  review_after TEXT,              -- NULL for stable
  review_interval_days INTEGER,   -- the original interval, used by confirm
  pending_proposal TEXT,          -- proposal id that wants to supersede this fact (FR-59)
  source TEXT NOT NULL,           -- JSON: {agent, date, session?, document?, episode?}
  use_count INTEGER NOT NULL DEFAULT 0,
  last_used TEXT,
  created_at TEXT NOT NULL,
  text_key TEXT                   -- hash of the normalized text (migration 0003), indexed.
                                  -- Internal: never copied into the log or a proposal
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
-- Migration 0002_semantic_and_structure.sql:
fact_embeddings(fact_id TEXT PRIMARY KEY, model TEXT, dims INTEGER, vec BLOB, updated_at TEXT)
note_embeddings(note_id INTEGER PRIMARY KEY, model TEXT, dims INTEGER, vec BLOB, updated_at TEXT)
aliases(id INTEGER PK, term TEXT, canonical TEXT, created_at TEXT, UNIQUE(term, canonical))
edges(id INTEGER PK, src_id TEXT, dst_id TEXT, relation TEXT, created_at TEXT, UNIQUE(src_id, dst_id, relation))
-- Migration 0003_fact_text_key.sql: facts.text_key and its index (see above).
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

## `ingatan.tokens`

HTTP-mode access tokens (FR-58, see `HTTP_MODE.md`). Pure standard library: it must not import `mcp`, `starlette` or `pydantic`. A token is `igt_` plus 32 random bytes (URL-safe base64, no padding) and maps to one `(business, agent)`. Only its SHA-256 hash is stored, in `<INGATAN_HOME>/tokens.json` (`{"tokens": [{id, hash, business, agent, label, created}]}`, id `t_` + 8 hex), rewritten atomically (temp file, `os.replace`, mode 0600 where supported).

- `Token` (frozen dataclass), `add(home, business, agent, label="") -> (raw, Token)`, `list_tokens(home)`, `revoke(home, token_id) -> Token`, `hash_token(raw)`, `tokens_path(home)`. Names pass `validate_business` / `validate_agent`.
- `TokenStore(home)`: `lookup(raw) -> Scope | None` (hash compared with `hmac.compare_digest`) and `count()`. It re-reads the file when its `(mtime_ns, size, inode)` changes, so a revoke takes effect on the next request. An unreadable file lets nobody in.

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
- Signals, fused with reciprocal rank fusion (`k = 60`) over the union of candidates: BM25 on `facts_fts`, trigram match rank, entity match (number of the fact's entities that appear in the query, case-insensitive). A signal contributes nothing for a fact it did not match, so an entity match always lifts a fact above an otherwise equal one.
- Freshness is not a rank signal. The fused score of a `volatile` fact is multiplied by `1 + 0.04 * (0.5 ** (age_days / 30) - 0.5)`, a factor between 0.98 and 1.02 that depends on the fact's age and not on what else matched. A `stable` fact keeps its score. Ties still fall back to the newest `valid_from`.
- Usage is not a rank signal either. The fused score is multiplied by `1 + 0.01 * often * recent` (`_usage_factor`): `often = 1 - 0.5 ** (use_count / 5)`, `recent = 0.5 ** (days_since_last_used / 30)`. A fact never recalled keeps its score. What the 1% bound gives: it is smaller than the gap between two neighbouring ranks of one signal among the first forty (1/100 against 1/101), so within a single signal usage cannot reorder two facts. Where several signals disagree, fused scores can be closer than that, and there usage decides; that is what it is for. The count saturates and the effect decays because `recall` raises `use_count` itself. A fact whose status is `possible conflict` gets no boost (`_open_conflict_ids`, the rule of `_in_open_conflict` for the whole pool: the other fact of the pair is still valid and visible to the caller), so a much recalled value does not move ahead of a newer fact that was paired with it, and a recall `as_of` a past moment ignores usage. What it does not give: when two facts that contradict each other were never paired (no shared entity, or saved with `check_conflicts=false`), the older, much recalled one can lead a newer one that is only a few days younger. The scenarios `usage-01` and `usage-02` hold the paired case and the tie; both were run without an embedding model only.
- A weak single-term hit is dropped when its BM25 is under a quarter of the best hit of the same index in the same file. Raw BM25 values are never compared across indexes or files: they depend on the tokenizer, the column weights and the size of the file.
- Fact extraction (`extract/`, `ingatan extract`, `[llm]`, off by default): `extract_pending(ctx, limit, dry_run)` reads the caller's pending episodes with `episodes.pending` (no offer is counted), redacts them again, and asks the configured model once per episode (`extract/llm.py`: Ollama `/api/chat` or an OpenAI-compatible `/chat/completions`, standard library only, outside any transaction). The answer is read by `prompt.parse_answer` and used for four fields only (`text`, `keywords`, `entities`, `kind`) of at most `MAX_FACTS_PER_EPISODE` items; an item with a number the episode does not contain is dropped (`extract.numbers` puts each number into one spelling first, so thousands separators and either decimal mark compare by value), and one that repeats a valid fact, a pending proposal or an extracted proposal the user rejected, word for word, is skipped. Each kept item goes through `facts.remember(..., tier="business", as_proposal=True, episode=id, source={"extracted_by": "provider/model"})`, so it is a proposal whatever `require_approval` says and never a fact by itself. With each text the model is shown up to `MAX_KNOWN_FACTS` facts that `recall` finds for it (`touch=False`, so this is not use; by words alone, with embeddings switched off for that one lookup, so the text is never sent to the embedding provider as a query), and may return `relations` from one of its facts to one of those ids; a relation is kept only for an id it was shown and one of the two relations of `edges`, and it rides on the proposal as `links`, so it becomes an edge only on approval, and not even then when the new fact is a possible conflict with the target (a link would resolve that pair). The episode is marked processed when it was read; one the model failed on stays pending and gets one more in `extract_failures` (migration 0006, on `episodes` and `notes`), and after `MAX_READ_FAILURES` it is left out unless `retry_failed` is set. `llm.Unreachable` (no answer at all: down, slow, busy, key refused) is counted against no text, and `MAX_UNREACHABLE` of them in a row end the pass; a request the endpoint refuses (HTTP 400, 413, 422), an empty answer and an answer that is not JSON are about the text. With `documents=True` the texts are ingested notes whose `extracted_at` (migration 0005) is NULL, the proposal's source names the document, and a note that was read gets the timestamp; a changed file is ingested as new rows and so is read again. The server and the MCP tools never import `extract`.
- Related facts in `recall` (`_related_hits`, `MAX_HOPS = 2`, `MAX_RELATED = 3`): a breadth-first walk over `edges` from the hits. Every direct neighbour comes before anything two links away, so a second hop only fills the room the first leaves. A fact that is closed, or that the caller cannot see, is neither returned nor walked through. Edges live in the business file, so a walk never leaves the business. The scenario `graph-01-dua-lompatan` needs the fact two links away and fails with its links removed.
- Review queue (`facts/review.py`, `cli/review.py`, `ingatan conflicts review`): `open_pairs` lists the unresolved pairs whose facts are both valid and visible, with `newer_id` and `older_id` by `valid_from`; `resolve_pair(ctx, pair, action)` applies `keep_both` (`confirm` on the newer fact naming the older, then a `related_to` link), `keep_newer` or `keep_older` (`supersede`), all logged and undoable, and skips a pair that changed since it was listed. In approval mode the queued change comes back as a proposal id, which the command approves at once (`facts` cannot import `proposals`). Forgetting one of the two stays in the command, with the confirmation of `ingatan forget`, and is never applied to several pairs.
- Shared model process (`search/shared.py`, `[embeddings] shared`, off by default): one process per data folder holds the local model and the servers ask it for vectors (`embed._embed_local_batch` calls `shared.embed` instead of `local.embed`). Loopback TCP on an OS-chosen port; pid, port and a random token in `<home>/run/embed.json` (mode 0600, never text); length-prefixed frames; a mutual HMAC-SHA256 challenge before any text is sent; one deadline for the handshake and one for the request with its answer (`_Deadline`, applied before every send and read, so a peer that trickles bytes cannot stretch them). `shared.embed` starts the process when none answers (`python -m ingatan.cli embeddings serve`, streams on the null device) and retries once; a start that does not lead to an answer is not repeated for 30 seconds per caller, while a process that answered and later left on idleness is started again at once. Every failure is `shared.Unavailable`, an `OSError`, so the caller gets no vector and answers from FTS5; the model is never loaded in-process as a fallback. The process exits after `idle_unload_seconds`, on `embeddings stop`, or when its state file is gone or names another process (a file that cannot be read for a moment is neither). `probe_embedding`, the health check of `doctor`, `setup` and `embeddings enable`, loads the model in its own process also with `shared` on, so a check never leaves the shared process running.
- Deferred embedding (`facts/backfill.py`, `[embeddings] defer`, off by default): `remember` saves without a vector and leaves a row in `embed_queue` (fact id and whether a conflict check is owed, never text); one daemon thread per server process (`start_worker` / `notify` / `stop_worker`, wired through `mcp.server.background_fill` and `HTTPApp.lifespan`) opens its own `Context`, embeds outside any transaction, stores the vector only while the fact row exists, and runs `similar_facts` with it, as the agent that wrote the fact (the server that fills may belong to another agent, whose private facts are no part of that check). Pairs found then get a `conflict` log entry, which `undo` reverses and which never blocks undoing the fact's other entries. Facts that were never queued and have no vector at all are embedded too, without a check; a vector of another model is left to `ingatan reindex`. Each worker has its own stop flag and its own list of scopes, so a thread told to stop ends after the batch it is in and never runs beside its successor; a pass that fails as a whole (an exception, or no vector for anything) is run again after `RETRY_SECONDS`, at most `MAX_RETRIES` times.
- Embeddings are optional (`search/embed.py`, `search/local.py`; provider `none` by default). With a provider, the query vector is scored against the `fact_embeddings` rows of the same model, and hits at or above the model family's `recall_floor` join the fusion as one more signal (weight 1.5). Every cut-off comes from `embed.similarity_profile(cfg)`, never from a bare constant, because cosine scores are not comparable between model families; the measurements are in `docs/EMBEDDINGS.md`. The local provider is ONNX Runtime + SentencePiece: one text per forward pass, CPU memory arena off, the model dropped after `idle_unload_seconds`, files under `<INGATAN_HOME>/models` fetched only by `ingatan embeddings pull` (and by `embeddings enable` and `setup`, which call it). An embedding is computed before any write lock is taken and at most once per call; a failure returns no vector and recall carries on with FTS5.
- `similar_facts` takes its candidates from indexes instead of reading every fact row into Python: the facts whose `entities` column mentions one of the new fact's entities (at most 200 per file, those sharing topic words first) and, for a profile that has a `conflict_strict` value, the facts at or above it. Each candidate's own vector is then scored, so the 50-row cut of a nearest-neighbour list cannot hide the fact that shares the entity. One counting pass over the `entities` column still runs inside SQLite (from 10 valid facts on), to find the entities too common to signal a conflict. A candidate is a conflict when it shares a specific (non-generic) entity and either overlaps in topic words or reaches `conflict`. Similarity alone counts only if the profile has a `conflict_strict` value; the E5 family has none, because two facts on one template about different subjects score as high as a real update. Facts that name different external resources are never a conflict unless their wording is nearly the same.
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

- `remember(..., dedup=True)`: a text that is the same as a valid stored fact of the same tier and category after `db.normalize_fact_text`, with the same entities, refreshes that fact (one lookup in the `text_key` index) and creates nothing. Closeness in meaning never merges: such a fact is saved and the earlier one is returned in `similar`.
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

Facts in each section are ordered by `use_count` desc, then `last_used` desc; facts that tie in every respect are ordered by insertion (`seq`, newest first), never by their random id, so two stores with the same content build the same brief. A fact with `possible conflict` carries that tag on its line. Over budget: drop the least used facts first, across the three sections. The header, the pending count and the whole "check first" section are kept. If episodes push it over, drop episodes from the end. The final text is never over `scope.token_budget`. Notes never appear.

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

`ingatan/mcp/http.py`: `build_app(home) -> HTTPApp` (an ASGI app) and `run_http(host="127.0.0.1", port=8765)`. `/healthz` answers without auth. Every request to `/mcp` needs `Authorization: Bearer <token>`; `TokenStore.lookup` turns it into a `Scope`, and the request goes to the MCP server of that scope: `build_server(open_context(scope))`, built on first use and cached per `(business, agent)`. Each one uses the SDK's `streamable_http_app(stateless_http=True, json_response=True, max_request_body_size=1 MiB, transport_security=<DNS-rebinding check off>)`; the app uses that call's `session_manager` and routes `/mcp` itself. One dedicated asyncio task per scope holds `session_manager.run()` open (anyio task groups must be entered and left in one task); `HTTPApp.lifespan()` (also driven by the ASGI `lifespan` scope under uvicorn) stops those tasks and closes the contexts. Missing, malformed or unknown tokens all get the same `401` body. `run_http` refuses to start without a token and warns on stderr when the bind host is not loopback. Everything runs on the event-loop thread, so the tools stay `async def`.

## `ingatan.cli`

`ingatan` with no subcommand starts the MCP server (OQ-20, `uvx ingatan`). Commands: `serve [--http [--host H] [--port P]]` (stdio by default), `init [--business --agent]`, `setup [--business --agent --semantic/--no-semantic --client --yes]` (init, optional `embeddings enable`, the client's MCP entry, then `doctor`; logic in `cli/setup.py`, the Claude Code file change in `integrations/claude_code/install.py`), `doctor`, `log [--limit N] [--fact ID]`, `diff <log_id>`, `undo <log_id>`, `proposals list|approve <id>|reject <id>`, `forget <fact_id>` (the user's forget, any tier, asks for confirmation unless `--yes`), `ingest <path>`, `reindex [--force]`, `embeddings pull [--model NAME]` (downloads the optional local model), `embeddings enable [--model --global --no-reindex]` (pull, set `provider = "local"`, reindex, probe; with `setup` the only other command that downloads), `embeddings serve [--idle-seconds N]` and `embeddings stop` (the shared model process), `extract [--limit N --dry-run --documents --retry-failed]` (optional fact extraction into the proposal queue), `conflicts review [--apply ACTION --all]` (settle open pairs), `conflicts` (the NFR-9 report), `token add --business --agent [--label]|list|revoke <id>` (HTTP-mode tokens), `hook <event>` (Claude Code hook entry, see `integrations`). Exit code 1 with a one-line message on `IngatanError`.

- Human commands open the context of the configured business and agent: they see `business` and `global` facts and that agent's private entries only. `log`, `diff`, `undo`, `forget` and `conflicts` take `--agent NAME` to open the context as another agent, to audit that agent's private entries. The visibility filter itself is never weakened.
- `hook <event>` always exits 0 and prints nothing of its own. When the hook module is missing or `run` raises, it writes one line to `<data home>/capture-errors.log` if it can (NFR-14). It never exits 2, which would block the agent.

## `evals/`

Not shipped in the wheel. `uv run python -m evals` runs every scenario in `evals/scenarios/*.yaml`-style files (use JSON or TOML so no extra dependency is needed) against the real modules in a temporary `INGATAN_HOME` with a frozen clock, prints recall@5, stale rate, leak rate and tokens per brief, writes `evals/reports/latest.md`, and exits non-zero when leak rate is above 0. Scenario text is Indonesian. `--embeddings local` runs the same scenarios with the local model switched on (it must already be downloaded) and writes `evals/reports/latest-semantic.md`; a scenario with `needs = "embeddings"` is skipped without it.
