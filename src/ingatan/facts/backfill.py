"""Give facts the vector they were saved without, and run the check that waited for it.

With `[embeddings] defer = true`, `remember` saves a fact at once and leaves a row in
`embed_queue` instead of calling the model. `fill_pending` is the other half: it embeds
those facts, stores the vectors and runs the meaning-based conflict check `remember` would
have run with the vector in hand. A pair found this way is recorded like any other and gets
a `conflict` log entry of its own, so it shows in `recall`, in `ingatan conflicts` and in
the conflict report.

Facts that have no vector at all for another reason (saved while embeddings were off,
imported) are embedded as well, the way `ingatan reindex` does it: without a conflict scan,
which stays the job of `ingatan conflicts scan`. A fact that holds a vector of another model
is left as it is. Replacing vectors after a change of model is a decision, and with a paid
provider a bill: that stays `ingatan reindex`.

The model is never called inside a transaction, and a vector is stored only while its fact
still exists, so a `forget` that lands between the model call and the write leaves nothing.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ingatan import audit
from ingatan.db import transaction
from ingatan.facts._util import loads_list, now_iso, record_conflicts
from ingatan.scope import Context, Scope, open_context
from ingatan.search import embed as embed_store
from ingatan.search import is_valid_sql, similar_facts

__all__ = [
    "FillReport",
    "enqueue",
    "fill_pending",
    "last_worker_error",
    "notify",
    "start_worker",
    "stop_worker",
]

WORKER_NAME = "ingatan-backfill"
# A pass that failed as a whole (the database was locked, the provider gave no answer) is
# run again this many seconds later, at most this many times. After that its rows wait in
# the file for the next `notify` or the next server start.
RETRY_SECONDS = 5.0
MAX_RETRIES = 3


@dataclass(frozen=True)
class FillReport:
    """What one pass did. `errors` counts facts the provider returned no vector for."""

    embedded: int
    conflicts_opened: int
    errors: int


def enqueue(conn: sqlite3.Connection, fact_id: str, *, check_conflicts: bool) -> None:
    """Note that `fact_id` was saved without a vector. Must run inside `transaction(conn)`.

    `conn` is the file the fact lives in. `check_conflicts` is what the caller of `remember`
    asked for, carried over to the moment the vector exists.
    """
    conn.execute(
        "INSERT INTO embed_queue (fact_id, check_conflicts, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(fact_id) DO UPDATE SET check_conflicts = excluded.check_conflicts",
        (fact_id, 1 if check_conflicts else 0, now_iso()),
    )


def _drop_dead_rows(ctx: Context, conn: sqlite3.Connection) -> None:
    """Queue rows whose fact is gone, or closed and visible to this caller: nothing to do."""
    closed = "(f.superseded_by IS NOT NULL OR (f.valid_to IS NOT NULL AND f.valid_to <= :now))"
    with transaction(conn):
        conn.execute(
            "DELETE FROM embed_queue WHERE fact_id NOT IN (SELECT id FROM facts) "
            f"OR fact_id IN (SELECT f.id FROM facts f WHERE {closed} "
            f"AND {ctx.visible_sql(conn, alias='f')})",
            {"now": now_iso(), **ctx.params},
        )


def _batch(ctx: Context, conn: sqlite3.Connection, size: int, skip: set[str]) -> list[sqlite3.Row]:
    """The next facts to embed in this file: queued ones first, then any without a vector.

    Each row is a fact row plus `queued_check`: the queue's flag, or NULL for a fact that
    was never queued. A fact that was never queued and holds a vector, of whatever model, is
    not taken.
    """
    where = f"{is_valid_sql('f')} AND {ctx.visible_sql(conn, alias='f')}"
    not_skipped = "f.id NOT IN (SELECT value FROM json_each(:skip))"
    params = {"now": now_iso(), "skip": json.dumps(sorted(skip)), **ctx.params}
    rows = conn.execute(
        "SELECT f.*, q.check_conflicts AS queued_check FROM embed_queue q "
        f"JOIN facts f ON f.id = q.fact_id WHERE {where} AND {not_skipped} "
        "ORDER BY q.created_at ASC, f.seq ASC LIMIT :n",
        {**params, "n": size},
    ).fetchall()
    if len(rows) < size:
        rows += conn.execute(
            f"SELECT f.*, NULL AS queued_check FROM facts f WHERE {where} AND {not_skipped} "
            "AND NOT EXISTS (SELECT 1 FROM fact_embeddings e WHERE e.fact_id = f.id) "
            "AND NOT EXISTS (SELECT 1 FROM embed_queue q WHERE q.fact_id = f.id) "
            "ORDER BY f.seq ASC LIMIT :n",
            {**params, "n": size - len(rows)},
        ).fetchall()
    return rows


def _late_conflicts(
    ctx: Context, conn: sqlite3.Connection, row: sqlite3.Row, vector: Sequence[float]
) -> int:
    """Run the conflict check for a fact whose vector just arrived. Returns pairs opened.

    It is the check `remember` would have run, so it looks with the eyes of the fact's
    writer. The server that does the fill may belong to another agent: that agent's private
    facts are no business of this fact, and the writer's own are.
    """
    fact_id = str(row["id"])
    ctx = Context(dataclasses.replace(ctx.scope, agent=str(row["agent"])), ctx.business, ctx.glob)
    hits = similar_facts(
        ctx,
        str(row["text"]),
        loads_list(str(row["entities"])),
        loads_list(str(row["keywords"])),
        exclude_id=fact_id,
        query_vec=vector,
        kind=str(row["kind"]),
    )
    if not hits:
        return 0
    # Conflict pairs live in the business file, even for a global fact.
    with transaction(ctx.business), transaction(conn):
        # The model ran outside any lock, so the fact may have been forgotten or closed by
        # now. Looked at again here, under the write lock, where nothing can change it: a
        # pair or a log entry for a fact that is gone would be a trace of it.
        still_open = conn.execute(
            f"SELECT 1 FROM facts f WHERE f.id = :id AND {is_valid_sql('f')}",
            {"id": fact_id, "now": now_iso()},
        ).fetchone()
        if still_open is None:
            return 0
        # Pairs the word check opened at write time are skipped: only new ones come back.
        added = record_conflicts(ctx, fact_id, [hit.id for hit in hits])
        if added:
            audit.write(
                conn,
                agent=ctx.scope.agent,
                op="conflict",
                fact_id=fact_id,
                tier=str(row["tier"]),
                owner=str(row["agent"]),
                before={"conflicts_with": []},
                after={"conflicts_with": added, "found": "after the vector arrived"},
            )
    return len(added)


def fill_pending(
    ctx: Context, *, limit: int | None = None, should_stop: Callable[[], bool] | None = None
) -> FillReport:
    """Embed the facts this caller can see that are owed a vector.

    Queued facts come first, oldest first, and get their conflict check; then the facts
    that have no vector at all. `limit` caps how many facts one call takes on. A provider
    that answers nothing for a whole batch ends the pass: the rows stay queued for the next
    one. `should_stop` is asked before every batch, and a true answer ends the pass the same
    way.
    """
    cfg = ctx.scope.embeddings
    model = embed_store.effective_model(cfg)
    if not embed_store.is_enabled(cfg) or not model:
        return FillReport(0, 0, 0)
    embedded = opened = errors = 0
    budget = limit
    for conn in ctx.connections():
        _drop_dead_rows(ctx, conn)
        skip: set[str] = set()
        while budget is None or budget > 0:
            if should_stop is not None and should_stop():
                return FillReport(embedded, opened, errors)
            size = embed_store.REINDEX_BATCH_SIZE
            if budget is not None:
                size = min(size, budget)
            rows = _batch(ctx, conn, size, skip)
            if not rows:
                break
            # The model runs before any write lock is taken, never inside one.
            vectors = embed_store.embed_batch(
                cfg, [str(row["text"]) for row in rows], is_query=False
            )
            if not any(vectors):
                return FillReport(embedded, opened, errors + len(rows))
            for row, vector in zip(rows, vectors, strict=True):
                fact_id = str(row["id"])
                if not vector:
                    errors += 1
                    skip.add(fact_id)
                    continue
                stored = embed_store.store_fact_embedding(
                    conn, fact_id, str(row["text"]), cfg, vector=vector
                )
                if stored:
                    embedded += 1
                    if row["queued_check"] == 1:
                        opened += _late_conflicts(ctx, conn, row, vector)
                else:
                    skip.add(fact_id)
                if row["queued_check"] is not None:
                    with transaction(conn):
                        conn.execute("DELETE FROM embed_queue WHERE fact_id = ?", (fact_id,))
            if budget is not None:
                budget -= len(rows)
    return FillReport(embedded, opened, errors)


# ---- the worker ---------------------------------------------------------------------------
#
# One daemon thread per server process. `remember` wakes it with the scope it wrote in; the
# thread opens its own `Context` for that scope, because an SQLite connection stays on the
# thread that opened it, and runs `fill_pending`. A process without a worker (a CLI command,
# a test) loses nothing: `notify` does nothing there, the queue row stays in the file, and
# the next server to start picks it up.
#
# Each worker has its own stop flag and its own list of scopes. A thread that was told to
# stop while it was inside a model call ends after that batch, whatever was started since,
# and a pass holds `_pass_lock`: a successor started meanwhile waits for that batch instead
# of sending the same queued facts to the model a second time.


class _Worker:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.pending: set[Scope] = set()  # guarded by `_lock`
        self.thread = threading.Thread(target=self._run, name=WORKER_NAME, daemon=True)

    def _run(self) -> None:
        global _last_error
        tries: dict[Scope, int] = {}
        retry_at: dict[Scope, float] = {}
        while not self.stop.is_set():
            due = min(retry_at.values(), default=None)
            self.wake.wait(None if due is None else max(0.0, due - time.monotonic()))
            self.wake.clear()
            now = time.monotonic()
            with _lock:
                ready = self.pending | {scope for scope, at in retry_at.items() if at <= now}
                self.pending.clear()
            for scope in ready:
                if self.stop.is_set():
                    return
                retry_at.pop(scope, None)
                try:
                    with _pass_lock:
                        if self.stop.is_set():
                            return
                        with open_context(scope) as ctx:
                            report = fill_pending(ctx, should_stop=self.stop.is_set)
                except Exception as exc:  # one bad scope must not end the worker
                    message = f"{type(exc).__name__}: {exc}"
                    failed = True
                else:
                    message = ""
                    failed = report.embedded == 0 and report.errors > 0
                if failed and tries.get(scope, 0) < MAX_RETRIES:
                    tries[scope] = tries.get(scope, 0) + 1
                    retry_at[scope] = time.monotonic() + RETRY_SECONDS
                else:
                    tries.pop(scope, None)
                with _lock:
                    _last_error = message


_lock = threading.Lock()
# Held by a worker for the whole of one pass. One process fills with one thread at a time.
_pass_lock = threading.Lock()
_worker: _Worker | None = None
_last_error = ""


def last_worker_error() -> str:
    """Why the worker's most recent pass failed as a whole ("" when it did not)."""
    with _lock:
        return _last_error


def notify(scope: Scope) -> None:
    """Ask the worker to fill what `scope` can see. Does nothing when no worker runs."""
    with _lock:
        worker = _worker
        if worker is None or not worker.thread.is_alive():
            return
        worker.pending.add(scope)
    worker.wake.set()


def start_worker() -> None:
    """Start the worker thread of this process. Safe to call when it already runs."""
    global _worker
    with _lock:
        if _worker is not None and _worker.thread.is_alive():
            return
        _worker = _Worker()
        _worker.thread.start()


def stop_worker(timeout: float = 2.0) -> None:
    """Ask the worker to end and wait up to `timeout` seconds for it.

    A pass that is inside a model call finishes that call and the batch it belongs to, then
    ends, also when the wait here was over sooner. Whatever it did not reach stays queued
    in the database.
    """
    global _worker
    with _lock:
        worker = _worker
        _worker = None
    if worker is None:
        return
    worker.stop.set()
    worker.wake.set()
    worker.thread.join(timeout)
