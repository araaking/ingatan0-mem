"""Filling in vectors after the write, and the conflict check that has to wait for them."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from ingatan import audit, facts, search
from ingatan.cli.conflicts import open_conflicts
from ingatan.db import transaction
from ingatan.facts import backfill
from ingatan.scope import Context, EmbeddingConfig, Scope
from ingatan.search import embed

MakeContext = Callable[..., Context]

EMBED = EmbeddingConfig(provider="local")
# Two statements of one thing that share an entity and no topic word: only a model can tell
# that the second replaces the first (the texts of the eval scenario `semantic-01`).
OLD_RATE = "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler."
NEW_RATE = "Tarif ongkir Bandung sekarang delapan belas ribu rupiah tiap kilogram."
OTHER = "Stok gula di gudang tinggal 20 karung."


@pytest.fixture
def plain(make_ctx: MakeContext) -> Context:
    """No embeddings: what it saves has no vector yet."""
    return make_ctx()


@pytest.fixture
def embed_ctx(make_ctx: MakeContext) -> Context:
    """The same files with the local provider switched on."""
    return make_ctx(embeddings=EMBED)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A stand-in model: shipping-rate texts point one way, everything else another."""
    seen: list[str] = []

    def fake(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        seen.append(text)
        return [1.0, 0.0, 0.0] if "ongkir" in text.casefold() else [0.0, 1.0, 0.0]

    monkeypatch.setattr(embed, "_embed_raw", fake)
    return seen


def save(ctx: Context, text: str, *, queue: bool | None = True, **kw: object) -> str:
    """Remember without a vector. `queue`: True/False enqueue with that conflict flag."""
    kw.setdefault("entities", ["ongkir Bandung"] if "ongkir" in text.casefold() else ["gula"])
    result = facts.remember(ctx, text, **kw)  # type: ignore[arg-type]
    assert result.fact_id is not None
    if queue is not None:
        conn = ctx.conn_for_tier(str(kw.get("tier", "agent")))
        with transaction(conn):
            backfill.enqueue(conn, result.fact_id, check_conflicts=queue)
    return result.fact_id


def vectors(ctx: Context) -> set[str]:
    return {
        str(r[0])
        for conn in ctx.connections()
        for r in conn.execute("SELECT fact_id FROM fact_embeddings")
    }


def queued(ctx: Context) -> set[str]:
    return {
        str(r[0])
        for conn in ctx.connections()
        for r in conn.execute("SELECT fact_id FROM embed_queue")
    }


# ---- the race with forget ---------------------------------------------------------------


def test_vector_is_not_stored_for_a_fact_forgotten_meanwhile(ctx: Context) -> None:
    saved = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])
    assert saved.fact_id is not None
    facts.forget(ctx, saved.fact_id)
    stored = embed.store_fact_embedding(ctx.business, saved.fact_id, "x", EMBED, vector=[1.0, 0.0])
    assert stored is False
    assert ctx.business.execute("SELECT count(*) FROM fact_embeddings").fetchone()[0] == 0


def test_forget_drops_the_queue_row(plain: Context) -> None:
    fact_id = save(plain, OLD_RATE)
    assert queued(plain) == {fact_id}
    facts.forget(plain, fact_id)
    assert queued(plain) == set()


# ---- filling ----------------------------------------------------------------------------


def test_fills_only_facts_without_a_current_vector(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    first = save(plain, OLD_RATE, queue=None)
    second = save(plain, OTHER, queue=None)
    third = save(plain, "Stok tepung di gudang tinggal 20 karung.", queue=None, entities=["tepung"])
    embed.store_fact_embedding(plain.business, first, OLD_RATE, EMBED, vector=[1.0, 0.0, 0.0])
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.errors) == (2, 0)
    assert vectors(embed_ctx) == {first, second, third}
    assert OLD_RATE not in " ".join(model)


def test_a_second_pass_finds_nothing_to_do(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    save(plain, OLD_RATE)
    backfill.fill_pending(embed_ctx)
    assert backfill.fill_pending(embed_ctx) == backfill.FillReport(0, 0, 0)


def test_disabled_embeddings_do_nothing(plain: Context, model: list[str]) -> None:
    fact_id = save(plain, OLD_RATE)
    assert backfill.fill_pending(plain) == backfill.FillReport(0, 0, 0)
    assert queued(plain) == {fact_id}
    assert model == []


def test_limit_caps_the_work_of_one_pass(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    for n in range(3):
        save(plain, f"Stok barang nomor {n} tinggal sedikit.", queue=None, entities=[f"barang {n}"])
    assert backfill.fill_pending(embed_ctx, limit=2).embedded == 2
    assert len(vectors(embed_ctx)) == 2
    assert backfill.fill_pending(embed_ctx).embedded == 1


def test_model_call_is_outside_any_transaction(
    plain: Context, embed_ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        nonlocal calls
        calls += 1
        assert not any(conn.in_transaction for conn in embed_ctx.connections())
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(embed, "_embed_raw", fake)
    save(plain, OLD_RATE)
    save(plain, NEW_RATE)
    assert backfill.fill_pending(embed_ctx).embedded == 2
    assert calls >= 2


def test_other_agents_private_facts_are_left_alone(
    make_ctx: MakeContext, embed_ctx: Context, model: list[str]
) -> None:
    other = make_ctx(agent="hermes")
    private = save(other, OLD_RATE)
    report = backfill.fill_pending(embed_ctx)
    assert report.embedded == 0
    assert private not in vectors(embed_ctx)
    assert queued(embed_ctx) == {private}
    assert model == []


def test_embed_failure_counts_an_error_and_keeps_the_queue_row(
    plain: Context, embed_ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        raise OSError("model file is missing")

    monkeypatch.setattr(embed, "_embed_raw", broken)
    fact_id = save(plain, OLD_RATE)
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.errors) == (0, 1)
    assert queued(embed_ctx) == {fact_id}
    monkeypatch.setattr(embed, "_embed_raw", lambda cfg, text, *, is_query: [1.0, 0.0, 0.0])
    assert backfill.fill_pending(embed_ctx).embedded == 1
    assert queued(embed_ctx) == set()


def test_queue_row_of_a_closed_fact_is_dropped_without_a_vector(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    old = save(plain, OLD_RATE)
    new = save(plain, NEW_RATE, queue=None)
    facts.supersede(plain, old, new)
    backfill.fill_pending(embed_ctx)
    assert vectors(embed_ctx) == {new}
    assert queued(embed_ctx) == set()


# ---- the conflict check that waited for the vector --------------------------------------


def test_late_semantic_conflict_is_recorded(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    old = save(plain, OLD_RATE, keywords=["ongkos kirim", "paket"])
    new = save(plain, NEW_RATE, keywords=["biaya pengiriman", "harga baru"])
    assert open_conflicts(plain) == []  # the word index alone sees no conflict
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.conflicts_opened) == (2, 1)
    pairs = open_conflicts(embed_ctx)
    assert [{p.fact_a, p.fact_b} for p in pairs] == [{old, new}]
    statuses = {hit.id: hit.status for hit in search.recall(embed_ctx, "ongkir Bandung").facts}
    assert statuses == {old: "possible conflict", new: "possible conflict"}
    assert queued(embed_ctx) == set()


def test_late_conflict_is_logged_and_undoable(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    old = save(plain, OLD_RATE)
    new = save(plain, NEW_RATE)
    backfill.fill_pending(embed_ctx)
    entries = [e for e in audit.entries(embed_ctx) if e.op == "conflict"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.fact_id == new
    assert entry.change is not None
    assert entry.change["after"]["conflicts_with"] == [old]
    audit.undo(embed_ctx, entry.id)
    assert open_conflicts(embed_ctx) == []


def test_create_can_still_be_undone_after_a_late_conflict_entry(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    save(plain, OLD_RATE)
    new = save(plain, NEW_RATE)
    backfill.fill_pending(embed_ctx)
    created = next(e for e in audit.entries(embed_ctx) if e.op == "create" and e.fact_id == new)
    audit.undo(embed_ctx, created.id)  # the later `conflict` entry must not block this
    assert open_conflicts(embed_ctx) == []
    assert [hit.id for hit in search.recall(embed_ctx, "ongkir Bandung").facts] != [new]


def test_fact_saved_without_conflict_check_gets_a_vector_but_no_pair(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    old = save(plain, OLD_RATE, queue=False, check_conflicts=False)
    new = save(plain, NEW_RATE, queue=False, check_conflicts=False)
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.conflicts_opened) == (2, 0)
    assert vectors(embed_ctx) == {old, new}
    assert open_conflicts(embed_ctx) == []
    assert queued(embed_ctx) == set()


def test_unqueued_facts_are_filled_without_a_conflict_check(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    # Facts that were never queued (saved while embeddings were off, or imported) get their
    # vector, as `ingatan reindex` gives it, and no conflict scan: that is `conflicts scan`.
    save(plain, OLD_RATE, queue=None)
    save(plain, NEW_RATE, queue=None)
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.conflicts_opened) == (2, 0)
    assert open_conflicts(embed_ctx) == []


def test_a_pair_the_word_index_already_found_is_not_opened_twice(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    first = facts.remember(plain, "Harga kopi susu Rp 15.000.", entities=["kopi susu"])
    second = facts.remember(plain, "Harga kopi susu Rp 18.000.", entities=["kopi susu"])
    assert second.similar, "the word check should already flag this pair"
    for result in (first, second):
        assert result.fact_id is not None
        with transaction(plain.business):
            backfill.enqueue(plain.business, result.fact_id, check_conflicts=True)
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.conflicts_opened) == (2, 0)
    assert len(open_conflicts(embed_ctx)) == 1
    assert [e for e in audit.entries(embed_ctx) if e.op == "conflict"] == []


def test_undo_of_a_late_conflict_leaves_the_facts_other_pairs_open(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    old = save(plain, OLD_RATE)
    # Same wording as NEW_RATE with another value: the word check flags this pair at once.
    twin = save(
        plain, "Tarif ongkir Bandung sekarang dua puluh ribu rupiah tiap kilogram.", queue=None
    )
    new = save(plain, NEW_RATE)
    assert [{p.fact_a, p.fact_b} for p in open_conflicts(plain)] == [{new, twin}]
    backfill.fill_pending(embed_ctx)
    entry = next(e for e in audit.entries(embed_ctx) if e.op == "conflict")
    assert entry.change is not None
    assert (entry.fact_id, entry.change["after"]["conflicts_with"]) == (new, [old])
    audit.undo(embed_ctx, entry.id)
    assert [{p.fact_a, p.fact_b} for p in open_conflicts(embed_ctx)] == [{new, twin}]


# ---- the worker a server runs ------------------------------------------------------------

DEFER = EmbeddingConfig(provider="local", defer=True)


@pytest.fixture
def worker() -> Iterator[None]:
    backfill.start_worker()
    yield
    backfill.stop_worker()


def worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "ingatan-backfill" and t.is_alive()]


def wait_for(condition: Callable[[], bool], seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_notify_without_a_worker_does_nothing(make_ctx: MakeContext, model: list[str]) -> None:
    ctx = make_ctx(embeddings=DEFER)
    result = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])
    backfill.notify(ctx.scope)
    time.sleep(0.1)
    assert worker_threads() == []
    assert vectors(ctx) == set()
    assert queued(ctx) == {result.fact_id}


def test_worker_fills_after_notify(make_ctx: MakeContext, model: list[str], worker: None) -> None:
    ctx = make_ctx(embeddings=DEFER)
    result = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])  # notifies by itself
    assert result.fact_id is not None
    assert wait_for(lambda: vectors(ctx) == {result.fact_id})
    assert wait_for(lambda: queued(ctx) == set())


def test_worker_finds_the_late_conflict(
    make_ctx: MakeContext, model: list[str], worker: None
) -> None:
    ctx = make_ctx(embeddings=DEFER)
    old = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])
    new = facts.remember(ctx, NEW_RATE, entities=["ongkir Bandung"])
    assert new.similar == []  # nothing to report at write time
    assert wait_for(lambda: len(open_conflicts(ctx)) == 1)
    assert {open_conflicts(ctx)[0].fact_a, open_conflicts(ctx)[0].fact_b} == {
        old.fact_id,
        new.fact_id,
    }


def test_worker_catches_up_on_start(make_ctx: MakeContext, model: list[str]) -> None:
    ctx = make_ctx(embeddings=DEFER)
    result = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])  # no worker yet
    assert vectors(ctx) == set()
    backfill.start_worker()
    try:
        backfill.notify(ctx.scope)
        assert wait_for(lambda: vectors(ctx) == {result.fact_id})
    finally:
        backfill.stop_worker()


def test_start_worker_twice_runs_one_thread_and_stop_ends_it() -> None:
    backfill.start_worker()
    backfill.start_worker()
    try:
        assert len(worker_threads()) == 1
    finally:
        backfill.stop_worker()
    assert worker_threads() == []


def test_worker_survives_a_scope_it_cannot_open(
    make_ctx: MakeContext, model: list[str], worker: None, ingatan_home: object
) -> None:
    ctx = make_ctx(embeddings=DEFER)
    broken = Scope(business="Not A Valid Name", agent="x", home=ctx.scope.home, embeddings=DEFER)
    backfill.notify(broken)
    assert wait_for(lambda: backfill.last_worker_error() != "")
    result = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])
    assert wait_for(lambda: vectors(ctx) == {result.fact_id})
    assert len(worker_threads()) == 1


# ---- found by the final review: forget during the late check --------------------------------


def test_forget_during_the_late_check_leaves_no_pair_and_no_log_entry(
    plain: Context, embed_ctx: Context, model: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    old = save(plain, OLD_RATE, queue=None)
    embed.store_fact_embedding(plain.business, old, OLD_RATE, EMBED, vector=[1.0, 0.0, 0.0])
    new = save(plain, NEW_RATE)
    real = backfill.similar_facts

    def forgotten_meanwhile(*args: Any, **kwargs: Any) -> Any:
        hits = real(*args, **kwargs)
        # Another process forgets the fact after its vector was stored and its look-alikes
        # were found, just before the pair would be written.
        facts.forget(plain, new)
        return hits

    monkeypatch.setattr(backfill, "similar_facts", forgotten_meanwhile)
    report = backfill.fill_pending(embed_ctx)
    assert report.conflicts_opened == 0
    assert plain.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0
    assert [e for e in audit.entries(plain) if e.op == "conflict"] == []
    assert vectors(plain) == {old}
    assert queued(plain) == set()


# ---- minor findings of the final review ---------------------------------------------------

OTHER_MODEL = EmbeddingConfig(provider="local", model="intfloat/multilingual-e5-small")


def stock(n: int) -> str:
    return f"Stok barang nomor {n} tinggal sedikit."


def test_a_pass_ends_between_batches_when_asked_to_stop(
    plain: Context, embed_ctx: Context, model: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embed, "REINDEX_BATCH_SIZE", 1)
    for n in range(3):
        save(plain, stock(n), entities=[f"barang {n}"])
    report = backfill.fill_pending(embed_ctx, should_stop=lambda: len(model) >= 1)
    assert report.embedded == 1
    assert len(queued(embed_ctx)) == 2  # nothing is lost: the rest waits in the file


def test_a_worker_told_to_stop_ends_and_does_not_run_beside_its_successor(
    make_ctx: MakeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx(embeddings=DEFER)
    monkeypatch.setattr(embed, "REINDEX_BATCH_SIZE", 1)
    entered, release = threading.Event(), threading.Event()

    def slow(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        entered.set()
        release.wait(5)
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(embed, "_embed_raw", slow)
    backfill.start_worker()
    try:
        old = worker_threads()[0]
        for n in range(3):
            facts.remember(ctx, stock(n), entities=[f"barang {n}"])
        assert entered.wait(3)
        backfill.stop_worker(timeout=0.05)  # gives up waiting: the thread is inside the model
        assert old.is_alive()
        backfill.start_worker()
        release.set()
        old.join(3)
        assert not old.is_alive(), "a stopped thread ends after the batch it was in"
        assert len(worker_threads()) == 1
        assert len(vectors(ctx)) + len(queued(ctx)) == 3
    finally:
        release.set()
        backfill.stop_worker()


def test_a_pass_that_fails_is_tried_again_without_another_notify(
    make_ctx: MakeContext, model: list[str], monkeypatch: pytest.MonkeyPatch, worker: None
) -> None:
    ctx = make_ctx(embeddings=DEFER)
    real = backfill.fill_pending
    calls = 0

    def locked_once(inner: Context, **kwargs: Any) -> backfill.FillReport:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(inner, **kwargs)

    monkeypatch.setattr(backfill, "fill_pending", locked_once)
    monkeypatch.setattr(backfill, "RETRY_SECONDS", 0.05, raising=False)
    result = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])  # the only notify
    assert wait_for(lambda: vectors(ctx) == {result.fact_id})
    assert wait_for(lambda: backfill.last_worker_error() == "")


def test_a_provider_that_answers_nothing_is_asked_again_without_another_notify(
    make_ctx: MakeContext, monkeypatch: pytest.MonkeyPatch, worker: None
) -> None:
    ctx = make_ctx(embeddings=DEFER)
    calls = 0

    def down_once(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("the model process is still starting")
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(embed, "_embed_raw", down_once)
    monkeypatch.setattr(backfill, "RETRY_SECONDS", 0.05, raising=False)
    result = facts.remember(ctx, OLD_RATE, entities=["ongkir Bandung"])
    assert wait_for(lambda: vectors(ctx) == {result.fact_id})


def test_a_scope_that_keeps_failing_is_given_up_after_a_few_tries(
    make_ctx: MakeContext, model: list[str], monkeypatch: pytest.MonkeyPatch, worker: None
) -> None:
    ctx = make_ctx(embeddings=DEFER)
    calls = 0

    def always_locked(inner: Context, **kwargs: Any) -> backfill.FillReport:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(backfill, "fill_pending", always_locked)
    monkeypatch.setattr(backfill, "RETRY_SECONDS", 0.02, raising=False)
    backfill.notify(ctx.scope)
    tries = 1 + backfill.MAX_RETRIES
    assert wait_for(lambda: calls == tries)
    time.sleep(0.3)
    assert calls == tries  # and then it waits for the next notify
    assert "locked" in backfill.last_worker_error()
    assert len(worker_threads()) == 1


def test_the_late_check_sees_what_the_writer_of_the_fact_sees(
    make_ctx: MakeContext, embed_ctx: Context, model: list[str]
) -> None:
    """Another agent's server does the fill; the check is the one `remember` would have run."""
    writer = make_ctx(agent="hermes")
    old = save(writer, OLD_RATE, queue=None)  # private to the writer
    embed.store_fact_embedding(writer.business, old, OLD_RATE, EMBED, vector=[1.0, 0.0, 0.0])
    new = save(writer, NEW_RATE, tier="business")
    report = backfill.fill_pending(embed_ctx)
    assert report.conflicts_opened == 1
    as_writer = make_ctx(agent="hermes", embeddings=EMBED)
    assert [{p.fact_a, p.fact_b} for p in open_conflicts(as_writer)] == [{old, new}]
    entry = next(e for e in audit.entries(as_writer) if e.op == "conflict")
    assert (entry.agent, entry.fact_id) == ("hermes", new)


def test_the_late_check_does_not_pair_a_shared_fact_with_a_private_fact_of_the_filling_agent(
    make_ctx: MakeContext, plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    mine = save(plain, OLD_RATE, queue=None)  # private to the agent whose server fills
    embed.store_fact_embedding(plain.business, mine, OLD_RATE, EMBED, vector=[1.0, 0.0, 0.0])
    writer = make_ctx(agent="hermes")
    save(writer, NEW_RATE, tier="business")
    report = backfill.fill_pending(embed_ctx)
    assert (report.embedded, report.conflicts_opened) == (1, 0)
    assert plain.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0
    assert [e for e in audit.entries(plain) if e.op == "conflict"] == []


def test_a_vector_of_another_model_is_left_for_reindex(
    plain: Context, embed_ctx: Context, model: list[str]
) -> None:
    """The catch-up gives a vector to a fact that has none. Replacing one is `reindex`."""
    fact_id = save(plain, OLD_RATE, queue=None)
    embed.store_fact_embedding(plain.business, fact_id, OLD_RATE, OTHER_MODEL, vector=[1.0, 0.0])
    assert backfill.fill_pending(embed_ctx) == backfill.FillReport(0, 0, 0)
    assert model == []
    stored = plain.business.execute(
        "SELECT model FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    assert stored[0] == "intfloat/multilingual-e5-small"
