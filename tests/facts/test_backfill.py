"""Filling in vectors after the write, and the conflict check that has to wait for them."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ingatan import audit, facts, search
from ingatan.cli.conflicts import open_conflicts
from ingatan.db import transaction
from ingatan.facts import backfill
from ingatan.scope import Context, EmbeddingConfig
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
