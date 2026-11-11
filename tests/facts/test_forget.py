from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from ingatan import audit, facts, proposals, search
from ingatan.errors import NotFoundError, ScopeError
from ingatan.scope import Context

MakeContext = Callable[..., Context]

TEXT_MARK = "zzuniktekszz"
ENTITY_MARK = "zzunikentzz"
KEYWORD_MARK = "zzunikkwzz"
REASON_MARK = "zzunikalasanzz"
ALL_MARKS = [TEXT_MARK, ENTITY_MARK, KEYWORD_MARK, REASON_MARK]


def raw_bytes(home: Path, name: str) -> bytes:
    """The database file, its WAL and its shared-memory file, as one blob."""
    blob = b""
    for suffix in ("", "-wal", "-shm"):
        path = home / f"{name}.db{suffix}"
        if path.exists():
            blob += path.read_bytes()
    return blob


def encodings(marker: str) -> list[bytes]:
    return [marker.encode("utf-8"), marker.encode("utf-16-le"), marker.encode("utf-16-be")]


def present(blob: bytes, marks: list[str]) -> list[str]:
    return [m for m in marks if any(enc in blob for enc in encodings(m))]


def store_and_touch(ctx: Context, tier_flow: str) -> str:
    """Store a fact carrying all markers and write several log entries about it."""
    if tier_flow == "agent":
        result = facts.remember(
            ctx,
            f"Kode gudang {TEXT_MARK} ada di rak belakang",
            keywords=[KEYWORD_MARK, "gudang"],
            entities=[ENTITY_MARK],
            kind="volatile",
        )
        fact_id = result.fact_id
    else:
        result = facts.remember(
            ctx,
            f"Kode gudang {TEXT_MARK} ada di rak belakang",
            keywords=[KEYWORD_MARK, "gudang"],
            entities=[ENTITY_MARK],
            tier=tier_flow,
        )
        assert result.proposal_id is not None
        fact_id = proposals.approve(ctx, result.proposal_id)
    assert fact_id is not None
    facts.confirm(ctx, fact_id, f"dicek ulang, {REASON_MARK}")
    search.recall(ctx, TEXT_MARK)
    return fact_id


def test_the_markers_really_are_on_disk_before_forget(ctx: Context, ingatan_home: Path) -> None:
    store_and_touch(ctx, "agent")
    blob = raw_bytes(ingatan_home, "tokoku")
    # Guards the test itself: with nothing on disk the later assertions would prove nothing.
    assert set(present(blob, ALL_MARKS)) == set(ALL_MARKS)


@pytest.mark.parametrize("second_context_open", [False, True])
def test_forget_agent_fact_leaves_no_bytes(
    make_ctx: MakeContext, ingatan_home: Path, second_context_open: bool
) -> None:
    ctx = make_ctx()
    other = make_ctx() if second_context_open else None
    fact_id = store_and_touch(ctx, "agent")
    # A second fact superseding the first adds a supersede entry, then it is forgotten too.
    newer = facts.remember(ctx, "Kode gudang baru", keywords=["gudang"], entities=["gudang"])
    assert newer.fact_id is not None
    facts.supersede(ctx, fact_id, newer.fact_id)
    if other is not None:
        search.recall(other, "gudang")

    facts.forget(ctx, fact_id)

    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []
    ctx.close()
    if other is not None:
        assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []
        other.close()
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []


@pytest.mark.parametrize("second_context_open", [False, True])
def test_forget_global_fact_by_user_leaves_no_bytes(
    make_ctx: MakeContext, ingatan_home: Path, second_context_open: bool
) -> None:
    ctx = make_ctx()
    other = make_ctx(agent="hermes") if second_context_open else None
    fact_id = store_and_touch(ctx, "global")
    assert present(raw_bytes(ingatan_home, "global"), ALL_MARKS)

    facts.forget(ctx, fact_id, by_user=True)

    assert present(raw_bytes(ingatan_home, "global"), ALL_MARKS) == []
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []
    ctx.close()
    if other is not None:
        assert present(raw_bytes(ingatan_home, "global"), ALL_MARKS) == []
        other.close()
    assert present(raw_bytes(ingatan_home, "global"), ALL_MARKS) == []
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []


def test_forget_business_fact_by_user_leaves_no_bytes(
    make_ctx: MakeContext, ingatan_home: Path
) -> None:
    ctx = make_ctx()
    fact_id = store_and_touch(ctx, "business")
    facts.forget(ctx, fact_id, by_user=True)
    ctx.close()
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []
    assert present(raw_bytes(ingatan_home, "global"), ALL_MARKS) == []


def test_forget_scrubs_the_proposal_that_created_the_fact(ctx: Context, ingatan_home: Path) -> None:
    result = facts.remember(
        ctx, f"Rahasia {TEXT_MARK}", keywords=["rahasia"], entities=[ENTITY_MARK], tier="business"
    )
    assert result.proposal_id is not None
    fact_id = proposals.approve(ctx, result.proposal_id)
    facts.forget(ctx, fact_id, by_user=True)
    row = ctx.business.execute(
        "SELECT payload, status FROM proposals WHERE id = ?", (result.proposal_id,)
    ).fetchone()
    assert TEXT_MARK not in row["payload"]
    assert row["status"] == "approved"
    entries = ctx.business.execute(
        "SELECT change FROM log WHERE fact_id = ?", (result.proposal_id,)
    ).fetchall()
    assert entries
    assert all(e["change"] is None for e in entries)
    ctx.close()
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []


def test_forget_scrubs_a_pending_copy_made_by_supersede_and_rejects_it(
    ctx: Context, ingatan_home: Path
) -> None:
    old = facts.remember(ctx, "Harga lama", keywords=["harga"], tier="business")
    assert old.proposal_id is not None
    old_id = proposals.approve(ctx, old.proposal_id)
    mine = facts.remember(
        ctx, f"Harga baru {TEXT_MARK}", keywords=["harga"], entities=[ENTITY_MARK]
    )
    assert mine.fact_id is not None
    queued = facts.supersede(ctx, old_id, mine.fact_id)
    assert queued.proposal_id is not None
    assert [p.id for p in proposals.pending(ctx)] == [queued.proposal_id]

    facts.forget(ctx, mine.fact_id)

    assert proposals.pending_count(ctx) == 0
    # The old fact no longer waits for a proposal that is gone.
    row = ctx.business.execute(
        "SELECT pending_proposal FROM facts WHERE id = ?", (old_id,)
    ).fetchone()
    assert row["pending_proposal"] is None
    assert search.recall(ctx, "harga").facts[0].status == "valid"
    ctx.close()
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []


def test_pending_proposal_that_supersedes_a_forgotten_fact_keeps_no_dangling_link(
    ctx: Context,
) -> None:
    old = facts.remember(ctx, "Tarif lama", keywords=["tarif"], tier="business")
    assert old.proposal_id is not None
    old_id = proposals.approve(ctx, old.proposal_id)
    new = facts.remember(ctx, "Tarif baru", keywords=["tarif"], tier="business")
    assert new.proposal_id is not None
    facts.supersede(ctx, old_id, new.proposal_id)

    facts.forget(ctx, old_id, by_user=True)

    row = ctx.business.execute(
        "SELECT supersedes, status FROM proposals WHERE id = ?", (new.proposal_id,)
    ).fetchone()
    assert row["supersedes"] is None
    assert row["status"] == "pending"
    # It can still be approved, as a plain new fact.
    fact_id = proposals.approve(ctx, new.proposal_id)
    assert search.recall(ctx, "tarif").facts[0].id == fact_id


def test_after_forget_nothing_returns_content_and_history_is_content_free(
    ctx: Context,
) -> None:
    fact_id = store_and_touch(ctx, "agent")
    facts.forget(ctx, fact_id)

    assert search.recall(ctx, TEXT_MARK, include_closed=True).facts == []
    assert search.recall(ctx, ENTITY_MARK, include_closed=True).facts == []
    assert facts.history(ctx, ENTITY_MARK) == []
    entries = facts.history(ctx, fact_id)
    assert entries
    assert all(e.change is None for e in entries)
    assert [e.op for e in entries].count("forget") == 1
    assert [e.op for e in entries] == ["create", "confirm", "forget"]
    assert ctx.business.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == 0
    assert ctx.business.execute("SELECT count(*) FROM facts_tri").fetchone()[0] == 0
    last = entries[-1]
    assert (last.fact_id, last.agent, last.tier, last.owner) == (
        fact_id,
        "claude-code",
        "agent",
        "claude-code",
    )


def test_forget_drops_conflict_pairs_and_the_other_fact_loses_its_flag(ctx: Context) -> None:
    first = facts.remember(
        ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"]
    )
    second = facts.remember(
        ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"]
    )
    assert first.fact_id and second.fact_id
    assert search.recall(ctx, "Selasa").facts[0].status == "possible conflict"
    facts.forget(ctx, first.fact_id)
    assert ctx.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0
    assert search.recall(ctx, "Selasa").facts[0].status == "valid"


def test_forgetting_a_superseding_fact_leaves_the_older_one_closed(ctx: Context) -> None:
    old = facts.remember(ctx, "Alamat toko Jalan Mawar", keywords=["alamat"])
    new = facts.remember(ctx, "Alamat toko Jalan Melati", keywords=["alamat"])
    assert old.fact_id and new.fact_id
    facts.supersede(ctx, old.fact_id, new.fact_id)
    facts.forget(ctx, new.fact_id)
    row = ctx.business.execute(
        "SELECT valid_to, superseded_by FROM facts WHERE id = ?", (old.fact_id,)
    ).fetchone()
    assert row["superseded_by"] == new.fact_id  # OQ-17: stays closed, link kept
    assert row["valid_to"] is not None
    assert search.recall(ctx, "alamat").facts == []


def test_agent_cannot_forget_business_or_global_facts(make_ctx: MakeContext) -> None:
    ctx = make_ctx()
    biz = facts.remember(ctx, "Jam buka 09.00", keywords=["jam"], tier="business")
    glo = facts.remember(ctx, "Pemilik bernama Budi", keywords=["pemilik"], tier="global")
    assert biz.proposal_id and glo.proposal_id
    biz_id = proposals.approve(ctx, biz.proposal_id)
    glo_id = proposals.approve(ctx, glo.proposal_id)
    log_counts = [
        c.execute("SELECT count(*) FROM log").fetchone()[0] for c in (ctx.business, ctx.glob)
    ]
    for fact_id in (biz_id, glo_id):
        with pytest.raises(ScopeError):
            facts.forget(ctx, fact_id)
    assert [
        c.execute("SELECT count(*) FROM log").fetchone()[0] for c in (ctx.business, ctx.glob)
    ] == log_counts
    assert search.recall(ctx, "jam").facts and search.recall(ctx, "pemilik").facts
    # The user can.
    facts.forget(ctx, biz_id, by_user=True)
    facts.forget(ctx, glo_id, by_user=True)
    assert search.recall(ctx, "jam").facts == [] and search.recall(ctx, "pemilik").facts == []


def test_forget_unknown_id_raises_not_found(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        facts.forget(ctx, "f_000000000000")


def test_forget_entry_is_content_free_and_log_cannot_be_undone(ctx: Context) -> None:
    fact_id = store_and_touch(ctx, "agent")
    facts.forget(ctx, fact_id)
    forget_entry = next(e for e in audit.entries(ctx, fact_id=fact_id) if e.op == "forget")
    assert forget_entry.change is None


def test_forget_works_while_a_reader_connection_is_idle(
    make_ctx: MakeContext, ingatan_home: Path
) -> None:
    ctx = make_ctx()
    other = make_ctx()
    fact_id = store_and_touch(ctx, "agent")
    # The other connection read the row (so it may cache pages) and stays open.
    assert search.recall(other, TEXT_MARK).facts
    facts.forget(ctx, fact_id)
    assert search.recall(other, TEXT_MARK, include_closed=True).facts == []
    raw = sqlite3.connect(str(ingatan_home / "tokoku.db"))
    try:
        assert raw.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    finally:
        raw.close()
    assert present(raw_bytes(ingatan_home, "tokoku"), ALL_MARKS) == []
