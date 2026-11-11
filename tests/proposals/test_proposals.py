from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from ingatan import audit, facts, proposals, search
from ingatan.errors import NotFoundError, ValidationError
from ingatan.scope import Context

MakeContext = Callable[..., Context]


def propose(ctx: Context, text: str, tier: str = "business", **kw: Any) -> str:
    result = facts.remember(ctx, text, keywords=kw.pop("keywords", ["x"]), tier=tier, **kw)
    assert result.proposal_id is not None
    return result.proposal_id


def row(ctx: Context, fact_id: str) -> Any:
    for conn in ctx.connections():
        found = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if found is not None:
            return found
    raise AssertionError(fact_id)


def count(conn: Any, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# ---- the queue (FR-21, FR-23, FR-26) ------------------------------------------------


def test_business_remember_returns_a_proposal_id_and_creates_no_fact(ctx: Context) -> None:
    result = facts.remember(ctx, "Retur maksimal 7 hari", keywords=["retur"], tier="business")
    assert result.proposal_id and result.fact_id is None
    assert count(ctx.business, "facts") == 0 and count(ctx.glob, "facts") == 0
    assert search.recall(ctx, "retur").facts == []


def test_pending_text_is_never_returned_by_recall(ctx: Context) -> None:
    propose(ctx, "Teks rahasia zzusulanzz", keywords=["zzusulanzz"])
    propose(ctx, "Teks global zzusulanzz", "global", keywords=["zzusulanzz"])
    for query in ("zzusulanzz", "Teks rahasia", "global"):
        result = search.recall(ctx, query, include_closed=True)
        assert result.facts == [] and result.notes == []


def test_pending_lists_both_files_oldest_first_with_the_writing_agent(
    make_ctx: MakeContext, frozen_clock: Any
) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    first = propose(a, "Satu", "global")
    frozen_clock.advance(minutes=1)
    second = propose(b, "Dua", "business")
    frozen_clock.advance(minutes=1)
    third = propose(a, "Tiga", "business")
    queue = proposals.pending(a)
    assert [p.id for p in queue] == [first, second, third]
    assert [(p.tier, p.agent, p.text, p.status) for p in queue] == [
        ("global", "agent-a", "Satu", "pending"),
        ("business", "agent-b", "Dua", "pending"),
        ("business", "agent-a", "Tiga", "pending"),
    ]
    assert queue[0].created_at == "2026-11-11T09:00:00Z"
    assert proposals.pending_count(a) == 3 == proposals.pending_count(b)


def test_pending_count_drops_when_a_proposal_is_decided(ctx: Context) -> None:
    one = propose(ctx, "Satu")
    two = propose(ctx, "Dua", "global")
    assert proposals.pending_count(ctx) == 2
    proposals.approve(ctx, one)
    assert proposals.pending_count(ctx) == 1
    proposals.reject(ctx, two)
    assert proposals.pending_count(ctx) == 0 and proposals.pending(ctx) == []


def test_business_x_does_not_see_the_pending_proposals_of_business_y(
    make_ctx: MakeContext,
) -> None:
    x = make_ctx(business="usaha-x")
    y = make_ctx(business="usaha-y")
    propose(y, "Usulan Y")
    assert proposals.pending(x) == [] and proposals.pending_count(x) == 0


# ---- approve (FR-24) ----------------------------------------------------------------


@pytest.mark.parametrize(("tier", "file"), [("business", "business"), ("global", "glob")])
def test_approve_creates_the_fact_in_the_requested_tier(
    make_ctx: MakeContext, frozen_clock: Any, tier: str, file: str
) -> None:
    writer = make_ctx(agent="agent-a")
    user = make_ctx(agent="owner")
    pid = propose(
        writer,
        "Aturan retur 7 hari",
        tier,
        keywords=["retur"],
        entities=["Retur"],
        kind="stable",
    )
    frozen_clock.advance(days=2)
    fact_id = proposals.approve(user, pid)
    created = row(user, fact_id)
    conn = getattr(user, file)
    assert conn.execute("SELECT count(*) FROM facts WHERE id = ?", (fact_id,)).fetchone()[0] == 1
    assert created["tier"] == tier
    assert created["agent"] == "agent-a"
    assert json.loads(created["source"])["agent"] == "agent-a"
    assert created["valid_from"] == "2026-11-13T09:00:00Z"
    assert json.loads(created["entities"]) == ["Retur"] and created["kind"] == "stable"
    assert [h.id for h in search.recall(writer, "retur").facts] == [fact_id]
    decided = conn.execute("SELECT * FROM proposals WHERE id = ?", (pid,)).fetchone()
    assert (decided["status"], decided["fact_id"]) == ("approved", fact_id)
    assert decided["decided_at"] == "2026-11-13T09:00:00Z"


def test_approved_volatile_fact_gets_its_review_date_from_the_approval(
    ctx: Context, frozen_clock: Any
) -> None:
    pid = propose(ctx, "Harga bahan 50 ribu", keywords=["harga"])
    frozen_clock.advance(days=5)
    fact_id = proposals.approve(ctx, pid)
    created = row(ctx, fact_id)
    assert created["kind"] == "volatile" and created["review_interval_days"] == 30
    assert created["review_after"] == "2026-12-16T09:00:00Z"  # approval day + 30


def test_approve_logs_the_proposal_id_the_created_fact_and_what_was_closed(
    ctx: Context,
) -> None:
    old_prop = propose(ctx, "Jam buka 09.00", keywords=["jam"])
    old = proposals.approve(ctx, old_prop)
    mine = facts.remember(ctx, "Jam buka 08.00", keywords=["jam"]).fact_id
    assert mine
    queued = facts.supersede(ctx, old, mine)
    new = proposals.approve(ctx, queued.proposal_id or "")
    entry = next(e for e in audit.entries(ctx, fact_id=new) if e.op == "approve")
    assert (entry.tier, entry.owner, entry.agent) == ("business", "claude-code", "claude-code")
    assert entry.change is not None and entry.change["before"] is None
    after = entry.change["after"]
    assert after["proposal_id"] == queued.proposal_id
    assert after["id"] == new and after["text"] == "Jam buka 08.00"
    assert after["keywords"] == ["jam"] and after["tier"] == "business"
    assert after["closed"]["fact_id"] == old
    assert after["closed"]["before"] == {
        "valid_to": None,
        "superseded_by": None,
        "pending_proposal": queued.proposal_id,
    }
    assert after["closed"]["after"]["superseded_by"] == new
    assert after["closed"]["after"]["pending_proposal"] is None
    assert after["affects"] == [old]


def test_approve_closes_the_superseded_fact_and_clears_check_first(ctx: Context) -> None:
    old = proposals.approve(ctx, propose(ctx, "Jam buka 09.00", keywords=["jam"]))
    mine = facts.remember(ctx, "Jam buka 08.00", keywords=["jam"]).fact_id
    assert mine
    queued = facts.supersede(ctx, old, mine)
    assert {h.id: h.status for h in search.recall(ctx, "jam buka").facts}[old] == "check first"
    new = proposals.approve(ctx, queued.proposal_id or "")
    closed = row(ctx, old)
    assert closed["superseded_by"] == new and closed["valid_to"] is not None
    assert closed["pending_proposal"] is None
    visible = {h.id: h.status for h in search.recall(ctx, "jam buka").facts}
    assert old not in visible and visible[new] in ("valid", "possible conflict")
    everything = {h.id: h.status for h in search.recall(ctx, "jam buka", include_closed=True).facts}
    assert everything[old] == "superseded"


def test_approve_closes_a_global_fact_in_the_global_file(ctx: Context) -> None:
    old = proposals.approve(ctx, propose(ctx, "Pemilik di Medan", "global", keywords=["pemilik"]))
    queued = facts.supersede(
        ctx, old, propose(ctx, "Pemilik di Bandung", "global", keywords=["pemilik"])
    )
    new = proposals.approve(ctx, queued.proposal_id or "")
    assert (
        ctx.glob.execute("SELECT superseded_by FROM facts WHERE id = ?", (old,)).fetchone()[0]
        == new
    )
    assert count(ctx.business, "facts") == 0


def test_approve_records_conflict_pairs_from_the_similar_list(ctx: Context) -> None:
    mine = facts.remember(
        ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"]
    ).fact_id
    pid = propose(ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"])
    new = proposals.approve(ctx, pid)
    pair = ctx.business.execute("SELECT * FROM conflicts").fetchall()
    assert len(pair) == 1 and {pair[0]["fact_a"], pair[0]["fact_b"]} == {mine, new}
    assert {h.status for h in search.recall(ctx, "supplier Budi").facts} == {"possible conflict"}


def test_approve_skips_similar_facts_that_no_longer_exist(ctx: Context) -> None:
    mine = facts.remember(
        ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"]
    ).fact_id
    assert mine
    pid = propose(ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"])
    facts.forget(ctx, mine)
    proposals.approve(ctx, pid)
    assert count(ctx.business, "conflicts") == 0


def test_approve_unknown_or_decided_proposals_fail(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        proposals.approve(ctx, "p_000000000000")
    pid = propose(ctx, "Satu")
    proposals.approve(ctx, pid)
    facts_before = count(ctx.business, "facts")
    log_before = count(ctx.business, "log")
    with pytest.raises(ValidationError, match="already approved"):
        proposals.approve(ctx, pid)
    with pytest.raises(ValidationError, match="already approved"):
        proposals.reject(ctx, pid)
    assert count(ctx.business, "facts") == facts_before
    assert count(ctx.business, "log") == log_before


def test_approve_does_not_close_a_target_that_was_closed_meanwhile(ctx: Context) -> None:
    old = proposals.approve(ctx, propose(ctx, "Jam buka 09.00", keywords=["jam"]))
    mine = facts.remember(ctx, "Jam buka 08.00", keywords=["jam"]).fact_id
    assert mine
    queued = facts.supersede(ctx, old, mine)
    facts.forget(ctx, old, by_user=True)  # the target is gone before the user decides
    new = proposals.approve(ctx, queued.proposal_id or "")
    assert row(ctx, new)["valid_to"] is None


# ---- reject (FR-25) -----------------------------------------------------------------


def test_reject_creates_no_fact_and_logs_it(ctx: Context) -> None:
    pid = propose(ctx, "Usulan ditolak")
    proposals.reject(ctx, pid)
    assert count(ctx.business, "facts") == 0
    assert search.recall(ctx, "usulan").facts == []
    stored = ctx.business.execute("SELECT status, decided_at FROM proposals").fetchone()
    assert stored["status"] == "rejected" and stored["decided_at"] is not None
    ops = [e.op for e in audit.entries(ctx, fact_id=pid)]
    assert ops == ["propose", "reject"]


def test_reject_clears_the_pending_mark_on_the_fact_it_wanted_to_replace(ctx: Context) -> None:
    old = proposals.approve(ctx, propose(ctx, "Jam buka 09.00", keywords=["jam"]))
    mine = facts.remember(ctx, "Jam buka 08.00", keywords=["jam"]).fact_id
    assert mine
    queued = facts.supersede(ctx, old, mine)
    assert row(ctx, old)["pending_proposal"] == queued.proposal_id
    proposals.reject(ctx, queued.proposal_id or "")
    after = row(ctx, old)
    assert after["pending_proposal"] is None and after["valid_to"] is None
    entry = audit.entries(ctx, fact_id=queued.proposal_id)[-1]
    assert entry.op == "reject" and entry.change is not None
    assert entry.change["after"]["affects"] == [old]
    assert "reject" in [e.op for e in facts.history(ctx, old)]
    # The old fact can be replaced again afterwards.
    again = facts.remember(ctx, "Jam buka 07.00", keywords=["jam"]).fact_id
    assert again and facts.supersede(ctx, old, again).closed is False


def test_reject_leaves_a_fact_with_another_pending_mark_alone(ctx: Context) -> None:
    old = proposals.approve(ctx, propose(ctx, "Jam buka 09.00", keywords=["jam"]))
    first = propose(ctx, "Jam buka 08.00", keywords=["jam"])
    facts.supersede(ctx, old, first)
    # Confirming cleared nothing here; a stray proposal pointing at old must not clear it.
    other = propose(ctx, "Jam buka 07.00", keywords=["jam"])
    ctx.business.execute("UPDATE proposals SET supersedes = ? WHERE id = ?", (old, other))
    proposals.reject(ctx, other)
    assert row(ctx, old)["pending_proposal"] == first


def test_reject_unknown_proposal_fails(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        proposals.reject(ctx, "p_000000000000")
