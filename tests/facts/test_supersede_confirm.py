from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ingatan import audit, facts, proposals, search
from ingatan.errors import NotFoundError, ValidationError
from ingatan.scope import Context

MakeContext = Callable[..., Context]


def rem(ctx: Context, text: str, keywords: list[str] | None = None, **kw: Any) -> str:
    result = facts.remember(ctx, text, keywords=keywords or ["x"], **kw)
    assert result.fact_id is not None
    return result.fact_id


def approved(ctx: Context, text: str, tier: str = "business", **kw: Any) -> str:
    result = facts.remember(ctx, text, keywords=kw.pop("keywords", ["x"]), tier=tier, **kw)
    assert result.proposal_id is not None
    return proposals.approve(ctx, result.proposal_id)


def row(ctx: Context, fact_id: str) -> Any:
    for conn in ctx.connections():
        found = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if found is not None:
            return found
    raise AssertionError(fact_id)


def snapshot(ctx: Context) -> list[Any]:
    out: list[Any] = []
    for conn in ctx.connections():
        for table in ("facts", "log", "conflicts", "proposals"):
            out.append([tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")])
    return out


def status(ctx: Context, query: str) -> dict[str, str]:
    return {h.id: h.status for h in search.recall(ctx, query, touch=False).facts}


# ---- supersede, agent facts (FR-12) -------------------------------------------------


def test_supersede_closes_the_old_fact_and_links_the_new_one(
    ctx: Context, frozen_clock: Any
) -> None:
    old = rem(ctx, "Harga kopi 50 ribu", ["harga"])
    new = rem(ctx, "Harga kopi 55 ribu", ["harga"])
    frozen_clock.advance(hours=1)
    result = facts.supersede(ctx, old, new)
    assert (result.old_id, result.new_id, result.closed, result.proposal_id) == (
        old,
        new,
        True,
        None,
    )
    closed = row(ctx, old)
    assert closed["valid_to"] == "2026-11-11T10:00:00Z" and closed["superseded_by"] == new
    assert row(ctx, new)["valid_to"] is None


def test_after_supersede_default_recall_excludes_the_old_fact_and_history_shows_it(
    ctx: Context,
) -> None:
    old = rem(ctx, "Harga kopi 50 ribu", ["harga"], entities=["Kopi"])
    new = rem(ctx, "Harga kopi 55 ribu", ["harga"], entities=["Kopi"])
    facts.supersede(ctx, old, new)
    assert [h.id for h in search.recall(ctx, "harga kopi").facts] == [new]
    assert old in [h.id for h in search.recall(ctx, "harga kopi", include_closed=True).facts]
    ops = [(e.fact_id, e.op) for e in facts.history(ctx, old)]
    assert ops == [(old, "create"), (old, "supersede")]
    assert {e.fact_id for e in facts.history(ctx, "kopi")} == {old, new}


def test_supersede_logs_before_and_after_validity_and_the_new_id(
    ctx: Context, frozen_clock: Any
) -> None:
    old = rem(ctx, "Harga kopi 50 ribu", ["harga"], entities=["Kopi"])
    new = rem(ctx, "Harga kopi 55 ribu", ["harga"], entities=["Kopi"])
    facts.supersede(ctx, old, new)
    entry = audit.entries(ctx, fact_id=old)[-1]
    assert (entry.op, entry.tier, entry.owner) == ("supersede", "agent", "claude-code")
    assert entry.change is not None
    assert entry.change["before"] == {"valid_to": None, "superseded_by": None}
    after = entry.change["after"]
    assert after["valid_to"] == "2026-11-11T09:00:00Z" and after["superseded_by"] == new
    assert len(after["resolved_conflicts"]) == 1


def test_supersede_resolves_the_conflict_pair(ctx: Context) -> None:
    old = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    new = rem(ctx, "Supplier Budi kirim Selasa", ["supplier"], entities=["Budi"])
    assert status(ctx, "supplier Budi") == {old: "possible conflict", new: "possible conflict"}
    facts.supersede(ctx, old, new)
    assert status(ctx, "supplier Budi") == {new: "valid"}
    pair = ctx.business.execute("SELECT resolved_at, resolution FROM conflicts").fetchone()
    assert pair["resolved_at"] is not None and pair["resolution"] == "supersede"


@pytest.mark.parametrize(("old", "new"), [("f_000000000000", "real"), ("real", "f_000000000000")])
def test_supersede_with_a_missing_id_fails_and_changes_nothing(
    ctx: Context, old: str, new: str
) -> None:
    real = rem(ctx, "Harga kopi", ["harga"])
    before = snapshot(ctx)
    with pytest.raises(NotFoundError):
        facts.supersede(ctx, real if old == "real" else old, real if new == "real" else new)
    assert snapshot(ctx) == before


def test_supersede_with_a_missing_proposal_id_fails(ctx: Context) -> None:
    real = rem(ctx, "Harga kopi", ["harga"])
    before = snapshot(ctx)
    with pytest.raises(NotFoundError):
        facts.supersede(ctx, real, "p_000000000000")
    assert snapshot(ctx) == before


def test_supersede_rejects_nonsense_without_changing_anything(ctx: Context) -> None:
    first = rem(ctx, "A", ["a"])
    second = rem(ctx, "B", ["b"])
    facts.supersede(ctx, first, second)
    third = rem(ctx, "C", ["c"])
    before = snapshot(ctx)
    with pytest.raises(ValidationError):
        facts.supersede(ctx, third, third)
    with pytest.raises(ValidationError):  # the old fact is already closed
        facts.supersede(ctx, first, third)
    with pytest.raises(ValidationError):  # the new fact is closed
        facts.supersede(ctx, third, first)
    assert snapshot(ctx) == before


def test_an_agent_fact_cannot_be_superseded_by_a_proposal(ctx: Context) -> None:
    mine = rem(ctx, "Harga kopi 50 ribu", ["harga"])
    prop = facts.remember(ctx, "Harga kopi 55 ribu", keywords=["harga"], tier="business")
    before = snapshot(ctx)
    with pytest.raises(ValidationError):
        facts.supersede(ctx, mine, prop.proposal_id or "")
    assert snapshot(ctx) == before


# ---- supersede, business and global facts (FR-59) -----------------------------------


@pytest.mark.parametrize(("tier", "file"), [("business", "business"), ("global", "glob")])
def test_supersede_of_a_shared_fact_with_an_agent_fact_queues_a_proposal(
    ctx: Context, tier: str, file: str
) -> None:
    old = approved(ctx, "Jam buka 09.00", tier, keywords=["jam"])
    mine = rem(ctx, "Jam buka 08.00", ["jam"])
    result = facts.supersede(ctx, old, mine)
    conn = getattr(ctx, file)
    assert result.closed is False and result.proposal_id is not None
    assert "approval" in result.message
    # Nothing is closed; the old fact shows "check first" at once.
    old_row = row(ctx, old)
    assert old_row["valid_to"] is None and old_row["superseded_by"] is None
    assert old_row["pending_proposal"] == result.proposal_id
    assert status(ctx, "jam buka")[old] == "check first"
    # The proposal lives in the old fact's file and carries the agent fact's content.
    prop = conn.execute("SELECT * FROM proposals WHERE id = ?", (result.proposal_id,)).fetchone()
    assert (prop["tier"], prop["supersedes"], prop["status"]) == (tier, old, "pending")
    assert "08.00" in prop["payload"]
    assert [p.id for p in proposals.pending(ctx)] == [result.proposal_id]
    # The agent's own fact is untouched.
    assert row(ctx, mine)["valid_to"] is None


def test_supersede_of_a_shared_fact_with_a_pending_proposal_links_it(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    prop = facts.remember(ctx, "Jam buka 08.00", keywords=["jam"], tier="business")
    result = facts.supersede(ctx, old, prop.proposal_id or "")
    assert result.proposal_id == prop.proposal_id and result.closed is False
    assert row(ctx, old)["pending_proposal"] == prop.proposal_id
    linked = ctx.business.execute(
        "SELECT supersedes FROM proposals WHERE id = ?", (prop.proposal_id,)
    ).fetchone()
    assert linked["supersedes"] == old
    assert status(ctx, "jam buka")[old] == "check first"
    # And no pending text is ever shown.
    assert all("08.00" not in h.text for h in search.recall(ctx, "jam buka").facts)


def test_supersede_logs_the_queued_change_on_the_old_fact(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    mine = rem(ctx, "Jam buka 08.00", ["jam"])
    result = facts.supersede(ctx, old, mine)
    entry = audit.entries(ctx, fact_id=old)[-1]
    assert (entry.op, entry.tier) == ("supersede", "business")
    assert entry.change == {
        "before": {"pending_proposal": None},
        "after": {"pending_proposal": result.proposal_id, "proposal_from_fact": mine},
    }


def test_a_shared_fact_with_a_change_waiting_cannot_get_a_second_one(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    first = rem(ctx, "Jam buka 08.00", ["jam"])
    second = rem(ctx, "Jam buka 10.00", ["jam"])
    facts.supersede(ctx, old, first)
    before = snapshot(ctx)
    with pytest.raises(ValidationError, match="waiting"):
        facts.supersede(ctx, old, second)
    assert snapshot(ctx) == before


def test_shared_fact_supersede_input_checks(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    other_shared = approved(ctx, "Jam tutup 17.00", keywords=["jam"])
    global_prop = facts.remember(ctx, "Jam global", keywords=["jam"], tier="global")
    before = snapshot(ctx)
    with pytest.raises(ValidationError):  # a shared fact is not a valid replacement
        facts.supersede(ctx, old, other_shared)
    with pytest.raises(ValidationError):  # proposal for another tier
        facts.supersede(ctx, old, global_prop.proposal_id or "")
    assert snapshot(ctx) == before


def test_another_agents_proposal_cannot_be_used_to_supersede(make_ctx: MakeContext) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    old = approved(a, "Jam buka 09.00", keywords=["jam"])
    prop = facts.remember(a, "Jam buka 08.00", keywords=["jam"], tier="business")
    before = snapshot(b)
    with pytest.raises(NotFoundError):
        facts.supersede(b, old, prop.proposal_id or "")
    assert snapshot(b) == before


# ---- confirm (FR-13, FR-14, FR-60) --------------------------------------------------


def test_confirm_moves_review_after_by_the_original_interval(
    ctx: Context, frozen_clock: Any
) -> None:
    fid = rem(ctx, "Harga kopi 50 ribu", ["harga"], review_after="2026-11-21")  # 10 days
    frozen_clock.advance(days=12)
    assert status(ctx, "harga kopi")[fid] == "check first"
    result = facts.confirm(ctx, fid, "Dicek langsung dengan supplier")
    assert result.review_after == "2026-12-03T09:00:00Z"
    assert row(ctx, fid)["review_after"] == "2026-12-03T09:00:00Z"  # now + 10 days
    assert row(ctx, fid)["review_interval_days"] == 10
    assert status(ctx, "harga kopi")[fid] == "valid"


def test_confirm_leaves_a_stable_fact_without_a_review_date(ctx: Context) -> None:
    fid = rem(ctx, "Pendiri Budi", ["pendiri"], kind="stable")
    facts.confirm(ctx, fid, "masih benar")
    assert row(ctx, fid)["review_after"] is None


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_confirm_with_an_empty_reason_fails_and_changes_nothing(
    ctx: Context, frozen_clock: Any, reason: str
) -> None:
    fid = rem(ctx, "Harga kopi", ["harga"])
    frozen_clock.advance(days=40)
    before = snapshot(ctx)
    with pytest.raises(ValidationError, match="reason"):
        facts.confirm(ctx, fid, reason)
    assert snapshot(ctx) == before
    assert status(ctx, "harga kopi")[fid] == "check first"


def test_confirm_with_a_missing_id_fails_and_changes_nothing(ctx: Context) -> None:
    rem(ctx, "Harga kopi", ["harga"])
    before = snapshot(ctx)
    with pytest.raises(NotFoundError):
        facts.confirm(ctx, "f_000000000000", "ok")
    assert snapshot(ctx) == before


def test_confirm_takes_a_fact_out_of_check_first(ctx: Context, frozen_clock: Any) -> None:
    fid = rem(ctx, "Harga kopi 50 ribu", ["harga"])
    frozen_clock.advance(days=31)
    assert status(ctx, "harga kopi")[fid] == "check first"
    facts.confirm(ctx, fid, "harga masih sama")
    assert status(ctx, "harga kopi")[fid] == "valid"
    frozen_clock.advance(days=31)
    assert status(ctx, "harga kopi")[fid] == "check first"  # the doubt comes back later


def test_confirm_logs_the_reason_with_before_and_after_review_date(
    ctx: Context, frozen_clock: Any
) -> None:
    fid = rem(ctx, "Harga kopi 50 ribu", ["harga"])
    frozen_clock.advance(days=31)
    facts.confirm(ctx, fid, "harga masih sama")
    entry = audit.entries(ctx, fact_id=fid)[-1]
    assert (entry.op, entry.tier, entry.owner, entry.agent) == (
        "confirm",
        "agent",
        "claude-code",
        "claude-code",
    )
    assert entry.change is not None
    assert entry.change["before"]["review_after"] == "2026-12-11T09:00:00Z"
    assert entry.change["after"]["review_after"] == "2027-01-11T09:00:00Z"
    assert entry.change["after"]["reason"] == "harga masih sama"


def test_confirm_with_a_reason_naming_the_other_fact_clears_the_pair(ctx: Context) -> None:
    first = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    second = rem(ctx, "Supplier Budi kirim Selasa", ["supplier"], entities=["Budi"])
    assert set(status(ctx, "supplier Budi").values()) == {"possible conflict"}
    result = facts.confirm(ctx, second, f"Keduanya benar, {first} untuk gudang A")
    assert result.resolved_conflicts == 1
    assert status(ctx, "supplier Budi") == {first: "valid", second: "valid"}
    pair = ctx.business.execute("SELECT resolution FROM conflicts").fetchone()
    assert pair["resolution"] == "confirm"


def test_confirm_without_naming_the_other_fact_keeps_the_pair(ctx: Context) -> None:
    first = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    second = rem(ctx, "Supplier Budi kirim Selasa", ["supplier"], entities=["Budi"])
    facts.confirm(ctx, second, "Saya yakin ini benar")
    assert status(ctx, "supplier Budi") == {
        first: "possible conflict",
        second: "possible conflict",
    }


def test_confirm_naming_an_unrelated_fact_resolves_nothing(ctx: Context) -> None:
    first = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    second = rem(ctx, "Supplier Budi kirim Selasa", ["supplier"], entities=["Budi"])
    third = rem(ctx, "Tentang hal lain sama sekali", ["lain"])
    result = facts.confirm(ctx, second, f"mirip dengan {third}")
    assert result.resolved_conflicts == 0
    assert status(ctx, "supplier Budi")[first] == "possible conflict"


@pytest.mark.parametrize(("tier", "file"), [("business", "business"), ("global", "glob")])
def test_confirm_on_a_shared_fact_is_applied_directly_and_logged(
    ctx: Context, frozen_clock: Any, tier: str, file: str
) -> None:
    fid = approved(ctx, "Pajak dibayar tiap bulan", tier, keywords=["pajak"])
    frozen_clock.advance(days=31)
    assert status(ctx, "pajak")[fid] == "check first"
    facts.confirm(ctx, fid, "Dicek dengan akuntan")
    assert status(ctx, "pajak")[fid] == "valid"
    conn = getattr(ctx, file)
    entry = conn.execute("SELECT * FROM log WHERE fact_id = ? AND op = 'confirm'", (fid,))
    found = entry.fetchall()
    assert len(found) == 1 and found[0]["tier"] == tier


def test_confirm_clears_the_check_first_of_a_pending_supersede(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    mine = rem(ctx, "Jam buka 08.00", ["jam"])
    facts.supersede(ctx, old, mine)
    assert status(ctx, "jam buka")[old] == "check first"
    facts.confirm(ctx, old, "Pemilik bilang jam buka tetap 09.00")
    assert status(ctx, "jam buka")[old] == "valid"
    assert row(ctx, old)["pending_proposal"] is None
    entry = audit.entries(ctx, fact_id=old)[-1]
    assert entry.change is not None
    assert entry.change["before"]["pending_proposal"] is not None
    assert entry.change["after"]["pending_proposal"] is None


def test_confirm_refuses_a_closed_fact(ctx: Context) -> None:
    old = rem(ctx, "A", ["a"])
    new = rem(ctx, "B", ["b"])
    facts.supersede(ctx, old, new)
    before = snapshot(ctx)
    with pytest.raises(ValidationError):
        facts.confirm(ctx, old, "masih benar")
    assert snapshot(ctx) == before


# ---- check first for a pending supersede ends with reject ---------------------------


def test_check_first_of_a_pending_supersede_is_cleared_by_reject(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    mine = rem(ctx, "Jam buka 08.00", ["jam"])
    result = facts.supersede(ctx, old, mine)
    assert status(ctx, "jam buka")[old] == "check first"
    proposals.reject(ctx, result.proposal_id or "")
    assert status(ctx, "jam buka")[old] == "valid"


# ---- history (FR-15) ----------------------------------------------------------------


def test_history_by_id_is_oldest_first_and_includes_every_operation(
    ctx: Context, frozen_clock: Any
) -> None:
    fid = rem(ctx, "Harga kopi", ["harga"])
    frozen_clock.advance(hours=1)
    facts.confirm(ctx, fid, "oke")
    frozen_clock.advance(hours=1)
    new = rem(ctx, "Harga kopi baru", ["harga"])
    facts.supersede(ctx, fid, new)
    entries = facts.history(ctx, fid)
    assert [e.op for e in entries] == ["create", "confirm", "supersede"]
    assert [e.ts for e in entries] == sorted(e.ts for e in entries)


def test_history_by_entity_covers_every_fact_linked_to_it(ctx: Context) -> None:
    one = rem(ctx, "Budi kirim Senin", ["supplier"], entities=["Budi"])
    two = rem(ctx, "Budi kirim Selasa", ["supplier"], entities=["Budi", "Gudang"])
    rem(ctx, "Ani kirim Rabu", ["supplier"], entities=["Ani"])
    facts.confirm(ctx, two, "oke")
    entries = facts.history(ctx, "bUDI")  # case-insensitive
    assert [(e.fact_id, e.op) for e in entries] == [
        (one, "create"),
        (two, "create"),
        (two, "confirm"),
    ]
    assert facts.history(ctx, "tidak-ada") == []


def test_history_of_an_unknown_id_is_not_found(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        facts.history(ctx, "f_000000000000")
    with pytest.raises(NotFoundError):
        facts.history(ctx, "p_000000000000")


def test_history_of_a_proposal_belongs_to_its_author_only(make_ctx: MakeContext) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    prop = facts.remember(a, "Rahasia usulan", keywords=["rahasia"], tier="business")
    assert [e.op for e in facts.history(a, prop.proposal_id or "")] == ["propose"]
    with pytest.raises(NotFoundError):
        facts.history(b, prop.proposal_id or "")


def test_history_of_a_replaced_fact_shows_the_approval_that_closed_it(ctx: Context) -> None:
    old = approved(ctx, "Jam buka 09.00", keywords=["jam"])
    mine = rem(ctx, "Jam buka 08.00", ["jam"])
    queued = facts.supersede(ctx, old, mine)
    new = proposals.approve(ctx, queued.proposal_id or "")
    assert [e.op for e in facts.history(ctx, old)] == ["approve", "supersede", "approve"]
    assert [e.op for e in facts.history(ctx, new)] == ["approve"]
