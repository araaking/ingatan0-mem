from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ingatan import audit, facts, proposals, search
from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.scope import Context

MakeContext = Callable[..., Context]


def rem(ctx: Context, text: str, keywords: list[str] | None = None, **kw: Any) -> str:
    kw.setdefault("entities", ["kopi"])  # a shared entity makes two facts similar
    result = facts.remember(ctx, text, keywords=keywords or ["harga"], **kw)
    assert result.fact_id is not None
    return result.fact_id


def propose(ctx: Context, text: str, tier: str = "business", **kw: Any) -> str:
    result = facts.remember(ctx, text, keywords=kw.pop("keywords", ["harga"]), tier=tier, **kw)
    assert result.proposal_id is not None
    return result.proposal_id


def approved(ctx: Context, text: str, tier: str = "business") -> str:
    return proposals.approve(ctx, propose(ctx, text, tier))


def row(ctx: Context, fact_id: str) -> Any:
    for conn in ctx.connections():
        found = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if found is not None:
            return found
    raise AssertionError(fact_id)


def table(ctx: Context, name: str) -> list[tuple[Any, ...]]:
    return [tuple(r) for c in ctx.connections() for r in c.execute(f"SELECT * FROM {name}")]


def snapshot(ctx: Context) -> list[list[tuple[Any, ...]]]:
    return [table(ctx, n) for n in ("facts", "log", "conflicts", "proposals")]


def last_log(ctx: Context, fact_id: str, op: str) -> int:
    found = [e for e in audit.entries(ctx, fact_id=fact_id) if e.op == op]
    return found[-1].id


def recalled(ctx: Context, query: str) -> set[str]:
    return {h.id for h in search.recall(ctx, query, touch=False).facts}


def open_pairs(ctx: Context) -> int:
    return ctx.business.execute(
        "SELECT count(*) FROM conflicts WHERE resolved_at IS NULL"
    ).fetchone()[0]


# ---- create --------------------------------------------------------------------------


def test_undo_create_closes_the_fact_and_it_is_no_longer_recalled(
    ctx: Context, frozen_clock: Any
) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    assert fact in recalled(ctx, "harga kopi")
    log_id = last_log(ctx, fact, "create")
    old_entry = audit.get(ctx, log_id)
    frozen_clock.advance(minutes=5)

    new_id = audit.undo(ctx, log_id)

    assert recalled(ctx, "harga kopi") == set()
    r = row(ctx, fact)
    assert r["valid_to"] == "2026-11-11T09:05:00Z" and r["superseded_by"] is None
    entry = audit.get(ctx, new_id)
    assert entry.op == "undo" and entry.fact_id == fact
    assert entry.change is not None and entry.change["after"]["undoes"] == log_id
    assert audit.get(ctx, log_id) == old_entry  # the old entry is untouched
    # history still makes sense, by id and by entity
    for key in (fact, "kopi"):
        assert [e.op for e in facts.history(ctx, key)] == ["create", "undo"]
    assert [h.status for h in search.recall(ctx, "harga kopi", include_closed=True).facts] == [
        "expired"
    ]


def test_undo_create_resolves_the_conflict_pairs_of_the_fact(ctx: Context) -> None:
    rem(ctx, "Harga kopi 50 ribu")
    new = rem(ctx, "Harga kopi 55 ribu")
    assert open_pairs(ctx) == 1
    audit.undo(ctx, last_log(ctx, new, "create"))
    assert open_pairs(ctx) == 0
    resolution = ctx.business.execute("SELECT resolution FROM conflicts").fetchone()[0]
    assert resolution == "undo"


def test_undo_create_is_refused_when_a_later_entry_exists_for_the_fact(ctx: Context) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    facts.confirm(ctx, fact, "masih berlaku")
    before = snapshot(ctx)
    with pytest.raises(ValidationError, match="later entries"):
        audit.undo(ctx, last_log(ctx, fact, "create"))
    assert snapshot(ctx) == before


def test_undo_create_is_refused_when_another_fact_was_superseded_by_it(ctx: Context) -> None:
    old = rem(ctx, "Harga kopi 50 ribu")
    new = rem(ctx, "Harga kopi 55 ribu")
    facts.supersede(ctx, old, new)
    with pytest.raises(ValidationError, match="later entries"):
        audit.undo(ctx, last_log(ctx, new, "create"))


def test_changes_can_be_unwound_newest_first(ctx: Context, frozen_clock: Any) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    create_id = last_log(ctx, fact, "create")
    facts.confirm(ctx, fact, "sudah dicek")
    confirm_id = last_log(ctx, fact, "confirm")
    with pytest.raises(ValidationError):
        audit.undo(ctx, create_id)
    audit.undo(ctx, confirm_id)
    audit.undo(ctx, create_id)  # the confirm no longer stands in the way
    assert recalled(ctx, "kopi") == set()


def test_an_entry_cannot_be_undone_twice(ctx: Context) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    log_id = last_log(ctx, fact, "create")
    undo_id = audit.undo(ctx, log_id)
    with pytest.raises(ValidationError, match=f"already undone by entry {undo_id}"):
        audit.undo(ctx, log_id)


# ---- supersede -----------------------------------------------------------------------


def test_undo_supersede_reopens_the_old_fact_and_its_conflict_pair(
    ctx: Context, frozen_clock: Any
) -> None:
    old = rem(ctx, "Harga kopi 50 ribu")
    new = rem(ctx, "Harga kopi 55 ribu")
    before_old = tuple(row(ctx, old))
    facts.supersede(ctx, old, new)
    assert open_pairs(ctx) == 0
    assert recalled(ctx, "harga") == {new}

    audit.undo(ctx, last_log(ctx, old, "supersede"))

    assert tuple(row(ctx, old)) == before_old
    assert recalled(ctx, "harga") == {old, new}
    assert open_pairs(ctx) == 1  # the pair is open again
    assert [e.op for e in facts.history(ctx, old)] == ["create", "supersede", "undo"]


def test_undo_supersede_is_refused_when_the_old_fact_changed_later(ctx: Context) -> None:
    old = rem(ctx, "Harga kopi 50 ribu")
    new = rem(ctx, "Harga kopi 55 ribu")
    facts.supersede(ctx, old, new)
    log_id = last_log(ctx, old, "supersede")
    facts.forget(ctx, old)
    with pytest.raises(ValidationError, match="forgotten"):
        audit.undo(ctx, log_id)


def test_undo_supersede_of_a_shared_fact_drops_the_proposal_it_created(ctx: Context) -> None:
    shared = approved(ctx, "Harga kopi 50 ribu")
    mine = rem(ctx, "Harga kopi 55 ribu")
    result = facts.supersede(ctx, shared, mine)
    assert result.proposal_id is not None and row(ctx, shared)["pending_proposal"]

    audit.undo(ctx, last_log(ctx, shared, "supersede"))

    assert row(ctx, shared)["pending_proposal"] is None
    assert proposals.pending(ctx) == []
    assert row(ctx, mine)["valid_to"] is None  # the agent fact is untouched


def test_undo_supersede_of_a_shared_fact_unlinks_an_existing_proposal(ctx: Context) -> None:
    shared = approved(ctx, "Harga kopi 50 ribu")
    pid = propose(ctx, "Harga kopi 55 ribu")
    facts.supersede(ctx, shared, pid)

    audit.undo(ctx, last_log(ctx, shared, "supersede"))

    assert row(ctx, shared)["pending_proposal"] is None
    (p,) = proposals.pending(ctx)
    assert p.id == pid and p.supersedes is None
    # approving it now no longer closes the shared fact
    proposals.approve(ctx, pid)
    assert row(ctx, shared)["valid_to"] is None


def test_undo_supersede_is_refused_after_the_proposal_was_decided(ctx: Context) -> None:
    shared = approved(ctx, "Harga kopi 50 ribu")
    pid = propose(ctx, "Harga kopi 55 ribu")
    facts.supersede(ctx, shared, pid)
    log_id = last_log(ctx, shared, "supersede")
    proposals.reject(ctx, pid)
    before = snapshot(ctx)
    with pytest.raises(ValidationError, match="later entries"):
        audit.undo(ctx, log_id)
    assert snapshot(ctx) == before


# ---- confirm -------------------------------------------------------------------------


def test_undo_confirm_restores_review_after_and_reopens_the_pair(
    ctx: Context, frozen_clock: Any
) -> None:
    a = rem(ctx, "Harga kopi 50 ribu", kind="volatile")
    b = rem(ctx, "Harga kopi 55 ribu", kind="volatile")
    frozen_clock.advance(days=40)
    before_a = tuple(row(ctx, a))
    facts.confirm(ctx, a, f"keduanya benar, lihat {b}")
    assert open_pairs(ctx) == 0
    assert row(ctx, a)["review_after"] != before_a[tuple(row(ctx, a).keys()).index("review_after")]

    audit.undo(ctx, last_log(ctx, a, "confirm"))

    assert tuple(row(ctx, a)) == before_a
    assert open_pairs(ctx) == 1


def test_undo_confirm_restores_the_pending_mark_of_a_shared_fact(ctx: Context) -> None:
    shared = approved(ctx, "Harga kopi 50 ribu")
    pid = propose(ctx, "Harga kopi 55 ribu")
    facts.supersede(ctx, shared, pid)
    facts.confirm(ctx, shared, "masih benar")
    assert row(ctx, shared)["pending_proposal"] is None

    audit.undo(ctx, last_log(ctx, shared, "confirm"))

    assert row(ctx, shared)["pending_proposal"] == pid


# ---- approve -------------------------------------------------------------------------


def test_undo_approve_removes_the_fact_and_puts_the_proposal_back(ctx: Context) -> None:
    pid = propose(ctx, "Retur maksimal 7 hari", keywords=["retur"])
    fact = proposals.approve(ctx, pid)
    assert fact in recalled(ctx, "retur")

    audit.undo(ctx, last_log(ctx, fact, "approve"))

    assert recalled(ctx, "retur") == set()
    assert row(ctx, fact)["valid_to"] is not None
    (p,) = proposals.pending(ctx)
    assert p.id == pid and p.status == "pending"
    # the proposal can be approved again, giving a second, valid fact
    again = proposals.approve(ctx, pid)
    assert again != fact and recalled(ctx, "retur") == {again}


def test_undo_approve_reopens_the_fact_the_approval_closed(ctx: Context) -> None:
    shared = approved(ctx, "Harga kopi 50 ribu")
    pid = propose(ctx, "Harga kopi 55 ribu")
    facts.supersede(ctx, shared, pid)
    before_shared = tuple(row(ctx, shared))
    new = proposals.approve(ctx, pid)
    assert row(ctx, shared)["superseded_by"] == new

    audit.undo(ctx, last_log(ctx, new, "approve"))

    assert tuple(row(ctx, shared)) == before_shared  # open again, `check first` again
    assert row(ctx, shared)["pending_proposal"] == pid
    assert recalled(ctx, "harga") == {shared}
    assert [e.op for e in facts.history(ctx, shared)][-1] == "undo"
    # and the state is consistent enough to approve once more
    final = proposals.approve(ctx, pid)
    assert row(ctx, shared)["superseded_by"] == final


def test_undo_approve_of_a_global_proposal_works_across_files(ctx: Context) -> None:
    pid = propose(ctx, "Jam buka toko jam 9", "global", keywords=["jam"])
    fact = proposals.approve(ctx, pid)
    audit.undo(ctx, last_log(ctx, fact, "approve"))
    assert recalled(ctx, "jam buka") == set()
    assert ctx.glob.execute("SELECT status FROM proposals").fetchone()[0] == "pending"
    assert audit.entries(ctx, fact_id=fact)[-1].op == "undo"


def test_undo_approve_is_refused_when_the_new_fact_changed_later(ctx: Context) -> None:
    pid = propose(ctx, "Retur maksimal 7 hari", keywords=["retur"])
    fact = proposals.approve(ctx, pid)
    log_id = last_log(ctx, fact, "approve")
    facts.confirm(ctx, fact, "ok")
    with pytest.raises(ValidationError, match="later entries"):
        audit.undo(ctx, log_id)


# ---- refusals ------------------------------------------------------------------------


def test_forget_cannot_be_undone_and_nothing_changes(ctx: Context) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    facts.forget(ctx, fact)
    forget_id = last_log(ctx, fact, "forget")
    before = snapshot(ctx)
    with pytest.raises(ValidationError) as caught:
        audit.undo(ctx, forget_id)
    assert caught.value.message == "a forget cannot be undone"
    assert snapshot(ctx) == before


def test_the_create_entry_of_a_forgotten_fact_is_refused(ctx: Context) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    create_id = last_log(ctx, fact, "create")
    facts.forget(ctx, fact)
    with pytest.raises(ValidationError, match="forgotten"):
        audit.undo(ctx, create_id)


def test_undo_of_an_undo_is_refused(ctx: Context) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    undo_id = audit.undo(ctx, last_log(ctx, fact, "create"))
    with pytest.raises(ValidationError, match="cannot be undone"):
        audit.undo(ctx, undo_id)


def test_propose_reject_and_ingest_entries_are_refused_with_a_clear_message(ctx: Context) -> None:
    pid = propose(ctx, "Retur maksimal 7 hari", keywords=["retur"])
    propose_id = last_log(ctx, pid, "propose")
    with pytest.raises(ValidationError, match="proposals reject"):
        audit.undo(ctx, propose_id)
    proposals.reject(ctx, pid)
    with pytest.raises(ValidationError, match="rejected proposal"):
        audit.undo(ctx, last_log(ctx, pid, "reject"))
    with transaction(ctx.business):
        ingest_id = audit.write(
            ctx.business,
            agent="claude-code",
            op="ingest",
            fact_id=None,
            tier="business",
            owner="claude-code",
            before=None,
            after={"path": "a.md"},
        )
    with pytest.raises(ValidationError, match="ingest"):
        audit.undo(ctx, ingest_id)


def test_unknown_log_id_is_not_found(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        audit.undo(ctx, 999)


def test_another_agents_private_entry_behaves_like_a_missing_one(make_ctx: MakeContext) -> None:
    mine = make_ctx(agent="agen-a")
    other = make_ctx(agent="agen-b")
    fact = rem(mine, "Rahasia agen a", ["rahasia"])
    log_id = last_log(mine, fact, "create")
    with pytest.raises(NotFoundError):
        audit.undo(other, log_id)
    assert row(mine, fact)["valid_to"] is None


def test_undo_writes_exactly_one_entry_and_never_edits_old_ones(ctx: Context) -> None:
    fact = rem(ctx, "Harga kopi 50 ribu")
    old = table(ctx, "log")
    audit.undo(ctx, last_log(ctx, fact, "create"))
    new = table(ctx, "log")
    assert new[: len(old)] == old and len(new) == len(old) + 1
