from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ingatan import facts, proposals, search
from ingatan.errors import NotFoundError
from ingatan.scope import Context

MakeContext = Callable[..., Context]


def snapshot(ctx: Context) -> list[Any]:
    out: list[Any] = []
    for conn in ctx.connections():
        for table in ("facts", "log", "conflicts", "proposals", "facts_fts"):
            out.append([tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")])
    return out


@pytest.fixture
def two_agents(make_ctx: MakeContext) -> tuple[Context, Context, str, str]:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    secret = facts.remember(
        a, "Margin rahasia agen A", keywords=["margin"], entities=["Rahasia"]
    ).fact_id
    own = facts.remember(b, "Catatan milik agen B", keywords=["catatan"]).fact_id
    assert secret and own
    return a, b, secret, own


def test_agent_b_cannot_recall_agent_a_fact(
    two_agents: tuple[Context, Context, str, str],
) -> None:
    a, b, secret, _ = two_agents
    assert search.recall(b, "margin rahasia").facts == []
    assert [h.id for h in search.recall(a, "margin rahasia").facts] == [secret]


def test_agent_b_cannot_supersede_agent_a_fact_in_either_position(
    two_agents: tuple[Context, Context, str, str],
) -> None:
    _, b, secret, own = two_agents
    before = snapshot(b)
    with pytest.raises(NotFoundError):
        facts.supersede(b, secret, own)
    with pytest.raises(NotFoundError):
        facts.supersede(b, own, secret)
    assert snapshot(b) == before


def test_agent_b_cannot_confirm_agent_a_fact(
    two_agents: tuple[Context, Context, str, str],
) -> None:
    _, b, secret, _ = two_agents
    before = snapshot(b)
    with pytest.raises(NotFoundError):
        facts.confirm(b, secret, "saya yakin")
    assert snapshot(b) == before


@pytest.mark.parametrize("by_user", [False, True])
def test_agent_b_cannot_forget_agent_a_fact(
    two_agents: tuple[Context, Context, str, str], by_user: bool
) -> None:
    a, b, secret, _ = two_agents
    before = snapshot(b)
    with pytest.raises(NotFoundError):
        facts.forget(b, secret, by_user=by_user)
    assert snapshot(b) == before
    assert [h.id for h in search.recall(a, "margin").facts] == [secret]


def test_agent_b_cannot_see_the_history_of_agent_a_fact(
    two_agents: tuple[Context, Context, str, str],
) -> None:
    a, b, secret, _ = two_agents
    with pytest.raises(NotFoundError):
        facts.history(b, secret)
    assert facts.history(b, "Rahasia") == []  # by entity: nothing visible
    assert [e.op for e in facts.history(a, secret)] == ["create"]
    assert [e.op for e in facts.history(a, "rahasia")] == ["create"]


def test_agent_b_cannot_see_the_forget_entry_of_agent_a_fact(
    two_agents: tuple[Context, Context, str, str],
) -> None:
    a, b, secret, _ = two_agents
    facts.forget(a, secret)
    with pytest.raises(NotFoundError):
        facts.history(b, secret)


def test_a_conflict_with_another_agents_fact_is_not_revealed(
    make_ctx: MakeContext,
) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    prop = facts.remember(
        a, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"], tier="business"
    )
    assert prop.proposal_id
    shared = proposals.approve(a, prop.proposal_id)
    facts.remember(a, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"])
    assert search.recall(a, "supplier Budi").facts[0].status == "possible conflict"
    # Agent B sees the shared fact, but not that a private fact of A disagrees with it.
    seen = search.recall(b, "supplier Budi").facts
    assert [(h.id, h.status) for h in seen] == [(shared, "valid")]


def test_business_x_cannot_touch_business_y_by_id(make_ctx: MakeContext) -> None:
    x = make_ctx(business="usaha-x")
    y = make_ctx(business="usaha-y")
    fid = facts.remember(y, "Data Y", keywords=["data"]).fact_id
    prop = facts.remember(y, "Aturan Y", keywords=["aturan"], tier="business")
    biz = proposals.approve(y, prop.proposal_id or "")
    mine = facts.remember(x, "Data X", keywords=["data"]).fact_id
    assert fid and mine
    before = snapshot(x)
    for target in (fid, biz):
        with pytest.raises(NotFoundError):
            facts.supersede(x, target, mine)
        with pytest.raises(NotFoundError):
            facts.confirm(x, target, "ok")
        with pytest.raises(NotFoundError):
            facts.forget(x, target, by_user=True)
        with pytest.raises(NotFoundError):
            facts.history(x, target)
    assert snapshot(x) == before
