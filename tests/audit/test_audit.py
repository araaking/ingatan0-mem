from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from ingatan import audit
from ingatan.db import transaction
from ingatan.errors import NotFoundError
from ingatan.scope import Context

MakeContext = Callable[..., Context]


def put(
    conn: sqlite3.Connection,
    *,
    agent: str = "claude-code",
    op: str = "create",
    fact_id: str | None = "f_1",
    tier: str | None = "agent",
    owner: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> int:
    with transaction(conn):
        return audit.write(
            conn,
            agent=agent,
            op=op,
            fact_id=fact_id,
            tier=tier,
            owner=agent if owner is None and tier == "agent" else owner,
            before=before,
            after=after,
        )


def test_write_outside_a_transaction_raises(ctx: Context) -> None:
    with pytest.raises(RuntimeError, match="transaction"):
        audit.write(
            ctx.business,
            agent="claude-code",
            op="create",
            fact_id="f_1",
            tier="agent",
            owner="claude-code",
            before=None,
            after={"text": "x"},
        )
    assert ctx.business.execute("SELECT count(*) FROM log").fetchone()[0] == 0


def test_write_appends_exactly_one_entry_and_returns_its_id(
    ctx: Context, frozen_clock: object
) -> None:
    first = put(ctx.business, after={"text": "kopi"})
    second = put(ctx.business, op="confirm", before={"n": 1}, after={"n": 2})
    assert second == first + 1
    assert ctx.business.execute("SELECT count(*) FROM log").fetchone()[0] == 2
    entry = audit.get(ctx, first)
    assert entry.ts == "2026-11-11T09:00:00Z"
    assert (entry.agent, entry.op, entry.fact_id, entry.tier, entry.owner) == (
        "claude-code",
        "create",
        "f_1",
        "agent",
        "claude-code",
    )
    assert entry.change == {"before": None, "after": {"text": "kopi"}}


def test_write_inside_a_rolled_back_transaction_leaves_nothing(ctx: Context) -> None:
    with pytest.raises(RuntimeError), transaction(ctx.business):
        audit.write(
            ctx.business,
            agent="a",
            op="create",
            fact_id="f_1",
            tier="agent",
            owner="a",
            before=None,
            after={},
        )
        raise RuntimeError("fail after logging")
    assert ctx.business.execute("SELECT count(*) FROM log").fetchone()[0] == 0


def test_unknown_op_is_refused(ctx: Context) -> None:
    with pytest.raises(ValueError, match="Unknown log op"), transaction(ctx.business):
        audit.write(
            ctx.business,
            agent="a",
            op="explode",
            fact_id=None,
            tier=None,
            owner=None,
            before=None,
            after=None,
        )


def test_unicode_change_is_stored_unescaped(ctx: Context) -> None:
    text = "Harga kopi susu naik jadi Rp 18.000 — kata Bu Sári ‘mulai Senin’ ✓"
    log_id = put(ctx.business, after={"text": text})
    raw = ctx.business.execute("SELECT change FROM log WHERE id = ?", (log_id,)).fetchone()[0]
    assert "Sári" in raw and "\\u" not in raw
    assert audit.get(ctx, log_id).change == {"before": None, "after": {"text": text}}


def test_entries_filtering_order_and_limit(ctx: Context, frozen_clock: Any) -> None:
    ids = []
    for n in range(4):
        ids.append(put(ctx.business, fact_id="f_a" if n % 2 == 0 else "f_b", after={"n": n}))
        frozen_clock.advance(seconds=1)
    assert [e.id for e in audit.entries(ctx)] == ids
    assert [e.id for e in audit.entries(ctx, newest_first=True)] == ids[::-1]
    assert [e.id for e in audit.entries(ctx, fact_id="f_a")] == [ids[0], ids[2]]
    assert [e.id for e in audit.entries(ctx, fact_ids=["f_a", "f_b"], limit=3)] == ids[:3]
    assert [e.id for e in audit.entries(ctx, newest_first=True, limit=1)] == [ids[3]]
    assert audit.entries(ctx, fact_id="f_missing") == []


def test_entries_are_scope_filtered_by_tier_and_owner(make_ctx: MakeContext) -> None:
    alice = make_ctx(agent="alice")
    bob = make_ctx(agent="bob")
    private = put(alice.business, agent="alice", fact_id="f_priv", tier="agent")
    shared = put(alice.business, agent="alice", fact_id="f_shared", tier="business", owner="alice")
    global_entry = put(alice.glob, agent="alice", fact_id="f_glob", tier="global", owner="alice")
    untagged = put(alice.business, agent="alice", fact_id="f_none", tier=None, owner=None)

    assert {e.id for e in audit.entries(alice)} == {private, shared, global_entry}
    assert {e.id for e in audit.entries(bob)} == {shared, global_entry}
    assert untagged not in {e.id for e in audit.entries(alice)}
    assert audit.entries(bob, fact_id="f_priv") == []
    with pytest.raises(NotFoundError):
        audit.get(bob, private)
    with pytest.raises(NotFoundError):
        audit.diff(bob, private)
    assert audit.get(bob, shared).fact_id == "f_shared"


def test_entries_merge_both_files_and_ids_never_collide(make_ctx: MakeContext) -> None:
    ctx = make_ctx()
    local = put(ctx.business, fact_id="f_l", tier="business", owner="x")
    shared = put(ctx.glob, fact_id="f_g", tier="global", owner="x")
    assert local != shared
    assert audit.get(ctx, local).fact_id == "f_l"
    assert audit.get(ctx, shared).fact_id == "f_g"
    assert {e.fact_id for e in audit.entries(ctx)} == {"f_l", "f_g"}


def test_diff_returns_before_and_after(ctx: Context) -> None:
    log_id = put(
        ctx.business, op="supersede", before={"valid_to": None}, after={"valid_to": "2026-11-11"}
    )
    result = audit.diff(ctx, log_id)
    assert isinstance(result, audit.Diff)
    assert (result.log_id, result.op, result.fact_id) == (log_id, "supersede", "f_1")
    assert result.before == {"valid_to": None}
    assert result.after == {"valid_to": "2026-11-11"}


def test_get_unknown_id_raises_not_found(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        audit.get(ctx, 12345)


def test_scrub_clears_content_of_one_fact_only_and_keeps_the_entries(ctx: Context) -> None:
    mine = [put(ctx.business, fact_id="f_x", after={"text": "rahasia"}) for _ in range(2)]
    other = put(ctx.business, fact_id="f_y", after={"text": "tetap ada"})
    with transaction(ctx.business):
        audit.scrub(ctx.business, "f_x")
    for log_id in mine:
        entry = audit.get(ctx, log_id)
        assert entry.change is None
        assert (entry.op, entry.fact_id, entry.agent) == ("create", "f_x", "claude-code")
        diff = audit.diff(ctx, log_id)
        assert diff.before is None and diff.after is None
    assert audit.get(ctx, other).change == {"before": None, "after": {"text": "tetap ada"}}


def test_forget_style_entry_has_no_content(ctx: Context) -> None:
    log_id = put(ctx.business, op="forget", fact_id="f_x", before=None, after=None)
    raw = ctx.business.execute("SELECT change FROM log WHERE id = ?", (log_id,)).fetchone()[0]
    assert raw is None


def test_the_log_is_append_only_at_the_sql_level(ctx: Context) -> None:
    log_id = put(ctx.business, after={"text": "x"})
    with pytest.raises(sqlite3.DatabaseError):
        ctx.business.execute("DELETE FROM log WHERE id = ?", (log_id,))
    with pytest.raises(sqlite3.DatabaseError):
        ctx.business.execute("UPDATE log SET ts = 'now' WHERE id = ?", (log_id,))
    ctx.business.execute("UPDATE log SET change = NULL WHERE id = ?", (log_id,))
    assert audit.get(ctx, log_id).change is None


def test_undo_of_an_unknown_entry_is_not_found(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        audit.undo(ctx, 1)


def test_no_public_function_edits_or_deletes_entries_except_scrub() -> None:
    public = {n for n in dir(audit) if not n.startswith("_") and callable(getattr(audit, n))}
    assert {"write", "entries", "get", "scrub", "diff", "undo"} <= public
    assert not {n for n in public if "delete" in n or "update" in n or "remove" in n}
