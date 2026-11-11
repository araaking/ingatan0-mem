"""PLAN T-6: a test fails if a write path skips the log.

Each public write operation runs once and must add exactly one log row, in the file where
the row it changed lives, and none in the other file.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from ingatan import facts, proposals
from ingatan.scope import Context

Action = Callable[[Context], None]


def remember_fact(ctx: Context, text: str = "Harga kopi 50 ribu") -> str:
    fact_id = facts.remember(ctx, text, keywords=["harga"]).fact_id
    assert fact_id
    return fact_id


def approved(ctx: Context, tier: str) -> str:
    proposal = facts.remember(ctx, "Aturan toko", keywords=["aturan"], tier=tier).proposal_id
    assert proposal
    return proposals.approve(ctx, proposal)


def pending(ctx: Context, tier: str) -> str:
    proposal = facts.remember(ctx, "Aturan baru", keywords=["aturan"], tier=tier).proposal_id
    assert proposal
    return proposal


# name -> (setup returning state, action(ctx, state), file that must gain one entry)
CASES: dict[str, tuple[Callable[[Context], Any], Callable[[Context, Any], object], str]] = {
    "remember agent tier": (
        lambda ctx: None,
        lambda ctx, s: facts.remember(ctx, "x", keywords=["x"]),
        "business",
    ),
    "remember business tier": (
        lambda ctx: None,
        lambda ctx, s: facts.remember(ctx, "x", keywords=["x"], tier="business"),
        "business",
    ),
    "remember global tier": (
        lambda ctx: None,
        lambda ctx, s: facts.remember(ctx, "x", keywords=["x"], tier="global"),
        "glob",
    ),
    "supersede agent fact": (
        lambda ctx: (remember_fact(ctx), remember_fact(ctx, "Harga kopi 55 ribu")),
        lambda ctx, s: facts.supersede(ctx, s[0], s[1]),
        "business",
    ),
    "supersede business fact with an agent fact": (
        lambda ctx: (approved(ctx, "business"), remember_fact(ctx)),
        lambda ctx, s: facts.supersede(ctx, s[0], s[1]),
        "business",
    ),
    "supersede business fact with a proposal": (
        lambda ctx: (approved(ctx, "business"), pending(ctx, "business")),
        lambda ctx, s: facts.supersede(ctx, s[0], s[1]),
        "business",
    ),
    "supersede global fact with an agent fact": (
        lambda ctx: (approved(ctx, "global"), remember_fact(ctx)),
        lambda ctx, s: facts.supersede(ctx, s[0], s[1]),
        "glob",
    ),
    "supersede global fact with a proposal": (
        lambda ctx: (approved(ctx, "global"), pending(ctx, "global")),
        lambda ctx, s: facts.supersede(ctx, s[0], s[1]),
        "glob",
    ),
    "confirm agent fact": (
        remember_fact,
        lambda ctx, s: facts.confirm(ctx, s, "masih benar"),
        "business",
    ),
    "confirm business fact": (
        lambda ctx: approved(ctx, "business"),
        lambda ctx, s: facts.confirm(ctx, s, "masih benar"),
        "business",
    ),
    "confirm global fact": (
        lambda ctx: approved(ctx, "global"),
        lambda ctx, s: facts.confirm(ctx, s, "masih benar"),
        "glob",
    ),
    "confirm resolving a conflict pair": (
        lambda ctx: (
            facts.remember(ctx, "Budi Senin", keywords=["a"], entities=["Budi"]).fact_id,
            facts.remember(ctx, "Budi Selasa", keywords=["a"], entities=["Budi"]).fact_id,
        ),
        lambda ctx, s: facts.confirm(ctx, s[1], f"keduanya benar, lihat {s[0]}"),
        "business",
    ),
    "forget agent fact": (
        remember_fact,
        lambda ctx, s: facts.forget(ctx, s),
        "business",
    ),
    "forget business fact by the user": (
        lambda ctx: approved(ctx, "business"),
        lambda ctx, s: facts.forget(ctx, s, by_user=True),
        "business",
    ),
    "forget global fact by the user": (
        lambda ctx: approved(ctx, "global"),
        lambda ctx, s: facts.forget(ctx, s, by_user=True),
        "glob",
    ),
    "approve business proposal": (
        lambda ctx: pending(ctx, "business"),
        lambda ctx, s: proposals.approve(ctx, s),
        "business",
    ),
    "approve global proposal": (
        lambda ctx: pending(ctx, "global"),
        lambda ctx, s: proposals.approve(ctx, s),
        "glob",
    ),
    "approve a proposal that replaces a fact": (
        lambda ctx: _queued_replacement(ctx, "business"),
        lambda ctx, s: proposals.approve(ctx, s),
        "business",
    ),
    "reject business proposal": (
        lambda ctx: pending(ctx, "business"),
        lambda ctx, s: proposals.reject(ctx, s),
        "business",
    ),
    "reject global proposal": (
        lambda ctx: pending(ctx, "global"),
        lambda ctx, s: proposals.reject(ctx, s),
        "glob",
    ),
    "reject a proposal that replaces a fact": (
        lambda ctx: _queued_replacement(ctx, "global"),
        lambda ctx, s: proposals.reject(ctx, s),
        "glob",
    ),
}


def _queued_replacement(ctx: Context, tier: str) -> str:
    old = approved(ctx, tier)
    mine = remember_fact(ctx)
    proposal = facts.supersede(ctx, old, mine).proposal_id
    assert proposal
    return proposal


def log_counts(ctx: Context) -> dict[str, int]:
    return {
        "business": ctx.business.execute("SELECT count(*) FROM log").fetchone()[0],
        "glob": ctx.glob.execute("SELECT count(*) FROM log").fetchone()[0],
    }


@pytest.mark.parametrize("name", list(CASES))
def test_every_write_operation_adds_exactly_one_log_row(ctx: Context, name: str) -> None:
    setup, action, expected_file = CASES[name]
    state = setup(ctx)
    before = log_counts(ctx)
    action(ctx, state)
    after = log_counts(ctx)
    other_file = "glob" if expected_file == "business" else "business"
    assert after[expected_file] == before[expected_file] + 1, name
    assert after[other_file] == before[other_file], name


@pytest.mark.parametrize("name", list(CASES))
def test_the_new_log_row_has_tier_and_owner_filled(ctx: Context, name: str) -> None:
    setup, action, expected_file = CASES[name]
    state = setup(ctx)
    conn = getattr(ctx, expected_file)
    last_before = conn.execute("SELECT coalesce(max(id), 0) FROM log").fetchone()[0]
    action(ctx, state)
    rows = conn.execute("SELECT * FROM log WHERE id > ?", (last_before,)).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["tier"] in ("agent", "business", "global") and row["owner"], name
    assert row["agent"] == "claude-code" and row["fact_id"]
    assert (row["tier"] == "global") == (expected_file == "glob")


@pytest.mark.parametrize(
    "name",
    [
        "remember agent tier",
        "supersede agent fact",
        "confirm agent fact",
        "confirm business fact",
        "approve business proposal",
        "supersede business fact with an agent fact",
    ],
)
def test_a_failing_write_leaves_no_row_in_the_log(ctx: Context, name: str) -> None:
    """The row and its log entry commit together: a failure inside rolls both back."""
    setup, action, _ = CASES[name]
    state = setup(ctx)
    before = log_counts(ctx)
    facts_before = ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0]
    ctx.business.create_function("explode", 0, _boom)
    ctx.business.execute(
        "CREATE TEMP TRIGGER explode_on_log BEFORE INSERT ON main.log "
        "WHEN new.op != 'xx' BEGIN SELECT explode(); END"
    )
    try:
        with pytest.raises(sqlite3.OperationalError):
            action(ctx, state)
    finally:
        ctx.business.execute("DROP TRIGGER temp.explode_on_log")
    assert log_counts(ctx) == before
    assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == facts_before
    assert not ctx.business.in_transaction


def _boom() -> int:
    raise RuntimeError("boom")
