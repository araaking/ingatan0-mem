"""Resolving a `possible conflict` pair in one call (the logic behind `conflicts review`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ingatan import audit, facts
from ingatan.errors import ValidationError
from ingatan.facts import review
from ingatan.scope import Context

MakeContext = Callable[..., Context]

OLD = "Harga kopi susu Rp 15.000."
NEW = "Harga kopi susu Rp 18.000."


def save(ctx: Context, text: str, **kw: Any) -> str:
    kw.setdefault("entities", ["kopi susu"])
    result = facts.remember(ctx, text, **kw)
    assert result.fact_id is not None
    return result.fact_id


@pytest.fixture
def pair(ctx: Context, frozen_clock: Any) -> review.OpenConflict:
    """One open pair: an older and a newer statement of one price."""
    save(ctx, OLD)
    frozen_clock.advance(days=1)
    save(ctx, NEW)
    pairs = review.open_pairs(ctx)
    assert len(pairs) == 1
    return pairs[0]


def row(ctx: Context, fact_id: str) -> Any:
    return ctx.business.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()


def edges(ctx: Context) -> list[tuple[str, str, str]]:
    return [
        (str(r["src_id"]), str(r["dst_id"]), str(r["relation"]))
        for r in ctx.business.execute("SELECT src_id, dst_id, relation FROM edges")
    ]


# ---- which one is newer -------------------------------------------------------------------


def test_newer_is_the_one_valid_from_later(ctx: Context, pair: review.OpenConflict) -> None:
    assert (pair.text_a, pair.text_b) != ("", "")
    assert str(row(ctx, pair.newer_id)["text"]) == NEW
    assert str(row(ctx, pair.older_id)["text"]) == OLD
    assert {pair.newer_id, pair.older_id} == {pair.fact_a, pair.fact_b}


def test_a_tie_on_the_date_goes_to_the_row_written_later(ctx: Context, frozen_clock: Any) -> None:
    first = save(ctx, OLD)
    second = save(ctx, NEW)  # the same frozen second
    (found,) = review.open_pairs(ctx)
    assert (found.older_id, found.newer_id) == (first, second)


def test_a_backdated_fact_is_the_older_one_even_when_written_last(
    ctx: Context, frozen_clock: Any
) -> None:
    current = save(ctx, NEW)
    backdated = save(ctx, OLD, valid_from="2026-01-05")
    (found,) = review.open_pairs(ctx)
    assert (found.older_id, found.newer_id) == (backdated, current)


# ---- the three answers --------------------------------------------------------------------


def test_keep_both_confirms_then_links(ctx: Context, pair: review.OpenConflict) -> None:
    outcome = review.resolve_pair(ctx, pair, "keep_both")
    assert outcome.proposal_id is None
    assert review.open_pairs(ctx) == []
    confirms = [e for e in audit.entries(ctx) if e.op == "confirm"]
    assert len(confirms) == 1
    assert confirms[0].change is not None
    assert pair.older_id in confirms[0].change["after"]["reason"]
    assert edges(ctx) == [(pair.newer_id, pair.older_id, "related_to")]
    # Both stay valid, and the decision can be taken back.
    assert row(ctx, pair.older_id)["superseded_by"] is None
    audit.undo(ctx, confirms[0].id)
    assert [p.id for p in review.open_pairs(ctx)] == [pair.id]


def test_keep_newer_supersedes_the_older(ctx: Context, pair: review.OpenConflict) -> None:
    outcome = review.resolve_pair(ctx, pair, "keep_newer")
    assert outcome.proposal_id is None
    assert row(ctx, pair.older_id)["superseded_by"] == pair.newer_id
    assert row(ctx, pair.newer_id)["superseded_by"] is None
    assert review.open_pairs(ctx) == []
    assert pair.older_id in outcome.message and pair.newer_id in outcome.message


def test_keep_older_supersedes_the_newer(ctx: Context, pair: review.OpenConflict) -> None:
    review.resolve_pair(ctx, pair, "keep_older")
    assert row(ctx, pair.newer_id)["superseded_by"] == pair.older_id
    assert row(ctx, pair.older_id)["superseded_by"] is None
    assert review.open_pairs(ctx) == []


def test_a_supersede_from_review_is_logged_and_undoable(
    ctx: Context, pair: review.OpenConflict
) -> None:
    review.resolve_pair(ctx, pair, "keep_newer")
    entry = next(e for e in audit.entries(ctx) if e.op == "supersede")
    audit.undo(ctx, entry.id)
    assert row(ctx, pair.older_id)["superseded_by"] is None
    assert [p.id for p in review.open_pairs(ctx)] == [pair.id]


def test_approval_mode_returns_the_proposal_id(make_ctx: MakeContext, frozen_clock: Any) -> None:
    direct = make_ctx()
    older = save(direct, OLD, tier="business")
    frozen_clock.advance(days=1)
    save(direct, NEW, tier="business")
    approving = make_ctx(require_approval=True)
    (found,) = review.open_pairs(approving)
    outcome = review.resolve_pair(approving, found, "keep_newer")
    assert outcome.proposal_id is not None and outcome.proposal_id.startswith("p_")
    assert outcome.proposal_id in outcome.message
    assert row(approving, older)["superseded_by"] is None  # nothing closed yet


# ---- pairs that changed since they were listed --------------------------------------------


def test_pair_whose_fact_was_closed_meanwhile_is_skipped(
    ctx: Context, pair: review.OpenConflict
) -> None:
    replacement = save(ctx, "Harga kopi susu Rp 20.000.", check_conflicts=False)
    facts.supersede(ctx, pair.newer_id, replacement)
    before = len(audit.entries(ctx))
    outcome = review.resolve_pair(ctx, pair, "keep_both")
    assert "Skipped" in outcome.message
    assert outcome.proposal_id is None
    assert len(audit.entries(ctx)) == before
    assert edges(ctx) == []


def test_pair_whose_fact_was_forgotten_meanwhile_is_skipped(
    ctx: Context, pair: review.OpenConflict
) -> None:
    facts.forget(ctx, pair.older_id)
    outcome = review.resolve_pair(ctx, pair, "keep_newer")
    assert "Skipped" in outcome.message
    assert row(ctx, pair.newer_id)["superseded_by"] is None


def test_pair_that_was_already_resolved_is_skipped(ctx: Context, pair: review.OpenConflict) -> None:
    review.resolve_pair(ctx, pair, "keep_both")
    outcome = review.resolve_pair(ctx, pair, "keep_newer")
    assert "Skipped" in outcome.message
    assert row(ctx, pair.older_id)["superseded_by"] is None


def test_unknown_action_is_a_validation_error(ctx: Context, pair: review.OpenConflict) -> None:
    with pytest.raises(ValidationError, match="keep_both, keep_newer, keep_older"):
        review.resolve_pair(ctx, pair, "forget")
    assert [p.id for p in review.open_pairs(ctx)] == [pair.id]


def test_the_old_import_path_still_works(ctx: Context, pair: review.OpenConflict) -> None:
    from ingatan.cli.conflicts import OpenConflict, open_conflicts

    assert OpenConflict is review.OpenConflict
    assert open_conflicts(ctx) == [pair]
