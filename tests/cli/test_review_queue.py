"""`ingatan conflicts review`: resolve every open pair without typing a fact id."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import Result

from ingatan import facts, proposals
from ingatan.facts import review
from ingatan.scope import Context

Invoke = Callable[..., Result]
MakeContext = Callable[..., Context]

PAIRS = [
    ("kopi susu", "Harga kopi susu Rp 15.000.", "Harga kopi susu Rp 18.000."),
    ("roti bakar", "Harga roti bakar Rp 12.000.", "Harga roti bakar Rp 14.000."),
    ("teh tarik", "Harga teh tarik Rp 8.000.", "Harga teh tarik Rp 9.000."),
]


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingatan.cli.commands._is_interactive", lambda: True)


def save(ctx: Context, text: str, entity: str, **kw: Any) -> str:
    result = facts.remember(ctx, text, entities=[entity], **kw)
    assert result.fact_id is not None
    return result.fact_id


@pytest.fixture
def three_pairs(cli_ctx: Context, frozen_clock: Any) -> list[tuple[str, str]]:
    """Three unrelated open pairs, each as (older id, newer id)."""
    made: list[tuple[str, str]] = []
    for entity, old, new in PAIRS:
        older = save(cli_ctx, old, entity)
        frozen_clock.advance(days=1)
        newer = save(cli_ctx, new, entity)
        made.append((older, newer))
    assert len(review.open_pairs(cli_ctx)) == 3
    return made


def superseded_by(ctx: Context, fact_id: str) -> str | None:
    row = ctx.business.execute(
        "SELECT superseded_by FROM facts WHERE id = ?", (fact_id,)
    ).fetchone()
    return None if row is None else row[0]


def exists(ctx: Context, fact_id: str) -> bool:
    return (
        ctx.business.execute("SELECT 1 FROM facts WHERE id = ?", (fact_id,)).fetchone() is not None
    )


# ---- answering one by one -----------------------------------------------------------------


def test_review_resolves_every_pair_without_an_id(
    invoke: Invoke, terminal: None, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review", input="k\nn\no\n")
    assert result.exit_code == 0, result.output
    (kopi_old, kopi_new), (roti_old, roti_new), (teh_old, teh_new) = three_pairs
    assert superseded_by(cli_ctx, kopi_old) is None and superseded_by(cli_ctx, kopi_new) is None
    assert superseded_by(cli_ctx, roti_old) == roti_new
    assert superseded_by(cli_ctx, teh_new) == teh_old
    listed = invoke("conflicts", "list")
    assert "No open conflicts." in listed.output


def test_review_shows_both_facts_in_full_with_labels(
    invoke: Invoke, terminal: None, cli_ctx: Context, frozen_clock: Any
) -> None:
    long_old = "Harga kopi susu Rp 15.000 " + "untuk semua cabang di kota ini " * 8 + "sejak awal."
    save(cli_ctx, long_old, "kopi susu")
    frozen_clock.advance(days=1)
    save(cli_ctx, "Harga kopi susu Rp 18.000.", "kopi susu")
    result = invoke("conflicts", "review", input="s\n")
    assert result.exit_code == 0, result.output
    assert "Pair 1 of 1" in result.output
    assert "NEWER" in result.output and "OLDER" in result.output
    assert long_old in result.output  # not shortened: the reviewer decides on the whole text
    assert "2026-11-11" in result.output and "2026-11-12" in result.output
    assert "[agent]" in result.output


def test_capital_answer_applies_to_the_rest(
    invoke: Invoke, terminal: None, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review", input="N\n")
    assert result.exit_code == 0, result.output
    for older, newer in three_pairs:
        assert superseded_by(cli_ctx, older) == newer
    assert review.open_pairs(cli_ctx) == []


def test_skip_and_quit_leave_pairs_open(
    invoke: Invoke, terminal: None, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review", input="s\nq\n")
    assert result.exit_code == 0, result.output
    assert len(review.open_pairs(cli_ctx)) == 3
    assert "3 still open" in result.output


def test_an_answer_that_is_not_understood_is_asked_again(
    invoke: Invoke, terminal: None, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review", input="x\n\nK\n")
    assert result.exit_code == 0, result.output
    assert review.open_pairs(cli_ctx) == []


def test_a_pair_settled_by_an_earlier_answer_is_not_asked_again(
    invoke: Invoke, terminal: None, cli_ctx: Context, frozen_clock: Any
) -> None:
    # Three prices of one thing: three pairs that all share facts.
    for price in ("15.000", "18.000", "20.000"):
        save(cli_ctx, f"Harga kopi susu Rp {price}.", "kopi susu")
        frozen_clock.advance(days=1)
    assert len(review.open_pairs(cli_ctx)) == 3
    # Two answers are enough; a third question would abort on the exhausted input.
    result = invoke("conflicts", "review", input="n\nn\n")
    assert result.exit_code == 0, result.output
    assert review.open_pairs(cli_ctx) == []
    assert "settled by an earlier answer" in result.output


def test_approval_mode_change_is_applied_at_once(
    invoke: Invoke,
    terminal: None,
    cli_ctx: Context,
    frozen_clock: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = save(cli_ctx, "Harga kopi susu Rp 15.000.", "kopi susu", tier="business")
    frozen_clock.advance(days=1)
    newer = save(cli_ctx, "Harga kopi susu Rp 18.000.", "kopi susu", tier="business")
    monkeypatch.setenv("INGATAN_REQUIRE_APPROVAL", "1")
    result = invoke("conflicts", "review", input="n\n")
    assert result.exit_code == 0, result.output
    assert superseded_by(cli_ctx, older) == newer
    assert proposals.pending(cli_ctx) == []


# ---- forgetting one of the two ----------------------------------------------------------------


def test_forget_asks_which_and_confirms(
    invoke: Invoke, terminal: None, cli_ctx: Context, frozen_clock: Any
) -> None:
    older = save(cli_ctx, "Harga kopi susu Rp 15.000.", "kopi susu")
    frozen_clock.advance(days=1)
    newer = save(cli_ctx, "Harga kopi susu Rp 18.000.", "kopi susu")

    declined = invoke("conflicts", "review", input="f\nolder\nn\n")
    assert declined.exit_code == 0, declined.output
    assert "cannot be undone" in declined.output
    assert "Nothing was forgotten." in declined.output
    assert exists(cli_ctx, older) and exists(cli_ctx, newer)
    assert len(review.open_pairs(cli_ctx)) == 1

    confirmed = invoke("conflicts", "review", input="f\nolder\ny\n")
    assert confirmed.exit_code == 0, confirmed.output
    assert not exists(cli_ctx, older) and exists(cli_ctx, newer)
    assert review.open_pairs(cli_ctx) == []


def test_forget_is_never_applied_in_bulk(
    invoke: Invoke, terminal: None, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    refused = invoke("conflicts", "review", "--apply", "forget", "--all")
    assert refused.exit_code == 1, refused.output
    assert "keep_both, keep_newer, keep_older" in refused.output
    # A capital F is not "forget in all that remain": it is refused and the pair asked again.
    result = invoke("conflicts", "review", input="F\nq\n")
    assert result.exit_code == 0, result.output
    assert "never applied to several pairs" in result.output
    for older, newer in three_pairs:
        assert exists(cli_ctx, older) and exists(cli_ctx, newer)


# ---- without a terminal ------------------------------------------------------------------------


def test_apply_all_needs_no_terminal(
    invoke: Invoke, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review", "--apply", "keep_newer", "--all")
    assert result.exit_code == 0, result.output
    for older, newer in three_pairs:
        assert superseded_by(cli_ctx, older) == newer
    assert "3 pair(s) resolved" in result.output


def test_apply_without_all_is_refused(
    invoke: Invoke, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review", "--apply", "keep_newer")
    assert result.exit_code == 1, result.output
    assert "--all" in result.output
    assert len(review.open_pairs(cli_ctx)) == 3


def test_all_without_apply_is_refused(invoke: Invoke, cli_ctx: Context) -> None:
    result = invoke("conflicts", "review", "--all")
    assert result.exit_code == 1, result.output
    assert "--apply" in result.output


def test_review_without_terminal_and_without_apply_fails_clearly(
    invoke: Invoke, cli_ctx: Context, three_pairs: list[tuple[str, str]]
) -> None:
    result = invoke("conflicts", "review")
    assert result.exit_code == 1, result.output
    assert "--apply" in result.output and "no terminal" in result.output
    assert len(review.open_pairs(cli_ctx)) == 3


def test_review_with_nothing_open_says_so(invoke: Invoke, cli_ctx: Context) -> None:
    result = invoke("conflicts", "review")
    assert result.exit_code == 0, result.output
    assert "No open conflicts." in result.output
