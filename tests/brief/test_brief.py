from __future__ import annotations

from typing import Any

import pytest
from tests.conftest import FrozenClock, InsertFact, MakeContext

from ingatan import brief, episodes, facts, proposals
from ingatan.scope import Context

PAST = "2026-10-01T00:00:00Z"
FUTURE = "2026-12-01T00:00:00Z"


def use_counts(ctx: Context) -> list[tuple[str, int, str | None]]:
    rows: list[tuple[str, int, str | None]] = []
    for conn in ctx.connections():
        rows += [
            (r["id"], r["use_count"], r["last_used"])
            for r in conn.execute("SELECT id, use_count, last_used FROM facts ORDER BY id")
        ]
    return rows


def sections(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


def test_estimate_tokens_is_ceil_of_length_over_four() -> None:
    assert brief.estimate_tokens("") == 0
    assert brief.estimate_tokens("a") == 1
    assert brief.estimate_tokens("abcd") == 1
    assert brief.estimate_tokens("abcde") == 2


def test_empty_brief_has_header_and_pending_count_zero(ctx: Context) -> None:
    text = brief.build(ctx)
    assert text.splitlines()[0] == "# Ingatan brief: tokoku / claude-code"
    assert "Pending proposals waiting for the user: 0" in text
    assert sections(text) == []


def test_section_order_is_check_first_global_business_agent(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    insert_fact(ctx.glob, text="Pemilik suka kopi hitam", tier="global")
    insert_fact(ctx.business, text="Toko buka jam 8", tier="business")
    insert_fact(ctx.business, text="Agent ini mencatat stok", tier="agent")
    insert_fact(
        ctx.business,
        text="Harga kopi Rp 18.000",
        tier="business",
        kind="volatile",
        review_after=PAST,
    )
    text = brief.build(ctx)
    assert sections(text) == ["## Check first", "## Global", "## Business", "## Agent"]
    assert text.index("Pending proposals") < text.index("## Check first")


def test_empty_sections_are_omitted(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    insert_fact(ctx.business, text="Hanya satu fakta bisnis", tier="business")
    assert sections(brief.build(ctx)) == ["## Business"]


def test_facts_are_ordered_by_use_count_then_last_used(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    unused = insert_fact(ctx.business, text="never used", tier="business")
    old_use = insert_fact(
        ctx.business,
        text="used twice long ago",
        tier="business",
        use_count=2,
        last_used="2026-10-01T00:00:00Z",
    )
    new_use = insert_fact(
        ctx.business,
        text="used twice recently",
        tier="business",
        use_count=2,
        last_used="2026-11-10T00:00:00Z",
    )
    top = insert_fact(
        ctx.business,
        text="used five times",
        tier="business",
        use_count=5,
        last_used="2026-09-01T00:00:00Z",
    )
    text = brief.build(ctx)
    order = [text.index(f"[{i}]") for i in (top, new_use, old_use, unused)]
    assert order == sorted(order)


def test_overdue_volatile_fact_appears_only_in_check_first(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    overdue = insert_fact(
        ctx.business,
        text="Harga gula Rp 15.000",
        tier="business",
        kind="volatile",
        review_after=PAST,
    )
    fresh = insert_fact(
        ctx.business,
        text="Harga teh Rp 9.000",
        tier="business",
        kind="volatile",
        review_after=FUTURE,
    )
    stable = insert_fact(ctx.business, text="Toko di Bandung", tier="business")
    text = brief.build(ctx)
    assert text.count(f"[{overdue}]") == 1
    first = text.split("## Check first")[1].split("\n## ")[0]
    assert f"[{overdue}]" in first
    assert "review was due 2026-10-01" in first
    rest = text.split("## Business")[1]
    assert f"[{fresh}]" in rest
    assert f"[{stable}]" in rest
    assert f"[{overdue}]" not in rest


def test_fact_with_pending_supersede_is_check_first_only(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    old = insert_fact(ctx.business, text="Supplier utama PT Maju", tier="business")
    new = facts.remember(ctx, "Supplier utama PT Jaya", keywords=["supplier", "pemasok"])
    assert new.fact_id is not None
    facts.supersede(ctx, old, new.fact_id)
    text = brief.build(ctx)
    assert text.count(f"[{old}]") == 1
    check = text.split("## Check first")[1].split("\n## ")[0]
    assert f"[{old}]" in check
    assert "waiting for the user's approval" in check
    assert "Pending proposals waiting for the user: 1" in text


def test_pending_count_zero_and_n_and_pending_text_never_shown(
    ctx: Context, frozen_clock: FrozenClock
) -> None:
    assert "waiting for the user: 0" in brief.build(ctx)
    facts.remember(ctx, "Rahasia usulan bisnis", keywords=["usulan"], tier="business")
    facts.remember(ctx, "Rahasia usulan global", keywords=["usulan"], tier="global")
    facts.remember(ctx, "Rahasia usulan kedua", keywords=["usulan"], tier="business")
    text = brief.build(ctx)
    assert "waiting for the user: 3" in text
    assert "Rahasia" not in text
    assert proposals.pending_count(ctx) == 3
    assert sections(text) == []


def test_document_notes_never_appear(ctx: Context, frozen_clock: FrozenClock) -> None:
    ctx.business.execute(
        "INSERT INTO notes (path, heading, text, file_hash, created_at) "
        "VALUES ('c:/docs/a.md', 'Harga', 'catatan dokumen rahasia', 'h', '2026-11-11T09:00:00Z')"
    )
    assert "catatan dokumen" not in brief.build(ctx)


def test_possible_conflict_is_tagged_in_the_line(ctx: Context, frozen_clock: FrozenClock) -> None:
    facts.remember(
        ctx, "Kopi susu Rp 18.000", keywords=["kopi"], entities=["kopi susu"], kind="stable"
    )
    facts.remember(
        ctx, "Kopi susu Rp 20.000", keywords=["kopi"], entities=["kopi susu"], kind="stable"
    )
    text = brief.build(ctx)
    lines = [line for line in text.splitlines() if line.startswith("- [")]
    assert len(lines) == 2
    assert all("possible conflict" in line for line in lines)


def test_other_agents_private_facts_are_not_shown(
    make_ctx: MakeContext, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    mine = make_ctx(agent="claude-code")
    other = make_ctx(agent="hermes")
    insert_fact(other.business, text="catatan pribadi hermes", tier="agent", agent="hermes")
    insert_fact(mine.business, text="catatan pribadi saya", tier="agent", agent="claude-code")
    text = brief.build(mine)
    assert "pribadi saya" in text
    assert "pribadi hermes" not in text


def test_closed_facts_are_not_shown(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    insert_fact(
        ctx.business,
        text="sudah diganti",
        tier="business",
        superseded_by="f_other",
        valid_to="2026-11-01T00:00:00Z",
    )
    insert_fact(
        ctx.business, text="sudah kedaluwarsa", tier="business", valid_to="2026-11-02T00:00:00Z"
    )
    assert sections(brief.build(ctx)) == []


def test_brief_never_raises_use_count(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    insert_fact(ctx.business, text="satu", tier="business", use_count=3, last_used=PAST)
    insert_fact(ctx.glob, text="dua", tier="global")
    insert_fact(ctx.business, text="tiga", tier="business", kind="volatile", review_after=PAST)
    before = use_counts(ctx)
    brief.build(ctx)
    brief.build(ctx)
    assert use_counts(ctx) == before


# ----------------------------------------------------------------- budget


def fill(ctx: Context, insert_fact: InsertFact, count: int, overdue: int = 0) -> None:
    for n in range(count):
        tier = ("global", "business", "agent")[n % 3]
        conn = ctx.glob if tier == "global" else ctx.business
        extra: dict[str, Any] = {}
        if n < overdue:
            extra = {"kind": "volatile", "review_after": PAST}
        insert_fact(
            conn,
            text=f"Fakta nomor {n}: pelanggan tetap membeli kopi susu dan roti bakar tiap pagi",
            tier=tier,
            use_count=n % 17,
            last_used=f"2026-11-{1 + n % 10:02d}T00:00:00Z" if n % 17 else None,
            **extra,
        )


def test_budget_is_never_exceeded_with_300_facts(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    fill(ctx, insert_fact, 300, overdue=4)
    text = brief.build(ctx)
    assert brief.estimate_tokens(text) <= 1000
    assert brief.estimate_tokens(text) > 800  # it really uses the room
    assert text.count("## Check first") == 1
    assert text.count("review was due") == 4


def test_over_budget_drops_the_least_used_facts_first(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    fill(ctx, insert_fact, 300)
    text = brief.build(ctx)
    shown = [line for line in text.splitlines() if line.startswith("- [")]
    counts: dict[str, int] = {}
    for conn in ctx.connections():
        for r in conn.execute("SELECT id, use_count FROM facts"):
            counts[r["id"]] = r["use_count"]
    shown_counts = [counts[line[3 : line.index("]")]] for line in shown]
    dropped_max = max(counts[i] for i in counts if f"[{i}]" not in text)
    assert min(shown_counts) >= dropped_max
    assert len(shown) < 300


def test_tiny_budget_is_never_exceeded(
    make_ctx: MakeContext, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    ctx = make_ctx(token_budget=120)
    fill(ctx, insert_fact, 300, overdue=10)
    text = brief.build(ctx)
    assert brief.estimate_tokens(text) <= 120
    assert "## Check first" in text
    assert "more fact(s) to check" in text
    assert "recall" in text
    assert text.splitlines()[0].startswith("# Ingatan brief")
    assert "Pending proposals waiting for the user: 0" in text


def test_check_first_is_kept_whole_when_it_fits(
    make_ctx: MakeContext, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    ctx = make_ctx(token_budget=300)
    fill(ctx, insert_fact, 120, overdue=3)
    text = brief.build(ctx)
    assert brief.estimate_tokens(text) <= 300
    assert text.count("review was due") == 3
    assert "more fact(s) to check" not in text


@pytest.mark.parametrize("budget", [1, 5, 20, 40, 60, 90])
def test_absurd_budgets_still_hold(
    make_ctx: MakeContext, insert_fact: InsertFact, frozen_clock: FrozenClock, budget: int
) -> None:
    ctx = make_ctx(token_budget=budget)
    fill(ctx, insert_fact, 30, overdue=5)
    assert brief.estimate_tokens(brief.build(ctx)) <= budget


# --------------------------------------------------------------- episodes


def record_many(ctx: Context, count: int) -> list[int]:
    return [episodes.record(ctx, f"pesan {n}", f"jawaban {n}") for n in range(count)]


def offered_ids(text: str) -> list[int]:
    return [int(line.split()[2]) for line in text.splitlines() if line.startswith("### Episode")]


def test_ten_pending_episodes_offer_three(ctx: Context, frozen_clock: FrozenClock) -> None:
    ids = record_many(ctx, 10)
    text = brief.build(ctx)
    assert "## Episodes to digest" in text
    assert "remember(..., episode=<id>)" in text
    assert offered_ids(text) == ids[:3]
    assert "User: pesan 0" in text
    assert "Assistant: jawaban 0" in text
    assert brief.estimate_tokens(text) <= 1000


def test_episodes_come_last(
    ctx: Context, insert_fact: InsertFact, frozen_clock: FrozenClock
) -> None:
    insert_fact(ctx.business, text="fakta bisnis", tier="business")
    record_many(ctx, 1)
    assert sections(brief.build(ctx)) == ["## Business", "## Episodes to digest"]


def test_a_digested_episode_is_not_offered_again(ctx: Context, frozen_clock: FrozenClock) -> None:
    ids = record_many(ctx, 4)
    assert offered_ids(brief.build(ctx)) == ids[:3]
    facts.remember(ctx, "Pelanggan suka kopi", keywords=["kopi"], episode=ids[0])
    second = offered_ids(brief.build(ctx))
    assert ids[0] not in second
    assert second == [ids[1], ids[2], ids[3]]


def test_an_episode_offered_three_times_stops_being_offered(
    ctx: Context, frozen_clock: FrozenClock
) -> None:
    [only] = record_many(ctx, 1)
    for _ in range(3):
        assert offered_ids(brief.build(ctx)) == [only]
    assert offered_ids(brief.build(ctx)) == []
    assert "Episodes to digest" not in brief.build(ctx)


def test_offer_episodes_false_offers_nothing_and_counts_nothing(
    ctx: Context, frozen_clock: FrozenClock
) -> None:
    record_many(ctx, 3)
    assert "Episodes to digest" not in brief.build(ctx, offer_episodes=False)
    rows = ctx.business.execute("SELECT offered FROM episodes").fetchall()
    assert [r["offered"] for r in rows] == [0, 0, 0]


def test_other_agents_episodes_are_not_offered(
    make_ctx: MakeContext, frozen_clock: FrozenClock
) -> None:
    mine, other = make_ctx(agent="claude-code"), make_ctx(agent="hermes")
    episodes.record(other, "rahasia hermes", "balasan hermes")
    assert "rahasia" not in brief.build(mine)


def test_processed_episodes_are_purged_after_retention(
    ctx: Context, frozen_clock: FrozenClock
) -> None:
    [old] = record_many(ctx, 1)
    episodes.mark_processed(ctx, old)
    frozen_clock.advance(days=episodes.RETENTION_DAYS + 1)
    brief.build(ctx)
    assert ctx.business.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0


def test_episodes_fit_the_budget_and_unseen_ones_are_not_counted(
    make_ctx: MakeContext, frozen_clock: FrozenClock
) -> None:
    ctx = make_ctx(token_budget=400)
    for n in range(3):
        episodes.record(ctx, f"u{n} " + "x" * 1500, "a" * 1500)
    text = brief.build(ctx)
    assert brief.estimate_tokens(text) <= 400
    shown = offered_ids(text)
    counts = {
        r["id"]: r["offered"] for r in ctx.business.execute("SELECT id, offered FROM episodes")
    }
    assert all(counts[i] == 1 for i in shown)
    assert all(counts[i] == 0 for i in counts if i not in shown)
    assert len(shown) == 1  # too long to fit whole: the first one is cut down


def test_no_network_is_used(monkeypatch: pytest.MonkeyPatch, ctx: Context) -> None:
    import socket

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    record_many(ctx, 2)
    assert "Episodes to digest" in brief.build(ctx)
