from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from ingatan import audit, episodes, facts, proposals, search
from ingatan.errors import NotFoundError, ValidationError
from ingatan.redact import MARKER
from ingatan.scope import Context

MakeContext = Callable[..., Context]


def count(conn: Any, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def dump_all_text(ctx: Context) -> str:
    """Every stored string in the business and global files, for leak checks."""
    parts: list[str] = []
    for conn in ctx.connections():
        for table in ("facts", "log", "proposals", "conflicts", "facts_fts", "facts_tri"):
            for row in conn.execute(f"SELECT * FROM {table}"):
                parts.extend(str(v) for v in tuple(row))
    return "\n".join(parts)


# ---- defaults (FR-10) ---------------------------------------------------------------


def test_defaults_are_volatile_with_review_after_thirty_days(
    ctx: Context, frozen_clock: Any
) -> None:
    result = facts.remember(ctx, "Harga kopi 50 ribu", keywords=["harga"])
    row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (result.fact_id,)).fetchone()
    assert row["kind"] == "volatile"
    assert row["valid_from"] == "2026-11-11T09:00:00Z"
    assert row["review_after"] == "2026-12-11T09:00:00Z"  # 30 days later
    assert row["review_interval_days"] == 30
    assert row["tier"] == "agent" and row["agent"] == "claude-code"
    assert row["valid_to"] is None and row["superseded_by"] is None
    assert row["pending_proposal"] is None
    assert (row["use_count"], row["last_used"]) == (0, None)
    assert row["created_at"] == "2026-11-11T09:00:00Z"
    assert row["id"].startswith("f_") and len(row["id"]) == 14


def test_stable_fact_has_no_review_date(ctx: Context) -> None:
    result = facts.remember(ctx, "Pemilik toko Budi", keywords=["pemilik"], kind="stable")
    row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (result.fact_id,)).fetchone()
    assert row["kind"] == "stable"
    assert row["review_after"] is None and row["review_interval_days"] is None


def test_stable_fact_ignores_a_review_after_it_was_given(ctx: Context) -> None:
    result = facts.remember(
        ctx, "Pemilik toko Budi", keywords=["pemilik"], kind="stable", review_after="2027-01-01"
    )
    row = ctx.business.execute("SELECT review_after FROM facts WHERE id = ?", (result.fact_id,))
    assert row.fetchone()[0] is None


def test_explicit_review_after_is_normalised_and_sets_the_interval(
    ctx: Context, frozen_clock: Any
) -> None:
    result = facts.remember(ctx, "Promo berlaku", keywords=["promo"], review_after="2026-11-21")
    row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (result.fact_id,)).fetchone()
    assert row["review_after"] == "2026-11-21T00:00:00Z"
    assert row["review_interval_days"] == 10
    assert row["kind"] == "volatile"


def test_review_after_accepts_a_full_timestamp(ctx: Context) -> None:
    result = facts.remember(
        ctx, "Promo berlaku", keywords=["promo"], review_after="2027-02-03T04:05:06Z"
    )
    row = ctx.business.execute("SELECT review_after FROM facts WHERE id = ?", (result.fact_id,))
    assert row.fetchone()[0] == "2027-02-03T04:05:06Z"


def test_a_bad_review_after_is_a_validation_error_and_saves_nothing(ctx: Context) -> None:
    with pytest.raises(ValidationError):
        facts.remember(ctx, "x", keywords=["x"], review_after="minggu depan")
    assert count(ctx.business, "facts") == 0 and count(ctx.business, "log") == 0


# ---- keywords are required (FR-10) --------------------------------------------------


@pytest.mark.parametrize("keywords", [None, [], [""], ["  ", ""], ""])
def test_missing_keywords_error_asks_for_them_and_saves_nothing(
    ctx: Context, keywords: Any
) -> None:
    with pytest.raises(ValidationError, match="keywords") as info:
        facts.remember(ctx, "Pelanggan membayar tunai", keywords=keywords)
    assert "synonym" in str(info.value) and "root" in str(info.value)
    for conn in ctx.connections():
        assert count(conn, "facts") == 0
        assert count(conn, "log") == 0
        assert count(conn, "proposals") == 0
        assert count(conn, "facts_fts") == 0
    assert count(ctx.business, "conflicts") == 0


@pytest.mark.parametrize("tier", ["business", "global"])
def test_missing_keywords_saves_no_proposal_either(ctx: Context, tier: str) -> None:
    with pytest.raises(ValidationError):
        facts.remember(ctx, "Aturan toko", keywords=[], tier=tier)
    assert proposals.pending_count(ctx) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tier": "dunia"}, "tier"),
        ({"kind": "forever"}, "kind"),
    ],
)
def test_bad_tier_or_kind_is_a_validation_error(
    ctx: Context, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        facts.remember(ctx, "x", keywords=["x"], **kwargs)
    assert count(ctx.business, "log") == 0


def test_empty_text_is_a_validation_error(ctx: Context) -> None:
    with pytest.raises(ValidationError):
        facts.remember(ctx, "   ", keywords=["x"])
    assert count(ctx.business, "facts") == 0


# ---- redaction (FR-32) --------------------------------------------------------------

SECRET = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX1234"
PASSWORD = "hunter2zz"


def test_a_secret_is_absent_from_facts_and_log_tables(ctx: Context) -> None:
    result = facts.remember(
        ctx,
        f"Kunci API toko adalah {SECRET} dan password={PASSWORD}",
        keywords=[f"token={PASSWORD}", "kunci", SECRET],
        entities=[f"api_key={PASSWORD}"],
    )
    assert result.fact_id is not None
    facts.confirm(ctx, result.fact_id, f"sudah dicek, sandi: {PASSWORD}")
    blob = dump_all_text(ctx)
    assert SECRET not in blob
    assert PASSWORD not in blob
    assert "ABCDEFGHIJKLMNOPQRSTUVWX1234" not in blob
    row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (result.fact_id,)).fetchone()
    assert MARKER in row["text"] and MARKER in row["keywords"] and MARKER in row["entities"]
    created = audit.entries(ctx, fact_id=result.fact_id)[0]
    assert created.change is not None and MARKER in created.change["after"]["text"]


@pytest.mark.parametrize("tier", ["business", "global"])
def test_a_secret_is_absent_from_proposals_and_their_log(ctx: Context, tier: str) -> None:
    result = facts.remember(
        ctx, f"Token {SECRET}", keywords=["token", f"password={PASSWORD}"], tier=tier
    )
    assert result.proposal_id
    assert SECRET not in dump_all_text(ctx) and PASSWORD not in dump_all_text(ctx)
    proposals.approve(ctx, result.proposal_id)
    assert SECRET not in dump_all_text(ctx) and PASSWORD not in dump_all_text(ctx)


def test_a_secret_in_source_is_redacted_too(ctx: Context) -> None:
    result = facts.remember(
        ctx, "x", keywords=["x"], source={"document": f"notes.md?token={PASSWORD}", "session": "s1"}
    )
    row = ctx.business.execute(
        "SELECT source FROM facts WHERE id = ?", (result.fact_id,)
    ).fetchone()
    assert PASSWORD not in row["source"]
    assert json.loads(row["source"])["session"] == "s1"


def test_the_source_always_names_the_calling_agent(ctx: Context, frozen_clock: Any) -> None:
    spoof = facts.remember(ctx, "x", keywords=["x"], source={"agent": "orang-lain"})
    doc = facts.remember(ctx, "y", keywords=["y"], source="README.md")
    rows = {
        r["id"]: json.loads(r["source"])
        for r in ctx.business.execute("SELECT id, source FROM facts")
    }
    assert rows[spoof.fact_id] == {"agent": "claude-code", "date": "2026-11-11"}
    assert rows[doc.fact_id] == {
        "agent": "claude-code",
        "date": "2026-11-11",
        "document": "README.md",
    }


def test_indonesian_text_round_trips_unchanged(ctx: Context) -> None:
    text = "Café “Kopi Nusantara” — harga Rp 25.000, buka pukul 08.00–21.00 ✓"
    result = facts.remember(ctx, text, keywords=["kafe", "harga"], entities=["Kopi Nusantara"])
    hit = search.recall(ctx, "Nusantara").facts[0]
    assert hit.id == result.fact_id and hit.text == text


# ---- similar facts and conflicts (FR-8, FR-9, FR-11) --------------------------------


def test_similar_fact_never_blocks_the_save_and_is_returned(ctx: Context) -> None:
    first = facts.remember(
        ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"]
    )
    second = facts.remember(
        ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"]
    )
    assert first.fact_id and second.fact_id and first.fact_id != second.fact_id
    assert first.similar == []
    assert second.similar == [
        {"id": first.fact_id, "text": "Supplier Budi kirim Senin", "status": "valid"}
    ]
    assert "supersede" in second.message and "confirm" in second.message
    assert count(ctx.business, "facts") == 2


def test_the_pair_shows_possible_conflict_on_both_facts(ctx: Context) -> None:
    first = facts.remember(
        ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"]
    )
    second = facts.remember(
        ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"]
    )
    hits = {h.id: h.status for h in search.recall(ctx, "supplier Budi").facts}
    assert hits == {first.fact_id: "possible conflict", second.fact_id: "possible conflict"}
    pair = ctx.business.execute("SELECT * FROM conflicts").fetchall()
    assert len(pair) == 1
    assert {pair[0]["fact_a"], pair[0]["fact_b"]} == {first.fact_id, second.fact_id}
    assert pair[0]["resolved_at"] is None


def test_facts_without_a_shared_entity_are_not_paired(ctx: Context) -> None:
    facts.remember(ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"])
    second = facts.remember(
        ctx, "Supplier Ani kirim Selasa", keywords=["supplier"], entities=["Ani"]
    )
    assert second.similar == [] and count(ctx.business, "conflicts") == 0


def test_at_most_three_similar_facts_are_paired(ctx: Context) -> None:
    for n in range(5):
        facts.remember(
            ctx, f"Supplier Budi kirim hari {n}", keywords=["supplier"], entities=["Budi"]
        )
    last = facts.remember(ctx, "Supplier Budi kirim lagi", keywords=["supplier"], entities=["Budi"])
    assert len(last.similar) == 3
    pairs = ctx.business.execute(
        "SELECT count(*) FROM conflicts WHERE fact_a = ?", (last.fact_id,)
    ).fetchone()[0]
    assert pairs == 3


def test_a_pair_with_a_global_fact_lives_in_the_business_file(ctx: Context) -> None:
    prop = facts.remember(
        ctx, "Pemilik tinggal di Medan", keywords=["pemilik"], entities=["Budi"], tier="global"
    )
    assert prop.proposal_id
    glo = proposals.approve(ctx, prop.proposal_id)
    mine = facts.remember(ctx, "Budi tinggal di Bandung", keywords=["tinggal"], entities=["Budi"])
    assert [s["id"] for s in mine.similar] == [glo]
    assert count(ctx.glob, "conflicts") == 0
    assert count(ctx.business, "conflicts") == 1
    assert {h.status for h in search.recall(ctx, "Budi tinggal").facts} == {"possible conflict"}


def test_a_proposal_returns_similar_facts_but_records_no_pair_yet(ctx: Context) -> None:
    mine = facts.remember(
        ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"]
    )
    prop = facts.remember(
        ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"], tier="business"
    )
    assert [s["id"] for s in prop.similar] == [mine.fact_id]
    assert count(ctx.business, "conflicts") == 0
    new_id = proposals.approve(ctx, prop.proposal_id or "")
    assert count(ctx.business, "conflicts") == 1
    assert {h.id for h in search.recall(ctx, "supplier Budi").facts} == {mine.fact_id, new_id}


# ---- tier routing (FR-21, FR-22) ----------------------------------------------------


def test_agent_tier_creates_a_valid_fact_at_once(ctx: Context) -> None:
    result = facts.remember(ctx, "Catatan pribadi", keywords=["catatan"])
    assert result.fact_id and result.proposal_id is None
    assert [h.id for h in search.recall(ctx, "catatan").facts] == [result.fact_id]


@pytest.mark.parametrize(("tier", "file"), [("business", "business"), ("global", "glob")])
def test_business_and_global_tiers_create_only_a_pending_proposal(
    ctx: Context, tier: str, file: str
) -> None:
    result = facts.remember(ctx, "Aturan retur 7 hari", keywords=["retur"], tier=tier)
    conn = getattr(ctx, file)
    assert result.fact_id is None and result.proposal_id and result.proposal_id.startswith("p_")
    assert count(ctx.business, "facts") == 0 and count(ctx.glob, "facts") == 0
    row = conn.execute("SELECT * FROM proposals").fetchone()
    assert (row["id"], row["tier"], row["status"], row["agent"]) == (
        result.proposal_id,
        tier,
        "pending",
        "claude-code",
    )
    assert json.loads(row["payload"])["text"] == "Aturan retur 7 hari"
    assert "approval" in result.message
    assert search.recall(ctx, "retur").facts == []


# ---- episodes (FR-50) ---------------------------------------------------------------


@pytest.mark.parametrize("tier", ["agent", "business", "global"])
def test_remember_with_an_episode_marks_it_processed(ctx: Context, tier: str) -> None:
    episode_id = episodes.record(ctx, "Tolong ingat harga kopi 50 ribu", "Baik, dicatat")
    facts.remember(ctx, "Harga kopi 50 ribu", keywords=["harga"], tier=tier, episode=episode_id)
    row = ctx.business.execute("SELECT status, processed_at FROM episodes").fetchone()
    assert row["status"] == "processed" and row["processed_at"] is not None


def test_an_unknown_episode_is_not_found_and_saves_nothing(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        facts.remember(ctx, "x", keywords=["x"], episode=99)
    assert count(ctx.business, "facts") == 0 and count(ctx.business, "log") == 0


def test_another_agents_episode_cannot_be_used(make_ctx: MakeContext) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    episode_id = episodes.record(a, "halo", "hai")
    with pytest.raises(NotFoundError):
        facts.remember(b, "x", keywords=["x"], episode=episode_id)
    assert count(b.business, "facts") == 0
    assert a.business.execute("SELECT status FROM episodes").fetchone()[0] == "pending"


# ---- the log entry (FR-35) ----------------------------------------------------------


def test_create_writes_one_entry_with_the_full_fact(ctx: Context) -> None:
    result = facts.remember(
        ctx, "Harga kopi 50 ribu", keywords=["harga"], entities=["Kopi"], kind="stable"
    )
    entries = audit.entries(ctx, fact_id=result.fact_id)
    assert len(entries) == 1
    entry = entries[0]
    assert (entry.op, entry.tier, entry.owner, entry.agent) == (
        "create",
        "agent",
        "claude-code",
        "claude-code",
    )
    assert entry.change is not None and entry.change["before"] is None
    after = entry.change["after"]
    for key, value in {
        "id": result.fact_id,
        "text": "Harga kopi 50 ribu",
        "entities": ["Kopi"],
        "keywords": ["harga"],
        "kind": "stable",
        "tier": "agent",
        "agent": "claude-code",
    }.items():
        assert after[key] == value
    assert "valid_from" in after and "source" in after and "seq" not in after


def test_proposal_writes_a_propose_entry_in_the_tier_file(ctx: Context) -> None:
    biz = facts.remember(ctx, "Aturan A", keywords=["a"], tier="business")
    glo = facts.remember(ctx, "Aturan B", keywords=["b"], tier="global")
    biz_log = ctx.business.execute("SELECT * FROM log").fetchall()
    glo_log = ctx.glob.execute("SELECT * FROM log").fetchall()
    assert [(r["op"], r["fact_id"], r["tier"], r["owner"]) for r in biz_log] == [
        ("propose", biz.proposal_id, "business", "claude-code")
    ]
    assert [(r["op"], r["fact_id"], r["tier"], r["owner"]) for r in glo_log] == [
        ("propose", glo.proposal_id, "global", "claude-code")
    ]
