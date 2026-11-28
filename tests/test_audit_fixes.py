"""Regression tests for audit findings: recall ranking/noise, dated snapshots, brief cap,
conflicts dismiss/prune/scan, doctor default_agent, Hermes plugin agent, HTTP approval env,
MCP compact notes/docs, and CLI reject/import/move guards.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ingatan import audit, brief, facts, search, tokens
from ingatan.cli import app
from ingatan.cli import doctor as doctor_module
from ingatan.cli.conflicts import compute_report, prune_conflicts, scan_conflicts
from ingatan.facts import review
from ingatan.integrations.hermes import IngatanProvider
from ingatan.mcp.server import build_multi_server, build_server
from ingatan.scope import Context, EmbeddingConfig, Scope, open_context
from ingatan.search import embed as embed_store
from ingatan.search.stopwords import distinct_dated_snapshots, extract_date_markers

MakeContext = Callable[..., Context]
runner = CliRunner()


def _unit(*vals: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vals))
    return [v / norm for v in vals]


# ---- 1. Recall ranking, generic entity filtering, & off-topic collision gate ---------


def test_exact_keyword_new_fact_outranks_old_facts_with_business_entity(
    make_ctx: MakeContext,
) -> None:
    ctx = make_ctx(business="proyekku", agent="asisten")
    for day in range(7, 15):
        facts.remember(
            ctx,
            f"Jadwal konten proyekku {day} Nov tentang tips menulis dan format laporan.",
            entities=["proyekku", "menulis", "jadwal"],
            keywords=["jadwal", "konten", "menulis", "Nov", "proyekku"],
            tier="business",
            category="procedure",
            valid_from=f"2026-11-{day:02d}T00:00:00Z",
            check_conflicts=False,
        )
    newest = facts.remember(
        ctx,
        "Konten 28 Nov terjadwal: video rumus sampel, buku tebal, pembimbing beda pendapat.",
        entities=[],
        keywords=["28 Nov", "terjadwal", "rumus sampel", "buku tebal", "pembimbing beda pendapat"],
        tier="business",
        valid_from="2026-11-27T14:00:00Z",
        check_conflicts=False,
    )
    assert newest.fact_id is not None
    res = search.recall(
        ctx,
        "proyekku 28 Nov terjadwal rumus sampel buku tebal pembimbing",
        limit=5,
        touch=False,
    )
    assert res.facts
    assert res.facts[0].id == newest.fact_id


def test_off_topic_single_word_collision_returns_empty_for_facts_and_notes(
    ctx: Context,
) -> None:
    for i in range(8):
        facts.remember(
            ctx,
            f"Catatan operasional proyek nomor {i} tentang server cadangan.",
            entities=["tokoku"],
            tier="business",
            kind="stable",
            check_conflicts=False,
        )
    r1 = facts.remember(
        ctx,
        "Resep template video animasi standar berdurasi tiga puluh detik.",
        entities=["video"],
        tier="business",
        kind="stable",
    )
    facts.remember(
        ctx,
        "Resep opening singkat tiga detik untuk klip media sosial.",
        entities=["klip"],
        tier="business",
        kind="stable",
    )
    facts.remember(
        ctx,
        "Resep pacing edit cepat untuk video pendek.",
        entities=["video"],
        tier="business",
        kind="stable",
    )
    for i in range(4):
        ctx.business.execute(
            "INSERT INTO notes (path, heading, text, file_hash, created_at) "
            "VALUES (?, ?, ?, 'h', '2026-11-11T09:00:00Z')",
            (f"/docs/doc_{i}.md", f"Panduan {i}", "Resep template animasi standar tim produksi."),
        )

    off = search.recall(ctx, "resep rendang padang", limit=5, touch=False)
    assert off.facts == []
    assert off.notes == []

    on_topic = search.recall(ctx, "resep template video", limit=5, touch=False)
    assert r1.fact_id and on_topic.facts and on_topic.facts[0].id == r1.fact_id


def test_overdue_check_first_fact_yields_to_fresh_valid_fact_on_tie(
    ctx: Context, frozen_clock: Any
) -> None:
    old = facts.remember(
        ctx,
        "Estimasi waktu kirim kurir kota sekitar dua hari kerja.",
        keywords=["estimasi", "waktu", "kirim"],
        kind="volatile",
        review_after="2026-11-15",
        check_conflicts=False,
    )
    frozen_clock.advance(days=10)
    fresh = facts.remember(
        ctx,
        "Estimasi waktu kirim kurir kota sekitar satu hari kerja.",
        keywords=["estimasi", "waktu", "kirim"],
        kind="volatile",
        check_conflicts=False,
    )
    assert old.fact_id and fresh.fact_id
    hits = search.recall(ctx, "estimasi waktu kirim kurir kota", touch=False).facts
    assert [h.id for h in hits[:2]] == [fresh.fact_id, old.fact_id]
    assert hits[0].status == "valid" and hits[1].status == "check first"


# ---- 2. Periodic dated snapshot conflict guard ---------------------------------------


def test_extract_date_markers_and_distinct_dated_snapshots() -> None:
    assert "28/11" in extract_date_markers("Update performa iklan 28 Nov: spend Rp 500rb")
    assert "w47" in extract_date_markers("Worklog W47: rapikan halaman utama")
    assert distinct_dated_snapshots(
        "Update performa iklan 22 Okt: spend Rp 400rb, 9 prospek.",
        "Update performa iklan 28 Nov: spend Rp 520rb, 14 prospek.",
    )
    assert distinct_dated_snapshots(
        "Worklog W46: audit kecepatan halaman.",
        "Worklog W47: perbaiki tautan rusak.",
    )
    # Effective-date standing updates must NOT be treated as periodic snapshots.
    assert not distinct_dated_snapshots(
        "Harga kopi mulai 1 Nov adalah Rp 15.000.",
        "Harga kopi mulai 10 Nov adalah Rp 18.000.",
    )
    # Same-date updates must still conflict.
    assert not distinct_dated_snapshots(
        "Update performa iklan 28 Nov: spend Rp 400rb.",
        "Update performa iklan 28 Nov: spend Rp 520rb.",
    )


def test_volatile_dated_snapshots_do_not_conflict_even_with_high_semantic_similarity(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def same_vec(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        return _unit(1.0, 0.0, 0.0)

    monkeypatch.setattr(embed_store, "_embed_raw", same_vec)
    scope = Scope(
        business="tokoku",
        agent="asisten",
        home=ingatan_home,
        embeddings=EmbeddingConfig(provider="local", model="xenova/multilingual-e5-small"),
    )
    with open_context(scope) as ctx:
        f1 = facts.remember(
            ctx,
            "Update performa iklan 22 Okt: belanja Rp 400.000, prospek 9.",
            entities=["Iklan"],
            tier="business",
            kind="volatile",
        )
        f2 = facts.remember(
            ctx,
            "Update performa iklan 28 Nov: belanja Rp 520.000, prospek 14.",
            entities=["Iklan"],
            tier="business",
            kind="volatile",
        )
        assert f1.fact_id and f2.fact_id
        assert f2.similar == []
        assert review.open_pairs(ctx) == []

        # Standing rule/price update with effective date still flags conflict.
        p1 = facts.remember(
            ctx,
            "Harga paket Pro mulai 1 Nov adalah Rp 49.000.",
            entities=["Paket Pro"],
            tier="business",
            kind="volatile",
        )
        p2 = facts.remember(
            ctx,
            "Harga paket Pro mulai 10 Nov adalah Rp 59.000.",
            entities=["Paket Pro"],
            tier="business",
            kind="volatile",
        )
        assert p1.fact_id and p2.fact_id
        assert [s["id"] for s in p2.similar] == [p1.fact_id]


# ---- 3. Brief check-first cap preserves pinned / core facts --------------------------


def test_brief_caps_check_first_backlog_so_pinned_and_core_facts_survive(
    make_ctx: MakeContext, frozen_clock: Any
) -> None:
    ctx = make_ctx(token_budget=250)
    pinned = facts.remember(
        ctx,
        "Aturan utama: harga paket premium Rp 49.000 per bulan.",
        entities=["paket premium"],
        tier="business",
        kind="stable",
        category="rule",
        pinned=True,
    )
    for i in range(15):
        facts.remember(
            ctx,
            f"Catatan harian lama ke-{i}: metrik iklan tanggal {i + 1} Oktober Rp {100 + i}rb.",
            entities=[f"log-{i}"],
            tier="business",
            kind="volatile",
            review_after="2026-10-20",
            check_conflicts=False,
        )
    text = brief.build(ctx)
    assert brief.estimate_tokens(text) <= 250
    assert pinned.fact_id and pinned.fact_id in text
    assert "## Check first" in text
    assert "more fact(s) to check" in text
    assert "## Business" in text


def test_brief_caps_check_first_item_count_on_large_budget(
    make_ctx: MakeContext, frozen_clock: Any
) -> None:
    ctx = make_ctx(token_budget=1000)
    core = facts.remember(
        ctx,
        "Panduan inti layanan untuk seluruh agen bisnis.",
        tier="business",
        kind="stable",
        pinned=True,
    )
    for i in range(30):
        facts.remember(
            ctx,
            f"Catatan harian panjang nomor {i:02d}: ringkasan metrik harian yang belum diperiksa "
            "ulang sejak bulan lalu oleh tim operasional.",
            tier="business",
            kind="volatile",
            review_after="2026-10-15",
            check_conflicts=False,
        )
    text = brief.build(ctx)
    assert brief.estimate_tokens(text) <= 1000
    assert core.fact_id and core.fact_id in text
    assert text.count("review was due") <= brief.CHECK_FIRST_MAX_ITEMS
    assert "more fact(s) to check" in text


# ---- 4. Conflicts dismiss, prune, and scan vector reuse + audit log ------------------


def test_review_dismiss_closes_conflict_without_creating_edge_and_is_undoable(
    ctx: Context, frozen_clock: Any
) -> None:
    f1 = facts.remember(ctx, "Harga kopi susu Rp 15.000.", entities=["kopi susu"])
    frozen_clock.advance(days=1)
    f2 = facts.remember(ctx, "Harga kopi susu Rp 18.000.", entities=["kopi susu"])
    assert f1.fact_id and f2.fact_id
    (pair,) = review.open_pairs(ctx)

    outcome = review.resolve_pair(ctx, pair, "dismiss")
    assert "not linked" in outcome.message
    assert review.open_pairs(ctx) == []
    assert ctx.business.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
    assert compute_report(audit.entries(ctx)).resolved == 1

    # Undo restores the open conflict.
    confirm_entry = next(e for e in audit.entries(ctx) if e.op == "confirm")
    audit.undo(ctx, confirm_entry.id)
    assert [p.id for p in review.open_pairs(ctx)] == [pair.id]


def test_cli_conflicts_review_dismiss_and_prune_legacy_false_positives(
    ingatan_home: Path,
) -> None:
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text('default_agent = "asisten"\n', encoding="utf-8")
    scope = Scope(business="tokoku", agent="asisten", home=ingatan_home)
    with open_context(scope) as ctx:
        # Two completely unrelated facts forced into a legacy open conflict pair.
        a = facts.remember(
            ctx, "Jadwal perawatan mesin kopi tiap Senin.", entities=["mesin"], tier="business"
        )
        b = facts.remember(
            ctx,
            "Domain situs utama diperpanjang sampai 2028.",
            entities=["domain"],
            tier="business",
        )
        # Two genuinely conflicting facts.
        c1 = facts.remember(
            ctx, "Tarif kurir JNE Bandung Rp 12.000.", entities=["JNE"], tier="business"
        )
        c2 = facts.remember(
            ctx, "Tarif kurir JNE Bandung Rp 15.000.", entities=["JNE"], tier="business"
        )
        assert a.fact_id and b.fact_id and c1.fact_id and c2.fact_id
        ctx.business.execute(
            "INSERT INTO conflicts (fact_a, fact_b, created_at) VALUES (?, ?, ?)",
            (a.fact_id, b.fact_id, "2026-11-11T09:00:00Z"),
        )
        assert len(review.open_pairs(ctx)) == 2

    dry = runner.invoke(app, ["conflicts", "prune", "--dry-run", "-b", "tokoku"])
    assert dry.exit_code == 0, dry.output
    assert "1 false positive(s) would be dismissed, 1 would remain open" in dry.stdout

    pruned = runner.invoke(app, ["conflicts", "prune", "-b", "tokoku"])
    assert pruned.exit_code == 0, pruned.output
    assert "dismissed 1 false positive(s) (not linked), 1 still open" in pruned.stdout

    with open_context(scope) as ctx:
        # Legacy false positive was dismissed with ZERO bogus edges; real conflict stays open.
        assert ctx.business.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
        remaining = review.open_pairs(ctx)
        assert len(remaining) == 1
        assert {remaining[0].fact_a, remaining[0].fact_b} == {c1.fact_id, c2.fact_id}
        assert prune_conflicts(ctx, dry_run=True).false_positives == 0

    # Dismiss the remaining pair via CLI `--apply dismiss --all`.
    dismiss_all = runner.invoke(
        app, ["conflicts", "review", "--apply", "dismiss", "--all", "-b", "tokoku"]
    )
    assert dismiss_all.exit_code == 0, dismiss_all.output
    assert "not linked" in dismiss_all.stdout
    with open_context(scope) as ctx:
        assert review.open_pairs(ctx) == []
        assert ctx.business.execute("SELECT count(*) FROM edges").fetchone()[0] == 0


def test_scan_conflicts_reuses_stored_vectors_and_writes_audit_log(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embed_calls: list[str] = []

    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        embed_calls.append(text)
        return _unit(1.0, 0.0, 0.0)

    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)
    scope = Scope(
        business="tokoku",
        agent="asisten",
        home=ingatan_home,
        embeddings=EmbeddingConfig(provider="local", model="test-generic-model"),
    )
    with open_context(scope) as ctx:
        # Entity-less semantically conflicting facts saved with check_conflicts=False.
        f1 = facts.remember(
            ctx,
            "Jam buka gerai dimulai pukul 08:00 pagi.",
            entities=[],
            tier="business",
            check_conflicts=False,
        )
        f2 = facts.remember(
            ctx,
            "Waktu operasional toko dimulai pukul 10:00 siang.",
            entities=[],
            tier="business",
            check_conflicts=False,
        )
        assert f1.fact_id and f2.fact_id
        assert len(embed_calls) == 2

        # Scanning reuses stored vectors (0 new embed calls!) and still catches entity-less pair.
        embed_calls.clear()
        res = scan_conflicts(ctx)
        assert embed_calls == []
        assert res.pairs_recorded == 1
        # Audit log recorded the scanned conflict so report & undo see it.
        conflict_entries = [e for e in audit.entries(ctx) if e.op == "conflict"]
        assert len(conflict_entries) == 1
        assert compute_report(audit.entries(ctx)).candidates == 1


# ---- 5. Doctor default_agent & Hermes plugin agent resolution ------------------------


def test_doctor_reads_default_agent_from_home_config_without_contradiction(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INGATAN_AGENT", raising=False)
    monkeypatch.delenv("INGATAN_BUSINESS", raising=False)
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text('default_agent = "asisten"\n', encoding="utf-8")

    checks = doctor_module.run_checks(business="proyekku", probe_embeddings=False)
    cfg = next(c for c in checks if c.name == "config")
    assert cfg.status == "pass"
    assert "agent asisten" in cfg.detail
    rendered = "\n".join(cfg.notes)
    assert "agent: asisten (from " in rendered and "config.toml" in rendered
    assert "agent: not set" not in rendered


def test_hermes_plugin_honours_home_config_default_agent(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")
    monkeypatch.delenv("INGATAN_AGENT", raising=False)
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text('default_agent = "asisten"\n', encoding="utf-8")

    provider = IngatanProvider()
    provider.initialize("sess-cfg", agent_context="primary")
    provider.sync_turn("Harga kopi?", "Rp 18.000")

    with open_context(Scope(business="tokoku", agent="asisten", home=ingatan_home)) as ctx:
        rows = ctx.business.execute("SELECT agent FROM episodes").fetchall()
        assert [r["agent"] for r in rows] == ["asisten"]


# ---- 6. HTTP TokenStore honours INGATAN_REQUIRE_APPROVAL -----------------------------


def test_token_store_honours_require_approval_env_override(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text("require_approval = false\n", encoding="utf-8")
    raw, _ = tokens.add(ingatan_home, "tokoku", "asisten")
    store = tokens.TokenStore(ingatan_home)

    monkeypatch.setenv("INGATAN_REQUIRE_APPROVAL", "1")
    scope_on = store.lookup(raw)
    assert scope_on is not None and scope_on.require_approval is True

    (ingatan_home / "config.toml").write_text("require_approval = true\n", encoding="utf-8")
    monkeypatch.setenv("INGATAN_REQUIRE_APPROVAL", "0")
    scope_off = store.lookup(raw)
    assert scope_off is not None and scope_off.require_approval is False


# ---- 7. MCP tool docs & compact recall default notes=0 -------------------------------


def test_mcp_tool_docs_name_parameters_and_compact_defaults_notes_zero(
    ctx: Context, ingatan_home: Path
) -> None:
    from mcp import Client

    facts.remember(ctx, "Harga kopi susu Rp 18.000", keywords=["kopi", "harga"])
    ctx.business.execute(
        "INSERT INTO notes (path, heading, text, file_hash, created_at) "
        "VALUES ('/docs/menu.md', 'Menu Kopi', 'Penjelasan panjang soal kopi susu', 'h', "
        "'2026-11-11T09:00:00Z')"
    )

    async def exercise() -> None:
        async with Client(build_server(ctx)) as client:
            tools_map = {t.name: t.description or "" for t in (await client.list_tools()).tools}
            assert tools_map["remember"].startswith("Remember(text,")
            assert tools_map["supersede"].startswith("Supersede(old, new)")
            assert tools_map["history"].startswith("History(id_or_entity)")
            assert "not embeddings" not in tools_map["recall"].lower()

            # Default compact=False includes excerpts.
            full_res = await client.call_tool("recall", {"query": "kopi"})
            assert "Document excerpts" in getattr(full_res.content[0], "text", "")

            # compact=True without explicit notes defaults notes=0.
            compact_res = await client.call_tool("recall", {"query": "kopi", "compact": True})
            compact_text = getattr(compact_res.content[0], "text", "")
            assert "Harga kopi susu" in compact_text
            assert "Document excerpts" not in compact_text

            # Explicit notes=1 still works with compact=True.
            compact_notes = await client.call_tool(
                "recall", {"query": "kopi", "compact": True, "notes": 1}
            )
            assert "Document excerpts" in getattr(compact_notes.content[0], "text", "")

        server, opened = build_multi_server(["tokoku"], agent="claude-code", home=ingatan_home)
        try:
            async with Client(server) as mclient:
                mcompact = await mclient.call_tool(
                    "recall", {"business": "tokoku", "query": "kopi", "compact": True}
                )
                assert "Document excerpts" not in getattr(mcompact.content[0], "text", "")
        finally:
            for c in opened.values():
                c.close()

    asyncio.run(exercise())


# ---- 8. CLI proposals reject --all, import --dry-run, and move closed guard ----------


def test_cli_proposals_reject_multi_and_all(ingatan_home: Path) -> None:
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text(
        'default_agent = "asisten"\nrequire_approval = true\n', encoding="utf-8"
    )
    scope = Scope(business="tokoku", agent="asisten", home=ingatan_home, require_approval=True)
    with open_context(scope) as ctx:
        p1 = facts.remember(ctx, "Usulan satu", tier="business").proposal_id
        p2 = facts.remember(ctx, "Usulan dua", tier="business").proposal_id
        p3 = facts.remember(ctx, "Usulan tiga", tier="business").proposal_id
        assert p1 and p2 and p3

    multi = runner.invoke(app, ["proposals", "reject", p1, p2, "-b", "tokoku"])
    assert multi.exit_code == 0, multi.output
    assert f"Rejected {p1}" in multi.stdout and f"Rejected {p2}" in multi.stdout

    rest = runner.invoke(app, ["proposals", "reject", "--all", "-b", "tokoku"])
    assert rest.exit_code == 0, rest.output
    assert f"[1/1] Rejected {p3}" in rest.stdout

    empty = runner.invoke(app, ["proposals", "reject", "--all", "-b", "tokoku"])
    assert empty.exit_code == 0 and "No pending proposals." in empty.stdout


def test_cli_import_dry_run_and_move_refuses_closed_fact(
    ingatan_home: Path, tmp_path: Path
) -> None:
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text('default_agent = "asisten"\n', encoding="utf-8")
    scope = Scope(business="toko-a", agent="asisten", home=ingatan_home)
    with open_context(scope) as ctx:
        old = facts.remember(ctx, "Harga lama Rp 10.000", tier="business")
        new = facts.remember(ctx, "Harga baru Rp 15.000", tier="business")
        assert old.fact_id and new.fact_id
        facts.supersede(ctx, old.fact_id, new.fact_id)

    # Moving a closed fact must fail and leave destination untouched.
    bad_move = runner.invoke(app, ["move", old.fact_id, "-b", "toko-a", "--to", "toko-b", "--yes"])
    assert bad_move.exit_code == 1
    assert "closed" in bad_move.stderr.lower()
    assert not (ingatan_home / "toko-b.db").exists()

    # Dry-run import validates and previews exact duplicate count without writing.
    jsonl = tmp_path / "bulk.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"text": "Harga baru Rp 15.000", "tier": "business"}),
                json.dumps({"text": "Jam buka toko 08.00 WIB", "tier": "business"}),
            ]
        ),
        encoding="utf-8",
    )
    dry = runner.invoke(app, ["import", str(jsonl), "--dry-run", "-b", "toko-a"])
    assert dry.exit_code == 0, dry.output
    assert "1 new, 1 exact duplicate(s) to refresh" in dry.stdout
    with open_context(scope) as ctx:
        assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 2

    real = runner.invoke(app, ["import", str(jsonl), "-b", "toko-a"])
    assert real.exit_code == 0, real.output
    assert "Imported 2 fact(s) into toko-a. (1 refreshed)" in real.stdout
    with open_context(scope) as ctx:
        assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 3
