"""Regression & feature tests for general search, conflict, CLI, and MCP improvements."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from ingatan import audit, brief, facts, proposals, search, tokens
from ingatan.cli import app
from ingatan.cli.conflicts import compute_report, open_conflicts, scan_conflicts
from ingatan.ingest import ingest_path
from ingatan.mcp.server import NOTE_EXCERPT_CHARS, _recall_text, build_multi_server
from ingatan.scope import Context, resolve_allowed_businesses, resolve_scope

MakeContext = Callable[..., Context]


# ---- 1. Conflict detection precision & lifecycle bug fixes ---------------------------


def test_unrelated_facts_with_common_entity_and_boilerplate_do_not_conflict(ctx: Context) -> None:
    facts.remember(
        ctx,
        "Transaksi bank 10 Nov transfer Rp 500.000 untuk pembayaran listrik. "
        "Disimpan oleh asisten.",
        keywords=["bank", "transaksi", "listrik"],
        entities=["tokoku", "Budi"],
    )
    second = facts.remember(
        ctx,
        "Postingan artikel soal pembaruan halaman utama bulan ini sudah tayang. "
        "Disimpan oleh asisten.",
        keywords=["artikel", "pembaruan", "halaman"],
        entities=["tokoku", "Budi"],
    )
    assert second.similar == []
    assert ctx.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0


def test_distinct_google_sheet_ids_do_not_conflict(ctx: Context) -> None:
    facts.remember(
        ctx,
        "Lokasi Google Sheet data cuaca gunung id 1AbCdEfGhIjKlMnOpQrStUvWxYz123456.",
        keywords=["cuaca", "gunung", "sheet"],
        entities=["Google Sheet", "Gunung"],
    )
    second = facts.remember(
        ctx,
        "Lokasi Google Sheet pembukuan bulanan id 1ZyXwVuTsRqPoNmLkJiHgFeDcBa987654.",
        keywords=["pembukuan", "bulanan", "sheet"],
        entities=["Google Sheet"],
    )
    assert second.similar == []
    assert ctx.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0


def test_different_subtopics_with_same_entity_do_not_conflict(ctx: Context) -> None:
    facts.remember(
        ctx,
        "Prakiraan cuaca puncak Gunung 21 Nov hujan ringan dan angin kencang 25 km/jam.",
        keywords=["cuaca", "puncak", "hujan"],
        entities=["Gunung"],
    )
    jacket = facts.remember(
        ctx,
        "Jaket pendakian Gunung memakai merek outdoor ukuran L warna hitam.",
        keywords=["jaket", "outdoor", "perlengkapan"],
        entities=["Gunung"],
    )
    assert jacket.similar == []


def test_check_conflicts_false_skips_conflict_detection(ctx: Context) -> None:
    facts.remember(
        ctx, "Log harian spend iklan 14 Nov Rp 200rb", keywords=["log", "spend"], entities=["Ads"]
    )
    second = facts.remember(
        ctx,
        "Log harian spend iklan 15 Nov Rp 220rb",
        keywords=["log", "spend"],
        entities=["Ads"],
        check_conflicts=False,
    )
    assert second.similar == []
    assert ctx.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0


def test_expired_partner_does_not_keep_fact_stuck_in_possible_conflict(
    ctx: Context, frozen_clock: Any
) -> None:
    a = facts.remember(ctx, "Supplier Budi kirim Senin", keywords=["supplier"], entities=["Budi"])
    b = facts.remember(ctx, "Supplier Budi kirim Selasa", keywords=["supplier"], entities=["Budi"])
    assert a.fact_id and b.fact_id
    assert search.recall(ctx, "supplier Budi", touch=False).facts[0].status == "possible conflict"
    # Expire `a` directly (simulating valid_to passing without explicit supersede).
    ctx.business.execute(
        "UPDATE facts SET valid_to = '2026-11-11T08:00:00Z' WHERE id = ?", (a.fact_id,)
    )
    hits = search.recall(ctx, "supplier Budi", touch=False).facts
    assert [(h.id, h.status) for h in hits] == [(b.fact_id, "valid")]


def test_approve_recomputes_similar_and_resolves_superseded_target_conflicts(
    approval_ctx: Context,
) -> None:
    ctx = approval_ctx
    # Queue two conflicting business proposals while DB is still empty (migration scenario).
    p1 = facts.remember(
        ctx,
        "Budget iklan bulanan Meta Rp 5 juta",
        keywords=["budget", "iklan"],
        entities=["Meta Ads"],
        tier="business",
    )
    p2 = facts.remember(
        ctx,
        "Budget iklan bulanan Meta Rp 8 juta",
        keywords=["budget", "iklan"],
        entities=["Meta Ads"],
        tier="business",
    )
    assert p1.proposal_id and p2.proposal_id
    f1 = proposals.approve(ctx, p1.proposal_id)
    f2 = proposals.approve(ctx, p2.proposal_id)
    # Approval of p2 re-checked against live f1 and opened a conflict pair!
    assert {h.status for h in search.recall(ctx, "budget iklan Meta", touch=False).facts} == {
        "possible conflict"
    }
    # Now supersede f1 with existing valid f2 (retire-only proposal) and approve.
    q = facts.supersede(ctx, f1, f2)
    assert q.proposal_id is not None
    kept = proposals.approve(ctx, q.proposal_id)
    assert kept == f2
    assert [
        (h.id, h.status) for h in search.recall(ctx, "budget iklan Meta", touch=False).facts
    ] == [(f2, "valid")]
    report = compute_report(audit.entries(ctx))
    assert report.candidates == 1 and report.resolved == 1


# ---- 2. Recall precision, identifier matching, & notes token control -----------------


def test_underscore_identifier_ranks_script_fact_first(ctx: Context, frozen_clock: Any) -> None:
    script = facts.remember(
        ctx,
        "Pipeline update Ads report memakai skrip sync_report.py.",
        keywords=["pipeline", "skrip", "ads"],
        entities=["Ads"],
        kind="stable",
    )
    frozen_clock.advance(days=5)
    facts.remember(
        ctx,
        "Update Ads report sheet campaign laporan harian selesai dicek.",
        keywords=["update", "ads", "report", "sheet", "campaign", "laporan"],
        entities=["Ads"],
        kind="volatile",
    )
    hits = search.recall(
        ctx, "update ads report sheet sync_report pipeline campaign laporan", limit=5, touch=False
    ).facts
    assert hits and hits[0].id == script.fact_id


def test_long_unrelated_query_returns_empty_instead_of_noise(ctx: Context) -> None:
    facts.remember(
        ctx,
        "Audit server bulanan selesai dibuat di dokumen internal.",
        keywords=["audit", "server", "dokumen"],
        entities=["Server"],
    )
    facts.remember(
        ctx,
        "Konfigurasi data MCP dibuat kemarin malam.",
        keywords=["mcp", "konfigurasi", "data"],
        entities=["MCP"],
    )
    result = search.recall(
        ctx,
        "product knowledge pengetahuan produk knowledge base treatment dokumen dibuat",
        limit=5,
        touch=False,
    )
    assert result.facts == []


def test_recall_notes_limit_and_compact_formatting(ctx: Context) -> None:
    facts.remember(ctx, "Harga kopi Rp 20.000", keywords=["kopi", "harga"])
    long_text = "Kopi arabika gayo " + ("penjelasan panjang " * 60)
    for i in range(4):
        ctx.business.execute(
            "INSERT INTO notes (path, heading, text, file_hash, created_at) "
            "VALUES (?, ?, ?, 'h', '2026-11-11T09:00:00Z')",
            (f"/docs/menu_{i}.md", f"Kopi {i}", long_text),
        )
    # Default notes_limit is 2, not limit=5.
    res_default = search.recall(ctx, "kopi", limit=5, touch=False)
    assert len(res_default.notes) == 2
    # notes_limit=0 disables excerpts entirely.
    res_none = search.recall(ctx, "kopi", limit=5, notes_limit=0, touch=False)
    assert res_none.notes == []
    # Excerpt clipping at NOTE_EXCERPT_CHARS (400) and compact mode.
    rendered = _recall_text("kopi", res_default, False)
    assert "…" in rendered
    assert NOTE_EXCERPT_CHARS == 400
    compact = _recall_text("kopi", res_none, False, compact=True)
    assert "source:" not in compact and "(valid)" in compact


# ---- 3. Brief stable-first cold-start ranking ----------------------------------------


def test_brief_prefers_stable_facts_over_newer_volatile_on_cold_start(
    make_ctx: MakeContext, frozen_clock: Any
) -> None:
    ctx = make_ctx(token_budget=120)
    stable = facts.remember(
        ctx,
        "Harga Premium resmi Rp49.000 per bulan.",
        keywords=["harga", "premium"],
        tier="business",
        kind="stable",
    )
    for i in range(5):
        frozen_clock.advance(hours=1)
        facts.remember(
            ctx,
            f"Log harian nomor {i}: catatan perubahan kecil jadwal posting yang cukup panjang.",
            keywords=["log", "jadwal"],
            tier="business",
            kind="volatile",
        )
    text = brief.build(ctx)
    assert stable.fact_id and stable.fact_id in text
    assert len(text) / 3.5 <= 120


# ---- 4. Preserving source.date, global config.toml, & TokenStore ---------------------


def test_remember_preserves_explicit_source_date_and_valid_from(ctx: Context) -> None:
    res = facts.remember(
        ctx,
        "Harga awal produk Rp 100.000",
        keywords=["harga"],
        source={"document": "import:legacy", "date": "2025-08-15"},
        valid_from="2025-08-15",
    )
    assert res.fact_id
    hit = search.recall(ctx, "harga awal", touch=False).facts[0]
    assert hit.date == "2025-08-15"
    row = ctx.business.execute(
        "SELECT valid_from FROM facts WHERE id = ?", (res.fact_id,)
    ).fetchone()
    assert row["valid_from"] == "2025-08-15T00:00:00Z"


def test_home_config_toml_applies_to_resolve_scope_and_token_store(
    ingatan_home: Path, tmp_path: Path
) -> None:
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text(
        'default_agent = "hermes"\nrequire_approval = true\n\n[brief]\ntoken_budget = 650\n',
        encoding="utf-8",
    )
    scope = resolve_scope(
        tmp_path, env={"INGATAN_HOME": str(ingatan_home), "INGATAN_BUSINESS": "toko-a"}
    )
    assert (scope.business, scope.agent, scope.token_budget, scope.require_approval) == (
        "toko-a",
        "hermes",
        650,
        True,
    )
    raw, _ = tokens.add(ingatan_home, "toko-a", "hermes")
    t_scope = tokens.TokenStore(ingatan_home).lookup(raw)
    assert t_scope is not None
    assert (t_scope.token_budget, t_scope.require_approval) == (650, True)


# ---- 5. CLI upgrades (--business, approve --all, import, conflicts, move, ingest) ----


def test_cli_business_flag_import_conflicts_scan_move_and_approve_all(
    ingatan_home: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    # Set global default_agent so `--business` works from any folder without .ingatan.toml.
    ingatan_home.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text('default_agent = "hermes"\n', encoding="utf-8")

    # 1. Bulk import JSONL with original dates + conflict scan.
    jsonl = tmp_path / "dump.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": "Tarif pengiriman kurir JNE kota Rp 12.000",
                        "entities": ["JNE"],
                        "date": "2026-09-01",
                        "kind": "stable",
                    }
                ),
                json.dumps(
                    {
                        "text": "Tarif pengiriman kurir JNE kota Rp 15.000",
                        "entities": ["JNE"],
                        "date": "2026-10-01",
                        "kind": "stable",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    imp = runner.invoke(app, ["import", str(jsonl), "--business", "toko-a", "--scan-conflicts"])
    assert imp.exit_code == 0, imp.output
    assert "Imported 2 fact(s) into toko-a." in imp.stdout
    assert "recorded 1 new conflict pair(s)" in imp.stdout

    # 2. `ingatan conflicts list --business toko-a`
    listed = runner.invoke(app, ["conflicts", "list", "--business", "toko-a"])
    assert listed.exit_code == 0, listed.output
    assert "Tarif pengiriman kurir JNE" in listed.stdout

    # 3. `ingatan conflicts scan --dry-run` finds 0 new pairs (already recorded).
    dry = runner.invoke(app, ["conflicts", "scan", "--dry-run", "--business", "toko-a"])
    assert dry.exit_code == 0 and "found 0 candidate pair(s)" in dry.stdout

    # 4. `ingatan move <id> --business toko-a --to toko-b --yes`
    log_out = runner.invoke(app, ["log", "--business", "toko-a"])
    assert log_out.exit_code == 0
    first_id = next(tok for tok in log_out.stdout.split() if tok.startswith("f_"))
    moved = runner.invoke(
        app, ["move", first_id, "--business", "toko-a", "--to", "toko-b", "--yes"]
    )
    assert moved.exit_code == 0, moved.output
    assert "Moved" in moved.stdout and "to toko-b" in moved.stdout

    # 5. `ingatan proposals approve --all --business toko-a`
    env_app = {"INGATAN_HOME": str(ingatan_home), "INGATAN_REQUIRE_APPROVAL": "1"}
    scope_app = resolve_scope(tmp_path, env={**env_app, "INGATAN_BUSINESS": "toko-a"})
    from ingatan.scope import open_context

    with open_context(scope_app) as sctx:
        facts.remember(sctx, "Aturan satu", keywords=["aturan"], tier="business")
        facts.remember(sctx, "Aturan dua", keywords=["aturan"], tier="business")
        assert len(open_conflicts(sctx)) == 0
        assert scan_conflicts(sctx, dry_run=True).pairs_found == 0
    app_all = runner.invoke(app, ["proposals", "approve", "--all", "--business", "toko-a"])
    assert app_all.exit_code == 0, app_all.output
    assert "[1/2] Approved" in app_all.stdout and "[2/2] Approved" in app_all.stdout


def test_ingest_skips_hidden_and_vendor_directories(ctx: Context, tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Good\nValid doc\n", encoding="utf-8")
    for hidden in (".git", ".venv", "node_modules", "__pycache__"):
        d = tmp_path / hidden
        d.mkdir()
        (d / "ignore.md").write_text("# Secret\nShould be skipped\n", encoding="utf-8")
    report = ingest_path(ctx, tmp_path)
    assert report.files == 1 and report.created == 1


# ---- 6. Multi-business MCP allowlist mode --------------------------------------------


def test_resolve_allowed_businesses_and_build_multi_server(
    ingatan_home: Path, tmp_path: Path
) -> None:
    import asyncio

    from mcp import Client

    multi = resolve_allowed_businesses(
        tmp_path,
        env={
            "INGATAN_HOME": str(ingatan_home),
            "INGATAN_BUSINESSES": "toko-a, toko-b, kantor",
            "INGATAN_AGENT": "hermes",
        },
    )
    assert multi is not None
    businesses, agent, home, budget, req = multi
    assert businesses == ["toko-a", "toko-b", "kantor"]
    assert agent == "hermes" and home == ingatan_home and budget == 1000 and req is False
    server, opened = build_multi_server(businesses, agent=agent, home=home)

    async def exercise() -> None:
        async with Client(server) as client:
            tools = (await client.list_tools()).tools
            assert len(tools) == 7
            # Save in toko-a, verify isolation from toko-b, reject unauthorized business.
            saved = await client.call_tool(
                "remember",
                {"business": "toko-a", "text": "Harga paket Pro Rp49.000", "tier": "business"},
            )
            assert not saved.is_error
            hit_a = await client.call_tool(
                "recall", {"business": "toko-a", "query": "harga paket pro"}
            )
            assert "Rp49.000" in getattr(hit_a.content[0], "text", "")
            hit_b = await client.call_tool(
                "recall", {"business": "toko-b", "query": "harga paket pro"}
            )
            assert "No facts found" in getattr(hit_b.content[0], "text", "")
            denied = await client.call_tool("recall", {"business": "rahasia", "query": "harga"})
            assert denied.is_error

    try:
        asyncio.run(exercise())
    finally:
        for ctx in opened.values():
            ctx.close()
