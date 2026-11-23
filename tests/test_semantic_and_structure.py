"""Tests for Phase 2: semantic hybrid search, deduplication, time-travel, categories/pin,
project aliases, relation edges, CLI commands, and MCP schema invariants.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ingatan import brief, db, facts, search, tokens
from ingatan.cli import app
from ingatan.cli import doctor as doctor_checks
from ingatan.errors import ConfigError, NotFoundError, ValidationError
from ingatan.ingest import ingest_path
from ingatan.mcp.server import _recall_text, build_server
from ingatan.scope import Context, EmbeddingConfig, Scope, open_context, resolve_scope
from ingatan.search import embed as embed_store

runner = CliRunner()


def _unit(*vals: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vals))
    return [v / norm for v in vals]


def _make_embed_ctx(
    ingatan_home: Path,
    *,
    business: str = "tokoku",
    agent: str = "claude-code",
    provider: str = "local",
    model: str = "test-model",
) -> Context:
    scope = Scope(
        business=business,
        agent=agent,
        home=ingatan_home,
        embeddings=EmbeddingConfig(provider=provider, model=model),
    )
    return open_context(scope)


# ---- 1. Schema migration v1 -> v2 & config parsing -----------------------------------


def test_migration_v1_to_v2_preserves_existing_rows_and_adds_tables(tmp_path: Path) -> None:
    migrations = db.load_migrations()
    assert len(migrations) >= 2
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        db.migrate(conn, migrations=[migrations[0]])
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        conn.execute(
            "INSERT INTO facts (id, text, entities, keywords, tier, agent, kind, "
            "valid_from, source, created_at) "
            "VALUES ('f_old000000001', 'Harga kopi Rp 20.000', '[]', '[\"kopi\"]', "
            "'business', 'claude-code', 'stable', '2026-11-01T00:00:00Z', '{}', "
            "'2026-11-01T00:00:00Z')"
        )
        applied = db.migrate(conn)
        assert 2 in applied
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.LATEST_VERSION
        row = conn.execute(
            "SELECT category, pinned FROM facts WHERE id = 'f_old000000001'"
        ).fetchone()
        assert row["category"] == "fact"
        assert row["pinned"] == 0
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {"fact_embeddings", "note_embeddings", "aliases", "edges"} <= tables
    finally:
        conn.close()


def test_default_embeddings_is_local_multilingual_e5_and_prefixes(
    ingatan_home: Path, tmp_path: Path
) -> None:
    proj = tmp_path / "default-proj"
    proj.mkdir()
    (proj / ".ingatan.toml").write_text('business = "tokoku"\nagent = "hermes"\n', encoding="utf-8")

    # With no [embeddings] section or env override, defaults to local multilingual-e5-small.
    scope = resolve_scope(start=proj, env={"INGATAN_HOME": str(ingatan_home)})
    assert scope.embeddings.provider == "local"
    assert embed_store.effective_model(scope.embeddings) == "intfloat/multilingual-e5-small"

    # Model-family prefix formatting for both E5 and EmbeddingGemma
    assert (
        embed_store._apply_model_prefix("intfloat/multilingual-e5-small", "biaya", is_query=True)
        == "query: biaya"
    )
    assert (
        embed_store._apply_model_prefix("intfloat/multilingual-e5-small", "tarif", is_query=False)
        == "passage: tarif"
    )
    assert (
        embed_store._apply_model_prefix("embeddinggemma", "biaya", is_query=True)
        == "task: search result | query: biaya"
    )
    assert (
        embed_store._apply_model_prefix("embeddinggemma", "tarif", is_query=False)
        == "title: none | text: tarif"
    )


def test_embedding_config_from_toml_env_and_tokens(
    ingatan_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ingatan_home / "config.toml").parent.mkdir(parents=True, exist_ok=True)
    (ingatan_home / "config.toml").write_text(
        'default_agent = "hermes"\n\n'
        "[embeddings]\n"
        'provider = "ollama"\n'
        'model = "bge-m3"\n'
        'api_base = "http://127.0.0.1:11434"\n',
        encoding="utf-8",
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".ingatan.toml").write_text('business = "tokoku"\n', encoding="utf-8")

    scope = resolve_scope(start=proj, env={"INGATAN_HOME": str(ingatan_home)})
    assert scope.embeddings.provider == "ollama"
    assert scope.embeddings.model == "bge-m3"
    assert scope.embeddings.api_base == "http://127.0.0.1:11434"

    # Env override wins over file
    env_scope = resolve_scope(
        start=proj,
        env={
            "INGATAN_HOME": str(ingatan_home),
            "INGATAN_EMBED_PROVIDER": "local",
            "INGATAN_EMBED_MODEL": "custom-e5",
        },
    )
    assert env_scope.embeddings.provider == "local"
    assert env_scope.embeddings.model == "custom-e5"

    # Invalid provider raises ConfigError
    with pytest.raises(ConfigError):
        resolve_scope(
            start=proj,
            env={"INGATAN_HOME": str(ingatan_home), "INGATAN_EMBED_PROVIDER": "unknown-provider"},
        )

    # TokenStore.lookup inherits [embeddings] from ~/.ingatan/config.toml
    monkeypatch.delenv("INGATAN_EMBED_PROVIDER", raising=False)
    raw, _ = tokens.add(ingatan_home, "tokoku", "hermes")
    looked = tokens.TokenStore(ingatan_home).lookup(raw)
    assert looked is not None
    assert looked.embeddings.provider == "ollama"
    assert looked.embeddings.model == "bge-m3"


# ---- 2. Pluggable semantic hybrid search ---------------------------------------------


def test_semantic_hybrid_recall_finds_zero_lexical_overlap(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deterministic 4-axis semantic space:
    # axis 0 = pricing/subscription, axis 1 = product catalog, axis 2 = shipping, axis 3 = unrelated
    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        low = text.lower()
        if any(w in low for w in ("tarif", "langganan", "biaya", "bulanan", "harga")):
            return _unit(1.0, 0.05, 0.0, 0.0)
        if any(w in low for w in ("katalog", "layanan", "product", "knowledge")):
            return _unit(0.05, 1.0, 0.0, 0.0)
        if any(w in low for w in ("pengiriman", "kurir", "ekspedisi")):
            return _unit(0.0, 0.0, 1.0, 0.0)
        return _unit(0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)

    ctx = _make_embed_ctx(ingatan_home)
    try:
        pricing = facts.remember(
            ctx,
            "Tarif langganan paket pro Rp 49.000 per bulan.",
            keywords=["tarif", "langganan"],
            tier="business",
            kind="stable",
        )
        catalog = facts.remember(
            ctx,
            "Katalog layanan tersedia di dokumen panduan utama.",
            keywords=["katalog", "layanan"],
            tier="business",
            kind="stable",
        )
        shipping = facts.remember(
            ctx,
            "Ekspedisi pengiriman memakai kurir reguler.",
            keywords=["ekspedisi", "kurir"],
            tier="business",
            kind="stable",
        )
        assert pricing.fact_id and catalog.fact_id and shipping.fact_id

        # Zero lexical overlap queries find the intended fact at rank #1
        res_price = search.recall(ctx, "biaya bulanan")
        assert res_price.facts
        assert res_price.facts[0].id == pricing.fact_id

        res_cat = search.recall(ctx, "product knowledge")
        assert res_cat.facts
        assert res_cat.facts[0].id == catalog.fact_id

        # Completely orthogonal query gets no semantic false positives (< 0.35 threshold)
        res_none = search.recall(ctx, "cuaca hujan badai")
        assert res_none.facts == []
    finally:
        ctx.close()


def test_semantic_graceful_fallback_to_fts5_on_error(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        raise RuntimeError("offline / model missing")

    monkeypatch.setattr(embed_store, "_embed_raw", boom)
    ctx = _make_embed_ctx(ingatan_home)
    try:
        saved = facts.remember(
            ctx,
            "Jadwal pengiriman setiap hari Selasa dan Kamis.",
            keywords=["pengiriman", "jadwal"],
            tier="business",
        )
        assert saved.fact_id
        # Recall still works purely on FTS5 even though embedder raises
        res = search.recall(ctx, "jadwal pengiriman")
        assert [h.id for h in res.facts] == [saved.fact_id]
    finally:
        ctx.close()


def test_reindex_and_forget_scrubs_embedding_bytes(
    ingatan_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Start with embeddings disabled, insert a fact and a note
    plain_ctx = open_context(
        Scope(
            business="tokoku",
            agent="claude-code",
            home=ingatan_home,
            embeddings=EmbeddingConfig(provider="none"),
        )
    )
    secret_marker = "RahasiaStrategiUnik998877"
    try:
        saved = facts.remember(
            plain_ctx,
            f"Strategi {secret_marker} hanya untuk internal.",
            keywords=["strategi"],
            tier="business",
            kind="stable",
        )
        assert saved.fact_id
        doc = tmp_path / "panduan.md"
        doc.write_text(
            "# Panduan Harga\n\nTarif langganan paket pro Rp 49.000.\n", encoding="utf-8"
        )
        ingest_path(plain_ctx, doc)
    finally:
        plain_ctx.close()

    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        # Distinct signature vector for the secret fact vs the ingested note
        if secret_marker in text:
            return _unit(0.73125, 0.41875, 0.53875)
        return _unit(0.1, 0.9, 0.2)

    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)

    ctx = _make_embed_ctx(ingatan_home)
    try:
        rep1 = search.reindex(ctx)
        assert rep1.facts_indexed == 1
        assert rep1.notes_indexed == 1
        assert rep1.errors == 0

        # Second reindex without force skips already indexed rows
        rep2 = search.reindex(ctx)
        assert rep2.facts_indexed == 0
        assert rep2.facts_skipped == 1
        assert rep2.notes_indexed == 0
        assert rep2.notes_skipped == 1

        # Force reindex recomputes all
        rep3 = search.reindex(ctx, force=True)
        assert rep3.facts_indexed == 1
        assert rep3.notes_indexed == 1

        # Record packed embedding bytes before forget
        emb_row = ctx.business.execute(
            "SELECT vec FROM fact_embeddings WHERE fact_id = ?", (saved.fact_id,)
        ).fetchone()
        assert emb_row is not None
        vec_bytes = bytes(emb_row["vec"])
        assert len(vec_bytes) == 12  # 3 * float32

        facts.forget(ctx, saved.fact_id, by_user=True)
        assert (
            ctx.business.execute(
                "SELECT count(*) FROM fact_embeddings WHERE fact_id = ?", (saved.fact_id,)
            ).fetchone()[0]
            == 0
        )
    finally:
        ctx.close()

    # Verify both fact text AND raw embedding bytes are gone from DB + WAL files
    db_file = ingatan_home / "tokoku.db"
    raw = db_file.read_bytes()
    wal = ingatan_home / "tokoku.db-wal"
    if wal.exists():
        raw += wal.read_bytes()
    assert secret_marker.encode("utf-8") not in raw
    assert vec_bytes not in raw


def test_http_embedding_providers_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeResp:
        def __init__(self, body: dict[str, Any]) -> None:
            self._buf = io.BytesIO(json.dumps(body).encode("utf-8"))

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float = 0.0) -> FakeResp:
        url = req.full_url
        payload = json.loads(req.data.decode("utf-8"))
        calls.append((url, payload))
        if "/api/embed" in url:
            return FakeResp({"embeddings": [[3.0, 4.0]]})
        if "/embeddings" in url:
            return FakeResp({"data": [{"embedding": [0.0, 5.0]}]})
        if ":embedContent" in url:
            return FakeResp({"embedding": {"values": [1.0, 0.0]}})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-fake")

    v_ollama = embed_store.embed_text(EmbeddingConfig(provider="ollama"), "tes", is_query=True)
    assert v_ollama is not None and abs(v_ollama[0] - 0.6) < 1e-5

    v_openai = embed_store.embed_text(EmbeddingConfig(provider="openai"), "tes", is_query=False)
    assert v_openai is not None and abs(v_openai[1] - 1.0) < 1e-5

    v_gemini = embed_store.embed_text(EmbeddingConfig(provider="gemini"), "tes", is_query=True)
    assert v_gemini is not None and abs(v_gemini[0] - 1.0) < 1e-5
    assert len(calls) == 3


def test_semantic_conflict_detection(ingatan_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        low = text.lower()
        if "jam" in low or "buka" in low or "operasional" in low:
            # High similarity (~0.90) — above 0.88 conflict threshold, below 0.94 dedup threshold
            if "08:00" in low:
                return _unit(1.0, 0.0, 0.0)
            return _unit(0.90, 0.43589, 0.0)
        return _unit(0.0, 0.0, 1.0)

    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)
    ctx = _make_embed_ctx(ingatan_home)
    try:
        f1 = facts.remember(
            ctx,
            "Jam buka toko dimulai pukul 08:00 pagi.",
            keywords=["jam", "buka"],
            tier="business",
        )
        f2 = facts.remember(
            ctx,
            "Waktu operasional gerai dimulai pukul 10:00 siang.",
            keywords=["operasional", "gerai"],
            tier="business",
        )
        assert f1.fact_id and f2.fact_id and f1.fact_id != f2.fact_id
        assert [s["id"] for s in f2.similar] == [f1.fact_id]
    finally:
        ctx.close()


# ---- 3. Near-duplicate deduplication -------------------------------------------------


def test_near_duplicate_exact_and_semantic_dedup(
    ingatan_home: Path, frozen_clock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        low = text.lower()
        if "margin" in low or "keuntungan" in low:
            # Cosine ~0.985 >= 0.94 semantic dedup threshold
            if "margin" in low:
                return _unit(1.0, 0.0, 0.0)
            return _unit(0.985, 0.1725, 0.0)
        return _unit(0.0, 1.0, 0.0)

    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)
    ctx = _make_embed_ctx(ingatan_home)
    try:
        first = facts.remember(
            ctx,
            "Target margin bersih produk minimal 25%.",
            keywords=["margin"],
            entities=["produk"],
            tier="business",
            kind="volatile",
            review_after="2026-12-01",
        )
        assert first.fact_id
        frozen_clock.advance(days=5)

        # Exact normalized duplicate (punctuation/case/whitespace differences) refreshes existing
        dup_exact = facts.remember(
            ctx,
            "  target   margin bersih produk minimal 25%  ",
            keywords=["profit"],
            entities=["produk"],
            tier="business",
        )
        assert dup_exact.fact_id == first.fact_id
        assert "refreshed" in dup_exact.message.lower()

        # Keywords were merged
        row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (first.fact_id,)).fetchone()
        assert "profit" in json.loads(row["keywords"])

        # Semantic near-duplicate (>= 0.94 cosine, >= 4 words) also refreshes instead of duplicating
        dup_sem = facts.remember(
            ctx,
            "Batas keuntungan bersih minimum barang adalah 25 persen.",
            keywords=["keuntungan"],
            tier="business",
        )
        assert dup_sem.fact_id == first.fact_id
        assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 1

        # Passing dedup=False forces a separate row
        forced = facts.remember(
            ctx,
            "Target margin bersih produk minimal 25%.",
            keywords=["margin"],
            entities=["produk"],
            tier="business",
            dedup=False,
        )
        assert forced.fact_id and forced.fact_id != first.fact_id
        assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 2
    finally:
        ctx.close()


# ---- 4. Time-travel queries (as_of) --------------------------------------------------


def test_recall_as_of_time_travel(ctx: Context, frozen_clock: Any) -> None:
    # Nov 11: initial price Rp 20.000
    old = facts.remember(
        ctx,
        "Harga kopi susu adalah Rp 20.000 per gelas.",
        keywords=["harga", "kopi"],
        entities=["kopi susu"],
        tier="business",
        kind="stable",
    )
    assert old.fact_id
    # Advance to Nov 20 and supersede with Rp 25.000
    frozen_clock.advance(days=9)
    new = facts.remember(
        ctx,
        "Harga kopi susu naik menjadi Rp 25.000 per gelas.",
        keywords=["harga", "kopi"],
        entities=["kopi susu"],
        tier="business",
        kind="stable",
    )
    assert new.fact_id
    facts.supersede(ctx, old.fact_id, new.fact_id)

    # Current recall returns only the new price
    now_hits = search.recall(ctx, "harga kopi susu")
    assert [h.id for h in now_hits.facts] == [new.fact_id]

    # Time-travel back to Nov 15 returns ONLY the old price, and does NOT bump use_count
    before_count = ctx.business.execute(
        "SELECT use_count FROM facts WHERE id = ?", (old.fact_id,)
    ).fetchone()[0]
    past_hits = search.recall(ctx, "harga kopi susu", as_of="2026-11-15")
    assert [h.id for h in past_hits.facts] == [old.fact_id]
    after_count = ctx.business.execute(
        "SELECT use_count FROM facts WHERE id = ?", (old.fact_id,)
    ).fetchone()[0]
    assert after_count == before_count

    # Time-travel before the fact existed returns nothing
    too_early = search.recall(ctx, "harga kopi susu", as_of="2026-11-01")
    assert too_early.facts == []


# ---- 5. Cognitive memory categories & pinning ----------------------------------------


def test_categories_and_pinned_prioritize_in_brief_and_filter_in_recall(
    ctx: Context, frozen_clock: Any
) -> None:
    pinned_rule = facts.remember(
        ctx,
        "Jangan pernah push langsung ke branch main tanpa review.",
        keywords=["git", "main", "aturan"],
        tier="business",
        kind="stable",
        category="rule",
        pinned=True,
    )
    proc = facts.remember(
        ctx,
        "Prosedur deploy: jalankan pytest lalu build container.",
        keywords=["deploy", "prosedur"],
        tier="business",
        kind="stable",
        category="procedure",
    )
    frozen_clock.advance(days=2)
    log_fact = facts.remember(
        ctx,
        "Log harian: server berjalan normal hari ini.",
        keywords=["log", "server"],
        tier="business",
        kind="volatile",
        category="fact",
    )
    assert pinned_rule.fact_id and proc.fact_id and log_fact.fact_id

    text = brief.build(ctx, offer_episodes=False)
    pos_pin = text.index(pinned_rule.fact_id)
    pos_proc = text.index(proc.fact_id)
    pos_log = text.index(log_fact.fact_id)
    assert pos_pin < pos_proc < pos_log
    assert "(rule; pinned)" in text
    assert "(procedure)" in text

    # Category filter in recall
    only_rules = search.recall(ctx, "aturan deploy server", category="rule")
    assert [h.id for h in only_rules.facts] == [pinned_rule.fact_id]
    assert only_rules.facts[0].category == "rule"
    assert only_rules.facts[0].pinned is True

    with pytest.raises(ValidationError):
        search.recall(ctx, "aturan", category="nonexistent")

    # set_pinned toggles pin
    facts.set_pinned(ctx, pinned_rule.fact_id, False)
    row = ctx.business.execute(
        "SELECT pinned FROM facts WHERE id = ?", (pinned_rule.fact_id,)
    ).fetchone()
    assert row["pinned"] == 0


# ---- 6. Project-scoped alias glossary ------------------------------------------------


def test_aliases_bidirectional_expansion_and_business_isolation(make_ctx: Any) -> None:
    ctx_a = make_ctx(business="toko-a")
    ctx_b = make_ctx(business="toko-b")

    assert search.add_alias(ctx_a, "LP", "landing page") == ("lp", "landing page")
    assert search.add_alias(ctx_a, "LP", "landing page") == ("lp", "landing page")  # idempotent
    assert search.list_aliases(ctx_a) == [("lp", "landing page")]
    assert search.list_aliases(ctx_b) == []  # strictly isolated per business

    f1 = facts.remember(
        ctx_a,
        "Konversi landing page minggu ini naik 12%.",
        keywords=["konversi", "halaman"],
        tier="business",
    )
    f2 = facts.remember(
        ctx_a,
        "Budget iklan LP bulan ini Rp 3 juta.",
        keywords=["budget", "iklan"],
        tier="business",
    )
    assert f1.fact_id and f2.fact_id

    # Querying "LP" finds "landing page" fact too
    hits_lp = search.recall(ctx_a, "konversi LP")
    assert f1.fact_id in [h.id for h in hits_lp.facts]

    # Querying "landing page" finds "LP" fact too
    hits_full = search.recall(ctx_a, "budget landing page")
    assert f2.fact_id in [h.id for h in hits_full.facts]

    assert search.remove_alias(ctx_a, "LP") == 1
    assert search.list_aliases(ctx_a) == []


# ---- 7. Lightweight relation edges ---------------------------------------------------


def test_relation_edges_suppress_conflicts_and_expand_1_hop(ctx: Context) -> None:
    f1 = facts.remember(
        ctx,
        "Anggaran pemasaran kuartal ini untuk kampanye media sosial Rp 10 juta.",
        keywords=["anggaran", "pemasaran", "kampanye"],
        entities=["Pemasaran"],
        tier="business",
    )
    assert f1.fact_id
    # Linked at remember time -> does NOT trigger conflict
    f2 = facts.remember(
        ctx,
        "Anggaran pemasaran kuartal ini untuk konten video Rp 5 juta.",
        keywords=["anggaran", "pemasaran", "video"],
        entities=["Pemasaran"],
        tier="business",
        related_to=[f1.fact_id],
    )
    assert f2.fact_id
    assert f2.similar == []
    assert ctx.business.execute("SELECT count(*) FROM conflicts").fetchone()[0] == 0

    # Recall on video hits f2 and surfaces 1-hop related fact f1
    res = search.recall(ctx, "konten video", limit=1, include_related=True)
    assert [h.id for h in res.facts] == [f2.fact_id]
    assert [h.id for h in res.related] == [f1.fact_id]
    rendered = _recall_text("konten video", res, False)
    assert "Related facts" in rendered
    assert f1.fact_id in rendered

    # include_related=False omits 1-hop related facts
    res_no_rel = search.recall(ctx, "konten video", limit=1, include_related=False)
    assert res_no_rel.related == []

    # Linking two conflicting facts resolves their open conflict
    c1 = facts.remember(
        ctx,
        "Jadwal pengiriman kurir ekspres setiap hari Senin pagi.",
        keywords=["jadwal", "pengiriman", "kurir"],
        entities=["Kurir"],
        tier="business",
    )
    c2 = facts.remember(
        ctx,
        "Jadwal pengiriman kurir ekspres setiap hari Kamis sore.",
        keywords=["jadwal", "pengiriman", "kurir"],
        entities=["Kurir"],
        tier="business",
    )
    assert c1.fact_id and c2.fact_id
    assert [s["id"] for s in c2.similar] == [c1.fact_id]
    assert "possible conflict" in search.status_of(
        ctx, ctx.business.execute("SELECT * FROM facts WHERE id = ?", (c2.fact_id,)).fetchone()
    )
    assert facts.link_facts(ctx, c1.fact_id, c2.fact_id, relation="part_of") is True
    assert "possible conflict" not in search.status_of(
        ctx, ctx.business.execute("SELECT * FROM facts WHERE id = ?", (c2.fact_id,)).fetchone()
    )

    # Forget cleans up edges
    facts.forget(ctx, f2.fact_id, by_user=True)
    assert facts.related_ids(ctx, f1.fact_id) == []

    # Validation checks
    with pytest.raises(ValidationError):
        facts.link_facts(ctx, f1.fact_id, f1.fact_id)
    with pytest.raises(NotFoundError):
        facts.link_facts(ctx, f1.fact_id, "f_missing00000")


# ---- 8. CLI commands & doctor & MCP invariants ---------------------------------------


def test_cli_alias_link_pin_reindex_import_and_doctor(ingatan_home: Path, tmp_path: Path) -> None:
    res_init = runner.invoke(app, ["init", "--business", "tokoku", "--agent", "claude-code"])
    assert res_init.exit_code == 0

    # Doctor reports embeddings provider
    checks = doctor_checks.run_checks(business="tokoku", agent="claude-code")
    rendered = doctor_checks.render(checks)
    assert "embeddings: none (FTS5 only)" in rendered

    # Reindex when embeddings disabled prints helpful message and exits 0
    res_reindex = runner.invoke(app, ["reindex", "--business", "tokoku"])
    assert res_reindex.exit_code == 0
    assert "disabled" in res_reindex.stdout.lower()

    # Alias CLI
    assert (
        runner.invoke(app, ["alias", "add", "ROAS", "return on ad spend", "-b", "tokoku"]).exit_code
        == 0
    )
    res_list = runner.invoke(app, ["alias", "list", "-b", "tokoku"])
    assert res_list.exit_code == 0 and "roas" in res_list.stdout
    assert runner.invoke(app, ["alias", "remove", "ROAS", "-b", "tokoku"]).exit_code == 0

    # Import with category & pinned
    jsonl = tmp_path / "seed.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "text": "Aturan utama: selalu tulis unit test sebelum merge.",
                "category": "rule",
                "pinned": True,
                "kind": "stable",
            }
        )
        + "\n"
        + json.dumps(
            {
                "text": "Checklist rilis: jalankan pytest dan evals.",
                "category": "procedure",
                "kind": "stable",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res_imp = runner.invoke(app, ["import", str(jsonl), "-b", "tokoku"])
    assert res_imp.exit_code == 0

    ctx = open_context(Scope(business="tokoku", agent="claude-code", home=ingatan_home))
    try:
        rows = ctx.business.execute(
            "SELECT id, category, pinned FROM facts ORDER BY seq ASC"
        ).fetchall()
        assert len(rows) == 2
        f_rule, f_proc = str(rows[0]["id"]), str(rows[1]["id"])
        assert rows[0]["category"] == "rule" and rows[0]["pinned"] == 1
        assert rows[1]["category"] == "procedure" and rows[1]["pinned"] == 0
    finally:
        ctx.close()

    # Pin / unpin / link / unlink CLI
    assert runner.invoke(app, ["pin", f_proc, "-b", "tokoku"]).exit_code == 0
    assert runner.invoke(app, ["unpin", f_proc, "-b", "tokoku"]).exit_code == 0
    assert (
        runner.invoke(
            app, ["link", f_rule, f_proc, "--relation", "part_of", "-b", "tokoku"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["unlink", f_rule, f_proc, "-b", "tokoku"]).exit_code == 0


def test_mcp_server_still_exactly_seven_tools_with_phase2_params(ctx: Context) -> None:
    server = build_server(ctx)
    tools = {t.name: t for t in server._tool_manager.list_tools()}  # type: ignore[attr-defined]
    assert set(tools) == {
        "brief",
        "recall",
        "remember",
        "supersede",
        "confirm",
        "forget",
        "history",
    }
    for name, tool in tools.items():
        props = tool.parameters.get("properties", {})
        assert "business" not in props, f"{name} must not expose business in single-business mode"
        assert "agent" not in props, f"{name} must not expose agent in single-business mode"
    recall_props = tools["recall"].parameters["properties"]
    assert {"as_of", "category", "include_related"} <= set(recall_props)
    remember_props = tools["remember"].parameters["properties"]
    assert {"category", "pinned", "related_to", "dedup"} <= set(remember_props)


# ---- 9. Memory optimizations & HTTP lifecycle ----------------------------------------


def test_embed_threads_idle_unload_and_query_cache(
    ingatan_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = tmp_path / "opt-proj"
    proj.mkdir()
    (proj / ".ingatan.toml").write_text(
        'business = "tokoku"\nagent = "hermes"\n\n'
        "[embeddings]\n"
        'provider = "local"\n'
        "threads = 1\n"
        "idle_unload_seconds = 60\n",
        encoding="utf-8",
    )
    scope = resolve_scope(start=proj, env={"INGATAN_HOME": str(ingatan_home)})
    assert scope.embeddings.threads == 1
    assert scope.embeddings.idle_unload_seconds == 60

    env_scope = resolve_scope(
        start=proj,
        env={
            "INGATAN_HOME": str(ingatan_home),
            "INGATAN_EMBED_THREADS": "4",
            "INGATAN_EMBED_IDLE_UNLOAD": "0",
        },
    )
    assert env_scope.embeddings.threads == 4
    assert env_scope.embeddings.idle_unload_seconds == 0

    with pytest.raises(ConfigError):
        resolve_scope(
            start=proj,
            env={"INGATAN_HOME": str(ingatan_home), "INGATAN_EMBED_THREADS": "0"},
        )

    # Verify idle auto-unload and manual unload_local_models()
    embed_store.unload_local_models()
    key = ("fake-model", 2)
    with embed_store._local_lock:
        embed_store._local_models[key] = object()
        embed_store._local_last_used[key] = 0.0
    embed_store._unload_if_idle(key, 1.0)
    assert key not in embed_store._local_models

    with embed_store._local_lock:
        embed_store._local_models[key] = object()
        embed_store._local_last_used[key] = 999999999.0
    assert embed_store.unload_local_models() == 1
    assert embed_store._local_models == {}

    # Verify bounded LRU query cache avoids redundant inference
    calls = 0

    def counting_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        nonlocal calls
        calls += 1
        return _unit(1.0, 0.0, 0.0)

    monkeypatch.setattr(embed_store, "_embed_raw", counting_embed)
    cfg = EmbeddingConfig(provider="local", model="cache-test")
    v1 = embed_store.embed_text(cfg, "ulang query", is_query=True)
    v2 = embed_store.embed_text(cfg, "ulang query", is_query=True)
    assert v1 == v2 and calls == 1


def test_vectorized_score_blobs_and_reindex_prunes_orphans(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = _unit(1.0, 0.0, 0.0)
    b_high = embed_store.pack_vector(_unit(0.95, 0.31, 0.0))
    b_mid = embed_store.pack_vector(_unit(0.6, 0.8, 0.0))
    b_low = embed_store.pack_vector(_unit(0.0, 1.0, 0.0))
    ranked = embed_store.score_blobs(
        q,
        [("low", b_low), ("mid", b_mid), ("high", b_high)],
        dims=3,
        min_similarity=0.5,
        top_k=2,
    )
    assert [k for k, _ in ranked] == ["high", "mid"]

    # Insert an orphan row in fact_embeddings and confirm reindex prunes it
    monkeypatch.setattr(
        embed_store, "_embed_raw", lambda cfg, text, *, is_query: _unit(1.0, 0.0, 0.0)
    )
    ctx = _make_embed_ctx(ingatan_home)
    try:
        with db.transaction(ctx.business):
            ctx.business.execute(
                "INSERT INTO fact_embeddings (fact_id, model, dims, vec, updated_at) "
                "VALUES ('f_orphan000001', 'test-model', 3, ?, '2026-11-11T00:00:00Z')",
                (b_high,),
            )
        search.reindex(ctx)
        assert (
            ctx.business.execute(
                "SELECT count(*) FROM fact_embeddings WHERE fact_id = 'f_orphan000001'"
            ).fetchone()[0]
            == 0
        )
    finally:
        ctx.close()


def test_http_app_reloads_scope_on_config_change_and_evicts_revoked(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from ingatan.mcp.http import build_app

    monkeypatch.delenv("INGATAN_EMBED_PROVIDER", raising=False)
    ingatan_home.mkdir(parents=True, exist_ok=True)
    cfg_path = ingatan_home / "config.toml"
    cfg_path.write_text("[brief]\ntoken_budget = 800\n", encoding="utf-8")
    raw, rec = tokens.add(ingatan_home, "tokoku", "claude-code")

    async def exercise() -> None:
        http_app = build_app(ingatan_home)
        async with http_app.lifespan():
            s1 = http_app.tokens.lookup(raw)
            assert s1 is not None and s1.token_budget == 800
            srv1 = await http_app._server_for(s1)

            # Update config.toml and bump mtime/size; next lookup reloads and swaps _ScopeServer.
            cfg_path.write_text("[brief]\ntoken_budget = 1400\n", encoding="utf-8")
            s2 = http_app.tokens.lookup(raw)
            assert s2 is not None and s2.token_budget == 1400
            srv2 = await http_app._server_for(s2)
            assert srv2 is not srv1
            assert srv2.context.scope.token_budget == 1400

            # Revoking the token evicts the cached _ScopeServer cleanly.
            tokens.revoke(ingatan_home, rec.id)
            assert http_app.tokens.lookup(raw) is None
            await http_app._evict_revoked_scopes()
            assert http_app._servers == {}

    asyncio.run(exercise())


def test_custom_e5_small_registration_doctor_probe_and_reindex_exit_code(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    # 1. Verify _ensure_custom_local_model registers intfloat/multilingual-e5-small idempotently
    registered: list[dict[str, Any]] = []

    class FakeTextEmbedding:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, Any]]:
            return [{"model": m["model"]} for m in registered]

        @classmethod
        def add_custom_model(cls, **kwargs: Any) -> None:
            registered.append(kwargs)

    fake_md = types.ModuleType("fastembed.common.model_description")
    fake_md.PoolingType = types.SimpleNamespace(MEAN="MEAN")  # type: ignore[attr-defined]
    fake_md.ModelSource = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", types.ModuleType("fastembed"))
    monkeypatch.setitem(sys.modules, "fastembed.common", types.ModuleType("fastembed.common"))
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_md)

    embed_store._ensure_custom_local_model(FakeTextEmbedding, "intfloat/multilingual-e5-small")
    embed_store._ensure_custom_local_model(FakeTextEmbedding, "intfloat/multilingual-e5-small")
    assert len(registered) == 1
    assert registered[0]["model"] == "intfloat/multilingual-e5-small"
    assert registered[0]["dim"] == 384
    assert registered[0]["model_file"] == "onnx/model.onnx"
    assert registered[0]["normalization"] is True
    assert registered[0]["sources"] == {"hf": "intfloat/multilingual-e5-small"}

    # 2. Initialize a business with 1 fact, then test doctor probe & reindex error handling
    assert (
        runner.invoke(app, ["init", "--business", "tokoku", "--agent", "claude-code"]).exit_code
        == 0
    )
    ctx = open_context(
        Scope(
            business="tokoku",
            agent="claude-code",
            home=ingatan_home,
            embeddings=EmbeddingConfig(provider="none"),
        )
    )
    try:
        facts.remember(ctx, "Tarif paket pro Rp 49.000.", keywords=["tarif"], tier="business")
    finally:
        ctx.close()

    # Enable local embeddings and force _embed_raw to fail
    monkeypatch.setenv("INGATAN_EMBED_PROVIDER", "local")

    def failing_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        raise ValueError("Model unsupported in runtime")

    monkeypatch.setattr(embed_store, "_embed_raw", failing_embed)

    # Doctor must FAIL with exit code 1 and surface the real error
    res_doc_fail = runner.invoke(app, ["doctor", "--business", "tokoku"])
    assert res_doc_fail.exit_code == 1
    assert "[FAIL] embeddings:" in res_doc_fail.stdout
    assert "Model unsupported in runtime" in res_doc_fail.stdout

    # Reindex must exit 1 and report the embedding error clearly (not silent exit 0!)
    res_re_fail = runner.invoke(app, ["reindex", "--business", "tokoku"])
    assert res_re_fail.exit_code == 1
    assert "errors=1" in res_re_fail.stdout
    assert "Model unsupported in runtime" in res_re_fail.output

    # Once embedder succeeds, both doctor probe and reindex exit 0
    monkeypatch.setattr(
        embed_store, "_embed_raw", lambda cfg, text, *, is_query: _unit(1.0, 0.0, 0.0)
    )
    res_doc_ok = runner.invoke(app, ["doctor", "--business", "tokoku"])
    assert res_doc_ok.exit_code == 0
    assert "[pass] embeddings: local (intfloat/multilingual-e5-small, 3 dims; probe OK)" in (
        res_doc_ok.stdout
    )
    res_re_ok = runner.invoke(app, ["reindex", "--business", "tokoku"])
    assert res_re_ok.exit_code == 0
    assert "facts indexed=1" in res_re_ok.stdout
