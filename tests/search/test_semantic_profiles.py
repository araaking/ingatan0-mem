"""Model-specific similarity cut-offs, and what `remember` and `recall` do with them.

The vectors are hand-made, so each test states the cosine it means. Model names decide the
profile: a name from the E5 family gets the cut-offs measured for it, anything else the
generic ones.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from ingatan import facts, proposals, search
from ingatan.errors import ConfigError
from ingatan.ingest import ingest_path
from ingatan.scope import (
    Context,
    EmbeddingConfig,
    Scope,
    load_embeddings,
    open_context,
    resolve_scope,
)
from ingatan.search import embed

E5 = "fake/multilingual-e5-small"
Embedder = Callable[..., list[float]]


def _at(cosine: float) -> list[float]:
    """A unit vector whose cosine with (1, 0) is `cosine`."""
    return [cosine, math.sqrt(1.0 - cosine * cosine)]


def _ctx(
    home: Path, model: str = E5, *, require_approval: bool = False, **overrides: float
) -> Context:
    cfg = EmbeddingConfig(provider="local", model=model, **overrides)  # type: ignore[arg-type]
    return open_context(
        Scope(
            business="tokoku",
            agent="asisten",
            home=home,
            require_approval=require_approval,
            embeddings=cfg,
        )
    )


def _embedder(table: dict[str, float], default: float = 0.0) -> Embedder:
    """Texts containing a key of `table` get that cosine against the (1, 0) axis."""

    def fake(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        low = text.casefold()
        for needle, cosine in table.items():
            if needle in low:
                return _at(cosine)
        return _at(default)

    return fake


# ---- profiles ------------------------------------------------------------------------


def test_profile_is_chosen_by_model_family_and_can_be_overridden() -> None:
    generic = embed.similarity_profile(EmbeddingConfig(provider="openai"))
    assert generic == embed.GENERIC_PROFILE
    for name in ("xenova/multilingual-e5-small", "intfloat/e5-base-v2", "multilingual-e5-large"):
        cfg = EmbeddingConfig(provider="ollama", model=name)
        assert embed.similarity_profile(cfg) == embed.E5_PROFILE
    # The E5 band is narrow and high: its cut-offs sit far above the generic ones, and
    # similarity alone never flags a conflict (same-template facts score like real updates).
    assert embed.E5_PROFILE.recall_floor > 0.8 > embed.GENERIC_PROFILE.recall_floor
    assert embed.E5_PROFILE.conflict > embed.GENERIC_PROFILE.conflict
    assert embed.E5_PROFILE.conflict_strict is None
    assert embed.GENERIC_PROFILE.conflict_strict is not None

    tuned = embed.similarity_profile(
        EmbeddingConfig(
            provider="ollama",
            model="bge-m3",
            recall_floor=0.5,
            conflict_similarity=0.7,
            conflict_strict_similarity=0.8,
        )
    )
    assert tuned == embed.SimilarityProfile(0.5, 0.7, 0.8)
    partly = embed.similarity_profile(EmbeddingConfig(provider="local", recall_floor=0.8))
    assert partly == embed.SimilarityProfile(0.8, embed.E5_PROFILE.conflict, None)
    opted_in = embed.similarity_profile(
        EmbeddingConfig(provider="local", conflict_strict_similarity=0.95)
    )
    assert opted_in.conflict_strict == 0.95


def test_similarity_settings_come_from_toml_and_are_validated(
    ingatan_home: Path, tmp_path: Path
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".ingatan.toml").write_text(
        'business = "tokoku"\nagent = "asisten"\n\n[embeddings]\nprovider = "ollama"\n'
        'model = "bge-m3"\nrecall_floor = 0.55\nconflict_similarity = 0.8\n'
        'conflict_strict_similarity = 0.9\nmodel_dir = "~/somewhere"\n',
        encoding="utf-8",
    )
    env = {"INGATAN_HOME": str(ingatan_home)}
    cfg = resolve_scope(start=proj, env=env).embeddings
    assert (cfg.recall_floor, cfg.conflict_similarity, cfg.conflict_strict_similarity) == (
        0.55,
        0.8,
        0.9,
    )
    assert cfg.model_dir == str(Path("~/somewhere").expanduser())
    # The same settings are readable without a business or agent name (for `embeddings pull`).
    assert load_embeddings(start=proj, env=env) == cfg
    assert load_embeddings(start=tmp_path, env=env) == EmbeddingConfig()
    assert load_embeddings(
        start=tmp_path, env={**env, "INGATAN_MODEL_DIR": str(tmp_path)}
    ).model_dir == str(tmp_path)

    for bad in ("1.5", '"high"', "true"):
        (proj / ".ingatan.toml").write_text(
            f'business = "tokoku"\nagent = "asisten"\n[embeddings]\nrecall_floor = {bad}\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="recall_floor"):
            resolve_scope(start=proj, env=env)

    # A relative model folder would be a different folder for every working directory, so
    # the one `ingatan embeddings pull` filled would not be the one the server reads.
    (proj / ".ingatan.toml").write_text(
        'business = "tokoku"\nagent = "asisten"\n[embeddings]\nmodel_dir = "models"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="absolute path"):
        resolve_scope(start=proj, env=env)
    with pytest.raises(ConfigError, match="absolute path"):
        load_embeddings(start=tmp_path, env={**env, "INGATAN_MODEL_DIR": "relatif/models"})


# ---- recall --------------------------------------------------------------------------


def test_recall_floor_keeps_out_semantic_noise_for_a_narrow_band_model(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the E5 family really does: text about another subject still scores about 0.8.
    vectors = {
        "ongkir": [0.90, math.sqrt(1 - 0.90**2), 0.0],
        "wifi": [0.80, 0.60, 0.0],
        "berapa biaya kirim": [1.0, 0.0, 0.0],  # 0.90 to the ongkir fact, 0.80 to the other
        "resep rendang": [0.90, 0.0, math.sqrt(1 - 0.90**2)],  # 0.81 and 0.72
    }

    def fake(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        return next(vec for needle, vec in vectors.items() if needle in text.casefold())

    monkeypatch.setattr(embed, "_embed_raw", fake)
    ctx = _ctx(ingatan_home)
    try:
        ongkir = facts.remember(
            ctx, "Ongkir ke Bandung Rp 15.000 per kg.", keywords=["ongkir"], kind="stable"
        )
        facts.remember(ctx, "Sandi wifi diganti tiap tiga bulan.", keywords=["wifi"], kind="stable")

        hits = search.recall(ctx, "berapa biaya kirim", touch=False).facts
        assert [h.id for h in hits] == [ongkir.fact_id]  # 0.90 passes the floor, 0.80 does not
        # A question about nothing in the store comes back empty ...
        assert search.recall(ctx, "resep rendang padang", touch=False).facts == []
    finally:
        ctx.close()
    # ... where one cut-off for every model (the generic 0.35) would return every fact.
    loose = _ctx(ingatan_home, recall_floor=0.35)
    try:
        assert len(search.recall(loose, "resep rendang padang", touch=False).facts) == 2
    finally:
        loose.close()


# ---- remember ------------------------------------------------------------------------


def test_a_changed_value_is_saved_and_flagged_never_merged(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Measured on the real model: "Rp 15.000" against "Rp 18.000" scores about 0.98.
    monkeypatch.setattr(embed, "_embed_raw", _embedder({"15.000": 1.0, "18.000": 0.98}))
    ctx = _ctx(ingatan_home)
    try:
        old = facts.remember(
            ctx,
            "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler.",
            keywords=["ongkir"],
            entities=["Bandung"],
            kind="volatile",
        )
        new = facts.remember(
            ctx,
            "Ongkir ke Bandung Rp 18.000 per kg lewat kurir reguler.",
            keywords=["ongkir"],
            entities=["Bandung"],
            kind="volatile",
        )
        assert new.fact_id and new.fact_id != old.fact_id
        assert [s["id"] for s in new.similar] == [old.fact_id]
        stored = [r["text"] for r in ctx.business.execute("SELECT text FROM facts ORDER BY seq")]
        assert stored == [
            "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler.",
            "Ongkir ke Bandung Rp 18.000 per kg lewat kurir reguler.",
        ]
        statuses = {h.id: h.status for h in search.recall(ctx, "ongkir Bandung", touch=False).facts}
        assert statuses == {old.fact_id: "possible conflict", new.fact_id: "possible conflict"}
    finally:
        ctx.close()


def test_conflict_cut_offs_follow_the_profile(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No two texts share a content word, so only the entity and the score can link them.
    # Each later text has its own axis: it is close to the first fact and to nothing else.
    closeness = {"vendor": 0.91, "rekanan": 0.88, "mitra": 0.93, "kolega": 0.91}

    def fake(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        vec = [0.0] * (len(closeness) + 1)
        for axis, (needle, cosine) in enumerate(closeness.items(), start=1):
            if needle in text.casefold():
                vec[0], vec[axis] = cosine, math.sqrt(1 - cosine * cosine)
                return vec
        vec[0] = 1.0
        return vec

    monkeypatch.setattr(embed, "_embed_raw", fake)

    def run(ctx: Context) -> dict[str, bool]:
        base = facts.remember(
            ctx, "Pemasok utama berganti.", keywords=["pemasok"], entities=["toko-a"], kind="stable"
        )

        def flagged(text: str, entities: list[str]) -> bool:
            keyword = text.split()[0].casefold()
            saved = facts.remember(
                ctx, text, keywords=[keyword], entities=entities, kind="stable", dedup=False
            )
            return [s["id"] for s in saved.similar] == [base.fact_id]

        return {
            "0.91 same entity": flagged("Vendor tetap sama.", ["toko-a"]),
            "0.88 same entity": flagged("Rekanan belum ditentukan.", ["toko-a"]),
            "0.93 no entity": flagged("Mitra sudah dipilih.", []),
            "0.91 other entity": flagged("Kolega ikut rapat.", ["toko-b"]),
        }

    # E5 family: a shared entity and 0.90 are both needed. Without the shared entity even
    # 0.93 is not evidence, because two facts on one template score that high.
    ctx = _ctx(ingatan_home)
    try:
        assert run(ctx) == {
            "0.91 same entity": True,
            "0.88 same entity": False,
            "0.93 no entity": False,
            "0.91 other entity": False,
        }
    finally:
        ctx.close()
    # A project can opt in to an entity-free cut-off for its own model.
    opted = open_context(
        Scope(
            business="toko-kedua",
            agent="asisten",
            home=ingatan_home,
            embeddings=EmbeddingConfig(provider="local", model=E5, conflict_strict_similarity=0.92),
        )
    )
    try:
        assert run(opted) == {
            "0.91 same entity": True,
            "0.88 same entity": False,
            "0.93 no entity": True,
            "0.91 other entity": False,
        }
    finally:
        opted.close()


def test_the_fact_that_shares_the_entity_is_scored_among_many_closer_look_alikes(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sixty facts about other things score 0.99 against the new text, the one fact about
    # the same entity 0.91. A nearest-neighbour list of the file (cut at fifty) would hold
    # only look-alikes, and the fact that matters would count as "not similar".
    assert embed.MAX_SEMANTIC_CANDIDATES < 60
    monkeypatch.setattr(
        embed, "_embed_raw", _embedder({"pengalih": 0.99, "pemasok": 0.91, "vendor": 1.0})
    )
    ctx = _ctx(ingatan_home)
    try:
        base = facts.remember(
            ctx, "Pemasok utama berganti.", keywords=["pemasok"], entities=["toko-a"], kind="stable"
        )
        for n in range(60):
            facts.remember(
                ctx,
                f"Pengalih nomor {n} membahas hal lain.",
                keywords=["pengalih"],
                entities=[f"rak-{n}"],
                kind="stable",
            )
        new = facts.remember(
            ctx, "Vendor tetap sama.", keywords=["vendor"], entities=["toko-a"], kind="stable"
        )
        assert new.fact_id and new.fact_id != base.fact_id
        assert [s["id"] for s in new.similar] == [base.fact_id]
    finally:
        ctx.close()


@pytest.mark.parametrize(
    "name",
    [
        'toko "sumber" rejeki',
        "A&W (pusat)",
        "NEAR/AND OR NOT",
        "kopi*",
        "col:umn",
        "^awal",
        "{kurawal} [siku]",
        "toko-a + toko-b",
        "C++",
        "cafe\u0301 senja",  # a combining accent, not the precomposed letter
        "++",
        "\U0001f525",  # no letter or digit at all: nothing for the word index to hold
        "-",
    ],
)
def test_an_entity_name_with_any_characters_is_looked_up_safely(
    ingatan_home: Path, name: str
) -> None:
    ctx = open_context(Scope(business="tokoku", agent="asisten", home=ingatan_home))
    try:
        base = facts.remember(
            ctx,
            "Harga sewa gudang naik bulan depan.",
            keywords=["sewa"],
            entities=[name],
            kind="stable",
        )
        facts.remember(
            ctx,
            "Jadwal piket kebersihan berlaku mulai Senin.",
            keywords=["piket"],
            entities=["piket"],
            kind="stable",
        )
        new = facts.remember(
            ctx,
            "Harga sewa gudang turun bulan depan.",
            keywords=["sewa"],
            entities=[f"  {name.upper()} "],  # entity names compare without case or padding
            kind="stable",
        )
        assert new.fact_id and new.fact_id != base.fact_id
        assert [s["id"] for s in new.similar] == [base.fact_id]
    finally:
        ctx.close()


def test_remember_reads_fact_rows_only_through_an_index(ingatan_home: Path) -> None:
    ctx = open_context(Scope(business="tokoku", agent="asisten", home=ingatan_home))
    try:
        for n in range(40):
            facts.remember(
                ctx,
                f"Catatan gudang nomor {n} tentang rak {n % 7}.",
                keywords=["gudang"],
                entities=["gudang", f"rak {n % 7}"],
                kind="stable",
            )
        seen: list[str] = []
        ctx.business.set_trace_callback(seen.append)
        ctx.glob.set_trace_callback(seen.append)
        try:
            dup = facts.remember(
                ctx,
                "catatan gudang nomor 7 tentang rak 0",
                keywords=["gudang"],
                entities=["gudang", "rak 0"],
                kind="stable",
            )
            new = facts.remember(
                ctx,
                "Catatan gudang nomor 7 tentang rak 3 sudah dipindah.",
                keywords=["gudang"],
                entities=["gudang", "rak 0"],
                kind="stable",
            )
        finally:
            ctx.business.set_trace_callback(None)
            ctx.glob.set_trace_callback(None)
        assert "refreshed" in dup.message  # found through the text_key index
        assert new.similar  # and the facts that share its entity through the word index
        # Every statement that returns fact rows is driven by an index: an id, the text key,
        # or the FTS table (whose statements select from facts_fts, not from facts).
        # Counting (count(*), GROUP BY) is done inside SQLite and returns no rows of text.
        for sql in seen:
            flat = re.sub(r"\s+", " ", sql)
            if re.match(r"SELECT (f\.)?\* FROM facts\b", flat):
                assert re.search(r"\bid = |\bid IN \(|\btext_key = ", flat), flat
    finally:
        ctx.close()


# ---- no model call while the database is locked for writing ----------------------------


def test_embedding_runs_before_the_write_lock_in_ingest_and_approve(
    ingatan_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(ingatan_home, require_approval=True)
    in_transaction: list[bool] = []

    def watching(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        in_transaction.append(ctx.business.in_transaction or ctx.glob.in_transaction)
        return _at(0.5)

    monkeypatch.setattr(embed, "_embed_raw", watching)
    try:
        doc = tmp_path / "panduan.md"
        doc.write_text("# Satu\n\nIsi pertama.\n\n# Dua\n\nIsi kedua.\n", encoding="utf-8")
        report = ingest_path(ctx, doc)
        assert report.created == 2
        assert ctx.business.execute("SELECT count(*) FROM note_embeddings").fetchone()[0] == 2

        queued = facts.remember(
            ctx, "Jam buka toko pukul 08.00.", keywords=["jam"], tier="business"
        )
        assert queued.proposal_id
        fact_id = proposals.approve(ctx, queued.proposal_id)
        assert (
            ctx.business.execute(
                "SELECT count(*) FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
            ).fetchone()[0]
            == 1
        )
        assert in_transaction and not any(in_transaction)
        # One call per text: the vector made for the conflict check is the one stored.
        assert len(in_transaction) == 2 + 1 + 1  # two sections, the proposal, its approval
    finally:
        ctx.close()


def test_a_failed_embedding_is_not_retried_inside_one_call(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def failing(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(embed, "_embed_raw", failing)
    ctx = _ctx(ingatan_home)
    try:
        saved = facts.remember(
            ctx, "Stok gula tinggal 20 karung.", keywords=["stok"], entities=["gula"], kind="stable"
        )
        assert saved.fact_id
        assert calls == 1  # not once for the conflict check and again for storage
        assert ctx.business.execute("SELECT count(*) FROM fact_embeddings").fetchone()[0] == 0
    finally:
        ctx.close()


def test_query_vectors_are_cached_packed_and_cleared_on_unload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counting(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        nonlocal calls
        calls += 1
        return _at(0.6)

    monkeypatch.setattr(embed, "_embed_raw", counting)
    cfg = EmbeddingConfig(provider="local", model="cache-test")
    first = embed.embed_text(cfg, "ulang", is_query=True)
    again = embed.embed_text(cfg, "ulang", is_query=True)
    assert calls == 1 and first is not None and again is not None
    assert again == pytest.approx(first, abs=1e-6)
    assert all(isinstance(value, bytes) for value in embed._query_cache.values())
    embed.embed_text(cfg, "ulang", is_query=False)  # stored text is never served from the cache
    assert calls == 2
    embed.unload_local_models()
    assert not embed._query_cache
    embed.embed_text(cfg, "ulang", is_query=True)
    assert calls == 3
