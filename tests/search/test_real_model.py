"""The real local model, end to end. Skipped unless the model files are on this machine.

These are the checks the hand-made vectors elsewhere cannot make: that the tokenizer feeds
the model the ids it was trained on, and that the calibrated cut-offs do on real text what
they are documented to do. Run them after `ingatan embeddings pull`:

    INGATAN_TEST_MODEL_DIR=~/.ingatan/models python -m pytest tests/search/test_real_model.py

The folder is only read. Facts go to the temporary data folder every test gets.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ingatan import facts, search
from ingatan.scope import Context, EmbeddingConfig, Scope, open_context
from ingatan.search import embed, local

MODEL_DIR = (os.environ.get("INGATAN_TEST_MODEL_DIR") or "").strip()


def _unavailable() -> str:
    if not MODEL_DIR:
        return "set INGATAN_TEST_MODEL_DIR to a folder filled by `ingatan embeddings pull`"
    if local.missing_packages():
        return f"missing packages: {', '.join(local.missing_packages())}"
    if local.missing_files(local.resolve(""), Path(MODEL_DIR).expanduser()):
        return f"the default model is not in {MODEL_DIR}"
    return ""


pytestmark = pytest.mark.skipif(bool(_unavailable()), reason=_unavailable() or "available")

CFG = EmbeddingConfig(provider="local", model_dir=str(Path(MODEL_DIR).expanduser()))


@pytest.fixture
def real_ctx(ingatan_home: Path) -> Iterator[Context]:
    ctx = open_context(Scope(business="tokoku", agent="asisten", home=ingatan_home, embeddings=CFG))
    try:
        yield ctx
    finally:
        ctx.close()
        embed.unload_local_models()


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_token_ids_are_the_ones_the_model_was_trained_on() -> None:
    model = local.resolve("")
    loaded = local._get(model, local.model_folder(CFG.model_dir, model), 2)

    def ids(text: str) -> list[int]:
        return local._token_ids(loaded.tokenizer.encode(text, out_type=int), model.max_tokens)

    # Reference ids from the model's own tokenizer.json (Hugging Face `tokenizers`).
    assert ids("query: berapa ongkir ke Bandung") == [
        0, 41, 1294, 12, 63279, 36, 176478, 311, 63500, 2,
    ]  # fmt: skip
    assert ids("passage: Toko tutup setiap hari Minggu.") == [
        0, 46692, 12, 95427, 112641, 7028, 2651, 72352, 5, 2,
    ]  # fmt: skip
    assert ids("passage: sync_report.py runs every 6 hours") == [
        0, 46692, 12, 6002, 238, 454, 79770, 5, 6493, 127877, 11907, 305, 21974, 2,
    ]  # fmt: skip


def test_vectors_are_unit_length_stable_and_independent_of_their_batch() -> None:
    text = "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler."
    alone = embed.embed_text(CFG, text)
    assert alone is not None and len(alone) == 384
    assert math.sqrt(sum(x * x for x in alone)) == pytest.approx(1.0, abs=1e-4)
    assert [round(x, 3) for x in (embed.embed_text(CFG, text) or [])] == [
        round(x, 3) for x in alone
    ]
    # Same text, embedded in one call with much longer and shorter ones: the neighbours must
    # not change it. (In a shared forward pass of the quantized model they do: 0.997.)
    others = ["a", "Catatan panjang. " * 60, "Toko tutup hari Minggu.", "x y z", text, "akhir"]
    batch = embed.embed_batch(CFG, others)
    assert batch[4] is not None and _cos(alone, batch[4]) > 0.99999
    # Leading and trailing whitespace is not part of the text.
    padded = embed.embed_text(CFG, f"  {text}\n")
    assert padded is not None and _cos(alone, padded) > 0.9999
    # A question and a stored text are embedded with different prefixes.
    as_query = embed.embed_text(CFG, text, is_query=True)
    assert as_query is not None and _cos(alone, as_query) < 0.999


def test_a_changed_value_is_kept_and_flagged_with_the_real_model(real_ctx: Context) -> None:
    old = facts.remember(
        real_ctx,
        "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler.",
        keywords=["ongkir", "bandung"],
        entities=["Bandung"],
        kind="volatile",
    )
    new = facts.remember(
        real_ctx,
        "Ongkir ke Bandung Rp 18.000 per kg lewat kurir reguler.",
        keywords=["ongkir", "bandung"],
        entities=["Bandung"],
        kind="volatile",
    )
    assert new.fact_id and new.fact_id != old.fact_id
    assert [s["id"] for s in new.similar] == [old.fact_id]
    texts = [r["text"] for r in real_ctx.business.execute("SELECT text FROM facts ORDER BY seq")]
    assert [("15.000" in t, "18.000" in t) for t in texts] == [(True, False), (False, True)]

    # A replacement in entirely different words shares no content word with the fact it
    # replaces, so the word rules cannot see it. The shared entity and the score do (0.94).
    reworded = facts.remember(
        real_ctx,
        "Tarif ongkir Bandung sekarang delapan belas ribu rupiah tiap kilogram.",
        keywords=["biaya pengiriman"],
        entities=["Bandung"],
        kind="volatile",
        check_conflicts=True,
    )
    assert {old.fact_id, new.fact_id} <= {s["id"] for s in reworded.similar}


def test_look_alike_facts_about_different_things_are_not_flagged(real_ctx: Context) -> None:
    # Measured 0.98: higher than any real update. The agent labelled them as different
    # things, and a similarity score is never allowed to overrule that.
    gula = facts.remember(
        real_ctx,
        "Stok gula di gudang tinggal 20 karung.",
        keywords=["persediaan"],
        entities=["gula"],
        kind="volatile",
    )
    tepung = facts.remember(
        real_ctx,
        "Stok tepung di gudang tinggal 20 karung.",
        keywords=["persediaan"],
        entities=["tepung"],
        kind="volatile",
    )
    assert tepung.fact_id != gula.fact_id and tepung.similar == []
    # Nor without any label at all: similarity alone is not evidence for this model family.
    first = facts.remember(real_ctx, "Cabang timur buka pukul 08.00.", keywords=[], kind="stable")
    second = facts.remember(real_ctx, "Cabang barat buka pukul 10.00.", keywords=[], kind="stable")
    assert first.fact_id != second.fact_id and second.similar == []


def test_recall_answers_in_other_words_and_stays_silent_off_topic(real_ctx: Context) -> None:
    stored = {
        "retur": "Pelanggan boleh retur barang maksimal tujuh hari setelah diterima.",
        "wifi": "Sandi wifi kantor diganti setiap tiga bulan oleh admin.",
        "gaji": "Gaji dibayarkan setiap tanggal 28.",
        "blog": "Artikel blog terbit setiap Selasa dan Kamis.",
        "backup": "Server dicadangkan otomatis tiap malam pukul 02.00.",
    }
    ids = {
        key: facts.remember(real_ctx, text, keywords=[key], kind="stable").fact_id
        for key, text in stored.items()
    }
    # No word of the question appears in the fact or its keyword. (Measured: 0.87 to the
    # right fact, 0.84 to the next one. The margins of this small model are thin; see
    # docs/EMBEDDINGS.md for how often a fact phrased in other words clears the floor.)
    hits = search.recall(real_ctx, "sampai kapan pembeli boleh mengembalikan pesanan", touch=False)
    assert hits.facts and hits.facts[0].id == ids["retur"]
    # Nothing in the store is about these (best score 0.74 to 0.79), and nothing is returned.
    for question in ("who wrote Hamlet", "pemenang piala dunia", "cara menanam cabai di pot"):
        assert search.recall(real_ctx, question, touch=False).facts == [], question
