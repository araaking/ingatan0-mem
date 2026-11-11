from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from ingatan import facts, proposals, search, timeutil
from ingatan.scope import Context

MakeContext = Callable[..., Context]
InsertFact = Callable[..., str]


def rem(ctx: Context, text: str, keywords: list[str] | None = None, **kw: Any) -> str:
    result = facts.remember(ctx, text, keywords=keywords or ["x"], **kw)
    assert result.fact_id is not None
    return result.fact_id


def ids(result: search.RecallResult) -> list[str]:
    return [h.id for h in result.facts]


def insert_note(conn: sqlite3.Connection, heading: str, text: str, path: str = "/n/a.md") -> int:
    cursor = conn.execute(
        "INSERT INTO notes (path, heading, text, file_hash, created_at) VALUES (?, ?, ?, 'h', 't')",
        (path, heading, text),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


# ---- validity (T-8: superseded and not-yet-valid facts are not returned) ------------


def test_superseded_fact_is_not_returned_by_default(ctx: Context) -> None:
    old = rem(ctx, "Harga paket basic 100 ribu", ["harga"])
    new = rem(ctx, "Harga paket basic 120 ribu", ["harga"])
    facts.supersede(ctx, old, new)
    assert ids(search.recall(ctx, "harga paket basic")) == [new]


def test_not_yet_valid_fact_is_not_returned_until_its_time(
    ctx: Context, insert_fact: InsertFact, frozen_clock: Any
) -> None:
    fid = insert_fact(
        ctx.business,
        text="Promo ramadan mulai",
        keywords=["promo"],
        valid_from="2026-12-01T00:00:00Z",
    )
    assert search.recall(ctx, "promo ramadan").facts == []
    assert search.recall(ctx, "promo ramadan", include_closed=True).facts == []
    frozen_clock.set(timeutil.parse("2026-12-01T00:00:01Z"))
    assert ids(search.recall(ctx, "promo ramadan")) == [fid]


def test_expired_fact_is_not_returned_by_default(
    ctx: Context, insert_fact: InsertFact, frozen_clock: Any
) -> None:
    insert_fact(
        ctx.business,
        text="Diskon akhir tahun",
        valid_to="2026-11-12T00:00:00Z",
    )
    assert len(search.recall(ctx, "diskon akhir tahun").facts) == 1
    frozen_clock.advance(days=2)
    assert search.recall(ctx, "diskon akhir tahun").facts == []


def test_include_closed_returns_superseded_and_expired_with_their_status(
    ctx: Context, insert_fact: InsertFact, frozen_clock: Any
) -> None:
    old = rem(ctx, "Ongkir Jakarta 10 ribu", ["ongkir"])
    new = rem(ctx, "Ongkir Jakarta 12 ribu", ["ongkir"])
    facts.supersede(ctx, old, new)
    expired = insert_fact(
        ctx.business, text="Ongkir Bandung gratis", valid_to="2026-11-11T09:00:00Z"
    )
    frozen_clock.advance(minutes=1)
    hits = {h.id: h for h in search.recall(ctx, "ongkir", include_closed=True).facts}
    assert hits[old].status == "superseded"
    assert hits[expired].status == "expired"
    assert hits[new].status == "valid"
    assert expired not in ids(search.recall(ctx, "ongkir"))


def test_is_valid_sql_is_the_shared_validity_filter(
    ctx: Context, insert_fact: InsertFact, frozen_clock: Any
) -> None:
    good = insert_fact(ctx.business)
    insert_fact(ctx.business, superseded_by="f_other0000000")
    insert_fact(ctx.business, valid_to="2026-01-01T00:00:00Z")
    insert_fact(ctx.business, valid_from="2999-01-01T00:00:00Z")
    rows = ctx.business.execute(
        f"SELECT id FROM facts WHERE {search.is_valid_sql()}",
        {"now": timeutil.iso(timeutil.now())},
    ).fetchall()
    assert [r["id"] for r in rows] == [good]


# ---- result shape -------------------------------------------------------------------


def test_hit_carries_id_text_tier_source_date_status_entities_keywords_score(
    ctx: Context,
) -> None:
    fid = rem(ctx, "Stok kopi tinggal 5 kg", ["kopi", "stok"], entities=["Kopi Gayo"])
    hit = search.recall(ctx, "kopi").facts[0]
    assert hit.id == fid
    assert hit.text == "Stok kopi tinggal 5 kg"
    assert hit.tier == "agent"
    assert hit.source["agent"] == "claude-code"
    assert hit.date == "2026-11-11" or len(hit.date) == 10
    assert hit.status == "valid"
    assert hit.entities == ["Kopi Gayo"]
    assert hit.keywords == ["kopi", "stok"]
    assert hit.score > 0


# ---- ranking ------------------------------------------------------------------------


def test_entity_match_outranks_an_equal_bm25_non_match(
    ctx: Context, insert_fact: InsertFact
) -> None:
    # Two facts whose index rows are identical (so identical BM25). Only one of them lists
    # the entity, and the other one is newer, which the recency signal prefers.
    with_entity = insert_fact(
        ctx.business,
        text="Pesanan Budi dikirim hari Senin",
        keywords=["pesanan"],
        entities=["Budi"],
        valid_from="2026-11-01T00:00:00Z",
    )
    without = insert_fact(
        ctx.business,
        text="Pesanan Budi dikirim hari Senin",
        keywords=["pesanan"],
        entities=[],
        valid_from="2026-11-05T00:00:00Z",
    )
    for table in ("facts_fts", "facts_tri"):
        ctx.business.execute(f"UPDATE {table} SET entities = ''")
    hits = search.recall(ctx, "Budi pesanan")
    assert ids(hits) == [with_entity, without]
    assert hits.facts[0].score > hits.facts[1].score


def test_more_matching_entities_rank_higher(ctx: Context) -> None:
    one = rem(ctx, "Kontrak supplier ditandatangani", ["kontrak"], entities=["Budi"])
    two = rem(ctx, "Kontrak supplier ditandatangani", ["kontrak"], entities=["Budi", "Ani"])
    assert ids(search.recall(ctx, "kontrak Budi Ani"))[0] == two
    assert one in ids(search.recall(ctx, "kontrak Budi Ani"))


def test_entity_names_match_case_insensitively(ctx: Context) -> None:
    fid = rem(ctx, "Pelanggan setia", ["pelanggan"], entities=["Toko Maju"])
    assert ids(search.recall(ctx, "toko maju"))[0] == fid


def test_newer_fact_wins_when_everything_else_is_equal(ctx: Context, frozen_clock: Any) -> None:
    older = rem(ctx, "Jadwal rapat mingguan", ["rapat"])
    frozen_clock.advance(days=3)
    newer = rem(ctx, "Jadwal rapat mingguan", ["rapat"])
    assert ids(search.recall(ctx, "jadwal rapat")) == [newer, older]


def test_limit_cuts_after_ranking(ctx: Context) -> None:
    for n in range(8):
        rem(ctx, f"Catatan stok gudang {n}", ["stok"])
    assert len(search.recall(ctx, "stok gudang", limit=3).facts) == 3
    assert search.recall(ctx, "stok gudang", limit=0).facts == []


def test_results_from_both_files_are_merged_before_the_cut(make_ctx: MakeContext) -> None:
    ctx = make_ctx()
    biz = facts.remember(ctx, "Pajak toko dibayar tiap bulan", keywords=["pajak"], tier="business")
    glo = facts.remember(ctx, "Pajak pribadi dibayar tiap tahun", keywords=["pajak"], tier="global")
    assert biz.proposal_id and glo.proposal_id
    proposals.approve(ctx, biz.proposal_id)
    proposals.approve(ctx, glo.proposal_id)
    hits = search.recall(ctx, "pajak dibayar", limit=2)
    assert {h.tier for h in hits.facts} == {"business", "global"}


# ---- Indonesian (FR-29, FR-30) ------------------------------------------------------


def test_inflected_query_finds_the_fact_through_its_keywords(ctx: Context) -> None:
    fid = rem(
        ctx,
        "Pelanggan membayar lewat transfer bank",
        ["bayar", "pembayaran", "transfer"],
    )
    assert ids(search.recall(ctx, "pembayaran")) == [fid]


def test_without_the_keyword_the_inflected_form_is_found_through_the_root(ctx: Context) -> None:
    # Used to be a gap. "pembayaran" now expands to "bayar", a fragment of "membayar".
    fid = rem(ctx, "Pelanggan membayar lewat transfer bank", ["transfer"])
    assert ids(search.recall(ctx, "pembayaran")) == [fid]


def test_an_unrelated_word_is_still_not_found(ctx: Context) -> None:
    rem(ctx, "Pelanggan membayar lewat transfer bank", ["transfer"])
    assert search.recall(ctx, "pengiriman").facts == []


def test_root_word_query_finds_the_inflected_text_by_trigram(ctx: Context) -> None:
    fid = rem(ctx, "Pembayaran dilakukan lewat transfer", ["transfer"])
    assert ids(search.recall(ctx, "bayar")) == [fid]


def test_partial_word_finds_the_whole_word_by_trigram(ctx: Context) -> None:
    fid = rem(ctx, "Kirim invoice setiap tanggal 25", ["tagihan"])
    assert ids(search.recall(ctx, "invo")) == [fid]


@pytest.mark.parametrize(
    ("text", "keywords", "query"),
    [
        ("Barang dikirim memakai JNE", ["kirim", "pengiriman"], "pengiriman"),
        ("Stok kopi menipis, perlu belanja", ["belanja", "pembelian"], "pembelian"),
        ("Karyawan baru bekerja mulai Senin", ["kerja", "pekerjaan"], "pekerjaan"),
        ("Kami menjual kopi gayo", ["jual", "penjualan"], "penjualan"),
        ("Tagihan listrik ditagih tiap bulan", ["tagih", "penagihan"], "penagihan"),
    ],
)
def test_five_indonesian_inflection_scenarios(
    ctx: Context, text: str, keywords: list[str], query: str
) -> None:
    fid = rem(ctx, text, keywords)
    assert ids(search.recall(ctx, query)) == [fid]


def test_accents_and_case_do_not_block_a_match(ctx: Context) -> None:
    fid = rem(ctx, "Menu spesial: crème brûlée", ["dessert"])
    assert ids(search.recall(ctx, "CREME brulee")) == [fid]


# ---- query-side expansion with affix-stripped roots (FR-30) ---------------------------


def test_inflected_query_finds_a_root_keyword_and_a_differently_inflected_text(
    ctx: Context,
) -> None:
    fid = rem(
        ctx,
        "Penukaran produk cacat gratis dan pelanggan tidak menanggung ongkos kirim.",
        ["tukar", "retur", "garansi"],
    )
    rem(ctx, "Stok kaos hitam ukuran L tinggal sepuluh pcs.", ["persediaan"])
    assert ids(search.recall(ctx, "bisa menukar pesanan"))[:1] == [fid]


def test_dikirim_finds_a_root_keyword_with_pengiriman_text(ctx: Context) -> None:
    fid = rem(
        ctx,
        "Pengiriman luar pulau memakai JNE Reguler dengan tarif Rp 35.000 per kilo.",
        ["kirim", "ekspedisi", "ongkos"],
    )
    rem(ctx, "Gudang tutup pukul 17.00 dan paket belum dikemas dikerjakan esok.", ["packing"])
    assert ids(search.recall(ctx, "kapan dikirim"))[:1] == [fid]


def test_reverse_direction_text_only_inflected_form_found_by_another_form(
    ctx: Context,
) -> None:
    # No keyword holds the root, and "kirim" is not inside "pengiriman" (the k fell away
    # when the prefix was added). The assimilated candidate "ngirim" is.
    fid = rem(ctx, "Pengiriman ke luar pulau butuh lima hari", ["x"])
    assert ids(search.recall(ctx, "dikirim")) == [fid]


def test_reverse_direction_other_forms_of_one_root(ctx: Context) -> None:
    fid = rem(ctx, "Penjualan kopi gayo naik bulan ini", ["x"])
    assert ids(search.recall(ctx, "dijual")) == [fid]
    fid2 = rem(ctx, "Pembayaran lewat transfer bank", ["x"])
    assert ids(search.recall(ctx, "membayar"))[:1] == [fid2]


def test_a_literal_match_outranks_a_root_only_match_even_when_the_root_fact_is_newer(
    ctx: Context, insert_fact: InsertFact
) -> None:
    literal = insert_fact(
        ctx.business,
        text="Paket dikirim setiap hari kerja",
        keywords=["paket"],
        valid_from="2026-11-01T00:00:00Z",
    )
    root_only = insert_fact(
        ctx.business,
        text="Ongkos pengiriman ditanggung toko",
        keywords=["ongkos"],
        valid_from="2026-11-09T00:00:00Z",
    )
    hits = search.recall(ctx, "dikirim")
    assert ids(hits) == [literal, root_only]
    assert hits.facts[0].score > hits.facts[1].score


def test_a_literal_match_outranks_a_root_only_match_in_a_crowd(
    ctx: Context, insert_fact: InsertFact
) -> None:
    root_ids = [
        insert_fact(
            ctx.business,
            text=f"Catatan pengiriman nomor {n}",
            keywords=["catatan"],
            valid_from=f"2026-11-{n + 2:02d}T00:00:00Z",
        )
        for n in range(1, 8)
    ]
    literal = insert_fact(
        ctx.business,
        text="Barang dikirim lewat kurir",
        keywords=["barang"],
        valid_from="2026-11-01T00:00:00Z",
    )
    got = ids(search.recall(ctx, "dikirim", limit=10))
    assert got[0] == literal
    assert set(got[1:]) == set(root_ids)


@pytest.mark.parametrize(
    ("query", "unrelated"),
    [
        ("invoice", "Meeting dengan vendor hari Selasa"),
        ("meeting", "Invoice dikirim tiap tanggal 25"),
        ("deadline", "Premium plan dibayar tahunan"),
        ("premium", "Mention akun resmi di postingan"),
        ("mention", "Deadline laporan hari Jumat"),
        ("members", "Bersama tim penjualan kami rapat"),
    ],
)
def test_english_words_do_not_drag_in_unrelated_facts(
    ctx: Context, query: str, unrelated: str
) -> None:
    rem(ctx, unrelated, ["x"])
    assert search.recall(ctx, query).facts == []


def test_expansion_never_changes_what_is_stored(ctx: Context) -> None:
    fid = rem(ctx, "Pengiriman luar pulau", ["kirim"])
    search.recall(ctx, "dikirim")
    row = ctx.business.execute("SELECT text, keywords FROM facts WHERE id = ?", (fid,)).fetchone()
    assert row["text"] == "Pengiriman luar pulau"
    assert row["keywords"] == '["kirim"]'


def test_notes_are_found_through_roots_too(ctx: Context) -> None:
    nid = insert_note(ctx.business, "Pengiriman", "Aturan ongkos dan ekspedisi")
    assert [n.id for n in search.recall(ctx, "dikirim").notes] == [nid]


# ---- queries that are not words -----------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        '"',
        "'",
        "AND",
        "AND OR NOT",
        "NEAR",
        "* ( ) :",
        "^ - + ~",
        '"" ""',
        "NOT (",
        "---",
        "()[]{}",
    ],
)
def test_query_without_real_words_returns_nothing_and_no_error(ctx: Context, query: str) -> None:
    rem(ctx, "Pengiriman and pembayaran or retur not diterima", ["x"])
    insert_note(ctx.business, "Judul", "teks and or not")
    result = search.recall(ctx, query)
    assert result.facts == [] and result.notes == []


@pytest.mark.parametrize(
    "query",
    ['"pembayaran', "pembayaran)", "pembayaran AND", "pem*", "bayar:", "x NEAR/2 y pembayaran"],
)
def test_hostile_syntax_never_raises_and_real_words_still_match(ctx: Context, query: str) -> None:
    fid = rem(ctx, "Pembayaran lewat transfer", ["pembayaran", "bayar"])
    assert fid in ids(search.recall(ctx, query))


def test_two_character_query_works_through_the_word_index_only(ctx: Context) -> None:
    fid = rem(ctx, "Kirim ke DI Yogyakarta", ["kirim"])
    assert ids(search.recall(ctx, "DI")) == [fid]


# ---- use_count and last_used (FR-31, OQ-6) ------------------------------------------


def test_recall_touches_returned_facts_only(ctx: Context, frozen_clock: Any) -> None:
    hit = rem(ctx, "Kode promo hemat", ["promo"])
    other = rem(ctx, "Alamat kantor pusat", ["alamat"])
    frozen_clock.advance(hours=2)
    search.recall(ctx, "promo hemat")
    search.recall(ctx, "promo hemat")
    rows = {r["id"]: r for r in ctx.business.execute("SELECT id, use_count, last_used FROM facts")}
    assert rows[hit]["use_count"] == 2
    assert rows[hit]["last_used"] == "2026-11-11T11:00:00Z"
    assert rows[other]["use_count"] == 0 and rows[other]["last_used"] is None


def test_touch_false_leaves_counters_alone(ctx: Context) -> None:
    fid = rem(ctx, "Kode promo hemat", ["promo"])
    search.recall(ctx, "promo", touch=False)
    assert (
        ctx.business.execute("SELECT use_count FROM facts WHERE id = ?", (fid,)).fetchone()[0] == 0
    )


def test_touch_updates_the_file_where_the_fact_lives(make_ctx: MakeContext) -> None:
    ctx = make_ctx()
    proposal = facts.remember(ctx, "Nama pemilik Budi", keywords=["pemilik"], tier="global")
    assert proposal.proposal_id
    glo_id = proposals.approve(ctx, proposal.proposal_id)
    biz_id = rem(ctx, "Nama toko Tokoku", ["toko"])
    search.recall(ctx, "nama")
    assert (
        ctx.glob.execute("SELECT use_count FROM facts WHERE id = ?", (glo_id,)).fetchone()[0] == 1
    )
    assert (
        ctx.business.execute("SELECT use_count FROM facts WHERE id = ?", (biz_id,)).fetchone()[0]
        == 1
    )
    assert (
        ctx.business.execute("SELECT count(*) FROM facts WHERE id = ?", (glo_id,)).fetchone()[0]
        == 0
    )


def test_touching_does_not_write_a_log_entry(ctx: Context) -> None:
    rem(ctx, "Kode promo hemat", ["promo"])
    before = ctx.business.execute("SELECT count(*) FROM log").fetchone()[0]
    search.recall(ctx, "promo")
    assert ctx.business.execute("SELECT count(*) FROM log").fetchone()[0] == before


def test_similar_facts_never_touch(ctx: Context) -> None:
    fid = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    search.similar_facts(ctx, "Supplier Budi", ["Budi"], ["supplier"])
    assert (
        ctx.business.execute("SELECT use_count FROM facts WHERE id = ?", (fid,)).fetchone()[0] == 0
    )


# ---- status (FR-7) ------------------------------------------------------------------


def test_check_first_for_a_volatile_fact_past_its_review_date(
    ctx: Context, frozen_clock: Any
) -> None:
    rem(ctx, "Harga bahan baku 50 ribu", ["harga"])
    assert search.recall(ctx, "harga bahan").facts[0].status == "valid"
    frozen_clock.advance(days=31)
    assert search.recall(ctx, "harga bahan").facts[0].status == "check first"


def test_a_stable_fact_never_needs_a_check(ctx: Context, frozen_clock: Any) -> None:
    rem(ctx, "Pendiri toko adalah Budi", ["pendiri"], kind="stable")
    frozen_clock.advance(days=3650)
    assert search.recall(ctx, "pendiri").facts[0].status == "valid"


def test_possible_conflict_and_check_first_together(ctx: Context, frozen_clock: Any) -> None:
    rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    rem(ctx, "Supplier Budi kirim Selasa", ["supplier"], entities=["Budi"])
    frozen_clock.advance(days=31)
    statuses = {h.status for h in search.recall(ctx, "supplier Budi").facts}
    assert statuses == {"check first, possible conflict"}


def test_status_of_takes_a_plain_row(ctx: Context, insert_fact: InsertFact) -> None:
    fid = insert_fact(ctx.business, text="x", pending_proposal="p_abc")
    row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
    assert search.status_of(ctx, row) == "check first"


# ---- scope (FR-19, FR-20, FR-23) ----------------------------------------------------


def test_agent_b_never_recalls_agent_a_private_facts(make_ctx: MakeContext) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    private = rem(a, "Catatan pribadi rahasia dagang", ["rahasia"])
    shared = rem(a, "Kantor buka jam 9", ["jam"], entities=["kantor"])
    assert private in ids(search.recall(a, "catatan rahasia"))
    assert search.recall(b, "catatan rahasia").facts == []
    assert search.recall(b, "catatan rahasia", include_closed=True).facts == []
    assert shared in ids(search.recall(a, "kantor"))
    assert search.recall(b, "kantor").facts == []
    assert search.similar_facts(b, "Kantor buka", ["kantor"], ["jam"]) == []


def test_business_facts_are_shared_between_agents_of_one_business(
    make_ctx: MakeContext,
) -> None:
    a = make_ctx(agent="agent-a")
    b = make_ctx(agent="agent-b")
    prop = facts.remember(a, "Rekening toko di Bank Maju", keywords=["rekening"], tier="business")
    assert prop.proposal_id
    fid = proposals.approve(a, prop.proposal_id)
    assert ids(search.recall(b, "rekening toko")) == [fid]


def test_business_x_never_sees_business_y(make_ctx: MakeContext) -> None:
    x = make_ctx(business="usaha-x")
    y = make_ctx(business="usaha-y")
    secret = rem(y, "Margin usaha Y tujuh puluh persen", ["margin"])
    note = insert_note(y.business, "Margin", "margin usaha Y tujuh puluh persen")
    prop = facts.remember(y, "Margin bisnis Y", keywords=["margin"], tier="business")
    assert prop.proposal_id
    biz = proposals.approve(y, prop.proposal_id)
    for query in ("margin", "usaha Y", "tujuh puluh"):
        result = search.recall(x, query, include_closed=True)
        assert result.facts == [] and result.notes == []
    assert {secret, biz} <= set(ids(search.recall(y, "margin")))
    assert note in [n.id for n in search.recall(y, "margin").notes]


def test_global_facts_are_visible_to_every_business(make_ctx: MakeContext) -> None:
    x = make_ctx(business="usaha-x")
    y = make_ctx(business="usaha-y")
    prop = facts.remember(x, "Pemilik tinggal di Medan", keywords=["pemilik"], tier="global")
    assert prop.proposal_id
    fid = proposals.approve(x, prop.proposal_id)
    assert ids(search.recall(y, "pemilik Medan")) == [fid]


def test_a_pending_proposal_is_never_recalled_not_even_by_its_author(ctx: Context) -> None:
    facts.remember(ctx, "Rahasia proposal zzpendingzz", keywords=["zzpendingzz"], tier="business")
    facts.remember(ctx, "Rahasia global zzpendingzz", keywords=["zzpendingzz"], tier="global")
    for query in ("zzpendingzz", "Rahasia proposal", "proposal"):
        assert search.recall(ctx, query).facts == []
        assert search.recall(ctx, query, include_closed=True).facts == []
    assert search.similar_facts(ctx, "Rahasia proposal", ["x"], ["zzpendingzz"]) == []


# ---- similar_facts (OQ-7) -----------------------------------------------------------


def test_similar_facts_need_a_shared_entity(ctx: Context) -> None:
    shared = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    rem(ctx, "Supplier Ani kirim Senin", ["supplier"], entities=["Ani"])
    rem(ctx, "Supplier tanpa entitas kirim Senin", ["supplier"])
    hits = search.similar_facts(ctx, "Supplier Budi kirim Selasa", ["budi"], ["supplier"])
    assert [h.id for h in hits] == [shared]
    assert hits[0].status == "valid"
    assert search.similar_facts(ctx, "Supplier kirim Selasa", [], ["supplier"]) == []


def test_similar_facts_are_the_top_three_and_can_exclude_one(ctx: Context) -> None:
    made = [
        rem(ctx, f"Supplier Budi kirim hari ke {n}", ["supplier"], entities=["Budi"])
        for n in range(5)
    ]
    assert len(search.similar_facts(ctx, "Supplier Budi kirim", ["Budi"], ["supplier"])) == 3
    top = search.similar_facts(ctx, "Supplier Budi kirim", ["Budi"], ["supplier"], limit=10)
    assert len(top) == 5
    without = search.similar_facts(
        ctx, "Supplier Budi kirim", ["Budi"], ["supplier"], exclude_id=made[0], limit=10
    )
    assert made[0] not in [h.id for h in without]


def test_similar_facts_only_look_at_valid_facts(ctx: Context) -> None:
    old = rem(ctx, "Supplier Budi kirim Senin", ["supplier"], entities=["Budi"])
    new = rem(ctx, "Supplier Budi kirim Rabu", ["supplier"], entities=["Budi"])
    facts.supersede(ctx, old, new)
    hits = search.similar_facts(ctx, "Supplier Budi kirim", ["Budi"], ["supplier"])
    assert [h.id for h in hits] == [new]


# ---- document notes (FR-62) ---------------------------------------------------------


def test_notes_are_returned_separately_from_facts(ctx: Context) -> None:
    fid = rem(ctx, "SOP refund maksimal tujuh hari", ["refund"])
    note = insert_note(ctx.business, "Prosedur refund", "Pelanggan mengajukan refund lewat email.")
    result = search.recall(ctx, "refund")
    assert ids(result) == [fid]
    assert [(n.id, n.path, n.heading) for n in result.notes] == [
        (note, "/n/a.md", "Prosedur refund")
    ]
    assert result.notes[0].text == "Pelanggan mengajukan refund lewat email."


def test_notes_match_by_trigram_and_need_three_characters(ctx: Context) -> None:
    note = insert_note(ctx.business, "Panduan", "Kirim invoice sebelum tanggal jatuh tempo")
    assert [n.id for n in search.recall(ctx, "invo").notes] == [note]
    assert [n.id for n in search.recall(ctx, "jatuh tem").notes] == [note]
    assert search.recall(ctx, "zq").notes == []


def test_a_heading_match_outranks_a_body_match(ctx: Context) -> None:
    body = insert_note(ctx.business, "Lain-lain", "Ada catatan soal retur barang rusak")
    head = insert_note(ctx.business, "Retur barang", "Hubungi gudang untuk detail")
    assert [n.id for n in search.recall(ctx, "retur barang").notes] == [head, body]


def test_notes_update_and_delete_keep_the_index_in_step(ctx: Context) -> None:
    note = insert_note(ctx.business, "Judul", "kata unikpertama di sini")
    ctx.business.execute("UPDATE notes SET text = 'kata unikkedua di sini' WHERE id = ?", (note,))
    assert search.recall(ctx, "unikpertama").notes == []
    assert [n.id for n in search.recall(ctx, "unikkedua").notes] == [note]
    ctx.business.execute("DELETE FROM notes WHERE id = ?", (note,))
    assert search.recall(ctx, "unikkedua").notes == []


def test_notes_are_limited_like_facts(ctx: Context) -> None:
    for n in range(6):
        insert_note(ctx.business, f"Catatan {n}", "isi catatan gudang", path=f"/n/{n}.md")
    assert len(search.recall(ctx, "catatan gudang", limit=2).notes) == 2


def test_recall_with_only_notes_matching_still_returns_them(ctx: Context) -> None:
    note = insert_note(ctx.business, "Kebijakan", "garansi satu tahun")
    result = search.recall(ctx, "garansi")
    assert result.facts == [] and [n.id for n in result.notes] == [note]
