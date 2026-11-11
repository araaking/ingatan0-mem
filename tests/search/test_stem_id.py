from __future__ import annotations

import pytest

from ingatan.search import stem_id

# (word, a root that must be among the candidates)
ROOTS = [
    ("pengiriman", "kirim"),
    ("penukaran", "tukar"),
    ("pembayaran", "bayar"),
    ("dikirimkan", "kirim"),
    ("menyewa", "sewa"),
    ("mengambil", "ambil"),
    ("memesan", "pesan"),
    ("penjualan", "jual"),
    ("berlangganan", "langgan"),
    ("menukar", "tukar"),
    ("dikirim", "kirim"),
    ("mengirim", "kirim"),
    ("pengirim", "kirim"),
    ("pelanggan", "langgan"),
    ("pekerjaan", "kerja"),
    ("bekerja", "kerja"),
    ("perjalanan", "jalan"),
    ("membeli", "beli"),
    ("pembelian", "beli"),
    ("mencari", "cari"),
    ("pencarian", "cari"),
    ("melihat", "lihat"),
    ("merasa", "rasa"),
    ("menanyakan", "tanya"),
    ("pemasaran", "pasar"),
    ("kehilangan", "hilang"),
    ("terbaru", "baru"),
    ("belajar", "ajar"),
    ("pelajar", "ajar"),
    ("diperbaiki", "baik"),
    ("mempertimbangkan", "timbang"),
    ("penyewaan", "sewa"),
    ("pemesanan", "pesan"),
    ("penagihan", "tagih"),
    ("penerimaan", "terima"),
    ("menagih", "tagih"),
    ("menerima", "terima"),
    ("bukunya", "buku"),
    ("makanlah", "makan"),
    ("pesanannya", "pesan"),
    ("pengembalian", "kembali"),
    ("memukul", "pukul"),
    ("menggantikan", "ganti"),
    ("menghitung", "hitung"),
    ("diselesaikan", "selesai"),
    ("pembatalan", "batal"),
    ("pengantaran", "antar"),
]


@pytest.mark.parametrize(("word", "root"), ROOTS)
def test_root_is_among_the_candidates(word: str, root: str) -> None:
    assert root in stem_id.candidates(word)


@pytest.mark.parametrize(
    "word",
    [
        "invoice", "meeting", "deadline", "premium", "mention", "members", "server", "settings",
        "penalty", "perfect", "people", "seperti", "sekolah", "tagihan", "kirim", "jual",
        "ab", "kirim123", "rp35000",
    ],
)  # fmt: skip
def test_other_words_and_roots_get_no_candidates(word: str) -> None:
    got = stem_id.candidates(word)
    if word == "tagihan":
        assert got == ["tagih"]
    else:
        assert got == []


@pytest.mark.parametrize("word", [w for w, _ in ROOTS] + ["invoice", "Pengiriman", "SERIAL"])
def test_candidates_are_capped_distinct_long_enough_and_never_the_word(word: str) -> None:
    got = stem_id.candidates(word)
    assert len(got) <= stem_id.MAX_CANDIDATES
    assert len(set(got)) == len(got)
    assert word.casefold() not in got
    assert all(len(c) >= stem_id.MIN_ROOT and c == c.casefold() for c in got)


@pytest.mark.parametrize(
    ("query_word", "stored_word"),
    [
        ("dikirim", "pengiriman"),
        ("ditukar", "penukaran"),
        ("disewa", "penyewaan"),
        ("dipesan", "pemesanan"),
        ("dijual", "penjualan"),
        ("membayar", "pembayaran"),
        ("ditagih", "penagihan"),
    ],
)
def test_a_candidate_is_a_substring_of_another_inflection_of_the_same_root(
    query_word: str, stored_word: str
) -> None:
    assert any(c in stored_word for c in stem_id.candidates(query_word))


def test_case_is_ignored() -> None:
    assert "kirim" in stem_id.candidates("PENGIRIMAN")
