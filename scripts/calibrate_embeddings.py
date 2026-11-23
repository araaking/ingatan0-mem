"""Measure how an embedding model scores Indonesian memory text, to set its cut-offs.

Cosine scores are not comparable between model families: one model scores two unrelated
sentences 0.15, another 0.85. Ingatan therefore keeps a `SimilarityProfile` per family
(`ingatan.search.embed`). This script produces the numbers behind a profile, for the model
in your configuration or the one named on the command line:

    python scripts/calibrate_embeddings.py
    python scripts/calibrate_embeddings.py --provider local --model intfloat/multilingual-e5-small
    python scripts/calibrate_embeddings.py --provider ollama --model embeddinggemma

It embeds a fixed set of facts, each with a paraphrase, a version whose value changed, and
two questions, plus questions about subjects no fact covers. It prints the score ranges per
kind of pair and what each candidate cut-off would do. Nothing is stored, no Ingatan data
folder is opened, and the only network use is whatever the chosen provider itself needs.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from ingatan.scope import load_embeddings
from ingatan.search import embed, stem_id

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Item:
    topic: str
    fact: str
    paraphrase: str  # the same statement in other words
    changed: str  # the same statement with a different value
    queries: tuple[str, str]  # one that shares words with the fact, one that does not


ITEMS: tuple[Item, ...] = (
    Item(
        "pengiriman",
        "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler.",
        "Biaya kirim reguler tujuan Bandung lima belas ribu rupiah tiap kilogram.",
        "Ongkir ke Bandung Rp 18.000 per kg lewat kurir reguler.",
        ("berapa ongkir ke Bandung", "tarif pengiriman paket tujuan Bandung"),
    ),
    Item(
        "pengiriman",
        "Paket dikirim paling lambat pukul 15.00 setiap hari kerja.",
        "Batas akhir pengiriman barang di hari kerja adalah jam tiga sore.",
        "Paket dikirim paling lambat pukul 17.00 setiap hari kerja.",
        ("jam berapa batas kirim paket", "sampai kapan pesanan masih diberangkatkan hari ini"),
    ),
    Item(
        "pengiriman",
        "Pengiriman ke luar Jawa memakai ekspedisi kargo dengan estimasi lima hari.",
        "Untuk tujuan di luar Pulau Jawa barang dikirim via kargo, perkiraan tiba lima hari.",
        "Pengiriman ke luar Jawa memakai ekspedisi kargo dengan estimasi sembilan hari.",
        ("berapa lama kirim ke luar Jawa", "ekspedisi apa untuk Kalimantan dan Sulawesi"),
    ),
    Item(
        "harga",
        "Harga kopi susu ukuran besar Rp 22.000 per gelas.",
        "Segelas kopi susu large dijual dua puluh dua ribu rupiah.",
        "Harga kopi susu ukuran besar Rp 25.000 per gelas.",
        ("harga kopi susu besar", "berapa biaya satu gelas minuman kopi dengan susu"),
    ),
    Item(
        "harga",
        "Diskon 10 persen berlaku untuk pembelian di atas Rp 500.000.",
        "Belanja lebih dari lima ratus ribu rupiah mendapat potongan sepuluh persen.",
        "Diskon 15 persen berlaku untuk pembelian di atas Rp 500.000.",
        ("syarat dapat diskon", "minimal belanja berapa supaya dapat potongan harga"),
    ),
    Item(
        "harga",
        "Harga grosir berlaku mulai pembelian 12 lusin.",
        "Tarif grosir baru bisa dipakai kalau membeli minimal dua belas lusin.",
        "Harga grosir berlaku mulai pembelian 24 lusin.",
        ("mulai berapa lusin harga grosir", "kapan pembeli dapat tarif partai besar"),
    ),
    Item(
        "jadwal",
        "Toko tutup setiap hari Minggu.",
        "Setiap Minggu toko tidak beroperasi.",
        "Toko tutup setiap hari Senin.",
        ("kapan toko libur", "hari apa gerai tidak buka"),
    ),
    Item(
        "jadwal",
        "Rapat mingguan tim diadakan Senin pukul 09.00.",
        "Tim bertemu rutin tiap Senin jam sembilan pagi.",
        "Rapat mingguan tim diadakan Rabu pukul 13.00.",
        ("jadwal rapat mingguan", "kapan pertemuan rutin seluruh anggota"),
    ),
    Item(
        "jadwal",
        "Stok opname dilakukan setiap akhir bulan.",
        "Penghitungan fisik persediaan dijadwalkan tiap penghujung bulan.",
        "Stok opname dilakukan setiap akhir kuartal.",
        ("kapan stok opname", "jadwal menghitung ulang persediaan gudang"),
    ),
    Item(
        "pembayaran",
        "Supplier kopi utama adalah toko-a dengan tempo pembayaran 14 hari.",
        "Pemasok kopi utama kita toko-a, tagihannya jatuh tempo dua minggu.",
        "Supplier kopi utama adalah toko-b dengan tempo pembayaran 30 hari.",
        ("siapa supplier kopi", "berapa lama tempo bayar ke pemasok biji kopi"),
    ),
    Item(
        "pembayaran",
        "Pembayaran lewat transfer bank dikonfirmasi manual oleh kasir.",
        "Kasir mengecek sendiri setiap pembayaran yang masuk via transfer bank.",
        "Pembayaran lewat transfer bank dikonfirmasi otomatis oleh sistem.",
        ("cara konfirmasi transfer", "siapa yang memverifikasi uang masuk ke rekening"),
    ),
    Item(
        "pembayaran",
        "Tagihan listrik dibayar sebelum tanggal 20 setiap bulan.",
        "Batas pembayaran listrik bulanan adalah sebelum tanggal dua puluh.",
        "Tagihan listrik dibayar sebelum tanggal 5 setiap bulan.",
        ("kapan bayar listrik", "tenggat pelunasan rekening PLN"),
    ),
    Item(
        "stok",
        "Stok gula di gudang tinggal 20 karung.",
        "Persediaan gula di gudang tersisa dua puluh karung.",
        "Stok gula di gudang tinggal 5 karung.",
        ("sisa stok gula", "berapa karung pemanis yang masih ada di tempat penyimpanan"),
    ),
    Item(
        "stok",
        "Pesanan ulang susu dibuat saat stok di bawah 30 liter.",
        "Kalau susu kurang dari tiga puluh liter, langsung pesan lagi.",
        "Pesanan ulang susu dibuat saat stok di bawah 80 liter.",
        ("kapan pesan ulang susu", "batas minimum persediaan sebelum order lagi"),
    ),
    Item(
        "stok",
        "Barang rusak dicatat di lembar retur gudang.",
        "Setiap barang cacat ditulis pada lembar retur milik gudang.",
        "Barang rusak dicatat di aplikasi kasir.",
        ("di mana mencatat barang rusak", "prosedur pendataan produk cacat"),
    ),
    Item(
        "pelanggan",
        "Pelanggan boleh retur barang maksimal tujuh hari setelah diterima.",
        "Batas pengembalian barang oleh pembeli adalah 7 hari sejak barang sampai.",
        "Pelanggan boleh retur barang maksimal tiga hari setelah diterima.",
        ("berapa hari batas retur", "sampai kapan pembeli boleh mengembalikan pesanan"),
    ),
    Item(
        "pelanggan",
        "Keluhan pelanggan dijawab paling lama 24 jam.",
        "Setiap komplain pembeli harus dibalas dalam satu hari.",
        "Keluhan pelanggan dijawab paling lama 48 jam.",
        ("berapa lama respon keluhan", "target waktu membalas komplain pembeli"),
    ),
    Item(
        "pelanggan",
        "Pelanggan baru mendapat gratis ongkir untuk pesanan pertama.",
        "Pesanan pertama dari pembeli baru bebas biaya kirim.",
        "Pelanggan baru mendapat voucher Rp 25.000 untuk pesanan pertama.",
        ("promo pelanggan baru", "keuntungan apa untuk pembeli yang baru mendaftar"),
    ),
    Item(
        "konten",
        "Artikel blog terbit setiap Selasa dan Kamis.",
        "Blog diperbarui dua kali seminggu, hari Selasa serta Kamis.",
        "Artikel blog terbit setiap Senin dan Jumat.",
        ("jadwal terbit artikel", "kapan tulisan baru dipublikasikan di situs"),
    ),
    Item(
        "konten",
        "Panjang artikel minimal 1.200 kata dengan satu kata kunci utama.",
        "Setiap tulisan paling sedikit seribu dua ratus kata dan fokus pada satu keyword.",
        "Panjang artikel minimal 800 kata dengan satu kata kunci utama.",
        ("panjang minimal artikel", "berapa kata sebaiknya satu tulisan blog"),
    ),
    Item(
        "konten",
        "Judul halaman tidak boleh lebih dari 60 karakter.",
        "Title halaman dibatasi maksimal enam puluh karakter.",
        "Judul halaman tidak boleh lebih dari 70 karakter.",
        ("batas panjang judul halaman", "maksimal karakter title tag"),
    ),
    Item(
        "teknis",
        "Server dicadangkan otomatis tiap malam pukul 02.00.",
        "Backup server berjalan sendiri setiap dini hari jam dua.",
        "Server dicadangkan otomatis tiap Minggu pukul 23.00.",
        ("kapan backup server", "jadwal pencadangan data otomatis"),
    ),
    Item(
        "teknis",
        "Skrip sync_report.py dijalankan cron setiap enam jam.",
        "Cron memanggil sync_report.py empat kali sehari, tiap enam jam.",
        "Skrip sync_report.py dijalankan cron setiap dua jam.",
        ("seberapa sering sync_report jalan", "interval penjadwalan sinkronisasi laporan"),
    ),
    Item(
        "teknis",
        "Sandi wifi kantor diganti setiap tiga bulan oleh admin.",
        "Admin memperbarui password wifi kantor tiap kuartal.",
        "Sandi wifi kantor diganti setiap bulan oleh admin.",
        ("kapan sandi wifi diganti", "siapa yang mengurus kata kunci jaringan nirkabel"),
    ),
    Item(
        "preferensi",
        "Budi lebih suka laporan dikirim lewat email setiap Senin pagi.",
        "Kirim laporan ke Budi via email tiap hari Senin pagi, itu preferensinya.",
        "Budi lebih suka laporan dikirim lewat grup chat setiap Jumat sore.",
        ("Budi mau laporan dikirim lewat apa", "cara Budi ingin menerima rekap mingguan"),
    ),
    Item(
        "preferensi",
        "Jawaban untuk Budi ditulis singkat dalam bahasa Indonesia.",
        "Budi minta balasan yang ringkas dan berbahasa Indonesia.",
        "Jawaban untuk Budi ditulis lengkap dalam bahasa Inggris.",
        ("gaya jawaban yang disukai Budi", "bahasa apa yang dipakai saat membalas Budi"),
    ),
    Item(
        "preferensi",
        "Jangan menghubungi Budi di luar jam kerja kecuali darurat.",
        "Budi hanya mau dikontak di luar jam kantor kalau ada keadaan darurat.",
        "Budi boleh dihubungi kapan saja termasuk akhir pekan.",
        ("boleh hubungi Budi malam hari", "aturan mengontak Budi setelah jam kantor"),
    ),
    Item(
        "karyawan",
        "Karyawan baru mendapat seragam pada hari pertama kerja.",
        "Seragam dibagikan kepada pegawai baru saat pertama masuk.",
        "Karyawan baru mendapat seragam pada akhir masa percobaan.",
        ("kapan karyawan baru dapat seragam", "pembagian baju kerja untuk pegawai baru"),
    ),
    Item(
        "karyawan",
        "Gaji dibayarkan setiap tanggal 28.",
        "Tanggal dua puluh delapan adalah hari gajian.",
        "Gaji dibayarkan setiap tanggal 1.",
        ("tanggal berapa gajian", "kapan upah bulanan ditransfer"),
    ),
    Item(
        "karyawan",
        "Lembur dihitung setelah delapan jam kerja per hari.",
        "Jam kerja di atas delapan jam sehari masuk hitungan lembur.",
        "Lembur dihitung setelah sembilan jam kerja per hari.",
        ("kapan mulai dihitung lembur", "aturan jam kerja tambahan"),
    ),
    Item(
        "english",
        "The staging server is redeployed every Friday at 6 pm.",
        "Each Friday evening at six the staging environment gets a fresh deploy.",
        "The staging server is redeployed every Monday at 9 am.",
        ("when is staging redeployed", "jadwal deploy ulang server staging"),
    ),
    Item(
        "english",
        "Refunds above $200 need approval from the finance lead.",
        "The head of finance must sign off on any refund larger than two hundred dollars.",
        "Refunds above $50 need approval from the finance lead.",
        ("who approves large refunds", "batas pengembalian dana yang perlu persetujuan keuangan"),
    ),
    Item(
        "english",
        "API keys are rotated every 90 days.",
        "We replace API credentials on a ninety-day cycle.",
        "API keys are rotated every 30 days.",
        ("how often are API keys rotated", "kapan kunci API diganti"),
    ),
)

OFF_TOPIC: tuple[str, ...] = (
    "resep rendang padang yang enak",
    "jadwal pertandingan bulu tangkis",
    "cara menanam cabai di pot",
    "ibu kota negara Australia",
    "cuaca hujan badai besok",
    "film terbaru di bioskop",
    "cara mengganti ban mobil",
    "sejarah candi Borobudur",
    "rumus luas lingkaran",
    "lirik lagu daerah",
    "harga tiket pesawat ke Tokyo",
    "gejala demam berdarah",
    "cara merawat kucing persia",
    "pemenang piala dunia",
    "tips belajar gitar untuk pemula",
    "jarak bumi ke bulan",
    "arti mimpi dikejar ular",
    "cara membuat akun media sosial",
    "ukuran lapangan sepak bola",
    "obat sakit gigi alami",
    "best hiking trails in the Alps",
    "how to bake sourdough bread",
    "symptoms of the common cold",
    "who wrote Hamlet",
)


# A fact, a fact on the same template about another subject, and an update of the fact.
# If a model scores the first two as high as the first and the third, its score cannot say
# whether two facts are about one thing.
TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "Stok gula di gudang tinggal 20 karung.",
        "Stok tepung di gudang tinggal 20 karung.",
        "Stok gula di gudang tinggal 5 karung.",
    ),
    (
        "Ongkir ke Bandung Rp 15.000 per kg lewat kurir reguler.",
        "Ongkir ke Surabaya Rp 21.000 per kg lewat kurir reguler.",
        "Ongkir ke Bandung Rp 18.000 per kg lewat kurir reguler.",
    ),
    (
        "Cabang timur buka pukul 08.00.",
        "Cabang barat buka pukul 10.00.",
        "Cabang timur buka pukul 10.00.",
    ),
    (
        "Gaji kasir dibayarkan setiap tanggal 28.",
        "Gaji kurir dibayarkan setiap tanggal 5.",
        "Gaji kasir dibayarkan setiap tanggal 5.",
    ),
    (
        "Harga kopi susu ukuran besar Rp 22.000 per gelas.",
        "Harga teh manis ukuran besar Rp 8.000 per gelas.",
        "Harga kopi susu ukuran besar Rp 25.000 per gelas.",
    ),
    (
        "Budi lebih suka laporan lewat email.",
        "Sari lebih suka laporan lewat grup chat.",
        "Budi lebih suka laporan lewat grup chat.",
    ),
    (
        "Toko tutup setiap hari Minggu.",
        "Gudang tutup setiap hari Minggu.",
        "Toko tutup setiap hari Senin.",
    ),
)

# Different facts about one entity: what an entity-anchored conflict cut-off must not flag.
ENTITY_GROUPS: dict[str, tuple[str, ...]] = {
    "Budi": (
        "Budi lebih suka laporan dikirim lewat email setiap Senin pagi.",
        "Jawaban untuk Budi ditulis singkat dalam bahasa Indonesia.",
        "Jangan menghubungi Budi di luar jam kerja kecuali darurat.",
        "Budi tinggal di Bandung dan bekerja dari rumah.",
        "Budi alergi kacang dan tidak minum kopi.",
    ),
    "gudang": (
        "Stok gula di gudang tinggal 20 karung.",
        "Barang rusak dicatat di lembar retur gudang.",
        "Stok opname gudang dilakukan setiap akhir bulan.",
        "Kunci gudang dipegang oleh kepala toko.",
        "Gudang dibersihkan setiap Jumat sore.",
    ),
    "toko-a": (
        "Supplier kopi utama adalah toko-a dengan tempo pembayaran 14 hari.",
        "Toko-a mengirim barang setiap hari Selasa.",
        "Kontak toko-a adalah bagian penjualan lewat email.",
        "Toko-a memberi diskon 5 persen untuk pembelian di atas 50 kg.",
    ),
}


def _shares_a_word(query: str, fact: str) -> bool:
    """Would the word index already connect this question to this fact?

    True when a word of the question, or one of its Indonesian roots, occurs in the fact.
    """
    body = fact.casefold()
    words = _WORD.findall(query.casefold())
    roots = [root for word in words for root in stem_id.candidates(word)]
    return any(len(term) >= 3 and term in body for term in [*words, *roots])


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))]


def _row(label: str, values: list[float]) -> str:
    return (
        f"  {label:36s} n={len(values):4d}  min {min(values):.3f}  p05 "
        f"{_percentile(values, 0.05):.3f}  median {_percentile(values, 0.5):.3f}  p95 "
        f"{_percentile(values, 0.95):.3f}  max {max(values):.3f}"
    )


def _share(values: list[float], cut: float) -> str:
    return f"{100 * sum(1 for v in values if v >= cut) / len(values):5.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--provider", help="override [embeddings] provider")
    parser.add_argument("--model", help="override [embeddings] model")
    parser.add_argument("--model-dir", help="override the local model folder")
    args = parser.parse_args()

    cfg = load_embeddings()
    changes = {
        key: value
        for key, value in (
            ("provider", args.provider),
            ("model", args.model),
            ("model_dir", args.model_dir),
        )
        if value
    }
    cfg = dataclasses.replace(cfg, **changes)
    if not embed.is_enabled(cfg):
        print("Embeddings are off (provider = 'none'). Pass --provider.", file=sys.stderr)
        return 2
    profile = embed.similarity_profile(cfg)
    print(f"provider {cfg.provider}, model {embed.effective_model(cfg)}")
    print(
        f"profile in use: recall_floor {profile.recall_floor}, conflict {profile.conflict}, "
        f"conflict_strict {profile.conflict_strict}\n"
    )

    def passages(texts: list[str]) -> list[list[float]]:
        vectors = embed.embed_batch(cfg, texts, is_query=False)
        if any(v is None for v in vectors):
            raise SystemExit(f"embedding failed: {embed.last_embed_error()}")
        return [v for v in vectors if v is not None]

    def queries(texts: list[str]) -> list[list[float]]:
        vectors = embed.embed_batch(cfg, texts, is_query=True)
        if any(v is None for v in vectors):
            raise SystemExit(f"embedding failed: {embed.last_embed_error()}")
        return [v for v in vectors if v is not None]

    facts = passages([item.fact for item in ITEMS])
    paraphrases = passages([item.paraphrase for item in ITEMS])
    changed = passages([item.changed for item in ITEMS])
    asked = queries([q for item in ITEMS for q in item.queries])
    off_topic = queries(list(OFF_TOPIC))

    # -- fact against fact: what conflict detection compares ---------------------------
    para_scores = [_cos(f, p) for f, p in zip(facts, paraphrases, strict=True)]
    changed_scores = [_cos(f, c) for f, c in zip(facts, changed, strict=True)]
    same_topic: list[float] = []
    unrelated: list[float] = []
    for i, first in enumerate(ITEMS):
        for j in range(i + 1, len(ITEMS)):
            score = _cos(facts[i], facts[j])
            (same_topic if first.topic == ITEMS[j].topic else unrelated).append(score)
    print("fact against fact")
    print(_row("same fact, value changed", changed_scores))
    print(_row("same fact, other words", para_scores))
    print(_row("same topic, different fact", same_topic))
    print(_row("different topic", unrelated))
    updates_above = sum(1 for c, p in zip(changed_scores, para_scores, strict=True) if c > p)
    print(
        f"  a changed value scores higher than a paraphrase of the same fact in "
        f"{updates_above} of {len(ITEMS)} items"
    )
    siblings: list[float] = []
    updates: list[float] = []
    for base, sibling, update in TEMPLATES:
        b, s_, u = passages([base, sibling, update])
        siblings.append(_cos(b, s_))
        updates.append(_cos(b, u))
    print(_row("one template, another subject", siblings))
    print(_row("one template, the same subject", updates))
    same_entity: list[float] = []
    for texts in ENTITY_GROUPS.values():
        vectors = passages(list(texts))
        same_entity += [
            _cos(vectors[i], vectors[j])
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
    print(_row("one entity, different facts", same_entity))
    for cut in (0.88, 0.90, 0.92):
        flagged = sum(1 for v in same_entity if v >= cut)
        print(f"    at {cut:.2f}: {flagged} of {len(same_entity)} one-entity pairs are flagged")
    print("\n  share of pairs at or above a cut-off")
    print("  cut-off   value changed   other words   same topic   different topic")
    for step in range(80, 100, 2):
        cut = step / 100
        print(
            f"  {cut:.2f}      {_share(changed_scores, cut)}          "
            f"{_share(para_scores, cut)}        {_share(same_topic, cut)}       "
            f"{_share(unrelated, cut)}"
        )

    # -- question against fact: what recall compares -----------------------------------
    gold: list[float] = []
    best_other: list[float] = []
    top1 = 0
    # The questions the word index cannot answer: no word in common with their fact.
    wordless: list[float] = []
    wordless_top1 = 0
    questions = [q for item in ITEMS for q in item.queries]
    for index, vector in enumerate(asked):
        owner = index // 2
        scores = [_cos(vector, fact) for fact in facts]
        gold.append(scores[owner])
        others = [s for k, s in enumerate(scores) if k != owner]
        best_other.append(max(others))
        first = scores[owner] >= max(others)
        top1 += first
        if not _shares_a_word(questions[index], ITEMS[owner].fact):
            wordless.append(scores[owner])
            wordless_top1 += first
    off_best = [max(_cos(vector, fact) for fact in facts) for vector in off_topic]
    print("\nquestion against fact")
    print(_row("question and its own fact", gold))
    print(_row("question and the best other fact", best_other))
    print(_row("off-topic question, best fact", off_best))
    print(f"  the right fact ranks first for {top1} of {len(asked)} questions")
    if wordless:
        kept = sum(1 for v in wordless if v >= profile.recall_floor)
        print(_row("... question sharing no word with it", wordless))
        print(
            f"  of those {len(wordless)} questions (the ones only a model can answer) the "
            f"right fact ranks first for {wordless_top1} and clears the floor of "
            f"{profile.recall_floor} for {kept}"
        )
    print("\n  what a recall floor would do")
    print("  floor   right fact kept   off-topic questions answered with nothing")
    for step in range(76, 92, 1):
        cut = step / 100
        silent = 100 * sum(1 for v in off_best if v < cut) / len(off_best)
        print(f"  {cut:.2f}    {_share(gold, cut)}            {silent:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
