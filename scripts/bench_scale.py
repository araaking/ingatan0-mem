"""Do `remember` and `recall` get slower as the store grows? Time both at several sizes.

    python scripts/bench_scale.py [SIZE ...]        # default: 1000 4000 16000

For each size a throwaway data folder is filled with that many synthetic Indonesian facts,
then 300 more are saved the normal way (so conflict pairs exist), and the median time of 60
`recall` calls and 60 further `remember` calls is printed. No embeddings, no network.

The corpus is a hard case on purpose: it draws on about sixty words, so every query word
occurs in roughly a tenth of all facts, and every fact shares its entities with many others.
Real text has a far larger vocabulary and shorter posting lists.
"""

from __future__ import annotations

import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from ingatan import facts, search
from ingatan.scope import EmbeddingConfig, Scope, open_context

# One string is easier to read and edit here than fifty one-word lines.
WORDS = (  # noqa: SIM905
    "pelanggan kopi susu roti bakar harga stok supplier gula teh toko cabang kasir "
    "pembayaran tunai transfer promo diskon pengiriman gudang pesanan laporan "
    "faktur retur ongkir kurir jadwal libur karyawan gaji lembur sewa listrik pajak "
    "margin target penjualan iklan konten artikel kata kunci halaman tautan audit "
    "server domain cadangan sandi akses kontrak vendor tagihan cicilan diskusi "
    "rapat"
).split()
CHECKED = 300
TIMED = 60


def fact(n: int, rng: random.Random) -> tuple[str, list[str], list[str]]:
    picks = rng.sample(WORDS, 6)
    text = (
        f"Catatan {n}: {picks[0]} untuk cabang {n % 40} dipesan bersama {picks[1]} dan "
        f"{picks[2]}, {picks[3]} Rp {10000 + n * 25} sejak minggu {n % 52} lewat {picks[4]}."
    )
    return text, picks[:4], [f"cabang {n % 40}", picks[5]]


def measure(size: int) -> tuple[int, float, float]:
    """(facts in the store, median remember ms, median recall ms) for a store of `size`."""
    rng = random.Random(7)
    with tempfile.TemporaryDirectory(prefix="ingatan-bench-", ignore_cleanup_errors=True) as home:
        scope = Scope(
            business="tokoku",
            agent="asisten",
            home=Path(home),
            embeddings=EmbeddingConfig(provider="none"),
        )
        ctx = open_context(scope)
        try:
            for n in range(size):
                text, keywords, entities = fact(n, rng)
                facts.remember(
                    ctx,
                    text,
                    keywords=keywords,
                    entities=entities,
                    kind="stable",
                    check_conflicts=False,
                    dedup=False,
                )
            for n in range(size, size + CHECKED):
                text, keywords, entities = fact(n, rng)
                facts.remember(ctx, text, keywords=keywords, entities=entities, kind="stable")
            recall_ms: list[float] = []
            for _ in range(TIMED):
                query = " ".join(rng.sample(WORDS, 3))
                start = time.perf_counter()
                search.recall(ctx, query, limit=5, touch=False)
                recall_ms.append((time.perf_counter() - start) * 1000)
            remember_ms: list[float] = []
            for n in range(size + CHECKED, size + CHECKED + TIMED):
                text, keywords, entities = fact(n, rng)
                start = time.perf_counter()
                facts.remember(ctx, text, keywords=keywords, entities=entities, kind="stable")
                remember_ms.append((time.perf_counter() - start) * 1000)
        finally:
            ctx.close()
    return size + CHECKED, statistics.median(remember_ms), statistics.median(recall_ms)


def main() -> int:
    sizes = [int(arg) for arg in sys.argv[1:]] or [1000, 4000, 16000]
    print("facts | remember ms (median) | recall ms (median)")
    for size in sizes:
        stored, remember_ms, recall_ms = measure(size)
        print(f"{stored:5d} | {remember_ms:8.1f} | {recall_ms:8.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
