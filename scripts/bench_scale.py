"""Do `remember` and `recall` get slower as the store grows? Time both at several sizes.

    python scripts/bench_scale.py [SIZE ...]        # default: 1000 4000 16000
    python scripts/bench_scale.py --embeddings local [--defer] [--model-dir DIR] [SIZE ...]

For each size a throwaway data folder is filled with that many synthetic Indonesian facts,
then 300 more are saved the normal way (so conflict pairs exist), and the median time of 60
`recall` calls and 60 further `remember` calls is printed. No network.

By default there are no embeddings. `--embeddings local` runs the same with the local model,
which must already be on disk (`ingatan embeddings pull`): every `remember` then waits for
the model. Adding `--defer` saves each fact without a vector, as `[embeddings] defer = true`
does, and fills the vectors in afterwards; the last column is what that fill cost per fact.
Run all three to see what deferring buys. With a model, start with small sizes: every stored
fact is embedded once.

The corpus is a hard case on purpose: it draws on about sixty words, so every query word
occurs in roughly a tenth of all facts, and every fact shares its entities with many others.
Real text has a far larger vocabulary and shorter posting lists.
"""

from __future__ import annotations

import argparse
import random
import statistics
import tempfile
import time
from pathlib import Path

from ingatan import facts, search
from ingatan.facts import backfill
from ingatan.scope import EmbeddingConfig, Scope, open_context
from ingatan.search import embed, local

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


def embedding_config(mode: str, defer: bool, model_dir: str) -> EmbeddingConfig:
    """The settings for one run, or `SystemExit` saying what is missing for the local model."""
    if mode == "none":
        if defer:
            raise SystemExit("--defer needs --embeddings local")
        return EmbeddingConfig(provider="none")
    model = local.resolve("")
    # Resolved here: the run itself works in a throwaway data folder.
    folder = (Path(model_dir) if model_dir else local.default_model_dir()).resolve()
    absent = local.missing_packages()
    if absent:
        raise SystemExit(f"--embeddings local needs {', '.join(absent)}: {local.INSTALL_HINT}")
    if local.missing_files(model, folder):
        raise SystemExit(
            f"--embeddings local: {model.name} is not in {folder}. "
            "Run `ingatan embeddings pull` or pass --model-dir."
        )
    return EmbeddingConfig(
        provider="local",
        model=model.name,
        model_dir=str(folder),
        idle_unload_seconds=0,
        defer=defer,
    )


def measure(size: int, cfg: EmbeddingConfig) -> tuple[int, float, float, float | None]:
    """(facts stored, median remember ms, median recall ms, fill ms per fact) for `size`.

    The last value is None unless `cfg.defer` is on: then the timed `remember` calls save
    without a vector, and it is the time the fill afterwards took for each of them.
    """
    rng = random.Random(7)
    fill_ms: float | None = None
    with tempfile.TemporaryDirectory(prefix="ingatan-bench-", ignore_cleanup_errors=True) as home:
        scope = Scope(business="tokoku", agent="asisten", home=Path(home), embeddings=cfg)
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
            if cfg.defer:
                # Not timed: the store that is searched and written to below has its vectors.
                backfill.fill_pending(ctx)
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
            if cfg.defer:
                start = time.perf_counter()
                report = backfill.fill_pending(ctx)
                if report.embedded:
                    fill_ms = (time.perf_counter() - start) * 1000 / report.embedded
        finally:
            ctx.close()
            embed.unload_local_models()
    return (
        size + CHECKED,
        statistics.median(remember_ms),
        statistics.median(recall_ms),
        fill_ms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sizes", nargs="*", type=int, metavar="SIZE")
    parser.add_argument("--embeddings", choices=("none", "local"), default="none")
    parser.add_argument(
        "--defer", action="store_true", help="with --embeddings local: fill vectors afterwards"
    )
    parser.add_argument("--model-dir", default="", help="model folder with --embeddings local")
    args = parser.parse_args(argv)
    cfg = embedding_config(args.embeddings, args.defer, args.model_dir)
    label = "none (FTS5 only)" if cfg.provider == "none" else f"local ({cfg.model})"
    print(f"embeddings: {label}{', deferred' if cfg.defer else ''}")
    header = "facts | remember ms (median) | recall ms (median)"
    print(header + (" | fill ms per fact" if cfg.defer else ""))
    for size in args.sizes or [1000, 4000, 16000]:
        stored, remember_ms, recall_ms, fill_ms = measure(size, cfg)
        line = f"{stored:5d} | {remember_ms:8.1f} | {recall_ms:8.1f}"
        if fill_ms is not None:
            line += f" | {fill_ms:8.1f}"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
