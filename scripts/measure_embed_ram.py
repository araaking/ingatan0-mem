"""Measure what the local embedding model costs in memory, one fresh process per model.

    python scripts/measure_embed_ram.py [--model-dir DIR] [--model NAME ...] [--json OUT]

For the FTS-only baseline and for each local model that has been downloaded, a child
process walks through what a server does (import, load, one query, a batch of facts, a
batch of long notes, unload) and records its own resident memory after each step. The
numbers in docs/RAM.md come from this script and from scripts/measure_ram.py.

Resident memory is the working set on Windows and VmRSS on Linux. "private" is the commit
charge on Windows (memory the process owns even when it is paged out); it is not reported
elsewhere. Nothing here opens an Ingatan data folder or the network.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# One string is easier to read and edit here than fifty one-word lines.
WORDS = (  # noqa: SIM905
    "pelanggan kopi susu roti bakar harga stok supplier gula teh toko cabang kasir "
    "pembayaran tunai transfer promo diskon pengiriman gudang pesanan laporan "
    "faktur retur ongkir kurir jadwal libur karyawan gaji lembur sewa listrik pajak "
    "margin target penjualan iklan konten artikel halaman tautan audit server "
    "domain cadangan"
).split()


def _sentence(seed: int, words: int) -> str:
    return " ".join(WORDS[(seed * 7 + i * 13) % len(WORDS)] for i in range(words))


QUERIES = [_sentence(n, 4) for n in range(21)]
FACTS = [f"Catatan {n}: {_sentence(n, 22)} sejak minggu {n % 52}." for n in range(32)]
# Long enough to reach the model's 512-token limit, like a section of an ingested document.
NOTES = [f"Bagian {n}. " + ". ".join(_sentence(n + k, 30) for k in range(14)) for n in range(32)]


def memory() -> dict[str, float | None]:
    """Resident, peak resident and private memory of the current process, in MB."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(f"K32GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
        mb = 1024 * 1024
        return {
            "rss": round(counters.WorkingSetSize / mb, 1),
            "peak": round(counters.PeakWorkingSetSize / mb, 1),
            "private": round(counters.PrivateUsage / mb, 1),
        }
    if sys.platform.startswith("linux"):
        values: dict[str, float] = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                values[key] = round(int(rest.split()[0]) / 1024, 1)
        return {"rss": values.get("VmRSS"), "peak": values.get("VmHWM"), "private": None}
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    rss = round(int(out.stdout.strip()) / 1024, 1)
    return {"rss": rss, "peak": rss, "private": None}


def run_child(model_name: str, model_dir: str) -> dict[str, Any]:
    """The stages of one setup. `model_name` empty means the FTS-only baseline."""
    from ingatan.search.local import limit_math_threads

    limit_math_threads()  # what `ingatan` itself does first, at every entry point
    stages: list[dict[str, Any]] = []

    def mark(stage: str, **extra: Any) -> None:
        stages.append({"stage": stage, **memory(), **extra})

    mark("python started")
    import ingatan.facts
    import ingatan.search  # noqa: F401
    from ingatan.search import embed, local

    heavy = [m for m in ("numpy", "onnxruntime", "sentencepiece") if m in sys.modules]
    mark("ingatan imported", heavy_modules_loaded=heavy)
    if not model_name:
        return {"stages": stages}

    def vectors(texts: list[str], *, is_query: bool) -> int:
        prepared = [embed._apply_model_prefix(model_name, t, is_query=is_query) for t in texts]  # pyright: ignore[reportPrivateUsage]
        found = local.embed(
            prepared, model_name=model_name, model_dir=model_dir, idle_unload_seconds=0
        )
        return len(found)

    started = time.perf_counter()
    vectors(QUERIES[:1], is_query=True)
    mark("model loaded, first query", seconds=round(time.perf_counter() - started, 2))
    started = time.perf_counter()
    for query in QUERIES[1:]:
        vectors([query], is_query=True)
    each = (time.perf_counter() - started) * 1000 / (len(QUERIES) - 1)
    mark("20 warm queries", ms_each=round(each, 1))
    started = time.perf_counter()
    vectors(FACTS, is_query=False)
    mark("32 facts embedded", seconds=round(time.perf_counter() - started, 2))
    started = time.perf_counter()
    vectors(NOTES, is_query=False)
    mark("32 long notes embedded", seconds=round(time.perf_counter() - started, 2))
    local.unload()
    mark("model unloaded")
    return {"stages": stages}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model-dir", help="model folder (default: the configured one)")
    parser.add_argument("--model", action="append", help="measure only this local model")
    parser.add_argument("--json", dest="json_out", help="also write the raw numbers here")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    from ingatan.scope import load_embeddings
    from ingatan.search import local

    model_dir = str(
        Path(args.model_dir or load_embeddings().model_dir or local.default_model_dir()).resolve()
    )
    if args.child is not None:
        print("RESULT=" + json.dumps(run_child(args.child, model_dir)), flush=True)
        return 0

    print(
        f"{platform.system()} {platform.release()} ({platform.machine()}), "
        f"Python {platform.python_version()}, {os.cpu_count()} logical CPUs"
    )
    print(f"model folder: {model_dir}")
    names = [local.resolve(n).name for n in args.model] if args.model else list(local.LOCAL_MODELS)
    collected: dict[str, Any] = {}
    for name in ["", *names]:
        label = name or "fts-only (no model)"
        print(f"\n== {label}")
        if name and local.missing_packages():
            print(f"   skipped: missing packages ({local.INSTALL_HINT})")
            continue
        if name and local.missing_files(local.resolve(name), model_dir):
            print(f"   skipped: not downloaded (ingatan embeddings pull --model {name})")
            continue
        proc = subprocess.run(
            [sys.executable, __file__, "--child", name, "--model-dir", model_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")), None)
        if proc.returncode != 0 or line is None:
            print(f"   FAILED (exit {proc.returncode})")
            print("   " + "\n   ".join(proc.stderr.strip().splitlines()[-12:]))
            continue
        result = json.loads(line[len("RESULT=") :])
        collected[label] = result
        print(f"   {'stage':28s} {'rss MB':>8s} {'peak MB':>8s} {'private MB':>11s}  detail")
        for stage in result["stages"]:
            detail = ", ".join(
                f"{key}={value}"
                for key, value in stage.items()
                if key not in ("stage", "rss", "peak", "private")
            )
            private = "-" if stage["private"] is None else f"{stage['private']:.1f}"
            print(
                f"   {stage['stage']:28s} {stage['rss']:8.1f} {stage['peak']:8.1f} "
                f"{private:>11s}  {detail}"
            )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(collected, indent=2), encoding="utf-8")
        print(f"\nraw numbers written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
