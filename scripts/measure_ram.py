"""Measure the resident memory (RSS) of one Ingatan stdio server process (NFR-4, T-16).

Starts one `ingatan.mcp.server` subprocess in a temporary data folder, talks to it with
plain JSON-RPC over its stdin and stdout, and reads its resident set size three times:
idle (after the handshake), after N facts were saved through the `remember` tool and a
few `recall` and `brief` calls, and the peak the operating system recorded.

Standard library only. Usage:

    python scripts/measure_ram.py [--facts N] [--no-write]

It prints the numbers and appends a dated row to docs/RAM.md (unless --no-write).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RAM_DOC = REPO / "docs" / "RAM.md"
DOC_HEADER = """# Memory use of one stdio server process (NFR-4)

Resident set size (RSS) of one `ingatan.mcp.server` process, measured by
`scripts/measure_ram.py`. "Loaded" means N facts saved through the `remember` tool, then a few
`recall` and `brief` calls. "Peak" is the highest value the operating system recorded.
The threshold is OQ-19 and is the owner's call; these are measurements only.

| Date | Machine | Python | mcp | Facts | Idle MB | Loaded MB | Peak MB | Command |
|---|---|---|---|---|---|---|---|---|
"""
PID_MARK = "INGATAN_MEASURE_PID="
# Prints the real interpreter's pid on stderr first. On Windows a virtualenv's python.exe can
# be a launcher that starts the real interpreter as a child, so Popen.pid would be wrong.
BOOTSTRAP = (
    "import os, sys; "
    f"sys.stderr.write('{PID_MARK}' + str(os.getpid()) + chr(10)); sys.stderr.flush(); "
    "from ingatan.mcp.server import run; run()"
)

WORDS = [
    "pelanggan",
    "kopi",
    "susu",
    "roti",
    "bakar",
    "harga",
    "stok",
    "supplier",
    "gula",
    "teh",
    "toko",
    "cabang",
    "kasir",
    "pembayaran",
    "tunai",
    "transfer",
    "promo",
    "diskon",
    "pengiriman",
    "gudang",
    "pesanan",
    "laporan",
]


# ----------------------------------------------------------------- RSS readers


def rss_bytes(pid: int) -> tuple[int, int | None]:
    """(current RSS, peak RSS or None) of a process, in bytes."""
    if sys.platform == "win32":
        return _rss_windows(pid)
    if sys.platform.startswith("linux"):
        return _rss_linux(pid)
    return _rss_ps(pid), None


def _rss_linux(pid: int) -> tuple[int, int | None]:
    values: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        match = re.match(r"(VmRSS|VmHWM):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values["VmRSS"], values.get("VmHWM")


def _rss_ps(pid: int) -> int:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) * 1024


def _rss_windows(pid: int) -> tuple[int, int | None]:
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
        ]

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    query_limited_and_vm_read = 0x1000 | 0x0010  # PROCESS_QUERY_LIMITED_INFORMATION | VM_READ
    handle = kernel32.OpenProcess(query_limited_and_vm_read, False, pid)
    if not handle:
        raise OSError(f"OpenProcess failed for pid {pid}: {ctypes.get_last_error()}")
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(f"K32GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


# ------------------------------------------------------------- JSON-RPC client


class Server:
    """One server subprocess spoken to over newline-delimited JSON-RPC."""

    def __init__(self, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", BOOTSTRAP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=env["INGATAN_HOME"],
        )
        assert self.proc.stdin and self.proc.stdout and self.proc.stderr
        self.lines: queue.Queue[str] = queue.Queue()
        self.stderr_tail: list[str] = []
        self.pid_queue: queue.Queue[int] = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self.pid = self.pid_queue.get(timeout=60)
        self._next_id = 0

    def _pump_stdout(self) -> None:
        assert self.proc.stdout
        for raw in self.proc.stdout:
            self.lines.put(raw.decode("utf-8", "replace"))

    def _pump_stderr(self) -> None:
        assert self.proc.stderr
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith(PID_MARK):
                self.pid_queue.put(int(line[len(PID_MARK) :]))
            else:
                self.stderr_tail = [*self.stderr_tail[-20:], line]

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin
        self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        wanted = self._next_id
        self._send({"jsonrpc": "2.0", "id": wanted, "method": method, "params": params})
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=1)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited: {self.stderr_tail}") from None
                continue
            reply = json.loads(line)
            if reply.get("id") == wanted:
                if "error" in reply:
                    raise RuntimeError(f"{method} failed: {reply['error']}")
                return reply["result"]
        raise TimeoutError(f"no answer to {method}")

    def handshake(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "measure_ram", "version": "0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = self.request("tools/list", {})["tools"]
        assert len(tools) == 7, f"expected seven tools, got {len(tools)}"

    def call(self, tool: str, **arguments: Any) -> str:
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        text = "".join(part.get("text", "") for part in result.get("content", []))
        if result.get("isError"):
            raise RuntimeError(f"{tool} returned an error: {text}")
        return text

    def close(self) -> None:
        with contextlib.suppress(Exception):
            assert self.proc.stdin
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ------------------------------------------------------------------- the run


def fact_text(n: int) -> tuple[str, list[str], list[str]]:
    a, b, c = (
        WORDS[n % len(WORDS)],
        WORDS[(n * 7 + 3) % len(WORDS)],
        WORDS[(n * 13 + 5) % len(WORDS)],
    )
    text = (
        f"Catatan {n}: {a} untuk cabang {n % 12} biasanya dipesan bersama {b} dan {c}, "
        f"harga Rp {10000 + n * 25} per paket sejak minggu {n % 52}."
    )
    return text, [a, b, c, f"cabang {n % 12}"], [f"cabang {n % 12}"]


def megabytes(value: int) -> float:
    return round(value / (1024 * 1024), 1)


def measure(fact_count: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ingatan-ram-") as home:
        env = {
            **os.environ,
            "INGATAN_HOME": home,
            "INGATAN_BUSINESS": "measure",
            "INGATAN_AGENT": "measure-agent",
        }
        server = Server(env)
        try:
            server.handshake()
            idle, _ = rss_bytes(server.pid)
            for n in range(fact_count):
                text, keywords, entities = fact_text(n)
                server.call(
                    "remember", text=text, keywords=keywords, entities=entities, kind="stable"
                )
            for query in ("harga kopi susu", "pelanggan cabang 3", "pembayaran tunai", "stok"):
                server.call("recall", query=query, limit=10)
            server.call("brief")
            loaded, peak = rss_bytes(server.pid)
        finally:
            server.close()
    return {"idle": idle, "loaded": loaded, "peak": peak if peak is not None else loaded}


def machine() -> str:
    system = platform.system()
    if system == "Windows":
        # platform.release() still says "10" on Windows 11; the build number tells them apart.
        build = platform.version().split(".")[-1]
        release = "11" if build.isdigit() and int(build) >= 22000 else platform.release()
        return f"Windows {release} ({platform.machine()})"
    if system == "Linux":
        return f"Linux {platform.release()} ({platform.machine()})"
    return f"{system} {platform.release()} ({platform.machine()})"


def mcp_version() -> str:
    try:
        return metadata.version("mcp")
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--facts", type=int, default=500, help="facts to load (default 500)")
    parser.add_argument("--no-write", action="store_true", help="do not touch docs/RAM.md")
    args = parser.parse_args()

    result = measure(args.facts)
    idle, loaded, peak = (megabytes(result[k]) for k in ("idle", "loaded", "peak"))
    command = "python scripts/measure_ram.py" + (
        "" if args.facts == 500 else f" --facts {args.facts}"
    )
    print(f"idle   : {idle} MB")
    print(f"loaded : {loaded} MB  ({args.facts} facts through the MCP tools)")
    print(f"peak   : {peak} MB")

    if not args.no_write:
        row = (
            f"| {datetime.now(timezone.utc).date().isoformat()} | {machine()} | "
            f"{platform.python_version()} | {mcp_version()} | {args.facts} | "
            f"{idle} | {loaded} | {peak} | `{command}` |\n"
        )
        RAM_DOC.parent.mkdir(parents=True, exist_ok=True)
        if not RAM_DOC.exists():
            RAM_DOC.write_text(DOC_HEADER, encoding="utf-8")
        with RAM_DOC.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(row)
        print(f"appended a row to {RAM_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
