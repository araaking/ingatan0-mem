"""One process holds the embedding model; every Ingatan server asks it for vectors.

Each stdio server otherwise loads its own copy of the model. With `[embeddings] shared`,
the servers of one data folder share a single process instead: `serve` is that process,
`request` is how a server asks it.

How they find and trust each other:

- The process listens on 127.0.0.1 only, on a port the operating system picks, and writes
  `<INGATAN_HOME>/run/embed.json` (pid, port and a random token), readable by the user
  alone. The file never holds fact text, and nothing else is written to disk.
- A caller always connects to 127.0.0.1, whatever the file says, so text cannot be sent off
  the machine. Before it sends any text it makes the listener prove that it holds the token
  (an HMAC over fresh random bytes from both sides), and proves the same in return. A port
  that another program took over after a crash therefore receives a few random bytes and
  nothing else. The token itself never travels.
- Every wait has a time limit. Whatever goes wrong surfaces as `Unavailable`, an `OSError`,
  which the embedding layer already turns into "no vector": the server then answers from
  the word index, exactly as it does when the model is missing.

Frames are a 4-byte big-endian length and that many bytes. A request is one JSON frame; the
answer is one JSON frame (`ok`, and `error` or `dims` and `count`) followed, for vectors, by
one frame of little-endian float32. Texts arrive already prefixed for their model, so the
process only turns texts into vectors. It loads the model on the first request, never
downloads one, and leaves after `idle_unload_seconds` without a request.
"""

from __future__ import annotations

import array
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingatan.scope import data_home
from ingatan.search import local

__all__ = ["HOST", "SharedStatus", "Unavailable", "request", "serve", "status", "stop"]

HOST = "127.0.0.1"
STATE_DIR = "run"
STATE_NAME = "embed.json"
PROTOCOL = 1
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_VECTOR_BYTES = 256 * 1024 * 1024
_NONCE_BYTES = 32
_PROOF_BYTES = 32
_HANDSHAKE_SECONDS = 5.0
_REQUEST_SECONDS = 60.0
_CONTROL_SECONDS = 2.0
_TICK_SECONDS = 0.25
_OWNERSHIP_CHECK_SECONDS = 1.0


class Unavailable(OSError):
    """The shared process cannot be reached, cannot be trusted, or could not embed."""


@dataclass(frozen=True)
class SharedStatus:
    pid: int
    port: int


@dataclass(frozen=True)
class _State:
    pid: int
    port: int
    token: bytes


# ---- the state file -------------------------------------------------------------------------


def _state_path() -> Path:
    return data_home() / STATE_DIR / STATE_NAME


def _read_state() -> _State | None:
    try:
        raw: Any = json.loads(_state_path().read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("v") != PROTOCOL:
            return None
        port, pid = raw.get("port"), raw.get("pid")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        token = bytes.fromhex(str(raw.get("token")))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except (OSError, ValueError):
        return None
    if not isinstance(port, int) or not isinstance(pid, int) or isinstance(port, bool):
        return None
    if not 0 < port < 65536 or len(token) != _NONCE_BYTES:
        return None
    return _State(pid=pid, port=port, token=token)


def _write_state(path: Path, state: _State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {"v": PROTOCOL, "pid": state.pid, "port": state.port, "token": state.token.hex()},
            handle,
        )
    os.replace(temporary, path)


def _state_is_mine(token: bytes) -> bool:
    current = _read_state()
    return current is not None and hmac.compare_digest(current.token, token)


# ---- frames and the handshake ---------------------------------------------------------------


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _receive_exactly(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = sock.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ConnectionError("the other side closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive(sock: socket.socket, limit: int) -> bytes:
    (length,) = struct.unpack(">I", _receive_exactly(sock, 4))
    if length > limit:
        raise ValueError(f"a frame of {length} bytes is over the limit of {limit}")
    return _receive_exactly(sock, length)


def _proof(token: bytes, role: bytes, first: bytes, second: bytes) -> bytes:
    return hmac.new(token, role + first + second, hashlib.sha256).digest()


def _connect(state: _State, timeout: float) -> socket.socket:
    """A connection to a listener that has proved it holds the token. No text sent yet."""
    sock = socket.create_connection((HOST, state.port), timeout=timeout)
    try:
        sock.settimeout(timeout)
        mine = secrets.token_bytes(_NONCE_BYTES)
        _send(sock, mine)
        answer = _receive(sock, _NONCE_BYTES + _PROOF_BYTES)
        theirs, proof = answer[:_NONCE_BYTES], answer[_NONCE_BYTES:]
        if len(answer) != _NONCE_BYTES + _PROOF_BYTES or not hmac.compare_digest(
            proof, _proof(state.token, b"server", mine, theirs)
        ):
            raise Unavailable("the listener on the recorded port is not the shared process")
        _send(sock, _proof(state.token, b"client", theirs, mine))
    except BaseException:
        sock.close()
        raise
    return sock


def _call(
    state: _State, payload: dict[str, Any], *, timeout: float, vectors: bool = False
) -> tuple[dict[str, Any], bytes]:
    """One request and its answer. Every failure becomes `Unavailable`."""
    try:
        with _connect(state, timeout) as sock:
            _send(sock, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            header: Any = json.loads(_receive(sock, MAX_HEADER_BYTES))
            if not isinstance(header, dict):
                raise ValueError("the answer is not a JSON object")
            if not header.get("ok"):  # pyright: ignore[reportUnknownMemberType]
                raise Unavailable(str(header.get("error") or "the shared process refused"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            blob = _receive(sock, MAX_VECTOR_BYTES) if vectors else b""
    except Unavailable:
        raise
    except (OSError, ValueError) as exc:
        raise Unavailable(f"the shared embedding process did not answer: {exc}") from exc
    return header, blob  # pyright: ignore[reportUnknownVariableType]


# ---- the caller's side ----------------------------------------------------------------------


def request(
    texts: Sequence[str],
    *,
    model_name: str,
    model_dir: Path | str,
    threads: int,
    timeout: float,
) -> list[list[float]]:
    """Vectors for `texts`, in order, from the running process. Never starts one.

    Raises `Unavailable` when no process runs, when it cannot be trusted, when it does not
    answer within `timeout` seconds, or when the model failed there (the message says why).
    """
    if not texts:
        return []
    state = _read_state()
    if state is None:
        raise Unavailable(f"the shared embedding process is not running (no {_state_path()})")
    header, blob = _call(
        state,
        {
            "op": "embed",
            "model": model_name,
            "model_dir": str(model_dir),
            "threads": threads,
            "texts": list(texts),
        },
        timeout=timeout,
        vectors=True,
    )
    dims, count = header.get("dims"), header.get("count")
    if (
        not isinstance(dims, int)
        or not isinstance(count, int)
        or count != len(texts)
        or len(blob) != dims * count * 4
    ):
        raise Unavailable("the shared embedding process sent vectors of the wrong shape")
    flat = array.array("f")
    flat.frombytes(blob)
    if sys.byteorder == "big":
        flat.byteswap()
    return [list(flat[index * dims : (index + 1) * dims]) for index in range(count)]


def status() -> SharedStatus | None:
    """The process that answers for this data folder, or None when none does."""
    state = _read_state()
    if state is None:
        return None
    try:
        header, _ = _call(state, {"op": "ping"}, timeout=_CONTROL_SECONDS)
    except Unavailable:
        return None
    pid = header.get("pid")
    return SharedStatus(pid=pid if isinstance(pid, int) else state.pid, port=state.port)


def stop() -> bool:
    """Ask the running process to leave. False when none answered."""
    state = _read_state()
    if state is None:
        return False
    try:
        _call(state, {"op": "stop"}, timeout=_CONTROL_SECONDS)
    except Unavailable:
        return False
    return True


# ---- the process ------------------------------------------------------------------------------


def _listen() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((HOST, 0))
        listener.listen(16)
    except BaseException:
        listener.close()
        raise
    return listener


class _Server:
    def __init__(self, token: bytes) -> None:
        self.token = token
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()  # one forward pass at a time
        self._busy = 0
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        """How long nothing has been embedded; 0 while a request is being answered."""
        with self._lock:
            return 0.0 if self._busy else time.monotonic() - self._last_used

    def handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.settimeout(_HANDSHAKE_SECONDS)
                theirs = _receive(conn, _NONCE_BYTES)
                mine = secrets.token_bytes(_NONCE_BYTES)
                _send(conn, mine + _proof(self.token, b"server", theirs, mine))
                proof = _receive(conn, _PROOF_BYTES)
                if not hmac.compare_digest(proof, _proof(self.token, b"client", mine, theirs)):
                    return  # a caller without the token gets no answer
                conn.settimeout(_REQUEST_SECONDS)
                message: Any = json.loads(_receive(conn, MAX_REQUEST_BYTES))
                if isinstance(message, dict):
                    self._answer(conn, message)  # pyright: ignore[reportUnknownArgumentType]
            except (OSError, ValueError):
                return

    def _answer(self, conn: socket.socket, message: dict[str, Any]) -> None:
        def header(**fields: Any) -> None:
            _send(conn, json.dumps(fields).encode("utf-8"))

        op = message.get("op")
        if op == "ping":
            header(ok=True, pid=os.getpid())
        elif op == "stop":
            header(ok=True)
            self.stop.set()
        elif op == "embed":
            texts = message.get("texts")
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):  # pyright: ignore[reportUnknownVariableType]
                header(ok=False, error="`texts` must be a list of strings")
                return
            with self._lock:
                self._busy += 1
            try:
                with self._model_lock:
                    vectors = local.embed(
                        [str(t) for t in texts],  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                        model_name=str(message.get("model") or ""),
                        model_dir=str(message.get("model_dir") or ""),
                        threads=int(message.get("threads") or 2),
                        # The process leaves as a whole when idle; until then the model stays.
                        idle_unload_seconds=0,
                    )
            except Exception as exc:  # the caller falls back to the word index
                header(ok=False, error=f"{type(exc).__name__}: {exc}")
                return
            finally:
                with self._lock:
                    self._busy -= 1
                    self._last_used = time.monotonic()
            dims = len(vectors[0]) if vectors else 0
            if any(len(vector) != dims for vector in vectors):
                header(ok=False, error="the model returned vectors of different lengths")
                return
            flat = array.array("f")
            for vector in vectors:
                flat.extend(vector)
            if sys.byteorder == "big":
                flat.byteswap()
            header(ok=True, dims=dims, count=len(vectors))
            _send(conn, flat.tobytes())
        else:
            header(ok=False, error=f"unknown request {op!r}")


def serve(*, idle_unload_seconds: int) -> None:
    """Be the shared process for this data folder until idle, stopped or replaced.

    Returns at once when another process already answers. `idle_unload_seconds` of 0 means
    never leave on idleness. The process also leaves when its state file is deleted or
    names another process, which settles two that started at the same moment.
    """
    if status() is not None:
        return
    token = secrets.token_bytes(_NONCE_BYTES)
    listener = _listen()
    server = _Server(token)
    try:
        _write_state(
            _state_path(), _State(pid=os.getpid(), port=listener.getsockname()[1], token=token)
        )
        listener.settimeout(_TICK_SECONDS)
        checked = time.monotonic()
        while not server.stop.is_set():
            try:
                conn, _address = listener.accept()
            except TimeoutError:
                now = time.monotonic()
                if idle_unload_seconds > 0 and server.idle_seconds() >= idle_unload_seconds:
                    break
                if now - checked >= _OWNERSHIP_CHECK_SECONDS:
                    checked = now
                    if not _state_is_mine(token):
                        break
                continue
            except OSError:
                break
            threading.Thread(target=server.handle, args=(conn,), daemon=True).start()
    finally:
        listener.close()
        if _state_is_mine(token):
            with contextlib.suppress(OSError):
                _state_path().unlink()
        local.unload()
