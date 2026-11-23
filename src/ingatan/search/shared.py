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
- Every exchange has one deadline, not a limit per read. Whatever goes wrong surfaces as
  `Unavailable`, an `OSError`, which the embedding layer already turns into "no vector":
  the server then answers from the word index, exactly as it does when the model is missing.

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
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingatan.scope import data_home
from ingatan.search import local

__all__ = [
    "HOST",
    "Refused",
    "SharedStatus",
    "Unavailable",
    "embed",
    "request",
    "serve",
    "state_file",
    "status",
    "stop",
]

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
# `timeout` is how long a caller waits for the process to be there and to shake hands. How
# long the model may take is a different question: it grows with the number of texts, and
# the first request also loads the model. So the answer gets this much per text on top.
ANSWER_SECONDS_PER_TEXT = 5.0
# A caller that finds no process starts one, but not more often than this: when the process
# cannot start at all, every tool call would otherwise launch another.
SPAWN_INTERVAL_SECONDS = 30.0


class Unavailable(OSError):
    """The shared process cannot be reached, cannot be trusted, or could not embed."""


class Refused(Unavailable):
    """The process answered, and the answer was an error (a missing model file, say).

    Starting another process would not help, so the caller does not try.
    """


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


def state_file() -> Path:
    """Where the process of this data folder records its pid, port and token."""
    return data_home() / STATE_DIR / STATE_NAME


def _state_path() -> Path:
    return state_file()


def _state_text() -> str:
    return _state_path().read_text(encoding="utf-8")


def _parse_state(text: str) -> _State | None:
    try:
        raw: Any = json.loads(text)
        if not isinstance(raw, dict) or raw.get("v") != PROTOCOL:
            return None
        port, pid = raw.get("port"), raw.get("pid")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        token = bytes.fromhex(str(raw.get("token")))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except ValueError:
        return None
    if not isinstance(port, int) or not isinstance(pid, int) or isinstance(port, bool):
        return None
    if not 0 < port < 65536 or len(token) != _NONCE_BYTES:
        return None
    return _State(pid=pid, port=port, token=token)


def _read_state() -> _State | None:
    try:
        return _parse_state(_state_text())
    except OSError:
        return None


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


def _ownership(token: bytes) -> bool | None:
    """Whether the state file names the process that holds `token`.

    True or False when the file could be read (a missing file is False: it was taken away).
    None when it could not be read just now: on Windows a file that is being replaced or
    scanned refuses a read for a moment, and that is no reason for the process to leave.
    """
    try:
        text = _state_text()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    current = _parse_state(text)
    return current is not None and hmac.compare_digest(current.token, token)


# ---- frames and the handshake ---------------------------------------------------------------


class _Deadline:
    """A moment by which a whole exchange must be over.

    A socket timeout alone bounds each read: a peer that sends one byte just in time, again
    and again, keeps the other side waiting as long as it likes. Every send and every read
    of an exchange takes what is left until its deadline instead.
    """

    def __init__(self, seconds: float) -> None:
        self._at = time.monotonic() + seconds

    def apply(self, sock: socket.socket) -> None:
        left = self._at - time.monotonic()
        if left <= 0:
            raise TimeoutError("the exchange took longer than its time limit")
        sock.settimeout(left)


def _send(sock: socket.socket, payload: bytes, deadline: _Deadline) -> None:
    deadline.apply(sock)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _receive_exactly(sock: socket.socket, count: int, deadline: _Deadline) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        deadline.apply(sock)
        chunk = sock.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ConnectionError("the other side closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive(sock: socket.socket, limit: int, deadline: _Deadline) -> bytes:
    (length,) = struct.unpack(">I", _receive_exactly(sock, 4, deadline))
    if length > limit:
        raise ValueError(f"a frame of {length} bytes is over the limit of {limit}")
    return _receive_exactly(sock, length, deadline)


def _proof(token: bytes, role: bytes, first: bytes, second: bytes) -> bytes:
    return hmac.new(token, role + first + second, hashlib.sha256).digest()


def _connect(state: _State, timeout: float) -> socket.socket:
    """A connection to a listener that has proved it holds the token. No text sent yet."""
    deadline = _Deadline(timeout)  # the connection and the whole handshake together
    sock = socket.create_connection((HOST, state.port), timeout=timeout)
    try:
        mine = secrets.token_bytes(_NONCE_BYTES)
        _send(sock, mine, deadline)
        answer = _receive(sock, _NONCE_BYTES + _PROOF_BYTES, deadline)
        theirs, proof = answer[:_NONCE_BYTES], answer[_NONCE_BYTES:]
        if len(answer) != _NONCE_BYTES + _PROOF_BYTES or not hmac.compare_digest(
            proof, _proof(state.token, b"server", mine, theirs)
        ):
            raise Unavailable("the listener on the recorded port is not the shared process")
        _send(sock, _proof(state.token, b"client", theirs, mine), deadline)
    except BaseException:
        sock.close()
        raise
    return sock


def _call(
    state: _State,
    payload: dict[str, Any],
    *,
    timeout: float,
    vectors: bool = False,
    answer_timeout: float | None = None,
) -> tuple[dict[str, Any], bytes]:
    """One request and its answer. Every failure becomes `Unavailable`.

    `timeout` bounds the connection and the handshake together; `answer_timeout` (default:
    the same) bounds sending the request and receiving the whole answer. Each is one
    deadline, not a limit per read.
    """
    try:
        with _connect(state, timeout) as sock:
            deadline = _Deadline(timeout if answer_timeout is None else answer_timeout)
            _send(sock, json.dumps(payload, ensure_ascii=False).encode("utf-8"), deadline)
            header: Any = json.loads(_receive(sock, MAX_HEADER_BYTES, deadline))
            if not isinstance(header, dict):
                raise ValueError("the answer is not a JSON object")
            if not header.get("ok"):  # pyright: ignore[reportUnknownMemberType]
                raise Refused(str(header.get("error") or "the shared process refused"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            blob = _receive(sock, MAX_VECTOR_BYTES, deadline) if vectors else b""
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
        answer_timeout=timeout + ANSWER_SECONDS_PER_TEXT * len(texts),
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


_spawn_lock = threading.Lock()
_last_spawn = float("-inf")


def _spawn(idle_unload_seconds: int) -> None:
    """Start the process, detached from this one, with nothing connected to its streams.

    A stdio MCP server speaks its protocol on stdout, so the child must never inherit it.
    """
    options: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        options["start_new_session"] = True
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ingatan.cli",
            "embeddings",
            "serve",
            "--idle-seconds",
            str(idle_unload_seconds),
        ],
        **options,
    )
    # Collect its exit status when it leaves, so it never lingers as a defunct process.
    threading.Thread(target=child.wait, name="ingatan-shared-reaper", daemon=True).start()


def _process_answered() -> None:
    """The process was there and answered. The spawn limit is for a process that cannot
    start; one that came up and later left on idleness is started again without waiting."""
    global _last_spawn
    with _spawn_lock:
        _last_spawn = float("-inf")


def _start_process(idle_unload_seconds: int) -> bool:
    """Start the process unless this caller tried a moment ago. True if it was started."""
    global _last_spawn
    with _spawn_lock:
        now = time.monotonic()
        if now - _last_spawn < SPAWN_INTERVAL_SECONDS:
            return False
        _last_spawn = now
    try:
        _spawn(idle_unload_seconds)
    except OSError:
        return False
    return True


def embed(
    texts: Sequence[str],
    *,
    model_name: str,
    model_dir: Path | str,
    threads: int,
    idle_unload_seconds: int,
    timeout: float,
) -> list[list[float]]:
    """Vectors for `texts` from the shared process, starting it when it is not there.

    One attempt to start it, at most once per `SPAWN_INTERVAL_SECONDS` in this process, a
    wait of at most `timeout` seconds for it to answer, then one retry. Raises `Unavailable`
    when that is not enough; the model is never loaded into the calling process instead.
    """
    try:
        vectors = request(
            texts, model_name=model_name, model_dir=model_dir, threads=threads, timeout=timeout
        )
        _process_answered()
        return vectors
    except Refused:
        raise
    except Unavailable:
        # A process that is there and answers a ping did not fail for want of being started:
        # it is slow, or busy with someone else's texts. Starting another, or sending the
        # same texts again behind the first, would only add to its queue.
        if status() is not None or not _start_process(idle_unload_seconds):
            raise
    deadline = time.monotonic() + timeout
    while status() is None:
        if time.monotonic() >= deadline:
            raise Unavailable("the shared embedding process was started but did not answer")
        time.sleep(0.05)
    vectors = request(
        texts, model_name=model_name, model_dir=model_dir, threads=threads, timeout=timeout
    )
    _process_answered()
    return vectors


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
                handshake = _Deadline(_HANDSHAKE_SECONDS)
                theirs = _receive(conn, _NONCE_BYTES, handshake)
                mine = secrets.token_bytes(_NONCE_BYTES)
                _send(conn, mine + _proof(self.token, b"server", theirs, mine), handshake)
                proof = _receive(conn, _PROOF_BYTES, handshake)
                if not hmac.compare_digest(proof, _proof(self.token, b"client", mine, theirs)):
                    return  # a caller without the token gets no answer
                message: Any = json.loads(
                    _receive(conn, MAX_REQUEST_BYTES, _Deadline(_REQUEST_SECONDS))
                )
                if isinstance(message, dict):
                    self._answer(conn, message)  # pyright: ignore[reportUnknownArgumentType]
            except (OSError, ValueError):
                return

    def _answer(self, conn: socket.socket, message: dict[str, Any]) -> None:
        def header(**fields: Any) -> None:
            _send(conn, json.dumps(fields).encode("utf-8"), _Deadline(_REQUEST_SECONDS))

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
            _send(conn, flat.tobytes(), _Deadline(_REQUEST_SECONDS))
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
                    if _ownership(token) is False:
                        break
                continue
            except OSError:
                break
            threading.Thread(target=server.handle, args=(conn,), daemon=True).start()
    finally:
        listener.close()
        if _ownership(token):
            with contextlib.suppress(OSError):
                _state_path().unlink()
        local.unload()
