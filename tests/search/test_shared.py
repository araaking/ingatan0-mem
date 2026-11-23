"""The shared model process: one process holds the model, the servers ask it for vectors.

The process is run in a thread of the test and the model is a stand-in, so nothing here
loads a model, downloads anything or leaves the machine.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from ingatan.scope import EmbeddingConfig
from ingatan.search import embed, local, shared

CALL = {"model_name": "", "model_dir": "", "threads": 2, "timeout": 5.0}


def wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """A stand-in for `local.embed`: the vector of a text is `[its length, its word count]`."""
    batches: list[list[str]] = []

    def fake(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        batches.append(list(texts))
        return [[float(len(t)), float(len(t.split()))] for t in texts]

    monkeypatch.setattr(local, "embed", fake)
    return batches


class Process:
    """`shared.serve` running in a thread, as the process would run it."""

    def __init__(self, idle_unload_seconds: int = 0) -> None:
        self.thread = threading.Thread(
            target=shared.serve, kwargs={"idle_unload_seconds": idle_unload_seconds}, daemon=True
        )

    def start(self) -> Process:
        self.thread.start()
        assert wait_for(lambda: shared.status() is not None), "the process did not come up"
        return self

    def stop(self) -> None:
        shared.stop()
        self.thread.join(5)


@pytest.fixture
def process(model: list[list[str]]) -> Iterator[Process]:
    running = Process().start()
    yield running
    running.stop()


def state_file(home: Path) -> Path:
    return home / "run" / "embed.json"


# ---- asking for vectors ------------------------------------------------------------------


def test_round_trip_returns_vectors_in_order(process: Process, model: list[list[str]]) -> None:
    vectors = shared.request(["satu", "dua kata", "tiga kata saja"], **CALL)
    assert vectors == [[4.0, 1.0], [8.0, 2.0], [14.0, 3.0]]
    assert model == [["satu", "dua kata", "tiga kata saja"]]


def test_a_large_batch_survives_the_trip(process: Process) -> None:
    texts = [f"catatan nomor {n} " + "kata " * 400 for n in range(64)]
    vectors = shared.request(texts, **CALL)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]


def test_the_model_settings_reach_the_process(
    process: Process, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake(texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        seen.update(kwargs)
        return [[1.0] for _ in texts]

    monkeypatch.setattr(local, "embed", fake)
    shared.request(["a"], model_name="some/model", model_dir="/models", threads=3, timeout=5.0)
    assert (seen["model_name"], seen["model_dir"], seen["threads"]) == ("some/model", "/models", 3)
    # The process decides for itself when it has been idle: the model stays loaded meanwhile.
    assert seen["idle_unload_seconds"] == 0


def test_status_names_the_process(process: Process) -> None:
    status = shared.status()
    assert status is not None
    assert status.pid == os.getpid()
    assert 0 < status.port < 65536


# ---- when it is not there ------------------------------------------------------------------


def test_request_without_a_process_is_unavailable_and_starts_nothing(
    ingatan_home: Path, model: list[list[str]]
) -> None:
    before = threading.active_count()
    with pytest.raises(shared.Unavailable, match="not running"):
        shared.request(["a"], **CALL)
    assert shared.status() is None
    assert threading.active_count() == before
    assert not state_file(ingatan_home).exists()


def test_a_stale_state_file_is_unavailable(ingatan_home: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    state_file(ingatan_home).parent.mkdir(parents=True)
    state_file(ingatan_home).write_text(
        json.dumps({"v": 1, "pid": 1, "port": free_port, "token": "00" * 32}), encoding="utf-8"
    )
    with pytest.raises(shared.Unavailable):
        shared.request(["a"], **CALL)
    assert shared.status() is None


def test_a_model_error_is_an_error_answer_not_a_crash(
    process: Process, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        raise RuntimeError("could not run the model")

    monkeypatch.setattr(local, "embed", broken)
    with pytest.raises(shared.Unavailable, match="could not run the model"):
        shared.request(["a"], **CALL)
    assert shared.status() is not None  # still serving


def test_missing_model_file_is_an_error_answer_not_a_download(
    process: Process, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_file(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        raise FileNotFoundError(
            "model.onnx is missing. Download the model once with: ingatan embeddings pull"
        )

    def no_download(*_args: Any, **_kwargs: Any) -> list[Path]:
        raise AssertionError("the process must never download a model")

    monkeypatch.setattr(local, "embed", no_file)
    monkeypatch.setattr(local, "pull", no_download)
    with pytest.raises(shared.Unavailable, match="ingatan embeddings pull"):
        shared.request(["a"], **CALL)


# ---- who may talk to it ----------------------------------------------------------------------


def test_binds_loopback_only() -> None:
    listener = shared._listen()  # pyright: ignore[reportPrivateUsage]
    try:
        assert listener.getsockname()[0] == "127.0.0.1"
    finally:
        listener.close()


def test_wrong_token_is_refused(
    process: Process, ingatan_home: Path, model: list[list[str]]
) -> None:
    path = state_file(ingatan_home)
    good = path.read_text(encoding="utf-8")
    state = json.loads(good)
    path.write_text(json.dumps({**state, "token": "ab" * 32}), encoding="utf-8")
    try:
        with pytest.raises(shared.Unavailable):
            shared.request(["rahasia toko"], **CALL)
        assert model == []  # the text never reached the model
    finally:
        path.write_text(good, encoding="utf-8")
    assert shared.request(["a"], **CALL) == [[1.0, 1.0]]


def test_text_is_not_sent_to_a_listener_that_cannot_prove_the_token(ingatan_home: Path) -> None:
    """A stale state file whose port another program took over must not receive fact text."""
    received = bytearray()
    rogue = socket.socket()
    rogue.bind(("127.0.0.1", 0))
    rogue.listen(1)

    def serve_garbage() -> None:
        conn, _ = rogue.accept()
        with conn:
            conn.settimeout(2)
            try:
                received.extend(conn.recv(4096))
                conn.sendall((70).to_bytes(4, "big") + b"x" * 70)  # a "proof" that is wrong
                received.extend(conn.recv(65536))
            except OSError:
                pass

    thread = threading.Thread(target=serve_garbage, daemon=True)
    thread.start()
    state_file(ingatan_home).parent.mkdir(parents=True)
    state_file(ingatan_home).write_text(
        json.dumps({"v": 1, "pid": 1, "port": rogue.getsockname()[1], "token": "cd" * 32}),
        encoding="utf-8",
    )
    try:
        with pytest.raises(shared.Unavailable):
            shared.request(["harga rahasia Rp 15.000"], **CALL)
    finally:
        thread.join(5)
        rogue.close()
    assert b"rahasia" not in bytes(received)
    assert b"cd" * 32 not in bytes(received)  # nor the token itself


# ---- what it leaves on disk ------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_state_file_is_user_only(process: Process, ingatan_home: Path) -> None:
    assert state_file(ingatan_home).stat().st_mode & 0o777 == 0o600


def test_state_file_holds_no_text(process: Process, ingatan_home: Path) -> None:
    shared.request(["ongkir Bandung lima belas ribu"], **CALL)
    on_disk = "".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (ingatan_home / "run").rglob("*")
        if p.is_file()
    )
    assert set(json.loads(state_file(ingatan_home).read_text(encoding="utf-8"))) == {
        "v",
        "pid",
        "port",
        "token",
    }
    for word in ("ongkir", "Bandung", "ribu"):
        assert word not in on_disk


# ---- its lifetime ------------------------------------------------------------------------------


def test_stop_ends_the_process_and_removes_the_state_file(
    ingatan_home: Path, model: list[list[str]]
) -> None:
    running = Process().start()
    assert shared.stop() is True
    running.thread.join(5)
    assert not running.thread.is_alive()
    assert not state_file(ingatan_home).exists()
    assert shared.stop() is False  # nothing left to stop


def test_exits_after_idle(ingatan_home: Path, model: list[list[str]]) -> None:
    running = Process(idle_unload_seconds=1).start()
    shared.request(["a"], **CALL)
    running.thread.join(4)
    assert not running.thread.is_alive()
    assert not state_file(ingatan_home).exists()


def test_a_request_pushes_the_idle_exit_back(ingatan_home: Path, model: list[list[str]]) -> None:
    running = Process(idle_unload_seconds=1).start()
    for _ in range(4):
        time.sleep(0.5)
        shared.request(["a"], **CALL)
    assert running.thread.is_alive()  # two seconds in, kept alive by use
    running.stop()


def test_second_instance_exits(process: Process, model: list[list[str]]) -> None:
    second = threading.Thread(target=shared.serve, kwargs={"idle_unload_seconds": 0}, daemon=True)
    second.start()
    second.join(3)
    assert not second.is_alive()
    assert shared.request(["a"], **CALL) == [[1.0, 1.0]]  # the first one still answers


def test_it_exits_when_its_state_file_is_taken_away(
    ingatan_home: Path, model: list[list[str]]
) -> None:
    running = Process().start()
    state_file(ingatan_home).unlink()
    running.thread.join(5)
    assert not running.thread.is_alive()


# ---- the caller that starts the process when it is not there ---------------------------------

EMBED_CALL = {**CALL, "idle_unload_seconds": 300}


@pytest.fixture
def spawns(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the attempts to start a process, without starting one."""
    calls: list[int] = []
    monkeypatch.setattr(shared, "_spawn", calls.append)
    monkeypatch.setattr(shared, "_last_spawn", float("-inf"))
    return calls


def test_embed_asks_the_running_process(process: Process, spawns: list[int]) -> None:
    assert shared.embed(["dua kata"], **EMBED_CALL) == [[8.0, 2.0]]
    assert spawns == []


def test_embed_starts_the_process_and_retries(
    model: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Process] = []

    def start(idle_unload_seconds: int) -> None:
        running = Process(idle_unload_seconds=0)
        running.thread.start()
        started.append(running)

    monkeypatch.setattr(shared, "_spawn", start)
    monkeypatch.setattr(shared, "_last_spawn", float("-inf"))
    try:
        assert shared.embed(["dua kata"], **EMBED_CALL) == [[8.0, 2.0]]
        assert len(started) == 1
    finally:
        for running in started:
            running.stop()


def test_embed_gives_up_when_the_process_does_not_come_up(spawns: list[int]) -> None:
    started = time.monotonic()
    with pytest.raises(shared.Unavailable):
        shared.embed(["a"], **{**EMBED_CALL, "timeout": 0.5})
    assert spawns == [300]
    assert time.monotonic() - started < 3


def test_spawn_is_rate_limited(spawns: list[int]) -> None:
    for _ in range(2):
        with pytest.raises(shared.Unavailable):
            shared.embed(["a"], **{**EMBED_CALL, "timeout": 0.3})
    assert len(spawns) == 1


def test_a_model_error_does_not_start_another_process(
    process: Process, spawns: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        raise FileNotFoundError("model.onnx is missing")

    monkeypatch.setattr(local, "embed", broken)
    with pytest.raises(shared.Unavailable, match=r"model\.onnx is missing"):
        shared.embed(["a"], **EMBED_CALL)
    assert spawns == []


def test_the_spawn_command_is_the_serve_command_and_stays_off_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            seen["argv"] = argv
            seen.update(kwargs)

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    shared._spawn(120)  # pyright: ignore[reportPrivateUsage]
    assert seen["argv"] == [
        sys.executable,
        "-m",
        "ingatan.cli",
        "embeddings",
        "serve",
        "--idle-seconds",
        "120",
    ]
    # A stdio MCP server speaks its protocol on stdout: the child must never write there.
    assert seen["stdout"] == subprocess.DEVNULL
    assert seen["stderr"] == subprocess.DEVNULL
    assert seen["stdin"] == subprocess.DEVNULL


def test_the_process_really_starts_as_a_subprocess(ingatan_home: Path) -> None:
    shared._spawn(30)  # pyright: ignore[reportPrivateUsage]
    try:
        assert wait_for(lambda: shared.status() is not None, 20)
        status = shared.status()
        assert status is not None and status.pid != os.getpid()
    finally:
        assert shared.stop() is True
    assert wait_for(lambda: not state_file(ingatan_home).exists(), 10)


# ---- the embedding layer picks the process when `shared` is on -----------------------------


def test_embed_uses_the_process_when_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[list[str]] = []

    def through_process(texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        asked.append(list(texts))
        assert kwargs["timeout"] == 15.0 and kwargs["idle_unload_seconds"] == 300
        return [[1.0, 0.0] for _ in texts]

    def in_process(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        raise AssertionError("with `shared` on, this process must not load the model")

    monkeypatch.setattr(shared, "embed", through_process)
    monkeypatch.setattr(local, "embed", in_process)
    cfg = EmbeddingConfig(provider="local", shared=True)
    assert embed.embed_text(cfg, "ongkir Bandung", is_query=False) == [1.0, 0.0]
    assert embed.embed_batch(cfg, ["satu", "dua"]) == [[1.0, 0.0], [1.0, 0.0]]
    # The model's own prefix is added before the text leaves this process.
    assert asked == [["passage: ongkir Bandung"], ["passage: satu", "passage: dua"]]


def test_embed_stays_in_this_process_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def through_process(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        raise AssertionError("without `shared`, nothing may reach for another process")

    def in_process(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(shared, "embed", through_process)
    monkeypatch.setattr(local, "embed", in_process)
    cfg = EmbeddingConfig(provider="local")
    assert embed.embed_text(cfg, "ongkir Bandung", is_query=False) == [0.0, 1.0]


# ---- found by the final review: a long batch, and a slow answer ------------------------------


def test_a_batch_that_takes_longer_than_the_timeout_still_gets_its_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`timeout` says how long to wait for the process, not how long the model may think."""

    def slow(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        time.sleep(0.15 * len(texts))
        return [[float(len(t)), 1.0] for t in texts]

    def no_spawn(idle_unload_seconds: int) -> None:
        raise AssertionError("the process is running: nothing may start another")

    monkeypatch.setattr(local, "embed", slow)
    monkeypatch.setattr(shared, "_spawn", no_spawn)
    running = Process().start()
    try:
        cfg = EmbeddingConfig(provider="local", shared=True, timeout=0.5)
        texts = [f"catatan nomor {n}" for n in range(12)]
        vectors = embed.embed_batch(cfg, texts)
    finally:
        running.stop()
    assert all(vector is not None for vector in vectors), embed.last_embed_error()
    assert len(vectors) == 12


def test_a_slow_answer_is_not_asked_for_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def very_slow(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        calls.append(len(texts))
        time.sleep(1.0)
        return [[1.0, 0.0] for _ in texts]

    spawned: list[int] = []
    monkeypatch.setattr(local, "embed", very_slow)
    monkeypatch.setattr(shared, "_spawn", spawned.append)
    monkeypatch.setattr(shared, "_last_spawn", float("-inf"))
    monkeypatch.setattr(shared, "ANSWER_SECONDS_PER_TEXT", 0.0)
    running = Process().start()
    try:
        with pytest.raises(shared.Unavailable):
            shared.embed(["a"], **{**EMBED_CALL, "timeout": 0.3})
        time.sleep(1.2)  # let the abandoned call finish before the process is stopped
    finally:
        running.stop()
    # The process was alive and working: starting another, or sending the same texts again
    # behind the first, would only add to the queue.
    assert calls == [1]
    assert spawned == []


# ---- deferred review findings: one deadline per call, short idle times, a busy state file ----


def test_a_listener_that_trickles_bytes_cannot_hold_a_caller_past_its_timeout(
    ingatan_home: Path,
) -> None:
    """`timeout` is for the whole exchange, not for each byte that arrives."""
    rogue = socket.socket()
    rogue.bind(("127.0.0.1", 0))
    rogue.listen(1)

    def trickle() -> None:
        conn, _ = rogue.accept()
        with conn:
            try:
                conn.recv(4096)
                conn.sendall((64).to_bytes(4, "big"))
                for _ in range(64):  # 3.2 seconds for one small frame
                    conn.sendall(b"x")
                    time.sleep(0.05)
            except OSError:
                pass

    thread = threading.Thread(target=trickle, daemon=True)
    thread.start()
    state_file(ingatan_home).parent.mkdir(parents=True)
    state_file(ingatan_home).write_text(
        json.dumps({"v": 1, "pid": 1, "port": rogue.getsockname()[1], "token": "cd" * 32}),
        encoding="utf-8",
    )
    started = time.monotonic()
    try:
        with pytest.raises(shared.Unavailable):
            shared.request(["a"], **{**CALL, "timeout": 0.3})
        elapsed = time.monotonic() - started
    finally:
        rogue.close()
        thread.join(5)
    assert elapsed < 1.5, f"the call was held for {elapsed:.1f}s"


def test_the_process_is_started_again_right_after_an_idle_exit(
    model: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that came up and left on idleness is not a process that cannot start."""
    started: list[Process] = []

    def start(idle_unload_seconds: int) -> None:
        running = Process(idle_unload_seconds=1)
        running.thread.start()
        started.append(running)

    monkeypatch.setattr(shared, "_spawn", start)
    monkeypatch.setattr(shared, "_last_spawn", float("-inf"))
    try:
        assert shared.embed(["dua kata"], **EMBED_CALL) == [[8.0, 2.0]]
        # Not `status()` in a loop: every answered ping counts as use and keeps it alive.
        started[0].thread.join(5)
        assert not started[0].thread.is_alive(), "it should leave after a second"
        # Well inside SPAWN_INTERVAL_SECONDS of the first start.
        assert shared.embed(["dua kata"], **EMBED_CALL) == [[8.0, 2.0]]
        assert len(started) == 2
    finally:
        for running in started:
            running.stop()


def test_a_state_file_that_cannot_be_read_for_a_moment_does_not_end_the_process(
    ingatan_home: Path, model: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows a file being replaced or scanned can refuse a read. That is not "gone"."""
    running = Process().start()
    real = shared._state_text  # pyright: ignore[reportPrivateUsage]
    until = time.monotonic() + 1.6

    def busy() -> str:
        if time.monotonic() < until:
            raise PermissionError("the file is in use by another process")
        return real()

    monkeypatch.setattr(shared, "_state_text", busy)
    try:
        time.sleep(1.9)  # the process looks at its state file once a second
        assert running.thread.is_alive()
        assert shared.request(["a"], **CALL) == [[1.0, 1.0]]
    finally:
        running.stop()
