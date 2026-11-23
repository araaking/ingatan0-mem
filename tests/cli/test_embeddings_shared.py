"""`ingatan embeddings serve` and `ingatan embeddings stop`: the shared model process."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

from ingatan.search import local, shared

Invoke = Callable[..., Result]


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(local, "embed", fake)


def test_stop_says_so_when_nothing_runs(invoke: Invoke) -> None:
    result = invoke("embeddings", "stop")
    assert result.exit_code == 0, result.output
    assert "not running" in result.output


def test_serve_runs_until_idle_and_reports_where_it_listened(
    invoke: Invoke, model: None, ingatan_home: Path
) -> None:
    result = invoke("embeddings", "serve", "--idle-seconds", "1")
    assert result.exit_code == 0, result.output
    assert "127.0.0.1" in result.output
    assert not (ingatan_home / "run" / "embed.json").exists()


def test_stop_ends_a_running_process(invoke: Invoke, model: None) -> None:
    runner = threading.Thread(target=lambda: invoke("embeddings", "serve"), daemon=True)
    runner.start()
    deadline = time.monotonic() + 5
    while shared.status() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert shared.status() is not None
    result = invoke("embeddings", "stop")
    assert result.exit_code == 0, result.output
    assert "Stopped" in result.output
    runner.join(5)
    assert not runner.is_alive()


def test_serve_says_so_when_one_already_runs(invoke: Invoke, model: None) -> None:
    first = threading.Thread(target=shared.serve, kwargs={"idle_unload_seconds": 0}, daemon=True)
    first.start()
    deadline = time.monotonic() + 5
    while shared.status() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        result = invoke("embeddings", "serve")
        assert result.exit_code == 0, result.output
        assert "already running" in result.output
    finally:
        shared.stop()
        first.join(5)
