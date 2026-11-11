from __future__ import annotations

import itertools
import json
import sqlite3
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ingatan import timeutil
from ingatan.scope import Context, Scope, open_context

DEFAULT_BUSINESS = "tokoku"
DEFAULT_AGENT = "claude-code"
FROZEN_START = datetime(2026, 11, 11, 9, 0, 0, tzinfo=timezone.utc)

MakeContext = Callable[..., Context]


@pytest.fixture(autouse=True)
def ingatan_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own data folder; the real ~/.ingatan is never touched."""
    home = tmp_path / "ingatan-home"
    monkeypatch.setenv("INGATAN_HOME", str(home))
    monkeypatch.delenv("INGATAN_BUSINESS", raising=False)
    monkeypatch.delenv("INGATAN_AGENT", raising=False)
    # Belt and braces: even a bug that falls back to `~` stays inside tmp_path.
    fake_user = tmp_path / "user-home"
    fake_user.mkdir()
    monkeypatch.setenv("HOME", str(fake_user))
    monkeypatch.setenv("USERPROFILE", str(fake_user))
    # And no stray .ingatan.toml is found from the working folder.
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return home


@pytest.fixture
def make_ctx(ingatan_home: Path) -> Iterator[MakeContext]:
    """Open a context in the shared test home for any agent and business.

    Contexts opened for the same business share one file, so two agents (or two
    businesses) can be exercised side by side. All are closed after the test.
    """
    opened: list[Context] = []

    def factory(
        agent: str = DEFAULT_AGENT, business: str = DEFAULT_BUSINESS, token_budget: int = 1000
    ) -> Context:
        ctx = open_context(
            Scope(business=business, agent=agent, home=ingatan_home, token_budget=token_budget)
        )
        opened.append(ctx)
        return ctx

    yield factory
    for ctx in opened:
        ctx.close()


@pytest.fixture
def ctx(make_ctx: MakeContext) -> Context:
    """Business "tokoku", agent "claude-code"."""
    return make_ctx()


class FrozenClock:
    def set(self, when: datetime) -> None:
        timeutil.set_now(when)

    def advance(self, **delta: float) -> None:
        timeutil.set_now(timeutil.now() + timedelta(**delta))


@pytest.fixture
def frozen_clock() -> Iterator[FrozenClock]:
    """Clock frozen at 2026-11-11T09:00:00Z. Always unfrozen afterwards."""
    timeutil.set_now(FROZEN_START)
    try:
        yield FrozenClock()
    finally:
        timeutil.set_now(None)


InsertFact = Callable[..., str]


@pytest.fixture
def insert_fact() -> InsertFact:
    """Insert a fact row with plain SQL (no facts module yet). Returns its id."""
    counter = itertools.count(1)

    def insert(conn: sqlite3.Connection, **fields: object) -> str:
        n = next(counter)
        row: dict[str, object] = {
            "id": f"f_{n:012x}",
            "text": f"fact number {n}",
            "entities": [],
            "keywords": [],
            "tier": "agent",
            "agent": DEFAULT_AGENT,
            "kind": "stable",
            "valid_from": "2026-11-11T09:00:00Z",
            "source": {"agent": DEFAULT_AGENT, "date": "2026-11-11"},
            "created_at": "2026-11-11T09:00:00Z",
        }
        row.update(fields)
        for key in ("entities", "keywords", "source"):
            if not isinstance(row[key], str):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        names = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        conn.execute(f"INSERT INTO facts ({names}) VALUES ({marks})", row)
        return str(row["id"])

    return insert
