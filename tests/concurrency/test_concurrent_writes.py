"""NFR-12: two processes writing to one business file lose nothing and corrupt nothing."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from ingatan.scope import Scope, db_path, open_context

WRITER = Path(__file__).with_name("_writer.py")
OPENER = Path(__file__).with_name("_opener.py")
PER_PROCESS = 150


def test_two_processes_write_300_facts_without_loss(ingatan_home: Path) -> None:
    # The files exist before the writers start, as they do after `ingatan init`. The race
    # between two first opens is a separate matter, see the test below.
    open_context(Scope(business="tokoku", agent="setup", home=ingatan_home)).close()
    start = time.time() + 3.0  # both children are ready (imports done) by then
    env = {**os.environ, "INGATAN_HOME": str(ingatan_home)}
    children = [
        subprocess.Popen(
            [
                sys.executable,
                str(WRITER),
                str(ingatan_home),
                "tokoku",
                agent,
                str(PER_PROCESS),
                str(start),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        for agent in ("agent-a", "agent-b")
    ]
    outputs = [child.communicate(timeout=240) for child in children]
    for child, (_out, err) in zip(children, outputs, strict=True):
        assert child.returncode == 0, err.decode("utf-8", "replace")

    conn = sqlite3.connect(db_path(ingatan_home, "tokoku"))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2 * PER_PROCESS
        assert (
            conn.execute("SELECT count(*) FROM log WHERE op = 'create'").fetchone()[0]
            == 2 * PER_PROCESS
        )
        per_agent = dict(conn.execute("SELECT agent, count(*) FROM facts GROUP BY agent"))
        assert per_agent == {"agent-a": PER_PROCESS, "agent-b": PER_PROCESS}
        # The full-text indexes stayed in step with the table.
        assert conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == 2 * PER_PROCESS
        assert conn.execute("SELECT count(*) FROM facts_tri").fetchone()[0] == 2 * PER_PROCESS
    finally:
        conn.close()


def test_four_processes_opening_a_fresh_home_at_once(tmp_path: Path) -> None:
    failures: list[str] = []
    for round_number in range(2):
        home = tmp_path / f"fresh-{round_number}"
        start = time.time() + 1.5
        env = {**os.environ, "INGATAN_HOME": str(home)}
        children = [
            subprocess.Popen(
                [sys.executable, str(OPENER), str(home), "tokoku", f"agent-{n}", str(start)],
                stderr=subprocess.PIPE,
                env=env,
            )
            for n in range(4)
        ]
        for child in children:
            _out, err = child.communicate(timeout=120)
            if child.returncode != 0:
                failures.append(err.decode("utf-8", "replace").strip())
    assert not failures, failures
