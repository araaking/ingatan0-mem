"""SQLite connections, transactions and numbered migrations (stdlib `sqlite3` only)."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ingatan.errors import SchemaError

__all__ = [
    "GLOBAL_LOG_OFFSET",
    "LATEST_VERSION",
    "Migration",
    "check_capabilities",
    "connect",
    "load_migrations",
    "migrate",
    "transaction",
]

# Log ids are AUTOINCREMENT per file, and an entry can come from the business file or
# from global.db. Ids in global.db start above this offset so a bare log id names
# exactly one entry (needed by `audit.get`, `diff` and `undo`).
GLOBAL_LOG_OFFSET = 1_000_000_000

_MIGRATION_NAME = re.compile(r"^(\d{4})_[A-Za-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


def load_migrations() -> list[Migration]:
    """Read the numbered `.sql` files shipped inside the package, in order."""
    folder = resources.files("ingatan.db").joinpath("migrations")
    found: list[Migration] = []
    for item in folder.iterdir():
        match = _MIGRATION_NAME.match(item.name)
        if match and item.is_file():
            found.append(
                Migration(int(match.group(1)), item.name, item.read_text(encoding="utf-8"))
            )
    found.sort(key=lambda m: m.version)
    versions = [m.version for m in found]
    if versions != list(range(1, len(versions) + 1)):
        raise SchemaError(f"Migration files are not numbered 0001, 0002, ...: {versions}")
    return found


LATEST_VERSION: int = len(load_migrations())

_capabilities_ok = False


def check_capabilities(conn: sqlite3.Connection | None = None) -> None:
    """Raise `SchemaError` if this SQLite build lacks FTS5, the trigram tokenizer,
    FTS5 secure-delete (3.42+) or JSON.

    `conn` exists so tests can probe a fake build; normally it is left out.
    """
    global _capabilities_ok
    if conn is None and _capabilities_ok:
        return
    probe = conn if conn is not None else sqlite3.connect(":memory:")
    try:
        try:
            probe.execute("CREATE VIRTUAL TABLE t_fts USING fts5(x)")
        except sqlite3.Error as exc:
            raise SchemaError(
                "This Python's SQLite was built without FTS5, which Ingatan needs "
                f"(SQLite {sqlite3.sqlite_version}). Use a Python build with FTS5 enabled."
            ) from exc
        try:
            probe.execute("CREATE VIRTUAL TABLE t_tri USING fts5(x, tokenize = 'trigram')")
        except sqlite3.Error as exc:
            raise SchemaError(
                "This Python's SQLite has no FTS5 trigram tokenizer, which Ingatan needs "
                f"(SQLite {sqlite3.sqlite_version}, trigram needs 3.34 or newer). "
                "Use a Python build with a newer SQLite."
            ) from exc
        try:
            probe.execute("INSERT INTO t_fts (t_fts, rank) VALUES ('secure-delete', 1)")
        except sqlite3.Error as exc:
            raise SchemaError(
                "This Python's SQLite does not support the FTS5 secure-delete option, which "
                "Ingatan needs so that a forgotten fact leaves no trace in the search index "
                f"(SQLite {sqlite3.sqlite_version}, secure-delete needs 3.42 or newer). "
                "Use a Python build with a newer SQLite."
            ) from exc
        try:
            probe.execute("SELECT group_concat(value, ' ') FROM json_each('[\"a\"]')").fetchall()
        except sqlite3.Error as exc:
            raise SchemaError(
                "This Python's SQLite was built without the JSON functions Ingatan needs."
            ) from exc
    finally:
        if conn is None:
            probe.close()
    if conn is None:
        _capabilities_ok = True


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _statements(sql: str) -> Iterator[str]:
    """Split a script into complete statements (triggers keep their inner semicolons)."""
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            yield buffer
            buffer = ""
    if buffer.strip() and sqlite3.complete_statement(buffer + ";"):
        yield buffer


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] | None = None) -> list[int]:
    """Apply every migration above `PRAGMA user_version`. Returns the versions applied.

    A database newer than the code raises `SchemaError`. `migrations` defaults to the
    files shipped in the package; tests pass their own list.
    """
    plan = list(load_migrations() if migrations is None else migrations)
    newest = plan[-1].version if plan else 0
    current = _user_version(conn)
    if current > newest:
        raise SchemaError(
            f"This database is at schema version {current}, but this Ingatan only knows "
            f"version {newest}. Upgrade Ingatan."
        )
    applied: list[int] = []
    for step in plan:
        if step.version <= current:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Another process may have migrated while we waited for the lock.
            if _user_version(conn) >= step.version:
                conn.execute("COMMIT")
                continue
            for statement in _statements(step.sql):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {int(step.version)}")
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        applied.append(step.version)
    return applied


def _seed_global_log_ids(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'log'").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sqlite_sequence (name, seq) VALUES ('log', ?)", (GLOBAL_LOG_OFFSET,)
            )
        elif row[0] < GLOBAL_LOG_OFFSET:
            conn.execute(
                "UPDATE sqlite_sequence SET seq = ? WHERE name = 'log'", (GLOBAL_LOG_OFFSET,)
            )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _enable_wal(conn: sqlite3.Connection, patience: float = 10.0) -> None:
    """`PRAGMA journal_mode = WAL`, retried while another process holds the new file.

    On a brand-new file this pragma fails at once with "database is locked" when several
    processes open it together: SQLite does not honour `busy_timeout` for it. Once one of
    them has switched the file to WAL the others succeed.
    """
    deadline = time.monotonic() + patience
    delay = 0.01
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as exc:
            text = str(exc).lower()
            if ("locked" not in text and "busy" not in text) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.2)


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (and create) a database file, set the pragmas and run `migrate`."""
    check_capabilities()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        _enable_wal(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA secure_delete = ON")
        migrate(conn)
        if target.stem == "global":
            _seed_global_log_ids(conn)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """`BEGIN IMMEDIATE` ... `COMMIT`, rolled back if the block raises.

    If the connection is already inside a transaction, the block joins it and the
    outer `transaction` decides the outcome.
    """
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
