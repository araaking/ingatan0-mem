from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ingatan import db
from ingatan.errors import SchemaError

InsertFact = Callable[..., str]

TABLES = {
    "facts",
    "facts_fts",
    "facts_tri",
    "conflicts",
    "log",
    "proposals",
    "episodes",
    "notes",
    "notes_fts",
    "notes_tri",
}


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def insert_log(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO log (ts, agent, op) VALUES ('t', 'a', 'create')")


# ---- migrations -------------------------------------------------------------------


def test_new_database_is_created_at_latest_version(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "tokoku.db"
    conn = db.connect(path)
    try:
        assert path.exists()
        assert db.LATEST_VERSION >= 1
        assert user_version(conn) == db.LATEST_VERSION
        assert table_names(conn) >= TABLES
    finally:
        conn.close()


def test_connect_sets_the_documented_pragmas(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        assert conn.isolation_level is None
        assert conn.row_factory is sqlite3.Row
    finally:
        conn.close()


def test_migration_files_are_loaded_as_package_resources() -> None:
    migrations = db.load_migrations()
    assert [m.version for m in migrations] == list(range(1, db.LATEST_VERSION + 1))
    assert "CREATE TABLE facts" in migrations[0].sql


def test_up_to_date_database_applies_nothing(tmp_path: Path) -> None:
    path = tmp_path / "a.db"
    db.connect(path).close()
    conn = db.connect(path)
    try:
        assert db.migrate(conn) == []
        assert user_version(conn) == db.LATEST_VERSION
    finally:
        conn.close()


def test_old_database_gets_only_the_newer_migrations(tmp_path: Path) -> None:
    # A database at the current version holds data. A later release adds one more
    # migration: only that one may run, and the data must survive (0001 is not re-run).
    path = tmp_path / "a.db"
    first = db.connect(path)
    insert_log(first)
    first.close()

    next_version = db.LATEST_VERSION + 1
    future = [
        *db.load_migrations(),
        db.Migration(
            next_version,
            f"{next_version:04d}_extra.sql",
            "CREATE TABLE extra (x TEXT);\nINSERT INTO extra VALUES ('hi; there');\n",
        ),
    ]
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        assert user_version(conn) == db.LATEST_VERSION
        assert db.migrate(conn, future) == [next_version]
        assert user_version(conn) == next_version
        assert conn.execute("SELECT x FROM extra").fetchone()[0] == "hi; there"
        assert conn.execute("SELECT count(*) FROM log").fetchone()[0] == 1
        assert db.migrate(conn, future) == []
    finally:
        conn.close()


def test_existing_unversioned_file_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "a.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE unrelated (x)")
    raw.commit()
    raw.close()
    conn = db.connect(path)
    try:
        assert user_version(conn) == db.LATEST_VERSION
        assert {"unrelated", "facts"} <= table_names(conn)
    finally:
        conn.close()


def test_database_newer_than_the_code_raises_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "a.db"
    db.connect(path).close()
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {db.LATEST_VERSION + 1}")
    raw.close()
    with pytest.raises(SchemaError, match="Upgrade Ingatan"):
        db.connect(path)


def test_failed_migration_rolls_back_completely(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "a.db", isolation_level=None)
    try:
        bad = [db.Migration(1, "0001_bad.sql", "CREATE TABLE half (x);\nCREATE TABLE half (y);")]
        with pytest.raises(sqlite3.OperationalError):
            db.migrate(conn, bad)
        assert user_version(conn) == 0
        assert "half" not in table_names(conn)
        assert not conn.in_transaction
    finally:
        conn.close()


# ---- capabilities -----------------------------------------------------------------


def test_real_sqlite_has_the_needed_capabilities() -> None:
    db.check_capabilities()


class _FailsOn(sqlite3.Connection):
    needle = ""

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        if self.needle in sql:
            raise sqlite3.OperationalError("simulated missing feature")
        return super().execute(sql, parameters)


class _NoTrigram(_FailsOn):
    needle = "trigram"


class _NoFts5(_FailsOn):
    needle = "fts5"


def test_missing_trigram_tokenizer_gives_a_clear_error() -> None:
    fake = sqlite3.connect(":memory:", factory=_NoTrigram)
    with pytest.raises(SchemaError, match="trigram"):
        db.check_capabilities(fake)


def test_missing_fts5_gives_a_clear_error() -> None:
    fake = sqlite3.connect(":memory:", factory=_NoFts5)
    with pytest.raises(SchemaError, match="FTS5"):
        db.check_capabilities(fake)


class _NoSecureDelete(_FailsOn):
    needle = "secure-delete"


def test_old_sqlite_without_fts5_secure_delete_gives_a_clear_error() -> None:
    fake = sqlite3.connect(":memory:", factory=_NoSecureDelete)
    with pytest.raises(SchemaError, match=r"secure-delete.*3\.42"):
        db.check_capabilities(fake)


@pytest.mark.parametrize("table", ["facts_fts", "facts_tri", "notes_fts", "notes_tri"])
def test_every_fts_table_has_secure_delete_switched_on(tmp_path: Path, table: str) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        value = conn.execute(f"SELECT v FROM {table}_config WHERE k = 'secure-delete'").fetchone()
        assert value is not None and value[0] == 1
    finally:
        conn.close()


def test_facts_rowid_is_an_explicit_stable_column(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        columns = {r["name"]: r for r in conn.execute("PRAGMA table_info(facts)")}
        assert columns["seq"]["pk"] == 1 and columns["seq"]["type"] == "INTEGER"
        assert columns["id"]["pk"] == 0 and columns["id"]["notnull"] == 1
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'facts'").fetchone()[0]
        assert "AUTOINCREMENT" in sql
        with pytest.raises(sqlite3.IntegrityError):
            for _ in range(2):
                conn.execute(
                    "INSERT INTO facts (id, text, entities, keywords, tier, agent, kind, "
                    "valid_from, source, created_at) VALUES ('f_1', 't', '[]', '[]', 'agent', "
                    "'a', 'stable', 'x', '{}', 'x')"
                )
    finally:
        conn.close()


def test_fts_rows_follow_the_seq_column_through_a_vacuum(
    tmp_path: Path, insert_fact: InsertFact
) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        ids = [insert_fact(conn, text=f"kata unik{n}") for n in range(5)]
        conn.execute("DELETE FROM facts WHERE id = ?", (ids[1],))
        conn.execute("VACUUM")
        for n in (0, 2, 3, 4):
            row = conn.execute(
                "SELECT f.id FROM facts_fts JOIN facts f ON f.seq = facts_fts.rowid "
                "WHERE facts_fts MATCH ?",
                (f'"unik{n}"',),
            ).fetchone()
            assert row["id"] == ids[n]
    finally:
        conn.close()


# ---- transaction ------------------------------------------------------------------


def test_transaction_commits_and_rolls_back(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        with db.transaction(conn):
            assert conn.in_transaction
            insert_log(conn)
        assert not conn.in_transaction
        with pytest.raises(RuntimeError), db.transaction(conn):
            insert_log(conn)
            raise RuntimeError("boom")
        assert not conn.in_transaction
        assert conn.execute("SELECT count(*) FROM log").fetchone()[0] == 1
    finally:
        conn.close()


def test_nested_transaction_joins_the_outer_one(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        with pytest.raises(RuntimeError), db.transaction(conn):
            with db.transaction(conn):
                insert_log(conn)
            assert conn.in_transaction
            raise RuntimeError("outer fails")
        assert conn.execute("SELECT count(*) FROM log").fetchone()[0] == 0
    finally:
        conn.close()


def test_global_file_log_ids_do_not_collide_with_business_ids(tmp_path: Path) -> None:
    business = db.connect(tmp_path / "tokoku.db")
    glob = db.connect(tmp_path / "global.db")
    try:
        insert_log(business)
        insert_log(glob)
        assert business.execute("SELECT max(id) FROM log").fetchone()[0] == 1
        assert glob.execute("SELECT max(id) FROM log").fetchone()[0] == db.GLOBAL_LOG_OFFSET + 1
    finally:
        business.close()
        glob.close()
    again = db.connect(tmp_path / "global.db")  # re-opening must not reset the sequence
    try:
        insert_log(again)
        assert again.execute("SELECT max(id) FROM log").fetchone()[0] == db.GLOBAL_LOG_OFFSET + 2
    finally:
        again.close()


# ---- Indonesian text through facts and both FTS indexes ------------------------------

INDONESIAN = "Harga kopi susu naik jadi Rp 18.000 — kata Bu Sári ‘mulai Senin’ ✓"


def fts_ids(conn: sqlite3.Connection, table: str, query: str) -> list[str]:
    rows = conn.execute(
        f"SELECT f.id FROM {table} JOIN facts f ON f.rowid = {table}.rowid WHERE {table} MATCH ?",
        (query,),
    )
    return [r[0] for r in rows]


def test_indonesian_text_round_trips_through_facts_and_both_indexes(
    tmp_path: Path, insert_fact: InsertFact
) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        fid = insert_fact(
            conn,
            text=INDONESIAN,
            entities=["Bu Sári", "kopi susu"],
            keywords=["harga", "naik", "menaikkan"],
        )
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
        assert row["text"] == INDONESIAN
        assert json.loads(row["entities"]) == ["Bu Sári", "kopi susu"]
        # Word index: diacritics are folded, so the plain spelling finds "Sári".
        assert fts_ids(conn, "facts_fts", "kopi") == [fid]
        assert fts_ids(conn, "facts_fts", "sari") == [fid]
        assert fts_ids(conn, "facts_fts", "Senin") == [fid]
        assert fts_ids(conn, "facts_fts", "entities:sari") == [fid]
        assert fts_ids(conn, "facts_fts", "keywords:menaikkan") == [fid]
        assert fts_ids(conn, "facts_fts", "udang") == []
        # Trigram index: partial words, and the exact accented spelling.
        assert fts_ids(conn, "facts_tri", '"nai"') == [fid]
        assert fts_ids(conn, "facts_tri", '"Sári"') == [fid]
        assert fts_ids(conn, "facts_tri", '"mulai"') == [fid]
        assert fts_ids(conn, "facts_tri", '"menaik"') == [fid]
    finally:
        conn.close()


def test_indonesian_text_round_trips_through_the_notes_indexes(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        conn.execute(
            "INSERT INTO notes (path, heading, text, file_hash, created_at) "
            "VALUES ('/n/harga.md', 'Harga kopi', ?, 'h', 't')",
            (INDONESIAN,),
        )
        row = conn.execute("SELECT text FROM notes").fetchone()
        assert row["text"] == INDONESIAN
        for table, query in (
            ("notes_fts", "sari"),
            ("notes_fts", "heading:kopi"),
            ("notes_tri", '"Sári"'),
            ("notes_tri", '"mulai"'),
        ):
            found = conn.execute(
                f"SELECT n.path FROM {table} JOIN notes n ON n.id = {table}.rowid "
                f"WHERE {table} MATCH ?",
                (query,),
            ).fetchall()
            assert [r[0] for r in found] == ["/n/harga.md"], (table, query)
    finally:
        conn.close()


def test_indexes_follow_updates_and_deletes(tmp_path: Path, insert_fact: InsertFact) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        fid = insert_fact(conn, text="kopi susu", keywords=["minuman"])
        conn.execute("UPDATE facts SET text = 'teh manis' WHERE id = ?", (fid,))
        assert fts_ids(conn, "facts_fts", "kopi") == []
        assert fts_ids(conn, "facts_fts", "teh") == [fid]
        assert fts_ids(conn, "facts_tri", '"man"') == [fid]
        conn.execute("DELETE FROM facts WHERE id = ?", (fid,))
        assert conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM facts_tri").fetchone()[0] == 0
    finally:
        conn.close()


# ---- log protection (FR-36) ----------------------------------------------------------


def test_log_cannot_be_deleted(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        insert_log(conn)
        with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
            conn.execute("DELETE FROM log WHERE id = 1")
        with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
            conn.execute("DELETE FROM log")
        assert conn.execute("SELECT count(*) FROM log").fetchone()[0] == 1
    finally:
        conn.close()


def test_log_rows_cannot_be_rewritten_but_change_can_be_cleared(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "a.db")
    try:
        conn.execute(
            "INSERT INTO log (ts, agent, op, fact_id, tier, owner, change) "
            "VALUES ('2026-11-11T09:00:00Z', 'a', 'create', 'f_1', 'agent', 'a', '{\"x\": 1}')"
        )
        for column, value in (
            ("ts", "'2030-01-01T00:00:00Z'"),
            ("agent", "'b'"),
            ("op", "'forget'"),
            ("fact_id", "'f_2'"),
            ("tier", "'global'"),
            ("owner", "'b'"),
            ("change", "'{}'"),
            ("id", "7"),
        ):
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                conn.execute(f"UPDATE log SET {column} = {value}")
        # Setting change to NULL together with a changed column is also refused.
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE log SET change = NULL, ts = 'x'")
        row = conn.execute("SELECT * FROM log").fetchone()
        assert (row["ts"], row["op"], row["change"]) == (
            "2026-11-11T09:00:00Z",
            "create",
            '{"x": 1}',
        )

        conn.execute("UPDATE log SET change = NULL")
        row = conn.execute("SELECT * FROM log").fetchone()
        assert row["change"] is None
        assert (row["ts"], row["agent"], row["op"], row["fact_id"]) == (
            "2026-11-11T09:00:00Z",
            "a",
            "create",
            "f_1",
        )
    finally:
        conn.close()


def test_enable_wal_retries_while_the_file_is_locked() -> None:
    calls: list[str] = []

    class Flaky:
        def execute(self, sql: str) -> None:
            calls.append(sql)
            if len(calls) < 3:
                raise sqlite3.OperationalError("database is locked")

    db._enable_wal(Flaky(), patience=5)  # pyright: ignore[reportArgumentType]
    assert len(calls) == 3


def test_enable_wal_gives_up_on_other_errors_and_after_the_patience_runs_out() -> None:
    class Broken:
        def execute(self, sql: str) -> None:
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        db._enable_wal(Broken(), patience=5)  # pyright: ignore[reportArgumentType]

    class AlwaysLocked:
        def execute(self, sql: str) -> None:
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        db._enable_wal(AlwaysLocked(), patience=0.05)  # pyright: ignore[reportArgumentType]
