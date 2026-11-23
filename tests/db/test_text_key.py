"""`facts.text_key` (schema version 3): the indexed column behind exact-repeat detection."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingatan import db

_INSERT = (
    "INSERT INTO facts (id, text, entities, keywords, tier, agent, kind, valid_from, source, "
    "created_at) VALUES (?, ?, '[]', '[]', 'business', 'asisten', 'stable', "
    "'2026-11-01T00:00:00Z', '{}', '2026-11-01T00:00:00Z')"
)


def test_key_ignores_case_spacing_and_trailing_punctuation_and_nothing_else() -> None:
    key = db.fact_text_key("Target margin bersih produk minimal 25%.")
    assert key == db.fact_text_key("  target   MARGIN bersih\tproduk minimal 25%  ")
    assert key == db.fact_text_key("Target margin bersih produk minimal 25%!?")
    assert len(key) == 32
    # A different value, sign, comparison or inner punctuation is a different statement.
    for other in (
        "Target margin bersih produk minimal 20%.",
        "Target margin bersih produk minimal -25%.",
        "Target margin bersih produk maksimal 25%.",
        "Target margin bersih: produk minimal 25%.",
    ):
        assert db.fact_text_key(other) != key
    assert db.fact_text_key("Stok < 10") != db.fact_text_key("Stok > 10")


def test_a_database_from_version_2_gets_its_keys_when_it_is_opened(tmp_path: Path) -> None:
    path = tmp_path / "tokoku.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        db.migrate(conn, migrations=db.load_migrations()[:2])
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")}
        assert "text_key" not in columns
        for n, text in enumerate(("Harga kopi Rp 20.000.", "Toko tutup hari Minggu."), start=1):
            conn.execute(_INSERT, (f"f_old00000000{n}", text))
    finally:
        conn.close()

    opened = db.connect(path)
    try:
        assert opened.execute("PRAGMA user_version").fetchone()[0] == db.LATEST_VERSION >= 3
        rows = opened.execute("SELECT text, text_key FROM facts ORDER BY seq").fetchall()
        assert [r["text_key"] for r in rows] == [db.fact_text_key(r["text"]) for r in rows]
        plan = opened.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM facts WHERE text_key = 'x'"
        ).fetchall()
        assert any("facts_text_key" in str(row[-1]) for row in plan)
        # A row that arrives without a key (written by an older Ingatan sharing the file)
        # is filled in the next time the file is opened, and only that row is touched.
        opened.execute(_INSERT, ("f_late00000001", "Gudang pindah ke lantai dua."))
    finally:
        opened.close()

    again = db.connect(path)
    try:
        keys = dict(again.execute("SELECT id, text_key FROM facts").fetchall())
        assert keys["f_late00000001"] == db.fact_text_key("Gudang pindah ke lantai dua.")
        assert all(keys.values()) and len(set(keys.values())) == 3
    finally:
        again.close()
