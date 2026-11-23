"""`facts.text_key` as `remember` uses it. The column itself is covered in `tests/db`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ingatan import db, facts
from ingatan.facts._util import fact_dict
from ingatan.scope import Context, Scope, open_context


def test_rows_written_before_the_key_existed_are_found_as_repeats(tmp_path: Path) -> None:
    path = tmp_path / "tokoku.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        db.migrate(conn, migrations=db.load_migrations()[:2])
        for n, text in enumerate(("Harga kopi Rp 20.000.", "Toko tutup hari Minggu."), start=1):
            conn.execute(
                "INSERT INTO facts (id, text, entities, keywords, tier, agent, kind, "
                "valid_from, source, created_at) VALUES (?, ?, '[]', '[]', 'business', "
                "'asisten', 'stable', '2026-11-01T00:00:00Z', '{}', '2026-11-01T00:00:00Z')",
                (f"f_old00000000{n}", text),
            )
    finally:
        conn.close()

    # Opening the file upgrades it; the old rows then behave like rows of this version.
    ctx = open_context(Scope(business="tokoku", agent="asisten", home=tmp_path))
    try:
        again = facts.remember(ctx, "harga kopi Rp 20.000", keywords=["harga"], tier="business")
        assert again.fact_id == "f_old000000001" and "refreshed" in again.message
        assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 2
    finally:
        ctx.close()


def test_the_key_stays_in_the_row_and_out_of_the_log(ctx: Context) -> None:
    saved = facts.remember(ctx, "Sandi wifi diganti tiap tiga bulan.", keywords=["wifi"])
    assert saved.fact_id
    row = ctx.business.execute("SELECT * FROM facts WHERE id = ?", (saved.fact_id,)).fetchone()
    key = str(row["text_key"])
    assert key == db.fact_text_key("Sandi wifi diganti tiap tiga bulan.")
    assert "text_key" not in fact_dict(row) and "seq" not in fact_dict(row)
    for entry in ctx.business.execute("SELECT change FROM log"):
        assert "text_key" not in json.loads(entry["change"])["after"]
        assert key not in entry["change"]

    # So forgetting the fact removes the key with the row: no trace of it stays in the file.
    facts.forget(ctx, saved.fact_id)
    ctx.business.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = Path(ctx.scope.home / "tokoku.db").read_bytes()
    assert key.encode("ascii") not in raw
