from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ingatan import audit, facts, search
from ingatan.errors import NotFoundError
from ingatan.ingest import ingest_path
from ingatan.ingest.markdown import split_sections
from ingatan.redact import MARKER
from ingatan.scope import Context

KEY = "sk-Ab12Cd34Ef56Gh78Ij90Kl12Mn34"

DOC = """Pengantar toko.

# Harga
Kopi susu Rp 18.000 per gelas.

## Promo
Beli dua gratis satu hari Jumat.

# Pengiriman
Kurir lokal jam sepuluh pagi.
"""


def notes(ctx: Context) -> list[tuple[str, str]]:
    return [
        (r["heading"], r["text"])
        for r in ctx.business.execute("SELECT heading, text FROM notes ORDER BY id")
    ]


def test_one_note_per_heading_section(ctx: Context, tmp_path: Path) -> None:
    doc = tmp_path / "toko.md"
    doc.write_text(DOC, encoding="utf-8")
    report = ingest_path(ctx, doc)
    assert (report.files, report.created, report.replaced, report.skipped) == (1, 4, 0, [])
    assert notes(ctx) == [
        ("toko.md", "Pengantar toko."),
        ("Harga", "Kopi susu Rp 18.000 per gelas."),
        ("Promo", "Beli dua gratis satu hari Jumat."),
        ("Pengiriman", "Kurir lokal jam sepuluh pagi."),
    ]


def test_path_is_resolved_absolute_with_forward_slashes_and_hash_is_sha256(
    ctx: Context, tmp_path: Path
) -> None:
    (tmp_path / "sub").mkdir()
    doc = tmp_path / "sub" / ".." / "toko.md"
    doc.write_bytes(DOC.encode("utf-8"))
    ingest_path(ctx, doc)
    row = ctx.business.execute("SELECT path, file_hash FROM notes LIMIT 1").fetchone()
    assert row["path"] == (tmp_path / "toko.md").resolve().as_posix()
    assert "\\" not in row["path"]
    assert Path(row["path"]).is_absolute()
    assert row["file_hash"] == hashlib.sha256(DOC.encode("utf-8")).hexdigest()


def test_unchanged_file_is_skipped_and_creates_nothing(ctx: Context, tmp_path: Path) -> None:
    doc = tmp_path / "toko.md"
    doc.write_text(DOC, encoding="utf-8")
    ingest_path(ctx, doc)
    before = ctx.business.execute("SELECT COUNT(*) FROM log").fetchone()[0]
    again = ingest_path(ctx, doc)
    assert (again.files, again.created, again.replaced) == (1, 0, 0)
    assert [s.reason for s in again.skipped] == ["unchanged"]
    assert len(notes(ctx)) == 4
    assert ctx.business.execute("SELECT COUNT(*) FROM log").fetchone()[0] == before


def test_changed_file_replaces_its_notes_only(ctx: Context, tmp_path: Path) -> None:
    doc = tmp_path / "toko.md"
    other = tmp_path / "lain.md"
    doc.write_text(DOC, encoding="utf-8")
    other.write_text("# Lain\nCatatan lain.\n", encoding="utf-8")
    ingest_path(ctx, tmp_path)
    doc.write_text("# Harga\nKopi susu Rp 20.000 per gelas.\n", encoding="utf-8")
    report = ingest_path(ctx, tmp_path)
    assert (report.files, report.created, report.replaced) == (2, 1, 4)
    assert sorted(notes(ctx)) == [
        ("Harga", "Kopi susu Rp 20.000 per gelas."),
        ("Lain", "Catatan lain."),
    ]
    # The old text is gone from the search index too.
    assert search.recall(ctx, "Pengiriman kurir").notes == []


def test_folder_is_walked_recursively_in_sorted_order(ctx: Context, tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "z.md").write_text("# Z\nisi z\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\nisi a\n", encoding="utf-8")
    (tmp_path / "b" / "c.MD").write_text("# C\nisi c\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("# T\nbukan markdown\n", encoding="utf-8")
    report = ingest_path(ctx, tmp_path)
    assert report.files == 3
    assert [h for h, _ in notes(ctx)] == ["A", "C", "Z"]
    paths = [r["path"] for r in ctx.business.execute("SELECT path FROM notes ORDER BY id")]
    assert paths == [(tmp_path / p).resolve().as_posix() for p in ("a.md", "b/c.MD", "b/z.md")]


def test_secret_in_a_file_never_reaches_the_database(
    ctx: Context, tmp_path: Path, ingatan_home: Path
) -> None:
    doc = tmp_path / "rahasia.md"
    doc.write_text(f"# Akses {KEY}\nkunci api: {KEY}\npassword=hunter2xyz\n", encoding="utf-8")
    ingest_path(ctx, doc)
    [(heading, text)] = notes(ctx)
    assert KEY not in heading + text
    assert "hunter2xyz" not in text
    assert MARKER in text
    blob = b""
    for suffix in ("", "-wal", "-shm"):
        p = ingatan_home / f"tokoku.db{suffix}"
        if p.exists():
            blob += p.read_bytes()
    assert KEY.encode() not in blob
    assert b"hunter2xyz" not in blob
    for entry in audit.entries(ctx):
        assert KEY not in str(entry.change)


def test_one_ingest_log_entry_per_note(ctx: Context, tmp_path: Path) -> None:
    doc = tmp_path / "toko.md"
    doc.write_text(DOC, encoding="utf-8")
    ingest_path(ctx, doc)
    entries = [e for e in audit.entries(ctx) if e.op == "ingest"]
    assert len(entries) == 4
    ids = [r["id"] for r in ctx.business.execute("SELECT id FROM notes ORDER BY id")]
    assert [e.fact_id for e in entries] == [f"n_{i}" for i in ids]
    for entry in entries:
        assert entry.tier == "business"
        assert entry.change is not None
        assert entry.change["before"] is None
        after = entry.change["after"]
        assert after["path"] == doc.resolve().as_posix()
        assert {"heading", "text"} <= set(after)
    assert entries[1].change is not None
    assert entries[1].change["after"]["text"] == "Kopi susu Rp 18.000 per gelas."


def test_non_utf8_file_is_skipped_with_a_reason(ctx: Context, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_bytes(b"# Judul\nkopi \xff\xfe rusak\n")
    (tmp_path / "good.md").write_text("# Baik\nisi baik\n", encoding="utf-8")
    report = ingest_path(ctx, tmp_path)
    assert (report.files, report.created) == (2, 1)
    [skip] = report.skipped
    assert skip.path.endswith("bad.md")
    assert "UTF-8" in skip.reason
    assert notes(ctx) == [("Baik", "isi baik")]


def test_deleted_file_leaves_its_notes_alone(ctx: Context, tmp_path: Path) -> None:
    doc = tmp_path / "toko.md"
    doc.write_text(DOC, encoding="utf-8")
    ingest_path(ctx, doc)
    doc.unlink()
    (tmp_path / "baru.md").write_text("# Baru\nisi baru\n", encoding="utf-8")
    ingest_path(ctx, tmp_path)
    assert len(notes(ctx)) == 5


def test_missing_path_raises_not_found(ctx: Context, tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        ingest_path(ctx, tmp_path / "tidak-ada.md")


def test_recall_returns_note_as_document_excerpt(ctx: Context, tmp_path: Path) -> None:
    doc = tmp_path / "toko.md"
    doc.write_text(DOC, encoding="utf-8")
    ingest_path(ctx, doc)
    result = search.recall(ctx, "gratis Jumat")
    assert result.facts == []
    [hit] = result.notes
    assert hit.heading == "Promo"
    assert hit.path == doc.resolve().as_posix()
    assert "Beli dua gratis" in hit.text


def test_notes_never_affect_recalled_facts(ctx: Context, tmp_path: Path) -> None:
    facts.remember(ctx, "Harga kopi susu 18000", keywords=["kopi", "harga"])
    before = [h.id for h in search.recall(ctx, "kopi susu harga", touch=False).facts]
    doc = tmp_path / "toko.md"
    doc.write_text(DOC, encoding="utf-8")
    ingest_path(ctx, doc)
    result = search.recall(ctx, "kopi susu harga", touch=False)
    assert [h.id for h in result.facts] == before
    assert result.notes  # the note is found, but only under `notes`
    assert ctx.business.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1


def test_brief_never_shows_notes(ctx: Context, tmp_path: Path) -> None:
    brief = pytest.importorskip("ingatan.brief")
    if not hasattr(brief, "build"):
        pytest.skip("brief.build is not finished yet")
    doc = tmp_path / "toko.md"
    doc.write_text("# Rahasia dokumen\nzzcatatanunikzz\n", encoding="utf-8")
    ingest_path(ctx, doc)
    assert "zzcatatanunikzz" not in brief.build(ctx)


class TestSplitSections:
    def test_hash_lines_in_fenced_code_are_not_headings(self) -> None:
        md = "# Atas\nteks\n```bash\n# komentar shell\necho hi\n```\nlanjut\n## Bawah\nisi\n"
        sections = split_sections(md, "f.md")
        assert [s.heading for s in sections] == ["Atas", "Bawah"]
        assert "# komentar shell" in sections[0].text

    def test_tilde_fence_and_longer_closing_fence(self) -> None:
        md = "# A\n~~~\n# x\n~~~\n````\n# y\n```\n# masih kode\n````\n# B\nb\n"
        assert [s.heading for s in split_sections(md, "f.md")] == ["A", "B"]

    def test_empty_sections_are_skipped(self) -> None:
        md = "# Kosong\n\n# Ada\nisi\n## Kosong lagi\n"
        assert [(s.heading, s.text) for s in split_sections(md, "f.md")] == [("Ada", "isi")]

    def test_levels_closing_hashes_and_non_headings(self) -> None:
        md = "###### Enam ######\na\n####### tujuh\n#tanpa spasi\n    # kode berindent\n"
        sections = split_sections(md, "f.md")
        assert [s.heading for s in sections] == ["Enam"]
        assert "####### tujuh" in sections[0].text
        assert "#tanpa spasi" in sections[0].text

    def test_preamble_gets_file_name_and_crlf_is_handled(self) -> None:
        sections = split_sections("awal\r\n# H\r\nisi\r\n", "catatan.md")
        assert [(s.heading, s.text) for s in sections] == [("catatan.md", "awal"), ("H", "isi")]

    def test_no_headings_gives_one_section(self) -> None:
        assert len(split_sections("hanya teks", "x.md")) == 1
        assert split_sections("   \n\n", "x.md") == []
