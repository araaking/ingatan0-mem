"""Markdown ingest: one document note per heading section (FR-53, FR-54, FR-62, OQ-15).

Notes are searchable through `search.recall` (as `RecallResult.notes`) and are never part
of a brief. They skip the proposal queue because the user ran the command.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ingatan import audit, timeutil
from ingatan.db import transaction
from ingatan.errors import NotFoundError
from ingatan.ingest.markdown import split_sections
from ingatan.redact import redact
from ingatan.scope import Context

__all__ = ["IngestReport", "Skipped", "ingest_path"]


@dataclass(frozen=True)
class Skipped:
    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path} ({self.reason})"


@dataclass(frozen=True)
class IngestReport:
    """What one run did.

    `files`: Markdown files looked at. `created`: notes written. `replaced`: old notes
    deleted because their file changed. `skipped`: files not ingested, with the reason
    (unchanged, not UTF-8, unreadable).
    """

    files: int = 0
    created: int = 0
    replaced: int = 0
    skipped: list[Skipped] = field(default_factory=lambda: [])


def _files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() == ".md" and p.is_file())
    return [path]


def ingest_path(ctx: Context, path: str | Path) -> IngestReport:
    """Ingest one Markdown file, or every `.md` file under a folder (sorted order)."""
    target = Path(path).expanduser()
    if not target.exists():
        raise NotFoundError(f"No file or folder at {path}.")
    files = created = replaced = 0
    skipped: list[Skipped] = []
    for file in _files(target):
        files += 1
        stored = file.resolve().as_posix()
        try:
            raw = file.read_bytes()
        except OSError as exc:
            skipped.append(Skipped(stored, f"cannot read: {exc.strerror or exc}"))
            continue
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            skipped.append(Skipped(stored, "not valid UTF-8"))
            continue
        digest = hashlib.sha256(raw).hexdigest()
        hashes = {
            r["file_hash"]
            for r in ctx.business.execute(
                "SELECT DISTINCT file_hash FROM notes WHERE path = ?", (stored,)
            )
        }
        if hashes == {digest}:
            skipped.append(Skipped(stored, "unchanged"))
            continue
        sections = split_sections(content, file.name)
        now = timeutil.iso(timeutil.now())
        with transaction(ctx.business):
            removed = ctx.business.execute("DELETE FROM notes WHERE path = ?", (stored,)).rowcount
            for section in sections:
                heading = redact(section.heading)
                text = redact(section.text)
                note_id = ctx.business.execute(
                    "INSERT INTO notes (path, heading, text, file_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (stored, heading, text, digest, now),
                ).lastrowid
                audit.write(
                    ctx.business,
                    agent=ctx.scope.agent,
                    op="ingest",
                    fact_id=f"n_{note_id}",
                    tier="business",
                    owner=ctx.scope.agent,
                    before=None,
                    after={"path": stored, "heading": heading, "text": text},
                )
        created += len(sections)
        replaced += removed
    return IngestReport(files=files, created=created, replaced=replaced, skipped=skipped)
