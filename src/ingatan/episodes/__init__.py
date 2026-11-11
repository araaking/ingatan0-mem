"""Episodes: raw conversation turns kept so an agent can digest them into facts later."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from ingatan import timeutil
from ingatan.db import transaction
from ingatan.errors import NotFoundError, ValidationError
from ingatan.redact import redact
from ingatan.scope import Context, Scope, data_home, open_context, resolve_scope

__all__ = [
    "MAX_OFFER",
    "MAX_OFFERS_PER_EPISODE",
    "RETENTION_DAYS",
    "TRUNCATE_AT",
    "Episode",
    "capture",
    "mark_processed",
    "offer",
    "purge",
    "record",
]

TRUNCATE_AT = 2000  # characters per message (OQ-13)
RETENTION_DAYS = 30
MAX_OFFER = 3
MAX_OFFERS_PER_EPISODE = 3

ERROR_LOG_NAME = "capture-errors.log"


@dataclass(frozen=True)
class Episode:
    id: int
    agent: str
    session: str | None
    user_text: str
    assistant_text: str
    created_at: str
    status: str
    offered: int
    processed_at: str | None


def _episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        agent=row["agent"],
        session=row["session"],
        user_text=row["user_text"],
        assistant_text=row["assistant_text"],
        created_at=row["created_at"],
        status=row["status"],
        offered=row["offered"],
        processed_at=row["processed_at"],
    )


def _clean(text: str) -> str:
    # Redact before cutting: a cut in the middle of a key would hide it from the patterns.
    return redact(text)[:TRUNCATE_AT]


def record(ctx: Context, user_text: str, assistant_text: str, *, session: str | None = None) -> int:
    """Store one turn for the calling agent and return its id. Raises on any failure."""
    if not user_text.strip() and not assistant_text.strip():
        raise ValidationError("Both messages are empty; there is nothing to record.")
    with transaction(ctx.business):
        cursor = ctx.business.execute(
            "INSERT INTO episodes (agent, session, user_text, assistant_text, created_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'pending')",
            (
                ctx.scope.agent,
                None if session is None else redact(session),
                _clean(user_text),
                _clean(assistant_text),
                timeutil.iso(timeutil.now()),
            ),
        )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _log_failure(home: Path | None, exc: BaseException) -> None:
    try:
        folder = home if home is not None else data_home()
        folder.mkdir(parents=True, exist_ok=True)
        message = redact(f"{type(exc).__name__}: {exc}").replace("\r", " ").replace("\n", " ")
        line = f"{timeutil.iso(timeutil.now())}\t{message}\n"
        with (folder / ERROR_LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:  # the caller must never see a capture failure
        pass


def capture(
    scope_or_none: Scope | None,
    user_text: str,
    assistant_text: str,
    *,
    session: str | None = None,
) -> bool:
    """Record a turn without ever raising. Returns False (and logs) on any failure."""
    home: Path | None = None
    try:
        scope = scope_or_none if scope_or_none is not None else resolve_scope()
        home = scope.home
        ctx = open_context(scope)
        try:
            record(ctx, user_text, assistant_text, session=session)
        finally:
            ctx.close()
        return True
    except Exception as exc:  # NFR-14: a hook failure must not reach the agent
        _log_failure(home, exc)
        return False


def offer(ctx: Context, limit: int = MAX_OFFER) -> list[Episode]:
    """The calling agent's oldest pending episodes, each with its `offered` count raised.

    An episode offered `MAX_OFFERS_PER_EPISODE` times is marked processed in the same
    call (it is returned this last time): the agent judged it not worth a fact.
    """
    now = timeutil.iso(timeutil.now())
    with transaction(ctx.business):
        rows = ctx.business.execute(
            "SELECT * FROM episodes WHERE agent = ? AND status = 'pending' ORDER BY id ASC LIMIT ?",
            (ctx.scope.agent, max(limit, 0)),
        ).fetchall()
        offered: list[Episode] = []
        for row in rows:
            count = row["offered"] + 1
            done = count >= MAX_OFFERS_PER_EPISODE
            ctx.business.execute(
                "UPDATE episodes SET offered = ?, status = ?, processed_at = ? WHERE id = ?",
                (count, "processed" if done else "pending", now if done else None, row["id"]),
            )
            offered.append(
                _episode(
                    ctx.business.execute(
                        "SELECT * FROM episodes WHERE id = ?", (row["id"],)
                    ).fetchone()
                )
            )
    return offered


def mark_processed(ctx: Context, episode_id: int) -> None:
    """Mark one of the calling agent's episodes processed. Idempotent."""
    with transaction(ctx.business):
        row = ctx.business.execute(
            "SELECT status FROM episodes WHERE id = ? AND agent = ?",
            (episode_id, ctx.scope.agent),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"No episode {episode_id}.")
        if row["status"] != "processed":
            ctx.business.execute(
                "UPDATE episodes SET status = 'processed', processed_at = ? WHERE id = ?",
                (timeutil.iso(timeutil.now()), episode_id),
            )


def purge(ctx: Context) -> int:
    """Delete the calling agent's processed episodes older than `RETENTION_DAYS`."""
    cutoff = timeutil.iso(timeutil.now() - timedelta(days=RETENTION_DAYS))
    with transaction(ctx.business):
        cursor = ctx.business.execute(
            "DELETE FROM episodes WHERE agent = ? AND status = 'processed' AND processed_at < ?",
            (ctx.scope.agent, cutoff),
        )
    return cursor.rowcount
