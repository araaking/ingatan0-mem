"""Optional fact extraction: a model reads captured episodes and proposes facts.

Ingatan remembers what an agent chooses to pass to `remember`. A long conversation holds
facts nobody wrote down, and the hooks capture those conversations as episodes. With a model
configured under `[llm]`, `extract_pending` lets that model read the pending episodes and
turn what they state into proposals.

What keeps this safe to switch on:

- Off by default. With `[llm] provider = "none"` nothing is read and nothing is sent.
- A model can invent things. What it returns is never written as a fact: every item becomes
  a proposal in the `business` tier, marked with the model that produced it, and the user
  approves or rejects it. A number that does not occur in the episode is dropped outright.
- An episode is data. Whatever it says, and whatever the model answers, the only thing this
  code does with the answer is read `text`, `keywords`, `entities` and `kind` of at most
  `MAX_FACTS_PER_EPISODE` items.
- Text is redacted again before it is sent, and the model is called outside any transaction.
- It is its own command (`ingatan extract`). The server and `remember` never call a model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ingatan import episodes, facts, search, timeutil
from ingatan.db import fact_text_key, transaction
from ingatan.errors import IngatanError
from ingatan.extract import llm, prompt
from ingatan.redact import redact
from ingatan.scope import Context
from ingatan.search import is_valid_sql

__all__ = ["MAX_FACTS_PER_EPISODE", "MAX_FACT_CHARS", "ExtractReport", "extract_pending"]

# A conversation turn rarely states more. The cap is what bounds the queue when a model, or
# text written to steer it, returns a long list.
MAX_FACTS_PER_EPISODE = 8
# A fact is one statement. Longer than this is a summary of the episode, not a fact.
MAX_FACT_CHARS = 400
# How many facts already in memory the model is shown, to relate a new fact to.
MAX_KNOWN_FACTS = 5
_RELATIONS = ("related_to", "part_of")

_NUMBER = re.compile(r"\d[\d.,]*")
_KINDS = ("stable", "volatile")


@dataclass(frozen=True)
class ExtractReport:
    """What one pass did. `rejected` are items of the model that were dropped, `known` are
    items that repeat a fact or a pending proposal word for word, `errors` are episodes or
    notes the model could not be read for (they are read again next time). `texts` are the
    facts proposed, or, in a dry run, the ones that would be. A pass reads episodes or, with
    `documents`, ingested notes: one of the two counts is always 0."""

    episodes_read: int
    proposed: int
    rejected: int
    errors: int
    known: int = 0
    last_error: str = ""
    texts: tuple[str, ...] = ()
    model: str = ""
    notes_read: int = 0
    relations: int = 0


def _numbers(text: str) -> set[str]:
    """Every number in `text`, without its separators: "Rp 15.000" and "Rp15,000" are one."""
    return {match.group(0).replace(".", "").replace(",", "") for match in _NUMBER.finditer(text)}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]  # pyright: ignore[reportUnknownVariableType]


def _known_keys(ctx: Context) -> set[str]:
    """Text keys of what the user would see twice: pending proposals of every agent."""
    keys: set[str] = set()
    for conn in ctx.connections():
        for row in conn.execute(
            "SELECT json_extract(payload, '$.text') AS text FROM proposals WHERE status = 'pending'"
        ):
            if row["text"]:
                keys.add(fact_text_key(str(row["text"])))
    return keys


def _is_stored(ctx: Context, key: str) -> bool:
    """Whether a valid fact this caller can see says exactly this already."""
    params = {"key": key, "now": timeutil.iso(timeutil.now()), **ctx.params}
    for conn in ctx.connections():
        row = conn.execute(
            f"SELECT 1 FROM facts WHERE text_key = :key AND {is_valid_sql()} "
            f"AND {ctx.visible_sql(conn)} LIMIT 1",
            params,
        ).fetchone()
        if row is not None:
            return True
    return False


def _known_facts(ctx: Context, text: str) -> list[tuple[str, str]]:
    """Facts in memory about what `text` talks about, as `(id, text)`, for the model to
    relate a new fact to. Not counted as use: nobody asked for them."""
    hits = search.recall(
        ctx, text, limit=MAX_KNOWN_FACTS, notes_limit=0, touch=False, include_related=False
    ).facts
    return [(hit.id, redact(hit.text)) for hit in hits]


def _links(relations: list[Any], count: int, offered: set[str]) -> dict[int, list[tuple[str, str]]]:
    """The model's relations that can be used, by the position of the fact they belong to.

    A relation is kept only when it names a fact of this answer, an id the model was shown,
    and one of the two relations there are. One link per target: the first one stands.
    """
    links: dict[int, list[tuple[str, str]]] = {}
    for item in relations:
        if not isinstance(item, dict):
            continue
        index, target, relation = item.get("fact"), item.get("to"), item.get("relation")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            continue
        if target not in offered or relation not in _RELATIONS:
            continue
        chosen = links.setdefault(index, [])
        if all(target != linked for linked, _ in chosen):
            chosen.append((str(target), str(relation)))  # pyright: ignore[reportUnknownArgumentType]
    return links


@dataclass(frozen=True)
class _Reading:
    """One text for the model: what to send, what it states, and how to record the result."""

    message: str
    stated_text: str
    source: dict[str, Any]
    episode: int | None
    mark_read: Callable[[], None]


def _episode_readings(ctx: Context, limit: int) -> list[_Reading]:
    readings: list[_Reading] = []
    for episode in episodes.pending(ctx, limit):

        def mark(episode_id: int = episode.id) -> None:
            episodes.mark_processed(ctx, episode_id)

        readings.append(
            _Reading(
                # Redacted once more: an episode written by an older version or another
                # tool may hold what today's patterns would catch.
                message=prompt.user_message(
                    redact(episode.user_text), redact(episode.assistant_text)
                ),
                stated_text=f"{episode.user_text}\n{episode.assistant_text}",
                source={"session": episode.session},
                episode=episode.id,
                mark_read=mark,
            )
        )
    return readings


def _note_readings(ctx: Context, limit: int) -> list[_Reading]:
    """Ingested notes the model has not read. `extract` sits above `ingest`, whose rows
    these are; there is no function for this there, so it is plain SQL."""
    rows = ctx.business.execute(
        "SELECT id, path, heading, text FROM notes WHERE extracted_at IS NULL "
        "ORDER BY id ASC LIMIT ?",
        (max(limit, 0),),
    ).fetchall()
    readings: list[_Reading] = []
    for row in rows:

        def mark(note_id: int = int(row["id"])) -> None:
            with transaction(ctx.business):
                ctx.business.execute(
                    "UPDATE notes SET extracted_at = ? WHERE id = ?",
                    (timeutil.iso(timeutil.now()), note_id),
                )

        heading, text = str(row["heading"]), str(row["text"])
        readings.append(
            _Reading(
                message=prompt.document_message(str(row["path"]), redact(heading), redact(text)),
                stated_text=f"{heading}\n{text}",
                source={"document": str(row["path"])},
                episode=None,
                mark_read=mark,
            )
        )
    return readings


def extract_pending(
    ctx: Context, *, limit: int = 10, dry_run: bool = False, documents: bool = False
) -> ExtractReport:
    """Let the configured model read pending texts, oldest first, and propose facts.

    By default the texts are the caller's pending episodes; with `documents` they are the
    ingested notes of the business that were not read yet. Each fact the model states
    becomes a `business` proposal that carries `source.extracted_by`. A text that was read
    is not sent again: the episode is marked processed, the note gets `extracted_at`. A
    text the model could not be read for is left as it is. With `dry_run` nothing is
    written: the report says what would be proposed.

    Raises `IngatanError` when no model is configured.
    """
    cfg = ctx.scope.llm
    llm.ensure_configured(cfg)
    label = llm.model_label(cfg)
    proposed = rejected = errors = known = linked = 0
    last_error = ""
    texts: list[str] = []
    seen = _known_keys(ctx)
    readings = _note_readings(ctx, limit) if documents else _episode_readings(ctx, limit)

    for reading in readings:
        offered = _known_facts(ctx, reading.stated_text)
        try:
            # Outside any transaction: nothing is locked while the model thinks.
            answer = llm.chat(cfg, prompt.SYSTEM, reading.message + prompt.known_block(offered))
            items = prompt.parse_answer(answer)
        except (IngatanError, ValueError) as exc:
            errors += 1
            last_error = exc.message if isinstance(exc, IngatanError) else f"JSON: {exc}"
            continue

        stated = _numbers(reading.stated_text)
        links = _links(prompt.parse_relations(answer), len(items), {fid for fid, _ in offered})
        kept = 0
        marked = False
        for index, item in enumerate(items):
            text = item.get("text") if isinstance(item, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_FACT_CHARS:
                rejected += 1
                continue
            clean = redact(text).strip()
            if not _numbers(clean) <= stated:
                rejected += 1  # a value the text does not contain was made up
                continue
            key = fact_text_key(clean)
            if key in seen or _is_stored(ctx, key):
                known += 1
                continue
            if kept >= MAX_FACTS_PER_EPISODE:
                rejected += 1
                continue
            seen.add(key)
            kind = item.get("kind")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            item_links = links.get(index, [])
            if not dry_run:
                try:
                    facts.remember(
                        ctx,
                        clean,
                        keywords=_strings(item.get("keywords")),  # pyright: ignore[reportUnknownMemberType]
                        entities=_strings(item.get("entities")),  # pyright: ignore[reportUnknownMemberType]
                        kind=kind if kind in _KINDS else None,  # pyright: ignore[reportUnknownArgumentType]
                        tier="business",
                        as_proposal=True,
                        episode=reading.episode,
                        source={**reading.source, "extracted_by": label},
                        # A relation is reviewed like the fact it rides on: it becomes an
                        # edge only when the proposal is approved.
                        related_to=[t for t, r in item_links if r == "related_to"],
                        part_of=[t for t, r in item_links if r == "part_of"],
                    )
                except IngatanError:
                    rejected += 1
                    continue
                # `remember` marks the episode itself when it is given one.
                marked = marked or reading.episode is not None
            kept += 1
            proposed += 1
            linked += len(item_links)
            texts.append(clean)
        if not dry_run and not marked:
            # Read: do not send it to the model again, whether or not it held a fact.
            reading.mark_read()

    return ExtractReport(
        episodes_read=0 if documents else len(readings),
        proposed=proposed,
        rejected=rejected,
        errors=errors,
        known=known,
        last_error=last_error,
        texts=tuple(texts),
        model=label,
        notes_read=len(readings) if documents else 0,
        relations=linked,
    )
