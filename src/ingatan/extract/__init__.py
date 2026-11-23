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
from dataclasses import dataclass
from typing import Any

from ingatan import episodes, facts, timeutil
from ingatan.db import fact_text_key
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

_NUMBER = re.compile(r"\d[\d.,]*")
_KINDS = ("stable", "volatile")


@dataclass(frozen=True)
class ExtractReport:
    """What one pass did. `rejected` are items of the model that were dropped, `known` are
    items that repeat a fact or a pending proposal word for word, `errors` are episodes the
    model could not be read for (they stay pending). `texts` are the facts proposed, or,
    in a dry run, the ones that would be."""

    episodes_read: int
    proposed: int
    rejected: int
    errors: int
    known: int = 0
    last_error: str = ""
    texts: tuple[str, ...] = ()
    model: str = ""


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


def extract_pending(ctx: Context, *, limit: int = 10, dry_run: bool = False) -> ExtractReport:
    """Let the configured model read the caller's pending episodes, oldest first.

    Each fact the model states becomes a `business` proposal that carries
    `source.extracted_by`; the episode is then marked processed. An episode the model found
    nothing in is marked processed too. An episode the model could not be read for stays
    pending. With `dry_run` nothing is written: the report says what would be proposed.

    Raises `IngatanError` when no model is configured.
    """
    cfg = ctx.scope.llm
    llm.ensure_configured(cfg)
    label = llm.model_label(cfg)
    read = proposed = rejected = errors = known = 0
    last_error = ""
    texts: list[str] = []
    seen = _known_keys(ctx)

    for episode in episodes.pending(ctx, limit):
        read += 1
        try:
            # Outside any transaction, and redacted once more: an episode written by an
            # older version or another tool may hold what today's patterns would catch.
            answer = llm.chat(
                cfg,
                prompt.SYSTEM,
                prompt.user_message(redact(episode.user_text), redact(episode.assistant_text)),
            )
            items = prompt.parse_answer(answer)
        except (IngatanError, ValueError) as exc:
            errors += 1
            last_error = exc.message if isinstance(exc, IngatanError) else f"JSON: {exc}"
            continue

        stated = _numbers(f"{episode.user_text}\n{episode.assistant_text}")
        kept = 0
        for item in items:
            text = item.get("text") if isinstance(item, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_FACT_CHARS:
                rejected += 1
                continue
            clean = redact(text).strip()
            if not _numbers(clean) <= stated:
                rejected += 1  # a value the episode does not contain was made up
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
                        episode=episode.id,
                        source={"session": episode.session, "extracted_by": label},
                    )
                except IngatanError:
                    rejected += 1
                    continue
            kept += 1
            proposed += 1
            texts.append(clean)
        if not dry_run and kept == 0:
            # Read, and nothing in it to propose: do not send it to the model again.
            episodes.mark_processed(ctx, episode.id)

    return ExtractReport(
        episodes_read=read,
        proposed=proposed,
        rejected=rejected,
        errors=errors,
        known=known,
        last_error=last_error,
        texts=tuple(texts),
        model=label,
    )
