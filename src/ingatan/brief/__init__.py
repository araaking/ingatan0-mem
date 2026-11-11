"""The session-start brief: what an agent should know before it starts working.

Layout, budget rules and ordering follow ARCHITECTURE.md. The brief only reads facts: it
never raises `use_count`. Pending proposals and document notes never appear in it.
"""

from __future__ import annotations

import contextlib
import re
import sqlite3
from dataclasses import dataclass

from ingatan import episodes, proposals, timeutil
from ingatan.scope import Context
from ingatan.search import is_valid_sql, status_of

__all__ = ["build", "estimate_tokens"]

CHARS_PER_TOKEN = 4
# Below this many spare characters an episode is not worth showing, even cut short.
MIN_EPISODE_CHARS = 120

_CHECK_INTRO = (
    "Do not act on these facts before checking them. Check only those relevant to the "
    "current task: confirm(id, reason) if still true, supersede(old, new) if it changed."
)
_EPISODE_INTRO = (
    "Turn what is worth keeping into facts with remember(..., episode=<id>). "
    "An episode you do not digest is dropped after a few sessions."
)
_SECTIONS = (("global", "## Global"), ("business", "## Business"), ("agent", "## Agent"))
_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """`ceil(len / 4)`: the rough token count used for the budget."""
    return -(-len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class _Item:
    # Smaller means less used, and dropped first when the brief is over budget.
    rank: tuple[int, str, str, str]
    line: str


def _flat(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _item(row: sqlite3.Row, tags: list[str]) -> _Item:
    suffix = f" ({'; '.join(tags)})" if tags else ""
    return _Item(
        rank=(row["use_count"], row["last_used"] or "", row["valid_from"], row["id"]),
        line=f"- [{row['id']}] {_flat(row['text'])}{suffix}",
    )


def _facts(ctx: Context) -> tuple[list[_Item], dict[str, list[_Item]]]:
    """Valid, visible facts: (check-first items, items per tier), most used first."""
    now = timeutil.iso(timeutil.now())
    check: list[_Item] = []
    tiers: dict[str, list[_Item]] = {name: [] for name, _ in _SECTIONS}
    for conn in ctx.connections():
        rows = conn.execute(
            f"SELECT * FROM facts WHERE {is_valid_sql()} AND {ctx.visible_sql(conn)}",
            {"now": now, **ctx.params},
        ).fetchall()
        for row in rows:
            status = status_of(ctx, row)
            tags: list[str] = []
            if "check first" in status:
                if row["kind"] == "volatile" and row["review_after"] and row["review_after"] < now:
                    tags.append(f"review was due {row['review_after'][:10]}")
                if row["pending_proposal"]:
                    tags.append("a change is waiting for the user's approval")
            if "possible conflict" in status:
                tags.append("possible conflict")
            target = check if "check first" in status else tiers[row["tier"]]
            target.append(_item(row, tags))
    check.sort(key=lambda i: i.rank, reverse=True)
    for items in tiers.values():
        items.sort(key=lambda i: i.rank, reverse=True)
    return check, tiers


def _core(header: str, check: list[_Item], omitted: int, tiers: dict[str, list[_Item]]) -> str:
    blocks = [header]
    if check or omitted:
        lines = ["## Check first", _CHECK_INTRO, *(i.line for i in check)]
        if omitted:
            lines.append(
                f"- {omitted} more fact(s) to check are not listed here; "
                'recall shows them with status "check first".'
            )
        blocks.append("\n".join(lines))
    for name, title in _SECTIONS:
        if tiers[name]:
            blocks.append("\n".join([title, *(i.line for i in tiers[name])]))
    return "\n\n".join(blocks)


def _fit_core(header: str, check: list[_Item], tiers: dict[str, list[_Item]], budget: int) -> str:
    limit = budget * CHARS_PER_TOKEN
    tiers = {name: list(items) for name, items in tiers.items()}

    # Drop the least used facts across the three sections. First a coarse cut: whatever lies
    # past `limit` characters of lines can never fit, which keeps this fast for huge stores.
    everyone = sorted(
        ((i, name) for name, items in tiers.items() for i in items),
        key=lambda pair: pair[0].rank,
        reverse=True,
    )
    used = 0
    kept: set[int] = set()
    for position, (item, _) in enumerate(everyone):
        used += len(item.line) + 1
        if used > limit:
            break
        kept.add(position)
    for position, (item, name) in enumerate(everyone):
        if position not in kept:
            tiers[name].remove(item)
    while True:
        text = _core(header, check, 0, tiers)
        if len(text) <= limit:
            return text
        victims = [(items[-1], name) for name, items in tiers.items() if items]
        if not victims:
            break
        _, name = min(victims, key=lambda pair: pair[0].rank)
        tiers[name].pop()

    # Header and "check first" alone are too big: trim that list from the least used.
    shown = list(check)
    used = len(header) + len(_CHECK_INTRO) + 80
    cut = 0
    for position, item in enumerate(shown):
        used += len(item.line) + 1
        if used > limit:
            cut = position
            break
    else:
        cut = len(shown)
    shown = shown[:cut]
    while True:
        text = _core(header, shown, len(check) - len(shown), tiers)
        if len(text) <= limit or not shown:
            break
        shown.pop()
    if len(text) > limit:
        # Even the bare header and the note do not fit. The budget is a hard limit.
        text = text[:limit].rstrip()
    return text


def _peek_episodes(ctx: Context, limit: int) -> list[episodes.Episode]:
    """The episodes `episodes.offer` would return next, without counting an offer."""
    rows = ctx.business.execute(
        "SELECT * FROM episodes WHERE agent = ? AND status = 'pending' ORDER BY id ASC LIMIT ?",
        (ctx.scope.agent, limit),
    ).fetchall()
    return [
        episodes.Episode(
            id=r["id"],
            agent=r["agent"],
            session=r["session"],
            user_text=r["user_text"],
            assistant_text=r["assistant_text"],
            created_at=r["created_at"],
            status=r["status"],
            offered=r["offered"],
            processed_at=r["processed_at"],
        )
        for r in rows
    ]


def _episode_block(
    ep: episodes.Episode, user_limit: int | None, assistant_limit: int | None
) -> str:
    def clip(text: str, limit: int | None) -> str:
        flat = text.strip()
        if limit is None or len(flat) <= limit:
            return flat
        return flat[: max(limit - 1, 0)].rstrip() + "…"

    return (
        f"### Episode {ep.id} ({ep.created_at[:10]})\n"
        f"User: {clip(ep.user_text, user_limit)}\n"
        f"Assistant: {clip(ep.assistant_text, assistant_limit)}"
    )


def _with_episodes(core: str, found: list[episodes.Episode], budget: int) -> tuple[str, int]:
    """`core` plus as many episodes as fit. Returns the text and how many were used."""
    limit = budget * CHARS_PER_TOKEN
    head = f"{core}\n\n## Episodes to digest\n{_EPISODE_INTRO}"
    for count in range(len(found), 0, -1):
        text = head + "\n" + "\n".join(_episode_block(e, None, None) for e in found[:count])
        if len(text) <= limit:
            return text, count
    if found:
        # Not even one whole episode fits. Cut the first one down so it is not starved
        # forever by a long conversation; the agent can still judge it from the start.
        first = found[0]
        spare = limit - len(head) - len(_episode_block(first, 0, 0)) - 2
        if spare >= MIN_EPISODE_CHARS:
            half = spare // 2
            return head + "\n" + _episode_block(first, half, half), 1
    return core, 0


def build(ctx: Context, *, offer_episodes: bool = True) -> str:
    """The brief for the context's business and agent, never over `scope.token_budget`."""
    budget = max(ctx.scope.token_budget, 1)
    with contextlib.suppress(sqlite3.Error):  # housekeeping must not stop the brief
        episodes.purge(ctx)

    header = (
        f"# Ingatan brief: {ctx.scope.business} / {ctx.scope.agent}\n"
        f"Pending proposals waiting for the user: {proposals.pending_count(ctx)}"
    )
    check, tiers = _facts(ctx)
    text = _core(header, check, 0, tiers)
    if estimate_tokens(text) > budget:
        text = _fit_core(header, check, tiers, budget)

    if offer_episodes:
        _, count = _with_episodes(text, _peek_episodes(ctx, episodes.MAX_OFFER), budget)
        if count:
            # Only what fits is offered, so nothing is counted as offered without being shown.
            final, _ = _with_episodes(text, episodes.offer(ctx, limit=count), budget)
            text = final
    return text
