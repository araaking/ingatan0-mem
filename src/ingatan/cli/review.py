"""Showing one open conflict pair and reading one answer (`ingatan conflicts review`).

What an answer does lives in `facts.review`. This module only puts a pair on the screen and
turns what was typed into one of the things the command can do.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingatan.facts._util import loads_dict
from ingatan.facts.review import OpenConflict
from ingatan.scope import Context, find_fact

__all__ = ["ANSWERS", "MENU", "Answer", "describe", "parse_answer", "parse_side"]

# The letters that resolve a pair, and the action of `facts.review.ACTIONS` each stands for.
ANSWERS = {"k": "keep_both", "n": "keep_newer", "o": "keep_older"}
MENU = (
    "[k]eep both and link them, [n]ewer replaces older, [o]lder replaces newer, "
    "[f]orget one, [s]kip, [q]uit. A capital K, N or O answers for every pair that remains."
)


@dataclass(frozen=True)
class Answer:
    """One typed answer: a letter of `ANSWERS` or `f`, `s`, `q`, and whether it was typed
    as a capital (which means "and for all that remain")."""

    letter: str
    for_all: bool


def describe(ctx: Context, pair: OpenConflict, position: int, total: int) -> list[str]:
    """Both facts of `pair` in full, the newer one first, labelled so no id has to be typed."""
    lines = [f"Pair {position} of {total}  (#{pair.id}, opened {pair.created_at[:10]})"]
    for label, fact_id in (("NEWER", pair.newer_id), ("OLDER", pair.older_id)):
        found = find_fact(ctx, fact_id)
        if found is None:
            lines.append(f"  {label}  {fact_id}  (no longer there)")
            continue
        row = found[1]
        source = loads_dict(str(row["source"]))
        written_by = source.get("agent") or row["agent"]
        origin = source.get("document") or source.get("session")
        head = f"  {label}  {str(row['valid_from'])[:10]}  [{row['tier']}]  by {written_by}"
        if origin:
            head += f"  from {origin}"
        lines.append(f"{head}  ({fact_id})")
        lines.append(f"      {row['text']}")
    return lines


def parse_answer(raw: str) -> Answer | None:
    """The answer in `raw`, or None when its first letter is not one of the menu's."""
    typed = raw.strip()[:1]
    letter = typed.lower()
    if letter not in (*ANSWERS, "f", "s", "q"):
        return None
    return Answer(letter=letter, for_all=typed.isupper())


def parse_side(raw: str) -> str | None:
    """`newer` or `older` from what was typed (`n`, `o` or the word), else None."""
    typed = raw.strip().lower()
    if typed in ("n", "newer"):
        return "newer"
    if typed in ("o", "older"):
        return "older"
    return None
