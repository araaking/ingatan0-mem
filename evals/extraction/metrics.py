"""Cases with a known answer, and the judging of what was extracted against them.

A case is a few captured exchanges, the facts a careful reader would take from them, and
things that must not come out. Judging uses no model: an extracted fact is correct when it
contains every word and every value of one expected fact, and invented otherwise.

- correct share: extracted facts that match an expected fact, out of all extracted.
- invented share: extracted facts that match none (or contain a forbidden word or value).
- found share: expected facts that at least one extracted fact matches, out of all expected.

"Extracted" is what reached the proposal queue, after Ingatan's own checks: that is what a
user would be asked to approve.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Case",
    "CaseResult",
    "Expected",
    "Totals",
    "judge",
    "load_cases",
    "numbers",
    "parse_case",
    "totals",
]

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_NUMBER = re.compile(r"\d[\d.,]*")
_FIELDS = {"id", "title", "episodes", "expected", "forbidden"}


def numbers(text: str) -> set[str]:
    """Every number in `text` without separators, as `ingatan.extract` compares them."""
    return {m.group(0).replace(".", "").replace(",", "") for m in _NUMBER.finditer(text)}


@dataclass(frozen=True)
class Expected:
    """One fact the episodes state: the words and the values a statement of it must hold."""

    name: str
    all_of: tuple[str, ...]
    values: tuple[str, ...]

    def matches(self, text: str) -> bool:
        low = text.casefold()
        stated = numbers(text)
        return all(word.casefold() in low for word in self.all_of) and all(
            numbers(value) <= stated for value in self.values
        )


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    episodes: tuple[tuple[str, str], ...]
    expected: tuple[Expected, ...]
    forbidden: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    """`dropped` are items of the model that Ingatan refused before the queue; `errors` are
    episodes the model could not be read for."""

    case_id: str
    extracted: tuple[str, ...]
    correct: int
    invented: int
    found: int
    expected: int
    dropped: int = 0
    errors: int = 0


@dataclass(frozen=True)
class Totals:
    extracted: int
    correct: int
    invented: int
    expected: int
    found: int
    dropped: int
    errors: int

    @property
    def correct_share(self) -> float | None:
        return self.correct / self.extracted if self.extracted else None

    @property
    def invented_share(self) -> float | None:
        return self.invented / self.extracted if self.extracted else None

    @property
    def found_share(self) -> float | None:
        return self.found / self.expected if self.expected else None


def _strings(where: str, name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):  # pyright: ignore[reportUnknownVariableType]
        raise ValueError(f"{where}: '{name}' must be a list of non-empty strings")
    return tuple(str(v) for v in value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def parse_case(raw: Any) -> Case:
    """Validate one case. Raises `ValueError` that names the case and the field."""
    if not isinstance(raw, dict):
        raise ValueError("a case must be one JSON object")
    data: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    where = str(data.get("id") or "<case without id>")
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown field(s): {', '.join(sorted(unknown))}")
    if not isinstance(data.get("id"), str) or not _ID.match(data["id"]):
        raise ValueError(f"{where}: 'id' must be lowercase letters, digits and dashes")
    episodes_raw = data.get("episodes")
    if not isinstance(episodes_raw, list) or not episodes_raw:
        raise ValueError(f"{where}: 'episodes' must be a non-empty list")
    episodes: list[tuple[str, str]] = []
    for item in episodes_raw:  # pyright: ignore[reportUnknownVariableType]
        if (
            not isinstance(item, dict)
            or set(item) != {"user", "assistant"}  # pyright: ignore[reportUnknownArgumentType]
            or not all(isinstance(v, str) for v in item.values())  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        ):
            raise ValueError(f"{where}: an episode needs exactly 'user' and 'assistant' texts")
        episodes.append((str(item["user"]), str(item["assistant"])))
    expected: list[Expected] = []
    expected_raw = data.get("expected", [])
    if not isinstance(expected_raw, list):
        raise ValueError(f"{where}: 'expected' must be a list")
    for item in expected_raw:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):  # pyright: ignore[reportUnknownMemberType]
            raise ValueError(f"{where}: an expected fact needs a 'name'")
        all_of = _strings(where, "all_of", item.get("all_of", []))  # pyright: ignore[reportUnknownMemberType]
        values = _strings(where, "values", item.get("values", []))  # pyright: ignore[reportUnknownMemberType]
        if not all_of and not values:
            raise ValueError(f"{where}: expected fact '{item['name']}' needs 'all_of' or 'values'")
        expected.append(Expected(str(item["name"]), all_of, values))
    return Case(
        id=str(data["id"]),
        title=str(data.get("title") or ""),
        episodes=tuple(episodes),
        expected=tuple(expected),
        forbidden=_strings(where, "forbidden", data.get("forbidden", [])),
    )


def load_cases(directory: Path) -> list[Case]:
    cases = [
        parse_case(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"case id '{case.id}' is used twice")
        seen.add(case.id)
    return cases


def judge(
    case: Case, extracted: list[str] | tuple[str, ...], *, dropped: int = 0, errors: int = 0
) -> CaseResult:
    """Count the correct and the invented among `extracted`, and the expected that were found."""
    correct = invented = 0
    found: set[str] = set()
    for text in extracted:
        low = text.casefold()
        stated = numbers(text)
        barred = any(
            (numbers(item) and numbers(item) <= stated)
            or (not numbers(item) and item.casefold() in low)
            for item in case.forbidden
        )
        matched = [] if barred else [e.name for e in case.expected if e.matches(text)]
        if matched:
            correct += 1
            found.update(matched)
        else:
            invented += 1
    return CaseResult(
        case_id=case.id,
        extracted=tuple(extracted),
        correct=correct,
        invented=invented,
        found=len(found),
        expected=len(case.expected),
        dropped=dropped,
        errors=errors,
    )


def totals(results: list[CaseResult]) -> Totals:
    return Totals(
        extracted=sum(len(r.extracted) for r in results),
        correct=sum(r.correct for r in results),
        invented=sum(r.invented for r in results),
        expected=sum(r.expected for r in results),
        found=sum(r.found for r in results),
        dropped=sum(r.dropped for r in results),
        errors=sum(r.errors for r in results),
    )
