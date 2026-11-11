"""Reading a brief back: sections, fact lines and the pending-proposal count."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["BriefLine", "ParsedBrief", "parse_brief"]

_HEADING = re.compile(r"^##\s+(.+?)\s*$")
_FACT_LINE = re.compile(r"^-\s*\[(f_[0-9a-f]+)\]\s*(.*)$")
_PENDING = re.compile(r"Pending proposals[^:\n]*:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class BriefLine:
    id: str
    text: str
    section: str


@dataclass(frozen=True)
class ParsedBrief:
    lines: list[BriefLine] = field(default_factory=lambda: [])
    pending: int = 0

    def ids_in(self, section: str) -> list[str]:
        return [ln.id for ln in self.lines if ln.section == section]

    def all_ids(self) -> list[str]:
        return [ln.id for ln in self.lines]


def parse_brief(text: str) -> ParsedBrief:
    """Section names are lower-cased (`check first`, `global`, `business`, `agent`, ...).

    Only fact lines (`- [f_...] text`) count. A brief that does not state a pending count
    has none.
    """
    section = ""
    lines: list[BriefLine] = []
    for raw in text.splitlines():
        heading = _HEADING.match(raw)
        if heading:
            section = heading.group(1).casefold()
            continue
        fact = _FACT_LINE.match(raw.strip())
        if fact and section != "episodes to digest":
            lines.append(BriefLine(fact.group(1), fact.group(2), section))
    pending = _PENDING.search(text)
    return ParsedBrief(lines, int(pending.group(1)) if pending else 0)
