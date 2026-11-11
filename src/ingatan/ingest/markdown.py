"""Split Markdown into heading sections (ATX headings only)."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Section", "split_sections"]

_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


def split_sections(markdown: str, fallback_heading: str) -> list[Section]:
    """One section per heading. Text before the first heading is headed `fallback_heading`.

    `#` lines inside fenced code blocks are not headings. Sections whose body is empty
    are left out.
    """
    sections: list[Section] = []
    heading = fallback_heading
    body: list[str] = []
    fence: str | None = None  # the opening fence string while inside a code block

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(Section(heading, text))

    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        fence_match = _FENCE.match(line)
        if fence is not None:
            if (
                fence_match
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= len(fence)
                and not line.strip().strip(fence[0])
            ):
                fence = None
            body.append(line)
            continue
        if fence_match:
            fence = fence_match.group(1)
            body.append(line)
            continue
        heading_match = _HEADING.match(line)
        if heading_match:
            flush()
            title = _CLOSING_HASHES.sub("", (heading_match.group(2) or "").strip()).strip()
            heading = title or fallback_heading
            body = []
            continue
        body.append(line)
    flush()
    return sections
