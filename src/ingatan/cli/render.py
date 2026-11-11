"""Plain-text rendering of log entries and diffs for the CLI."""

from __future__ import annotations

import json
from typing import Any

from ingatan.audit import Diff, LogEntry

__all__ = ["format_value", "log_line", "render_diff", "shorten"]

NO_CONTENT = "(no content: the fact was forgotten)"


def shorten(text: str, width: int = 70) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _summary(entry: LogEntry) -> str:
    if entry.change is None:
        return NO_CONTENT
    after: dict[str, Any] = entry.change.get("after") or {}
    before: dict[str, Any] = entry.change.get("before") or {}
    text = after.get("text") or (after.get("payload") or {}).get("text") or before.get("text")
    if text:
        return shorten(str(text))
    if entry.op == "confirm" and after.get("reason"):
        return "reason: " + shorten(str(after["reason"]))
    if entry.op == "supersede":
        if after.get("superseded_by"):
            return f"superseded by {after['superseded_by']}"
        if after.get("pending_proposal"):
            return f"change waits for approval ({after['pending_proposal']})"
    if entry.op == "undo":
        return f"undoes entry {after.get('undoes')} ({after.get('undone_op')})"
    return ""


def log_line(entry: LogEntry) -> str:
    target = entry.fact_id or "-"
    tier = entry.tier or "-"
    head = f"#{entry.id}  {entry.ts}  {entry.agent}  {entry.op}  {target}  [{tier}]"
    summary = _summary(entry)
    return f"{head}  {summary}" if summary else head


def render_diff(diff: Diff, entry: LogEntry) -> str:
    lines = [
        f"Log entry #{entry.id}: {entry.op} of {entry.fact_id or '-'}",
        f"When: {entry.ts}   By: {entry.agent}   Tier: {entry.tier or '-'}",
        "",
    ]
    if diff.before is None and diff.after is None:
        lines.append("No content is stored for this entry: the content was forgotten.")
        lines.append("There is nothing to compare.")
        return "\n".join(lines)
    before = diff.before
    after = diff.after
    if before is None:
        lines.append("Before: nothing (this entry created something).")
    if after is None:
        lines.append("After: nothing (this entry removed something).")
    keys: list[str] = list(dict.fromkeys([*(before or {}).keys(), *(after or {}).keys()]))
    unchanged: list[str] = []
    for key in keys:
        old_present = before is not None and key in before
        new_present = after is not None and key in after
        old = before[key] if old_present and before is not None else None
        new = after[key] if new_present and after is not None else None
        if old_present and new_present and old == new:
            unchanged.append(key)
            continue
        lines.append(f"{key}:")
        if old_present:
            lines.append(f"  - {format_value(old)}")
        if new_present:
            lines.append(f"  + {format_value(new)}")
    if unchanged:
        lines.append("")
        lines.append("Unchanged: " + ", ".join(unchanged))
    return "\n".join(lines)
