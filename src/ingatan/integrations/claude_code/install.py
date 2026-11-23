"""Add the Ingatan MCP server to Claude Code's project file `.mcp.json`.

Nothing here writes on its own. `mcp_entry_change` works out the change, the caller shows
it (`diff_lines`) and asks, and only then calls `write_change`. The file belongs to another
application, so everything in it that is not the `ingatan` entry is carried over as it is.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingatan.errors import ConfigError

__all__ = [
    "MCP_FILE",
    "SERVER_ENTRY",
    "SERVER_NAME",
    "ConfigChange",
    "diff_lines",
    "mcp_entry_change",
    "write_change",
]

MCP_FILE = ".mcp.json"
SERVER_NAME = "ingatan"
SERVER_ENTRY: dict[str, Any] = {"command": "ingatan", "args": ["serve"]}
BY_HAND = "Add the server yourself: claude mcp add --scope project ingatan -- ingatan serve"

_FIRST_INDENT = re.compile(r"^([ \t]+)\S", re.MULTILINE)


@dataclass(frozen=True)
class ConfigChange:
    """A file as it is now and as it would be. `before` is empty when the file is missing."""

    path: Path
    before: str
    after: str


def mcp_entry_change(project_dir: Path) -> ConfigChange | None:
    """The change that makes `<project_dir>/.mcp.json` run `ingatan serve`.

    None when the file already does. An existing `ingatan` entry keeps its other keys (an
    `env` block, for example). Raises `ConfigError` when the file cannot be merged into.
    """
    path = project_dir / MCP_FILE
    try:
        before = path.read_bytes().decode("utf-8") if path.is_file() else ""
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}. {BY_HAND}") from exc
    data: dict[str, Any] = {}
    if before.strip():
        try:
            parsed = json.loads(before)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON ({exc}). {BY_HAND}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"{path} does not hold a JSON object. {BY_HAND}")
        data = parsed  # pyright: ignore[reportUnknownVariableType]
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"`mcpServers` in {path} is not a JSON object. {BY_HAND}")
    existing = servers.get(SERVER_NAME)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    entry: dict[str, Any] = (
        {**existing, **SERVER_ENTRY} if isinstance(existing, dict) else dict(SERVER_ENTRY)
    )
    if existing == entry:
        return None
    servers[SERVER_NAME] = entry
    indent_match = _FIRST_INDENT.search(before)
    indent: str | int = indent_match.group(1) if indent_match else 2
    after = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    if "\r\n" in before:
        after = after.replace("\n", "\r\n")
    return ConfigChange(path=path, before=before, after=after)


def diff_lines(change: ConfigChange) -> list[str]:
    """The change as a unified diff, one line per item, to show before asking."""
    return list(
        difflib.unified_diff(
            change.before.splitlines(),
            change.after.splitlines(),
            fromfile=f"{change.path} (now)",
            tofile=f"{change.path} (after)",
            lineterm="",
        )
    )


def write_change(change: ConfigChange) -> None:
    """Write `change.after` to `change.path`, replacing the file in one step."""
    try:
        change.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = change.path.with_name(change.path.name + ".tmp")
        temporary.write_bytes(change.after.encode("utf-8"))
        os.replace(temporary, change.path)
    except OSError as exc:
        raise ConfigError(f"Cannot write {change.path}: {exc}") from exc
