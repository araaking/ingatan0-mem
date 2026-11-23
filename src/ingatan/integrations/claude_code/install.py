"""Add the Ingatan MCP server to Claude Code's project file `.mcp.json`.

Nothing here writes on its own. `mcp_entry_change` works out the change, the caller shows
it (`diff_lines`) and asks, and only then calls `write_change`. The file belongs to another
application, so everything in it that is not the `ingatan` entry is carried over as it is.

An `ingatan` entry that already starts the server, in whatever way (`uvx ingatan serve`, a
full path, a URL), is carried over as well: the path or the launcher in it is somebody's
choice, and may be the thing that makes it work. `own_start` says how such an entry starts
the server, so that the caller can report it.
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
    "own_start",
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


def _read(project_dir: Path) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    """The file's path, its text, its content and its `mcpServers` object (added if absent)."""
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
    return path, before, data, servers  # pyright: ignore[reportUnknownVariableType]


def _start_of(entry: Any) -> str | None:
    """How an entry starts a server: its command line or its URL. None when it has neither."""
    if not isinstance(entry, dict):
        return None
    command, args, url = entry.get("command"), entry.get("args"), entry.get("url")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(command, str) and command.strip():
        words = [str(arg) for arg in args] if isinstance(args, list) else []  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return " ".join([command, *words])
    if isinstance(url, str) and url.strip():
        return url
    return None


_DEFAULT_START = _start_of(SERVER_ENTRY)


def own_start(project_dir: Path) -> str | None:
    """How the `ingatan` entry of `<project_dir>/.mcp.json` starts the server, when that is
    not `ingatan serve`: its command line, or its URL. None when there is no such entry.

    Raises `ConfigError` when the file cannot be read as JSON.
    """
    start = _start_of(_read(project_dir)[3].get(SERVER_NAME))
    return None if start == _DEFAULT_START else start


def mcp_entry_change(project_dir: Path) -> ConfigChange | None:
    """The change that makes `<project_dir>/.mcp.json` run `ingatan serve`.

    None when the file already has an `ingatan` entry that starts a server: the default
    one, or one of the user's own (see `own_start`), which is never rewritten. An entry
    that starts nothing (an `env` block alone, for example) is completed and keeps its
    other keys. Raises `ConfigError` when the file cannot be merged into.
    """
    path, before, data, servers = _read(project_dir)
    existing = servers.get(SERVER_NAME)
    if _start_of(existing) is not None:
        return None
    entry: dict[str, Any] = (
        {**existing, **SERVER_ENTRY} if isinstance(existing, dict) else dict(SERVER_ENTRY)
    )
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
