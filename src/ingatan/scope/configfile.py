"""Change one key of a TOML config file, leaving every other byte as it was.

The standard library reads TOML but cannot write it, and writing a file back from the
parsed dict would lose the user's comments, order and line endings. So the text is edited
by line and the result is parsed again: unless it says exactly what the old text said plus
the one new value, nothing is written and the user is told what to put in by hand.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from ingatan.errors import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

__all__ = ["apply", "updated_text"]

_TABLE_START = re.compile(r"^\s*\[")


def _render(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # A JSON string is a valid TOML basic string: both escape `"`, `\` and control characters
    # the same way.
    return json.dumps(value, ensure_ascii=False)


def _parse(text: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"The file is not valid TOML: {exc}") from exc


def _split_ending(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body) :]


def _split_comment(rest: str) -> tuple[str, str]:
    """Split what follows `key =` into the value and the spaces and comment after it."""
    quote = ""
    index = 0
    while index < len(rest):
        char = rest[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "#":
            break
        index += 1
    value = rest[:index].rstrip()
    return value, rest[len(value) :]


def updated_text(text: str, section: str, key: str, value: str | int | bool) -> str:
    """`text` with `key = value` under `[section]`, everything else untouched.

    The key is replaced where it stands, added at the end of an existing section, or the
    section is appended to the file. Raises `ConfigError` when `text` is not valid TOML or
    when the edit cannot be made without changing what the rest of the file says (an inline
    table, a dotted key, a value that spans lines).
    """
    before = _parse(text)
    newline = "\r\n" if "\r\n" in text else "\n"
    rendered = _render(value)
    lines = text.splitlines(keepends=True)
    header = re.compile(rf"^\s*\[\s*{re.escape(section)}\s*\]\s*(#.*)?$")
    key_line = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*)$")

    start = next((i for i, line in enumerate(lines) if header.match(_split_ending(line)[0])), None)
    if start is None:
        out = text
        if out and not out.endswith("\n"):
            out += newline
        if out.strip():
            out += newline
        out += f"[{section}]{newline}{key} = {rendered}{newline}"
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if _TABLE_START.match(lines[i])), len(lines)
        )
        for index in range(start + 1, end):
            body, ending = _split_ending(lines[index])
            found = key_line.match(body)
            if found is not None:
                _old, tail = _split_comment(found.group(2))
                lines[index] = f"{found.group(1)}{rendered}{tail}{ending}"
                break
        else:
            last = max(i for i in range(start, end) if lines[i].strip())
            if not lines[last].endswith("\n"):
                lines[last] += newline
            lines.insert(last + 1, f"{key} = {rendered}{newline}")
        out = "".join(lines)

    by_hand = (
        f"Cannot set {key} under [{section}] without disturbing the rest of the file. "
        f"Edit it by hand: under [{section}] put `{key} = {rendered}`."
    )
    expected = copy.deepcopy(before)
    table = expected.setdefault(section, {})
    if not isinstance(table, dict):
        raise ConfigError(by_hand)
    table[key] = value
    try:
        after = tomllib.loads(out)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(by_hand) from exc
    if after != expected:
        raise ConfigError(by_hand)
    return out


def apply(path: Path, section: str, key: str, value: str | int | bool) -> bool:
    """Set `key = value` under `[section]` in the file at `path`, creating it if missing.

    Returns True when the file was written, False when it already said so. On `ConfigError`
    the file is exactly as it was.
    """
    exists = path.is_file()
    try:
        text = path.read_bytes().decode("utf-8") if exists else ""
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    try:
        new_text = updated_text(text, section, key, value)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc.message}") from exc
    if exists and new_text == text:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(new_text.encode("utf-8"))
        if exists:
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except OSError as exc:
        raise ConfigError(f"Cannot write {path}: {exc}") from exc
    return True
