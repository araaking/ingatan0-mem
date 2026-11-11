"""Claude Code hook entry points (`ingatan hook <event>`), T-19.

Events:

- `session-start`: print the brief as SessionStart `additionalContext` (FR-47). With the
  digest offer (FR-49) except when the session continues (`source` is `compact` or `resume`).
- `user-prompt`: keep the (redacted) prompt in `<home>/pending/<session_id>.json`.
- `stop`: pair the stored prompt with the final reply and call `episodes.capture` (FR-44).
  Only those two texts are ever stored: no tool input or output, no file contents.

Rules (NFR-14): `run` always returns 0 (never 2, which would block), never prints a
traceback, prints nothing but the intended output, and logs failures to
`<home>/capture-errors.log`. A hook with no resolvable scope does nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ingatan import episodes, timeutil
from ingatan.errors import ConfigError
from ingatan.redact import redact
from ingatan.scope import Scope, data_home, find_config, open_context, resolve_scope

__all__ = ["EVENTS", "run"]

ERROR_LOG_NAME = "capture-errors.log"
PENDING_DIR = "pending"
PENDING_MAX_AGE_SECONDS = 7 * 24 * 3600
# Sources on which the session continues: inject the brief but do not offer episodes again.
NO_OFFER_SOURCES = frozenset({"compact", "resume"})
# Transcript fallback: read the file from the end in growing chunks up to this size.
_TAIL_START = 256 * 1024
_TAIL_MAX = 64 * 1024 * 1024


# --------------------------------------------------------------------------- plumbing


def _read_payload() -> dict[str, Any]:
    """Hook JSON from stdin. Bytes, decoded as UTF-8 (Windows console code pages are not
    trusted). Anything that is not a JSON object gives an empty payload."""
    stream = sys.stdin
    raw: bytes
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        raw = buffer.read()
    else:
        text = stream.read()
        raw = text.encode("utf-8", errors="replace") if isinstance(text, str) else bytes(text)
    data = json.loads(raw.decode("utf-8", errors="replace").lstrip("﻿"))
    return data if isinstance(data, dict) else {}  # pyright: ignore[reportUnknownVariableType]


def _log_failure(home: Path | None, exc: BaseException, where: str) -> None:
    try:
        folder = home if home is not None else data_home()
        folder.mkdir(parents=True, exist_ok=True)
        message = redact(f"hook {where}: {type(exc).__name__}: {exc}")
        message = message.replace("\r", " ").replace("\n", " ")
        with (folder / ERROR_LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(f"{timeutil.iso(timeutil.now())}\t{message}\n")
    except Exception:  # a hook must never fail loudly
        pass


def _scope(payload: Mapping[str, Any]) -> Scope | None:
    """Scope from the hook's `cwd` (nearest `.ingatan.toml`) with env overrides, or None."""
    cwd = payload.get("cwd")
    start = Path(cwd) if isinstance(cwd, str) and cwd.strip() else None
    try:
        return resolve_scope(start, os.environ)
    except ConfigError as exc:
        # No configuration at all is the normal state of a project that does not use
        # Ingatan: stay silent. A configuration that exists but is broken is logged.
        configured = (
            find_config(start) is not None
            or bool(os.environ.get("INGATAN_BUSINESS"))
            or bool(os.environ.get("INGATAN_AGENT"))
        )
        if configured:
            _log_failure(None, exc, "scope")
        return None


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _safe_name(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:128]


def _pending_path(scope: Scope, session_id: str) -> Path:
    return scope.home / PENDING_DIR / f"{_safe_name(session_id)}.json"


# --------------------------------------------------------------------------- handlers


def _session_start(payload: dict[str, Any], scope: Scope) -> str | None:
    from ingatan import brief  # lazy: the brief module is not needed by capture events

    offer = _text(payload.get("source")) not in NO_OFFER_SOURCES
    ctx = open_context(scope)
    try:
        text = brief.build(ctx, offer_episodes=offer)
        try:
            episodes.purge(ctx)  # retention (OQ-13); never worth failing the injection for
        except Exception as exc:
            _log_failure(scope.home, exc, "purge")
    finally:
        ctx.close()
    if not text or not text.strip():
        return None
    # ASCII-escaped JSON stays valid whatever encoding the console or pipe uses.
    return json.dumps(
        {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}
    )


def _user_prompt(payload: dict[str, Any], scope: Scope) -> str | None:
    session_id = _text(payload.get("session_id")).strip()
    prompt = _text(payload.get("prompt"))
    if not session_id or not prompt.strip():
        return None
    path = _pending_path(scope, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _prune_pending(path.parent)
    stored = redact(prompt)[: episodes.TRUNCATE_AT]
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps({"prompt": stored, "ts": timeutil.iso(timeutil.now())}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp, path)
    return None


def _prune_pending(folder: Path) -> None:
    """Drop pending files nobody came back for (an interrupted turn never reaches Stop)."""
    cutoff = time.time() - PENDING_MAX_AGE_SECONDS
    for old in folder.glob("*.json"):
        with contextlib.suppress(OSError):
            if old.stat().st_mtime < cutoff:
                old.unlink()


def _stop(payload: dict[str, Any], scope: Scope) -> str | None:
    if payload.get("stop_hook_active") is True:
        return None
    session_id = _text(payload.get("session_id")).strip()
    if not session_id:
        return None
    path = _pending_path(scope, session_id)
    try:
        pending = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    user_text = _text(pending.get("prompt")) if isinstance(pending, dict) else ""  # pyright: ignore[reportUnknownMemberType]
    try:
        if not user_text.strip():
            return None
        reply = _text(payload.get("last_assistant_message"))
        if not reply.strip():
            reply = _last_assistant_text(_text(payload.get("transcript_path")))
        # `capture` redacts and truncates, logs its own failures and never raises.
        episodes.capture(scope, user_text, reply, session=session_id)
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
    return None


def _content_text(content: Any) -> str:
    """Only `text` blocks. `thinking`, `tool_use` and `tool_result` are never read."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:  # pyright: ignore[reportUnknownVariableType]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"  # pyright: ignore[reportUnknownMemberType]
            and isinstance(block.get("text"), str)  # pyright: ignore[reportUnknownMemberType]
        ):
            parts.append(block["text"].strip())  # pyright: ignore[reportUnknownMemberType]
    return "\n".join(p for p in parts if p)


def _last_assistant_text(transcript_path: str) -> str:
    """Fallback when Stop carries no `last_assistant_message`: scan the transcript backward.

    Takes the last non-sidechain `assistant` line that has text blocks. Transcripts can be
    huge, so the file is read from the end in growing chunks.
    """
    if not transcript_path:
        return ""
    try:
        with open(transcript_path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            chunk = _TAIL_START
            while True:
                start = max(0, size - chunk)
                handle.seek(start)
                data = handle.read(size - start)
                lines = data.split(b"\n")
                if start > 0:
                    lines = lines[1:]  # the first line may be cut in half
                found = _scan_lines(lines)
                if found or start == 0 or chunk >= _TAIL_MAX:
                    return found
                chunk *= 4
    except OSError:
        return ""


def _scan_lines(lines: list[bytes]) -> str:
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":  # pyright: ignore[reportUnknownMemberType]
            continue
        if entry.get("isSidechain"):  # pyright: ignore[reportUnknownMemberType]
            continue
        message = entry.get("message")  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(message, dict):
            continue
        text = _content_text(message.get("content"))  # pyright: ignore[reportUnknownMemberType]
        if text:
            return text
    return ""


_Handler = Callable[[dict[str, Any], Scope], "str | None"]
EVENTS: dict[str, _Handler] = {
    "session-start": _session_start,
    "user-prompt": _user_prompt,
    "stop": _stop,
}


# --------------------------------------------------------------------------- entry point


def run(event: str) -> int:
    """Run one hook event. Always returns 0 and prints only the intended output."""
    output: str | None = None
    home: Path | None = None
    try:
        handler = EVENTS.get(event)
        if handler is None:
            return 0
        payload = _read_payload()
        scope = _scope(payload)
        if scope is None:
            return 0
        home = scope.home
        # Anything a library prints by accident must not reach Claude Code's stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            output = handler(payload, scope)
    except Exception as exc:
        _log_failure(home, exc, event)
        output = None
    if output:
        try:
            with contextlib.suppress(Exception):
                sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
            sys.stdout.write(output)
            sys.stdout.flush()
        except Exception as exc:
            _log_failure(home, exc, event)
    return 0
