"""Static bearer tokens for HTTP mode (FR-58). Each token maps to one (business, agent) scope.

This is the HTTP equivalent of `.ingatan.toml`: the client cannot choose its scope. Only the
SHA-256 hash of a token is stored, in `<INGATAN_HOME>/tokens.json`. This module is pure
standard library (no `mcp`, `starlette` or `pydantic`), so the CLI can manage tokens cheaply.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ingatan import timeutil
from ingatan.errors import ConfigError, NotFoundError
from ingatan.scope import Scope, validate_agent, validate_business

__all__ = [
    "TOKEN_PREFIX",
    "Token",
    "TokenStore",
    "add",
    "hash_token",
    "list_tokens",
    "revoke",
    "tokens_path",
]

TOKEN_PREFIX = "igt_"
FILE_NAME = "tokens.json"


@dataclass(frozen=True)
class Token:
    """One stored record. `hash` is the SHA-256 hex digest of the raw token."""

    id: str
    hash: str
    business: str
    agent: str
    label: str
    created: str


def tokens_path(home: Path) -> Path:
    return home / FILE_NAME


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _record(item: object) -> Token:
    if not isinstance(item, dict):
        raise ConfigError("A record in tokens.json is not an object.")
    data: dict[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
    try:
        values = {name: data[name] for name in ("id", "hash", "business", "agent", "created")}
    except KeyError as exc:
        raise ConfigError(f"A record in tokens.json has no {exc.args[0]!r} field.") from exc
    label = data.get("label", "")
    if not all(isinstance(v, str) for v in (*values.values(), label)):
        raise ConfigError("A record in tokens.json has a field that is not text.")
    return Token(label=label, **values)


def _read(path: Path) -> list[Token]:
    """The records in the file. A missing file means no tokens."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    items = document.get("tokens") if isinstance(document, dict) else None  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(items, list):
        raise ConfigError(f'{path} must hold an object with a "tokens" list.')
    return [_record(item) for item in items]  # pyright: ignore[reportUnknownVariableType]


def _write(path: Path, records: list[Token]) -> None:
    """Atomic rewrite: a temp file in the same folder, then `os.replace`.

    `mkstemp` creates the file with mode 0600 on POSIX; Windows has no such mode bits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"tokens": [asdict(r) for r in records]}, ensure_ascii=False, indent=2, sort_keys=True
    )
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".tokens-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def new_secret() -> str:
    """`igt_` plus 32 random bytes, URL-safe base64 without padding."""
    body = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    return TOKEN_PREFIX + body


def add(home: Path, business: str, agent: str, label: str = "") -> tuple[str, Token]:
    """Create a token for a scope. Returns the raw token (shown once) and its stored record."""
    business = validate_business(business)
    agent = validate_agent(agent)
    path = tokens_path(home)
    records = _read(path)
    taken = {r.id for r in records}
    token_id = f"t_{secrets.token_hex(4)}"
    while token_id in taken:  # pragma: no cover - 1 in 4 billion
        token_id = f"t_{secrets.token_hex(4)}"
    raw = new_secret()
    record = Token(
        id=token_id,
        hash=hash_token(raw),
        business=business,
        agent=agent,
        label=label.strip(),
        created=timeutil.iso(timeutil.now()),
    )
    _write(path, [*records, record])
    return raw, record


def list_tokens(home: Path) -> list[Token]:
    return _read(tokens_path(home))


def revoke(home: Path, token_id: str) -> Token:
    """Remove a record. Raises `NotFoundError` for an unknown id."""
    path = tokens_path(home)
    records = _read(path)
    kept = [r for r in records if r.id != token_id]
    if len(kept) == len(records):
        raise NotFoundError(f"No token {token_id}. Run `ingatan token list` to see the ids.")
    removed = next(r for r in records if r.id == token_id)
    _write(path, kept)
    return removed


class TokenStore:
    """Resolves a raw token to its `Scope`, re-reading the file when it changes on disk."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self._signature: tuple[int, int, int] | None = None
        self._loaded = False
        self._records: list[Token] = []

    def _refresh(self) -> None:
        path = tokens_path(self.home)
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            signature = None
        if self._loaded and signature == self._signature:
            return
        try:
            records = _read(path) if signature is not None else []
        except ConfigError:
            records = []  # an unreadable file lets nobody in
        self._records = records
        self._signature = signature
        self._loaded = True

    def count(self) -> int:
        self._refresh()
        return len(self._records)

    def lookup(self, raw: str) -> Scope | None:
        """The scope of a valid token, else None. Compares hashes in constant time."""
        self._refresh()
        digest = hash_token(raw)
        match: Token | None = None
        for record in self._records:  # no early exit: the time does not depend on the match
            if hmac.compare_digest(record.hash, digest):
                match = record
        if match is None:
            return None
        try:
            return Scope(
                business=validate_business(match.business),
                agent=validate_agent(match.agent),
                home=self.home,
            )
        except ConfigError:
            return None  # a hand-edited record with a bad name routes nowhere
