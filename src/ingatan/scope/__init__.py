"""Who is calling (business, agent), where the data lives, and what each caller may see."""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from ingatan import db
from ingatan.errors import ConfigError, ValidationError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

__all__ = [
    "CONFIG_NAME",
    "GLOBAL_NAME",
    "NAME_PATTERN",
    "VISIBLE_SQL",
    "VISIBLE_SQL_GLOBAL",
    "Context",
    "Scope",
    "data_home",
    "db_path",
    "find_config",
    "find_fact",
    "open_context",
    "resolve_scope",
    "validate_agent",
    "validate_business",
    "visible_sql",
]

CONFIG_NAME = ".ingatan.toml"
GLOBAL_NAME = "global"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
DEFAULT_TOKEN_BUDGET = 1000

# WHERE clause for facts in the business file. Bind `:agent` to the caller's agent name.
VISIBLE_SQL = "(tier = 'business' OR (tier = 'agent' AND agent = :agent))"
# WHERE clause for the global file.
VISIBLE_SQL_GLOBAL = "tier = 'global'"


@dataclass(frozen=True)
class Scope:
    business: str
    agent: str
    home: Path
    token_budget: int = DEFAULT_TOKEN_BUDGET


def _check_name(kind: str, value: object) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.match(value):
        raise ConfigError(
            f"Invalid {kind} name {value!r}. Use 1 to 64 characters: lowercase letters, "
            "digits, '_' or '-', starting with a letter or digit."
        )
    return value


def validate_business(name: object) -> str:
    """Return `name` if it is a valid business name, else raise `ConfigError`."""
    value = _check_name("business", name)
    if value == GLOBAL_NAME:
        raise ConfigError(f"'{GLOBAL_NAME}' is reserved and cannot be used as a business name.")
    return value


def validate_agent(name: object) -> str:
    """Return `name` if it is a valid agent name, else raise `ConfigError`."""
    return _check_name("agent", name)


def data_home(env: Mapping[str, str] | None = None) -> Path:
    """`$INGATAN_HOME` if set, else `~/.ingatan`."""
    source = os.environ if env is None else env
    configured = (source.get("INGATAN_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".ingatan"


def db_path(home: Path, name: str) -> Path:
    """Database file for a business name, or for `"global"`."""
    if name != GLOBAL_NAME:
        validate_business(name)
    return home / f"{name}.db"


def find_config(start: Path | None = None) -> Path | None:
    """The nearest `.ingatan.toml` in `start` (default: current folder) or a parent."""
    folder = (start if start is not None else Path.cwd()).resolve()
    for candidate in (folder, *folder.parents):
        path = candidate / CONFIG_NAME
        if path.is_file():
            return path
    return None


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc


def resolve_scope(start: Path | None = None, env: Mapping[str, str] | None = None) -> Scope:
    """Business and agent from `INGATAN_BUSINESS` / `INGATAN_AGENT`, else `.ingatan.toml`.

    Each of the two is resolved on its own: the environment variable wins over the file.
    """
    source = os.environ if env is None else env
    config_file = find_config(start)
    config = _read_config(config_file) if config_file is not None else {}

    def pick(key: str, env_name: str) -> str | None:
        from_env = (source.get(env_name) or "").strip()
        if from_env:
            return from_env
        value = config.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    business = pick("business", "INGATAN_BUSINESS")
    agent = pick("agent", "INGATAN_AGENT")
    for label, value, env_name in (
        ("business", business, "INGATAN_BUSINESS"),
        ("agent", agent, "INGATAN_AGENT"),
    ):
        if value is None:
            raise ConfigError(
                f'No {label} name is set. Put `{label} = "..."` in {CONFIG_NAME} '
                f"(in the project folder or a parent) or set the {env_name} environment variable."
            )
    assert business is not None and agent is not None

    budget = DEFAULT_TOKEN_BUDGET
    brief_section = config.get("brief")
    if isinstance(brief_section, dict) and "token_budget" in brief_section:
        raw = brief_section["token_budget"]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ConfigError(f"[brief] token_budget in {config_file} must be a positive integer.")
        budget = raw

    return Scope(
        business=validate_business(business),
        agent=validate_agent(agent),
        home=data_home(source),
        token_budget=budget,
    )


def visible_sql(*, is_global: bool, alias: str = "", owner_column: str = "agent") -> str:
    """WHERE clause for the rows a caller may see, in the business or the global file.

    Works for `facts` (owner column `agent`) and `log` (owner column `owner`).
    Bind `:agent` when `is_global` is false.
    """
    prefix = f"{alias}." if alias else ""
    if is_global:
        return f"{prefix}tier = 'global'"
    return (
        f"({prefix}tier = 'business' OR "
        f"({prefix}tier = 'agent' AND {prefix}{owner_column} = :agent))"
    )


class Context:
    """The object every module takes: the scope plus both open database files."""

    scope: Scope
    business: sqlite3.Connection
    glob: sqlite3.Connection

    def __init__(
        self, scope: Scope, business: sqlite3.Connection, glob: sqlite3.Connection
    ) -> None:
        self.scope = scope
        self.business = business
        self.glob = glob

    def conn_for_tier(self, tier: str) -> sqlite3.Connection:
        if tier == "global":
            return self.glob
        if tier in ("business", "agent"):
            return self.business
        raise ValidationError(f"Unknown tier '{tier}'. Use agent, business or global.")

    def is_global(self, conn: sqlite3.Connection) -> bool:
        return conn is self.glob

    def connections(self) -> list[sqlite3.Connection]:
        """Both files, business first."""
        return [self.business, self.glob]

    def visible_sql(
        self, conn: sqlite3.Connection, *, alias: str = "", owner_column: str = "agent"
    ) -> str:
        """The visibility WHERE clause for rows in `conn` (one of this context's files)."""
        return visible_sql(is_global=self.is_global(conn), alias=alias, owner_column=owner_column)

    @property
    def params(self) -> dict[str, str]:
        """Named parameters that every visibility clause needs."""
        return {"agent": self.scope.agent}

    def close(self) -> None:
        for conn in (self.business, self.glob):
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    def __enter__(self) -> Context:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_context(scope: Scope) -> Context:
    """Open the business file and `global.db` under `scope.home`, creating them if needed."""
    business_name = validate_business(scope.business)
    validate_agent(scope.agent)
    scope.home.mkdir(parents=True, exist_ok=True)
    business = db.connect(db_path(scope.home, business_name))
    try:
        glob = db.connect(db_path(scope.home, GLOBAL_NAME))
    except BaseException:
        business.close()
        raise
    return Context(scope, business, glob)


def find_fact(ctx: Context, fact_id: str) -> tuple[sqlite3.Connection, sqlite3.Row] | None:
    """The fact with this id in either file, if the caller may see it.

    An invisible fact behaves exactly like a missing one.
    """
    for conn in ctx.connections():
        where = ctx.visible_sql(conn)
        row = conn.execute(
            f"SELECT * FROM facts WHERE id = :id AND {where}",
            {"id": fact_id, **ctx.params},
        ).fetchone()
        if row is not None:
            return conn, row
    return None
