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
    "DEFAULT_TOKEN_BUDGET",
    "GLOBAL_NAME",
    "HOME_CONFIG_NAME",
    "NAME_PATTERN",
    "VISIBLE_SQL",
    "VISIBLE_SQL_GLOBAL",
    "Context",
    "Scope",
    "data_home",
    "db_path",
    "find_config",
    "find_fact",
    "load_home_config",
    "open_context",
    "parse_bool",
    "resolve_allowed_businesses",
    "resolve_scope",
    "validate_agent",
    "validate_business",
    "visible_sql",
]

CONFIG_NAME = ".ingatan.toml"
HOME_CONFIG_NAME = "config.toml"
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
    require_approval: bool = False


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


def load_home_config(home: Path) -> dict[str, Any]:
    """Read `<home>/config.toml` if present, else an empty dict."""
    path = home / HOME_CONFIG_NAME
    if not path.is_file():
        return {}
    return _read_config(path)


def parse_bool(value: object, *, label: str) -> bool:
    """Parse a boolean from TOML or an environment variable string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().casefold()
        if cleaned in ("1", "true", "yes", "on"):
            return True
        if cleaned in ("0", "false", "no", "off"):
            return False
    raise ConfigError(f"{label} must be true or false (got {value!r}).")


def _parse_budget(config: Mapping[str, Any], origin: str) -> int | None:
    brief_section = config.get("brief")
    if isinstance(brief_section, dict) and "token_budget" in brief_section:
        raw = brief_section["token_budget"]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ConfigError(f"[brief] token_budget in {origin} must be a positive integer.")
        return raw
    return None


def _parse_approval(
    source: Mapping[str, str],
    project_cfg: Mapping[str, Any],
    project_origin: str,
    home_cfg: Mapping[str, Any],
    home_origin: str,
) -> bool:
    raw_env = (source.get("INGATAN_REQUIRE_APPROVAL") or "").strip()
    if raw_env:
        return parse_bool(raw_env, label="INGATAN_REQUIRE_APPROVAL")
    if "require_approval" in project_cfg:
        return parse_bool(
            project_cfg["require_approval"], label=f"require_approval in {project_origin}"
        )
    if "require_approval" in home_cfg:
        return parse_bool(home_cfg["require_approval"], label=f"require_approval in {home_origin}")
    return False


def resolve_scope(start: Path | None = None, env: Mapping[str, str] | None = None) -> Scope:
    """Business and agent from `INGATAN_BUSINESS` / `INGATAN_AGENT`, else `.ingatan.toml`,
    falling back to `<INGATAN_HOME>/config.toml` for defaults (`default_agent`, `token_budget`,
    `require_approval`).

    Each setting is resolved on its own: environment wins over `.ingatan.toml`, which wins over
    `<INGATAN_HOME>/config.toml`.
    """
    source = os.environ if env is None else env
    home = data_home(source)
    home_cfg = load_home_config(home)
    config_file = find_config(start)
    config = _read_config(config_file) if config_file is not None else {}

    def pick(key: str, env_name: str, home_keys: tuple[str, ...] = ()) -> str | None:
        from_env = (source.get(env_name) or "").strip()
        if from_env:
            return from_env
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        for hk in home_keys:
            hval = home_cfg.get(hk)
            if isinstance(hval, str) and hval.strip():
                return hval.strip()
        return None

    business = pick("business", "INGATAN_BUSINESS")
    agent = pick("agent", "INGATAN_AGENT", ("default_agent", "agent"))
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

    home_origin = str(home / HOME_CONFIG_NAME)
    project_origin = str(config_file) if config_file is not None else CONFIG_NAME
    budget = _parse_budget(config, project_origin)
    if budget is None:
        budget = _parse_budget(home_cfg, home_origin)
    if budget is None:
        budget = DEFAULT_TOKEN_BUDGET

    require_approval = _parse_approval(source, config, project_origin, home_cfg, home_origin)

    return Scope(
        business=validate_business(business),
        agent=validate_agent(agent),
        home=home,
        token_budget=budget,
        require_approval=require_approval,
    )


def resolve_allowed_businesses(
    start: Path | None = None, env: Mapping[str, str] | None = None
) -> tuple[list[str], str, Path, int, bool] | None:
    """Resolve optional multi-business MCP server configuration.

    Triggered when `INGATAN_BUSINESSES` (comma-separated) is set, or when `allowed_businesses`
    is configured in `.ingatan.toml` / `<INGATAN_HOME>/config.toml` and no single business is
    explicitly pinned via `INGATAN_BUSINESS` or `business = ...`.
    Returns `(businesses, agent, home, token_budget, require_approval)` or `None`.
    """
    source = os.environ if env is None else env
    home = data_home(source)
    home_cfg = load_home_config(home)
    config_file = find_config(start)
    config = _read_config(config_file) if config_file is not None else {}

    raw_env_list = (source.get("INGATAN_BUSINESSES") or "").strip()
    raw_list: list[str] | None = None
    if raw_env_list:
        raw_list = [part.strip() for part in raw_env_list.split(",") if part.strip()]
    else:
        single_pinned = bool((source.get("INGATAN_BUSINESS") or "").strip()) or isinstance(
            config.get("business"), str
        )
        if not single_pinned:
            cfg_list = config.get("allowed_businesses") or home_cfg.get("allowed_businesses")
            if isinstance(cfg_list, list) and cfg_list:
                raw_list = [str(item).strip() for item in cfg_list if str(item).strip()]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]

    if not raw_list:
        return None

    seen: set[str] = set()
    businesses: list[str] = []
    for name in raw_list:
        valid = validate_business(name)
        if valid not in seen:
            seen.add(valid)
            businesses.append(valid)

    agent = (source.get("INGATAN_AGENT") or "").strip()
    if not agent:
        for candidate in (config.get("agent"), home_cfg.get("default_agent"), home_cfg.get("agent")):
            if isinstance(candidate, str) and candidate.strip():
                agent = candidate.strip()
                break
    if not agent:
        raise ConfigError(
            f'No agent name is set. Put `agent = "..."` in {CONFIG_NAME} '
            "or set the INGATAN_AGENT environment variable."
        )

    home_origin = str(home / HOME_CONFIG_NAME)
    project_origin = str(config_file) if config_file is not None else CONFIG_NAME
    budget = _parse_budget(config, project_origin) or _parse_budget(home_cfg, home_origin)
    if budget is None:
        budget = DEFAULT_TOKEN_BUDGET
    require_approval = _parse_approval(source, config, project_origin, home_cfg, home_origin)
    return businesses, validate_agent(agent), home, budget, require_approval


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
