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
    "EMBED_PROVIDERS",
    "GLOBAL_NAME",
    "HOME_CONFIG_NAME",
    "NAME_PATTERN",
    "VISIBLE_SQL",
    "VISIBLE_SQL_GLOBAL",
    "Context",
    "EmbeddingConfig",
    "Scope",
    "data_home",
    "db_path",
    "find_config",
    "find_fact",
    "load_embeddings",
    "load_home_config",
    "open_context",
    "parse_bool",
    "parse_embeddings",
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
EMBED_PROVIDERS = ("none", "local", "ollama", "openai", "gemini")

# WHERE clause for facts in the business file. Bind `:agent` to the caller's agent name.
VISIBLE_SQL = "(tier = 'business' OR (tier = 'agent' AND agent = :agent))"
# WHERE clause for the global file.
VISIBLE_SQL_GLOBAL = "tier = 'global'"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Semantic search settings. The default is no embeddings at all: pure FTS5.

    `model_dir` is where the local provider keeps its model files (empty means
    `<INGATAN_HOME>/models`). With `defer`, `remember` saves a fact without waiting for the
    model and a background step of the server fills the vector in afterwards. The three
    similarity fields override the cut-offs calibrated for the model family; None keeps the
    calibrated value.
    """

    provider: str = "none"
    model: str = ""
    api_base: str = ""
    api_key_env: str = ""
    timeout: float = 15.0
    threads: int = 2
    idle_unload_seconds: int = 300
    model_dir: str = ""
    defer: bool = False
    recall_floor: float | None = None
    conflict_similarity: float | None = None
    conflict_strict_similarity: float | None = None


@dataclass(frozen=True)
class Scope:
    business: str
    agent: str
    home: Path
    token_budget: int = DEFAULT_TOKEN_BUDGET
    require_approval: bool = False
    embeddings: EmbeddingConfig = EmbeddingConfig()


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


def parse_embeddings(
    source: Mapping[str, str],
    project_cfg: Mapping[str, Any],
    home_cfg: Mapping[str, Any],
) -> EmbeddingConfig:
    """Resolve `[embeddings]` config: env > `.ingatan.toml` > `<home>/config.toml` > default."""
    home_sec = home_cfg.get("embeddings")
    proj_sec = project_cfg.get("embeddings")
    merged: dict[str, Any] = {}
    if isinstance(home_sec, dict):
        merged.update(home_sec)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(proj_sec, dict):
        merged.update(proj_sec)  # pyright: ignore[reportUnknownArgumentType]

    def pick_str(key: str, env_name: str) -> str:
        from_env = (source.get(env_name) or "").strip()
        if from_env:
            return from_env
        val = merged.get(key)
        return val.strip() if isinstance(val, str) else ""

    provider = (pick_str("provider", "INGATAN_EMBED_PROVIDER") or "none").casefold()
    if provider not in EMBED_PROVIDERS:
        raise ConfigError(
            f"Unknown embeddings provider {provider!r}. Use one of: {', '.join(EMBED_PROVIDERS)}."
        )
    model = pick_str("model", "INGATAN_EMBED_MODEL")
    api_base = pick_str("api_base", "INGATAN_EMBED_API_BASE")
    api_key_env = pick_str("api_key_env", "INGATAN_EMBED_API_KEY_ENV")
    raw_timeout = merged.get("timeout", 15.0)
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or raw_timeout <= 0
    ):
        raise ConfigError("[embeddings] timeout must be a positive number.")

    def pick_int(key: str, env_name: str, default: int, *, minimum: int) -> int:
        from_env = (source.get(env_name) or "").strip()
        raw_val: object = from_env if from_env else merged.get(key, default)
        if isinstance(raw_val, bool):
            raise ConfigError(f"[embeddings] {key} must be an integer >= {minimum}.")
        if isinstance(raw_val, str):
            try:
                parsed = int(raw_val)
            except ValueError as exc:
                raise ConfigError(f"[embeddings] {key} must be an integer >= {minimum}.") from exc
        elif isinstance(raw_val, int):
            parsed = raw_val
        else:
            raise ConfigError(f"[embeddings] {key} must be an integer >= {minimum}.")
        if parsed < minimum:
            raise ConfigError(f"[embeddings] {key} must be an integer >= {minimum}.")
        return parsed

    def pick_bool(key: str, env_name: str) -> bool:
        from_env = (source.get(env_name) or "").strip()
        if from_env:
            return parse_bool(from_env, label=env_name)
        if key in merged:
            return parse_bool(merged[key], label=f"[embeddings] {key}")
        return False

    def pick_similarity(key: str) -> float | None:
        if key not in merged:
            return None
        raw_val = merged[key]
        if (
            isinstance(raw_val, bool)
            or not isinstance(raw_val, (int, float))
            or not -1.0 <= raw_val <= 1.0
        ):
            raise ConfigError(f"[embeddings] {key} must be a cosine similarity between -1 and 1.")
        return float(raw_val)

    threads = pick_int("threads", "INGATAN_EMBED_THREADS", 2, minimum=1)
    idle_unload = pick_int("idle_unload_seconds", "INGATAN_EMBED_IDLE_UNLOAD", 300, minimum=0)
    model_dir = pick_str("model_dir", "INGATAN_MODEL_DIR")
    if model_dir:
        expanded = Path(model_dir).expanduser()
        if not expanded.is_absolute():
            # A relative path would be another folder for every working directory: the one
            # `ingatan embeddings pull` filled is then not the one the server looks in.
            raise ConfigError(
                f"[embeddings] model_dir (or INGATAN_MODEL_DIR) must be an absolute path, "
                f"got {model_dir!r}."
            )
        model_dir = str(expanded)
    return EmbeddingConfig(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        timeout=float(raw_timeout),
        threads=threads,
        idle_unload_seconds=idle_unload,
        model_dir=model_dir,
        defer=pick_bool("defer", "INGATAN_EMBED_DEFER"),
        recall_floor=pick_similarity("recall_floor"),
        conflict_similarity=pick_similarity("conflict_similarity"),
        conflict_strict_similarity=pick_similarity("conflict_strict_similarity"),
    )


def load_embeddings(
    start: Path | None = None, env: Mapping[str, str] | None = None
) -> EmbeddingConfig:
    """The `[embeddings]` settings alone, for commands that need no business or agent name."""
    source = os.environ if env is None else env
    config_file = find_config(start)
    project_cfg = _read_config(config_file) if config_file is not None else {}
    return parse_embeddings(source, project_cfg, load_home_config(data_home(source)))


def resolve_scope(start: Path | None = None, env: Mapping[str, str] | None = None) -> Scope:
    """Business and agent from `INGATAN_BUSINESS` / `INGATAN_AGENT`, else `.ingatan.toml`,
    falling back to `<INGATAN_HOME>/config.toml` for defaults (`default_agent`, `token_budget`,
    `require_approval`, `[embeddings]`).

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
    embeddings = parse_embeddings(source, config, home_cfg)

    return Scope(
        business=validate_business(business),
        agent=validate_agent(agent),
        home=home,
        token_budget=budget,
        require_approval=require_approval,
        embeddings=embeddings,
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
        for candidate in (
            config.get("agent"),
            home_cfg.get("default_agent"),
            home_cfg.get("agent"),
        ):
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
