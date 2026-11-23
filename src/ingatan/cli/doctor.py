"""`ingatan doctor`: one pass or fail line per check (FR-52).

Every check catches its own failure, so `doctor` still reports clearly when there is no
config, a broken config, an old database or a SQLite build without a feature.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ingatan import db, timeutil, tokens
from ingatan.errors import ConfigError, IngatanError
from ingatan.scope import (
    GLOBAL_NAME,
    EmbeddingConfig,
    Scope,
    data_home,
    db_path,
    find_config,
    resolve_scope,
    validate_business,
    visible_sql,
)

__all__ = ["Check", "render", "run_checks"]

MIN_PYTHON = (3, 10)


@dataclass
class Check:
    name: str
    status: str  # "pass", "FAIL", or "skip" (not run because the config gave no business)
    detail: str
    notes: list[str] = field(default_factory=lambda: [])

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def _sqlite_probe() -> sqlite3.Connection:
    """A scratch in-memory database. Tests replace this to fake a limited SQLite build."""
    return sqlite3.connect(":memory:")


def _check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] >= MIN_PYTHON:
        return Check("python", "pass", f"Python {version}")
    wanted = ".".join(str(part) for part in MIN_PYTHON)
    return Check("python", "FAIL", f"Python {version}. Ingatan needs {wanted} or newer.")


def _check_feature(name: str, label: str, statements: tuple[str, ...], hint: str) -> Check:
    try:
        probe = _sqlite_probe()
    except sqlite3.Error as exc:
        return Check(name, "FAIL", f"cannot open a scratch database: {exc}")
    try:
        for sql in statements:
            probe.execute(sql)
    except sqlite3.Error as exc:
        return Check(name, "FAIL", f"{label} is not available ({exc}). {hint}")
    finally:
        probe.close()
    return Check(name, "pass", f"{label} (SQLite {sqlite3.sqlite_version})")


def _sources(
    env: dict[str, str],
) -> tuple[list[str], dict[str, str | None]]:
    """Where business and agent come from, mirroring `resolve_scope`. Never raises."""
    notes: list[str] = []
    values: dict[str, str | None] = {}
    path = find_config()
    config: dict[str, object] = {}
    if path is None:
        notes.append("config file: none found in the current folder or a parent")
    else:
        notes.append(f"config file: {path}")
        try:
            from ingatan.scope import _read_config  # pyright: ignore[reportPrivateUsage]

            config = _read_config(path)
        except ConfigError as exc:
            notes.append(f"config file problem: {exc.message}")
    for key, env_name in (("business", "INGATAN_BUSINESS"), ("agent", "INGATAN_AGENT")):
        from_env = (env.get(env_name) or "").strip()
        from_file = config.get(key)
        if from_env:
            values[key] = from_env
            notes.append(f"{key}: {from_env} (from environment variable {env_name})")
        elif isinstance(from_file, str) and from_file.strip():
            values[key] = from_file.strip()
            notes.append(f"{key}: {from_file.strip()} (from {path})")
        else:
            values[key] = None
            notes.append(f"{key}: not set (set {env_name} or put it in .ingatan.toml)")
    return notes, values


def _check_config(env: dict[str, str]) -> tuple[Check, str | None, Scope | None]:
    """The config check, business name to inspect, and resolved Scope (if valid)."""
    notes, values = _sources(env)
    try:
        scope = resolve_scope(env=env)
    except ConfigError as exc:
        return (
            Check("config", "FAIL", exc.message, notes),
            _valid_business(values["business"]),
            None,
        )
    notes.append(f"token budget: {scope.token_budget}")
    mode = "required (proposal queue)" if scope.require_approval else "write-then-undo (immediate)"
    notes.append(f"approval mode: {mode}")
    emb = scope.embeddings
    if emb.provider == "none":
        notes.append("embeddings: none (FTS5 only)")
    else:
        from ingatan.search.embed import effective_model

        model_name = effective_model(emb)
        if emb.provider == "local":
            idle_label = (
                f"idle unload {emb.idle_unload_seconds}s"
                if emb.idle_unload_seconds > 0
                else "no idle unload"
            )
            notes.append(f"embeddings: local ({model_name}; threads={emb.threads}, {idle_label})")
        else:
            notes.append(f"embeddings: {emb.provider} ({model_name})")
        if emb.defer:
            notes.append("defer: on (remember saves first, the server embeds afterwards)")
        if emb.shared and emb.provider == "local":
            notes.append(f"shared model process: {_shared_process_note()}")
    notes.append(f"data folder: {scope.home}")
    detail = f"business {scope.business}, agent {scope.agent}"
    return Check("config", "pass", detail, notes), scope.business, scope


def _shared_process_note() -> str:
    """Whether the one process that holds the model for this data folder answers."""
    from ingatan.search import shared

    running = shared.status()
    if running is not None:
        return f"running (pid {running.pid}, port {running.port} on 127.0.0.1)"
    leftover = shared.state_file()
    if leftover.is_file():
        return (
            f"does not answer ({leftover} is left over from an earlier one; "
            "a server starts a new one on first use)"
        )
    return "not running (a server starts it on first use)"


_FALLBACK_NOTE = (
    "Recall will fall back to FTS5 only until this is resolved "
    "(or set [embeddings] provider = 'none' / INGATAN_EMBED_PROVIDER=none)."
)


def _local_setup_problem(emb: EmbeddingConfig) -> str | None:
    """What still has to be installed or downloaded for the local provider, if anything.

    Looks at package names and file sizes only: it loads no model and uses no network.
    """
    from ingatan.search import local

    try:
        model = local.resolve(emb.model)
    except IngatanError as exc:
        return exc.message
    absent = local.missing_packages()
    if absent:
        return (
            f"the packages {', '.join(absent)} are not installed. "
            f"Install them in the environment Ingatan runs in: {local.INSTALL_HINT}"
        )
    if local.missing_files(model, emb.model_dir):
        folder = local.model_folder(emb.model_dir, model)
        return (
            f"the model files are not in {folder}. Download them once with: ingatan embeddings pull"
        )
    return None


def _vector_notes(scope: Scope, model_name: str) -> list[str]:
    """Which valid facts still lack a vector of the current model. Never raises.

    Only what `ingatan reindex` would fix is counted as its job: the facts this business and
    agent can see. Another agent's private facts are reported separately, because a reindex
    as this agent does not reach them. Vectors left on closed facts are not counted at all:
    nothing recomputes them and nothing reads them.
    """
    now = timeutil.iso(timeutil.now())
    valid = (
        "f.valid_from <= :now AND (f.valid_to IS NULL OR f.valid_to > :now) "
        "AND f.superseded_by IS NULL"
    )
    no_vector = (
        "NOT EXISTS (SELECT 1 FROM fact_embeddings e WHERE e.fact_id = f.id AND e.model = :model)"
    )
    old_vector = (
        "EXISTS (SELECT 1 FROM fact_embeddings e WHERE e.fact_id = f.id AND e.model != :model)"
    )
    params = {"now": now, "model": model_name, "agent": scope.agent}
    total = missing = replaced = others = 0
    for name in (scope.business, GLOBAL_NAME):
        path = db_path(scope.home, name)
        if not path.is_file():
            continue
        try:
            conn = sqlite3.connect(str(path))
        except sqlite3.Error:
            continue

        def count(where: str, conn: sqlite3.Connection = conn) -> int:
            sql = f"SELECT count(*) FROM facts f WHERE {valid} AND {where}"
            return int(conn.execute(sql, params).fetchone()[0])

        try:
            mine = visible_sql(is_global=name == GLOBAL_NAME, alias="f")
            total += count(mine)
            missing += count(f"{mine} AND {no_vector}")
            replaced += count(f"{mine} AND {no_vector} AND {old_vector}")
            others += count(f"NOT ({mine}) AND {no_vector}")
        except sqlite3.Error:
            continue
        finally:
            conn.close()
    notes: list[str] = []
    if missing:
        note = f"{missing} of {total} valid fact(s) have no vector for this model"
        if replaced:
            note += f" ({replaced} of them have one from another model)"
        bare = missing - replaced
        until = " Until then those facts are found by words only."
        if not scope.embeddings.defer:
            note += ". Run `ingatan reindex`; until then those facts are found by words only."
        elif not replaced:
            note += (
                ". With `defer` on, the server embeds them in the background; "
                "`ingatan reindex` does it now." + until
            )
        elif bare:
            note += (
                f". With `defer` on, the server embeds the facts without any vector ({bare}) "
                f"in the background. A vector of another model ({replaced}) is never replaced "
                "that way: run `ingatan reindex`." + until
            )
        else:
            note += (
                ". A vector of another model is never replaced in the background, also with "
                "`defer` on: run `ingatan reindex`." + until
            )
        notes.append(note)
    if others:
        notes.append(
            f"{others} private fact(s) of other agents have no vector either. "
            "Run `ingatan reindex --agent NAME` for each of those agents."
        )
    return notes


def _check_embeddings(scope: Scope | None, *, probe: bool = True) -> Check:
    """Verify the configured embedding provider by embedding one short test sentence.

    Nothing here downloads a model. For the local provider a missing package or model file
    is reported with the command that fixes it.
    """
    if scope is None:
        return Check("embeddings", "skip", "no valid config (see config)")
    emb = scope.embeddings
    if emb.provider == "none":
        return Check("embeddings", "pass", "none (FTS5 only)")
    from ingatan.search.embed import effective_model, probe_embedding

    model_name = effective_model(emb)
    if not probe:
        problem = _local_setup_problem(emb) if emb.provider == "local" else None
        if problem is not None:
            return Check("embeddings", "FAIL", f"local ({model_name}): {problem}", [_FALLBACK_NOTE])
        return Check(
            "embeddings",
            "pass",
            f"{emb.provider} ({model_name}; probe skipped)",
            _vector_notes(scope, model_name),
        )
    try:
        dims = probe_embedding(emb)
    except IngatanError as exc:
        return Check("embeddings", "FAIL", exc.message, [_FALLBACK_NOTE])
    return Check(
        "embeddings",
        "pass",
        f"{emb.provider} ({model_name}, {dims} dims; probe OK)",
        _vector_notes(scope, model_name),
    )


def _valid_business(name: str | None) -> str | None:
    if name is None:
        return None
    try:
        return validate_business(name)
    except ConfigError:
        return None


def _check_database(label: str, path: Path) -> Check:
    name = f"database {label}"
    if not path.is_file():
        return Check(name, "FAIL", f"{path} does not exist. Run `ingatan init`.")
    # Plain sqlite3, not `db.connect`: connecting would migrate an old file and hide it.
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as exc:
        return Check(name, "FAIL", f"{path} cannot be opened: {exc}")
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.Error as exc:
        return Check(name, "FAIL", f"{path} is not a usable database: {exc}")
    finally:
        conn.close()
    latest = db.LATEST_VERSION
    if version == latest:
        return Check(name, "pass", f"{path} at schema version {version}")
    if version < latest:
        return Check(
            name,
            "FAIL",
            f"{path} is at schema version {version}, the code expects {latest}. "
            "It is migrated the next time Ingatan opens it.",
        )
    return Check(
        name,
        "FAIL",
        f"{path} is at schema version {version}, newer than this Ingatan ({latest}). "
        "Upgrade Ingatan.",
    )


def _check_writable(home: Path) -> Check:
    probe_dir = home
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    suffix = "" if probe_dir == home else f" (it will be created under {probe_dir})"
    try:
        with tempfile.NamedTemporaryFile(dir=probe_dir, prefix=".ingatan-doctor-"):
            pass
    except OSError as exc:
        return Check("data folder", "FAIL", f"{home} is not writable: {exc}")
    return Check("data folder", "pass", f"{home} is writable{suffix}")


def _check_tokens(home: Path) -> Check:
    """How many HTTP tokens exist. None is fine: HTTP mode is optional."""
    try:
        count = len(tokens.list_tokens(home))
    except ConfigError as exc:
        return Check("tokens", "FAIL", exc.message)
    return Check("tokens", "pass", f"{count} HTTP access token(s) in {tokens.tokens_path(home)}")


def run_checks(
    *,
    business: str | None = None,
    agent: str | None = None,
    probe_embeddings: bool = True,
) -> list[Check]:
    env = dict(os.environ)
    if business is not None:
        env["INGATAN_BUSINESS"] = business
    if agent is not None:
        env["INGATAN_AGENT"] = agent
    checks = [
        _check_python(),
        _check_feature(
            "fts5",
            "FTS5",
            ("CREATE VIRTUAL TABLE t_fts USING fts5(x)",),
            "Use a Python build whose SQLite has FTS5.",
        ),
        _check_feature(
            "trigram",
            "FTS5 trigram tokenizer",
            ("CREATE VIRTUAL TABLE t_tri USING fts5(x, tokenize = 'trigram')",),
            "It needs SQLite 3.34 or newer.",
        ),
        _check_feature(
            "fts secure-delete",
            "FTS5 secure-delete",
            (
                "CREATE VIRTUAL TABLE t_sd USING fts5(x)",
                "INSERT INTO t_sd (t_sd, rank) VALUES ('secure-delete', 1)",
            ),
            "It needs SQLite 3.42 or newer, and `forget` relies on it.",
        ),
    ]
    config, resolved_business, scope = _check_config(env)
    checks.append(config)
    checks.append(_check_embeddings(scope, probe=probe_embeddings))
    home = data_home(env)
    if resolved_business is not None:
        checks.append(_check_database(resolved_business, db_path(home, resolved_business)))
    else:
        checks.append(Check("database business", "skip", "no valid business name (see config)"))
    checks.append(_check_database(GLOBAL_NAME, db_path(home, GLOBAL_NAME)))
    checks.append(_check_writable(home))
    checks.append(_check_tokens(home))
    return checks


def render(checks: list[Check]) -> str:
    lines: list[str] = []
    for check in checks:
        lines.append(f"[{check.status}] {check.name}: {check.detail}")
        lines.extend(f"       {note}" for note in check.notes)
    failed = sum(1 for check in checks if check.failed)
    lines.append("")
    lines.append("All checks passed." if failed == 0 else f"{failed} check(s) failed.")
    return "\n".join(lines)
