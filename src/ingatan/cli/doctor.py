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

from ingatan import db, tokens
from ingatan.errors import ConfigError
from ingatan.scope import (
    GLOBAL_NAME,
    data_home,
    db_path,
    find_config,
    resolve_scope,
    validate_business,
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


def _sources() -> tuple[list[str], dict[str, str | None]]:
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
        from_env = (os.environ.get(env_name) or "").strip()
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


def _check_config() -> tuple[Check, str | None]:
    """The config check, and the business name to look for a database under (if any)."""
    notes, values = _sources()
    try:
        scope = resolve_scope()
    except ConfigError as exc:
        return Check("config", "FAIL", exc.message, notes), _valid_business(values["business"])
    notes.append(f"token budget: {scope.token_budget}")
    notes.append(f"data folder: {scope.home}")
    detail = f"business {scope.business}, agent {scope.agent}"
    return Check("config", "pass", detail, notes), scope.business


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


def run_checks() -> list[Check]:
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
    config, business = _check_config()
    checks.append(config)
    home = data_home()
    if business is not None:
        checks.append(_check_database(business, db_path(home, business)))
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
