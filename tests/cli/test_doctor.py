"""T-26: `ingatan doctor` fails the right check and exits non-zero."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

from ingatan import db
from ingatan.cli import doctor as doctor_module

Invoke = Callable[..., Result]


def statuses(output: str) -> dict[str, str]:
    """Check name -> pass / FAIL / skip, from the `[status] name: detail` lines."""
    found = re.findall(r"^\[(pass|FAIL|skip)\] ([^:]+):", output, re.MULTILINE)
    return {name: status for status, name in found}


@pytest.fixture
def healthy(invoke: Invoke, ingatan_home: Path) -> Path:
    assert invoke("init", "--business", "tokoku", "--agent", "claude-code").exit_code == 0
    return ingatan_home


def test_a_healthy_setup_passes_every_check_and_exits_zero(invoke: Invoke, healthy: Path) -> None:
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    found = statuses(result.stdout)
    assert set(found.values()) == {"pass"}
    assert set(found) == {
        "python",
        "fts5",
        "trigram",
        "fts secure-delete",
        "config",
        "embeddings",
        "database tokoku",
        "database global",
        "data folder",
        "tokens",
    }
    assert "0 HTTP access token(s)" in result.stdout
    assert "All checks passed." in result.stdout


def test_doctor_shows_the_resolved_config_and_which_source_won(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_only = invoke("doctor").stdout
    assert "business: tokoku (from " in file_only and ".ingatan.toml" in file_only
    assert f"data folder: {healthy}" in file_only and "token budget: 1000" in file_only
    monkeypatch.setenv("INGATAN_AGENT", "hermes")
    mixed = invoke("doctor").stdout
    assert "agent: hermes (from environment variable INGATAN_AGENT)" in mixed
    assert "business: tokoku (from " in mixed and ".ingatan.toml" in mixed


def test_doctor_without_any_config_reports_clearly_and_still_runs_the_other_checks(
    invoke: Invoke, ingatan_home: Path
) -> None:
    result = invoke("doctor")
    assert result.exit_code == 1
    found = statuses(result.stdout)
    assert found["config"] == "FAIL" and found["python"] == "pass" and found["fts5"] == "pass"
    assert found["database business"] == "skip" and found["database global"] == "FAIL"
    assert "No business name is set" in result.stdout
    assert "config file: none found" in result.stdout
    assert "business: not set" in result.stdout and "Traceback" not in result.output
    assert "check(s) failed" in result.stdout


def test_a_broken_config_fails_the_config_check_only(invoke: Invoke, healthy: Path) -> None:
    (Path.cwd() / ".ingatan.toml").write_text("business = [", encoding="utf-8")
    result = invoke("doctor")
    assert result.exit_code == 1
    found = statuses(result.stdout)
    assert found["config"] == "FAIL" and "not valid TOML" in result.stdout
    assert found["fts5"] == found["trigram"] == found["database global"] == "pass"


def test_an_invalid_business_name_fails_the_config_check(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "Toko Ku")
    result = invoke("doctor")
    found = statuses(result.stdout)
    assert result.exit_code == 1 and found["config"] == "FAIL"
    assert "Invalid business name" in result.stdout
    assert found["database business"] == "skip"


def test_an_old_schema_fails_the_database_checks(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "LATEST_VERSION", db.LATEST_VERSION + 1)  # the code is ahead
    result = invoke("doctor")
    found = statuses(result.stdout)
    assert result.exit_code == 1
    assert found["database tokoku"] == found["database global"] == "FAIL"
    assert found["config"] == "pass" and found["fts5"] == "pass"
    assert f"schema version {db.LATEST_VERSION - 1}, the code expects {db.LATEST_VERSION}" in (
        result.stdout
    )


def test_a_database_newer_than_the_code_fails_its_check(invoke: Invoke, healthy: Path) -> None:
    raw = sqlite3.connect(str(healthy / "tokoku.db"))
    raw.execute(f"PRAGMA user_version = {db.LATEST_VERSION + 5}")
    raw.close()
    result = invoke("doctor")
    found = statuses(result.stdout)
    assert result.exit_code == 1
    assert found["database tokoku"] == "FAIL" and found["database global"] == "pass"
    assert "newer than this Ingatan" in result.stdout


def test_a_missing_database_file_fails_with_a_hint_to_run_init(
    invoke: Invoke, healthy: Path
) -> None:
    for leftover in healthy.glob("global.db*"):
        leftover.unlink()
    result = invoke("doctor")
    found = statuses(result.stdout)
    assert result.exit_code == 1 and found["database global"] == "FAIL"
    assert "ingatan init" in result.stdout and found["database tokoku"] == "pass"


def test_a_file_that_is_not_a_database_fails_without_a_traceback(
    invoke: Invoke, healthy: Path
) -> None:
    (healthy / "tokoku.db").write_bytes(b"this is not an sqlite file at all" * 20)
    for leftover in healthy.glob("tokoku.db-*"):
        leftover.unlink()
    result = invoke("doctor")
    assert result.exit_code == 1 and statuses(result.stdout)["database tokoku"] == "FAIL"
    assert "Traceback" not in result.output


class LimitedSqlite:
    """A scratch connection that refuses statements mentioning a missing feature."""

    def __init__(self, missing: str) -> None:
        self.real = sqlite3.connect(":memory:")
        self.missing = missing

    def execute(self, sql: str, *args: Any) -> Any:
        if self.missing in sql:
            raise sqlite3.OperationalError(f"no such tokenizer: {self.missing}")
        return self.real.execute(sql, *args)

    def close(self) -> None:
        self.real.close()


def test_a_missing_tokenizer_fails_the_trigram_check_and_exits_non_zero(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_module, "_sqlite_probe", lambda: LimitedSqlite("trigram"))
    result = invoke("doctor")
    found = statuses(result.stdout)
    assert result.exit_code == 1
    assert found["trigram"] == "FAIL" and "no such tokenizer: trigram" in result.stdout
    assert found["fts5"] == "pass" and found["fts secure-delete"] == "pass"
    assert found["database tokoku"] == "pass"


def test_missing_fts5_fails_the_fts5_check(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_module, "_sqlite_probe", lambda: LimitedSqlite("fts5"))
    found = statuses(invoke("doctor").stdout)
    assert found["fts5"] == found["trigram"] == found["fts secure-delete"] == "FAIL"


def test_missing_secure_delete_fails_its_own_check(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_module, "_sqlite_probe", lambda: LimitedSqlite("secure-delete"))
    result = invoke("doctor")
    found = statuses(result.stdout)
    assert result.exit_code == 1 and found["fts secure-delete"] == "FAIL"
    assert found["fts5"] == found["trigram"] == "pass"


def test_an_old_python_fails_the_python_check(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_module, "MIN_PYTHON", (99, 0))
    result = invoke("doctor")
    assert result.exit_code == 1 and statuses(result.stdout)["python"] == "FAIL"


def test_a_data_folder_that_cannot_be_written_fails_its_check(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**_: Any) -> Any:
        raise PermissionError("read-only folder")

    monkeypatch.setattr(doctor_module.tempfile, "NamedTemporaryFile", refuse)
    result = invoke("doctor")
    assert result.exit_code == 1 and statuses(result.stdout)["data folder"] == "FAIL"
    assert "read-only folder" in result.stdout


def test_doctor_does_not_create_files_or_migrate(
    invoke: Invoke, ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")
    monkeypatch.setenv("INGATAN_AGENT", "claude-code")
    result = invoke("doctor")
    assert result.exit_code == 1  # the databases do not exist yet
    assert not ingatan_home.exists()


# ---- embeddings ----------------------------------------------------------------------


def test_doctor_says_embeddings_are_off_by_default(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INGATAN_EMBED_PROVIDER", raising=False)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "[pass] embeddings: none (FTS5 only)" in result.stdout


def test_doctor_names_the_command_that_fixes_a_missing_local_model(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ingatan.search import local

    monkeypatch.setenv("INGATAN_EMBED_PROVIDER", "local")
    monkeypatch.setattr(local, "missing_packages", lambda: [])
    for args in (("doctor",), ("doctor", "--no-probe-embeddings")):
        result = invoke(*args)
        assert result.exit_code == 1, result.output
        assert statuses(result.stdout)["embeddings"] == "FAIL"
        assert "ingatan embeddings pull" in result.stdout
        assert "fall back to FTS5 only" in result.stdout
    # Nothing was downloaded or created by looking.
    assert not (healthy / "models").exists()

    monkeypatch.setattr(local, "missing_packages", lambda: ["onnxruntime", "sentencepiece"])
    for args in (("doctor",), ("doctor", "--no-probe-embeddings")):
        result = invoke(*args)
        assert result.exit_code == 1
        # A command that works from a checkout and from an installed tool alike.
        assert local.INSTALL_HINT in result.stdout
        assert "onnxruntime, sentencepiece" in result.stdout

    monkeypatch.setenv("INGATAN_EMBED_MODEL", "not/a-model")
    result = invoke("doctor", "--no-probe-embeddings")
    assert result.exit_code == 1
    assert "Unknown local embedding model" in result.stdout


def _add_fact(
    home: Path,
    fact_id: str,
    *,
    tier: str = "business",
    agent: str = "claude-code",
    closed: bool = False,
    vector_of: str | None = None,
) -> None:
    """Write one fact row (and optionally a vector) straight into tokoku.db."""
    conn = sqlite3.connect(str(home / "tokoku.db"), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO facts (id, text, entities, keywords, tier, agent, kind, valid_from, "
            "valid_to, source, created_at) VALUES (?, ?, '[]', '[]', ?, ?, 'stable', "
            "'2020-01-01T00:00:00Z', ?, '{}', '2020-01-01T00:00:00Z')",
            (
                fact_id,
                f"Catatan {fact_id}.",
                tier,
                agent,
                "2020-02-01T00:00:00Z" if closed else None,
            ),
        )
        if vector_of is not None:
            conn.execute(
                "INSERT INTO fact_embeddings (fact_id, model, dims, vec, updated_at) "
                "VALUES (?, ?, 1, ?, '2020-01-01T00:00:00Z')",
                (fact_id, vector_of, b"\x00\x00\x80\x3f"),
            )
    finally:
        conn.close()


def test_doctor_counts_only_the_vectors_a_reindex_would_write(
    invoke: Invoke, healthy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = "xenova/multilingual-e5-small"
    monkeypatch.setenv("INGATAN_EMBED_PROVIDER", "local")
    monkeypatch.setattr(doctor_module, "_local_setup_problem", lambda emb: None)

    def notes() -> str:
        result = invoke("doctor", "--no-probe-embeddings")
        assert result.exit_code == 0, result.output
        return result.stdout

    assert "have no vector" not in notes()  # an empty store has nothing to index

    _add_fact(healthy, "f_done", vector_of=current)
    _add_fact(healthy, "f_new")
    _add_fact(healthy, "f_switched", vector_of="some/older-model")
    _add_fact(healthy, "f_mine", tier="agent")
    # Closed facts are never searched and never reindexed, whatever vector they still hold.
    _add_fact(healthy, "f_closed_bare", closed=True)
    _add_fact(healthy, "f_closed_old", closed=True, vector_of="some/older-model")
    out = notes()
    assert "3 of 4 valid fact(s) have no vector for this model" in out
    assert "(1 of them have one from another model)" in out
    assert "Run `ingatan reindex`" in out
    assert "other agents" not in out

    # Another agent's private facts are out of reach of a reindex as this agent: they are
    # reported on their own line, with the command that does reach them.
    _add_fact(healthy, "f_theirs", tier="agent", agent="hermes")
    _add_fact(healthy, "f_theirs_done", tier="agent", agent="hermes", vector_of=current)
    out = notes()
    assert "3 of 4 valid fact(s) have no vector for this model" in out
    assert "1 private fact(s) of other agents have no vector either" in out
    assert "ingatan reindex --agent NAME" in out

    # Seen as that agent, the same row is its own and is counted in the first line.
    monkeypatch.setenv("INGATAN_AGENT", "hermes")
    out = notes()
    assert "3 of 5 valid fact(s) have no vector for this model" in out
    assert "1 private fact(s) of other agents have no vector either" in out  # f_mine
