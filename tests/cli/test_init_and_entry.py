"""T-15: skeleton, `init`, entry point behaviour (server dispatch, errors, UTF-8)."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

import ingatan
from ingatan import cli, facts
from ingatan.errors import ConfigError
from ingatan.scope import Context

Invoke = Callable[..., Result]
MakeContext = Callable[..., Context]


def tree(*roots: Path) -> dict[str, tuple[int, int]]:
    """Every file under the roots with its size and modification time."""
    found: dict[str, tuple[int, int]] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                found[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return found


# ---- init ------------------------------------------------------------------------------


def test_init_in_an_empty_folder_creates_the_config_and_both_databases(
    invoke: Invoke, ingatan_home: Path
) -> None:
    result = invoke("init", "--business", "tokoku", "--agent", "claude-code")
    assert result.exit_code == 0, result.output
    config = Path.cwd() / ".ingatan.toml"
    assert config.read_text(encoding="utf-8") == 'business = "tokoku"\nagent = "claude-code"\n'
    assert (ingatan_home / "tokoku.db").is_file() and (ingatan_home / "global.db").is_file()
    assert "Created" in result.output


def test_a_second_init_changes_no_file_and_exits_zero(invoke: Invoke, ingatan_home: Path) -> None:
    assert invoke("init", "--business", "tokoku", "--agent", "claude-code").exit_code == 0
    before = tree(Path.cwd(), ingatan_home)
    assert len(before) >= 3
    again = invoke("init", "--business", "tokoku", "--agent", "claude-code")
    assert again.exit_code == 0, again.output
    assert tree(Path.cwd(), ingatan_home) == before
    assert "left unchanged" in again.output


def test_init_never_overwrites_an_existing_config(invoke: Invoke, ingatan_home: Path) -> None:
    config = Path.cwd() / ".ingatan.toml"
    config.write_text('business = "toko-lama"\nagent = "hermes"\n', encoding="utf-8")
    result = invoke("init", "--business", "lainnya", "--agent", "x")
    assert result.exit_code == 0, result.output
    assert config.read_text(encoding="utf-8") == 'business = "toko-lama"\nagent = "hermes"\n'
    assert (ingatan_home / "toko-lama.db").is_file()
    assert not (ingatan_home / "lainnya.db").exists()
    assert "ignored" in result.output


def test_init_without_values_and_without_a_terminal_is_a_clear_error(
    invoke: Invoke, ingatan_home: Path
) -> None:
    result = invoke("init")
    assert result.exit_code == 1
    assert "--business" in result.stderr and "INGATAN_BUSINESS" in result.stderr
    assert not (Path.cwd() / ".ingatan.toml").exists() and not ingatan_home.exists()


def test_init_asks_on_a_terminal_for_missing_values(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, ingatan_home: Path
) -> None:
    monkeypatch.setattr("ingatan.cli.commands._is_interactive", lambda: True)
    result = invoke("init", "--agent", "claude-code", input="tokoku\n")
    assert result.exit_code == 0, result.output
    assert "Business name" in result.output and "Agent name" not in result.output
    assert (
        (Path.cwd() / ".ingatan.toml").read_text(encoding="utf-8").startswith('business = "tokoku"')
    )


def test_init_takes_missing_values_from_the_environment_without_asking(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ingatan.cli.commands._is_interactive", lambda: True)
    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")
    monkeypatch.setenv("INGATAN_AGENT", "hermes")
    result = invoke("init")  # no input: a prompt would abort
    assert result.exit_code == 0, result.output
    assert 'agent = "hermes"' in (Path.cwd() / ".ingatan.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(("business", "agent"), [("Toko Ku", "a"), ("global", "a"), ("ok", "A B")])
def test_init_rejects_invalid_names_and_creates_nothing(
    invoke: Invoke, ingatan_home: Path, business: str, agent: str
) -> None:
    result = invoke("init", "--business", business, "--agent", agent)
    assert result.exit_code == 1 and "name" in result.stderr
    assert not (Path.cwd() / ".ingatan.toml").exists() and not ingatan_home.exists()


def test_init_with_a_broken_existing_config_fails_cleanly(invoke: Invoke) -> None:
    (Path.cwd() / ".ingatan.toml").write_text("business = [", encoding="utf-8")
    result = invoke("init")
    assert result.exit_code == 1 and "not valid TOML" in result.stderr
    assert "Traceback" not in result.output


# ---- entry point -----------------------------------------------------------------------


def fake_server(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    module = types.ModuleType("ingatan.mcp.server")
    module.run = lambda: calls.append("run")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ingatan.mcp.server", module)
    return calls


def test_bare_ingatan_starts_the_server(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_server(monkeypatch)
    assert invoke().exit_code == 0
    assert calls == ["run"]


def test_serve_starts_the_server(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_server(monkeypatch)
    assert invoke("serve").exit_code == 0
    assert calls == ["run"]


def test_a_missing_server_module_is_a_clean_error(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ingatan.mcp.server", None)  # import raises ImportError
    result = invoke("serve")
    assert result.exit_code == 1 and "not available" in result.stderr
    assert "Traceback" not in result.output


def test_version_and_help(invoke: Invoke) -> None:
    result = invoke("--version")
    assert result.exit_code == 0 and result.stdout.strip() == f"ingatan {ingatan.__version__}"
    helped = invoke("--help")
    assert helped.exit_code == 0
    for command in (
        "serve",
        "init",
        "doctor",
        "log",
        "diff",
        "undo",
        "proposals",
        "forget",
        "ingest",
        "conflicts",
        "hook",
    ):
        assert command in helped.output


def test_an_ingatan_error_is_one_line_on_stderr_with_exit_code_one(
    invoke: Invoke, configured: None
) -> None:
    result = invoke("undo", "999")
    assert result.exit_code == 1
    assert result.stderr.strip() == "ingatan: No log entry 999."
    assert result.stdout == ""


def test_main_turns_an_ingatan_error_into_one_line_and_exit_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(**_: Any) -> None:
        raise ConfigError("no business set")

    monkeypatch.setattr(cli, "app", failing)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1
    assert capsys.readouterr().err == "ingatan: no business set\n"


# ---- real process: entry point and UTF-8 (NFR-17) ---------------------------------------

INDONESIAN = "Café Ibu Sri – pesanan “kopi Gayo” dikirim ke Jl. Mawar No. 5… besok"


def run_module(*args: str, input: str | None = None) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"  # a legacy console codepage must not leak into the output
    env.pop("PYTHONIOENCODING", None)
    return subprocess.run(
        [sys.executable, "-m", "ingatan.cli", *args],
        capture_output=True,
        input=None if input is None else input.encode("utf-8"),
        env=env,
        cwd=Path.cwd(),
        timeout=120,
        check=False,
    )


def test_python_m_ingatan_cli_runs_the_entry_point() -> None:
    result = run_module("--version")
    assert result.returncode == 0
    assert result.stdout.decode("utf-8").strip() == f"ingatan {ingatan.__version__}"


def test_a_real_process_prints_indonesian_text_as_utf8(
    cli_ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = facts.remember(cli_ctx, INDONESIAN, keywords=["pesanan", "kopi"])
    assert result.fact_id is not None
    shown = run_module("log")
    assert shown.returncode == 0, shown.stderr
    assert INDONESIAN in shown.stdout.decode("utf-8")
    assert b"\xe2\x80\x93" in shown.stdout  # the en dash is UTF-8 bytes, not '?'
    full = run_module("diff", "1")
    assert full.returncode == 0 and INDONESIAN in full.stdout.decode("utf-8")


def test_a_real_process_reports_errors_in_one_line_with_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_module("log")  # no business and no agent configured
    assert result.returncode == 1
    lines = result.stderr.decode("utf-8").strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("ingatan: No business name is set")


def test_a_real_process_init_then_doctor_passes(make_ctx: MakeContext) -> None:
    created = run_module("init", "--business", "tokoku", "--agent", "claude-code")
    assert created.returncode == 0, created.stderr
    checked = run_module("doctor")
    assert checked.returncode == 0, checked.stdout.decode("utf-8")
    assert "All checks passed." in checked.stdout.decode("utf-8")
