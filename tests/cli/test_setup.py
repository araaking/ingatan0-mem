"""`ingatan setup`: project, semantic search, client and a health check in one command."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

from ingatan.scope import EmbeddingConfig, resolve_scope
from ingatan.search import embed as embed_store
from ingatan.search import local as local_models

Invoke = Callable[..., Result]

BASE = ("setup", "--business", "tokoku", "--agent", "claude-code")
OTHER_SERVERS = (
    '{\n  "mcpServers": {\n    "other": {\n      "command": "other-server"\n    }\n  }\n}\n'
)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingatan.cli.commands._is_interactive", lambda: True)


def _files(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def test_setup_with_all_flags_asks_nothing(invoke: Invoke, ingatan_home: Path) -> None:
    result = invoke(*BASE, "--no-semantic", "--client", "none")
    assert result.exit_code == 0, result.output
    config = Path.cwd() / ".ingatan.toml"
    assert config.read_text(encoding="utf-8") == 'business = "tokoku"\nagent = "claude-code"\n'
    assert (ingatan_home / "tokoku.db").is_file()
    assert (ingatan_home / "global.db").is_file()
    assert "[pass] config: business tokoku, agent claude-code" in result.output
    assert "All checks passed." in result.output


def test_setup_without_terminal_and_missing_flag_fails_clearly(invoke: Invoke) -> None:
    result = invoke(*BASE, "--no-semantic")
    assert result.exit_code == 1, result.output
    assert "--client" in result.output
    assert "no terminal" in result.output


def test_setup_without_terminal_needs_the_semantic_flag(invoke: Invoke) -> None:
    result = invoke(*BASE, "--client", "none")
    assert result.exit_code == 1, result.output
    assert "--semantic" in result.output


def test_setup_without_terminal_needs_the_names(invoke: Invoke) -> None:
    result = invoke("setup", "--no-semantic", "--client", "none")
    assert result.exit_code == 1, result.output
    assert "setup needs a business name" in result.output


def test_setup_rejects_an_unknown_client(invoke: Invoke) -> None:
    result = invoke(*BASE, "--no-semantic", "--client", "emacs")
    assert result.exit_code == 1, result.output
    assert "claude-code, hermes, none" in result.output


def test_setup_asks_on_a_terminal_for_what_was_not_passed(
    invoke: Invoke, terminal: None, ingatan_home: Path
) -> None:
    result = invoke("setup", input="tokoku\nclaude-code\nn\nnone\n")
    assert result.exit_code == 0, result.output
    assert resolve_scope().business == "tokoku"
    assert "All checks passed." in result.output


def test_setup_shows_diff_and_writes_nothing_on_no(invoke: Invoke, terminal: None) -> None:
    mcp = Path.cwd() / ".mcp.json"
    mcp.write_text(OTHER_SERVERS, encoding="utf-8")
    result = invoke(*BASE, "--no-semantic", "--client", "claude-code", input="n\n")
    assert result.exit_code == 0, result.output
    assert mcp.read_text(encoding="utf-8") == OTHER_SERVERS
    assert '+    "ingatan": {' in result.output
    assert "claude mcp add" in result.output


def test_setup_writes_the_claude_code_entry_on_yes(invoke: Invoke, terminal: None) -> None:
    mcp = Path.cwd() / ".mcp.json"
    mcp.write_text(OTHER_SERVERS, encoding="utf-8")
    result = invoke(*BASE, "--no-semantic", "--client", "claude-code", input="y\n")
    assert result.exit_code == 0, result.output
    servers = json.loads(mcp.read_text(encoding="utf-8"))["mcpServers"]
    assert list(servers) == ["other", "ingatan"]
    assert servers["ingatan"] == {"command": "ingatan", "args": ["serve"]}


def test_setup_yes_flag_writes_without_asking(invoke: Invoke) -> None:
    result = invoke(*BASE, "--no-semantic", "--client", "claude-code", "--yes")
    assert result.exit_code == 0, result.output
    data = json.loads((Path.cwd() / ".mcp.json").read_text(encoding="utf-8"))
    assert data == {"mcpServers": {"ingatan": {"command": "ingatan", "args": ["serve"]}}}
    assert '+    "ingatan": {' in result.output  # the change is shown even when not asked


def test_setup_without_terminal_and_without_yes_writes_no_client_file(invoke: Invoke) -> None:
    result = invoke(*BASE, "--no-semantic", "--client", "claude-code")
    assert result.exit_code == 1, result.output
    assert "--yes" in result.output
    assert not (Path.cwd() / ".mcp.json").exists()


def test_setup_twice_changes_nothing(invoke: Invoke) -> None:
    first = invoke(*BASE, "--no-semantic", "--client", "claude-code", "--yes")
    assert first.exit_code == 0, first.output
    before = {name: (Path.cwd() / name).read_bytes() for name in _files(Path.cwd())}
    # No --yes this time: with nothing to write there is nothing to confirm.
    second = invoke(*BASE, "--no-semantic", "--client", "claude-code")
    assert second.exit_code == 0, second.output
    assert "already" in second.output
    assert {name: (Path.cwd() / name).read_bytes() for name in _files(Path.cwd())} == before


def test_setup_leaves_an_entry_that_starts_the_server_another_way(invoke: Invoke) -> None:
    mine = '{"mcpServers": {"ingatan": {"command": "uvx", "args": ["ingatan", "serve"]}}}'
    (Path.cwd() / ".mcp.json").write_text(mine, encoding="utf-8")
    result = invoke(*BASE, "--no-semantic", "--client", "claude-code", "--yes")
    assert result.exit_code == 0, result.output
    assert (Path.cwd() / ".mcp.json").read_text(encoding="utf-8") == mine
    assert "started with `uvx ingatan serve`" in result.output
    assert "Left as it is" in result.output


def test_setup_hermes_prints_block_and_writes_no_file(
    invoke: Invoke, tmp_path: Path, ingatan_home: Path
) -> None:
    result = invoke(*BASE, "--no-semantic", "--client", "hermes")
    assert result.exit_code == 0, result.output
    assert "mcp_servers:" in result.output
    assert 'INGATAN_BUSINESS: "tokoku"' in result.output
    assert 'INGATAN_AGENT: "claude-code"' in result.output
    # Hermes hands a server only the variables of this block: without the data folder in
    # it, the server would open another store than the one this command just set up.
    assert f"INGATAN_HOME: '{ingatan_home}'" in result.output
    assert _files(Path.cwd()) == [".ingatan.toml"]
    # Nothing was written into the (fake) user home, where a Hermes folder would live.
    assert _files(tmp_path / "user-home") == []


def test_setup_with_semantic_switches_the_local_model_on(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_pull(model_name: str, model_dir: object, **_kwargs: Any) -> list[Path]:
        return []

    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(local_models, "pull", fake_pull)
    monkeypatch.setattr(local_models, "missing_packages", lambda: [])
    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)
    monkeypatch.delenv("INGATAN_EMBED_PROVIDER")
    result = invoke(*BASE, "--semantic", "--client", "none")
    assert result.exit_code == 0, result.output
    assert resolve_scope().embeddings.provider == "local"
    assert "Semantic search is on" in result.output
    assert "All checks passed." in result.output


def test_setup_exits_non_zero_when_a_check_fails(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The project already asks for the local model, but its packages are not installed.
    (Path.cwd() / ".ingatan.toml").write_text(
        'business = "tokoku"\nagent = "claude-code"\n\n[embeddings]\nprovider = "local"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("INGATAN_EMBED_PROVIDER")
    monkeypatch.setattr(local_models, "missing_packages", lambda: ["onnxruntime"])
    result = invoke("setup", "--no-semantic", "--client", "none")
    assert result.exit_code == 1, result.output
    assert "check(s) failed." in result.output
