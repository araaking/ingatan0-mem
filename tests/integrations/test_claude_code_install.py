"""The change `ingatan setup` proposes for Claude Code's project file `.mcp.json`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingatan.errors import ConfigError
from ingatan.integrations.claude_code import install

ENTRY = {"command": "ingatan", "args": ["serve"]}


def test_change_for_a_missing_file_holds_only_the_ingatan_server(tmp_path: Path) -> None:
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    assert change.path == tmp_path / ".mcp.json"
    assert change.before == ""
    assert json.loads(change.after) == {"mcpServers": {"ingatan": ENTRY}}
    assert change.after.endswith("\n")


def test_computing_a_change_writes_nothing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    assert install.mcp_entry_change(project) is not None
    assert list(project.iterdir()) == []


def test_claude_code_change_keeps_other_servers(tmp_path: Path) -> None:
    path = tmp_path / ".mcp.json"
    path.write_text(
        "{\n"
        '    "mcpServers": {\n'
        '        "other": {"command": "other-server", "args": []}\n'
        "    },\n"
        '    "extra": true\n'
        "}\n",
        encoding="utf-8",
    )
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    data = json.loads(change.after)
    assert list(data) == ["mcpServers", "extra"]
    assert list(data["mcpServers"]) == ["other", "ingatan"]
    assert data["mcpServers"]["other"] == {"command": "other-server", "args": []}
    assert data["mcpServers"]["ingatan"] == ENTRY
    # The file's own indentation is kept, so the diff shows only what is added.
    assert '\n    "mcpServers": {\n' in change.after


def test_a_file_without_servers_gains_the_key(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text('{"extra": 1}\n', encoding="utf-8")
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    assert json.loads(change.after) == {"extra": 1, "mcpServers": {"ingatan": ENTRY}}


def test_no_change_when_the_server_is_already_there(tmp_path: Path) -> None:
    entry = {**ENTRY, "env": {"INGATAN_BUSINESS": "tokoku"}}
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"ingatan": entry}}), encoding="utf-8"
    )
    assert install.mcp_entry_change(tmp_path) is None


@pytest.mark.parametrize(
    ("entry", "started_with"),
    [
        ({"command": "uvx", "args": ["ingatan", "serve"]}, "uvx ingatan serve"),
        ({"command": "/opt/tools/bin/ingatan", "args": ["serve"]}, "/opt/tools/bin/ingatan serve"),
        ({"command": "ingatan"}, "ingatan"),
        ({"command": "ingatan", "args": ["serve", "--http"]}, "ingatan serve --http"),
        ({"type": "http", "url": "http://127.0.0.1:8765/mcp"}, "http://127.0.0.1:8765/mcp"),
    ],
)
def test_an_entry_that_starts_the_server_another_way_is_left_alone(
    tmp_path: Path, entry: dict[str, object], started_with: str
) -> None:
    """A path or a launcher in that entry is somebody's choice, and may be what makes it work."""
    entry = {**entry, "env": {"INGATAN_AGENT": "claude-code"}}
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"ingatan": entry}}), encoding="utf-8"
    )
    assert install.mcp_entry_change(tmp_path) is None
    assert install.own_start(tmp_path) == started_with


def test_own_start_is_none_for_no_entry_and_for_the_default_one(tmp_path: Path) -> None:
    assert install.own_start(tmp_path) is None
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"ingatan": ENTRY, "other": {"command": "x"}}}), encoding="utf-8"
    )
    assert install.own_start(tmp_path) is None


def test_an_entry_that_starts_nothing_is_completed_and_keeps_its_env(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"ingatan": {"env": {"INGATAN_AGENT": "claude-code"}}}}),
        encoding="utf-8",
    )
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    assert json.loads(change.after)["mcpServers"]["ingatan"] == {
        **ENTRY,
        "env": {"INGATAN_AGENT": "claude-code"},
    }
    assert install.own_start(tmp_path) is None


@pytest.mark.parametrize("content", ["[]", '{"mcpServers": []}', "{", '"text"'])
def test_a_file_that_cannot_be_merged_is_refused(tmp_path: Path, content: str) -> None:
    path = tmp_path / ".mcp.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\.mcp\.json"):
        install.mcp_entry_change(tmp_path)
    assert path.read_text(encoding="utf-8") == content


def test_crlf_line_endings_are_kept(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_bytes(b'{\r\n  "mcpServers": {}\r\n}\r\n')
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    assert "\n" not in change.after.replace("\r\n", "")


def test_write_change_writes_exactly_the_after_text(tmp_path: Path) -> None:
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    install.write_change(change)
    assert change.path.read_bytes() == change.after.encode("utf-8")
    assert install.mcp_entry_change(tmp_path) is None


def test_diff_shows_only_added_lines_for_a_new_server(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(
        '{\n  "mcpServers": {\n    "other": {\n      "command": "other-server"\n    }\n  }\n}\n',
        encoding="utf-8",
    )
    change = install.mcp_entry_change(tmp_path)
    assert change is not None
    lines = install.diff_lines(change)
    removed = [ln for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    # Nothing of the other server is taken away or rewritten.
    assert removed == []
    assert '+    "ingatan": {' in added
    assert not any("other" in ln for ln in added)
