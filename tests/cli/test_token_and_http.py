"""`ingatan token ...`, `ingatan serve --http` and the doctor line for tokens (FR-58)."""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

from ingatan import tokens

Invoke = Callable[..., Result]

TOKEN = re.compile(r"igt_[A-Za-z0-9_-]{43}")


def fake_http_module(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    module = types.ModuleType("ingatan.mcp.http")
    module.run_http = lambda **kwargs: calls.append(kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ingatan.mcp.http", module)
    return calls


# ---- serve --http ----------------------------------------------------------------------


def test_serve_http_without_a_token_exits_non_zero_and_names_the_command(
    invoke: Invoke, ingatan_home: Path
) -> None:
    result = invoke("serve", "--http", "--port", "9")
    assert result.exit_code != 0
    assert "ingatan token add" in result.output
    assert "Traceback" not in result.output


def test_serve_http_passes_host_and_port_on(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_http_module(monkeypatch)
    assert invoke("serve", "--http").exit_code == 0
    assert invoke("serve", "--http", "--host", "0.0.0.0", "--port", "9000").exit_code == 0
    assert calls == [{}, {"host": "0.0.0.0", "port": 9000}]


def test_host_and_port_need_http(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_http_module(monkeypatch)
    for args in (("--host", "0.0.0.0"), ("--port", "9000")):
        result = invoke("serve", *args)
        assert result.exit_code == 1
        assert "--http" in result.output


def test_a_port_out_of_range_is_refused(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_http_module(monkeypatch)
    assert invoke("serve", "--http", "--port", "70000").exit_code != 0
    assert calls == []


# ---- token add, list, revoke -----------------------------------------------------------


def test_token_add_prints_the_token_once_and_stores_only_its_hash(
    invoke: Invoke, ingatan_home: Path
) -> None:
    result = invoke(
        "token", "add", "--business", "kopi", "--agent", "claude-mac", "--label", "MacBook"
    )
    assert result.exit_code == 0, result.output
    found = TOKEN.findall(result.output)
    assert len(found) == 1
    assert result.output.count(found[0]) == 1
    stored = tokens.list_tokens(ingatan_home)
    assert len(stored) == 1
    assert stored[0].id in result.output
    assert found[0].encode() not in tokens.tokens_path(ingatan_home).read_bytes()
    assert tokens.TokenStore(ingatan_home).lookup(found[0]) is not None


def test_token_list_never_shows_the_token_or_its_hash(invoke: Invoke, ingatan_home: Path) -> None:
    added = invoke(
        "token", "add", "--business", "kopi", "--agent", "claude-mac", "--label", "MacBook"
    )
    raw = TOKEN.findall(added.output)[0]
    record = tokens.list_tokens(ingatan_home)[0]
    result = invoke("token", "list")
    assert result.exit_code == 0, result.output
    assert raw not in result.output and raw[4:] not in result.output
    assert record.hash not in result.output
    for expected in (record.id, "kopi", "claude-mac", "MacBook", record.created):
        assert expected in result.output


def test_token_list_with_no_tokens(invoke: Invoke, ingatan_home: Path) -> None:
    result = invoke("token", "list")
    assert result.exit_code == 0
    assert "No tokens" in result.output


def test_token_add_rejects_invalid_names(invoke: Invoke, ingatan_home: Path) -> None:
    for business, agent in (("Kopi Shop", "claude"), ("global", "claude"), ("kopi", "Mac Book")):
        result = invoke("token", "add", "--business", business, "--agent", agent)
        assert result.exit_code == 1, result.output
        assert "Invalid" in result.output or "reserved" in result.output
        assert not TOKEN.search(result.output)
    assert tokens.list_tokens(ingatan_home) == []


def test_token_add_needs_both_names(invoke: Invoke, ingatan_home: Path) -> None:
    assert invoke("token", "add", "--business", "kopi").exit_code != 0
    assert invoke("token", "add", "--agent", "claude").exit_code != 0


def test_token_revoke_removes_the_token(invoke: Invoke, ingatan_home: Path) -> None:
    added = invoke("token", "add", "--business", "kopi", "--agent", "claude-mac")
    raw = TOKEN.findall(added.output)[0]
    token_id = tokens.list_tokens(ingatan_home)[0].id
    result = invoke("token", "revoke", token_id)
    assert result.exit_code == 0, result.output
    assert token_id in result.output
    assert tokens.list_tokens(ingatan_home) == []
    assert tokens.TokenStore(ingatan_home).lookup(raw) is None


def test_token_revoke_of_an_unknown_id_fails_cleanly(invoke: Invoke, ingatan_home: Path) -> None:
    result = invoke("token", "revoke", "t_deadbeef")
    assert result.exit_code == 1
    assert "No token t_deadbeef" in result.output
    assert "Traceback" not in result.output


def test_a_corrupt_token_file_is_a_clean_error(invoke: Invoke, ingatan_home: Path) -> None:
    ingatan_home.mkdir(parents=True)
    tokens.tokens_path(ingatan_home).write_text("{not json", encoding="utf-8")
    for args in (("token", "list"), ("token", "add", "--business", "kopi", "--agent", "a")):
        result = invoke(*args)
        assert result.exit_code == 1
        assert "not valid JSON" in result.output
        assert "Traceback" not in result.output


# ---- doctor ----------------------------------------------------------------------------


def test_doctor_reports_the_token_count_and_zero_is_not_an_error(
    invoke: Invoke, ingatan_home: Path
) -> None:
    assert invoke("init", "--business", "tokoku", "--agent", "claude-code").exit_code == 0
    none = invoke("doctor")
    assert none.exit_code == 0, none.output
    assert re.search(r"^\[pass\] tokens: 0 HTTP access token\(s\)", none.stdout, re.MULTILINE)
    assert invoke("token", "add", "--business", "kopi", "--agent", "claude-mac").exit_code == 0
    one = invoke("doctor")
    assert one.exit_code == 0, one.output
    assert re.search(r"^\[pass\] tokens: 1 HTTP access token\(s\)", one.stdout, re.MULTILINE)
    assert not TOKEN.search(one.output)


def test_doctor_fails_on_a_corrupt_token_file(invoke: Invoke, ingatan_home: Path) -> None:
    assert invoke("init", "--business", "tokoku", "--agent", "claude-code").exit_code == 0
    tokens.tokens_path(ingatan_home).write_text("{not json", encoding="utf-8")
    result = invoke("doctor")
    assert result.exit_code == 1
    assert re.search(r"^\[FAIL\] tokens:", result.stdout, re.MULTILINE)
