"""The seven tools, in-process (fast, detailed) and as a real subprocess (end to end)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

import pytest

try:
    from mcp.client.stdio import StdioServerParameters

    from mcp import Client
except ImportError as exc:  # pragma: no cover - only on machines that block pydantic_core
    if "Application Control" in str(exc):
        pytest.skip(f"the mcp SDK cannot be imported here: {exc}", allow_module_level=True)
    raise

from ingatan import episodes
from ingatan.mcp.server import build_server
from ingatan.scope import Context, Scope, open_context

T = TypeVar("T")
TOOLS = ["brief", "recall", "remember", "supersede", "confirm", "forget", "history"]
FACT_ID = re.compile(r"f_[0-9a-f]{12}")


def run_async(awaitable: Awaitable[T], timeout: float = 90) -> T:
    async def guarded() -> T:
        return await asyncio.wait_for(awaitable, timeout)

    return asyncio.run(guarded())


def text_of(result: Any) -> str:
    return "\n".join(part.text for part in result.content if part.type == "text")


def property_names(schema: Any) -> list[str]:
    """Every property name anywhere inside a JSON schema, at any depth."""
    names: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():  # pyright: ignore[reportUnknownVariableType]
            if key == "properties" and isinstance(value, dict):
                names.extend(value)  # pyright: ignore[reportUnknownArgumentType]
            names.extend(property_names(value))
    elif isinstance(schema, list):
        for item in schema:  # pyright: ignore[reportUnknownVariableType]
            names.extend(property_names(item))
    return names


def forbidden(names: list[str]) -> list[str]:
    return [n for n in names if re.search(r"business|agent|tenant|scope", n, re.IGNORECASE)]


def call(ctx: Context, name: str, arguments: dict[str, Any] | None = None) -> Any:
    async def go() -> Any:
        async with Client(build_server(ctx)) as client:
            return await client.call_tool(name, arguments or {})

    return run_async(go())


# ----------------------------------------------------------- in-process


def test_exactly_seven_tools_with_descriptions(ctx: Context) -> None:
    async def go() -> list[Any]:
        async with Client(build_server(ctx)) as client:
            return (await client.list_tools()).tools

    tools = run_async(go())
    assert sorted(t.name for t in tools) == sorted(TOOLS)
    for tool in tools:
        assert tool.description
        assert len(tool.description) > 80, tool.name


def test_no_schema_has_a_business_or_agent_field(ctx: Context) -> None:
    async def go() -> list[Any]:
        async with Client(build_server(ctx)) as client:
            return (await client.list_tools()).tools

    for tool in run_async(go()):
        assert forbidden(property_names(tool.input_schema)) == [], tool.name
    # `tier` is the only scope-like choice, and keywords is optional in the schema.
    schemas = {t.name: t.input_schema for t in run_async(go())}
    assert "tier" in schemas["remember"]["properties"]
    assert "keywords" not in schemas["remember"].get("required", [])
    assert schemas["remember"]["required"] == ["text"]


def test_schema_walker_catches_nested_fields() -> None:
    nested = {"properties": {"a": {"items": {"properties": {"Business_Name": {}}}}}}
    assert forbidden(property_names(nested)) == ["Business_Name"]


def test_descriptions_teach_the_agent(ctx: Context) -> None:
    async def go() -> dict[str, str]:
        async with Client(build_server(ctx)) as client:
            return {t.name: t.description or "" for t in (await client.list_tools()).tools}

    docs = run_async(go())
    remember = docs["remember"].lower()
    for word in ("synonym", "root word", "indonesian", "supersede", "confirm", "other fact id"):
        assert word in remember
    assert "check first" in docs["brief"].lower()
    assert "once" in docs["brief"].lower()
    assert "possible conflict" in docs["recall"].lower()
    assert "reason" in docs["confirm"].lower()
    assert "cannot be undone" in docs["forget"].lower()


def test_remember_without_keywords_is_an_error_result_and_saves_nothing(ctx: Context) -> None:
    result = call(ctx, "remember", {"text": "Harga kopi Rp 18.000"})
    assert result.is_error
    assert "keywords" in text_of(result)
    assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_recall_lists_id_text_source_date_status(ctx: Context) -> None:
    saved = call(ctx, "remember", {"text": "Harga kopi susu Rp 18.000", "keywords": ["kopi"]})
    fact_id = FACT_ID.search(text_of(saved))
    assert fact_id is not None
    out = text_of(call(ctx, "recall", {"query": "kopi"}))
    assert f"[{fact_id.group(0)}] Harga kopi susu Rp 18.000" in out
    assert "source: claude-code" in out
    assert re.search(r"date: \d{4}-\d{2}-\d{2}", out)
    assert "status: valid" in out
    assert "Document excerpts" not in out


def test_recall_with_no_match_suggests_other_wordings(ctx: Context) -> None:
    out = text_of(call(ctx, "recall", {"query": "tidak ada apa apa"}))
    assert "No facts found" in out
    assert "synonyms" in out


def test_document_excerpts_come_in_a_separate_labelled_block(ctx: Context) -> None:
    ctx.business.execute(
        "INSERT INTO notes (path, heading, text, file_hash, created_at) VALUES "
        "('c:/docs/menu.md', 'Menu kopi', 'Daftar kopi: espresso dan latte', 'h', "
        "'2026-11-11T09:00:00Z')"
    )
    call(ctx, "remember", {"text": "Kopi favorit pelanggan adalah latte", "keywords": ["kopi"]})
    out = text_of(call(ctx, "recall", {"query": "kopi"}))
    facts_part, _, notes_part = out.partition("Document excerpts")
    assert "Kopi favorit pelanggan" in facts_part
    assert "espresso" not in facts_part
    assert "not facts" in notes_part
    assert "c:/docs/menu.md > Menu kopi" in notes_part
    assert "espresso dan latte" in notes_part


def test_conflict_and_check_first_hints_are_in_recall(ctx: Context) -> None:
    call(
        ctx,
        "remember",
        {
            "text": "Kopi susu Rp 18.000",
            "keywords": ["kopi"],
            "entities": ["kopi susu"],
            "kind": "stable",
        },
    )
    second = text_of(
        call(
            ctx,
            "remember",
            {
                "text": "Kopi susu Rp 20.000",
                "keywords": ["kopi"],
                "entities": ["kopi susu"],
                "kind": "stable",
            },
        )
    )
    assert "Similar facts:" in second
    out = text_of(call(ctx, "recall", {"query": "kopi susu"}))
    assert "status: possible conflict" in out
    assert "supersede(old, new)" in out


def test_business_tier_remember_returns_a_proposal_and_brief_counts_it(ctx: Context) -> None:
    out = text_of(
        call(
            ctx,
            "remember",
            {"text": "Jam buka toko 08.00", "keywords": ["jam"], "tier": "business"},
        )
    )
    assert re.search(r"p_[0-9a-f]{12}", out)
    assert "waiting for the user's approval" in out
    assert "No facts found" in text_of(call(ctx, "recall", {"query": "jam buka"}))
    assert "waiting for the user: 1" in text_of(call(ctx, "brief"))


def test_forget_and_history_and_errors(ctx: Context) -> None:
    saved = text_of(call(ctx, "remember", {"text": "Nomor rahasia 12345", "keywords": ["nomor"]}))
    fact_id = FACT_ID.search(saved)
    assert fact_id is not None
    forgotten = call(ctx, "forget", {"id": fact_id.group(0)})
    assert not forgotten.is_error
    assert text_of(forgotten).startswith(f"Forgot {fact_id.group(0)}")
    assert "12345" not in text_of(call(ctx, "history", {"id_or_entity": fact_id.group(0)}))
    missing = call(ctx, "forget", {"id": "f_000000000000"})
    assert missing.is_error
    assert "f_000000000000" in text_of(missing)


def test_forget_of_a_business_fact_is_refused(ctx: Context, insert_fact: Any) -> None:
    fact_id = insert_fact(ctx.business, text="fakta bisnis", tier="business")
    result = call(ctx, "forget", {"id": fact_id})
    assert result.is_error
    assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 1


def test_confirm_without_a_reason_is_an_error(ctx: Context) -> None:
    saved = text_of(call(ctx, "remember", {"text": "Harga gula Rp 15.000", "keywords": ["gula"]}))
    fact_id = FACT_ID.search(saved)
    assert fact_id is not None
    result = call(ctx, "confirm", {"id": fact_id.group(0), "reason": "  "})
    assert result.is_error


def test_brief_tool_offers_episodes(ctx: Context) -> None:
    episodes.record(ctx, "tolong catat harga kopi naik", "Baik, harga kopi naik jadi 20 ribu")
    out = text_of(call(ctx, "brief"))
    assert "## Episodes to digest" in out
    assert "harga kopi naik" in out


def test_tools_use_only_the_fixed_scope(ctx: Context, make_ctx: Any) -> None:
    other = make_ctx(agent="hermes")
    call(other, "remember", {"text": "rahasia hermes", "keywords": ["rahasia"]})
    assert "No facts found" in text_of(call(ctx, "recall", {"query": "rahasia hermes"}))
    assert "rahasia hermes" not in text_of(call(ctx, "brief"))


# --------------------------------------------------------------- end to end


def server_env(home: Path, **extra: str) -> dict[str, str]:
    env = {"INGATAN_HOME": str(home)}
    for key in ("PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP"):  # PYTHONPATH: odd dev setups
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(extra)
    return env


def params(home: Path, cwd: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ingatan.mcp.server"],
        env=server_env(home, INGATAN_BUSINESS="demo", INGATAN_AGENT="e2e-agent"),
        cwd=cwd,
    )


def test_end_to_end_over_stdio(ingatan_home: Path, tmp_path: Path) -> None:
    async def scenario() -> None:
        async with Client(params(ingatan_home, tmp_path)) as client:
            tools = (await client.list_tools()).tools
            assert sorted(t.name for t in tools) == sorted(TOOLS)
            assert len(tools) == 7
            for tool in tools:
                assert forbidden(property_names(tool.input_schema)) == [], tool.name

            async def use(name: str, **arguments: Any) -> Any:
                return await client.call_tool(name, arguments)

            bad = await use("remember", text="Harga kopi susu Rp 18.000")
            assert bad.is_error
            assert "keywords" in text_of(bad)

            first = text_of(
                await use(
                    "remember",
                    text="Harga kopi susu Rp 18.000",
                    keywords=["kopi", "harga", "price"],
                    entities=["kopi susu"],
                )
            )
            old_id = FACT_ID.search(first)
            assert old_id is not None
            second = text_of(
                await use(
                    "remember",
                    text="Harga kopi susu Rp 20.000",
                    keywords=["kopi", "harga", "price"],
                    entities=["kopi susu"],
                )
            )
            new_id = FACT_ID.search(second)
            assert new_id is not None
            assert old_id.group(0) in second  # listed as similar

            recalled = text_of(await use("recall", query="harga kopi susu"))
            assert old_id.group(0) in recalled
            assert new_id.group(0) in recalled
            assert "possible conflict" in recalled

            brief = text_of(await use("brief"))
            assert brief.startswith("# Ingatan brief: demo / e2e-agent")
            assert "waiting for the user: 0" in brief
            assert new_id.group(0) in brief

            done = await use("supersede", old=old_id.group(0), new=new_id.group(0))
            assert not done.is_error
            after = text_of(await use("recall", query="harga kopi susu"))
            assert old_id.group(0) not in after
            assert new_id.group(0) in after

            confirmed = await use(
                "confirm", id=new_id.group(0), reason="Dicek langsung dengan pemilik toko"
            )
            assert not confirmed.is_error

            history = text_of(await use("history", id_or_entity=old_id.group(0)))
            assert "supersede" in history
            assert "create" in history
            by_entity = text_of(await use("history", id_or_entity="kopi susu"))
            assert new_id.group(0) in by_entity

            gone = await use("forget", id=new_id.group(0))
            assert not gone.is_error
            assert new_id.group(0) not in text_of(await use("recall", query="harga kopi susu"))

    run_async(scenario(), timeout=180)


def test_stdout_carries_only_protocol(ingatan_home: Path, tmp_path: Path) -> None:
    env = {**os.environ, **server_env(ingatan_home, INGATAN_BUSINESS="demo", INGATAN_AGENT="a1")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "ingatan.mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
    )
    assert proc.stdin is not None
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0"},
        },
    }
    proc.stdin.write((json.dumps(request) + "\n").encode())
    proc.stdin.flush()
    proc.stdin.close()
    out, _err = proc.communicate(timeout=60)
    lines = [line for line in out.decode("utf-8").splitlines() if line.strip()]
    assert lines, "the server answered nothing"
    for line in lines:
        assert json.loads(line)["jsonrpc"] == "2.0"


def test_config_error_goes_to_stderr_with_a_nonzero_exit(
    ingatan_home: Path, tmp_path: Path
) -> None:
    env = {**os.environ, **server_env(ingatan_home)}
    env.pop("INGATAN_BUSINESS", None)
    env.pop("INGATAN_AGENT", None)
    done = subprocess.run(
        [sys.executable, "-m", "ingatan.mcp.server"],
        input=b"",
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert done.returncode != 0
    assert done.stdout == b""
    err = done.stderr.decode("utf-8")
    assert "INGATAN_BUSINESS" in err
    assert ".ingatan.toml" in err


def test_server_module_does_not_open_a_scope_on_import() -> None:
    from ingatan.mcp import server

    assert callable(server.run)
    assert callable(server.build_server)


def test_open_context_scope_matches_the_server_scope(ingatan_home: Path) -> None:
    ctx = open_context(Scope(business="x", agent="y", home=ingatan_home))
    try:
        assert build_server(ctx).name == "ingatan"
    finally:
        ctx.close()


def test_opening_is_retried_while_another_process_creates_the_files(
    monkeypatch: pytest.MonkeyPatch, ingatan_home: Path
) -> None:
    from ingatan.mcp import server

    real = server.open_context
    calls: list[int] = []

    def flaky(scope: Scope) -> Context:
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")
        return real(scope)

    monkeypatch.setattr(server, "open_context", flaky)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    ctx = server._open_with_retry(Scope(business="x", agent="y", home=ingatan_home))  # pyright: ignore[reportPrivateUsage]
    ctx.close()
    assert len(calls) == 3


def test_other_database_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, ingatan_home: Path
) -> None:
    from ingatan.mcp import server

    def broken(scope: Scope) -> Context:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(server, "open_context", broken)
    with pytest.raises(sqlite3.OperationalError):
        server._open_with_retry(Scope(business="x", agent="y", home=ingatan_home))  # pyright: ignore[reportPrivateUsage]
