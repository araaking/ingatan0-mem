from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ingatan import episodes
from ingatan.integrations import hermes
from ingatan.integrations.hermes import IngatanProvider
from ingatan.scope import Context

KEY = "sk-Ab12Cd34Ef56Gh78Ij90Kl12Mn34"
PLUGIN_DIR = Path(hermes.__file__).parent


@pytest.fixture(autouse=True)
def hermes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")


def rows(ctx: Context) -> list[sqlite3.Row]:
    return ctx.business.execute("SELECT * FROM episodes ORDER BY id").fetchall()


def started(**kwargs: Any) -> IngatanProvider:
    provider = IngatanProvider()
    provider.initialize("sess-1", hermes_home="/tmp/h", platform="cli", **kwargs)
    return provider


def test_sync_turn_records_the_turn_for_agent_hermes(make_ctx: Any) -> None:
    provider = started(agent_context="primary")
    assert provider.sync_turn("Berapa harga kopi?", "Rp 18.000 — ✓", session_id="sess-9") is None
    [row] = rows(make_ctx(agent="hermes"))
    assert (row["agent"], row["session"]) == ("hermes", "sess-9")
    assert (row["user_text"], row["assistant_text"]) == ("Berapa harga kopi?", "Rp 18.000 — ✓")


def test_sync_turn_stores_only_the_two_texts_not_messages(make_ctx: Any) -> None:
    provider = started()
    messages = [
        {"role": "tool", "content": "zztooloutputzz"},
        {"role": "assistant", "tool_calls": [{"function": {"arguments": "zzargszz"}}]},
    ]
    provider.sync_turn(
        "halo", "hai", session_id="s", messages=messages, turn_author={"name": "zzauthorzz"}
    )
    [row] = rows(make_ctx(agent="hermes"))
    assert "zz" not in row["user_text"] + row["assistant_text"] + (row["session"] or "")


def test_secret_redacted_and_long_text_truncated(make_ctx: Any) -> None:
    provider = started()
    provider.sync_turn(f"kunci {KEY}", "a" * 5000)
    [row] = rows(make_ctx(agent="hermes"))
    assert KEY not in row["user_text"]
    assert len(row["assistant_text"]) == episodes.TRUNCATE_AT


@pytest.mark.parametrize("context", ["subagent", "cron", "flush"])
def test_non_primary_contexts_are_skipped(make_ctx: Any, context: str) -> None:
    provider = started(agent_context=context)
    provider.sync_turn("halo", "hai")
    assert rows(make_ctx(agent="hermes")) == []


def test_agent_name_comes_from_env_when_set(monkeypatch: pytest.MonkeyPatch, make_ctx: Any) -> None:
    monkeypatch.setenv("INGATAN_AGENT", "hermes-vps")
    started().sync_turn("halo", "hai")
    assert [r["agent"] for r in rows(make_ctx(agent="hermes-vps"))] == ["hermes-vps"]


def test_unknown_and_missing_arguments_never_raise(make_ctx: Any) -> None:
    provider = IngatanProvider()  # initialize never called
    provider.sync_turn("halo", "hai", future_keyword=1)
    provider.sync_turn()
    provider.sync_turn(None, None)  # type: ignore[arg-type]
    provider.initialize(session_id="s", unknown="x", agent_context=None, resumed=None)
    assert provider.get_tool_schemas(foo=1) == []
    assert provider.prefetch("q", session_id="s", extra=1) == ""
    assert provider.queue_prefetch("q", session_id="s", extra=1) is None
    assert provider.on_turn_start(3, "hi", extra=1) is None
    assert provider.on_session_end([{"role": "user"}], extra=1) is None
    assert provider.on_session_end() is None
    assert provider.on_session_switch("s2", parent_session_id="s", reset=True, extra=1) is None
    assert provider.on_pre_compress([]) == ""
    assert provider.on_memory_write("add", "memory", "x", metadata={}) is None
    assert provider.shutdown() is None
    assert "mcp_ingatan" in provider.handle_tool_call("anything", {})
    assert len(rows(make_ctx(agent="hermes"))) == 1  # only the one real turn


def test_sync_turn_without_scope_or_with_broken_database_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, make_ctx: Any, ingatan_home: Path
) -> None:
    monkeypatch.delenv("INGATAN_BUSINESS")
    provider = started()
    assert provider.is_available() is False
    provider.sync_turn("halo", "hai")
    assert not ingatan_home.exists()

    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")
    assert provider.is_available() is True

    def locked(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("ingatan.db.connect", locked)
    provider.sync_turn("halo", "hai")
    assert "database is locked" in (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")


def test_a_failing_capture_call_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(episodes, "capture", boom)
    started().sync_turn("halo", "hai")  # must not raise


def test_tool_schemas_are_empty_and_name_is_ingatan() -> None:
    provider = IngatanProvider()
    assert provider.get_tool_schemas() == []
    assert provider.name == "ingatan"


# ------------------------------------------------------------------ system prompt block


def test_system_prompt_block_is_the_brief_and_built_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_build(ctx: Context, *, offer_episodes: bool = True) -> str:
        calls.append((ctx.scope.agent, offer_episodes))
        return "# Ingatan brief: tokoku / hermes"

    monkeypatch.setattr("ingatan.brief.build", fake_build, raising=False)
    provider = started(agent_context="primary")
    assert provider.system_prompt_block() == "# Ingatan brief: tokoku / hermes"
    assert provider.system_prompt_block(extra=1) == "# Ingatan brief: tokoku / hermes"
    assert calls == [("hermes", True)]
    provider.on_session_switch("sess-2")
    provider.system_prompt_block()
    assert len(calls) == 2


@pytest.mark.parametrize("kwargs", [{"agent_context": "subagent"}, {"resumed": True}])
def test_no_digest_offer_for_subagents_and_resumed_sessions(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    flags: list[bool] = []

    def fake_build(ctx: Context, *, offer_episodes: bool = True) -> str:
        flags.append(offer_episodes)
        return "brief"

    monkeypatch.setattr("ingatan.brief.build", fake_build, raising=False)
    assert started(**kwargs).system_prompt_block() == "brief"
    assert flags == [False]


def test_system_prompt_block_failure_gives_empty_string(
    monkeypatch: pytest.MonkeyPatch, ingatan_home: Path
) -> None:
    def boom(ctx: Context, **kwargs: Any) -> str:
        raise RuntimeError("no brief today")

    monkeypatch.setattr("ingatan.brief.build", boom, raising=False)
    assert started().system_prompt_block() == ""
    assert "no brief today" in (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")


def test_system_prompt_block_without_scope_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGATAN_BUSINESS")
    assert started().system_prompt_block() == ""


# ------------------------------------------------------------------ packaging and loading


def test_register_hands_a_new_provider_to_hermes() -> None:
    registered: list[Any] = []

    class FakeCtx:
        def register_memory_provider(self, provider: Any) -> None:
            registered.append(provider)

    hermes.register(FakeCtx())
    hermes.register(FakeCtx())
    assert len(registered) == 2
    assert registered[0] is not registered[1]
    assert isinstance(registered[0], IngatanProvider)


def test_plugin_loaded_from_a_copied_folder_uses_the_hermes_base_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copy the folder like `$HERMES_HOME/plugins/ingatan/` and load it by file path, with a
    fake `agent.memory_provider` in place of Hermes."""
    import shutil

    class FakeBase:
        marker = "hermes-base"

    agent = types.ModuleType("agent")
    module = types.ModuleType("agent.memory_provider")
    module.MemoryProvider = FakeBase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.memory_provider", module)

    target = tmp_path / "plugins" / "ingatan"
    shutil.copytree(PLUGIN_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location(
        "hermes_plugin_ingatan_copy",
        target / "__init__.py",
        submodule_search_locations=[str(target)],
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hermes_plugin_ingatan_copy", loaded)
    spec.loader.exec_module(loaded)
    provider_class: Any = loaded.IngatanProvider
    assert issubclass(provider_class, FakeBase)
    instance: Any = provider_class()
    assert instance.name == "ingatan"


def test_manifest_is_a_static_file_with_the_expected_fields() -> None:
    text = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8")
    fields = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line and not line.startswith(("#", " ", "-")) and ":" in line
    }
    assert fields["name"] == "ingatan"
    assert fields["manifest_version"] == "1"
    assert fields["kind"] == "exclusive"
    assert "sync_turn" in text
