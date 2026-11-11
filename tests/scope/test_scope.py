from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ingatan import db
from ingatan.errors import ConfigError, ValidationError
from ingatan.scope import (
    Context,
    Scope,
    data_home,
    db_path,
    find_fact,
    open_context,
    resolve_scope,
    validate_agent,
    validate_business,
    visible_sql,
)

MakeContext = Callable[..., Context]
InsertFact = Callable[..., str]

TOML = 'business = "tokoku"\nagent = "claude-code"\n\n[brief]\ntoken_budget = 750\n'


# ---- names ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tokoku", "a", "toko-1_b", "0abc", "x" * 64])
def test_valid_names(name: str) -> None:
    assert validate_business(name) == name
    assert validate_agent(name) == name


@pytest.mark.parametrize(
    "name", ["", "Tokoku", "-toko", "_toko", "toko kopi", "toko/kopi", "..", "x" * 65, "tókó"]
)
def test_invalid_names(name: str) -> None:
    with pytest.raises(ConfigError):
        validate_business(name)
    with pytest.raises(ConfigError):
        validate_agent(name)


def test_global_is_reserved_for_businesses_but_fine_for_agents() -> None:
    with pytest.raises(ConfigError, match="reserved"):
        validate_business("global")
    assert validate_agent("global") == "global"


# ---- data home and paths ----------------------------------------------------------


def test_data_home_from_env_and_default(tmp_path: Path) -> None:
    assert data_home({"INGATAN_HOME": str(tmp_path / "x")}) == tmp_path / "x"
    assert data_home({}) == Path.home() / ".ingatan"
    assert data_home({"INGATAN_HOME": "  "}) == Path.home() / ".ingatan"


def test_data_home_uses_the_process_environment_by_default(ingatan_home: Path) -> None:
    assert data_home() == ingatan_home


def test_db_path(tmp_path: Path) -> None:
    assert db_path(tmp_path, "tokoku") == tmp_path / "tokoku.db"
    assert db_path(tmp_path, "global") == tmp_path / "global.db"
    with pytest.raises(ConfigError):
        db_path(tmp_path, "../evil")


# ---- config resolution ------------------------------------------------------------


def test_scope_from_config_file(tmp_path: Path) -> None:
    (tmp_path / ".ingatan.toml").write_text(TOML, encoding="utf-8")
    scope = resolve_scope(tmp_path, env={"INGATAN_HOME": str(tmp_path / "h")})
    assert scope == Scope("tokoku", "claude-code", tmp_path / "h", 750)


def test_config_file_is_found_in_a_parent_folder(tmp_path: Path) -> None:
    (tmp_path / ".ingatan.toml").write_text(TOML, encoding="utf-8")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert resolve_scope(deep, env={}).business == "tokoku"


def test_nearest_config_wins(tmp_path: Path) -> None:
    (tmp_path / ".ingatan.toml").write_text(TOML, encoding="utf-8")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / ".ingatan.toml").write_text('business = "other"\nagent = "x"\n', encoding="utf-8")
    assert resolve_scope(inner, env={}).business == "other"


def test_default_token_budget(tmp_path: Path) -> None:
    (tmp_path / ".ingatan.toml").write_text('business = "b"\nagent = "a"\n', encoding="utf-8")
    assert resolve_scope(tmp_path, env={}).token_budget == 1000


def test_environment_overrides_the_config_file(tmp_path: Path) -> None:
    (tmp_path / ".ingatan.toml").write_text(TOML, encoding="utf-8")
    both = resolve_scope(tmp_path, env={"INGATAN_BUSINESS": "warung", "INGATAN_AGENT": "hermes"})
    assert (both.business, both.agent) == ("warung", "hermes")
    # Each variable overrides on its own.
    only_business = resolve_scope(tmp_path, env={"INGATAN_BUSINESS": "warung"})
    assert (only_business.business, only_business.agent) == ("warung", "claude-code")
    only_agent = resolve_scope(tmp_path, env={"INGATAN_AGENT": "hermes"})
    assert (only_agent.business, only_agent.agent) == ("tokoku", "hermes")


def test_environment_alone_is_enough(tmp_path: Path) -> None:
    scope = resolve_scope(tmp_path, env={"INGATAN_BUSINESS": "warung", "INGATAN_AGENT": "hermes"})
    assert (scope.business, scope.agent) == ("warung", "hermes")


def test_process_environment_is_used_when_env_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "warung")
    monkeypatch.setenv("INGATAN_AGENT", "hermes")
    assert resolve_scope(tmp_path).business == "warung"


@pytest.mark.parametrize(
    ("toml", "env", "missing"),
    [
        (None, {}, "business"),
        ('business = "tokoku"\n', {}, "agent"),
        ('agent = "claude-code"\n', {}, "business"),
        (None, {"INGATAN_BUSINESS": "tokoku"}, "agent"),
    ],
)
def test_missing_business_or_agent_names_both_ways_to_set_it(
    tmp_path: Path, toml: str | None, env: dict[str, str], missing: str
) -> None:
    if toml is not None:
        (tmp_path / ".ingatan.toml").write_text(toml, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        resolve_scope(tmp_path, env=env)
    message = str(caught.value)
    assert missing in message
    assert ".ingatan.toml" in message
    assert f"INGATAN_{missing.upper()}" in message


def test_bad_names_and_bad_files_raise_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_scope(tmp_path, env={"INGATAN_BUSINESS": "Bad Name", "INGATAN_AGENT": "a"})
    with pytest.raises(ConfigError, match="reserved"):
        resolve_scope(tmp_path, env={"INGATAN_BUSINESS": "global", "INGATAN_AGENT": "a"})
    (tmp_path / ".ingatan.toml").write_text("business = [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        resolve_scope(tmp_path, env={})
    (tmp_path / ".ingatan.toml").write_text(
        'business = "b"\nagent = "a"\n[brief]\ntoken_budget = "lots"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="token_budget"):
        resolve_scope(tmp_path, env={})


# ---- context and files ------------------------------------------------------------


def test_open_context_creates_both_files_at_latest_version(
    make_ctx: MakeContext, ingatan_home: Path
) -> None:
    ctx = make_ctx()
    assert (ingatan_home / "tokoku.db").exists()
    assert (ingatan_home / "global.db").exists()
    for conn in (ctx.business, ctx.glob):
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.LATEST_VERSION
    assert ctx.conn_for_tier("agent") is ctx.business
    assert ctx.conn_for_tier("business") is ctx.business
    assert ctx.conn_for_tier("global") is ctx.glob
    with pytest.raises(ValidationError):
        ctx.conn_for_tier("everyone")


def test_open_context_rejects_a_bad_scope(ingatan_home: Path) -> None:
    with pytest.raises(ConfigError):
        open_context(Scope("global", "a", ingatan_home))
    with pytest.raises(ConfigError):
        open_context(Scope("tokoku", "Bad Agent", ingatan_home))


def test_close_closes_both_connections(make_ctx: MakeContext) -> None:
    ctx = make_ctx()
    ctx.close()
    with pytest.raises(Exception, match="closed"):
        ctx.business.execute("SELECT 1")
    with pytest.raises(Exception, match="closed"):
        ctx.glob.execute("SELECT 1")
    ctx.close()  # closing twice is harmless


def test_two_businesses_use_different_files(
    make_ctx: MakeContext, ingatan_home: Path, insert_fact: InsertFact
) -> None:
    tokoku = make_ctx(business="tokoku")
    warung = make_ctx(business="warung")
    assert tokoku.business is not warung.business
    assert (ingatan_home / "tokoku.db").exists()
    assert (ingatan_home / "warung.db").exists()

    secret_id = insert_fact(tokoku.business, tier="business", text="harga rahasia tokoku")
    assert find_fact(tokoku, secret_id) is not None
    assert find_fact(warung, secret_id) is None
    assert warung.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    # The global file is shared by every business.
    global_id = insert_fact(tokoku.glob, tier="global", text="nama pemilik")
    assert find_fact(warung, global_id) is not None


# ---- visibility -------------------------------------------------------------------


def test_visible_sql_shapes() -> None:
    assert visible_sql(is_global=True) == "tier = 'global'"
    assert visible_sql(is_global=False, alias="f") == (
        "(f.tier = 'business' OR (f.tier = 'agent' AND f.agent = :agent))"
    )
    assert "owner = :agent" in visible_sql(is_global=False, owner_column="owner")


def test_agent_tier_rows_are_invisible_to_another_agent(
    make_ctx: MakeContext, insert_fact: InsertFact
) -> None:
    alice = make_ctx(agent="alice")
    bob = make_ctx(agent="bob")
    mine = insert_fact(alice.business, tier="agent", agent="alice", text="catatan alice")
    shared = insert_fact(alice.business, tier="business", agent="alice", text="info bersama")

    def visible_ids(ctx: Context) -> set[str]:
        where = ctx.visible_sql(ctx.business)
        rows = ctx.business.execute(f"SELECT id FROM facts WHERE {where}", ctx.params)
        return {r[0] for r in rows}

    assert visible_ids(alice) == {mine, shared}
    assert visible_ids(bob) == {shared}

    assert find_fact(alice, mine) is not None
    assert find_fact(bob, mine) is None  # behaves exactly like a missing id
    assert find_fact(bob, "f_000000000999") is None
    found = find_fact(bob, shared)
    assert found is not None and found[1]["text"] == "info bersama"


def test_global_file_only_shows_global_tier(make_ctx: MakeContext, insert_fact: InsertFact) -> None:
    ctx = make_ctx()
    visible = insert_fact(ctx.glob, tier="global", text="visible")
    stray = insert_fact(ctx.glob, tier="agent", text="stray row that must not show")
    assert find_fact(ctx, visible) is not None
    assert find_fact(ctx, stray) is None


def test_find_fact_returns_the_connection_the_row_lives_in(
    make_ctx: MakeContext, insert_fact: InsertFact
) -> None:
    ctx = make_ctx()
    local = insert_fact(ctx.business, tier="business")
    shared = insert_fact(ctx.glob, tier="global")
    found_local = find_fact(ctx, local)
    found_shared = find_fact(ctx, shared)
    assert found_local is not None and found_local[0] is ctx.business
    assert found_shared is not None and found_shared[0] is ctx.glob
