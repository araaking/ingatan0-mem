"""HTTP-mode tokens: storage (hash only, atomic), validation, revoke, reload."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ingatan import tokens
from ingatan.errors import ConfigError, NotFoundError

RAW_FORMAT = re.compile(r"^igt_[A-Za-z0-9_-]{43}$")  # 32 bytes, base64url, no padding


def test_add_returns_a_well_formed_token_and_a_record(ingatan_home: Path) -> None:
    raw, record = tokens.add(ingatan_home, "kopi", "claude-mac", "MacBook")
    assert RAW_FORMAT.match(raw)
    assert len(base64.urlsafe_b64decode(raw[4:] + "=")) == 32
    assert re.fullmatch(r"t_[0-9a-f]{8}", record.id)
    assert (record.business, record.agent, record.label) == ("kopi", "claude-mac", "MacBook")
    assert record.hash == hashlib.sha256(raw.encode()).hexdigest()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record.created)


def test_every_token_is_different(ingatan_home: Path) -> None:
    first, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    second, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    assert first != second
    assert len({r.id for r in tokens.list_tokens(ingatan_home)}) == 2


def test_only_the_hash_is_stored(ingatan_home: Path) -> None:
    raw, record = tokens.add(ingatan_home, "kopi", "claude-mac")
    data = tokens.tokens_path(ingatan_home).read_bytes()
    assert raw.encode() not in data
    assert raw[4:].encode() not in data
    assert record.hash.encode() in data
    stored = json.loads(data)["tokens"][0]
    assert set(stored) == {"id", "hash", "business", "agent", "label", "created"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_the_file_is_private(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "claude-mac")
    assert tokens.tokens_path(ingatan_home).stat().st_mode & 0o777 == 0o600


def test_list_is_empty_without_a_file(ingatan_home: Path) -> None:
    assert tokens.list_tokens(ingatan_home) == []


def test_list_keeps_the_order_of_creation(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "a1")
    tokens.add(ingatan_home, "teh", "a2", "  spaced  ")
    listed = tokens.list_tokens(ingatan_home)
    assert [(r.business, r.agent, r.label) for r in listed] == [
        ("kopi", "a1", ""),
        ("teh", "a2", "spaced"),
    ]


@pytest.mark.parametrize(
    ("business", "agent"),
    [
        ("Kopi", "claude"),  # upper case
        ("global", "claude"),  # reserved
        ("kopi shop", "claude"),
        ("", "claude"),
        ("kopi", "Claude Mac"),
        ("kopi", ""),
        ("../kopi", "claude"),
    ],
)
def test_invalid_names_are_rejected_and_nothing_is_written(
    ingatan_home: Path, business: str, agent: str
) -> None:
    with pytest.raises(ConfigError):
        tokens.add(ingatan_home, business, agent)
    assert not tokens.tokens_path(ingatan_home).exists()


def test_revoke_removes_only_that_record(ingatan_home: Path) -> None:
    _, keep = tokens.add(ingatan_home, "kopi", "a1")
    _, drop = tokens.add(ingatan_home, "kopi", "a2")
    removed = tokens.revoke(ingatan_home, drop.id)
    assert removed.id == drop.id
    assert [r.id for r in tokens.list_tokens(ingatan_home)] == [keep.id]


def test_revoking_an_unknown_id_fails_and_changes_nothing(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "a1")
    before = tokens.tokens_path(ingatan_home).read_bytes()
    with pytest.raises(NotFoundError):
        tokens.revoke(ingatan_home, "t_00000000")
    assert tokens.tokens_path(ingatan_home).read_bytes() == before


def test_a_failed_rewrite_leaves_the_old_file_and_no_temp_file(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens.add(ingatan_home, "kopi", "a1")
    before = tokens.tokens_path(ingatan_home).read_bytes()

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk says no")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk says no"):
        tokens.add(ingatan_home, "kopi", "a2")
    monkeypatch.undo()
    assert tokens.tokens_path(ingatan_home).read_bytes() == before
    assert sorted(p.name for p in ingatan_home.iterdir()) == ["tokens.json"]


def test_a_successful_rewrite_leaves_no_temp_file(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "a1")
    tokens.add(ingatan_home, "kopi", "a2")
    assert sorted(p.name for p in ingatan_home.iterdir()) == ["tokens.json"]


def test_a_corrupt_file_is_a_config_error_for_the_cli_functions(ingatan_home: Path) -> None:
    ingatan_home.mkdir(parents=True)
    tokens.tokens_path(ingatan_home).write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        tokens.list_tokens(ingatan_home)
    with pytest.raises(ConfigError):
        tokens.add(ingatan_home, "kopi", "a1")


# ---- TokenStore ------------------------------------------------------------------------


def test_lookup_resolves_a_token_to_its_scope(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    scope = tokens.TokenStore(ingatan_home).lookup(raw)
    assert scope is not None
    assert (scope.business, scope.agent, scope.home) == ("kopi", "claude-mac", ingatan_home)


def test_lookup_rejects_unknown_and_similar_tokens(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    store = tokens.TokenStore(ingatan_home)
    other = tokens.new_secret()
    for bad in ("", "igt_", other, raw + "x", raw[:-1], raw.upper(), raw[4:]):
        assert store.lookup(bad) is None


def test_a_store_without_a_file_has_no_tokens(ingatan_home: Path) -> None:
    store = tokens.TokenStore(ingatan_home)
    assert store.count() == 0
    assert store.lookup(tokens.new_secret()) is None


def test_the_store_sees_a_token_added_after_it_started(ingatan_home: Path) -> None:
    store = tokens.TokenStore(ingatan_home)
    assert store.count() == 0
    raw, _ = tokens.add(ingatan_home, "kopi", "a1")
    assert store.count() == 1
    assert store.lookup(raw) is not None


def test_the_store_stops_accepting_a_revoked_token(ingatan_home: Path) -> None:
    keep_raw, _ = tokens.add(ingatan_home, "kopi", "a1")
    drop_raw, drop = tokens.add(ingatan_home, "kopi", "a2")
    store = tokens.TokenStore(ingatan_home)
    assert store.lookup(drop_raw) is not None
    tokens.revoke(ingatan_home, drop.id)
    assert store.lookup(drop_raw) is None
    assert store.lookup(keep_raw) is not None


def test_an_unreadable_file_lets_nobody_in(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "a1")
    store = tokens.TokenStore(ingatan_home)
    assert store.lookup(raw) is not None
    tokens.tokens_path(ingatan_home).write_text("{not json", encoding="utf-8")
    assert store.lookup(raw) is None


def test_a_hand_edited_record_with_a_bad_name_routes_nowhere(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "a1")
    path = tokens.tokens_path(ingatan_home)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["tokens"][0]["business"] = "Not Valid"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert tokens.TokenStore(ingatan_home).lookup(raw) is None


def test_the_tokens_module_does_not_pull_in_the_http_stack() -> None:
    code = (
        "import sys, ingatan.tokens; "
        "print([m for m in ('mcp', 'starlette', 'pydantic') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout
