from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ingatan import episodes
from ingatan.errors import NotFoundError, ValidationError
from ingatan.redact import MARKER
from ingatan.scope import Context, Scope

MakeContext = Callable[..., Context]

KEY = "sk-Ab12Cd34Ef56Gh78Ij90Kl12Mn34"


def rows(ctx: Context) -> list[Any]:
    return ctx.business.execute("SELECT * FROM episodes ORDER BY id").fetchall()


def test_record_stores_one_pending_episode(ctx: Context, frozen_clock: object) -> None:
    episode_id = episodes.record(ctx, "Berapa harga kopi susu?", "Rp 18.000.", session="s1")
    [row] = rows(ctx)
    assert row["id"] == episode_id
    assert (row["agent"], row["session"], row["status"], row["offered"]) == (
        "claude-code",
        "s1",
        "pending",
        0,
    )
    assert row["user_text"] == "Berapa harga kopi susu?"
    assert row["created_at"] == "2026-11-11T09:00:00Z"
    assert row["processed_at"] is None


def test_unicode_survives(ctx: Context) -> None:
    text = "Harga kopi susu naik jadi Rp 18.000 — kata Bu Sári ‘mulai Senin’ ✓"
    episodes.record(ctx, text, text)
    assert rows(ctx)[0]["user_text"] == text
    assert rows(ctx)[0]["assistant_text"] == text


def test_overlong_messages_are_truncated_at_the_limit(ctx: Context) -> None:
    assert episodes.TRUNCATE_AT == 2000
    episodes.record(ctx, "a" * 5000, "ü" * 2001)
    [row] = rows(ctx)
    assert row["user_text"] == "a" * 2000
    assert row["assistant_text"] == "ü" * 2000


def test_a_message_of_exactly_the_limit_is_kept_whole(ctx: Context) -> None:
    episodes.record(ctx, "b" * 2000, "ok")
    assert rows(ctx)[0]["user_text"] == "b" * 2000


def test_secrets_are_redacted_in_both_messages_and_the_session(ctx: Context) -> None:
    episodes.record(
        ctx,
        f"kunci saya {KEY}, password=hunter2",
        "Baik. Authorization: Bearer abcDEF1234567890xyz",
        session=f"token={KEY}",
    )
    [row] = rows(ctx)
    blob = " ".join(str(row[k]) for k in ("user_text", "assistant_text", "session"))
    for secret in (KEY, "hunter2", "abcDEF1234567890xyz"):
        assert secret not in blob
    assert MARKER in row["user_text"]
    assert MARKER in row["assistant_text"]


def test_a_secret_cut_by_truncation_cannot_survive(ctx: Context) -> None:
    # Redaction runs before the cut, so a key straddling position 2000 is still removed.
    episodes.record(ctx, "x" * 1994 + " " + KEY, "ok")
    stored = rows(ctx)[0]["user_text"]
    assert "sk-" not in stored
    assert len(stored) <= 2000


def test_record_refuses_an_empty_turn(ctx: Context) -> None:
    with pytest.raises(ValidationError):
        episodes.record(ctx, "  ", "")
    assert rows(ctx) == []


# ---- capture ----------------------------------------------------------------------


def test_capture_returns_true_and_stores(ingatan_home: Path) -> None:
    scope = Scope("tokoku", "claude-code", ingatan_home)
    assert episodes.capture(scope, "halo", "hai") is True
    from ingatan.scope import open_context

    with open_context(scope) as ctx:
        assert [r["user_text"] for r in rows(ctx)] == ["halo"]
    assert not (ingatan_home / "capture-errors.log").exists()


def test_capture_resolves_the_scope_itself_when_given_none(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "warung")
    monkeypatch.setenv("INGATAN_AGENT", "hermes")
    assert episodes.capture(None, "halo", "hai", session="s") is True
    from ingatan.scope import open_context

    with open_context(Scope("warung", "hermes", ingatan_home)) as ctx:
        assert [(r["agent"], r["session"]) for r in rows(ctx)] == [("hermes", "s")]


def test_capture_truncates_and_redacts(ingatan_home: Path) -> None:
    scope = Scope("tokoku", "claude-code", ingatan_home)
    assert episodes.capture(scope, "y" * 2500 + " " + KEY, f"pakai {KEY} ya")
    from ingatan.scope import open_context

    with open_context(scope) as ctx:
        [row] = rows(ctx)
    assert len(row["user_text"]) == 2000
    assert KEY not in row["assistant_text"] and MARKER in row["assistant_text"]


def test_capture_with_an_unusable_database_returns_false_and_logs(ingatan_home: Path) -> None:
    ingatan_home.mkdir(parents=True)
    (ingatan_home / "tokoku.db").mkdir()  # a folder where the database file should be
    scope = Scope("tokoku", "claude-code", ingatan_home)
    assert episodes.capture(scope, "halo", f"rahasia {KEY}") is False
    log = (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")
    assert log.count("\n") == 1
    assert "Error" in log
    assert KEY not in log
    # A second failure appends another line.
    assert episodes.capture(scope, "halo", "lagi") is False
    assert (ingatan_home / "capture-errors.log").read_text(encoding="utf-8").count("\n") == 2


def test_capture_with_no_config_returns_false_and_logs_in_the_data_home(
    ingatan_home: Path,
) -> None:
    # No INGATAN_BUSINESS/INGATAN_AGENT and no .ingatan.toml: resolving the scope fails.
    assert episodes.capture(None, "halo", "hai") is False
    log = (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")
    assert "ConfigError" in log


def test_capture_never_raises_even_when_the_log_cannot_be_written(tmp_path: Path) -> None:
    broken_home = tmp_path / "a-file"
    broken_home.write_text("not a folder", encoding="utf-8")
    scope = Scope("tokoku", "claude-code", broken_home)
    assert episodes.capture(scope, "halo", "hai") is False


def test_capture_with_an_empty_turn_returns_false(ingatan_home: Path) -> None:
    scope = Scope("tokoku", "claude-code", ingatan_home)
    assert episodes.capture(scope, "", "") is False


# ---- offer / mark_processed / purge ------------------------------------------------


def test_offer_returns_oldest_pending_first_up_to_the_limit(ctx: Context) -> None:
    ids = [episodes.record(ctx, f"u{n}", f"a{n}") for n in range(5)]
    offered = episodes.offer(ctx)
    assert [e.id for e in offered] == ids[:3]
    assert [e.offered for e in offered] == [1, 1, 1]
    assert [e.id for e in episodes.offer(ctx, limit=1)] == [ids[0]]
    assert episodes.offer(ctx, limit=0) == []


def test_offer_with_ten_pending_episodes_offers_three(ctx: Context) -> None:
    for n in range(10):
        episodes.record(ctx, f"u{n}", f"a{n}")
    assert len(episodes.offer(ctx)) == 3 == episodes.MAX_OFFER


def test_an_episode_offered_three_times_is_processed(ctx: Context, frozen_clock: Any) -> None:
    first = episodes.record(ctx, "u0", "a0")
    second = episodes.record(ctx, "u1", "a1")
    for expected in (1, 2):
        assert [e.offered for e in episodes.offer(ctx, limit=1)] == [expected]
        assert rows(ctx)[0]["status"] == "pending"
    last = episodes.offer(ctx, limit=1)
    assert [(e.id, e.offered, e.status) for e in last] == [(first, 3, "processed")]
    assert last[0].processed_at == "2026-11-11T09:00:00Z"
    # The first episode is gone from the offers; the second is next.
    assert [e.id for e in episodes.offer(ctx, limit=1)] == [second]


def test_offer_skips_processed_episodes(ctx: Context) -> None:
    done = episodes.record(ctx, "u0", "a0")
    todo = episodes.record(ctx, "u1", "a1")
    episodes.mark_processed(ctx, done)
    assert [e.id for e in episodes.offer(ctx)] == [todo]


def test_mark_processed_sets_status_and_is_idempotent(ctx: Context, frozen_clock: Any) -> None:
    episode_id = episodes.record(ctx, "u", "a")
    episodes.mark_processed(ctx, episode_id)
    first = rows(ctx)[0]
    assert (first["status"], first["processed_at"]) == ("processed", "2026-11-11T09:00:00Z")
    frozen_clock.advance(days=1)
    episodes.mark_processed(ctx, episode_id)
    assert rows(ctx)[0]["processed_at"] == "2026-11-11T09:00:00Z"


def test_mark_processed_unknown_id_raises(ctx: Context) -> None:
    with pytest.raises(NotFoundError):
        episodes.mark_processed(ctx, 999)


def test_episodes_are_private_to_the_agent_that_captured_them(make_ctx: MakeContext) -> None:
    alice = make_ctx(agent="alice")
    bob = make_ctx(agent="bob")
    mine = episodes.record(alice, "rahasia alice", "ok")
    assert episodes.offer(bob) == []
    with pytest.raises(NotFoundError):
        episodes.mark_processed(bob, mine)
    assert rows(alice)[0]["status"] == "pending"  # bob's attempt changed nothing
    assert [e.id for e in episodes.offer(alice)] == [mine]


def test_purge_deletes_only_old_processed_episodes(ctx: Context, frozen_clock: Any) -> None:
    old_done = episodes.record(ctx, "old", "x")
    old_pending = episodes.record(ctx, "pending", "x")
    episodes.mark_processed(ctx, old_done)
    frozen_clock.advance(days=episodes.RETENTION_DAYS - 1)
    recent_done = episodes.record(ctx, "recent", "x")
    episodes.mark_processed(ctx, recent_done)
    assert episodes.purge(ctx) == 0  # nothing is older than the retention yet
    frozen_clock.advance(days=2)
    assert episodes.purge(ctx) == 1
    assert [r["id"] for r in rows(ctx)] == [old_pending, recent_done]
    assert episodes.purge(ctx) == 0


def test_purge_leaves_other_agents_episodes_alone(make_ctx: MakeContext, frozen_clock: Any) -> None:
    alice = make_ctx(agent="alice")
    bob = make_ctx(agent="bob")
    bob_id = episodes.record(bob, "u", "a")
    episodes.mark_processed(bob, bob_id)
    frozen_clock.advance(days=episodes.RETENTION_DAYS + 5)
    assert episodes.purge(alice) == 0
    assert len(rows(bob)) == 1
    assert episodes.purge(bob) == 1
