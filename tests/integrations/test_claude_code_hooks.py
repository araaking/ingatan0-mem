from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from ingatan import episodes, facts
from ingatan.integrations.claude_code import hooks
from ingatan.scope import Context

KEY = "sk-Ab12Cd34Ef56Gh78Ij90Kl12Mn34"
SESSION = "sess-1234"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "proyek" / "sub"
    folder.mkdir(parents=True)
    (tmp_path / "proyek" / ".ingatan.toml").write_text(
        'business = "tokoku"\nagent = "claude-code"\n', encoding="utf-8"
    )
    return folder


def feed(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data), encoding="cp1252"))


def call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    event: str,
    payload: dict[str, Any] | bytes,
) -> tuple[int, str, str]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    feed(monkeypatch, data)
    code = hooks.run(event)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def episodes_in(ctx: Context) -> list[sqlite3.Row]:
    return ctx.business.execute("SELECT * FROM episodes ORDER BY id").fetchall()


def prompt_and_stop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    prompt: str,
    stop: dict[str, Any],
) -> None:
    base = {"session_id": SESSION, "cwd": str(project)}
    assert call(monkeypatch, capsys, "user-prompt", {**base, "prompt": prompt})[0] == 0
    assert call(monkeypatch, capsys, "stop", {**base, **stop})[0] == 0


# ------------------------------------------------------------------ capture


def test_prompt_and_reply_become_one_episode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
    ingatan_home: Path,
) -> None:
    prompt = "Berapa harga kopi susu? Bu Sári bilang Rp 18.000 — benar?"
    reply = "Harga kopi susu Rp 18.000 per gelas ✓"
    base = {"session_id": SESSION, "cwd": str(project)}
    code, out, err = call(monkeypatch, capsys, "user-prompt", {**base, "prompt": prompt})
    assert (code, out, err) == (0, "", "")
    pending = ingatan_home / "pending" / f"{SESSION}.json"
    assert json.loads(pending.read_text(encoding="utf-8"))["prompt"] == prompt

    code, out, err = call(
        monkeypatch,
        capsys,
        "stop",
        {**base, "last_assistant_message": reply, "stop_hook_active": False},
    )
    assert (code, out, err) == (0, "", "")
    [row] = episodes_in(ctx)
    assert (row["user_text"], row["assistant_text"]) == (prompt, reply)
    assert (row["agent"], row["session"], row["status"]) == ("claude-code", SESSION, "pending")
    assert not pending.exists()


def test_bom_and_invalid_bytes_in_stdin_are_tolerated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
) -> None:
    body = json.dumps({"session_id": SESSION, "cwd": str(project), "prompt": "Halo Sári"})
    assert call(monkeypatch, capsys, "user-prompt", b"\xef\xbb\xbf" + body.encode("utf-8"))[0] == 0
    stop = json.dumps(
        {"session_id": SESSION, "cwd": str(project), "last_assistant_message": "ok \xe9"}
    )
    # A stray invalid byte inside a string must not lose the whole turn.
    raw = stop.encode("utf-8").replace(b"ok ", b"ok \xff ")
    assert call(monkeypatch, capsys, "stop", raw)[0] == 0
    [row] = episodes_in(ctx)
    assert row["user_text"] == "Halo Sári"
    assert row["assistant_text"].startswith("ok ")


TRANSCRIPT_LINES: list[dict[str, Any]] = [
    {"type": "queue-operation", "operation": "enqueue", "content": "zzqueuezz"},
    {"type": "user", "message": {"role": "user", "content": "Cek stok kopi"}},
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "zzthinkingzz rahasia"}],
        },
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Saya cek dulu."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "zzfilepathzz.txt"},
                },
            ],
        },
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "zztooloutputzz isi file"}
            ],
        },
    },
    {
        "type": "assistant",
        "isSidechain": True,
        "message": {"role": "assistant", "content": [{"type": "text", "text": "zzsidechainzz"}]},
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "zzthinking2zz"},
                {"type": "text", "text": "Stok kopi tinggal 12 bungkus."},
                {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "zzcmdzz"}},
            ],
        },
    },
    {"type": "last-prompt", "lastPrompt": "zzlastpromptzz"},
]


def test_transcript_fallback_stores_only_prompt_and_final_reply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in TRANSCRIPT_LINES) + "\n",
        encoding="utf-8",
    )
    prompt_and_stop(
        monkeypatch,
        capsys,
        project,
        "Cek stok kopi",
        {"transcript_path": str(transcript)},
    )
    [row] = episodes_in(ctx)
    assert row["user_text"] == "Cek stok kopi"
    assert row["assistant_text"] == "Stok kopi tinggal 12 bungkus."
    everything = json.dumps([dict(r) for r in episodes_in(ctx)])
    for leaked in (
        "zzthinkingzz",
        "zzthinking2zz",
        "zzfilepathzz",
        "zztooloutputzz",
        "zzcmdzz",
        "zzsidechainzz",
        "zzqueuezz",
        "zzlastpromptzz",
        "Saya cek dulu",
    ):
        assert leaked not in everything


def test_transcript_fallback_reads_a_large_file_from_the_end(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
    tmp_path: Path,
) -> None:
    filler = {"type": "user", "message": {"role": "user", "content": "x" * 5000}}
    lines = [json.dumps(filler)] * 100  # about 500 KB, more than the first tail chunk
    # The only assistant text sits before the filler, beyond the first chunk.
    early = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "jawaban awal"}]},
    }
    transcript = tmp_path / "big.jsonl"
    transcript.write_text("\n".join([json.dumps(early), *lines]) + "\n", encoding="utf-8")
    prompt_and_stop(monkeypatch, capsys, project, "tanya", {"transcript_path": str(transcript)})
    [row] = episodes_in(ctx)
    assert row["assistant_text"] == "jawaban awal"


def test_secret_in_prompt_is_redacted_in_pending_file_and_episode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
    ingatan_home: Path,
) -> None:
    base = {"session_id": SESSION, "cwd": str(project)}
    call(monkeypatch, capsys, "user-prompt", {**base, "prompt": f"kunci saya {KEY} tolong simpan"})
    pending = (ingatan_home / "pending" / f"{SESSION}.json").read_text(encoding="utf-8")
    assert KEY not in pending
    call(
        monkeypatch,
        capsys,
        "stop",
        {**base, "last_assistant_message": f"Kuncinya {KEY} sudah dicatat"},
    )
    [row] = episodes_in(ctx)
    assert KEY not in row["user_text"] + row["assistant_text"]
    assert "[redacted]" in row["user_text"]


def test_overlong_text_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
) -> None:
    prompt_and_stop(
        monkeypatch,
        capsys,
        project,
        "p" * 5000,
        {"last_assistant_message": "a" * 5000},
    )
    [row] = episodes_in(ctx)
    assert len(row["user_text"]) == episodes.TRUNCATE_AT == 2000
    assert len(row["assistant_text"]) == 2000


def test_stop_hook_active_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
) -> None:
    prompt_and_stop(
        monkeypatch,
        capsys,
        project,
        "halo",
        {"last_assistant_message": "hai", "stop_hook_active": True},
    )
    assert episodes_in(ctx) == []


def test_stop_without_stored_prompt_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
) -> None:
    payload = {"session_id": SESSION, "cwd": str(project), "last_assistant_message": "hai"}
    assert call(monkeypatch, capsys, "stop", payload) == (0, "", "")
    assert episodes_in(ctx) == []


def test_pending_prompts_are_kept_per_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
) -> None:
    for sid, text in (("a", "dari a"), ("b", "dari b")):
        call(
            monkeypatch,
            capsys,
            "user-prompt",
            {"session_id": sid, "cwd": str(project), "prompt": text},
        )
    call(
        monkeypatch,
        capsys,
        "stop",
        {"session_id": "a", "cwd": str(project), "last_assistant_message": "balas a"},
    )
    [row] = episodes_in(ctx)
    assert (row["user_text"], row["assistant_text"], row["session"]) == ("dari a", "balas a", "a")


def test_session_id_cannot_escape_the_pending_folder(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ingatan_home: Path,
) -> None:
    payload = {"session_id": "../../evil", "cwd": str(project), "prompt": "x"}
    assert call(monkeypatch, capsys, "user-prompt", payload)[0] == 0
    assert not (ingatan_home.parent / "evil.json").exists()
    assert [p.parent.name for p in ingatan_home.rglob("*.json")] == ["pending"]


def test_scope_comes_from_cwd_and_env_overrides_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    make_ctx: Any,
) -> None:
    monkeypatch.setenv("INGATAN_AGENT", "hermes")
    prompt_and_stop(monkeypatch, capsys, project, "halo", {"last_assistant_message": "hai"})
    ctx = make_ctx(agent="hermes")
    [row] = episodes_in(ctx)
    assert row["agent"] == "hermes"


# ------------------------------------------------------------------ never noisy, never failing

EVENT_PAYLOADS: dict[str, dict[str, Any]] = {
    "session-start": {"session_id": SESSION, "source": "startup"},
    "user-prompt": {"session_id": SESSION, "prompt": "halo"},
    "stop": {"session_id": SESSION, "last_assistant_message": "hai"},
}


@pytest.mark.parametrize("event", ["session-start", "user-prompt", "stop"])
@pytest.mark.parametrize(
    "raw",
    [b"", b"not json {", b"[1, 2]", b"null", b"\xff\xfe\x00garbage", b"{}", b'{"cwd": 5}'],
)
def test_malformed_or_empty_stdin_returns_zero_quietly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
    event: str,
    raw: bytes,
) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")
    monkeypatch.setenv("INGATAN_AGENT", "claude-code")
    # A valid-but-empty payload still resolves a scope from the env, so a real brief would
    # legitimately print; the point here is that nothing else does.
    monkeypatch.setattr("ingatan.brief.build", lambda c, **k: "", raising=False)
    assert call(monkeypatch, capsys, event, raw) == (0, "", "")
    assert episodes_in(ctx) == []


@pytest.mark.parametrize("event", ["session-start", "user-prompt", "stop"])
def test_missing_fields_return_zero_quietly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
    event: str,
) -> None:
    monkeypatch.setattr("ingatan.brief.build", lambda c, **k: "", raising=False)
    assert call(monkeypatch, capsys, event, {"cwd": str(project)}) == (0, "", "")
    assert call(monkeypatch, capsys, event, {"cwd": str(project), "session_id": ""}) == (0, "", "")
    assert episodes_in(ctx) == []


def test_unknown_event_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert call(monkeypatch, capsys, "nonsense", {}) == (0, "", "")


@pytest.mark.parametrize("event", ["session-start", "user-prompt", "stop"])
def test_unresolvable_scope_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    ingatan_home: Path,
    event: str,
) -> None:
    bare = tmp_path / "tanpa-konfigurasi"
    bare.mkdir()
    payload = {**EVENT_PAYLOADS[event], "cwd": str(bare)}
    assert call(monkeypatch, capsys, event, payload) == (0, "", "")
    assert not ingatan_home.exists()  # not even the data folder was created


@pytest.mark.parametrize("event", ["session-start", "user-prompt", "stop"])
def test_database_failures_return_zero_quietly_and_are_logged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ingatan_home: Path,
    event: str,
) -> None:
    ingatan_home.mkdir(parents=True)
    (ingatan_home / "tokoku.db").write_bytes(b"this is not a sqlite database" * 100)
    monkeypatch.setattr("ingatan.brief.build", lambda c, **k: "brief", raising=False)
    # user-prompt needs no database, so give stop a stored prompt to reach capture.
    base = {"session_id": SESSION, "cwd": str(project)}
    call(monkeypatch, capsys, "user-prompt", {**base, "prompt": "halo"})
    payload = {**EVENT_PAYLOADS[event], "cwd": str(project)}
    assert call(monkeypatch, capsys, event, payload) == (0, "", "")
    if event != "user-prompt":
        log = (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")
        assert log.strip()


def test_locked_database_returns_zero_quietly_and_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ingatan_home: Path,
) -> None:
    base = {"session_id": SESSION, "cwd": str(project)}
    call(monkeypatch, capsys, "user-prompt", {**base, "prompt": "halo"})

    def locked(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("ingatan.db.connect", locked)
    assert call(monkeypatch, capsys, "stop", {**base, "last_assistant_message": "hai"}) == (
        0,
        "",
        "",
    )
    assert "database is locked" in (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------ session-start


def test_session_start_prints_valid_hook_json_with_the_brief(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], project: Path
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_build(ctx: Context, **kwargs: Any) -> str:
        seen.append({"agent": ctx.scope.agent, **kwargs})
        return "# Ingatan brief: tokoku / claude-code\n- [f_1] Kopi susu Rp 18.000 — ✓"

    monkeypatch.setattr("ingatan.brief.build", fake_build, raising=False)
    code, out, err = call(
        monkeypatch,
        capsys,
        "session-start",
        {"session_id": SESSION, "cwd": str(project), "source": "startup"},
    )
    assert (code, err) == (0, "")
    data = json.loads(out)  # exactly one JSON object and nothing else
    assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert data["hookSpecificOutput"]["additionalContext"].startswith("# Ingatan brief")
    assert "Rp 18.000 — ✓" in data["hookSpecificOutput"]["additionalContext"]
    assert seen == [{"agent": "claude-code", "offer_episodes": True}]


@pytest.mark.parametrize(
    ("source", "offers"),
    [
        ("startup", True),
        ("clear", True),
        ("fork", True),
        (None, True),
        ("compact", False),
        ("resume", False),
    ],
)
def test_session_start_offers_episodes_only_for_new_sessions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    source: str | None,
    offers: bool,
) -> None:
    flags: list[bool] = []

    def fake_build(ctx: Context, *, offer_episodes: bool = True) -> str:
        flags.append(offer_episodes)
        return "brief"

    monkeypatch.setattr("ingatan.brief.build", fake_build, raising=False)
    payload: dict[str, Any] = {"session_id": SESSION, "cwd": str(project)}
    if source is not None:
        payload["source"] = source
    code, out, _ = call(monkeypatch, capsys, "session-start", payload)
    assert code == 0
    assert flags == [offers]
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"] == "brief"


def test_session_start_with_a_failing_brief_prints_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ingatan_home: Path,
) -> None:
    def boom(ctx: Context, **kwargs: Any) -> str:
        raise RuntimeError("brief exploded")

    monkeypatch.setattr("ingatan.brief.build", boom, raising=False)
    assert call(monkeypatch, capsys, "session-start", {"cwd": str(project)}) == (0, "", "")
    assert "brief exploded" in (ingatan_home / "capture-errors.log").read_text(encoding="utf-8")


def test_session_start_ignores_stray_prints(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], project: Path
) -> None:
    def noisy(ctx: Context, **kwargs: Any) -> str:
        print("stray debug output")
        return "brief"

    monkeypatch.setattr("ingatan.brief.build", noisy, raising=False)
    _, out, _ = call(monkeypatch, capsys, "session-start", {"cwd": str(project)})
    assert "stray" not in out
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"] == "brief"


def test_session_start_with_the_real_brief_when_it_exists(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    project: Path,
    ctx: Context,
) -> None:
    brief = pytest.importorskip("ingatan.brief")
    if not hasattr(brief, "build"):
        pytest.skip("brief.build is not finished yet")
    facts.remember(ctx, "Harga kopi susu 18000 rupiah", keywords=["kopi"])
    episodes.record(ctx, "tolong catat zzepisodeunikzz", "siap")
    code, out, _ = call(
        monkeypatch, capsys, "session-start", {"cwd": str(project), "source": "startup"}
    )
    assert code == 0
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "Harga kopi susu 18000 rupiah" in context
    assert "zzepisodeunikzz" in context  # the digest offer
    code, out, _ = call(
        monkeypatch, capsys, "session-start", {"cwd": str(project), "source": "compact"}
    )
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "Harga kopi susu 18000 rupiah" in context
    assert "zzepisodeunikzz" not in context
