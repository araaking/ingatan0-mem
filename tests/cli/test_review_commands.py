"""T-23, T-24, T-25: log, diff, undo, proposals, forget, ingest and hook through the CLI."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

from ingatan import audit, facts, proposals, search
from ingatan.scope import Context

Invoke = Callable[..., Result]
MakeContext = Callable[..., Context]

KOPI = "Harga kopi Gayo “spesial” Rp 50.000 – berlaku sampai akhir bulan…"


def rem(ctx: Context, text: str, **kw: Any) -> str:
    kw.setdefault("keywords", ["harga"])
    result = facts.remember(ctx, text, **kw)
    assert result.fact_id is not None
    return result.fact_id


def propose(ctx: Context, text: str, tier: str = "business", **kw: Any) -> str:
    kw.setdefault("keywords", ["harga"])
    result = facts.remember(ctx, text, tier=tier, **kw)
    assert result.proposal_id is not None
    return result.proposal_id


def recalled(ctx: Context, query: str) -> set[str]:
    return {h.id for h in search.recall(ctx, query, touch=False).facts}


def log_ids(output: str) -> list[int]:
    return [
        int(line.split()[0].lstrip("#")) for line in output.splitlines() if line.startswith("#")
    ]


def snapshot(ctx: Context) -> list[list[tuple[Any, ...]]]:
    return [
        [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1")]
        for conn in ctx.connections()
        for t in ("facts", "log", "conflicts", "proposals")
    ]


# ---- log and diff (T-23) -----------------------------------------------------------------


def test_log_lists_entries_newest_first(
    cli_ctx: Context, invoke: Invoke, frozen_clock: Any
) -> None:
    ids = []
    for text in ("pertama", "kedua", "ketiga"):
        ids.append(rem(cli_ctx, text))
        frozen_clock.advance(minutes=1)
    result = invoke("log")
    assert result.exit_code == 0, result.output
    shown = log_ids(result.stdout)
    assert shown == sorted(shown, reverse=True) and len(shown) == 3
    order = [result.stdout.index(t) for t in ("ketiga", "kedua", "pertama")]
    assert order == sorted(order)
    assert all(fact_id in result.stdout for fact_id in ids)


def test_log_limit_and_fact_filter(cli_ctx: Context, invoke: Invoke) -> None:
    a = rem(cli_ctx, "fakta a")
    rem(cli_ctx, "fakta b")
    rem(cli_ctx, "fakta c")
    facts.confirm(cli_ctx, a, "masih benar")
    assert len(log_ids(invoke("log", "--limit", "2").stdout)) == 2
    only_a = invoke("log", "--fact", a)
    assert only_a.exit_code == 0
    lines = [ln for ln in only_a.stdout.splitlines() if ln.startswith("#")]
    assert len(lines) == 2 and "confirm" in lines[0] and "create" in lines[1]
    assert invoke("log", "--fact", "f_000000000000").exit_code == 1


def test_log_of_an_empty_database_says_so(configured: None, invoke: Invoke) -> None:
    result = invoke("log")
    assert result.exit_code == 0 and "empty" in result.stdout


def test_diff_of_a_create_shows_the_text_it_added(cli_ctx: Context, invoke: Invoke) -> None:
    rem(cli_ctx, KOPI)
    result = invoke("diff", "1")
    assert result.exit_code == 0, result.output
    assert "create" in result.stdout and "+ " + KOPI in result.stdout
    assert "Before: nothing" in result.stdout


def test_diff_of_a_supersede_shows_each_field_before_and_after(
    cli_ctx: Context, invoke: Invoke, frozen_clock: Any
) -> None:
    old = rem(cli_ctx, "Harga lama", entities=["kopi"])
    new = rem(cli_ctx, "Harga baru", entities=["kopi"])
    frozen_clock.advance(hours=2)
    facts.supersede(cli_ctx, old, new)
    entry = next(e for e in audit.entries(cli_ctx, fact_id=old) if e.op == "supersede")
    out = invoke("diff", str(entry.id)).stdout
    assert "valid_to:\n  - null\n  + 2026-11-11T11:00:00Z" in out
    assert f"superseded_by:\n  - null\n  + {new}" in out


def test_diff_of_a_forgotten_fact_contains_no_content(cli_ctx: Context, invoke: Invoke) -> None:
    fact = rem(cli_ctx, KOPI)
    facts.forget(cli_ctx, fact)
    created = invoke("diff", "1")
    assert created.exit_code == 0
    assert "forgotten" in created.stdout and "Harga kopi" not in created.stdout
    listing = invoke("log")
    assert "Harga kopi" not in listing.stdout and "forgotten" in listing.stdout
    assert "Harga kopi" not in invoke("log", "--fact", fact).stdout


def test_diff_of_an_unknown_entry_fails_with_one_line(cli_ctx: Context, invoke: Invoke) -> None:
    result = invoke("diff", "42")
    assert result.exit_code == 1 and result.stderr.strip() == "ingatan: No log entry 42."


def test_another_agents_private_entries_need_the_agent_option(
    cli_ctx: Context, make_ctx: MakeContext, invoke: Invoke
) -> None:
    other = make_ctx(agent="hermes")
    private = rem(other, "Catatan pribadi hermes", keywords=["catatan"])
    shared = rem(cli_ctx, "Catatan claude")
    entry = audit.entries(other, fact_id=private)[0]

    default = invoke("log")
    assert private not in default.stdout and shared in default.stdout
    assert invoke("diff", str(entry.id)).exit_code == 1

    as_other = invoke("log", "--agent", "hermes")
    assert private in as_other.stdout and shared not in as_other.stdout
    assert "Catatan pribadi hermes" in invoke("diff", str(entry.id), "--agent", "hermes").stdout
    assert invoke("log", "--agent", "Not Valid").exit_code == 1


# ---- undo (T-24) -------------------------------------------------------------------------


def test_undo_of_a_create_through_the_cli(cli_ctx: Context, invoke: Invoke) -> None:
    fact = rem(cli_ctx, KOPI, entities=["kopi"])
    assert fact in recalled(cli_ctx, "kopi")
    result = invoke("undo", "1")
    assert result.exit_code == 0, result.output
    assert "New log entry: 2" in result.stdout
    assert recalled(cli_ctx, "kopi") == set()
    assert [e.op for e in audit.entries(cli_ctx, fact_id=fact)] == ["create", "undo"]
    assert "undo" in invoke("log").stdout.splitlines()[0]


def test_undo_of_a_supersede_and_a_confirm_through_the_cli(
    cli_ctx: Context, invoke: Invoke
) -> None:
    old = rem(cli_ctx, "Harga lama", entities=["kopi"])
    new = rem(cli_ctx, "Harga baru", entities=["kopi"])
    facts.supersede(cli_ctx, old, new)
    facts.confirm(cli_ctx, new, "benar")
    by_op = {e.op: e.id for e in audit.entries(cli_ctx) if e.op in ("supersede", "confirm")}
    assert invoke("undo", str(by_op["confirm"])).exit_code == 0
    assert invoke("undo", str(by_op["supersede"])).exit_code == 0
    assert recalled(cli_ctx, "kopi") == {old, new}


def test_undo_of_a_forget_fails_with_the_documented_message_and_changes_nothing(
    cli_ctx: Context, invoke: Invoke
) -> None:
    fact = rem(cli_ctx, KOPI)
    facts.forget(cli_ctx, fact)
    forget_id = audit.entries(cli_ctx, fact_id=fact)[-1].id
    before = snapshot(cli_ctx)
    result = invoke("undo", str(forget_id))
    assert result.exit_code == 1
    assert result.stderr.strip() == "ingatan: a forget cannot be undone"
    assert snapshot(cli_ctx) == before


def test_undo_with_a_later_change_is_refused_and_changes_nothing(
    cli_ctx: Context, invoke: Invoke
) -> None:
    fact = rem(cli_ctx, KOPI)
    facts.confirm(cli_ctx, fact, "masih benar")
    before = snapshot(cli_ctx)
    result = invoke("undo", "1")
    assert result.exit_code == 1 and "later entries" in result.stderr
    assert snapshot(cli_ctx) == before


def test_undo_writes_a_new_entry_and_leaves_the_old_one(cli_ctx: Context, invoke: Invoke) -> None:
    rem(cli_ctx, KOPI)
    old = audit.get(cli_ctx, 1)
    assert invoke("undo", "1").exit_code == 0
    assert audit.get(cli_ctx, 1) == old and audit.get(cli_ctx, 2).op == "undo"


# ---- proposals (T-25) --------------------------------------------------------------------


def test_proposals_list_shows_id_tier_text_agent_date_and_what_it_replaces(
    cli_ctx: Context, make_ctx: MakeContext, invoke: Invoke, frozen_clock: Any
) -> None:
    shared = proposals.approve(cli_ctx, propose(cli_ctx, "Retur maksimal 7 hari"))
    other = make_ctx(agent="hermes")
    first = propose(other, "Jam buka toko “09.00” – 21.00", tier="global")
    second = propose(cli_ctx, "Retur maksimal 14 hari")
    facts.supersede(cli_ctx, shared, second)

    result = invoke("proposals", "list")
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert f"{first}  [global]  hermes  2026-11-11" in out
    assert "Jam buka toko “09.00” – 21.00" in out
    assert f"{second}  [business]  claude-code  2026-11-11" in out
    assert f'would supersede {shared}: "Retur maksimal 7 hari"' in out


def test_proposals_list_when_empty(cli_ctx: Context, invoke: Invoke) -> None:
    assert "No pending proposals." in invoke("proposals", "list").stdout


def test_approve_makes_the_fact_valid_recallable_and_keeps_the_writing_agent(
    cli_ctx: Context, make_ctx: MakeContext, invoke: Invoke
) -> None:
    other = make_ctx(agent="hermes")
    pid = propose(other, "Retur maksimal 7 hari", keywords=["retur"])
    assert recalled(cli_ctx, "retur") == set()
    result = invoke("proposals", "approve", pid)
    assert result.exit_code == 0, result.output
    (hit,) = search.recall(cli_ctx, "retur", touch=False).facts
    assert hit.id in result.stdout and hit.tier == "business" and hit.status == "valid"
    assert hit.source["agent"] == "hermes"
    assert "approve" in [e.op for e in audit.entries(cli_ctx, fact_id=hit.id)]


def test_reject_creates_no_fact_and_both_actions_are_logged(
    cli_ctx: Context, invoke: Invoke
) -> None:
    keep = propose(cli_ctx, "Retur 7 hari", keywords=["retur"])
    drop = propose(cli_ctx, "Retur 30 hari", keywords=["retur"])
    assert invoke("proposals", "reject", drop).exit_code == 0
    assert invoke("proposals", "approve", keep).exit_code == 0
    assert len(recalled(cli_ctx, "retur")) == 1
    assert [e.op for e in audit.entries(cli_ctx, fact_id=drop)] == ["propose", "reject"]
    assert "approve" in invoke("log").stdout and "reject" in invoke("log").stdout


def test_proposal_commands_fail_cleanly_for_unknown_or_decided_ids(
    cli_ctx: Context, invoke: Invoke
) -> None:
    pid = propose(cli_ctx, "Retur 7 hari")
    assert invoke("proposals", "approve", "p_000000000000").exit_code == 1
    assert invoke("proposals", "approve", pid).exit_code == 0
    again = invoke("proposals", "approve", pid)
    assert again.exit_code == 1 and "already approved" in again.stderr


# ---- forget (T-25) -----------------------------------------------------------------------


def approved(ctx: Context, text: str, tier: str) -> str:
    return proposals.approve(ctx, propose(ctx, text, tier=tier, keywords=["rahasia"]))


@pytest.mark.parametrize("tier", ["business", "global"])
def test_the_cli_forgets_a_shared_fact_with_the_same_guarantees(
    cli_ctx: Context, invoke: Invoke, tier: str
) -> None:
    fact = approved(cli_ctx, "Kode gudang zzrahasiazz 77", tier)
    result = invoke("forget", fact, "--yes")
    assert result.exit_code == 0, result.output
    for conn in cli_ctx.connections():
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
        blob = " ".join(str(v) for r in conn.execute("SELECT * FROM log") for v in tuple(r))
        assert "zzrahasiazz" not in blob
    assert recalled(cli_ctx, "zzrahasiazz") == set()
    assert audit.entries(cli_ctx, fact_id=fact)[-1].op == "forget"


def test_forget_asks_first_and_shows_the_fact_text(cli_ctx: Context, invoke: Invoke) -> None:
    fact = approved(cli_ctx, "Kode gudang 77", "business")
    declined = invoke("forget", fact, input="n\n")
    assert declined.exit_code == 1
    assert "Kode gudang 77" in declined.stdout and "Nothing was forgotten" in declined.stdout
    assert recalled(cli_ctx, "kode gudang") == {fact}
    accepted = invoke("forget", fact, input="y\n")
    assert accepted.exit_code == 0 and recalled(cli_ctx, "kode gudang") == set()


def test_forget_without_a_terminal_and_without_yes_does_not_forget(
    cli_ctx: Context, invoke: Invoke
) -> None:
    fact = approved(cli_ctx, "Kode gudang 77", "business")
    assert invoke("forget", fact).exit_code != 0  # no input: the prompt is aborted
    assert recalled(cli_ctx, "kode gudang") == {fact}


def test_forget_unknown_and_other_agents_private_fact(
    cli_ctx: Context, make_ctx: MakeContext, invoke: Invoke
) -> None:
    assert invoke("forget", "f_000000000000", "--yes").exit_code == 1
    private = rem(make_ctx(agent="hermes"), "Catatan pribadi hermes", keywords=["catatan"])
    assert invoke("forget", private, "--yes").exit_code == 1  # invisible, like a missing fact
    assert invoke("forget", private, "--yes", "--agent", "hermes").exit_code == 0


# ---- ingest and hook (collaborators are monkeypatched) -----------------------------------


@dataclass
class FakeReport:
    files: int
    created: int
    replaced: int
    skipped: list[str]


def test_ingest_calls_ingest_path_and_prints_the_report(
    cli_ctx: Context, invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[Context, Path]] = []

    def fake(ctx: Context, path: Path) -> FakeReport:
        seen.append((ctx, path))
        return FakeReport(2, 5, 1, ["x.md (empty)"])

    monkeypatch.setattr("ingatan.ingest.ingest_path", fake)
    folder = tmp_path / "docs"
    folder.mkdir()
    result = invoke("ingest", str(folder))
    assert result.exit_code == 0, result.output
    assert seen[0][1] == folder and seen[0][0].scope.business == "tokoku"
    assert "Files: 2  created: 5  replaced: 1  skipped: 1" in result.stdout
    assert "x.md (empty)" in result.stdout


def test_ingest_of_a_missing_path_is_a_clean_error(cli_ctx: Context, invoke: Invoke) -> None:
    result = invoke("ingest", "tidak-ada.md")
    assert result.exit_code == 1 and "No such file" in result.stderr


def fake_hooks(monkeypatch: pytest.MonkeyPatch, run: Callable[[str], int]) -> None:
    module = types.ModuleType("ingatan.integrations.claude_code.hooks")
    module.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ingatan.integrations.claude_code.hooks", module)


def test_hook_passes_the_event_and_stdin_and_exits_with_the_returned_code(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []

    def run(event: str) -> int:
        seen.append((event, sys.stdin.read()))
        return 0

    fake_hooks(monkeypatch, run)
    result = invoke("hook", "SessionStart", input='{"prompt": "Selamat pagi"}')
    assert result.exit_code == 0
    assert seen == [("SessionStart", '{"prompt": "Selamat pagi"}')]


def test_hook_exits_zero_whatever_code_run_returns(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_hooks(monkeypatch, lambda event: 3)
    assert invoke("hook", "Stop").exit_code == 0


def error_log(tmp_path: Path) -> Path:
    return tmp_path / "capture-errors.log"


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyError("x"), ValueError("bad json")])
def test_hook_that_raises_exits_zero_prints_nothing_and_logs_one_line(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, exc: Exception, tmp_path: Path
) -> None:
    def run(event: str) -> int:
        raise exc

    monkeypatch.setenv("INGATAN_HOME", str(tmp_path))
    fake_hooks(monkeypatch, run)
    result = invoke("hook", "Stop")
    assert result.exit_code == 0
    assert result.stdout == "" and result.stderr == ""
    lines = error_log(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "hook Stop" in lines[0] and type(exc).__name__ in lines[0]


def test_hook_with_a_missing_module_exits_zero_and_logs_it(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INGATAN_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "ingatan.integrations.claude_code.hooks", None)
    result = invoke("hook", "Stop")
    assert result.exit_code == 0 and result.stdout == "" and result.stderr == ""
    assert "is not available" in error_log(tmp_path).read_text(encoding="utf-8")


def test_hook_survives_a_data_home_it_cannot_write_to(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("INGATAN_HOME", str(blocker / "home"))

    def run(event: str) -> int:
        raise ZeroDivisionError

    fake_hooks(monkeypatch, run)
    result = invoke("hook", "Stop")
    assert result.exit_code == 0 and result.stdout == ""


def test_hook_that_calls_sys_exit_still_exits_zero(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def run(event: str) -> int:
        raise SystemExit(2)

    monkeypatch.setenv("INGATAN_HOME", str(tmp_path))
    fake_hooks(monkeypatch, run)
    assert invoke("hook", "Stop").exit_code == 0
