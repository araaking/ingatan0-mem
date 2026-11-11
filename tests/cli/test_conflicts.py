"""T-28 / NFR-9: the conflict-resolution report, from a prepared log and from real use."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import Result

from ingatan import audit, facts
from ingatan.audit import LogEntry
from ingatan.cli.conflicts import ConflictReport, compute_report, render_report
from ingatan.scope import Context

Invoke = Callable[..., Result]


class Log:
    """Build a prepared log, oldest first, with sequential ids."""

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    def add(self, op: str, fact_id: str | None, after: dict[str, Any] | None) -> int:
        entry_id = len(self.entries) + 1
        change = None if after is None else {"before": None, "after": after}
        self.entries.append(
            LogEntry(entry_id, "2026-11-11T09:00:00Z", "a", op, fact_id, "agent", "a", change)
        )
        return entry_id

    def create(self, fact_id: str, *conflicts_with: str, op: str = "create") -> int:
        return self.add(op, fact_id, {"id": fact_id, "conflicts_with": list(conflicts_with)})

    def supersede(self, fact_id: str, new: str) -> int:
        return self.add("supersede", fact_id, {"superseded_by": new, "resolved_conflicts": [1]})

    def confirm(self, fact_id: str, reason: str) -> int:
        return self.add("confirm", fact_id, {"reason": reason, "review_after": None})

    def undo(self, target: int, fact_id: str) -> int:
        return self.add("undo", fact_id, {"undoes": target})


def test_a_prepared_log_gives_the_expected_counts_and_share() -> None:
    log = Log()
    log.create("f_a")
    log.create("f_b", "f_a")  # candidate 1
    log.create("f_c", "f_a")  # candidate 2
    log.create("f_d", "f_c")  # candidate 3
    log.create("f_e", "f_d")  # candidate 4
    log.supersede("f_a", "f_b")  # resolves 1 and 2
    log.confirm("f_c", "Keduanya benar, lihat F_D juga")  # resolves 3 (id named, any case)
    report = compute_report(log.entries)
    assert report == ConflictReport(candidates=4, resolved=3)
    assert report.share == 0.75 and not report.below_half


def test_approve_entries_count_as_candidates_too() -> None:
    log = Log()
    log.create("f_a")
    log.create("f_b", "f_a", op="approve")
    assert compute_report(log.entries) == ConflictReport(1, 0)


def test_a_confirm_that_does_not_name_the_other_fact_resolves_nothing() -> None:
    log = Log()
    log.create("f_a")
    log.create("f_b", "f_a")
    log.confirm("f_b", "masih benar")
    log.confirm("f_b", "bandingkan dengan f_zzz")
    assert compute_report(log.entries) == ConflictReport(1, 0)


def test_a_queued_supersede_of_a_shared_fact_resolves_nothing() -> None:
    log = Log()
    log.create("f_a")
    log.create("f_b", "f_a")
    log.add("supersede", "f_a", {"pending_proposal": "p_1"})
    assert compute_report(log.entries) == ConflictReport(1, 0)


def test_a_supersede_before_the_pair_existed_does_not_resolve_it() -> None:
    log = Log()
    log.create("f_a")
    log.supersede("f_a", "f_x")
    log.create("f_b", "f_a")
    assert compute_report(log.entries) == ConflictReport(1, 0)


def test_undone_entries_do_not_count() -> None:
    log = Log()
    log.create("f_a")
    log.create("f_b", "f_a")
    undone_create = log.create("f_c", "f_a")
    log.undo(undone_create, "f_c")
    undone_supersede = log.supersede("f_a", "f_b")
    log.undo(undone_supersede, "f_a")
    assert compute_report(log.entries) == ConflictReport(1, 0)


def test_entries_without_content_and_pairs_listed_twice_are_handled() -> None:
    log = Log()
    log.add("create", "f_gone", None)  # forgotten
    log.create("f_a")
    log.create("f_b", "f_a", "f_a")
    assert compute_report(log.entries).candidates == 1


def test_the_failure_note_appears_only_below_one_half() -> None:
    below = render_report(ConflictReport(candidates=5, resolved=2))
    assert "Conflict candidates: 5" in below and "Resolved by supersede or confirm: 2" in below
    assert "Share resolved: 40.0%" in below and "API key" in below
    exactly_half = render_report(ConflictReport(candidates=4, resolved=2))
    assert "Share resolved: 50.0%" in exactly_half and "API key" not in exactly_half
    assert "API key" not in render_report(ConflictReport(candidates=3, resolved=3))


def test_an_empty_log_has_no_share_and_no_failure_note() -> None:
    text = render_report(compute_report([]))
    assert "Conflict candidates: 0" in text and "n/a" in text and "API key" not in text


# ---- the command, on a real log ----------------------------------------------------------


def rem(ctx: Context, text: str) -> str:
    result = facts.remember(ctx, text, keywords=["harga"], entities=["kopi"])
    assert result.fact_id is not None
    return result.fact_id


def test_the_conflicts_command_reports_share_from_real_use(
    cli_ctx: Context, invoke: Invoke
) -> None:
    old = rem(cli_ctx, "Harga kopi 50 ribu")
    new = rem(cli_ctx, "Harga kopi 55 ribu")  # candidate 1
    third = rem(cli_ctx, "Harga kopi 60 ribu")  # candidates 2 and 3 (similar to both)
    facts.supersede(cli_ctx, old, new)  # resolves (new, old) and (third, old)
    result = invoke("conflicts")
    assert result.exit_code == 0, result.output
    assert third and "Conflict candidates: 3" in result.stdout
    assert "Resolved by supersede or confirm: 2" in result.stdout
    assert "Share resolved: 66.7%" in result.stdout and "API key" not in result.stdout


def test_the_conflicts_command_prints_the_failure_note_under_one_half(
    cli_ctx: Context, invoke: Invoke
) -> None:
    rem(cli_ctx, "Harga kopi 50 ribu")
    rem(cli_ctx, "Harga kopi 55 ribu")
    result = invoke("conflicts")
    assert "Share resolved: 0.0%" in result.stdout and "digest with an API key" in result.stdout


def test_undoing_a_supersede_takes_it_out_of_the_resolved_count(
    cli_ctx: Context, invoke: Invoke
) -> None:
    old = rem(cli_ctx, "Harga kopi 50 ribu")
    new = rem(cli_ctx, "Harga kopi 55 ribu")
    facts.supersede(cli_ctx, old, new)
    assert "Share resolved: 100.0%" in invoke("conflicts").stdout
    entry = next(e for e in audit.entries(cli_ctx) if e.op == "supersede")
    audit.undo(cli_ctx, entry.id)
    assert "Share resolved: 0.0%" in invoke("conflicts").stdout


def test_the_conflicts_command_on_an_empty_log(configured: None, invoke: Invoke) -> None:
    result = invoke("conflicts")
    assert result.exit_code == 0 and "Conflict candidates: 0" in result.stdout


@pytest.mark.parametrize("flag", [[], ["--agent", "claude-code"]])
def test_the_conflicts_command_accepts_the_agent_option(
    cli_ctx: Context, invoke: Invoke, flag: list[str]
) -> None:
    rem(cli_ctx, "Harga kopi 50 ribu")
    assert invoke("conflicts", *flag).exit_code == 0
