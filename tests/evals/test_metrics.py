from __future__ import annotations

import pytest
from evals.briefs import parse_brief
from evals.metrics import (
    CheckRecord,
    compute,
    leak_rate,
    recall_at_5,
    stale_rate,
    tokens_per_brief,
)


def recall(top5_ok: bool | None = True, *, stale: bool = False, items: int = 3, leaked: int = 0):
    return CheckRecord("s", "recall", "q", top5_ok, stale, items, leaked)


def brief(tokens: int | None, *, items: int = 4, leaked: int = 0):
    return CheckRecord("s", "brief", "tokoku/asisten", None, False, items, leaked, tokens, 1000)


def test_recall_at_5_counts_only_checks_with_expected_facts():
    records = [recall(True), recall(True), recall(False), recall(None), brief(100)]
    assert recall_at_5(records) == pytest.approx(2 / 3)


def test_recall_at_5_has_no_value_without_expectations():
    assert recall_at_5([recall(None)]) is None
    assert recall_at_5([]) is None


def test_stale_rate_is_share_of_recall_checks():
    records = [recall(stale=True), recall(), recall(), recall(), brief(10)]
    assert stale_rate(records) == pytest.approx(0.25)


def test_leak_rate_counts_results_of_recalls_and_briefs():
    records = [recall(items=4, leaked=1), brief(120, items=6, leaked=1), recall(items=10)]
    assert leak_rate(records) == pytest.approx(2 / 20)


def test_leak_rate_is_zero_when_nothing_leaks():
    assert leak_rate([recall(items=5), brief(50, items=5)]) == 0


def test_leak_rate_has_no_value_without_results():
    assert leak_rate([recall(items=0)]) is None


def test_tokens_per_brief_mean_and_max():
    assert tokens_per_brief([brief(100), brief(300), recall()]) == (200.0, 300)


def test_tokens_per_brief_ignores_unmeasured_briefs():
    assert tokens_per_brief([brief(None)]) is None
    assert tokens_per_brief([brief(None), brief(80)]) == (80.0, 80)


def test_compute_bundles_everything():
    m = compute([recall(True, items=2, leaked=1), recall(False, stale=True), brief(40)])
    assert m.recall_at_5 == 0.5 and m.recall_checks == 2
    assert m.stale_rate == 0.5 and m.stale_checks == 2
    assert m.leaked_items == 1 and m.total_items == 9
    assert (m.tokens_mean, m.tokens_max, m.briefs) == (40.0, 40, 1)


BRIEF_TEXT = """# Ingatan brief: tokoku / asisten
Pending proposals waiting for the user: 2

## Check first
Do not act on these facts before checking them.
- [f_0123456789ab] harga lama (review was due 2026-10-01)

## Global
- [f_aaaaaaaaaaaa] pemilik santai

## Agent
- [f_bbbbbbbbbbbb] catatan agen possible conflict

## Episodes to digest
### Episode 1 (2026-11-10)
- [f_cccccccccccc] ini bukan fakta
"""


def test_parse_brief_sections_and_pending_count():
    parsed = parse_brief(BRIEF_TEXT)
    assert parsed.pending == 2
    assert parsed.ids_in("check first") == ["f_0123456789ab"]
    assert parsed.ids_in("global") == ["f_aaaaaaaaaaaa"]
    assert parsed.ids_in("agent") == ["f_bbbbbbbbbbbb"]
    assert parsed.ids_in("business") == []
    assert "f_cccccccccccc" not in parsed.all_ids()


def test_parse_brief_without_pending_line_means_zero():
    assert parse_brief("# Ingatan brief: a / b\n\n## Agent\n- [f_bbbbbbbbbbbb] x\n").pending == 0
