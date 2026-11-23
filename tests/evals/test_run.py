from __future__ import annotations

import contextlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from evals.__main__ import main
from evals.driver import Hit, InProcessDriver, MemoryDriver
from evals.metrics import compute
from evals.netguard import NetworkBlockedError, NetworkGuard
from evals.report import exit_reasons, format_metrics, write_report
from evals.runner import FAIL, PASS, SKIP, XFAIL, XPASS, RunResult, run_all, run_scenario
from evals.scenarios import Actor, Scenario, load_dir, parse_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "evals" / "scenarios"

TOKO = {"business": "tokoku", "agent": "asisten"}
SEO = {"business": "tokoku", "agent": "seo-agent"}

LEAK_SCENARIO: dict[str, Any] = {
    "id": "leak-agent-fake",
    "title": "agen lain mencari catatan pribadi",
    "category": "leak-agent",
    "source": "pattern",
    "actors": {"asisten": TOKO, "seo": SEO},
    "steps": [
        {
            "op": "remember",
            "key": "priv",
            "actor": "asisten",
            "text": "Catatan pribadi: Pak Wahyu minta diskon 15 persen.",
            "keywords": ["negosiasi", "potongan harga"],
        },
        {"op": "recall", "actor": "seo", "query_text_of": "priv", "expect_absent": ["priv"]},
    ],
}

PLAIN_SCENARIO: dict[str, Any] = {
    "id": "stale-fake",
    "title": "fakta baru ditemukan",
    "category": "stale",
    "source": "pattern",
    "actors": {"asisten": TOKO},
    "steps": [
        {
            "op": "remember",
            "key": "k",
            "text": "Harga kaos polos Rp 65.000 per pcs.",
            "keywords": ["tarif"],
        },
        {"op": "recall", "query": "harga kaos polos", "expect_top5": ["k"]},
    ],
}


class LeakyDriver(InProcessDriver):
    """A deliberately broken memory: every recall runs as the agent that wrote last,
    so callers see facts from outside their scope."""

    def __init__(self, home: Path, *, token_budget: int = 1000) -> None:
        super().__init__(home, token_budget=token_budget)
        self._writer: Actor | None = None

    def remember(self, actor: Actor, text: str, **kwargs: Any):  # type: ignore[override]
        self._writer = actor
        return super().remember(actor, text, **kwargs)

    def recall(self, actor: Actor, query: str, *, limit: int, include_closed: bool) -> list[Hit]:
        return super().recall(
            self._writer or actor, query, limit=limit, include_closed=include_closed
        )


def leaky_factory(home: Path, scenario: Scenario) -> MemoryDriver:
    return LeakyDriver(home, token_budget=scenario.token_budget)


def write_scenarios(folder: Path, *scenarios: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for data in scenarios:
        (folder / f"{data['id']}.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


def test_leaking_driver_produces_leak_rate_above_zero():
    run = run_all([parse_scenario(LEAK_SCENARIO)], leaky_factory)
    metrics = compute(run.records)
    assert metrics.leak_rate is not None and metrics.leak_rate > 0
    assert metrics.leaked_items >= 1
    assert run.results[0].status == FAIL
    assert any("LEAK" in f for f in run.results[0].failures)
    assert any("leak rate" in r for r in exit_reasons(run))


def test_leaking_driver_exits_non_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = write_scenarios(tmp_path / "scenarios", LEAK_SCENARIO)
    code = main(
        ["--scenarios-dir", str(folder), "--report", str(tmp_path / "r.md")],
        driver_factory=leaky_factory,
    )
    assert code == 1
    assert "leak rate" in capsys.readouterr().err


def test_a_leak_fails_the_run_even_when_the_scenario_is_xfail(tmp_path: Path):
    data = {**LEAK_SCENARIO, "xfail": "pretend this is a known gap"}
    run = run_all([parse_scenario(data)], leaky_factory)
    assert run.results[0].status == XFAIL
    assert exit_reasons(run)


def test_correct_driver_has_no_leak_on_the_same_scenario():
    run = run_all([parse_scenario(LEAK_SCENARIO)])
    assert run.results[0].status == PASS
    assert compute(run.records).leaked_items == 0


def test_main_exits_zero_and_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = write_scenarios(tmp_path / "scenarios", PLAIN_SCENARIO, LEAK_SCENARIO)
    report = tmp_path / "out" / "latest.md"
    assert main(["--scenarios-dir", str(folder), "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out and "leak rate" in out and "OK" in out
    text = report.read_text(encoding="utf-8")
    assert "## Metrics" in text and "leak rate" in text and "stale-fake" in text


def test_main_scenario_flag_runs_only_that_scenario(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    folder = write_scenarios(tmp_path / "s", PLAIN_SCENARIO, LEAK_SCENARIO)
    assert main(["--scenarios-dir", str(folder), "--scenario", "stale-fake", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "stale-fake" in out and "leak-agent-fake" not in out


def test_main_unknown_scenario_and_invalid_files_exit_two(tmp_path: Path):
    folder = write_scenarios(tmp_path / "s", PLAIN_SCENARIO)
    assert main(["--scenarios-dir", str(folder), "--scenario", "nope"]) == 2
    (folder / "broken.json").write_text("{", encoding="utf-8")
    assert main(["--scenarios-dir", str(folder)]) == 2


def test_failing_scenario_exits_non_zero_but_xfail_does_not(tmp_path: Path):
    failing = {
        **PLAIN_SCENARIO,
        "id": "stale-wrong",
        "steps": [
            PLAIN_SCENARIO["steps"][0],
            {"op": "recall", "query": "harga kaos polos", "expect_absent": ["k"]},
        ],
    }
    folder = write_scenarios(tmp_path / "a", failing)
    assert main(["--scenarios-dir", str(folder), "--report", str(tmp_path / "r.md")]) == 1
    marked = write_scenarios(tmp_path / "b", {**failing, "xfail": "known gap"})
    assert main(["--scenarios-dir", str(marked), "--report", str(tmp_path / "r.md")]) == 0


def test_xfail_that_passes_is_reported_as_xpass():
    run = run_all([parse_scenario({**PLAIN_SCENARIO, "xfail": "was a gap"})])
    assert run.results[0].status == XPASS
    assert exit_reasons(run) == []


def test_expect_error_must_match_the_raised_error():
    data = {
        **PLAIN_SCENARIO,
        "id": "forget-fake",
        "steps": [
            PLAIN_SCENARIO["steps"][0],
            {"op": "forget", "key": "k", "actor": "asisten", "expect_error": "ScopeError"},
        ],
    }
    run = run_all([parse_scenario(data)])
    assert run.results[0].status == FAIL
    assert "expected ScopeError, but the call succeeded" in run.results[0].failures[0]


def test_stale_result_is_detected():
    """A driver that hides the 'check first' status counts as stale."""

    class HidingDriver(InProcessDriver):
        def recall(
            self, actor: Actor, query: str, *, limit: int, include_closed: bool
        ) -> list[Hit]:
            hits = super().recall(actor, query, limit=limit, include_closed=include_closed)
            return [Hit(h.id, h.text, "valid", h.tier) for h in hits]

    data = {
        **PLAIN_SCENARIO,
        "id": "stale-hidden",
        "steps": [
            PLAIN_SCENARIO["steps"][0],
            {"op": "advance_days", "days": 31},
            {"op": "recall", "query": "harga kaos polos", "expect_top5": ["k"]},
        ],
    }
    run = run_all(
        [parse_scenario(data)], lambda home, sc: HidingDriver(home, token_budget=sc.token_budget)
    )
    assert compute(run.records).stale_rate == 1.0
    assert run.results[0].status == FAIL


# -- the no-network guard (NFR-3) ---------------------------------------------


def test_network_guard_blocks_and_records_sockets():
    original = socket.socket
    with NetworkGuard() as guard:
        with pytest.raises(NetworkBlockedError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("example.com", 80))
    assert len(guard.attempts) == 2
    assert socket.socket is original


def test_a_socket_opened_during_a_run_fails_the_run():
    class NetworkDriver(InProcessDriver):
        def recall(
            self, actor: Actor, query: str, *, limit: int, include_closed: bool
        ) -> list[Hit]:
            # The memory swallows the error; the guard still saw the attempt.
            with contextlib.suppress(OSError):
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            return super().recall(actor, query, limit=limit, include_closed=include_closed)

    run = run_all(
        [parse_scenario(PLAIN_SCENARIO)],
        lambda home, sc: NetworkDriver(home, token_budget=sc.token_budget),
    )
    assert run.network_attempts
    assert any("network" in r for r in exit_reasons(run))


def test_expect_none_passes_on_silence_and_fails_on_any_result():
    data = dict(PLAIN_SCENARIO, id="noise-fake", category="noise")
    data["steps"] = [
        PLAIN_SCENARIO["steps"][0],
        {"op": "recall", "query": "jadwal pertandingan bulu tangkis", "expect_none": True},
    ]
    assert run_all([parse_scenario(data)]).results[0].status == PASS
    data["steps"] = [
        PLAIN_SCENARIO["steps"][0],
        {"op": "recall", "query": "harga kaos polos", "expect_none": True},
    ]
    result = run_all([parse_scenario(data)]).results[0]
    assert result.status == FAIL
    assert "expected no result at all" in result.failures[0]


def test_a_scenario_that_needs_embeddings_is_skipped_without_a_model(tmp_path: Path):
    needy = [s for s in load_dir(SCENARIO_DIR) if s.needs == "embeddings"]
    assert needy, "at least one shipped scenario exercises the embedding model"
    for scenario in needy:
        skipped = run_scenario(scenario, tmp_path / f"{scenario.id}-a")
        assert skipped.status == SKIP and skipped.failures == []
        assert "needs an embedding model" in skipped.notes[0]
        # It really does depend on one: made to run on words alone, it fails.
        forced = run_scenario(scenario, tmp_path / f"{scenario.id}-b", honor_needs=False)
        assert forced.status == FAIL, scenario.id
    # A skipped scenario never fails the run.
    run = run_all(needy)
    assert run.count(SKIP) == len(needy) and exit_reasons(run) == []


def test_embeddings_mode_refuses_to_start_without_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = main(["--embeddings", "local", "--model-dir", str(tmp_path / "nothing-here")])
    assert code == 2
    err = capsys.readouterr().err
    assert "ingatan embeddings pull" in err or "pip install" in err


# -- the whole shipped suite, in process ------------------------------------------


@pytest.fixture(scope="module")
def suite() -> RunResult:
    return run_all(load_dir(SCENARIO_DIR))


def test_whole_suite_runs_without_unexpected_failures_or_leaks(suite: RunResult):
    failed = {r.scenario.id: r.failures for r in suite.results if r.status == FAIL}
    assert failed == {}
    metrics = compute(suite.records)
    assert metrics.leak_rate == 0
    assert metrics.total_items > 0
    assert suite.network_attempts == []
    assert exit_reasons(suite) == []


def test_whole_suite_metrics_are_reported(suite: RunResult, tmp_path: Path):
    metrics = compute(suite.records)
    assert metrics.recall_at_5 is not None and metrics.recall_at_5 > 0.8
    assert metrics.stale_rate is not None and metrics.stale_rate < 0.05
    lines = format_metrics(suite)
    assert len(lines) == 4
    report = tmp_path / "latest.md"
    write_report(report, suite)
    text = report.read_text(encoding="utf-8")
    assert "recall@5" in text and "Briefs measured" in text
    assert "Embeddings: none (FTS5 only)" in text
    for result in suite.results:
        assert result.scenario.id in text


def test_every_brief_is_within_its_budget(suite: RunResult):
    briefs = [r for r in suite.records if r.kind == "brief" and r.tokens is not None]
    assert all(r.budget is not None and (r.tokens or 0) <= r.budget for r in briefs)


# ---- deferred embedding in the harness ------------------------------------------------------


def test_defer_flag_needs_the_local_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--defer"]) == 2
    assert "--defer needs --embeddings local" in capsys.readouterr().err


@pytest.mark.parametrize(("fill", "expected"), [(True, PASS), (False, FAIL)])
def test_in_process_driver_fills_deferred_vectors_after_each_remember(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fill: bool, expected: str
) -> None:
    """With `defer`, the harness does at once what a server's worker does a moment later.

    The scenario needs a conflict only a model can see. Every text gets the same vector
    here, so the pair that shares an entity is flagged as soon as both vectors exist, and
    without the fill it never is.
    """
    from ingatan.facts import backfill
    from ingatan.scope import EmbeddingConfig
    from ingatan.search import embed as embed_store

    def fake(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(embed_store, "_embed_raw", fake)
    if not fill:
        monkeypatch.setattr(backfill, "fill_pending", lambda ctx, **_: backfill.FillReport(0, 0, 0))
    cfg = EmbeddingConfig(provider="local", defer=True)

    def factory(home: Path, scenario: Scenario) -> MemoryDriver:
        return InProcessDriver(home, token_budget=scenario.token_budget, embeddings=cfg)

    scenario = next(
        s for s in load_dir(SCENARIO_DIR) if s.id == "semantic-01-konflik-entitas-sama-kata-berbeda"
    )
    result = run_scenario(scenario, tmp_path / "home", factory)
    assert result.status == expected, result.failures


# ---- expect_first and repeat --------------------------------------------------------------------

ORDER_SCENARIO: dict[str, Any] = {
    "id": "usage-fake",
    "title": "urutan hasil",
    "category": "usage",
    "source": "pattern",
    "actors": {"asisten": TOKO},
    "steps": [
        {
            "op": "remember",
            "key": "tepat",
            "text": "Supplier gula pasir adalah toko-a.",
            "keywords": ["pemasok"],
            "entities": ["gula pasir"],
            "kind": "stable",
        },
        {
            "op": "remember",
            "key": "lain",
            "text": "Jadwal libur karyawan diumumkan tiap Jumat.",
            "keywords": ["cuti"],
            "entities": ["karyawan"],
            "kind": "stable",
        },
    ],
}


def _ordered(first: str, **extra: Any) -> Scenario:
    data = json.loads(json.dumps(ORDER_SCENARIO))
    data["steps"].append(
        {"op": "recall", "query": "supplier gula pasir", "expect_first": first, **extra}
    )
    return parse_scenario(data)


def test_expect_first_passes_when_that_fact_leads_and_fails_otherwise(tmp_path: Path) -> None:
    right = run_scenario(_ordered("tepat"), tmp_path / "right")
    assert right.status == PASS, right.failures
    wrong = run_scenario(_ordered("lain"), tmp_path / "wrong")
    assert wrong.status == FAIL
    assert "expected first: lain" in wrong.failures[0]


def test_repeat_recalls_that_many_times_and_judges_once(tmp_path: Path) -> None:
    calls: list[str] = []

    class Counting(InProcessDriver):
        def recall(
            self, actor: Actor, query: str, *, limit: int, include_closed: bool
        ) -> list[Hit]:
            calls.append(query)
            return super().recall(actor, query, limit=limit, include_closed=include_closed)

    def factory(home: Path, scenario: Scenario) -> MemoryDriver:
        return Counting(home, token_budget=scenario.token_budget)

    result = run_scenario(_ordered("tepat", repeat=4), tmp_path / "home", factory)
    assert result.status == PASS, result.failures
    assert calls == ["supplier gula pasir"] * 4
    assert len([r for r in result.records if r.kind == "recall"]) == 1


# ---- links and related facts ---------------------------------------------------------------------

GRAPH = "graph-01-dua-lompatan"


def test_the_graph_scenario_passes_with_its_links_and_fails_without(tmp_path: Path) -> None:
    path = SCENARIO_DIR / f"{GRAPH}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    with_links = run_scenario(parse_scenario(data), tmp_path / "with")
    assert with_links.status == PASS, with_links.failures
    # The same scenario without its `link` steps: the fact two hops away is out of reach.
    data["steps"] = [step for step in data["steps"] if step["op"] != "link"]
    without = run_scenario(parse_scenario(data), tmp_path / "without")
    assert without.status == FAIL
    assert "expected among the related facts" in without.failures[0]


def test_a_related_fact_is_judged_for_leaks_like_any_result(tmp_path: Path) -> None:
    class LeakyRelated(InProcessDriver):
        """Hands back, as "related", a private fact of another agent."""

        def recall(
            self, actor: Actor, query: str, *, limit: int, include_closed: bool
        ) -> list[Hit]:
            hits = super().recall(actor, query, limit=limit, include_closed=include_closed)
            other = Actor("tokoku", "seo-agent")
            rows = self.ctx(other).business.execute(
                "SELECT id, text, tier FROM facts WHERE agent = 'seo-agent'"
            )
            self.last_related = [Hit(r["id"], r["text"], "valid", r["tier"]) for r in rows]
            return hits

    def factory(home: Path, scenario: Scenario) -> MemoryDriver:
        return LeakyRelated(home, token_budget=scenario.token_budget)

    data = {
        "id": "leak-related-fake",
        "title": "fakta terkait milik agen lain",
        "category": "leak-agent",
        "source": "pattern",
        "actors": {"asisten": TOKO, "seo": SEO},
        "steps": [
            {"op": "remember", "key": "mine", "text": "Harga kaos Rp 65.000.", "keywords": ["x"]},
            {
                "op": "remember",
                "key": "theirs",
                "actor": "seo",
                "text": "Catatan pribadi agen SEO.",
                "keywords": ["y"],
            },
            {"op": "recall", "query": "harga kaos", "expect_top5": ["mine"]},
        ],
    }
    result = run_scenario(parse_scenario(data), tmp_path / "home", factory)
    assert result.status == FAIL
    assert any("LEAK" in failure for failure in result.failures)
    assert sum(r.leaked for r in result.records) == 1
