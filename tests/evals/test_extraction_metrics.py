"""The extraction eval: judging what a model extracted against a known answer.

CI has no model. These tests cover the judging, the cases that ship, and one run end to end
with a stand-in model.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from evals.extraction import metrics
from evals.extraction.__main__ import main, only_endpoint, run_case

from ingatan.extract import llm
from ingatan.scope import LlmConfig

CASES_DIR = Path(__file__).resolve().parents[2] / "evals" / "extraction" / "cases"
LLM = LlmConfig(provider="ollama", model="local-model")

CASE: dict[str, Any] = {
    "id": "ex-test",
    "title": "contoh",
    "episodes": [
        {
            "user": "Ongkir ke Bandung sekarang Rp 15.000 per kg. Stok gula tinggal 20 karung.",
            "assistant": "Baik, saya catat keduanya.",
        }
    ],
    "expected": [
        {"name": "ongkir", "all_of": ["ongkir", "bandung"], "values": ["15.000"]},
        {"name": "stok", "all_of": ["stok", "gula"], "values": ["20"]},
    ],
    "forbidden": ["kata sandi"],
}


def case(**changes: Any) -> metrics.Case:
    return metrics.parse_case({**CASE, **changes})


def answer(*texts: str) -> str:
    return json.dumps({"facts": [{"text": text} for text in texts]})


# ---- judging --------------------------------------------------------------------------------


def test_a_fact_with_the_words_and_the_value_is_correct() -> None:
    result = metrics.judge(case(), ["Ongkir ke Bandung Rp15,000 per kg."])
    assert (result.correct, result.invented, result.found, result.expected) == (1, 0, 1, 2)


def test_a_fact_with_another_value_is_invented() -> None:
    result = metrics.judge(case(), ["Ongkir ke Bandung Rp 18.000 per kg."])
    assert (result.correct, result.invented, result.found) == (0, 1, 0)


def test_a_fact_that_pairs_the_wrong_value_with_a_subject_is_invented() -> None:
    result = metrics.judge(case(), ["Stok gula tinggal 15.000 karung."])
    assert (result.correct, result.invented) == (0, 1)


def test_a_fact_about_something_else_is_invented() -> None:
    result = metrics.judge(case(), ["Toko buka tiap hari."])
    assert (result.correct, result.invented, result.found) == (0, 1, 0)


def test_a_forbidden_word_makes_a_fact_invented_whatever_else_it_says() -> None:
    result = metrics.judge(case(), ["Ongkir ke Bandung Rp 15.000, kata sandi admin rahasia."])
    assert (result.correct, result.invented, result.found) == (0, 1, 0)


def test_two_facts_for_one_expected_are_both_correct_and_found_once() -> None:
    texts = ["Ongkir ke Bandung Rp 15.000 per kg.", "Tarif ongkir Bandung: Rp 15.000."]
    result = metrics.judge(case(), texts)
    assert (result.correct, result.invented, result.found) == (2, 0, 1)


def test_nothing_extracted_finds_nothing_and_invents_nothing() -> None:
    result = metrics.judge(case(), [])
    assert (result.correct, result.invented, result.found, result.expected) == (0, 0, 0, 2)


def test_totals_and_shares() -> None:
    results = [
        metrics.judge(case(), ["Ongkir ke Bandung Rp 15.000.", "Toko buka tiap hari."]),
        metrics.judge(case(id="ex-b"), ["Stok gula 20 karung."]),
    ]
    totals = metrics.totals(results)
    assert (totals.extracted, totals.correct, totals.invented) == (3, 2, 1)
    assert (totals.expected, totals.found) == (4, 2)
    assert totals.correct_share == pytest.approx(2 / 3)
    assert totals.invented_share == pytest.approx(1 / 3)
    assert totals.found_share == pytest.approx(0.5)
    empty = metrics.totals([metrics.judge(case(expected=[]), [])])
    assert empty.correct_share is None and empty.found_share is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "Bad Id"}, "'id' must be"),
        ({"episodes": []}, "'episodes' must be"),
        ({"episodes": [{"user": "x"}]}, "'user' and 'assistant'"),
        ({"expected": [{"name": "a"}]}, "'all_of' or 'values'"),
        ({"colour": "red"}, "unknown field"),
    ],
)
def test_a_malformed_case_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        case(**changes)


# ---- the cases that ship -------------------------------------------------------------------


def test_shipped_cases_load_and_hold_both_kinds_of_trap() -> None:
    cases = metrics.load_cases(CASES_DIR)
    assert len(cases) >= 6
    assert len({c.id for c in cases}) == len(cases)
    assert any(not c.expected for c in cases), "a case where the right answer is nothing"
    assert any(c.forbidden for c in cases), "a case with something that must not come out"
    for item in cases:
        # Every value a case expects is stated in its own episodes: a case cannot ask for
        # something Ingatan's number check would drop.
        said = " ".join(f"{u} {a}" for u, a in item.episodes)
        for expected in item.expected:
            for value in expected.values:
                assert metrics.numbers(value) <= metrics.numbers(said), (item.id, value)


# ---- one run with a stand-in model ---------------------------------------------------------


def test_run_case_counts_what_reaches_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def chat(cfg: LlmConfig, system: str, user: str) -> str:
        return answer(
            "Ongkir ke Bandung Rp 15.000 per kg.",
            "Ongkir ke Bandung Rp 99.000 per kg.",  # Ingatan drops this: 99.000 was never said
            "Toko buka tiap hari.",  # nothing in it is checkable: it reaches the queue
        )

    monkeypatch.setattr(llm, "chat", chat)
    result = run_case(case(), LLM, tmp_path / "home")
    assert sorted(result.extracted) == [
        "Ongkir ke Bandung Rp 15.000 per kg.",
        "Toko buka tiap hari.",
    ]
    assert (result.correct, result.invented, result.found, result.expected) == (1, 1, 1, 2)
    assert (result.dropped, result.errors) == (1, 0)


def test_main_without_a_model_says_how_to_set_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("INGATAN_LLM_PROVIDER", raising=False)
    assert main([]) == 2
    assert "[llm]" in capsys.readouterr().err


def test_main_writes_a_report_with_the_model_and_the_shares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INGATAN_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("INGATAN_LLM_MODEL", "local-model")

    def chat(cfg: LlmConfig, system: str, user: str) -> str:
        return answer()  # a model that finds nothing: nothing invented, nothing found

    monkeypatch.setattr(llm, "chat", chat)
    report = tmp_path / "report.md"
    assert main(["--report", str(report), "--no-network-guard"]) == 0
    text = report.read_text(encoding="utf-8")
    assert "ollama/local-model" in text
    assert "found" in text and "invented" in text
    assert "0 of" in text
    assert "report written to" in capsys.readouterr().out


# ---- the one endpoint a run may reach ------------------------------------------------------


def test_only_the_configured_endpoint_may_be_reached() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with only_endpoint("127.0.0.1", port) as attempts:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            with pytest.raises(OSError, match="not the configured endpoint"):
                socket.create_connection(("192.0.2.1", 80), timeout=1)
        assert attempts == ["192.0.2.1:80"]
    finally:
        listener.close()
    # Outside the block the standard function is back.
    assert socket.create_connection.__module__ == "socket"
