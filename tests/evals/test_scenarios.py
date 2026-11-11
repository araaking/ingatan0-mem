from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from evals.scenarios import (
    CATEGORIES,
    ScenarioError,
    lint_dependencies,
    load_dir,
    load_file,
    parse_scenario,
)

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "evals" / "scenarios"

VALID: dict[str, Any] = {
    "id": "stale-test",
    "title": "contoh",
    "category": "stale",
    "source": "pattern",
    "actors": {"a": {"business": "tokoku", "agent": "asisten"}},
    "steps": [
        {"op": "remember", "key": "k", "text": "Harga kaos Rp 65.000.", "keywords": ["tarif"]},
        {"op": "advance_days", "days": 31},
        {"op": "recall", "query": "harga kaos", "expect_top5": ["k"]},
    ],
}


def broken(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


def with_step(step: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    data["steps"].append(step)
    return data


def test_valid_scenario_parses():
    scenario = parse_scenario(VALID)
    assert scenario.id == "stale-test"
    assert scenario.default_actor == "a"
    assert [s.op for s in scenario.steps] == ["remember", "advance_days", "recall"]


CASES: list[tuple[str, dict[str, Any], str]] = [
    ("not an object", [], "one JSON object"),  # type: ignore[list-item]
    ("missing id", {k: v for k, v in VALID.items() if k != "id"}, "missing field 'id'"),
    ("missing steps", {k: v for k, v in VALID.items() if k != "steps"}, "missing field 'steps'"),
    ("unknown field", broken(colour="red"), "unknown field(s): colour"),
    ("bad id", broken(id="Stale Test"), "'id' must be lowercase"),
    ("bad category", broken(category="misc"), "'category' must be one of"),
    ("bad source", broken(source="guess"), "'source' must be one of"),
    ("empty xfail", broken(xfail=" "), "'xfail' must be a non-empty reason"),
    ("bad needs", broken(needs="magic"), "'needs' must be one of"),
    ("bad budget", broken(token_budget=0), "'token_budget' must be a positive integer"),
    ("no actors", broken(actors={}), "'actors' must be an object"),
    ("bad actor", broken(actors={"a": {"business": "x"}}), "needs exactly 'business' and 'agent'"),
    ("no steps", broken(steps=[]), "'steps' must be a non-empty list"),
    ("unknown op", with_step({"op": "teleport"}), "'op' must be one of"),
    ("unknown actor", with_step({"op": "brief", "actor": "z"}), "unknown actor 'z'"),
    (
        "missing remember field",
        with_step({"op": "remember", "key": "q", "text": "x"}),
        "missing field(s): keywords",
    ),
    (
        "empty keywords",
        with_step({"op": "remember", "key": "q", "text": "x", "keywords": []}),
        "'keywords' must not be empty",
    ),
    (
        "duplicate key",
        with_step({"op": "remember", "key": "k", "text": "x", "keywords": ["y"]}),
        "key 'k' is defined twice",
    ),
    (
        "bad tier",
        with_step({"op": "remember", "key": "q", "text": "x", "keywords": ["y"], "tier": "x"}),
        "'tier' must be one of",
    ),
    (
        "extra field",
        with_step({"op": "advance_days", "days": 1, "weeks": 2}),
        "unexpected field(s): weeks",
    ),
    ("negative days", with_step({"op": "advance_days", "days": -1}), "'days' must be a positive"),
    (
        "supersede unknown key",
        with_step({"op": "supersede", "old": "k", "new": "nope"}),
        "unknown key 'nope'",
    ),
    (
        "confirm reason names unknown key",
        with_step({"op": "confirm", "key": "k", "reason": "lihat {nope}"}),
        "unknown key '{nope}'",
    ),
    ("approve a fact", with_step({"op": "approve", "key": "k"}), "is not a proposal"),
    ("recall without expectation", with_step({"op": "recall", "query": "x"}), "needs expect_top5"),
    (
        "recall query and query_text_of",
        with_step({"op": "recall", "query": "x", "query_text_of": "k", "expect_absent": ["k"]}),
        "exactly one of 'query' and 'query_text_of'",
    ),
    (
        "recall unknown expected key",
        with_step({"op": "recall", "query": "x", "expect_top5": ["ghost"]}),
        "unknown key 'ghost'",
    ),
    (
        "recall limit under five",
        with_step({"op": "recall", "query": "x", "limit": 3, "expect_absent": ["k"]}),
        "'limit' must be an integer of at least 5",
    ),
    (
        "recall six expected",
        with_step({"op": "recall", "query": "x", "expect_top5": ["k", "k", "k", "k", "k", "k"]}),
        "more than five keys",
    ),
    (
        "brief unknown section",
        with_step({"op": "brief", "expect_in_section": {"basement": ["k"]}}),
        "unknown section 'basement'",
    ),
    (
        "brief bad max_tokens",
        with_step({"op": "brief", "max_tokens": "many"}),
        "'max_tokens' must be a positive integer",
    ),
    (
        "expect_error with key",
        with_step(
            {
                "op": "remember",
                "key": "z",
                "text": "x",
                "keywords": [],
                "expect_error": "ValidationError",
            }
        ),
        "defines no key",
    ),
]


@pytest.mark.parametrize(("name", "data", "message"), CASES, ids=[c[0] for c in CASES])
def test_invalid_scenarios_are_rejected_with_a_clear_message(
    name: str, data: dict[str, Any], message: str
):
    with pytest.raises(ScenarioError) as info:
        parse_scenario(data, source_name="x.json")
    assert message in str(info.value)
    assert str(info.value).startswith("x.json")


def test_error_names_the_step():
    with pytest.raises(ScenarioError, match=r"step 4 \(advance_days\)"):
        parse_scenario(with_step({"op": "advance_days", "days": 0}))


def test_remember_with_expect_error_may_have_no_keywords():
    data = with_step(
        {"op": "remember", "text": "x", "keywords": [], "expect_error": "ValidationError"}
    )
    assert parse_scenario(data).steps[-1].expect_error == "ValidationError"


def test_a_proposal_key_becomes_a_fact_key_after_approve():
    data = broken(
        steps=[
            {"op": "remember", "key": "p", "text": "x", "keywords": ["y"], "tier": "business"},
            {"op": "supersede", "old": "p", "new": "p"},
        ]
    )
    with pytest.raises(ScenarioError, match="is a proposal, not a fact"):
        parse_scenario(data)
    data["steps"].insert(1, {"op": "approve", "key": "p"})
    assert parse_scenario(data).steps[-1].op == "supersede"


def test_load_file_rejects_bad_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScenarioError, match=r"bad.json: not valid JSON"):
        load_file(path)


def test_load_file_requires_file_name_to_match_id(tmp_path: Path):
    path = tmp_path / "other-name.json"
    path.write_text(json.dumps(VALID), encoding="utf-8")
    with pytest.raises(ScenarioError, match="file name must be the scenario id"):
        load_file(path)


def test_load_dir_rejects_duplicate_ids(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    for folder in (tmp_path, tmp_path / "sub"):
        (folder / "stale-test.json").write_text(json.dumps(VALID), encoding="utf-8")
    with pytest.raises(ScenarioError, match="used by two files"):
        load_dir(tmp_path)


def test_load_dir_reads_subfolders_and_ignores_other_files(tmp_path: Path):
    (tmp_path / "incidents").mkdir()
    (tmp_path / "incidents" / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "stale-test.json").write_text(json.dumps(VALID), encoding="utf-8")
    assert [s.id for s in load_dir(tmp_path)] == ["stale-test"]


# -- the shipped scenario set -------------------------------------------------


@pytest.fixture(scope="module")
def shipped():
    return load_dir(SCENARIO_DIR)


def test_shipped_scenarios_load(shipped: list[Any]):
    assert 36 <= len(shipped) <= 40


def test_shipped_scenarios_cover_every_category(shipped: list[Any]):
    assert {s.category for s in shipped} == set(CATEGORIES)


def test_shipped_scenarios_are_all_patterns_never_invented_incidents(shipped: list[Any]):
    assert {s.source for s in shipped} == {"pattern"}


def test_at_least_five_inflection_scenarios_depend_on_keywords(shipped: list[Any]):
    runnable = [s for s in shipped if s.category == "inflection" and s.needs == "keywords"]
    assert len(runnable) >= 5
    assert len([s for s in runnable if not s.xfail]) >= 5


def test_a_few_partial_word_scenarios_depend_on_trigram(shipped: list[Any]):
    assert len([s for s in shipped if s.category == "partial-word" and s.needs == "trigram"]) >= 3


def test_shipped_scenarios_depend_on_what_they_claim(shipped: list[Any]):
    problems = {s.id: lint_dependencies(s) for s in shipped if lint_dependencies(s)}
    assert problems == {}


def test_every_xfail_has_a_reason(shipped: list[Any]):
    assert all(s.xfail is None or len(s.xfail) > 20 for s in shipped)


def test_lint_flags_a_scenario_that_does_not_need_keywords():
    data = broken(needs="keywords")
    problems = lint_dependencies(parse_scenario(data))
    assert any("already matches the text" in p for p in problems)


def test_lint_flags_a_query_stuffed_into_keywords():
    data = copy.deepcopy(VALID)
    data["needs"] = "keywords"
    data["steps"][0]["keywords"] = ["pesan kaos"]
    data["steps"][2]["query"] = "pesan kaos"
    problems = lint_dependencies(parse_scenario(data))
    assert any("stuffed" in p for p in problems)


def test_lint_checks_trigram_scenarios():
    data = copy.deepcopy(VALID)
    data["needs"] = "trigram"
    data["steps"][2]["query"] = "harga"  # a whole word, so no trigram needed
    assert any("without the trigram index" in p for p in lint_dependencies(parse_scenario(data)))
    data["steps"][2]["query"] = "kao"  # a fragment of 'kaos'
    assert lint_dependencies(parse_scenario(data)) == []
