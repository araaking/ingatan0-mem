"""Scenario files: loading, validation and a static dependency lint.

A scenario is one JSON file. See `evals/README.md` for the format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BRIEF_SECTIONS",
    "CATEGORIES",
    "OPS",
    "Actor",
    "Scenario",
    "ScenarioError",
    "Step",
    "lint_dependencies",
    "load_dir",
    "load_file",
    "parse_scenario",
]

CATEGORIES = (
    "stale",
    "supersede",
    "leak-agent",
    "leak-business",
    "inflection",
    "partial-word",
    "brief",
    "conflict",
    "proposal",
    "forget",
)
SOURCES = ("pattern", "incident")
NEEDS = ("keywords", "trigram")
BRIEF_SECTIONS = ("check first", "global", "business", "agent")
OPS = (
    "remember",
    "supersede",
    "confirm",
    "forget",
    "approve",
    "reject",
    "advance_days",
    "recall",
    "brief",
)
TIERS = ("agent", "business", "global")
KINDS = ("stable", "volatile")

_KEY = re.compile(r"^[A-Za-z0-9_]+$")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Fields each op accepts besides `op`, `actor` and `expect_error`: (required, optional).
_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "remember": (
        {"text", "keywords"},
        {"key", "entities", "tier", "kind", "review_in_days"},
    ),
    "supersede": ({"old", "new"}, {"as"}),
    "confirm": ({"key", "reason"}, set()),
    "forget": ({"key"}, {"by_user"}),
    "approve": ({"key"}, set()),
    "reject": ({"key"}, set()),
    "advance_days": ({"days"}, set()),
    "recall": (
        set(),
        {
            "query",
            "query_text_of",
            "limit",
            "include_closed",
            "expect_top5",
            "expect_absent",
            "expect_status",
        },
    ),
    "brief": (
        set(),
        {"expect_in_section", "expect_not_in", "max_tokens", "expect_pending"},
    ),
}


class ScenarioError(ValueError):
    """A scenario file is malformed. The message names the file and, if known, the step."""


@dataclass(frozen=True)
class Actor:
    business: str
    agent: str


@dataclass(frozen=True)
class Step:
    index: int  # 1-based position in the file, used in messages
    op: str
    actor: str | None
    data: dict[str, Any]
    expect_error: str | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    category: str
    source: str
    actors: dict[str, Actor]
    steps: list[Step]
    xfail: str | None = None
    needs: str | None = None
    token_budget: int = 1000
    path: Path | None = field(default=None, compare=False)

    @property
    def default_actor(self) -> str:
        return next(iter(self.actors))


# ---------------------------------------------------------------------------
# parsing


def _fail(where: str, message: str) -> ScenarioError:
    return ScenarioError(f"{where}: {message}")


def _str_list(where: str, name: str, value: Any, *, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):  # pyright: ignore[reportUnknownVariableType]
        raise _fail(where, f"'{name}' must be a list of non-empty strings")
    items: list[str] = list(value)  # pyright: ignore[reportUnknownArgumentType]
    if non_empty and not items:
        raise _fail(where, f"'{name}' must not be empty")
    return items


def _check_key(where: str, name: str, value: Any) -> str:
    if not isinstance(value, str) or not _KEY.match(value):
        raise _fail(where, f"'{name}' must be a key made of letters, digits and '_'")
    return value


def parse_scenario(
    raw: Any, *, source_name: str = "<scenario>", path: Path | None = None
) -> Scenario:
    """Validate decoded JSON and build a `Scenario`. Raises `ScenarioError`."""
    if not isinstance(raw, dict):
        raise _fail(source_name, "the file must hold one JSON object")
    data: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    known = {
        "id",
        "title",
        "category",
        "source",
        "xfail",
        "needs",
        "token_budget",
        "actors",
        "steps",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        raise _fail(source_name, f"unknown field(s): {', '.join(unknown)}")
    for name in ("id", "title", "category", "source", "actors", "steps"):
        if name not in data:
            raise _fail(source_name, f"missing field '{name}'")
    sid = data["id"]
    if not isinstance(sid, str) or not re.match(r"^[a-z0-9][a-z0-9-]*$", sid):
        raise _fail(source_name, "'id' must be lowercase letters, digits and '-'")
    where = f"{source_name} [{sid}]"
    title = data["title"]
    if not isinstance(title, str) or not title.strip():
        raise _fail(where, "'title' must be a non-empty string")
    if data["category"] not in CATEGORIES:
        raise _fail(where, f"'category' must be one of: {', '.join(CATEGORIES)}")
    if data["source"] not in SOURCES:
        raise _fail(where, f"'source' must be one of: {', '.join(SOURCES)}")
    xfail = data.get("xfail")
    if xfail is not None and (not isinstance(xfail, str) or not xfail.strip()):
        raise _fail(where, "'xfail' must be a non-empty reason string")
    needs = data.get("needs")
    if needs is not None and needs not in NEEDS:
        raise _fail(where, f"'needs' must be one of: {', '.join(NEEDS)}")
    budget = data.get("token_budget", 1000)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise _fail(where, "'token_budget' must be a positive integer")

    actors = _parse_actors(where, data["actors"])
    steps = _parse_steps(where, data["steps"], actors)
    return Scenario(
        id=sid,
        title=title,
        category=str(data["category"]),
        source=str(data["source"]),
        actors=actors,
        steps=steps,
        xfail=xfail,
        needs=needs,
        token_budget=budget,
        path=path,
    )


def _parse_actors(where: str, raw: Any) -> dict[str, Actor]:
    if not isinstance(raw, dict) or not raw:
        raise _fail(where, "'actors' must be an object with at least one actor")
    actors: dict[str, Actor] = {}
    for name, spec in raw.items():  # pyright: ignore[reportUnknownVariableType]
        _check_key(where, "actor name", name)
        if (
            not isinstance(spec, dict)
            or set(spec) != {"business", "agent"}  # pyright: ignore[reportUnknownArgumentType]
            or not all(isinstance(v, str) and v for v in spec.values())  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        ):
            raise _fail(where, f"actor '{name}' needs exactly 'business' and 'agent' strings")
        actors[str(name)] = Actor(business=spec["business"], agent=spec["agent"])  # pyright: ignore[reportUnknownArgumentType]
    return actors


def _parse_steps(where: str, raw: Any, actors: dict[str, Actor]) -> list[Step]:
    if not isinstance(raw, list) or not raw:
        raise _fail(where, "'steps' must be a non-empty list")
    steps: list[Step] = []
    keys: dict[str, str] = {}  # key -> "fact" | "proposal"
    for i, item in enumerate(raw, start=1):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        at = f"{where} step {i}"
        if not isinstance(item, dict):
            raise _fail(at, "a step must be an object")
        body: dict[str, Any] = dict(item)  # pyright: ignore[reportUnknownArgumentType]
        op = body.pop("op", None)
        if op not in OPS:
            raise _fail(at, f"'op' must be one of: {', '.join(OPS)}")
        at = f"{at} ({op})"
        actor = body.pop("actor", None)
        if actor is not None and actor not in actors:
            raise _fail(at, f"unknown actor '{actor}'")
        expect_error = body.pop("expect_error", None)
        if expect_error is not None and not isinstance(expect_error, str):
            raise _fail(at, "'expect_error' must be an error class name")
        required, optional = _FIELDS[op]
        missing = sorted(required - set(body))
        if missing:
            raise _fail(at, f"missing field(s): {', '.join(missing)}")
        extra = sorted(set(body) - required - optional)
        if extra:
            raise _fail(at, f"unexpected field(s): {', '.join(extra)}")
        _validate_step(at, op, body, keys, expect_error)
        steps.append(Step(i, op, actor, body, expect_error))
    return steps


def _validate_step(
    at: str, op: str, body: dict[str, Any], keys: dict[str, str], expect_error: str | None
) -> None:
    def known(name: str, *, fact: bool = False) -> None:
        key = _check_key(at, name, body[name])
        if key not in keys:
            raise _fail(at, f"'{name}' refers to unknown key '{key}' (define it earlier)")
        if fact and keys[key] != "fact":
            raise _fail(at, f"'{name}' key '{key}' is a proposal, not a fact")

    def define(key: str, kind: str) -> None:
        _check_key(at, "key", key)
        if key in keys:
            raise _fail(at, f"key '{key}' is defined twice")
        keys[key] = kind

    if op == "remember":
        text = body["text"]
        if not isinstance(text, str) or not text.strip():
            raise _fail(at, "'text' must be a non-empty string")
        words = _str_list(at, "keywords", body["keywords"])
        if not words and expect_error is None:
            raise _fail(at, "'keywords' must not be empty (unless 'expect_error' is set)")
        if "entities" in body:
            _str_list(at, "entities", body["entities"])
        if body.get("tier", "agent") not in TIERS:
            raise _fail(at, f"'tier' must be one of: {', '.join(TIERS)}")
        if body.get("kind", "volatile") not in KINDS:
            raise _fail(at, f"'kind' must be one of: {', '.join(KINDS)}")
        days = body.get("review_in_days")
        if days is not None and (isinstance(days, bool) or not isinstance(days, int)):
            raise _fail(at, "'review_in_days' must be an integer")
        if expect_error is None:
            if "key" not in body:
                raise _fail(at, "'key' is required (a name to refer to the fact later)")
            define(body["key"], "proposal" if body.get("tier", "agent") != "agent" else "fact")
        elif "key" in body:
            raise _fail(at, "a step with 'expect_error' defines no key")
    elif op == "supersede":
        known("old", fact=True)
        known("new")
        if "as" in body:
            define(body["as"], "proposal")
    elif op == "confirm":
        known("key", fact=True)
        if not isinstance(body["reason"], str):
            raise _fail(at, "'reason' must be a string")
        for ref in re.findall(r"\{(\w+)\}", body["reason"]):
            if ref not in keys:
                raise _fail(at, f"'reason' mentions unknown key '{{{ref}}}'")
    elif op == "forget":
        known("key", fact=True)
        if not isinstance(body.get("by_user", False), bool):
            raise _fail(at, "'by_user' must be true or false")
    elif op in ("approve", "reject"):
        known("key")
        if keys[body["key"]] != "proposal":
            raise _fail(at, f"key '{body['key']}' is not a proposal")
        # After approval the key names the new fact; after a rejection it names nothing.
        keys[body["key"]] = "fact" if op == "approve" else "rejected"
    elif op == "advance_days":
        days = body["days"]
        if isinstance(days, bool) or not isinstance(days, (int, float)) or days <= 0:
            raise _fail(at, "'days' must be a positive number")
    elif op == "recall":
        _validate_recall(at, body, keys)
    elif op == "brief":
        _validate_brief(at, body, keys)


def _validate_recall(at: str, body: dict[str, Any], keys: dict[str, str]) -> None:
    has_query = "query" in body
    has_of = "query_text_of" in body
    if has_query == has_of:
        raise _fail(at, "give exactly one of 'query' and 'query_text_of'")
    if has_query and (not isinstance(body["query"], str) or not body["query"].strip()):
        raise _fail(at, "'query' must be a non-empty string")
    if has_of and body["query_text_of"] not in keys:
        raise _fail(at, f"'query_text_of' refers to unknown key '{body['query_text_of']}'")
    limit = body.get("limit", 10)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 5:
        raise _fail(at, "'limit' must be an integer of at least 5")
    if not isinstance(body.get("include_closed", False), bool):
        raise _fail(at, "'include_closed' must be true or false")
    for name in ("expect_top5", "expect_absent"):
        if name in body:
            for key in _str_list(at, name, body[name], non_empty=True):
                if key not in keys:
                    raise _fail(at, f"'{name}' refers to unknown key '{key}'")
    if "expect_top5" in body and len(body["expect_top5"]) > 5:
        raise _fail(at, "'expect_top5' lists more than five keys")
    if "expect_status" in body:
        status = body["expect_status"]
        if (
            not isinstance(status, dict)
            or not status
            or not all(isinstance(v, str) for v in status.values())  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        ):
            raise _fail(at, "'expect_status' must map keys to status strings")
        for key in status:  # pyright: ignore[reportUnknownVariableType]
            if key not in keys:
                raise _fail(at, f"'expect_status' refers to unknown key '{key}'")
    if not {"expect_top5", "expect_absent", "expect_status"} & set(body):
        raise _fail(at, "a recall check needs expect_top5, expect_absent or expect_status")


def _validate_brief(at: str, body: dict[str, Any], keys: dict[str, str]) -> None:
    def section_map(name: str) -> None:
        value = body[name]
        if not isinstance(value, dict) or not value:
            raise _fail(at, f"'{name}' must map section names to lists of keys")
        for section, listed in value.items():  # pyright: ignore[reportUnknownVariableType]
            if section not in BRIEF_SECTIONS:
                raise _fail(at, f"'{name}': unknown section '{section}'")
            for key in _str_list(at, name, listed, non_empty=True):
                if key not in keys:
                    raise _fail(at, f"'{name}' refers to unknown key '{key}'")

    if "expect_in_section" in body:
        section_map("expect_in_section")
    if "expect_not_in" in body:
        value = body["expect_not_in"]
        if isinstance(value, dict):
            section_map("expect_not_in")
        else:
            for key in _str_list(at, "expect_not_in", value, non_empty=True):
                if key not in keys:
                    raise _fail(at, f"'expect_not_in' refers to unknown key '{key}'")
    if "max_tokens" in body:
        tokens = body["max_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise _fail(at, "'max_tokens' must be a positive integer")
    if "expect_pending" in body:
        pending = body["expect_pending"]
        if isinstance(pending, bool) or not isinstance(pending, int) or pending < 0:
            raise _fail(at, "'expect_pending' must be an integer of 0 or more")


# ---------------------------------------------------------------------------
# loading


def load_file(path: Path) -> Scenario:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{path.name}: not valid JSON ({exc})") from exc
    except OSError as exc:
        raise ScenarioError(f"{path.name}: cannot read file ({exc})") from exc
    scenario = parse_scenario(raw, source_name=path.name, path=path)
    if path.stem != scenario.id:
        raise ScenarioError(f"{path.name}: the file name must be the scenario id '{scenario.id}'")
    return scenario


def load_dir(directory: Path) -> list[Scenario]:
    """Every `*.json` under `directory` (sub-folders included), sorted by id."""
    scenarios = [load_file(p) for p in sorted(directory.rglob("*.json"))]
    seen: dict[str, Path | None] = {}
    for scenario in scenarios:
        if scenario.id in seen:
            raise ScenarioError(f"scenario id '{scenario.id}' is used by two files")
        seen[scenario.id] = scenario.path
    return sorted(scenarios, key=lambda s: s.id)


# ---------------------------------------------------------------------------
# lint: does a scenario really depend on what it says it depends on?


def _words(*texts: str) -> set[str]:
    return {w for text in texts for w in _WORD.findall(text.casefold())}


def lint_dependencies(scenario: Scenario) -> list[str]:
    """Problems with a scenario's `needs` claim, found without running anything.

    `needs = "keywords"`: the query must share no word (and no 3+ letter fragment)
    with the expected fact's text or entities, but must hit its keywords.
    `needs = "trigram"`: the query must share no whole word with the fact (text,
    entities, keywords) and no fragment with its keywords, but be a fragment of its text.
    """
    if scenario.needs is None:
        return []
    remembered = {
        s.data["key"]: s.data for s in scenario.steps if s.op == "remember" and "key" in s.data
    }
    problems: list[str] = []
    checked = 0
    for step in scenario.steps:
        if step.op != "recall" or "expect_top5" not in step.data or "query" not in step.data:
            continue
        query = str(step.data["query"])
        tokens = _WORD.findall(query.casefold())
        for key in step.data["expect_top5"]:
            fact = remembered.get(key)
            if fact is None:
                continue
            checked += 1
            text = str(fact["text"]).casefold()
            entities = " ".join(fact.get("entities", [])).casefold()
            keywords = " ".join(fact["keywords"]).casefold()
            body = f"{text} {entities}"
            body_words = _words(text, entities)
            kw_words = _words(keywords)
            for token in tokens:
                long = len(token) >= 3
                if scenario.needs == "keywords":
                    if token in body_words or (long and token in body):
                        problems.append(f"query word '{token}' already matches the text of {key}")
                elif token in body_words | kw_words or (long and token in keywords):
                    problems.append(f"query word '{token}' matches {key} without the trigram index")
            if scenario.needs == "keywords":
                if not any(t in kw_words or (len(t) >= 3 and t in keywords) for t in tokens):
                    problems.append(f"no query word reaches the keywords of {key}")
                if query.casefold().strip() in keywords:
                    problems.append(f"the whole query is stuffed into the keywords of {key}")
            elif not any(len(t) >= 3 and t in body for t in tokens):
                problems.append(f"no query word is a fragment of the text of {key}")
    if checked == 0:
        problems.append("needs is set but no recall step with a literal query expects a fact")
    return problems
