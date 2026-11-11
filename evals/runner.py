"""Run scenarios against a `MemoryDriver` and collect results and metric records."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ingatan import timeutil

from .briefs import parse_brief
from .driver import InProcessDriver, MemoryDriver
from .metrics import CheckRecord
from .netguard import NetworkGuard
from .scenarios import Actor, Scenario, Step

__all__ = [
    "FROZEN_START",
    "DriverFactory",
    "RunResult",
    "ScenarioResult",
    "default_driver_factory",
    "run_all",
    "run_scenario",
]

FROZEN_START = datetime(2026, 11, 11, 9, 0, 0, tzinfo=timezone.utc)

DriverFactory = Callable[[Path, Scenario], MemoryDriver]

PASS, FAIL, XFAIL, XPASS, SKIP = "PASS", "FAIL", "XFAIL", "XPASS", "SKIP"


def default_driver_factory(home: Path, scenario: Scenario) -> MemoryDriver:
    return InProcessDriver(home, token_budget=scenario.token_budget)


@dataclass
class ScenarioResult:
    scenario: Scenario
    status: str
    failures: list[str] = field(default_factory=lambda: [])
    notes: list[str] = field(default_factory=lambda: [])
    records: list[CheckRecord] = field(default_factory=lambda: [])
    log: list[str] = field(default_factory=lambda: [])


@dataclass
class RunResult:
    results: list[ScenarioResult]
    network_attempts: list[str]

    @property
    def records(self) -> list[CheckRecord]:
        return [r for result in self.results for r in result.records]

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


@dataclass
class _Ref:
    fact_id: str | None
    proposal_id: str | None
    text: str
    actor: Actor


class _Stop(Exception):
    """A step failed in a way that makes the rest of the scenario meaningless."""


def _short(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


class _Run:
    def __init__(self, scenario: Scenario, driver: MemoryDriver) -> None:
        self.scenario = scenario
        self.driver = driver
        self.refs: dict[str, _Ref] = {}
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.records: list[CheckRecord] = []
        self.log: list[str] = []
        self.skipped = False

    # -- helpers --------------------------------------------------------

    def fail(self, step: Step, message: str) -> None:
        self.failures.append(f"step {step.index} ({step.op}): {message}")

    def actor(self, step: Step) -> Actor:
        return self.scenario.actors[step.actor or self.scenario.default_actor]

    def fact_id(self, key: str) -> str:
        ref = self.refs[key]
        if ref.fact_id is None:
            raise _Stop(f"'{key}' is still a pending proposal, it has no fact id")
        return ref.fact_id

    def name_of(self, fact_id: str) -> str:
        for key, ref in self.refs.items():
            if ref.fact_id == fact_id:
                return key
        return fact_id

    def describe(self, hits: list[Any]) -> str:
        return ", ".join(f"{self.name_of(h.id)}[{h.status}]" for h in hits) or "nothing"

    # -- steps ------------------------------------------------------------

    def run(self, step: Step) -> None:
        handler = getattr(self, f"_op_{step.op}")
        if step.expect_error is not None:
            try:
                handler(step)
            except _Stop:
                raise
            except Exception as exc:
                names = [c.__name__ for c in type(exc).__mro__]
                if step.expect_error not in names:
                    self.fail(
                        step,
                        f"expected {step.expect_error}, got {type(exc).__name__}: {exc}",
                    )
                else:
                    self.log.append(f"  -> {type(exc).__name__}: {_short(str(exc), 80)} (expected)")
            else:
                self.fail(step, f"expected {step.expect_error}, but the call succeeded")
            return
        handler(step)

    def _op_remember(self, step: Step) -> None:
        d = step.data
        review_after = None
        if d.get("review_in_days") is not None:
            review_after = timeutil.iso(timeutil.now() + timedelta(days=d["review_in_days"]))
        actor = self.actor(step)
        result = self.driver.remember(
            actor,
            d["text"],
            keywords=list(d["keywords"]),
            entities=list(d["entities"]) if "entities" in d else None,
            tier=d.get("tier", "agent"),
            kind=d.get("kind"),
            review_after=review_after,
        )
        if "key" in d:
            self.refs[d["key"]] = _Ref(result.fact_id, result.proposal_id, d["text"], actor)
        self.log.append(f"  remember {d.get('key', '')} -> {result.fact_id or result.proposal_id}")

    def _op_supersede(self, step: Step) -> None:
        d = step.data
        old = self.fact_id(d["old"])
        new_ref = self.refs[d["new"]]
        new_id = new_ref.fact_id or new_ref.proposal_id
        assert new_id is not None
        proposal_id = self.driver.supersede(self.actor(step), old, new_id)
        if "as" in d and proposal_id is not None:
            self.refs[d["as"]] = _Ref(None, proposal_id, new_ref.text, self.actor(step))
        elif "as" in d:
            raise _Stop(f"supersede did not queue a proposal, so '{d['as']}' has no id")

    def _op_confirm(self, step: Step) -> None:
        d = step.data

        def sub(match: re.Match[str]) -> str:
            ref = self.refs[match.group(1)]
            return ref.fact_id or ref.proposal_id or ""

        reason = re.sub(r"\{(\w+)\}", sub, d["reason"])
        self.driver.confirm(self.actor(step), self.fact_id(d["key"]), reason)

    def _op_forget(self, step: Step) -> None:
        d = step.data
        self.driver.forget(
            self.actor(step), self.fact_id(d["key"]), by_user=bool(d.get("by_user", False))
        )

    def _proposal(self, step: Step) -> tuple[Actor, _Ref]:
        ref = self.refs[step.data["key"]]
        if ref.proposal_id is None:
            raise _Stop(f"'{step.data['key']}' is not a proposal")
        return (self.scenario.actors[step.actor] if step.actor else ref.actor), ref

    def _op_approve(self, step: Step) -> None:
        actor, ref = self._proposal(step)
        assert ref.proposal_id is not None
        ref.fact_id = self.driver.approve(actor, ref.proposal_id)

    def _op_reject(self, step: Step) -> None:
        actor, ref = self._proposal(step)
        assert ref.proposal_id is not None
        self.driver.reject(actor, ref.proposal_id)

    def _op_advance_days(self, step: Step) -> None:
        timeutil.set_now(timeutil.now() + timedelta(days=step.data["days"]))
        self.log.append(f"  clock is now {timeutil.iso(timeutil.now())}")

    # -- checks -----------------------------------------------------------

    def _op_recall(self, step: Step) -> None:
        d = step.data
        actor = self.actor(step)
        query = d["query"] if "query" in d else self.refs[d["query_text_of"]].text
        hits = self.driver.recall(
            actor,
            query,
            limit=int(d.get("limit", 10)),
            include_closed=bool(d.get("include_closed")),
        )
        top5 = hits[:5]
        top5_ids = {h.id for h in top5}
        problems: list[str] = []
        top5_ok: bool | None = None

        if "expect_top5" in d:
            missing = [k for k in d["expect_top5"] if self.refs[k].fact_id not in top5_ids]
            top5_ok = not missing
            if missing:
                problems.append(
                    f"expected in top 5: {', '.join(missing)}; top 5 was {self.describe(top5)}"
                )
        for key in d.get("expect_absent", []):
            ref = self.refs[key]
            if any(h.id == ref.fact_id or h.text == ref.text for h in hits):
                problems.append(f"{key} must not be returned, but it was ({self.describe(hits)})")
        for key, wanted in d.get("expect_status", {}).items():
            fid = self.refs[key].fact_id
            hit = next((h for h in hits if h.id == fid), None)
            if hit is None:
                problems.append(f"{key} should have status '{wanted}' but was not returned")
            elif hit.status != wanted:
                problems.append(f"{key} has status '{hit.status}', expected '{wanted}'")

        leaked = 0
        stale = False
        for hit in hits:
            seen = self.driver.inspect(actor, hit.id, hit.text)
            if seen.leaked:
                leaked += 1
                problems.append(
                    f"LEAK: {hit.id} ({seen.reason}) was returned to {actor.business}/{actor.agent}"
                )
            if seen.closed and hit.status not in ("superseded", "expired"):
                stale = True
                problems.append(
                    f"STALE: closed fact {self.name_of(hit.id)} returned as '{hit.status}'"
                )
            if seen.overdue and not seen.closed and "check first" not in hit.status:
                stale = True
                problems.append(
                    f"STALE: {self.name_of(hit.id)} is past review but has status '{hit.status}'"
                )
        for p in problems:
            self.fail(step, f"query {query!r}: {p}")
        self.log.append(f"  recall {query!r} as {actor.agent} -> {self.describe(hits)}")
        self.records.append(
            CheckRecord(
                scenario=self.scenario.id,
                kind="recall",
                label=query,
                top5_ok=top5_ok,
                stale=stale,
                items=len(hits),
                leaked=leaked,
            )
        )

    def _op_brief(self, step: Step) -> None:
        d = step.data
        actor = self.actor(step)
        result = self.driver.brief(actor)
        if result.text is None or result.tokens is None:
            self.skipped = True
            note = "brief not available (ingatan.brief.build is missing): brief check skipped"
            if note not in self.notes:
                self.notes.append(note)
            return
        text = result.text
        parsed = parse_brief(text)
        problems: list[str] = []

        leaked = 0
        items = 0
        clean_texts: list[str] = []
        for line in parsed.lines:
            items += 1
            seen = self.driver.inspect(actor, line.id, line.text)
            if seen.leaked:
                leaked += 1
                problems.append(f"LEAK: brief line {line.id} ({seen.reason})")
            else:
                clean_texts.append(line.text)
        for pending in self.driver.pending_texts():
            if pending and pending in text and not any(pending in t for t in clean_texts):
                items += 1
                leaked += 1
                problems.append(
                    f"LEAK: a pending proposal appears in the brief: {_short(pending)!r}"
                )

        budget = self.scenario.token_budget
        if result.tokens > budget:
            problems.append(f"the brief is {result.tokens} tokens, over the budget of {budget}")
        if "max_tokens" in d and result.tokens > d["max_tokens"]:
            problems.append(
                f"the brief is {result.tokens} tokens, expected at most {d['max_tokens']}"
            )
        if "expect_pending" in d and parsed.pending != d["expect_pending"]:
            problems.append(
                f"the brief says {parsed.pending} pending proposals, expected {d['expect_pending']}"
            )
        for section, keys in d.get("expect_in_section", {}).items():
            ids = parsed.ids_in(section)
            for key in keys:
                if self.refs[key].fact_id not in ids:
                    problems.append(
                        f"{key} should be in the '{section}' section (it has: {self._named(ids)})"
                    )
        absent = d.get("expect_not_in")
        if isinstance(absent, dict):
            for section, keys in absent.items():
                ids = parsed.ids_in(section)
                for key in keys:
                    if self.refs[key].fact_id in ids:
                        problems.append(f"{key} must not be in the '{section}' section")
        elif absent:
            for key in absent:
                ref = self.refs[key]
                if (ref.fact_id and ref.fact_id in text) or ref.text in text:
                    problems.append(f"{key} must not appear in the brief, but it does")
        for p in problems:
            self.fail(step, f"brief for {actor.business}/{actor.agent}: {p}")
        self.log.append(
            f"  brief as {actor.agent}: {result.tokens} tokens, {len(parsed.lines)} fact lines, "
            f"{parsed.pending} pending"
        )
        self.records.append(
            CheckRecord(
                scenario=self.scenario.id,
                kind="brief",
                label=f"{actor.business}/{actor.agent}",
                items=items,
                leaked=leaked,
                tokens=result.tokens,
                budget=budget,
            )
        )

    def _named(self, ids: list[str]) -> str:
        return ", ".join(self.name_of(i) for i in ids) or "nothing"


def run_scenario(
    scenario: Scenario, home: Path, driver_factory: DriverFactory = default_driver_factory
) -> ScenarioResult:
    """Run one scenario in `home` with the clock frozen at `FROZEN_START`."""
    timeutil.set_now(FROZEN_START)
    driver = driver_factory(home, scenario)
    run = _Run(scenario, driver)
    try:
        for step in scenario.steps:
            try:
                run.run(step)
            except _Stop as stop:
                run.fail(step, str(stop))
                break
            except Exception as exc:  # a failed scenario must not end the whole run
                run.fail(step, f"unexpected {type(exc).__name__}: {exc}")
                break
    finally:
        driver.close()
        timeutil.set_now(None)

    if run.failures:
        status = XFAIL if scenario.xfail else FAIL
    elif run.skipped:
        status = SKIP
    else:
        status = XPASS if scenario.xfail else PASS
    return ScenarioResult(scenario, status, run.failures, run.notes, run.records, run.log)


def run_all(
    scenarios: list[Scenario],
    driver_factory: DriverFactory = default_driver_factory,
    *,
    on_result: Callable[[ScenarioResult], None] | None = None,
) -> RunResult:
    """Run every scenario, each in its own data folder under one temporary INGATAN_HOME.

    The whole run happens under a `NetworkGuard`; its attempts are returned in the result.
    """
    results: list[ScenarioResult] = []
    previous = os.environ.get("INGATAN_HOME")
    with (
        tempfile.TemporaryDirectory(prefix="ingatan-evals-", ignore_cleanup_errors=True) as tmp,
        NetworkGuard() as guard,
    ):
        os.environ["INGATAN_HOME"] = tmp
        try:
            for scenario in scenarios:
                result = run_scenario(scenario, Path(tmp) / scenario.id, driver_factory)
                results.append(result)
                if on_result is not None:
                    on_result(result)
        finally:
            timeutil.set_now(None)
            if previous is None:
                os.environ.pop("INGATAN_HOME", None)
            else:
                os.environ["INGATAN_HOME"] = previous
    return RunResult(results, list(guard.attempts))
