"""The interface the harness drives the memory through, and the in-process driver.

`MemoryDriver` is deliberately small and speaks only plain values (ids, text, status
strings). `InProcessDriver` calls the real modules directly. A future driver can send the
same calls through the MCP server instead; the runner and the metrics do not change.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ingatan import facts, proposals, search, timeutil
from ingatan.scope import Context, Scope, open_context

from .scenarios import Actor

__all__ = [
    "BriefResult",
    "Hit",
    "InProcessDriver",
    "Inspection",
    "MemoryDriver",
    "Remembered",
]


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    status: str
    tier: str


@dataclass(frozen=True)
class Remembered:
    fact_id: str | None
    proposal_id: str | None


@dataclass(frozen=True)
class BriefResult:
    """`text` is None when the brief module is not available (not a failure of the memory)."""

    text: str | None
    tokens: int | None


@dataclass(frozen=True)
class Inspection:
    """The harness's own view of a returned item, taken from the stored data (the oracle)."""

    found: bool
    leaked: bool
    closed: bool
    overdue: bool
    reason: str = ""


class MemoryDriver(Protocol):
    def remember(
        self,
        actor: Actor,
        text: str,
        *,
        keywords: list[str],
        entities: list[str] | None,
        tier: str,
        kind: str | None,
        review_after: str | None,
    ) -> Remembered: ...

    def supersede(self, actor: Actor, old_id: str, new_id: str) -> str | None:
        """Returns the proposal id when the change was queued instead of applied."""
        ...

    def confirm(self, actor: Actor, fact_id: str, reason: str) -> None: ...

    def forget(self, actor: Actor, fact_id: str, *, by_user: bool) -> None: ...

    def approve(self, actor: Actor, proposal_id: str) -> str:
        """The user approves a proposal. Returns the new fact id."""
        ...

    def reject(self, actor: Actor, proposal_id: str) -> None: ...

    def recall(
        self, actor: Actor, query: str, *, limit: int, include_closed: bool
    ) -> list[Hit]: ...

    def brief(self, actor: Actor) -> BriefResult: ...

    def inspect(self, actor: Actor, item_id: str, text: str) -> Inspection: ...

    def pending_texts(self) -> list[str]:
        """Text of every pending proposal anywhere in the data folder (oracle)."""
        ...

    def close(self) -> None: ...


class InProcessDriver:
    """Calls the real modules, one open `Context` per actor, all in one data folder."""

    def __init__(self, home: Path, *, token_budget: int = 1000) -> None:
        self.home = home
        self.token_budget = token_budget
        self._contexts: dict[Actor, Context] = {}

    # -- plumbing ---------------------------------------------------------

    def ctx(self, actor: Actor) -> Context:
        found = self._contexts.get(actor)
        if found is None:
            scope = Scope(
                business=actor.business,
                agent=actor.agent,
                home=self.home,
                token_budget=self.token_budget,
            )
            found = self._contexts[actor] = open_context(scope)
        return found

    def close(self) -> None:
        for ctx in self._contexts.values():
            ctx.close()
        self._contexts.clear()

    # -- the memory calls ---------------------------------------------------

    def remember(
        self,
        actor: Actor,
        text: str,
        *,
        keywords: list[str],
        entities: list[str] | None,
        tier: str,
        kind: str | None,
        review_after: str | None,
    ) -> Remembered:
        result = facts.remember(
            self.ctx(actor),
            text,
            keywords=keywords,
            entities=entities,
            tier=tier,
            kind=kind,
            review_after=review_after,
        )
        return Remembered(result.fact_id, result.proposal_id)

    def supersede(self, actor: Actor, old_id: str, new_id: str) -> str | None:
        result = facts.supersede(self.ctx(actor), old_id, new_id)
        return result.proposal_id

    def confirm(self, actor: Actor, fact_id: str, reason: str) -> None:
        facts.confirm(self.ctx(actor), fact_id, reason)

    def forget(self, actor: Actor, fact_id: str, *, by_user: bool) -> None:
        facts.forget(self.ctx(actor), fact_id, by_user=by_user)

    def approve(self, actor: Actor, proposal_id: str) -> str:
        return proposals.approve(self.ctx(actor), proposal_id)

    def reject(self, actor: Actor, proposal_id: str) -> None:
        proposals.reject(self.ctx(actor), proposal_id)

    def recall(self, actor: Actor, query: str, *, limit: int, include_closed: bool) -> list[Hit]:
        result = search.recall(self.ctx(actor), query, limit=limit, include_closed=include_closed)
        return [Hit(h.id, h.text, h.status, h.tier) for h in result.facts]

    def brief(self, actor: Actor) -> BriefResult:
        try:
            module = importlib.import_module("ingatan.brief")
        except ImportError:
            return BriefResult(None, None)
        build = getattr(module, "build", None)
        estimate = getattr(module, "estimate_tokens", None)
        if build is None or estimate is None:
            return BriefResult(None, None)
        text = str(build(self.ctx(actor)))
        return BriefResult(text, int(estimate(text)))

    # -- the oracle ---------------------------------------------------------

    def _databases(self) -> list[tuple[str, sqlite3.Connection]]:
        opened: list[tuple[str, sqlite3.Connection]] = []
        for path in sorted(self.home.glob("*.db")):
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            opened.append((path.stem, conn))
        return opened

    def inspect(self, actor: Actor, item_id: str, text: str) -> Inspection:
        """Judge a returned item from the raw data, not from what the memory says about it."""
        now = timeutil.iso(timeutil.now())
        databases = self._databases()
        try:
            for name, conn in databases:
                row = conn.execute(
                    "SELECT tier, agent, superseded_by, valid_to, kind, review_after "
                    "FROM facts WHERE id = ?",
                    (item_id,),
                ).fetchone()
                if row is None:
                    continue
                tier = str(row["tier"])
                if tier == "global":
                    visible = name == "global"
                elif tier == "business":
                    visible = name == actor.business
                else:
                    visible = name == actor.business and row["agent"] == actor.agent
                closed = row["superseded_by"] is not None or (
                    row["valid_to"] is not None and str(row["valid_to"]) <= now
                )
                overdue = (
                    row["kind"] == "volatile"
                    and row["review_after"] is not None
                    and str(row["review_after"]) < now
                )
                reason = ""
                if not visible:
                    owner = f"{name}/{row['agent']}" if tier == "agent" else name
                    reason = f"{tier}-tier fact of {owner}"
                return Inspection(True, not visible, closed, overdue, reason)
            # Not a stored fact. A pending proposal's text, or something unknown, is a leak.
            if text in self.pending_texts():
                return Inspection(False, True, False, False, "pending proposal")
            return Inspection(False, True, False, False, "id not found in any database")
        finally:
            for _, conn in databases:
                conn.close()

    def pending_texts(self) -> list[str]:
        found: list[str] = []
        for _, conn in self._databases():
            try:
                for row in conn.execute("SELECT payload FROM proposals WHERE status = 'pending'"):
                    found.append(str(json.loads(row["payload"]).get("text", "")))
            finally:
                conn.close()
        return found
