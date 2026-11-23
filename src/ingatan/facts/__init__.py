"""Facts: remember, supersede, confirm, forget and history."""

from __future__ import annotations

from ingatan.facts.change import ConfirmResult, SupersedeResult, confirm, supersede
from ingatan.facts.create import RememberResult, create_fact, remember
from ingatan.facts.forget import forget
from ingatan.facts.history import history
from ingatan.facts.links import RELATIONS, link_facts, related_ids, set_pinned, unlink_facts

__all__ = [
    "RELATIONS",
    "ConfirmResult",
    "RememberResult",
    "SupersedeResult",
    "confirm",
    "create_fact",
    "forget",
    "history",
    "link_facts",
    "related_ids",
    "remember",
    "set_pinned",
    "supersede",
    "unlink_facts",
]
