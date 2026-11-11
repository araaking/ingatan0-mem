"""Facts: remember, supersede, confirm, forget and history."""

from __future__ import annotations

from ingatan.facts.change import ConfirmResult, SupersedeResult, confirm, supersede
from ingatan.facts.create import RememberResult, create_fact, remember
from ingatan.facts.forget import forget
from ingatan.facts.history import history

__all__ = [
    "ConfirmResult",
    "RememberResult",
    "SupersedeResult",
    "confirm",
    "create_fact",
    "forget",
    "history",
    "remember",
    "supersede",
]
