"""Optional fact extraction: a model reads captured episodes and proposes facts.

Off by default (`[llm] provider = "none"`), never part of the server, never inside
`remember`. See `ingatan extract`.
"""

from __future__ import annotations
