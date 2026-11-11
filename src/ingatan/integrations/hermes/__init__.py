"""Ingatan memory provider for Hermes Agent (T-20).

Copy this folder to `$HERMES_HOME/plugins/ingatan/`. It captures each completed turn as an
episode (FR-45) and returns the brief from `system_prompt_block()` (FR-48). It adds no
tools: the seven tools come from the Ingatan MCP server (`mcp_servers.ingatan`).

Written from docs/HOOK_FINDINGS.md section A, not tested against a real Hermes. Every method
takes `**kwargs` because the interface grows, and nothing here raises into Hermes (NFR-14).
Scope: `INGATAN_BUSINESS` and `INGATAN_AGENT` from the environment of the Hermes process
(agent defaults to "hermes"); data lives in `INGATAN_HOME` / `~/.ingatan`, the same place the
MCP server uses.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from typing import Any


def _hermes_base() -> Any:
    """Hermes' `MemoryProvider` when running inside Hermes, else a plain `object`."""
    try:
        return importlib.import_module("agent.memory_provider").MemoryProvider
    except Exception:
        return object


_Base: Any = _hermes_base()

__all__ = ["DEFAULT_AGENT", "IngatanProvider", "register"]

DEFAULT_AGENT = "hermes"


class IngatanProvider(_Base):
    def __init__(self) -> None:
        self._session_id = ""
        self._primary = True
        self._resumed = False
        self._block: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        return "ingatan"

    def _scope(self) -> Any:
        """The scope from the environment, or None. Imported lazily; never raises."""
        try:
            from ingatan.scope import resolve_scope

            env = dict(os.environ)
            if not env.get("INGATAN_AGENT", "").strip():
                env["INGATAN_AGENT"] = DEFAULT_AGENT
            return resolve_scope(None, env)
        except Exception:
            return None

    def is_available(self, **kwargs: Any) -> bool:
        """No network and no database access: only checks that a business name is set."""
        return self._scope() is not None

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        try:
            self._session_id = str(session_id or "")
            context = kwargs.get("agent_context")
            self._primary = context in (None, "", "primary")
            self._resumed = bool(kwargs.get("resumed"))
            self._block = None
        except Exception:
            pass

    def get_tool_schemas(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def system_prompt_block(self, **kwargs: Any) -> str:
        """The brief. Built once per session: building it bumps the digest-offer counters."""
        try:
            with self._lock:
                if self._block is None:
                    self._block = self._build_brief()
                return self._block
        except Exception:
            return ""

    def _build_brief(self) -> str:
        scope = self._scope()
        if scope is None:
            return ""
        try:
            from ingatan import brief
            from ingatan.scope import open_context

            ctx = open_context(scope)
            try:
                # Subagents, cron runs and resumed sessions get the facts but no digest offer.
                offer = self._primary and not self._resumed
                return brief.build(ctx, offer_episodes=offer) or ""
            finally:
                ctx.close()
        except Exception as exc:
            self._log(scope, exc, "system_prompt_block")
            return ""

    def prefetch(self, query: str = "", **kwargs: Any) -> str:
        return ""

    def queue_prefetch(self, query: str = "", **kwargs: Any) -> None:
        return None

    def on_turn_start(self, turn_number: int = 0, message: str = "", **kwargs: Any) -> None:
        return None

    def on_session_switch(self, new_session_id: str = "", **kwargs: Any) -> None:
        self._session_id = str(new_session_id or "")
        self._block = None

    def on_pre_compress(self, messages: Any = None, **kwargs: Any) -> str:
        return ""

    def on_memory_write(self, *args: Any, **kwargs: Any) -> None:
        return None

    def handle_tool_call(self, tool_name: str = "", args: Any = None, **kwargs: Any) -> str:
        return json.dumps(
            {"error": "The ingatan provider has no tools. Use the mcp_ingatan_* tools."}
        )

    def on_session_end(self, messages: Any = None, **kwargs: Any) -> None:
        # Hermes discards the result and cannot hand work back (HOOK_FINDINGS A4). Digesting
        # happens at the next session start, through the brief.
        return None

    def shutdown(self, **kwargs: Any) -> None:
        return None

    # ------------------------------------------------------------------ capture

    def sync_turn(
        self,
        user_content: str = "",
        assistant_content: str = "",
        *,
        session_id: str = "",
        messages: Any = None,
        turn_author: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record one turn. Hermes calls this on its own background worker, so the write is
        done inline. Only the two strings are stored: `messages` (which may hold tool output)
        is never read."""
        try:
            if not self._primary:
                return
            user = user_content if isinstance(user_content, str) else ""
            reply = assistant_content if isinstance(assistant_content, str) else ""
            if not user.strip() and not reply.strip():
                return
            scope = self._scope()
            if scope is None:
                return
            from ingatan import episodes

            episodes.capture(scope, user, reply, session=session_id or self._session_id or None)
        except Exception as exc:
            self._log(None, exc, "sync_turn")

    @staticmethod
    def _log(scope: Any, exc: BaseException, where: str) -> None:
        try:
            from ingatan import timeutil
            from ingatan.redact import redact
            from ingatan.scope import data_home

            home = scope.home if scope is not None else data_home()
            home.mkdir(parents=True, exist_ok=True)
            message = redact(f"hermes {where}: {type(exc).__name__}: {exc}").replace("\n", " ")
            with (home / "capture-errors.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{timeutil.iso(timeutil.now())}\t{message}\n")
        except Exception:
            pass


def register(ctx: Any) -> None:
    """Hermes entry point. A new instance per call, not a singleton."""
    ctx.register_memory_provider(IngatanProvider())
