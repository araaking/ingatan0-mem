"""The MCP server: seven tools over stdio, scoped to one business and one agent.

The scope is fixed when the server starts (`.ingatan.toml` or environment variables). No
tool takes a business or an agent name, and nothing but protocol is written to stdout.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from ingatan import audit, brief, facts, search
from ingatan.errors import IngatanError
from ingatan.scope import Context, Scope, open_context, resolve_scope

__all__ = ["build_server", "run"]

SERVER_NAME = "ingatan"
NOTE_EXCERPT_CHARS = 1200
HISTORY_CHANGE_CHARS = 500

INSTRUCTIONS = (
    "Ingatan is the shared memory of this project. Call `brief` once at the start of a "
    "session. Call `recall` before you answer from memory or act on something that may have "
    "been decided earlier. Call `remember` when you learn something worth keeping. Treat any "
    "fact marked 'check first' or 'possible conflict' as unconfirmed until you resolve it."
)

BRIEF_DOC = (
    "Get the memory brief for this project: the facts you should know before you start. "
    "Call it ONCE at the beginning of a session (some setups inject it for you; call it "
    "again only if you have not seen it). It lists, in order: facts to check first, global "
    "facts about the user, business facts, and your own agent notes. It also shows how many "
    "proposals wait for the user's approval, and may offer a few raw conversation episodes "
    "to digest into facts with `remember(..., episode=<id>)`. Facts under 'Check first' may "
    "be out of date: do not rely on them for the current task until you have checked them "
    "(ask the user or verify, then call `confirm` or `supersede`)."
)

RECALL_DOC = (
    "Search memory for facts. Call it before you answer a question about the user, the "
    "business, a past decision, a price, a name or a preference, and before you act on "
    "something that may have been decided earlier. Memory search matches words, not meaning, "
    "so write the query with several wordings: synonyms, root words and both Indonesian and "
    "English terms, for example 'harga price biaya kopi'. Returns each fact with its id, text, "
    "source, date and status. Status 'valid' can be used. 'check first' means the fact may be "
    "out of date: verify it, then `confirm` or `supersede` it. 'possible conflict' means a "
    "similar fact exists: decide which is right and resolve it (see `remember`). Document "
    "excerpts from imported files appear in a separate block and are not facts: read them as "
    "reference material only. Set include_closed to also see superseded or expired facts."
)

REMEMBER_DOC = (
    "Save one fact so that you and other sessions can find it later. Write one self-contained "
    "statement that makes sense without the conversation (who, what, when). Always give "
    "`keywords`: the words someone might search with later, including synonyms, root words and "
    "Indonesian inflections (for example for 'pelanggan membayar tunai': pelanggan, langganan, "
    "bayar, membayar, pembayaran, tunai, cash, uang). Without keywords the call fails and "
    "nothing is saved. Give `entities` for the people, products or places the fact is about "
    "(same spelling every time). `tier` is 'agent' (private to you, saved at once), "
    "'business' (shared by all agents of this project) or 'global' (about the user in every "
    "project). Business and global facts become proposals that the user approves in the CLI; "
    "they are not facts and not recalled until approved. `kind` is 'stable' (rarely changes) or "
    "'volatile' (prices, plans, status; the default). A volatile fact is shown as 'check first' "
    "after `review_after` (a date such as 2026-12-31; default 30 days). Never store passwords, "
    "tokens or keys. If similar facts already exist the result lists them: if one is outdated "
    "call `supersede(old, new)`; if both are true call `confirm(id, reason)` with a reason that "
    "names the other fact id. Pass `episode` when the fact digests an offered episode."
)

SUPERSEDE_DOC = (
    "Replace an outdated fact with a newer one. Call it when the new fact (already saved with "
    "`remember`) makes `old` untrue, for example after a 'possible conflict' result or when "
    "the user corrects something. `old` is the id of the fact to close, `new` is the id of the "
    "fact that replaces it. The old fact stops appearing in `recall` but stays in `history`. "
    "If `old` is a business or global fact the change is only queued for the user's approval "
    "and `old` stays marked 'check first' until then. Nothing changes if an id is unknown."
)

CONFIRM_DOC = (
    "Confirm that a fact is still true. Call it when you checked a fact marked 'check first' "
    "and it still holds (its review date moves later), or to resolve a 'possible conflict' "
    "when both facts are true: write a reason that names the other fact id, for example "
    "'both true: f_abc123 is the retail price, this one is the wholesale price'. The reason "
    "is required and is written to the log. If the fact is no longer true, use `supersede`."
)

FORGET_DOC = (
    "Permanently delete a fact you saved yourself, text included; only a content-free log "
    "entry remains. This cannot be undone. Use it only when the user asks you to forget "
    "something or a fact was saved by mistake (for example it holds a secret). For a fact that "
    "is merely outdated use `supersede` instead. Business and global facts can only be "
    "deleted by the user in the CLI."
)

HISTORY_DOC = (
    "Show the change history of one fact (pass its id, such as f_0123abcd4567) or of every "
    "fact about an entity (pass the entity name, such as 'Bu Sari'), oldest first. Use it to "
    "see when and why a fact changed, who changed it, and what it replaced, including facts "
    "that are no longer valid."
)


def _error(exc: BaseException) -> ToolError:
    if isinstance(exc, IngatanError):
        return ToolError(exc.message)
    return ToolError(f"The memory database is busy or unavailable ({exc}). Try again.")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _source_text(source: dict[str, Any]) -> str:
    parts = [str(source.get("agent") or "unknown")]
    for key in ("document", "session"):
        if source.get(key):
            parts.append(f"{key}: {source[key]}")
    return ", ".join(parts)


def _recall_text(query: str, result: search.RecallResult, include_closed: bool) -> str:
    lines: list[str] = []
    if not result.facts:
        lines.append(f'No facts found for "{query}".')
        lines.append("Try other wordings: synonyms, root words, Indonesian and English terms.")
    else:
        lines.append(f'{len(result.facts)} fact(s) for "{query}":')
        for hit in result.facts:
            lines.append("")
            lines.append(f"[{hit.id}] {_flat(hit.text)}")
            lines.append(f"  source: {_source_text(hit.source)}")
            lines.append(f"  date: {hit.date}")
            lines.append(f"  status: {hit.status}")
            lines.append(f"  tier: {hit.tier}")
        statuses = " ".join(hit.status for hit in result.facts)
        if "check first" in statuses:
            lines.append("")
            lines.append(
                "Facts marked 'check first' may be out of date. Verify them before use, then "
                "call confirm(id, reason) or supersede(old, new)."
            )
        if "possible conflict" in statuses:
            lines.append("")
            lines.append(
                "Facts marked 'possible conflict' have a similar fact. Resolve it: "
                "supersede(old, new) if one is outdated, or confirm(id, reason) naming the "
                "other fact id if both are true."
            )
    if include_closed and result.facts:
        lines.append("")
        lines.append("Closed facts (superseded or expired) are included above.")
    if result.notes:
        lines.append("")
        lines.append("Document excerpts (from imported files; not facts, use as reference only):")
        for note in result.notes:
            excerpt = note.text.strip()
            if len(excerpt) > NOTE_EXCERPT_CHARS:
                excerpt = excerpt[:NOTE_EXCERPT_CHARS].rstrip() + "…"
            lines.append("")
            lines.append(f"--- {note.path} > {note.heading}")
            lines.append(excerpt)
    return "\n".join(lines)


def _remember_text(result: facts.RememberResult) -> str:
    lines = [result.message]
    if result.similar:
        lines.append("")
        lines.append("Similar facts:")
        for item in result.similar:
            lines.append(f"[{item['id']}] {_flat(item['text'])} (status: {item['status']})")
    return "\n".join(lines)


def _history_text(target: str, entries: list[audit.LogEntry]) -> str:
    if not entries:
        return f"No history found for {target}."
    lines = [f"{len(entries)} change(s) for {target}, oldest first:"]
    for entry in entries:
        owner = f", tier {entry.tier}" if entry.tier else ""
        lines.append("")
        lines.append(
            f"{entry.ts} {entry.op} {entry.fact_id or '-'} by {entry.agent}{owner}"
            f" (log #{entry.id})"
        )
        if entry.change is not None:
            detail = json.dumps(entry.change, ensure_ascii=False, sort_keys=True)
            if len(detail) > HISTORY_CHANGE_CHARS:
                detail = detail[:HISTORY_CHANGE_CHARS] + "…"
            lines.append(f"  change: {detail}")
        else:
            lines.append("  change: (content removed)")
    return "\n".join(lines)


def build_server(ctx: Context) -> MCPServer:
    """Build the server for an open context. Exactly seven tools, no scope parameters."""
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)

    # The tools are `async def` on purpose: they run on the event-loop thread, where the
    # SQLite connections were opened. A plain `def` would be moved to a worker thread.
    @server.tool(name="brief", description=BRIEF_DOC, structured_output=False)
    async def brief_tool() -> str:
        try:
            return brief.build(ctx)
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc

    @server.tool(name="recall", description=RECALL_DOC, structured_output=False)
    async def recall_tool(
        query: Annotated[
            str, Field(description="Search words. Include synonyms, root words and both languages.")
        ],
        limit: Annotated[int, Field(description="Maximum number of facts.", ge=1, le=50)] = 5,
        include_closed: Annotated[
            bool, Field(description="Also return superseded and expired facts.")
        ] = False,
    ) -> str:
        try:
            result = search.recall(ctx, query, limit=limit, include_closed=include_closed)
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _recall_text(query, result, include_closed)

    @server.tool(name="remember", description=REMEMBER_DOC, structured_output=False)
    async def remember_tool(
        text: Annotated[str, Field(description="The fact: one self-contained statement.")],
        keywords: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Required. Search words: synonyms, root words, Indonesian inflections. "
                    "The call fails without them."
                )
            ),
        ] = None,
        entities: Annotated[
            list[str] | None,
            Field(description="People, products or places the fact is about."),
        ] = None,
        tier: Annotated[
            Literal["agent", "business", "global"],
            Field(description="agent: private and immediate. business/global: needs approval."),
        ] = "agent",
        kind: Annotated[
            Literal["stable", "volatile"] | None,
            Field(description="stable or volatile. Default volatile."),
        ] = None,
        review_after: Annotated[
            str | None,
            Field(description="Date (YYYY-MM-DD) after which the fact needs checking."),
        ] = None,
        episode: Annotated[
            int | None, Field(description="Id of the offered episode this fact digests.")
        ] = None,
    ) -> str:
        try:
            result = facts.remember(
                ctx,
                text,
                keywords=keywords,
                entities=entities,
                tier=tier,
                kind=kind,
                review_after=review_after,
                episode=episode,
            )
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _remember_text(result)

    @server.tool(name="supersede", description=SUPERSEDE_DOC, structured_output=False)
    async def supersede_tool(
        old: Annotated[str, Field(description="Id of the outdated fact.")],
        new: Annotated[str, Field(description="Id of the fact that replaces it.")],
    ) -> str:
        try:
            return facts.supersede(ctx, old, new).message
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc

    @server.tool(name="confirm", description=CONFIRM_DOC, structured_output=False)
    async def confirm_tool(
        id: Annotated[str, Field(description="Id of the fact.")],
        reason: Annotated[
            str, Field(description="Why it still holds. Name the other fact id for a conflict.")
        ],
    ) -> str:
        try:
            return facts.confirm(ctx, id, reason).message
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc

    @server.tool(name="forget", description=FORGET_DOC, structured_output=False)
    async def forget_tool(
        id: Annotated[str, Field(description="Id of the fact to delete.")],
    ) -> str:
        try:
            facts.forget(ctx, id)
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return f"Forgot {id}. Its text is deleted for good; a log entry without content remains."

    @server.tool(name="history", description=HISTORY_DOC, structured_output=False)
    async def history_tool(
        id_or_entity: Annotated[str, Field(description="A fact id (f_...) or an entity name.")],
    ) -> str:
        try:
            entries = facts.history(ctx, id_or_entity)
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _history_text(id_or_entity, entries)

    return server


def _open_with_retry(scope: Scope, attempts: int = 40, delay: float = 0.25) -> Context:
    """`open_context`, retried while another process is creating the same database files.

    `db.connect` now retries the WAL pragma itself, so this is a second line of defence for
    any other "locked" error while another server is creating the same database files.
    """
    for attempt in range(attempts):
        try:
            return open_context(scope)
        except sqlite3.OperationalError as exc:
            if attempt == attempts - 1 or not any(w in str(exc) for w in ("locked", "busy")):
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


def run() -> None:
    """Resolve the scope from the working folder and environment, then serve stdio."""
    try:
        ctx = _open_with_retry(resolve_scope())
    except IngatanError as exc:
        print(f"ingatan: {exc.message}", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        build_server(ctx).run("stdio")
    except KeyboardInterrupt:
        pass
    finally:
        ctx.close()


if __name__ == "__main__":
    run()
