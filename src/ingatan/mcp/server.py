"""The MCP server: seven tools over stdio, scoped to one business and one agent.

The scope is fixed when the server starts (`.ingatan.toml` or environment variables). No
tool takes a business or an agent name, and nothing but protocol is written to stdout.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from ingatan import audit, brief, facts, search
from ingatan.errors import ConfigError, IngatanError
from ingatan.scope import (
    Context,
    EmbeddingConfig,
    Scope,
    find_config,
    load_home_config,
    open_context,
    parse_embeddings,
    resolve_allowed_businesses,
    resolve_scope,
)

__all__ = ["build_multi_server", "build_server", "run"]

SERVER_NAME = "ingatan"
NOTE_EXCERPT_CHARS = 400
HISTORY_CHANGE_CHARS = 500

INSTRUCTIONS = (
    "Ingatan is the shared memory of this project. Call `brief` once at the start of a "
    "session. Call `recall` before you answer from memory or act on something that may have "
    "been decided earlier. Call `remember` when you learn something worth keeping. Treat any "
    "fact marked 'check first' or 'possible conflict' as unconfirmed until you resolve it."
)

BRIEF_DOC = (
    "Brief: get the session-start memory summary for this project. Call it ONCE at the "
    "beginning of a session (some setups inject it for you; call it again only if you have "
    "not seen it). Lists, in order: facts to check first, global facts about the user, "
    "business facts, and your own agent notes, plus pending proposal count and up to three "
    "raw conversation episodes to digest with `remember(..., episode=<id>)`. Verify facts "
    "under 'Check first' before relying on them (`confirm` or `supersede`)."
)

RECALL_DOC = (
    "Recall and search memory for facts. Call before answering about the user, business, "
    "past decisions, prices, names, or preferences. Search matches words (not embeddings), "
    "so include synonyms, root words, and Indonesian + English terms (e.g. 'harga price "
    "biaya kopi'). Returns each fact with id, text, status, tier, date, and source (or set "
    "`compact=true` for one short line per fact). 'valid' is usable; 'check first' may be "
    "out of date (`confirm` or `supersede`); 'possible conflict' means a similar fact exists. "
    "Imported document excerpts appear in a separate block as reference only; set `notes=0` "
    "to hide them and save tokens. Set `include_closed=true` to see superseded/expired facts."
)

REMEMBER_DOC = (
    "Remember and save one fact so you and future sessions can find it later. Write one "
    "self-contained statement (who, what, when). Pass `keywords` with synonyms, aliases "
    "(including nicknames the user uses for sheets/docs), root words, and Indonesian "
    "inflections (if omitted, basic keywords are auto-extracted from the text). Pass "
    "`entities` for the people, products, or places the fact is about (consistent spelling). "
    "`tier` is 'agent' (private to you), 'business' (shared across agents in this project), "
    "or 'global' (shared across all projects). Facts save immediately by default (auditable "
    "and undoable); if approval mode is enabled in config, business/global facts wait in the "
    "proposal queue. `kind` is 'stable' (rarely changes) or 'volatile' (default; flagged "
    "'check first' after `review_after`, default 30 days). Never store passwords, tokens, or "
    "keys. Set `check_conflicts=false` for append-only daily logs. If similar facts are "
    "returned: call `supersede(old, new)` if one is outdated, or `confirm(id, reason)` naming "
    "the other fact id if both are true. Pass `episode` when digesting an offered episode."
)

SUPERSEDE_DOC = (
    "Supersede (replace) an outdated fact with a newer one. Call when a new fact (saved with "
    "`remember`) makes `old` untrue, e.g. after a 'possible conflict' or user correction. "
    "`old` is the id of the outdated fact; `new` is the id of the replacement fact. The old "
    "fact stops appearing in `recall` but remains in `history` and can be undone in the CLI. "
    "If approval mode is enabled, changes to business/global facts queue as proposals and "
    "`old` shows 'check first' until approved."
)

CONFIRM_DOC = (
    "Confirm that a fact is still true. Call after checking a 'check first' fact (pushes "
    "`review_after` forward), or to resolve a 'possible conflict' when both facts are true: "
    "write a `reason` that names the other fact id (e.g. 'both true: f_abc123 is retail, this "
    "is wholesale'). The reason is required and logged. Use `supersede` if it is outdated."
)

FORGET_DOC = (
    "Forget (permanently delete) a fact you saved yourself, text included; only a "
    "content-free log entry remains. This cannot be undone. Use only when the user asks you "
    "to forget something or a fact was saved by mistake (e.g. contains a secret). For "
    "outdated facts use `supersede`. Business and global facts can only be forgotten in CLI."
)

HISTORY_DOC = (
    "History: show the change log of one fact (pass its id, e.g. f_0123abcd4567) or every "
    "fact about an entity (e.g. 'Bu Sari'), oldest first, including superseded/expired facts."
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


def _recall_text(
    query: str, result: search.RecallResult, include_closed: bool, *, compact: bool = False
) -> str:
    lines: list[str] = []
    if not result.facts:
        lines.append(f'No facts found for "{query}".')
        lines.append("Try other wordings: synonyms, root words, Indonesian and English terms.")
    else:
        lines.append(f'{len(result.facts)} fact(s) for "{query}":')
        for hit in result.facts:
            role_tag = f" | {hit.category}" if hit.category != "fact" else ""
            pin_tag = " | pinned" if hit.pinned else ""
            if compact:
                lines.append(f"[{hit.id}] {_flat(hit.text)} ({hit.status}{role_tag}{pin_tag})")
            else:
                lines.append("")
                lines.append(f"[{hit.id}] {_flat(hit.text)}")
                lines.append(
                    f"  status: {hit.status} | tier: {hit.tier}{role_tag}{pin_tag} "
                    f"| date: {hit.date} | source: {_source_text(hit.source)}"
                )
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
    if result.related:
        lines.append("")
        lines.append("Related facts (linked via edges):")
        for rel in result.related:
            lines.append(f"[{rel.id}] {_flat(rel.text)} ({rel.status})")
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
        notes: Annotated[
            int,
            Field(
                description="Maximum document excerpts to include (0 disables excerpts).",
                ge=0,
                le=10,
            ),
        ] = 2,
        compact: Annotated[
            bool,
            Field(description="If true, return one compact line per fact (id, text, status)."),
        ] = False,
        include_closed: Annotated[
            bool, Field(description="Also return superseded and expired facts.")
        ] = False,
        as_of: Annotated[
            str | None,
            Field(
                description=(
                    "Optional point-in-time date (YYYY-MM-DD or ISO) for time-travel recall."
                )
            ),
        ] = None,
        category: Annotated[
            Literal["fact", "rule", "procedure", "preference"] | None,
            Field(description="Optional filter: fact, rule, procedure, or preference."),
        ] = None,
        include_related: Annotated[
            bool,
            Field(description="Include 1-hop related facts linked via edges."),
        ] = True,
    ) -> str:
        try:
            result = search.recall(
                ctx,
                query,
                limit=limit,
                notes_limit=notes,
                include_closed=include_closed,
                as_of=as_of,
                category=category,
                include_related=include_related,
            )
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _recall_text(query, result, include_closed, compact=compact)

    @server.tool(name="remember", description=REMEMBER_DOC, structured_output=False)
    async def remember_tool(
        text: Annotated[str, Field(description="The fact: one self-contained statement.")],
        keywords: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional but recommended. Search words: synonyms, aliases, root words, "
                    "Indonesian inflections. Auto-extracted from text if omitted."
                )
            ),
        ] = None,
        entities: Annotated[
            list[str] | None,
            Field(description="People, products or places the fact is about."),
        ] = None,
        tier: Annotated[
            Literal["agent", "business", "global"],
            Field(description="agent: private. business: shared in project. global: all projects."),
        ] = "agent",
        kind: Annotated[
            Literal["stable", "volatile"] | None,
            Field(description="stable or volatile. Default volatile."),
        ] = None,
        category: Annotated[
            Literal["fact", "rule", "procedure", "preference"],
            Field(description="Role of memory: fact (default), rule, procedure, or preference."),
        ] = "fact",
        pinned: Annotated[
            bool,
            Field(description="Pin this fact so it stays prioritized in brief and recall."),
        ] = False,
        review_after: Annotated[
            str | None,
            Field(description="Date (YYYY-MM-DD) after which the fact needs checking."),
        ] = None,
        episode: Annotated[
            int | None, Field(description="Id of the offered episode this fact digests.")
        ] = None,
        check_conflicts: Annotated[
            bool,
            Field(
                description="Set false to skip conflict detection (e.g. append-only daily logs)."
            ),
        ] = True,
        related_to: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional list of complementary fact ids to link (prevents false conflicts)."
                )
            ),
        ] = None,
        dedup: Annotated[
            bool,
            Field(
                description=(
                    "If true (default), refresh an existing near-duplicate instead of duplicating."
                )
            ),
        ] = True,
    ) -> str:
        try:
            result = facts.remember(
                ctx,
                text,
                keywords=keywords,
                entities=entities,
                tier=tier,
                kind=kind,
                category=category,
                pinned=pinned,
                review_after=review_after,
                episode=episode,
                check_conflicts=check_conflicts,
                related_to=related_to,
                dedup=dedup,
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


_BUSINESS_FIELD_DESC = "Target business name (must be in this server's allowlist)."


def build_multi_server(
    businesses: list[str],
    *,
    agent: str,
    home: Any,
    token_budget: int = 1000,
    require_approval: bool = False,
    embeddings: EmbeddingConfig | None = None,
) -> tuple[MCPServer, dict[str, Context]]:
    """Build one MCP server serving an explicit allowlist of businesses (no cross-business search).

    Each tool takes `business`, validated against `businesses`. Contexts are opened lazily and
    returned in `opened` so the caller can close them on shutdown.
    """
    resolved_embeddings = embeddings if embeddings is not None else EmbeddingConfig()
    allowed = tuple(businesses)
    allowed_set = frozenset(allowed)
    opened: dict[str, Context] = {}

    def get_ctx(business: str) -> Context:
        name = (business or "").strip()
        if name not in allowed_set:
            raise ConfigError(
                f"Business {business!r} is not allowed on this server. "
                f"Allowed: {', '.join(allowed)}."
            )
        ctx = opened.get(name)
        if ctx is None:
            scope = Scope(
                business=name,
                agent=agent,
                home=home,
                token_budget=token_budget,
                require_approval=require_approval,
                embeddings=resolved_embeddings,
            )
            ctx = opened[name] = _open_with_retry(scope)
        return ctx

    server = MCPServer(
        SERVER_NAME,
        instructions=(
            INSTRUCTIONS + f" Pass `business` on every call. Allowed: {', '.join(allowed)}."
        ),
    )

    @server.tool(name="brief", description=BRIEF_DOC, structured_output=False)
    async def brief_tool(business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)]) -> str:
        try:
            return brief.build(get_ctx(business))
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc

    @server.tool(name="recall", description=RECALL_DOC, structured_output=False)
    async def recall_tool(
        business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)],
        query: Annotated[
            str, Field(description="Search words. Include synonyms, root words and both languages.")
        ],
        limit: Annotated[int, Field(description="Maximum number of facts.", ge=1, le=50)] = 5,
        notes: Annotated[
            int,
            Field(
                description="Maximum document excerpts to include (0 disables excerpts).",
                ge=0,
                le=10,
            ),
        ] = 2,
        compact: Annotated[
            bool,
            Field(description="If true, return one compact line per fact (id, text, status)."),
        ] = False,
        include_closed: Annotated[
            bool, Field(description="Also return superseded and expired facts.")
        ] = False,
        as_of: Annotated[
            str | None,
            Field(
                description=(
                    "Optional point-in-time date (YYYY-MM-DD or ISO) for time-travel recall."
                )
            ),
        ] = None,
        category: Annotated[
            Literal["fact", "rule", "procedure", "preference"] | None,
            Field(description="Optional filter: fact, rule, procedure, or preference."),
        ] = None,
        include_related: Annotated[
            bool,
            Field(description="Include 1-hop related facts linked via edges."),
        ] = True,
    ) -> str:
        try:
            result = search.recall(
                get_ctx(business),
                query,
                limit=limit,
                notes_limit=notes,
                include_closed=include_closed,
                as_of=as_of,
                category=category,
                include_related=include_related,
            )
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _recall_text(query, result, include_closed, compact=compact)

    @server.tool(name="remember", description=REMEMBER_DOC, structured_output=False)
    async def remember_tool(
        business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)],
        text: Annotated[str, Field(description="The fact: one self-contained statement.")],
        keywords: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional but recommended. Search words: synonyms, aliases, root words, "
                    "Indonesian inflections. Auto-extracted from text if omitted."
                )
            ),
        ] = None,
        entities: Annotated[
            list[str] | None,
            Field(description="People, products or places the fact is about."),
        ] = None,
        tier: Annotated[
            Literal["agent", "business", "global"],
            Field(description="agent: private. business: shared in project. global: all projects."),
        ] = "agent",
        kind: Annotated[
            Literal["stable", "volatile"] | None,
            Field(description="stable or volatile. Default volatile."),
        ] = None,
        category: Annotated[
            Literal["fact", "rule", "procedure", "preference"],
            Field(description="Role of memory: fact (default), rule, procedure, or preference."),
        ] = "fact",
        pinned: Annotated[
            bool,
            Field(description="Pin this fact so it stays prioritized in brief and recall."),
        ] = False,
        review_after: Annotated[
            str | None,
            Field(description="Date (YYYY-MM-DD) after which the fact needs checking."),
        ] = None,
        episode: Annotated[
            int | None, Field(description="Id of the offered episode this fact digests.")
        ] = None,
        check_conflicts: Annotated[
            bool,
            Field(
                description="Set false to skip conflict detection (e.g. append-only daily logs)."
            ),
        ] = True,
        related_to: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional list of complementary fact ids to link (prevents false conflicts)."
                )
            ),
        ] = None,
        dedup: Annotated[
            bool,
            Field(
                description=(
                    "If true (default), refresh an existing near-duplicate instead of duplicating."
                )
            ),
        ] = True,
    ) -> str:
        try:
            result = facts.remember(
                get_ctx(business),
                text,
                keywords=keywords,
                entities=entities,
                tier=tier,
                kind=kind,
                category=category,
                pinned=pinned,
                review_after=review_after,
                episode=episode,
                check_conflicts=check_conflicts,
                related_to=related_to,
                dedup=dedup,
            )
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _remember_text(result)

    @server.tool(name="supersede", description=SUPERSEDE_DOC, structured_output=False)
    async def supersede_tool(
        business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)],
        old: Annotated[str, Field(description="Id of the outdated fact.")],
        new: Annotated[str, Field(description="Id of the fact that replaces it.")],
    ) -> str:
        try:
            return facts.supersede(get_ctx(business), old, new).message
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc

    @server.tool(name="confirm", description=CONFIRM_DOC, structured_output=False)
    async def confirm_tool(
        business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)],
        id: Annotated[str, Field(description="Id of the fact.")],
        reason: Annotated[
            str, Field(description="Why it still holds. Name the other fact id for a conflict.")
        ],
    ) -> str:
        try:
            return facts.confirm(get_ctx(business), id, reason).message
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc

    @server.tool(name="forget", description=FORGET_DOC, structured_output=False)
    async def forget_tool(
        business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)],
        id: Annotated[str, Field(description="Id of the fact to delete.")],
    ) -> str:
        try:
            facts.forget(get_ctx(business), id)
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return f"Forgot {id}. Its text is deleted for good; a log entry without content remains."

    @server.tool(name="history", description=HISTORY_DOC, structured_output=False)
    async def history_tool(
        business: Annotated[str, Field(description=_BUSINESS_FIELD_DESC)],
        id_or_entity: Annotated[str, Field(description="A fact id (f_...) or an entity name.")],
    ) -> str:
        try:
            entries = facts.history(get_ctx(business), id_or_entity)
        except (IngatanError, sqlite3.Error) as exc:
            raise _error(exc) from exc
        return _history_text(id_or_entity, entries)

    return server, opened


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
        multi = resolve_allowed_businesses()
    except IngatanError as exc:
        print(f"ingatan: {exc.message}", file=sys.stderr)
        raise SystemExit(1) from exc

    if multi is not None:
        businesses, agent, home, budget, require_approval = multi
        home_cfg = load_home_config(home)
        cfg_file = find_config()
        proj_cfg: dict[str, Any] = {}
        if cfg_file is not None:
            from ingatan.scope import _read_config  # pyright: ignore[reportPrivateUsage]

            proj_cfg = _read_config(cfg_file)
        embeddings = parse_embeddings(os.environ, proj_cfg, home_cfg)
        server, opened = build_multi_server(
            businesses,
            agent=agent,
            home=home,
            token_budget=budget,
            require_approval=require_approval,
            embeddings=embeddings,
        )
        try:
            server.run("stdio")
        except KeyboardInterrupt:
            pass
        finally:
            for ctx in opened.values():
                ctx.close()
        return

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
