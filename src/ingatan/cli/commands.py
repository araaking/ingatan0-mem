"""The `ingatan` commands (typer).

Human commands act as the user. They open the context of the configured business and agent,
so they see `business` and `global` facts, every agent's proposals, and the log entries of
those tiers. Another agent's private (`agent` tier) entries stay hidden, exactly as for MCP.
To audit them, a command that touches the log takes `--agent NAME`: it opens the context
as that agent. The visibility filter itself is never weakened.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from ingatan import __version__, audit, facts, proposals, search
from ingatan import tokens as token_store
from ingatan.cli import doctor as doctor_checks
from ingatan.cli.conflicts import compute_report, open_conflicts, render_report, scan_conflicts
from ingatan.cli.render import log_line, render_diff, shorten
from ingatan.errors import ConfigError, IngatanError, NotFoundError, ValidationError
from ingatan.facts._util import fact_dict
from ingatan.scope import (
    CONFIG_NAME,
    Context,
    Scope,
    data_home,
    db_path,
    find_fact,
    load_embeddings,
    open_context,
    resolve_scope,
    validate_agent,
    validate_business,
)
from ingatan.search import local as local_models

F = TypeVar("F", bound=Callable[..., Any])

app = typer.Typer(
    name="ingatan",
    help="Local-first shared memory for AI agents. Without a command it starts the MCP server.",
    invoke_without_command=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=False,
)
proposals_app = typer.Typer(
    help="Review the proposals agents made for the business and global tiers.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(proposals_app, name="proposals")
token_app = typer.Typer(
    help="Manage the access tokens of HTTP mode. Each token is tied to one business and agent.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(token_app, name="token")
conflicts_app = typer.Typer(
    help="Inspect conflict resolution report, list open conflicts, or scan existing facts.",
    invoke_without_command=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(conflicts_app, name="conflicts")
alias_app = typer.Typer(
    help="Manage project-scoped glossary aliases for bidirectional recall expansion.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(alias_app, name="alias")
embeddings_app = typer.Typer(
    help="Set up the optional local embedding model for semantic search.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(embeddings_app, name="embeddings")

AgentOption = Annotated[
    str | None,
    typer.Option(
        "--agent",
        metavar="NAME",
        help="Open the context as this agent, to see its private (agent tier) entries.",
    ),
]
BusinessOption = Annotated[
    str | None,
    typer.Option(
        "--business",
        "-b",
        metavar="NAME",
        help="Target business name (overrides .ingatan.toml and INGATAN_BUSINESS).",
    ),
]

_ROOT_BUSINESS: str | None = None


def cli_command(func: F) -> F:
    """Turn an `IngatanError` into one line on stderr and exit code 1."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except IngatanError as exc:
            typer.echo(f"ingatan: {exc.message}", err=True)
            raise typer.Exit(1) from exc

    return wrapper  # type: ignore[return-value]


def _pick_business(business: str | None) -> str | None:
    return business if business is not None else _ROOT_BUSINESS


def _open(agent: str | None = None, business: str | None = None) -> Context:
    env = dict(os.environ)
    resolved_business = _pick_business(business)
    if resolved_business is not None:
        env["INGATAN_BUSINESS"] = validate_business(resolved_business)
    if agent is not None:
        env["INGATAN_AGENT"] = validate_agent(agent)
    return open_context(resolve_scope(env=env))


def _import(module: str, name: str) -> Any:
    """Import a collaborator lazily, so the CLI works while that module is unfinished."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise IngatanError(f"{module}.{name} is not available in this installation.") from exc


def _is_interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ingatan {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
    business: BusinessOption = None,
) -> None:
    global _ROOT_BUSINESS
    _ROOT_BUSINESS = business
    if context.invoked_subcommand is None:
        _serve()


@cli_command
def _serve() -> None:
    _import("ingatan.mcp.server", "run")()


@app.command()
@cli_command
def serve(
    http: Annotated[
        bool,
        typer.Option("--http", help="Serve MCP over HTTP (bearer tokens) instead of stdio."),
    ] = False,
    host: Annotated[
        str | None, typer.Option(help="Bind address with --http. Default 127.0.0.1.")
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(min=1, max=65535, help="Port with --http. Default 8765."),
    ] = None,
) -> None:
    """Start the MCP server: on stdio, or with --http on a port (see docs/setup-http.md)."""
    if not http:
        if host is not None or port is not None:
            raise ConfigError("--host and --port only apply together with --http.")
        _serve()
        return
    run_http = _import("ingatan.mcp.http", "run_http")
    kwargs: dict[str, Any] = {}
    if host is not None:
        kwargs["host"] = host
    if port is not None:
        kwargs["port"] = port
    run_http(**kwargs)


# ---- init ----------------------------------------------------------------------------


def _ask(label: str, flag: str | None, env_name: str, validate: Callable[[object], str]) -> str:
    value = (flag or os.environ.get(env_name) or "").strip()
    if not value:
        if not _is_interactive():
            raise ConfigError(
                f"init needs a {label} name. Pass --{label} NAME or set {env_name} "
                "(there is no terminal to ask on)."
            )
        value = typer.prompt(f"{label.capitalize()} name").strip()
    return validate(value)


@app.command()
@cli_command
def init(
    business: Annotated[str | None, typer.Option(help="Business name for this project.")] = None,
    agent: Annotated[str | None, typer.Option(help="Agent name for this project.")] = None,
) -> None:
    """Create .ingatan.toml here (if missing) and both database files. Safe to run twice."""
    config_path = Path.cwd() / CONFIG_NAME
    if config_path.is_file():
        typer.echo(f"{config_path} already exists, left unchanged.")
        if business or agent:
            typer.echo("The --business and --agent values were ignored.")
    else:
        business_name = _ask("business", business, "INGATAN_BUSINESS", validate_business)
        agent_name = _ask("agent", agent, "INGATAN_AGENT", validate_agent)
        try:
            with config_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(f'business = "{business_name}"\nagent = "{agent_name}"\n')
        except FileExistsError:
            typer.echo(f"{config_path} already exists, left unchanged.")
        else:
            typer.echo(f"Created {config_path}")
    scope = resolve_scope()
    existed = [db_path(scope.home, n).exists() for n in (scope.business, "global")]
    with open_context(scope):
        pass
    for name, was_there in zip((scope.business, "global"), existed, strict=True):
        path = db_path(scope.home, name)
        typer.echo(f"{'Database ready' if was_there else 'Created database'}: {path}")
    typer.echo("Next: run `ingatan doctor` to check the setup.")


# ---- doctor --------------------------------------------------------------------------


@app.command()
def doctor(
    business: BusinessOption = None,
    agent: AgentOption = None,
    probe_embeddings: Annotated[
        bool,
        typer.Option(
            "--probe-embeddings/--no-probe-embeddings",
            help="Test embedding one short sentence when embeddings are enabled.",
        ),
    ] = True,
) -> None:
    """Check Python, SQLite features, config, embeddings, and the databases."""
    checks = doctor_checks.run_checks(
        business=_pick_business(business),
        agent=agent,
        probe_embeddings=probe_embeddings,
    )
    typer.echo(doctor_checks.render(checks))
    if any(check.failed for check in checks):
        raise typer.Exit(1)


# ---- token ---------------------------------------------------------------------------


@token_app.command("add")
@cli_command
def token_add(
    business: Annotated[str, typer.Option(help="Business the token gives access to.")],
    agent: Annotated[str, typer.Option(help="Agent name the token acts as.")],
    label: Annotated[str, typer.Option(help="Free text to tell tokens apart.")] = "",
) -> None:
    """Create a token. It is shown once and only its hash is stored."""
    raw, record = token_store.add(data_home(), business, agent, label)
    typer.echo(f"Created token {record.id} for business {record.business}, agent {record.agent}.")
    typer.echo("Copy it now. It cannot be shown again:")
    typer.echo(raw)


@token_app.command("list")
@cli_command
def token_list() -> None:
    """List tokens: id, business, agent, created, label. Never the token itself."""
    records = token_store.list_tokens(data_home())
    if not records:
        typer.echo("No tokens. Create one with `ingatan token add`.")
    for r in records:
        typer.echo(f"{r.id}  {r.business}  {r.agent}  {r.created}  {r.label}".rstrip())


@token_app.command("revoke")
@cli_command
def token_revoke(
    token_id: Annotated[str, typer.Argument(help="Token id, from `token list`.")],
) -> None:
    """Remove a token. A running server stops accepting it on its next request."""
    record = token_store.revoke(data_home(), token_id)
    typer.echo(f"Revoked {record.id} (business {record.business}, agent {record.agent}).")


# ---- log, diff, undo -----------------------------------------------------------------


@app.command("log")
@cli_command
def log_command(
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="How many entries.")] = 20,
    fact: Annotated[str | None, typer.Option("--fact", help="Only this fact or proposal.")] = None,
    agent: AgentOption = None,
    business: BusinessOption = None,
) -> None:
    """List log entries, newest first."""
    with _open(agent, business) as ctx:
        if fact is not None and fact.startswith("f_"):
            found = list(reversed(facts.history(ctx, fact)))[:limit]
        elif fact is not None:
            found = audit.entries(ctx, fact_id=fact, limit=limit, newest_first=True)
            if not found:
                raise NotFoundError(f"No log entries for {fact}.")
        else:
            found = audit.entries(ctx, limit=limit, newest_first=True)
        if not found:
            typer.echo("The log is empty.")
        for entry in found:
            typer.echo(log_line(entry))


@app.command()
@cli_command
def diff(
    log_id: Annotated[int, typer.Argument(help="Log entry id, as shown by `ingatan log`.")],
    agent: AgentOption = None,
    business: BusinessOption = None,
) -> None:
    """Show the before and after of one log entry, field by field."""
    with _open(agent, business) as ctx:
        entry = audit.get(ctx, log_id)
        typer.echo(render_diff(audit.diff(ctx, log_id), entry))


@app.command()
@cli_command
def undo(
    log_id: Annotated[int, typer.Argument(help="Log entry id, as shown by `ingatan log`.")],
    agent: AgentOption = None,
    business: BusinessOption = None,
) -> None:
    """Reverse one log entry (create, supersede, confirm or approve)."""
    with _open(agent, business) as ctx:
        entry = audit.get(ctx, log_id)
        new_id = audit.undo(ctx, log_id)
    typer.echo(f"Undid entry {log_id} ({entry.op} of {entry.fact_id}). New log entry: {new_id}.")


# ---- proposals, forget ---------------------------------------------------------------


def _replaced_text(ctx: Context, fact_id: str) -> str | None:
    for conn in ctx.connections():
        row = conn.execute("SELECT text FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if row is not None:
            return str(row["text"])
    return None


@proposals_app.command("list")
@cli_command
def proposals_list(business: BusinessOption = None, agent: AgentOption = None) -> None:
    """List pending proposals from every agent of this business and of the global tier."""
    with _open(agent, business) as ctx:
        pending = proposals.pending(ctx)
        if not pending:
            typer.echo("No pending proposals.")
        for p in pending:
            typer.echo(f"{p.id}  [{p.tier}]  {p.agent}  {p.created_at[:10]}")
            typer.echo(f"    {p.text}")
            if p.supersedes:
                old = _replaced_text(ctx, p.supersedes)
                shown = f': "{shorten(old, 80)}"' if old is not None else " (no longer exists)"
                typer.echo(f"    would supersede {p.supersedes}{shown}")


@proposals_app.command("approve")
@cli_command
def proposals_approve(
    proposal_ids: Annotated[
        list[str] | None, typer.Argument(help="One or more proposal ids (omit with --all).")
    ] = None,
    approve_all_flag: Annotated[
        bool, typer.Option("--all", help="Approve every pending proposal in order.")
    ] = False,
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Approve one, several, or all pending proposals."""
    ids = list(proposal_ids or [])
    if approve_all_flag and ids:
        raise ValidationError("Pass either proposal ids or --all, not both.")
    if not approve_all_flag and not ids:
        raise ValidationError("Pass at least one proposal id, or use --all.")
    with _open(agent, business) as ctx:
        if approve_all_flag:
            pending_items = proposals.pending(ctx)
            if not pending_items:
                typer.echo("No pending proposals.")
                return
            total = len(pending_items)
            for idx, p in enumerate(pending_items, start=1):
                fact_id = proposals.approve(ctx, p.id)
                typer.echo(f"[{idx}/{total}] Approved {p.id}. The new fact is {fact_id}.")
            return
        for pid in ids:
            fact_id = proposals.approve(ctx, pid)
            typer.echo(f"Approved {pid}. The new fact is {fact_id}.")


@proposals_app.command("reject")
@cli_command
def proposals_reject(
    proposal_id: Annotated[str, typer.Argument(help="Proposal id.")],
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Reject a proposal: no fact is created."""
    with _open(agent, business) as ctx:
        proposals.reject(ctx, proposal_id)
    typer.echo(f"Rejected {proposal_id}. No fact was created.")


@app.command()
@cli_command
def forget(
    fact_id: Annotated[str, typer.Argument(help="Fact id.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    agent: AgentOption = None,
    business: BusinessOption = None,
) -> None:
    """Forget a fact of any tier for good, as the user. This cannot be undone."""
    with _open(agent, business) as ctx:
        found = find_fact(ctx, fact_id)
        if found is None:
            raise NotFoundError(f"No fact {fact_id}.")
        row = found[1]
        if not yes:
            typer.echo(f"{fact_id}  [{row['tier']}]  {row['agent']}")
            typer.echo(f"    {shorten(str(row['text']), 300)}")
            if not typer.confirm("Forget this fact for good? This cannot be undone."):
                typer.echo("Nothing was forgotten.")
                raise typer.Exit(1)
        facts.forget(ctx, fact_id, by_user=True)
    typer.echo(f"Forgotten {fact_id}. Its content is gone from the database and the log.")


# ---- ingest, import, move, conflicts, hook -------------------------------------------


def _count(value: object) -> object:
    return len(value) if isinstance(value, (list, tuple, set, frozenset)) else value  # pyright: ignore[reportUnknownArgumentType]


@app.command()
@cli_command
def ingest(
    path: Annotated[Path, typer.Argument(help="A Markdown file or a folder.")],
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Read Markdown documents into searchable notes."""
    ingest_path = _import("ingatan.ingest", "ingest_path")
    if not path.exists():
        raise NotFoundError(f"No such file or folder: {path}")
    with _open(agent, business) as ctx:
        report = ingest_path(ctx, path)
    typer.echo(
        f"Files: {_count(report.files)}  created: {_count(report.created)}  "
        f"replaced: {_count(report.replaced)}  skipped: {_count(report.skipped)}"
    )
    skipped = report.skipped
    if isinstance(skipped, (list, tuple)):
        for item in skipped:  # pyright: ignore[reportUnknownVariableType]
            typer.echo(f"  skipped: {item}")


def _load_import_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValidationError(f"{path} must contain a JSON array or JSONL lines.")
        return [item for item in data if isinstance(item, dict)]  # pyright: ignore[reportUnknownVariableType]
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except ValueError as exc:
            raise ValidationError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValidationError(f"{path}:{line_no} must be a JSON object.")
        records.append(obj)  # pyright: ignore[reportUnknownArgumentType]
    return records


@app.command("import")
@cli_command
def import_facts(
    path: Annotated[Path, typer.Argument(help="JSONL or JSON array file of facts to import.")],
    default_tier: Annotated[
        str, typer.Option("--tier", help="Default tier when a record omits `tier`.")
    ] = "business",
    scan: Annotated[
        bool,
        typer.Option(
            "--scan-conflicts", help="Scan for conflicts across imported facts at the end."
        ),
    ] = False,
    check_conflicts: Annotated[
        bool,
        typer.Option("--check-conflicts/--no-check-conflicts", help="Check conflicts per item."),
    ] = True,
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Bulk-import facts from a JSONL or JSON file, preserving original dates."""
    if not path.is_file():
        raise NotFoundError(f"No such file: {path}")
    records = _load_import_records(path)
    if not records:
        typer.echo("No records to import.")
        return
    with _open(agent, business) as ctx:
        imported = 0
        for idx, rec in enumerate(records, start=1):
            text = str(rec.get("text") or "").strip()
            if not text:
                raise ValidationError(f"Record #{idx} has empty `text`.")
            src = rec.get("source")
            source_dict: dict[str, Any] = (
                dict(src)
                if isinstance(src, dict)
                else ({"document": src} if isinstance(src, str) and src else {})
            )
            raw_date = rec.get("date") or source_dict.get("date")
            if isinstance(raw_date, str) and raw_date.strip():
                source_dict["date"] = raw_date.strip()
            valid_from = rec.get("valid_from") or (
                raw_date if isinstance(raw_date, str) and raw_date.strip() else None
            )
            facts.remember(
                ctx,
                text,
                keywords=rec.get("keywords"),
                entities=rec.get("entities"),
                tier=str(rec.get("tier") or default_tier),
                kind=rec.get("kind"),
                category=str(rec.get("category") or "fact"),
                pinned=bool(rec.get("pinned", False)),
                review_after=rec.get("review_after"),
                source=source_dict or None,
                valid_from=valid_from,
                check_conflicts=check_conflicts and not scan,
                related_to=rec.get("related_to"),
                dedup=bool(rec.get("dedup", True)),
            )
            imported += 1
        typer.echo(f"Imported {imported} fact(s) into {ctx.scope.business}.")
        if scan:
            res = scan_conflicts(ctx)
            typer.echo(
                f"Conflict scan: scanned {res.facts_scanned} fact(s), "
                f"recorded {res.pairs_recorded} new conflict pair(s)."
            )


@app.command()
@cli_command
def move(
    fact_id: Annotated[str, typer.Argument(help="Fact id to move.")],
    to_business: Annotated[str, typer.Option("--to", help="Destination business name.")],
    tier: Annotated[
        str | None, typer.Option("--tier", help="Destination tier (default: keep original tier).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Move a misplaced fact from one business database to another."""
    dest_name = validate_business(to_business)
    with _open(agent, business) as src_ctx:
        if dest_name == src_ctx.scope.business:
            raise ValidationError(f"Source and destination business are both '{dest_name}'.")
        found = find_fact(src_ctx, fact_id)
        if found is None:
            raise NotFoundError(f"No fact {fact_id}.")
        row = found[1]
        if row["tier"] == "global":
            raise ValidationError("Global facts are already shared across all businesses.")
        data = fact_dict(row)
        dest_tier = tier or str(data["tier"])
        if not yes:
            typer.echo(
                f"Move {fact_id} [{data['tier']}] from {src_ctx.scope.business} -> "
                f"{dest_name} [{dest_tier}]:"
            )
            typer.echo(f"    {shorten(str(data['text']), 300)}")
            if not typer.confirm("Copy to destination and forget from source?"):
                typer.echo("Nothing was moved.")
                raise typer.Exit(1)
        dest_scope = Scope(
            business=dest_name,
            agent=str(data["agent"]),
            home=src_ctx.scope.home,
            token_budget=src_ctx.scope.token_budget,
            require_approval=False,
            embeddings=src_ctx.scope.embeddings,
        )
        with open_context(dest_scope) as dst_ctx:
            saved = facts.remember(
                dst_ctx,
                str(data["text"]),
                keywords=data["keywords"],
                entities=data["entities"],
                tier=dest_tier,
                kind=str(data["kind"]),
                category=str(data.get("category") or "fact"),
                pinned=bool(data.get("pinned", False)),
                review_after=data["review_after"],
                source=data["source"],
                valid_from=str(data["valid_from"]),
                dedup=False,
            )
        facts.forget(src_ctx, fact_id, by_user=True)
    typer.echo(f"Moved {fact_id} from {src_ctx.scope.business} to {dest_name} as {saved.fact_id}.")


@conflicts_app.callback()
@cli_command
def conflicts_main(
    context: typer.Context,
    agent: AgentOption = None,
    business: BusinessOption = None,
) -> None:
    """Report how many conflict candidates were resolved (NFR-9), from the audit log."""
    if context.invoked_subcommand is not None:
        return
    with _open(agent, business) as ctx:
        report = compute_report(audit.entries(ctx))
    typer.echo(render_report(report))


@conflicts_app.command("list")
@cli_command
def conflicts_list(agent: AgentOption = None, business: BusinessOption = None) -> None:
    """List currently open conflict pairs between valid facts."""
    with _open(agent, business) as ctx:
        items = open_conflicts(ctx)
        if not items:
            typer.echo("No open conflicts.")
            return
        for item in items:
            typer.echo(f"#{item.id}  {item.fact_a} <-> {item.fact_b}  ({item.created_at[:10]})")
            typer.echo(f"    A [{item.fact_a}]: {shorten(item.text_a, 100)}")
            typer.echo(f"    B [{item.fact_b}]: {shorten(item.text_b, 100)}")


@conflicts_app.command("scan")
@cli_command
def conflicts_scan(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report matching pairs without writing them.")
    ] = False,
    agent: AgentOption = None,
    business: BusinessOption = None,
) -> None:
    """Scan existing valid facts for unflagged conflicts (e.g. after bulk import)."""
    with _open(agent, business) as ctx:
        res = scan_conflicts(ctx, dry_run=dry_run)
        if dry_run:
            typer.echo(
                f"Scanned {res.facts_scanned} fact(s): "
                f"found {res.pairs_found} candidate pair(s) (dry run)."
            )
        else:
            typer.echo(
                f"Scanned {res.facts_scanned} fact(s): found {res.pairs_found}, "
                f"recorded {res.pairs_recorded} new conflict pair(s)."
            )
        for a, b in res.pairs:
            typer.echo(f"  {a} <-> {b}")


# ---- reindex, alias, link, pin -------------------------------------------------------


@app.command()
@cli_command
def reindex(
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Recompute embeddings even if already indexed for this model."
        ),
    ] = False,
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Compute or refresh semantic embeddings for all valid facts and ingested notes."""
    with _open(agent, business) as ctx:
        provider = ctx.scope.embeddings.provider
        if provider == "none":
            typer.echo(
                "Embeddings are disabled (provider = 'none'). "
                "Enable [embeddings] in .ingatan.toml or ~/.ingatan/config.toml "
                "(or set INGATAN_EMBED_PROVIDER) first. For the offline local model, "
                "run `ingatan embeddings pull` once."
            )
            return
        report = search.reindex(ctx, force=force)
    typer.echo(
        f"Reindex ({provider} / {report.model}): "
        f"facts indexed={report.facts_indexed} skipped={report.facts_skipped}, "
        f"notes indexed={report.notes_indexed} skipped={report.notes_skipped}"
        + (f", errors={report.errors}" if report.errors else "")
    )
    if report.errors > 0:
        if report.last_error:
            typer.echo(f"ingatan: embedding error: {report.last_error}", err=True)
        raise typer.Exit(1)


@embeddings_app.command("pull")
@cli_command
def embeddings_pull(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            metavar="NAME",
            help="Local model to download. Default: the configured one, else the built-in default.",
        ),
    ] = None,
) -> None:
    """Download the local embedding model once and verify its checksums.

    This is the only Ingatan command that downloads anything. The server itself never
    does: without the model it simply answers from the word index.
    """
    cfg = load_embeddings()
    name = (model or (cfg.model if cfg.provider == "local" else "")).strip()
    spec = local_models.resolve(name)
    folder = local_models.model_folder(cfg.model_dir, spec)
    typer.echo(f"Model:  {spec.name} ({spec.summary})")
    typer.echo(f"Folder: {folder}")

    shown: dict[str, int] = {}

    def progress(file: local_models.ModelFile, received: int) -> None:
        tenth = received * 10 // max(1, file.size)
        if shown.get(file.name) != tenth:
            shown[file.name] = tenth
            typer.echo(f"  {file.name}: {received / 1e6:.0f} of {file.size / 1e6:.0f} MB", err=True)

    fetched = local_models.pull(spec.name, cfg.model_dir, progress=progress)
    if fetched:
        size = sum(path.stat().st_size for path in fetched) / 1e6
        typer.echo(f"Downloaded {len(fetched)} file(s), {size:.0f} MB. Checksums verified.")
    else:
        typer.echo("Already present and verified. Nothing was downloaded.")
    missing = local_models.missing_packages()
    if missing:
        typer.echo(
            f"Still needed: the packages {', '.join(missing)}. "
            f"Install them in the environment Ingatan runs in: {local_models.INSTALL_HINT}"
        )
    configured_model = cfg.model.strip().casefold() or local_models.DEFAULT_LOCAL_MODEL
    if cfg.provider != "local" or configured_model != spec.name:
        typer.echo(
            "To use it, set this in .ingatan.toml or in config.toml of the data folder:\n"
            "  [embeddings]\n"
            '  provider = "local"\n'
            f'  model = "{spec.name}"'
        )
    typer.echo("Then run `ingatan reindex` to embed the facts that already exist.")


@alias_app.command("add")
@cli_command
def alias_add(
    term: Annotated[str, typer.Argument(help="Short alias, acronym, or local term (e.g. 'LP').")],
    canonical: Annotated[str, typer.Argument(help="Canonical phrase (e.g. 'landing page').")],
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Register a project-scoped alias pair for bidirectional recall expansion."""
    with _open(agent, business) as ctx:
        t, c = search.add_alias(ctx, term, canonical)
        typer.echo(f"Saved alias in {ctx.scope.business}: {t} <-> {c}")


@alias_app.command("list")
@cli_command
def alias_list(business: BusinessOption = None, agent: AgentOption = None) -> None:
    """List all glossary aliases in the configured business."""
    with _open(agent, business) as ctx:
        items = search.list_aliases(ctx)
        if not items:
            typer.echo("No aliases configured.")
            return
        for term, canonical in items:
            typer.echo(f"{term}  <->  {canonical}")


@alias_app.command("remove")
@cli_command
def alias_remove(
    term: Annotated[str, typer.Argument(help="Alias term to remove.")],
    canonical: Annotated[
        str | None, typer.Argument(help="Optional specific canonical phrase to remove.")
    ] = None,
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Remove one or all alias mappings for a term."""
    with _open(agent, business) as ctx:
        removed = search.remove_alias(ctx, term, canonical)
    if removed == 0:
        typer.echo(f"No matching alias for '{term}'.")
    else:
        typer.echo(f"Removed {removed} alias mapping(s) for '{term}'.")


@app.command()
@cli_command
def link(
    src_id: Annotated[str, typer.Argument(help="Source fact id.")],
    dst_id: Annotated[str, typer.Argument(help="Target fact id.")],
    relation: Annotated[
        str, typer.Option("--relation", "-r", help="Edge relation: related_to or part_of.")
    ] = "related_to",
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Link two complementary facts (also resolves any open conflict between them)."""
    with _open(agent, business) as ctx:
        created = facts.link_facts(ctx, src_id, dst_id, relation=relation)
    if created:
        typer.echo(f"Linked {src_id} -[{relation}]-> {dst_id}.")
    else:
        typer.echo(f"Link already exists: {src_id} -[{relation}]-> {dst_id}.")


@app.command()
@cli_command
def unlink(
    src_id: Annotated[str, typer.Argument(help="Source fact id.")],
    dst_id: Annotated[str, typer.Argument(help="Target fact id.")],
    relation: Annotated[
        str | None, typer.Option("--relation", "-r", help="Only remove this relation type.")
    ] = None,
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Remove relation edges between two facts."""
    with _open(agent, business) as ctx:
        removed = facts.unlink_facts(ctx, src_id, dst_id, relation=relation)
    if removed == 0:
        typer.echo(f"No edge found between {src_id} and {dst_id}.")
    else:
        typer.echo(f"Removed {removed} edge(s) between {src_id} and {dst_id}.")


@app.command()
@cli_command
def pin(
    fact_id: Annotated[str, typer.Argument(help="Fact id to pin.")],
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Pin a fact so it stays prioritized in `brief` and boosted in `recall`."""
    with _open(agent, business) as ctx:
        facts.set_pinned(ctx, fact_id, True)
    typer.echo(f"Pinned {fact_id}.")


@app.command()
@cli_command
def unpin(
    fact_id: Annotated[str, typer.Argument(help="Fact id to unpin.")],
    business: BusinessOption = None,
    agent: AgentOption = None,
) -> None:
    """Remove the pin from a fact."""
    with _open(agent, business) as ctx:
        facts.set_pinned(ctx, fact_id, False)
    typer.echo(f"Unpinned {fact_id}.")


def _log_hook_failure(event: str, exc: BaseException) -> None:
    """One line in `<data home>/capture-errors.log`, if that is possible at all."""
    try:
        from ingatan.redact import redact
        from ingatan.scope import data_home
        from ingatan.timeutil import iso, now

        message = exc.message if isinstance(exc, IngatanError) else f"{type(exc).__name__}: {exc}"
        line = redact(f"hook {event}: {message}").replace("\r", " ").replace("\n", " ")
        folder = data_home()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "capture-errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{iso(now())}\t{line}\n")
    except Exception:  # the hook must stay silent whatever happens
        pass


@app.command()
def hook(event: Annotated[str, typer.Argument(help="Claude Code hook event name.")]) -> None:
    """Claude Code hook entry point. Reads the hook JSON on stdin. Always exits 0.

    A capture hook must never disturb the agent session (NFR-14): when the hook module is
    missing or fails, one line goes to `capture-errors.log` and nothing to stdout or stderr.
    """
    try:
        run = _import("ingatan.integrations.claude_code.hooks", "run")
        run(event)
    except (Exception, SystemExit) as exc:
        _log_hook_failure(event, exc)
    raise typer.Exit(0)
