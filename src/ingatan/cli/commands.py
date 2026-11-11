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
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from ingatan import __version__, audit, facts, proposals
from ingatan import tokens as token_store
from ingatan.cli import doctor as doctor_checks
from ingatan.cli.conflicts import compute_report, render_report
from ingatan.cli.render import log_line, render_diff, shorten
from ingatan.errors import ConfigError, IngatanError, NotFoundError
from ingatan.scope import (
    CONFIG_NAME,
    Context,
    data_home,
    db_path,
    find_fact,
    open_context,
    resolve_scope,
    validate_agent,
    validate_business,
)

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

AgentOption = Annotated[
    str | None,
    typer.Option(
        "--agent",
        metavar="NAME",
        help="Open the context as this agent, to see its private (agent tier) entries.",
    ),
]


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


def _open(agent: str | None = None) -> Context:
    env = dict(os.environ)
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
) -> None:
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
def doctor() -> None:
    """Check Python, SQLite features, the resolved config and the databases."""
    checks = doctor_checks.run_checks()
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
) -> None:
    """List log entries, newest first."""
    with _open(agent) as ctx:
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
) -> None:
    """Show the before and after of one log entry, field by field."""
    with _open(agent) as ctx:
        entry = audit.get(ctx, log_id)
        typer.echo(render_diff(audit.diff(ctx, log_id), entry))


@app.command()
@cli_command
def undo(
    log_id: Annotated[int, typer.Argument(help="Log entry id, as shown by `ingatan log`.")],
    agent: AgentOption = None,
) -> None:
    """Reverse one log entry (create, supersede, confirm or approve)."""
    with _open(agent) as ctx:
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
def proposals_list() -> None:
    """List pending proposals from every agent of this business and of the global tier."""
    with _open() as ctx:
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
def proposals_approve(proposal_id: Annotated[str, typer.Argument(help="Proposal id.")]) -> None:
    """Approve a proposal: the fact becomes valid in its tier."""
    with _open() as ctx:
        fact_id = proposals.approve(ctx, proposal_id)
    typer.echo(f"Approved {proposal_id}. The new fact is {fact_id}.")


@proposals_app.command("reject")
@cli_command
def proposals_reject(proposal_id: Annotated[str, typer.Argument(help="Proposal id.")]) -> None:
    """Reject a proposal: no fact is created."""
    with _open() as ctx:
        proposals.reject(ctx, proposal_id)
    typer.echo(f"Rejected {proposal_id}. No fact was created.")


@app.command()
@cli_command
def forget(
    fact_id: Annotated[str, typer.Argument(help="Fact id.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    agent: AgentOption = None,
) -> None:
    """Forget a fact of any tier for good, as the user. This cannot be undone."""
    with _open(agent) as ctx:
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


# ---- ingest, conflicts, hook ---------------------------------------------------------


def _count(value: object) -> object:
    return len(value) if isinstance(value, (list, tuple, set, frozenset)) else value  # pyright: ignore[reportUnknownArgumentType]


@app.command()
@cli_command
def ingest(path: Annotated[Path, typer.Argument(help="A Markdown file or a folder.")]) -> None:
    """Read Markdown documents into searchable notes."""
    ingest_path = _import("ingatan.ingest", "ingest_path")
    if not path.exists():
        raise NotFoundError(f"No such file or folder: {path}")
    with _open() as ctx:
        report = ingest_path(ctx, path)
    typer.echo(
        f"Files: {_count(report.files)}  created: {_count(report.created)}  "
        f"replaced: {_count(report.replaced)}  skipped: {_count(report.skipped)}"
    )
    skipped = report.skipped
    if isinstance(skipped, (list, tuple)):
        for item in skipped:  # pyright: ignore[reportUnknownVariableType]
            typer.echo(f"  skipped: {item}")


@app.command()
@cli_command
def conflicts(agent: AgentOption = None) -> None:
    """Report how many conflict candidates were resolved (NFR-9), from the audit log."""
    with _open(agent) as ctx:
        report = compute_report(audit.entries(ctx))
    typer.echo(render_report(report))


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
