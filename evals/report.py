"""Console table, markdown report and the exit-code rule."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .metrics import Metrics, compute
from .runner import FAIL, PASS, SKIP, XFAIL, XPASS, RunResult

__all__ = ["exit_reasons", "format_metrics", "format_table", "write_report"]

BRIEF_UNAVAILABLE = "n/a (brief not available)"


def _pct(value: float | None, checks: str) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}% ({checks})"


def _brief_missing(run: RunResult) -> bool:
    return any("brief not available" in n for r in run.results for n in r.notes)


def _brief_tokens(metrics: Metrics, run: RunResult) -> str:
    if metrics.tokens_mean is not None:
        return f"mean {metrics.tokens_mean:.0f}, max {metrics.tokens_max} ({metrics.briefs} briefs)"
    return BRIEF_UNAVAILABLE if _brief_missing(run) else "n/a (no brief was built)"


def _rows(run: RunResult) -> list[tuple[str, str]]:
    m = compute(run.records)
    return [
        ("recall@5", _pct(m.recall_at_5, f"{m.recall_checks} checks")),
        ("stale rate", _pct(m.stale_rate, f"{m.stale_checks} recall checks")),
        ("leak rate", _pct(m.leak_rate, f"{m.leaked_items} of {m.total_items} results")),
        ("tokens per brief", _brief_tokens(m, run)),
    ]


def format_metrics(run: RunResult) -> list[str]:
    return [f"{name:<17}{value}" for name, value in _rows(run)]


def _summary(run: RunResult) -> str:
    return (
        f"{len(run.results)} scenarios: {run.count(PASS)} pass, {run.count(FAIL)} fail, "
        f"{run.count(XFAIL)} xfail, {run.count(XPASS)} xpass, {run.count(SKIP)} skipped"
    )


def _first_problem(failures: list[str], notes: list[str], width: int) -> str:
    if failures:
        text = failures[0] if len(failures) == 1 else f"{failures[0]} (+{len(failures) - 1})"
    elif notes:
        text = notes[0]
    else:
        return ""
    return text if len(text) <= width else text[: width - 1] + "…"


def format_table(run: RunResult) -> list[str]:
    width = max((len(r.scenario.id) for r in run.results), default=8)
    lines = [f"{'scenario':<{width}}  {'category':<13} {'status':<6}  detail"]
    for r in run.results:
        detail = _first_problem(r.failures, r.notes, 90)
        lines.append(f"{r.scenario.id:<{width}}  {r.scenario.category:<13} {r.status:<6}  {detail}")
    lines.append("")
    lines.append(_summary(run))
    return lines


def exit_reasons(run: RunResult) -> list[str]:
    """Why the run must exit non-zero. Empty means success."""
    reasons: list[str] = []
    m = compute(run.records)
    if m.leak_rate:
        reasons.append(f"leak rate is {m.leak_rate * 100:.1f}% ({m.leaked_items} leaked results)")
    failed = [r.scenario.id for r in run.results if r.status == FAIL]
    if failed:
        reasons.append(f"{len(failed)} scenario(s) failed: {', '.join(failed)}")
    if run.network_attempts:
        reasons.append(f"network access was attempted: {', '.join(run.network_attempts)}")
    return reasons


def write_report(path: Path, run: RunResult) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    targets = {"recall@5": "over 80%", "stale rate": "under 5%", "leak rate": "0"}
    out: list[str] = [
        "# Ingatan eval report",
        "",
        f"Date: {today}  ",
        f"Result: {_summary(run)}  ",
        f"Network attempts during the run: {len(run.network_attempts)}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---|---|",
    ]
    for name, value in _rows(run):
        out.append(f"| {name} | {value} | {targets.get(name, 'about 1,000')} |")
    out += [
        "",
        "Definitions are in `evals/README.md`.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Category | Source | Status | First problem |",
        "|---|---|---|---|---|",
    ]
    for r in run.results:
        first = _first_problem(r.failures, r.notes, 200).replace("|", "\\|")
        sc = r.scenario
        out.append(f"| {sc.id} | {sc.category} | {sc.source} | {r.status} | {first} |")

    out += ["", "## Expected failures (xfail)", ""]
    expected = [r for r in run.results if r.scenario.xfail]
    if not expected:
        out.append("None.")
    for r in expected:
        out += [f"### {r.scenario.id} ({r.status})", "", f"Reason: {r.scenario.xfail}", ""]
        out += [f"- {f}" for f in r.failures] or ["- (it passes now: remove its `xfail`)"]
        out.append("")

    unexpected = [r for r in run.results if r.status == FAIL]
    if unexpected:
        out += ["## Unexpected failures", ""]
        for r in unexpected:
            out += [f"### {r.scenario.id}", ""] + [f"- {f}" for f in r.failures] + [""]

    out += ["## Briefs measured", ""]
    briefs = [rec for rec in run.records if rec.kind == "brief" and rec.tokens is not None]
    if briefs:
        out += ["| Scenario | Caller | Tokens | Budget |", "|---|---|---|---|"]
        out += [f"| {b.scenario} | {b.label} | {b.tokens} | {b.budget} |" for b in briefs]
    else:
        out.append(BRIEF_UNAVAILABLE if _brief_missing(run) else "No brief was built.")
    notes = sorted({n for r in run.results for n in r.notes})
    if notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
