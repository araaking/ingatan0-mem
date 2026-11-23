"""Run the extraction eval with the model configured under `[llm]`.

    python -m evals.extraction [--case ID ...] [--report PATH] [--cases-dir DIR]

Every case gets its own throwaway data folder: its episodes are recorded, `ingatan extract`
reads them with the configured model, and what reached the proposal queue is judged against
the case (see `metrics`). The report goes to `evals/reports/latest-extraction.md`.

The run talks to one place only: the host and port of `[llm] api_base` (or the provider's
default). Anything else that tries to open a connection is refused and fails the run.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import sys
import tempfile
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evals.extraction import metrics
from ingatan import episodes, extract, proposals
from ingatan.errors import IngatanError
from ingatan.extract import llm
from ingatan.scope import (
    LlmConfig,
    Scope,
    data_home,
    find_config,
    load_home_config,
    open_context,
    parse_llm,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases"
DEFAULT_REPORT = HERE.parent / "reports" / "latest-extraction.md"


def configured_llm() -> LlmConfig:
    """`[llm]` as `ingatan extract` would resolve it here: environment, project, home."""
    project: dict[str, Any] = {}
    config_file = find_config()
    if config_file is not None:
        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover - Python 3.10 only
            import tomli as tomllib
        with config_file.open("rb") as handle:
            project = tomllib.load(handle)
    return parse_llm(os.environ, project, load_home_config(data_home()))


@contextlib.contextmanager
def only_endpoint(host: str, port: int) -> Iterator[list[str]]:
    """Let connections go to `host:port` and nowhere else. Yields the refused attempts."""
    attempts: list[str] = []
    original = socket.create_connection

    def guarded(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if (str(address[0]), int(address[1])) != (host, port):
            attempts.append(f"{address[0]}:{address[1]}")
            raise OSError(f"{address[0]}:{address[1]} is not the configured endpoint")
        return original(address, *args, **kwargs)

    socket.create_connection = guarded  # type: ignore[assignment]
    try:
        yield attempts
    finally:
        socket.create_connection = original


def endpoint(cfg: LlmConfig) -> tuple[str, int]:
    parts = urlsplit(cfg.api_base or llm.DEFAULT_BASES[cfg.provider])
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.hostname or "localhost", port


def run_case(case: metrics.Case, cfg: LlmConfig, home: Path) -> metrics.CaseResult:
    """Record the case's episodes in `home`, extract, and judge what reached the queue."""
    scope = Scope(business="tokoku", agent="asisten", home=home, llm=cfg)
    with open_context(scope) as ctx:
        for user_text, assistant_text in case.episodes:
            episodes.record(ctx, user_text, assistant_text)
        report = extract.extract_pending(ctx, limit=len(case.episodes))
        queued = [item.text for item in proposals.pending(ctx)]
    return metrics.judge(case, queued, dropped=report.rejected, errors=report.errors)


def _share(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{100 * part / whole:.1f}% ({part} of {whole})"


def write_report(path: Path, label: str, results: list[metrics.CaseResult]) -> None:
    total = metrics.totals(results)
    lines = [
        "# Ingatan extraction eval",
        "",
        f"Date: {date.today().isoformat()}  ",
        f"Model: {label}  ",
        f"Cases: {len(results)}  ",
        "",
        "What is counted is what reached the proposal queue, after Ingatan's own checks.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| correct (matches a fact the text states) | {_share(total.correct, total.extracted)} |",
        f"| wrong (forbidden, another value, or nothing was expected) | "
        f"{_share(total.wrong, total.extracted)} |",
        f"| unlisted (matches nothing listed: true or invented, not judged) | "
        f"{_share(total.unlisted, total.extracted)} |",
        f"| found (expected facts that were extracted) | {_share(total.found, total.expected)} |",
        f"| dropped by Ingatan before the queue | {total.dropped} |",
        f"| episodes the model could not be read for | {total.errors} |",
        "",
        "## Cases",
        "",
        "| Case | Extracted | Correct | Wrong | Unlisted | Found | Dropped | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.case_id} | {len(r.extracted)} | {r.correct} | {r.wrong} | {r.unlisted} | "
            f"{r.found} of {r.expected} | {r.dropped} | {r.errors} |"
        )
    lines += ["", "Definitions are in `evals/README.md`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.extraction", description=__doc__)
    parser.add_argument("--case", action="append", metavar="ID", help="run only this case")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--no-network-guard",
        action="store_true",
        help="do not restrict connections to the configured endpoint (for a stand-in model)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = configured_llm()
        llm.ensure_configured(cfg)
        cases = metrics.load_cases(args.cases_dir)
    except IngatanError as exc:
        print(f"evals.extraction: {exc.message}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"case error: {exc}", file=sys.stderr)
        return 2
    if args.case:
        unknown = sorted(set(args.case) - {c.id for c in cases})
        if unknown:
            print(f"unknown case(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in set(args.case)]
    label = llm.model_label(cfg)
    host, port = endpoint(cfg)
    guard = contextlib.nullcontext([]) if args.no_network_guard else only_endpoint(host, port)
    results: list[metrics.CaseResult] = []
    with guard as refused:
        for case in cases:
            with tempfile.TemporaryDirectory(
                prefix="ingatan-extract-", ignore_cleanup_errors=True
            ) as home:
                result = run_case(case, cfg, Path(home))
            results.append(result)
            print(
                f"{case.id:40s} extracted {len(result.extracted)}, correct {result.correct}, "
                f"wrong {result.wrong}, unlisted {result.unlisted}, "
                f"found {result.found} of {result.expected}"
            )
    total = metrics.totals(results)
    print()
    print(f"model     {label}")
    print(f"correct   {_share(total.correct, total.extracted)}")
    print(f"wrong     {_share(total.wrong, total.extracted)}")
    print(f"unlisted  {_share(total.unlisted, total.extracted)}")
    print(f"found     {_share(total.found, total.expected)}")
    report = args.report
    if report is None and not args.case:
        report = DEFAULT_REPORT
    if report is not None:
        write_report(report, label, results)
        print(f"\nreport written to {report}")
    if refused:
        print(f"\nFAILED: tried to reach {', '.join(refused)}", file=sys.stderr)
        return 1
    if total.errors and total.errors == sum(len(c.episodes) for c in cases):
        print("\nFAILED: the model answered for no episode", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
