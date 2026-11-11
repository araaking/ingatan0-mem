"""`python -m evals`: run every scenario, print the table and the metrics, write the report."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

from .report import exit_reasons, format_metrics, format_table, write_report
from .runner import DriverFactory, ScenarioResult, default_driver_factory, run_all
from .scenarios import ScenarioError, load_dir

HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = HERE / "scenarios"
DEFAULT_REPORT = HERE / "reports" / "latest.md"


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def main(
    argv: Sequence[str] | None = None,
    *,
    driver_factory: DriverFactory = default_driver_factory,
) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__)
    parser.add_argument(
        "--scenario", action="append", metavar="NAME", help="run only this scenario id (repeatable)"
    )
    parser.add_argument("--verbose", action="store_true", help="print every step and result")
    parser.add_argument("--scenarios-dir", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="where to write the markdown report (default evals/reports/latest.md; "
        "a --scenario run writes none unless this is given)",
    )
    args = parser.parse_args(argv)

    try:
        scenarios = load_dir(args.scenarios_dir)
    except ScenarioError as exc:
        print(f"scenario error: {exc}", file=sys.stderr)
        return 2
    if args.scenario:
        known = {s.id for s in scenarios}
        unknown = [n for n in args.scenario if n not in known]
        if unknown:
            print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.id in wanted]
    if not scenarios:
        print("no scenarios found", file=sys.stderr)
        return 2

    def show(result: ScenarioResult) -> None:
        if args.verbose:
            print(f"== {result.scenario.id}: {result.scenario.title} [{result.status}]")
            for line in result.log:
                print(line)
            for failure in result.failures:
                print(f"  ! {failure}")

    run = run_all(scenarios, driver_factory, on_result=show)

    print()
    print("\n".join(format_table(run)))
    if not args.verbose:
        for result in run.results:
            if result.failures:
                print(f"\n{result.scenario.id} ({result.status}):")
                for failure in result.failures:
                    print(f"  - {failure}")
    print()
    print("\n".join(format_metrics(run)))

    report = args.report
    if report is None and not args.scenario:
        report = DEFAULT_REPORT
    if report is not None:
        write_report(report, run)
        print(f"\nreport written to {report}")

    reasons = exit_reasons(run)
    if reasons:
        print("\nFAILED: " + "; ".join(reasons), file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
