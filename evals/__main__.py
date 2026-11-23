"""`python -m evals`: run every scenario, print the table and the metrics, write the report."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

from ingatan.errors import IngatanError
from ingatan.scope import EmbeddingConfig, load_embeddings
from ingatan.search import embed, local

from .driver import InProcessDriver, MemoryDriver
from .report import exit_reasons, format_metrics, format_table, write_report
from .runner import DriverFactory, ScenarioResult, default_driver_factory, run_all
from .scenarios import Scenario, ScenarioError, load_dir

HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = HERE / "scenarios"
DEFAULT_REPORT = HERE / "reports" / "latest.md"
SEMANTIC_REPORT = HERE / "reports" / "latest-semantic.md"
NO_EMBEDDINGS = "none (FTS5 only)"


def _local_embeddings(model_name: str, model_dir: Path | None) -> EmbeddingConfig:
    """The local model as an explicit config, or `IngatanError` saying what is missing.

    The folder is resolved here, before the run points INGATAN_HOME at a temporary folder.
    """
    base = load_embeddings()
    model = local.resolve(model_name or (base.model if base.provider == "local" else ""))
    folder = (model_dir or Path(base.model_dir or local.default_model_dir())).resolve()
    absent = local.missing_packages()
    if absent:
        raise IngatanError(f"--embeddings local needs {', '.join(absent)}: {local.INSTALL_HINT}")
    if local.missing_files(model, folder):
        raise IngatanError(
            f"--embeddings local: {model.name} is not in {folder}. "
            "Run `ingatan embeddings pull` or pass --model-dir."
        )
    return EmbeddingConfig(
        provider="local", model=model.name, model_dir=str(folder), idle_unload_seconds=0
    )


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
        "--embeddings",
        choices=("none", "local"),
        default="none",
        help="run every scenario with the local embedding model switched on (it must "
        "already be downloaded; the run still makes no network call). Default: none",
    )
    parser.add_argument("--model", default="", help="local model name with --embeddings local")
    parser.add_argument(
        "--model-dir", type=Path, default=None, help="model folder with --embeddings local"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="where to write the markdown report (default evals/reports/latest.md, or "
        "latest-semantic.md with --embeddings local; a --scenario run writes none unless "
        "this is given)",
    )
    args = parser.parse_args(argv)

    embeddings_label = NO_EMBEDDINGS
    if args.embeddings == "local":
        try:
            cfg = _local_embeddings(args.model, args.model_dir)
        except IngatanError as exc:
            print(f"evals: {exc.message}", file=sys.stderr)
            return 2
        embeddings_label = f"local ({embed.effective_model(cfg)})"

        def with_model(home: Path, scenario: Scenario) -> MemoryDriver:
            return InProcessDriver(
                home,
                token_budget=scenario.token_budget,
                require_approval=scenario.require_approval,
                embeddings=cfg,
            )

        driver_factory = with_model

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

    try:
        run = run_all(scenarios, driver_factory, on_result=show)
    finally:
        embed.unload_local_models()

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
    print(f"embeddings       {embeddings_label}")

    report = args.report
    if report is None and not args.scenario:
        report = SEMANTIC_REPORT if args.embeddings == "local" else DEFAULT_REPORT
    if report is not None:
        write_report(report, run, embeddings=embeddings_label)
        print(f"\nreport written to {report}")

    reasons = exit_reasons(run)
    if reasons:
        print("\nFAILED: " + "; ".join(reasons), file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
