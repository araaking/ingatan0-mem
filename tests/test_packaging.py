"""The package metadata and the release workflow agree with each other."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

import ingatan

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _release_workflow() -> str:
    return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def test_dunder_version_matches_pyproject() -> None:
    # The version lives in two files. A release built from one and tagged from the other
    # would publish a number that cannot be replaced.
    assert ingatan.__version__ == _project()["version"]


def test_project_links_to_its_changelog_and_documentation() -> None:
    urls = _project()["urls"]
    assert isinstance(urls, dict)
    assert {"Repository", "Issues", "Changelog", "Documentation"} <= set(urls)


def test_release_workflow_can_rehearse_on_testpypi() -> None:
    text = _release_workflow()
    assert "workflow_dispatch:" in text
    assert "environment: testpypi" in text
    assert "repository-url: https://test.pypi.org/legacy/" in text


def test_release_workflow_publishes_to_pypi_only_on_a_tag_push() -> None:
    text = _release_workflow()
    publish = text.split("\n  publish:\n", 1)[1]
    assert "if: github.event_name == 'push'" in publish
    assert "environment: pypi" in publish


def test_release_workflow_refuses_a_tag_that_is_not_the_package_version() -> None:
    text = _release_workflow()
    build = text.split("\n  build:\n", 1)[1].split("\n  publish", 1)[0]
    assert "Tag matches the package version" in build
    assert "if: github.event_name == 'push'" in build
    assert '"v$VERSION" = "$GITHUB_REF_NAME"' in build


# ---- the clean-install proof (CI runs it from the built wheel on three systems) -------

SMOKE = ROOT / "scripts" / "smoke_round_trip.py"


def test_clean_install_job_runs_setup_and_the_round_trip() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = text.split("\n  clean-install:\n", 1)[1]
    assert "ingatan setup --business demo --agent ci --no-semantic --client none" in job
    assert "scripts/smoke_round_trip.py" in job


def test_smoke_round_trip_saves_finds_and_removes_one_fact(
    monkeypatch: pytest.MonkeyPatch, ingatan_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INGATAN_BUSINESS", "demo")
    monkeypatch.setenv("INGATAN_AGENT", "ci")
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SMOKE), run_name="__main__")
    assert stop.value.code == 0
    assert "round trip ok" in capsys.readouterr().out
    from ingatan.scope import open_context, resolve_scope

    with open_context(resolve_scope()) as ctx:
        assert ctx.business.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_smoke_round_trip_refuses_to_run_in_the_default_data_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("INGATAN_HOME")
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SMOKE), run_name="__main__")
    assert stop.value.code == 2
    assert "INGATAN_HOME" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("anchor", "first_section"),
    [
        ('<a id="bahasa-indonesia"></a>', '<a id="id-ringkasan"></a>'),
        ('<a id="english"></a>', '<a id="en-overview"></a>'),
    ],
)
def test_readme_opens_each_language_with_the_three_commands(
    anchor: str, first_section: str
) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    opening = text.split(anchor, 1)[1].split(first_section, 1)[0]
    assert "uv tool install ingatan\n" in opening
    assert "ingatan setup\n" in opening


def test_the_llm_extra_exists_and_brings_no_package() -> None:
    # The extraction client uses the standard library. The extra is a name to install by
    # (`ingatan[llm]`, PRD FR-57), not a set of packages.
    extras = _project()["optional-dependencies"]
    assert isinstance(extras, dict)
    assert extras["llm"] == []
    assert len(extras["vector"]) == 3


def test_the_layering_rule_names_the_extract_module_in_both_guides() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "ingest -> extract -> mcp" in agents
    assert "`ingest`, `extract`, `mcp`" in contributing
