"""The package metadata and the release workflow agree with each other."""

from __future__ import annotations

import sys
from pathlib import Path

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
