from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from ingatan.cli import app
from ingatan.scope import Context

MakeContext = Callable[..., Context]
Invoke = Callable[..., Result]


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Business tokoku and agent claude-code from the environment, as the CLI resolves them."""
    monkeypatch.setenv("INGATAN_BUSINESS", "tokoku")
    monkeypatch.setenv("INGATAN_AGENT", "claude-code")


@pytest.fixture
def invoke() -> Invoke:
    runner = CliRunner()

    def run(*args: str, input: str | None = None, **kwargs: Any) -> Result:
        return runner.invoke(app, list(args), input=input, **kwargs)

    return run


@pytest.fixture
def cli_ctx(configured: None, make_ctx: MakeContext) -> Context:
    """A context for the same business and agent the CLI resolves (data for CLI tests)."""
    return make_ctx()
