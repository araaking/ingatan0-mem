"""`ingatan embeddings enable`: packages, model, setting and reindex in one step."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import Result

from ingatan import facts
from ingatan.scope import Context, EmbeddingConfig, load_embeddings, resolve_scope
from ingatan.search import embed as embed_store
from ingatan.search import local as local_models

Invoke = Callable[..., Result]
MakeContext = Callable[..., Context]

PROJECT = 'business = "tokoku"\nagent = "claude-code"\n'
OTHER_MODEL = "intfloat/multilingual-e5-small"


@pytest.fixture
def pulls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """No download, no missing package, and a model that answers without being loaded."""
    calls: list[str] = []

    def fake_pull(model_name: str, model_dir: object, **_kwargs: Any) -> list[Path]:
        calls.append(model_name)
        return []

    def fake_embed(cfg: EmbeddingConfig, text: str, *, is_query: bool) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(local_models, "pull", fake_pull)
    monkeypatch.setattr(local_models, "missing_packages", lambda: [])
    monkeypatch.setattr(embed_store, "_embed_raw", fake_embed)
    # The autouse fixture pins the provider to "none" through the environment.
    monkeypatch.delenv("INGATAN_EMBED_PROVIDER")
    return calls


@pytest.fixture
def project() -> Path:
    path = Path.cwd() / ".ingatan.toml"
    path.write_text(PROJECT, encoding="utf-8", newline="\n")
    return path


def _vector_count(ctx: Context) -> int:
    return int(ctx.business.execute("SELECT count(*) FROM fact_embeddings").fetchone()[0])


def test_enable_stops_before_download_when_packages_are_missing(
    invoke: Invoke, pulls: list[str], project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_models, "missing_packages", lambda: ["onnxruntime"])
    result = invoke("embeddings", "enable")
    assert result.exit_code == 1, result.output
    assert "onnxruntime" in result.output
    assert "ingatan[vector]" in result.output
    assert pulls == []
    assert project.read_text(encoding="utf-8") == PROJECT


def test_enable_pulls_sets_provider_and_reindexes(
    invoke: Invoke, pulls: list[str], project: Path, make_ctx: MakeContext
) -> None:
    ctx = make_ctx()
    facts.remember(ctx, "Ongkir ke Bandung Rp 15.000 per kg.", entities=["ongkir Bandung"])
    result = invoke("embeddings", "enable")
    assert result.exit_code == 0, result.output
    assert pulls == [local_models.DEFAULT_LOCAL_MODEL]
    assert resolve_scope().embeddings.provider == "local"
    assert "Reindex (local" in result.output
    assert "facts indexed=1" in result.output
    assert _vector_count(ctx) == 1


def test_enable_keeps_the_rest_of_the_project_file(
    invoke: Invoke, pulls: list[str], project: Path
) -> None:
    result = invoke("embeddings", "enable")
    assert result.exit_code == 0, result.output
    assert project.read_text(encoding="utf-8") == PROJECT + '\n[embeddings]\nprovider = "local"\n'


def test_enable_global_writes_home_config(
    invoke: Invoke, pulls: list[str], project: Path, ingatan_home: Path
) -> None:
    result = invoke("embeddings", "enable", "--global")
    assert result.exit_code == 0, result.output
    home_config = ingatan_home / "config.toml"
    assert home_config.read_text(encoding="utf-8") == '[embeddings]\nprovider = "local"\n'
    assert project.read_text(encoding="utf-8") == PROJECT
    assert load_embeddings().provider == "local"


def test_enable_without_a_project_file_writes_the_home_config(
    invoke: Invoke, pulls: list[str], ingatan_home: Path
) -> None:
    result = invoke("embeddings", "enable")
    assert result.exit_code == 0, result.output
    assert (ingatan_home / "config.toml").is_file()
    assert not (Path.cwd() / ".ingatan.toml").exists()
    assert "ingatan reindex" in result.output
    assert "Reindex (" not in result.output


def test_enable_with_no_reindex_leaves_existing_facts_without_a_vector(
    invoke: Invoke, pulls: list[str], project: Path, make_ctx: MakeContext
) -> None:
    ctx = make_ctx()
    facts.remember(ctx, "Ongkir ke Bandung Rp 15.000 per kg.", entities=["ongkir Bandung"])
    result = invoke("embeddings", "enable", "--no-reindex")
    assert result.exit_code == 0, result.output
    assert "Reindex (" not in result.output
    assert _vector_count(ctx) == 0
    assert resolve_scope().embeddings.provider == "local"


def test_enable_replaces_the_model_of_another_provider(invoke: Invoke, pulls: list[str]) -> None:
    path = Path.cwd() / ".ingatan.toml"
    path.write_text(
        PROJECT + '\n[embeddings]\nprovider = "ollama"\nmodel = "nomic-embed-text"\n',
        encoding="utf-8",
        newline="\n",
    )
    result = invoke("embeddings", "enable")
    assert result.exit_code == 0, result.output
    cfg = load_embeddings()
    assert cfg.provider == "local"
    assert cfg.model == local_models.DEFAULT_LOCAL_MODEL


def test_enable_with_a_named_model_records_it(
    invoke: Invoke, pulls: list[str], project: Path
) -> None:
    result = invoke("embeddings", "enable", "--model", OTHER_MODEL)
    assert result.exit_code == 0, result.output
    assert pulls == [OTHER_MODEL]
    cfg = load_embeddings()
    assert (cfg.provider, cfg.model) == ("local", OTHER_MODEL)


def test_enable_rejects_an_unknown_model_before_changing_anything(
    invoke: Invoke, pulls: list[str], project: Path
) -> None:
    result = invoke("embeddings", "enable", "--model", "no/such-model")
    assert result.exit_code == 1, result.output
    assert "Unknown local embedding model" in result.output
    assert pulls == []
    assert project.read_text(encoding="utf-8") == PROJECT


def test_enable_twice_changes_nothing_the_second_time(
    invoke: Invoke, pulls: list[str], project: Path
) -> None:
    assert invoke("embeddings", "enable").exit_code == 0
    after_first = project.read_bytes()
    second = invoke("embeddings", "enable")
    assert second.exit_code == 0, second.output
    assert project.read_bytes() == after_first


def test_enable_warns_when_env_overrides(
    invoke: Invoke, pulls: list[str], project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGATAN_EMBED_PROVIDER", "none")
    result = invoke("embeddings", "enable")
    assert result.exit_code == 1, result.output
    assert "INGATAN_EMBED_PROVIDER" in result.output
    assert 'provider = "local"' in project.read_text(encoding="utf-8")
    assert "Reindex (" not in result.output
