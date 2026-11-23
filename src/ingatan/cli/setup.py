"""The steps behind `ingatan embeddings enable` and `ingatan setup`.

The commands in `commands.py` only parse options and print. What is decided, in which order,
and what is written where lives here, so it can be tested without a terminal.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from ingatan import search
from ingatan.errors import ConfigError, IngatanError
from ingatan.scope import (
    CONFIG_NAME,
    HOME_CONFIG_NAME,
    EmbeddingConfig,
    configfile,
    data_home,
    find_config,
    load_embeddings,
    open_context,
    resolve_scope,
)
from ingatan.search import local as local_models

Echo = Callable[[str], None]


def reindex_summary(provider: str, report: search.ReindexReport) -> str:
    """The one line `ingatan reindex` prints for a finished run."""
    return (
        f"Reindex ({provider} / {report.model}): "
        f"facts indexed={report.facts_indexed} skipped={report.facts_skipped}, "
        f"notes indexed={report.notes_indexed} skipped={report.notes_skipped}"
        + (f", errors={report.errors}" if report.errors else "")
    )


def pull_model(spec: local_models.LocalModel, model_dir: str, progress_echo: Echo) -> list[Path]:
    """Download the files of `spec` that are not there yet, reporting each tenth of a file."""
    shown: dict[str, int] = {}

    def progress(file: local_models.ModelFile, received: int) -> None:
        tenth = received * 10 // max(1, file.size)
        if shown.get(file.name) != tenth:
            shown[file.name] = tenth
            progress_echo(f"  {file.name}: {received / 1e6:.0f} of {file.size / 1e6:.0f} MB")

    return local_models.pull(spec.name, model_dir, progress=progress)


def _configured_local_model(cfg: EmbeddingConfig) -> str:
    return cfg.model.strip().casefold() or local_models.DEFAULT_LOCAL_MODEL


def _why_not_in_effect(target: Path, project_file: Path | None) -> str:
    written = f"The setting was written to {target}, but "
    for name in ("INGATAN_EMBED_PROVIDER", "INGATAN_EMBED_MODEL"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return (
                f"{written}{name}={value} in the environment overrides it. "
                "Unset it and run this again."
            )
    if project_file is not None and project_file != target:
        return (
            f"{written}{project_file} has its own [embeddings] settings, which win. "
            "Change them there, or run this without --global."
        )
    return f"{written}it is not what Ingatan resolves. Run `ingatan doctor` to see why."


def enable_semantic(*, model: str | None, use_home_config: bool, reindex: bool, echo: Echo) -> None:
    """Switch the offline semantic search on: model files, `provider = "local"`, vectors.

    In this order, stopping at the first step that cannot be done: the runtime packages
    must be installed (nothing is downloaded or written otherwise), the model is downloaded
    if it is not on disk, the setting is written to `.ingatan.toml` (or to `config.toml`
    of the data folder with `use_home_config`, or when there is no project file), the
    facts that already exist are embedded, and one sentence is embedded as a proof.
    """
    cfg = load_embeddings()
    spec = local_models.resolve((model or (cfg.model if cfg.provider == "local" else "")).strip())
    missing = local_models.missing_packages()
    if missing:
        raise ConfigError(
            f"Semantic search needs the packages {', '.join(missing)}, which are not "
            "installed. Install Ingatan with its `vector` extra "
            '(`uv tool install "ingatan[vector]"`), or add them to the environment Ingatan '
            f"runs in ({local_models.INSTALL_HINT}), then run this again. "
            "Nothing was downloaded or changed."
        )

    echo(f"Model: {spec.name} ({spec.summary})")
    fetched = pull_model(spec, cfg.model_dir, echo)
    if fetched:
        size = sum(path.stat().st_size for path in fetched) / 1e6
        echo(f"Downloaded {len(fetched)} file(s), {size:.0f} MB. Checksums verified.")
    else:
        echo("Model files are present and verified. Nothing was downloaded.")

    project_file = find_config()
    if use_home_config or project_file is None:
        target = data_home() / HOME_CONFIG_NAME
        if not use_home_config:
            echo(f"No {CONFIG_NAME} here, so the setting goes to {target} (every project).")
    else:
        target = project_file
    changed = configfile.apply(target, "embeddings", "provider", "local")
    if _configured_local_model(load_embeddings()) != spec.name:
        changed = configfile.apply(target, "embeddings", "model", spec.name) or changed
    current = load_embeddings()
    echo(f'{"Set" if changed else "Already set"}: [embeddings] provider = "local" in {target}')
    if current.provider != "local" or _configured_local_model(current) != spec.name:
        raise ConfigError(_why_not_in_effect(target, project_file))

    if not reindex:
        echo("Skipped embedding the existing facts. Run `ingatan reindex` when you want them.")
    else:
        try:
            scope = resolve_scope()
        except ConfigError as exc:
            echo(
                f"No fact was embedded: {exc.message} "
                "Run `ingatan reindex` inside a project to embed the facts it already has."
            )
        else:
            with open_context(scope) as ctx:
                report = search.reindex(ctx)
            echo(reindex_summary(current.provider, report))
            if report.errors:
                raise IngatanError(f"embedding error: {report.last_error or 'unknown'}")

    dims = search.probe_embedding(current)
    echo(f"Semantic search is on: {spec.name}, {dims} dimensions.")
