from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ingatan.errors import ConfigError
from ingatan.scope import configfile

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib


def test_adds_section_when_missing() -> None:
    out = configfile.updated_text('business = "tokoku"\n', "embeddings", "provider", "local")
    assert tomllib.loads(out) == {"business": "tokoku", "embeddings": {"provider": "local"}}


def test_adds_section_to_an_empty_text() -> None:
    out = configfile.updated_text("", "embeddings", "provider", "local")
    assert out == '[embeddings]\nprovider = "local"\n'


def test_adds_section_after_a_last_line_without_newline() -> None:
    out = configfile.updated_text('business = "tokoku"', "embeddings", "provider", "local")
    assert out == 'business = "tokoku"\n\n[embeddings]\nprovider = "local"\n'


def test_replaces_existing_key_only() -> None:
    text = (
        'business = "tokoku"\n'
        "\n"
        "[embeddings]\n"
        "# which model answers\n"
        'provider = "none"\n'
        "threads = 4\n"
        "\n"
        "[brief]\n"
        "token_budget = 750\n"
    )
    out = configfile.updated_text(text, "embeddings", "provider", "local")
    assert out == text.replace('provider = "none"', 'provider = "local"')


def test_adds_key_at_the_end_of_an_existing_section() -> None:
    text = "[embeddings]\nthreads = 4\n\n[brief]\ntoken_budget = 750\n"
    out = configfile.updated_text(text, "embeddings", "provider", "local")
    assert out == '[embeddings]\nthreads = 4\nprovider = "local"\n\n[brief]\ntoken_budget = 750\n'


def test_keeps_crlf_and_comments() -> None:
    text = (
        "# my project\r\n"
        'business = "tokoku"\r\n'
        "[embeddings]\r\n"
        'provider = "none"   # off for now\r\n'
        "threads = 4\r\n"
    )
    out = configfile.updated_text(text, "embeddings", "provider", "local")
    before, after = text.split("\r\n"), out.split("\r\n")
    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert changed == [3]
    assert after[3] == 'provider = "local"   # off for now'
    assert "\n" not in out.replace("\r\n", "")


def test_new_lines_use_the_line_ending_of_the_file() -> None:
    out = configfile.updated_text('business = "tokoku"\r\n', "embeddings", "provider", "local")
    assert out == 'business = "tokoku"\r\n\r\n[embeddings]\r\nprovider = "local"\r\n'


def test_a_hash_inside_the_old_string_is_not_taken_for_a_comment() -> None:
    text = '[embeddings]\nmodel = "a#b"  # note\n'
    out = configfile.updated_text(text, "embeddings", "model", "c")
    assert out == '[embeddings]\nmodel = "c"  # note\n'


def test_booleans_and_integers_are_written_as_toml() -> None:
    out = configfile.updated_text("", "embeddings", "defer", True)
    out = configfile.updated_text(out, "embeddings", "threads", 4)
    assert tomllib.loads(out) == {"embeddings": {"defer": True, "threads": 4}}


def test_a_string_with_quotes_survives() -> None:
    out = configfile.updated_text("", "embeddings", "model", 'odd "name"')
    assert tomllib.loads(out)["embeddings"]["model"] == 'odd "name"'


def test_the_same_value_changes_nothing() -> None:
    text = '[embeddings]\nprovider = "local"\n'
    assert configfile.updated_text(text, "embeddings", "provider", "local") == text


def test_refuses_text_it_cannot_round_trip() -> None:
    # The section exists as an inline table, so adding `[embeddings]` would define it twice.
    text = 'embeddings = { provider = "none" }\n'
    with pytest.raises(ConfigError, match="by hand"):
        configfile.updated_text(text, "embeddings", "provider", "local")


def test_refuses_a_value_that_spans_lines() -> None:
    text = '[embeddings]\nprovider = """\nnone"""\n'
    with pytest.raises(ConfigError, match="by hand"):
        configfile.updated_text(text, "embeddings", "provider", "local")


def test_refuses_text_that_is_not_toml() -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        configfile.updated_text("business = \n", "embeddings", "provider", "local")


def test_apply_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.toml"
    assert configfile.apply(path, "embeddings", "provider", "local") is True
    assert path.read_bytes() == b'[embeddings]\nprovider = "local"\n'


def test_apply_keeps_the_bytes_it_does_not_change(tmp_path: Path) -> None:
    path = tmp_path / ".ingatan.toml"
    path.write_bytes(b'business = "tokoku"\r\n[embeddings]\r\nprovider = "none"\r\n')
    assert configfile.apply(path, "embeddings", "provider", "local") is True
    assert path.read_bytes() == b'business = "tokoku"\r\n[embeddings]\r\nprovider = "local"\r\n'


def test_apply_reports_no_change(tmp_path: Path) -> None:
    path = tmp_path / ".ingatan.toml"
    path.write_bytes(b'[embeddings]\nprovider = "local"\n')
    assert configfile.apply(path, "embeddings", "provider", "local") is False


def test_apply_leaves_the_file_alone_when_it_refuses(tmp_path: Path) -> None:
    path = tmp_path / ".ingatan.toml"
    original = b'embeddings = { provider = "none" }\n'
    path.write_bytes(original)
    with pytest.raises(ConfigError):
        configfile.apply(path, "embeddings", "provider", "local")
    assert path.read_bytes() == original
