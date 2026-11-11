from __future__ import annotations

from pathlib import Path

import pytest

from ingatan.redact import MARKER, redact

HERE = Path(__file__).parent


def _lines(name: str) -> list[str]:
    text = (HERE / name).read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


# "{}" in the corpus splits the fake keys so secret scanners do not flag the file.
SECRET_CASES = [ln.replace("{}", "").split("\t") for ln in _lines("corpus_secrets.txt")]
PLAIN_CASES = _lines("corpus_plain.txt")


def test_marker_is_documented_literal() -> None:
    assert MARKER == "[redacted]"


def test_corpora_are_not_empty() -> None:
    assert len(SECRET_CASES) >= 30
    assert len(PLAIN_CASES) >= 25
    assert all(len(case) == 2 and case[1] for case in SECRET_CASES)


@pytest.mark.parametrize(("text", "secret"), SECRET_CASES)
def test_every_secret_in_the_corpus_is_redacted(text: str, secret: str) -> None:
    text = text.replace("\n", "\n")
    result = redact(text)
    assert result != text
    assert MARKER in result
    assert secret.replace("\n", "\n") not in result


@pytest.mark.parametrize("text", PLAIN_CASES)
def test_ordinary_text_is_untouched(text: str) -> None:
    assert redact(text) == text


def test_assignment_keeps_the_key_name() -> None:
    assert redact("password=hunter2") == f"password={MARKER}"
    assert redact("kata sandi: rahasia") == f"kata sandi: {MARKER}"
    assert redact('{"api_key": "abc"}') == '{"api_key": ' + MARKER + "}"


def test_bearer_keeps_the_word() -> None:
    assert redact("Authorization: Bearer abcDEF1234567890xyz") == f"Authorization: Bearer {MARKER}"


def test_pem_block_is_replaced_whole() -> None:
    pem = "before\n-----BEGIN PRIVATE KEY-----\nAAAA\nBBBB\n-----END PRIVATE KEY-----\nafter"
    assert redact(pem) == f"before\n{MARKER}\nafter"


def test_redact_is_idempotent_and_handles_empty() -> None:
    once = redact("token=abc123 and sk-Ab12Cd34Ef56Gh78Ij90Kl12")
    assert redact(once) == once
    assert redact("") == ""


def test_password_assignment_does_not_swallow_the_next_line() -> None:
    assert redact("Password:\nLangkah berikutnya") == "Password:\nLangkah berikutnya"


def test_indonesian_unicode_survives_around_a_secret() -> None:
    result = redact("Bu Sári bilang ‘kata sandi: rahasia’ ✓")
    assert "Bu Sári bilang ‘kata sandi: " in result
    assert "rahasia" not in result
