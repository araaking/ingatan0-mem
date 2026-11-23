"""The chat client of the optional fact extraction: request shapes, errors, no surprises."""

from __future__ import annotations

import http.client
import urllib.error
import urllib.request
from typing import Any

import pytest

from ingatan.errors import IngatanError
from ingatan.extract import llm
from ingatan.scope import LlmConfig


@pytest.fixture
def posts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every request instead of sending it, and answer like the provider would."""
    sent: list[dict[str, Any]] = []

    def fake(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        sent.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        if "/api/chat" in url:
            return {"message": {"role": "assistant", "content": "  jawaban ollama  "}}
        return {"choices": [{"message": {"role": "assistant", "content": "jawaban openai"}}]}

    monkeypatch.setattr(llm, "_post_json", fake)
    return sent


def test_provider_none_raises_before_any_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("nothing may be sent while the provider is none")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(IngatanError, match=r"\[llm\] provider"):
        llm.chat(LlmConfig(), "system", "user")


def test_a_provider_without_a_model_name_is_an_error(posts: list[dict[str, Any]]) -> None:
    with pytest.raises(IngatanError, match=r"\[llm\] model"):
        llm.chat(LlmConfig(provider="ollama"), "system", "user")
    assert posts == []


def test_ollama_request_shape(posts: list[dict[str, Any]]) -> None:
    cfg = LlmConfig(provider="ollama", model="local-model", timeout=12.0)
    assert llm.chat(cfg, "aturan", "teks episode") == "jawaban ollama"
    (sent,) = posts
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["payload"] == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "aturan"},
            {"role": "user", "content": "teks episode"},
        ],
        "stream": False,
    }
    assert sent["headers"] == {}
    assert sent["timeout"] == 12.0


def test_openai_request_shape_with_the_key_in_a_header(
    posts: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_LLM_KEY", "sk-test-123")
    cfg = LlmConfig(
        provider="openai",
        model="small-model",
        api_base="http://localhost:8080/v1/",
        api_key_env="MY_LLM_KEY",
    )
    assert llm.chat(cfg, "aturan", "teks") == "jawaban openai"
    (sent,) = posts
    assert sent["url"] == "http://localhost:8080/v1/chat/completions"
    assert "sk-test-123" not in sent["url"]
    assert sent["headers"] == {"Authorization": "Bearer sk-test-123"}
    assert sent["payload"]["model"] == "small-model"
    assert [m["role"] for m in sent["payload"]["messages"]] == ["system", "user"]


def test_openai_defaults_and_no_header_without_a_key(
    posts: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm.chat(LlmConfig(provider="openai", model="m"), "s", "u")
    assert posts[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert posts[0]["headers"] == {}


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ValueError("bad"),
        http.client.IncompleteRead(b"half an answer"),
        http.client.InvalidURL("nonnumeric port: 'abc'"),
    ],
)
def test_a_failed_request_becomes_one_clear_error(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(llm, "_post_json", broken)
    with pytest.raises(IngatanError, match="ollama"):
        llm.chat(LlmConfig(provider="ollama", model="m"), "s", "u")


@pytest.mark.parametrize("answer", [{}, {"message": {}}, {"message": {"content": "   "}}])
def test_an_empty_answer_is_an_error(monkeypatch: pytest.MonkeyPatch, answer: Any) -> None:
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: answer)
    with pytest.raises(IngatanError, match="empty"):
        llm.chat(LlmConfig(provider="ollama", model="m"), "s", "u")


# ---- which failures say something about the text ----------------------------------------------


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://localhost:11434/api/chat", code, "x", None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        http.client.RemoteDisconnected("closed"),
        _http_error(401),
        _http_error(429),
        _http_error(503),
        ValueError("the endpoint answered with a page, not JSON"),
    ],
)
def test_an_endpoint_that_gives_no_answer_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(llm, "_post_json", broken)
    with pytest.raises(llm.Unreachable):
        llm.chat(LlmConfig(provider="ollama", model="m"), "s", "u")


@pytest.mark.parametrize("code", [400, 413, 422])
def test_a_request_the_endpoint_refuses_is_about_the_text(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    def refused(*_args: Any, **_kwargs: Any) -> Any:
        raise _http_error(code)

    monkeypatch.setattr(llm, "_post_json", refused)
    with pytest.raises(IngatanError, match=str(code)) as caught:
        llm.chat(LlmConfig(provider="ollama", model="m"), "s", "u")
    assert not isinstance(caught.value, llm.Unreachable)


def test_an_empty_answer_is_about_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: {"message": {"content": ""}})
    with pytest.raises(IngatanError) as caught:
        llm.chat(LlmConfig(provider="ollama", model="m"), "s", "u")
    assert not isinstance(caught.value, llm.Unreachable)
