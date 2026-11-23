"""One chat call to the configured model, over the standard library alone.

Two providers, the same choice the embedding providers offer for a model of your own:
`ollama` (a model on this machine, `/api/chat`) and `openai` (any OpenAI-compatible
endpoint, `/chat/completions`). Nothing is sent unless a provider and a model are set, the
key travels in a header and never in a URL, and every failure is one `IngatanError`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ingatan.errors import IngatanError
from ingatan.scope import LlmConfig

__all__ = ["DEFAULT_BASES", "DEFAULT_KEY_ENV", "chat", "model_label"]

DEFAULT_BASES = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
}
DEFAULT_KEY_ENV = "OPENAI_API_KEY"


def model_label(cfg: LlmConfig) -> str:
    """`provider/model`, the name an extracted fact carries as its origin."""
    return f"{cfg.provider}/{cfg.model.strip()}"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _content(provider: str, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    message: Any = None
    if provider == "ollama":
        message = data.get("message")  # pyright: ignore[reportUnknownMemberType]
    else:
        choices = data.get("choices")  # pyright: ignore[reportUnknownMemberType]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(message, dict):
        return ""
    content = message.get("content")  # pyright: ignore[reportUnknownMemberType]
    return content.strip() if isinstance(content, str) else ""


def chat(cfg: LlmConfig, system: str, user: str) -> str:
    """The model's answer to one system and one user message. Raises `IngatanError`."""
    if cfg.provider == "none":
        raise IngatanError(
            'Fact extraction is off: [llm] provider is "none". To use it, set [llm] provider '
            "(ollama or openai) and model in .ingatan.toml or in config.toml of the data folder."
        )
    model = cfg.model.strip()
    if not model:
        raise IngatanError(
            f"[llm] model is not set. Name the {cfg.provider} model that should read the "
            "conversations: Ingatan picks none for you."
        )
    base = (cfg.api_base or DEFAULT_BASES[cfg.provider]).rstrip("/")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    headers: dict[str, str] = {}
    if cfg.provider == "ollama":
        url = f"{base}/api/chat"
    else:
        url = f"{base}/chat/completions"
        key = (os.environ.get(cfg.api_key_env or DEFAULT_KEY_ENV) or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
    try:
        data = _post_json(url, payload, headers, cfg.timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise IngatanError(
            f"The {cfg.provider} model {model} at {base} did not answer: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    answer = _content(cfg.provider, data)
    if not answer:
        raise IngatanError(f"The {cfg.provider} model {model} returned an empty answer.")
    return answer
