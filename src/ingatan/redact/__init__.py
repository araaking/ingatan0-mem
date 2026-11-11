"""Replace secrets (API keys, tokens, passwords) with a marker before anything is stored.

The marker is the literal text `[redacted]` (constant `MARKER`). Provider keys, JWTs,
PEM private keys and bearer tokens are replaced whole. In assignments such as
`password=hunter2` or `kata sandi: rahasia` the key name stays and only the value is
replaced. The pattern list is fixed (no allow-list in the MVP), so a sentence like
`token: habis` is redacted too.
"""

from __future__ import annotations

import re

__all__ = ["MARKER", "redact"]

MARKER = "[redacted]"

_WHOLE_PATTERNS = [
    # PEM private key blocks, including a block that was cut off before its END line.
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----|[\s\S]*\Z)",
    # JWT: three base64url parts, header and payload start with "eyJ" ({"...).
    r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}",
    # Provider key prefixes.
    r"\bsk-ant-[A-Za-z0-9_-]{16,}",
    r"\bsk-[A-Za-z0-9_-]{20,}",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"\bAIza[A-Za-z0-9_-]{35}",
    r"\bglpat-[A-Za-z0-9_-]{20,}",
    r"\bhf_[A-Za-z0-9]{30,}",
    r"\b(?:sk|rk|pk)_live_[A-Za-z0-9]{16,}",
]
_WHOLE = re.compile("|".join(f"(?:{p})" for p in _WHOLE_PATTERNS), re.DOTALL)

# "Bearer <token>": the token must contain a digit or be long, so prose such as
# "bearer authentication" is left alone. The word "Bearer" stays.
_TOKEN_CHARS = r"[A-Za-z0-9._~+/=-]"
_BEARER = re.compile(
    rf"(\bBearer[ \t]+)(?:(?=[A-Za-z0-9._~+/=-]*\d){_TOKEN_CHARS}{{8,}}|{_TOKEN_CHARS}{{20,}})",
    re.IGNORECASE,
)

# key (name stays) + separator + value (replaced). Spaces and tabs only, never a newline.
_ASSIGNMENT = re.compile(
    r"(?P<key>\b[\w.-]*?(?:password|passwd|secret|token|api[_-]?key|sandi))"
    r"(?P<sep>[\"']?[ \t]*[:=][ \t]*)"
    r"(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s\"']+)",
    re.IGNORECASE,
)


def _assignment(match: re.Match[str]) -> str:
    return f"{match.group('key')}{match.group('sep')}{MARKER}"


def redact(text: str) -> str:
    """Return `text` with every secret-looking part replaced by `MARKER`."""
    text = _WHOLE.sub(MARKER, text)
    text = _BEARER.sub(lambda m: f"{m.group(1)}{MARKER}", text)
    return _ASSIGNMENT.sub(_assignment, text)
