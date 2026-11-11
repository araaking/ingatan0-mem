"""The only source of time. Tests and evals freeze it with `set_now`."""

from __future__ import annotations

from datetime import datetime, timezone

from ingatan.errors import ValidationError

_frozen: datetime | None = None


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now() -> datetime:
    """Current time, timezone-aware, UTC."""
    if _frozen is not None:
        return _frozen
    return datetime.now(timezone.utc)


def set_now(dt: datetime | None) -> None:
    """Freeze the clock at `dt` (naive values are taken as UTC). `None` unfreezes."""
    global _frozen
    _frozen = None if dt is None else _as_utc(dt)


def iso(dt: datetime) -> str:
    """`YYYY-MM-DDTHH:MM:SSZ`. These strings sort correctly as text."""
    return _as_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(value: str) -> datetime:
    """Parse a date (`YYYY-MM-DD`) or an ISO timestamp. A bare date means 00:00:00 UTC."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return _as_utc(datetime.fromisoformat(text))
    except ValueError:
        raise ValidationError(
            f"'{value}' is not a date. Use YYYY-MM-DD or a full ISO timestamp."
        ) from None


def normalize(value: str) -> str:
    """Turn a date or ISO timestamp into the stored full form."""
    return iso(parse(value))
