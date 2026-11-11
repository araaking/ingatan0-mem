"""Exceptions that carry a user-facing message."""

from __future__ import annotations


class IngatanError(Exception):
    """Base class. `str(error)` is safe to show to the user."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(IngatanError):
    """Missing or invalid configuration."""


class NotFoundError(IngatanError):
    """An id does not exist, or exists but is not visible to the caller."""


class ValidationError(IngatanError):
    """Bad input from the caller."""


class ScopeError(IngatanError):
    """The caller is not allowed to do this in its scope."""


class SchemaError(IngatanError):
    """The database schema or the SQLite build cannot be used."""
