"""Proof of NFR-3: while the guard is on, opening a socket is recorded and refused."""

from __future__ import annotations

import socket
from types import TracebackType
from typing import Any

__all__ = ["NetworkBlockedError", "NetworkGuard"]


class NetworkBlockedError(OSError):
    """Raised in place of any network access while the guard is active."""


class NetworkGuard:
    """Patch `socket.socket` (and the name lookups) for the length of a `with` block.

    Every attempt is appended to `attempts`; the caller decides what a non-empty list means.
    """

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._saved: dict[str, Any] = {}

    def _refuse(self, what: str) -> NetworkBlockedError:
        self.attempts.append(what)
        return NetworkBlockedError(f"network access is blocked during evals ({what})")

    def __enter__(self) -> NetworkGuard:
        guard = self
        original = socket.socket

        class BlockedSocket(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise guard._refuse("socket.socket()")

        def blocked(name: str) -> Any:
            def refuse(*_args: Any, **_kwargs: Any) -> Any:
                raise guard._refuse(f"socket.{name}()")

            return refuse

        for name in ("create_connection", "getaddrinfo", "gethostbyname"):
            self._saved[name] = getattr(socket, name)
            setattr(socket, name, blocked(name))
        self._saved["socket"] = original
        socket.socket = BlockedSocket  # type: ignore[misc]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for name, value in self._saved.items():
            setattr(socket, name, value)
        self._saved.clear()
