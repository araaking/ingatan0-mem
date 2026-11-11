"""HTTP mode (FR-58): the same seven tools over MCP streamable HTTP, one scope per token.

An outer ASGI app authenticates each request to `/mcp`, resolves the bearer token to a
`Scope` and hands the request to the MCP server of that scope. Those servers are built
lazily with `build_server(ctx)` and cached, one per `(business, agent)`. Stdio mode
(`mcp/server.py`) is not touched.

Nothing here logs or prints a token.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings

from ingatan.errors import ConfigError
from ingatan.mcp.server import build_server
from ingatan.scope import Context, Scope, data_home, open_context
from ingatan.tokens import TokenStore

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "HTTPApp", "build_app", "is_loopback", "run_http"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MCP_PATH = "/mcp"
HEALTH_PATH = "/healthz"
MAX_BODY_BYTES = 1024 * 1024
MAX_TOKEN_CHARS = 512

# The SDK's own Host/Origin check (DNS-rebinding protection) is switched off on purpose. By
# default the SDK only allows loopback Host headers when it binds to loopback, and with any
# other bind host it applies whatever allow-list it is given, so a server on 0.0.0.0 behind
# a reverse proxy or Tailscale would answer 421 for its real hostname. Ingatan cannot know
# that hostname. The check protects unauthenticated local services from a web page that
# rebinds DNS; here every request to `/mcp` already needs a bearer token that such a page
# does not have. The SDK still enforces the Content-Type of POST requests.
TRANSPORT_SECURITY = TransportSecuritySettings(enable_dns_rebinding_protection=False)

ASGIScope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


def is_loopback(host: str) -> bool:
    """True for `localhost` and for loopback IP addresses (127.0.0.0/8, ::1)."""
    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


async def _respond(
    send: Send,
    status: int,
    body: object,
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
    *,
    head_only: bool = False,
) -> None:
    payload = json.dumps(body).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
                *extra_headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": b"" if head_only else payload})


async def _unauthorized(send: Send) -> None:
    # One answer for a missing, malformed and unknown token: the caller learns nothing.
    await _respond(send, 401, {"error": "unauthorized"}, ((b"www-authenticate", b"Bearer"),))


def _bearer_token(scope: ASGIScope) -> str | None:
    """The token from `Authorization: Bearer <token>`, or None if absent or malformed."""
    for name, value in scope.get("headers", ()):
        if name != b"authorization":
            continue
        parts = value.decode("latin-1").split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        token = parts[1]
        if not token or len(token) > MAX_TOKEN_CHARS:
            return None
        return token
    return None


@dataclass
class _ScopeServer:
    """The running MCP server of one scope."""

    context: Context
    manager: StreamableHTTPSessionManager
    stop: asyncio.Event
    task: asyncio.Task[None]


class HTTPApp:
    """The ASGI application. Use `async with app.lifespan():` when you drive it yourself."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.tokens = TokenStore(home)
        self._servers: dict[tuple[str, str], _ScopeServer] = {}
        self._lock: asyncio.Lock | None = None

    # ---- lifecycle ---------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Allow scope servers to start; on exit stop them all and close their contexts."""
        self._lock = asyncio.Lock()
        try:
            yield
        finally:
            self._lock = None
            servers = list(self._servers.values())
            self._servers.clear()
            for server in servers:
                server.stop.set()
            await asyncio.gather(*(s.task for s in servers), return_exceptions=True)
            for server in servers:
                server.context.close()

    async def _lifespan_protocol(self, receive: Receive, send: Send) -> None:
        await receive()  # lifespan.startup
        stack = contextlib.AsyncExitStack()
        try:
            await stack.enter_async_context(self.lifespan())
        except Exception as exc:
            await send({"type": "lifespan.startup.failed", "message": str(exc)})
            return
        await send({"type": "lifespan.startup.complete"})
        await receive()  # lifespan.shutdown
        await stack.aclose()
        await send({"type": "lifespan.shutdown.complete"})

    async def _server_for(self, scope: Scope) -> _ScopeServer:
        key = (scope.business, scope.agent)
        found = self._servers.get(key)
        if found is not None:
            return found
        lock = self._lock
        if lock is None:
            raise RuntimeError("The HTTP app is not running. Enter `lifespan()` first.")
        async with lock:
            found = self._servers.get(key)
            if found is None:
                found = await self._start(scope)
                self._servers[key] = found
            return found

    async def _start(self, scope: Scope) -> _ScopeServer:
        # On the event-loop thread, like every tool call: the SQLite connections belong here.
        context = open_context(scope)
        stop = asyncio.Event()
        try:
            mcp_server = build_server(context)
            # Stateless: every request stands alone, which fits per-request token auth and
            # needs no session table. JSON responses: no server-initiated messages exist.
            mcp_server.streamable_http_app(
                stateless_http=True,
                json_response=True,
                max_request_body_size=MAX_BODY_BYTES,
                transport_security=TRANSPORT_SECURITY,
            )
            manager = mcp_server.session_manager
            ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()

            # `manager.run()` holds an anyio task group, which must be entered and left in
            # the same task. So one dedicated task owns it for the life of the server.
            async def run_manager() -> None:
                try:
                    async with manager.run():
                        ready.set_result(None)
                        await stop.wait()
                except Exception as exc:
                    if ready.done():
                        raise
                    ready.set_exception(exc)

            task = asyncio.create_task(run_manager())
            await ready
        except BaseException:
            stop.set()  # a manager task that already started winds down
            context.close()
            raise
        return _ScopeServer(context=context, manager=manager, stop=stop, task=task)

    # ---- requests ----------------------------------------------------------------------

    async def __call__(self, scope: ASGIScope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind == "lifespan":
            await self._lifespan_protocol(receive, send)
        elif kind == "http":
            await self._http(scope, receive, send)
        elif kind == "websocket":
            await send({"type": "websocket.close"})

    async def _http(self, scope: ASGIScope, receive: Receive, send: Send) -> None:
        path = scope["path"]
        if path == HEALTH_PATH:
            method = scope["method"]
            if method in ("GET", "HEAD"):
                await _respond(send, 200, {"status": "ok"}, head_only=method == "HEAD")
            else:
                await _respond(send, 405, {"error": "method not allowed"}, ((b"allow", b"GET"),))
            return
        if path != MCP_PATH:
            await _respond(send, 404, {"error": "not found"})
            return
        raw = _bearer_token(scope)
        who = self.tokens.lookup(raw) if raw is not None else None
        if who is None:
            await _unauthorized(send)
            return
        server = await self._server_for(who)
        # The SDK refuses bodies above MAX_BODY_BYTES with 413 before parsing them.
        await server.manager.handle_request(scope, receive, send)


def build_app(home: Path) -> HTTPApp:
    """The ASGI app for the data folder `home`. Tokens are read from `home/tokens.json`."""
    return HTTPApp(home)


def run_http(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve HTTP until interrupted. Refuses to start when no token exists."""
    home = data_home()
    if TokenStore(home).count() == 0:
        raise ConfigError(
            "No access token exists, so nobody could connect in HTTP mode. "
            "Create one with `ingatan token add --business NAME --agent NAME`."
        )
    if not is_loopback(host):
        print(
            f"ingatan: warning: listening on {host} without encryption. Put a reverse proxy "
            "with TLS or a private network (Tailscale, WireGuard) in front.",
            file=sys.stderr,
        )
    import uvicorn

    config = uvicorn.Config(
        build_app(home), host=host, port=port, log_level="warning", server_header=False
    )
    uvicorn.Server(config).run()
