"""HTTP mode, in process: the ASGI app is driven through `httpx2.ASGITransport`, no socket.

`httpx2` is the HTTP client the `mcp` SDK itself depends on. Requests are raw JSON-RPC over
the SDK's stateless streamable HTTP, so each call is one POST and the tests do not depend
on how a client negotiates a session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any, TypeVar

import pytest

try:
    import httpx2

    from ingatan.mcp.http import build_app, is_loopback
except ImportError as exc:  # pragma: no cover - only on machines that block pydantic_core
    if "Application Control" in str(exc):
        pytest.skip(f"the mcp SDK cannot be imported here: {exc}", allow_module_level=True)
    raise

from ingatan import tokens
from ingatan.scope import Context

T = TypeVar("T")
MakeContext = Callable[..., Context]
InsertFact = Callable[..., str]

TOOLS = ["brief", "recall", "remember", "supersede", "confirm", "forget", "history"]
MAX_BODY = 1024 * 1024
JSON_HEADERS = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
}
UNAUTHORIZED = {"error": "unauthorized"}


def run_async(awaitable: Awaitable[T], timeout: float = 90) -> T:
    async def guarded() -> T:
        return await asyncio.wait_for(awaitable, timeout)

    return asyncio.run(guarded())


@contextlib.asynccontextmanager
async def serving(home: Path) -> AsyncIterator[httpx2.AsyncClient]:
    """The app with its lifespan running, and a client wired straight to it."""
    app = build_app(home)
    async with app.lifespan():
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


async def rpc(
    client: httpx2.AsyncClient,
    token: str,
    method: str,
    params: dict[str, Any] | None = None,
    request_id: int | None = 1,
) -> httpx2.Response:
    body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        body["id"] = request_id
    return await client.post(
        "/mcp", content=json.dumps(body).encode(), headers={**JSON_HEADERS, **bearer(token)}
    )


async def initialize(client: httpx2.AsyncClient, token: str) -> dict[str, Any]:
    response = await rpc(
        client,
        token,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "ingatan-test", "version": "0"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def list_tools(client: httpx2.AsyncClient, token: str) -> list[dict[str, Any]]:
    response = await rpc(client, token, "tools/list")
    assert response.status_code == 200, response.text
    return response.json()["result"]["tools"]


async def tool(client: httpx2.AsyncClient, token: str, name: str, arguments: dict[str, Any]) -> str:
    response = await rpc(client, token, "tools/call", {"name": name, "arguments": arguments})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert not result.get("isError"), result
    return "\n".join(part["text"] for part in result["content"] if part["type"] == "text")


# ---------------------------------------------------------------- authentication


def test_missing_malformed_and_unknown_tokens_get_the_same_401(ingatan_home: Path) -> None:
    valid, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    unknown = tokens.new_secret()

    async def go() -> list[httpx2.Response]:
        async with serving(ingatan_home) as client:
            attempts: list[dict[str, str]] = [
                {},  # missing
                {"authorization": ""},
                {"authorization": "Bearer"},
                {"authorization": "Bearer "},
                {"authorization": f"Bearer {valid} extra"},
                {"authorization": f"Basic {valid}"},  # right secret, wrong scheme
                {"authorization": f"Token {valid}"},
                {"authorization": valid},  # no scheme at all
                {"authorization": f"Bearer {'x' * 5000}"},  # absurdly long
                {"authorization": f"Bearer {unknown}"},  # well formed, not stored
                {"authorization": f"Bearer {valid[:-1]}"},  # one character short
            ]
            return [
                await client.post("/mcp", content=b"{}", headers={**JSON_HEADERS, **headers})
                for headers in attempts
            ]

    responses = run_async(go())
    assert len(responses) == 11
    for response in responses:
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["content-type"] == "application/json"
        assert response.json() == UNAUTHORIZED
    assert len({r.content for r in responses}) == 1  # byte for byte the same body


def test_every_method_on_mcp_needs_a_token(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> list[int]:
        async with serving(ingatan_home) as client:
            return [
                (await client.request(method, "/mcp")).status_code
                for method in ("GET", "POST", "DELETE", "OPTIONS", "PUT")
            ]

    assert run_async(go()) == [401] * 5


def test_a_server_with_no_tokens_lets_nobody_in(ingatan_home: Path) -> None:
    async def go() -> httpx2.Response:
        async with serving(ingatan_home) as client:
            return await rpc(client, tokens.new_secret(), "tools/list")

    assert run_async(go()).status_code == 401


def test_healthz_answers_without_a_token_and_says_nothing_else(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> tuple[httpx2.Response, httpx2.Response, httpx2.Response]:
        async with serving(ingatan_home) as client:
            return (
                await client.get("/healthz"),
                await client.head("/healthz"),
                await client.post("/healthz"),
            )

    got, head, posted = run_async(go())
    assert got.status_code == 200
    assert got.json() == {"status": "ok"}
    assert head.status_code == 200 and head.content == b""
    assert posted.status_code == 405


def test_other_paths_are_404_without_leaking_anything(ingatan_home: Path) -> None:
    async def go() -> list[httpx2.Response]:
        async with serving(ingatan_home) as client:
            return [await client.get(path) for path in ("/", "/mcp/", "/admin", "/tokens.json")]

    for response in run_async(go()):
        assert response.status_code == 404
        assert response.json() == {"error": "not found"}


# ---------------------------------------------------------------- tools over HTTP


def test_a_full_round_trip_with_a_valid_token(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> tuple[dict[str, Any], list[dict[str, Any]], str, str, int]:
        async with serving(ingatan_home) as client:
            info = await initialize(client, raw)
            note = await rpc(client, raw, "notifications/initialized", request_id=None)
            tools_listed = await list_tools(client, raw)
            saved = await tool(
                client,
                raw,
                "remember",
                {
                    "text": "Harga kopi susu naik jadi Rp 18.000 mulai Senin",
                    "keywords": ["harga", "kopi", "susu", "price"],
                },
            )
            found = await tool(client, raw, "recall", {"query": "harga kopi susu"})
            return info, tools_listed, saved, found, note.status_code

    info, listed, saved, found, note_status = run_async(go())
    assert info["serverInfo"]["name"] == "ingatan"
    assert note_status == 202
    assert sorted(t["name"] for t in listed) == sorted(TOOLS)
    assert saved.startswith("Saved as f_")
    assert "Harga kopi susu naik jadi Rp 18.000 mulai Senin" in found
    assert "source: claude-mac" in found
    assert "status: valid" in found


def test_exactly_seven_tools_and_no_scope_parameters(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> list[dict[str, Any]]:
        async with serving(ingatan_home) as client:
            return await list_tools(client, raw)

    listed = run_async(go())
    assert len(listed) == 7
    assert sorted(t["name"] for t in listed) == sorted(TOOLS)
    for entry in listed:
        assert entry["description"]
        schema_text = json.dumps(entry["inputSchema"]).lower()
        for word in ("business", "agent", "tenant", "scope"):
            assert word not in schema_text, entry["name"]


def test_two_agents_of_one_business_do_not_see_each_others_agent_facts(
    ingatan_home: Path,
) -> None:
    first, _ = tokens.add(ingatan_home, "kopi", "agent-one")
    second, _ = tokens.add(ingatan_home, "kopi", "agent-two")

    async def go() -> tuple[str, str, str, str]:
        async with serving(ingatan_home) as client:
            await tool(
                client,
                first,
                "remember",
                {"text": "Catatan pribadi satu: pemasok zebraone", "keywords": ["zebraone"]},
            )
            await tool(
                client,
                second,
                "remember",
                {"text": "Catatan pribadi dua: pemasok zebratwo", "keywords": ["zebratwo"]},
            )
            return (
                await tool(client, first, "recall", {"query": "zebraone"}),
                await tool(client, first, "recall", {"query": "zebratwo"}),
                await tool(client, second, "recall", {"query": "zebratwo"}),
                await tool(client, second, "recall", {"query": "zebraone"}),
            )

    own_one, other_one, own_two, other_two = run_async(go())
    assert "zebraone" in own_one and "zebratwo" not in own_one
    assert "No facts found" in other_one and "zebratwo" not in other_one
    assert "zebratwo" in own_two
    assert "No facts found" in other_two and "zebraone" not in other_two


def test_business_facts_are_shared_per_business_and_global_facts_by_all(
    ingatan_home: Path, make_ctx: MakeContext, insert_fact: InsertFact
) -> None:
    kopi_a, _ = tokens.add(ingatan_home, "kopi", "agent-one")
    kopi_b, _ = tokens.add(ingatan_home, "kopi", "agent-two")
    teh_a, _ = tokens.add(ingatan_home, "teh", "agent-one")
    kopi_ctx = make_ctx(agent="agent-one", business="kopi")
    insert_fact(kopi_ctx.business, text="Fakta bisnis kopi: cabang giraffekopi", tier="business")
    insert_fact(kopi_ctx.glob, text="Fakta global: kebiasaan okapiglobal", tier="global")

    async def go() -> dict[str, str]:
        async with serving(ingatan_home) as client:
            return {
                "kopi_a business": await tool(client, kopi_a, "recall", {"query": "giraffekopi"}),
                "kopi_b business": await tool(client, kopi_b, "recall", {"query": "giraffekopi"}),
                "teh_a business": await tool(client, teh_a, "recall", {"query": "giraffekopi"}),
                "kopi_a global": await tool(client, kopi_a, "recall", {"query": "okapiglobal"}),
                "teh_a global": await tool(client, teh_a, "recall", {"query": "okapiglobal"}),
            }

    got = run_async(go())
    assert "giraffekopi" in got["kopi_a business"]
    assert "giraffekopi" in got["kopi_b business"]
    assert "giraffekopi" not in got["teh_a business"]
    assert "No facts found" in got["teh_a business"]
    assert "okapiglobal" in got["kopi_a global"]
    assert "okapiglobal" in got["teh_a global"]


def test_the_scope_comes_from_the_token_only(ingatan_home: Path) -> None:
    """Query strings, headers and body fields cannot move a request to another scope."""
    raw, _ = tokens.add(ingatan_home, "kopi", "agent-one")
    tokens.add(ingatan_home, "teh", "agent-two")

    async def go() -> str:
        async with serving(ingatan_home) as client:
            await tool(
                client,
                raw,
                "remember",
                {"text": "Rahasia kopi: pemasok lemurkopi", "keywords": ["lemurkopi"]},
            )
            response = await client.post(
                "/mcp?business=teh&agent=agent-two",
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "recall",
                            "arguments": {"query": "lemurkopi"},
                            "business": "teh",
                            "agent": "agent-two",
                        },
                    }
                ).encode(),
                headers={
                    **JSON_HEADERS,
                    **bearer(raw),
                    "x-ingatan-business": "teh",
                    "x-ingatan-agent": "agent-two",
                },
            )
            assert response.status_code == 200, response.text
            return response.text

    assert "lemurkopi" in run_async(go())


def test_a_revoked_token_gets_401_on_the_next_request(ingatan_home: Path) -> None:
    keep, _ = tokens.add(ingatan_home, "kopi", "agent-one")
    doomed, record = tokens.add(ingatan_home, "kopi", "agent-two")

    async def go() -> tuple[int, int, int]:
        async with serving(ingatan_home) as client:
            before = await rpc(client, doomed, "tools/list")
            tokens.revoke(ingatan_home, record.id)
            after = await rpc(client, doomed, "tools/list")
            kept = await rpc(client, keep, "tools/list")
            return before.status_code, after.status_code, kept.status_code

    assert run_async(go()) == (200, 401, 200)


def test_a_token_created_while_the_server_runs_works(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "agent-one")

    async def go() -> int:
        async with serving(ingatan_home) as client:
            late, _ = tokens.add(ingatan_home, "teh", "agent-late")
            return (await rpc(client, late, "tools/list")).status_code

    assert run_async(go()) == 200


def test_a_body_above_one_mib_is_refused_with_413(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    padding = " " * (MAX_BODY + 1)

    async def go() -> tuple[int, int]:
        async with serving(ingatan_home) as client:
            big = await client.post(
                "/mcp",
                content=('{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' + padding).encode(),
                headers={**JSON_HEADERS, **bearer(raw)},
            )
            small = await rpc(client, raw, "tools/list")
            return big.status_code, small.status_code

    assert run_async(go()) == (413, 200)


def test_an_oversized_body_without_a_token_is_still_401(ingatan_home: Path) -> None:
    tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> int:
        async with serving(ingatan_home) as client:
            response = await client.post(
                "/mcp", content=b" " * (MAX_BODY + 1), headers=JSON_HEADERS
            )
            return response.status_code

    assert run_async(go()) == 401


def test_a_non_loopback_host_header_is_not_rejected_by_the_sdk(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> list[int]:
        async with serving(ingatan_home) as client:
            statuses: list[int] = []
            for host in ("memory.example.net", "100.64.0.7:8765", "10.0.0.5"):
                response = await client.post(
                    "/mcp",
                    content=b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}',
                    headers={**JSON_HEADERS, **bearer(raw), "host": host},
                )
                statuses.append(response.status_code)
            return statuses

    assert run_async(go()) == [200, 200, 200]


def test_the_token_never_appears_in_a_response(ingatan_home: Path) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")

    async def go() -> list[httpx2.Response]:
        async with serving(ingatan_home) as client:
            return [
                await rpc(client, raw, "tools/list"),
                await rpc(client, raw, "no/such/method"),
                await client.post(
                    "/mcp", content=b"not json", headers={**JSON_HEADERS, **bearer(raw)}
                ),
                await client.post("/mcp", content=b"{}", headers=bearer(f"{raw}x")),
            ]

    for response in run_async(go()):
        assert raw not in response.text
        assert raw not in json.dumps(dict(response.headers))


def test_the_context_of_a_scope_is_opened_once_and_closed_on_shutdown(
    ingatan_home: Path,
) -> None:
    raw, _ = tokens.add(ingatan_home, "kopi", "claude-mac")
    app = build_app(ingatan_home)

    async def go() -> tuple[int, int]:
        async with app.lifespan():
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(transport=transport, base_url="http://t") as client:
                for _ in range(3):
                    assert (await rpc(client, raw, "tools/list")).status_code == 200
            running = len(app._servers)  # pyright: ignore[reportPrivateUsage]
        return running, len(app._servers)  # pyright: ignore[reportPrivateUsage]

    assert run_async(go()) == (1, 0)


def test_the_lifespan_protocol_starts_and_stops_cleanly(ingatan_home: Path) -> None:
    """What uvicorn does: a `lifespan` scope, startup, then shutdown."""
    app = build_app(ingatan_home)

    async def go() -> list[str]:
        sent: list[str] = []
        inbox: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        await inbox.put({"type": "lifespan.startup"})

        async def receive() -> dict[str, str]:
            return await inbox.get()

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(message["type"])
            if message["type"] == "lifespan.startup.complete":
                await inbox.put({"type": "lifespan.shutdown"})

        await app({"type": "lifespan"}, receive, send)
        return sent

    assert run_async(go()) == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.5.5.5", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("100.64.0.7", False),
        ("memory.example.net", False),
        ("", False),
    ],
)
def test_is_loopback(host: str, expected: bool) -> None:
    assert is_loopback(host) is expected
