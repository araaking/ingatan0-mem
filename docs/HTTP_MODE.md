# HTTP mode (FR-58)

Status: spec for implementation. Local stdio mode stays the default and does not change.

## Goal

One Ingatan server on a machine the user controls (a VPS or a home server) serves the same
memory to every device and harness the user works on. A laptop connects with a URL and a
token; it does not need the `ingatan` package installed.

## Non-goals for this version

- No sync between two databases, no offline cache, no conflict merging.
- No TLS termination inside Ingatan. The user puts it behind a reverse proxy or a private
  network (Tailscale, WireGuard). The docs say so.
- No remote mode for the Claude Code hooks or the Hermes plugin. They keep reading the local
  database. Over HTTP the agent calls `brief` itself, as the tool description already says.
- No remote CLI. `ingatan proposals`, `log`, `undo`, `forget` run on the server machine.
- No user accounts, no OAuth. Static bearer tokens only.

## Behaviour

### Starting the server

```
ingatan serve                      # stdio, unchanged
ingatan serve --http               # 127.0.0.1:8765
ingatan serve --http --host 0.0.0.0 --port 9000
```

- Transport: MCP streamable HTTP from the `mcp` SDK, at path `/mcp`.
- `GET /healthz` answers `200 {"status": "ok"}` with no auth and no other information.
- The server refuses to start in HTTP mode when no token exists. The message names
  `ingatan token add`.
- A bind host other than a loopback address prints one warning line to stderr that the
  traffic is not encrypted unless a proxy or private network is in front.
- HTTP mode does not read `.ingatan.toml`, `INGATAN_BUSINESS` or `INGATAN_AGENT`. The scope
  of each request comes from its token. `INGATAN_HOME` still selects the data folder.
- The server makes no outbound network call (NFR-3 holds in both modes).

### Tokens

A token maps to exactly one `(business, agent)` scope. This is the HTTP equivalent of
`.ingatan.toml`: the client cannot choose or change its scope.

```
ingatan token add --business kopi --agent claude-mac [--label "MacBook"]
ingatan token list
ingatan token revoke <token-id>
```

- `add` prints the token once. Format: `igt_` followed by 32 bytes from `secrets`, URL-safe
  base64 without padding. Only its SHA-256 hash is stored.
- Storage: `<INGATAN_HOME>/tokens.json`, written atomically (temp file, then `os.replace`),
  mode `0600` where the platform supports it. Each record: `id` (`t_` + 8 hex chars),
  `hash`, `business`, `agent`, `label`, `created`.
- `business` and `agent` pass `validate_business` / `validate_agent`.
- `list` shows id, business, agent, label, created. Never the token or the hash.
- `revoke` removes the record. A running server stops accepting the token on the next
  request (it re-reads the file when its mtime changes).
- The token file is not part of any business database and is never returned by a tool.

### Requests

- Every request to `/mcp` needs `Authorization: Bearer <token>`.
- Missing, malformed or unknown token: `401` with `WWW-Authenticate: Bearer` and a JSON body
  `{"error": "unauthorized"}`. The body does not say which of the three it was. Token lookup
  is by hash, compared with `hmac.compare_digest`.
- The token is never written to a log, an error message or stderr.
- A valid token routes the request to the MCP server of its scope. The seven tools, their
  descriptions and their results are identical to stdio mode. No tool gains a scope
  parameter.
- Two tokens for the same business and different agents see the same business and global
  facts and their own agent notes, exactly like two local agents today.
- Request bodies above 1 MiB are refused with `413`.

### Concurrency

- One process, one event loop. A `Context` is opened per scope on first use, on the
  event-loop thread, and reused. Contexts are closed on shutdown.
- All SQLite work stays on the event-loop thread (the tools are `async def` for this
  reason; keep it).
- The CLI on the same machine may run while the server runs (WAL already allows it).

## Code layout

- `src/ingatan/tokens/__init__.py`: `Token` record, `add`, `list_tokens`, `revoke`,
  `TokenStore` with `lookup(raw) -> Scope | None` and mtime-based reload. No import of
  `mcp`, `starlette` or `pydantic`.
- `src/ingatan/mcp/http.py`: `build_app(home) -> ASGI app` and `run_http(host, port)`.
  Auth, `/healthz`, body limit, per-scope server cache. `mcp/server.py` keeps `build_server`
  and stdio `run` as they are.
- `src/ingatan/cli/commands.py`: `--http/--host/--port` on `serve`, and the `token`
  sub-commands.
- `ingatan doctor`: one new line that reports the number of tokens (0 is fine and is not an
  error).

## Tests

- `tests/tokens/`: add/list/revoke, hash-only storage (the raw token does not occur in the
  file bytes), atomic rewrite, invalid names rejected, reload after revoke.
- `tests/mcp/test_http.py`, in process against the ASGI app (no real socket where the SDK
  allows it): 401 for missing, malformed and unknown tokens with identical bodies; healthz
  without auth; a full `initialize` + `tools/list` + `remember` + `recall` round trip with a
  valid token; two tokens with different agents do not see each other's agent facts; two
  tokens with different businesses do not see each other's business facts; a revoked token
  gets 401 on the next request; 413 for an oversized body; exactly seven tools.
- `tests/cli/`: `serve --http` with no token exits non-zero and names `ingatan token add`;
  `token add` output contains the token once; `token list` output does not contain it.

## Documentation

- `docs/setup-http.md`: server setup, token creation, a reverse-proxy or Tailscale note,
  the client configuration for Claude Code (`claude mcp add --transport http ...` with an
  `Authorization` header) and what does not work yet (hooks, Hermes plugin, remote CLI).
- `README.md`: a short section in both languages that links to it.
- `docs/PRD.md`: FR-58 marked as implemented in a first version; the non-goals above listed.
- `docs/ARCHITECTURE.md`: the two new modules.
