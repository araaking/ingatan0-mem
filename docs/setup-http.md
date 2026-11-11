# Set up Ingatan in HTTP mode

One Ingatan server on a machine you control (a VPS or a home server) serves the same memory
to every device and harness you work on. A laptop connects with a URL and a token; it does
not need the `ingatan` package.

The design is in [HTTP_MODE.md](HTTP_MODE.md) (FR-58). Stdio mode stays the default and is
unchanged: `ingatan serve` with no flag still starts the stdio server.

Marks used below follow the other setup guides: **verified** means a test in this repository
exercises it (the HTTP tests drive the server in process, with no real socket). **not
verified** means nobody ran it end to end yet: no real client, reverse proxy or Tailscale
network has been used with this server.

## 1. Install and create a token

On the server machine:

```
uv tool install ingatan          # or: pip install ingatan
ingatan token add --business kopi --agent claude-mac --label "MacBook"
```

`token add` prints the token once, on its own line. Copy it then. Only its SHA-256 hash is
stored, in `<INGATAN_HOME>/tokens.json` (default `~/.ingatan/tokens.json`), which is not part
of any business database and is never returned by a tool.

A token maps to exactly one business and one agent. It is the HTTP equivalent of
`.ingatan.toml`: the client cannot choose or change its scope. Make one token per device or
harness, so you can revoke one without touching the others.

```
ingatan token list               # id, business, agent, created, label. Never the token.
ingatan token revoke t_1a2b3c4d  # the running server rejects it from the next request on
```

Two tokens for the same business and different agents see the same business and global
facts and their own agent notes, like two local agents. `ingatan doctor` reports how many
tokens exist (zero is fine).

## 2. Start the server

```
ingatan serve --http                              # 127.0.0.1:8765, this machine only
ingatan serve --http --host 0.0.0.0 --port 9000   # all interfaces
```

- The MCP endpoint is `/mcp`. `GET /healthz` answers `200 {"status": "ok"}` without a token
  and says nothing else.
- The server refuses to start when no token exists. The message names `ingatan token add`.
- A bind host other than a loopback address prints one warning line to stderr: the traffic is
  not encrypted unless a reverse proxy or a private network is in front.
- HTTP mode does not read `.ingatan.toml`, `INGATAN_BUSINESS` or `INGATAN_AGENT`. The scope of
  each request comes from its token. `INGATAN_HOME` still selects the data folder.
- Requests above 1 MiB are refused with `413`. A missing, malformed or unknown token gets
  `401` with `WWW-Authenticate: Bearer`; the body does not say which of the three it was.
- The server makes no outbound network call.

Run it under your service manager so that it restarts, for example a systemd unit whose
`ExecStart` is `ingatan serve --http` (*not verified*).

## 3. Encryption: reverse proxy or private network

Ingatan does not terminate TLS. A bearer token sent over plain HTTP across the internet can
be read by anyone on the path. Use one of these:

- **Private network (Tailscale, WireGuard).** Keep the default `127.0.0.1` bind and let the
  network layer reach it, or bind to the address of the private interface with `--host`.
  Nothing is exposed to the public internet. (*not verified*)
- **Reverse proxy with TLS.** Keep Ingatan on `127.0.0.1` and put Caddy, nginx or similar in
  front. The proxy must pass the path `/mcp` through unchanged and must forward the
  `Authorization` header. A minimal Caddyfile: (*not verified*)

  ```
  memory.example.net {
      reverse_proxy 127.0.0.1:8765
  }
  ```

The SDK's own Host and Origin check (DNS-rebinding protection) is switched off on purpose,
because the real hostname is not known to Ingatan and every request to `/mcp` already needs
a token that a web page does not have.

## 4. Connect Claude Code

On the laptop, no `ingatan` install needed:

```
claude mcp add --transport http ingatan https://memory.example.net/mcp \
  --header "Authorization: Bearer <token>"
```

For a server on a private network use its address instead, for example
`http://100.64.0.7:8765/mcp`. Add `--scope project` or `--scope user` as you would for any
MCP server. (*not verified*: the flags follow the Claude Code documentation; no real session
has used this server yet.)

Any other MCP client that supports streamable HTTP and a custom `Authorization` header works
the same way: URL `.../mcp`, header `Authorization: Bearer <token>`.

The agent has to call `brief` itself at the start of a session, as the tool description says.

## What does not work yet

- **Hooks.** The Claude Code hooks keep reading the local database. They do not talk to the
  HTTP server, so there is no automatic brief injection and no automatic capture of turns
  over HTTP.
- **Hermes plugin.** Same: it reads the local database.
- **Remote CLI.** `ingatan proposals`, `log`, `diff`, `undo` and `forget` run on the server
  machine, with `INGATAN_HOME` pointing at the server's data folder. Business and global
  facts that agents propose over HTTP wait there for your approval.
- **Sync and offline use.** There is one database on the server. Two databases are not
  merged, and a laptop without a connection has no memory.
- **Accounts and OAuth.** Static bearer tokens only.
