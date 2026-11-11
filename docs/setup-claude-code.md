# Set up Ingatan in Claude Code

Two pieces: the MCP server (the seven tools) and the hooks (capture and brief injection).
The server alone works. The hooks add the automatic brief at session start (FR-47) and the
safety-net capture of turns (FR-44).

Marks used below: **verified** means HOOK_FINDINGS.md read it in the Claude Code hook docs
(https://code.claude.com/docs/en/hooks) or in a real hooks file. **not verified** means it
was not checked, or it is standard behaviour that nobody tested with Ingatan on this machine.
No real Claude Code session has run these hooks yet (PLAN T-19, first checkbox).

## 1. Install and give the project a scope

```
uv tool install ingatan          # or: pip install ingatan
```

*not verified*: the package is not on PyPI yet. From a checkout use
`uv tool install D:\path\to\ingatan0-mem`. The hooks need an `ingatan` command on the `PATH`
that Claude Code starts (whether a bare command name is found on Windows is *not verified*).

In the project folder create `.ingatan.toml` (`ingatan init` writes it):

```toml
business = "tokoku"
agent = "claude-code"
```

The hooks read it from the `cwd` field Claude Code sends, walking up to the nearest parent.
`INGATAN_BUSINESS` and `INGATAN_AGENT` override each field. A project with no scope at all is
left alone: the hooks print nothing and exit 0.

## 2. The MCP server

Project file `.mcp.json` (shared with the team, or use `--scope local` for yourself):

```json
{
  "mcpServers": {
    "ingatan": {
      "command": "ingatan",
      "args": ["serve"]
    }
  }
}
```

Or from a terminal:

```
claude mcp add --scope project ingatan -- ingatan serve
```

Add `"env": { "INGATAN_BUSINESS": "tokoku", "INGATAN_AGENT": "claude-code" }` if you do not
use `.ingatan.toml`. *not verified*: the `.mcp.json` shape and the `claude mcp add` flags are
Claude Code's standard ones and were not fetched in HOOK_FINDINGS.md. Check with `/mcp` inside
Claude Code. *not verified*: the exact tool prefix Claude Code shows (`mcp__ingatan__brief`
is the usual form).

## 3. The hooks

Put this in `~/.claude/settings.json` (all projects) or `.claude/settings.json` (one project):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact|fork",
        "hooks": [
          { "type": "command", "command": "ingatan", "args": ["hook", "session-start"], "timeout": 30 }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "ingatan", "args": ["hook", "user-prompt"], "timeout": 15 }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "ingatan", "args": ["hook", "stop"], "timeout": 30, "async": true }
        ]
      }
    ]
  }
}
```

What each one does:

| Event | Command | Effect |
|---|---|---|
| SessionStart | `ingatan hook session-start` | Prints `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "<brief>"}}`. On `startup`, `clear` and `fork` the brief ends with up to three episodes to digest. On `compact` and `resume` the brief is injected again without the episodes. |
| UserPromptSubmit | `ingatan hook user-prompt` | Stores the redacted prompt in `<data home>/pending/<session_id>.json`. Prints nothing. |
| Stop | `ingatan hook stop` | Pairs the stored prompt with `last_assistant_message` and records one episode. Prints nothing. Skipped when `stop_hook_active` is true or no prompt is stored. |

Stored text: only the user prompt and Claude's final reply, redacted and cut at 2000
characters each. Tool calls, tool results, thinking and file contents are never read. If
`last_assistant_message` is missing, the hook falls back to the last text block of the last
assistant line of `transcript_path`.

Every hook exits 0 and prints nothing but the output above. A failure (locked database, bad
input, no brief) is appended to `<data home>/capture-errors.log` and the session goes on.
A project without `.ingatan.toml` and without the env variables is skipped silently.

### Verified

- Events, stdin fields (`session_id`, `cwd`, `source`, `prompt`, `last_assistant_message`,
  `stop_hook_active`, `transcript_path`), and that SessionStart cannot block.
- SessionStart accepts plain stdout or the JSON `hookSpecificOutput.additionalContext` form
  (Ingatan uses the JSON form). Matcher values: `startup`, `resume`, `clear`, `compact`, `fork`.
- Stop fires once per turn, not on an interrupt. `async: true` runs a hook in the background
  without enforcing `timeout`. The `command` plus `args` form is used by claude-mem.
- Exit 0 never blocks; exit 2 does on some events, which is why these hooks never use it.
- Transcript recipe for the last assistant text (one local transcript).

### Not verified

- A real Claude Code session end to end: episodes without tool output, brief in context.
- Stdin encoding of hook processes on Windows. The hooks read bytes and decode UTF-8
  (invalid bytes replaced, a BOM dropped); tests cover this with Indonesian text but not on a
  real Windows hook run.
- Whether a bare `ingatan` (or `uv run ...`) works as `command` on Windows.
- Whether SessionStart output has a size cap, so a large brief might be cut.
- How to tell a `UserPromptSubmit` that Claude Code started on its own from a typed prompt:
  such prompts are stored like any other.
- Where settings files live and how hooks merge across them. If both a user and a project file
  declare the hooks, they may run twice; keep them in one place.
- `async: true` on Stop: the docs say it is supported, but nobody has measured whether the
  pending-prompt file of the next turn can be written before a slow Stop has read the old one
  (pairing is per session id, so a rare mispair is possible).
