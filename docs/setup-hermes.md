# Set up Ingatan in Hermes Agent

Two pieces: a memory-provider plugin (captures turns, puts the brief in the system prompt)
and the MCP server (the seven tools). Hermes allows one external memory provider at a time;
Ingatan takes that slot. The built-in `MEMORY.md` / `USER.md` memory stays on. The MCP server
does not use the slot, so Ingatan keeps working through MCP alone if you ever switch provider.

**Read the list at the end first: nothing here has run against a real Hermes yet.**

## 1. Install Ingatan where Hermes can import it

The plugin imports the `ingatan` package, so install it into the Python environment that runs
Hermes (and keep an `ingatan` command available for the MCP server):

```
pip install ingatan      # or: uv tool / pip install from a checkout
```

## 2. Copy the plugin

The plugin folder ships inside the package: `ingatan/integrations/hermes/` with `plugin.yaml`
and `__init__.py`. Find it and copy it:

```
python -c "import ingatan.integrations.hermes as h, pathlib; print(pathlib.Path(h.__file__).parent)"
cp -r <that folder> $HERMES_HOME/plugins/ingatan      # default HERMES_HOME is ~/.hermes
```

Remove any `__pycache__` in the copy. From a source checkout the folder is
`src/ingatan/integrations/hermes/`.

## 3. Choose the provider and the scope

In `$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: ingatan
```

or `hermes config set memory.provider ingatan`, or `hermes memory setup`. Check with
`hermes memory status`.

The plugin reads the scope from the environment of the Hermes process:

```
INGATAN_BUSINESS=tokoku        # required
INGATAN_AGENT=hermes           # optional, default "hermes"
INGATAN_HOME=...               # optional, default ~/.ingatan
```

Without `INGATAN_BUSINESS` (and no `.ingatan.toml` in Hermes' working folder) `is_available()`
returns false and the plugin does nothing.

## 4. The MCP server

Hermes passes an MCP server only the `env` you list plus a safe baseline, not your shell
environment, so the scope must be in the block. In `$HERMES_HOME/config.yaml`:

```yaml
mcp_servers:
  ingatan:
    command: ingatan
    args: ["serve"]
    env:
      INGATAN_BUSINESS: tokoku
      INGATAN_AGENT: hermes
      # INGATAN_HOME: /home/you/.ingatan   # must match the Hermes process if you set it there
    enabled: true
```

Apply with `/reload-mcp` and test with `hermes mcp test ingatan`. The tools appear with a
prefix: `mcp_ingatan_brief`, `mcp_ingatan_recall`, `mcp_ingatan_remember`,
`mcp_ingatan_supersede`, `mcp_ingatan_confirm`, `mcp_ingatan_forget`, `mcp_ingatan_history`.
The plugin and the server must use the same business, agent and data folder: the plugin writes
episodes, the server's `brief` and `remember(..., episode=<id>)` read and close them.

## What the plugin does

- `sync_turn(user_content, assistant_content, ...)`: records one episode (redacted, cut at
  2000 characters each). `messages` is never read, so no tool output is stored. Skipped when
  `initialize` said `agent_context` is `subagent`, `cron` or `flush`. Hermes already runs this
  hook on its own worker, so the write happens inline.
- `system_prompt_block()`: the brief, built once per session. Subagent, cron and resumed
  sessions get the brief without the "episodes to digest" section.
- `get_tool_schemas()`: `[]`. `on_session_end()`: does nothing (Hermes discards its result).
- Every method accepts `**kwargs` and none raises. Failures go to
  `<INGATAN_HOME>/capture-errors.log`.

## Not yet verified against a real Hermes install

From HOOK_FINDINGS.md section A and the "Not verified" summary. All of the plugin is written
from upstream docs and source read on the web, plus third-party plugins:

- Everything end to end: that Hermes loads this plugin, calls these methods with these
  arguments, and that a session records episodes and receives the brief.
- The installed Hermes version (claude-mem asks for `>=0.21`; `plugin.yaml` sets no
  `requires_hermes`).
- What `kind: exclusive` in `plugin.yaml` does. Only the one-provider rule in
  `MemoryManager.add_provider` is verified.
- Whether Hermes puts the plugin folder on `sys.path` under the name `ingatan`. If it does,
  `import ingatan` inside the plugin would find the plugin folder instead of the installed
  package and the plugin would silently stay idle (see `capture-errors.log` or run
  `hermes memory status`). Rename the copied folder if that happens.
- Whether `initialize` really receives `agent_context` and `resumed` (read from upstream docs;
  older Hermes may omit them, and the plugin then treats the session as primary).
- Hermes on Windows and its default `HERMES_HOME` there. The owner runs Hermes on a VPS.
- That MCP tool names are exactly `mcp_<server>_<tool>`, and that `env` is the only way the
  server gets its scope. `system_prompt_block()` being called once per session is also from
  the docs; the plugin caches its result in case it is called more often.
- `sync_turn` can arrive after the next turn started (seen in claude-mem). Ordering inside
  Ingatan is by insertion, so a rare swap is possible.
- Python 3.10+ is assumed to be what Hermes runs under.
