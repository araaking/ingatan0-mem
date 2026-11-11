# Hook findings (PLAN task T-2)

Date: 2026-11-11. Purpose: let an implementer build the Claude Code hooks (T-19) and the Hermes memory provider (T-20) without guessing.

How to read the marks:

- **verified (local)**: seen in a file on this machine, path given.
- **verified (web)**: seen on a documentation page or in upstream source through a fetch tool, URL given. The fetch tool returned a summary of the page, not a byte-exact copy, so for exact signatures re-read the upstream file before coding.
- **verified against plugin X, not against Hermes itself**: seen only in a third-party plugin.
- **not verified**: could not check. Do not rely on it.

Hermes was not run and no Hermes source tree exists on this machine. Everything about Hermes is therefore from upstream docs and source on the web, plus third-party plugins under `D:\memory-resource`. Claude Code hooks were not run either; they come from the official docs plus one local transcript.

## A. Hermes Agent

### A1. Memory provider interface

Base class `MemoryProvider` in `agent/memory_provider.py`. Import: `from agent.memory_provider import MemoryProvider`.

- **verified (web)** https://raw.githubusercontent.com/NousResearch/hermes-agent/main/agent/memory_provider.py
- **verified against plugin claude-mem, not against Hermes itself**: `D:\memory-resource\claude-mem\hermes\claude-mem\__init__.py` line 25 imports it and line 101 subclasses it. `D:\memory-resource\honcho\hermes-plugin-honcho\__init__.py` does the same.
- Note: `D:\memory-resource\basic-memory\integrations\hermes\tests\stubs\agent\memory_provider.py` is only a test stub, an empty class. It proves nothing about the interface.

Signatures (verified (web) from the file above; cross-checked against claude-mem `__init__.py` and honcho `__init__.py` where noted):

| Member | Signature | Required | Called |
|---|---|---|---|
| `name` | `@property name(self) -> str` | abstract | always |
| `is_available` | `is_available(self) -> bool` | abstract | at startup, before activation. No network calls |
| `initialize` | `initialize(self, session_id: str, **kwargs) -> None` | abstract | once at agent startup |
| `get_tool_schemas` | `get_tool_schemas(self) -> list[dict]` | abstract | after init. OpenAI function-calling format. Return `[]` for none |
| `system_prompt_block` | `system_prompt_block(self) -> str` | optional | while the system prompt is built. Empty string skips. Static text, not recalled context |
| `prefetch` | `prefetch(self, query: str, *, session_id: str = "") -> str` | optional | before each turn. Must be fast. Empty string if nothing |
| `queue_prefetch` | `queue_prefetch(self, query, *, session_id="") -> None` | optional | after a turn, to warm the next prefetch |
| `sync_turn` | `sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: Optional[list[dict]] = None, turn_author: Optional[dict] = None) -> None` | optional | after each completed turn. Must not block |
| `on_turn_start` | `on_turn_start(self, turn_number: int, message: str, **kwargs) -> None` | optional | start of each turn |
| `on_session_end` | `on_session_end(self, messages: list[dict]) -> None` | optional | real session boundaries only, not per turn |
| `on_session_switch` | `on_session_switch(self, new_session_id: str, *, parent_session_id="", reset=False, rewound=False, **kwargs) -> None` | optional | session id changes without teardown |
| `on_pre_compress` | `on_pre_compress(self, messages) -> str` | optional | before context compression. The returned string feeds the summary prompt |
| `on_memory_write` | `on_memory_write(self, action, target, content, metadata=None) -> None` | optional | after the built-in memory tool writes |
| `handle_tool_call` | `handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str` | optional | when the agent calls a provider tool. Returns a JSON string |
| `shutdown` | `shutdown(self) -> None` | optional | clean exit. Flush and close |
| `get_config_schema` / `save_config(values, hermes_home)` | | optional | `hermes memory setup` |

Other optional methods listed upstream: `unavailable_reason`, `recall_status`, `identity_signature`, `on_delegation`, `backup_paths`, `on_profile_created`, `on_profile_renamed`, `read_config_fields`, `write_config_fields` (verified (web), same file; signatures not read).

`initialize` kwargs (verified (web), same file; read them with `kwargs.get`, older Hermes may omit some): `hermes_home` (always), `platform` (always), `agent_context` (`primary`, `subagent`, `cron`, `flush`; non-primary should skip writes), `agent_identity`, `agent_workspace`, `parent_session_id`, `user_id`, `user_id_alt`, `resumed`, `message_count`. claude-mem additionally reads `cwd` (`__init__.py` line 124) which is not in the upstream list I read: **not verified** for Hermes itself.

Behaviours that matter for capture:

- `sync_turn` runs on a single background worker so writes stay ordered; it receives optional keyword args only if the signature accepts them. verified (web) https://raw.githubusercontent.com/NousResearch/hermes-agent/main/agent/memory_manager.py. Declare `**kwargs` so new keywords do not break the plugin. claude-mem comment at `__init__.py` lines 170-173 says `sync_turn` can run after the next turn started: verified against plugin claude-mem, not against Hermes itself.
- Threads: upstream guide says spawn them with `agent.memory_provider.spawn_context_thread`, not bare `threading.Thread`, or the thread loses the profile context. verified (web) https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
- Storage must use the `hermes_home` kwarg, not a hardcoded `~/.hermes`. verified (web), same page.
- Shutdown budget: claude-mem comment says Hermes drains its queue for 5 seconds (`__init__.py` line 227). verified against plugin claude-mem, not against Hermes itself.
- Provider `register(ctx)` is the entry point and calls `ctx.register_memory_provider(Provider())`. verified (web) developer-guide page above; verified against plugin claude-mem at `__init__.py` lines 253-258. Build a new instance per call, not a singleton (web).

### A2. Manifest, install location, enabling

- Layout: `__init__.py` (provider class and `register(ctx)`), `plugin.yaml`, optional `README.md`, `cli.py`, `config_schema.py`. verified (web) developer-guide page.
- Fields seen in real manifests: `name`, `version`, `description`, `author`, `kind: exclusive`, `requires_hermes: ">=0.21"`, `hooks: [...]`. verified against plugin claude-mem: `D:\memory-resource\claude-mem\hermes\claude-mem\plugin.yaml` lines 1-8.
- basic-memory's manifest adds `manifest_version: 1`, `api_version: 1`, `python_dependencies`, and says released Hermes installers refuse `manifest_version` above 1 while the runtime loader reads the v2 fields. verified against plugin basic-memory, not against Hermes itself: `D:\memory-resource\basic-memory\integrations\hermes\plugin.yaml` lines 4-11 (it cites NousResearch/hermes-agent#85893, which I did not open). Use `manifest_version: 1`.
- Meaning of `kind: exclusive` inside plugin.yaml: **not verified**. The plugins docs page mentions kinds standalone, backend, exclusive but I could not see where Hermes reads the value. What is verified is the rule in code (A3).
- Install paths (verified (web) developer-guide page): bundled `plugins/memory/<name>/`; user `$HERMES_HOME/plugins/<name>/` (default `~/.hermes/plugins/<name>/`, claude-mem README line 36 agrees: verified against plugin claude-mem); project `./.hermes/plugins/<name>/` only with `HERMES_ENABLE_PROJECT_PLUGINS=1`; or a pip package exposing a `hermes_agent.memory_providers` entry point.
- Enable: set `memory.provider: <name>` in `$HERMES_HOME/config.yaml`, or `hermes config set memory.provider <name>`, or `hermes memory setup`. Check with `hermes memory status`; turn off with `hermes memory off`. verified (web) https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers ; the key is also seen in claude-mem README lines 39-41 and `__init__.py` lines 8-11, honcho `README.md` line 30, Memori README line 198 (verified against those plugins).
- Memory providers do not need `plugins.enabled`; `memory.provider` controls activation. verified (web) https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins
- On Windows the default `HERMES_HOME`: **not verified** (Hermes on Windows was not checked at all; the owner's Hermes runs on a VPS).

### A3. The exclusive rule (OQ-1)

- Only one external memory provider is active at a time. The built-in memory (`MEMORY.md`, `USER.md`) stays active next to it. verified (web) https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers ("Only one external provider can be active at a time").
- Enforcement: `MemoryManager.add_provider` always accepts the built-in provider and rejects a second external one with a warning. verified (web) https://raw.githubusercontent.com/NousResearch/hermes-agent/main/agent/memory_manager.py ; the developer guide says the same.
- So Ingatan as a memory provider plus the built-in memory: allowed. Ingatan as a memory provider plus claude-mem, honcho, mem0, etc.: not allowed.
- Ingatan as an MCP server is not a memory provider and does not use that slot (A5).

### A4. Can `on_session_end()` return work for the agent?

No. The signature returns `None`, and the manager calls it through `_each_provider` and discards the result; failures are logged and swallowed. verified (web) memory_manager.py URL above and developer guide. claude-mem's `on_session_end` also returns nothing (`__init__.py` line 210: verified against plugin claude-mem). The only provider hook with a string return that Hermes uses is `on_pre_compress`, and it feeds the compression summary, not the agent's next task.

### A5. Hermes and an external stdio MCP server

- File: `~/.hermes/config.yaml` (that is `$HERMES_HOME/config.yaml`), key `mcp_servers`, one entry per server name. verified (web) https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
- Stdio keys: `command`, `args`, `env` (Hermes passes only what you list plus a safe baseline, not your full shell environment), `cwd`, `timeout`, `connect_timeout`, `enabled`, `lazy`, `idle_timeout_seconds`, `max_lifetime_seconds`, `supports_parallel_tool_calls`, `tools: {include, exclude, prompts, resources}`. verified (web), same page.
- Real example: claude-mem writes `mcp_servers.claude-mem` with `command`, `args`, `enabled: true`: `D:\memory-resource\claude-mem\hermes\claude-mem\README.md` lines 42-48 (verified against plugin claude-mem).
- Tool names become `mcp_<server>_<tool>`, e.g. `mcp_ingatan_brief`. verified (web), same page. The prompts and docs for the agent must use that prefixed name.
- Consequence for scope: because `env` is not inherited, the Ingatan business and agent names must be set in the `env` block (or in the config file the server reads). Inference from the above, not tested.
- `/reload-mcp` applies config changes; `hermes mcp test <name>` checks a server (web, same page).

## B. Claude Code hooks

All of B is verified (web) from https://code.claude.com/docs/en/hooks (fetched in three parts; the SessionStart/UserPromptSubmit text sat in characters 100000-200000, Stop and SessionEnd in 200000+). The full text of the part with Stop is saved locally at a local copy of the fetched page (line numbers below refer to it, and it is a temp file, not part of the project). Real usage: `D:\memory-resource\claude-mem\plugin\hooks\hooks.json` (verified (local)).

### B1. SessionStart

- Stdin JSON: common fields `session_id`, `transcript_path`, `cwd`, `hook_event_name`, `permission_mode`, `prompt_id`, plus `source` (`startup`, `resume`, `clear`, `compact`, `fork`) and optional `model`, `agent_type`, `session_title`. On `resume` and `fork` some extra cache fields are included.
- Matcher values: `startup`, `resume`, `clear`, `compact`, and `fork` (the page lists five; the task listed four).
- Inject context, two ways: plain text on stdout with exit 0 is added as context Claude sees; or JSON `{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"..."}}`. Other output fields: `initialUserMessage`, `sessionTitle`, `watchPaths`, `reloadSkills`.
- Cannot block. Exit 2 only shows stderr to the user.
- claude-mem runs its context hook on `startup|clear|compact`, not on `resume` (`hooks.json` lines 35-48: verified (local)). That is a choice, not a rule.
- Whether injected context is capped in size: **not verified**.

### B2. UserPromptSubmit

- Stdin JSON: common fields plus `prompt` (the submitted text), `session_title` if set. Example in the doc has `session_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, `prompt`.
- Also fires for turns Claude Code starts on its own, so `prompt` may not be typed by a human (web). How to tell the two apart: **not verified**.
- Can block (exit 2, or `decision: "block"`). Can inject with stdout or `additionalContext`. Default timeout 30 s for command hooks on this event.

### B3. Stop and SessionEnd

- Stop fires when the main agent finishes responding, so once per turn, not once per session. It does not fire on a user interrupt; API errors fire `StopFailure` instead (doc lines 202-206). This confirms DESIGN.md.
- Stop stdin: common fields plus `stop_hook_active`, `last_assistant_message`, `background_tasks`, `session_crons` (doc line 214). **`last_assistant_message` exists** and holds the text of Claude's final response; the docs tell hooks to use it instead of reading the transcript because the transcript file may not yet include the final message (doc lines 218). `stop_hook_active` is `true` when Claude Code is already continuing because of a stop hook.
- Hand work back: yes. `{"decision":"block","reason":"..."}` on stdout (exit 0), or exit 2 with the reason on stderr, keeps Claude working; `hookSpecificOutput.additionalContext` also continues the turn but is labelled as hook feedback, not an error (doc lines 276-295). Loop protection: after 8 consecutive continuations Claude Code overrides the block (cap changeable with `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`; resets when Claude calls a tool). Check `stop_hook_active` to avoid blocking forever.
- SessionEnd: stdin has `reason` (`clear`, `resume`, `logout`, `prompt_input_exit`, `other`). It cannot block and cannot hand work back; exit 2 only shows stderr to the user. Its hooks share a 1.5 second budget by default; a per-hook `timeout` raises it, up to 60 s. It does not run reliably on a crash or kill: **not verified** (no statement read either way).

### B4. Transcript JSONL, last assistant text

Verified (local) by counting block types in one real transcript from this project (`~/.claude/projects/<project>/<session>.jsonl`, current Claude Code version; format can change):

- One JSON object per line. Conversation lines have `type` `user` or `assistant`, a `message` object, `isSidechain`, `uuid`, `parentUuid`, `sessionId`, `cwd`, `timestamp`. Many other `type` values exist (`attachment`, `system`, `queue-operation`, `last-prompt`, ...) and must be skipped.
- Each assistant content block was on its own line in this file: lines with only `thinking`, only `text`, or only `tool_use` blocks. Not guaranteed by any doc I read, so do not depend on it; iterate over `message.content` blocks.
- Typed prompts: `type:"user"` with `message.content` a plain string (24 lines). Tool results: `type:"user"` with content array of `tool_result` blocks (69 lines). One user line had an array with a `text` block (not inspected further).
- Recipe for the last assistant text: scan backward, take lines with `type == "assistant"` (and `isSidechain` false), join `block.text` for blocks with `block.type == "text"`, skip `thinking`, `tool_use`; stop at the first non-empty result. claude-mem does the same: `D:\memory-resource\claude-mem\src\shared\transcript-parser.ts` lines 453-533 and `contentToText` lines 303-316 (verified (local)); it reads backward in chunks (lines 72-117) because transcripts can reach gigabytes.
- But prefer `last_assistant_message` from Stop stdin; use the transcript only as a fallback or for the user side.
- User message for the turn: take `prompt` from UserPromptSubmit. Filtering transcript user lines (skip `tool_result`, skip system reminders and hook-injected text) is possible but I did not verify what meta or injected user lines look like. claude-mem strips `<system-reminder>` blocks (transcript-parser.ts line 505) which suggests they appear in user text: verified (local), reason not verified.

### B5. settings.json registration, timeouts, exit codes

Snippet shape (from the docs example and claude-mem `hooks.json`; verified (web)/(local)):

```json
{
  "hooks": {
    "SessionStart": [
      { "matcher": "startup|resume|clear|compact",
        "hooks": [ { "type": "command", "command": "ingatan-hook", "args": ["session-start"], "timeout": 30 } ] }
    ],
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "ingatan-hook", "args": ["prompt"], "timeout": 15 } ] }
    ],
    "Stop": [
      { "hooks": [ { "type": "command", "command": "ingatan-hook", "args": ["stop"], "timeout": 30, "async": true } ] }
    ]
  }
}
```

(`ingatan-hook` is a placeholder name, not an existing command.) Matcher is omitted for Stop (the page says Stop has no matcher support). `args` form is used by claude-mem for all hooks; with `args`, `shell` is ignored (web).

- `timeout` is in seconds. Defaults: 600 for command hooks, dropping to 30 on UserPromptSubmit. `async: true` runs the hook in the background without blocking and without enforcing `timeout`; async hooks ignore `onFailure`.
- Exit codes: 0 success (stdout parsed as JSON if one object, else plain text); 2 blocking (only effective on events that can block; on SessionStart and SessionEnd it just shows stderr); any other code is a non-blocking error: the action proceeds and a notice shows the first line of stderr. Exit 1 does not block unless `onFailure: "block"` is set.
- So: a hook that always exits 0 (or any code other than 2) never interrupts the session. For capture, never exit 2. Wrap the whole hook in try/except and exit 0.
- Where the settings file lives (`~/.claude/settings.json`, `.claude/settings.json`, `.claude/settings.local.json`) and whether hooks merge across them: **not verified** in this pass (not read in the page text I saw).

### B6. Windows

- Default hook shell is `bash` via Git Bash; on Windows without Git Bash the default is `powershell`. `"shell": "powershell"` per hook is supported (`pwsh.exe` first, then `powershell.exe` 5.1). `CLAUDE_CODE_GIT_BASH_PATH` points at Git Bash if it is not found. verified (web).
- Do not write bare `$CLAUDE_PROJECT_DIR` in a PowerShell hook (web, doc line 1691).
- Stdin encoding on Windows (UTF-8 or the console code page, BOM): **not verified**. Not found in the docs I read, and no test was run. Mitigation to design in, untested: read stdin as raw bytes and decode UTF-8 explicitly (`sys.stdin.buffer.read().decode("utf-8")`), and test with Indonesian and non-ASCII text on Windows in T-19.
- The `command` plus `args` array form avoids shell quoting issues (claude-mem uses it everywhere). Whether `command` can be a bare `python` or `uv run` on a Windows PATH: **not verified**.

## C. Conclusions

### OQ-1 (can Ingatan run next to the built-in memory?)

Answered: yes. Only one external provider may be active; the built-in `MEMORY.md`/`USER.md` memory is separate and stays on (A3). Ingatan cannot be the memory provider at the same time as another external provider such as claude-mem or honcho. Practical split: the provider plugin (T-20) takes the single slot and does capture and injection; the seven tools come through `mcp_servers.ingatan` and do not use the slot. If the owner ever wants a different provider, Ingatan keeps working through MCP alone. Caveat: this is read from upstream docs and source, not from the owner's installed Hermes version. Version is unknown (PLAN.md line 366 says the owner has not supplied it); re-check against it. The claude-mem plugin requires Hermes `>=0.21` (`plugin.yaml` line 6), which is the only version number seen.

### OQ-2 (can a hook make the agent digest?)

Answered: partly, and it is the wrong tool.

- Hermes: no. `on_session_end` returns nothing and its result is discarded (A4).
- Claude Code: `Stop` can hand work back with `decision:"block"` + `reason` or `additionalContext`, but it fires after every turn, forces the agent to continue, adds a visible hook-feedback turn, and is capped at 8 consecutive continuations (B3). `SessionEnd` cannot hand back anything.
- So digest at the start of the next session (DESIGN.md's default) stays correct and is the only approach that works on both agents. A `Stop`-based digest on Claude Code is possible later as an optional extra, not for the MVP.

### Recommended mapping for Ingatan

Claude Code:
- Inject: `SessionStart`, matcher `startup|clear|compact` (add `resume` if the brief should refresh on resume; claude-mem skips it). Print the brief, plus at most three pending episodes to digest, as `additionalContext` (or plain stdout).
- Capture: `UserPromptSubmit` stores the prompt (or spools it keyed by `session_id`); `Stop` stores the reply from `last_assistant_message` and pairs it with the spooled prompt. Run `Stop` with `async: true` and always exit 0. Transcript is a fallback only (it may lag at Stop time).
- Never block, never exit 2.

Hermes:
- Capture: `sync_turn(user_content, assistant_content, *, session_id="", **kwargs)`, enqueue on a background thread (use `spawn_context_thread`), skip when `agent_context` is not `primary`.
- Inject: `system_prompt_block()` for the brief (static, once per session); `prefetch(query)` only if per-turn recall is wanted. The digest offer for pending episodes can go in `system_prompt_block()`.
- Declare `kind: exclusive` as other plugins do, `manifest_version: 1`, `hooks` listing the methods used.
- Tools via `mcp_servers.ingatan` in `config.yaml`, with business and agent scope in `env`.

### What contradicts or refines DESIGN.md

Nothing contradicts DESIGN.md's lines 90-101; its claims are now verified. Refinements the implementer needs:

1. `sync_turn` is not `(user, assistant)` only. It has keyword args `session_id`, `messages`, `turn_author` and may gain more; use `**kwargs`.
2. `exclusive` is enforced in the memory manager's `add_provider`; what the manifest `kind` itself does is not verified.
3. Claude Code `Stop` does receive `last_assistant_message`; no transcript parsing is required for capture.
4. `SessionStart` has a fifth source, `fork`.
5. `Stop` can hand work back, so "not yet checked" in DESIGN.md becomes: yes, but costly (see OQ-2). Update the "Hook findings so far" block when the owner accepts this note.
6. Hermes MCP tool names are prefixed `mcp_<server>_<tool>`, so agent-facing instructions written for Claude Code (`mcp__ingatan__brief` style) differ on Hermes. The Claude Code prefix form was not checked in this pass.

## Not verified, summary

- Hermes read directly from source or a running install (only web docs and third-party plugins). Installed Hermes version.
- Meaning of `kind: exclusive` in plugin.yaml; whether `cwd` is passed to `initialize` by Hermes itself.
- Hermes behaviour on Windows, default `HERMES_HOME` there.
- Stdin encoding of hook processes on Windows; bare `python` as hook command.
- SessionStart context size cap; how to tell a self-started UserPromptSubmit from a typed one; SessionEnd on crash.
- Settings file locations and merge order for hooks.
- Transcript guarantees beyond the one local file (one block per line, shape of injected user lines).
