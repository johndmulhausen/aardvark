# AGENTS.md

Authoritative guide for AI coding agents working in this repository. Read this file in full before making any change. If a request would conflict with anything documented here, **update this file as part of the same change** so it stays the source of truth.

## What this project is

A locally-run Streamlit desktop app that turns any model served by the [W&B Inference](https://docs.wandb.ai/inference) service into a tool-calling code editing agent. The user supplies their W&B API key, picks a model from the live `/v1/models` list, points the agent at a working directory on disk, and chats with it. The agent autonomously calls a small set of tools (`list_files`, `read_file`, `write_file`, `edit_file`, `run_shell`) to read and modify code, with arguments and unified diffs shown inline in the chat for full transparency. The user is responsible for choosing a working directory whose contents they're comfortable letting the agent run shell commands inside; there is no in-product gate on filesystem writes or shell execution beyond the working-directory sandbox.

Every chat-completion call to the W&B Inference service, every agent turn, and every tool dispatch is traced to [W&B Weave](https://docs.wandb.ai/weave/) — initialized in the connect flow with the same API key and `team/project` the user pastes for inference attribution — so the user gets a structured timeline of inputs, outputs, latency, and token usage in their W&B workspace without any extra setup.

The app ships in two equivalent forms: directly via `streamlit run streamlit_app.py` for development, and as a packaged desktop application built by [`scripts/build_desktop.py`](scripts/build_desktop.py) (a `.app` on macOS, an "onedir" launcher on Linux/Windows). Both render through the same `streamlit_app.py` entry point.

## Module map - read this before adding files

There are exactly seven runtime Python modules (`streamlit_app.py`, `agent.py`, `tools.py`, `wb_client.py`, `mcp_servers.py`, `project_context.py`, `chat_input.py`), plus the build script and the Streamlit config file. Each row below has a single responsibility. **Do not create new top-level modules without first re-reading this section and confirming the change does not duplicate existing functionality. If you add a module, document it here in the same edit.**

| File | Responsibility | Do NOT |
| --- | --- | --- |
| [`streamlit_app.py`](streamlit_app.py) | Streamlit entry point. Page config, session-state init, sidebar (API key, project, Connect button, Weave tracing status caption, an **MCP servers** expander that lists configured servers with per-server enable toggles + edit buttons backed by an `@st.dialog` add/edit modal, a "File changes" panel that summarizes successful `write_file`/`edit_file` results from `ui_turns` with cumulative +/- counts and per-file diff expanders, and clear chat). All session controls render *below* the chat input so the chat input itself sits directly under the conversation history. The page body in `main()` follows a strict positional contract: (1) a forward-declared `conversation_area = st.container()` is created first, with `_render_history()` invoked inside it; (2) the chat input is wrapped in a `st.container()` so Streamlit renders it inline (not pinned to viewport bottom) and is `disabled=not wd_ok` while the working directory is missing; (3) `chat_input.mount_slash_autocomplete(...)` is mounted immediately after the chat input so typing `/` opens an inline filterable dropdown of available skills; (4) `_render_workdir_controls` renders the **Working directory** dropdown + folder-picker button + **Project context indicator**; (5) a "choose a valid working directory" warning appears when `wd_ok` is false; (6) `_render_model_controls` renders the **Mode** + **Model** selectors + **Skills popover** + a labeled "model card" caption (bold model name, `context` / `params` chips, em-dash, then the W&B docs description) sourced from `MODEL_METADATA`. When a turn fires, `_run_turn(prompt)` is invoked inside `with conversation_area:` so both the user message it appends and the assistant `st.chat_message` it opens for live streaming render in the forward-declared container above the chat input — *not* at the bottom of the page in document-flow order, which would put streaming tokens beneath the workdir/mode/model controls. The working-directory dropdown is backed by a persistent recents list at `~/.wb_coding_agent/recent_dirs.json` and a native folder picker via `osascript` on macOS / `tkinter` subprocess elsewhere; the project-context indicator and skills popover read from `project_context.summary(...)` cached for ~5s. Also owns the `MODEL_METADATA` dict that backs the model dropdown, chat history rendering, live event rendering during a turn including the `skills_loaded` caption, and `_render_mcp_result` which case-walks an MCP `CallToolResult`'s content blocks. After a successful Connect, also calls `wb_client.init_weave` so subsequent agent turns are traced. | Put inference logic, tool execution, or filesystem access here. UI only. Define new CCv2 components here (those live in `chat_input.py` or another dedicated UI module). Render anything turn-related (history, streaming, errors mid-turn) outside `conversation_area` — that's the *only* container guaranteed to sit above the chat input. |
| [`chat_input.py`](chat_input.py) | The slash-command autocomplete enhancer. Owns one CCv2 component (declared at module import via `st.components.v2.component(...)`) plus the public `mount_slash_autocomplete(skills, *, placeholder_hint="")` helper that the UI calls on every script run. The component renders nothing visible — its inline JS finds the page's `st.chat_input` `<textarea>` (via `document.querySelector` against multiple defensive selectors), attaches an `input` listener, and pops a floating dropdown above the textarea while the user types `/<query>`. The dropdown lives directly under `document.body` (not in the component's shadow root) so it can overlay the chat input from anywhere. Keyboard contract: arrow keys navigate, Tab / Enter insert + close, Escape closes; the listener uses the capture phase so Enter-to-insert never bubbles up to React's chat-input submit handler. Idempotent across reruns thanks to a `__wbSlashAutocompleteAttached` marker on the textarea + a `window.__wbSlashAutocomplete` singleton. The skill list is delivered via `data={"skills": [...]}` every run so `.cursor/skills` edits surface on the next rerun without a refresh. | Render visible UI inside the component host (it must stay zero-height). Reach into `streamlit_app.py` (this module is a leaf — the UI imports from it, never the other way). Read or write files. |
| [`agent.py`](agent.py) | The tool-calling loop. Exports `run_agent_turn(...)`, a generator that yields `skills_loaded` (per-turn skill selection), `assistant_text_delta` (streamed token chunks), `assistant_text` (full message for replay), `tool_call`, `tool_result`, and `error` events. Owns the system prompts, the per-turn `project_context.scan` + `select_skills_for_turn` call that produces the system-prompt addendum, the `stream=True` chat-completion call, the per-`index` tool-call delta accumulator, the dispatch fork that routes `mcp__*` tool names to `mcp_servers.dispatch` and everything else to `tools.dispatch`, and conversation-message bookkeeping. Both `run_agent_turn` (kind=`agent`) and `_stream_one_call` (kind=`llm`) are decorated with the local `@_op` shim — a thin wrapper that delegates to `@weave.op` after dropping `kind=`/`color=` kwargs unsupported by older `weave` (see "weave.op compat shim" below) — so each turn becomes a single Weave trace tree; `_strip_client` is the `postprocess_inputs` hook that prevents the OpenAI client (and its embedded API key) from being logged. | Call `st.*` from here. Add Streamlit imports. Touch the filesystem directly. Decorate with `@weave.op(...)` directly when `kind=`/`color=` are passed — use the local `@_op(...)` so older weave doesn't crash on import. |
| [`tools.py`](tools.py) | OpenAI-format tool schemas (`TOOL_SCHEMAS`) and their sandboxed executors. Owns the `ToolContext` dataclass and the `dispatch(name, args_json, ctx)` entry point, which is decorated with the local `@_op(kind="tool")` shim (delegates to `@weave.op` after dropping `kind=`/`color=` on older `weave`) so tool calls show up as siblings of inference calls in Weave traces. All path containment checks live here. | Call the LLM. Import `streamlit`. Define MCP-backed tools (those live under `mcp__*` names and never touch this module). Decorate with `@weave.op(..., kind=..., color=...)` directly — use the local `@_op(...)` for compat. |
| [`wb_client.py`](wb_client.py) | Thin wrapper around the OpenAI SDK pointed at `https://api.inference.wandb.ai/v1`, plus the Weave bootstrap. Exports `make_client(api_key, project=None)`, `list_models(client)`, and `init_weave(api_key, project=None)` — the last sets `WANDB_API_KEY` in the environment and calls `weave.init(project)` so the OpenAI client is auto-patched and `@weave.op`-decorated functions actually log. | Add caching, retries, or any product logic. Keep it boring. |
| [`mcp_servers.py`](mcp_servers.py) | All Model Context Protocol runtime + on-disk config. Owns the `ServerConfig` dataclass, the `MCPRegistry` singleton (accessed via `get_registry()` and cached at the UI layer with `@st.cache_resource`), the daemon-thread asyncio loop that owns every live MCP `ClientSession`, the on-disk config at `~/.wb_coding_agent/mcp.json` (mode 0600 — header tokens stored), and the `dispatch(name, arguments_json)` entry point decorated with the local `@_op(name="mcp_dispatch_tool", kind="tool")` compat shim. Tool names are namespaced as `mcp__<server_id>__<tool_name>` (sanitized to `[A-Za-z0-9_]`, ≤64 chars) so `agent.py` can route via `name.startswith(TOOL_NAME_PREFIX)`. | Import `streamlit`. Build OpenAI clients. Edit the working directory directly. Use `weave.init` (use the local `@_op` shim only). Decorate with `@weave.op(..., kind=..., color=...)` directly — use the local `@_op(...)` for compat. |
| [`project_context.py`](project_context.py) | Auto-detect AGENTS.md / CLAUDE.md / CONVENTIONS.md / .cursor/rules / .cursor/skills and decide what's spliced into the system prompt for a turn. Pure functions, no I/O outside scanning. Owns `scan(working_dir)`, `parse_slash_commands`, `match_skills`, `select_skills_for_turn`, `build_system_addendum`, `summary`, plus the constants `MAX_AUTO_SKILLS`, `SKILL_CONTENT_CAP`, `AGENTS_MD_CAP`, `RULES_FILE_CAP`. | Import `streamlit`, `mcp_servers`, or `agent`. Read or write files outside the user's working dir or `~/.cursor/skills`. Splice into `messages` directly (return strings; the agent loop is the splicer). |
| [`scripts/build_desktop.py`](scripts/build_desktop.py) | Build entry point for the packaged desktop app. Wraps `streamlit-desktop-app build` with our pinned PyInstaller options (`--windowed`, bundled-module `--add-data` flags, `--collect-all openai weave mcp httpx`) and the runtime Streamlit options that match `.streamlit/config.toml`. | Add product logic. Import the app modules. |
| [`scripts/build_desktop.sh`](scripts/build_desktop.sh) | Bootstrap wrapper for the desktop build. Idempotently creates `.venv-build` (Python 3.12 via `uv venv`), runs `uv pip install -e '.[desktop]'` to sync deps, then invokes `scripts/build_desktop.py` from that venv. The shell layer only handles environment bootstrapping; all PyInstaller flags stay in the Python script. | Duplicate PyInstaller logic. Add product behavior. |
| [`.streamlit/config.toml`](.streamlit/config.toml) | Streamlit options for the local-dev workflow (`streamlit run streamlit_app.py`). Hides the toolbar / Deploy button and disables telemetry. **Mirror in `STREAMLIT_OPTIONS` in `scripts/build_desktop.py`** because the bundled `.app`'s cwd at launch is `/`, not the project root, so this file is not read at runtime in packaged builds. | Drift from the build script. |

The constant `WB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"` lives in `wb_client.py`. **Do not hard-code that URL anywhere else.**

The OpenAI tool schemas list lives as `TOOL_SCHEMAS` in `tools.py`. **Do not redefine tool schemas in `agent.py` or `streamlit_app.py`.** When adding a new tool, follow the dual-edit pattern below.

## Architecture (call graph)

```mermaid
flowchart LR
 User -->|chat| UI[streamlit_app.py]
 UI -->|"mount_slash_autocomplete<br/>(skills payload)"| Chat[chat_input.py]
 Chat -.->|"floating dropdown<br/>over chat-input textarea"| User
 UI -->|run_agent_turn<br/>@weave.op kind=agent| Agent[agent.py]
 Agent -->|"scan + select_skills_for_turn"| Ctx[project_context.py]
 Ctx -->|"AGENTS.md, .cursor/rules, .cursor/skills"| FS[(Working dir)]
 Ctx -->|"~/.cursor/skills"| Home[("~/.cursor")]
 UI -->|"summary(...)"| Ctx
 Agent -->|"chat.completions.create<br/>tools = TOOL_SCHEMAS + MCP schemas"| WB[W&B Inference<br/>api.inference.wandb.ai/v1]
 WB -->|streamed chunks<br/>content + tool_calls| Agent
 Agent -->|"local tools<br/>@weave.op kind=tool"| Tools[tools.py]
 Agent -->|"mcp__* tools<br/>@weave.op kind=tool"| MCP[mcp_servers.py]
 Tools -->|read/write/list/shell| FS
 MCP -->|"stdio / streamable-HTTP"| External["External MCP servers"]
 MCP -.->|persist| Cfg[("~/.wb_coding_agent/mcp.json")]
 Tools -->|JSON result| Agent
 MCP -->|JSON result| Agent
 Agent -->|delta + final events| UI
 Agent -.->|trace tree| Weave[(W&B Weave<br/>traces)]
 Tools -.->|trace tree| Weave
 MCP -.->|trace tree| Weave
```

Hard rules:

- The chat-completion request goes out from **exactly one place**: `agent.py`'s `run_agent_turn`.
- The OpenAI client is built **only** by `wb_client.make_client`.
- Filesystem reads and writes against the user's working directory live **only** in `tools.py`. Read-only scans for guidance files (`AGENTS.md`, `.cursor/rules`, `.cursor/skills`) live **only** in `project_context.py`.
- The path-containment check (`_resolve_inside`) is the single safety boundary for filesystem tools. Any new file-touching tool **must** call it before reading or writing.
- W&B Weave is initialized **only** by `wb_client.init_weave`, called once per Connect from `streamlit_app.py`. Anywhere else that needs Weave should rely on the `@weave.op` decorator and the auto-patched OpenAI client; do not call `weave.init` from `agent.py`, `tools.py`, `mcp_servers.py`, or anywhere else.
- MCP servers are connected and dispatched **only** through `mcp_servers.MCPRegistry`. No other module opens MCP `ClientSession`s, parses MCP tool schemas, or writes `~/.wb_coding_agent/mcp.json`.
- Skill discovery + selection lives **only** in `project_context.py`. The agent loop calls `select_skills_for_turn(...)` once per turn and splices the returned text; nothing else mines `.cursor/skills`.
- CCv2 components live in dedicated UI modules (today: only `chat_input.py`). Do not call `st.components.v2.component(...)` from `streamlit_app.py`; if you need a new component, add a new module beside `chat_input.py` and import its public mount function.

## How a turn works

1. UI appends `{"role": "user", "content": prompt}` to `st.session_state.messages` and to `st.session_state.ui_turns`.
2. UI calls `run_agent_turn(client, model, messages, working_dir, mode)`. `mode` is either `"agent"` (full tool set, default) or `"ask"` (read-only tools and an ask-only system prompt). The agent runs `project_context.scan(working_dir)` and then `select_skills_for_turn(last_user_message, ctx)` (see "Skill selection contract" below). It yields a single `skills_loaded` event so the UI can show which skills (and via which trigger) were spliced in, and rewrites `messages[0]` with the base system prompt + the project-context addendum (AGENTS.md / CLAUDE.md / .cursor/rules eagerly + the matched skills' bodies). Mid-conversation mode switches and edits to AGENTS.md both take effect on the next turn for free. The tool-call loop runs `while True` until the model produces a final answer with no tool calls — there is no per-turn iteration cap.
3. The agent loop sends `messages + tools=tools_for_mode(mode) + mcp_servers.get_registry().openai_tool_schemas() + tool_choice="auto" + stream=True` to W&B Inference. (MCP tools are only included in Agent mode; Ask mode's contract stays "purely sandbox-read".) The streaming response is consumed by `_stream_one_call`, which yields `assistant_text_delta` events for each `delta.content` chunk and accumulates `delta.tool_calls` fragments by `index` (id, name, and arguments arrive in separate chunks).
4. Once the stream ends, `_stream_one_call` appends the assembled `{"role": "assistant", "content": ..., "tool_calls": [...]}` message to `messages`, then yields one `assistant_text` event with the full content (if any) followed by a `tool_call` event per assembled call.
5. While there are tool calls, the outer loop yields each `tool_call` event, dispatches it through `mcp_servers.dispatch` when the name starts with `mcp__` and `tools.dispatch` otherwise, yields the `tool_result` event, appends a `{"role": "tool", "tool_call_id": ..., "content": json.dumps(result)}` message, and re-queries (back to step 3 with another streamed call).
6. When a streamed response contains no tool calls, the loop yields the final `assistant_text` event and returns.
7. UI streams events into a live `st.chat_message("assistant")` container that's opened inside the page-level `conversation_area` (a forward-declared `st.container()` that lives above the chat input). `_run_turn` is invoked from `main()` as `with conversation_area: _run_turn(prompt)` precisely so its `_render_user_turn(...)` call and the assistant `chat_message` it opens both render *above* the chat input, not at the bottom of the page. The `skills_loaded` event renders as a compact caption above everything else; deltas accumulate into a single `st.empty()` placeholder rendered with `markdown`; tool calls render their own placeholders (with `_render_mcp_result` walking content blocks for `mcp__*` results); a "Thinking..." caption sits below the live content while waiting on the model and is hidden as soon as a delta or tool call arrives.

`skills_loaded`, `assistant_text`, `tool_call`, `tool_result`, and `error` events are persisted to `st.session_state.ui_turns` for replay - `assistant_text_delta` events are display-only and intentionally dropped, since the trailing `assistant_text` carries the full content needed to re-render history.

`st.session_state.messages` is the OpenAI-format conversation (used for the API). `st.session_state.ui_turns` is the structured render log (used for replay). Keep them parallel - if you add a new event type, replay must understand it too.

Concurrently with steps 2–6, every call is captured in W&B Weave as a single nested trace: `run_agent_turn` (kind=`agent`, purple) is the parent op; each invocation of `_stream_one_call` (kind=`llm`, blue) is a child, with the auto-patched `client.chat.completions.create` underneath; each `tools.dispatch` invocation (kind=`tool`, green) is a sibling of the inference calls. The `client` argument is stripped from logged inputs by `_strip_client` so the W&B API key is never serialized into a trace. See "Tracing with W&B Weave" below for the contract.

## Tracing with W&B Weave

`wb_client.init_weave(api_key, project)` is called by `streamlit_app.py._connect` after a successful Connect. It:

1. Sets `WANDB_API_KEY` in the process environment (so `weave.init` does not block on stdin auth — important inside the packaged desktop app where there is no terminal).
2. Calls `weave.init(project or DEFAULT_WEAVE_PROJECT)`, which patches `openai.OpenAI` so every `chat.completions.create` becomes a child of any active `@weave.op`.
3. Returns `(WeaveClient, resolved_project)` so the UI can show `Tracing turns to W&B Weave at <project>` in the sidebar.

`DEFAULT_WEAVE_PROJECT = "wandb-coding-agent"` (in `wb_client.py`) is the fallback project used when the user leaves the optional `team/project` field blank; Weave creates it under the user's default entity on first call.

If init fails (network error, project access, etc.) the UI shows `Weave tracing disabled: <error>` instead. The `@weave.op` decorators in `agent.py` and `tools.py` are no-ops without an initialized Weave client, so failure is non-fatal — the agent runs unchanged, just without tracing.

When you add a new function that calls the W&B Inference service (or that should appear in the trace tree), decorate it with `@_op(...)` (the local compat wrapper — see "weave.op compat shim" below) using one of the standard `kind` values:

- `kind="agent"` — top-level orchestration of a user turn.
- `kind="llm"` — a function whose body issues `client.chat.completions.create` (or another model call).
- `kind="tool"` — tool dispatch / executor.
- `kind="search"` — retrieval helpers, when you eventually add them.

If your decorated function takes an OpenAI client (or anything else with embedded credentials) as an argument, also pass `postprocess_inputs=_strip_client` (or an equivalent function) so secrets stay out of the trace. The auto-patched child OpenAI op already captures every relevant request field (model, messages, tools), so dropping the client argument from the parent's logged inputs costs nothing.

### weave.op compat shim

`agent.py`, `tools.py`, and `mcp_servers.py` each define a tiny local `_op(...)` wrapper that delegates to `weave.op` after stripping `kind=` / `color=` kwargs that older `weave` (< 0.52.24) doesn't accept. The kwargs were added partway through the 0.52 series, and on older installs `@weave.op(..., kind="tool", color="green")` raises `TypeError` at decorator-evaluation time — i.e. at module import — which crashes the whole app before Streamlit ever paints. Feature-detecting once at module load (via `inspect.signature(weave.op).parameters`) means upgrade-laggers still get a correct trace tree; they just lose the Weave UI's color/category badges. `pyproject.toml` pins `weave>=0.52.24` so fresh installs get the full UX, but the shim is the safety net.

When adding a new `@_op`-decorated function, always go through `_op` — never call `@weave.op(..., kind=..., color=...)` directly. If a future weave release rejects yet another kwarg, extend `_WEAVE_OP_DROP` in each of the three modules to include it.

## Session state contract

These keys are owned by `streamlit_app.py` and initialized in `_init_state()`. **Update `_init_state` and this table together** if you add or rename a key.

| Key | Type | Meaning |
| --- | --- | --- |
| `api_key` | `str` | W&B API key. Session-only, never persisted. |
| `project` | `str` | Optional `team/project` for usage attribution. |
| `client` | `openai.OpenAI \| None` | Built by `make_client` after Connect. |
| `models` | `list[str]` | Model IDs from `list_models`, sorted alphabetically by their display label (case-insensitive) so the dropdown reads in user-visible order. |
| `model` | `str \| None` | Currently selected model ID. Chosen via the Model dropdown rendered *below* the chat input (not in the sidebar). The dropdown binds to this key with `key="model"` so Streamlit owns the value end-to-end — **do not** revert to the `ss.model = st.selectbox(..., index=...)` pattern; that mixed ownership lets the user's selection get silently dismissed across reruns. `_connect` reseeds this from inside its `on_click` callback when the new models list doesn't contain the prior id. |
| `mode` | `str` | Either `"agent"` (full tool set, default) or `"ask"` (read-only tools, ask-only system prompt). Chosen via the Mode dropdown rendered *below* the chat input, alongside Model and Skills. Bound to the dropdown via `key="mode"` for the same reason as `model`. |
| `working_dir` | `str` | User-chosen working directory. Selected via the Working directory dropdown rendered *below* the chat input (free text + recents) or the adjacent folder-picker button that launches a native OS chooser. Validated as an existing dir before any turn — when the path doesn't exist, the chat input is disabled and a warning rendered between the workdir picker and the model controls explains why. |
| `recent_dirs` | `list[str]` | Most-recent-first list of absolute working-directory paths, loaded from `~/.wb_coding_agent/recent_dirs.json` on startup and re-saved whenever the user picks a new directory. Capped at `MAX_RECENT_DIRS`. |
| `messages` | `list[dict]` | OpenAI-format conversation. |
| `ui_turns` | `list[dict]` | Structured render log: each item is `{"role": "user", "content": ...}` or `{"role": "assistant", "events": [...]}`. |
| `connect_error` | `str \| None` | Last connection error, displayed in the sidebar. |
| `weave_project` | `str \| None` | Resolved Weave project (`team/project` or `DEFAULT_WEAVE_PROJECT`) that turns are being traced to. Set by `_connect` after `init_weave` succeeds; displayed as a "Tracing turns to W&B Weave at ..." caption under the connection status. |
| `weave_error` | `str \| None` | Error message from the most recent `init_weave` failure. Set when Connect succeeds but Weave init does not; surfaces a "Weave tracing disabled" caption. Mutually exclusive with `weave_project`. |
| `conn_open` | `bool` | Whether the sidebar Connection expander is open. Defaults to `True` so first-time users see the form, then is flipped to `False` by `_on_connect` after a successful Connect so the panel collapses to a one-line "Connected" header (green check icon). The expander is bound via `key="conn_open"` + `on_change="rerun"`, so manual chevron toggles by the user write back here and persist across reruns. |
| `mcp_dialog_open` | `bool` | Gates the `@st.dialog("MCP server")` add/edit modal. Set to `True` by `_open_add_mcp_dialog` / `_open_edit_mcp_dialog` and back to `False` after Save/Cancel/Delete. The dialog persists configs through `MCPRegistry.add` / `update` / `remove`; the registry itself lives in the `@st.cache_resource`-backed singleton in `mcp_servers.get_registry()`, not in session state. |
| `mcp_dialog_editing` | `str \| None` | When the dialog is open, this is either `None` (Add) or the id of the server being edited. Used by `_mcp_server_dialog` to pre-fill the form. |

## Adding a new tool (the dual-edit pattern)

Tools touch filesystem and shell, so getting them right matters. Follow these steps in order:

1. **Add the schema** in `TOOL_SCHEMAS` in `tools.py`. Use OpenAI function-calling format. Required fields belong in `parameters.required`. Write a description that tells the model exactly when to use the tool and what it returns.
2. **Implement the executor** as `_my_tool(ctx: ToolContext, ...) -> dict[str, Any]`. Always:
   - Call `_resolve_inside(ctx.working_dir, path)` for any path argument.
   - Return `{"error": ...}` for recoverable failures so the model can adjust. Do not raise.
   - Keep the result JSON-serializable - it is round-tripped through `json.dumps` for the next API call.
3. **Register it** in the `_DISPATCH` map in `tools.py`.
4. **Pick a UI icon** in `TOOL_ICONS` in `streamlit_app.py` (Material icon name).
5. **Render its result** in `_render_tool_event` in `streamlit_app.py` if it has a non-trivial display (diffs, listings, stdout, etc).
6. **Update the system prompt** in `agent.py` (`SYSTEM_PROMPT`) if the tool changes the recommended workflow.
7. **Update the README and this file's tool table** below if the tool is user-visible.

Current tools:

| Tool | Required | Returns |
| --- | --- | --- |
| `list_files(path=".", max_depth=3)` | - | `{"listing": str}` (tree) |
| `read_file(path, start_line=1, end_line=None)` | `path` | `{"content": str, "total_lines": int, ...}` (line-numbered) |
| `write_file(path, content)` | `path`, `content` | `{"ok": True, "diff": str, ...}` (unified diff vs prior) |
| `edit_file(path, old_string, new_string)` | all | `{"ok": True, "diff": str}` after a unique replace |
| `run_shell(command, timeout=30)` | `command` | `{"exit_code", "stdout", "stderr"}`. Runs with `cwd=working_dir`; no in-product gate. |

In **Ask only** mode, only `list_files` and `read_file` are exposed to the model — `write_file`, `edit_file`, and `run_shell` are withheld at the API boundary by `tools.tools_for_mode("ask")`. The `READONLY_TOOL_NAMES` set in `tools.py` is the source of truth for which tools remain available in Ask mode; update it (and the system prompt for Ask mode in `agent.py`) if you add a new read-only tool. **MCP-backed tools are also withheld in Ask mode for v1**: `agent.py` only adds `mcp_servers.get_registry().openai_tool_schemas()` to the tools list when `mode != "ask"`, because we don't yet have per-MCP-tool read-only metadata.

### MCP-backed tools

External MCP server tools are exposed to the model under `mcp__<server_id>__<tool_name>` names. They never enter `TOOL_SCHEMAS`. The dual-edit pattern above does **not** apply — adding an MCP tool is purely a user-side action (pointing the agent at the right MCP server), not a code change. To make a new MCP server type easier to use, edit `_render_mcp_result` in `streamlit_app.py` if its result content blocks need a richer rendering than the default text/image/resource walk.

## Skill selection contract

`project_context.py` is the single source of truth for what `.cursor/skills` look like to the agent. Every turn, `agent.py` calls `project_context.scan(working_dir)` and then `select_skills_for_turn(last_user_message, ctx)`. The selection rule is:

1. **Slash commands** — any `/<slug>` token in the user's message (matched at word boundaries via `_SLASH_RE` in `project_context.py`) force-loads that skill for the turn. Slash commands always win and are not subject to the per-turn cap.
2. **Keyword match** — for every skill, we mine trigger phrases from (a) the YAML frontmatter `triggers:` field, (b) any `Triggers: a, b, c` sentence inside the frontmatter `description`, and (c) word-parts of the skill's `name` ≥ 3 characters. A skill matches if any phrase appears in the user message (whole-word for word-character phrases, substring for phrases containing punctuation like `st.`).
3. Keyword matches fill the remaining budget up to `MAX_AUTO_SKILLS` (currently 5), ordered by trigger-match count.
4. Each loaded SKILL.md body is capped at `SKILL_CONTENT_CAP` characters (currently 6000).

**Authoring a new SKILL.md.** Frontmatter must include `name:` (used as the slash slug after slugification). To make the skill reliably keyword-trigger, pack a `Triggers: a, b, c` line into `description` — this matches the convention the existing Cursor skills already use. Write triggers as the words / phrases users naturally type (e.g. `dashboard, metrics, KPI`) rather than the skill's internal terminology.

**Workspace vs. user skills.** `.cursor/skills/**/SKILL.md` inside the working directory and `~/.cursor/skills/**/SKILL.md` in the user's home both feed into the same selection pass. On slug collision the workspace skill wins (project context overrides global) and the conflict is recorded in `summary(...)["slug_conflicts"]` so the UI can warn.

## Coding standards

All future code additions must:

- **Be commented and documented.** Every module needs a top-of-file docstring explaining its responsibility. Every public function needs a docstring describing arguments, return shape, and any side effects. Add inline comments for non-obvious branches, safety boundaries, and protocol decisions - **do not** add narrative comments that just restate the code (`# increment counter`).
- **Be typed.** Use `from __future__ import annotations` and PEP-604 (`X | None`) types. Public function signatures must have full annotations.
- **Stay inside the module map above.** If a new responsibility genuinely does not fit, update both the code and the table in this file.
- **Not introduce a new dependency** without adding it to `[project].dependencies` in [`pyproject.toml`](pyproject.toml) with a sensible lower bound (`>=`). Prefer the OpenAI SDK and the standard library; reach for new packages only when they materially simplify the code.
- **Keep paths relative to the working directory** when interacting with the user's project. Never widen the sandbox.
- **Show users what the agent did.** Tool results that mutate state must include a `diff` (or equivalent before/after evidence) so `_render_tool_event` can display it.

Streamlit-specific rules:

- Read [`/Users/john/streamlit-inference/.cursor/skills/developing-with-streamlit/SKILL.md`](.cursor/skills/developing-with-streamlit/SKILL.md) and the relevant sub-skill before non-trivial UI changes.
- Use Material icons (`:material/icon_name:`) over emojis.
- Use sentence case for titles and labels.
- Prefer `st.caption` / `st.toast` over `st.info` for low-importance notes.
- Pin `streamlit>=1.53.0`. Many features used here (Material icons, `st.space`, `horizontal_alignment`, modern `st.container` options) require this.

## Anti-duplication checklist

Before creating any new file or function, verify:

- [ ] Does this responsibility already live somewhere in the **module map** above? If yes, edit there.
- [ ] Am I about to write a second OpenAI client constructor? Use `wb_client.make_client`.
- [ ] Am I about to redefine a tool schema? It belongs in `TOOL_SCHEMAS` in `tools.py`.
- [ ] Am I hard-coding the W&B Inference base URL? Import `WB_INFERENCE_BASE_URL` from `wb_client`.
- [ ] Am I about to read or write the user's filesystem outside `tools.py`? Stop. Add a tool instead.
- [ ] Am I about to call `client.chat.completions.create` outside `agent.py`? Stop. Extend `run_agent_turn` instead.
- [ ] Am I adding a new function that calls a W&B Inference model? Decorate it with the local `@_op(...)` shim (`kind="llm"` or `kind="agent"`) and use `postprocess_inputs=_strip_client` if it takes the OpenAI client as an arg. See "Tracing with W&B Weave" + "weave.op compat shim".
- [ ] Am I adding a `@weave.op(..., kind=..., color=...)` decorator directly? Stop. Use the local `@_op(...)` shim instead — older `weave` (< 0.52.24) raises `TypeError` at import time on those kwargs. The shim feature-detects and gracefully drops them.
- [ ] Am I about to call `weave.init` outside `wb_client.init_weave`? Stop. Add it once at Connect; everything else relies on the auto-patched OpenAI client and `@weave.op`.
- [ ] Am I about to redefine model display labels or descriptions? They live only in `MODEL_METADATA` in `streamlit_app.py`, transcribed from the W&B [Available models](https://docs.wandb.ai/inference/models) page. Edit there.
- [ ] Am I about to add another OS file/folder picker? Use `_pick_directory()` in `streamlit_app.py` (osascript on macOS, `tkinter` subprocess elsewhere). Don't import `tkinter` at top level — it must stay inside the subprocess snippet so the main Streamlit script thread is unaffected.
- [ ] Am I about to introduce a second on-disk preferences file? Persisted state lives under `~/.wb_coding_agent/`: `recent_dirs.json` (working-directory recents) and `mcp.json` (MCP server configs, mode 0600). Add new persisted state to that same directory and document it in the session-state table.
- [ ] Am I opening an MCP `ClientSession`, calling `list_tools`/`call_tool`, or reading/writing `~/.wb_coding_agent/mcp.json` outside `mcp_servers.py`? Stop. Extend `MCPRegistry` instead.
- [ ] Am I scanning the working directory (or `~/.cursor/skills`) for guidance / skills outside `project_context.py`? Stop. Extend `project_context.scan` / `summary` / `select_skills_for_turn` instead.
- [ ] Am I about to redefine the slash-command regex or the keyword-trigger logic? It's centralized in `project_context.py`. Edit there.
- [ ] Am I adding a new read-only tool? Add its name to `READONLY_TOOL_NAMES` in `tools.py` so it's available in Ask mode, and update `SYSTEM_PROMPT_ASK` in `agent.py` if the workflow guidance needs to mention it.
- [ ] Am I about to register a new CCv2 component (`st.components.v2.component(...)`)? Define it in a new dedicated UI module beside `chat_input.py`, expose a single `mount_<thing>(...)` helper, and import that helper from `streamlit_app.py`. Don't sprinkle component declarations across `streamlit_app.py`.
- [ ] Am I about to write JavaScript that reaches into the parent page's DOM to enhance another Streamlit widget? See `chat_input.py` for the pattern: idempotent attach via a marker property on the target element, dropdown / overlay rendered into `document.body` (not the component's shadow root), capture-phase listeners when you need to intercept React's delegated handlers.
- [ ] Am I creating a new state key in `st.session_state`? Update `_init_state` **and** the session-state table above.
- [ ] Am I changing a Streamlit runtime option? Update **both** `.streamlit/config.toml` and `STREAMLIT_OPTIONS` in `scripts/build_desktop.py`.
- [ ] Am I adding a new top-level Python module or third-party package? Update `BUNDLED_MODULES` / `COLLECT_ALL_PACKAGES` in `scripts/build_desktop.py` so the packaged build still imports it.

## Running and verifying

The expected dev loop:

```bash
cd /Users/john/streamlit-inference
source .venv/bin/activate         # created by uv venv
streamlit run streamlit_app.py    # opens http://localhost:8501
```

For a quick smoke test without spending a W&B API call:

```bash
source .venv/bin/activate
python -c "from tools import dispatch, ToolContext; from pathlib import Path; import json; \
  print(dispatch('list_files', json.dumps({'path': '.', 'max_depth': 1}), ToolContext(Path('.'))))"
```

To exercise write/edit/read end-to-end against a temp directory, see the verification snippet pattern used during initial implementation: create a `tempfile.TemporaryDirectory`, build a `ToolContext`, then dispatch `write_file` -> `read_file` -> `edit_file` -> `read_file` and confirm the diff field shape.

When testing the live UI, the browser MCP can be used to render the running app at `http://localhost:8501` and inspect the snapshot. The empty-state path ("Enter your W&B API key and click Connect") is the canonical first-render check.

## Desktop build

The repo ships a packaged desktop app on top of [`streamlit-desktop-app`](https://pypi.org/project/streamlit-desktop-app/). The build pipeline is owned by [`scripts/build_desktop.py`](scripts/build_desktop.py); do not invoke `streamlit-desktop-app build` directly. Architecture at runtime:

- The packaged binary spawns Streamlit on a random localhost port (headless).
- `pywebview` opens a native window pointed at that URL (WKWebView on macOS).
- The user-visible window is therefore the same Streamlit UI, with the toolbar suppressed via `client.toolbarMode = "minimal"`.

Build prerequisites:

- Use a CPython 3.12 interpreter; `streamlit-desktop-app` 0.3.4 caps Python at 3.12.
- Install the build extras into a separate venv (e.g. `uv venv --python 3.12 .venv-build && uv pip install -e '.[desktop]'`).

Build invocation (from repo root, with the build venv active):

```bash
python scripts/build_desktop.py
```

Or use the bootstrap wrapper, which works from a clean shell — no venv activation required, and it (re-)syncs the build venv against `pyproject.toml` first so the build can never run with stale deps:

```bash
./scripts/build_desktop.sh
```

Output:

- macOS: `dist/WB Coding Agent.app` (real `.app` bundle; drag to `/Applications`). Unsigned, so first launch needs right-click → Open or `xattr -d com.apple.quarantine`.
- Linux / Windows: `dist/WB Coding Agent/` (PyInstaller onedir layout). Distribute as a zip.

When adding a new top-level Python module imported by `streamlit_app.py`, append it to `BUNDLED_MODULES` in `scripts/build_desktop.py`. When adding a new third-party dependency that is only imported transitively from a bundled module, append it to `COLLECT_ALL_PACKAGES`. PyInstaller's static import graph cannot see imports inside files added via `--add-data`, so this manual step is unavoidable.

When changing any Streamlit runtime option, update **both** `.streamlit/config.toml` (for local dev) **and** `STREAMLIT_OPTIONS` in `scripts/build_desktop.py` (for packaged builds). The two are not auto-synced because the bundled app cannot read the config file at runtime.

Out of scope for the desktop build today: code signing, notarization, auto-update. The artifact is intended for the user's own machine; if you ship it externally, expect Gatekeeper friction.

## Out of scope (intentional)

These are explicitly **not** part of v1. If a future request asks for one of these, treat it as a new feature, design it deliberately, and update this section once it lands:

- Persistent chat history across browser sessions.
- Multi-tab or multi-project sessions.
- Per-tool click-to-approve UI, and any in-product gating of file writes or shell commands. Current safety model: the user picks a working directory they trust the agent to operate inside; file ops auto-approve and show diffs; shell commands run with that directory as their cwd and a 30s timeout. There is no toggle, allow-list, or confirm dialog.
- Code signing, notarization, and auto-update for the desktop build (see "Desktop build" above for the current unsigned-distribution caveats).
- Hosted multi-tenant deployment (Streamlit Community Cloud, etc.). The agent's tools mutate the working directory; on a hosted server that directory is shared between visitors. Designing this safely would require per-session sandboxes and is intentionally not v1.

## Updating this file

This document is the contract between humans and agents working on this repo. **You must update it whenever you:**

- Add, remove, or rename a module, tool, session-state key, or system-prompt rule.
- Change the architectural boundaries above (e.g. start streaming, add a second LLM caller, move filesystem access).
- Add a new dependency or change minimum versions.
- Move something from the "Out of scope" list into the product.
- Notice that the current code or behavior contradicts something written here. The **code is not** automatically authoritative; if the conflict is a regression, fix the code; if it is intentional, fix this file. Either way, do not leave the conflict.

When in doubt, update AGENTS.md in the same commit as the code change so the contract never drifts.
