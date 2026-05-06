# W&B Inference Code Editing Agent

A local Streamlit app that turns any [W&B Inference](https://docs.wandb.ai/inference) model into a code editing agent. The agent uses OpenAI-compatible tool calling to autonomously read, edit, and run shell commands inside a working directory you choose.

## Features

- Bring your own W&B API key (kept only in session memory, never written to disk).
- Live-fetched model list from `https://api.inference.wandb.ai/v1/models`, with descriptions from the W&B docs — pick whichever model you want to code with from the dropdown below the chat input.
- Two modes: **Agent** (full read/write/edit, plus optional shell) and **Ask only** (read-only — the model can list and read files but cannot modify the project).
- Working-directory dropdown below the chat input remembers your recent project folders and includes a folder icon that opens a native OS picker. The recents list is persisted to `~/.wb_coding_agent/recent_dirs.json`.
- File tools: `list_files`, `read_file`, `write_file`, `edit_file`.
- `run_shell` tool runs commands inside the working directory with a 30-second timeout (Agent mode only).
- **MCP server support** — connect any number of [Model Context Protocol](https://modelcontextprotocol.io/) servers (stdio subprocesses or remote streamable-HTTP URLs) via the sidebar. Their tools are namespaced as `mcp__<server>__<tool>` and joined to the local tool list, so the model sees one combined toolbox. Configs persist to `~/.wb_coding_agent/mcp.json` (mode 0600).
- **Project context auto-injection** — at the start of every turn we scan the working directory for `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md` and `.cursor/rules/*.mdc` and splice them into the system prompt; we also detect `.cursor/skills/**/SKILL.md` (workspace) and `~/.cursor/skills/**/SKILL.md` (user-level), surface them as `/<slug>` slash commands, and auto-load any whose trigger keywords match the user's message. A popover next to the Mode/Model selectors (below the chat input) lists every available slash command.
- **Slash-command autocomplete** — type `/` in the chat input and a floating, keyboard-navigable dropdown of available skills appears; arrow keys to navigate, Tab/Enter to insert, Escape to dismiss. The dropdown filters as you type and inherits your Streamlit theme.
- Every tool call is shown inline with arguments and a unified diff for writes/edits.
- Sidebar **File changes** panel summarizes every successful `write_file` / `edit_file` from the conversation with cumulative +/− line counts and a per-file diff expander, so you can audit what the agent touched without scrolling back through chat history.
- The model's reply streams in token-by-token, so you see the agent's reasoning as it's produced instead of waiting for a full response — a "Thinking..." indicator only appears between tool calls while the model is still working.
- Every turn is automatically traced to [W&B Weave](https://docs.wandb.ai/weave/) so you get a structured timeline of inference calls, tool dispatches, latency, and token usage in your W&B workspace. Tracing uses the same API key and `team/project` you enter on Connect; if you leave the project blank, traces go to a `wandb-coding-agent` project under your default entity. The sidebar shows where traces are being written.

## Setup

Requires Python 3.11+.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
streamlit run streamlit_app.py
```

Then in the app:

1. Paste a W&B API key from [wandb.ai/settings](https://wandb.ai/settings).
2. (Optional) Enter `team/project` for usage attribution.
3. Click **Connect** to fetch the model list.
4. Below the chat input, pick a **Working directory** (use the folder icon to browse, or pick a recent), a **Mode** (Agent or Ask only), and a **Model** — then start chatting.

## Desktop build (optional)

If you'd rather launch the agent as a real desktop app instead of typing `streamlit run` every time, you can build a packaged binary that wraps the same Streamlit UI in a native window via [`streamlit-desktop-app`](https://pypi.org/project/streamlit-desktop-app/) + `pywebview`.

Use a separate Python 3.12 build environment — the packager only supports CPython 3.9–3.12:

```bash
uv venv --python 3.12 .venv-build
source .venv-build/bin/activate
uv pip install -e '.[desktop]'
python scripts/build_desktop.py
```

Or use the bootstrap wrapper, which works from a clean shell — no venv activation required, and it (re-)syncs the build venv against `pyproject.toml` first so the build can never run against stale deps:

```bash
./scripts/build_desktop.sh
```

Output by platform:

- **macOS**: `dist/WB Coding Agent.app` (real `.app` bundle). Drag it to `/Applications`. The build is unsigned, so the first launch will be blocked by Gatekeeper — right-click → **Open**, or run `xattr -d com.apple.quarantine "dist/WB Coding Agent.app"` once.
- **Linux / Windows**: `dist/WB Coding Agent/` (a folder with the launcher binary inside). Zip and ship.

The packaged app hides Streamlit's hamburger menu and "Deploy" button so it looks like a desktop tool, not a hosted web app. The same options live in [`.streamlit/config.toml`](.streamlit/config.toml) for the `streamlit run` workflow.

## MCP servers

In the sidebar, expand **MCP servers** and click **Add server** to connect one. Two transports are supported:

- **Stdio** — runs a local subprocess. Example: connect the [filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) reference server.
  - Name: `Filesystem`
  - Transport: `Stdio`
  - Command: `npx`
  - Arguments: one per line — `-y`, `@modelcontextprotocol/server-filesystem`, `/Users/me/projects`
- **HTTP (remote)** — connects to a streamable-HTTP MCP endpoint. Example: a hosted retrieval server.
  - Name: `My MCP`
  - Transport: `HTTP`
  - URL: `https://example.com/mcp`
  - Headers: `Authorization: Bearer ...`

Each row in the **MCP servers** sidebar panel shows the server's name, transport, connection status (or error), an enable/disable checkbox, and an edit button. Configurations are persisted to `~/.wb_coding_agent/mcp.json` (mode 0600 — header tokens are stored in plaintext, so only enter credentials you're comfortable having on disk).

Note: the **W&B Inference service does not need to know anything about MCP**. The local app holds the MCP session, lists each server's tools, and exposes them as ordinary OpenAI-format function schemas in the `tools` array of every chat-completion request. When the model returns a tool call named `mcp__<server>__<tool>`, the app dispatches it to the right MCP session locally and feeds the result back to the model. This is the same bridge pattern Claude Code, Cursor, and the OpenAI Agents SDK use.

In **Ask only** mode, MCP tools are withheld for v1 (Ask mode's contract is "purely sandbox-read"); they're available in **Agent** mode.

## Project context

To make the agent more reliable in your project, drop guidance files where it expects them:

- `AGENTS.md`, `CLAUDE.md`, or `CONVENTIONS.md` at the working-directory root — fully injected into the system prompt on every turn (capped at 12k chars each).
- `.cursor/rules/*.mdc` — fully injected on every turn (capped at 4k chars each).
- `.cursor/skills/**/SKILL.md` (workspace) and `~/.cursor/skills/**/SKILL.md` (user-level) — **conditionally loaded** based on either:
  - **Slash command**: type `/<skill-slug>` anywhere in your message to force-load the skill for that turn (e.g. `/building-streamlit-chat-ui can you wire this up?`). Typing `/` opens the autocomplete dropdown directly over the chat input, or open the **Skills popover** below the chat input for the full list.
  - **Keyword match**: each SKILL.md's frontmatter `description` is mined for trigger phrases (or you can add a `Triggers: a, b, c` line). When your message contains a trigger, the skill auto-loads. Up to 5 keyword-matched skills load per turn (each capped at 6k chars).

A "Project context" expander below the chat input shows what was detected, and a `:material/auto_fix_high: Loaded N skills` caption appears above each assistant turn so you know which skills actually fired.

## Safety

- All file paths are resolved against the working directory; paths that escape the directory are rejected.
- `run_shell` is always available in Agent mode and runs with `cwd` set to your working directory and a 30-second timeout. It is a real shell — only point the agent at directories whose contents (and surrounding environment) you're comfortable letting it touch. Switch to Ask only mode to disable shell, writes, and edits when you just want the model to look at code.
- The W&B API key is held in `st.session_state` for the duration of the browser session only. It is not persisted.
- Always review the unified diffs the agent emits before relying on its edits.

## Project layout

- `streamlit_app.py` — UI entry point.
- `chat_input.py` — Slash-command autocomplete enhancer. CCv2 component that attaches a floating dropdown to `st.chat_input` while typing `/`.
- `agent.py` — Tool-calling agent loop. Decorated with `@weave.op` so each turn is a single Weave trace.
- `tools.py` — Tool schemas and sandboxed executors. `dispatch` is decorated with `@weave.op(kind="tool")`.
- `wb_client.py` — OpenAI-client wrapper for W&B Inference, plus the Weave bootstrap (`init_weave`).
- `mcp_servers.py` — MCP runtime: `ServerConfig`, the registry singleton, the daemon-thread asyncio loop that owns every live MCP session, and the on-disk config at `~/.wb_coding_agent/mcp.json`. `dispatch` is decorated with `@weave.op(kind="tool")`.
- `project_context.py` — Auto-detects `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md`, `.cursor/rules`, and `.cursor/skills` and selects which guidance to splice into each turn's system prompt.
- `scripts/build_desktop.py` — Packaged-desktop-app build script.
- `scripts/build_desktop.sh` — Bootstrap wrapper that creates / re-syncs the build venv, then invokes `build_desktop.py`.
- `.streamlit/config.toml` — Streamlit runtime options for local dev (mirrored in the build script for packaged builds).
- `AGENTS.md` — Authoritative contract for AI agents working on this repo. Read it before contributing.
