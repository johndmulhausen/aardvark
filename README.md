# W&B Inference Code Editing Agent

A local Streamlit desktop application that transforms any model served by the [W&B Inference](https://docs.wandb.ai/inference) service into a powerful code editing agent. The agent uses OpenAI-compatible tool calling to autonomously read, edit, and run shell commands within a working directory of your choice, providing a seamless AI-assisted coding experience with full transparency and control.

## Project Overview

This application provides a sophisticated interface for AI-assisted coding with the following architecture:

- **Multi-page Streamlit App**: Organized into Chat, Usage, and Settings pages with shared sidebar
- **Real-time Agent System**: Tool-calling agent that streams responses and shows inline diffs
- **Comprehensive Tool Suite**: File operations, shell execution, and MCP server integration
- **Persistent State Management**: Chat history, usage tracking, and user preferences
- **Git Integration**: Full git workflow support with conflict resolution and push capabilities
- **W&B Ecosystem Integration**: Native tracing, model selection, and cost tracking

## Features

- Bring your own W&B API key. By default it's session-only; tick **Remember on this machine** in the settings popover and it persists to `~/.wb_coding_agent/credentials.json` (mode 0600) so future launches connect automatically.
- **Settings tab in the top nav** — verify a GitHub Personal Access Token (your GitHub avatar then renders in the page header and the agent commits as you), pick the app theme from a **System / Light / Dark** segmented control (the page reloads once to apply your choice; the preference persists across sessions), and connect to W&B Inference with optional opt-in API-key persistence.
- **Usage and cost dashboard** as a separate page — KPI cards for today / 7-day token use and dollar cost, daily token volume + cost line charts, cost-by-model bar chart, and a recent-turns table. Driven by live `usage` chunks the W&B Inference service emits during each streamed completion, priced via the per-model rates published at [wandb.ai/site/pricing/inference](https://wandb.ai/site/pricing/inference). Persisted to `~/.wb_coding_agent/usage.jsonl` so the data survives restarts.
- Live-fetched model list from `https://api.inference.wandb.ai/v1/models`, with descriptions from the W&B docs — pick whichever model you want to code with from the dropdown below the chat input.
- Two modes: **Agent** (full read/write/edit, plus optional shell) and **Ask only** (read-only — the model can list and read files but cannot modify the project).
- Working-directory dropdown below the chat input remembers your recent project folders and includes a folder icon that opens a native OS picker. The recents list is persisted to `~/.wb_coding_agent/recent_dirs.json`.
- File tools: `list_files`, `read_file`, `write_file`, `edit_file`.
- `run_shell` tool runs commands inside the working directory with a 30-second timeout (Agent mode only).
- **MCP server support** — connect any number of [Model Context Protocol](https://modelcontextprotocol.io/) servers (stdio subprocesses or remote streamable-HTTP URLs) via the sidebar. Their tools are namespaced as `mcp__<server>__<tool>` and joined to the local tool list, so the model sees one combined toolbox. Configs persist to `~/.wb_coding_agent/mcp.json` (mode 0600).
- **Project context auto-injection** — at the start of every turn we scan the working directory for `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md` and `.cursor/rules/*.mdc` and splice them into the system prompt; we also detect `SKILL.md` files under either `.cursor/skills/**/` or `.claude/skills/**/` (workspace) and `~/.cursor/skills/**/` or `~/.claude/skills/**/` (user-level), surface them as `/<slug>` slash commands, and auto-load any whose trigger keywords match the user's message. A popover next to the Mode/Model selectors (below the chat input) lists every available slash command.
- **Slash-command autocomplete** — type `/` in the chat input and a floating, keyboard-navigable dropdown of available skills appears; arrow keys to navigate, Tab/Enter to insert, Escape to dismiss. The dropdown filters as you type and inherits your Streamlit theme.
- Every tool call is shown inline with arguments and a unified diff for writes/edits.
- Sidebar **File changes** panel summarizes every successful `write_file` / `edit_file` from the conversation with cumulative +/− line counts and a per-file diff expander, so you can audit what the agent touched without scrolling back through chat history.
- The model's reply streams in token-by-token, so you see the agent's reasoning as it's produced instead of waiting for a full response — a "Thinking..." indicator only appears between tool calls while the model is still working.
- Every turn is automatically traced to [W&B Weave](https://docs.wandb.ai/weave/) so you get a structured timeline of inference calls, tool dispatches, latency, and token usage in your W&B workspace. Tracing uses the same API key and `team/project` you enter on Connect; if you leave the project blank, traces go to a `wandb-coding-agent` project under your default entity. The sidebar shows where traces are being written.

## Setup

### Prerequisites

- **Python 3.11+** (required for modern async/await patterns and type hints)
- **W&B API Key** from [wandb.ai/settings](https://wandb.ai/settings)
- **Git** (optional, for version control integration)

### Installation

#### Using uv (recommended)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install with development dependencies
uv pip install -e .
```

#### Using pip

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e .
```

### Key Dependencies

The application relies on several core packages:

- **Streamlit (>=1.53.0)**: Web framework for the multi-page interface
- **OpenAI (>=1.40.0)**: Client library for W&B Inference API compatibility
- **Weave (>=0.52.24)**: W&B's tracing and observability framework
- **MCP (>=1.19.0)**: Model Context Protocol for external tool integration
- **Streamlit Desktop App (optional)**: For packaged desktop builds

## Run

```bash
streamlit run streamlit_app.py
```

Then in the app:

1. Open the **Settings** tab in the top nav.
2. Paste a W&B API key from [wandb.ai/settings](https://wandb.ai/settings). (Optional: tick **Remember on this machine** to save it.)
3. (Optional) Enter `team/project` for usage attribution and Weave tracing.
4. Click **Connect** to fetch the model list.
5. Switch to the **Chat** tab. Below the chat input, pick a **Working directory** (use the folder icon to browse, or pick a recent), a **Mode** (Agent or Ask only), and a **Model** — then start chatting.
6. Switch to the **Usage** tab any time to see token use and dollar cost across every turn you've run.

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

The packaged app hides Streamlit's toolbar entirely (hamburger menu, Settings, Deploy button) so it reads as a self-contained desktop tool rather than a hosted web app. The Light / Dark / System toggle lives on the in-app **Settings** tab instead. The same options live in [`.streamlit/config.toml`](.streamlit/config.toml) for the `streamlit run` workflow.

## MCP servers

Open the **Settings** tab in the top nav, scroll to the **MCP servers** card, and click **Add server** to connect one. Two transports are supported:

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

Each row in the **MCP servers** card on the Settings page shows the server's name, transport, connection status (or error), an enable/disable checkbox, and an edit button. Configurations are persisted to `~/.wb_coding_agent/mcp.json` (mode 0600 — header tokens are stored in plaintext, so only enter credentials you're comfortable having on disk).

Note: the **W&B Inference service does not need to know anything about MCP**. The local app holds the MCP session, lists each server's tools, and exposes them as ordinary OpenAI-format function schemas in the `tools` array of every chat-completion request. When the model returns a tool call named `mcp__<server>__<tool>`, the app dispatches it to the right MCP session locally and feeds the result back to the model. This is the same bridge pattern Claude Code, Cursor, and the OpenAI Agents SDK use.

In **Ask only** mode, MCP tools are withheld for v1 (Ask mode's contract is "purely sandbox-read"); they're available in **Agent** mode.

## Project context

To make the agent more reliable in your project, drop guidance files where it expects them:

- `AGENTS.md`, `CLAUDE.md`, or `CONVENTIONS.md` at the working-directory root — fully injected into the system prompt on every turn (capped at 12k chars each).
- `.cursor/rules/*.mdc` — fully injected on every turn (capped at 4k chars each).
- `.cursor/skills/**/SKILL.md` or `.claude/skills/**/SKILL.md` (workspace) and `~/.cursor/skills/**/SKILL.md` or `~/.claude/skills/**/SKILL.md` (user-level) — **conditionally loaded** based on either:
  - **Slash command**: type `/<skill-slug>` anywhere in your message to force-load the skill for that turn (e.g. `/building-streamlit-chat-ui can you wire this up?`). Typing `/` opens the autocomplete dropdown directly over the chat input, or open the **Skills popover** below the chat input for the full list.
  - **Keyword match**: each SKILL.md's frontmatter `description` is mined for trigger phrases (or you can add a `Triggers: a, b, c` line). When your message contains a trigger, the skill auto-loads. Up to 5 keyword-matched skills load per turn (each capped at 6k chars).

A "Project context" expander below the chat input shows what was detected, and a `:material/auto_fix_high: Loaded N skills` caption appears above each assistant turn so you know which skills actually fired.

## Settings and GitHub identity

The **Settings** tab in the top nav manages everything per-user. The flow:

- **Avatar.** When you verify a GitHub PAT (below), your GitHub avatar is downloaded once and cached for the rest of the Streamlit session, and renders in the GitHub identity card. Without a verified PAT, an inline-SVG `account_circle` glyph is shown instead. There is no separate avatar uploader.
- **GitHub identity.** Generate a fine-grained personal access token at [github.com/settings/personal-access-tokens/new](https://github.com/settings/personal-access-tokens/new). Recommended permissions:
  - `read:user` and `user:email` — required for the verify step (`GET /user`) to return your username and primary email.
  - `repo` — only if you want to grant the agent push access to repositories under your control. This isn't required for verification or for stamping commit authorship.

  Paste the PAT, click **Verify and save**, and the app calls GitHub's `/user` endpoint to confirm the token + pull your username, email, and avatar. The PAT is stored in `~/.wb_coding_agent/credentials.json` (mode 0600), the non-secret identity fields in `~/.wb_coding_agent/preferences.json`. When you chat in a directory that's a git repo, the agent runs `git config --local user.name`/`user.email` so commits it makes via `run_shell` are authored as you.
- **Theme.** A **System / Light / Dark** segmented control on the Settings page is the canonical theme toggle for this app. Picking a value writes the preference to `~/.wb_coding_agent/preferences.json` and reloads the page once so the new theme applies. **System** follows your operating system's light / dark setting. Streamlit's own toolbar Settings menu is hidden in this app — the in-app switcher is the only theme control you'll see.
- **W&B Inference connection.** API key + optional `team/project` + Connect button. Tick **Remember on this machine** to persist the API key (mode 0600); leave it unchecked for the original session-only behavior.

## Usage and cost

Click **Usage** in the top nav (next to **Chat**) to see a dashboard of every turn you've run:

- KPI cards for today's tokens, today's cost, the trailing 7-day tokens, and the trailing 7-day cost (with deltas vs the prior period).
- A 30-day stacked line chart of prompt vs completion tokens.
- A 30-day line chart of daily USD cost.
- A horizontal bar chart of cost-by-model.
- A scrollable table of the most recent 100 turns with timestamp, model, mode, prompt/completion/total tokens, USD cost, and latency.

Cost is computed from the per-model rates published at [wandb.ai/site/pricing/inference](https://wandb.ai/site/pricing/inference); for any model without published pricing the cost cell renders `-`. Data is persisted to `~/.wb_coding_agent/usage.jsonl` (one JSON object per turn, append-only) so it survives restarts. The same `usage.jsonl` powers the per-turn footer caption in chat.

To capture this data the app passes `stream_options={"include_usage": True}` on every chat completion — the W&B Inference service responds with a final usage chunk containing prompt / completion / total token counts, which we accumulate across all inference rounds in the turn.

## Safety

- All file paths are resolved against the working directory; paths that escape the directory are rejected.
- `run_shell` is always available in Agent mode and runs with `cwd` set to your working directory and a 30-second timeout. It is a real shell — only point the agent at directories whose contents (and surrounding environment) you're comfortable letting it touch. Switch to Ask only mode to disable shell, writes, and edits when you just want the model to look at code.
- The W&B API key is held in `st.session_state` for the duration of the browser session by default. If you tick **Remember on this machine** on the Settings page, it's saved to `~/.wb_coding_agent/credentials.json` (mode 0600); anyone with read access to your home directory could read it. Untick the box and click **Forget saved API key** to remove it.
- The GitHub PAT (when set) is also saved at `~/.wb_coding_agent/credentials.json` (mode 0600). The token is sent only to `https://api.github.com/user` for verification and is otherwise inert; the agent itself never makes GitHub API calls with it.
- Always review the unified diffs the agent emits before relying on its edits.

## Project Architecture

### Module Structure

The application is organized into several specialized modules with clear responsibilities:

#### Core Application Modules
- **`streamlit_app.py`** - Main entry point with shared sidebar and navigation
- **`app_pages/chat.py`** - Chat interface with real-time agent interaction; also hosts the live working-tree diff in a "Changes" modal opened from above the chat input
- **`app_pages/usage.py`** - Token usage and cost dashboard
- **`app_pages/settings.py`** - User preferences and connection management

#### Agent & Tool System
- **`agent.py`** - Tool-calling loop with Weave tracing integration
- **`tools.py`** - File operations and shell execution tools
- **`mcp_servers.py`** - Model Context Protocol server management
- **`project_context.py`** - Skill and guidance file auto-detection

#### Service Integration
- **`wb_client.py`** - W&B Inference API client with Weave initialization
- **`git_ops.py`** - Git operations and repository management
- **`account.py`** - GitHub authentication and credential management
- **`models.py`** - Model metadata and pricing information

#### UI Components
- **`chat_input.py`** - Slash command autocomplete component
- **`theme_switcher.py`** - Theme management component
- **`actions.py`** - Cross-page callback handlers

#### Data Management
- **`chats.py`** - Multi-chat persistence and threading
- **`usage.py`** - Token usage tracking and cost calculation

### Data Flow

1. **User Interaction** → Chat page captures input
2. **Agent Processing** → Tool calls and model responses
3. **Tool Execution** → File operations, shell commands, or MCP calls
4. **Result Processing** → Updates to chat history and UI
5. **Persistence** → Chat history, usage data, and preferences saved
6. **Observability** → Weave tracing for all operations

### Key Design Patterns

- **Separation of Concerns**: Each module has a single responsibility
- **Immutable State**: Chat objects are locked during background processing
- **Streaming Architecture**: Real-time token streaming with event-based updates
- **Persistent Storage**: All user data survives app restarts
- **Cross-Platform**: Works in both web and packaged desktop modes

## Project layout

- `streamlit_app.py` — Entry point. Page config, session-state init, shared sidebar (file changes), and `st.navigation` between the chat, usage, and settings pages.
- `app_pages/chat.py` — The chat page (history + chat input + workdir + mode/model controls). Captures token usage from each turn and persists it.
- `app_pages/usage.py` — The usage and cost dashboard.
- `app_pages/settings.py` — GitHub PAT verify-and-save, theme info, W&B Inference Connect / Disconnect / Forget, and the MCP servers card + add/edit dialog.
- `actions.py` — Cross-page callbacks (recents, folder picker, Connect, GitHub PAT) imported by every page.
- `chat_input.py` — Slash-command autocomplete enhancer. CCv2 component that attaches a floating dropdown to `st.chat_input` while typing `/`.
- `theme_switcher.py` — In-app Light / Dark / System theme switcher. CCv2 component that reads/writes the `localStorage` key Streamlit's frontend reads on app boot, so the segmented control on the Settings page applies new themes via a single page reload (Streamlit has no programmatic theme API as of 1.57).
- `agent.py` — Tool-calling agent loop. Decorated with `@weave.op` so each turn is a single Weave trace. Streams with `stream_options={"include_usage": True}` so the dashboard has token counts to render.
- `tools.py` — Tool schemas and sandboxed executors. `dispatch` is decorated with `@weave.op(kind="tool")`.
- `wb_client.py` — OpenAI-client wrapper for W&B Inference, plus the Weave bootstrap (`init_weave`).
- `mcp_servers.py` — MCP runtime: `ServerConfig`, the registry singleton, the daemon-thread asyncio loop that owns every live MCP session, and the on-disk config at `~/.wb_coding_agent/mcp.json`. `dispatch` is decorated with `@weave.op(kind="tool")`.
- `project_context.py` — Auto-detects `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md`, `.cursor/rules`, and `SKILL.md` files under both `.cursor/skills/` and `.claude/skills/`, and selects which guidance to splice into each turn's system prompt.
- `models.py` — Single source of truth for model display labels, descriptions, context windows, parameter counts, and per-million-token pricing.
- `account.py` — GitHub PAT verification, opt-in W&B API key persistence, theme + avatar preferences. Pure stdlib.
- `usage.py` — Token-usage capture, cost compute, and aggregation. Owns `~/.wb_coding_agent/usage.jsonl`.
- `scripts/build_desktop.py` — Packaged-desktop-app build script.
- `scripts/build_desktop.sh` — Bootstrap wrapper that creates / re-syncs the build venv, then invokes `build_desktop.py`.
- `.streamlit/config.toml` — Streamlit runtime options for local dev (mirrored in the build script for packaged builds).
- `AGENTS.md` — Authoritative contract for AI agents working on this repo. Read it before contributing.

## Troubleshooting & Common Issues

### Connection Problems

**"Failed to connect to W&B Inference"**
- Verify your W&B API key is valid and has inference permissions
- Check your internet connection
- Ensure you're using a supported Python version (3.11+)

**"Model list empty after connect"**
- Your account may not have access to any inference models
- Contact W&B support for model access

### Performance Issues

**Slow response times**
- Large working directories can slow down file scanning
- Complex MCP servers may add latency
- Consider using smaller models for faster iteration

**High memory usage**
- The app maintains chat history and tool call context
- Archive or delete old chats to reduce memory footprint

### File System & Git Issues

**"Path escapes working directory" errors**
- The agent is restricted to the chosen working directory
- Use absolute paths within the working directory only

**Git operations failing**
- Ensure git is installed and in your PATH
- Verify you have appropriate permissions for the repository
- Check for merge conflicts that need manual resolution

### Theme & UI Issues

**Theme not applying correctly**
- The page reloads once after theme changes
- Some browsers may require a hard refresh (Ctrl+F5)

**Slash commands not working**
- Ensure skills are placed in the correct directories
- Check skill file formatting and trigger definitions

### Desktop Build Issues

**macOS Gatekeeper blocking app**
- Right-click the app and select "Open" for first launch
- Or run: `xattr -d com.apple.quarantine "dist/WB Coding Agent.app"`

**Build failures**
- Use Python 3.12 for desktop builds
- Ensure all build dependencies are installed
- Check the build script for specific error messages

## Getting Help

- **Documentation**: Refer to this README and `AGENTS.md`
- **W&B Support**: Contact support for inference-related issues
- **GitHub Issues**: Check for existing issues or create new ones
- **Community**: Join W&B community forums for discussion

## Contributing

Before making changes to this repository:

1. Read `AGENTS.md` thoroughly - it's the authoritative guide
2. Follow the module boundaries and responsibilities outlined
3. Test changes in both development and desktop build modes
4. Update documentation (README, AGENTS.md) alongside code changes
5. Ensure Weave tracing continues to work correctly

## Development & Contribution

### Development Practices

#### Code Organization
- Follow the module boundaries defined in `AGENTS.md`
- Keep Streamlit imports isolated to UI modules
- Use pure functions for business logic when possible
- Maintain clear separation between frontend and background processing

#### Testing Guidelines
- Test changes in both development and desktop build modes
- Verify Weave tracing continues to work correctly
- Test with both Agent and Ask only modes
- Validate MCP server integration when relevant

#### Performance Considerations
- Minimize blocking operations in the main thread
- Use appropriate caching for expensive operations
- Be mindful of token usage and cost implications
- Optimize file scanning for large working directories

### Building for Production

#### Desktop App Packaging
- Use the dedicated build environment (Python 3.12)
- Follow the bootstrap script for reproducible builds
- Test the packaged app on target platforms
- Handle code signing and notarization for distribution

#### Configuration Management
- Keep `.streamlit/config.toml` and build script options synchronized
- Maintain backward compatibility for user data files
- Handle credential migration carefully between versions

## License & Attribution

This project is built on:
- [Streamlit](https://streamlit.io/) for the web interface
- [W&B Inference](https://docs.wandb.ai/inference) for model serving
- [Weave](https://docs.wandb.ai/weave) for observability
- [MCP](https://modelcontextprotocol.io/) for tool extensibility

Refer to individual package licenses for specific terms.
