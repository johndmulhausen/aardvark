# W&B Inference Code Editing Agent

A local Streamlit app that turns any [W&B Inference](https://docs.wandb.ai/inference) model into a code editing agent. The agent uses OpenAI-compatible tool calling to autonomously read, edit, and run shell commands inside a working directory you choose.

## Features

- Bring your own W&B API key (kept only in session memory, never written to disk).
- Live-fetched model list from `https://api.inference.wandb.ai/v1/models`, with descriptions from the W&B docs — pick whichever model you want to code with from the dropdown above the chat input.
- Two modes: **Agent** (full read/write/edit, plus optional shell) and **Ask only** (read-only — the model can list and read files but cannot modify the project).
- Working-directory dropdown above the chat input remembers your recent project folders and includes a folder icon that opens a native OS picker. The recents list is persisted to `~/.wb_coding_agent/recent_dirs.json`.
- File tools: `list_files`, `read_file`, `write_file`, `edit_file`.
- `run_shell` tool runs commands inside the working directory with a 30-second timeout (Agent mode only).
- Every tool call is shown inline with arguments and a unified diff for writes/edits.
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
4. Above the chat input, pick a **Working directory** (use the folder icon to browse, or pick a recent), a **Mode** (Agent or Ask only), and a **Model** — then start chatting.

## Desktop build (optional)

If you'd rather launch the agent as a real desktop app instead of typing `streamlit run` every time, you can build a packaged binary that wraps the same Streamlit UI in a native window via [`streamlit-desktop-app`](https://pypi.org/project/streamlit-desktop-app/) + `pywebview`.

Use a separate Python 3.12 build environment — the packager only supports CPython 3.9–3.12:

```bash
uv venv --python 3.12 .venv-build
source .venv-build/bin/activate
uv pip install -e '.[desktop]'
python scripts/build_desktop.py
```

Output by platform:

- **macOS**: `dist/WB Coding Agent.app` (real `.app` bundle). Drag it to `/Applications`. The build is unsigned, so the first launch will be blocked by Gatekeeper — right-click → **Open**, or run `xattr -d com.apple.quarantine "dist/WB Coding Agent.app"` once.
- **Linux / Windows**: `dist/WB Coding Agent/` (a folder with the launcher binary inside). Zip and ship.

The packaged app hides Streamlit's hamburger menu and "Deploy" button so it looks like a desktop tool, not a hosted web app. The same options live in [`.streamlit/config.toml`](.streamlit/config.toml) for the `streamlit run` workflow.

## Safety

- All file paths are resolved against the working directory; paths that escape the directory are rejected.
- `run_shell` is always available in Agent mode and runs with `cwd` set to your working directory and a 30-second timeout. It is a real shell — only point the agent at directories whose contents (and surrounding environment) you're comfortable letting it touch. Switch to Ask only mode to disable shell, writes, and edits when you just want the model to look at code.
- The W&B API key is held in `st.session_state` for the duration of the browser session only. It is not persisted.
- Always review the unified diffs the agent emits before relying on its edits.

## Project layout

- `streamlit_app.py` — UI entry point.
- `agent.py` — Tool-calling agent loop. Decorated with `@weave.op` so each turn is a single Weave trace.
- `tools.py` — Tool schemas and sandboxed executors. `dispatch` is decorated with `@weave.op(kind="tool")`.
- `wb_client.py` — OpenAI-client wrapper for W&B Inference, plus the Weave bootstrap (`init_weave`).
- `scripts/build_desktop.py` — Packaged-desktop-app build script.
- `.streamlit/config.toml` — Streamlit runtime options for local dev (mirrored in the build script for packaged builds).
