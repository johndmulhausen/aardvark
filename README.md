# W&B Inference Code Editing Agent

A local Streamlit app that turns any [W&B Inference](https://docs.wandb.ai/inference) model into a code editing agent. The agent uses OpenAI-compatible tool calling to autonomously read, edit, and (optionally) run shell commands inside a working directory you choose.

## Features

- Bring your own W&B API key (kept only in session memory, never written to disk).
- Live-fetched model list from `https://api.inference.wandb.ai/v1/models` — pick whichever model you want to code with.
- File tools: `list_files`, `read_file`, `write_file`, `edit_file`.
- Optional `run_shell` tool (opt-in via sidebar toggle).
- Every tool call is shown inline with arguments and a unified diff for writes/edits.

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
4. Pick a model, set your working directory, and start chatting.

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
- `run_shell` is disabled by default. When enabled, commands run with a 30-second timeout and stdout/stderr are captured.
- The W&B API key is held in `st.session_state` for the duration of the browser session only. It is not persisted.
- Always review the unified diffs the agent emits before relying on its edits.

## Project layout

- `streamlit_app.py` — UI entry point.
- `agent.py` — Tool-calling agent loop.
- `tools.py` — Tool schemas and sandboxed executors.
- `wb_client.py` — OpenAI-client wrapper for W&B Inference.
- `scripts/build_desktop.py` — Packaged-desktop-app build script.
- `.streamlit/config.toml` — Streamlit runtime options for local dev (mirrored in the build script for packaged builds).
