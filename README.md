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
