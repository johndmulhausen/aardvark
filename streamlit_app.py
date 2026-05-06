"""Streamlit UI for the W&B Inference code editing agent."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from agent import run_agent_turn
from wb_client import init_weave, list_models, make_client

# Persistent recent-working-directories list. We store this in the user's home
# rather than session state so the dropdown is populated across app restarts
# (Streamlit's session state is per-browser-tab and cleared on reload).
RECENT_DIRS_FILE = Path.home() / ".wb_coding_agent" / "recent_dirs.json"
MAX_RECENT_DIRS = 10

st.set_page_config(
    page_title="W&B Coding Agent",
    page_icon=":material/smart_toy:",
    layout="wide",
)

PREFERRED_CODING_MODELS = [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "openai/gpt-oss-120b",
    "MiniMaxAI/MiniMax-M2.5",
    "zai-org/GLM-5.1",
    "deepseek-ai/DeepSeek-V3.1",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "openai/gpt-oss-20b",
]

# Model metadata transcribed from the W&B Inference "Available models" page:
# https://docs.wandb.ai/inference/models
#
# This is the single source of truth for model display labels and descriptions
# in the UI. The dropdown still uses the live `/v1/models` list as its option
# set; this map only attaches a friendly label and description when we have
# one. Models the API returns that we don't recognize fall back to the
# trailing slug as their label and render with no description.
MODEL_METADATA: dict[str, dict[str, str]] = {
    "deepseek-ai/DeepSeek-V3.1": {
        "label": "DeepSeek V3.1",
        "description": "A large hybrid model that supports both thinking and non-thinking modes via prompt templates.",
        "context": "161k",
        "params": "37B-671B (Active-Total)",
    },
    "google/gemma-4-31B-it": {
        "label": "Google Gemma 4 31B",
        "description": "Gemma 4 31B Dense is designed for advanced reasoning, agentic workflows, and longer context and is natively trained on 140+ languages.",
        "context": "262k",
        "params": "31B (Total)",
    },
    "ibm-granite/granite-4.1-8b": {
        "label": "IBM Granite 4.1 8B",
        "description": "Granite 4.1 8B is a long-context instruct model capable of enhanced tool calling, instruction following, and chat capabilities.",
        "context": "131k",
        "params": "8B (Total)",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "label": "Meta Llama 3.3 70B",
        "description": "Multilingual model excelling in conversational tasks, detailed instruction-following, and coding.",
        "context": "128k",
        "params": "70B (Total)",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "label": "Meta Llama 3.1 70B",
        "description": "Efficient conversational model optimized for responsive multilingual chatbot interactions.",
        "context": "128k",
        "params": "70B (Total)",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "label": "Meta Llama 3.1 8B",
        "description": "Efficient conversational model optimized for responsive multilingual chatbot interactions.",
        "context": "128k",
        "params": "8B (Total)",
    },
    "microsoft/Phi-4-mini-instruct": {
        "label": "Microsoft Phi 4 Mini 3.8B",
        "description": "Compact, efficient model ideal for fast responses in resource-constrained environments.",
        "context": "128k",
        "params": "3.8B (Total)",
    },
    "MiniMaxAI/MiniMax-M2.5": {
        "label": "MiniMax M2.5",
        "description": "MoE model with a highly sparse architecture designed for high-throughput and low latency with strong coding capabilities.",
        "context": "197k",
        "params": "10B-230B (Active-Total)",
    },
    "moonshotai/Kimi-K2.6": {
        "label": "Moonshot AI Kimi K2.6",
        "description": "Kimi K2.6 is a multimodal Mixture-of-Experts language model featuring 32 billion activated parameters and a total of 1 trillion parameters.",
        "context": "262k",
        "params": "32B-1T (Active-Total)",
    },
    "moonshotai/Kimi-K2.5": {
        "label": "Moonshot AI Kimi K2.5",
        "description": "Kimi K2.5 is a multimodal Mixture-of-Experts language model featuring 32 billion activated parameters and a total of 1 trillion parameters.",
        "context": "262k",
        "params": "32B-1T (Active-Total)",
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": {
        "label": "NVIDIA Nemotron 3 Super 120B",
        "description": "Nemotron 3 is a LatentMoE model designed to deliver strong agentic, reasoning, and conversational capabilities.",
        "context": "262k",
        "params": "12B-120B (Active-Total)",
    },
    "openai/gpt-oss-120b": {
        "label": "OpenAI GPT OSS 120B",
        "description": "Efficient Mixture-of-Experts model designed for high-reasoning, agentic and general-purpose use cases.",
        "context": "131k",
        "params": "5.1B-117B (Active-Total)",
    },
    "openai/gpt-oss-20b": {
        "label": "OpenAI GPT OSS 20B",
        "description": "Lower latency Mixture-of-Experts model trained on OpenAI's Harmony response format with reasoning capabilities.",
        "context": "131k",
        "params": "3.6B-20B (Active-Total)",
    },
    "OpenPipe/Qwen3-14B-Instruct": {
        "label": "OpenPipe Qwen3 14B Instruct",
        "description": "An efficient multilingual, dense, instruction-tuned model, optimized by OpenPipe for building agents with finetuning.",
        "context": "32.8k",
        "params": "14.8B (Total)",
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "label": "Qwen3.5 35B A3B",
        "description": "Qwen3.5-35B-A3B is an open-weights multimodal MoE model built for efficient, high-throughput inference across chat, reasoning, and agentic tasks.",
        "context": "262k",
        "params": "3B-35B (Active-Total)",
    },
    "Qwen/Qwen3-235B-A22B-Thinking-2507": {
        "label": "Qwen3 235B A22B Thinking-2507",
        "description": "High-performance Mixture-of-Experts model optimized for structured reasoning, math, and long-form generation.",
        "context": "262k",
        "params": "22B-235B (Active-Total)",
    },
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "label": "Qwen3 235B A22B-2507",
        "description": "Efficient multilingual, Mixture-of-Experts, instruction-tuned model, optimized for logical reasoning.",
        "context": "262k",
        "params": "22B-235B (Active-Total)",
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "label": "Qwen3 30B A3B",
        "description": "Qwen3-30B-A3B-Instruct-2507 is a 30.5B MoE instruction-tuned model with enhanced reasoning, coding, and long-context understanding.",
        "context": "262k",
        "params": "3.3B-30.5B (Active-Total)",
    },
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
        "label": "Qwen3 Coder 480B A35B",
        "description": "Mixture-of-Experts model optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning.",
        "context": "262k",
        "params": "35B-480B (Active-Total)",
    },
    "zai-org/GLM-5.1": {
        "label": "Z.AI GLM 5.1",
        "description": "Powerful MoE model for long-horizon agentic engineering and advanced reasoning.",
        "context": "203k",
        "params": "40B-744B (Active-Total)",
    },
    "deepseek-ai/DeepSeek-V4-Flash": {
        "label": "DeepSeek V4-Flash (experimental)",
        "description": "DeepSeek V4-Flash is an MoE model with 1M context length great for coding, reasoning, and agentic workloads.",
        "context": "1000k",
        "params": "13B-284B (Active-Total)",
    },
    "Qwen/Qwen3.5-27B": {
        "label": "Qwen3.5 27B (experimental)",
        "description": "Qwen3.5-27B is a dense model from the Qwen3.5 family built for high performance across a large range of benchmarks.",
        "context": "262k",
        "params": "27B (Total)",
    },
}


def _model_label(model_id: str) -> str:
    """Return the friendly display label for a model id, or its slug fallback."""
    meta = MODEL_METADATA.get(model_id)
    return meta["label"] if meta else model_id.split("/")[-1]


def _model_description(model_id: str) -> str | None:
    """Return the W&B docs description for a model id, or ``None`` if unknown."""
    meta = MODEL_METADATA.get(model_id)
    return meta["description"] if meta else None


def _load_recent_dirs() -> list[str]:
    """Load recently used working directories from disk.

    Returns an empty list on any failure (file missing, malformed JSON, etc.) so
    the UI degrades to "no recents yet" rather than crashing on startup.
    """
    try:
        raw = json.loads(RECENT_DIRS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw if isinstance(p, str)][:MAX_RECENT_DIRS]


def _save_recent_dirs(dirs: list[str]) -> None:
    """Persist the recent-directories list to disk; failures are best-effort."""
    try:
        RECENT_DIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_DIRS_FILE.write_text(json.dumps(dirs, indent=2), encoding="utf-8")
    except OSError:
        pass


def _record_recent_dir(path: str) -> None:
    """Move ``path`` to the front of the recents list and persist.

    The list is deduplicated, capped at :data:`MAX_RECENT_DIRS`, and stored as
    absolute paths so entries remain valid even if the app is later launched
    from a different cwd.
    """
    ss = st.session_state
    abs_path = str(Path(path).expanduser().resolve())
    existing = [d for d in ss.recent_dirs if d != abs_path]
    ss.recent_dirs = ([abs_path] + existing)[:MAX_RECENT_DIRS]
    _save_recent_dirs(ss.recent_dirs)


def _pick_directory(initial: str | None = None) -> str | None:
    """Open a native folder picker and return the chosen absolute path.

    Returns ``None`` if the user cancelled or the picker could not be launched.
    On macOS we shell out to ``osascript`` (no extra deps, no threading
    concerns); on other platforms we run a tiny ``tkinter.filedialog`` snippet
    in a subprocess so it doesn't have to share a thread with Streamlit.
    """
    if sys.platform == "darwin":
        script = "POSIX path of (choose folder with prompt \"Choose working directory\")"
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        chosen = result.stdout.strip().rstrip("/")
        return chosen or None

    snippet = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        f"path = filedialog.askdirectory(initialdir={initial or ''!r})\n"
        "print(path or '')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    chosen = result.stdout.strip()
    return chosen or None


TOOL_ICONS = {
    "list_files": ":material/folder_open:",
    "read_file": ":material/description:",
    "write_file": ":material/edit_note:",
    "edit_file": ":material/edit:",
    "run_shell": ":material/terminal:",
}


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("api_key", "")
    ss.setdefault("project", "")
    ss.setdefault("client", None)
    ss.setdefault("models", [])
    ss.setdefault("model", None)
    ss.setdefault("mode", "agent")
    ss.setdefault("recent_dirs", _load_recent_dirs())
    default_wd = ss.recent_dirs[0] if ss.recent_dirs else os.getcwd()
    ss.setdefault("working_dir", default_wd)
    ss.setdefault("messages", [])
    ss.setdefault("ui_turns", [])
    ss.setdefault("connect_error", None)
    ss.setdefault("weave_project", None)
    ss.setdefault("weave_error", None)
    # Drives the sidebar Connection expander. Starts True so first-time users
    # see the form, then is flipped to False by ``_on_connect`` after a
    # successful Connect so the panel collapses out of the way. The expander
    # uses ``key="conn_open"`` + ``on_change="rerun"`` so manual toggles by the
    # user write back here and persist across reruns.
    ss.setdefault("conn_open", True)


def _sort_models(models: list[str]) -> list[str]:
    preferred = [m for m in PREFERRED_CODING_MODELS if m in models]
    rest = sorted(m for m in models if m not in preferred)
    return preferred + rest


def _connect(api_key: str, project: str) -> None:
    api_key = api_key.strip()
    project = project.strip()
    if not api_key:
        st.session_state.connect_error = "Enter a W&B API key first."
        return
    try:
        client = make_client(api_key=api_key, project=project or None)
        models = list_models(client)
    except Exception as e:
        st.session_state.client = None
        st.session_state.models = []
        st.session_state.connect_error = f"Could not connect: {e}"
        return
    st.session_state.client = client
    st.session_state.models = _sort_models(models)
    st.session_state.connect_error = None
    if st.session_state.model not in st.session_state.models:
        st.session_state.model = st.session_state.models[0] if st.session_state.models else None

    # Initialize Weave so every chat-completion call from agent.py is captured
    # in the user's W&B project. Failures here (auth, network, project access)
    # don't fail the connect — the @weave.op decorators are no-ops without an
    # initialized Weave client, so the agent still runs unchanged.
    try:
        _, resolved_project = init_weave(api_key=api_key, project=project or None)
        st.session_state.weave_project = resolved_project
        st.session_state.weave_error = None
    except Exception as e:
        st.session_state.weave_project = None
        st.session_state.weave_error = str(e)


def _on_connect() -> None:
    """Connect button callback.

    Runs ``_connect`` and, on success, collapses the sidebar Connection
    expander by clearing ``conn_open`` so the panel stops occupying space once
    the user is connected. We do this in an ``on_click`` callback (rather than
    inline after the button) because callbacks fire *before* the next script
    rerun, which means the expander reads the updated ``conn_open`` value on
    its very first render of that rerun and renders collapsed immediately.
    """
    ss = st.session_state
    _connect(ss.api_key, ss.project)
    if ss.client is not None and ss.connect_error is None:
        ss.conn_open = False


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.ui_turns = []


def _short_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if isinstance(v, str):
            preview = v.replace("\n", " ")
            if len(preview) > 40:
                preview = preview[:40] + "..."
            parts.append(f'{k}="{preview}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _render_tool_event(call_event: dict[str, Any], result_event: dict[str, Any] | None) -> None:
    name = call_event["name"]
    args = call_event.get("args", {}) or {}
    icon = TOOL_ICONS.get(name, ":material/build:")
    label = f"{icon} `{name}`({_short_args(args)})"

    expanded = result_event is None
    with st.expander(label, expanded=expanded):
        st.markdown("**Arguments**")
        st.code(json.dumps(args, indent=2), language="json")

        if result_event is None:
            st.caption("Running...")
            return

        result = result_event.get("result", {}) or {}
        if "error" in result:
            st.error(result["error"], icon=":material/error:")
            return

        diff = result.get("diff")
        if diff and diff != "(no change)":
            st.markdown("**Diff**")
            if diff == "(new file)":
                st.caption("New file created.")
            else:
                st.code(diff, language="diff")

        if name == "list_files" and "listing" in result:
            st.markdown("**Listing**")
            st.code(result["listing"], language="text")
        elif name == "read_file" and "content" in result:
            total = result.get("total_lines")
            shown = result.get("shown_lines")
            caption = f"{total} lines total"
            if shown:
                caption += f"; showing {shown[0]}-{shown[1]}"
            st.caption(caption)
            st.code(result["content"], language="text")
        elif name == "run_shell":
            cols = st.columns([1, 1, 4])
            cols[0].metric("Exit code", result.get("exit_code", "?"))
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if stdout:
                st.markdown("**stdout**")
                st.code(stdout, language="text")
            if stderr:
                st.markdown("**stderr**")
                st.code(stderr, language="text")
        elif name in ("write_file", "edit_file"):
            if result.get("ok"):
                msg_parts = [f"Wrote `{result.get('path')}`"]
                if "bytes_written" in result:
                    msg_parts.append(f"({result['bytes_written']} bytes)")
                st.caption(" ".join(msg_parts))


def _render_assistant_turn(turn: dict[str, Any]) -> None:
    events: list[dict[str, Any]] = turn.get("events", [])

    pending_calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev["type"] == "tool_call":
            pending_calls[ev["id"]] = ev
        elif ev["type"] == "tool_result":
            results[ev["id"]] = ev

    with st.chat_message("assistant"):
        for ev in events:
            if ev["type"] == "assistant_text":
                content = ev.get("content") or ""
                if content.strip():
                    st.markdown(content)
            elif ev["type"] == "tool_call":
                _render_tool_event(ev, results.get(ev["id"]))
            elif ev["type"] == "tool_result":
                continue
            elif ev["type"] == "error":
                st.error(ev["message"], icon=":material/error:")


def _render_user_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(turn["content"])


def _render_history() -> None:
    for turn in st.session_state.ui_turns:
        if turn["role"] == "user":
            _render_user_turn(turn)
        else:
            _render_assistant_turn(turn)


def _run_turn(prompt: str) -> None:
    ss = st.session_state
    ss.messages.append({"role": "user", "content": prompt})
    ss.ui_turns.append({"role": "user", "content": prompt})

    _render_user_turn({"role": "user", "content": prompt})

    assistant_turn: dict[str, Any] = {"role": "assistant", "events": []}
    ss.ui_turns.append(assistant_turn)

    working_dir = Path(ss.working_dir).expanduser().resolve()
    short_model = ss.model.split("/")[-1] if ss.model else ""

    with st.chat_message("assistant"):
        live_container = st.container()
        # The status placeholder is created AFTER ``live_container`` so the
        # "Thinking..." caption always renders below the most recent streamed
        # text or tool call, like the cursor of a terminal that's still
        # waiting on output. It's hidden as soon as the next event arrives
        # and re-shown after each tool result while the model thinks again.
        status_ph = st.empty()
        thinking_visible = {"on": False}

        def _show_thinking() -> None:
            if not thinking_visible["on"]:
                status_ph.caption(
                    f":material/auto_awesome: Thinking with `{short_model}`..."
                )
                thinking_visible["on"] = True

        def _hide_thinking() -> None:
            if thinking_visible["on"]:
                status_ph.empty()
                thinking_visible["on"] = False

        _show_thinking()

        results_by_id: dict[str, dict[str, Any]] = {}
        call_placeholders: dict[str, Any] = {}
        call_args: dict[str, dict[str, Any]] = {}

        # Streaming state for the assistant text segment currently being
        # produced. ``text_ph`` is the live placeholder receiving deltas; both
        # are reset after the segment is finalized (by an ``assistant_text``
        # event) or interrupted (by a ``tool_call`` event).
        text_ph: Any = None
        text_buf = ""

        try:
            events_iter = run_agent_turn(
                client=ss.client,
                model=ss.model,
                messages=ss.messages,
                working_dir=working_dir,
                mode=ss.mode,
            )
            for event in events_iter:
                etype = event["type"]
                if etype == "assistant_text_delta":
                    _hide_thinking()
                    if text_ph is None:
                        with live_container:
                            text_ph = st.empty()
                        text_buf = ""
                    text_buf += event.get("content") or ""
                    text_ph.markdown(text_buf)
                elif etype == "assistant_text":
                    _hide_thinking()
                    content = event.get("content") or ""
                    if text_ph is not None:
                        if content:
                            text_ph.markdown(content)
                        else:
                            text_ph.empty()
                        text_ph = None
                        text_buf = ""
                    elif content.strip():
                        with live_container:
                            st.markdown(content)
                    assistant_turn["events"].append(event)
                elif etype == "tool_call":
                    _hide_thinking()
                    text_ph = None
                    text_buf = ""
                    assistant_turn["events"].append(event)
                    call_args[event["id"]] = event
                    with live_container:
                        ph = st.empty()
                    call_placeholders[event["id"]] = ph
                    with ph.container():
                        _render_tool_event(event, None)
                elif etype == "tool_result":
                    assistant_turn["events"].append(event)
                    results_by_id[event["id"]] = event
                    ph = call_placeholders.get(event["id"])
                    call_ev = call_args.get(event["id"])
                    if ph is not None and call_ev is not None:
                        with ph.container():
                            _render_tool_event(call_ev, event)
                    # After a tool result, the agent re-queries the model.
                    # Show the thinking indicator while we wait for either the
                    # next stream of deltas or another tool call.
                    _show_thinking()
                elif etype == "error":
                    _hide_thinking()
                    assistant_turn["events"].append(event)
                    with live_container:
                        st.error(event["message"], icon=":material/error:")
        except Exception as e:
            with live_container:
                st.error(f"Agent crashed: {e}", icon=":material/error:")
            assistant_turn["events"].append({"type": "error", "message": str(e)})
        finally:
            _hide_thinking()


def _shorten_path(path: str) -> str:
    """Collapse the user's home directory to ``~`` for compact display."""
    if not path:
        return path
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _on_working_dir_select() -> None:
    """Selectbox callback: persist the freshly chosen working directory.

    Runs before the next rerun renders, so by the time the selectbox is drawn
    again the new value is already in ``recent_dirs`` and will appear as a
    dropdown option (preventing it from being silently dropped).
    """
    chosen = st.session_state.get("wd_select")
    if chosen:
        st.session_state.working_dir = chosen
        _record_recent_dir(chosen)


def _render_chat_controls() -> None:
    """Render the working-directory, mode, and model selectors above chat input.

    Layout (top to bottom, all directly above the sticky chat input):

    1. Working-directory dropdown (recents + free text) plus a folder-picker
       button that launches a native OS directory chooser.
    2. Mode and Model dropdowns side by side.
    3. Optional caption with the selected model's W&B docs description.
    """
    ss = st.session_state

    wd_options: list[str] = []
    if ss.working_dir:
        wd_options.append(ss.working_dir)
    for d in ss.recent_dirs:
        if d not in wd_options:
            wd_options.append(d)

    wd_cols = st.columns([10, 1], vertical_alignment="bottom")
    with wd_cols[0]:
        st.selectbox(
            "Working directory",
            options=wd_options,
            index=wd_options.index(ss.working_dir) if ss.working_dir in wd_options else 0,
            key="wd_select",
            on_change=_on_working_dir_select,
            format_func=_shorten_path,
            accept_new_options=True,
            placeholder="Choose or paste a directory",
            help=(
                "Recent working directories. Pick from the list, paste a "
                "custom path, or click the folder icon to browse."
            ),
        )
    with wd_cols[1]:
        if st.button(
            "",
            icon=":material/folder_open:",
            key="wd_pick_btn",
            help="Browse for a working directory",
            width="stretch",
        ):
            chosen = _pick_directory(initial=ss.working_dir)
            if chosen:
                ss.working_dir = chosen
                _record_recent_dir(chosen)
                st.rerun()

    cols = st.columns([1, 2], vertical_alignment="bottom")
    with cols[0]:
        mode_options = ["agent", "ask"]
        ss.mode = st.selectbox(
            "Mode",
            options=mode_options,
            index=mode_options.index(ss.mode) if ss.mode in mode_options else 0,
            format_func=lambda m: "Agent" if m == "agent" else "Ask only",
            help=(
                "Agent can read, write, edit files (and run shell if enabled). "
                "Ask only is read-only — the model can list and read files but "
                "cannot modify the project."
            ),
        )
    with cols[1]:
        if ss.models:
            ss.model = st.selectbox(
                "Model",
                options=ss.models,
                index=ss.models.index(ss.model) if ss.model in ss.models else 0,
                format_func=_model_label,
                help="Switch which W&B Inference model handles the next turn.",
            )
        else:
            st.selectbox(
                "Model",
                options=["Connect to load models"],
                disabled=True,
            )
    desc = _model_description(ss.model) if ss.model else None
    if desc:
        st.caption(desc)


def _count_diff_lines(diff: str) -> tuple[int, int]:
    """Count added/removed lines in a unified diff string.

    Header lines (``--- a/...``, ``+++ b/...``) are excluded so they don't
    inflate the counts. The two sentinel diffs produced by ``tools.py`` —
    ``"(no change)"`` and ``"(new file)"`` — return ``(0, 0)`` because they
    don't carry per-line information.
    """
    if not diff or diff in ("(no change)", "(new file)"):
        return 0, 0
    additions = 0
    deletions = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _collect_file_changes(ui_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate successful write/edit tool results into per-file summaries.

    Walks every assistant turn's events in order and folds successful
    ``write_file`` / ``edit_file`` ``tool_result`` events into one entry per
    path. The returned list is ordered most-recent-first (the file most
    recently touched sits at the top), so the sidebar reflects the agent's
    latest activity without the user having to scroll.

    Each entry has shape::

        {
            "path": str,
            "additions": int,   # cumulative +lines across all edits
            "deletions": int,   # cumulative -lines across all edits
            "created": bool,    # any op against this path created it
            "edits": int,       # total successful write/edit ops
            "latest_diff": str, # diff text from the most recent op
        }
    """
    summaries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for turn in ui_turns:
        if turn.get("role") != "assistant":
            continue
        for ev in turn.get("events", []):
            if ev.get("type") != "tool_result":
                continue
            if ev.get("name") not in ("write_file", "edit_file"):
                continue
            result = ev.get("result") or {}
            if not result.get("ok"):
                continue
            path = result.get("path")
            if not path:
                continue
            diff = result.get("diff") or ""
            adds, dels = _count_diff_lines(diff)
            entry = summaries.get(path)
            if entry is None:
                entry = {
                    "path": path,
                    "additions": 0,
                    "deletions": 0,
                    "created": False,
                    "edits": 0,
                    "latest_diff": "",
                }
                summaries[path] = entry
            entry["additions"] += adds
            entry["deletions"] += dels
            entry["edits"] += 1
            if result.get("created") or diff == "(new file)":
                entry["created"] = True
            entry["latest_diff"] = diff
            if path in order:
                order.remove(path)
            order.append(path)
    return [summaries[p] for p in reversed(order)]


def _render_file_changes() -> None:
    """Render the sidebar "File changes" panel.

    Reads ``st.session_state.ui_turns`` and renders nothing when no successful
    write/edit has happened yet, so quiet sessions (read-only Ask mode, or
    pre-first-edit Agent mode) don't show an empty panel. When there is at
    least one change, an expander summarizes each touched file with cumulative
    +/- counts and a nested expander revealing the most recent diff.
    """
    changes = _collect_file_changes(st.session_state.ui_turns)
    if not changes:
        return

    label = f"File changes \u00b7 {len(changes)} file{'s' if len(changes) != 1 else ''}"
    with st.expander(label, icon=":material/edit_note:", expanded=True):
        for i, entry in enumerate(changes):
            if i > 0:
                st.divider()
            icon = (
                ":material/add_circle:" if entry["created"] else ":material/edit_note:"
            )
            st.markdown(f"{icon} `{entry['path']}`")

            caption_parts: list[str] = []
            if entry["created"]:
                caption_parts.append(":green[New file]")
            if entry["additions"] or entry["deletions"]:
                caption_parts.append(
                    f":green[+{entry['additions']}] :red[\u2212{entry['deletions']}]"
                )
            if entry["edits"] > 1:
                caption_parts.append(f"{entry['edits']} edits")
            if caption_parts:
                st.caption(" \u00b7 ".join(caption_parts))

            diff = entry["latest_diff"]
            if diff and diff not in ("(no change)", "(new file)"):
                with st.expander("Diff", expanded=False):
                    st.code(diff, language="diff")


def _render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### :material/smart_toy: W&B Coding Agent")
        st.caption("A code editing agent powered by W&B Inference.")

        # The Connection panel doubles as a status indicator: when not yet
        # connected (or after an error) it renders expanded with the form
        # visible; once connected, ``_on_connect`` flips ``conn_open`` to
        # False so it collapses to a one-line "Connected" header with a green
        # check icon. The user can still click the chevron to re-open it any
        # time — ``on_change="rerun"`` writes that toggle back into
        # ``conn_open`` so it persists across reruns.
        is_connected = ss.client is not None and ss.connect_error is None
        if is_connected:
            conn_label = f":green[Connected] · {len(ss.models)} models"
            conn_icon = ":material/check_circle:"
        else:
            conn_label = "Connection"
            conn_icon = ":material/link:"

        with st.expander(
            conn_label,
            icon=conn_icon,
            expanded=ss.conn_open,
            key="conn_open",
            on_change="rerun",
        ):
            st.text_input(
                "W&B API key",
                key="api_key",
                type="password",
                help="Get one at wandb.ai/settings. Held in session memory only.",
            )
            st.text_input(
                "Project (optional)",
                key="project",
                placeholder="team/project",
                help="Optional. Used for usage attribution.",
            )
            st.button(
                "Connect",
                icon=":material/link:",
                width="stretch",
                type="primary",
                on_click=_on_connect,
            )

            if ss.connect_error:
                st.error(ss.connect_error, icon=":material/error:")
            elif ss.client is not None:
                st.success(
                    f"Connected. {len(ss.models)} models available.",
                    icon=":material/check_circle:",
                )
                if ss.weave_project:
                    st.caption(
                        f":material/sensors: Tracing turns to W&B Weave at "
                        f"`{ss.weave_project}`."
                    )
                elif ss.weave_error:
                    st.caption(
                        f":material/sensors_off: Weave tracing disabled: "
                        f"{ss.weave_error}"
                    )

        _render_file_changes()

        st.button("Clear chat", icon=":material/delete:", width="stretch", on_click=_clear_chat)


def main() -> None:
    _init_state()
    _render_sidebar()

    ss = st.session_state

    st.title("Code editing agent")
    st.caption(
        "Powered by [W&B Inference](https://docs.wandb.ai/inference). "
        "Point it at a working directory, pick a mode and model below the chat, "
        "and ask it to read or modify your code."
    )

    ready = ss.client is not None and ss.model is not None

    if not ready:
        st.info(
            "Enter your W&B API key and click **Connect** in the sidebar to get started.",
            icon=":material/login:",
        )
        return

    _render_history()
    _render_chat_controls()

    wd_ok = Path(ss.working_dir).expanduser().is_dir()
    if not wd_ok:
        st.warning(
            "Choose a valid working directory from the picker above before chatting.",
            icon=":material/folder_off:",
        )
        return

    prompt = st.chat_input("Ask the agent to read or modify your code...")
    if prompt:
        _run_turn(prompt)
        st.rerun()


main()
