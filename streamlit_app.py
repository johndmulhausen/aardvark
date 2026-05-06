"""Streamlit UI for the W&B Inference code editing agent."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

import mcp_servers
import project_context
from agent import run_agent_turn
from chat_input import mount_slash_autocomplete
from mcp_servers import MCPRegistry, ServerConfig, make_server_id
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

# Fallback icon for any ``mcp__<server>__<tool>`` tool name. Looked up in
# ``_render_tool_event`` after the per-name lookup misses; lets the UI
# render every MCP tool with a consistent badge instead of the generic
# ``:material/build:`` fallback.
MCP_TOOL_ICON = ":material/extension:"


@st.cache_resource
def _get_mcp_registry() -> MCPRegistry:
    """Return the process-wide MCP registry, lazily initialized.

    Wrapped in ``@st.cache_resource`` so the registry (and the daemon
    thread + live MCP sessions it holds) survives Streamlit reruns. The
    underlying singleton in ``mcp_servers`` would survive anyway, but
    routing through the cache makes the intent explicit and keeps the
    dependency boundary tidy.
    """
    return mcp_servers.get_registry()


@st.cache_data(ttl=5, show_spinner=False)
def _scan_project_summary(working_dir: str) -> dict[str, Any]:
    """Cached UI summary of the project context for ``working_dir``.

    The 5-second TTL keeps the dropdown / popover responsive without
    re-globbing the working directory on every Streamlit rerun. The
    actual per-turn scan that drives the system prompt happens inside
    ``agent.run_agent_turn`` and is not cached, so the model always sees
    a fresh ``AGENTS.md`` if the user just edited it.
    """
    if not working_dir:
        return {
            "agents_md": [],
            "cursor_rules": [],
            "workspace_skills": [],
            "user_skills": [],
            "all_skills": [],
            "slug_conflicts": [],
        }
    ctx = project_context.scan(Path(working_dir))
    return project_context.summary(ctx)


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
    # MCP server config dialog state. ``mcp_dialog_open`` gates the modal
    # and ``mcp_dialog_editing`` is either ``None`` (add) or the id of an
    # existing server being edited.
    ss.setdefault("mcp_dialog_open", False)
    ss.setdefault("mcp_dialog_editing", None)


def _sort_models(models: list[str]) -> list[str]:
    """Sort model IDs alphabetically by the label shown in the dropdown.

    Sorting by the displayed label (case-insensitive) keeps the dropdown
    visually alphabetical for the user, even when the model id casing
    (``OpenPipe/...`` vs. ``openai/...``) would produce a surprising order.
    """
    return sorted(models, key=lambda m: _model_label(m).casefold())


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
    is_mcp = name.startswith(mcp_servers.TOOL_NAME_PREFIX)
    if is_mcp:
        icon = MCP_TOOL_ICON
    else:
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

        if is_mcp:
            _render_mcp_result(result)
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


def _render_mcp_result(result: dict[str, Any]) -> None:
    """Render an MCP ``CallToolResult`` payload.

    Walks ``content`` blocks in order; each block has a ``type`` field
    (``text``, ``image``, ``resource``, ...) that we case on. Unknown
    block types fall back to a JSON dump so the user can still see what
    the server returned.
    """
    if result.get("isError") or result.get("is_error"):
        st.warning("Server reported an error.", icon=":material/warning:")
    blocks = result.get("content") or []
    if not isinstance(blocks, list):
        st.code(json.dumps(result, indent=2), language="json")
        return
    for block in blocks:
        if not isinstance(block, dict):
            st.write(block)
            continue
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "")
            st.markdown(text) if text and len(text) < 1000 else st.code(text or "", language="text")
        elif btype == "image":
            data = block.get("data")
            mime = block.get("mimeType", "image/png")
            if data:
                st.image(f"data:{mime};base64,{data}")
        elif btype == "resource":
            resource = block.get("resource") or {}
            uri = resource.get("uri", "")
            st.caption(f":material/link: {uri}")
            text = resource.get("text")
            if text:
                st.code(text, language="text")
        else:
            st.code(json.dumps(block, indent=2), language="json")
    structured = result.get("structuredContent") or result.get("structured_content")
    if structured:
        st.markdown("**Structured content**")
        st.code(json.dumps(structured, indent=2), language="json")


def _render_skills_loaded(event: dict[str, Any]) -> None:
    """Show a caption listing skills the auto-loader picked for this turn.

    Persisted on every assistant turn so replays match what the user saw
    live. Kept compact: a single ``st.caption`` that lists each selected
    skill (with its trigger reason) and any unknown ``/foo`` slugs the
    user typed so they catch typos.
    """
    selected = event.get("selected") or []
    unknown = event.get("unknown_slash") or []
    if not selected and not unknown:
        return
    parts: list[str] = []
    if selected:
        chips = []
        for s in selected:
            slug = s.get("slug", "")
            reason = s.get("trigger_reason", "")
            chips.append(f"`/{slug}` ({reason})")
        parts.append(
            f":material/auto_fix_high: Loaded {len(selected)} skill"
            f"{'s' if len(selected) != 1 else ''}: " + ", ".join(chips)
        )
    if unknown:
        parts.append(
            ":material/help: Unknown slash command"
            f"{'s' if len(unknown) != 1 else ''}: "
            + ", ".join(f"`/{u}`" for u in unknown)
        )
    st.caption(" \u00b7 ".join(parts))


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
            if ev["type"] == "skills_loaded":
                _render_skills_loaded(ev)
            elif ev["type"] == "assistant_text":
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
                if etype == "skills_loaded":
                    assistant_turn["events"].append(event)
                    selected = event.get("selected") or []
                    unknown = event.get("unknown_slash") or []
                    if selected or unknown:
                        with live_container:
                            _render_skills_loaded(event)
                    continue
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


def _render_project_context_indicator(working_dir: str) -> None:
    """Render a compact "guidance files detected" expander.

    Shown directly below the working-directory selector. Splits into two
    sections: eagerly-loaded files (AGENTS.md / CLAUDE.md / .cursor/rules)
    and conditionally-loaded skills with their slash command. When
    nothing is detected, the function renders nothing — quiet workspaces
    don't need a panel.
    """
    summary = _scan_project_summary(working_dir or "")
    eager: list[dict[str, Any]] = list(summary.get("agents_md", [])) + list(
        summary.get("cursor_rules", [])
    )
    all_skills: list[dict[str, Any]] = summary.get("all_skills", [])
    if not eager and not all_skills:
        return

    pieces: list[str] = []
    if eager:
        pieces.append(f"{len(eager)} guidance file{'s' if len(eager) != 1 else ''}")
    if all_skills:
        pieces.append(f"{len(all_skills)} skill{'s' if len(all_skills) != 1 else ''}")
    label = "Project context \u00b7 " + ", ".join(pieces)

    with st.expander(label, icon=":material/menu_book:", expanded=False):
        if eager:
            st.markdown("**Eagerly loaded** (always sent to the model)")
            for entry in eager:
                marker = " :gray-badge[truncated]" if entry.get("truncated") else ""
                st.markdown(f"- `{entry['path']}`{marker}")
        if all_skills:
            if eager:
                st.divider()
            st.markdown("**Conditionally loaded skills**")
            st.caption(
                "Auto-loaded when your message matches the keywords, or "
                "force-loaded with `/<slug>` (type `/` in the chat input "
                "for inline autocomplete)."
            )
            for skill in all_skills:
                scope = skill.get("scope", "workspace")
                badge = (
                    ":blue-badge[workspace]" if scope == "workspace" else ":gray-badge[user]"
                )
                slug = skill.get("slug", "")
                desc = skill.get("description", "")
                st.markdown(f"- `/{slug}` {badge} \u2014 {desc}")
                triggers = skill.get("triggers") or []
                if triggers:
                    preview = ", ".join(f"`{t}`" for t in triggers[:8])
                    if len(triggers) > 8:
                        preview += f", ... +{len(triggers) - 8} more"
                    st.caption(f"Triggers: {preview}")
        conflicts = summary.get("slug_conflicts") or []
        if conflicts:
            st.warning(
                "User skills shadowed by workspace skills with the same slug: "
                + ", ".join(f"`/{c}`" for c in conflicts),
                icon=":material/warning:",
            )


def _render_skills_popover(working_dir: str) -> None:
    """Render a popover next to the chat input listing every skill slug.

    Complements the inline ``/`` autocomplete (rendered by
    :func:`chat_input.mount_slash_autocomplete`) — the popover is the
    "browse everything in one shot" affordance for users who want to see
    the full skill catalog with descriptions before typing. Hidden when
    no skills are detected.
    """
    summary = _scan_project_summary(working_dir or "")
    all_skills: list[dict[str, Any]] = summary.get("all_skills", [])
    if not all_skills:
        return
    with st.popover(
        f":material/auto_fix_high: {len(all_skills)} skills",
        help="Skills auto-load when your message matches their keywords. "
        "Type `/` in the chat input for inline autocomplete, or pick from "
        "the full list here.",
    ):
        st.caption(
            "Type `/` in the chat input for inline autocomplete, or pick a "
            "command below to see its description."
        )
        for skill in all_skills:
            scope = skill.get("scope", "workspace")
            badge = (
                ":blue-badge[workspace]" if scope == "workspace" else ":gray-badge[user]"
            )
            slug = skill.get("slug", "")
            desc = skill.get("description", "")
            st.markdown(f"`/{slug}` {badge} \u2014 {desc}")


def _render_workdir_controls() -> None:
    """Render the working-directory selector + project-context indicator.

    Renders *below* the chat input, immediately above
    :func:`_render_model_controls`, so all session controls are docked at
    the bottom of the page beneath the conversation history and chat
    input. The working directory is still a precondition for chatting —
    when it's invalid, the chat input itself is disabled and a warning
    appears between this block and the model controls.

    Layout (top to bottom):

    1. Working-directory dropdown (recents + free text) plus a folder-picker
       button that launches a native OS directory chooser.
    2. Project-context indicator showing detected guidance files / skills.
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

    _render_project_context_indicator(ss.working_dir)


def _render_model_controls() -> None:
    """Render the mode, model, and skills selectors at the very bottom.

    These are the per-turn knobs the user reaches for occasionally: the
    chat input is the primary affordance and stays directly below the
    conversation history, with all session settings (workdir + these)
    docked at the bottom of the page. Pairs with
    :func:`_render_workdir_controls`, which renders directly above this
    block.

    Layout (top to bottom):

    1. Mode + Model + Skills dropdowns side by side.
    2. Optional "model card" caption that labels the selected model with
       its name in bold, the context length and parameter count chipped
       alongside, and the W&B docs description after an em-dash. The
       label is what makes this block readable as info *about the chosen
       model* instead of a stray sentence floating at the bottom of the
       page.
    """
    ss = st.session_state

    # Both selectboxes bind to session_state via ``key=`` rather than the
    # ``ss.x = st.selectbox(..., index=...)`` pattern. Mixing an ``index=``
    # computed from session state with a separate write-back to that same
    # key is a documented Streamlit footgun: the user's selection can get
    # silently dismissed when the widget's anonymous identity shifts
    # between reruns (e.g. after a turn submit + ``st.rerun()``), causing
    # the dropdown to "snap back" to the prior value. ``key=`` makes
    # Streamlit the single owner of the value and is the canonical
    # remedy. ``_init_state`` seeds defaults, and ``_connect`` re-seeds
    # ``ss.model`` from inside its on_click callback (which runs before
    # the next rerun, so the widget picks the new value up cleanly).
    cols = st.columns([1, 2, 1], vertical_alignment="bottom")
    with cols[0]:
        mode_options = ["agent", "ask"]
        st.selectbox(
            "Mode",
            options=mode_options,
            key="mode",
            format_func=lambda m: "Agent" if m == "agent" else "Ask only",
            help=(
                "Agent can read, write, edit files (and run shell if enabled). "
                "Ask only is read-only — the model can list and read files but "
                "cannot modify the project."
            ),
        )
    with cols[1]:
        if ss.models:
            st.selectbox(
                "Model",
                options=ss.models,
                key="model",
                format_func=_model_label,
                help="Switch which W&B Inference model handles the next turn.",
            )
        else:
            st.selectbox(
                "Model",
                options=["Connect to load models"],
                disabled=True,
            )
    with cols[2]:
        _render_skills_popover(ss.working_dir)

    meta = MODEL_METADATA.get(ss.model) if ss.model else None
    if meta:
        chips: list[str] = []
        if meta.get("context"):
            chips.append(f"{meta['context']} context")
        if meta.get("params"):
            chips.append(f"{meta['params']} params")
        header = f":material/info: **{meta['label']}**"
        if chips:
            header += " \u00b7 " + " \u00b7 ".join(chips)
        desc = meta.get("description", "")
        st.caption(f"{header} \u2014 {desc}" if desc else header)


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


def _parse_kv_lines(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` or ``Header: value`` lines into a dict.

    Used by the Add MCP server dialog for env vars (stdio) and headers
    (HTTP). Each line is split on the *first* ``=`` or ``:``, whichever
    appears earlier, so bearer tokens with base64 ``=`` padding survive
    intact: ``Authorization: Bearer eyJ...xyz==`` partitions on the ``:``
    (position 13) before the ``=`` (position 30), yielding the correct
    ``("Authorization", "Bearer eyJ...xyz==")`` pair instead of the
    silently-corrupted ``("Authorization: Bearer eyJ...xyz", "=")`` you
    get from a naive ``=``-first partition. Env var values like
    ``PATH=/usr/bin:/bin`` still work because ``=`` comes first there.

    Empty lines and lines with neither separator are ignored so the user
    can write comments or blank lines without breaking validation.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        eq_pos = line.find("=")
        co_pos = line.find(":")
        if eq_pos == -1 and co_pos == -1:
            continue
        if eq_pos == -1:
            sep = ":"
        elif co_pos == -1:
            sep = "="
        else:
            sep = "=" if eq_pos < co_pos else ":"
        k, _, v = line.partition(sep)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def _format_kv_lines(items: dict[str, str], header_style: bool = False) -> str:
    """Inverse of :func:`_parse_kv_lines` for pre-filling the edit form.

    With ``header_style=True``, lines are written as ``Header: value`` so
    the rendered text matches what the user pasted from MCP server
    documentation (and the placeholder we show in the headers field).
    Otherwise we use ``KEY=value`` for env vars.
    """
    sep = ": " if header_style else "="
    return "\n".join(f"{k}{sep}{v}" for k, v in items.items())


@st.dialog("MCP server", width="large")
def _mcp_server_dialog() -> None:
    """Add or edit an MCP server config.

    Decided by ``st.session_state.mcp_dialog_editing``: ``None`` adds a new
    server, otherwise it's the id of the server being edited (we look it
    up in the registry to pre-fill the form). Save reconciles the live
    sessions; Delete prompts an explicit confirm.
    """
    registry = _get_mcp_registry()
    editing_id: str | None = st.session_state.mcp_dialog_editing

    existing: ServerConfig | None = None
    if editing_id is not None:
        existing = next((c for c in registry.configs if c.id == editing_id), None)

    title = "Edit MCP server" if existing else "Add MCP server"
    st.markdown(f"### {title}")
    st.caption(
        "Connect an external Model Context Protocol server. Stdio servers "
        "run as a local subprocess; HTTP servers are remote."
    )

    name = st.text_input(
        "Name",
        value=existing.name if existing else "",
        placeholder="My filesystem",
        help="Display label. We derive a sanitized id from this for the tool namespace.",
    )

    transport_options = ["stdio", "http"]
    default_transport = existing.transport if existing else "stdio"
    transport = st.segmented_control(
        "Transport",
        options=transport_options,
        default=default_transport,
        format_func=lambda t: "Stdio (local subprocess)" if t == "stdio" else "HTTP (remote)",
    ) or default_transport

    if transport == "stdio":
        command = st.text_input(
            "Command",
            value=existing.command if existing and existing.transport == "stdio" else "",
            placeholder="npx",
            help="Executable to run.",
        )
        args_default = "\n".join(existing.args) if existing and existing.transport == "stdio" else ""
        args_text = st.text_area(
            "Arguments (one per line)",
            value=args_default,
            placeholder="-y\n@modelcontextprotocol/server-filesystem\n/Users/me/projects",
            height=120,
        )
        env_default = (
            _format_kv_lines(existing.env)
            if existing and existing.transport == "stdio"
            else ""
        )
        env_text = st.text_area(
            "Environment variables (KEY=value, one per line)",
            value=env_default,
            placeholder="API_TOKEN=secret\nDEBUG=1",
            height=80,
        )
        url = ""
        headers_text = ""
    else:
        command = ""
        args_text = ""
        env_text = ""
        url = st.text_input(
            "URL",
            value=existing.url if existing and existing.transport == "http" else "",
            placeholder="https://example.com/mcp",
        )
        headers_default = (
            _format_kv_lines(existing.headers, header_style=True)
            if existing and existing.transport == "http"
            else ""
        )
        headers_text = st.text_area(
            "Headers (Header: value, one per line)",
            value=headers_default,
            placeholder="Authorization: Bearer ...",
            height=100,
            help="Auth headers stored on disk in plaintext (mode 0600).",
        )

    enabled = st.checkbox(
        "Enabled",
        value=existing.enabled if existing else True,
        help="Disabled servers stay configured but aren't connected.",
    )

    cols = st.columns([1, 1, 2])
    save_clicked = cols[0].button(
        "Save",
        icon=":material/save:",
        type="primary",
        width="stretch",
    )
    cancel_clicked = cols[1].button(
        "Cancel",
        icon=":material/close:",
        width="stretch",
    )
    delete_clicked = False
    if existing is not None:
        delete_clicked = cols[2].button(
            "Delete server",
            icon=":material/delete:",
            width="stretch",
        )

    if cancel_clicked:
        st.session_state.mcp_dialog_open = False
        st.session_state.mcp_dialog_editing = None
        st.rerun()

    if delete_clicked and existing is not None:
        try:
            registry.remove(existing.id)
        except Exception as e:
            st.error(f"Could not delete: {e}", icon=":material/error:")
            return
        st.session_state.mcp_dialog_open = False
        st.session_state.mcp_dialog_editing = None
        st.rerun()

    if save_clicked:
        if not name.strip():
            st.error("Name is required.", icon=":material/error:")
            return
        if transport == "stdio" and not command.strip():
            st.error("Command is required for stdio servers.", icon=":material/error:")
            return
        if transport == "http" and not url.strip():
            st.error("URL is required for HTTP servers.", icon=":material/error:")
            return

        server_id = existing.id if existing else make_server_id(name)
        config = ServerConfig(
            id=server_id,
            name=name.strip(),
            transport=transport,
            command=command.strip(),
            args=[a.strip() for a in args_text.splitlines() if a.strip()],
            env=_parse_kv_lines(env_text),
            url=url.strip(),
            headers=_parse_kv_lines(headers_text),
            enabled=enabled,
        )

        try:
            if existing is None:
                registry.add(config)
            else:
                registry.update(config)
        except Exception as e:
            st.error(f"Could not save: {e}", icon=":material/error:")
            return

        # Toast the save outcome so the user actually sees feedback. We can't
        # use ``st.warning``/``st.error`` here because ``st.rerun()`` below
        # aborts the script before they paint; toasts survive the rerun.
        status = registry.statuses.get(config.id)
        if not config.enabled:
            st.toast(f"Saved '{config.name}' (disabled)", icon=":material/save:")
        elif status and status.connected:
            n = len(status.tools)
            st.toast(
                f"Connected to '{config.name}' \u00b7 "
                f"{n} tool{'s' if n != 1 else ''}",
                icon=":material/check_circle:",
            )
        elif status and status.error:
            st.toast(
                f"'{config.name}' failed to connect \u2014 see sidebar.",
                icon=":material/error:",
            )
        else:
            st.toast(f"Saved '{config.name}'", icon=":material/save:")

        st.session_state.mcp_dialog_open = False
        st.session_state.mcp_dialog_editing = None
        st.rerun()


def _open_add_mcp_dialog() -> None:
    st.session_state.mcp_dialog_editing = None
    st.session_state.mcp_dialog_open = True


def _open_edit_mcp_dialog(server_id: str) -> None:
    st.session_state.mcp_dialog_editing = server_id
    st.session_state.mcp_dialog_open = True


def _toggle_mcp_enabled(server_id: str) -> None:
    """Checkbox callback: persist the new ``enabled`` flag and reconcile."""
    registry = _get_mcp_registry()
    cfg = next((c for c in registry.configs if c.id == server_id), None)
    if cfg is None:
        return
    new_enabled = bool(st.session_state.get(f"mcp_enabled_{server_id}", cfg.enabled))
    if new_enabled == cfg.enabled:
        return
    cfg.enabled = new_enabled
    registry.save()
    registry.reconcile()


def _render_mcp_panel() -> None:
    """Render the sidebar MCP servers expander.

    One row per configured server: name + transport badge + per-server
    enable toggle + edit button + tool count (or error). An "Add server"
    button at the bottom opens the modal. The expander is auto-expanded
    only when the user has at least one server already, so first-time
    users see the empty-state copy without an extra click.
    """
    registry = _get_mcp_registry()
    configs = list(registry.configs)
    label = (
        f"MCP servers \u00b7 {len(configs)}"
        if configs
        else "MCP servers"
    )

    with st.expander(label, icon=":material/extension:", expanded=bool(configs)):
        if not configs:
            st.caption(
                "Connect external Model Context Protocol servers to expose "
                "their tools to the agent. Stdio servers run as a local "
                "subprocess; HTTP servers are remote."
            )
        for cfg in configs:
            status = registry.statuses.get(cfg.id)
            row = st.container(border=True)
            with row:
                top = st.columns([5, 1, 1], vertical_alignment="center")
                with top[0]:
                    transport_badge = (
                        ":blue-badge[stdio]" if cfg.transport == "stdio" else ":violet-badge[http]"
                    )
                    st.markdown(f"**{cfg.name}** {transport_badge}")
                    if status is not None and status.connected:
                        n = len(status.tools)
                        st.caption(
                            f":green[Connected] \u00b7 {n} tool{'s' if n != 1 else ''}"
                        )
                    elif status is not None and status.error:
                        st.caption(f":red[Error] \u00b7 {status.error}")
                    elif not cfg.enabled:
                        st.caption("Disabled")
                    else:
                        st.caption("Not connected")
                with top[1]:
                    st.checkbox(
                        "Enabled",
                        value=cfg.enabled,
                        key=f"mcp_enabled_{cfg.id}",
                        on_change=_toggle_mcp_enabled,
                        args=(cfg.id,),
                        label_visibility="collapsed",
                        help="Enable or disable this server.",
                    )
                with top[2]:
                    st.button(
                        "",
                        icon=":material/edit:",
                        key=f"mcp_edit_{cfg.id}",
                        help="Edit this server.",
                        on_click=_open_edit_mcp_dialog,
                        args=(cfg.id,),
                        width="stretch",
                    )

        st.button(
            "Add server",
            icon=":material/add:",
            type="primary" if not configs else "secondary",
            on_click=_open_add_mcp_dialog,
            width="stretch",
        )

    if st.session_state.get("mcp_dialog_open"):
        _mcp_server_dialog()


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

        _render_mcp_panel()

        _render_file_changes()

        st.button("Clear chat", icon=":material/delete:", width="stretch", on_click=_clear_chat)


def main() -> None:
    _init_state()
    _render_sidebar()

    ss = st.session_state

    st.title("Code editing agent")
    st.caption(
        "Powered by [W&B Inference](https://docs.wandb.ai/inference). "
        "Point it at a working directory and pick a mode and model below the chat, "
        "and ask it to read or modify your code."
    )

    ready = ss.client is not None and ss.model is not None

    if not ready:
        st.info(
            "Enter your W&B API key and click **Connect** in the sidebar to get started.",
            icon=":material/login:",
        )
        return

    # Conversation area — forward-declared *before* the chat input so that
    # both the replayed history and the live-streaming current turn render
    # ABOVE the chat input + session controls, never below them. The
    # ``with conversation_area:`` block below ``_run_turn`` is what keeps
    # streaming tokens flowing into this container instead of getting
    # appended at the very bottom of the page (under workdir / mode /
    # model). All session controls (working dir + project context, mode,
    # model, skills) sit *below* the chat input so the chat input itself
    # is the divider between conversation and settings.
    conversation_area = st.container()
    with conversation_area:
        _render_history()

    # The chat input is wrapped in an explicit ``st.container()`` so
    # Streamlit renders it inline rather than pinning it to the viewport
    # bottom (its default behavior at the top level of an app). ``wd_ok``
    # is computed *before* the chat input so we can disable submissions
    # when the working directory is missing — that way the user gets a
    # clear "this is why pressing enter does nothing" signal instead of
    # silently dropped turns, and the validity warning that follows the
    # workdir picker explains how to fix it.
    wd_ok = Path(ss.working_dir).expanduser().is_dir()

    with st.container():
        prompt = st.chat_input(
            "Ask the agent to read or modify your code...",
            disabled=not wd_ok,
        )

    # Mount the slash-command autocomplete enhancer immediately after the
    # chat input. The component is invisible (zero-height host); its JS
    # finds the chat input's textarea on the page and attaches a floating
    # dropdown that filters as the user types after a "/". We only mount
    # when there's at least one skill so empty workspaces don't pay for
    # an unused component.
    summary = _scan_project_summary(ss.working_dir or "")
    autocomplete_skills = summary.get("all_skills", []) or []
    if autocomplete_skills:
        mount_slash_autocomplete(
            autocomplete_skills,
            placeholder_hint=(
                "Try a different prefix, or open the Skills popover below "
                "for the full list."
            ),
        )

    _render_workdir_controls()

    if not wd_ok:
        st.warning(
            "Choose a valid working directory above before chatting.",
            icon=":material/folder_off:",
        )

    _render_model_controls()

    if prompt and wd_ok:
        # Run the turn *inside* ``conversation_area`` so the user message
        # + the assistant ``st.chat_message`` that ``_run_turn`` opens for
        # live streaming both render in the conversation pane above the
        # chat input. Without the ``with`` block, both would be appended
        # to the bottom of the page in document order (i.e. below the
        # workdir / mode / model controls), which looks broken.
        with conversation_area:
            _run_turn(prompt)
        st.rerun()


main()
