"""Streamlit UI for the W&B Inference code editing agent."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from agent import run_agent_turn
from wb_client import list_models, make_client

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
    ss.setdefault("working_dir", os.getcwd())
    ss.setdefault("max_iters", 12)
    ss.setdefault("shell_enabled", False)
    ss.setdefault("messages", [])
    ss.setdefault("ui_turns", [])
    ss.setdefault("connect_error", None)


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

    with st.chat_message("assistant"):
        live_container = st.container()
        results_by_id: dict[str, dict[str, Any]] = {}
        call_placeholders: dict[str, Any] = {}
        call_args: dict[str, dict[str, Any]] = {}

        with st.spinner(f"Thinking with `{ss.model}`...", show_time=True):
            try:
                events_iter = run_agent_turn(
                    client=ss.client,
                    model=ss.model,
                    messages=ss.messages,
                    working_dir=working_dir,
                    shell_enabled=ss.shell_enabled,
                    max_iters=ss.max_iters,
                )
                for event in events_iter:
                    assistant_turn["events"].append(event)
                    etype = event["type"]
                    if etype == "assistant_text":
                        content = event.get("content") or ""
                        if content.strip():
                            with live_container:
                                st.markdown(content)
                    elif etype == "tool_call":
                        call_args[event["id"]] = event
                        with live_container:
                            ph = st.empty()
                        call_placeholders[event["id"]] = ph
                        with ph.container():
                            _render_tool_event(event, None)
                    elif etype == "tool_result":
                        results_by_id[event["id"]] = event
                        ph = call_placeholders.get(event["id"])
                        call_ev = call_args.get(event["id"])
                        if ph is not None and call_ev is not None:
                            with ph.container():
                                _render_tool_event(call_ev, event)
                    elif etype == "error":
                        with live_container:
                            st.error(event["message"], icon=":material/error:")
            except Exception as e:
                with live_container:
                    st.error(f"Agent crashed: {e}", icon=":material/error:")
                assistant_turn["events"].append({"type": "error", "message": str(e)})


def _render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### :material/smart_toy: W&B Coding Agent")
        st.caption("A code editing agent powered by W&B Inference.")

        with st.container(border=True):
            st.markdown("**Connection**")
            ss.api_key = st.text_input(
                "W&B API key",
                value=ss.api_key,
                type="password",
                help="Get one at wandb.ai/settings. Held in session memory only.",
            )
            ss.project = st.text_input(
                "Project (optional)",
                value=ss.project,
                placeholder="team/project",
                help="Optional. Used for usage attribution.",
            )
            connect_clicked = st.button(
                "Connect",
                icon=":material/link:",
                width="stretch",
                type="primary",
            )
            if connect_clicked:
                _connect(ss.api_key, ss.project)

            if ss.connect_error:
                st.error(ss.connect_error, icon=":material/error:")
            elif ss.client is not None:
                st.success(
                    f"Connected. {len(ss.models)} models available.",
                    icon=":material/check_circle:",
                )

        with st.container(border=True):
            st.markdown("**Model**")
            if ss.models:
                ss.model = st.selectbox(
                    "Model",
                    options=ss.models,
                    index=ss.models.index(ss.model) if ss.model in ss.models else 0,
                    label_visibility="collapsed",
                )
            else:
                st.caption("Connect to load models.")

        with st.container(border=True):
            st.markdown("**Workspace**")
            ss.working_dir = st.text_input(
                "Working directory",
                value=ss.working_dir,
                help="The agent operates only on files inside this directory.",
            )
            wd = Path(ss.working_dir).expanduser()
            if not wd.exists():
                st.error("Directory does not exist.", icon=":material/error:")
            elif not wd.is_dir():
                st.error("Path is not a directory.", icon=":material/error:")
            else:
                st.caption(f":material/folder: `{wd.resolve()}`")

        with st.container(border=True):
            st.markdown("**Settings**")
            ss.max_iters = st.slider(
                "Max tool-call iterations per turn",
                min_value=1,
                max_value=30,
                value=ss.max_iters,
            )
            ss.shell_enabled = st.toggle(
                "Allow shell commands",
                value=ss.shell_enabled,
                help="When enabled, the agent can run shell commands inside the working directory.",
            )
            if ss.shell_enabled:
                st.warning(
                    "Shell access is enabled. The agent can execute arbitrary commands.",
                    icon=":material/warning:",
                )

        st.button("Clear chat", icon=":material/delete:", width="stretch", on_click=_clear_chat)


def main() -> None:
    _init_state()
    _render_sidebar()

    ss = st.session_state

    st.title("Code editing agent")
    st.caption(
        "Powered by [W&B Inference](https://docs.wandb.ai/inference). "
        "Pick a model in the sidebar, point it at a working directory, and ask it to read or modify your code."
    )

    ready = ss.client is not None and ss.model is not None
    wd_ok = Path(ss.working_dir).expanduser().is_dir()

    if not ready:
        st.info(
            "Enter your W&B API key and click **Connect** in the sidebar to get started.",
            icon=":material/login:",
        )
        return
    if not wd_ok:
        st.warning(
            "Set a valid working directory in the sidebar before chatting.",
            icon=":material/folder_off:",
        )
        return

    _render_history()

    if not ss.ui_turns:
        st.caption(
            f"Active model: `{ss.model}` · Working directory: `{Path(ss.working_dir).expanduser().resolve()}`"
        )

    prompt = st.chat_input("Ask the agent to read or modify your code...")
    if prompt:
        _run_turn(prompt)
        st.rerun()


main()
