"""Streamlit entry point for the W&B Inference code editing agent.

Runs as a multi-page app via ``st.navigation``: ``Chat`` (the existing
chat experience), ``Usage`` (the token-and-cost dashboard), and
``Settings`` (GitHub identity + W&B Inference connection + theme info).
The sidebar is shared chrome (MCP servers panel + file-changes panel +
clear-chat button) rendered on every page; everything else is per-page.

This module owns:

- ``_init_state`` — the session-state contract for the whole app. Pages
  read shared state (``client``, ``model``, ``working_dir``, etc.) but
  don't initialize it.
- The MCP server panel + add/edit dialog.
- The "File changes" sidebar panel.

Cross-cutting callbacks (Connect / Disconnect, GitHub PAT verify,
recent-dirs, folder picker) live in :mod:`actions` so page modules can
import them without re-triggering this entry script's import — Streamlit
loads the entry as ``__main__``, and a ``from streamlit_app import ...``
from a sub-page would re-run ``main()`` and re-render the sidebar,
producing duplicate-widget-key errors.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

import account
from actions import load_recent_dirs as _load_recent_dirs

st.set_page_config(
    page_title="W&B Coding Agent",
    page_icon=":material/smart_toy:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    """Seed session state from disk + defaults.

    Loads the persisted profile (GitHub identity, avatar path, theme pref)
    via ``account.load_profile`` and any opt-in saved credentials via
    ``account.load_credentials``. Persisted state is purposefully loaded
    once per session to avoid re-reading the same JSON on every rerun.
    """
    ss = st.session_state

    profile = account.load_profile()
    creds = account.load_credentials()

    # Pre-fill the API key from disk if the user opted in to "remember on
    # this machine"; the box stays editable.
    ss.setdefault("api_key", creds.get("wb_api_key", ""))
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
    ss.setdefault("conn_open", False)

    # MCP dialog state.
    ss.setdefault("mcp_dialog_open", False)
    ss.setdefault("mcp_dialog_editing", None)

    # "Start a new project" dialog state. ``new_project_dialog_open`` gates
    # the modal mounted by ``app_pages/chat.py``; ``new_proj_parent`` is the
    # sticky parent directory the form pre-fills with (defaults to ``~``).
    ss.setdefault("new_project_dialog_open", False)
    ss.setdefault("new_proj_parent", str(Path.home()))

    # Settings state.
    ss.setdefault("remember_wb_key", bool(creds.get("wb_api_key")))
    ss.setdefault("github_pat", creds.get("github_pat", ""))
    ss.setdefault("github_identity", _identity_from_profile(profile))
    ss.setdefault("github_pat_error", None)
    # Avatar bytes are populated only when the user verifies a GitHub PAT
    # (we cache the bytes downloaded from ``identity.avatar_url``); there is
    # no upload affordance and no on-disk avatar file by design.
    ss.setdefault("avatar_bytes", None)
    ss.setdefault("usage_session_total", {"total_tokens": 0, "cost_usd": 0.0, "turns": 0})
    ss.setdefault("git_identity_applied", set())


def _identity_from_profile(profile: account.Profile) -> dict[str, Any] | None:
    """Translate the persisted profile dataclass into the in-session identity dict.

    Returns ``None`` when the profile has no GitHub login (i.e. the user has
    not yet verified a PAT). The shape matches what
    ``account.verify_github_pat`` returns so the rest of the UI can read
    either source uniformly.
    """
    if not profile.github_username:
        return None
    return {
        "login": profile.github_username,
        "name": profile.github_username,
        "email": profile.github_email,
        "avatar_url": profile.github_avatar_url,
        "scopes": list(profile.github_scopes),
    }


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.ui_turns = []
    st.session_state.usage_session_total = {"total_tokens": 0, "cost_usd": 0.0, "turns": 0}




# ---------------------------------------------------------------------------
# File changes panel (unchanged from prior version)
# ---------------------------------------------------------------------------
def _count_diff_lines(diff: str) -> tuple[int, int]:
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


# ---------------------------------------------------------------------------
# Sidebar (shared chrome on every page)
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """Render the per-page sidebar chrome.

    Both Settings (GitHub, theme, W&B Inference, MCP) and the chat-time
    controls (workdir, model, mode) live on their own pages. The sidebar
    keeps app branding, the file-changes summary, and the clear-chat
    button — small, persistent affordances that are useful from any page.
    """
    with st.sidebar:
        st.markdown("### :material/smart_toy: W&B Coding Agent")
        st.caption("A code editing agent powered by W&B Inference.")

        _render_file_changes()
        st.button("Clear chat", icon=":material/delete:", width="stretch", on_click=_clear_chat)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    _render_sidebar()

    chat_page = st.Page(
        "app_pages/chat.py",
        title="Chat",
        icon=":material/chat:",
        default=True,
    )
    usage_page = st.Page(
        "app_pages/usage.py",
        title="Usage",
        icon=":material/insights:",
    )
    settings_page = st.Page(
        "app_pages/settings.py",
        title="Settings",
        icon=":material/settings:",
    )
    page = st.navigation(
        [chat_page, usage_page, settings_page],
        position="top",
    )
    page.run()


main()
