"""Streamlit entry point for the W&B Inference code editing agent.

Runs as a multi-page app via ``st.navigation``: ``Agent`` (the existing
chat experience), ``Usage`` (the token-and-cost dashboard), and
``Settings`` (GitHub identity + W&B Inference connection + theme info).
The sidebar is shared chrome (MCP servers panel + file-changes panel)
rendered on every page; everything else is per-page.

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
import chats
import git_ops
from actions import (
    load_recent_dirs as _load_recent_dirs,
    on_connect as _auto_on_connect,
    theme_detected as _theme_detected,
)
from agent import DEEPSEEK_MODEL
from font_size_switcher import mount_font_size_switcher as _mount_font_size_switcher
from theme_switcher import mount_theme_switcher as _mount_theme_switcher

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
    # Multi-chat session state. ``chats`` is the dict of all persisted
    # conversations (loaded from ``~/.wb_coding_agent/chats/`` once per
    # session); ``active_chat_id`` tracks which one the chat page is
    # currently rendering. We seed a fresh empty chat if the user has
    # none yet so the chat page always has something to render against.
    # The active id is sourced from a tiny on-disk pointer
    # (``chats.load_active_chat_id``) so reloading the app lands on the
    # chat the user was last *looking at* rather than whichever chat
    # happens to have the freshest ``updated_at`` (which is often a
    # placeholder ``+ New chat`` that was never used). When the pointer
    # is absent or stale, ``chats.best_default_chat_id`` prefers a
    # chat with actual content over an empty placeholder.
    # ``delete_chat_confirm_id`` gates the @st.dialog confirm for
    # destructive deletes.
    if "chats" not in ss:
        ss.chats = chats.load_all_chats()
    if not ss.chats:
        seed = chats.new_chat(
            model=ss.model or "",
            mode=ss.mode,
            working_dir=ss.working_dir,
        )
        ss.chats[seed.id] = seed
    if "active_chat_id" not in ss or ss.active_chat_id not in ss.chats:
        ss.active_chat_id = (
            chats.best_default_chat_id(ss.chats) or next(iter(ss.chats))
        )
        # Persist the resolved id so subsequent reloads agree on it.
        chats.save_active_chat_id(ss.active_chat_id)
    ss.setdefault("delete_chat_confirm_id", None)
    # Gates the @st.dialog("Changes") modal mounted at the bottom of
    # ``app_pages/chat.py.render()``. Set to True by the "Changes"
    # button on the chat page (the only entry point); cleared by the
    # dialog's Close handler.
    ss.setdefault("diff_dialog_open", False)
    ss.setdefault("connect_error", None)
    ss.setdefault("weave_project", None)
    ss.setdefault("weave_url", None)
    ss.setdefault("weave_error", None)
    ss.setdefault("conn_open", False)
    # Tracks whether we've already attempted the one-shot auto-connect at
    # session startup. See ``_maybe_auto_connect`` for the contract — the
    # flag is set *before* the connect call so a transient failure can't
    # kick off a retry storm on every rerun.
    ss.setdefault("auto_connect_attempted", False)

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
    # Theme preference. ``theme_pref`` carries the user's explicit Light /
    # Dark / System choice (mirrors :data:`Profile.theme`); ``theme_explicit``
    # is True iff the user has actually picked. The theme switcher
    # component reads both: when ``theme_explicit`` is False it falls back
    # to whatever's already in browser ``localStorage`` so users who set
    # Dark via Streamlit's pre-app toolbar menu keep their preference.
    ss.setdefault("theme_pref", profile.theme or "")
    ss.setdefault("theme_explicit", bool(profile.theme))
    # Font-size preference. ``font_size_pref`` carries the user's
    # explicit choice (one of "Extra small" / "Small" / "Medium" /
    # "Large" / "Extra large") or ``""`` when the user has never
    # picked. The font size switcher component reads this value on
    # every mount; an empty string is treated as "no override" so the
    # page renders at whatever ``baseFontSize``
    # ``.streamlit/config.toml`` ships with.
    ss.setdefault("font_size_pref", profile.font_size or "")
    ss.setdefault("github_pat", creds.get("github_pat", ""))
    ss.setdefault("github_identity", _identity_from_profile(profile))
    ss.setdefault("github_pat_error", None)
    # Avatar bytes are populated only when the user verifies a GitHub PAT
    # (we cache the bytes downloaded from ``identity.avatar_url``); there is
    # no upload affordance and no on-disk avatar file by design.
    ss.setdefault("avatar_bytes", None)
    ss.setdefault("usage_session_total", {"total_tokens": 0, "cost_usd": 0.0, "turns": 0})
    ss.setdefault("git_identity_applied", set())

    # Git integration state. ``git_state_nonce`` is bumped after every
    # mutating git op (checkout, commit, push, conflict resolution) so the
    # cached repo scan refreshes on the next rerun. The bottom-of-chat
    # git row owns the new-branch / publish-branch modals (gated by
    # ``new_branch_dialog_open`` / ``first_push_dialog_open``) and queues
    # the push pipeline through ``pending_push_request`` so the actual
    # network + LLM work runs inside ``app_pages/chat.py.render()`` (where
    # ``st.toast`` / ``st.rerun`` work normally) rather than inside an
    # on-click callback. ``merge_conflict`` is set to
    # ``{"files": [...], "operation": "rebase"|"merge"}`` after a conflict
    # during the push flow's ``pull --rebase`` and drives the sidebar
    # warning + "Resolve with DeepSeek" button. ``pending_conflict_resolution``
    # is the cross-page handoff: the sidebar (in this module) sets it; the
    # chat page (in ``app_pages/chat.py``) drains it and runs an agent turn
    # with ``override_model=DEEPSEEK_MODEL``.
    ss.setdefault("git_state_nonce", 0)
    ss.setdefault("new_branch_dialog_open", False)
    ss.setdefault("first_push_dialog_open", False)
    ss.setdefault("pending_push_request", None)
    ss.setdefault("merge_conflict", None)
    ss.setdefault("pending_conflict_resolution", None)


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


def _maybe_auto_connect() -> None:
    """Auto-connect on the first script run when a saved API key is present.

    Runs exactly once per session: ``auto_connect_attempted`` is flipped to
    ``True`` *before* the connect call so a transient failure (e.g. an
    expired key, a network blip) does not re-trigger on every rerun. The
    user can still manually click **Connect** on the Settings page if the
    auto-attempt failed — the error surfaces there via ``ss.connect_error``.
    """
    ss = st.session_state
    if ss.auto_connect_attempted:
        return
    ss.auto_connect_attempted = True
    if ss.client is not None:
        return
    if not (ss.api_key or "").strip():
        return
    _auto_on_connect()
    if ss.client is not None and ss.connect_error is None:
        n = len(ss.models)
        st.toast(
            f"Connected — {n} model{'s' if n != 1 else ''} available.",
            icon=":material/check_circle:",
        )




# ---------------------------------------------------------------------------
# Git scan: cached, nonce-busted on every mutating op
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3, show_spinner=False)
def _cached_git_scan(working_dir: str, _nonce: int) -> dict[str, Any]:
    """Cached :func:`git_ops.scan`.

    ``_nonce`` is part of the cache key (so callers can force a refresh
    by bumping ``ss.git_state_nonce``) but the value isn't used inside
    the function. The 3-second TTL keeps the panel cheap on rapid
    rerenders without needing to bust the cache for every Streamlit
    interaction.

    Returns the safe-defaults dict from :func:`git_ops.scan` whenever
    git is missing, the directory does not exist, or the directory is
    not a git working tree, so callers don't have to special-case those.
    """
    if not working_dir:
        return git_ops.scan(Path.home())
    p = Path(working_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return {
            "installed": git_ops.is_git_installed(),
            "in_repo": False,
            "current_branch": None,
            "branches": [],
            "remote_branches": [],
            "default_branch": "main",
            "status": [],
            "dirty": False,
            "in_merge_or_rebase": False,
            "operation": None,
            "conflicted_files": [],
            "remote_url": None,
        }
    return git_ops.scan(p)


def _git_state() -> dict[str, Any]:
    """Return the current git-state dict for the active working directory.

    Thin wrapper that pulls the nonce + workdir out of session state and
    forwards them to the cached scanner. Used by every git-aware UI
    helper in this module so they share one cache entry per render pass.
    """
    return _cached_git_scan(
        st.session_state.get("working_dir") or "",
        int(st.session_state.get("git_state_nonce") or 0),
    )


def _bump_git_nonce() -> None:
    """Force the next :func:`_git_state` call to re-scan."""
    st.session_state.git_state_nonce = (
        int(st.session_state.get("git_state_nonce") or 0) + 1
    )


# Per-file diff rendering and the push flow both live on the chat page
# (``app_pages/chat.py``): the unified-diff modal is opened by the
# "Changes" button just above the chat input, and the new bottom-of-chat
# git row owns branch switching, new-branch creation, fetch, and the
# one-click "generate commit message + commit + push" pipeline. This
# module is responsible only for the cross-page sidebar warning when
# the working tree is mid-merge/rebase (so users see it from any page);
# everything else moved into the chat page.


# ---------------------------------------------------------------------------
# Git warnings (sidebar)
# ---------------------------------------------------------------------------
def _request_conflict_resolution() -> None:
    """Sidebar callback: hand a synthesized prompt off to the chat page.

    The actual agent turn runs from ``app_pages/chat.py`` (it owns
    ``_run_turn`` + the streaming UI). We can't just call ``_run_turn``
    here without recreating the chat scaffolding, so we drop a payload
    in ``ss.pending_conflict_resolution`` and let the chat page pick it
    up on its next render.
    """
    conflict = st.session_state.get("merge_conflict") or {}
    files = conflict.get("files") or []
    operation = conflict.get("operation") or "merge"
    if not files:
        return
    listed = "\n".join(f"- `{p}`" for p in files)
    continue_cmd = "git rebase --continue" if operation == "rebase" else "git merge --continue"
    prompt = (
        f"There is a merge conflict during a `{operation}`. "
        f"The conflicted files are:\n\n{listed}\n\n"
        "For each file: read it, replace the conflict markers "
        "(`<<<<<<<` / `=======` / `>>>>>>>`) with a coherent merge that "
        "keeps the intent of both sides, then run "
        f"`git add <file>`. Once every conflicted file is staged, run "
        f"`{continue_cmd}` to finish the operation. Do not run any other "
        "git commands. Use the read_file / edit_file / run_shell tools."
    )
    st.session_state.pending_conflict_resolution = {
        "prompt": prompt,
        "model": DEEPSEEK_MODEL,
    }


def _render_git_warnings() -> None:
    """Sidebar warnings: git-not-installed and merge-conflict states.

    Rendered before the file-changes panel so a user mid-rebase sees the
    "Resolve with DeepSeek" affordance immediately, without scrolling
    through the diff list to find it.
    """
    if not git_ops.is_git_installed():
        st.error(
            "Git is not installed on PATH. Install git to enable branch "
            "switching, diffs, and the push flow.",
            icon=":material/error:",
        )
        return

    state = _git_state()
    if not state.get("in_repo"):
        return

    # Refresh stored conflict info from the live repo state — this keeps
    # the warning honest if the user (or the agent) finished the rebase
    # outside the dialog flow.
    conflict = st.session_state.get("merge_conflict")
    if state.get("in_merge_or_rebase") and state.get("conflicted_files"):
        st.session_state.merge_conflict = {
            "files": list(state["conflicted_files"]),
            "operation": state.get("operation") or "merge",
        }
        conflict = st.session_state.merge_conflict
    elif not state.get("in_merge_or_rebase") and conflict is not None:
        st.session_state.merge_conflict = None
        conflict = None

    if not conflict:
        return

    files = conflict.get("files") or []
    op = conflict.get("operation") or "merge"
    with st.container(border=True):
        st.markdown(
            f":material/warning: **Merge conflict during `{op}`**"
        )
        st.caption(f"{len(files)} conflicted file{'s' if len(files) != 1 else ''}:")
        for f in files[:8]:
            st.markdown(f"- `{f}`")
        if len(files) > 8:
            st.caption(f"... and {len(files) - 8} more")
        st.button(
            "Resolve with DeepSeek",
            icon=":material/auto_fix_high:",
            type="primary",
            width="stretch",
            key="resolve_conflict_btn",
            on_click=_request_conflict_resolution,
            help=(
                f"Run an agent turn with {DEEPSEEK_MODEL.split('/')[-1]} "
                "that reads each conflicted file, edits the markers into a "
                f"coherent merge, then runs `git {op} --continue`."
            ),
        )


# ---------------------------------------------------------------------------
# Multi-chat sidebar
# ---------------------------------------------------------------------------
# The sidebar is the "chat history" panel: every persisted conversation
# is rendered as a row with [activate, archive, delete] icon controls.
# Branch switching, new-branch creation, fetch, and the push pipeline
# all moved into the bottom-of-chat git row in ``app_pages/chat.py``;
# the file-by-file diff list lives on the chat page in the Changes
# modal. A collapsed "Archive" expander hosts archived chats. Every
# callback below mutates ``ss.chats`` / ``ss.active_chat_id`` and
# delegates persistence to :mod:`chats`.

# Material icon for each chat status. Lookup is forgiving: an unknown
# status falls back to the "new" icon so a future status (e.g.
# "queued") doesn't render as nothing while we add UI for it.
_STATUS_ICON: dict[str, str] = {
    chats.STATUS_NEW: ":material/chat_bubble_outline:",
    chats.STATUS_RUNNING: ":material/progress_activity:",
    chats.STATUS_IDLE: ":material/check_circle:",
    chats.STATUS_ERROR: ":material/cancel:",
}


def _sorted_chats(*, archived: bool) -> list[chats.Chat]:
    """Return chats matching ``archived`` flag, most-recent-first."""
    return sorted(
        (c for c in st.session_state.chats.values() if bool(c.archived) == archived),
        key=lambda c: c.updated_at,
        reverse=True,
    )


def _activate_chat(chat_id: str) -> None:
    """Sidebar row callback: mark this chat as the active one."""
    if chat_id in st.session_state.chats:
        st.session_state.active_chat_id = chat_id
        chats.save_active_chat_id(chat_id)
        # Forget the dropdown-sync sentinel so the next chat-page render
        # re-copies the chat's model / mode / working_dir into the flat
        # ss.* keys driving the dropdowns. See app_pages/chat.py.
        st.session_state.pop("_last_active_chat_id", None)


def _new_chat() -> None:
    """`+ New chat` button callback.

    If a blank ``+ New chat`` placeholder already exists (default title,
    no user / assistant turns, not archived), activates that chat instead
    of minting another empty row — clicking the button repeatedly should
    never accumulate a pile of identical placeholders. The reused chat
    has its model / mode / working_dir refreshed to the user's current
    dropdown picks so it behaves like a freshly seeded chat.

    Otherwise creates a new chat seeded with the current dropdown values.
    """
    ss = st.session_state
    blank_id = chats.find_blank_chat(ss.chats)
    if blank_id is not None:
        existing = ss.chats[blank_id]
        with existing._lock:
            existing.model = ss.model or ""
            existing.mode = ss.mode or "agent"
            existing.working_dir = ss.working_dir or ""
        chats.save_chat(existing)
        ss.active_chat_id = blank_id
        chats.save_active_chat_id(blank_id)
        ss.pop("_last_active_chat_id", None)
        return
    chat = chats.new_chat(
        model=ss.model or "",
        mode=ss.mode or "agent",
        working_dir=ss.working_dir or "",
    )
    ss.chats[chat.id] = chat
    ss.active_chat_id = chat.id
    chats.save_active_chat_id(chat.id)
    ss.pop("_last_active_chat_id", None)


def _seed_default_chat() -> str:
    """Mint a fresh chat (used when delete drains the sidebar empty).

    Returns the new chat's id so the caller can immediately activate it.
    """
    ss = st.session_state
    chat = chats.new_chat(
        model=ss.model or "",
        mode=ss.mode or "agent",
        working_dir=ss.working_dir or "",
    )
    ss.chats[chat.id] = chat
    return chat.id


def _archive_chat(chat_id: str) -> None:
    """Archive icon callback: flip ``archived=True``, persist, re-pick active."""
    ss = st.session_state
    chat = ss.chats.get(chat_id)
    if chat is None:
        return
    chats.archive_chat(chat)
    if ss.active_chat_id == chat_id:
        # Pick the most recent non-archived survivor; if there are none
        # left, mint a fresh chat so the chat page always has something
        # to render against.
        successors = _sorted_chats(archived=False)
        if successors:
            ss.active_chat_id = successors[0].id
        else:
            ss.active_chat_id = _seed_default_chat()
        chats.save_active_chat_id(ss.active_chat_id)
        ss.pop("_last_active_chat_id", None)


def _unarchive_chat(chat_id: str) -> None:
    """Unarchive icon callback: flip ``archived=False`` and surface in the live list."""
    ss = st.session_state
    chat = ss.chats.get(chat_id)
    if chat is None:
        return
    chats.unarchive_chat(chat)


def _open_delete_confirm(chat_id: str) -> None:
    """Delete icon callback: stash the target id; the dialog renders next pass."""
    st.session_state.delete_chat_confirm_id = chat_id


def _close_delete_chat_dialog() -> None:
    """Drop the gating id so subsequent reruns don't re-mount the modal.

    Wired to ``@st.dialog(..., on_dismiss=_close_delete_chat_dialog)``
    so X / Esc / click-outside dismissals clear the gating id —
    without this, the dialog re-opens on the very next rerun (e.g.
    the next chat submission on the chat page) because
    ``delete_chat_confirm_id`` stays set. The Cancel / Delete /
    "OK on running" handlers inside the dialog body clear the same
    id and call ``st.rerun()`` directly.
    """
    st.session_state.delete_chat_confirm_id = None


@st.dialog("Delete chat?", width="small", on_dismiss=_close_delete_chat_dialog)
def _delete_chat_dialog() -> None:
    """Confirm modal for the per-chat delete icon.

    Refuses to delete a running chat (the background thread is still
    holding the chat's lock + its on-disk file is its only persistence
    handle). Otherwise removes the chat from the session dict + its
    JSON file, picks a successor active chat, and reruns.

    ``on_dismiss=_close_delete_chat_dialog`` is mandatory so X / Esc /
    click-outside dismissal clears ``ss.delete_chat_confirm_id``;
    otherwise the modal re-opens on the next rerun (e.g. when the
    user navigates to the chat page and sends a message). See
    :func:`app_pages.chat._diff_dialog` for the full rationale.
    """
    ss = st.session_state
    chat_id = ss.delete_chat_confirm_id
    chat = ss.chats.get(chat_id) if chat_id else None
    if chat is None:
        ss.delete_chat_confirm_id = None
        return

    if chat.status == chats.STATUS_RUNNING:
        st.warning(
            "This chat is still running a turn. Wait for it to finish "
            "before deleting.",
            icon=":material/hourglass:",
        )
        if st.button("OK", width="stretch", key="delete_chat_running_ok"):
            ss.delete_chat_confirm_id = None
            st.rerun()
        return

    title = chat.title or "(untitled)"
    st.markdown(f"Delete **{title}**?")
    st.caption("This permanently removes the chat from disk. It cannot be undone.")
    cols = st.columns([1, 1])
    if cols[0].button(
        "Cancel",
        width="stretch",
        key="delete_chat_cancel_btn",
        icon=":material/close:",
    ):
        ss.delete_chat_confirm_id = None
        st.rerun()
    if cols[1].button(
        "Delete",
        icon=":material/delete:",
        type="primary",
        width="stretch",
        key="delete_chat_confirm_btn",
    ):
        was_active = ss.active_chat_id == chat_id
        try:
            chats.delete_chat(ss.chats, chat_id)
        except RuntimeError as e:
            st.error(str(e), icon=":material/error:")
            return
        if was_active:
            successors = _sorted_chats(archived=False)
            if successors:
                ss.active_chat_id = successors[0].id
            else:
                ss.active_chat_id = _seed_default_chat()
            chats.save_active_chat_id(ss.active_chat_id)
            ss.pop("_last_active_chat_id", None)
        ss.delete_chat_confirm_id = None
        st.rerun()


def _truncate(text: str, limit: int) -> str:
    """Tail-truncate ``text`` to ``limit`` characters with a unicode ellipsis."""
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "\u2026"


def _render_active_chat_body(chat: chats.Chat) -> None:
    """The body block rendered directly under the active chat's row.

    Contents (in order): the AI-generated description, and a status
    caption when the chat is running or last-errored. The archive +
    delete icons used to render here as a separate row but moved
    inline into the chat row itself (see :func:`_render_chat_row`),
    so this body now stays focused on read-only context. Branch
    switching, fetch, new branch, and push all live in the
    bottom-of-chat git row in ``app_pages/chat.py`` — the sidebar just
    has to track which chat is active so the chat page points at the
    right working directory.

    Only ever called for the **active** chat; inactive live rows render
    just the header row so a long chat list collapses cleanly.
    """
    description = (chat.description or "").strip()
    if description:
        st.caption(description)

    if chat.status == chats.STATUS_ERROR and chat.error_message:
        st.caption(f":red[:material/error: {chat.error_message}]")
    elif chat.status == chats.STATUS_RUNNING:
        st.caption(":material/progress_activity: Running a turn...")


def _render_chat_row(chat: chats.Chat, *, archived: bool) -> None:
    """One sidebar row.

    Both live and archived rows share the same 3-column layout:
    ``[main control | archive/unarchive icon | delete icon]``. Putting
    the destructive icons inline with the main control means they're
    reachable from any chat without first activating it, and keeps the
    active chat's body focused on read-only context (description +
    status caption) instead of repeating action buttons.

    For **live** rows the main control is a stretched ``st.button``;
    clicking it activates the chat (which loads its history into the
    chat page) and, since the body only renders for the active chat,
    simultaneously expands the row. The active row's main button is
    rendered as ``type="primary"`` so the user can see at a glance which
    chat the chat page is showing.

    For **archived** rows the main control is plain markdown (no
    activate affordance) — archived chats are intentionally minimized;
    the user can unarchive to inspect detail.
    """
    ss = st.session_state
    icon = _STATUS_ICON.get(chat.status, _STATUS_ICON[chats.STATUS_NEW])
    # Cap shorter than the historic 40 chars because the title now sits
    # in a ~75%-wide column (the icons take the other 25%) and a longer
    # title wraps onto a second line inside the narrower button box.
    title = _truncate(chat.title or "(untitled)", 30)

    cols = st.columns([6, 1, 1], vertical_alignment="center")

    if archived:
        with cols[0]:
            st.markdown(f"{icon} {title}")
        with cols[1]:
            st.button(
                "",
                icon=":material/unarchive:",
                key=f"chat_unarchive_{chat.id}",
                help="Unarchive",
                on_click=_unarchive_chat,
                args=(chat.id,),
            )
        with cols[2]:
            st.button(
                "",
                icon=":material/delete:",
                key=f"chat_delete_{chat.id}",
                help="Delete",
                on_click=_open_delete_confirm,
                args=(chat.id,),
            )
        return

    is_active = ss.active_chat_id == chat.id
    with cols[0]:
        st.button(
            title,
            icon=icon,
            key=f"chat_row_{chat.id}",
            type="primary" if is_active else "tertiary",
            width="stretch",
            on_click=_activate_chat,
            args=(chat.id,),
            help=(
                f"Status: {chat.status}"
                + (f" · {chat.error_message}" if chat.error_message else "")
                + ("" if is_active else " — click to load this chat's history.")
            ),
        )
    with cols[1]:
        st.button(
            "",
            icon=":material/archive:",
            key=f"chat_archive_{chat.id}",
            help="Archive",
            on_click=_archive_chat,
            args=(chat.id,),
        )
    with cols[2]:
        st.button(
            "",
            icon=":material/delete:",
            key=f"chat_delete_{chat.id}",
            help="Delete",
            on_click=_open_delete_confirm,
            args=(chat.id,),
        )
    if is_active:
        _render_active_chat_body(chat)


# CSS scoped to the chat-row buttons in the sidebar. Streamlit centers
# button labels by default (and there's no built-in alignment param —
# tracked at https://github.com/streamlit/streamlit/issues/10770), so we
# left-align via the documented ``st-key-<key>`` class Streamlit attaches
# to keyed widgets. Scoped to ``[data-testid="stSidebar"]`` + the
# ``chat_row_`` key prefix so the "New chat" action button + the
# icon-only archive / delete / unarchive buttons keep their default
# centering — only the chat history rows themselves shift left, which
# is what reads naturally for a list of titles.
_CHAT_LIST_BUTTON_CSS = (
    "<style>"
    '[data-testid="stSidebar"] [class*="st-key-chat_row_"] button{'
    "justify-content:flex-start;text-align:left;"
    "}"
    "</style>"
)


def _render_chat_list_panel() -> None:
    """The whole chat-history block: New button + live rows + Archive expander."""
    ss = st.session_state
    live = _sorted_chats(archived=False)
    archived = _sorted_chats(archived=True)

    st.html(_CHAT_LIST_BUTTON_CSS)

    st.button(
        "New chat",
        icon=":material/add:",
        key="new_chat_btn",
        type="primary",
        width="stretch",
        on_click=_new_chat,
        help="Start a fresh conversation. The current chat is preserved.",
    )

    for chat in live:
        _render_chat_row(chat, archived=False)

    if archived:
        with st.expander(
            f"Archive \u00b7 {len(archived)}",
            icon=":material/inventory_2:",
            expanded=False,
        ):
            st.caption(
                "Archived chats stay on disk but are out of the way. "
                "Click the unarchive icon to bring one back."
            )
            for chat in archived:
                _render_chat_row(chat, archived=True)


# ---------------------------------------------------------------------------
# Sidebar (shared chrome on every page)
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """Render the per-page sidebar chrome.

    The sidebar is the chat-history panel plus the merge-conflict
    warning (which is small and useful on every page). Settings,
    workdir / model / mode controls, and per-file diffs all live on
    their own pages — the sidebar exists to let users navigate between
    chats without leaving whatever page they're on.
    """
    with st.sidebar:
        st.markdown("### :material/smart_toy: W&B Coding Agent")
        st.caption("A code editing agent powered by W&B Inference.")

        _render_chat_list_panel()
        _render_git_warnings()

    if st.session_state.get("delete_chat_confirm_id"):
        _delete_chat_dialog()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    _maybe_auto_connect()

    # Tighten the default ~6rem top padding Streamlit reserves above the
    # main block container so the page title sits directly below the top
    # navigation bar instead of being separated by a wide empty band.
    # Applied globally (via ``main()``, not from any single page) so the
    # rule is in effect on Chat / Diff / Usage / Settings — each page's
    # title was visually orphaned from the nav at the default padding.
    # Targets the documented ``[data-testid="stMainBlockContainer"]``
    # selector so a Streamlit upgrade that changes internal class names
    # doesn't silently regress the rule. ``app_pages/chat.py`` adds its
    # own chat-page-specific flex/height rules on top of this; those use
    # ``!important`` and don't conflict with this padding-only override.
    st.html(
        "<style>"
        '[data-testid="stMainBlockContainer"]{padding-top:1.5rem;}'
        "</style>"
    )

    # Mount the in-app theme switcher before any page content renders. The
    # component is zero-height (``display: none``) so it doesn't affect
    # layout; its inline JS reads/writes the same browser ``localStorage``
    # key Streamlit's frontend reads on app boot, so picking a theme on the
    # Settings page applies the new theme via a single page reload (the
    # only mechanism Streamlit gives us to swap themes at runtime; there
    # is no programmatic API). When the user has not yet picked, the
    # component reports any pre-existing localStorage value back via
    # ``_theme_detected`` so legacy choices made via Streamlit's now-hidden
    # toolbar menu carry over without surprise.
    pref = st.session_state.theme_pref if st.session_state.theme_explicit else ""
    _mount_theme_switcher(pref, on_detected=_theme_detected)

    # Mount the in-app font-size switcher right after the theme one. It
    # injects a ``<style>`` tag into ``document.head`` overriding the
    # root ``html`` font size whenever the user has picked a non-empty
    # preference; an empty preference clears any prior override so the
    # bundled ``baseFontSize`` config value applies. Mounted from the
    # entry script (rather than from the Settings page) so the override
    # is in effect on every page in the multi-page app.
    _mount_font_size_switcher(st.session_state.font_size_pref or "")

    _render_sidebar()

    chat_page = st.Page(
        "app_pages/chat.py",
        title="Agent",
        icon=":material/auto_awesome:",
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
    docs_page = st.Page(
        "app_pages/docs.py",
        title="Docs",
        icon=":material/menu_book:",
    )
    page = st.navigation(
        [chat_page, usage_page, settings_page, docs_page],
        position="top",
    )
    page.run()


main()
