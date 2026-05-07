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
import providers
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
    saved_provider_keys = account.load_provider_keys()
    saved_provider_remember = account.load_provider_remember()

    # ------------------------------------------------------------------
    # Multi-provider session state (Phase 1).
    #
    # ``provider_keys[provider_id] -> str``: API keys for every persisted
    #   provider, pre-filled from disk for users who opted in to
    #   "Remember on this machine" per provider. Empty string means the
    #   user hasn't entered a key for that provider yet (or has cleared
    #   it). The dual-key pattern in the Settings card mirrors each
    #   widget back into this dict via the `_<id>_key_input` widget key.
    # ``provider_remember[provider_id] -> bool``: per-provider opt-in
    #   for persistence across sessions. Default False; flipped to True
    #   when the user ticks "Remember on this machine" before clicking
    #   Connect.
    # ``clients[provider_id] -> Any | None``: the persistent client
    #   object. ``openai.OpenAI`` for ``openai_native`` AND for the 9
    #   ``openai_compat`` providers (same SDK, per-provider
    #   ``base_url``); ``anthropic.Anthropic`` for ``anthropic_native``;
    #   ``google.genai.Client`` for ``google_native``. ``None`` means
    #   the user hasn't connected that provider yet.
    # ``provider_models[provider_id] -> list[str]``: raw model ids
    #   returned by ``client.models.list()`` for each provider, sorted
    #   by display label (the chat picker shows qualified
    #   ``<provider>:<raw>`` ids; this dict holds just the raw ids). The
    #   model catalog (Phase 2) populates richer ``ModelInfo`` objects
    #   on the side; this list is the lightweight per-provider view.
    # ``connect_errors[provider_id] -> str | None``: most recent connect
    #   error per provider, surfaced as ``st.error`` inside that
    #   provider's Settings card.
    # ------------------------------------------------------------------
    if "provider_keys" not in ss:
        ss.provider_keys = {pid: saved_provider_keys.get(pid, "") for pid in providers.PROVIDERS}
    if "provider_remember" not in ss:
        ss.provider_remember = {
            pid: bool(saved_provider_remember.get(pid, bool(saved_provider_keys.get(pid))))
            for pid in providers.PROVIDERS
        }
    if "clients" not in ss:
        ss.clients = {pid: None for pid in providers.PROVIDERS}
    if "provider_models" not in ss:
        ss.provider_models = {pid: [] for pid in providers.PROVIDERS}
    if "connect_errors" not in ss:
        ss.connect_errors = {pid: None for pid in providers.PROVIDERS}

    # Pre-fill the W&B API key field from disk if the user opted in to
    # "remember on this machine"; the box stays editable. ``ss.api_key``
    # is kept as a back-compat alias for ``ss.provider_keys["wandb"]``
    # so any code path that still reads it sees the same value.
    ss.setdefault("api_key", ss.provider_keys.get("wandb", ""))
    ss.setdefault("project", "")
    # ``ss.client`` is the W&B-specific legacy alias retained for one
    # release; the multi-provider ``ss.clients["wandb"]`` is the
    # canonical key going forward. W&B Inference is ``openai_compat``,
    # dispatched through the OpenAI SDK with W&B's ``base_url``, so
    # both keys hold the same ``openai.OpenAI`` instance after a
    # successful connect.
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
    # Legacy W&B-only alias kept around for one release; the canonical
    # value lives in ``ss.connect_errors["wandb"]``.
    ss.setdefault("connect_error", None)
    ss.setdefault("weave_project", None)
    ss.setdefault("weave_url", None)
    ss.setdefault("weave_error", None)
    ss.setdefault("conn_open", False)
    # Multi-provider model-picker modal state. ``model_picker_open``
    # gates the @st.dialog("Choose a model") modal mounted from the
    # chat page; ``_model_picker_search`` carries the user's filter
    # string across reruns; ``model_catalog_refreshing`` is True
    # while a daemon-thread refresh is in flight (the chat page
    # polls a 0.5s @st.fragment to re-render once it flips back);
    # ``model_catalog_last_refreshed_at`` holds a monotonic
    # timestamp for the "Last refreshed Nm ago" caption (the on-disk
    # cache holds the wall-clock equivalent).
    ss.setdefault("model_picker_open", False)
    ss.setdefault("_model_picker_search", "")
    ss.setdefault("model_catalog_refreshing", False)
    ss.setdefault("model_catalog_last_refreshed_at", None)
    # Phase 5: lightbox modal state for inline image / audio / video
    # previews. ``lightbox_open`` gates the @st.dialog("Preview")
    # modal mounted from the chat page; ``lightbox_payload`` is the
    # ``{"kind", "path", "alt", "caption"}`` dict the dialog reads.
    ss.setdefault("lightbox_open", False)
    ss.setdefault("lightbox_payload", None)
    # Phase 6: pending attachments staged for the next chat-input
    # submission. Cleared after the user sends or removes them.
    ss.setdefault("pending_attachments", [])
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

    # Settings state. ``remember_wb_key`` is the legacy W&B-only flag;
    # the multi-provider ``ss.provider_remember["wandb"]`` is the
    # canonical key going forward but the legacy alias is kept around
    # for one release so existing call sites keep working.
    ss.setdefault("remember_wb_key", bool(ss.provider_remember.get("wandb", False)))
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
    """Auto-connect every provider with a saved key on the first script run.

    Runs exactly once per session: ``auto_connect_attempted`` is flipped
    to ``True`` *before* any connect calls so a transient failure
    (e.g. an expired key, a network blip) does not re-trigger on every
    rerun. The user can still manually click **Connect** in Settings
    after a failure — the error surfaces in the relevant provider's
    ``ss.connect_errors`` slot.

    Auto-connect order: every provider with a saved API key is
    connected synchronously (so the chat picker has at least one
    provider's models populated by the time the first render
    completes). After that, :func:`model_catalog.refresh_all_async`
    runs in a daemon thread to refresh every connected provider's
    catalog (live ``/v1/models`` listing + OpenRouter description
    enrichment for the ``openrouter:*`` namespace) without blocking
    the UI.
    """
    ss = st.session_state
    if ss.auto_connect_attempted:
        return
    ss.auto_connect_attempted = True

    from actions import connect_provider

    saved = account.load_provider_keys()
    connected_count = 0
    for pid, key in saved.items():
        if not key:
            continue
        if pid not in providers.PROVIDERS:
            continue
        # Mirror the saved key into session state so the connect
        # callback finds it (the connect helper reads from
        # ``ss.provider_keys`` after syncing).
        keys = dict(ss.provider_keys)
        keys[pid] = key
        ss.provider_keys = keys
        flags = dict(ss.provider_remember)
        flags[pid] = True
        ss.provider_remember = flags
        try:
            connect_provider(pid)
        except Exception:  # noqa: BLE001 — auto-connect must never crash startup
            continue
        if not ss.connect_errors.get(pid):
            connected_count += 1

    if connected_count > 0:
        st.toast(
            f"Connected {connected_count} provider"
            f"{'s' if connected_count != 1 else ''}.",
            icon=":material/check_circle:",
        )
        # Kick off a background refresh so providers we successfully
        # auto-connected get their full catalog refresh (live
        # ``/v1/models`` listing + OpenRouter enrichment for
        # ``openrouter:*``) without blocking the first render. The
        # chat page's polling @st.fragment will re-render the modal
        # once ``model_catalog_refreshing`` flips back to False.
        import model_catalog
        ss.model_catalog_refreshing = True

        def _on_done() -> None:
            # Background callback — can't touch ss directly from here
            # because Streamlit's session-state proxy is bound to the
            # script thread. The flag flip happens on the next rerun
            # via the polling fragment that watches the catalog's
            # ``newest_refresh()`` timestamp; we just record completion
            # via a module-level flag inside model_catalog. The
            # fragment in app_pages/chat.py reads
            # ``model_catalog.newest_refresh()`` and rerruns when it
            # changes.
            pass

        clients_for_refresh = dict(ss.clients)
        model_catalog.refresh_all_async(
            clients_for_refresh, on_done=_on_done
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
# "Changes" button below the chat input (sitting directly under the
# bottom-of-chat git row), and that git row owns branch switching,
# new-branch creation, fetch, and the one-click "generate commit
# message + commit + push" pipeline. This module is responsible only
# for the cross-page sidebar warning when the working tree is
# mid-merge/rebase (so users see it from any page); everything else
# moved into the chat page.


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
    """Sidebar row callback: mark this chat as the active one.

    Also queues a page switch to the Agent tab (``app_pages/chat.py``)
    via ``ss["_pending_page_switch"]`` so a click from any other tab
    (Settings / Usage / Docs) lands the user on the chat itself rather
    than leaving them looking at the chat list while the actual chat
    history is on a different page. The drain in :func:`main` (after
    :func:`st.navigation` registers the pages) runs the actual
    :func:`st.switch_page` call.

    We can't call :func:`st.switch_page` directly from this callback:
    it raises ``RerunException`` internally (after queueing the rerun
    and yielding via ``st.empty()``) and Streamlit surfaces a yellow
    "Calling st.rerun() within a callback is a no-op." warning at the
    top of the page when a callback raises that. The navigation
    *does* still work in that flow, but the warning is loud and
    confusing to users. Routing through ``_pending_page_switch``
    avoids the warning entirely while preserving the cross-tab
    navigation UX.
    """
    if chat_id in st.session_state.chats:
        st.session_state.active_chat_id = chat_id
        chats.save_active_chat_id(chat_id)
        # Forget the dropdown-sync sentinel so the next chat-page render
        # re-copies the chat's model / mode / working_dir into the flat
        # ss.* keys driving the dropdowns. See app_pages/chat.py.
        st.session_state.pop("_last_active_chat_id", None)
        # Queue a page switch to the Agent tab for this same rerun so a
        # click from any other tab (Settings / Usage / Docs) lands the
        # user on the chat itself. Calling :func:`st.switch_page`
        # directly from a callback raises ``RerunException`` which
        # Streamlit catches and surfaces as a "Calling st.rerun() within
        # a callback is a no-op" warning at the top of the page (it
        # **does** still navigate, but the warning is loud); deferring
        # to ``main()`` (after ``st.navigation`` registers the pages)
        # keeps the navigation working without the warning.
        st.session_state["_pending_page_switch"] = "app_pages/chat.py"


def _new_chat() -> None:
    """`+ New chat` button callback.

    If a blank ``+ New chat`` placeholder already exists (default title,
    no user / assistant turns, not archived), activates that chat instead
    of minting another empty row — clicking the button repeatedly should
    never accumulate a pile of identical placeholders. The reused chat
    has its model / mode / working_dir refreshed to the user's current
    dropdown picks so it behaves like a freshly seeded chat.

    Otherwise creates a new chat seeded with the current dropdown values.

    On either path, the callback queues a page switch to the Agent tab
    via ``ss["_pending_page_switch"]`` so a click from a non-Agent tab
    (Settings / Usage / Docs) lands the user on the chat input
    immediately. The drain in :func:`main` runs the actual
    :func:`st.switch_page` call after :func:`st.navigation` has
    registered the pages, sidestepping the "Calling st.rerun() within
    a callback is a no-op" warning that direct ``st.switch_page``
    calls from this callback would trigger. See :func:`_activate_chat`
    for the full rationale.
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
    else:
        chat = chats.new_chat(
            model=ss.model or "",
            mode=ss.mode or "agent",
            working_dir=ss.working_dir or "",
        )
        ss.chats[chat.id] = chat
        ss.active_chat_id = chat.id
        chats.save_active_chat_id(chat.id)
        ss.pop("_last_active_chat_id", None)
    # Same deferred page-switch pattern as :func:`_activate_chat` —
    # calling :func:`st.switch_page` directly from a callback raises
    # ``RerunException`` and Streamlit shows a "Calling st.rerun()
    # within a callback is a no-op" warning. The drain in :func:`main`
    # runs the switch after :func:`st.navigation` has registered the
    # pages, so navigation still works but the warning never fires.
    ss["_pending_page_switch"] = "app_pages/chat.py"


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
    Stop-and-delete handlers inside the dialog body clear the same
    id and call ``st.rerun()`` directly.
    """
    st.session_state.delete_chat_confirm_id = None


def _finalize_chat_deletion(chat_id: str) -> None:
    """Pick a successor active chat after ``chat_id`` has been deleted.

    Shared between the idle and running delete paths so they don't
    drift. Caller is expected to have already removed ``chat_id``
    from ``ss.chats`` via :func:`chats.delete_chat`. When the deleted
    chat was the active one we hand the chat page over to the most
    recent surviving live chat, falling back to a fresh placeholder
    when the sidebar has been drained empty.
    """
    ss = st.session_state
    successors = _sorted_chats(archived=False)
    if successors:
        ss.active_chat_id = successors[0].id
    else:
        ss.active_chat_id = _seed_default_chat()
    chats.save_active_chat_id(ss.active_chat_id)
    ss.pop("_last_active_chat_id", None)


@st.dialog("Delete chat?", width="small", on_dismiss=_close_delete_chat_dialog)
def _delete_chat_dialog() -> None:
    """Confirm modal for the per-chat delete icon.

    Two body shapes depending on the chat's status:

    - **Idle / errored / new**: a plain "Delete `<title>`?" prompt
      with Cancel and Delete buttons. Calls
      :func:`chats.delete_chat` which unlinks the on-disk JSON file.
    - **Running**: a warning that deleting will stop the in-flight
      turn (interrupting the streamed chat-completion call so we
      stop spending tokens) plus Cancel and **Stop and delete**
      buttons. Stop and delete calls
      :func:`chats.delete_chat` with ``force=True``, which signals
      :func:`chats.request_cancel` before unlinking the file. The
      background thread checks the cancel event between agent events
      and before every persistence call, so the file we just unlinked
      stays gone.

    In both cases, when the deleted chat was the active one we pick
    a successor (most recent surviving live chat, falling back to a
    fresh ``+ New chat`` placeholder when the sidebar drains empty).

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

    title = chat.title or "(untitled)"
    is_running = chat.status == chats.STATUS_RUNNING

    if is_running:
        st.markdown(f"Stop turn and delete **{title}**?")
        st.warning(
            "This chat is still running a turn. Deleting will stop the "
            "model immediately so you don't spend more tokens, but you'll "
            "lose any in-flight reply. This cannot be undone.",
            icon=":material/warning:",
        )
        confirm_label = "Stop and delete"
        confirm_icon = ":material/cancel:"
    else:
        st.markdown(f"Delete **{title}**?")
        st.caption(
            "This permanently removes the chat from disk. It cannot be undone."
        )
        confirm_label = "Delete"
        confirm_icon = ":material/delete:"

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
        confirm_label,
        icon=confirm_icon,
        type="primary",
        width="stretch",
        key="delete_chat_confirm_btn",
    ):
        was_active = ss.active_chat_id == chat_id
        try:
            chats.delete_chat(ss.chats, chat_id, force=is_running)
        except RuntimeError as e:
            # Defensive: ``force=True`` should never raise on the
            # running path, and idle chats can't transition into
            # ``STATUS_RUNNING`` from a different page (turns are only
            # spawned from the chat page). Surface anything that
            # slips through so the user can retry rather than the
            # dialog silently no-op'ing.
            st.error(str(e), icon=":material/error:")
            return
        if was_active:
            _finalize_chat_deletion(chat_id)
        ss.delete_chat_confirm_id = None
        st.rerun()


def _truncate(text: str, limit: int) -> str:
    """Tail-truncate ``text`` to ``limit`` characters with a unicode ellipsis."""
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "\u2026"


def _render_chat_row(chat: chats.Chat, *, archived: bool) -> None:
    """One sidebar row.

    Both live and archived rows share the same 3-column layout:
    ``[main control | archive/unarchive icon | delete icon]``. Putting
    the destructive icons inline with the main control means they're
    reachable from any chat without first activating it.

    For **live** rows the main control is a stretched ``st.button``;
    clicking it activates the chat (which loads its history into the
    chat page). The currently-active row is rendered as
    ``type="secondary"`` (outlined) and ``disabled=True`` so the user
    can see at a glance which chat the chat page is showing **and**
    can't re-click their own selection. Inactive rows are
    ``type="tertiary"`` (borderless / text-only) so a long chat list
    reads as a tidy list of titles rather than a wall of buttons.

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
            type="secondary" if is_active else "tertiary",
            disabled=is_active,
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

    # Drain any deferred page-switch request from sidebar callbacks
    # (``_activate_chat`` / ``_new_chat``). They can't call
    # :func:`st.switch_page` directly because Streamlit raises a
    # ``RerunException`` from inside ``switch_page`` and surfaces a
    # "Calling st.rerun() within a callback is a no-op" warning when
    # that happens inside a widget callback. Running the switch here
    # — after :func:`st.navigation` has registered the pages, but
    # before :func:`page.run` paints anything — keeps the cross-tab
    # "click chat row from Settings / Usage / Docs jumps you to the
    # Agent page" UX intact, without the warning. ``st.switch_page``
    # itself raises after queueing the rerun, so anything below is
    # only reached on the no-pending-switch path.
    pending_switch = st.session_state.pop("_pending_page_switch", None)
    if pending_switch:
        st.switch_page(pending_switch)

    page.run()


main()
