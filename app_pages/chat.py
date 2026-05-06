"""Chat page: multi-chat tool-calling UI backed by ``chats.start_turn``.

This page renders **the active chat** out of ``ss.chats`` (a dict of
:class:`chats.Chat`); see ``streamlit_app._render_sidebar`` for the
chat-list panel that picks the active id. Turns run on background
daemon threads so multiple chats can be in flight at once; live token
streaming + tool-call updates flow through an ``@st.fragment`` that
re-renders every 0.25s while the active chat is in ``"running"``
status, then settles to a one-shot render once it's idle.

State of the world this page expects (initialized in ``streamlit_app.py``):

- ``ss.client`` is a connected OpenAI-pointed-at-W&B-Inference client.
- ``ss.chats`` maps chat ids to :class:`chats.Chat` instances.
- ``ss.active_chat_id`` is the id of one of those chats.
- ``ss.working_dir`` / ``ss.model`` / ``ss.mode`` are the *flat* keys
  that drive the dropdowns; on chat switch we sync them from the
  active chat (and on user edit we sync them back).

Side effects:

- :func:`chats.start_turn` appends per-turn token totals to
  ``~/.wb_coding_agent/usage.jsonl`` via :func:`usage.record_usage` so
  the Usage dashboard picks them up.
- If a verified GitHub identity is present in ``ss.github_identity``
  and the active chat's ``working_dir`` is a git repo,
  ``account.apply_git_identity`` is called once per (session,
  working_dir) pair (tracked in ``ss.git_identity_applied``) so
  commits the agent makes via ``run_shell`` are authored as that user.
"""
from __future__ import annotations

import json
import re
import webbrowser
from pathlib import Path
from typing import Any

import streamlit as st

import account
import chats
import commit_ai
import git_ops
import mcp_servers
import project_context
import usage as usage_log
from agent import DEEPSEEK_MODEL
from chat_input import mount_slash_autocomplete
from git_ops import GitError
from models import MODEL_METADATA, model_label

TOOL_ICONS = {
    "list_files": ":material/folder_open:",
    "read_file": ":material/description:",
    "write_file": ":material/edit_note:",
    "edit_file": ":material/edit:",
    "run_shell": ":material/terminal:",
}

MCP_TOOL_ICON = ":material/extension:"

# Pixel height of the scrollable chat history container. ``st.container``
# only enables internal scrolling (and the autoscroll-on-new-`st.chat_message`
# behaviour we lean on during streaming) when ``height`` is a fixed integer
# — the ``"stretch"`` variant only matches parent height and does not turn on
# scrolling, per the [`st.container` docs](https://docs.streamlit.io/develop/api-reference/layout/st.container).
# 380px is calibrated against a typical desktop viewport (≈900px tall) so
# the chat input, workdir picker, and model selector below the conversation
# area stay in view without the user having to scroll the whole page —
# i.e. the controls "stay pinned at the bottom" of the visible viewport for
# the common case. On taller viewports there will be empty space below the
# controls; on shorter viewports the page itself gains a scrollbar. We
# accept that trade-off because trying to make the height truly responsive
# via CSS ``calc(100vh - …)`` requires fighting with Streamlit-internal
# class names on the height-bearing inner element, which is unstable across
# upgrades — empirically tested and reverted, see git history. Tweak this
# integer if you change the natural height of anything outside the
# conversation area on the chat page (title above, chat input + workdir
# row + project context expander + model row + model card caption below).
_CHAT_HISTORY_HEIGHT_PX = 330


@st.cache_data(ttl=5, show_spinner=False)
def _scan_project_summary(working_dir: str) -> dict[str, Any]:
    """Cached UI summary of project context — TTL so AGENTS.md edits surface fast."""
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
    icon = MCP_TOOL_ICON if is_mcp else TOOL_ICONS.get(name, ":material/build:")
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


def _render_usage_event(
    event: dict[str, Any], *, weave_trace_url: str | None = None
) -> None:
    """Render the per-turn footer caption: tokens, cost, model, and
    (optionally) a deep link to this turn's W&B Weave trace.

    The model label is included so a mid-chat model switch is
    immediately visible on every turn the user scrolls past, rather
    than only on the model dropdown for the *next* turn — without it,
    a user who accidentally swapped from a $0.01/1M model to a
    $0.55/1M one wouldn't see the per-turn cost change attributed to
    the model that produced it.

    The trace URL is sourced from the per-turn ``weave_trace`` event the
    agent yields once at the start of every turn (when Weave is
    initialized); the caller is expected to pre-scan for it and pass it
    in so this renderer can fold it into the same single caption rather
    than emitting a second one underneath.
    """
    total = int(event.get("total_tokens") or 0)
    cost = event.get("cost_usd")
    parts = [f":material/data_usage: {usage_log.format_tokens(total)} tokens"]
    if cost is not None:
        parts.append(usage_log.format_cost(cost))
    rounds = int(event.get("rounds") or 0)
    if rounds > 1:
        parts.append(f"{rounds} rounds")
    model = str(event.get("model") or "")
    if model:
        parts.append(model_label(model))
    if weave_trace_url:
        parts.append(f"[:material/sensors: View trace]({weave_trace_url})")
    st.caption(" \u00b7 ".join(parts))


def _render_trace_only(weave_trace_url: str) -> None:
    """Render a standalone trace-link caption when there's no usage row.

    Used for turns that errored before any usage was recorded — without
    this the trace link (which the agent yields immediately on entry,
    before any inference call) would silently disappear from the UI.
    """
    st.caption(f":material/sensors: [View trace in Weave]({weave_trace_url})")


def _render_assistant_turn(turn: dict[str, Any]) -> None:
    events: list[dict[str, Any]] = turn.get("events", [])

    pending_calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    weave_trace_url: str | None = None
    has_turn_usage = False
    for ev in events:
        etype = ev["type"]
        if etype == "tool_call":
            pending_calls[ev["id"]] = ev
        elif etype == "tool_result":
            results[ev["id"]] = ev
        elif etype == "weave_trace":
            url = ev.get("url")
            if isinstance(url, str) and url:
                weave_trace_url = url
        elif etype == "turn_usage":
            has_turn_usage = True

    with st.chat_message("assistant"):
        for ev in events:
            etype = ev["type"]
            if etype == "skills_loaded":
                _render_skills_loaded(ev)
            elif etype == "assistant_text":
                content = ev.get("content") or ""
                if content.strip():
                    st.markdown(content)
            elif etype == "tool_call":
                _render_tool_event(ev, results.get(ev["id"]))
            elif etype == "tool_result":
                continue
            elif etype == "turn_usage":
                _render_usage_event(ev, weave_trace_url=weave_trace_url)
            elif etype == "weave_trace":
                # Folded into the turn_usage caption above; rendered
                # standalone only when no turn_usage was recorded
                # (handled after the loop).
                continue
            elif etype == "error":
                st.error(ev["message"], icon=":material/error:")
        if weave_trace_url and not has_turn_usage:
            _render_trace_only(weave_trace_url)


def _render_user_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(turn["content"])


def _maybe_apply_git_identity(working_dir: Path) -> None:
    """Stamp git author once per (session, working_dir) when GitHub is verified.

    We track applied (working_dir, login) pairs in
    ``ss.git_identity_applied`` so opening a new project mid-session re-runs
    the stamp on that project's ``.git`` config. Failures are silent (e.g.
    the working dir isn't a git repo) — the dashboard's GitHub section is
    informational, not load-bearing.
    """
    ss = st.session_state
    identity = ss.get("github_identity") or {}
    login = identity.get("login")
    name = identity.get("name") or login or ""
    email = identity.get("email") or ""
    if not (login and email):
        return
    applied = ss.setdefault("git_identity_applied", set())
    key = (str(working_dir), login)
    if key in applied:
        return
    ok, _msg = account.apply_git_identity(working_dir, name, email)
    if ok:
        applied.add(key)


def _start_turn(prompt: str, *, override_model: str | None = None) -> None:
    """Spawn a background turn for the active chat.

    Thin wrapper that pulls the active :class:`chats.Chat` out of session
    state, syncs the user's flat dropdown picks (model / mode / working
    dir) into the chat object so the background thread sees the
    user's intent, runs the per-(session, working_dir) git identity
    stamp, and delegates to :func:`chats.start_turn`. The fragment in
    :func:`_render_active_chat` picks up the streaming events on its
    next 0.25s tick.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id)
    if chat is None:
        st.error("No active chat. Click **New chat** in the sidebar.", icon=":material/error:")
        return
    # Sync the user's current dropdown picks back to the chat so the
    # background thread sees them. Persist *after* :func:`chats.start_turn`
    # appends the user message — that helper already calls
    # :func:`chats.save_chat` once it has flipped the status.
    chat.model = ss.model or chat.model
    chat.mode = ss.mode or chat.mode
    chat.working_dir = ss.working_dir or chat.working_dir
    working_dir = Path(chat.working_dir).expanduser().resolve()
    _maybe_apply_git_identity(working_dir)
    try:
        chats.start_turn(
            chat,
            prompt,
            ss.client,
            override_model=override_model,
        )
    except RuntimeError as e:
        st.error(str(e), icon=":material/error:")


def _shorten_path(path: str) -> str:
    if not path:
        return path
    home = str(Path.home())
    if path == home:
        return "~"
    import os as _os
    if path.startswith(home + _os.sep):
        return "~" + path[len(home):]
    return path


def _on_working_dir_select() -> None:
    """Working-directory selectbox callback: persist + sync to active chat.

    Recents and folder-picker live in ``actions.py`` so pages can
    import them without dragging the entry-point script back through
    Python's import machinery (Streamlit loads the entry as
    ``__main__``; importing it from a sub-page re-executes ``main()``
    and re-renders the sidebar, blowing up with duplicate-widget-key
    errors).
    """
    chosen = st.session_state.get("wd_select")
    if not chosen:
        return
    import actions

    st.session_state.working_dir = chosen
    actions.record_recent_dir(chosen)
    _persist_active_chat_setting("working_dir", chosen)


def _on_mode_change() -> None:
    """Mode dropdown callback: sync widget value → canonical state + active chat.

    ``ss._chat_mode_input`` is the widget key (Streamlit owns it and
    will strip it on widget unmount); ``ss.mode`` is the canonical
    non-widget key that survives navigation. The dual-key dance is
    documented in ``AGENTS.md`` as the canonical fix for "widget
    state vanishes when the user visits another page".
    """
    new_mode = st.session_state.get("_chat_mode_input") or "agent"
    st.session_state.mode = new_mode
    _persist_active_chat_setting("mode", new_mode)


def _on_model_change() -> None:
    """Model dropdown callback: sync widget value → canonical state + active chat."""
    new_model = st.session_state.get("_chat_model_input") or ""
    st.session_state.model = new_model
    _persist_active_chat_setting("model", new_model)


def _persist_active_chat_setting(field: str, value: str) -> None:
    """Mirror a flat ``ss.*`` value onto the active chat and persist.

    Called from the on-change hooks above. The chat is the source of
    truth for "what settings does this conversation use" — the flat
    ``ss.*`` keys are just the dropdown's view of the active chat. We
    also bump ``updated_at`` so the sidebar re-sorts the chat row to
    the top after the edit.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.get("active_chat_id") else None
    if chat is None:
        return
    setattr(chat, field, value)
    try:
        chats.save_chat(chat)
    except OSError:
        pass


def _render_project_context_indicator(working_dir: str) -> None:
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
    ss = st.session_state

    wd_options: list[str] = []
    if ss.working_dir:
        wd_options.append(ss.working_dir)
    for d in ss.recent_dirs:
        if d not in wd_options:
            wd_options.append(d)

    # The branch switcher relocated to the per-chat sidebar row (so each
    # chat can pin its own branch without sharing a single dropdown);
    # this row stays the simpler 3-column layout: workdir + browse +
    # new-project.
    wd_cols = st.columns([10, 1, 1], vertical_alignment="bottom")

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
                "custom path, click the folder icon to browse, or the plus "
                "icon to start a new project."
            ),
        )

    browse_col = wd_cols[1]
    new_proj_col = wd_cols[2]

    with browse_col:
        if st.button(
            "",
            icon=":material/folder_open:",
            key="wd_pick_btn",
            help="Browse for a working directory",
            width="stretch",
        ):
            import actions
            chosen = actions.pick_directory(initial=ss.working_dir)
            if chosen:
                ss.working_dir = chosen
                actions.record_recent_dir(chosen)
                st.rerun()
    with new_proj_col:
        st.button(
            "",
            icon=":material/create_new_folder:",
            key="wd_new_proj_btn",
            help="Start a new project (create a folder, git init, optionally wire an upstream).",
            on_click=_open_new_project_dialog,
            width="stretch",
        )

    _render_project_context_indicator(ss.working_dir)

    if ss.get("new_project_dialog_open"):
        _new_project_dialog()


# ---------------------------------------------------------------------------
# Start-a-new-project dialog
# ---------------------------------------------------------------------------
# This dialog lives next to ``_render_workdir_controls`` because it's an
# affordance of the working-directory picker, not a standalone page-level
# concern. All filesystem / GitHub / git work is delegated to ``account.py``
# (see "Project bootstrap helpers" there); this module only owns the form
# fields, validation, and orchestration.

_UPSTREAM_NONE = "None"
_UPSTREAM_REMOTE = "Link existing remote"
_UPSTREAM_CLONE = "Clone GitHub repo"
_UPSTREAM_CREATE = "Create on GitHub"


def _open_new_project_dialog() -> None:
    """Button callback: flip the dialog open flag."""
    st.session_state.new_project_dialog_open = True


def _close_new_project_dialog() -> None:
    """Drop the dialog flag so subsequent reruns don't re-mount the modal."""
    st.session_state.new_project_dialog_open = False


@st.cache_data(ttl=120, show_spinner=False)
def _cached_user_repos(pat_hash: str, _pat: str) -> list[dict[str, Any]]:
    """Fetch + cache the user's GitHub repos.

    Cache key is ``pat_hash`` (a SHA-256 of the PAT) so the actual token is
    not used as a hashable cache argument. The leading-underscore parameter
    ``_pat`` is intentionally not hashed by Streamlit — it carries the
    secret value through to ``account.list_user_repos`` without persisting
    it as a cache key.
    """
    return account.list_user_repos(_pat)


def _format_repo_option(repo: dict[str, Any]) -> str:
    """Compact ``selectbox`` label: ``owner/name (private · last updated)``."""
    visibility = "private" if repo.get("private") else "public"
    bits = [repo.get("full_name") or ""]
    extra: list[str] = [visibility]
    updated = repo.get("updated_at") or ""
    if updated:
        extra.append(updated.split("T", 1)[0])
    return f"{bits[0]} ({' \u00b7 '.join(extra)})"


@st.dialog("Start a new project", width="large")
def _new_project_dialog() -> None:
    """Modal that creates a project folder, ``git init``s it, and wires an upstream.

    The four upstream modes (None / paste-remote / clone / create-on-GitHub)
    map to short orchestrations against ``account.py``. Errors raised by
    those helpers (validation, network, git failures) surface as a single
    ``st.error`` and leave the dialog open so the user can fix and retry.
    """
    import actions

    ss = st.session_state

    st.caption(
        "Create a new folder, initialize it as a git repo, and optionally "
        "link or create an upstream. The new directory becomes the agent's "
        "working directory once it's ready."
    )

    parent_default = ss.get("new_proj_parent") or str(Path.home())
    p_cols = st.columns([10, 1], vertical_alignment="bottom")
    with p_cols[0]:
        parent_str = st.text_input(
            "Parent directory",
            value=parent_default,
            key="new_proj_parent",
            help="Where to create the new folder. Defaults to your home directory.",
        )
    with p_cols[1]:
        if st.button(
            "",
            icon=":material/folder_open:",
            key="new_proj_parent_btn",
            help="Browse for a parent directory",
            width="stretch",
        ):
            chosen = actions.pick_directory(initial=parent_str)
            if chosen:
                ss.new_proj_parent = chosen
                st.rerun()

    folder_name = st.text_input(
        "Folder name",
        key="new_proj_name",
        placeholder="my-new-project",
        help="A single folder name (no slashes). Created inside the parent above.",
    )

    parent_path: Path | None = None
    parent_str_clean = (parent_str or "").strip()
    if parent_str_clean:
        try:
            parent_path = Path(parent_str_clean).expanduser().resolve()
        except (OSError, RuntimeError):
            parent_path = None

    if parent_path and folder_name:
        st.caption(
            f":material/folder: Will create at `{parent_path / folder_name.strip()}`"
        )

    upstream = st.segmented_control(
        "Upstream repo",
        options=[_UPSTREAM_NONE, _UPSTREAM_REMOTE, _UPSTREAM_CLONE, _UPSTREAM_CREATE],
        default=ss.get("new_proj_upstream", _UPSTREAM_NONE),
        key="new_proj_upstream",
    ) or _UPSTREAM_NONE

    pat = (account.load_credentials().get("github_pat") or "").strip()
    needs_pat = upstream in (_UPSTREAM_CLONE, _UPSTREAM_CREATE)

    selected_repo: dict[str, Any] | None = None
    remote_url = ""
    new_repo_name = ""
    new_repo_desc = ""
    new_repo_private = True

    if upstream == _UPSTREAM_REMOTE:
        remote_url = st.text_input(
            "Remote URL",
            key="new_proj_remote_url",
            placeholder="https://github.com/owner/repo.git",
            help="Any git remote URL — added as 'origin' after git init.",
        )
    elif needs_pat and not pat:
        st.warning(
            "This option needs a GitHub personal access token. Verify a PAT "
            "in **Settings \u2192 GitHub** first.",
            icon=":material/warning:",
        )
    elif upstream == _UPSTREAM_CLONE:
        try:
            import hashlib

            pat_hash = hashlib.sha256(pat.encode("utf-8")).hexdigest()
            repos = _cached_user_repos(pat_hash, pat)
        except ValueError as e:
            st.error(f"Could not list your repos: {e}", icon=":material/error:")
            repos = []
        if repos:
            options = list(range(len(repos)))
            picked = st.selectbox(
                "GitHub repo",
                options=options,
                key="new_proj_clone_idx",
                format_func=lambda i: _format_repo_option(repos[i]),
                help="Cloned via HTTPS using your PAT for authentication.",
            )
            if picked is not None:
                selected_repo = repos[picked]
                if selected_repo.get("description"):
                    st.caption(selected_repo["description"])
                if not (folder_name or "").strip() and selected_repo.get("name"):
                    st.caption(
                        f":material/info: Folder name will default to `{selected_repo['name']}`."
                    )
        elif pat:
            st.caption("No repositories found on this account.")
    elif upstream == _UPSTREAM_CREATE:
        new_repo_name = st.text_input(
            "Repository name",
            value=(folder_name or "").strip(),
            key="new_proj_create_name",
            placeholder="my-new-project",
            help="The name of the repo created on GitHub. Defaults to the folder name.",
        )
        new_repo_desc = st.text_input(
            "Description (optional)",
            key="new_proj_create_desc",
            placeholder="What this project does",
        )
        visibility = st.segmented_control(
            "Visibility",
            options=["Private", "Public"],
            default="Private",
            key="new_proj_create_vis",
        ) or "Private"
        new_repo_private = visibility == "Private"

    cols = st.columns([1, 1])
    cancel_clicked = cols[0].button(
        "Cancel",
        icon=":material/close:",
        key="new_proj_cancel_btn",
        width="stretch",
    )
    create_clicked = cols[1].button(
        "Create",
        icon=":material/check:",
        key="new_proj_create_btn",
        type="primary",
        width="stretch",
    )

    if cancel_clicked:
        _close_new_project_dialog()
        st.rerun()

    if not create_clicked:
        return

    if parent_path is None or not parent_path.is_dir():
        st.error("Pick a valid parent directory.", icon=":material/error:")
        return

    # For the clone path we let the repo's name fill in if the user didn't
    # type one; everywhere else, folder name is required up front.
    effective_name = (folder_name or "").strip()
    if upstream == _UPSTREAM_CLONE and not effective_name and selected_repo:
        effective_name = (selected_repo.get("name") or "").strip()

    if not effective_name:
        st.error("Folder name is required.", icon=":material/error:")
        return

    try:
        if upstream == _UPSTREAM_CLONE:
            if selected_repo is None:
                st.error("Pick a GitHub repo to clone.", icon=":material/error:")
                return
            clone_url = (selected_repo.get("clone_url") or "").strip()
            if not clone_url:
                st.error("Selected repo has no clone URL.", icon=":material/error:")
                return
            dest = (parent_path / effective_name).resolve()
            if dest.exists():
                st.error(
                    f"{dest} already exists. Pick a different folder name.",
                    icon=":material/error:",
                )
                return
            account.git_clone(pat, clone_url, dest)
        else:
            dest = account.create_project_directory(parent_path, effective_name)
            account.git_init(dest)
            if upstream == _UPSTREAM_REMOTE:
                if not (remote_url or "").strip():
                    st.error("Remote URL is required.", icon=":material/error:")
                    return
                account.git_add_remote(dest, "origin", remote_url.strip())
            elif upstream == _UPSTREAM_CREATE:
                if not (new_repo_name or "").strip():
                    st.error("Repository name is required.", icon=":material/error:")
                    return
                repo = account.create_user_repo(
                    pat,
                    new_repo_name.strip(),
                    description=new_repo_desc,
                    private=new_repo_private,
                )
                clone_url = str(repo.get("clone_url") or "")
                if not clone_url:
                    st.error(
                        "GitHub created the repo but did not return a clone URL.",
                        icon=":material/error:",
                    )
                    return
                account.git_add_remote(dest, "origin", clone_url)
    except ValueError as e:
        st.error(str(e), icon=":material/error:")
        return
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}", icon=":material/error:")
        return

    ss.working_dir = str(dest)
    actions.record_recent_dir(str(dest))
    # Reset the per-dialog form fields so the next open starts clean. The
    # parent stays sticky (in ``new_proj_parent``) by design.
    for k in (
        "new_proj_name",
        "new_proj_remote_url",
        "new_proj_clone_idx",
        "new_proj_create_name",
        "new_proj_create_desc",
        "new_proj_create_vis",
        "new_proj_upstream",
    ):
        ss.pop(k, None)

    st.toast(f"Created '{effective_name}'", icon=":material/check_circle:")
    _close_new_project_dialog()
    st.rerun()


def _render_model_controls() -> None:
    ss = st.session_state

    # Dual-key pattern (per ``AGENTS.md``): the widgets bind to their
    # own ``_chat_*_input`` keys so Streamlit can freely strip them on
    # widget unmount, while the canonical ``ss.mode`` / ``ss.model``
    # keys are non-widget and survive navigation. Re-seed the widget
    # keys from canonical on every render so a remount after a Diff /
    # Usage / Settings visit picks the right initial selection.
    ss["_chat_mode_input"] = ss.mode if ss.mode in ("agent", "ask") else "agent"
    if ss.models and ss.model in ss.models:
        ss["_chat_model_input"] = ss.model

    cols = st.columns([1, 2, 1], vertical_alignment="bottom")
    with cols[0]:
        st.selectbox(
            "Mode",
            options=["agent", "ask"],
            key="_chat_mode_input",
            on_change=_on_mode_change,
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
                key="_chat_model_input",
                on_change=_on_model_change,
                format_func=model_label,
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
        # Pricing chip — surfaces input/output $/M alongside context/params so
        # the user knows what each turn will cost before sending it.
        inp = meta.get("input_price_per_1m")
        out = meta.get("output_price_per_1m")
        if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
            chips.append(f"${inp:g}/${out:g} per 1M in/out")
        header = f":material/info: **{meta['label']}**"
        if chips:
            header += " \u00b7 " + " \u00b7 ".join(chips)
        desc = meta.get("description", "")
        st.caption(f"{header} \u2014 {desc}" if desc else header)


# ---------------------------------------------------------------------------
# Working-tree diff (button + dialog overlay)
# ---------------------------------------------------------------------------
# The chat page is the single rendering site for the live working-tree
# unified diff. Surface area:
#   - ``_render_chat_diff_button``: a pinned "Changes" button at the top
#     of the sticky bottom-controls block; hidden when there are no
#     changes / not a git repo.
#   - ``_diff_dialog`` (mounted from ``render()``): a modal overlay
#     showing the TOC + per-file diff sections, gated by
#     ``ss.diff_dialog_open``.
#
# We re-implement the cached git scan locally (rather than import the
# one in ``streamlit_app.py``) so this page module never re-runs the
# entry script — Streamlit loads ``streamlit_app.py`` as ``__main__``
# and importing it from a sub-page would trigger ``main()`` again.
@st.cache_data(ttl=3, show_spinner=False)
def _cached_diff_git_scan(working_dir: str, _nonce: int) -> dict[str, Any]:
    """Cached :func:`git_ops.scan` keyed off ``working_dir`` + ``_nonce``."""
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


def _diff_git_state(working_dir: str) -> dict[str, Any]:
    """Pull the cached git scan dict for the chat-page diff helpers."""
    return _cached_diff_git_scan(
        working_dir,
        int(st.session_state.get("git_state_nonce") or 0),
    )


def _diff_entry_state_label(entry: git_ops.StatusEntry) -> str:
    """Compact ``[staged|...]`` label hint shown alongside +/- counts."""
    if entry.is_untracked:
        return ":green[Untracked]"
    if entry.is_deleted:
        return ":red[Deleted]"
    if entry.is_renamed:
        return ":blue[Renamed]"
    parts: list[str] = []
    if entry.staged_status not in (" ", "?"):
        parts.append("staged")
    if entry.unstaged_status not in (" ", "?"):
        parts.append("unstaged")
    return ":gray[" + ", ".join(parts) + "]" if parts else ""


def _diff_for_entry(working_dir: Path, entry: git_ops.StatusEntry) -> str:
    """Return a unified diff string, robust to ``git diff`` failures."""
    try:
        return git_ops.diff_for_path(working_dir, entry.path, untracked=entry.is_untracked)
    except GitError as e:
        return f"(git diff failed: {e.stderr})"


def _render_diff_file_section(
    entry: git_ops.StatusEntry,
    counts: dict[str, tuple[int, int]],
    working_dir: Path,
) -> None:
    """Render one file as a collapsed-by-default ``st.expander``.

    The expander label carries everything the user needs to triage a
    file without expanding it: path (or ``orig → new`` for renames), a
    ``:green[+adds] :red[−dels]`` chip, and a state badge for
    untracked / deleted / renamed / staged-only / unstaged-only files.
    Expanding the expander shows the unified ``git diff`` body. We
    deliberately render every file collapsed (``expanded=False``)
    because long working-tree diffs were dominating the modal — the
    user explicitly asked for "collapsed by default" so the dialog
    reads as a list of files first, with the line-by-line walk
    available on demand.
    """
    adds, dels = counts.get(entry.path, (0, 0))
    if entry.is_untracked and adds == 0:
        adds = git_ops.untracked_line_count(working_dir, entry.path)

    if entry.is_renamed and entry.orig_path:
        path_part = f"`{entry.orig_path}` → `{entry.path}`"
    else:
        path_part = f"`{entry.path}`"

    label_parts: list[str] = [path_part]
    if adds or dels:
        label_parts.append(f":green[+{adds}] :red[\u2212{dels}]")
    state_label = _diff_entry_state_label(entry)
    if state_label:
        label_parts.append(state_label)
    label = "  \u00b7  ".join(label_parts)

    with st.expander(label, expanded=False):
        diff = _diff_for_entry(working_dir, entry)
        if diff.strip():
            st.code(diff, language="diff")
        else:
            st.caption("(no textual diff — likely a binary file)")


def _open_diff_dialog() -> None:
    """Changes-button callback: flip the dialog open."""
    st.session_state.diff_dialog_open = True


def _close_diff_dialog() -> None:
    """Drop the dialog flag so subsequent reruns don't re-mount the modal."""
    st.session_state.diff_dialog_open = False


@st.dialog("Changes", width="large")
def _diff_dialog() -> None:
    """Modal overlay showing the live working-tree diff for the active chat.

    Body shape: a per-chat caption (chat title · branch · file count)
    followed by one collapsed-by-default ``st.expander`` per changed
    file. Each expander's label carries the path + ``+adds −dels``
    chip + state badge, so the dialog reads as a compact triage list
    on open; the user expands the files they care about to see the
    line-by-line unified diff. The dialog handles its own internal
    scroll, so opening it never pushes the chat input or controls
    out of the way.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    working_dir_str = (chat.working_dir if chat else "") or ss.working_dir or ""

    if not working_dir_str:
        st.info(
            "Pick a working directory below the chat input to see file diffs.",
            icon=":material/folder_open:",
        )
        if st.button("Close", key="diff_dlg_close_no_wd", width="stretch"):
            _close_diff_dialog()
            st.rerun()
        return

    working_dir = Path(working_dir_str).expanduser()
    if not working_dir.is_dir():
        st.warning(
            f"Working directory `{working_dir_str}` does not exist.",
            icon=":material/folder_off:",
        )
        if st.button("Close", key="diff_dlg_close_no_dir", width="stretch"):
            _close_diff_dialog()
            st.rerun()
        return

    if not git_ops.is_git_installed():
        st.error(
            "Git is not installed on PATH. Install git to see diffs.",
            icon=":material/error:",
        )
        if st.button("Close", key="diff_dlg_close_no_git", width="stretch"):
            _close_diff_dialog()
            st.rerun()
        return

    state = _diff_git_state(working_dir_str)
    if not state.get("in_repo"):
        st.info(
            f"`{working_dir_str}` is not a git repository.",
            icon=":material/info:",
        )
        if st.button("Close", key="diff_dlg_close_no_repo", width="stretch"):
            _close_diff_dialog()
            st.rerun()
        return

    entries: list[git_ops.StatusEntry] = list(state.get("status") or [])
    branch = state.get("current_branch") or "(detached HEAD)"
    in_progress = bool(state.get("in_merge_or_rebase"))

    chat_label = chat.title if chat and chat.title else "active chat"
    if entries:
        st.caption(
            f"Chat: **{chat_label}** \u00b7 branch `{branch}` \u00b7 "
            f"{len(entries)} file{'s' if len(entries) != 1 else ''} changed"
        )
    else:
        st.caption(f"Chat: **{chat_label}** \u00b7 branch `{branch}`")
        st.success(
            "Working tree clean. Nothing to push.",
            icon=":material/check_circle:",
        )
        if st.button("Close", key="diff_dlg_close_clean", width="stretch"):
            _close_diff_dialog()
            st.rerun()
        return

    working_dir_resolved = working_dir.resolve()
    counts = git_ops.summary_diff_counts(
        working_dir_resolved, [e.path for e in entries if not e.is_untracked]
    )

    # Each file renders as its own collapsed-by-default `st.expander`
    # via `_render_diff_file_section`, so the dialog body reads as a
    # compact list of file headers up front; the user expands the
    # ones they actually want to inspect. Expanders sit flush against
    # each other — no `st.divider()` between them — so the list stays
    # tight even with many files.
    for entry in entries:
        _render_diff_file_section(entry, counts, working_dir_resolved)

    if in_progress:
        st.divider()
        st.warning(
            "Resolve the in-progress "
            f"`{state.get('operation') or 'merge'}` before pushing. "
            "See the merge-conflict warning in the sidebar.",
            icon=":material/warning:",
        )

    st.divider()
    if st.button(
        "Close",
        icon=":material/close:",
        key="diff_dlg_close_btn",
        width="stretch",
    ):
        _close_diff_dialog()
        st.rerun()


# ---------------------------------------------------------------------------
# Bottom-of-chat git row: branch picker + new branch + fetch + Push
# ---------------------------------------------------------------------------
# Pinned just above the "Changes" button. Owns the per-chat branch
# switching, new-branch creation, fetch, and the one-click "generate
# commit message + commit + push" pipeline. Modals (`New branch`,
# `Publish branch`) are mounted from ``render()`` alongside
# ``_diff_dialog`` so they overlay the page rather than reflowing it.
#
# This is the **single entry point** for git push in the app — the
# sidebar's old per-chat git block was removed and the push dialog
# in ``streamlit_app.py`` was deleted. The merge-conflict warning still
# lives in the sidebar (so it remains visible from any page), but
# branch ops + push live exclusively here.

# Sentinel name used in the branch dropdown for "I want to create a new
# branch". Selecting it opens the new-branch dialog and reverts the
# selectbox to the live current branch so the sentinel never sticks.
# Plain text rather than a `:material/...:` token because Streamlit's
# selectbox option labels render as raw strings — Material icon tokens
# only render inside `icon=` props on buttons / sidebar items / similar,
# not inside selectbox options.
_NEW_BRANCH_SENTINEL = "+ New branch..."

# Strict branch-name validator. Mirrors a useful subset of git's
# refname rules: no whitespace, no double-dot, no leading/trailing
# slash, no leading dash, no characters git outright rejects. We don't
# try to be exhaustive — git itself will surface the precise error if
# anything we missed slips through.
_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def _validate_branch_name(name: str) -> str | None:
    """Return an error message for ``name`` or ``None`` if it's valid."""
    if not name:
        return "Branch name is required."
    if name.startswith("-"):
        return "Branch name cannot start with a dash."
    if name.startswith("/") or name.endswith("/"):
        return "Branch name cannot start or end with a slash."
    if ".." in name:
        return "Branch name cannot contain `..`."
    if not _BRANCH_NAME_RE.match(name):
        return (
            "Branch name can only contain letters, digits, "
            "`.`, `_`, `-`, and `/`."
        )
    return None


def _bump_chat_git_nonce() -> None:
    """Force the next cached git scan to re-read the working tree.

    Mirrors ``streamlit_app._bump_git_nonce`` but lives here so the
    chat page never imports the entry script (which would re-run
    ``main()``).
    """
    st.session_state.git_state_nonce = (
        int(st.session_state.get("git_state_nonce") or 0) + 1
    )


def _toast_git_error(prefix: str, exc: Exception) -> None:
    """Surface a git failure as a toast with a sane fallback message."""
    msg = getattr(exc, "stderr", None) or str(exc)
    st.toast(f"{prefix}: {msg}", icon=":material/error:")


def _on_chat_git_branch_change(chat_id: str) -> None:
    """Branch-selectbox callback inside the bottom-of-chat git row.

    Refuses on a dirty working tree (toast + revert), falls through to
    ``git_ops.checkout`` for both local and remote-tracking names
    (git resolves ``origin/feature`` -> a new local tracking branch
    automatically), bumps the global git nonce, and toasts on success.
    Selecting the sentinel "New branch..." entry opens the new-branch
    dialog instead of running a checkout.
    """
    ss = st.session_state
    select_key = f"chat_bottom_branch_select_{chat_id}"
    sentinel_key = f"_chat_bottom_branch_active_{chat_id}"
    chosen = ss.get(select_key)
    chat = ss.chats.get(chat_id)
    if not chosen or chat is None or not chat.working_dir:
        return

    if chosen == _NEW_BRANCH_SENTINEL:
        # Revert the selectbox so the sentinel doesn't stick around as
        # the displayed value while the dialog is open.
        ss[select_key] = ss.get(sentinel_key) or ss.get(select_key)
        ss.new_branch_dialog_open = True
        return

    if chosen == ss.get(sentinel_key):
        return

    working_dir = Path(chat.working_dir).expanduser().resolve()
    try:
        if git_ops.working_tree_dirty(working_dir):
            st.toast(
                "Cannot switch branches with uncommitted changes. "
                "Commit or stash first.",
                icon=":material/warning:",
            )
            ss[select_key] = ss.get(sentinel_key)
            return
        git_ops.checkout(working_dir, chosen)
    except GitError as e:
        _toast_git_error("git checkout failed", e)
        ss[select_key] = ss.get(sentinel_key)
        return
    ss[sentinel_key] = chosen
    _bump_chat_git_nonce()
    st.toast(f"Switched to `{chosen}`", icon=":material/check_circle:")


def _on_chat_git_fetch_clicked() -> None:
    """Fetch button callback: ``git fetch origin`` in the active chat's workdir."""
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        return
    working_dir = Path(chat.working_dir).expanduser().resolve()
    try:
        git_ops.fetch(working_dir)
    except GitError as e:
        _toast_git_error("git fetch failed", e)
        return
    _bump_chat_git_nonce()
    st.toast("Fetched from origin", icon=":material/cloud_download:")


def _on_chat_git_new_branch_clicked() -> None:
    """`+ New branch` button callback: open the modal."""
    st.session_state.new_branch_dialog_open = True


def _on_chat_git_push_clicked() -> None:
    """Push button callback: branch on upstream + open the right next step.

    - With an upstream: enqueue a one-click push (no PR) for the current
      script run to consume after the rerun. We can't run the pipeline
      directly inside this on-click callback because callbacks fire
      *before* the script runs; doing the pipeline here would block the
      callback for the duration of a network round-trip + an LLM call,
      and any exception would render through Streamlit's ugly
      uncaught-exception traceback. Routing through the
      ``pending_push_request`` queue keeps the pipeline inside
      ``render()`` where ``st.toast`` / ``st.rerun`` work normally.
    - Without an upstream (first push of this branch): open the
      "Publish branch" modal so the user can opt into a PR.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        return
    working_dir = Path(chat.working_dir).expanduser().resolve()
    branch = git_ops.current_branch(working_dir)
    if not branch:
        st.toast(
            "Cannot push from a detached HEAD. Check out a branch first.",
            icon=":material/error:",
        )
        return
    if git_ops.has_upstream(working_dir, branch):
        ss.pending_push_request = {"create_pr": False}
    else:
        ss.first_push_dialog_open = True


def _run_push_pipeline(
    working_dir: Path,
    branch: str,
    *,
    create_pr: bool,
) -> None:
    """One-click stage -> commit -> fetch -> rebase -> push (-> optional PR URL).

    Ports the body of the (former) ``streamlit_app._push_dialog`` action
    handler. Status surfaces via :func:`st.toast` calls so the user
    sees progress without us having to host a status pane in any
    dialog. On a ``pull --rebase`` conflict we set ``ss.merge_conflict``
    and the sidebar's existing merge-conflict warning takes over —
    same handoff the old dialog used.

    All dirty paths (tracked + untracked) are staged unconditionally;
    file-by-file selection is deliberately not exposed in v2 of this
    flow — see the plan's "Out of scope" section.
    """
    ss = st.session_state

    if not commit_ai.is_deepseek_available(ss.get("models")):
        st.toast(
            f"Push needs `{DEEPSEEK_MODEL}` for the commit message.",
            icon=":material/error:",
        )
        return

    try:
        entries = git_ops.status_entries(working_dir)
    except GitError as e:
        _toast_git_error("git status failed", e)
        return
    paths = [e.path for e in entries]
    if not paths:
        st.toast("Working tree is clean. Nothing to push.", icon=":material/info:")
        return

    in_progress, op = git_ops.is_in_merge_or_rebase(working_dir)
    if in_progress:
        st.toast(
            f"In-progress `{op or 'merge'}` — resolve before pushing "
            "(see the warning in the sidebar).",
            icon=":material/warning:",
        )
        return

    st.toast("Generating commit message...", icon=":material/auto_awesome:")
    try:
        commit_msg = commit_ai.generate_commit_message(
            ss.get("client"), working_dir, paths
        )
    except Exception as e:
        st.toast(f"DeepSeek failed: {e}", icon=":material/error:")
        return
    commit_msg = (commit_msg or "").strip()
    if not commit_msg:
        st.toast(
            "DeepSeek did not return a commit message.",
            icon=":material/error:",
        )
        return

    pr_title = ""
    pr_body = ""
    if create_pr:
        st.toast(
            "Generating pull request title and body...",
            icon=":material/auto_awesome:",
        )
        try:
            target = git_ops.default_branch(working_dir)
            pr_title, pr_body = commit_ai.generate_pr_description(
                ss.get("client"), working_dir, paths, branch, target
            )
        except Exception as e:
            st.toast(f"DeepSeek PR generation failed: {e}", icon=":material/warning:")

    try:
        git_ops.unstage_all(working_dir)
        git_ops.stage(working_dir, paths)
        git_ops.commit(working_dir, commit_msg)
        _bump_chat_git_nonce()
    except GitError as e:
        _toast_git_error("commit failed", e)
        return

    if git_ops.has_upstream(working_dir, branch):
        try:
            git_ops.fetch(working_dir)
        except GitError:
            # fetch failures aren't fatal; push will surface the same
            # condition with a clearer error if it matters.
            pass
        try:
            if git_ops.is_behind_upstream(working_dir):
                pull = git_ops.pull_rebase(working_dir)
                _bump_chat_git_nonce()
                if not pull.ok and pull.conflict:
                    ss.merge_conflict = {
                        "files": pull.files,
                        "operation": pull.operation,
                    }
                    st.toast(
                        "Rebase produced merge conflicts. "
                        "See the sidebar warning to resolve with DeepSeek.",
                        icon=":material/error:",
                    )
                    return
        except GitError as e:
            _toast_git_error("git pull --rebase failed", e)
            return

    push_result = git_ops.push(working_dir, branch=branch)
    _bump_chat_git_nonce()
    if not push_result.ok:
        st.toast(
            f"Push failed: {push_result.stderr.strip() or 'unknown error'}",
            icon=":material/error:",
        )
        return

    short_msg = commit_msg.splitlines()[0][:80]
    st.toast(
        f"Pushed `{branch}`: {short_msg}",
        icon=":material/check_circle:",
    )

    if create_pr:
        target = git_ops.default_branch(working_dir)
        url = git_ops.remote_compare_url(
            working_dir,
            branch,
            target,
            title=pr_title,
            body=pr_body,
        )
        if url is None:
            url = git_ops.extract_pr_link_from_stderr(push_result.stderr)
        if url:
            try:
                webbrowser.open(url)
            except Exception:
                pass
            st.toast(
                "Opened pull request draft in your browser.",
                icon=":material/link:",
            )
        else:
            st.toast(
                "Pushed, but the remote did not return a recognized "
                "PR-creation URL — open one manually on your hosting "
                "platform.",
                icon=":material/warning:",
            )


def _close_new_branch_dialog() -> None:
    st.session_state.new_branch_dialog_open = False


def _close_first_push_dialog() -> None:
    st.session_state.first_push_dialog_open = False


@st.dialog("New branch", width="small")
def _new_branch_dialog() -> None:
    """Modal for creating a new branch off HEAD.

    Uses ``git checkout -b`` so any uncommitted working-tree changes
    come along with the new branch — matches what most users mean
    when they say "let me put this on a new branch first".
    Validation errors render inline; the dialog stays open so the
    user can fix and retry.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        st.error(
            "No active chat / working directory.",
            icon=":material/error:",
        )
        if st.button("Close", key="new_branch_close_no_chat", width="stretch"):
            _close_new_branch_dialog()
            st.rerun()
        return

    st.caption(
        "Create a new local branch off the current HEAD. Any uncommitted "
        "changes in the working tree come along."
    )
    name = st.text_input(
        "Branch name",
        key="new_branch_name_input",
        placeholder="feature/my-change",
    )

    cols = st.columns([1, 1])
    cancel_clicked = cols[0].button(
        "Cancel",
        icon=":material/close:",
        key="new_branch_cancel_btn",
        width="stretch",
    )
    create_clicked = cols[1].button(
        "Create",
        icon=":material/check:",
        type="primary",
        key="new_branch_create_btn",
        width="stretch",
    )

    if cancel_clicked:
        ss.pop("new_branch_name_input", None)
        _close_new_branch_dialog()
        st.rerun()

    if not create_clicked:
        return

    name = (name or "").strip()
    err = _validate_branch_name(name)
    if err:
        st.error(err, icon=":material/error:")
        return

    working_dir = Path(chat.working_dir).expanduser().resolve()
    try:
        git_ops.create_branch(working_dir, name, checkout=True)
    except GitError as e:
        st.error(e.stderr or str(e), icon=":material/error:")
        return

    _bump_chat_git_nonce()
    st.toast(f"Created branch `{name}`", icon=":material/check_circle:")
    ss.pop("new_branch_name_input", None)
    _close_new_branch_dialog()
    st.rerun()


@st.dialog("Publish branch", width="medium")
def _first_push_dialog() -> None:
    """Modal shown on the first push of a branch (no upstream yet).

    Two-option radio: just push the branch, or push and open a
    pre-filled pull-request draft. Selecting the PR option triggers
    an extra DeepSeek call to draft a title + body, then opens the
    GitHub compare URL with both pre-filled — the user can review
    and edit before submitting on the platform side.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        st.error("No active chat / working directory.", icon=":material/error:")
        if st.button("Close", key="first_push_close_no_chat", width="stretch"):
            _close_first_push_dialog()
            st.rerun()
        return

    working_dir = Path(chat.working_dir).expanduser().resolve()
    branch = git_ops.current_branch(working_dir) or "(unknown)"

    st.caption(
        f"Branch `{branch}` does not have an upstream yet. "
        "Pushing will set the upstream to `origin/{0}` so future pushes "
        "go through with one click.".format(branch)
    )

    choice = st.radio(
        "What would you like to do?",
        options=["Just push the branch", "Push and open a pull request"],
        key="first_push_choice",
        index=1,
    )

    cols = st.columns([1, 1])
    cancel_clicked = cols[0].button(
        "Cancel",
        icon=":material/close:",
        key="first_push_cancel_btn",
        width="stretch",
    )
    confirm_clicked = cols[1].button(
        "Confirm",
        icon=":material/upload:",
        type="primary",
        key="first_push_confirm_btn",
        width="stretch",
    )

    if cancel_clicked:
        _close_first_push_dialog()
        st.rerun()

    if not confirm_clicked:
        return

    create_pr = choice == "Push and open a pull request"
    ss.pending_push_request = {"create_pr": create_pr}
    _close_first_push_dialog()
    st.rerun()


def _render_chat_git_row() -> None:
    """The compact bottom-of-chat git row: branch / new / fetch / push.

    Hidden entirely when the active chat has no workdir, when the
    workdir is not a git repo, or when git isn't installed — the
    Changes button below already degrades the same way, so there's
    nothing useful for this row to do in those cases.

    On a detached HEAD we render a single caption explaining why the
    full controls are missing instead of an empty row.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    working_dir_str = (chat.working_dir if chat else "") or ss.working_dir or ""
    if not working_dir_str:
        return
    if not git_ops.is_git_installed():
        return

    state = _diff_git_state(working_dir_str)
    if not state.get("in_repo"):
        return

    branch = state.get("current_branch")
    if not branch:
        st.caption(
            ":material/warning: Detached HEAD. Switch to a branch via the "
            "command line to enable branch operations and push."
        )
        return

    # Build the dropdown options. Local branches first, then any
    # remote-only ones (suffixed `(remote)` so users see at a glance
    # which entries are remote-tracking — selectbox option labels
    # render as raw strings, so Material icon tokens are not an
    # option), then the sentinel "+ New branch..." entry.
    local_branches = list(state.get("branches") or [])
    remote_branches = list(state.get("remote_branches") or [])
    remote_only = [
        rb for rb in remote_branches
        if rb.split("/", 1)[-1] not in local_branches
    ]
    options: list[str] = []
    if branch not in local_branches:
        options.append(branch)
    options.extend(local_branches)
    options.extend(remote_only)
    options.append(_NEW_BRANCH_SENTINEL)

    select_key = f"chat_bottom_branch_select_{chat.id}"
    sentinel_key = f"_chat_bottom_branch_active_{chat.id}"
    ss[sentinel_key] = branch
    ss[select_key] = branch if branch in options else options[0]

    in_progress = bool(state.get("in_merge_or_rebase"))
    dirty = bool(state.get("status"))
    deepseek_ok = commit_ai.is_deepseek_available(ss.get("models"))

    cols = st.columns([4, 1, 1, 4], vertical_alignment="bottom")

    with cols[0]:
        st.selectbox(
            "Branch",
            options=options,
            key=select_key,
            on_change=_on_chat_git_branch_change,
            args=(chat.id,),
            label_visibility="collapsed",
            format_func=lambda b: (
                b if b == _NEW_BRANCH_SENTINEL
                # Selectbox option labels render as raw strings — Material
                # tokens don't work here, so use a plain text marker for
                # remote-only refs.
                else (f"{b}  (remote)" if b in remote_only else b)
            ),
            help=(
                "Switch this chat's working-directory branch. Pick a "
                "`(remote)` entry to check out a remote-only branch as a "
                "new local tracking branch. Switching is refused with a "
                "toast if you have uncommitted changes."
            ),
        )

    with cols[1]:
        st.button(
            "",
            icon=":material/add:",
            key=f"chat_bottom_new_branch_{chat.id}",
            help="Create a new branch off the current HEAD.",
            on_click=_on_chat_git_new_branch_clicked,
            width="stretch",
        )

    with cols[2]:
        st.button(
            "",
            icon=":material/cloud_download:",
            key=f"chat_bottom_fetch_{chat.id}",
            help="Fetch upstream branches from origin.",
            on_click=_on_chat_git_fetch_clicked,
            width="stretch",
        )

    with cols[3]:
        push_disabled = (not dirty) or in_progress or (not deepseek_ok)
        if not dirty:
            push_help = "Working tree is clean. Nothing to push."
        elif in_progress:
            push_help = (
                f"In-progress `{state.get('operation') or 'merge'}` — "
                "resolve before pushing."
            )
        elif not deepseek_ok:
            push_help = (
                f"Push needs `{DEEPSEEK_MODEL}` to draft the commit "
                "message; that model isn't available on this account."
            )
        else:
            push_help = (
                "Generate a commit message with DeepSeek and push to "
                "origin. First push of a branch asks whether to also "
                "open a pull request."
            )
        st.button(
            "Push",
            icon=":material/upload:",
            type="primary",
            key=f"chat_bottom_push_{chat.id}",
            on_click=_on_chat_git_push_clicked,
            disabled=push_disabled,
            width="stretch",
            help=push_help,
        )


def _drain_pending_push_request() -> None:
    """Run any queued push pipeline. Called once per ``render()``.

    Push callbacks (the bottom-row Push button + the Publish-branch
    modal Confirm button) drop their intent into
    ``ss.pending_push_request`` instead of running the pipeline
    inline; this drain step turns that into a single in-render call
    so :func:`st.toast` / :func:`st.rerun` work normally and any
    DeepSeek / git failure surfaces through the toast layer rather
    than a Streamlit uncaught-exception trace.
    """
    ss = st.session_state
    req = ss.pop("pending_push_request", None)
    if not req:
        return
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        return
    working_dir = Path(chat.working_dir).expanduser().resolve()
    branch = git_ops.current_branch(working_dir)
    if not branch:
        st.toast(
            "Cannot push from a detached HEAD.",
            icon=":material/error:",
        )
        return
    _run_push_pipeline(working_dir, branch, create_pr=bool(req.get("create_pr")))


def _render_chat_diff_button() -> None:
    """Render the pinned "Changes" button above the chat input.

    Hidden entirely when the chat's working directory is empty / not a
    git repo / has no changes / git is missing — so the bottom controls
    stay compact when there's nothing to look at.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    working_dir_str = (chat.working_dir if chat else "") or ss.working_dir or ""
    if not working_dir_str:
        return
    if not git_ops.is_git_installed():
        return

    state = _diff_git_state(working_dir_str)
    if not state.get("in_repo"):
        return
    entries: list[git_ops.StatusEntry] = list(state.get("status") or [])
    if not entries:
        return

    working_dir_resolved = Path(working_dir_str).expanduser().resolve()
    counts = git_ops.summary_diff_counts(
        working_dir_resolved, [e.path for e in entries if not e.is_untracked]
    )
    total_adds = 0
    total_dels = 0
    for entry in entries:
        adds, dels = counts.get(entry.path, (0, 0))
        if entry.is_untracked and adds == 0:
            adds = git_ops.untracked_line_count(working_dir_resolved, entry.path)
        total_adds += adds
        total_dels += dels

    n = len(entries)
    label = (
        f"Changes  +{total_adds} \u2212{total_dels}  "
        f"\u00b7  {n} file{'s' if n != 1 else ''}"
    )
    st.button(
        label,
        icon=":material/difference:",
        key="chat_diff_open_btn",
        width="stretch",
        on_click=_open_diff_dialog,
        help="View the live working-tree diff in an overlay.",
    )


def _render_active_chat_static(chat: chats.Chat) -> None:
    """One-shot renderer for an idle / errored active chat.

    Draws every persisted turn out of ``chat.ui_turns`` plus a final
    status caption when the chat ended in error. Runs on a fresh
    Streamlit script run (no fragment), so reading under the lock is
    safe. Renders a helpful "this chat is empty" hint when the chat
    has no turns yet so brand-new chats don't look broken.
    """
    with chat._lock:
        turns = list(chat.ui_turns)
        status = chat.status
        error_message = chat.error_message
    if not turns:
        _render_empty_chat_hint(chat)
        return
    for turn in turns:
        if turn.get("role") == "user":
            _render_user_turn(turn)
        else:
            _render_assistant_turn(turn)
    if status == chats.STATUS_ERROR and error_message:
        st.error(f"Last turn failed: {error_message}", icon=":material/error:")


@st.fragment(run_every="0.25s")
def _render_active_chat_live(chat_id: str) -> None:
    """Live re-renderer for a running active chat.

    Uses ``@st.fragment(run_every="0.25s")`` so the chat panel polls
    itself while a turn is in flight without re-running the whole
    page. We re-look up the chat by id every tick (not closing over
    the chat object) so a delete/archive that fires on the script
    thread doesn't leave a dangling reference.

    When the chat's status flips out of ``"running"``, the fragment
    triggers a full rerun so the static renderer takes over (and the
    poll stops).
    """
    chat = st.session_state.chats.get(chat_id)
    if chat is None:
        return
    with chat._lock:
        turns = list(chat.ui_turns)
        partial = chat.partial_text
        status = chat.status
        model = chat.model

    for turn in turns:
        if turn.get("role") == "user":
            _render_user_turn(turn)
        else:
            _render_assistant_turn(turn)

    if partial:
        with st.chat_message("assistant"):
            st.markdown(partial)

    if status == chats.STATUS_RUNNING:
        short_model = model.split("/")[-1] if model else ""
        if short_model:
            st.caption(f":material/auto_awesome: Thinking with `{short_model}`...")
        else:
            st.caption(":material/auto_awesome: Thinking...")
    else:
        # Status flipped out of running; trigger a full rerun so the
        # static renderer takes over and this fragment stops polling.
        st.rerun()


def _sync_active_chat_settings(chat: chats.Chat) -> None:
    """Copy the active chat's settings into the flat ``ss.*`` dropdown keys.

    Streamlit's ``st.selectbox(..., key="model")`` pattern makes
    Streamlit the single owner of the value (good — it's the
    documented remedy for the "user picks A, sees B" footgun). To
    swap a chat's settings into those dropdowns when the user
    switches chats, we have to mutate ``ss.*`` *before* the dropdown
    renders for the new chat. We track the most-recently-synced chat
    id in ``ss._last_active_chat_id`` so we only do this on actual
    switches (not on every rerun).
    """
    ss = st.session_state
    if ss.get("_last_active_chat_id") == chat.id:
        return
    if chat.model and chat.model in (ss.models or []):
        ss.model = chat.model
    if chat.mode in ("agent", "ask"):
        ss.mode = chat.mode
    if chat.working_dir:
        ss.working_dir = chat.working_dir
    ss._last_active_chat_id = chat.id


def _render_welcome_steps() -> None:
    """The "Get started in 3 steps" three-card row.

    Shared visual building block for the chat-page zero state. The
    Docs page renders an equivalent block (see
    :func:`app_pages.docs._render_get_started`); the two are kept
    separate copies — they're each four lines of code, and avoiding
    a cross-page import keeps the page module a leaf in the import
    graph.
    """
    cols = st.columns(3, border=True)
    with cols[0]:
        st.markdown(":material/key: **1. Add your W&B API key**")
        st.caption(
            "Open the **Settings** tab in the top nav and paste a key "
            "from [wandb.ai/settings](https://wandb.ai/settings)."
        )
    with cols[1]:
        st.markdown(":material/folder_open: **2. Pick a folder**")
        st.caption(
            "Below the chat box, choose a project folder on your "
            "computer. The agent only touches files inside that folder."
        )
    with cols[2]:
        st.markdown(":material/chat: **3. Send your first message**")
        st.caption(
            "Ask the agent to read your code, fix a bug, or write a "
            "new feature. You'll see every step it takes."
        )


def _render_not_ready(ss: Any) -> None:
    """Zero-state UI when ``ss.client`` / ``ss.model`` aren't populated.

    Two genuine reasons the chat page reaches this branch:

    - The user hasn't connected yet (no saved API key + no in-session
      key). Direct them to the Settings page.
    - The startup auto-connect ran but failed (expired key, network
      blip). Surface the error and offer a one-click **Reconnect**
      button so the user can retry without leaving this page.

    The visual is a single bordered welcome card with a friendly
    headline, a one-paragraph plain-English pitch, a three-card "Get
    started" row, and (only when relevant) a status banner + Reconnect
    button. Copy throughout follows the same plain-language voice as
    the in-app **Docs** tab.

    Note: this branch should NOT fire just because the user navigated
    away from the chat page and back — the chat page's model/mode
    dropdowns use the dual-key pattern (widget keys ``_chat_*_input``
    + canonical ``ss.mode`` / ``ss.model``), so Streamlit's "strip
    widget state on unmount" behaviour can't wipe the canonical keys.
    """
    import actions

    has_saved_key = bool((ss.api_key or "").strip())
    error = ss.get("connect_error")

    with st.container(border=True):
        st.markdown(":material/smart_toy: ### Welcome to the W&B Coding Agent")
        st.markdown(
            "This is a coding assistant powered by AI models you choose. "
            "Point it at a folder on your computer, ask a question, and "
            "it will read your code, suggest changes, run commands, and "
            "show you exactly what it did."
        )

        st.markdown("**Get started in 3 steps**")
        _render_welcome_steps()

        if not has_saved_key:
            st.markdown(
                "Open the **Settings** tab in the top nav, paste your "
                "W&B API key, and click **Connect** to get going. New to "
                "the app? The **Docs** tab walks through every screen in "
                "plain English."
            )
            return

        # Past this point the user has a saved key — either auto-connect
        # is still in flight, or it ran and hit an error. Surface
        # whichever applies and offer a one-click Reconnect.
        if error:
            st.warning(
                f"We couldn't connect with your saved API key. {error} "
                "Try **Reconnect**, or update your key in the Settings tab.",
                icon=":material/sync_problem:",
            )
        else:
            st.info(
                "Connecting to W&B Inference with your saved API key...",
                icon=":material/sensors:",
            )
        st.button(
            "Reconnect",
            icon=":material/link:",
            type="primary",
            on_click=actions.on_connect,
            help="Retry the W&B Inference connection from this page.",
        )
        st.caption(
            "Or open the **Settings** tab to update your API key, switch "
            "projects, or sign out. The **Docs** tab has more help."
        )


def _render_empty_chat_hint(chat: chats.Chat) -> None:
    """Caption shown inside an empty chat to make it obvious it's just empty.

    Without this, a freshly-created ``+ New chat`` row activates a
    chat whose ``ui_turns`` list is empty and the conversation area
    renders nothing — which can read as "broken / disconnected"
    rather than "empty conversation, send a message". The caption is
    suppressed once the chat has any user or assistant content.
    """
    if chats.has_content(chat):
        return
    st.caption(
        ":material/chat_bubble_outline: This chat is empty. "
        "Send a message below to start the conversation."
    )


def render() -> None:
    """Page body for the Chat page (called by ``st.navigation`` -> ``page.run()``)."""
    ss = st.session_state

    # Look up the active chat *before* rendering the page header so the
    # title can mirror the chat the user is actually looking at. Once a
    # chat earns an AI-generated title (see ``chats.generate_title``),
    # surfacing that title in the page header gives the user a clear
    # at-a-glance sense of which thread they're in — especially when
    # bouncing between chats from the sidebar. Brand-new / blank
    # placeholder chats keep their :data:`chats.DEFAULT_TITLE` ("New
    # chat"), and we render that *same* string as the page header so
    # the page title and the sidebar row are visually consistent
    # rather than the page presenting a separate generic app name.
    # The chat-is-None defensive branch (zero-state before the seed
    # chat is created, etc.) also falls back to ``DEFAULT_TITLE`` so
    # the user always sees a coherent page title. The onboarding
    # caption is only rendered alongside the default-titled state so
    # first-run users still get a clear "what is this app" pitch
    # above the empty conversation area, while users with real
    # conversations don't see the same pitch repeated above every
    # thread.
    chat: chats.Chat | None = ss.chats.get(ss.active_chat_id)
    page_title = chat.title if (chat and chat.title) else chats.DEFAULT_TITLE
    st.title(page_title)
    if page_title == chats.DEFAULT_TITLE:
        st.caption(
            "Powered by [W&B Inference](https://docs.wandb.ai/inference). "
            "Point it at a working directory and pick a mode and model below the chat, "
            "and ask it to read or modify your code."
        )

    ready = ss.client is not None and ss.model is not None
    if not ready:
        _render_not_ready(ss)
        return

    if chat is None:
        st.info(
            "No active chat. Click **New chat** in the sidebar to start one.",
            icon=":material/chat_bubble_outline:",
        )
        return

    _sync_active_chat_settings(chat)

    # Scrollable chat history. The fixed pixel ``height`` flips
    # Streamlit into "fixed-height container + internal scroll +
    # autoscroll on new ``st.chat_message``" mode, which is what
    # keeps the chat input, workdir picker, and model selector below
    # this container visible at the bottom of the viewport on a
    # typical desktop window instead of being pushed below the fold
    # by a long conversation. See ``_CHAT_HISTORY_HEIGHT_PX`` for why
    # we use a fixed integer and how to retune it.
    conversation_area = st.container(
        height=_CHAT_HISTORY_HEIGHT_PX,
        border=False,
    )
    with conversation_area:
        if chat.status == chats.STATUS_RUNNING:
            _render_active_chat_live(chat.id)
        else:
            _render_active_chat_static(chat)

    wd_ok = Path(ss.working_dir).expanduser().is_dir() if ss.working_dir else False

    # Drain any push pipeline that a callback queued before this rerun.
    # Has to run after the active-chat sync above (so it sees the
    # right working_dir) but before the bottom controls render so the
    # toast for "pushed!" / "rebase conflict" appears on the same
    # rerun rather than the next one.
    _drain_pending_push_request()

    # Compact git row pinned just above the Changes button: branch
    # picker + new-branch + fetch + Push. Hidden entirely when the
    # workdir isn't a git repo / git is missing — same degradation the
    # Changes button has, so the bottom controls stay tight when
    # there's nothing useful to show.
    _render_chat_git_row()

    # "Changes" button sits directly above the chat input. Hidden when
    # the working tree is clean / not a git repo so it doesn't take up
    # space when there's nothing to view. Clicking the button opens
    # ``_diff_dialog`` (a modal overlay) so opening the diff never
    # reflows the chat input or any other control below.
    _render_chat_diff_button()

    # Chat input is wrapped in its own ``st.container()`` so Streamlit
    # renders it inline (rather than docking it to the viewport bottom
    # via the default top-level chat-input behaviour) — the *whole*
    # control stack below the chat history needs to stay together,
    # not the chat input alone.
    with st.container():
        prompt = st.chat_input(
            "Ask the agent to read or modify your code...",
            disabled=(not wd_ok) or chat.status == chats.STATUS_RUNNING,
        )

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

    # Handoff from the sidebar's "Resolve with DeepSeek" button: the
    # payload (synthesized prompt + override model) is set in
    # ``streamlit_app._request_conflict_resolution`` and consumed here so
    # the merge-conflict resolution turn renders into the same chat
    # transcript as everything else. We drain the key before running the
    # turn so a transient script-runner error can't loop on the same
    # payload forever.
    pending = st.session_state.pop("pending_conflict_resolution", None)
    if pending and wd_ok and not prompt and chat.status != chats.STATUS_RUNNING:
        _start_turn(pending["prompt"], override_model=pending.get("model"))
        st.rerun()

    if prompt and wd_ok and chat.status != chats.STATUS_RUNNING:
        _start_turn(prompt)
        st.rerun()

    # Mount the diff dialog last so its modal overlay sits above all
    # the page content. Gated by ``ss.diff_dialog_open`` (set by the
    # "Changes" button or by the sidebar's per-file deep-link); the
    # dialog itself clears the flag when the user closes it.
    if ss.get("diff_dialog_open"):
        _diff_dialog()

    # New-branch + first-push modals are mounted next to the diff dialog
    # so all three live at the same render level. Each is gated by its
    # own ``ss.*_dialog_open`` flag flipped from the bottom-row git
    # callbacks (and cleared by the dialog itself on Cancel / Confirm).
    if ss.get("new_branch_dialog_open"):
        _new_branch_dialog()
    if ss.get("first_push_dialog_open"):
        _first_push_dialog()


# Streamlit's st.navigation runs the page module top-to-bottom, so we call
# render() at module scope. ``streamlit_app.py`` initializes session state
# before navigation, so ss.* keys are present when this runs.
render()
