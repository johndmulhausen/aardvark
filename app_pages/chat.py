"""Chat page: the existing tool-calling chat UI, with token-usage capture.

Moved out of ``streamlit_app.py`` so the entry point can host
``st.navigation`` between the chat experience and the new Usage dashboard
page. The contract documented in ``AGENTS.md`` ("conversation history
renders inside a forward-declared ``conversation_area`` above the chat
input; workdir/mode/model controls render below the chat input") is
preserved verbatim — just relocated to this module.

State of the world this page expects (initialized in ``streamlit_app.py``):

- ``ss.client`` is a connected OpenAI-pointed-at-W&B-Inference client.
- ``ss.model`` is a non-empty model id from ``ss.models``.
- ``ss.working_dir`` is a directory path (validity is enforced here).
- ``ss.messages`` and ``ss.ui_turns`` are the conversation logs.

Side effects:

- After each successful turn, the per-turn token totals are appended to
  ``~/.wb_coding_agent/usage.jsonl`` via :func:`usage.record_usage` so the
  Usage dashboard picks them up. Cost is computed via :func:`usage.compute_cost`
  off the pricing table in ``models.py``.
- If a verified GitHub identity is present in ``ss.github_identity`` and
  ``ss.working_dir`` is a git repo, ``account.apply_git_identity`` is
  called once per session-and-workdir (tracked in ``ss.git_identity_applied``)
  so commits the agent makes via ``run_shell`` are authored as that user.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import streamlit as st

import account
import git_ops
import mcp_servers
import project_context
import usage as usage_log
from agent import run_agent_turn
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


def _render_usage_event(event: dict[str, Any]) -> None:
    """Render the per-turn usage caption: tokens + cost."""
    total = int(event.get("total_tokens") or 0)
    cost = event.get("cost_usd")
    parts = [f":material/data_usage: {usage_log.format_tokens(total)} tokens"]
    if cost is not None:
        parts.append(usage_log.format_cost(cost))
    rounds = int(event.get("rounds") or 0)
    if rounds > 1:
        parts.append(f"{rounds} rounds")
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
            elif ev["type"] == "turn_usage":
                _render_usage_event(ev)
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


def _run_turn(prompt: str, *, override_model: str | None = None) -> None:
    """Drive one chat turn: stream events, render live, persist usage.

    ``override_model`` lets callers (notably the merge-conflict
    resolution handoff from ``streamlit_app.py``) pin a specific model
    for this turn without having to mutate ``ss.model``. When ``None``
    we use the user's selected model from the model dropdown as before.
    """
    ss = st.session_state
    ss.messages.append({"role": "user", "content": prompt})
    ss.ui_turns.append({"role": "user", "content": prompt})

    _render_user_turn({"role": "user", "content": prompt})

    assistant_turn: dict[str, Any] = {"role": "assistant", "events": []}
    ss.ui_turns.append(assistant_turn)

    working_dir = Path(ss.working_dir).expanduser().resolve()
    _maybe_apply_git_identity(working_dir)
    turn_model = override_model or ss.model
    short_model = turn_model.split("/")[-1] if turn_model else ""

    # Per-turn usage accumulator. The agent yields one ``usage`` event per
    # inference round (a turn with N tool-calling rounds has N rounds); we
    # sum them here for the per-turn footer caption and the persisted log
    # entry.
    turn_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "rounds": 0,
    }
    turn_started = time.monotonic()

    with st.chat_message("assistant"):
        live_container = st.container()
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
        text_ph: Any = None
        text_buf = ""

        try:
            events_iter = run_agent_turn(
                client=ss.client,
                model=turn_model,
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
                    _show_thinking()
                elif etype == "usage":
                    # One per inference round; accumulate, don't persist
                    # individually. The aggregated ``turn_usage`` event we
                    # build below is what gets logged + replayed.
                    turn_usage["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
                    turn_usage["completion_tokens"] += int(event.get("completion_tokens") or 0)
                    turn_usage["total_tokens"] += int(event.get("total_tokens") or 0)
                    turn_usage["rounds"] += 1
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

        # Persist + render the per-turn usage summary if any inference call
        # produced token counts. Skip when zero (means the turn errored
        # before any model call landed).
        if turn_usage["total_tokens"] > 0 and turn_model:
            duration = time.monotonic() - turn_started
            entry = usage_log.build_entry(
                model=turn_model,
                prompt_tokens=turn_usage["prompt_tokens"],
                completion_tokens=turn_usage["completion_tokens"],
                total_tokens=turn_usage["total_tokens"],
                rounds=turn_usage["rounds"],
                duration_seconds=duration,
                mode=ss.mode,
            )
            usage_log.record_usage(entry)
            session_total = ss.setdefault(
                "usage_session_total",
                {"total_tokens": 0, "cost_usd": 0.0, "turns": 0},
            )
            session_total["total_tokens"] += entry["total_tokens"]
            cost = entry.get("cost_usd")
            if isinstance(cost, (int, float)):
                session_total["cost_usd"] += float(cost)
            session_total["turns"] += 1

            ui_event = {
                "type": "turn_usage",
                "model": entry["model"],
                "prompt_tokens": entry["prompt_tokens"],
                "completion_tokens": entry["completion_tokens"],
                "total_tokens": entry["total_tokens"],
                "cost_usd": entry.get("cost_usd"),
                "rounds": entry["rounds"],
                "duration_seconds": entry.get("duration_seconds"),
            }
            assistant_turn["events"].append(ui_event)
            with live_container:
                _render_usage_event(ui_event)


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
    chosen = st.session_state.get("wd_select")
    if chosen:
        # Recents and folder-picker live in ``actions.py`` so pages can
        # import them without dragging the entry-point script back through
        # Python's import machinery (Streamlit loads the entry as
        # ``__main__``; importing it from a sub-page re-executes
        # ``main()`` and re-renders the sidebar, blowing up with
        # duplicate-widget-key errors).
        import actions
        st.session_state.working_dir = chosen
        actions.record_recent_dir(chosen)


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


def _on_branch_change() -> None:
    """Selectbox callback: ``git checkout`` to the chosen branch.

    Refuses (with a toast + revert) when the working tree is dirty —
    silent loss of in-flight work would be the opposite of what the
    branch switcher is for. Bumps the git-state nonce so the file
    changes panel re-scans against the new branch on the next rerun.
    """
    ss = st.session_state
    chosen = ss.get("git_branch_select")
    if not chosen or chosen == ss.get("_git_active_branch"):
        return
    working_dir = Path(ss.working_dir).expanduser().resolve()
    try:
        if git_ops.working_tree_dirty(working_dir):
            st.toast(
                "Cannot switch branches with uncommitted changes. "
                "Commit or stash first.",
                icon=":material/warning:",
            )
            ss.git_branch_select = ss.get("_git_active_branch")
            return
        git_ops.checkout(working_dir, chosen)
    except GitError as e:
        st.toast(f"git checkout failed: {e.stderr}", icon=":material/error:")
        ss.git_branch_select = ss.get("_git_active_branch")
        return
    ss._git_active_branch = chosen
    ss.git_state_nonce = int(ss.get("git_state_nonce") or 0) + 1
    st.toast(f"Switched to `{chosen}`", icon=":material/check_circle:")


def _render_workdir_controls() -> None:
    ss = st.session_state

    wd_options: list[str] = []
    if ss.working_dir:
        wd_options.append(ss.working_dir)
    for d in ss.recent_dirs:
        if d not in wd_options:
            wd_options.append(d)

    # Probe the working dir for git state once per render. When it's a
    # repo we widen the row to include a branch switcher; otherwise we
    # keep the prior 3-column layout (workdir + browse + new-project).
    working_dir_path = (
        Path(ss.working_dir).expanduser() if ss.working_dir else None
    )
    git_repo = bool(
        working_dir_path
        and working_dir_path.is_dir()
        and git_ops.is_git_repo(working_dir_path)
    )

    if git_repo:
        wd_cols = st.columns([6, 4, 1, 1], vertical_alignment="bottom")
    else:
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

    if git_repo:
        with wd_cols[1]:
            try:
                branches = git_ops.list_branches(working_dir_path)
                current = git_ops.current_branch(working_dir_path)
            except GitError:
                branches = []
                current = None
            options = list(branches)
            if current and current not in options:
                options.insert(0, current)
            if not options:
                st.selectbox(
                    "Branch",
                    options=["(detached HEAD)"],
                    disabled=True,
                )
            else:
                idx = options.index(current) if current in options else 0
                # Track the active branch so ``_on_branch_change`` can
                # detect the actual delta (and revert on dirty-tree).
                ss._git_active_branch = current
                st.selectbox(
                    "Branch",
                    options=options,
                    index=idx,
                    key="git_branch_select",
                    on_change=_on_branch_change,
                    help=(
                        "Switch the local branch checked out in the "
                        "working directory. Refuses with a toast if "
                        "there are uncommitted changes."
                    ),
                )
        browse_col = wd_cols[2]
        new_proj_col = wd_cols[3]
    else:
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

    cols = st.columns([1, 2, 1], vertical_alignment="bottom")
    with cols[0]:
        st.selectbox(
            "Mode",
            options=["agent", "ask"],
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


def render() -> None:
    """Page body for the Chat page (called by ``st.navigation`` -> ``page.run()``)."""
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
            "Open the **Settings** tab in the top nav, paste your W&B API key, "
            "and click **Connect** to get started.",
            icon=":material/settings:",
        )
        return

    conversation_area = st.container()
    with conversation_area:
        _render_history()

    wd_ok = Path(ss.working_dir).expanduser().is_dir()

    with st.container():
        prompt = st.chat_input(
            "Ask the agent to read or modify your code...",
            disabled=not wd_ok,
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
    if pending and wd_ok and not prompt:
        with conversation_area:
            _run_turn(pending["prompt"], override_model=pending.get("model"))
        st.rerun()

    if prompt and wd_ok:
        with conversation_area:
            _run_turn(prompt)
        st.rerun()


# Streamlit's st.navigation runs the page module top-to-bottom, so we call
# render() at module scope. ``streamlit_app.py`` initializes session state
# before navigation, so ss.* keys are present when this runs.
render()
