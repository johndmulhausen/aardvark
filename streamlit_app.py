"""Streamlit entry point for the W&B Inference code editing agent.

Runs as a multi-page app via ``st.navigation``: ``Chat`` (the existing
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

import json
import os
import webbrowser
from pathlib import Path
from typing import Any

import streamlit as st

import account
import git_ops
from actions import (
    load_recent_dirs as _load_recent_dirs,
    on_connect as _auto_on_connect,
    theme_detected as _theme_detected,
)
from agent import DEEPSEEK_MODEL, generate_text
from git_ops import GitError
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
    ss.setdefault("messages", [])
    ss.setdefault("ui_turns", [])
    ss.setdefault("connect_error", None)
    ss.setdefault("weave_project", None)
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
    # cached repo scan refreshes on the next rerun. ``push_dialog_open``
    # gates the push modal. ``merge_conflict`` is set to
    # ``{"files": [...], "operation": "rebase"|"merge"}`` after a conflict
    # during the push flow's ``pull --rebase`` and drives the sidebar
    # warning + "Resolve with DeepSeek" button. ``pending_conflict_resolution``
    # is the cross-page handoff: the sidebar (in this module) sets it; the
    # chat page (in ``app_pages/chat.py``) drains it and runs an agent turn
    # with ``override_model=DEEPSEEK_MODEL``.
    ss.setdefault("git_state_nonce", 0)
    ss.setdefault("push_dialog_open", False)
    ss.setdefault("push_dialog_state", {})
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
# File changes panel
# ---------------------------------------------------------------------------
# Two render paths share the panel: when the working directory is a git
# working tree we drive the panel from ``git diff HEAD`` so the user sees
# the live filesystem state regardless of which agent turn produced each
# change (or whether the change came from outside the agent at all). When
# the workdir is not a git repo we fall back to aggregating successful
# ``write_file``/``edit_file`` tool results from the chat history — the
# legacy behavior, which gives at least *some* visibility on non-git
# workdirs.
def _count_diff_lines(diff: str) -> tuple[int, int]:
    """Count ``+`` and ``-`` lines in a unified diff (excluding headers)."""
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


def _render_tool_result_changes() -> None:
    """Legacy non-git fallback for the file-changes panel.

    Aggregates successful ``write_file`` / ``edit_file`` tool results into
    one entry per path, so users on non-git workdirs still see what the
    agent touched and can review the diff inline.
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


def _entry_icon(entry: git_ops.StatusEntry) -> str:
    """Material icon name for a :class:`git_ops.StatusEntry` row."""
    if entry.is_untracked:
        return ":material/add_circle:"
    if entry.is_deleted:
        return ":material/remove_circle:"
    if entry.is_renamed:
        return ":material/swap_horiz:"
    if entry.staged_status == "A":
        return ":material/add_circle:"
    return ":material/edit_note:"


def _entry_state_label(entry: git_ops.StatusEntry) -> str:
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


def _git_diff_for_entry(working_dir: Path, entry: git_ops.StatusEntry) -> str:
    """Return a unified diff string for a status entry, robust to errors."""
    try:
        return git_ops.diff_for_path(working_dir, entry.path, untracked=entry.is_untracked)
    except GitError as e:
        return f"(git diff failed: {e.stderr})"


def _render_git_file_changes(state: dict[str, Any]) -> None:
    """Render the git-driven file-changes panel.

    Uses ``git status`` + ``git diff HEAD`` as the source of truth so the
    panel reflects the actual filesystem regardless of which agent turn
    (or external tool) produced each change. The footer hosts the
    primary **Push changes** button — it's only enabled when the working
    tree is dirty *and* the repo isn't mid-rebase/merge (the merge-
    conflict warning takes over in that state).
    """
    entries: list[git_ops.StatusEntry] = list(state.get("status") or [])
    branch = state.get("current_branch") or "(detached HEAD)"
    in_progress = bool(state.get("in_merge_or_rebase"))

    if not entries:
        # Quiet repo. We still surface a tiny header so the user can see
        # the branch context (and so the panel doesn't disappear right
        # after they commit + push, leaving no breadcrumb).
        with st.expander(
            f"File changes \u00b7 clean on `{branch}`",
            icon=":material/check_circle:",
            expanded=False,
        ):
            st.caption("Working tree clean. Nothing to push.")
        return

    working_dir = Path(st.session_state.working_dir).expanduser().resolve()
    counts = git_ops.summary_diff_counts(working_dir, [e.path for e in entries if not e.is_untracked])

    label = (
        f"File changes \u00b7 {len(entries)} file"
        f"{'s' if len(entries) != 1 else ''} on `{branch}`"
    )
    with st.expander(label, icon=":material/edit_note:", expanded=True):
        for i, entry in enumerate(entries):
            if i > 0:
                st.divider()
            adds, dels = counts.get(entry.path, (0, 0))
            if entry.is_untracked and adds == 0:
                # ``git diff`` can't see untracked files; approximate the +
                # count from the file's current line count so the chip is
                # never blank for a brand-new file.
                adds = git_ops.untracked_line_count(working_dir, entry.path)

            st.markdown(f"{_entry_icon(entry)} `{entry.path}`")
            caption_parts: list[str] = []
            if adds or dels:
                caption_parts.append(f":green[+{adds}] :red[\u2212{dels}]")
            state_label = _entry_state_label(entry)
            if state_label:
                caption_parts.append(state_label)
            if entry.is_renamed and entry.orig_path:
                caption_parts.append(f"from `{entry.orig_path}`")
            if caption_parts:
                st.caption(" \u00b7 ".join(caption_parts))

            with st.expander("Diff", expanded=False):
                diff = _git_diff_for_entry(working_dir, entry)
                if diff.strip():
                    st.code(diff, language="diff")
                else:
                    st.caption("(no textual diff — likely a binary file)")

        if in_progress:
            # Push is unsafe while a merge/rebase is mid-flight; the
            # warning panel above handles the resolution affordance.
            st.caption(
                ":material/warning: Resolve the in-progress "
                f"{state.get('operation') or 'merge'} before pushing."
            )
        else:
            st.button(
                "Push changes",
                icon=":material/upload:",
                type="primary",
                width="stretch",
                key="open_push_dialog_btn",
                on_click=_open_push_dialog,
            )


def _render_file_changes() -> None:
    """Top-level dispatch: pick the git or tool-result panel.

    The git path is preferred whenever the working directory is a real
    git repo. We fall back to the legacy tool-result aggregation only
    when git isn't available — that way users on non-git scratch dirs
    still see what the agent did, while git users get the full
    filesystem-truth picture (including changes outside the chat).
    """
    state = _git_state()
    if state.get("in_repo"):
        _render_git_file_changes(state)
    else:
        _render_tool_result_changes()


# ---------------------------------------------------------------------------
# Git warnings (sidebar)
# ---------------------------------------------------------------------------
def _open_push_dialog() -> None:
    """Reset transient dialog state, then flip the dialog flag.

    Resetting clears any cached commit message / PR description from a
    prior open so the dialog regenerates fresh against the current diff.
    The set of checked-by-default files is recomputed from
    ``git status`` on first render.
    """
    st.session_state.push_dialog_state = {}
    st.session_state.push_dialog_open = True


def _close_push_dialog() -> None:
    st.session_state.push_dialog_open = False


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
# Push dialog
# ---------------------------------------------------------------------------
COMMIT_MSG_SYSTEM = (
    "You are a senior engineer writing a git commit message.\n"
    "Output a single conventional-commit message:\n"
    "- First line: `<type>: <subject>` (<= 72 chars), where <type> is one of "
    "feat, fix, refactor, docs, test, chore, perf, style, build, ci.\n"
    "- Blank line.\n"
    "- 1-4 bullet body explaining the *why* of the change.\n"
    "Output the message text only. No code fences, no preamble, no quotes."
)

PR_DESC_SYSTEM = (
    "You are a senior engineer writing a pull-request description.\n"
    "Reply with strictly valid JSON of the shape "
    '`{"title": "...", "body": "..."}`.\n'
    "- title: <= 80 chars, no trailing period.\n"
    "- body: GitHub-flavored markdown with three sections (in order):\n"
    "  `## Summary` (1-3 bullets describing what and why),\n"
    "  `## Changes` (per-file or per-area bullets),\n"
    "  `## Test plan` (a checklist with `- [ ]` items).\n"
    "Do not wrap the JSON in code fences. Do not include any text before or "
    "after the JSON object."
)


def _diff_for_paths(working_dir: Path, paths: list[str]) -> str:
    """Build the prompt context we feed DeepSeek for commit/PR generation.

    ``git diff HEAD`` is preferred (works for tracked changes); for
    untracked files we append a synthesized "(new file: <path>)" header
    plus the file body so the model still sees them. Caps the total at a
    generous 200 KB so very large diffs don't blow past V4-Flash's
    context.
    """
    chunks: list[str] = []
    tracked = [p for p in paths if (working_dir / p).exists()]
    if tracked:
        try:
            proc = git_ops._run(
                ["diff", "HEAD", "--no-color", "--no-renames", "--", *tracked],
                cwd=working_dir,
                check=False,
                timeout=30,
            )
            if proc.stdout:
                chunks.append(proc.stdout)
        except (GitError, OSError):
            pass

    state = _git_state()
    untracked_paths = {
        e.path for e in (state.get("status") or []) if e.is_untracked
    }
    for p in paths:
        if p not in untracked_paths:
            continue
        target = working_dir / p
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        chunks.append(f"\n--- /dev/null\n+++ b/{p}\n")
        for line in text.splitlines():
            chunks.append(f"+{line}\n")

    blob = "".join(chunks)
    cap = 200_000
    if len(blob) > cap:
        blob = blob[:cap] + f"\n\n[truncated, {len(blob) - cap} more chars]\n"
    return blob


def _generate_commit_message(working_dir: Path, paths: list[str]) -> str:
    """Ask DeepSeek for a commit message describing the staged changes."""
    client = st.session_state.get("client")
    if client is None:
        return ""
    diff = _diff_for_paths(working_dir, paths)
    if not diff.strip():
        return ""
    user = (
        "Write a conventional-commit message for the following diff:\n\n"
        "```diff\n" + diff + "\n```"
    )
    return generate_text(
        client=client,
        model=DEEPSEEK_MODEL,
        system=COMMIT_MSG_SYSTEM,
        user=user,
        max_tokens=500,
    )


def _generate_pr_description(
    working_dir: Path,
    paths: list[str],
    branch: str,
    base: str,
) -> tuple[str, str]:
    """Ask DeepSeek for a PR title + body. Returns ``(title, body)``.

    Parses the JSON response with a tolerant fallback: if parsing fails
    we treat the entire response as the body and synthesize a title from
    its first non-empty line. That way we never block the user on a
    malformed model output — they can edit either field before
    submitting.
    """
    client = st.session_state.get("client")
    if client is None:
        return "", ""
    diff = _diff_for_paths(working_dir, paths)
    if not diff.strip():
        return "", ""
    user = (
        f"Branch: `{branch}` -> `{base}`\n\n"
        "Summarize the following diff into a pull-request title and body:\n\n"
        "```diff\n" + diff + "\n```"
    )
    raw = generate_text(
        client=client,
        model=DEEPSEEK_MODEL,
        system=PR_DESC_SYSTEM,
        user=user,
        max_tokens=1500,
    )
    if not raw:
        return "", ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("title") or "").strip(), str(parsed.get("body") or "").strip()
    except json.JSONDecodeError:
        pass
    # Fallback: best-effort title from the first non-empty line.
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    title = lines[0][:80] if lines else "Untitled change"
    return title, raw


def _ensure_deepseek_available() -> tuple[bool, str | None]:
    """Verify ``DEEPSEEK_MODEL`` is in the connected account's model list."""
    models = st.session_state.get("models") or []
    if not models:
        return False, "Connect to W&B Inference first (Settings tab)."
    if DEEPSEEK_MODEL not in models:
        return False, (
            f"Model `{DEEPSEEK_MODEL}` is not available on this W&B "
            "Inference account. Commit messages and PR descriptions need "
            "DeepSeek; the rest of the push flow (commit/push) still works "
            "if you fill in the message yourself."
        )
    return True, None


@st.dialog("Push changes", width="large")
def _push_dialog() -> None:
    """Modal that stages, commits, and pushes the user's selected changes.

    Layout (top to bottom):
    1. Per-file checkboxes — sticky-defaulted to True so the dialog
       opens with everything checked, matching the user's mental model
       of "push my changes".
    2. AI-generated commit message with a regenerate button.
    3. Mode segmented control: push to branch vs. open a pull request.
    4. PR-mode-only: target branch + AI-generated title and body.
    5. Live status pane (the dialog stays open through the multi-step
       push so the user sees progress).
    6. Cancel + primary action button.

    The push handler runs in this same function (after the primary
    button is clicked) so we can update the status pane in place between
    steps. On a ``pull --rebase`` conflict we close the dialog and let
    the sidebar's merge-conflict warning take over.
    """
    ss = st.session_state
    state = _git_state()

    if not state.get("in_repo"):
        st.error("Working directory is not a git repository.", icon=":material/error:")
        if st.button("Close", key="push_close_not_repo"):
            _close_push_dialog()
            st.rerun()
        return

    branch = state.get("current_branch")
    if not branch:
        st.error(
            "Cannot push from a detached HEAD. Check out a branch first.",
            icon=":material/error:",
        )
        if st.button("Close", key="push_close_detached"):
            _close_push_dialog()
            st.rerun()
        return

    entries: list[git_ops.StatusEntry] = list(state.get("status") or [])
    if not entries:
        st.caption("Working tree is clean. Nothing to push.")
        if st.button("Close", key="push_close_clean"):
            _close_push_dialog()
            st.rerun()
        return

    working_dir = Path(ss.working_dir).expanduser().resolve()
    deepseek_ok, deepseek_msg = _ensure_deepseek_available()
    if not deepseek_ok:
        st.warning(deepseek_msg, icon=":material/warning:")

    dlg = ss.push_dialog_state
    dlg.setdefault("mode", "branch")
    files_state: dict[str, bool] = dlg.setdefault("files", {})
    for e in entries:
        files_state.setdefault(e.path, True)

    st.markdown("**Files to include**")
    st.caption(
        "Check the files you want to commit. Untracked files are "
        "available to add too."
    )
    counts = git_ops.summary_diff_counts(
        working_dir, [e.path for e in entries if not e.is_untracked]
    )
    for e in entries:
        adds, dels = counts.get(e.path, (0, 0))
        if e.is_untracked and adds == 0:
            adds = git_ops.untracked_line_count(working_dir, e.path)
        chip = f":green[+{adds}] :red[\u2212{dels}]" if (adds or dels) else ""
        label = f"`{e.path}` {chip}".strip()
        files_state[e.path] = st.checkbox(
            label,
            value=files_state.get(e.path, True),
            key=f"push_file_chk_{e.path}",
        )
    checked_paths = [p for p, v in files_state.items() if v]

    st.divider()

    st.markdown("**Commit message**")
    if "commit_msg" not in dlg and deepseek_ok and checked_paths:
        with st.spinner("Asking DeepSeek for a commit message...", show_time=True):
            try:
                dlg["commit_msg"] = _generate_commit_message(working_dir, checked_paths)
            except Exception as e:
                dlg["commit_msg"] = ""
                dlg["commit_msg_error"] = f"{type(e).__name__}: {e}"
    msg_value = dlg.get("commit_msg") or ""
    new_msg = st.text_area(
        "Commit message",
        value=msg_value,
        key="push_commit_msg_input",
        height=140,
        label_visibility="collapsed",
    )
    dlg["commit_msg"] = new_msg
    if dlg.get("commit_msg_error"):
        st.caption(f":red[{dlg['commit_msg_error']}]")
    msg_cols = st.columns([1, 4])
    if msg_cols[0].button(
        "Regenerate",
        icon=":material/refresh:",
        key="push_regen_msg_btn",
        disabled=not (deepseek_ok and checked_paths),
        width="stretch",
    ):
        with st.spinner("Asking DeepSeek for a commit message...", show_time=True):
            try:
                dlg["commit_msg"] = _generate_commit_message(working_dir, checked_paths)
                dlg.pop("commit_msg_error", None)
            except Exception as e:
                dlg["commit_msg_error"] = f"{type(e).__name__}: {e}"
        st.rerun()

    st.divider()

    mode_label = st.segmented_control(
        "Mode",
        options=["Push to branch", "Create pull request"],
        default="Push to branch" if dlg.get("mode", "branch") == "branch" else "Create pull request",
        key="push_mode_seg",
    ) or "Push to branch"
    dlg["mode"] = "pr" if mode_label == "Create pull request" else "branch"

    if dlg["mode"] == "pr":
        default_target = state.get("default_branch") or "main"
        dlg.setdefault("target_branch", default_target)
        dlg["target_branch"] = st.text_input(
            "Target branch",
            value=dlg.get("target_branch") or default_target,
            key="push_pr_target_input",
            help=(
                "Branch the pull request will merge *into*. Defaults to "
                "the repo's detected default branch."
            ),
        )

        if "pr_title" not in dlg and deepseek_ok and checked_paths:
            with st.spinner("Asking DeepSeek for a PR title and body...", show_time=True):
                try:
                    title, body = _generate_pr_description(
                        working_dir,
                        checked_paths,
                        branch,
                        dlg["target_branch"],
                    )
                    dlg["pr_title"] = title
                    dlg["pr_body"] = body
                except Exception as e:
                    dlg["pr_title"] = ""
                    dlg["pr_body"] = ""
                    dlg["pr_error"] = f"{type(e).__name__}: {e}"

        dlg["pr_title"] = st.text_input(
            "PR title",
            value=dlg.get("pr_title") or "",
            key="push_pr_title_input",
        )
        dlg["pr_body"] = st.text_area(
            "PR body",
            value=dlg.get("pr_body") or "",
            key="push_pr_body_input",
            height=240,
        )
        if dlg.get("pr_error"):
            st.caption(f":red[{dlg['pr_error']}]")
        if st.button(
            "Regenerate title and body",
            icon=":material/refresh:",
            key="push_regen_pr_btn",
            disabled=not (deepseek_ok and checked_paths),
        ):
            with st.spinner("Asking DeepSeek for a PR title and body...", show_time=True):
                try:
                    title, body = _generate_pr_description(
                        working_dir,
                        checked_paths,
                        branch,
                        dlg["target_branch"],
                    )
                    dlg["pr_title"] = title
                    dlg["pr_body"] = body
                    dlg.pop("pr_error", None)
                except Exception as e:
                    dlg["pr_error"] = f"{type(e).__name__}: {e}"
            st.rerun()

    status_box = st.container(border=False)
    last_status = dlg.get("status") or []
    if last_status:
        with status_box:
            for line in last_status:
                st.caption(line)

    st.divider()
    btn_cols = st.columns([1, 1, 2])
    cancel_clicked = btn_cols[0].button(
        "Cancel",
        icon=":material/close:",
        key="push_cancel_btn",
        width="stretch",
    )
    primary_label = "Create PR" if dlg["mode"] == "pr" else "Commit and push"
    primary_clicked = btn_cols[1].button(
        primary_label,
        icon=":material/upload:",
        type="primary",
        key="push_primary_btn",
        width="stretch",
        disabled=not checked_paths or not (dlg.get("commit_msg") or "").strip(),
    )

    if cancel_clicked:
        _close_push_dialog()
        st.rerun()

    if not primary_clicked:
        return

    status_lines: list[str] = []

    def status(line: str) -> None:
        status_lines.append(line)
        with status_box:
            st.caption(line)

    try:
        status(":material/playlist_add_check: Resetting staging area...")
        git_ops.unstage_all(working_dir)

        status(f":material/add: Staging {len(checked_paths)} file(s)...")
        git_ops.stage(working_dir, checked_paths)

        status(":material/save: Creating commit...")
        git_ops.commit(working_dir, dlg["commit_msg"].strip())
        _bump_git_nonce()

        if git_ops.has_upstream(working_dir, branch):
            status(":material/sync: Fetching upstream...")
            try:
                git_ops.fetch(working_dir)
            except GitError as e:
                # No-remote / auth failures shouldn't block push (push
                # itself will report the same problem with a clearer
                # message); just note it.
                status(f":material/warning: fetch warning: {e.stderr}")

            if git_ops.is_behind_upstream(working_dir):
                status(":material/sync_alt: Branch is behind; rebasing on upstream...")
                pull = git_ops.pull_rebase(working_dir)
                _bump_git_nonce()
                if not pull.ok and pull.conflict:
                    ss.merge_conflict = {
                        "files": pull.files,
                        "operation": pull.operation,
                    }
                    status(
                        ":material/error: Rebase produced merge conflicts. "
                        "Closing this dialog so you can resolve them with "
                        "DeepSeek from the sidebar warning."
                    )
                    dlg["status"] = status_lines
                    _close_push_dialog()
                    st.rerun()

        status(":material/cloud_upload: Pushing to origin...")
        push_result = git_ops.push(working_dir, branch=branch)
        _bump_git_nonce()
        if not push_result.ok:
            status(f":material/error: push failed: {push_result.stderr.strip()}")
            dlg["status"] = status_lines
            return

        if dlg["mode"] == "pr":
            target = (dlg.get("target_branch") or "").strip() or state.get("default_branch") or "main"
            url = git_ops.remote_compare_url(
                working_dir,
                branch,
                target,
                title=dlg.get("pr_title") or "",
                body=dlg.get("pr_body") or "",
            )
            if url is None:
                # Unknown host (self-hosted). Try to surface whatever the
                # remote returned in stderr; otherwise tell the user to
                # open the PR by hand.
                fallback = git_ops.extract_pr_link_from_stderr(push_result.stderr)
                if fallback:
                    url = fallback
            if url:
                status(f":material/link: Opening PR draft: {url}")
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
                with status_box:
                    st.success(
                        f"Pushed `{branch}`. [Open PR draft]({url}).",
                        icon=":material/check_circle:",
                    )
            else:
                with status_box:
                    st.success(
                        f"Pushed `{branch}` to origin. The remote did not "
                        "return a recognized PR-creation URL — open one "
                        "manually on your hosting platform.",
                        icon=":material/check_circle:",
                    )
            st.toast(f"Pushed `{branch}` and opened PR draft", icon=":material/check_circle:")
        else:
            with status_box:
                st.success(
                    f"Pushed `{branch}` to origin.",
                    icon=":material/check_circle:",
                )
            st.toast(f"Pushed `{branch}`", icon=":material/check_circle:")

    except GitError as e:
        status(f":material/error: {e.stderr or e}")
        dlg["status"] = status_lines
        return
    except Exception as e:
        status(f":material/error: {type(e).__name__}: {e}")
        dlg["status"] = status_lines
        return

    # Success — drop the cached state and close the dialog on the next
    # rerun so the file-changes panel re-scans the (now-clean) tree.
    dlg["status"] = status_lines
    ss.push_dialog_state = {}
    _close_push_dialog()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar (shared chrome on every page)
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """Render the per-page sidebar chrome.

    Both Settings (GitHub, theme, W&B Inference, MCP) and the chat-time
    controls (workdir, model, mode) live on their own pages. The sidebar
    keeps app branding and the file-changes summary — small, persistent
    affordances that are useful from any page.
    """
    with st.sidebar:
        st.markdown("### :material/smart_toy: W&B Coding Agent")
        st.caption("A code editing agent powered by W&B Inference.")

        _render_git_warnings()
        _render_file_changes()

    if st.session_state.get("push_dialog_open"):
        _push_dialog()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    _maybe_auto_connect()

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
