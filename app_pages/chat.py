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
import chat_input
import chats
import commit_ai
import git_ops
import mcp_servers
import project_context
import usage as usage_log
from chat_input import mount_slash_autocomplete
from git_ops import GitError
import model_catalog
import providers
from models import (
    MODEL_METADATA,
    RECOMMENDED_MODELS,
    model_label,
    model_provider,
    models_with_tag,
    unqualified,
    weak_tool_calling_issue_url,
)

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
# 330px is calibrated against a typical desktop viewport (≈900px tall) so
# the chat input + actions row + model selector below the conversation
# area stay in view without the user having to scroll the whole page —
# i.e. the controls "stay pinned at the bottom" of the visible viewport
# for the common case. On taller viewports there will be empty space
# below the controls; on shorter viewports the page itself gains a
# scrollbar. We accept that trade-off because trying to make the height
# truly responsive via CSS ``calc(100vh - …)`` requires fighting with
# Streamlit-internal class names on the height-bearing inner element,
# which is unstable across upgrades — empirically tested and reverted,
# see git history. Tweak this integer if you change the natural height
# of anything outside the conversation area on the chat page (title
# above; chat input + slash-autocomplete slot + single actions row
# [workdir | branch | Changes | Sync] + model row + model card
# caption below — Browse, Start a new project, New branch, and
# Fetch upstream branches all live as sentinel options at the top of
# the relevant dropdowns rather than as separate cells, so the row
# is compact). The project context is a modal triggered from the
# model row, not an inline expander, so toggling it never reshuffles
# this layout.
_CHAT_HISTORY_HEIGHT_PX = 330


# Allow-list of file extensions the chat input accepts as attachments.
# Covers images (vision-capable models), PDFs (Anthropic / Google
# native, others get text-extracted via pypdf), and a curated set of
# plain-text + code formats. We deliberately do **not** pass this
# list as ``file_type=`` to ``st.chat_input``: doing so would surface
# the full list as a tooltip on Streamlit's attach button (cluttering
# the chat UI). Instead we let the OS picker accept anything and
# validate manually after submit, so the user discovers the
# constraint via a clear error message that names the file they
# tried and the supported extensions.
_SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = (
    "png", "jpg", "jpeg", "webp", "gif",
    "pdf",
    "txt", "md", "json", "csv", "yaml", "yml", "toml",
    "py", "ts", "tsx", "js", "jsx", "html", "css",
    "sh", "rs", "go", "java", "cpp", "c", "h",
)


# CSS for Streamlit's chat-input attach button. Two things going on
# (the third — *repositioning* the button to live next to the submit
# arrow — is done in JS, not CSS, by ``chat_input.relocateFileButton``;
# see the docstring there for why ``order:`` doesn't work on its own).
#
# 1. **Match the submit button's resting visual exactly** so the two
#    buttons read as a single matched pair. Streamlit's submit button
#    measures ``20px × 20px`` with ``3.75px`` padding and a ``5px``
#    border-radius (verified by ``getComputedStyle`` against a live
#    chat input), and uses ``rgba(172, 177, 195, 0.15)`` as its
#    resting / disabled-state fill (a translucent theme-derived
#    color that reads correctly in both light and dark mode without
#    depending on a specific Streamlit CSS-variable name — those
#    rotate occasionally between minor versions). The paperclip
#    inherits exactly those values so the two buttons sit as
#    visually-identical pills inside the chat input's right cluster.
#    The submit button still bumps to ``--st-primary`` once the
#    textarea has text (Streamlit's own behavior); the paperclip
#    intentionally stays neutral, which mirrors the "secondary
#    action" UX pattern (Send is the primary action; Attach is a
#    helper).
# 2. **Swap glyph** from the default ``+`` SVG to ``attach_file``
#    from Material Symbols Rounded (the same icon font Streamlit
#    loads for its ``:material/...:`` shortcodes), at ``10px`` to
#    match the submit arrow's intrinsic icon size (the submit
#    button uses ``font-size: 10px`` for its inline arrow SVG).
#
# The SVG hide + ``::before`` glyph substitution keeps the existing
# button behavior (click handler + drag-and-drop target) intact —
# only the visual changes.
_CHAT_INPUT_ATTACH_BUTTON_CSS = (
    "<style>"
    # 1. Match submit button's exact dimensions + resting fill.
    '[data-testid="stChatInputFileUploadButton"]{'
    "width:20px!important;height:20px!important;"
    "padding:3.75px!important;"
    "border-radius:5px!important;"
    "background-color:rgba(172,177,195,0.15)!important;"
    "border:none!important;"
    "}"
    '[data-testid="stChatInputFileUploadButton"]:hover{'
    "background-color:rgba(172,177,195,0.30)!important;"
    "}"
    # 2. Replace the inline SVG ``+`` with the Material paperclip,
    # sized to match the submit button's arrow glyph.
    '[data-testid="stChatInputFileUploadButton"] svg{display:none!important;}'
    '[data-testid="stChatInputFileUploadButton"]::before{'
    'content:"attach_file";'
    'font-family:"Material Symbols Rounded";'
    "font-size:14px;font-weight:400;font-style:normal;"
    "line-height:1;letter-spacing:normal;text-transform:none;"
    "display:inline-block;white-space:nowrap;direction:ltr;"
    "color:inherit;"
    "-webkit-font-feature-settings:'liga';"
    "-webkit-font-smoothing:antialiased;"
    "}"
    "</style>"
)


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

        # Phase 5: media-generation tool results carry a ``kind`` field
        # ("image" / "audio" / "video") that drives inline HTML5
        # preview + lightbox + download affordances. The contract is
        # documented under ``tools.py``'s ``generate_*`` tools.
        kind = result.get("kind")
        if kind in ("image", "audio", "video"):
            _render_media_result(name, result)
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


# ---------------------------------------------------------------------------
# Phase 5: media result renderer + lightbox dialog
# ---------------------------------------------------------------------------
def _format_cost_usd(value: Any) -> str:
    """Format ``cost_usd`` consistently with the per-turn caption."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v < 0.01:
        return f"${v:.4f}"
    if v < 1:
        return f"${v:.3f}"
    return f"${v:.2f}"


def _open_lightbox(payload: dict[str, Any]) -> None:
    """Set the lightbox payload + flip the gating flag for the next render."""
    st.session_state.lightbox_payload = payload
    st.session_state.lightbox_open = True


def _close_lightbox() -> None:
    """Drop the gating flag + payload so the modal stops re-mounting.

    Wired to ``@st.dialog(..., on_dismiss=_close_lightbox)`` per the
    AGENTS.md @st.dialog rule — without this, X / Esc / click-outside
    dismissals leave ``lightbox_open=True`` and the modal re-opens on
    the next rerun (e.g. when the user sends another chat message).
    """
    st.session_state.lightbox_open = False
    st.session_state.lightbox_payload = None


def _kick_fullscreen() -> None:
    """Bump the fullscreen request counter so the CCv2 trigger fires."""
    st.session_state["_lightbox_fullscreen_counter"] = (
        int(st.session_state.get("_lightbox_fullscreen_counter") or 0) + 1
    )


@st.dialog("Preview", width="large", on_dismiss=_close_lightbox)
def _lightbox_dialog() -> None:
    """Lightbox modal for inline image / audio / video previews.

    Mounted from ``render()`` when ``ss.lightbox_open`` is True and
    ``ss.lightbox_payload`` carries the asset to display. The payload
    shape is::

        {
          "kind": "image" | "audio" | "video",
          "path": "<absolute or workdir-relative path>",
          "alt": "<accessibility label>",
          "caption": "<inline caption shown below the asset>",
        }

    For images, a "Fullscreen" button mounts the
    :mod:`fullscreen_trigger` CCv2 component which calls the HTML5
    Fullscreen API on the rendered ``<img>``. Video relies on the
    browser's native ``<video>`` fullscreen control.
    """
    payload = st.session_state.get("lightbox_payload") or {}
    kind = payload.get("kind") or ""
    path = payload.get("path") or ""
    caption = payload.get("caption") or ""

    if not path:
        st.info("No preview available.", icon=":material/info:")
        return

    # Resolve workdir-relative paths against the active chat's
    # working directory (the only folder we ever write artifacts to,
    # per the AGENTS.md hard rule). Absolute paths pass through.
    abs_path = path
    if not Path(path).is_absolute():
        wd = (st.session_state.get("working_dir") or "").strip()
        if wd:
            abs_path = str((Path(wd) / path).resolve())

    if not Path(abs_path).exists():
        st.warning(
            "The asset is no longer on disk.",
            icon=":material/visibility_off:",
        )
        st.caption(caption)
        return

    if kind == "image":
        # Inject a stable id on the rendered <img> so the Fullscreen
        # CCv2 trigger has a predictable selector.
        st.html(
            "<style>"
            "[data-testid='stMainBlockContainer'] img.wb-lightbox-img"
            "{display:block;margin:0 auto;max-width:100%;}"
            "</style>"
        )
        st.image(abs_path, use_container_width=True)
        if caption:
            st.caption(caption)
        cols = st.columns([1, 1, 2])
        with cols[0]:
            st.button(
                "Fullscreen",
                icon=":material/fullscreen:",
                on_click=_kick_fullscreen,
                width="stretch",
                key="_lightbox_fullscreen_btn",
            )
        with cols[1]:
            try:
                with open(abs_path, "rb") as f:
                    st.download_button(
                        "Download",
                        f.read(),
                        file_name=Path(abs_path).name,
                        mime="image/png",
                        icon=":material/download:",
                        width="stretch",
                        key="_lightbox_download_btn",
                    )
            except OSError:
                pass
        # Mount the fullscreen trigger. Selector targets the most
        # recent <img> in the dialog's main element — the dialog
        # contains exactly one image so this is unambiguous.
        from fullscreen_trigger import mount_fullscreen_trigger
        mount_fullscreen_trigger(
            "div[role='dialog'] img",
            request_id=int(st.session_state.get("_lightbox_fullscreen_counter") or 0),
        )
    elif kind == "audio":
        st.audio(abs_path)
        if caption:
            st.caption(caption)
        try:
            with open(abs_path, "rb") as f:
                st.download_button(
                    "Download",
                    f.read(),
                    file_name=Path(abs_path).name,
                    mime="audio/mpeg",
                    icon=":material/download:",
                    key="_lightbox_audio_download_btn",
                )
        except OSError:
            pass
    elif kind == "video":
        st.video(abs_path)
        if caption:
            st.caption(caption)
        try:
            with open(abs_path, "rb") as f:
                st.download_button(
                    "Download",
                    f.read(),
                    file_name=Path(abs_path).name,
                    mime="video/mp4",
                    icon=":material/download:",
                    key="_lightbox_video_download_btn",
                )
        except OSError:
            pass


def _render_media_result(tool_name: str, result: dict[str, Any]) -> None:
    """Render the inline preview row for a media-generation tool result.

    Dispatches off ``result.kind`` to the appropriate Streamlit
    HTML5 element. Each row also exposes a "View" button that
    opens the lightbox modal at full dialog width and a download
    button so users can grab the file directly. Replay-safe:
    when the saved file is missing, render a chip + the original
    prompt + cost so chat history stays informative.
    """
    kind = result.get("kind", "")
    model_used = result.get("model_used") or "—"
    cost = _format_cost_usd(result.get("cost_usd"))
    wd = (st.session_state.get("working_dir") or "").strip()

    def _abs(rel: str) -> str:
        if not rel:
            return rel
        return rel if Path(rel).is_absolute() else (
            str((Path(wd) / rel).resolve()) if wd else rel
        )

    if kind == "image":
        paths = result.get("paths") or []
        if not paths:
            st.warning("No images returned.", icon=":material/info:")
            return
        for idx, rel_path in enumerate(paths):
            abs_path = _abs(rel_path)
            if not Path(abs_path).exists():
                st.markdown(
                    f":gray-badge[file not found] `{rel_path}`"
                )
                continue
            st.image(abs_path, use_container_width=True)
            cols = st.columns([1, 1, 4])
            with cols[0]:
                st.button(
                    "View",
                    icon=":material/zoom_in:",
                    on_click=_open_lightbox,
                    args=({
                        "kind": "image",
                        "path": rel_path,
                        "alt": result.get("prompt") or "",
                        "caption": result.get("prompt") or "",
                    },),
                    key=f"_media_view_{tool_name}_{idx}_{rel_path}",
                    width="stretch",
                )
            with cols[1]:
                try:
                    with open(abs_path, "rb") as f:
                        st.download_button(
                            "Download",
                            f.read(),
                            file_name=Path(abs_path).name,
                            mime=result.get("mime_type") or "image/png",
                            icon=":material/download:",
                            key=f"_media_dl_{tool_name}_{idx}_{rel_path}",
                            width="stretch",
                        )
                except OSError:
                    pass
        size = result.get("size") or ""
        quality = result.get("quality") or ""
        chips = [c for c in [size, quality] if c]
        suffix = " \u00b7 ".join(chips)
        suffix = f"{suffix} \u00b7 " if suffix else ""
        st.caption(f"`{model_used}` \u00b7 {suffix}{cost}")

    elif kind == "audio":
        rel_path = result.get("path") or ""
        abs_path = _abs(rel_path)
        if not rel_path or not Path(abs_path).exists():
            st.markdown(
                f":gray-badge[file not found] `{rel_path or 'audio'}`"
            )
        else:
            st.audio(abs_path)
            try:
                with open(abs_path, "rb") as f:
                    st.download_button(
                        "Download",
                        f.read(),
                        file_name=Path(abs_path).name,
                        mime=result.get("mime_type") or "audio/mpeg",
                        icon=":material/download:",
                        key=f"_media_dl_audio_{rel_path}",
                    )
            except OSError:
                pass
        voice = result.get("voice") or ""
        chips = [c for c in [voice and f"voice {voice}"] if c]
        suffix = " \u00b7 ".join(chips)
        suffix = f"{suffix} \u00b7 " if suffix else ""
        st.caption(f"`{model_used}` \u00b7 {suffix}{cost}")

    elif kind == "video":
        rel_path = result.get("path") or ""
        abs_path = _abs(rel_path)
        if not rel_path or not Path(abs_path).exists():
            st.markdown(
                f":gray-badge[file not found] `{rel_path or 'video'}`"
            )
        else:
            st.video(abs_path)
            cols = st.columns([1, 1, 4])
            with cols[0]:
                st.button(
                    "View",
                    icon=":material/zoom_in:",
                    on_click=_open_lightbox,
                    args=({
                        "kind": "video",
                        "path": rel_path,
                        "alt": result.get("prompt") or "",
                        "caption": result.get("prompt") or "",
                    },),
                    key=f"_media_view_video_{rel_path}",
                    width="stretch",
                )
            with cols[1]:
                try:
                    with open(abs_path, "rb") as f:
                        st.download_button(
                            "Download",
                            f.read(),
                            file_name=Path(abs_path).name,
                            mime=result.get("mime_type") or "video/mp4",
                            icon=":material/download:",
                            key=f"_media_dl_video_{rel_path}",
                            width="stretch",
                        )
                except OSError:
                    pass
        duration = result.get("duration_seconds")
        ar = result.get("aspect_ratio") or ""
        chips = [c for c in [duration and f"{duration}s", ar] if c]
        suffix = " \u00b7 ".join(chips)
        suffix = f"{suffix} \u00b7 " if suffix else ""
        st.caption(f"`{model_used}` \u00b7 {suffix}{cost}")


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
            elif etype == "cancelled":
                # Subtle "Stopped by user" caption — appended by
                # ``chats._finalize_cancelled_turn`` when the user
                # clicked the Stop button (or "Stop and delete"
                # before delete-and-go won the race). Rendered as a
                # plain caption (not ``st.error``) because cancel is
                # an intentional user action, not a failure mode.
                st.caption(":material/stop_circle: :grey[Stopped by user.]")
            elif etype == "error":
                st.error(ev["message"], icon=":material/error:")
        if weave_trace_url and not has_turn_usage:
            _render_trace_only(weave_trace_url)


def _render_user_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        # Phase 6: render attachments above the prompt text. Each
        # attachment is one of the ``Attachment`` dicts persisted by
        # :func:`attachments.build_user_message`; the renderer
        # dispatches off ``kind`` for the right HTML5 element +
        # download / view affordance.
        attachments = turn.get("attachments") or []
        if attachments:
            _render_user_attachments(attachments)
        content = turn.get("content")
        if isinstance(content, str) and content:
            st.markdown(content)


def _render_user_attachments(attachments: list[dict[str, Any]]) -> None:
    """Render the attachments row above the user's prompt text.

    Images: small thumbnails (``st.image(..., width=160)``) with a
    "View" button that opens the lightbox modal. PDFs: a chip plus a
    download button. Text/code: a chip that expands an
    ``st.code`` block with the extracted content. Replay-safe:
    when a saved file is missing, render ``:gray-badge[file not
    found]`` plus the original filename.
    """
    ss = st.session_state
    wd = (ss.get("working_dir") or "").strip()

    def _abs(rel: str) -> str:
        if not rel:
            return rel
        return rel if Path(rel).is_absolute() else (
            str((Path(wd) / rel).resolve()) if wd else rel
        )

    for idx, att in enumerate(attachments):
        kind = att.get("kind") or "binary"
        filename = att.get("filename") or "(unnamed)"
        size = int(att.get("size_bytes") or 0)
        rel_path = att.get("path") or ""
        abs_path = _abs(rel_path)
        delivery = att.get("delivery") or ""

        if kind == "image":
            if Path(abs_path).exists():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(abs_path, width=160)
                with cols[1]:
                    st.caption(f":material/attach_file: `{filename}` ({_human_bytes(size)})")
                    st.button(
                        "View",
                        icon=":material/zoom_in:",
                        key=f"_user_att_view_{idx}_{rel_path}",
                        on_click=_open_lightbox,
                        args=({
                            "kind": "image",
                            "path": rel_path,
                            "alt": filename,
                            "caption": filename,
                        },),
                    )
            else:
                st.markdown(
                    f":gray-badge[file not found] :material/image: `{filename}` ({_human_bytes(size)})"
                )

        elif kind == "pdf":
            chip = (
                ":blue-badge[native PDF]"
                if delivery == "pdf_native"
                else ":gray-badge[text-extracted]"
            )
            cols = st.columns([3, 1])
            with cols[0]:
                if Path(abs_path).exists():
                    st.markdown(
                        f":material/picture_as_pdf: `{filename}` "
                        f"\u00b7 {_human_bytes(size)} \u00b7 {chip}"
                    )
                else:
                    st.markdown(
                        f":gray-badge[file not found] :material/picture_as_pdf: `{filename}`"
                    )
            with cols[1]:
                if Path(abs_path).exists():
                    try:
                        with open(abs_path, "rb") as f:
                            st.download_button(
                                "Download",
                                f.read(),
                                file_name=filename,
                                mime="application/pdf",
                                icon=":material/download:",
                                key=f"_user_att_dl_{idx}_{rel_path}",
                                width="stretch",
                            )
                    except OSError:
                        pass

        elif kind == "text":
            extracted = att.get("extracted_text")
            with st.expander(
                f":material/description: {filename} \u00b7 {_human_bytes(size)}",
                expanded=False,
            ):
                if extracted:
                    # Best-effort language hint based on filename.
                    ext = Path(filename).suffix.lower()
                    try:
                        from attachments import LANGUAGE_FOR_EXT
                        lang = LANGUAGE_FOR_EXT.get(ext, "text")
                    except ImportError:
                        lang = "text"
                    st.code(extracted, language=lang)
                else:
                    st.caption("No extracted text available.")

        else:
            st.markdown(
                f":gray-badge[unsupported] :material/attach_file: `{filename}` ({_human_bytes(size)})"
            )


def _human_bytes(n: int) -> str:
    """Format a byte count as ``"N B"`` / ``"N.N KB"`` / ``"N.N MB"``."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


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


def _start_turn(
    prompt: str,
    *,
    override_model: str | None = None,
    attachments: list[Any] | None = None,
) -> None:
    """Spawn a background turn for the active chat.

    Thin wrapper that pulls the active :class:`chats.Chat` out of session
    state, syncs the user's flat dropdown picks (model / mode / working
    dir) into the chat object so the background thread sees the
    user's intent, runs the per-(session, working_dir) git identity
    stamp, and delegates to :func:`chats.start_turn`. The fragment in
    :func:`_render_active_chat` picks up the streaming events on its
    next 0.25s tick.

    ``attachments`` (Phase 6): when non-empty, the multimodal user
    message is built via :func:`attachments.build_user_message`
    instead of the legacy plain-string path. The chosen model's
    capability flags (``supports_vision`` / ``supports_pdf_input``)
    drive whether image / PDF parts go through natively or are
    text-extracted by ``pypdf``.
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

    # Resolve which provider to call into based on the qualified id.
    # ``model_provider`` returns ``None`` for legacy bare ids; in that
    # case we fall back to W&B (the only previously-supported provider).
    turn_model = override_model or chat.model
    provider_id = model_provider(turn_model) if turn_model else None
    if not provider_id:
        provider_id = "wandb"
    client_for_turn = ss.clients.get(provider_id)
    api_key_for_turn = ss.provider_keys.get(provider_id, "")
    # Resolve the model's vision / PDF capability flags so
    # ``attachments.build_user_message`` knows whether to inline
    # images vs. drop them, and whether to pass PDFs as native
    # ``document`` blocks vs. running pypdf extraction.
    supports_vision = False
    supports_pdf_input = False
    if turn_model:
        try:
            import model_catalog as _mc
            info = _mc.get_info(turn_model)
            if info is not None:
                supports_vision = bool(info.supports_vision)
                supports_pdf_input = bool(info.supports_pdf_input)
        except Exception:  # noqa: BLE001
            pass

    try:
        chats.start_turn(
            chat,
            prompt,
            client_for_turn,
            override_model=override_model,
            provider_id=provider_id,
            api_key=api_key_for_turn,
            # Auto-titling uses W&B's DeepSeek; pass the W&B key
            # along regardless of the per-turn provider so the
            # title generator can fire on a non-W&B chat too.
            wandb_api_key=ss.provider_keys.get("wandb", ""),
            # Phase 5 media tools dispatch through ``ToolContext``;
            # pass the full provider_keys + clients dicts so e.g.
            # ``generate_image`` (which routes via OpenAI by default)
            # can find an OpenAI client even if the chat itself is
            # configured against W&B.
            provider_keys=dict(ss.provider_keys),
            clients=dict(ss.clients),
            attachments=attachments or [],
            supports_vision=supports_vision,
            supports_pdf_input=supports_pdf_input,
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


# Sentinel option labels at the top of the working-directory dropdown.
# Selecting one routes through ``_on_working_dir_select`` to either
# launch the native folder picker or open the new-project modal,
# instead of treating the sentinel string as a literal path. We keep
# them as plain text (with leading punctuation that's unlikely to
# collide with a real path) because Streamlit selectbox option labels
# render as raw strings — Material icon tokens don't work inside
# selectbox options. The branch picker uses the same pattern with
# ``_NEW_BRANCH_SENTINEL``.
#
# All sentinels share a leading ``›`` (U+203A SINGLE RIGHT-POINTING
# ANGLE QUOTATION MARK) glyph as a subtle unifying visual marker that
# delineates them from real path / branch entries below. The glyph is
# decorative — comparison sites use the constants directly so the
# sentinel-vs-real-value membership check stays exact, and the
# "meaningful name" inside the constant (e.g. ``Browse for a folder``)
# is what we surface in help text / docs / dialog copy.
_WD_BROWSE_SENTINEL = "›  Browse for a folder..."
_WD_NEW_PROJECT_SENTINEL = "›  Start a new project..."
_WD_SENTINELS: frozenset[str] = frozenset(
    [_WD_BROWSE_SENTINEL, _WD_NEW_PROJECT_SENTINEL]
)


def _on_working_dir_select() -> None:
    """Working-directory selectbox callback: persist + sync to active chat.

    Recents and folder-picker live in ``actions.py`` so pages can
    import them without dragging the entry-point script back through
    Python's import machinery (Streamlit loads the entry as
    ``__main__``; importing it from a sub-page re-executes ``main()``
    and re-renders the sidebar, blowing up with duplicate-widget-key
    errors).

    Two sentinel options live at the top of the dropdown:
      - ``_WD_BROWSE_SENTINEL`` — picking it opens the native folder
        picker. Selecting a folder updates the working directory; a
        Cancel reverts the dropdown to the previous value.
      - ``_WD_NEW_PROJECT_SENTINEL`` — picking it opens the
        ``_new_project_dialog`` modal (gated by
        ``ss.new_project_dialog_open``). The dialog itself updates
        ``ss.working_dir`` on success.

    Both sentinels revert the selectbox value to the previous working
    directory before triggering their action, so the sentinel string
    never sticks around as the visible value (matches the branch
    picker's ``_NEW_BRANCH_SENTINEL`` pattern).
    """
    ss = st.session_state
    chosen = ss.get("wd_select")
    if not chosen:
        return
    import actions

    if chosen in _WD_SENTINELS:
        # Revert the dropdown so the sentinel label doesn't stick
        # around. The action either updates ``ss.working_dir`` (which
        # the next render will re-seed into ``ss.wd_select``) or
        # leaves it alone if the user cancels.
        previous = ss.get("working_dir") or ""
        ss["wd_select"] = previous
        if chosen == _WD_BROWSE_SENTINEL:
            picked = actions.pick_directory(initial=previous)
            if picked:
                ss.working_dir = picked
                actions.record_recent_dir(picked)
                _persist_active_chat_setting("working_dir", picked)
        elif chosen == _WD_NEW_PROJECT_SENTINEL:
            ss.new_project_dialog_open = True
        return

    ss.working_dir = chosen
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


def _on_draft_change() -> None:
    """``on_draft_change`` callback for :mod:`chat_input`.

    Drains the latest ``setStateValue("draft", payload)`` push from the
    chat-input enhancer's JS singleton and persists it to disk. Streamlit
    fires this once per actual value change at the top of the next rerun
    (chat switch, tab navigation, button click, submit — every natural
    interaction triggers a rerun, which flushes the in-flight draft).

    The payload shape is ``{"chat_id": <id>, "text": <text>}`` — we
    resolve the chat by ``chat_id`` rather than by ``ss.active_chat_id``
    so a typed-then-immediately-switched flow ("type 'hello' in chat A,
    click chat B before pausing typing") still attributes the typing to
    chat A even though the rerun the callback fires under is the one
    that just made chat B the active chat.

    Race-free vs. background turn runners: the chat input is disabled
    while a chat is in ``STATUS_RUNNING`` (see ``st.chat_input(...,
    disabled=...)`` in :func:`render`), so the only chat whose
    ``draft_text`` this callback ever writes is one whose runner thread
    is **not** active. Concurrent ``save_chat`` from the runner +
    callback can't happen on the same chat.
    """
    ss = st.session_state
    state = ss.get(chat_input.STATE_KEY) or {}
    payload = getattr(state, "draft", None) or (
        state.get("draft") if isinstance(state, dict) else None
    )
    if not isinstance(payload, dict):
        return
    chat_id = payload.get("chat_id") or ""
    text = payload.get("text")
    if not chat_id or not isinstance(text, str):
        return
    chat = ss.chats.get(chat_id)
    if chat is None:
        return
    if chat.draft_text == text:
        return
    chat.draft_text = text
    try:
        chats.save_chat(chat)
    except OSError:
        pass


def _project_context_summary_pieces(summary: dict[str, Any]) -> list[str]:
    """Build the ``["N guidance files", "M skills"]`` chips for the button label."""
    eager: list[dict[str, Any]] = list(summary.get("agents_md", [])) + list(
        summary.get("cursor_rules", [])
    )
    all_skills: list[dict[str, Any]] = summary.get("all_skills", [])
    pieces: list[str] = []
    if eager:
        pieces.append(f"{len(eager)} guidance file{'s' if len(eager) != 1 else ''}")
    if all_skills:
        pieces.append(f"{len(all_skills)} skill{'s' if len(all_skills) != 1 else ''}")
    return pieces


def _render_project_context_button(working_dir: str) -> None:
    """Render the Project context button on the model row.

    Clicking the button opens the :func:`_project_context_dialog` modal
    so the user can peek at what's loaded without the layout-shifting
    inline expander we used to render at the tail of the workdir row.
    The button stays in place even when the working directory has no
    detected guidance / skills (rendered as disabled with a clear
    tooltip) so the model row keeps a stable layout regardless of the
    current workdir.
    """
    summary = _scan_project_summary(working_dir or "")
    pieces = _project_context_summary_pieces(summary)
    has_context = bool(pieces)
    # Label is **only** the counts (e.g. ``5 guidance · 12 skills``) —
    # no leading "Project context" word. The ``:material/menu_book:``
    # icon already conveys what the button is for, and dropping the
    # word lets the label fit comfortably in the narrower column.
    # Disabled state still surfaces a clear placeholder so the empty
    # button isn't a mystery — see the help-text branch below.
    if pieces:
        label = "  \u00b7  ".join(pieces)
        help_text = (
            "Show eagerly-loaded guidance files and conditionally-"
            "loaded skills the agent can pull in for this turn."
        )
    else:
        label = "No project context"
        help_text = (
            "No project context detected. Add an `AGENTS.md` or "
            "`.cursor/skills/` directory to your working directory "
            "to give the agent always-on guidance and named skill "
            "commands."
        )
    st.button(
        label,
        icon=":material/menu_book:",
        key="chat_project_context_btn",
        on_click=_open_project_context_dialog,
        disabled=not has_context,
        width="stretch",
        help=help_text,
    )


def _open_project_context_dialog() -> None:
    """Project-context-button callback: flip the dialog open."""
    st.session_state.project_context_dialog_open = True


def _close_project_context_dialog() -> None:
    """Drop the dialog flag so subsequent reruns don't re-mount the modal."""
    st.session_state.project_context_dialog_open = False


@st.dialog("Project context", width="large", on_dismiss=_close_project_context_dialog)
def _project_context_dialog() -> None:
    """Modal showing eagerly-loaded guidance + conditionally-loaded skills.

    Body shape mirrors what the now-deleted inline expander showed:
    a list of always-on guidance files (AGENTS.md, .cursor/rules), a
    list of skill commands the agent will auto-load when the user's
    message matches, and a slug-conflict warning when workspace skills
    shadow user-scope skills.

    Mounted via the modal pattern so toggling the dialog never reflows
    the chat input or the actions row — the user can peek at the
    context, dismiss, and keep typing.

    ``on_dismiss=_close_project_context_dialog`` is mandatory so X /
    Esc / click-outside dismissal clears
    ``ss.project_context_dialog_open``; otherwise the modal re-opens
    on the very next rerun (e.g. when the user presses Enter to send
    a chat message). See :func:`_diff_dialog` for the full rationale.
    """
    ss = st.session_state
    working_dir = ss.working_dir or ""
    summary = _scan_project_summary(working_dir)
    eager: list[dict[str, Any]] = list(summary.get("agents_md", [])) + list(
        summary.get("cursor_rules", [])
    )
    all_skills: list[dict[str, Any]] = summary.get("all_skills", [])

    if not eager and not all_skills:
        st.info(
            "No project context detected for this working directory.",
            icon=":material/info:",
        )
        st.caption(
            "Add an `AGENTS.md` (always-on guidance for the agent) or a "
            "`.cursor/skills/` directory (named slash-commands the agent "
            "can auto-load) to your working directory and reopen this "
            "dialog."
        )
        if st.button(
            "Close",
            icon=":material/close:",
            key="project_ctx_close_empty",
            width="stretch",
        ):
            _close_project_context_dialog()
            st.rerun()
        return

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

    st.divider()
    if st.button(
        "Close",
        icon=":material/close:",
        key="project_ctx_close_btn",
        width="stretch",
    ):
        _close_project_context_dialog()
        st.rerun()


def _render_chat_actions_row() -> None:
    """Single combined row of git + file actions below the chat input.

    Two layout shapes depending on whether a working directory has
    been picked:

    **No workdir selected yet** — the row collapses to just the
    workdir picker (rendered full-width). The git controls (branch
    picker, Changes, Sync) are hidden entirely because none of them
    are actionable until the user points the agent at a folder.
    Hiding (rather than disabling) keeps the brand-new-user state
    quiet — a row of 3 grayed-out cells next to the picker would
    just be noise. As soon as the user picks a folder via the
    dropdown's recents / **Browse for a folder** / **Start a new
    project** options, the next render falls into the full 4-cell
    layout below.

    **Workdir selected** — full 4-cell layout, left-to-right:
      1. Working directory selectbox — also hosts two sentinel
         options at the top of its list (**Browse for a folder** and
         **Start a new project**) that fold the v1 browse + new-
         project icon buttons into the dropdown itself.
      2. Branch selectbox — also hosts two sentinel options at the
         top of its list (**New branch** and **Fetch upstream
         branches**) that fold the v1 new-branch + fetch icon
         buttons into the dropdown itself, matching the workdir-
         dropdown convention. No separate icon buttons.
      3. Changes button (opens the live working-tree diff modal).
      4. Sync primary button (bidirectional commit/pull/push pipeline)
         — sits at the right end of the row as the primary action so
         the user's eye reads "what's pending? then sync it" in left-
         to-right order.

    In the 4-cell layout, the branch picker and Changes button are
    disabled rather than hidden when the workdir is set but isn't a
    git repo / git is missing / there's nothing to view, so the
    column layout stays stable as the user adds files or switches
    between git-and-non-git folders. On a detached HEAD the branch
    picker is replaced with a ``(detached HEAD)`` placeholder inside
    the same column — which means fetch is also unavailable from the
    UI in that state (since it now lives inside the dropdown that's
    disabled). That's an intentional, narrow trade-off: the user's
    workflow naturally requires checking out a branch before doing
    anything else (Sync is also disabled in detached HEAD), and once
    a branch is checked out the dropdown's Fetch sentinel becomes
    reachable again. The hide-vs-disable rule is therefore
    asymmetric: when there is no workdir at all, the git cells are
    hidden (no possible action); once there is a workdir, the git
    cells stay in place (disabled when the specific git op isn't
    applicable, with a tooltip explaining why).

    The new-project dialog is also mounted from this function (it's
    triggered when the user picks the ``_WD_NEW_PROJECT_SENTINEL``
    sentinel from the workdir dropdown). The new-branch dialog is
    mounted from ``render()`` alongside the other dialogs.
    """
    ss = st.session_state

    # Workdir dropdown options: sentinels first, then the current
    # workdir (if any), then recents. The sentinels are intercepted
    # in ``_on_working_dir_select`` and never persisted as the
    # working directory.
    wd_options: list[str] = [_WD_BROWSE_SENTINEL, _WD_NEW_PROJECT_SENTINEL]
    if ss.working_dir and ss.working_dir not in wd_options:
        wd_options.append(ss.working_dir)
    for d in ss.recent_dirs:
        if d not in wd_options:
            wd_options.append(d)

    initial_index = (
        wd_options.index(ss.working_dir)
        if ss.working_dir and ss.working_dir in wd_options
        else 0
    )

    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    working_dir_str = (chat.working_dir if chat else "") or ss.working_dir or ""

    # No workdir picked yet — short-circuit to a single full-width
    # workdir picker so the user's eye lands on the one action they
    # can usefully take. The git cells stay hidden entirely (rather
    # than rendered as a row of disabled grayed-out controls) — a
    # brand-new-user state is busy enough already without showing
    # 4 cells the user can't interact with. The new-project dialog
    # still mounts here because the workdir picker's sentinel
    # options can open it.
    if not working_dir_str:
        st.selectbox(
            "Working directory",
            options=wd_options,
            index=initial_index,
            key="wd_select",
            on_change=_on_working_dir_select,
            format_func=lambda v: v if v in _WD_SENTINELS else _shorten_path(v),
            accept_new_options=True,
            placeholder="Choose or paste a directory",
            help=(
                "Recent working directories. Pick from the list, paste a "
                "custom path, or pick **Browse for a folder...** / "
                "**Start a new project...** at the top of the dropdown "
                "(both prefixed with **›** to set them apart from real "
                "directories) to open the OS picker / the new-project "
                "modal."
            ),
        )
        if ss.get("new_project_dialog_open"):
            _new_project_dialog()
        return

    # Resolve git state once for the whole row so the branch picker,
    # Sync button, and Changes button all see the same snapshot.
    git_installed = git_ops.is_git_installed()
    state: dict[str, Any] = {}
    if git_installed:
        state = _diff_git_state(working_dir_str)
    in_repo = bool(state.get("in_repo"))
    branch = state.get("current_branch") if in_repo else None
    detached = in_repo and not branch
    in_progress = bool(state.get("in_merge_or_rebase"))
    entries: list[git_ops.StatusEntry] = list(state.get("status") or [])
    dirty = bool(entries)

    # Resolve the chat's qualified model id once for the Sync button
    # tooltip + the eventual ``_run_sync_pipeline`` drain. The Sync
    # flow's commit-message + PR-description steps go through
    # :func:`commit_ai.generate_commit_message` /
    # :func:`commit_ai.generate_pr_description`, which take the chat's
    # currently selected model rather than a hard-coded DeepSeek id.
    # ``_resolve_commit_ai_model`` returns the matching client +
    # model + a user-friendly error message when either is missing
    # (blank model, provider not connected, etc.).
    chat_model_str = chat.model if chat else ""
    _commit_client, _commit_model, commit_model_error = _resolve_commit_ai_model(
        chat_model_str
    )

    # Branch selectbox options. We build them up-front so the Sync /
    # Changes columns can still render even when the picker is empty.
    # The two sentinels sit at the **top** of the list (mirroring the
    # workdir dropdown's Browse / Start a new project sentinels):
    # ``_NEW_BRANCH_SENTINEL`` first (creating a branch is the more
    # common follow-up to looking at the list), then
    # ``_FETCH_BRANCHES_SENTINEL`` (a refresh action that doesn't
    # change the working tree). Both share the same leading ``›``
    # glyph as the workdir sentinels so all four pinned action items
    # read as a visually distinct group when the dropdown is open.
    # Both fold the v1 standalone icon buttons into the same dropdown
    # the user is already looking at — no separate icons needed.
    local_branches = list(state.get("branches") or []) if in_repo else []
    remote_branches = list(state.get("remote_branches") or []) if in_repo else []
    remote_only = [
        rb for rb in remote_branches
        if rb.split("/", 1)[-1] not in local_branches
    ]
    branch_options: list[str] = []
    if in_repo and not detached:
        branch_options.append(_NEW_BRANCH_SENTINEL)
        branch_options.append(_FETCH_BRANCHES_SENTINEL)
    if branch and branch not in local_branches:
        branch_options.append(branch)
    branch_options.extend(local_branches)
    branch_options.extend(remote_only)

    # Stable 4-cell layout. Wider weight on the workdir + branch
    # selectboxes (which carry text) and on the Sync + Changes buttons
    # (which carry text labels). The branch column absorbs the width
    # the v1 layout reserved for a standalone fetch icon — that
    # affordance now lives as a sentinel option at the top of the
    # branch dropdown alongside ``_NEW_BRANCH_SENTINEL``.
    cols = st.columns([5, 5, 3, 3], vertical_alignment="bottom")

    # Cell 1: working directory. The sentinel options
    # ``_WD_BROWSE_SENTINEL`` and ``_WD_NEW_PROJECT_SENTINEL`` live at
    # the top of the list; ``_on_working_dir_select`` intercepts them
    # and routes to the OS picker / the new-project modal
    # respectively. The current workdir / recents follow.
    with cols[0]:
        st.selectbox(
            "Working directory",
            options=wd_options,
            index=initial_index,
            key="wd_select",
            on_change=_on_working_dir_select,
            format_func=lambda v: v if v in _WD_SENTINELS else _shorten_path(v),
            accept_new_options=True,
            placeholder="Choose or paste a directory",
            help=(
                "Recent working directories. Pick from the list, paste a "
                "custom path, or pick **Browse for a folder...** / "
                "**Start a new project...** at the top of the dropdown "
                "(both prefixed with **›** to set them apart from real "
                "directories) to open the OS picker / the new-project "
                "modal."
            ),
        )

    # Cell 2: branch selectbox (or detached-HEAD caption / disabled
    # placeholder when the workdir isn't a usable git repo).
    with cols[1]:
        if not git_installed:
            st.selectbox(
                "Branch",
                options=["(git not installed)"],
                index=0,
                disabled=True,
                key="chat_actions_branch_disabled_no_git",
                label_visibility="collapsed",
                help="Install git to enable branch operations.",
            )
        elif not in_repo:
            st.selectbox(
                "Branch",
                options=["(not a git repo)"],
                index=0,
                disabled=True,
                key="chat_actions_branch_disabled_no_repo",
                label_visibility="collapsed",
                help=(
                    "The working directory is not a git repository. Run "
                    "`git init`, or pick a different directory, to enable "
                    "branch operations."
                ),
            )
        elif detached:
            st.selectbox(
                "Branch",
                options=["(detached HEAD)"],
                index=0,
                disabled=True,
                key=f"chat_actions_branch_detached_{chat.id if chat else 'no_chat'}",
                label_visibility="collapsed",
                help=(
                    "Detached HEAD. Switch to a branch via the command line "
                    "to enable branch operations and sync."
                ),
            )
        else:
            select_key = f"chat_bottom_branch_select_{chat.id}"
            sentinel_key = f"_chat_bottom_branch_active_{chat.id}"
            ss[sentinel_key] = branch
            # Defensive: never seed the visible value with one of the
            # sentinel entries — fall through to the first real
            # branch if we somehow can't find ``branch`` in the
            # option list (shouldn't happen, but keeps the dropdown
            # showing a real branch name even if our model drifts).
            if branch in branch_options:
                ss[select_key] = branch
            else:
                real_options = [
                    b for b in branch_options if b not in _BRANCH_SENTINELS
                ]
                ss[select_key] = real_options[0] if real_options else branch_options[0]
            st.selectbox(
                "Branch",
                options=branch_options,
                key=select_key,
                on_change=_on_chat_git_branch_change,
                args=(chat.id,),
                label_visibility="collapsed",
                format_func=lambda b: (
                    b if b in _BRANCH_SENTINELS
                    # Selectbox option labels render as raw strings —
                    # Material tokens don't work here, so use a plain
                    # text marker for remote-only refs.
                    else (f"{b}  (remote)" if b in remote_only else b)
                ),
                help=(
                    "Switch this chat's working-directory branch, create "
                    "a new branch, or fetch upstream branches from origin. "
                    "**New branch...** and **Fetch upstream branches** "
                    "(both prefixed with **›** at the top of the dropdown "
                    "to set them apart from real branch names) trigger "
                    "the create / fetch flows; everything below is a real "
                    "branch you can check out. Pick a `(remote)` entry "
                    "to check out a remote-only branch as a new local "
                    "tracking branch. Uncommitted changes come along on "
                    "the switch when there's no conflict; git surfaces a "
                    "clear error when the target branch would overwrite "
                    "a dirty file."
                ),
            )

    # Cell 3: Changes (live working-tree diff modal).
    with cols[2]:
        changes_disabled = not (in_repo and dirty)
        if not in_repo:
            # Empty placeholder label keeps the cell width — the
            # ``:material/difference:`` icon still conveys what the
            # button is for, and the disabled state + tooltip explain
            # the why.
            changes_label = " "
            changes_help = (
                "Pick a git working directory to view the working-tree diff."
            )
        elif not dirty:
            changes_label = " "
            changes_help = "Working tree is clean. Nothing to view."
        else:
            working_dir_resolved = Path(working_dir_str).expanduser().resolve()
            counts = git_ops.summary_diff_counts(
                working_dir_resolved,
                [e.path for e in entries if not e.is_untracked],
            )
            total_adds = 0
            total_dels = 0
            for entry in entries:
                adds, dels = counts.get(entry.path, (0, 0))
                if entry.is_untracked and adds == 0:
                    adds = git_ops.untracked_line_count(
                        working_dir_resolved, entry.path
                    )
                total_adds += adds
                total_dels += dels
            n = len(entries)
            # Compact label without the leading "Changes" word — the
            # ``:material/difference:`` icon already indicates this is
            # the changes button, and dropping the word lets the
            # +adds/−dels chip + file count fit on a single line in the
            # narrow column.
            changes_label = (
                f"+{total_adds} \u2212{total_dels}  "
                f"\u00b7  {n} file{'s' if n != 1 else ''}"
            )
            changes_help = "View the live working-tree diff in an overlay."
        st.button(
            changes_label,
            icon=":material/difference:",
            key="chat_diff_open_btn",
            width="stretch",
            on_click=_open_diff_dialog,
            help=changes_help,
            disabled=changes_disabled,
        )

    # Cell 4: Sync (bidirectional commit / pull / push). Sits at the
    # right end of the row as the primary action — eye lands on it
    # last, after the user has had a chance to glance at the Changes
    # button to confirm what's about to be synced.
    with cols[3]:
        sync_disabled, sync_help = _sync_button_state(
            in_repo=in_repo,
            detached=detached,
            in_progress=in_progress,
            operation=state.get("operation"),
            dirty=dirty,
            commit_model_error=commit_model_error,
            commit_model=_commit_model,
            working_dir_str=working_dir_str,
            branch=branch,
        )
        st.button(
            "Sync",
            icon=":material/sync:",
            type="primary",
            key=(
                f"chat_bottom_sync_{chat.id}"
                if chat else "chat_bottom_sync_no_chat"
            ),
            on_click=_on_chat_git_sync_clicked,
            disabled=sync_disabled,
            width="stretch",
            help=sync_help,
        )

    if ss.get("new_project_dialog_open"):
        _new_project_dialog()


def _sync_button_state(
    *,
    in_repo: bool,
    detached: bool,
    in_progress: bool,
    operation: str | None,
    dirty: bool,
    commit_model_error: str | None,
    commit_model: str | None,
    working_dir_str: str,
    branch: str | None,
) -> tuple[bool, str]:
    """Decide whether the Sync button is enabled and what its tooltip says.

    The state machine is the read-only mirror of
    :func:`_run_sync_pipeline`'s preconditions: anything that would
    cause the pipeline to early-return with a toast becomes a disabled
    + helpful-tooltip state on the button.

    ``commit_model_error`` / ``commit_model`` come from
    :func:`_resolve_commit_ai_model`. When the working tree is dirty
    we need a usable model for the commit-message draft; when it's
    clean we don't (Sync just fetches + pulls + pushes).

    Returns ``(disabled, help_text)``.
    """
    if not in_repo:
        return True, "Pick a git working directory to enable sync."
    if detached or branch is None:
        return True, "Detached HEAD. Check out a branch to enable sync."
    if in_progress:
        return (
            True,
            f"In-progress `{operation or 'merge'}` — resolve before syncing.",
        )
    if dirty and commit_model_error is not None:
        return True, commit_model_error

    has_upstream = False
    behind = False
    ahead = False
    if working_dir_str:
        working_dir = Path(working_dir_str).expanduser().resolve()
        try:
            has_upstream = git_ops.has_upstream(working_dir, branch)
            if has_upstream:
                behind = git_ops.is_behind_upstream(working_dir)
                ahead = git_ops.is_ahead_of_upstream(working_dir)
        except Exception:  # noqa: BLE001 — best-effort UI predicate
            pass

    if dirty:
        # ``commit_model`` is the qualified ``<provider>:<raw>`` id;
        # show only the raw bit in the tooltip so it reads naturally.
        model_label_short = (
            unqualified(commit_model) if commit_model else "the chat's model"
        )
        return (
            False,
            f"Sync: commit local changes (`{model_label_short}`-drafted "
            "message), fetch, pull --rebase if behind, and push if ahead.",
        )
    if not has_upstream:
        return (
            False,
            "Branch has no upstream yet. Sync opens the Publish branch "
            "dialog so you can push and optionally open a pull request.",
        )
    if behind and ahead:
        return (
            False,
            "Sync: pull --rebase upstream changes, then push your local "
            "commits.",
        )
    if behind:
        return False, "Sync: pull --rebase upstream changes from origin."
    if ahead:
        return False, "Sync: push local commits to origin."
    # Even with nothing to commit / pull / push, Sync still runs
    # `git fetch --prune` and bumps the git-state nonce so any new
    # remote branches show up in the branch dropdown (and any
    # branches deleted on origin disappear). That's a useful action
    # in its own right, so the button stays enabled here.
    return (
        False,
        f"Sync: fetch from origin to refresh the branch list "
        f"(already up to date with `origin/{branch}`).",
    )


# ---------------------------------------------------------------------------
# Start-a-new-project dialog
# ---------------------------------------------------------------------------
# This dialog is an affordance of the working-directory picker in
# ``_render_chat_actions_row`` (the new-project icon button) — opened
# via the ``ss.new_project_dialog_open`` flag and mounted from inside
# the actions-row helper. All filesystem / GitHub / git work is
# delegated to ``account.py`` (see "Project bootstrap helpers" there);
# this module only owns the form fields, validation, and orchestration.

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


@st.dialog("Start a new project", width="large", on_dismiss=_close_new_project_dialog)
def _new_project_dialog() -> None:
    """Modal that creates a project folder, ``git init``s it, and wires an upstream.

    The four upstream modes (None / paste-remote / clone / create-on-GitHub)
    map to short orchestrations against ``account.py``. Errors raised by
    those helpers (validation, network, git failures) surface as a single
    ``st.error`` and leave the dialog open so the user can fix and retry.

    ``on_dismiss=_close_new_project_dialog`` is mandatory so X / Esc /
    click-outside dismissal clears ``ss.new_project_dialog_open``;
    otherwise the modal re-opens on the next rerun (e.g. the next chat
    submission). See :func:`_diff_dialog` for the full rationale.
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


# Material badge color per provider. Used by the model picker rows + the
# inline current-model button. Defaults to gray for unknown providers.
_PROVIDER_BADGE_COLOR: dict[str, str] = {
    "wandb": "blue",
    "openai": "green",
    "anthropic": "orange",
    "gemini": "violet",
    "openrouter": "gray",
    "mistral": "violet",
    "xai": "blue",
}


def _provider_badge(provider_id: str | None) -> str:
    """Return a color-coded markdown badge for a provider id."""
    if not provider_id:
        return ":gray-badge[unknown]"
    provider = providers.get_provider(provider_id)
    label = provider.label if provider else provider_id
    color = _PROVIDER_BADGE_COLOR.get(provider_id, "gray")
    return f":{color}-badge[{label}]"


def _format_price(value: float | None) -> str:
    """Format ``$/M`` value tightly. ``None`` → ``-``."""
    if value is None:
        return "-"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:.2f}"


def _format_context(n: int | None) -> str:
    """Format a context-window int as ``128k`` / ``1M`` etc."""
    if not n:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n // 1000}k"
    return str(n)


def _open_model_picker() -> None:
    """Flip the gating flag so the modal mounts on the next render."""
    st.session_state.model_picker_open = True


def _close_model_picker() -> None:
    """Drop the gating flag so subsequent reruns don't re-mount the modal.

    Wired to ``@st.dialog(..., on_dismiss=_close_model_picker)`` per
    the AGENTS.md @st.dialog rule — without this, dismissing the modal
    via X / Esc / click-outside leaves the gating flag set and the
    modal re-opens on the very next rerun (e.g. when the user presses
    Enter to send a message).
    """
    st.session_state.model_picker_open = False


def _select_model(qualified_id: str) -> None:
    """Apply the picked model to canonical state + close the modal.

    Called from ``_render_model_row`` script-body context (NOT as an
    ``on_click`` callback): an ``@st.dialog`` only actually closes when
    ``st.rerun()`` is invoked from the dialog body, and ``st.rerun()``
    inside ``on_click`` callbacks is a no-op (per AGENTS.md). The caller
    is responsible for calling ``st.rerun()`` immediately after this
    helper returns; we keep the rerun out of here because tests and
    future non-dialog callers might want the state mutation without the
    forced rerun.
    """
    st.session_state.model = qualified_id
    _persist_active_chat_setting("model", qualified_id)
    _close_model_picker()


def _kick_catalog_refresh() -> None:
    """Header Refresh button: kick a daemon refresh + flip the spinner flag."""
    ss = st.session_state
    ss.model_catalog_refreshing = True
    clients = dict(ss.clients)

    def _on_done() -> None:
        # Background-thread callback — touching session_state from
        # here would race; the polling fragment in the chat body
        # picks up completion via ``model_catalog.newest_refresh()``.
        pass

    model_catalog.refresh_all_async(clients, on_done=_on_done)


def _format_last_refreshed() -> str:
    """Return a tiny ``Last refreshed Nm ago`` caption string."""
    last = model_catalog.newest_refresh()
    if last is None:
        return "Never refreshed"
    from datetime import datetime as _dt, timezone as _tz
    delta = _dt.now(_tz.utc) - last
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"Last refreshed {seconds}s ago"
    if seconds < 3600:
        return f"Last refreshed {seconds // 60}m ago"
    if seconds < 86400:
        return f"Last refreshed {seconds // 3600}h ago"
    return f"Last refreshed {seconds // 86400}d ago"


# Picker tabs. Order is the visual order, left to right. The first
# group (Recommended → Coding → Reasoning) is curated; the next
# group (Vision → Image gen → Audio gen → Video gen) is modality-
# based and auto-derived from per-model curated capability flags +
# OpenRouter's architecture-modality arrays for ``openrouter:*``
# ids; the last group (Long context → Fast & cheap → All) is
# generic-utility.
#
# Adding a new tab: append the label to ``_TAB_LABELS`` and (if it
# corresponds to a tag from ``models.ALLOWED_TAGS``) add it to
# ``_TAB_TAG_FOR``. The "All" / "Recommended" tabs have their own
# code paths in ``_filter_by_tab`` and don't use the tag mapping.
_TAB_LABELS = (
    "Recommended",
    "Coding",
    "Reasoning",
    "Vision",
    "Image gen",
    "Audio gen",
    "Video gen",
    "Long context",
    "Fast & cheap",
    "All",
)
_TAB_TAG_FOR: dict[str, str] = {
    "Coding": "coding",
    "Reasoning": "reasoning",
    "Vision": "vision",
    "Image gen": "image_gen",
    "Audio gen": "audio_gen",
    "Video gen": "video_gen",
    "Long context": "long_context",
    "Fast & cheap": "cheap",
}


# Column weights shared between the picker table header and each
# model row so the cells align. ``[Model | Context | $/M in |
# $/M out | Select]``. Skewed heavily toward Model because the
# description is the column users actually search by eye; the
# numeric columns only need room for short formatted values
# (``128k``, ``$0.50``). Keep these in sync with
# ``_render_picker_table_header`` and ``_render_model_row``.
_PICKER_TABLE_WEIGHTS = [8, 1, 1, 1, 1.2]


# Sortable columns: header label + the ``ModelInfo`` attribute the
# sort reads. Add a column by appending here AND widening
# ``_PICKER_TABLE_WEIGHTS`` by one slot.
_PICKER_SORT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("context", "Context", "context"),
    ("input", "$/M in", "input_price_per_1m"),
    ("output", "$/M out", "output_price_per_1m"),
)


def _cycle_picker_sort(column: str) -> None:
    """Cycle the picker's sort state for ``column``: inactive → asc → desc → cleared.

    Wired to each column-header button's ``on_click``. Mutating
    ``ss._model_picker_sort`` from a callback triggers Streamlit's
    automatic rerun, which re-renders the modal body with the new
    order applied via :func:`_apply_picker_sort`.
    """
    ss = st.session_state
    current = ss.get("_model_picker_sort")
    if not current or current.get("column") != column:
        ss._model_picker_sort = {"column": column, "direction": "asc"}
    elif current.get("direction") == "asc":
        ss._model_picker_sort = {"column": column, "direction": "desc"}
    else:
        ss._model_picker_sort = None


def _render_picker_table_header(*, key_prefix: str) -> None:
    """Render the sortable column-header row above each tab body.

    ``key_prefix`` scopes the per-column button keys to the active
    tab (mirrors :func:`_render_model_row`'s Select-button scoping)
    so a model that surfaces in multiple tabs doesn't collide on
    ``StreamlitDuplicateElementKey``. Cell 0 is a plain "Model"
    caption — alphabetical-by-label isn't a useful axis when the
    default order is the curated recommended list. Cells 1-3 are
    ``type="tertiary"`` buttons whose label carries a Material
    arrow when that column is the active sort.
    """
    ss = st.session_state
    sort = ss.get("_model_picker_sort") or {}
    active_column = sort.get("column")
    active_direction = sort.get("direction")

    cols = st.columns(_PICKER_TABLE_WEIGHTS, vertical_alignment="center")
    with cols[0]:
        st.caption("Model")
    for idx, (col_key, col_label, _attr) in enumerate(_PICKER_SORT_COLUMNS, start=1):
        with cols[idx]:
            if col_key == active_column:
                icon = (
                    ":material/arrow_upward:"
                    if active_direction == "asc"
                    else ":material/arrow_downward:"
                )
                btn_label = f"{icon} {col_label}"
            else:
                btn_label = col_label
            st.button(
                btn_label,
                key=f"_picker_sort_{key_prefix}_{col_key}",
                type="tertiary",
                on_click=_cycle_picker_sort,
                args=(col_key,),
                help=f"Sort by {col_label}.",
            )


def _apply_picker_sort(qualified_ids: list[str]) -> list[str]:
    """Apply ``ss._model_picker_sort`` to ``qualified_ids``.

    Returns the input unchanged when no column is active. Otherwise
    reads the configured ``ModelInfo`` attribute and sorts ascending
    or descending. The strict completeness gate guarantees chat
    models always carry context + both axis prices, but we still
    push ``None`` to the end defensively so a future media-mode
    surface in the picker doesn't crash on a missing field.
    """
    ss = st.session_state
    sort = ss.get("_model_picker_sort")
    if not sort:
        return qualified_ids
    column = sort.get("column")
    direction = sort.get("direction", "asc")
    attr = next(
        (a for k, _label, a in _PICKER_SORT_COLUMNS if k == column),
        None,
    )
    if attr is None:
        return qualified_ids
    descending = (direction == "desc")

    def _key(qid: str) -> tuple[int, float | int]:
        info = model_catalog.get_info(qid)
        if info is None:
            return (1, 0)
        value = getattr(info, attr, None)
        if value is None:
            return (1, 0)
        # Negate for descending sort while keeping the missing-value
        # sentinel ``(1, ...)`` after present values regardless of
        # direction.
        return (0, -value if descending else value)

    return sorted(qualified_ids, key=_key)


def _render_model_row(qualified_id: str, *, key_prefix: str) -> None:
    """Render one model row inside the picker modal.

    ``key_prefix`` disambiguates the row's Select button across tabs —
    a model can appear in multiple tabs (e.g. a coding *and* reasoning
    model surfaces in both tab bodies), so the per-row widget key
    needs the tab as a scope qualifier or Streamlit raises a
    ``StreamlitDuplicateElementKey`` error.

    A bordered container shaped as a 5-cell table row matching the
    column headers rendered by :func:`_render_picker_table_header`:
    ``[Model | Context | $/M in | $/M out | Select]``. The Select
    button is ``type="primary"`` for the currently-selected model so
    users can see at a glance what they're already using. Cached-input
    pricing (when set) renders as a small caption directly under the
    Input price — cache-hit pricing is an OpenAI / Anthropic
    convention that applies to *input* tokens, so it belongs under
    the Input column rather than earning its own column.
    """
    info = model_catalog.get_info(qualified_id)
    if info is None:
        # Catalog miss — surface the qualified id with a soft skeleton so
        # the row still renders something. Should be rare because the
        # picker filter only includes ids surfaced by the catalog.
        with st.container(border=True):
            st.caption(f"`{qualified_id}` — metadata not available.")
        return

    is_current = (st.session_state.model == qualified_id)

    with st.container(border=True):
        cols = st.columns(_PICKER_TABLE_WEIGHTS, vertical_alignment="center")
        with cols[0]:
            st.markdown(
                f"{_provider_badge(info.provider_id)} **{info.label}**"
            )
            st.caption(info.description)
        with cols[1]:
            st.markdown(_format_context(info.context))
        with cols[2]:
            st.markdown(f"{_format_price(info.input_price_per_1m)}/M")
            if info.cache_hit_price_per_1m is not None:
                st.caption(
                    f"{_format_price(info.cache_hit_price_per_1m)}/M cached"
                )
        with cols[3]:
            st.markdown(f"{_format_price(info.output_price_per_1m)}/M")
        with cols[4]:
            # NOTE: deliberately uses the "if button: ... st.rerun()"
            # pattern (mirrors ``_mcp_server_dialog`` in settings.py)
            # rather than ``on_click=_select_model``. ``@st.dialog``
            # modals only actually close when ``st.rerun()`` runs in
            # the dialog body — an auto-rerun triggered by an
            # ``on_click`` callback updates state but does NOT close
            # the modal in the user's browser, so picking a model
            # would silently apply the change while leaving the modal
            # stuck open. AGENTS.md's "do not call ``st.rerun`` inside
            # an ``on_click`` callback" rule (the rerun no-ops with a
            # warning) is what forces this body-pattern instead.
            picked = st.button(
                "Selected" if is_current else "Select",
                key=f"_pick_{key_prefix}_{qualified_id}",
                type="primary" if is_current else "secondary",
                disabled=is_current,
                width="stretch",
            )
            if picked:
                _select_model(qualified_id)
                st.rerun()


def _connected_qualified_ids() -> list[str]:
    """Return qualified ids reachable via a currently-connected provider.

    Combines the catalog membership (cleared the completeness gate)
    with per-provider connection status. A provider is "connected"
    when ``ss.provider_models[<id>]`` is non-empty (the connect flow
    populates this with the live raw-id list).
    """
    ss = st.session_state
    available = model_catalog.available_qualified_ids(ss.clients)
    # ``available_qualified_ids`` checks both connectivity (a
    # non-None client object in ``ss.clients[pid]``) and reachability
    # (the model's raw id appears in the live ``/v1/models`` listing).
    # Sort by recommended order if the qualified id is in
    # ``RECOMMENDED_MODELS`` so the more-prominent picks surface
    # first within each tab; otherwise alphabetical.
    rec_order = {qid: i for i, qid in enumerate(RECOMMENDED_MODELS)}
    return sorted(
        available,
        key=lambda qid: (rec_order.get(qid, len(rec_order)), qid.casefold()),
    )


def _filter_by_tab(qualified_ids: list[str], tab_label: str) -> list[str]:
    """Filter a qualified-id list to those tagged for the given tab."""
    if tab_label == "All":
        return qualified_ids
    if tab_label == "Recommended":
        # Intersect RECOMMENDED_MODELS with the connected set.
        connected = set(qualified_ids)
        return [m for m in RECOMMENDED_MODELS if m in connected]
    tag = _TAB_TAG_FOR.get(tab_label, "")
    if not tag:
        return qualified_ids
    out: list[str] = []
    for qid in qualified_ids:
        info = model_catalog.get_info(qid)
        if info is None:
            continue
        if tag in info.tags:
            out.append(qid)
    return out


def _filter_by_search(qualified_ids: list[str], query: str) -> list[str]:
    """Filter by a substring match on label / description / qualified id."""
    q = (query or "").strip().casefold()
    if not q:
        return qualified_ids
    out: list[str] = []
    for qid in qualified_ids:
        if q in qid.casefold():
            out.append(qid)
            continue
        info = model_catalog.get_info(qid)
        if info is None:
            continue
        if q in info.label.casefold() or q in info.description.casefold():
            out.append(qid)
            continue
        # Also match provider label so e.g. searching "anthropic" finds Claude rows.
        provider = providers.get_provider(info.provider_id)
        if provider and q in provider.label.casefold():
            out.append(qid)
    return out


@st.fragment(run_every="0.5s")
def _poll_catalog_refresh() -> None:
    """Poll for catalog-refresh completion and rerun once it lands.

    Runs only while ``ss.model_catalog_refreshing`` is True. The
    background daemon thread can't safely write to session state,
    so we watch ``model_catalog.newest_refresh()`` from this
    polling fragment: when its timestamp advances (or per-source
    errors accumulate), the refresh is done and we flip the flag
    back + rerun so the picker modal re-renders with the new
    catalog.
    """
    ss = st.session_state
    if not ss.get("model_catalog_refreshing"):
        return
    last_seen = ss.get("_model_catalog_refresh_baseline")
    current = model_catalog.newest_refresh()
    if last_seen is None:
        # First poll after kick — record the pre-refresh baseline so
        # we can detect the next bump.
        ss["_model_catalog_refresh_baseline"] = current
        return
    advanced = (current is not None) and (last_seen is None or current > last_seen)
    error_seen = bool(model_catalog.errors())
    if advanced or error_seen:
        ss.model_catalog_refreshing = False
        ss.pop("_model_catalog_refresh_baseline", None)
        st.rerun()


@st.dialog("Choose a model", width="large", on_dismiss=_close_model_picker)
def _model_picker_dialog() -> None:
    """The model-picker modal. Mounted from ``render()``.

    Layout:

    1. Header row: title + Refresh button + last-refreshed caption.
    2. Search box.
    3. Tabs over capability buckets.
    4. Per-tab sortable column-header row + list of model rows.
    5. Footer caption surfacing the count of hidden models.

    Per-source connect / refresh errors are intentionally NOT
    surfaced here — they live on the Settings provider card so the
    user sees the failure where they can fix the credentials and
    the picker stays on-task as a model-selection surface.

    The modal is gated by ``ss.model_picker_open`` and dismissed via
    ``on_dismiss=_close_model_picker`` (per AGENTS.md @st.dialog rule).
    """
    ss = st.session_state

    # Compute the connected qualified-id set once — every tab below
    # filters it, AND the header counts caption reads from it.
    connected = _connected_qualified_ids()
    n_connected = len(connected)
    n_providers = len(set(qid.split(":", 1)[0] for qid in connected))

    # Header row. Title on the left + (counts caption + Refresh button)
    # on the right. The counts caption gives the user an at-a-glance
    # "I have N models from M providers" — important when they've
    # connected a long-tail provider like OpenRouter (300+ models)
    # and the default Recommended tab only shows curated picks.
    header_cols = st.columns([4, 2, 2], vertical_alignment="center")
    with header_cols[0]:
        st.markdown("**Pick a model for your next message.**")
        if n_connected > 0:
            st.caption(
                f"{n_connected} model{'s' if n_connected != 1 else ''} "
                f"available across {n_providers} provider"
                f"{'s' if n_providers != 1 else ''}. The capability tabs "
                f"below show curated picks; **All** lists everything."
            )
    with header_cols[1]:
        st.caption(_format_last_refreshed())
    with header_cols[2]:
        if ss.model_catalog_refreshing:
            st.button(
                "Refreshing...",
                icon=":material/progress_activity:",
                disabled=True,
                width="stretch",
                key="_picker_refresh_disabled",
            )
        else:
            st.button(
                "Refresh",
                icon=":material/refresh:",
                on_click=_kick_catalog_refresh,
                width="stretch",
                key="_picker_refresh_btn",
            )

    # NOTE: per-source connect / refresh errors are intentionally NOT
    # rendered here. They live on the Settings provider card via
    # ``_render_provider_card``'s ``st.error(error)`` so the user sees
    # the failure where they can fix the credentials, and the picker
    # stays on-task as a model-selection surface.

    # Search box.
    st.text_input(
        "Search models",
        key="_model_picker_search",
        placeholder="Search by name, provider, or capability",
        label_visibility="collapsed",
    )

    # ``connected`` is computed at the top of this function; reuse it
    # so we don't double-walk the catalog.
    if not connected:
        st.info(
            "No connected provider has any picker-ready models. "
            "Open **Settings** and add a provider key.",
            icon=":material/info:",
        )
    else:
        searched = _filter_by_search(connected, ss.get("_model_picker_search") or "")

        tabs = st.tabs(list(_TAB_LABELS))
        for tab_idx, tab_label in enumerate(_TAB_LABELS):
            with tabs[tab_idx]:
                tab_ids = _filter_by_tab(searched, tab_label)
                if not tab_ids:
                    if tab_label == "All":
                        st.info(
                            "No models matched your search.",
                            icon=":material/search_off:",
                        )
                    else:
                        # Curated tabs (Recommended / Coding / Reasoning)
                        # only surface models that have those tags in
                        # ``MODEL_METADATA`` — auto-derived tags are
                        # limited to ``long_context`` / ``cheap`` /
                        # ``fast`` / ``multimodal``. So uncurated
                        # providers like OpenRouter (with 300+ models)
                        # never appear in Coding / Reasoning even
                        # though their models can do those things —
                        # we just don't have a verified opinion on
                        # which ones. Point users at **All** so they
                        # don't think their connected provider is
                        # broken.
                        n_total = len(connected)
                        if n_total > 0:
                            st.info(
                                f"This tab shows curated picks tagged "
                                f"**{tab_label.lower()}**. None of your "
                                f"connected providers' curated models "
                                f"carry that tag right now. Click "
                                f"**All** above to browse every "
                                f"available model "
                                f"({n_total} total) — or use the "
                                f"search box to filter by name.",
                                icon=":material/info:",
                            )
                        else:
                            st.info(
                                "No connected provider has any "
                                "picker-ready models yet.",
                                icon=":material/info:",
                            )
                    continue
                # Slugify the tab label into a stable key prefix.
                key_prefix = tab_label.replace(" ", "_").replace("&", "and").lower()
                _render_picker_table_header(key_prefix=key_prefix)
                for qid in _apply_picker_sort(tab_ids):
                    _render_model_row(qid, key_prefix=key_prefix)

    # Footer: hidden-models caption.
    hidden = model_catalog.hidden_models_summary(ss.clients)
    total_hidden = sum(hidden.values())
    if total_hidden > 0:
        breakdown = ", ".join(
            f"{providers.get_provider(pid).label if providers.get_provider(pid) else pid}: {n}"
            for pid, n in sorted(hidden.items())
            if n > 0
        )
        st.caption(
            f":material/visibility_off: {total_hidden} model"
            f"{'s' if total_hidden != 1 else ''} hidden — pricing or "
            f"description not yet verified. Click **Refresh** to retry."
            + (f"  \nBreakdown: {breakdown}" if breakdown else "")
        )


def _render_model_controls() -> None:
    """Render the Mode dropdown + the current-model button + Project context button.

    The model selectbox was replaced with a button that opens the
    ``@st.dialog`` model picker — its label shows the current
    model's provider + label so users see at a glance what they're
    using; clicking opens the modal.

    Column weights ``[1, 5, 2]`` make Mode the smallest cell (it
    only ever shows ``Agent`` / ``Ask only``), give the Model
    button the lion's share (it carries the longest text — provider
    + model name), and reserve a moderate slot for the Project
    context button (now a compact ``N guidance · M skills`` label
    with no "Project context" word).
    """
    ss = st.session_state

    # Dual-key pattern (per ``AGENTS.md``): the Mode dropdown widget
    # binds to ``_chat_mode_input``; the canonical ``ss.mode`` key is
    # non-widget and survives navigation.
    ss["_chat_mode_input"] = ss.mode if ss.mode in ("agent", "ask") else "agent"

    cols = st.columns([1, 5, 2], vertical_alignment="bottom")
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
        # Current-model button. Clicking opens the modal picker.
        any_provider_connected = any(
            bool(ids) for ids in (ss.provider_models or {}).values()
        )
        current_qid = ss.model if isinstance(ss.model, str) else ""
        if not any_provider_connected:
            st.button(
                "Open Settings to add a provider",
                icon=":material/settings:",
                disabled=True,
                width="stretch",
                key="_model_button_disabled",
            )
        else:
            current_provider = model_provider(current_qid)
            label_parts: list[str] = []
            if current_qid:
                label_parts.append(model_label(current_qid))
            else:
                label_parts.append("Choose a model")
            button_label = " ".join(label_parts) if label_parts else "Choose a model"
            help_text = "Click to switch models across all connected providers."
            if current_provider:
                provider = providers.get_provider(current_provider)
                if provider:
                    help_text = f"Currently using {provider.label}. " + help_text
            st.button(
                button_label,
                icon=":material/expand_more:",
                on_click=_open_model_picker,
                width="stretch",
                key="_model_button_open_picker",
                help=help_text,
            )
    with cols[2]:
        _render_project_context_button(ss.working_dir)

    # Inline model-card caption — surfaces the chips below the button so
    # the user has at-a-glance pricing info without opening the modal.
    info = model_catalog.get_info(ss.model) if ss.model else None
    if info is not None:
        chips: list[str] = [f"{_format_context(info.context)} context"]
        chips.append(
            f"{_format_price(info.input_price_per_1m)}/"
            f"{_format_price(info.output_price_per_1m)} per 1M in/out"
        )
        header = f":material/info: **{info.label}**"
        if chips:
            header += " \u00b7 " + " \u00b7 ".join(chips)
        st.caption(f"{header} \u2014 {info.description}")
    else:
        # Fall back to the curated MODEL_METADATA entry when the catalog
        # hasn't seen this id yet (first launch + no refresh).
        meta = MODEL_METADATA.get(ss.model) if ss.model else None
        if meta:
            chips: list[str] = []
            if meta.get("context"):
                chips.append(f"{meta['context']} context")
            if meta.get("params"):
                chips.append(f"{meta['params']} params")
            inp = meta.get("input_price_per_1m")
            out = meta.get("output_price_per_1m")
            if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                chips.append(f"${inp:g}/${out:g} per 1M in/out")
            header = f":material/info: **{meta['label']}**"
            if chips:
                header += " \u00b7 " + " \u00b7 ".join(chips)
            desc = meta.get("description", "")
            st.caption(f"{header} \u2014 {desc}" if desc else header)

    # Soft warning for models documented to mishandle tool calling. These
    # models often *describe* file edits in plain text without ever emitting
    # a structured ``tool_calls`` delta, so the agent loop has nothing to
    # dispatch and the user sees "I'll write the file..." that doesn't
    # actually edit anything. Source-of-truth lives in
    # ``models.weak_tool_calling_issue_url`` so the flag list stays in one
    # place. We deliberately link to the per-model upstream bug rather than
    # recommend specific alternative models — alternative recommendations
    # age poorly as the W&B catalog churns; a public bug URL is durable
    # and lets the user verify the claim themselves. Mode-agnostic on
    # purpose: even Ask mode's read-only flow needs the model to call
    # ``read_file`` / ``list_files`` to be useful.
    issue_url = weak_tool_calling_issue_url(ss.model)
    if issue_url:
        st.caption(
            ":orange[:material/warning: This model often *describes* file "
            f"edits without actually making them ([known issue]({issue_url}))."
            "]"
        )

    # Per-chat model-options toggles. Only rendered when the currently-
    # selected model exposes at least one applicable native feature
    # (so most chats see zero extra UI lines below the model card).
    _render_model_options()


# ---------------------------------------------------------------------------
# Per-chat model-options toggles
# ---------------------------------------------------------------------------
# These render directly under the model card caption. The set of toggles
# depends on the currently-selected model:
#
# - OpenAI o-series (o1, o3, o3-mini, o3-pro, o4, ...): a segmented
#   control for ``reasoning_effort`` (low / medium / high). Default:
#   medium (matches OpenAI's API default). Setting this passes
#   ``reasoning_effort=...`` on the chat completion call.
# - Google Gemini: a toggle for **Google Search grounding**. Off by
#   default. When on, the call config gets a ``Tool(google_search=...)``
#   so the model can hit real-time web at query time. Grounding adds
#   per-search costs above the picker's curated $/M-token figure
#   (~$14/1000 search queries beyond Google's free 5k/month tier).
# - xAI (Grok): a toggle for **Live Search**. Off by default. When on,
#   the OpenAI-SDK call gets ``extra_body={"search_parameters":
#   {"mode": "auto"}}`` so Grok can decide when to hit the real-time
#   web. Live Search has its own per-search costs (xAI publishes
#   those separately).
# - Anthropic Claude: no toggles — prompt caching is auto-applied by
#   ``chat_streams._stream_anthropic_native`` because it's a near-pure
#   win on the agent's long system prompt. Extended thinking is wired
#   as a hook in chat_streams but not surfaced as a UI toggle yet.
#
# All toggles persist per-chat via ``Chat.model_options`` and JSON-
# round-trip through chats.save_chat. Adding a new option is a
# 3-line change here + 1 line in chat_streams + a key in the
# Chat.model_options docstring.
# ---------------------------------------------------------------------------
def _render_model_options() -> None:
    """Render applicable per-chat option toggles inline under the model card.

    Reads ``ss.model`` to figure out the current provider + model id,
    then dispatches per provider. Persists changes to the active
    chat's ``model_options`` dict via :func:`chats.save_chat` so the
    toggle state survives chat switches and process restarts.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not ss.model:
        return

    provider_id = model_provider(ss.model)
    raw_id = ss.model.split(":", 1)[1] if ":" in ss.model else ss.model
    if provider_id is None:
        return

    if provider_id == "openai" and _is_openai_reasoning_model(raw_id):
        _render_openai_reasoning_effort_control(chat, raw_id)
    elif provider_id == "gemini":
        _render_gemini_grounding_toggle(chat)
    elif provider_id == "xai":
        _render_xai_live_search_toggle(chat)


def _is_openai_reasoning_model(raw_id: str) -> bool:
    """Match OpenAI o-series reasoning model ids (o1*, o3*, o4*).

    Mirrors :func:`chat_streams._is_openai_reasoning_model` so we
    surface the toggle for exactly the same model set the dispatch
    layer accepts the kwarg on.
    """
    if not isinstance(raw_id, str) or not raw_id:
        return False
    return raw_id.startswith("o1") or raw_id.startswith("o3") or raw_id.startswith("o4")


_REASONING_EFFORT_OPTIONS: tuple[str, ...] = ("low", "medium", "high")


def _render_openai_reasoning_effort_control(chat: chats.Chat, raw_id: str) -> None:
    """Segmented control for o-series ``reasoning_effort``."""
    current = chat.model_options.get("reasoning_effort", "medium")
    if current not in _REASONING_EFFORT_OPTIONS:
        current = "medium"

    widget_key = f"_reasoning_effort_{chat.id}"
    cols = st.columns([1, 4], vertical_alignment="center")
    with cols[0]:
        st.caption(":material/psychology: Reasoning effort")
    with cols[1]:
        picked = st.segmented_control(
            "Reasoning effort",
            options=list(_REASONING_EFFORT_OPTIONS),
            default=current,
            key=widget_key,
            label_visibility="collapsed",
            help=(
                "How much chain-of-thought the o-series model uses before "
                "answering. **Low** is fast and cheap; **high** is slow but "
                "best on hard problems. Output tokens charged at the model's "
                "standard rate either way."
            ),
        )
    if picked and picked != current:
        opts = dict(chat.model_options)
        opts["reasoning_effort"] = picked
        with chat._lock:
            chat.model_options = opts
        chats.save_chat(chat)


def _render_gemini_grounding_toggle(chat: chats.Chat) -> None:
    """Toggle for Gemini Google Search grounding (per-chat)."""
    current = bool(chat.model_options.get("grounding", False))
    widget_key = f"_grounding_{chat.id}"
    cols = st.columns([1, 4], vertical_alignment="center")
    with cols[0]:
        st.caption(":material/travel_explore: Google Search")
    with cols[1]:
        picked = st.toggle(
            "Google Search grounding",
            value=current,
            key=widget_key,
            label_visibility="collapsed",
            help=(
                "Let Gemini hit Google Search at query time for real-time "
                "facts. **Adds search costs** above Google's free tier "
                "(~$14 per 1000 queries beyond the first 5,000 / month "
                "shared across Gemini 3). When the agent already has "
                "function tools, grounding is silently dropped because "
                "Gemini's API can't combine the two."
            ),
        )
    if picked != current:
        opts = dict(chat.model_options)
        opts["grounding"] = bool(picked)
        with chat._lock:
            chat.model_options = opts
        chats.save_chat(chat)


def _render_xai_live_search_toggle(chat: chats.Chat) -> None:
    """Toggle for xAI Live Search (per-chat)."""
    current = bool(chat.model_options.get("live_search", False))
    widget_key = f"_live_search_{chat.id}"
    cols = st.columns([1, 4], vertical_alignment="center")
    with cols[0]:
        st.caption(":material/travel_explore: Live Search")
    with cols[1]:
        picked = st.toggle(
            "xAI Live Search",
            value=current,
            key=widget_key,
            label_visibility="collapsed",
            help=(
                "Let Grok hit real-time web at query time. **Adds per-search "
                "costs** (xAI publishes those separately from the per-token "
                "rate shown in the picker). Set to ``mode=\"auto\"`` so the "
                "model decides per-turn whether a search is needed."
            ),
        )
    if picked != current:
        opts = dict(chat.model_options)
        opts["live_search"] = bool(picked)
        with chat._lock:
            chat.model_options = opts
        chats.save_chat(chat)


# ---------------------------------------------------------------------------
# Working-tree diff (button + dialog overlay)
# ---------------------------------------------------------------------------
# The chat page is the single rendering site for the live working-tree
# unified diff. Surface area:
#   - The Changes button at the right of ``_render_chat_actions_row``;
#     disabled (rather than hidden) when the working tree is clean /
#     workdir isn't a git repo, so the row layout stays stable.
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


def _safe_path_key(path: str) -> str:
    """Sanitize a path string for use as a Streamlit widget key suffix.

    Paths can contain slashes, dots, and other characters that Streamlit
    accepts in keys but that we don't want littering the rendered key
    string (it shows up in the DOM as the row's ``data-testid`` slug).
    Replace any non-alphanumeric run with a single underscore so the
    resulting key is stable and unique across the per-file ``Discard``
    popover trigger + the inner Confirm button.
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", path).strip("_") or "root"


def _on_discard_one_file(
    working_dir_str: str, entry: git_ops.StatusEntry
) -> None:
    """Per-file Discard confirm callback — runs ``git_ops.discard_changes``.

    Streamlit ``on_click`` callback context: must NOT call ``st.rerun()``
    (Streamlit reruns automatically after callbacks per AGENTS.md). We
    bump the chat-page git nonce so the cached scan re-reads the
    working tree on the next render, and toast the result. ``st.toast``
    is allowed inside callbacks (only ``st.rerun`` / ``st.switch_page``
    are forbidden).
    """
    if not working_dir_str:
        return
    working_dir = Path(working_dir_str).expanduser().resolve()
    try:
        git_ops.discard_changes(working_dir, entry)
    except GitError as e:
        _toast_git_error("git restore failed", e)
        return
    _bump_chat_git_nonce()
    label = entry.orig_path if (entry.is_renamed and entry.orig_path) else entry.path
    st.toast(
        f"Discarded changes to `{label}`",
        icon=":material/undo:",
    )


def _on_discard_all_files(
    working_dir_str: str, entries: list[git_ops.StatusEntry]
) -> None:
    """Top-of-modal "Discard all" confirm callback — bulk discard.

    Routes through :func:`git_ops.discard_all_changes` which collapses
    the per-entry work into one ``git restore`` + one ``git clean``
    call so reverting a large working tree stays fast even on the
    on-click path. Same callback-context rules as
    :func:`_on_discard_one_file` (no ``st.rerun()``; ``st.toast`` OK).
    """
    if not working_dir_str or not entries:
        return
    working_dir = Path(working_dir_str).expanduser().resolve()
    try:
        git_ops.discard_all_changes(working_dir, entries)
    except GitError as e:
        _toast_git_error("git restore failed", e)
        return
    _bump_chat_git_nonce()
    n = len(entries)
    st.toast(
        f"Discarded changes to {n} file{'s' if n != 1 else ''}",
        icon=":material/undo:",
    )


def _render_diff_file_section(
    entry: git_ops.StatusEntry,
    counts: dict[str, tuple[int, int]],
    working_dir: Path,
    working_dir_str: str,
) -> None:
    """Render one file as a collapsed-by-default ``st.expander`` + Discard.

    Layout: a 2-column row whose left cell holds the expander (path +
    +adds/−dels chip + state badge in the label, unified diff in the
    body when expanded) and whose right cell holds a ``st.popover``
    trigger labeled **Discard** that opens a small confirmation pane
    inline. The popover is the inline-confirm pattern: a warning + a
    primary-styled **Confirm** button whose ``on_click`` runs
    ``git_ops.discard_changes`` against this single entry. We use a
    popover (rather than a second ``@st.dialog`` for the confirmation,
    or a 2-step button-label-changes-on-click pattern) because:

    1. Streamlit doesn't support nested ``@st.dialog``s — the Changes
       modal is itself a dialog, so a nested confirm dialog isn't an
       option.
    2. ``st.popover`` is exactly the right shape — a small overlay
       anchored to the trigger button, dismissible via click-outside
       — so the user gets a "are you sure?" step without losing
       sight of the file they're about to discard, and without
       reflowing the rest of the modal.
    3. The 2-step button approach (first click changes label to
       "Confirm?") doesn't read as confirmation in screenshots / docs
       and is easy to mis-click.

    We deliberately render every file collapsed (``expanded=False``)
    because long working-tree diffs were dominating the modal — the
    user explicitly asked for "collapsed by default" so the dialog
    reads as a list of files first, with the line-by-line walk
    available on demand. The discard button stays visible regardless
    of expander state because it's outside the expander.
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

    # 8/2 column split: the expander (with its diff body) gets the
    # bulk of the width so unified-diff lines stay readable when the
    # user expands a file; the Discard popover trigger sits on the
    # right end of the row at a comfortable touch target. The popover
    # body (which includes the warning + Confirm button) renders as a
    # floating overlay anchored to the trigger button — it doesn't
    # occupy column space.
    cols = st.columns([8, 2], vertical_alignment="top")
    with cols[0]:
        with st.expander(label, expanded=False):
            diff = _diff_for_entry(working_dir, entry)
            if diff.strip():
                st.code(diff, language="diff")
            else:
                st.caption("(no textual diff — likely a binary file)")
    with cols[1]:
        path_key = _safe_path_key(entry.path)
        with st.popover(
            "Discard",
            icon=":material/undo:",
            help="Discard changes to this file (cannot be undone)",
            width="stretch",
        ):
            display_path = (
                f"`{entry.orig_path}` → `{entry.path}`"
                if (entry.is_renamed and entry.orig_path)
                else f"`{entry.path}`"
            )
            st.markdown(f"Discard changes to {display_path}?")
            st.warning(
                "This permanently removes the changes from your working tree.",
                icon=":material/warning:",
            )
            st.button(
                "Confirm",
                icon=":material/undo:",
                type="primary",
                width="stretch",
                key=f"diff_dlg_discard_confirm_{path_key}",
                on_click=_on_discard_one_file,
                args=(working_dir_str, entry),
            )


def _open_diff_dialog() -> None:
    """Changes-button callback: flip the dialog open."""
    st.session_state.diff_dialog_open = True


def _close_diff_dialog() -> None:
    """Drop the dialog flag so subsequent reruns don't re-mount the modal."""
    st.session_state.diff_dialog_open = False


@st.dialog("Changes", width="large", on_dismiss=_close_diff_dialog)
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

    ``on_dismiss=_close_diff_dialog`` is mandatory: without it, when
    the user closes the modal via the built-in X / Esc / click-outside
    affordances, ``ss.diff_dialog_open`` stays ``True`` and the dialog
    re-opens on the very next rerun (e.g. when the user presses Enter
    to send a chat message). Streamlit calls the dismiss callback
    before the next rerun, so clearing the flag there guarantees the
    "X-then-send-a-message" flow doesn't loop the user back into the
    Changes modal.
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
    if not entries:
        st.caption(f"Chat: **{chat_label}** \u00b7 branch `{branch}`")
        st.success(
            "Working tree clean. Nothing to push.",
            icon=":material/check_circle:",
        )
        if st.button("Close", key="diff_dlg_close_clean", width="stretch"):
            _close_diff_dialog()
            st.rerun()
        return

    # Header row: caption on the left, "Discard all" popover on the
    # right. Column ratio [6, 2] keeps the caption comfortable on
    # typical chat titles while leaving the right column wide enough
    # for the labeled "Discard all" trigger button at width="stretch".
    # Using `vertical_alignment="center"` so the caption baseline lines
    # up with the button regardless of the caption's natural height.
    file_count = f"{len(entries)} file{'s' if len(entries) != 1 else ''} changed"
    header_cols = st.columns([6, 2], vertical_alignment="center")
    with header_cols[0]:
        st.caption(
            f"Chat: **{chat_label}** \u00b7 branch `{branch}` \u00b7 {file_count}"
        )
    with header_cols[1]:
        with st.popover(
            "Discard all",
            icon=":material/undo:",
            help="Discard every change in this list (cannot be undone)",
            width="stretch",
        ):
            st.markdown(f"Discard all **{file_count}**?")
            st.warning(
                "This permanently removes every change from your working "
                "tree. There is no in-product undo.",
                icon=":material/warning:",
            )
            st.button(
                "Confirm",
                icon=":material/undo:",
                type="primary",
                width="stretch",
                key="diff_dlg_discard_all_confirm",
                on_click=_on_discard_all_files,
                args=(working_dir_str, entries),
            )

    working_dir_resolved = working_dir.resolve()
    counts = git_ops.summary_diff_counts(
        working_dir_resolved, [e.path for e in entries if not e.is_untracked]
    )

    # Each file renders as its own collapsed-by-default `st.expander`
    # plus an inline ``Discard`` popover via `_render_diff_file_section`,
    # so the dialog body reads as a compact list of file headers up
    # front; the user expands the ones they actually want to inspect
    # (or clicks Discard to revert without expanding). Expanders sit
    # flush against each other — no `st.divider()` between them — so
    # the list stays tight even with many files.
    for entry in entries:
        _render_diff_file_section(
            entry, counts, working_dir_resolved, working_dir_str
        )

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
# Bottom-of-chat git row: branch picker (with new-branch + fetch
# sentinels) + Changes + Sync
# ---------------------------------------------------------------------------
# Rendered directly below the chat input. Owns the per-chat branch
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

# Sentinel names used in the branch dropdown to fold "create a new
# branch" and "fetch upstream branches" into the same control the user
# is already looking at — no separate icon buttons. Selecting either
# sentinel runs its action and reverts the selectbox to the live
# current branch so the sentinel string never sticks. Plain text
# rather than a `:material/...:` token because Streamlit's selectbox
# option labels render as raw strings — Material icon tokens only
# render inside `icon=` props on buttons / sidebar items / similar,
# not inside selectbox options.
#
# Both sentinels share the same leading ``›`` glyph as the workdir
# sentinels above (see ``_WD_BROWSE_SENTINEL``) so all four pinned
# action items in the bottom-of-chat actions row read as a visually
# distinct group when the dropdown is open.
_NEW_BRANCH_SENTINEL = "›  New branch..."
_FETCH_BRANCHES_SENTINEL = "›  Fetch upstream branches"
_BRANCH_SENTINELS: tuple[str, ...] = (
    _NEW_BRANCH_SENTINEL,
    _FETCH_BRANCHES_SENTINEL,
)

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


def _resolve_commit_ai_model(chat_model: str) -> tuple[Any | None, str | None, str | None]:
    """Resolve the chat's qualified model id for the Sync pipeline.

    Sync's commit-message + PR-description steps run :mod:`commit_ai`,
    which routes through :func:`agent.generate_text` and dispatches by
    qualified model id. To call them we need three things in sync:

    1. The qualified model id (``<provider>:<raw>``) — comes straight
       from ``chat.model``.
    2. The matching per-provider client object —
       ``ss.clients[provider_id]`` after a successful Connect.
    3. A user-friendly error string when either is missing, so the
       Sync button's tooltip and the pipeline's toast can explain
       *why* AI commits aren't ready (rather than blowing up later
       with a generic SDK exception).

    Returns ``(client, qualified_model, error)``:

    - On success, ``(client, qualified_model, None)``.
    - On any failure (blank model, malformed id, unknown provider,
      provider not connected), ``(None, None, error_msg)``.

    The error strings are user-facing — they reference the provider's
    display label (from :data:`providers.PROVIDERS`) so a user who
    picked an Anthropic model sees "Connect Anthropic..." rather
    than the cryptic "anthropic" id.
    """
    if not chat_model:
        return None, None, (
            "Pick a model in the dropdown below the chat input to "
            "enable AI-drafted commit messages."
        )
    provider_id = model_provider(chat_model)
    if not provider_id:
        # Bare ids (no ``provider:`` prefix) are legacy / migration
        # state. Surface a clear message rather than guessing a
        # provider — the user can pick from the model picker to fix.
        return None, None, (
            f"Model `{chat_model}` is not in qualified `<provider>:<model>` "
            "form. Open the model picker below the chat input to pick one."
        )
    provider = providers.get_provider(provider_id)
    if provider is None:
        return None, None, (
            f"Unknown provider id `{provider_id}`. Open the model picker "
            "to pick a model from a connected provider."
        )
    client = st.session_state.clients.get(provider_id)
    if client is None:
        return None, None, (
            f"Connect {provider.label} on the Settings page so Sync "
            "can use this chat's model to draft the commit message."
        )
    return client, chat_model, None


def _on_chat_git_branch_change(chat_id: str) -> None:
    """Branch-selectbox callback inside the bottom-of-chat git row.

    Three branches:

    1. ``_NEW_BRANCH_SENTINEL`` — revert the selectbox to the live
       branch (so the sentinel string doesn't stick) and open the new-
       branch modal.
    2. ``_FETCH_BRANCHES_SENTINEL`` — revert the selectbox, run
       ``git fetch origin`` in the active chat's workdir, bump the
       git nonce so the next render's branch dropdown reflects newly
       fetched remote branches (and any deleted on origin disappear),
       and toast on success / error.
    3. A real branch name — fall through to ``git_ops.checkout`` for
       both local and remote-tracking names (git resolves
       ``origin/feature`` -> a new local tracking branch
       automatically), bump the nonce, and toast on success. Don't
       refuse on a dirty working tree: ``git checkout <branch>``
       natively carries uncommitted changes when there's no conflict
       with the target branch, which matches what most users mean by
       "switch branches and bring my work along". When there *is* a
       real conflict (the target branch would overwrite a dirty file),
       git fails with a clear stderr that the GitError handler below
       surfaces verbatim.
    """
    ss = st.session_state
    select_key = f"chat_bottom_branch_select_{chat_id}"
    sentinel_key = f"_chat_bottom_branch_active_{chat_id}"
    chosen = ss.get(select_key)
    chat = ss.chats.get(chat_id)
    if not chosen or chat is None or not chat.working_dir:
        return

    if chosen == _NEW_BRANCH_SENTINEL:
        ss[select_key] = ss.get(sentinel_key) or ss.get(select_key)
        ss.new_branch_dialog_open = True
        return

    if chosen == _FETCH_BRANCHES_SENTINEL:
        ss[select_key] = ss.get(sentinel_key) or ss.get(select_key)
        working_dir = Path(chat.working_dir).expanduser().resolve()
        try:
            git_ops.fetch(working_dir)
        except GitError as e:
            _toast_git_error("git fetch failed", e)
            return
        _bump_chat_git_nonce()
        st.toast("Fetched from origin", icon=":material/cloud_download:")
        return

    if chosen == ss.get(sentinel_key):
        return

    working_dir = Path(chat.working_dir).expanduser().resolve()
    try:
        git_ops.checkout(working_dir, chosen)
    except GitError as e:
        _toast_git_error("git checkout failed", e)
        ss[select_key] = ss.get(sentinel_key)
        return
    ss[sentinel_key] = chosen
    _bump_chat_git_nonce()
    st.toast(f"Switched to `{chosen}`", icon=":material/check_circle:")


def _on_chat_git_new_branch_clicked() -> None:
    """**New branch** sentinel callback: open the modal."""
    st.session_state.new_branch_dialog_open = True


def _on_chat_git_sync_clicked() -> None:
    """Sync button callback: enqueue a bidirectional sync, or open first-push.

    Sync is bidirectional: if the working tree is dirty, draft a
    commit message via the chat's currently selected model, stage,
    commit; always fetch; pull ``--rebase`` if behind upstream
    (with the existing merge-conflict
    handoff); push if ahead. When the branch has no upstream yet, open
    the **Publish branch** modal first so the user can opt into a PR.

    - With an upstream: enqueue a one-click sync (no PR) for the
      current script run to consume after the rerun. We can't run the
      pipeline directly inside this on-click callback because callbacks
      fire *before* the script runs; doing the pipeline here would
      block the callback for the duration of a network round-trip + an
      LLM call, and any exception would render through Streamlit's
      ugly uncaught-exception traceback. Routing through the
      ``pending_sync_request`` queue keeps the pipeline inside
      ``render()`` where ``st.toast`` / ``st.rerun`` work normally.
    - Without an upstream + dirty or already-committed work locally:
      open the "Publish branch" modal so the user can opt into a PR.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        return
    working_dir = Path(chat.working_dir).expanduser().resolve()
    branch = git_ops.current_branch(working_dir)
    if not branch:
        st.toast(
            "Cannot sync from a detached HEAD. Check out a branch first.",
            icon=":material/error:",
        )
        return
    if git_ops.has_upstream(working_dir, branch):
        ss.pending_sync_request = {"create_pr": False}
    else:
        ss.first_push_dialog_open = True


def _run_sync_pipeline(
    working_dir: Path,
    branch: str,
    *,
    create_pr: bool,
) -> None:
    """One-click bidirectional sync: commit (if dirty) -> fetch -> pull --rebase (if behind) -> push (if ahead).

    The sync state machine:

    1. Refuse if a merge/rebase is in progress (the sidebar's existing
       conflict-resolution flow has to finish first).
    2. If the working tree is dirty: resolve the chat's selected model
       + matching per-provider client via
       :func:`_resolve_commit_ai_model`, ask **that** model for a
       commit message, stage all dirty paths, and commit. The chat's
       selected model (Anthropic / OpenAI / Google / W&B / etc.) is
       used here — not a hard-coded provider — so users who picked a
       non-W&B model don't have to also keep DeepSeek connected.
       Model resolution is only required when there's something to
       commit; pull-only / push-only syncs work without it.
    3. Always ``git fetch origin`` (best-effort; failure is non-fatal).
    4. If the branch has an upstream and is behind: ``git pull --rebase``.
       On a real merge conflict we set ``ss.merge_conflict`` and hand
       off to the sidebar's existing "Resolve with DeepSeek" affordance
       (the conflict-resolution turn still pins to DeepSeek because
       it's a multi-step agent loop with its own model contract).
    5. If the branch has an upstream and is ahead: ``git push``. (When
       the working tree was dirty in step 2, the new commit is what
       makes us ahead, so this is the same code path the old "Push"
       button took.)
    6. When ``create_pr=True`` (set by the Publish branch dialog on a
       branch with no upstream), generate PR title + body via the
       chat's selected model and open the platform compare URL.

    All dirty paths (tracked + untracked) are staged unconditionally;
    file-by-file selection is deliberately not exposed.
    """
    ss = st.session_state

    in_progress, op = git_ops.is_in_merge_or_rebase(working_dir)
    if in_progress:
        st.toast(
            f"In-progress `{op or 'merge'}` — resolve before syncing "
            "(see the warning in the sidebar).",
            icon=":material/warning:",
        )
        return

    try:
        entries = git_ops.status_entries(working_dir)
    except GitError as e:
        _toast_git_error("git status failed", e)
        return
    paths = [e.path for e in entries]
    dirty = bool(paths)

    # Resolve the chat's selected model + matching per-provider client
    # for AI-drafted commit messages and PR descriptions. Sync only
    # *requires* this when the working tree is dirty (the pull-only /
    # push-only paths work without an LLM round-trip); we still
    # resolve it up front so the create-PR step downstream — which
    # may run on a clean tree — can also reach for it.
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    chat_model_str = chat.model if chat else ""
    commit_client, commit_model, commit_model_error = _resolve_commit_ai_model(
        chat_model_str
    )

    commit_msg = ""
    if dirty:
        if commit_model_error is not None:
            st.toast(commit_model_error, icon=":material/error:")
            return

        # Pre-flight: build the diff blob ourselves so we can detect the
        # "dirty paths but no textual diff" case (submodules with only
        # internal changes, binary file changes, mode-only changes,
        # etc.) BEFORE calling the model. ``commit_ai.generate_commit_message``
        # short-circuits to ``""`` when the diff is empty and we'd
        # surface a misleading "model did not return" toast otherwise —
        # the model wasn't even called. Building it here costs nothing
        # extra because ``commit_ai.generate_commit_message`` would do
        # the same call internally; the diff is small relative to the
        # subsequent LLM round-trip.
        diff_blob = git_ops.combined_diff_for_paths(working_dir, paths)
        if not diff_blob.strip():
            # Most common cause is a submodule with only working-tree
            # changes (parent shows ` M <submodule>` but `git diff
            # HEAD` produces nothing because the submodule's commit
            # hash hasn't changed). Other causes: binary-only diffs,
            # mode-only changes, untracked files git can't read as
            # text. The `git commit` CLI rejects this state too — we
            # surface a clear toast rather than blame the model.
            sample_paths = ", ".join(f"`{p}`" for p in paths[:3])
            more = f" and {len(paths) - 3} more" if len(paths) > 3 else ""
            st.toast(
                "Nothing to commit from the working tree's textual "
                "diff (paths: " + sample_paths + more + "). This "
                "usually means a submodule with only internal "
                "changes, a binary-only change, or a mode-only "
                "change — none of which produce a textual diff for "
                "the model to summarize. Commit those directly with "
                "git or resolve the submodule first.",
                icon=":material/info:",
            )
            return

        st.toast("Generating commit message...", icon=":material/auto_awesome:")
        try:
            commit_msg = commit_ai.generate_commit_message(
                commit_client,
                working_dir,
                paths,
                model=commit_model,
            )
        except Exception as e:
            st.toast(
                f"Commit message generation failed: {e}",
                icon=":material/error:",
            )
            return
        commit_msg = (commit_msg or "").strip()
        if not commit_msg:
            # This branch is now narrowly "the model returned an
            # empty response despite a non-empty diff" — a real
            # model failure rather than a misleading
            # nothing-to-summarize misclassification (the empty-diff
            # case is caught by the pre-flight above).
            st.toast(
                "The model returned an empty commit message. Try "
                "again, or pick a different model in the chat input.",
                icon=":material/error:",
            )
            return

    pr_title = ""
    pr_body = ""
    if create_pr:
        # PR title/body uses the *current* dirty diff if there is one,
        # otherwise the diff of HEAD vs the default branch (covers the
        # "I committed earlier and now want to publish" case). Reuses
        # the same ``commit_client`` / ``commit_model`` we resolved
        # above; if model resolution failed we surface the same toast
        # rather than silently falling back to a blank PR body.
        if commit_model_error is not None:
            st.toast(commit_model_error, icon=":material/warning:")
        else:
            st.toast(
                "Generating pull request title and body...",
                icon=":material/auto_awesome:",
            )
            try:
                target = git_ops.default_branch(working_dir)
                pr_title, pr_body = commit_ai.generate_pr_description(
                    commit_client,
                    working_dir,
                    paths,
                    branch,
                    target,
                    model=commit_model,
                )
            except Exception as e:
                st.toast(
                    f"Pull request description generation failed: {e}",
                    icon=":material/warning:",
                )

    if dirty:
        try:
            git_ops.unstage_all(working_dir)
            git_ops.stage(working_dir, paths)
            git_ops.commit(working_dir, commit_msg)
            _bump_chat_git_nonce()
        except GitError as e:
            _toast_git_error("commit failed", e)
            return

    # Always fetch so behind/ahead checks below see fresh refs. We
    # also bump the git-state nonce on success so the branch
    # dropdown above the Sync button picks up any newly-fetched
    # remote branches (and any branches `--prune` just dropped) on
    # the same rerun — pulling the per-rerun "fetch" affordance into
    # Sync means users who hit Sync expecting the branch list to
    # stay current actually get that.
    fetched = False
    try:
        git_ops.fetch(working_dir)
        fetched = True
        _bump_chat_git_nonce()
    except GitError:
        # Fetch failures aren't fatal here; push/pull will surface
        # the same condition with a clearer error if it matters.
        pass

    has_upstream = git_ops.has_upstream(working_dir, branch)
    pulled = False
    if has_upstream:
        try:
            if git_ops.is_behind_upstream(working_dir):
                pull = git_ops.pull_rebase(working_dir)
                _bump_chat_git_nonce()
                pulled = pull.ok
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

    # Decide whether to push. With an upstream, only push when ahead;
    # without an upstream (first publish), push unconditionally so we
    # can set the upstream on origin in the same call.
    pushed = False
    push_result = None
    if has_upstream:
        if git_ops.is_ahead_of_upstream(working_dir):
            push_result = git_ops.push(working_dir, branch=branch)
            _bump_chat_git_nonce()
            if not push_result.ok:
                st.toast(
                    f"Push failed: {push_result.stderr.strip() or 'unknown error'}",
                    icon=":material/error:",
                )
                return
            pushed = True
    else:
        # No upstream + create_pr=True means the user opened the
        # Publish branch dialog explicitly and confirmed; always push
        # in that case. Otherwise (sync clicked on a no-upstream
        # branch with a clean tree), the on-click handler already
        # routes to the publish dialog rather than enqueueing a sync,
        # so this branch is reached after a confirmed publish or after
        # we just committed in step 2 above.
        push_result = git_ops.push(working_dir, branch=branch)
        _bump_chat_git_nonce()
        if not push_result.ok:
            st.toast(
                f"Push failed: {push_result.stderr.strip() or 'unknown error'}",
                icon=":material/error:",
            )
            return
        pushed = True

    if not (dirty or pulled or pushed):
        if fetched:
            st.toast(
                f"Fetched from origin — already in sync with `origin/{branch}`.",
                icon=":material/check_circle:",
            )
        else:
            st.toast(
                f"Already in sync with `origin/{branch}`. Nothing to do.",
                icon=":material/check_circle:",
            )
        return

    parts: list[str] = []
    if dirty:
        short_msg = commit_msg.splitlines()[0][:60] if commit_msg else "committed"
        parts.append(f"committed: {short_msg}")
    if pulled:
        parts.append("pulled upstream")
    if pushed:
        parts.append(f"pushed `{branch}`")
    st.toast(
        "Sync done — " + " · ".join(parts),
        icon=":material/check_circle:",
    )

    if create_pr and push_result is not None:
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


@st.dialog("New branch", width="small", on_dismiss=_close_new_branch_dialog)
def _new_branch_dialog() -> None:
    """Modal for creating a new branch off HEAD.

    Uses ``git checkout -b`` so any uncommitted working-tree changes
    come along with the new branch — matches what most users mean
    when they say "let me put this on a new branch first".
    Validation errors render inline; the dialog stays open so the
    user can fix and retry.

    ``on_dismiss=_close_new_branch_dialog`` is mandatory so X / Esc /
    click-outside dismissal clears ``ss.new_branch_dialog_open``;
    otherwise the modal re-opens on the next rerun (e.g. the next chat
    submission). See :func:`_diff_dialog` for the full rationale.
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


@st.dialog("Publish branch", width="medium", on_dismiss=_close_first_push_dialog)
def _first_push_dialog() -> None:
    """Modal shown on the first push of a branch (no upstream yet).

    Two-option radio: just push the branch, or push and open a
    pre-filled pull-request draft. Selecting the PR option triggers
    an extra round-trip to the chat's currently selected model to
    draft a title + body, then opens the GitHub compare URL with
    both pre-filled — the user can review and edit before submitting
    on the platform side.

    ``on_dismiss=_close_first_push_dialog`` is mandatory so X / Esc /
    click-outside dismissal clears ``ss.first_push_dialog_open``;
    otherwise the modal re-opens on the next rerun (e.g. the next chat
    submission). See :func:`_diff_dialog` for the full rationale.
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
    ss.pending_sync_request = {"create_pr": create_pr}
    _close_first_push_dialog()
    st.rerun()


def _drain_pending_sync_request() -> None:
    """Run any queued sync pipeline. Called once per ``render()``.

    Sync callbacks (the actions-row Sync button + the Publish-branch
    modal Confirm button) drop their intent into
    ``ss.pending_sync_request`` instead of running the pipeline
    inline; this drain step turns that into a single in-render call
    so :func:`st.toast` / :func:`st.rerun` work normally and any
    LLM / git failure surfaces through the toast layer rather than a
    Streamlit uncaught-exception trace.
    """
    ss = st.session_state
    req = ss.pop("pending_sync_request", None)
    if not req:
        return
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or not chat.working_dir:
        return
    working_dir = Path(chat.working_dir).expanduser().resolve()
    branch = git_ops.current_branch(working_dir)
    if not branch:
        st.toast(
            "Cannot sync from a detached HEAD.",
            icon=":material/error:",
        )
        return
    _run_sync_pipeline(working_dir, branch, create_pr=bool(req.get("create_pr")))


# ---------------------------------------------------------------------------
# Stop button: cancel an in-flight turn on demand
# ---------------------------------------------------------------------------
# Rendered inside the same ``st.container()`` as the chat input (just
# above it) so the user's eye lands on the stop affordance the moment
# they look at the disabled chat input. Hidden entirely when the
# active chat is not running, which matches the only state where
# cancellation is meaningful. The callback delegates to
# :func:`chats.request_cancel`, which is the **only** entry point in
# the app for asking the runner to stop — see ``AGENTS.md``'s
# anti-duplication checklist entry on cancellation.
def _on_chat_stop_clicked() -> None:
    """Stop button callback: signal the active chat's runner to abort.

    No-op when no chat is active or the chat is not currently running
    a turn (defensive: the button is hidden in those states, but
    Streamlit's button-click latency means a user could conceivably
    double-click during the brief window between the runner finishing
    and the rerun re-evaluating the button's visibility). The
    underlying ``chats.request_cancel`` is itself idempotent.

    Surfaces a toast so the user gets immediate feedback even before
    the runner has had a chance to flip status — the live fragment
    polls every 250ms, so there's a brief perceptual gap between
    click and the chat input re-enabling without the toast.
    """
    ss = st.session_state
    chat = ss.chats.get(ss.active_chat_id) if ss.active_chat_id else None
    if chat is None or chat.status != chats.STATUS_RUNNING:
        return
    chats.request_cancel(chat)
    st.toast("Stopping the current turn...", icon=":material/stop_circle:")


def _render_chat_stop_button(chat: chats.Chat | None) -> None:
    """Render the Stop button when the active chat has a turn in flight.

    Hidden entirely when the active chat is not running. Clicking
    signals the runner via :func:`chats.request_cancel`; the runner
    closes the streaming W&B Inference connection (so the server
    stops generating tokens), preserves any partial reply as a
    proper assistant_text event, appends a "Stopped by user" marker,
    and flips status back to IDLE so the live fragment stops polling
    and the chat input re-enables.

    Callers pass the already-resolved ``chat`` to avoid a second
    ``ss.chats.get(ss.active_chat_id)`` round trip — render() does
    that lookup once at the top of the page.
    """
    if chat is None or chat.status != chats.STATUS_RUNNING:
        return
    st.button(
        "Stop",
        icon=":material/stop:",
        type="primary",
        key="chat_stop_btn",
        on_click=_on_chat_stop_clicked,
        width="stretch",
        help=(
            "Stop the current turn. The model stops immediately so you "
            "don't spend more tokens, and any partial reply is kept in "
            "the chat history."
        ),
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

    # Drain any sync pipeline that a callback queued before this rerun.
    # Has to run after the active-chat sync above (so it sees the
    # right working_dir) but before the bottom controls render so the
    # toast for "synced!" / "rebase conflict" appears on the same
    # rerun rather than the next one, and so the actions row below
    # picks up the post-sync branch / dirty state on the same render.
    _drain_pending_sync_request()

    # Chat input is wrapped in its own ``st.container()`` so Streamlit
    # renders it inline (rather than docking it to the viewport bottom
    # via the default top-level chat-input behaviour) — the *whole*
    # control stack below the chat history needs to stay together,
    # not the chat input alone. The Stop button is rendered inside the
    # same container, just above the input, so the user's eye lands
    # on the cancel affordance immediately when they look at the
    # disabled chat input. The placeholder text also flips while a
    # turn is in flight so it's obvious why typing is locked out.
    chat_running = chat.status == chats.STATUS_RUNNING
    chat_input_placeholder = (
        "Click Stop above to interrupt the running turn..."
        if chat_running
        else "Ask the agent to read or modify your code..."
    )
    with st.container():
        _render_chat_stop_button(chat)
        # Inject the paperclip + outlined-button styling for Streamlit's
        # built-in attach button. Done inside the chat-input container
        # so the rule sits next to the element it targets.
        st.html(_CHAT_INPUT_ATTACH_BUTTON_CSS)
        # Phase 6: ``accept_file="multiple"`` turns the chat input
        # into a ``ChatInputValue`` dict-like with ``text`` and
        # ``files: list[UploadedFile]`` keys. We intentionally do **not**
        # pass ``file_type=...`` here: Streamlit surfaces that list as
        # a tooltip on the attach button which clutters the chat UI.
        # The OS picker accepts anything; we validate against
        # ``_SUPPORTED_FILE_EXTENSIONS`` after submit and surface a
        # clear ``st.error`` listing the supported types when the user
        # picks something we don't handle.
        chat_submission = st.chat_input(
            chat_input_placeholder,
            disabled=(not wd_ok) or chat_running,
            accept_file="multiple",
        )
        # Normalize: ``ChatInputValue`` is a dict-like for new-style
        # submissions; ``str`` for the legacy single-text path. We
        # keep both call sites simple by extracting (text, files).
        prompt: str | None = None
        submitted_files: list[Any] = []
        if isinstance(chat_submission, dict):
            prompt = (chat_submission.get("text") or "").strip() or None
            submitted_files = list(chat_submission.get("files") or [])
        elif isinstance(chat_submission, str):
            prompt = chat_submission
        elif chat_submission is not None and hasattr(chat_submission, "text"):
            # ChatInputValue object form
            prompt = (getattr(chat_submission, "text", "") or "").strip() or None
            submitted_files = list(getattr(chat_submission, "files", None) or [])

    # Mount the chat-input enhancer on every render. Two responsibilities:
    # (1) slash-command autocomplete (filters the skills list as the
    # user types ``/<query>``), and (2) per-chat draft persistence — the
    # JS restores ``chat.draft_text`` into the textarea on chat switch /
    # first attach, and pushes typing back via ``setStateValue`` so
    # ``_on_draft_change`` can save it to disk on the next rerun. We
    # mount unconditionally (rather than only when ``autocomplete_skills``
    # is non-empty as the historical code did) because draft persistence
    # has to work even in projects with no configured skills, and the
    # autocomplete dropdown silently shows "No matching skills" in that
    # case which is the right degradation.
    summary = _scan_project_summary(ss.working_dir or "")
    autocomplete_skills = summary.get("all_skills", []) or []
    mount_slash_autocomplete(
        autocomplete_skills,
        placeholder_hint=(
            "Try a different prefix, or open the Skills popover below "
            "for the full list."
        ),
        chat_id=chat.id,
        draft=chat.draft_text or "",
        on_draft_change=_on_draft_change,
    )

    # Single combined actions row directly below the chat input:
    # workdir picker (with browse + new-project sentinels) + branch
    # picker (with new-branch + fetch sentinels) + Changes + Sync.
    # Cells degrade to disabled (rather than hidden) when the workdir
    # isn't a git repo / there are no changes / git is missing, so
    # the column layout stays stable across workdir switches.
    _render_chat_actions_row()

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

    # Validate uploaded file extensions against the curated allow-
    # list. We surface unsupported types here (rather than in the OS
    # picker via ``file_type=``) so the chat input's attach button
    # doesn't render a long tooltip listing every supported
    # extension. ``submission_valid`` gates the actual turn-start
    # below; the dialog mounts at the bottom of ``render()`` still
    # need to run so we don't ``return`` early on a validation error.
    submission_valid = True
    if submitted_files:
        unsupported = [
            f for f in submitted_files
            if Path(getattr(f, "name", "") or "").suffix.lstrip(".").lower()
            not in _SUPPORTED_FILE_EXTENSIONS
        ]
        if unsupported:
            bad_names = ", ".join(
                f"`{getattr(f, 'name', '?')}`" for f in unsupported
            )
            supported_pretty = ", ".join(
                f".{ext}" for ext in _SUPPORTED_FILE_EXTENSIONS
            )
            st.error(
                f"Cannot attach {bad_names} \u2014 unsupported file "
                f"type. Supported: {supported_pretty}.",
                icon=":material/error:",
            )
            submission_valid = False
            # Restore the typed prompt as the chat draft so the next
            # render re-flows it into the textarea (st.chat_input has
            # already cleared its own value). The user can fix the
            # upload and re-send without retyping.
            if prompt:
                chat.draft_text = prompt
                try:
                    chats.save_chat(chat)
                except OSError:
                    pass

    if (
        submission_valid
        and (prompt or submitted_files)
        and wd_ok
        and chat.status != chats.STATUS_RUNNING
    ):
        # Phase 6: persist uploads into the chat's artifacts inbox so
        # the file metadata can be threaded through ``chats.start_turn``
        # along with the prompt text. Empty prompts with attachments
        # are still valid submissions ("look at this image and explain
        # what you see") so we treat any prompt-or-file combination as
        # a turn trigger.
        if submitted_files:
            try:
                import attachments as _attachments
                wd_path = Path(ss.working_dir).expanduser().resolve()
                saved_attachments = _attachments.save_uploads(
                    submitted_files, wd_path, chat.id
                )
                # Replace ss.pending_attachments with the saved list
                # so the next-render attachments-chips row reflects
                # what landed on disk (the chat input clears the
                # widget-side files automatically).
                ss.pending_attachments = saved_attachments
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not save attachments: {e}", icon=":material/error:")
                saved_attachments = []
        else:
            saved_attachments = list(ss.pending_attachments or [])
            ss.pending_attachments = []

        # Submitting the prompt consumes the draft — clear ``draft_text``
        # before we spawn the turn so the next render passes ``draft=""``
        # down to the chat-input enhancer (which keeps the textarea
        # empty after st.chat_input has already cleared it). Without
        # this, the saved draft would re-flow into the textarea on the
        # next chat switch and the user would see their just-sent
        # message reappear.
        if chat.draft_text:
            chat.draft_text = ""
            try:
                chats.save_chat(chat)
            except OSError:
                pass
        _start_turn(prompt or "", attachments=saved_attachments)
        st.rerun()

    # Mount the diff dialog last so its modal overlay sits above all
    # the page content. Gated by ``ss.diff_dialog_open`` (set by the
    # "Changes" button or by the sidebar's per-file deep-link); the
    # dialog itself clears the flag when the user closes it.
    if ss.get("diff_dialog_open"):
        _diff_dialog()

    # New-branch + first-push modals are mounted next to the diff dialog
    # so all three live at the same render level. Each is gated by its
    # own ``ss.*_dialog_open`` flag flipped from the actions-row git
    # callbacks (and cleared by the dialog itself on Cancel / Confirm).
    if ss.get("new_branch_dialog_open"):
        _new_branch_dialog()
    if ss.get("first_push_dialog_open"):
        _first_push_dialog()

    # Project-context modal — mounted alongside the other dialogs,
    # gated by ``ss.project_context_dialog_open`` (set by the
    # **Project context** button on the model row, cleared by the
    # dialog's body Close button or its on_dismiss callback).
    if ss.get("project_context_dialog_open"):
        _project_context_dialog()

    # Model picker modal — mounted alongside the other dialogs, gated
    # by ``ss.model_picker_open`` (set by the "current model" button
    # in the controls row, cleared by Select / Cancel / on_dismiss).
    if ss.get("model_picker_open"):
        _model_picker_dialog()

    # Phase 5: lightbox modal for media previews. Mounted alongside
    # the other dialogs at the same render level.
    if ss.get("lightbox_open"):
        _lightbox_dialog()

    # While a background catalog refresh is in flight, run a 0.5s
    # @st.fragment poll that watches the catalog's newest_refresh
    # timestamp + the per-source error dict for changes. When the
    # refresh thread finishes (or errors out), flip the flag back so
    # the modal re-renders with the new data.
    if ss.get("model_catalog_refreshing"):
        _poll_catalog_refresh()


# Streamlit's st.navigation runs the page module top-to-bottom, so we call
# render() at module scope. ``streamlit_app.py`` initializes session state
# before navigation, so ss.* keys are present when this runs.
render()
