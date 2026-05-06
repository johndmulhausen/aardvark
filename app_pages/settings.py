"""Settings page: GitHub identity, theme, W&B Inference, MCP servers.

Replaces the old sidebar settings popover and the MCP sidebar expander.
Rendered in the main column via ``st.navigation`` so the user gets the
full page width for forms and help text. Layout:

1. Page title with the circular avatar to the left.
2. **GitHub identity** card — PAT verify-and-save flow when unauthenticated;
   identity card + sign-out when verified.
3. **Theme** card — Light / Dark / System segmented control bound to
   ``ss.theme_pref`` (callback :func:`actions.set_theme_pref`). The
   actual at-runtime swap is done by :mod:`theme_switcher`, mounted from
   the entry script; this card is purely the user-facing control.
4. **W&B Inference** card — API key + opt-in "Remember on this machine" +
   project + Connect / Disconnect / Forget.
5. **MCP servers** card — list of configured MCP servers with per-server
   enable toggles + edit buttons + an "Add server" button that opens the
   add/edit dialog (an ``@st.dialog`` modal owned by this page). Stdio
   servers run as a local subprocess; HTTP servers are remote.

This page only renders + opens dialogs. All persistence goes through
:mod:`account` and :mod:`mcp_servers`; all callback-style state mutation
is in :mod:`actions`.
"""
from __future__ import annotations

import base64
from typing import Any

import streamlit as st

import account
import mcp_servers
from actions import (
    disconnect as _disconnect,
    forget_saved_wb_key as _forget_saved_wb_key,
    on_connect as _on_connect,
    set_theme_pref as _set_theme_pref,
    sign_out_github as _sign_out_github,
    verify_pat as _verify_pat,
)
from mcp_servers import MCPRegistry, ServerConfig, make_server_id


def _avatar_data_uri() -> str | None:
    """Return a base64 ``data:image/png;...`` URI for the cached GitHub avatar."""
    data = st.session_state.get("avatar_bytes")
    if not data:
        return None
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def _render_avatar(*, size: int = 96) -> None:
    """Render the circular avatar.

    Falls back to an inline-SVG ``account_circle`` glyph (with
    ``fill="currentColor"`` so it adapts to light/dark) when the user has
    not yet verified a PAT. Streamlit's ``:material/...:`` token is NOT
    used here because that token is only resolved in plain markdown text;
    inside raw HTML it renders as the literal string.
    """
    uri = _avatar_data_uri()
    if uri:
        html = (
            f"<img src='{uri}' alt='avatar' "
            f"style='width:{size}px;height:{size}px;border-radius:50%;"
            f"object-fit:cover;display:block;"
            f"border:1px solid var(--st-color-border-color, rgba(127,127,127,0.25));' />"
        )
    else:
        html = (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            f"width='{size}' height='{size}' fill='currentColor' "
            f"style='display:block;opacity:0.65;' aria-label='avatar'>"
            f"<path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10"
            f"S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3"
            f" 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08"
            f" 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z'/>"
            f"</svg>"
        )
    st.html(html)


def _render_github_card(identity: dict[str, Any] | None) -> None:
    """Render the GitHub identity card — PAT form when unauth, profile when verified."""
    ss = st.session_state
    with st.container(border=True):
        st.markdown("### :material/code: GitHub identity")
        if not identity:
            st.caption(
                "The agent uses this as the author when it makes commits via "
                "`run_shell`. Generate a fine-grained personal access token on "
                "GitHub, then paste it below."
            )
            st.caption(
                "Recommended permissions: " + ", ".join(
                    f"`{s}`" for s in account.RECOMMENDED_SCOPES
                )
            )
            st.link_button(
                "Generate a PAT on GitHub",
                account.GITHUB_PAT_CREATE_URL,
                icon=":material/open_in_new:",
            )
            st.text_input(
                "Personal access token",
                key="pat_input",
                type="password",
                placeholder="ghp_... or github_pat_...",
                help="Pasted token is verified against GitHub's `/user` endpoint.",
            )
            st.button(
                "Verify and save",
                icon=":material/verified_user:",
                on_click=_verify_pat,
                type="primary",
            )
            if ss.github_pat_error:
                st.error(ss.github_pat_error, icon=":material/error:")
        else:
            cols = st.columns([1, 4], vertical_alignment="center")
            with cols[0]:
                _render_avatar(size=72)
            with cols[1]:
                st.markdown(
                    f":material/check_circle: Signed in as **{identity.get('login', '')}**"
                )
                email = identity.get("email") or "(email hidden by GitHub)"
                st.caption(f":material/mail: {email}")
                scopes = identity.get("scopes") or []
                if scopes:
                    badges = " ".join(f":green-badge[{s}]" for s in scopes)
                    st.markdown(f"Scopes: {badges}")
                else:
                    st.caption(
                        "No legacy OAuth scopes — fine-grained PATs report "
                        "permissions per-resource rather than as scope strings."
                    )
            st.caption(
                "When you chat in a git repo, the agent stamps `user.name` / "
                "`user.email` into that repo's local config so commits it makes "
                "via `run_shell` are authored as you."
            )
            st.button(
                "Sign out of GitHub",
                icon=":material/logout:",
                on_click=_sign_out_github,
            )


def _theme_label(value: str) -> str:
    """Format a theme option for the segmented control with a Material icon.

    The ``:material/...:`` token renders inline inside widget option
    labels just like it does in plain markdown, which keeps the switcher
    visually consistent with the other Settings cards (palette /
    code / sensors / extension icons).
    """
    return {
        "System": ":material/contrast: System",
        "Light": ":material/light_mode: Light",
        "Dark": ":material/dark_mode: Dark",
    }.get(value, value)


def _render_theme_card() -> None:
    """Render the theme card — Light / Dark / System segmented control.

    The actual theme application is done by :mod:`theme_switcher`, which
    is mounted from ``streamlit_app.main()``. This page just owns the
    UI control and a one-line description of the contract; picking a
    new option fires :func:`actions.set_theme_pref` (persisting the
    choice to ``profile.theme``), the next rerun re-mounts the switcher
    component with the new explicit preference, and the component's JS
    writes to ``localStorage`` and reloads the page so Streamlit's boot
    code picks up the new theme.
    """
    with st.container(border=True):
        st.markdown("### :material/palette: Theme")
        st.segmented_control(
            "Theme",
            options=["System", "Light", "Dark"],
            selection_mode="single",
            format_func=_theme_label,
            key="theme_pref",
            on_change=_set_theme_pref,
            label_visibility="collapsed",
            width="stretch",
        )
        st.caption(
            "Switching reloads the page once so the new theme applies "
            "everywhere. **System** follows your operating system's "
            "light / dark setting."
        )


def _sync_api_key_input() -> None:
    """Mirror the API-key text input back to the canonical ``ss.api_key``."""
    st.session_state.api_key = st.session_state.get("_api_key_input", "")


def _sync_project_input() -> None:
    """Mirror the project text input back to the canonical ``ss.project``."""
    st.session_state.project = st.session_state.get("_project_input", "")


def _sync_remember_input() -> None:
    """Mirror the remember checkbox back to the canonical ``ss.remember_wb_key``."""
    st.session_state.remember_wb_key = bool(
        st.session_state.get("_remember_input", False)
    )


def _render_inference_card() -> None:
    """Render the W&B Inference connection card.

    The API key / project / remember widgets each use an internal widget
    key (``_api_key_input`` etc.) and seed their value from the canonical
    session-state key (``api_key`` etc.) via the ``value=`` parameter,
    syncing back via ``on_change``. The canonical keys are non-widget
    keys, so they survive Streamlit's "strip unused widget keys on
    unmount" behavior when the user navigates between pages — without
    this dance, going Settings -> Chat -> Settings would clear the
    visible API key field even while ``ss.client`` was still connected.
    """
    ss = st.session_state
    with st.container(border=True):
        st.markdown("### :material/sensors: W&B Inference")

        cols = st.columns([3, 2])
        with cols[0]:
            st.text_input(
                "API key",
                value=ss.api_key,
                key="_api_key_input",
                type="password",
                on_change=_sync_api_key_input,
                help=(
                    "Get one at [wandb.ai/settings](https://wandb.ai/settings). "
                    "Held only in session memory unless you tick **Remember on "
                    "this machine** below."
                ),
            )
        with cols[1]:
            st.text_input(
                "Project (optional)",
                value=ss.project,
                key="_project_input",
                placeholder="team/project",
                on_change=_sync_project_input,
                help="Used for usage attribution and Weave tracing.",
            )

        st.checkbox(
            "Remember on this machine",
            value=ss.remember_wb_key,
            key="_remember_input",
            on_change=_sync_remember_input,
            help=(
                "Saves to `~/.wb_coding_agent/credentials.json` (mode 0600). "
                "Anyone with read access to your home directory could read it."
            ),
        )

        is_connected = ss.client is not None and ss.connect_error is None
        if is_connected:
            n = len(ss.models)
            st.success(
                f"Connected. {n} models available.",
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
            st.button(
                "Disconnect",
                icon=":material/link_off:",
                on_click=_disconnect,
            )
        else:
            st.button(
                "Connect",
                icon=":material/link:",
                on_click=_on_connect,
                type="primary",
            )
            if ss.connect_error:
                st.error(ss.connect_error, icon=":material/error:")

        creds_on_disk = bool(account.load_credentials().get("wb_api_key"))
        if creds_on_disk:
            st.caption(
                ":material/save: API key currently saved on this machine."
            )
            st.button(
                "Forget saved API key",
                icon=":material/delete_sweep:",
                on_click=_forget_saved_wb_key,
            )


# ---------------------------------------------------------------------------
# MCP servers card + add/edit dialog
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_mcp_registry() -> MCPRegistry:
    """Cached singleton handle to the process-wide MCP registry.

    ``@st.cache_resource`` survives Streamlit reruns, which is what we
    want: the registry owns a daemon-thread asyncio loop and live
    ``ClientSession``s and must not be rebuilt on every interaction.
    """
    return mcp_servers.get_registry()


def _parse_kv_lines(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` or ``Header: value`` lines into a dict.

    Splits on whichever of ``=`` / ``:`` appears earlier on the line, so
    bearer tokens with base64 ``=`` padding (``Authorization: Bearer
    eyJ...xyz==``) survive intact while ``PATH=/usr/bin:/bin`` env vars
    still split correctly.
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
    """Inverse of :func:`_parse_kv_lines` for prefilling the edit form."""
    sep = ": " if header_style else "="
    return "\n".join(f"{k}{sep}{v}" for k, v in items.items())


@st.dialog("MCP server", width="large")
def _mcp_server_dialog() -> None:
    """Add or edit an MCP server config.

    Decided by ``ss.mcp_dialog_editing``: ``None`` adds a new server,
    otherwise it's the id of the server being edited (we look it up in the
    registry to pre-fill the form). Save reconciles live sessions; Delete
    is the explicit destructive action.
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
        args_default = (
            "\n".join(existing.args)
            if existing and existing.transport == "stdio"
            else ""
        )
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
        "Save", icon=":material/save:", type="primary", width="stretch"
    )
    cancel_clicked = cols[1].button(
        "Cancel", icon=":material/close:", width="stretch"
    )
    delete_clicked = False
    if existing is not None:
        delete_clicked = cols[2].button(
            "Delete server", icon=":material/delete:", width="stretch"
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

        # Toast the save outcome — ``st.warning`` / ``st.error`` won't paint
        # because ``st.rerun()`` below aborts the script before the next
        # redraw; toasts survive the rerun.
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
                f"'{config.name}' failed to connect — see Settings.",
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


def _render_mcp_card() -> None:
    """The MCP servers card. Per-server row + Add server + dialog mount."""
    registry = _get_mcp_registry()
    configs = list(registry.configs)
    with st.container(border=True):
        header_cols = st.columns([5, 2], vertical_alignment="center")
        with header_cols[0]:
            count = f" \u00b7 {len(configs)}" if configs else ""
            st.markdown(f"### :material/extension: MCP servers{count}")
            st.caption(
                "Connect external Model Context Protocol servers to expose "
                "their tools to the agent. Stdio servers run as a local "
                "subprocess; HTTP servers are remote. Configs persist to "
                "`~/.wb_coding_agent/mcp.json` (mode 0600)."
            )
        with header_cols[1]:
            st.button(
                "Add server",
                icon=":material/add:",
                type="primary" if not configs else "secondary",
                on_click=_open_add_mcp_dialog,
                width="stretch",
            )

        if not configs:
            return

        for cfg in configs:
            status = registry.statuses.get(cfg.id)
            row = st.container(border=True)
            with row:
                top = st.columns([6, 1, 1], vertical_alignment="center")
                with top[0]:
                    transport_badge = (
                        ":blue-badge[stdio]"
                        if cfg.transport == "stdio"
                        else ":violet-badge[http]"
                    )
                    st.markdown(f"**{cfg.name}** {transport_badge}")
                    if status is not None and status.connected:
                        n = len(status.tools)
                        st.caption(
                            f":green[Connected] \u00b7 {n} tool"
                            f"{'s' if n != 1 else ''}"
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


def _render_session_summary() -> None:
    """Render a compact session-usage caption when the user has run anything."""
    ss = st.session_state
    session_total = ss.usage_session_total
    if session_total["turns"] <= 0:
        return
    import usage as usage_log
    st.caption(
        f":material/data_usage: This session: "
        f"{usage_log.format_tokens(session_total['total_tokens'])} tokens "
        f"\u00b7 {usage_log.format_cost(session_total['cost_usd'])} "
        f"\u00b7 {session_total['turns']} turns"
    )


def render() -> None:
    """Page body.

    When the user is signed in to GitHub, the page header is a two-column
    avatar + identity strip (avatar to the left, login + email stacked to
    the right). When they're not signed in, the avatar is a generic
    placeholder and rendering it next to the title leaves a visible gutter
    between the title and the cards below it — so the signed-out path
    falls back to a full-width title that lines up with the card column.
    """
    ss = st.session_state
    identity = ss.github_identity or {}
    login = identity.get("login")

    if login:
        header = st.columns([1, 6], vertical_alignment="center")
        with header[0]:
            _render_avatar(size=72)
        with header[1]:
            st.title(login)
            email = identity.get("email")
            if email:
                st.caption(f":material/mail: {email}")
    else:
        st.title("Settings")
        st.caption(
            "Configure your GitHub identity, theme, and W&B Inference "
            "connection. Settings persist on this machine when you opt in."
        )

    _render_session_summary()
    _render_github_card(identity if identity else None)
    _render_theme_card()
    _render_inference_card()
    _render_mcp_card()

    # Mount the add/edit dialog last so it overlays whatever else is on
    # screen. The flag is set by ``_open_add_mcp_dialog`` /
    # ``_open_edit_mcp_dialog`` (button on_click callbacks).
    if st.session_state.get("mcp_dialog_open"):
        _mcp_server_dialog()


render()
