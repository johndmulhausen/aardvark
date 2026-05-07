"""Settings page: GitHub identity, appearance, providers, MCP servers.

Replaces the old sidebar settings popover and the MCP sidebar expander.
Rendered in the main column via ``st.navigation`` so the user gets the
full page width for forms and help text. Layout:

1. Page title with the circular avatar to the left.
2. **GitHub identity** card — PAT verify-and-save flow when unauthenticated;
   identity card + sign-out when verified.
3. **Appearance** card — two segmented controls:
   - Light / Dark / System for theme (bound to ``ss.theme_pref``,
     callback :func:`actions.set_theme_pref`). The actual at-runtime
     swap is done by :mod:`theme_switcher`, mounted from the entry
     script.
   - Extra small / Small / Medium / Large / Extra large for font size
     (bound to ``ss.font_size_pref``, callback
     :func:`actions.set_font_size_pref`).
     The actual at-runtime CSS override is done by
     :mod:`font_size_switcher`, also mounted from the entry script.
4. **Providers** — one card per provider directly in the page body
   (W&B Inference, OpenAI, Anthropic, Google Gemini, Mistral, xAI,
   OpenRouter). All 7 providers earn their top-of-page card on a
   different axis: W&B for tracing, OpenAI / Anthropic / Google /
   Mistral / xAI as native-SDK frontier-lab routes, and OpenRouter
   as the marked-up gateway covering the long tail. OpenRouter
   carries a persistent ``:gray-badge[marked-up gateway]`` plus a
   caveat caption explaining its 5–10% markup so users see the
   trade-off before opting in to the long-tail route.

   The :data:`providers.Provider.tier` field still exists (with
   ``"primary"`` / ``"more"`` literals) so a future expansion of the
   catalog can move less-prominent providers into a collapsed
   expander, but every entry today is ``tier="primary"``. The
   Settings page renders the "More providers" expander only when
   :func:`providers.more_providers` returns a non-empty list.

   Each card renders the same dual-key-pattern API key field
   (``_<id>_key_input`` widget key paired with the canonical
   ``ss.provider_keys[id]``), an ``st.link_button`` to the provider's
   key console (``provider.key_url``), a "Remember on this machine"
   checkbox bound to ``ss.provider_remember[id]``, and Connect /
   Disconnect / Forget buttons. The W&B card additionally renders the
   optional team/project field plus the Weave tracing caption when
   ``ss.weave_project`` is set; OpenRouter renders the markup caveat.
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
import providers
from actions import (
    connect_provider as _connect_provider,
    disconnect_provider as _disconnect_provider,
    forget_provider_key as _forget_provider_key,
    set_font_size_pref as _set_font_size_pref,
    set_theme_pref as _set_theme_pref,
    sign_out_github as _sign_out_github,
    verify_pat as _verify_pat,
)
from font_size_switcher import FONT_SIZE_OPTIONS
from mcp_servers import MCPRegistry, ServerConfig, make_server_id


# Provider cards render labels as plain text — no per-provider icons.
# An earlier iteration mapped each provider id to a Material Symbol
# (e.g. ``:material/diamond:`` for Gemini) but the icons were purely
# decorative and crowded the card header without conveying anything
# the label didn't already say. They've been removed; this comment
# is the only thing left so future readers don't reintroduce them
# without a real reason.


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


def _font_size_label(value: str) -> str:
    """Format a font-size option with a glyph hint at the option's relative size.

    Streamlit's segmented control doesn't let us style individual
    options by font-size directly, so we use a Material ``:material/text_fields:``
    glyph to hint that the row controls type sizing without duplicating
    the card's ``palette`` icon. Sentence case for the labels matches
    the rest of the Settings page.
    """
    return f":material/text_fields: {value}"


def _render_appearance_card() -> None:
    """Render the appearance card — theme + font size controls.

    Both controls follow the same pattern: bind the segmented control
    to a canonical session-state key via ``key=``, and wire ``on_change``
    to the matching callback in :mod:`actions`. The actual at-runtime
    application happens elsewhere on the next rerun (theme via
    :mod:`theme_switcher` writing to ``localStorage`` + a page reload;
    font size via :mod:`font_size_switcher` injecting a CSS rule into
    ``document.head``), so this card stays cleanly UI-only.
    """
    with st.container(border=True):
        st.markdown("### :material/palette: Appearance")

        st.markdown("**Theme**")
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

        st.markdown("**Font size**")
        st.segmented_control(
            "Font size",
            options=list(FONT_SIZE_OPTIONS),
            selection_mode="single",
            format_func=_font_size_label,
            key="font_size_pref",
            on_change=_set_font_size_pref,
            label_visibility="collapsed",
            width="stretch",
        )
        st.caption(
            "Adjusts the root font size used for body text, headings, "
            "captions, and code. **Small** matches the bundled default."
        )


def _make_sync_key_input(provider_id: str):
    """Return a no-args callback that mirrors the per-provider key widget."""
    widget_key = f"_{provider_id}_key_input"

    def _cb() -> None:
        ss = st.session_state
        keys = dict(ss.provider_keys)
        keys[provider_id] = (ss.get(widget_key, "") or "").strip()
        ss.provider_keys = keys
        if provider_id == "wandb":
            ss.api_key = keys[provider_id]

    return _cb


def _make_sync_remember_input(provider_id: str):
    """Return a no-args callback that mirrors the per-provider remember widget."""
    widget_key = f"_{provider_id}_remember_input"

    def _cb() -> None:
        ss = st.session_state
        flags = dict(ss.provider_remember)
        flags[provider_id] = bool(ss.get(widget_key, False))
        ss.provider_remember = flags
        if provider_id == "wandb":
            ss.remember_wb_key = flags[provider_id]

    return _cb


def _sync_wandb_project_input() -> None:
    """Mirror the W&B project text input back to ``ss.project``."""
    st.session_state.project = (st.session_state.get("_wandb_project_input", "") or "").strip()


def _provider_status(provider_id: str) -> tuple[bool, int, str | None]:
    """Return ``(is_connected, model_count, error_message)`` for ``provider_id``.

    Connectivity is signalled by a non-None client object in
    ``ss.clients[provider_id]`` (every provider gets a real client
    after a successful connect — for ``openai_compat`` it's an
    ``openai.OpenAI`` configured against the provider's ``base_url``).
    The model count comes from the live ``/v1/models`` listing the
    connect path stashed into ``ss.provider_models``.
    """
    ss = st.session_state
    error = ss.connect_errors.get(provider_id)
    models = ss.provider_models.get(provider_id) or []
    is_connected = (not error) and (ss.clients.get(provider_id) is not None)
    return is_connected, len(models), error


def _render_provider_card(provider: providers.Provider) -> None:
    """Render one provider Settings card — compact 3-row layout.

    Row 1: ``**Label**`` + inline status pill (`green-badge[Connected · N models]`,
    `red-badge[Error]`, `gray-badge[marked-up gateway]` for OpenRouter).
    The status pill replaces the previous ``st.success`` banner — pills
    are inline-markdown so they don't introduce a vertical block.

    Row 2: API key field + (W&B only) Project field + a "Get key"
    icon-button link to the provider's key console. The text inputs
    use ``label_visibility="collapsed"`` and a ``placeholder`` so the
    label-row vertical space is reclaimed.

    Row 3: ☑ Remember + Connect/Disconnect + Forget, all inline.

    Below the card body: an ``st.error`` for the most recent connect
    error (rendered only when present) and the W&B-specific Weave-
    tracing caption when W&B is connected. Both stay full-width
    because their content is variable-length.

    The dual-key pattern (per AGENTS.md) is preserved: each widget
    binds to its own ``_<id>_*_input`` key while the canonical state
    lives on ``ss.provider_keys[<id>]`` / ``ss.provider_remember[<id>]``.
    """
    ss = st.session_state
    pid = provider.id
    key_widget = f"_{pid}_key_input"
    remember_widget = f"_{pid}_remember_input"

    is_connected, n_models, error = _provider_status(pid)
    creds_on_disk = bool(account.load_provider_keys().get(pid))

    with st.container(border=True):
        # ---- Row 1: label + inline status pill ----
        header_parts: list[str] = [f"**{provider.label}**"]
        if pid == "openrouter":
            header_parts.append(":gray-badge[marked-up gateway]")
        if is_connected:
            header_parts.append(
                f":green-badge[Connected \u00b7 {n_models} model"
                f"{'s' if n_models != 1 else ''}]"
            )
        elif error:
            header_parts.append(":red-badge[Error]")
        st.markdown(" ".join(header_parts))

        # OpenRouter caveat — single-line caption.
        if provider.notes:
            st.caption(provider.notes)

        # ---- Row 2: key field(s) + key-link button ----
        if pid == "wandb":
            # W&B has the optional Project field, so the key + project +
            # link button live in a 3-column row. The link button gets a
            # narrow 1-unit column at the right.
            cols = st.columns([4, 3, 1], vertical_alignment="bottom")
            with cols[0]:
                st.text_input(
                    "API key",
                    value=ss.provider_keys.get(pid, ""),
                    key=key_widget,
                    type="password",
                    on_change=_make_sync_key_input(pid),
                    placeholder="API key",
                    label_visibility="collapsed",
                    help="Held only in session memory unless **Remember** is ticked.",
                )
            with cols[1]:
                st.text_input(
                    "Project",
                    value=ss.get("project", ""),
                    key="_wandb_project_input",
                    placeholder="team/project (optional)",
                    on_change=_sync_wandb_project_input,
                    label_visibility="collapsed",
                    help="W&B team/project for usage attribution + Weave tracing.",
                )
            with cols[2]:
                if provider.key_url:
                    st.link_button(
                        "Get key",
                        provider.key_url,
                        icon=":material/open_in_new:",
                        width="stretch",
                        help=f"Open {provider.label}'s API key console.",
                    )
        else:
            cols = st.columns([6, 1], vertical_alignment="bottom")
            with cols[0]:
                st.text_input(
                    "API key",
                    value=ss.provider_keys.get(pid, ""),
                    key=key_widget,
                    type="password",
                    on_change=_make_sync_key_input(pid),
                    placeholder="API key",
                    label_visibility="collapsed",
                    help="Held only in session memory unless **Remember** is ticked.",
                )
            with cols[1]:
                if provider.key_url:
                    st.link_button(
                        "Get key",
                        provider.key_url,
                        icon=":material/open_in_new:",
                        width="stretch",
                        help=f"Open {provider.label}'s API key console.",
                    )

        # ---- Row 3: Remember + Connect/Disconnect + Forget ----
        action_cols = st.columns([3, 2, 2], vertical_alignment="center")
        with action_cols[0]:
            st.checkbox(
                "Remember on this machine",
                value=bool(ss.provider_remember.get(pid, False)),
                key=remember_widget,
                on_change=_make_sync_remember_input(pid),
                help=(
                    "Saves to `~/.wb_coding_agent/credentials.json` "
                    "(mode 0600). Anyone with read access to your home "
                    "directory could read it."
                ),
            )
        with action_cols[1]:
            if is_connected:
                st.button(
                    "Disconnect",
                    icon=":material/link_off:",
                    on_click=_disconnect_provider,
                    args=(pid,),
                    key=f"_{pid}_disconnect_btn",
                    width="stretch",
                )
            else:
                st.button(
                    "Connect",
                    icon=":material/link:",
                    on_click=_connect_provider,
                    args=(pid,),
                    type="primary",
                    key=f"_{pid}_connect_btn",
                    width="stretch",
                )
        with action_cols[2]:
            if creds_on_disk:
                st.button(
                    "Forget key",
                    icon=":material/delete_sweep:",
                    on_click=_forget_provider_key,
                    args=(pid,),
                    key=f"_{pid}_forget_btn",
                    width="stretch",
                )

        # ---- W&B Weave tracing caption (variable-length, full-width) ----
        if pid == "wandb" and is_connected:
            if ss.weave_project:
                if ss.weave_url:
                    st.caption(
                        f":material/sensors: Tracing turns to W&B Weave at "
                        f"[`{ss.weave_project}`]({ss.weave_url})."
                    )
                else:
                    st.caption(
                        f":material/sensors: Tracing turns to W&B Weave at "
                        f"`{ss.weave_project}`."
                    )
            elif ss.weave_error:
                st.caption(
                    f":material/sensors_off: Weave tracing disabled: "
                    f"{ss.weave_error}"
                )

        # ---- Error banner (only on failure; full-width because the
        # message can be long) ----
        if not is_connected and error:
            st.error(error, icon=":material/error:")


def _render_providers_section() -> None:
    """Render the multi-provider section: one card per provider in catalog order.

    Every provider in :data:`providers.PROVIDERS` lives in the page
    body today — there is no "More providers" expander because the
    catalog is small enough (7 providers) that all of them earn
    above-the-fold treatment:

    - **W&B Inference** — Weave tracing kicks in only when W&B is
      connected, regardless of which provider you call for inference,
      so even users who plan to call other providers want a W&B key
      for observability.
    - **OpenAI**, **Anthropic**, **Google Gemini**, **Mistral**,
      **xAI** — frontier labs (they train and release their own
      foundation models). Each routes through a dedicated native-SDK
      dispatch path; OpenAI / Anthropic / Google / xAI have
      native-only features actively wired (``reasoning_effort``,
      auto prompt-caching, Google Search grounding, Live Search).
    - **OpenRouter** — labeled as a marked-up gateway, kept in the
      catalog because it's the cheapest one-key-fits-all path to
      dozens of open-model clouds without managing a separate key
      per backend.

    Pure inference-server providers (Together / Groq / Fireworks /
    DeepInfra / Cerebras) are intentionally NOT listed — OpenRouter
    routes to those backends already, and the only direct-key value-
    add is skipping OpenRouter's markup. Add a :class:`Provider`
    entry in :mod:`providers` if a future use case needs one of
    those routes specifically.

    The :data:`providers.Provider.tier` field still distinguishes
    ``"primary"`` from ``"more"`` so a future catalog expansion can
    fold less-prominent providers behind an expander; we render the
    expander only when :func:`providers.more_providers` returns a
    non-empty list.
    """
    st.markdown("### :material/hub: Providers")
    st.caption(
        "Add an API key from any provider you want to use. W&B "
        "Inference is the source of Weave tracing (works regardless "
        "of which provider you call); OpenAI, Anthropic, Google, "
        "Mistral, and xAI each route through their native SDK so "
        "you get features that OpenAI-compat HTTP can't model "
        "(reasoning effort, prompt caching, grounding, Live Search); "
        "OpenRouter is a one-key route to the long tail at a 5–10% "
        "markup."
    )

    for provider in providers.primary_providers():
        _render_provider_card(provider)

    # The "More providers" expander — only shown when the catalog
    # actually has tier="more" entries. Today the list is empty
    # (every provider lives above the fold), but the affordance
    # remains so a future catalog expansion can collapse less-
    # prominent additions without touching this rendering code.
    more = providers.more_providers()
    if more:
        with st.expander(":material/expand_more: More providers", expanded=False):
            st.caption(
                "Less-prominent providers. Add a key when you use that "
                "specific service heavily; otherwise the cards above "
                "cover almost everything."
            )
            for provider in more:
                _render_provider_card(provider)


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


def _close_mcp_dialog() -> None:
    """Drop the dialog flags so subsequent reruns don't re-mount the modal.

    Wired to ``@st.dialog(..., on_dismiss=_close_mcp_dialog)`` so X /
    Esc / click-outside dismissals clear the gating flag — without
    this, the dialog re-opens on the very next rerun (e.g. the next
    chat submission on the chat page) because ``mcp_dialog_open``
    stays ``True``. The Cancel / Save / Delete handlers inside the
    dialog body clear the same flags and call ``st.rerun()`` directly.
    """
    st.session_state.mcp_dialog_open = False
    st.session_state.mcp_dialog_editing = None


@st.dialog("MCP server", width="large", on_dismiss=_close_mcp_dialog)
def _mcp_server_dialog() -> None:
    """Add or edit an MCP server config.

    Decided by ``ss.mcp_dialog_editing``: ``None`` adds a new server,
    otherwise it's the id of the server being edited (we look it up in the
    registry to pre-fill the form). Save reconciles live sessions; Delete
    is the explicit destructive action.

    ``on_dismiss=_close_mcp_dialog`` is mandatory so X / Esc /
    click-outside dismissal clears ``ss.mcp_dialog_open`` /
    ``ss.mcp_dialog_editing``; otherwise the modal re-opens on the
    next rerun (e.g. when the user navigates to the chat page and
    sends a message). See :func:`app_pages.chat._diff_dialog` for
    the full rationale.
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
    _render_appearance_card()
    _render_providers_section()
    _render_mcp_card()

    # Mount the add/edit dialog last so it overlays whatever else is on
    # screen. The flag is set by ``_open_add_mcp_dialog`` /
    # ``_open_edit_mcp_dialog`` (button on_click callbacks).
    if st.session_state.get("mcp_dialog_open"):
        _mcp_server_dialog()


render()
