"""Shared callbacks and helpers used by the Streamlit page modules.

Streamlit loads the entry script (``streamlit_app.py``) as ``__main__``,
which means sub-page modules cannot ``from streamlit_app import …``
without re-importing — and re-importing re-runs ``main()`` and re-renders
the sidebar, causing ``StreamlitDuplicateElementKey`` errors. Putting
everything any page might need in a regular importable module avoids
that trap entirely. Add a callback here whenever a page (chat, usage,
settings, ...) needs to share it; never have pages import from
``streamlit_app``.

Owns:

- The on-disk recent-working-directories list (``recent_dirs.json``).
- The native folder picker (``osascript`` / ``tkinter`` subprocess).
- The multi-provider Connect / Disconnect / Forget callbacks
  (:func:`connect_provider`, :func:`disconnect_provider`,
  :func:`forget_provider_key`) — generalized in Phase 1 from the
  W&B-only flow. ``on_connect`` is retained as a thin shim that calls
  ``connect_provider("wandb", ...)`` so the auto-connect path in
  ``streamlit_app._maybe_auto_connect`` and the legacy callers in
  ``app_pages/settings.py`` keep working unchanged.
- The GitHub PAT verify-and-save / sign-out callbacks.
- The theme-switcher callbacks (segmented-control ``on_change`` +
  the migration-detection callback fired by ``theme_switcher``).
- The font-size switcher callback (segmented-control ``on_change``)
  fired from the Settings page's appearance card.

Pages keep their own *rendering* code; they call into this module purely
for state mutation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

import account
import model_catalog
import providers
from wb_client import init_weave


# ---------------------------------------------------------------------------
# Recent working directories
# ---------------------------------------------------------------------------
RECENT_DIRS_FILE = Path.home() / ".wb_coding_agent" / "recent_dirs.json"
MAX_RECENT_DIRS = 10


def load_recent_dirs() -> list[str]:
    """Read the persisted recents list, returning ``[]`` on any failure."""
    try:
        raw = json.loads(RECENT_DIRS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw if isinstance(p, str)][:MAX_RECENT_DIRS]


def save_recent_dirs(dirs: list[str]) -> None:
    """Persist the recents list. Best-effort: failures are swallowed."""
    try:
        RECENT_DIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_DIRS_FILE.write_text(json.dumps(dirs, indent=2), encoding="utf-8")
    except OSError:
        pass


def record_recent_dir(path: str) -> None:
    """Move ``path`` to the front of the recents list, dedupe, persist."""
    ss = st.session_state
    abs_path = str(Path(path).expanduser().resolve())
    existing = [d for d in ss.recent_dirs if d != abs_path]
    ss.recent_dirs = ([abs_path] + existing)[:MAX_RECENT_DIRS]
    save_recent_dirs(ss.recent_dirs)


def pick_directory(initial: str | None = None) -> str | None:
    """Open a native folder picker, returning an absolute path or ``None``.

    Uses ``osascript`` on macOS (no extra deps, no threading concerns) and
    a ``tkinter.filedialog`` subprocess elsewhere so the file dialog
    doesn't share a thread with Streamlit's script runner.
    """
    if sys.platform == "darwin":
        script = "POSIX path of (choose folder with prompt \"Choose working directory\")"
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        chosen = result.stdout.strip().rstrip("/")
        return chosen or None

    snippet = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        f"path = filedialog.askdirectory(initialdir={initial or ''!r})\n"
        "print(path or '')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    chosen = result.stdout.strip()
    return chosen or None


# ---------------------------------------------------------------------------
# Multi-provider Connect / Disconnect / Forget callbacks
# ---------------------------------------------------------------------------
def _sort_models(models: list[str]) -> list[str]:
    """Sort model ids by their display label, case-insensitive.

    Used for the legacy W&B-only ``ss.models`` list that the chat page
    previously read. The multi-provider model picker (Phase 2) sorts
    qualified ids via ``models.model_label(...)`` separately.
    """
    from models import model_label
    return sorted(models, key=lambda m: model_label(m).casefold())


def _sync_provider_widgets(provider_id: str) -> None:
    """Mirror the Settings page's per-provider widget keys into canonical state.

    The Settings page uses the dual-key pattern (per AGENTS.md) so the
    canonical ``ss.provider_keys[<id>]`` / ``ss.provider_remember[<id>]``
    survive page navigation while widgets bind to private
    ``_<id>_key_input`` / ``_<id>_remember_input`` keys. Each widget has
    an ``on_change`` sync callback, but we re-sync defensively here so
    a Connect click that fires *without* a prior ``on_change`` event
    (e.g. user clicks Connect immediately after typing without blurring
    the field) still picks up the latest typed value.

    For the W&B-specific ``project`` field we mirror an additional
    ``_wandb_project_input`` widget back into ``ss.project`` (the
    legacy single-source-of-truth key); future per-provider extras
    follow the same naming convention.
    """
    ss = st.session_state
    key_widget = f"_{provider_id}_key_input"
    if key_widget in ss:
        keys = dict(ss.provider_keys)
        keys[provider_id] = (ss[key_widget] or "").strip()
        ss.provider_keys = keys
    remember_widget = f"_{provider_id}_remember_input"
    if remember_widget in ss:
        flags = dict(ss.provider_remember)
        flags[provider_id] = bool(ss[remember_widget])
        ss.provider_remember = flags
    # W&B's optional team/project field uses its own widget key per the
    # ``extra_fields`` contract on :class:`providers.Provider`.
    if provider_id == "wandb":
        if "_wandb_project_input" in ss:
            ss.project = (ss["_wandb_project_input"] or "").strip()
        # Legacy alias used by the auto-connect path + older settings
        # widgets that pre-date the dual-key rename.
        if "_api_key_input" in ss:
            keys = dict(ss.provider_keys)
            keys["wandb"] = (ss._api_key_input or "").strip()
            ss.provider_keys = keys
            ss.api_key = keys["wandb"]
        if "_project_input" in ss:
            ss.project = (ss._project_input or "").strip()
        if "_remember_input" in ss:
            ss.remember_wb_key = bool(ss._remember_input)


def _persist_provider_key(provider_id: str) -> None:
    """Persist or clear ``ss.provider_keys[provider_id]`` based on the remember flag."""
    ss = st.session_state
    api_key = ss.provider_keys.get(provider_id, "").strip()
    remember = bool(ss.provider_remember.get(provider_id, False))
    account.save_provider_key(provider_id, api_key, remember=remember)


def connect_provider(provider_id: str) -> None:
    """Connect a single provider: build client, list models, persist.

    Reads from ``ss.provider_keys[provider_id]`` (already synced by
    :func:`_sync_provider_widgets`), constructs the per-kind client via
    :func:`providers.make_provider_client` (an ``openai.OpenAI`` for
    ``openai_native`` / ``openai_compat``, an ``anthropic.Anthropic``
    for ``anthropic_native``, a ``google.genai.Client`` for
    ``google_native``), and validates connectivity by listing the
    provider's models through :func:`model_catalog.refresh`.

    On success:

    - ``ss.clients[provider_id]`` carries the persistent client object
      (every provider gets a real client — for ``openai_compat`` the
      same ``openai.OpenAI`` SDK is used with a per-provider
      ``base_url``, the same pattern the app used for W&B Inference
      before the multi-provider migration).
    - ``ss.provider_models[provider_id]`` carries the raw model ids
      sorted alphabetically.
    - ``ss.connect_errors[provider_id]`` is cleared.
    - For W&B specifically, :func:`init_weave` is called with the
      same key + project so chat-completion calls show up as Weave
      traces.
    - The per-provider key is persisted to ``credentials.json`` when
      the user has ticked "Remember on this machine" for that provider.

    On failure the client / models are dropped and the error message is
    stashed for the Settings card to render.

    The model catalog refresh (with OpenRouter description enrichment
    for the ``openrouter:*`` namespace) runs synchronously inside
    :func:`model_catalog.refresh` — this connect path doubles as both
    the "is this API key live" connectivity check (a /v1/models call
    that returns 401/403 on a bad key) and the catalog-populator for
    the picker.
    """
    ss = st.session_state
    provider = providers.get_provider(provider_id)
    if provider is None:
        ss.connect_errors[provider_id] = f"Unknown provider: {provider_id!r}"
        return

    _sync_provider_widgets(provider_id)
    api_key = ss.provider_keys.get(provider_id, "").strip()

    # Mirror back into the (now-cleaned) dict so other code paths see
    # the trimmed value.
    keys = dict(ss.provider_keys)
    keys[provider_id] = api_key
    ss.provider_keys = keys

    if not api_key:
        errors = dict(ss.connect_errors)
        errors[provider_id] = f"Enter your {provider.label} API key first."
        ss.connect_errors = errors
        clients = dict(ss.clients)
        clients[provider_id] = None
        ss.clients = clients
        pm = dict(ss.provider_models)
        pm[provider_id] = []
        ss.provider_models = pm
        if provider_id == "wandb":
            # Mirror into the legacy back-compat fields so existing
            # zero-state UI keeps working.
            ss.client = None
            ss.models = []
            ss.connect_error = ss.connect_errors[provider_id]
        return

    # Build the client. ``make_provider_client`` raises on bad input;
    # surface the message verbatim.
    try:
        client = providers.make_provider_client(provider_id, api_key)
    except Exception as e:
        clients = dict(ss.clients)
        clients[provider_id] = None
        ss.clients = clients
        pm = dict(ss.provider_models)
        pm[provider_id] = []
        ss.provider_models = pm
        errors = dict(ss.connect_errors)
        errors[provider_id] = f"Could not initialize {provider.label} client: {e}"
        ss.connect_errors = errors
        if provider_id == "wandb":
            ss.client = None
            ss.models = []
            ss.connect_error = ss.connect_errors[provider_id]
        return

    # Validate connectivity AND refresh the model catalog in one pass.
    # ``model_catalog.refresh`` lists models (the de-facto "is this API
    # key live" connectivity check — /v1/models endpoints return 401 /
    # 403 on a bad key) and folds in OpenRouter description enrichment
    # for the ``openrouter:*`` namespace, populating the picker with
    # fresh data immediately. The synchronous call here is fine — it's
    # only run on user-initiated Connect clicks, not at startup; the
    # startup path uses ``refresh_all_async``.
    try:
        model_catalog.refresh(provider_id, client)
    except Exception as e:
        clients = dict(ss.clients)
        clients[provider_id] = None
        ss.clients = clients
        pm = dict(ss.provider_models)
        pm[provider_id] = []
        ss.provider_models = pm
        errors = dict(ss.connect_errors)
        errors[provider_id] = f"Could not connect to {provider.label}: {e}"
        ss.connect_errors = errors
        if provider_id == "wandb":
            ss.client = None
            ss.models = []
            ss.connect_error = ss.connect_errors[provider_id]
        return

    # The Settings card status caption + chat picker both consume the
    # raw-id list, while the picker's row layout reads the full
    # ``ModelInfo`` from ``model_catalog``. We keep ``provider_models``
    # populated with the raw ids so the Settings card's "N models
    # available" caption stays accurate even when some models were
    # dropped by the completeness gate.
    raw_ids = model_catalog._state.raw_ids_per_provider.get(provider_id, [])  # noqa: SLF001
    sorted_ids = _sort_models(raw_ids)
    # Touch the refreshing flag's monotonic timestamp so the modal
    # caption can show "Last refreshed Nm ago" without the daemon-
    # thread path having run yet (the synchronous connect path also
    # counts as a refresh from the user's perspective).
    import time as _time
    ss.model_catalog_last_refreshed_at = _time.monotonic()
    clients = dict(ss.clients)
    clients[provider_id] = client
    ss.clients = clients
    pm = dict(ss.provider_models)
    pm[provider_id] = sorted_ids
    ss.provider_models = pm
    errors = dict(ss.connect_errors)
    errors[provider_id] = None
    ss.connect_errors = errors

    # Legacy back-compat fields for the W&B path until the chat page
    # fully migrates to qualified ids in Phase 2. ``ss.client`` is the
    # legacy alias the chat page reads as a connectivity flag; we set
    # it to the real ``openai.OpenAI`` client (W&B Inference is now
    # ``openai_compat``, dispatched through the OpenAI SDK with
    # ``base_url`` set to the W&B endpoint).
    if provider_id == "wandb":
        ss.client = client
        ss.models = sorted_ids
        if ss.model not in ss.models:
            ss.model = ss.models[0] if ss.models else None
        ss.connect_error = None

    # W&B Inference also bootstraps Weave so chat completions get traced.
    # The other providers' Weave integrations attach via
    # ``weave-anthropic`` / ``weave-openai`` / ``weave-google-genai``
    # auto-patching on import, so they don't need a per-provider init.
    if provider_id == "wandb":
        try:
            _, resolved_project, weave_url = init_weave(
                api_key=api_key,
                project=ss.project.strip() or None,
            )
            ss.weave_project = resolved_project
            ss.weave_url = weave_url
            ss.weave_error = None
        except Exception as e:
            ss.weave_project = None
            ss.weave_url = None
            ss.weave_error = str(e)

    # Persist the key (or clear it if the user un-ticked Remember).
    _persist_provider_key(provider_id)


def disconnect_provider(provider_id: str) -> None:
    """Drop the live client + listed models for ``provider_id`` (key persists).

    Intentional design: clicking Disconnect removes the *runtime*
    connection (client, listed models, error message) but keeps the
    saved API key on disk if the user had ticked "Remember on this
    machine". To wipe the saved key from disk, call
    :func:`forget_provider_key` instead.
    """
    ss = st.session_state
    clients = dict(ss.clients)
    clients[provider_id] = None
    ss.clients = clients
    pm = dict(ss.provider_models)
    pm[provider_id] = []
    ss.provider_models = pm
    errors = dict(ss.connect_errors)
    errors[provider_id] = None
    ss.connect_errors = errors

    if provider_id == "wandb":
        ss.client = None
        ss.models = []
        ss.model = None
        ss.weave_project = None
        ss.weave_url = None
        ss.weave_error = None
        ss.connect_error = None


def forget_provider_key(provider_id: str) -> None:
    """Remove the persisted API key for ``provider_id`` from disk.

    Also clears the in-session value + un-ticks the "Remember on this
    machine" flag so the Settings card visibly reflects the change on
    the next render. If the provider is currently connected, the
    runtime client stays live until the user explicitly clicks
    Disconnect — forget_provider_key is "stop persisting", not
    "disconnect right now".
    """
    ss = st.session_state
    account.clear_credentials(provider_id=provider_id)
    keys = dict(ss.provider_keys)
    keys[provider_id] = ""
    ss.provider_keys = keys
    flags = dict(ss.provider_remember)
    flags[provider_id] = False
    ss.provider_remember = flags
    # Drop the dual-key widget values so the visible form clears on
    # rerun (per the dual-key pattern in AGENTS.md).
    for k in (f"_{provider_id}_key_input", f"_{provider_id}_remember_input"):
        if k in ss:
            del ss[k]
    # Legacy aliases for the W&B path.
    if provider_id == "wandb":
        ss.api_key = ""
        ss.remember_wb_key = False
        for k in ("_api_key_input", "_remember_input"):
            if k in ss:
                del ss[k]


# ---------------------------------------------------------------------------
# Legacy W&B-only callbacks (one-release shim layer)
# ---------------------------------------------------------------------------
# These keep ``streamlit_app._maybe_auto_connect`` and any settings code
# that hasn't been ported to the per-provider naming scheme working
# unchanged. Internally they all delegate to the multi-provider
# functions above.


def _sync_inference_inputs() -> None:
    """Pull the latest W&B widget values into canonical state.

    Legacy shim retained for one release. The per-provider equivalent is
    :func:`_sync_provider_widgets`, which this function delegates to
    after also handling the legacy ``_api_key_input`` / ``_project_input``
    / ``_remember_input`` widget keys.
    """
    _sync_provider_widgets("wandb")


def on_connect() -> None:
    """Connect the W&B Inference provider. Legacy shim for the auto-connect path.

    Honors the legacy ``ss.remember_wb_key`` flag by mirroring it into
    the per-provider ``ss.provider_remember["wandb"]`` before
    delegating to :func:`connect_provider`.
    """
    ss = st.session_state
    _sync_inference_inputs()
    # Mirror the legacy flag onto the per-provider one if a Settings
    # page from a previous build is still in use.
    if "remember_wb_key" in ss and "_remember_input" not in ss:
        flags = dict(ss.provider_remember)
        flags["wandb"] = bool(ss.remember_wb_key)
        ss.provider_remember = flags
    connect_provider("wandb")
    if ss.connect_errors.get("wandb") is None and ss.clients.get("wandb") is not None:
        ss.conn_open = False


def disconnect() -> None:
    """Legacy shim — disconnect the W&B Inference provider."""
    disconnect_provider("wandb")


def forget_saved_wb_key() -> None:
    """Legacy shim — forget the persisted W&B API key from disk."""
    forget_provider_key("wandb")


# ---------------------------------------------------------------------------
# GitHub identity
# ---------------------------------------------------------------------------
def _persist_profile_from_state() -> None:
    """Snapshot the relevant ``ss.*`` keys into a Profile and save it.

    Includes the theme preference if (and only if) the user has explicitly
    picked one via the Settings page switcher; an unset preference stays
    empty on disk so :mod:`theme_switcher` can fall back to whatever's
    already in browser ``localStorage``. The font-size preference is
    saved verbatim — it has no migration path and an empty string just
    means "keep the bundled ``baseFontSize`` default".
    """
    ss = st.session_state
    identity = ss.github_identity or {}
    theme = str(ss.get("theme_pref") or "")
    if theme not in ("System", "Light", "Dark"):
        theme = ""
    if not ss.get("theme_explicit"):
        theme = ""
    font_size = str(ss.get("font_size_pref") or "")
    if font_size not in ("Extra small", "Small", "Medium", "Large", "Extra large"):
        font_size = ""
    profile = account.Profile(
        github_username=identity.get("login", ""),
        github_email=identity.get("email", ""),
        github_avatar_url=identity.get("avatar_url", ""),
        github_scopes=list(identity.get("scopes", [])),
        theme=theme,
        font_size=font_size,
    )
    account.save_profile(profile)


def verify_pat() -> None:
    """Verify ``ss.pat_input`` against GitHub's ``/user`` endpoint and persist."""
    ss = st.session_state
    pat = (ss.get("pat_input") or "").strip()
    try:
        identity = account.verify_github_pat(pat)
    except ValueError as e:
        ss.github_pat_error = str(e)
        return
    ss.github_pat_error = None
    ss.github_pat = pat
    ss.github_identity = identity

    avatar_bytes = account.fetch_avatar_bytes(identity.get("avatar_url", ""))
    if avatar_bytes:
        ss.avatar_bytes = avatar_bytes

    creds = account.load_credentials()
    creds["github_pat"] = pat
    account.save_credentials(creds)
    _persist_profile_from_state()
    ss["pat_input"] = ""


def sign_out_github() -> None:
    """Clear the verified GitHub identity and saved PAT but keep avatar bytes off."""
    ss = st.session_state
    account.clear_credentials(github=True)
    ss.github_pat = ""
    ss.github_identity = None
    ss.github_pat_error = None
    ss.avatar_bytes = None
    ss.git_identity_applied = set()
    _persist_profile_from_state()


# ---------------------------------------------------------------------------
# Theme switcher
# ---------------------------------------------------------------------------
def set_theme_pref() -> None:
    """``on_change`` callback for the Settings page's theme segmented control.

    The widget binds to ``ss.theme_pref`` via ``key=``, so by the time this
    callback runs the new value is already in session state. We mark the
    choice as explicit so :mod:`theme_switcher` switches into write-and-
    reload mode on the next rerun, and we persist the preference to disk
    so it survives across sessions. The actual ``localStorage`` write +
    page reload happens inside the component on the next mount.
    """
    ss = st.session_state
    val = ss.get("theme_pref")
    if val not in ("System", "Light", "Dark"):
        return
    ss.theme_explicit = True
    _persist_profile_from_state()


def theme_detected() -> None:
    """``on_detected_change`` callback for :mod:`theme_switcher`.

    Fires when the component detects a pre-existing localStorage value
    (e.g. from Streamlit's legacy toolbar toggle) that differs from the
    Python-side preference. We adopt the detected value as the user's
    explicit choice so the next render's segmented control reflects what
    the page is actually painted as, and persist it. Streamlit only fires
    this callback when the trigger value actually changes, so it runs at
    most once per browser-session migration — there's no rerun storm.
    """
    ss = st.session_state
    state = ss.get("wb_theme_switcher") or {}
    detected = getattr(state, "detected", None) or (
        state.get("detected") if isinstance(state, dict) else None
    )
    if detected not in ("System", "Light", "Dark"):
        return
    ss.theme_pref = detected
    ss.theme_explicit = True
    _persist_profile_from_state()


# ---------------------------------------------------------------------------
# Font size switcher
# ---------------------------------------------------------------------------
def set_font_size_pref() -> None:
    """``on_change`` callback for the Settings page's font-size segmented control.

    The widget binds to ``ss.font_size_pref`` via ``key=``, so by the
    time this callback runs the new value is already in session state.
    We just persist the choice so it survives across sessions; the
    actual CSS injection happens on the next rerun, when
    :mod:`font_size_switcher` is re-mounted from ``streamlit_app.main()``
    with the new label.
    """
    ss = st.session_state
    val = ss.get("font_size_pref")
    if val not in ("Extra small", "Small", "Medium", "Large", "Extra large"):
        return
    _persist_profile_from_state()
