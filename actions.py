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
- The W&B Inference Connect / Disconnect / Forget callbacks (which build
  the OpenAI client, list models, and bootstrap Weave).
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
from wb_client import init_weave, list_models, make_client


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
# W&B Inference connection
# ---------------------------------------------------------------------------
def _sort_models(models: list[str]) -> list[str]:
    """Sort model ids by their display label, case-insensitive."""
    from models import model_label
    return sorted(models, key=lambda m: model_label(m).casefold())


def _connect(api_key: str, project: str) -> None:
    """Build the OpenAI client + bootstrap Weave. Mutates session state in place."""
    api_key = api_key.strip()
    project = project.strip()
    if not api_key:
        st.session_state.connect_error = "Enter a W&B API key first."
        return
    try:
        client = make_client(api_key=api_key, project=project or None)
        models = list_models(client)
    except Exception as e:
        st.session_state.client = None
        st.session_state.models = []
        st.session_state.connect_error = f"Could not connect: {e}"
        return
    st.session_state.client = client
    st.session_state.models = _sort_models(models)
    st.session_state.connect_error = None
    if st.session_state.model not in st.session_state.models:
        st.session_state.model = (
            st.session_state.models[0] if st.session_state.models else None
        )

    try:
        _, resolved_project, weave_url = init_weave(
            api_key=api_key, project=project or None
        )
        st.session_state.weave_project = resolved_project
        st.session_state.weave_url = weave_url
        st.session_state.weave_error = None
    except Exception as e:
        st.session_state.weave_project = None
        st.session_state.weave_url = None
        st.session_state.weave_error = str(e)


def _sync_inference_inputs() -> None:
    """Pull the latest values from the settings widget keys into the canonical
    state keys.

    The settings page uses the dual-key pattern (canonical state in
    ``api_key`` / ``project`` / ``remember_wb_key``; widget instances in
    ``_api_key_input`` / ``_project_input`` / ``_remember_input``) so the
    canonical values survive page navigation. Each widget has an
    ``on_change`` sync callback, but we re-sync defensively here so a
    Connect click that fires *without* a prior on_change (e.g. when the
    user clicks Connect immediately after typing without blurring) still
    sees the latest typed value.
    """
    ss = st.session_state
    if "_api_key_input" in ss:
        ss.api_key = ss._api_key_input
    if "_project_input" in ss:
        ss.project = ss._project_input
    if "_remember_input" in ss:
        ss.remember_wb_key = bool(ss._remember_input)


def on_connect() -> None:
    """Connect button callback. Honors the ``Remember on this machine`` toggle."""
    ss = st.session_state
    _sync_inference_inputs()
    _connect(ss.api_key, ss.project)
    if ss.client is not None and ss.connect_error is None:
        ss.conn_open = False
        creds = account.load_credentials()
        if ss.remember_wb_key and ss.api_key.strip():
            creds["wb_api_key"] = ss.api_key.strip()
            account.save_credentials(creds)
        elif not ss.remember_wb_key:
            account.clear_credentials(wb=True)


def disconnect() -> None:
    """Drop the live client + models without forgetting the saved API key."""
    ss = st.session_state
    ss.client = None
    ss.models = []
    ss.model = None
    ss.weave_project = None
    ss.weave_url = None
    ss.weave_error = None
    ss.connect_error = None


def forget_saved_wb_key() -> None:
    """Remove the persisted W&B API key from disk and clear the session value."""
    account.clear_credentials(wb=True)
    ss = st.session_state
    ss.api_key = ""
    ss.remember_wb_key = False
    # Drop the underlying widget keys so the visible form clears on rerun
    # (without this, the text input would still show the previously typed
    # value because Streamlit keeps the widget key as the source of truth
    # once it has been instantiated).
    for k in ("_api_key_input", "_remember_input"):
        if k in ss:
            del ss[k]


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
