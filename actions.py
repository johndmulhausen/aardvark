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
        _, resolved_project = init_weave(api_key=api_key, project=project or None)
        st.session_state.weave_project = resolved_project
        st.session_state.weave_error = None
    except Exception as e:
        st.session_state.weave_project = None
        st.session_state.weave_error = str(e)


def on_connect() -> None:
    """Connect button callback. Honors the ``Remember on this machine`` toggle."""
    ss = st.session_state
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
    ss.weave_error = None
    ss.connect_error = None


def forget_saved_wb_key() -> None:
    """Remove the persisted W&B API key from disk and clear the session value."""
    account.clear_credentials(wb=True)
    st.session_state.api_key = ""
    st.session_state.remember_wb_key = False


# ---------------------------------------------------------------------------
# GitHub identity
# ---------------------------------------------------------------------------
def _persist_profile_from_state() -> None:
    """Snapshot the relevant ``ss.*`` keys into a Profile and save it."""
    ss = st.session_state
    identity = ss.github_identity or {}
    profile = account.Profile(
        github_username=identity.get("login", ""),
        github_email=identity.get("email", ""),
        github_avatar_url=identity.get("avatar_url", ""),
        github_scopes=list(identity.get("scopes", [])),
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
