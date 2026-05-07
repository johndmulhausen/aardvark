"""Build the desktop binary using `streamlit-desktop-app`.

Run this from the repo root after installing the desktop extras::

    pip install -e '.[desktop]'
    python scripts/build_desktop.py

The script forwards to the ``streamlit-desktop-app build`` CLI with a fixed set
of flags so that local builds and CI builds always produce the same artifact
shape. The output lands under ``dist/``.

Artifact shape per platform:

- macOS: ``dist/WB Coding Agent.app`` - a real ``.app`` bundle with
  ``Contents/Info.plist`` and the binary at ``Contents/MacOS/...``. Drag to
  ``/Applications`` to install. First launch is gated by Gatekeeper because
  the build is unsigned; right-click -> Open the first time, or run
  ``xattr -d com.apple.quarantine "dist/WB Coding Agent.app"``.
- Linux / Windows: ``dist/WB Coding Agent/`` (PyInstaller "onedir" layout)
  with the launcher executable inside. Distribute as a zip / tarball.

The Streamlit chrome (hamburger menu and "Deploy" button) is hidden in the
packaged app via ``--streamlit-options --client.toolbarMode minimal``. The
same options live in ``.streamlit/config.toml`` for the local-dev workflow;
see that file for the rationale and keep the two in sync.

Note: ``streamlit-desktop-app`` 0.3.4 supports CPython 3.9-3.12 only. Build
this with a 3.12 interpreter (e.g. ``uv venv --python 3.12 .venv-build``).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

APP_NAME = "WB Coding Agent"
ENTRY_SCRIPT = "streamlit_app.py"

# Sibling modules imported by ``streamlit_app.py`` at runtime. These are plain
# .py files (not an installed package), so PyInstaller's static import graph
# does not pick them up - we must ship them as data files alongside the
# bundled entry script. Add to this list whenever a new top-level module is
# imported by ``streamlit_app.py``; see AGENTS.md.
#
# Paths may include a directory prefix (e.g. ``app_pages/chat.py``); the
# destination inside the bundle preserves the same relative location so
# ``st.navigation`` can find the Page modules at runtime.
BUNDLED_MODULES: tuple[str, ...] = (
    "agent.py",
    "tools.py",
    "wb_client.py",
    "mcp_servers.py",
    "project_context.py",
    "chat_input.py",
    "theme_switcher.py",
    "font_size_switcher.py",
    "models.py",
    "account.py",
    "actions.py",
    "usage.py",
    "git_ops.py",
    "chats.py",
    "commit_ai.py",
    # Phase 1–6 multi-provider modules.
    "providers.py",
    "chat_streams.py",
    "model_catalog.py",
    "fullscreen_trigger.py",
    "attachments.py",
    "_artifact_paths.py",
    "app_pages/__init__.py",
    "app_pages/chat.py",
    "app_pages/usage.py",
    "app_pages/settings.py",
    "app_pages/docs.py",
    # Repo-bundled LiteLLM pricing snapshot — the offline floor for
    # ``model_catalog._load_litellm_registry``. Without this, a fresh
    # install with no network connectivity would have no pricing data
    # for any model; the strict completeness gate would hide
    # everything from the picker until the first online refresh.
    "data/litellm_model_registry.json",
    "data/LITELLM_REGISTRY_SHA.txt",
)

# Third-party packages imported by ``BUNDLED_MODULES`` (not by
# ``streamlit_app.py`` directly). Because we ship those modules as data
# files, PyInstaller's import graph does not see their ``import`` statements,
# so it has no way to discover their transitive third-party dependencies. We
# force-collect those packages here. ``--collect-all`` pulls in submodules,
# data files, and dynamic libraries.
COLLECT_ALL_PACKAGES: tuple[str, ...] = (
    "openai",
    "weave",
    "mcp",
    "httpx",
    # Phase 1–6 deps. ``litellm`` is the call layer for 9 of the 12
    # providers; ``anthropic`` and ``google.genai`` are the native SDKs
    # for the other two; ``pypdf`` is used by ``attachments.extract_text``
    # for non-native PDF-input models; ``PIL`` is needed by
    # ``attachments.preprocess_image`` (transitive Streamlit dep, but
    # we collect it explicitly so PyInstaller picks up the bundled
    # image codecs).
    "anthropic",
    "google.genai",
    "litellm",
    "pypdf",
    "PIL",
)

# Streamlit CLI options forwarded to the bundled app at launch. These mirror
# the values in .streamlit/config.toml because that file is not picked up
# from inside the .app bundle (cwd at launch is /, not the project root).
# Keep this list in sync with .streamlit/config.toml.
# Note on ``theme.baseFontSize``: in ``.streamlit/config.toml`` we duplicate
# this into both ``[theme.light]`` and ``[theme.dark]`` so the app stays
# multi-mode (which is what makes the in-app theme switcher's
# ``localStorage`` writes actually take effect on boot). Streamlit's CLI
# does not expose per-mode variants for ``baseFontSize`` (only color
# options have ``--theme.light.X`` / ``--theme.dark.X`` flags), so the
# packaged build uses the top-level form. The in-app switcher on the
# Settings page still works because Streamlit's theme-mode selection
# logic still respects the localStorage key when both a top-level
# ``baseFontSize`` and the absence of color overrides leaves Light/Dark
# resolution to the user.
STREAMLIT_OPTIONS: tuple[tuple[str, str], ...] = (
    # ``"minimal"`` hides Streamlit's hamburger / Settings / Deploy chrome
    # entirely. The app's own Settings page exposes a Light / Dark /
    # System switcher that drives the same ``localStorage`` key Streamlit
    # reads on app boot, so users still control the theme without seeing
    # the framework's own menu. Mirror in ``.streamlit/config.toml``.
    ("client.toolbarMode", "minimal"),
    ("browser.gatherUsageStats", "false"),
    ("theme.baseFontSize", "12"),
    # Primary accent color. Mirrors the ``[theme.light].primaryColor``
    # / ``[theme.dark].primaryColor`` values in
    # ``.streamlit/config.toml``. Streamlit exposes per-mode CLI flags
    # for color options (unlike ``baseFontSize``), so we forward both
    # so Light and Dark modes stay in sync inside the packaged build.
    ("theme.light.primaryColor", "#0080FF"),
    ("theme.dark.primaryColor", "#0080FF"),
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"

ICON_BY_PLATFORM: dict[str, Path] = {
    "Darwin": ASSETS / "icon.icns",
    "Windows": ASSETS / "icon.ico",
    "Linux": ASSETS / "icon.png",
}


def _resolve_icon() -> Path | None:
    """Return the platform-specific icon path, or None if it is missing.

    PyInstaller fails the build if ``--icon`` is given but the file does not
    exist, so we skip the flag rather than break the build when assets have
    not been generated yet.
    """
    icon = ICON_BY_PLATFORM.get(platform.system())
    if icon is None or not icon.exists():
        return None
    return icon


def main() -> int:
    cmd: list[str] = [
        "streamlit-desktop-app",
        "build",
        ENTRY_SCRIPT,
        "--name",
        APP_NAME,
    ]

    icon = _resolve_icon()
    if icon is not None:
        cmd.extend(["--icon", str(icon)])
    else:
        print(
            f"[build_desktop] No icon found for {platform.system()}; "
            "building without one.",
            file=sys.stderr,
        )

    # --windowed (no console window; on macOS this is the flag that makes
    # PyInstaller emit a real .app bundle). We deliberately avoid --onefile
    # so the macOS output is a normal directory-style .app the user can drag
    # to /Applications, instead of a self-extracting blob with slow cold
    # start.
    pyinstaller_opts: list[str] = ["--windowed", "--noconfirm"]
    # PyInstaller's --add-data takes "SRC<sep>DEST". Separator is ':' on
    # POSIX and ';' on Windows (os.pathsep handles both).
    sep = os.pathsep
    for module in BUNDLED_MODULES:
        module_path = REPO_ROOT / module
        if not module_path.exists():
            print(
                f"[build_desktop] WARNING: bundled module {module} not found "
                f"at {module_path}; skipping.",
                file=sys.stderr,
            )
            continue
        # Preserve the file's relative directory inside the bundle so that
        # subpackages like ``app_pages/`` end up at the same import path
        # they have in the source tree. ``Path("foo.py").parent`` is ``.``,
        # which is the bundle root — exactly what we want for top-level
        # modules.
        rel_dir = str(Path(module).parent) or "."
        pyinstaller_opts.extend(["--add-data", f"{module_path}{sep}{rel_dir}"])

    for pkg in COLLECT_ALL_PACKAGES:
        pyinstaller_opts.extend(["--collect-all", pkg])

    cmd.append("--pyinstaller-options")
    cmd.extend(pyinstaller_opts)

    # The CLI splits "everything after --streamlit-options" off from the
    # pyinstaller options list, so streamlit options must come last.
    streamlit_opts: list[str] = []
    for key, value in STREAMLIT_OPTIONS:
        streamlit_opts.extend([f"--{key}", value])
    cmd.append("--streamlit-options")
    cmd.extend(streamlit_opts)

    print(f"[build_desktop] Running: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
