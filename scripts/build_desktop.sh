#!/usr/bin/env bash
# Build the desktop binary end-to-end from a clean shell.
#
# This wraps the documented manual flow in AGENTS.md ("Desktop build") so
# rebuilds are a one-liner. The script is idempotent — it can be re-run any
# time and only does work that's actually needed.
#
# What it does:
#   1. Ensures `.venv-build/` exists at the repo root, created with Python
#      3.12 via `uv` if missing (streamlit-desktop-app 0.3.4 caps Python at
#      3.12, so a project-local venv pinned to that interpreter is required).
#   2. Syncs the desktop-build dependencies into `.venv-build`:
#      `uv pip install -e '.[desktop]'`. This also picks up any new entries
#      in `[project].dependencies` (e.g. `weave`) the next time pyproject
#      changes, so the build venv never goes stale.
#   3. Invokes `python scripts/build_desktop.py` using `.venv-build`'s
#      interpreter. That Python script remains the single source of truth
#      for PyInstaller flags; this shell script only handles bootstrapping.
#
# Output lands under `dist/`. See `scripts/build_desktop.py` and AGENTS.md
# for artifact-shape details and the Gatekeeper note for macOS.
#
# Usage (from anywhere — paths are resolved relative to the script):
#   ./scripts/build_desktop.sh
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_dir="${repo_root}/.venv-build"

cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[build_desktop.sh] Error: 'uv' is required but not installed." >&2
  echo "[build_desktop.sh] Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -d "${venv_dir}" ]]; then
  echo "[build_desktop.sh] Creating ${venv_dir} (Python 3.12)..."
  uv venv --python 3.12 "${venv_dir}"
fi

echo "[build_desktop.sh] Syncing build dependencies into ${venv_dir}..."
VIRTUAL_ENV="${venv_dir}" uv pip install -e '.[desktop]'

# Activate the venv for the build step so the venv's `bin/` is on PATH and
# `build_desktop.py`'s `subprocess.call(["streamlit-desktop-app", ...])` can
# find the CLI. We don't `source bin/activate` because it isn't compatible
# with `set -u` (it references unbound vars on macOS); prepending PATH and
# setting VIRTUAL_ENV achieves the same result.
echo "[build_desktop.sh] Running scripts/build_desktop.py..."
PATH="${venv_dir}/bin:${PATH}" \
  VIRTUAL_ENV="${venv_dir}" \
  "${venv_dir}/bin/python" scripts/build_desktop.py

echo
echo "[build_desktop.sh] Build complete."
case "$(uname -s)" in
  Darwin)
    echo "[build_desktop.sh] Artifact: dist/WB Coding Agent.app"
    echo
    echo "[build_desktop.sh] First-launch hint (unsigned build, Gatekeeper):"
    echo "[build_desktop.sh]   right-click -> Open the first time, OR run"
    echo "[build_desktop.sh]   xattr -dr com.apple.quarantine \"dist/WB Coding Agent.app\""
    ;;
  *)
    echo "[build_desktop.sh] Artifact: dist/WB Coding Agent/"
    ;;
esac
