"""Account, GitHub identity, and credential helpers.

Pure-stdlib helpers backing the settings popover in ``streamlit_app.py``. This
module owns *every* read and write of the on-disk account state so the UI
layer never has to know the file layout. Specifically:

- ``~/.wb_coding_agent/credentials.json`` (mode 0600) — opt-in saved W&B API
  key and the GitHub personal access token. Either field is omitted when not
  saved.
- ``~/.wb_coding_agent/preferences.json`` (default mode) — non-secret profile
  data: GitHub display fields and scopes. (Avatars are sourced live from
  GitHub on PAT verify and cached only for the current session — there is
  no avatar uploader and no on-disk avatar file by design.)

GitHub authentication is plain personal-access-token auth: the user pastes a
fine-grained PAT and we verify it by calling ``GET /user`` against the
GitHub REST API via :mod:`urllib.request` (no new dependency). See the
settings popover for the user-facing copy that explains scopes.

This module deliberately has *no* Streamlit imports so it stays trivially
unit-testable and so non-UI callers (e.g. ``app_pages/chat.py``'s git-author
stamping) can use it freely.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".wb_coding_agent"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
PREFERENCES_FILE = CONFIG_DIR / "preferences.json"

# GitHub URLs surfaced in the UI. Centralized so the popover and the README
# can pull from the same place.
GITHUB_PAT_CREATE_URL = "https://github.com/settings/personal-access-tokens/new"
GITHUB_API_USER_URL = "https://api.github.com/user"
GITHUB_API_USER_REPOS_URL = "https://api.github.com/user/repos"

# Recommended scopes. We don't require any specific scope at verify time —
# ``GET /user`` works with the implicit "read" scope on a fine-grained PAT —
# but we surface these as a hint in the UI.
RECOMMENDED_SCOPES = [
    "read:user",
    "user:email",
    "repo (optional, only if you want the agent to push commits)",
]

@dataclass
class Profile:
    """Non-secret account preferences persisted to disk.

    Secrets (W&B API key, GitHub PAT) live in :data:`CREDENTIALS_FILE` and
    are loaded via :func:`load_credentials`; everything in this dataclass is
    safe to write at default file permissions.

    Theme is intentionally NOT a field here: Streamlit owns the theme
    toggle (in its toolbar Settings menu) and persists the user's choice
    in browser storage, so layering our own preference file on top would
    be redundant — and there's no programmatic way for the app to apply
    such a preference at startup anyway.

    Avatars are intentionally NOT a field here either: the popover does
    not support uploading a custom avatar, and the GitHub avatar is
    fetched fresh into in-memory bytes on PAT verify.
    """

    github_username: str = ""
    github_email: str = ""
    github_avatar_url: str = ""
    github_scopes: list[str] = field(default_factory=list)


def _ensure_config_dir() -> None:
    """Create ``~/.wb_coding_agent/`` if missing. Idempotent."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> Profile:
    """Read the persisted profile, returning a default-populated one on any failure."""
    try:
        raw = json.loads(PREFERENCES_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return Profile()
    if not isinstance(raw, dict):
        return Profile()
    scopes = raw.get("github_scopes")
    if not isinstance(scopes, list):
        scopes = []
    return Profile(
        github_username=str(raw.get("github_username") or ""),
        github_email=str(raw.get("github_email") or ""),
        github_avatar_url=str(raw.get("github_avatar_url") or ""),
        github_scopes=[str(s) for s in scopes],
    )


def save_profile(profile: Profile) -> None:
    """Persist ``profile`` to :data:`PREFERENCES_FILE`."""
    _ensure_config_dir()
    PREFERENCES_FILE.write_text(
        json.dumps(asdict(profile), indent=2),
        encoding="utf-8",
    )


def load_credentials() -> dict[str, str]:
    """Read the credentials file, returning ``{}`` on any failure.

    Possible keys: ``wb_api_key``, ``github_pat``. Missing keys mean the user
    has not opted in to persisting that secret.
    """
    try:
        raw = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if isinstance(v, str) and v}


def save_credentials(creds: dict[str, str]) -> None:
    """Write ``creds`` atomically with ``0600`` permissions.

    Empty values are dropped so calling ``save_credentials({"wb_api_key": ""})``
    effectively unsets that key. The file is created with mode 0600 from the
    start by writing into a temp file with that mode and renaming over the
    target — this avoids a window where the file exists with the umask's
    default permissions.
    """
    _ensure_config_dir()
    cleaned = {k: v for k, v in creds.items() if isinstance(v, str) and v}
    tmp = CREDENTIALS_FILE.with_suffix(".json.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.replace(tmp, CREDENTIALS_FILE)
    # ``os.replace`` preserves the source file's mode bits; assert anyway in
    # case a future edit changes that path.
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)
    except OSError:
        pass


def clear_credentials(*, wb: bool = False, github: bool = False) -> None:
    """Drop one or both stored secrets.

    Pass ``wb=True`` to forget the W&B API key, ``github=True`` to forget the
    GitHub PAT. With both False this is a no-op.
    """
    if not (wb or github):
        return
    creds = load_credentials()
    if wb:
        creds.pop("wb_api_key", None)
    if github:
        creds.pop("github_pat", None)
    if creds:
        save_credentials(creds)
    else:
        try:
            CREDENTIALS_FILE.unlink()
        except FileNotFoundError:
            pass


def verify_github_pat(pat: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """Call ``GET /user`` to validate a PAT and return identity fields.

    Raises ``ValueError`` with a user-friendly message on any failure (HTTP
    error, network error, non-JSON response). On success returns a dict with
    keys ``login``, ``name``, ``email``, ``avatar_url``, ``scopes`` (list).

    The ``scopes`` list is parsed from GitHub's ``X-OAuth-Scopes`` response
    header. Fine-grained PATs return an empty list there; classic PATs
    return their granted scope list. We use this purely to surface to the
    user what their token can do.

    The PAT itself is never logged. We pass it as the bearer in the
    Authorization header per the GitHub REST API docs.
    """
    pat = (pat or "").strip()
    if not pat:
        raise ValueError("Personal access token is empty.")

    req = urllib.request.Request(
        GITHUB_API_USER_URL,
        headers={
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "wb-coding-agent",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read()
            scopes_header = resp.headers.get("X-OAuth-Scopes", "") or ""
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ValueError("GitHub rejected the token (401 Unauthorized). Check it and try again.") from e
        if e.code == 403:
            raise ValueError("GitHub returned 403 Forbidden. The token may be missing required permissions.") from e
        raise ValueError(f"GitHub returned HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach GitHub: {e.reason}") from e
    except OSError as e:
        raise ValueError(f"Network error talking to GitHub: {e}") from e

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError("GitHub response was not valid JSON.") from e
    if not isinstance(body, dict):
        raise ValueError("GitHub response had unexpected shape.")

    scopes = [s.strip() for s in scopes_header.split(",") if s.strip()]
    return {
        "login": str(body.get("login") or ""),
        "name": str(body.get("name") or ""),
        "email": str(body.get("email") or ""),
        "avatar_url": str(body.get("avatar_url") or ""),
        "scopes": scopes,
    }


def fetch_avatar_bytes(avatar_url: str, *, timeout: float = 10.0) -> bytes | None:
    """Download a GitHub avatar URL into memory. Returns ``None`` on failure.

    Used by the popover to cache the verified user's avatar to disk so the
    sidebar can render it without doing an HTTP round-trip on every rerun.
    Failures are intentionally silent — a missing avatar simply falls back
    to a Material icon in the UI.
    """
    if not avatar_url:
        return None
    req = urllib.request.Request(
        avatar_url,
        headers={"User-Agent": "wb-coding-agent"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError):
        return None


def apply_git_identity(working_dir: Path, name: str, email: str) -> tuple[bool, str]:
    """Stamp ``user.name`` / ``user.email`` into the working dir's git config.

    Used so commits the agent makes via ``run_shell`` are authored as the
    GitHub identity the user verified. Runs ``git config --local user.name``
    and ``git config --local user.email`` inside ``working_dir``. Returns
    ``(ok, message)``: ``ok`` is False on any failure (not a git repo, git
    not installed, command failed) so the UI can show a non-fatal warning.

    No-op when ``name`` or ``email`` is empty — there is nothing to apply.
    """
    name = (name or "").strip()
    email = (email or "").strip()
    if not name and not email:
        return False, "No GitHub identity to apply."
    if shutil.which("git") is None:
        return False, "git is not installed on PATH."
    if not (working_dir / ".git").exists():
        return False, f"{working_dir} is not a git repository."
    try:
        if name:
            subprocess.run(
                ["git", "config", "--local", "user.name", name],
                cwd=str(working_dir),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        if email:
            subprocess.run(
                ["git", "config", "--local", "user.email", email],
                cwd=str(working_dir),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() or f"git config exited with {e.returncode}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"Could not run git config: {e}"
    return True, f"Set git author to {name} <{email}>."


# ---------------------------------------------------------------------------
# Project bootstrap helpers
# ---------------------------------------------------------------------------
# These back the "Start a new project" dialog in ``app_pages/chat.py``. They
# all raise :class:`ValueError` with a user-friendly message on any failure so
# the dialog can render a single ``st.error`` and let the user retry without
# half-finishing the bootstrap. The PAT is only ever passed in via argument
# (never read from disk here) so callers control the lifecycle.


def _github_request(
    pat: str,
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> tuple[dict[str, Any] | list[Any], dict[str, str]]:
    """Issue a GitHub REST API call as the user identified by ``pat``.

    Returns ``(parsed_body, headers)`` on success. Raises ``ValueError`` with
    a user-friendly message on any failure. The PAT is never logged.
    """
    pat = (pat or "").strip()
    if not pat:
        raise ValueError("GitHub personal access token is missing. Verify a PAT in the account menu first.")

    data: bytes | None = None
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "wb-coding-agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read()
            resp_headers = {k: v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        # Try to surface GitHub's error message for context (validation errors
        # from POST /user/repos in particular are useful — e.g. "name already
        # exists on this account").
        detail = ""
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            if isinstance(err_body, dict):
                detail = str(err_body.get("message") or "")
        except Exception:
            detail = ""
        if e.code == 401:
            raise ValueError("GitHub rejected the token (401 Unauthorized). Re-verify your PAT.") from e
        if e.code == 403:
            raise ValueError(
                f"GitHub returned 403 Forbidden{': ' + detail if detail else ''}. "
                "The token may be missing required permissions."
            ) from e
        if e.code == 422 and detail:
            raise ValueError(f"GitHub rejected the request: {detail}") from e
        raise ValueError(
            f"GitHub returned HTTP {e.code}: {e.reason}{': ' + detail if detail else ''}"
        ) from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach GitHub: {e.reason}") from e
    except OSError as e:
        raise ValueError(f"Network error talking to GitHub: {e}") from e

    try:
        parsed = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError("GitHub response was not valid JSON.") from e
    return parsed, resp_headers


def list_user_repos(pat: str, *, timeout: float = 15.0) -> list[dict[str, Any]]:
    """List repositories owned by the authenticated GitHub user.

    Paginates ``GET /user/repos`` (per_page=100, affiliation=owner,
    sort=updated) up to a hard cap of 5 pages (500 repos). Returns one dict
    per repo with the keys the dialog needs:

    - ``full_name`` (``owner/name``)
    - ``name``
    - ``clone_url``
    - ``ssh_url``
    - ``private`` (bool)
    - ``description`` (may be empty)
    - ``updated_at`` (ISO 8601 string)
    - ``default_branch``

    Raises ``ValueError`` on any HTTP / network / parsing failure.
    """
    out: list[dict[str, Any]] = []
    for page in range(1, 6):
        url = (
            f"{GITHUB_API_USER_REPOS_URL}"
            f"?per_page=100&affiliation=owner&sort=updated&page={page}"
        )
        body, _headers = _github_request(pat, url, timeout=timeout)
        if not isinstance(body, list):
            raise ValueError("Unexpected response shape from GitHub.")
        if not body:
            break
        for item in body:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "full_name": str(item.get("full_name") or ""),
                    "name": str(item.get("name") or ""),
                    "clone_url": str(item.get("clone_url") or ""),
                    "ssh_url": str(item.get("ssh_url") or ""),
                    "private": bool(item.get("private")),
                    "description": str(item.get("description") or ""),
                    "updated_at": str(item.get("updated_at") or ""),
                    "default_branch": str(item.get("default_branch") or ""),
                }
            )
        if len(body) < 100:
            break
    return out


def create_user_repo(
    pat: str,
    name: str,
    *,
    description: str = "",
    private: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Create a new repo on GitHub for the authenticated user.

    Calls ``POST /user/repos`` with ``auto_init=False`` so the local
    ``git init`` + first commit remain canonical (GitHub adding its own
    initial commit would force the user to merge before pushing). Returns
    the parsed response dict; the dialog uses ``clone_url`` to wire
    ``origin``.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Repository name is required.")
    payload: dict[str, Any] = {
        "name": name,
        "private": bool(private),
        "auto_init": False,
    }
    desc = (description or "").strip()
    if desc:
        payload["description"] = desc
    body, _headers = _github_request(
        pat,
        GITHUB_API_USER_REPOS_URL,
        method="POST",
        body=payload,
        timeout=timeout,
    )
    if not isinstance(body, dict):
        raise ValueError("Unexpected response shape from GitHub.")
    return body


def _require_git() -> None:
    """Raise :class:`ValueError` if ``git`` is not on PATH."""
    if shutil.which("git") is None:
        raise ValueError("git is not installed on PATH.")


def git_init(dest: Path) -> None:
    """Run ``git init`` inside ``dest``. Raises ``ValueError`` on failure.

    The caller is responsible for creating ``dest`` first; this function
    asserts the directory exists so a typo doesn't silently create a repo
    in an unexpected location.
    """
    _require_git()
    if not dest.exists() or not dest.is_dir():
        raise ValueError(f"Cannot init: {dest} is not an existing directory.")
    try:
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=str(dest),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"git init failed: {e.stderr.strip() or e.stdout.strip() or e.returncode}"
        ) from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"Could not run git init: {e}") from e


def git_clone(
    pat: str | None,
    https_url: str,
    dest: Path,
    *,
    timeout: float = 300.0,
) -> None:
    """Clone ``https_url`` into ``dest`` using ``pat`` for authentication.

    The PAT is supplied via ``git -c http.extraheader='Authorization: Bearer
    <pat>'`` so it never lands in the cloned repo's ``.git/config`` and never
    appears in URLs. Subsequent fetch/push goes through the user's regular
    git credential helper. When ``pat`` is ``None`` we fall back to plain
    ``git clone`` (e.g. for non-GitHub remotes the user pastes manually).

    ``dest`` must not already exist — git creates it. Raises ``ValueError``
    on any failure.
    """
    _require_git()
    https_url = (https_url or "").strip()
    if not https_url:
        raise ValueError("Clone URL is required.")
    if dest.exists():
        raise ValueError(f"Destination already exists: {dest}")
    if not dest.parent.exists():
        raise ValueError(f"Parent directory does not exist: {dest.parent}")

    cmd: list[str] = ["git"]
    if pat:
        cmd.extend(["-c", f"http.extraheader=Authorization: Bearer {pat.strip()}"])
    cmd.extend(["clone", "--quiet", https_url, str(dest)])

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        # Sanitize stderr in case git echoes the extraheader value back.
        msg = (e.stderr or e.stdout or "").strip()
        if pat:
            msg = msg.replace(pat.strip(), "<redacted>")
        raise ValueError(f"git clone failed: {msg or e.returncode}") from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"Could not run git clone: {e}") from e


def git_add_remote(working_dir: Path, name: str, url: str) -> None:
    """Run ``git remote add <name> <url>`` inside ``working_dir``.

    Validates that ``working_dir/.git`` exists so an accidental call against
    a non-repo gives a clear error. Raises ``ValueError`` on failure.
    """
    _require_git()
    name = (name or "").strip()
    url = (url or "").strip()
    if not name or not url:
        raise ValueError("Remote name and URL are required.")
    if not (working_dir / ".git").exists():
        raise ValueError(f"{working_dir} is not a git repository.")
    try:
        subprocess.run(
            ["git", "remote", "add", name, url],
            cwd=str(working_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"git remote add failed: {e.stderr.strip() or e.returncode}"
        ) from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"Could not run git remote add: {e}") from e


def create_project_directory(parent: Path, name: str) -> Path:
    """Create ``parent / name`` as a fresh empty directory and return it.

    Validates that ``name`` is a single non-empty path component (no path
    separators, not ``.``/``..``) so the user can't accidentally escape
    ``parent`` or stuff a partial path into the field. The parent must
    already exist — we do not auto-create intermediate directories. If the
    target already exists it must be empty (so the user can re-aim a typo
    without losing data); otherwise we raise.

    Returns the absolute resolved path. Raises ``ValueError`` on any
    validation or filesystem failure.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Folder name is required.")
    if name in (".", ".."):
        raise ValueError("Folder name cannot be '.' or '..'.")
    if any(sep in name for sep in ("/", "\\")):
        raise ValueError("Folder name cannot contain path separators.")
    if not parent.exists() or not parent.is_dir():
        raise ValueError(f"Parent directory does not exist: {parent}")

    dest = (parent / name).resolve()
    try:
        dest.relative_to(parent.resolve())
    except ValueError as e:
        raise ValueError("Folder name escapes the parent directory.") from e

    if dest.exists():
        try:
            entries = list(dest.iterdir())
        except OSError as e:
            raise ValueError(f"Could not inspect {dest}: {e}") from e
        if entries:
            raise ValueError(f"{dest} already exists and is not empty.")
        return dest

    try:
        dest.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        # Race against another process — treat the same as "exists".
        return dest
    except OSError as e:
        raise ValueError(f"Could not create {dest}: {e}") from e
    return dest
