"""Git command execution and platform compare-URL building.

Single boundary for every interaction with the ``git`` binary in the app.
Pure subprocess + stdlib; no streamlit, no openai. UI code (in
``streamlit_app.py``) reads/writes git state through this module so:

- All ``subprocess.run(['git', ...])`` calls live in one place and share
  consistent error handling (``GitError`` with stderr).
- The remote-compare-URL builder is unit-testable in isolation and the
  per-host quirks (GitHub vs GitLab vs Bitbucket) are co-located.
- Adding a new git operation later means editing exactly this file plus
  the UI helper that calls it.

All functions take an explicit ``working_dir: Path`` so the same registry
of helpers works across multiple workdirs without hidden global state.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


class GitError(RuntimeError):
    """Raised when a ``git`` subprocess returns a non-zero exit code.

    Carries the raw ``stderr`` so the UI can surface it verbatim — git's
    error messages are usually self-explanatory and re-wording them
    loses precision.
    """

    def __init__(self, returncode: int, stderr: str, command: list[str]) -> None:
        super().__init__(f"git {' '.join(command)} failed (exit {returncode}): {stderr.strip()}")
        self.returncode = returncode
        self.stderr = stderr.strip()
        self.command = command


@dataclass
class StatusEntry:
    """One file mentioned in ``git status --porcelain=v1``.

    The two-character XY code is preserved as ``staged_status`` /
    ``unstaged_status`` so callers can tell e.g. an unstaged-modify (``" M"``)
    from a staged-add (``"A "``) without re-running git.
    """

    path: str
    staged_status: str
    unstaged_status: str
    is_untracked: bool
    is_deleted: bool
    is_renamed: bool = False
    orig_path: str | None = None


@dataclass
class PullResult:
    ok: bool
    conflict: bool = False
    files: list[str] = field(default_factory=list)
    operation: str = ""
    stderr: str = ""


@dataclass
class PushResult:
    ok: bool
    stderr: str = ""
    set_upstream: bool = False


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int = 60,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a ``git ...`` subprocess, capturing stdout/stderr.

    Raises :class:`GitError` on non-zero exit when ``check`` is true.
    """
    cmd = ["git", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )
    if check and proc.returncode != 0:
        raise GitError(proc.returncode, proc.stderr, cmd)
    return proc


@lru_cache(maxsize=1)
def is_git_installed() -> bool:
    """Return True if ``git`` is on PATH and answers to ``--version``.

    Cached because the answer doesn't change during a single Streamlit
    process lifetime, and we check it on every UI rerender.
    """
    if shutil.which("git") is None:
        return False
    try:
        proc = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def is_git_repo(working_dir: Path) -> bool:
    """Return True if ``working_dir`` is inside a git working tree."""
    if not is_git_installed():
        return False
    try:
        proc = _run(
            ["rev-parse", "--is-inside-work-tree"],
            cwd=working_dir,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def repo_root(working_dir: Path) -> Path | None:
    """Return the top-level directory of the repo containing ``working_dir``.

    Returns ``None`` if not in a git repo.
    """
    if not is_git_repo(working_dir):
        return None
    try:
        proc = _run(["rev-parse", "--show-toplevel"], cwd=working_dir, timeout=5)
    except GitError:
        return None
    return Path(proc.stdout.strip())


def current_branch(working_dir: Path) -> str | None:
    """Return the current branch name, or ``None`` for detached HEAD."""
    proc = _run(
        ["symbolic-ref", "--short", "HEAD"],
        cwd=working_dir,
        check=False,
        timeout=5,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def list_branches(working_dir: Path) -> list[str]:
    """Return local branch names sorted alphabetically."""
    proc = _run(
        ["for-each-ref", "--format=%(refname:short)", "refs/heads/"],
        cwd=working_dir,
        timeout=10,
    )
    branches = [b for b in proc.stdout.splitlines() if b.strip()]
    return sorted(branches)


def list_remote_branches(working_dir: Path, remote: str = "origin") -> list[str]:
    """Return remote-tracking branch names like ``origin/feature``.

    Filters out the symbolic ``<remote>/HEAD`` ref so the chat-page
    branch dropdown doesn't surface it as a duplicate of the actual
    default branch.
    """
    proc = _run(
        ["for-each-ref", "--format=%(refname:short)", f"refs/remotes/{remote}/"],
        cwd=working_dir,
        check=False,
        timeout=10,
    )
    out: list[str] = []
    head_marker = f"{remote}/HEAD"
    for line in proc.stdout.splitlines():
        name = line.strip()
        if not name or name == head_marker:
            continue
        out.append(name)
    return sorted(out)


def checkout(working_dir: Path, branch: str) -> None:
    """Switch to ``branch`` (local or remote-tracking name).

    A remote-tracking name like ``origin/feature`` resolves automatically
    to a new local tracking branch named ``feature`` — that's plain git
    behaviour, and it's how the chat-page branch dropdown turns a
    remote-only entry into a local checkout in one click.
    """
    _run(["checkout", branch], cwd=working_dir, timeout=30)


def create_branch(working_dir: Path, name: str, *, checkout: bool = True) -> None:
    """Create a new branch off the current HEAD.

    With ``checkout=True`` (the default) we use ``git checkout -b`` so any
    uncommitted working-tree changes come along automatically — the
    common "make a new branch for what I'm currently working on" path.
    """
    if checkout:
        _run(["checkout", "-b", name], cwd=working_dir, timeout=15)
    else:
        _run(["branch", name], cwd=working_dir, timeout=15)


def default_branch(working_dir: Path) -> str:
    """Best-effort detection of the repo's default branch.

    Tries ``origin/HEAD`` first (most reliable when the repo was cloned).
    Falls back to common conventions (``main``, ``master``) when the
    symbolic ref is absent — e.g. brand-new repos where ``origin/HEAD``
    hasn't been set yet.
    """
    proc = _run(
        ["symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=working_dir,
        check=False,
        timeout=5,
    )
    if proc.returncode == 0:
        ref = proc.stdout.strip()
        if ref.startswith("refs/remotes/origin/"):
            return ref[len("refs/remotes/origin/") :]

    branches = set(list_branches(working_dir))
    for candidate in ("main", "master"):
        if candidate in branches:
            return candidate
    return "main"


def working_tree_dirty(working_dir: Path) -> bool:
    """Return True if there are any staged, unstaged, or untracked changes."""
    proc = _run(["status", "--porcelain"], cwd=working_dir, timeout=10)
    return bool(proc.stdout.strip())


def status_entries(working_dir: Path) -> list[StatusEntry]:
    """Parse ``git status -z --porcelain=v1`` into a list of :class:`StatusEntry`.

    Uses the NUL-separated form so paths with newlines or spaces don't
    break the parser. Renames span two NUL-separated records (the new
    path follows the original) — we collapse those into a single entry
    with both paths populated.
    """
    proc = _run(
        ["status", "-z", "--porcelain=v1"],
        cwd=working_dir,
        timeout=10,
    )
    raw = proc.stdout
    if not raw:
        return []

    parts = raw.split("\x00")
    out: list[StatusEntry] = []
    i = 0
    while i < len(parts):
        record = parts[i]
        i += 1
        if not record:
            continue
        if len(record) < 3:
            continue
        xy = record[:2]
        path = record[3:]
        staged, unstaged = xy[0], xy[1]
        is_untracked = xy == "??"
        is_renamed = staged in ("R", "C") or unstaged in ("R", "C")
        is_deleted = staged == "D" or unstaged == "D"
        orig_path: str | None = None
        # Renames in porcelain=v1 -z form emit the original path as the
        # immediately-following NUL-separated record. Consume it so we
        # don't accidentally treat it as a standalone entry.
        if is_renamed and i < len(parts):
            orig_path = parts[i]
            i += 1
        out.append(
            StatusEntry(
                path=path,
                staged_status=staged,
                unstaged_status=unstaged,
                is_untracked=is_untracked,
                is_deleted=is_deleted,
                is_renamed=is_renamed,
                orig_path=orig_path,
            )
        )
    return out


def diff_for_path(working_dir: Path, path: str, *, untracked: bool = False) -> str:
    """Return the unified diff for a single file vs HEAD.

    For untracked files we synthesize a diff against ``/dev/null`` via
    ``git diff --no-index``; this matches what GitHub shows for "added"
    files and gives the UI a uniform code-block render.
    """
    if untracked:
        proc = _run(
            ["diff", "--no-index", "--no-color", "--no-renames", "/dev/null", "--", path],
            cwd=working_dir,
            check=False,
            timeout=20,
        )
        # `git diff --no-index` returns exit 1 on differences, which is
        # the expected case here, so we only treat 2+ as an error.
        if proc.returncode >= 2:
            raise GitError(proc.returncode, proc.stderr, ["diff", "--no-index", path])
        return proc.stdout

    proc = _run(
        ["diff", "HEAD", "--no-color", "--no-renames", "--", path],
        cwd=working_dir,
        timeout=20,
    )
    return proc.stdout


def summary_diff_counts(
    working_dir: Path,
    paths: list[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Return ``{path: (additions, deletions)}`` for each path vs HEAD.

    Uses ``git diff --numstat HEAD`` which is much cheaper than asking
    for the full diff. Untracked files are not reported by ``git diff``;
    callers that need counts for those should fall back to counting the
    file's line count as additions.
    """
    args = ["diff", "--numstat", "HEAD"]
    if paths:
        args.append("--")
        args.extend(paths)
    proc = _run(args, cwd=working_dir, check=False, timeout=20)
    out: dict[str, tuple[int, int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        adds_raw, dels_raw, path = parts[0], parts[1], parts[2]
        # Binary files report '-' for both columns; treat as zero.
        adds = int(adds_raw) if adds_raw.isdigit() else 0
        dels = int(dels_raw) if dels_raw.isdigit() else 0
        out[path] = (adds, dels)
    return out


def untracked_line_count(working_dir: Path, path: str) -> int:
    """Count lines in an untracked file (treated as the +N for sidebar)."""
    target = (working_dir / path).resolve()
    try:
        with target.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# Cap for the combined diff blob we feed DeepSeek for commit-message /
# PR-body generation. Generous enough to cover most real changes
# without blowing past the model's context.
_COMBINED_DIFF_CAP = 200_000


def combined_diff_for_paths(working_dir: Path, paths: list[str]) -> str:
    """Build a single diff blob for ``paths`` (tracked + untracked).

    Tracked files come from ``git diff HEAD --`` so renames + binary
    handling match what GitHub shows. Untracked files (which
    ``git diff`` ignores) are appended as synthesized
    ``--- /dev/null / +++ b/<path>`` blocks so the model still sees
    new-file additions. The result is capped at
    :data:`_COMBINED_DIFF_CAP` characters so very large change sets
    don't overflow the model context — callers (today: the chat
    page's commit-message helper) can rely on this cap.
    """
    if not paths:
        return ""
    chunks: list[str] = []

    # Pre-compute which paths are untracked so we can split tracked vs
    # untracked in one walk without re-running ``git status``.
    untracked_set: set[str] = set()
    try:
        for entry in status_entries(working_dir):
            if entry.is_untracked:
                untracked_set.add(entry.path)
    except GitError:
        pass

    tracked = [p for p in paths if p not in untracked_set]
    if tracked:
        try:
            proc = _run(
                ["diff", "HEAD", "--no-color", "--no-renames", "--", *tracked],
                cwd=working_dir,
                check=False,
                timeout=30,
            )
            if proc.stdout:
                chunks.append(proc.stdout)
        except (GitError, OSError):
            pass

    for p in paths:
        if p not in untracked_set:
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
    if len(blob) > _COMBINED_DIFF_CAP:
        blob = (
            blob[:_COMBINED_DIFF_CAP]
            + f"\n\n[truncated, {len(blob) - _COMBINED_DIFF_CAP} more chars]\n"
        )
    return blob


def stage(working_dir: Path, paths: list[str]) -> None:
    """Stage the given paths.

    Uses ``git add -A -- <paths>`` so the same call handles new,
    modified, and deleted files uniformly. Empty input is a no-op so
    callers don't have to guard against it.
    """
    if not paths:
        return
    _run(["add", "-A", "--", *paths], cwd=working_dir, timeout=30)


def unstage_all(working_dir: Path) -> None:
    """Reset the staging area so it mirrors HEAD.

    Run before :func:`stage` in the push flow so the user's checkbox
    selection is the *only* thing that ends up in the commit, even if
    something was already staged before the dialog opened.
    """
    _run(["reset", "HEAD", "--", "."], cwd=working_dir, check=False, timeout=15)


def commit(working_dir: Path, message: str) -> None:
    """Create a commit with the given message.

    GPG signing is explicitly disabled with ``-c commit.gpgsign=false``
    so the desktop app doesn't deadlock on a passphrase prompt for
    users with a signed-commit default in their global config. Signed
    commits are listed as out-of-scope in AGENTS.md.
    """
    _run(
        ["-c", "commit.gpgsign=false", "commit", "-m", message],
        cwd=working_dir,
        timeout=30,
    )


def fetch(working_dir: Path, remote: str = "origin") -> None:
    """Run ``git fetch <remote>``; raises :class:`GitError` on failure.

    Tolerates "no remote" by raising — the UI catches and surfaces the
    error in the dialog status area.
    """
    _run(["fetch", remote], cwd=working_dir, timeout=120)


def has_upstream(working_dir: Path, branch: str | None = None) -> bool:
    """Return True if ``branch`` (or HEAD) has an upstream configured."""
    args = ["rev-parse", "--abbrev-ref", "--symbolic-full-name"]
    args.append(f"{branch}@{{u}}" if branch else "@{u}")
    proc = _run(args, cwd=working_dir, check=False, timeout=5)
    return proc.returncode == 0


def is_behind_upstream(working_dir: Path) -> bool:
    """Return True if HEAD is behind its upstream (i.e., a pull is needed)."""
    proc = _run(
        ["rev-list", "--count", "HEAD..@{u}"],
        cwd=working_dir,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        return False
    try:
        return int(proc.stdout.strip()) > 0
    except ValueError:
        return False


def conflicted_files(working_dir: Path) -> list[str]:
    """Return paths with unmerged conflicts (``--diff-filter=U``)."""
    proc = _run(
        ["diff", "--name-only", "--diff-filter=U"],
        cwd=working_dir,
        check=False,
        timeout=10,
    )
    return [p for p in proc.stdout.splitlines() if p.strip()]


def is_in_merge_or_rebase(working_dir: Path) -> tuple[bool, str | None]:
    """Detect whether the repo is mid-merge or mid-rebase.

    Returns ``(in_progress, operation)`` where ``operation`` is
    ``"merge"``, ``"rebase"``, or ``None``. Detection inspects the
    ``.git`` directory directly so we don't fork a subprocess just to
    answer a yes/no question on the hot path.
    """
    git_dir = working_dir / ".git"
    if not git_dir.exists():
        # Could be a worktree; ask git for the actual git-dir.
        proc = _run(["rev-parse", "--git-dir"], cwd=working_dir, check=False, timeout=5)
        if proc.returncode != 0:
            return False, None
        git_dir = (working_dir / proc.stdout.strip()).resolve()
    if (git_dir / "MERGE_HEAD").exists():
        return True, "merge"
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        return True, "rebase"
    return False, None


def pull_rebase(working_dir: Path) -> PullResult:
    """Run ``git pull --rebase`` and report whether conflicts occurred.

    A successful pull returns ``ok=True``. Any non-zero exit is
    inspected: if conflicted files are present we report them as a
    structured conflict so the UI can render the resolution affordance,
    otherwise we re-raise as :class:`GitError`.
    """
    proc = _run(["pull", "--rebase"], cwd=working_dir, check=False, timeout=180)
    if proc.returncode == 0:
        return PullResult(ok=True)

    in_progress, op = is_in_merge_or_rebase(working_dir)
    files = conflicted_files(working_dir)
    if in_progress and files:
        return PullResult(
            ok=False,
            conflict=True,
            files=files,
            operation=op or "rebase",
            stderr=proc.stderr,
        )
    raise GitError(proc.returncode, proc.stderr, ["pull", "--rebase"])


def push(
    working_dir: Path,
    *,
    branch: str | None = None,
    set_upstream: bool | None = None,
) -> PushResult:
    """Push the current branch to ``origin``.

    When ``set_upstream`` is ``None`` we auto-detect: if the branch has
    no upstream configured we add ``-u origin <branch>`` so the very
    first push of a freshly-created branch works without the user
    having to remember the flag.
    """
    branch = branch or current_branch(working_dir)
    if branch is None:
        return PushResult(ok=False, stderr="Cannot push from a detached HEAD.")

    if set_upstream is None:
        set_upstream = not has_upstream(working_dir, branch)

    args = ["push"]
    if set_upstream:
        args += ["-u", "origin", branch]
    proc = _run(args, cwd=working_dir, check=False, timeout=180)
    if proc.returncode != 0:
        return PushResult(ok=False, stderr=proc.stderr, set_upstream=set_upstream)
    # Push commonly emits useful info on stderr (the "Create a pull
    # request..." line that some hosts return). Surface it so the UI
    # can show it as a fallback when ``remote_compare_url`` returns None.
    return PushResult(ok=True, stderr=proc.stderr, set_upstream=set_upstream)


def remote_url(working_dir: Path, remote: str = "origin") -> str | None:
    """Return the configured URL for ``remote`` or ``None`` if absent."""
    proc = _run(["remote", "get-url", remote], cwd=working_dir, check=False, timeout=5)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


_REMOTE_PATTERNS = [
    # SSH: git@github.com:owner/repo.git
    re.compile(r"^git@(?P<host>[^:]+):(?P<path>.+?)(?:\.git)?/?$"),
    # https://github.com/owner/repo(.git)
    re.compile(r"^https?://(?:[^@]+@)?(?P<host>[^/]+)/(?P<path>.+?)(?:\.git)?/?$"),
    # ssh://git@github.com/owner/repo(.git)
    re.compile(r"^ssh://(?:[^@]+@)?(?P<host>[^/]+)/(?P<path>.+?)(?:\.git)?/?$"),
]


def parse_remote(url: str) -> tuple[str, str] | None:
    """Parse a remote URL into ``(host, owner_repo_path)``.

    Returns ``None`` if the URL doesn't match any of the supported
    patterns (SSH shorthand, https, ssh://). The path keeps any nested
    GitLab groups intact (e.g. ``mygroup/subgroup/project``).
    """
    for pat in _REMOTE_PATTERNS:
        m = pat.match(url.strip())
        if m:
            return m.group("host"), m.group("path")
    return None


def remote_compare_url(
    working_dir: Path,
    branch: str,
    base: str,
    *,
    title: str | None = None,
    body: str | None = None,
) -> str | None:
    """Build a pre-filled "open a PR" URL for the configured origin.

    Supports github.com, gitlab.com, and bitbucket.org explicitly; falls
    back to ``None`` for other hosts (including self-hosted instances)
    so the dialog can surface the raw "Create a pull request..." line
    that the host returns on push instead.

    Title and body are URL-encoded and pre-filled where each platform's
    "new PR" page accepts them via query string.
    """
    url = remote_url(working_dir)
    if not url:
        return None
    parsed = parse_remote(url)
    if parsed is None:
        return None
    host, path = parsed

    title_qs = urllib.parse.quote(title) if title else ""
    body_qs = urllib.parse.quote(body) if body else ""

    if host == "github.com" or host.endswith(".github.com"):
        base_url = f"https://github.com/{path}/compare/{urllib.parse.quote(base)}...{urllib.parse.quote(branch)}"
        params = ["quick_pull=1", "expand=1"]
        if title:
            params.append(f"title={title_qs}")
        if body:
            params.append(f"body={body_qs}")
        return base_url + "?" + "&".join(params)

    if host == "gitlab.com" or host.endswith(".gitlab.com"):
        params = [
            f"merge_request%5Bsource_branch%5D={urllib.parse.quote(branch)}",
            f"merge_request%5Btarget_branch%5D={urllib.parse.quote(base)}",
        ]
        if title:
            params.append(f"merge_request%5Btitle%5D={title_qs}")
        if body:
            params.append(f"merge_request%5Bdescription%5D={body_qs}")
        return f"https://gitlab.com/{path}/-/merge_requests/new?" + "&".join(params)

    if host == "bitbucket.org":
        # Bitbucket's "create pull request" URL accepts source/dest but
        # not title/body via query string.
        return (
            f"https://bitbucket.org/{path}/pull-requests/new?"
            f"source={urllib.parse.quote(branch)}&dest={urllib.parse.quote(base)}&t=1"
        )

    return None


def extract_pr_link_from_stderr(stderr: str) -> str | None:
    """Pull a "Create a pull request" URL out of ``git push`` stderr.

    GitHub, GitLab, and Bitbucket all print a hint URL on first push
    of a new branch. When :func:`remote_compare_url` doesn't recognize
    the host we surface this raw URL instead so the user still has a
    one-click path to the PR form.
    """
    for line in stderr.splitlines():
        line = line.strip()
        # Strip the "remote: " prefix that some servers add.
        if line.startswith("remote:"):
            line = line[len("remote:") :].strip()
        m = re.search(r"https?://\S+", line)
        if not m:
            continue
        url = m.group(0).rstrip(".,);")
        if any(needle in url for needle in ("compare", "merge_request", "pull-request", "/pr/")):
            return url
    return None


def scan(working_dir: Path) -> dict[str, Any]:
    """Collect everything the UI needs about the repo in one shot.

    Returns a dict with the same keys regardless of whether git is
    installed or the directory is a repo, so callers can branch on
    ``in_repo`` / ``installed`` without ``KeyError`` worries.
    """
    out: dict[str, Any] = {
        "installed": is_git_installed(),
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
    if not out["installed"]:
        return out
    if not is_git_repo(working_dir):
        return out

    out["in_repo"] = True
    try:
        out["current_branch"] = current_branch(working_dir)
        out["branches"] = list_branches(working_dir)
        out["remote_branches"] = list_remote_branches(working_dir)
        out["default_branch"] = default_branch(working_dir)
        out["status"] = status_entries(working_dir)
        out["dirty"] = bool(out["status"])
        in_progress, op = is_in_merge_or_rebase(working_dir)
        out["in_merge_or_rebase"] = in_progress
        out["operation"] = op
        if in_progress:
            out["conflicted_files"] = conflicted_files(working_dir)
        out["remote_url"] = remote_url(working_dir)
    except GitError:
        # A partially-initialized repo (e.g. brand new ``git init`` with
        # no commits) can fail some of these calls. Whatever fields we
        # already populated stay, the rest keep their safe defaults.
        pass
    return out
