"""Shared artifact-folder helpers for media generation + uploaded attachments.

Single owner of the ``<working_dir>/.wb_artifacts/<chat_id>/`` folder
structure used by both Phase 5 (media-generation tools writing into
the chat's outputs) and Phase 6 (uploaded attachments writing into
``<chat_id>/inbox/``). Centralising these helpers means folder
creation, ``.gitignore`` management, and collision-safe naming all
live in one place — neither ``tools.py`` nor ``attachments.py``
duplicates the logic.

Hard rules (mirrored in ``AGENTS.md``):

- All media bytes (image / audio / video / embeddings cached as bytes)
  AND user-uploaded attachment bytes land inside
  ``<working_dir>/.wb_artifacts/<chat_id>/`` via path-containment
  checks. Generated outputs go through ``tools.py``'s ``generate_*``
  tools; user uploads go through ``attachments.save_uploads``.
- The ``.wb_artifacts/`` line is appended to the workdir's
  ``.gitignore`` on first creation so multi-MB media files don't get
  accidentally committed. The append is idempotent (skipped if
  ``.wb_artifacts/`` is already present in the file).
- This module has **no Streamlit imports** so background threads can
  call into it safely.
"""
from __future__ import annotations

import re
from pathlib import Path

ARTIFACTS_DIR_NAME = ".wb_artifacts"


def _resolve_inside(working_dir: Path, sub_path: Path) -> Path:
    """Resolve ``sub_path`` against ``working_dir`` and reject escapes.

    Mirrors the safety check in ``tools.py`` for plain file ops; we
    duplicate it here so the artifact helpers don't have to import
    from tools.py (which is the agent-tools registry). Returns the
    absolute resolved path; raises :class:`ValueError` if the result
    sits outside ``working_dir``.
    """
    working_resolved = working_dir.expanduser().resolve()
    target = (working_resolved / sub_path).resolve()
    try:
        target.relative_to(working_resolved)
    except ValueError as e:
        raise ValueError(
            f"Refusing path that escapes working directory: {sub_path}"
        ) from e
    return target


def artifacts_root(working_dir: Path) -> Path:
    """Return the absolute path to ``<working_dir>/.wb_artifacts``."""
    return _resolve_inside(working_dir, Path(ARTIFACTS_DIR_NAME))


def chat_artifacts_dir(working_dir: Path, chat_id: str) -> Path:
    """Return the absolute path to ``<working_dir>/.wb_artifacts/<chat_id>``.

    Validates ``chat_id`` is a single safe path component (alnum +
    hyphen + underscore) so a malformed chat id can't escape the
    sandbox.
    """
    if not chat_id or not re.fullmatch(r"[A-Za-z0-9_-]+", chat_id):
        raise ValueError(f"Invalid chat id for artifact path: {chat_id!r}")
    return _resolve_inside(working_dir, Path(ARTIFACTS_DIR_NAME) / chat_id)


def ensure_artifacts_dir(working_dir: Path, chat_id: str) -> Path:
    """Create ``<working_dir>/.wb_artifacts/<chat_id>/`` if missing.

    Idempotent. Returns the absolute path. Side effect: appends the
    ``.wb_artifacts/`` line to the workdir's ``.gitignore`` on first
    creation (see :func:`ensure_gitignored`).
    """
    target = chat_artifacts_dir(working_dir, chat_id)
    fresh = not target.exists()
    target.mkdir(parents=True, exist_ok=True)
    if fresh:
        # First-creation only — the gitignore hook fires here so we
        # don't churn the file on every artifact write.
        ensure_gitignored(working_dir)
    return target


def ensure_inbox_dir(working_dir: Path, chat_id: str) -> Path:
    """Create ``<working_dir>/.wb_artifacts/<chat_id>/inbox/`` if missing.

    The inbox subfolder is reserved for user-uploaded attachments
    (Phase 6); generated outputs land in the chat folder directly.
    Separating the two makes "what did the agent produce vs what did
    the user attach" navigable on disk.
    """
    chat_dir = ensure_artifacts_dir(working_dir, chat_id)
    inbox = chat_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox


def ensure_gitignored(working_dir: Path) -> bool:
    """Idempotently append ``.wb_artifacts/`` to ``<working_dir>/.gitignore``.

    Returns ``True`` when the line was appended (first creation),
    ``False`` when the file already had the entry. Creates the
    ``.gitignore`` file if it doesn't exist. Best-effort: any I/O
    error is swallowed because failure to update gitignore should
    never break a media-generation call.
    """
    gitignore = working_dir / ".gitignore"
    line = f"{ARTIFACTS_DIR_NAME}/"
    try:
        if gitignore.exists():
            current = gitignore.read_text(encoding="utf-8")
            # Match either ``.wb_artifacts/`` (with trailing slash) or
            # ``.wb_artifacts`` (no trailing slash); both work as
            # gitignore patterns for our case.
            for ln in current.splitlines():
                stripped = ln.strip()
                if stripped in (line, ARTIFACTS_DIR_NAME):
                    return False
            sep = "" if current.endswith("\n") else "\n"
            gitignore.write_text(current + sep + line + "\n", encoding="utf-8")
            return True
        gitignore.write_text(line + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def collision_safe_path(target: Path) -> Path:
    """Return ``target`` if available, else the next ``name (N).ext`` variant.

    Used by attachment uploads where two users might attach
    ``screenshot.png`` to the same chat: the second upload becomes
    ``screenshot (2).png`` instead of clobbering the first. Caps at
    1000 attempts so a runaway loop can't lock up the script thread.
    """
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    for n in range(2, 1001):
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a collision-free path under {parent}")


def artifact_path(
    working_dir: Path,
    chat_id: str,
    *,
    kind: str,
    ext: str,
    idx: int | None = None,
) -> Path:
    """Build a deterministic output path for a generated artifact.

    Format: ``<chat_dir>/<kind>_<unix_ms>[_<idx>].<ext>``. The unix
    timestamp gives most files a unique name without needing a UUID;
    when a single tool call writes multiple files (e.g. an image-gen
    tool with ``n=4``), ``idx`` disambiguates within the millisecond.
    """
    import time
    chat_dir = ensure_artifacts_dir(working_dir, chat_id)
    ts = int(time.time() * 1000)
    suffix = ext.lstrip(".")
    name = f"{kind}_{ts}.{suffix}" if idx is None else f"{kind}_{ts}_{idx}.{suffix}"
    return chat_dir / name


__all__ = [
    "ARTIFACTS_DIR_NAME",
    "artifact_path",
    "artifacts_root",
    "chat_artifacts_dir",
    "collision_safe_path",
    "ensure_artifacts_dir",
    "ensure_gitignored",
    "ensure_inbox_dir",
]
