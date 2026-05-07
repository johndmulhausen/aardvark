"""User-attachment intake pipeline (images, PDFs, text/code files).

Phase 6 counterpart to Phase 5's media-generation tools: where the
agent writes generated assets into ``<chat_id>/`` directly, user
uploads land in ``<chat_id>/inbox/`` so the two directions of media
flow stay visually separable on disk. Streamlit's
``st.chat_input(accept_file="multiple")`` returns a dict-like with
``text`` + ``files: list[UploadedFile]`` — the chat page passes those
files through :func:`save_uploads`, which writes them into the
artifacts folder via the shared :mod:`_artifact_paths` helpers.

Hard rules (mirrored in ``AGENTS.md``):

- All upload bytes land inside
  ``<working_dir>/.wb_artifacts/<chat_id>/inbox/`` via
  :mod:`_artifact_paths`. No other module writes attachment bytes.
- Multimodal message construction goes through
  :func:`build_user_message`. ``chat_streams.py``'s per-provider
  translators are the last-line backstop, not a content-array
  generator.
- PDF text extraction goes through :func:`extract_text` (which
  imports ``pypdf``). No other module imports ``pypdf`` so swapping
  the extractor (e.g. for native Anthropic PDF support) lives in
  one place.

This module has **no Streamlit imports** — runs from background
threads safely, mirroring the chats / model_catalog / attachments
boundary pattern.
"""
from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import _artifact_paths


# ---------------------------------------------------------------------------
# File-classification tables
# ---------------------------------------------------------------------------
# Plain-text + code file extensions we route to ``"text"`` and inline
# as a fenced code block in the multimodal content array. The set is
# deliberately narrow — extending it is cheap (just add the
# extension here), but accepting unknown binary blobs as text is
# expensive (the model sees garbage).
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".rst", ".log",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".py", ".pyi", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sh", ".bash", ".zsh", ".fish",
    ".rs", ".go", ".java", ".kt", ".scala",
    ".cpp", ".cc", ".c", ".h", ".hpp", ".hxx",
    ".rb", ".php", ".pl", ".lua",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".makefile",
    ".csv", ".tsv",
    ".xml", ".svg",
    ".vue", ".svelte",
})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
})

# Programming-language hint per extension for the inline ``st.code``
# fence the user-turn renderer renders. Falls back to ``"text"``.
LANGUAGE_FOR_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".ipynb": "json",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mjs": "javascript", ".cjs": "javascript",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "scss", ".less": "less",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "bash",
    ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".cpp": "cpp", ".cc": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp", ".hxx": "cpp",
    ".rb": "ruby", ".php": "php", ".pl": "perl", ".lua": "lua",
    ".sql": "sql", ".graphql": "graphql", ".proto": "protobuf",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".env": "bash",
    ".md": "markdown", ".rst": "rst",
    ".dockerfile": "dockerfile", ".makefile": "makefile",
    ".xml": "xml", ".svg": "xml",
    ".csv": "text", ".tsv": "text",
    ".vue": "vue", ".svelte": "svelte",
}

# Hard cap on extracted-text payload per attachment so an overlong
# PDF or a multi-MB log doesn't blow the model's context budget.
EXTRACTED_TEXT_CAP_BYTES = 200_000

# Hard cap on PIL resize for images — matches OpenAI's recommended
# max image dimension for vision models. Larger inputs are downsized
# proportionally before being base64-encoded into the wire payload.
IMAGE_MAX_DIM_PX = 2048


@dataclass(frozen=True)
class Attachment:
    """One uploaded file ready to splice into the next user message.

    ``path`` is workdir-relative so the chat persists a portable
    reference (the user can move the project folder without
    breaking replay). ``extracted_text`` is populated only when
    :func:`extract_text` ran for the file (e.g. a PDF on a non-PDF-
    capable model, or a plain-text file).
    """

    path: str
    kind: str  # "image" / "pdf" / "text" / "audio" / "video" / "binary"
    mime_type: str
    filename: str
    size_bytes: int
    extracted_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence inside ``Chat.ui_turns``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Attachment":
        return cls(
            path=str(raw.get("path") or ""),
            kind=str(raw.get("kind") or "binary"),
            mime_type=str(raw.get("mime_type") or "application/octet-stream"),
            filename=str(raw.get("filename") or ""),
            size_bytes=int(raw.get("size_bytes") or 0),
            extracted_text=(
                str(raw["extracted_text"])
                if isinstance(raw.get("extracted_text"), str)
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Classification + IO
# ---------------------------------------------------------------------------
def classify(filename: str, mime_type: str = "") -> str:
    """Map ``(filename, mime_type)`` to an :class:`Attachment.kind` string.

    Mime type takes priority when it's a well-known image / pdf /
    text type; the file extension is the fallback. Anything we don't
    recognize lands as ``"binary"`` so the upload chip can warn the
    user that the file won't be readable by the model.
    """
    mt = (mime_type or "").lower().strip()
    if mt.startswith("image/"):
        return "image"
    if mt == "application/pdf" or filename.lower().endswith(".pdf"):
        return "pdf"
    if mt.startswith("text/"):
        return "text"
    if mt.startswith("audio/"):
        return "audio"
    if mt.startswith("video/"):
        return "video"

    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return "binary"


def save_uploads(
    uploads: Iterable[Any],
    working_dir: Path,
    chat_id: str,
) -> list[Attachment]:
    """Persist Streamlit ``UploadedFile`` instances into the chat inbox.

    Returns one :class:`Attachment` per saved upload, with collision-
    safe naming so re-uploading the same filename doesn't clobber
    a previous attachment. The ``inbox/`` subfolder under
    ``<chat_id>/`` is created on first call (idempotent).

    Streamlit's ``UploadedFile`` exposes ``.name``, ``.type``,
    ``.size``, and ``.getvalue()``; we use ``getvalue()`` because
    ``UploadedFile`` is a ``BytesIO`` and we want to materialize the
    full bytes on disk in one shot.
    """
    inbox = _artifact_paths.ensure_inbox_dir(working_dir, chat_id)
    out: list[Attachment] = []
    for upload in uploads:
        name = getattr(upload, "name", None) or "upload.bin"
        # Sanitize the filename to stop a malicious upload from
        # writing outside the inbox via path separators.
        safe_name = Path(name).name or "upload.bin"
        target = _artifact_paths.collision_safe_path(inbox / safe_name)
        try:
            data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
        except Exception:  # noqa: BLE001 — surfaced via empty bytes
            data = b""
        target.write_bytes(data)
        mime_type = getattr(upload, "type", None) or (
            mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )
        kind = classify(target.name, mime_type)
        # Resolve both paths through the same symlink-following pass
        # before computing the relative form. macOS's ``/var`` is a
        # symlink to ``/private/var`` so a naïve ``.relative_to``
        # fails when the working_dir hasn't been resolved.
        rel = str(target.relative_to(working_dir.resolve()))
        out.append(Attachment(
            path=rel,
            kind=kind,
            mime_type=mime_type,
            filename=safe_name,
            size_bytes=len(data),
        ))
    return out


# ---------------------------------------------------------------------------
# Text extraction (PDFs + plain text)
# ---------------------------------------------------------------------------
def extract_text(att: Attachment, working_dir: Path) -> str:
    """Extract a plain-text rendering of ``att`` for non-multimodal models.

    Only callable for ``kind in ("text", "pdf")``. PDFs are run
    through ``pypdf``'s page-extraction; plain-text files are read
    as UTF-8 with replacement on decode error. Output is capped at
    :data:`EXTRACTED_TEXT_CAP_BYTES` to keep prompts manageable —
    longer payloads are truncated with a clear note appended.
    """
    if att.kind not in ("text", "pdf"):
        return ""

    abs_path = (working_dir / att.path).resolve()
    if not abs_path.exists() or not abs_path.is_file():
        return ""

    try:
        if att.kind == "pdf":
            # ``pypdf`` is a pure-Python dep and the only place this
            # module imports it from; the module-level docstring + the
            # AGENTS.md hard rule both single this out as the single
            # call site.
            from pypdf import PdfReader

            reader = PdfReader(str(abs_path))
            chunks: list[str] = []
            for page in reader.pages:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001 — page-level failures are non-fatal
                    continue
            text = "\n\n".join(chunks)
        else:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    if len(text.encode("utf-8")) > EXTRACTED_TEXT_CAP_BYTES:
        # Cap on byte size, not character count, so the cap survives
        # multi-byte UTF-8 characters predictably.
        encoded = text.encode("utf-8")[:EXTRACTED_TEXT_CAP_BYTES]
        text = encoded.decode("utf-8", errors="ignore")
        text += "\n\n…[truncated — file exceeded extracted-text cap]"
    return text


def language_for(att: Attachment) -> str:
    """Return a Streamlit ``st.code`` language hint for a text attachment."""
    if att.kind != "text":
        return "text"
    ext = Path(att.filename).suffix.lower()
    return LANGUAGE_FOR_EXT.get(ext, "text")


# ---------------------------------------------------------------------------
# Image preprocessing (resize for wire payload; keep full-res on disk)
# ---------------------------------------------------------------------------
def preprocess_image(
    att: Attachment,
    working_dir: Path,
    *,
    max_dim: int = IMAGE_MAX_DIM_PX,
) -> bytes:
    """Resize ``att``'s file to ``<= max_dim`` pixels on the longest side.

    Returns PNG-encoded bytes suitable for inlining as a base64 data
    URL into a multimodal content part. The original file on disk is
    preserved at full resolution for replay + download; only the
    wire payload is downsized. ``Pillow`` is a transitive Streamlit
    dep so this doesn't add new install impact.

    Returns ``b""`` on any failure — the caller should fall through
    to the original file bytes (sub-optimal but model-acceptable).
    """
    abs_path = (working_dir / att.path).resolve()
    if not abs_path.exists():
        return b""

    try:
        from PIL import Image  # type: ignore
    except Exception:  # noqa: BLE001 — Pillow is transitive; defensive
        try:
            return abs_path.read_bytes()
        except OSError:
            return b""

    try:
        with Image.open(abs_path) as im:
            im.load()  # eager-load so we can close the file before resize
            # Convert palette / single-channel modes to RGB so the PNG
            # encoder doesn't choke. Preserve transparency for RGBA.
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if im.mode == "P" else "RGB")
            longest = max(im.size)
            if longest > max_dim:
                ratio = max_dim / float(longest)
                new_size = (int(im.size[0] * ratio), int(im.size[1] * ratio))
                im = im.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:  # noqa: BLE001 — fall back to raw bytes on any error
        try:
            return abs_path.read_bytes()
        except OSError:
            return b""


# ---------------------------------------------------------------------------
# Multimodal message construction
# ---------------------------------------------------------------------------
def build_user_message(
    prompt: str,
    attachments: list[Attachment],
    *,
    working_dir: Path,
    supports_vision: bool,
    supports_pdf_input: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build ``(openai_message, ui_turn_attachments)`` for a user turn.

    Returns:

    1. The OpenAI-shape user message to splice into ``chat.messages``.
       For models without attachments (or only text/PDF
       extraction), the ``content`` is a plain string so the
       majority of providers' chat-completion endpoints don't have
       to deal with multimodal arrays. For models with images, the
       ``content`` becomes a multimodal array of ``{"type": "text",
       "text": ...}`` and ``{"type": "image_url", "image_url": {...}}``
       parts.
    2. A list of dicts representing each attachment for ``Chat.ui_turns``
       persistence. Carries the ``Attachment`` data + a flag noting
       which path was taken (native PDF vs text-extracted) so the
       user-turn renderer can show the right chip ("native PDF" vs
       "text-extracted").

    Capability-aware:

    - When ``supports_vision`` is False, image attachments are
      stripped from the wire payload (the on-disk file stays put;
      replay still shows the thumbnail). A note is appended to the
      prompt text so the model knows the user attached images that
      weren't sent.
    - When ``supports_pdf_input`` is False, PDFs are run through
      :func:`extract_text` and the extracted text is folded into the
      prompt as a fenced code block. Pure-text attachments always
      take this path regardless of ``supports_pdf_input``.
    """
    image_parts: list[dict[str, Any]] = []
    document_parts: list[dict[str, Any]] = []
    text_blobs: list[str] = []
    skipped_images: list[str] = []
    persist_records: list[dict[str, Any]] = []

    for att in attachments:
        record = att.to_dict()
        if att.kind == "image":
            if not supports_vision:
                skipped_images.append(att.filename)
                record["delivery"] = "skipped_no_vision"
                persist_records.append(record)
                continue
            # Resize before base64-encoding so the wire payload is
            # bounded. Full-res file stays on disk for the user to
            # download.
            data = preprocess_image(att, working_dir)
            if not data:
                record["delivery"] = "skipped_unreadable"
                persist_records.append(record)
                continue
            b64 = base64.b64encode(data).decode("ascii")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
            record["delivery"] = "image_inline"
            persist_records.append(record)
            continue
        if att.kind == "pdf":
            abs_path = (working_dir / att.path).resolve()
            if not supports_pdf_input or not abs_path.exists():
                # Fall back to text extraction; text gets folded into
                # the prompt below.
                text = extract_text(att, working_dir)
                if text:
                    text_blobs.append(
                        f"--- attached PDF: {att.filename} ---\n{text}\n--- end ---"
                    )
                record["delivery"] = "text_extracted"
                record["extracted_text"] = text or None
                persist_records.append(record)
                continue
            # Native PDF support — pass the file bytes through as a
            # provider-shape document part. ``chat_streams`` reshapes
            # this for Anthropic / Google natively.
            try:
                data = abs_path.read_bytes()
            except OSError:
                data = b""
            if data:
                b64 = base64.b64encode(data).decode("ascii")
                document_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:application/pdf;base64,{b64}"},
                })
                record["delivery"] = "pdf_native"
            else:
                record["delivery"] = "skipped_unreadable"
            persist_records.append(record)
            continue
        if att.kind == "text":
            text = extract_text(att, working_dir)
            lang = language_for(att)
            if text:
                text_blobs.append(
                    f"--- attached file: {att.filename} ({lang}) ---\n```{lang}\n{text}\n```\n--- end ---"
                )
            record["delivery"] = "text_inline"
            record["extracted_text"] = text or None
            persist_records.append(record)
            continue
        # Audio / video / binary — out of scope for v1; record but skip.
        record["delivery"] = "skipped_unsupported"
        persist_records.append(record)

    # Compose the final prompt text. Text blobs come first (so the
    # model reads the file content before the question that may
    # reference it), then a skipped-images note (if any), then the
    # user's prompt verbatim.
    composed_parts: list[str] = []
    if text_blobs:
        composed_parts.append("\n\n".join(text_blobs))
    if skipped_images:
        composed_parts.append(
            "(The user attached the following images, but the chosen model "
            "doesn't support vision input: "
            + ", ".join(skipped_images)
            + ".)"
        )
    if prompt:
        composed_parts.append(prompt)
    composed_text = "\n\n".join(composed_parts)

    if image_parts or document_parts:
        # Multimodal content array: text first, then images / docs.
        content_array: list[dict[str, Any]] = []
        if composed_text:
            content_array.append({"type": "text", "text": composed_text})
        content_array.extend(image_parts)
        content_array.extend(document_parts)
        message: dict[str, Any] = {"role": "user", "content": content_array}
    else:
        # Plain-text: keep the OpenAI shape simple for downstream
        # providers that don't speak the multimodal array.
        message = {"role": "user", "content": composed_text or prompt}

    return message, persist_records


__all__ = [
    "Attachment",
    "EXTRACTED_TEXT_CAP_BYTES",
    "IMAGE_EXTENSIONS",
    "IMAGE_MAX_DIM_PX",
    "LANGUAGE_FOR_EXT",
    "TEXT_EXTENSIONS",
    "build_user_message",
    "classify",
    "extract_text",
    "language_for",
    "preprocess_image",
    "save_uploads",
]
