"""Chat data model, on-disk persistence, and per-chat threading harness.

A "chat" is one resumable conversation with the agent. The app keeps a
dict of these in ``st.session_state.chats`` so the user can have many
conversations running simultaneously, switch between them, and resume
work after a refresh / app restart.

Single source of truth for:

- The :class:`Chat` dataclass (the on-disk + in-memory shape).
- Reading and writing ``~/.wb_coding_agent/chats/<chat_id>.json``
  (atomic, idempotent, mode 0600 inherited from umask).
- Spawning the daemon thread that drives one user turn through
  :func:`agent.run_agent_turn` so the Streamlit script thread is
  never blocked while the model + tools run.
- Auto-titling a chat after its first turn via the DeepSeek model
  served by W&B Inference.

Hard rules (mirrored in ``AGENTS.md``):

- This module **does not import streamlit**. The Streamlit UI passes
  the ``client`` (and any other dependencies) in explicitly so this
  module stays unit-testable and so background threads — which cannot
  call ``st.*`` — only ever touch :class:`Chat` objects.
- Every read or write of ``chat.messages`` / ``chat.ui_turns`` /
  ``chat.status`` / ``chat.partial_text`` happens under
  ``chat._lock``. Background threads never resize ``ss.chats``; they
  only mutate the :class:`Chat` they were given.
- Each chat has at most one running thread. :func:`start_turn` raises
  :class:`RuntimeError` when called on a chat that's still running.
- A chat persisted with ``status == "running"`` at app startup is
  rewritten to ``"error"`` on load — the thread that owned it died
  with the previous process.
- A user-initiated cancel goes through :func:`request_cancel`, which
  sets a per-chat :class:`threading.Event`. The runner thread checks
  the event between agent events (and before every ``save_chat``)
  and routes through :func:`_finalize_cancelled_turn` to flip status
  back to ``"idle"``, preserve any in-flight streamed text as a
  proper ``assistant_text`` event for replay, append a
  ``{"type": "cancelled"}`` marker the renderer surfaces as a
  "Stopped by user" caption, and ensure ``chat.messages`` stays
  well-formed (no two consecutive user messages on the next turn).
  Two call sites today: the chat-page **Stop** button (the active
  chat stays around, just back at idle) and :func:`delete_chat` with
  ``force=True`` (the chat is deleted; the finalize call is a no-op
  for the file because :attr:`Chat._deleted` is set first).

Per-turn usage capture (token counts + cost) still flows through
:mod:`usage`; we mirror the bookkeeping the old synchronous chat page
did so the Usage dashboard keeps working.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import agent
import usage as usage_log

CHATS_DIR = Path.home() / ".wb_coding_agent" / "chats"

# Tiny pointer file recording the chat the user was most recently
# *looking at* (vs. ``updated_at`` which captures the most-recently-
# *modified* chat). Without this, reloading the app would land the user
# on whatever chat was last persisted — usually a freshly-clicked
# ``+ New chat`` row that's still empty, hiding the chat with the
# actual conversation. Keeping it in a separate file (rather than
# folding into a chat record) avoids contention with the per-chat
# atomic writes during a running turn.
ACTIVE_CHAT_FILE = CHATS_DIR / "_active.txt"

# The fallback title we use when a chat has no AI-generated title yet —
# kept as a constant so :func:`generate_title` can detect "this title is
# still placeholder, regenerate me" without us having to embed the
# string literal in three places.
DEFAULT_TITLE = "New chat"

# Chat statuses.
STATUS_NEW = "new"
STATUS_RUNNING = "running"
STATUS_IDLE = "idle"
STATUS_ERROR = "error"

# How often the background thread persists progress to disk during a
# long turn. We persist on every event by default so a crash mid-turn
# loses at most one event; that is cheap because each chat is one small
# JSON file.
_PERSIST_EVERY_N_EVENTS = 1


@dataclass
class Chat:
    """One resumable conversation.

    Most fields are JSON-serialized to ``CHATS_DIR/<id>.json``. The two
    leading-underscore fields (``_lock`` / ``_thread``) are runtime-only
    and never persisted — they're attached to the in-memory object so
    every mutation point can grab the lock cheaply.
    """

    id: str
    title: str = DEFAULT_TITLE
    messages: list[dict[str, Any]] = field(default_factory=list)
    ui_turns: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    mode: str = "agent"
    working_dir: str = ""
    status: str = STATUS_NEW
    error_message: str = ""
    partial_text: str = ""
    # In-progress chat-input text that hasn't been submitted yet. Saved
    # to disk on every keystroke (via the JS-side debounce + on-rerun
    # callback in :mod:`chat_input`) so navigating away from the chat
    # page, switching chats, reloading the app, or even a hard process
    # crash never silently drops what the user was typing. Cleared by
    # the submit handler in ``app_pages/chat.py`` once the prompt is
    # consumed by :func:`start_turn`.
    draft_text: str = ""
    archived: bool = False
    created_at: str = ""
    updated_at: str = ""

    # Runtime-only state — excluded from serialization.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    # Set by :func:`request_cancel` to ask the background turn thread
    # (if any) to stop ASAP. The runner checks ``is_set()`` between
    # events yielded by ``agent.run_agent_turn`` and before every
    # ``save_chat`` call, so once we set the event the runner will
    # neither spend more tokens nor write more state to disk.
    # ``start_turn`` clears the event before spawning a thread so a
    # stale signal from a previous turn doesn't kill a fresh one.
    _cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    # Set by :func:`delete_chat` (under ``chat._lock``) once the chat
    # has been popped from session state and unlinked from disk. From
    # that point on every :func:`save_chat` call against this chat
    # short-circuits, which closes the race where the runner is
    # mid-write when delete fires: ``delete_chat`` waits on the lock
    # so the in-flight save completes before the unlink, and any
    # subsequent save sees this flag and bails. Runtime-only.
    _deleted: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable subset of fields."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "ui_turns": self.ui_turns,
            "model": self.model,
            "mode": self.mode,
            "working_dir": self.working_dir,
            "status": self.status,
            "error_message": self.error_message,
            "partial_text": self.partial_text,
            "draft_text": self.draft_text,
            "archived": self.archived,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Chat":
        """Reconstruct a :class:`Chat` from its on-disk JSON.

        Tolerant of missing keys so older files (written before a new
        field was added) still load cleanly. Also tolerant of extra
        keys: any field we used to persist but no longer track (today:
        a removed AI-generated ``description`` field) is silently
        dropped — :func:`save_chat` immediately rewrites the file
        without it on the next persist.
        """
        chat_id = str(raw.get("id") or uuid.uuid4().hex)
        return cls(
            id=chat_id,
            title=str(raw.get("title") or DEFAULT_TITLE),
            messages=list(raw.get("messages") or []),
            ui_turns=list(raw.get("ui_turns") or []),
            model=str(raw.get("model") or ""),
            mode=str(raw.get("mode") or "agent"),
            working_dir=str(raw.get("working_dir") or ""),
            status=str(raw.get("status") or STATUS_NEW),
            error_message=str(raw.get("error_message") or ""),
            partial_text=str(raw.get("partial_text") or ""),
            draft_text=str(raw.get("draft_text") or ""),
            archived=bool(raw.get("archived") or False),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _ensure_chats_dir() -> None:
    """Create ``CHATS_DIR`` if missing. Idempotent."""
    CHATS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string with ``+00:00``."""
    return datetime.now(timezone.utc).isoformat()


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat: Chat) -> None:
    """Persist ``chat`` to ``CHATS_DIR/<id>.json`` atomically.

    Writes to a sibling ``.tmp`` file and then ``os.replace``\\s it onto
    the target so a crash mid-write can never leave a half-written
    JSON file on disk.

    No-op when :attr:`Chat._deleted` is True — the chat has been
    removed from session state and its on-disk file unlinked, and we
    deliberately swallow further writes that would resurrect it.
    The whole operation (snapshot + write + ``os.replace``) runs
    under ``chat._lock`` so :func:`delete_chat` (which acquires the
    same lock to set ``_deleted``) cannot interleave its
    pop-then-unlink between our snapshot and our rename — without
    the long-held lock, a concurrent delete could race the rename
    and leave an orphan file on disk.
    """
    _ensure_chats_dir()
    with chat._lock:
        if chat._deleted:
            return
        chat.updated_at = _now_iso()
        if not chat.created_at:
            chat.created_at = chat.updated_at
        snapshot = chat.to_dict()
        tmp = _chat_path(chat.id).with_suffix(".json.tmp")
        target = _chat_path(chat.id)
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)


def load_all_chats() -> dict[str, Chat]:
    """Read every ``*.json`` in :data:`CHATS_DIR`, returning a sorted dict.

    The returned dict is keyed by chat id and ordered most-recent-first
    by ``updated_at`` so the sidebar can iterate it directly. Files
    that fail to parse are skipped (we don't want one corrupted file to
    blank the whole list).

    Crash-recovery: any chat persisted with ``status == "running"`` is
    rewritten to ``"error"`` because the thread that owned it died
    with the previous process. We immediately persist the rewritten
    record so subsequent loads see the corrected state.

    **Multi-provider migration (Phase 2)**: chats persisted before the
    qualified-id schema (``<provider>:<raw>``) carry bare model ids
    in their ``model`` field — those are W&B Inference ids since W&B
    was the only previously-supported provider. We rewrite them to
    ``"wandb:<bare_id>"`` on load and persist immediately so the next
    launch sees the canonical form. The migration is one-shot per
    chat; subsequent loads short-circuit because the id is already
    qualified.
    """
    from models import is_qualified, qualify

    _ensure_chats_dir()
    chats: list[Chat] = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        chat = Chat.from_dict(raw)
        needs_save = False

        # Migrate bare W&B model ids → ``wandb:<bare>``.
        if chat.model and not is_qualified(chat.model):
            chat.model = qualify(chat.model, default_provider="wandb")
            needs_save = True

        # Crash-recovery: a "running" chat at startup means the previous
        # process died mid-turn. Mark it errored so the UI doesn't show
        # a phantom spinner forever, and clear partial state.
        if chat.status == STATUS_RUNNING:
            chat.status = STATUS_ERROR
            chat.error_message = "Interrupted by app restart."
            chat.partial_text = ""
            needs_save = True

        if needs_save:
            try:
                save_chat(chat)
            except OSError:
                # Best-effort; the in-memory rewrite is still authoritative.
                pass
        chats.append(chat)
    chats.sort(key=lambda c: c.updated_at, reverse=True)
    return {c.id: c for c in chats}


def request_cancel(chat: Chat) -> None:
    """Signal the background turn thread to stop ASAP.

    Sets ``chat._cancel_event``. The runner thread (spawned by
    :func:`start_turn`) checks the event between every agent event
    and before every persistence call, then routes through
    :func:`_finalize_cancelled_turn` to flip status back to
    ``"idle"`` (so the UI stops showing a phantom spinner), preserve
    any in-flight streamed text as a proper ``assistant_text`` event,
    append a ``{"type": "cancelled"}`` marker the renderer surfaces
    as a "Stopped by user" caption, and keep ``chat.messages``
    well-formed for the next turn.

    Idempotent: setting an already-set event is a no-op.

    Two call sites today:

    - The chat-page **Stop** button in :mod:`app_pages.chat`. The
      chat stays around — once the runner finalizes, the user can
      send a new prompt right away.
    - :func:`delete_chat` with ``force=True``. The runner still
      finalizes, but the chat is gone before the user can see it;
      the finalize call's ``save_chat`` becomes a no-op because
      :attr:`Chat._deleted` is set first.

    Returns immediately — it does **not** wait for the thread to
    exit. The streaming chat-completion connection closes shortly
    after the runner hits its next cancel check (once the OpenAI
    Stream object is closed via ``GeneratorExit``).
    """
    chat._cancel_event.set()


def _finalize_cancelled_turn(chat: Chat) -> None:
    """Common cleanup after the runner detects a cancel signal mid-turn.

    Called from every cancel tripwire in the runner spawned by
    :func:`start_turn`. Mutates the chat under its lock so the
    user-visible state ends up consistent regardless of where in
    the agent loop the cancel landed:

    1. **Preserve any in-flight streamed text.** ``chat.partial_text``
       carries content that hasn't yet been wrapped in an
       ``assistant_text`` event. We append a synthesized
       ``assistant_text`` event so the chat page replays whatever
       the model said before the stop.
    2. **Keep ``chat.messages`` well-formed.** When the cancel
       landed before ``_stream_one_call`` could append the assistant
       message (i.e., the last message is the user's prompt), we
       close out the conversation with a stub assistant message so
       the next turn doesn't send two consecutive user messages to
       the API. When the last message is ``"assistant"`` or
       ``"tool"`` the conversation is already in a state the model
       can naturally continue from, so we leave it alone.
    3. **Append a ``"cancelled"`` marker.** The chat page renders
       this as a subtle ":material/stop_circle: Stopped by user."
       caption inside the assistant turn so the user sees that the
       partial reply was cut off intentionally.
    4. **Flip status back to ``"idle"``** so the live fragment
       stops polling and the static renderer takes over. The user
       can immediately send a new prompt.

    The trailing ``save_chat`` is a no-op when
    :attr:`Chat._deleted` is True (the standard path for
    :func:`delete_chat` with ``force=True``), so the same finalize
    helper works for both the standalone Stop and the
    Stop-and-delete paths.
    """
    with chat._lock:
        # The in-flight assistant turn is the last entry in
        # ``ui_turns`` — start_turn appends user_turn then
        # assistant_turn, and the runner doesn't add more entries.
        # Defensive default to a fresh dict so a chat in a weird
        # state (e.g., manually mutated outside this module) still
        # gets its status flipped instead of crashing the runner.
        assistant_turn: dict[str, Any] = {}
        if chat.ui_turns:
            last = chat.ui_turns[-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                assistant_turn = last
        events = assistant_turn.setdefault("events", []) if assistant_turn else []

        partial = chat.partial_text
        if partial and assistant_turn:
            events.append({"type": "assistant_text", "content": partial})

        if chat.messages and chat.messages[-1].get("role") == "user":
            chat.messages.append(
                {
                    "role": "assistant",
                    "content": partial or "[turn cancelled by user]",
                }
            )

        if assistant_turn:
            events.append({"type": "cancelled"})

        chat.status = STATUS_IDLE
        chat.error_message = ""
        chat.partial_text = ""
    try:
        save_chat(chat)
    except OSError:
        pass


def delete_chat(
    chats_dict: dict[str, Chat],
    chat_id: str,
    *,
    force: bool = False,
) -> None:
    """Drop ``chat_id`` from ``chats_dict`` and unlink its on-disk file.

    Refuses by default to delete a chat that's currently running a
    turn — the caller is expected to gate this through a confirm
    dialog, and we still raise so the dialog can surface a useful
    error if the user somehow triggers delete on a freshly-running
    chat.

    With ``force=True``, the chat's cancel event is set first via
    :func:`request_cancel` so the background turn stops as soon as
    it can — the user gets to delete the chat without paying for any
    further tokens. The runner's :func:`_finalize_cancelled_turn`
    call's ``save_chat`` becomes a no-op because we then set
    :attr:`Chat._deleted` (under ``chat._lock``) before the unlink:

    1. Cancel signal goes up; the runner exits its event loop.
    2. We acquire ``chat._lock`` to set ``_deleted=True``. If the
       runner is mid-``save_chat`` the acquire blocks until that
       save's ``os.replace`` has completed.
    3. We pop from ``chats_dict`` and unlink the on-disk file. By
       this point any ``save_chat`` that was in flight when (1)
       fired has finished writing, and any ``save_chat`` that hasn't
       started yet sees ``_deleted=True`` and short-circuits. The
       unlink is the last word.

    Idempotent: a missing dict entry or missing file is not an error.
    """
    chat = chats_dict.get(chat_id)
    if chat is not None and chat.status == STATUS_RUNNING:
        if not force:
            raise RuntimeError(
                "Cannot delete a chat that is still running. "
                "Wait for the current turn to finish, or pass force=True "
                "to abort the turn."
            )
        request_cancel(chat)
    if chat is not None:
        # Take the lock to set ``_deleted`` so any in-flight
        # ``save_chat`` finishes its ``os.replace`` BEFORE we get
        # past this line, and any save_chat that hasn't started
        # yet sees ``_deleted=True`` and short-circuits. See
        # ``save_chat`` for the matching half of the contract.
        with chat._lock:
            chat._deleted = True
    chats_dict.pop(chat_id, None)
    try:
        _chat_path(chat_id).unlink()
    except FileNotFoundError:
        pass
    if load_active_chat_id() == chat_id:
        clear_active_chat_id()


def archive_chat(chat: Chat) -> None:
    """Flip ``chat.archived`` to True and persist."""
    with chat._lock:
        chat.archived = True
    save_chat(chat)


def unarchive_chat(chat: Chat) -> None:
    """Flip ``chat.archived`` to False and persist."""
    with chat._lock:
        chat.archived = False
    save_chat(chat)


def save_active_chat_id(chat_id: str) -> None:
    """Persist ``chat_id`` as the user's last-active chat.

    Best-effort: write failures are swallowed because losing the
    pointer is not fatal — :func:`load_active_chat_id` falls back to
    the most-recently-updated chat when the file is missing.
    """
    if not chat_id:
        return
    _ensure_chats_dir()
    try:
        ACTIVE_CHAT_FILE.write_text(chat_id, encoding="utf-8")
    except OSError:
        pass


def load_active_chat_id() -> str | None:
    """Return the last-active chat id from disk, or ``None`` if missing."""
    try:
        raw = ACTIVE_CHAT_FILE.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    return raw or None


def clear_active_chat_id() -> None:
    """Remove the active-chat pointer (used after a chat is deleted)."""
    try:
        ACTIVE_CHAT_FILE.unlink()
    except FileNotFoundError:
        pass


def has_content(chat: Chat) -> bool:
    """Return True iff the chat has any user or assistant turn yet.

    Used to skip empty placeholder chats when picking the default
    active chat at session start, so the user lands on the
    most-recently-touched conversation rather than a phantom
    ``+ New chat`` row that was never used.
    """
    with chat._lock:
        for turn in chat.ui_turns:
            if turn.get("role") == "user":
                return True
            events = turn.get("events") or []
            if any(ev.get("type") == "assistant_text" for ev in events):
                return True
    return False


def best_default_chat_id(chats_dict: dict[str, Chat]) -> str | None:
    """Pick the most sensible chat to land on at session start.

    Preference order:

    1. The chat the user was most recently *looking at* (persisted via
       :func:`save_active_chat_id`), if it's still around.
    2. The most-recently-updated chat that has at least one user
       message — i.e. a chat with actual content.
    3. The most-recently-updated chat overall (falling back to a
       placeholder ``+ New chat`` when that's all that exists).
    4. ``None`` if the dict is empty.
    """
    if not chats_dict:
        return None
    saved = load_active_chat_id()
    if saved and saved in chats_dict and not chats_dict[saved].archived:
        return saved

    live = [c for c in chats_dict.values() if not c.archived]
    if not live:
        return None

    with_content = [c for c in live if has_content(c)]
    pool = with_content or live
    pool.sort(key=lambda c: c.updated_at, reverse=True)
    return pool[0].id


def find_blank_chat(chats_dict: dict[str, Chat]) -> str | None:
    """Return the id of an existing blank ``+ New chat`` placeholder, or ``None``.

    A "blank" chat is one that:

    - Is not archived.
    - Still has the :data:`DEFAULT_TITLE` placeholder title (i.e. no
      AI-generated title yet, and the user hasn't sent a message that
      would trigger :func:`derive_title`).
    - Carries no user or assistant turn (:func:`has_content` is False).

    Used by the sidebar's ``+ New chat`` button to short-circuit the
    "create another empty placeholder" path: if a blank chat already
    exists, the caller activates that chat instead of accumulating a
    pile of empty rows. When several blank chats exist (e.g. left over
    from earlier sessions), the most-recently-updated one wins so the
    user lands on the freshest placeholder.
    """
    candidates: list[Chat] = []
    for chat in chats_dict.values():
        if chat.archived:
            continue
        if chat.title != DEFAULT_TITLE:
            continue
        if has_content(chat):
            continue
        candidates.append(chat)
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.updated_at, reverse=True)
    return candidates[0].id


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def new_chat(*, model: str = "", mode: str = "agent", working_dir: str = "") -> Chat:
    """Mint a fresh :class:`Chat`, persist it, and return it.

    The chat starts in ``"new"`` status with an empty conversation; the
    UI seeds it via :func:`new_chat` and then either lets the user type
    or auto-activates it.

    ``model`` is normalized to qualified form (``<provider>:<raw>``)
    when callers pass a bare id from the legacy schema — see
    :func:`models.qualify`. New code paths in the UI always pass
    qualified ids, but defending against the legacy form here means
    re-using `new_chat()` from a stale call site doesn't silently
    corrupt the chat file with a non-qualified ``model`` field.
    """
    from models import qualify

    if model:
        model = qualify(model, default_provider="wandb")
    now = _now_iso()
    chat = Chat(
        id=uuid.uuid4().hex,
        title=DEFAULT_TITLE,
        model=model,
        mode=mode,
        working_dir=working_dir,
        status=STATUS_NEW,
        created_at=now,
        updated_at=now,
    )
    save_chat(chat)
    return chat


# ---------------------------------------------------------------------------
# Title derivation
# ---------------------------------------------------------------------------
def derive_title(messages: list[dict[str, Any]], fallback: str = DEFAULT_TITLE) -> str:
    """Best-effort title from the first user message: collapse whitespace + cap to 60 chars."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        single = re.sub(r"\s+", " ", content).strip()
        if len(single) > 60:
            single = single[:57].rstrip() + "..."
        return single
    return fallback


def generate_title(chat: Chat, client: Any, *, api_key: str = "") -> str | None:
    """Ask DeepSeek for a short, descriptive title.

    Returns ``None`` on any failure (the W&B-DeepSeek path isn't
    configured, model not on the account, network blip, model
    returned blank). The caller is expected to fall back to
    :func:`derive_title` in that case.

    The ``api_key`` keyword arg is the W&B Inference API key — DeepSeek
    is currently a W&B-only model, so this should be the user's
    ``ss.provider_keys["wandb"]``. ``client`` is passed for back-compat
    but ignored on the LiteLLM-routed W&B path.

    We feed the model the first user message plus the first assistant
    text response so it has both the question and the agent's framing
    of the answer — that produces sharper titles than the user message
    alone (which is often a one-liner).
    """
    if not (api_key or "").strip():
        # No W&B key configured — auto-titling silently disabled.
        return None
    first_user = ""
    first_assistant = ""
    for msg in chat.messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user" and not first_user:
            first_user = content
        elif role == "assistant" and not first_assistant:
            first_assistant = content
        if first_user and first_assistant:
            break
    if not first_user:
        return None
    user_prompt = (
        "Summarize this chat in 5 words or fewer. Return the title only — "
        "no quotes, no punctuation at the end, no preamble.\n\n"
        f"User: {first_user}\n"
    )
    if first_assistant:
        user_prompt += f"\nAssistant: {first_assistant}\n"
    try:
        title = agent.generate_text(
            client=client,
            model=agent.DEEPSEEK_MODEL,
            system=(
                "You write concise, descriptive titles for chat sessions. "
                "Return only the title. Five words or fewer. "
                "Title case. No trailing punctuation."
            ),
            user=user_prompt,
            max_tokens=30,
            temperature=0.2,
            api_key=api_key,
        )
    except Exception:
        return None
    title = (title or "").strip().strip('"').strip("'")
    # Strip trailing periods/quotes the model might still add.
    title = re.sub(r"[.\s]+$", "", title)
    if not title:
        return None
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title


# ---------------------------------------------------------------------------
# Background turn runner
# ---------------------------------------------------------------------------
def start_turn(
    chat: Chat,
    prompt: str,
    client: Any,
    *,
    override_model: str | None = None,
    provider_id: str | None = None,
    api_key: str | None = None,
    wandb_api_key: str | None = None,
    provider_keys: dict[str, str] | None = None,
    clients: dict[str, Any] | None = None,
    attachments: list[Any] | None = None,
    supports_vision: bool = False,
    supports_pdf_input: bool = False,
) -> None:
    """Spawn a daemon thread that drives one full agent turn for ``chat``.

    The thread:

    1. Appends the user message to ``chat.messages`` + ``chat.ui_turns``
       under the chat's lock, flips status to ``"running"``, persists.
    2. Iterates :func:`agent.run_agent_turn`, accumulating
       ``assistant_text_delta`` events into ``chat.partial_text`` and
       appending every other event to the in-flight assistant turn's
       ``events`` list (under lock).
    3. After the loop finishes, if ``chat.title`` is still
       :data:`DEFAULT_TITLE` and we got at least one assistant
       message, asks DeepSeek for a 5-word title (silent fallback to
       :func:`derive_title` on failure).
    4. Sets status to ``"idle"`` (or ``"error"`` with
       ``error_message``) and clears ``partial_text``.

    Refuses (raises :class:`RuntimeError`) if ``chat.status ==
    "running"`` so the UI never accidentally double-fires a turn.

    ``provider_id`` and ``api_key`` are the multi-provider routing
    arguments. When ``provider_id`` is None it's auto-derived from the
    qualified model id (``<provider>:<raw>``). For ``litellm_compat``
    providers ``client`` may be ``None`` (LiteLLM is stateless) — only
    the API key is needed; we permit ``client is None`` for those.
    """
    from models import is_qualified, model_provider as _model_provider

    if chat.status == STATUS_RUNNING:
        raise RuntimeError(
            f"Chat {chat.id} is already running a turn. Wait for it to finish."
        )

    turn_model = override_model or chat.model
    if not turn_model:
        raise RuntimeError("Chat has no model selected; cannot start a turn.")

    # Auto-derive provider_id from the qualified id when not explicitly
    # passed. Bare ids fall back to "wandb" (legacy single-provider
    # default). The chat-file migration in :func:`load_all_chats`
    # rewrites bare ids to ``wandb:<bare>`` on load, so this fallback
    # only triggers for chats persisted between processes.
    if provider_id is None:
        derived = _model_provider(turn_model) if is_qualified(turn_model) else None
        provider_id = derived or "wandb"

    # ``litellm_compat`` providers don't need a persistent client
    # object — the call layer reads the API key at request time. We
    # only require ``client is not None`` for the three native paths.
    import providers as _providers
    provider = _providers.get_provider(provider_id)
    if provider is None:
        raise RuntimeError(f"Unknown provider id: {provider_id!r}")
    if provider.kind != "litellm_compat" and client is None:
        raise RuntimeError(
            f"Cannot start a turn for {provider.label}: no connected client."
        )
    if provider.kind == "litellm_compat" and not (api_key or "").strip():
        raise RuntimeError(
            f"Cannot start a turn for {provider.label}: no API key configured."
        )

    # Clear any leftover cancel signal from a prior turn. If the
    # previous turn was cancelled (via :func:`request_cancel`) the
    # event is still set; if we didn't clear it, the new runner would
    # exit on its first cancel check.
    chat._cancel_event.clear()

    # Seed the user turn + assistant placeholder synchronously, on the
    # caller's thread, so a freshly-rerendered UI sees the new content
    # immediately even before the background thread schedules.
    #
    # Attachments path: when ``attachments`` is non-empty we route
    # through :func:`attachments.build_user_message` to build a
    # multimodal content array for ``chat.messages`` plus a metadata
    # list for ``chat.ui_turns``. The single-string path (no
    # attachments) preserves the legacy plain-text shape so
    # downstream providers without multimodal support keep working.
    attachment_records: list[dict[str, Any]] = []
    if attachments:
        from pathlib import Path as _Path

        import attachments as _attachments

        wd_path = (
            _Path(chat.working_dir).expanduser().resolve()
            if chat.working_dir
            else _Path.cwd()
        )
        message, attachment_records = _attachments.build_user_message(
            prompt,
            attachments,
            working_dir=wd_path,
            supports_vision=supports_vision,
            supports_pdf_input=supports_pdf_input,
        )
        user_turn: dict[str, Any] = {
            "role": "user",
            "content": prompt,
            "attachments": attachment_records,
        }
    else:
        message = {"role": "user", "content": prompt}
        user_turn = {"role": "user", "content": prompt}

    assistant_turn: dict[str, Any] = {"role": "assistant", "events": []}
    with chat._lock:
        chat.messages.append(message)
        chat.ui_turns.append(user_turn)
        chat.ui_turns.append(assistant_turn)
        if not chat.title or chat.title == DEFAULT_TITLE:
            chat.title = derive_title(chat.messages, fallback=DEFAULT_TITLE)
        chat.status = STATUS_RUNNING
        chat.error_message = ""
        chat.partial_text = ""
    save_chat(chat)

    def _runner() -> None:
        events_seen = 0
        turn_started = time.monotonic()
        # Per-turn usage accumulator; matches the bookkeeping the
        # synchronous chat page used so the Usage dashboard sees the
        # same per-turn rows it always has.
        turn_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }

        # Tiny shorthand for the cancel check. ``request_cancel`` sets
        # this flag; every check site below routes through
        # :func:`_finalize_cancelled_turn` so the chat ends up at
        # IDLE (with a "Stopped by user" marker preserving any
        # partial reply) rather than stuck at RUNNING — the latter
        # would leave the live fragment polling forever and the
        # chat input disabled. ``save_chat`` becomes a no-op when
        # ``chat._deleted`` is True (the standard path for
        # :func:`delete_chat` with ``force=True``), so the same
        # finalize call works for the standalone Stop and the
        # Stop-and-delete paths.
        def _cancelled() -> bool:
            return chat._cancel_event.is_set()

        try:
            working_dir = (
                Path(chat.working_dir).expanduser().resolve()
                if chat.working_dir
                else Path.cwd()
            )
            events_iter = agent.run_agent_turn(
                client=client,
                model=turn_model,
                messages=chat.messages,
                working_dir=working_dir,
                mode=chat.mode,
                provider_id=provider_id,
                api_key=api_key,
                chat_id=chat.id,
                provider_keys=provider_keys,
                clients=clients,
            )
            for event in events_iter:
                # Cancellation tripwire #1: between every event from the
                # agent generator. ``events_iter.close()`` raises
                # ``GeneratorExit`` inside ``run_agent_turn`` ->
                # ``_stream_one_call``, which closes the streaming OpenAI
                # response and drops the W&B Inference connection so the
                # server stops generating tokens. ``_finalize_cancelled_turn``
                # then preserves whatever was already streamed as a
                # proper assistant_text event, appends a "cancelled"
                # marker, and flips status back to IDLE so the user
                # can immediately send a new prompt.
                if _cancelled():
                    try:
                        events_iter.close()
                    except Exception:
                        pass
                    _finalize_cancelled_turn(chat)
                    return
                etype = event.get("type")
                if etype == "assistant_text_delta":
                    with chat._lock:
                        chat.partial_text += event.get("content") or ""
                    # Don't persist on every delta — they fire fast and
                    # would thrash the disk. The `assistant_text` event
                    # that follows carries the full content for replay.
                    continue
                if etype == "usage":
                    turn_usage["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
                    turn_usage["completion_tokens"] += int(event.get("completion_tokens") or 0)
                    turn_usage["total_tokens"] += int(event.get("total_tokens") or 0)
                    turn_usage["rounds"] += 1
                    continue
                with chat._lock:
                    if etype == "assistant_text":
                        # The full assistant message landed; the
                        # streaming buffer can be cleared so the live
                        # view stops showing the duplicate.
                        chat.partial_text = ""
                    assistant_turn["events"].append(event)
                events_seen += 1
                if events_seen % _PERSIST_EVERY_N_EVENTS == 0:
                    # Cancellation tripwire #2: every persistence point.
                    # If we already wrote an event under the lock above
                    # but a cancel landed before we hit ``save_chat``,
                    # finalize so the chat ends up at IDLE rather than
                    # leaving the assistant_turn events list partially
                    # filled with no closing marker.
                    if _cancelled():
                        _finalize_cancelled_turn(chat)
                        return
                    try:
                        save_chat(chat)
                    except OSError:
                        pass
        except Exception as e:
            # Cancelling a streaming turn often surfaces as an exception
            # raised inside the OpenAI SDK as the connection is closed
            # (e.g. ``APIError`` / ``StreamError``). Route through the
            # cancel-finalize path when cancelled so we don't persist
            # a misleading "error" status — the user explicitly asked
            # to stop.
            if _cancelled():
                _finalize_cancelled_turn(chat)
                return
            with chat._lock:
                assistant_turn["events"].append({"type": "error", "message": str(e)})
                chat.status = STATUS_ERROR
                chat.error_message = f"{type(e).__name__}: {e}"
                chat.partial_text = ""
            try:
                save_chat(chat)
            except OSError:
                pass
            return

        # Cancellation tripwire #3: between the agent loop and the
        # post-turn finalization (usage row, auto-title). Each of
        # these is itself a small disk write or LLM call, and
        # "cancelled" means "skip every one".
        if _cancelled():
            _finalize_cancelled_turn(chat)
            return

        # Persist + emit per-turn usage just like the synchronous loop
        # used to. The dashboard reads ``~/.wb_coding_agent/usage.jsonl``
        # so it picks this up automatically; the matching ``turn_usage``
        # event lets the chat page render the per-turn caption on
        # replay without redoing the math.
        if turn_usage["total_tokens"] > 0:
            duration = time.monotonic() - turn_started
            try:
                entry = usage_log.build_entry(
                    model=turn_model,
                    prompt_tokens=turn_usage["prompt_tokens"],
                    completion_tokens=turn_usage["completion_tokens"],
                    total_tokens=turn_usage["total_tokens"],
                    rounds=turn_usage["rounds"],
                    duration_seconds=duration,
                    mode=chat.mode,
                )
                usage_log.record_usage(entry)
            except Exception:
                entry = None
            if entry is not None:
                with chat._lock:
                    assistant_turn["events"].append(
                        {
                            "type": "turn_usage",
                            "model": entry["model"],
                            "prompt_tokens": entry["prompt_tokens"],
                            "completion_tokens": entry["completion_tokens"],
                            "total_tokens": entry["total_tokens"],
                            "cost_usd": entry.get("cost_usd"),
                            "rounds": entry["rounds"],
                            "duration_seconds": entry.get("duration_seconds"),
                        }
                    )

        # Auto-title after the first successful turn. We only attempt
        # this when the title is still the default *and* the turn
        # produced an assistant message — partial errors keep their
        # default title so the next attempt can re-title cleanly.
        had_assistant = any(
            ev.get("type") == "assistant_text" and (ev.get("content") or "").strip()
            for ev in assistant_turn["events"]
        )
        with chat._lock:
            needs_title = chat.title == DEFAULT_TITLE or chat.title.startswith(
                "Untitled"
            )
        if had_assistant and needs_title and not _cancelled():
            # Auto-titling pins to the W&B-DeepSeek path (still the
            # one provider that ships the V4-Flash model). When the
            # user hasn't configured W&B (or the per-turn provider
            # is non-W&B and they didn't pass us a W&B key), fall
            # through to ``derive_title`` instead of trying.
            new_title = generate_title(
                chat,
                client,
                api_key=(wandb_api_key or "").strip(),
            )
            if new_title:
                with chat._lock:
                    chat.title = new_title
            else:
                # Fallback: use the user-message-derived title (which
                # may already be in chat.title from the seed step
                # above) and accept it as the final title.
                with chat._lock:
                    if not chat.title or chat.title == DEFAULT_TITLE:
                        chat.title = derive_title(chat.messages, fallback="Untitled")

        # Final cancellation tripwire before the status flip + save.
        if _cancelled():
            _finalize_cancelled_turn(chat)
            return

        with chat._lock:
            chat.status = STATUS_IDLE
            chat.error_message = ""
            chat.partial_text = ""
        try:
            save_chat(chat)
        except OSError:
            pass

    thread = threading.Thread(target=_runner, name=f"chat-turn-{chat.id}", daemon=True)
    chat._thread = thread
    thread.start()


__all__ = [
    "CHATS_DIR",
    "ACTIVE_CHAT_FILE",
    "DEFAULT_TITLE",
    "STATUS_NEW",
    "STATUS_RUNNING",
    "STATUS_IDLE",
    "STATUS_ERROR",
    "Chat",
    "save_chat",
    "load_all_chats",
    "delete_chat",
    "request_cancel",
    "archive_chat",
    "unarchive_chat",
    "save_active_chat_id",
    "load_active_chat_id",
    "clear_active_chat_id",
    "best_default_chat_id",
    "find_blank_chat",
    "has_content",
    "new_chat",
    "derive_title",
    "generate_title",
    "start_turn",
]
