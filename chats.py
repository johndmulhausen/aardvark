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
    description: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    ui_turns: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    mode: str = "agent"
    working_dir: str = ""
    status: str = STATUS_NEW
    error_message: str = ""
    partial_text: str = ""
    archived: bool = False
    created_at: str = ""
    updated_at: str = ""

    # Runtime-only state — excluded from serialization.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable subset of fields."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "messages": self.messages,
            "ui_turns": self.ui_turns,
            "model": self.model,
            "mode": self.mode,
            "working_dir": self.working_dir,
            "status": self.status,
            "error_message": self.error_message,
            "partial_text": self.partial_text,
            "archived": self.archived,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Chat":
        """Reconstruct a :class:`Chat` from its on-disk JSON.

        Tolerant of missing keys so older files (written before a new
        field was added) still load cleanly.
        """
        chat_id = str(raw.get("id") or uuid.uuid4().hex)
        return cls(
            id=chat_id,
            title=str(raw.get("title") or DEFAULT_TITLE),
            description=str(raw.get("description") or ""),
            messages=list(raw.get("messages") or []),
            ui_turns=list(raw.get("ui_turns") or []),
            model=str(raw.get("model") or ""),
            mode=str(raw.get("mode") or "agent"),
            working_dir=str(raw.get("working_dir") or ""),
            status=str(raw.get("status") or STATUS_NEW),
            error_message=str(raw.get("error_message") or ""),
            partial_text=str(raw.get("partial_text") or ""),
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
    JSON file on disk. We grab a *snapshot* of the chat under its lock
    before serialization so a background thread mutating the chat
    concurrently doesn't produce a torn read.
    """
    _ensure_chats_dir()
    with chat._lock:
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
    """
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
        # Crash-recovery: a "running" chat at startup means the previous
        # process died mid-turn. Mark it errored so the UI doesn't show
        # a phantom spinner forever, and clear partial state.
        if chat.status == STATUS_RUNNING:
            chat.status = STATUS_ERROR
            chat.error_message = "Interrupted by app restart."
            chat.partial_text = ""
            try:
                save_chat(chat)
            except OSError:
                # Best-effort; the in-memory rewrite is still authoritative.
                pass
        chats.append(chat)
    chats.sort(key=lambda c: c.updated_at, reverse=True)
    return {c.id: c for c in chats}


def delete_chat(chats_dict: dict[str, Chat], chat_id: str) -> None:
    """Drop ``chat_id`` from ``chats_dict`` and unlink its on-disk file.

    Refuses to delete a chat that's currently running a turn. The
    caller is expected to have already gated this through a confirm
    dialog; we still raise so the dialog can surface a useful error
    if the user somehow triggers delete on a freshly-running chat.

    Idempotent: a missing dict entry or missing file is not an error.
    """
    chat = chats_dict.get(chat_id)
    if chat is not None and chat.status == STATUS_RUNNING:
        raise RuntimeError(
            "Cannot delete a chat that is still running. "
            "Wait for the current turn to finish."
        )
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
    """
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


def generate_title(chat: Chat, client: Any) -> str | None:
    """Ask DeepSeek for a short, descriptive title.

    Returns ``None`` on any failure (no client, model not on the
    account, network blip, model returned blank). The caller is
    expected to fall back to :func:`derive_title` in that case.

    We feed the model the first user message plus the first assistant
    text response so it has both the question and the agent's framing
    of the answer — that produces sharper titles than the user message
    alone (which is often a one-liner).
    """
    if client is None:
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


def generate_description(chat: Chat, client: Any) -> str | None:
    """Ask DeepSeek for a one-sentence description of the chat.

    Used by the sidebar's collapsible chat row, where the longer
    description renders inside the expanded body (below the short
    title in the always-visible header). Re-generated after every
    successful turn so the description tracks the conversation as
    it evolves; the call is cheap because it goes to V4-Flash with
    a tiny prompt.

    Returns ``None`` on any failure (no client, model not on the
    account, network blip, model returned blank). The caller treats
    that as "leave the prior description in place".
    """
    if client is None:
        return None
    first_user = ""
    last_assistant = ""
    for msg in chat.messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user" and not first_user:
            first_user = content
        elif role == "assistant":
            last_assistant = content
    if not first_user:
        return None
    user_prompt = (
        "Write one short sentence (no more than 120 characters) "
        "describing what this chat is about. Use plain prose, no "
        "trailing period, no quotes, no preamble.\n\n"
        f"User: {first_user}\n"
    )
    if last_assistant:
        user_prompt += f"\nAssistant: {last_assistant}\n"
    try:
        desc = agent.generate_text(
            client=client,
            model=agent.DEEPSEEK_MODEL,
            system=(
                "You write concise one-sentence descriptions of coding "
                "chat sessions. Return only the sentence. At most 120 "
                "characters. No trailing punctuation."
            ),
            user=user_prompt,
            max_tokens=80,
            temperature=0.2,
        )
    except Exception:
        return None
    desc = (desc or "").strip().strip('"').strip("'")
    desc = re.sub(r"[.\s]+$", "", desc)
    if not desc:
        return None
    if len(desc) > 140:
        desc = desc[:137].rstrip() + "..."
    return desc


# ---------------------------------------------------------------------------
# Background turn runner
# ---------------------------------------------------------------------------
def start_turn(
    chat: Chat,
    prompt: str,
    client: Any,
    *,
    override_model: str | None = None,
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
    """
    if chat.status == STATUS_RUNNING:
        raise RuntimeError(
            f"Chat {chat.id} is already running a turn. Wait for it to finish."
        )
    if client is None:
        raise RuntimeError("Cannot start a turn without a connected W&B Inference client.")

    turn_model = override_model or chat.model
    if not turn_model:
        raise RuntimeError("Chat has no model selected; cannot start a turn.")

    # Seed the user turn + assistant placeholder synchronously, on the
    # caller's thread, so a freshly-rerendered UI sees the new content
    # immediately even before the background thread schedules.
    user_turn = {"role": "user", "content": prompt}
    assistant_turn: dict[str, Any] = {"role": "assistant", "events": []}
    with chat._lock:
        chat.messages.append({"role": "user", "content": prompt})
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
            )
            for event in events_iter:
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
                    try:
                        save_chat(chat)
                    except OSError:
                        pass
        except Exception as e:
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
        if had_assistant and needs_title:
            new_title = generate_title(chat, client)
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

        # Refresh the longer one-sentence description on every successful
        # turn so the sidebar's expanded chat row tracks the conversation
        # as it evolves (not just the first turn). Cheap because the
        # call is V4-Flash with a tiny prompt; failures silently leave
        # the prior description in place.
        if had_assistant:
            new_desc = generate_description(chat, client)
            if new_desc:
                with chat._lock:
                    chat.description = new_desc

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
    "generate_description",
    "start_turn",
]
