"""W&B Inference client wrapper.

Wraps the OpenAI Python SDK pointed at the W&B Inference base URL and provides
a thin helper for initializing W&B Weave so every chat-completion call that
flows through the OpenAI client is automatically traced.

Weave's contract: calling :func:`weave.init` patches ``openai.OpenAI`` at
import time so any subsequent ``client.chat.completions.create`` becomes a
child op of the currently active ``@weave.op`` (if any) and is logged with
inputs, outputs, latency, and token usage. The decorators in ``agent.py`` are
the parents that give those auto-traced calls a useful agent-loop hierarchy.

Compatibility shim: W&B Inference (and most other OpenAI-compatible providers)
emit ``finish_reason: null`` on the last chunk of a streamed completion rather
than the strict enum OpenAI itself uses. Weave's openai integration
reconstructs an :class:`openai.types.ChatCompletion` from the accumulated
stream and validates that field with pydantic, which crashes on ``None``. We
patch Weave's stream post-processor at init time to default a missing
``finish_reason`` to ``"tool_calls"`` (when tool calls were produced) or
``"stop"`` (otherwise) so traces are captured cleanly. See
:func:`_patch_weave_openai_finish_reason`.

Storage-pressure auto-prune
---------------------------
W&B Weave does not expose an account-level storage quota via the SDK, so we
can't proactively check "% full". Instead, :class:`_StorageHandler` attaches
to the ``weave`` logger tree at init time and watches for upload-pipeline
errors that look like storage / quota exhaustion (substrings ``storage``,
``quota``, ``insufficient``, ``exceeded``, or HTTP 403 paired with
``forbidden``/``limit``). When one fires, the handler sets a process-wide
``threading.Event``; the agent loop calls
:func:`prune_oldest_calls_if_pressured` at the end of each turn, which deletes
the oldest 200 root calls from the active project, clears the flag, and
debounces further prunes for 60 s so a burst of failed uploads doesn't
cascade into repeated deletions.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import weave
from openai import OpenAI
from weave.trace import urls as weave_urls
from weave.trace.context import weave_client_context
from weave.trace.weave_client import WeaveClient

WB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"

# Project Weave logs traces under when the user has not set the optional
# `team/project` field in the sidebar. Weave creates this project under the
# user's default entity on first call, so no upfront setup is required.
DEFAULT_WEAVE_PROJECT = "wandb-coding-agent"

# How many root calls to delete per prune cycle. The W&B Service API caps a
# single ``calls_delete`` request at 1000; 200 is a conservative default that
# clears enough headroom to keep tracing for several more turns without
# delaying the user noticeably.
DEFAULT_PRUNE_BATCH_SIZE = 200
MAX_PRUNE_BATCH_SIZE = 1000

# Don't re-run the prune more than once per minute even if the pressure flag
# keeps re-firing. Backend visibility lags after a delete; prune storms also
# eat into the user's ingest quota with their own delete traffic.
PRUNE_DEBOUNCE_SECONDS = 60.0

# Substrings (case-insensitive) on a Weave log record that we treat as
# evidence of storage / quota pressure. Deliberately conservative: 5xx
# transients and 413 ("payload too large") are NOT in this list because they
# don't indicate the user is out of storage.
_STORAGE_KEYWORDS = ("storage", "quota", "insufficient", "exceeded")

# Module-level state shared across threads. The Weave upload thread sets the
# flag from inside the logging handler; the agent loop reads it at end of
# turn from the main thread.
_storage_pressure_flag = threading.Event()
_storage_pressure_reason: str | None = None
_storage_pressure_lock = threading.Lock()
_last_prune_ts: float = 0.0
_storage_handler_attached = False


class _StorageHandler(logging.Handler):
    """Watches the ``weave`` logger for storage-quota signals.

    Weave's background upload thread funnels permanent upload failures through
    two log records:

    - ``weave.utils.retry`` at INFO ``"retry_failed"`` with the underlying
      exception attached on the record's ``exception`` extra field.
    - ``weave.trace_server_bindings.remote_http_trace_server`` at WARNING
      ``"Batch failed after max retries, requeueing batch ..."``.

    On every record we examine the formatted message plus any attached
    exception text for the keywords in :data:`_STORAGE_KEYWORDS` (or for
    HTTP 403 paired with the words ``forbidden`` / ``limit``). When matched,
    we set :data:`_storage_pressure_flag` and stash a short reason so the UI
    can display *why* the prune fired.

    All exceptions raised inside the handler are swallowed; a bug in detection
    must never crash Weave's upload pipeline.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            haystack_parts = [record.getMessage()]
            exc_extra = record.__dict__.get("exception")
            if exc_extra:
                haystack_parts.append(str(exc_extra))
            if record.exc_info:
                haystack_parts.append(repr(record.exc_info))
            haystack = " ".join(haystack_parts).lower()

            matched = False
            for kw in _STORAGE_KEYWORDS:
                if kw in haystack:
                    matched = True
                    break
            if not matched and "403" in haystack and (
                "forbidden" in haystack or "limit" in haystack
            ):
                matched = True

            if not matched:
                return

            # Compose a short reason combining the log message with any
            # attached exception. ``retry_failed`` records carry the actual
            # 4xx text on the ``exception`` extra, not in the message.
            reason = record.getMessage()
            if exc_extra:
                reason = f"{reason}: {exc_extra}"
            # Cap to keep the UI caption tidy.
            if len(reason) > 200:
                reason = reason[:197] + "..."

            global _storage_pressure_reason
            with _storage_pressure_lock:
                _storage_pressure_reason = reason
            _storage_pressure_flag.set()
        except Exception:
            # Detection bugs must never crash the Weave upload thread.
            pass

    def handleError(self, record: logging.LogRecord) -> None:
        # Default behavior writes to sys.stderr; we'd rather stay silent
        # because logging from inside a logging handler can re-enter Weave.
        pass


def _ensure_storage_handler_attached() -> None:
    """Idempotently attach :class:`_StorageHandler` to the ``weave`` logger.

    Mirrors :func:`_patch_weave_openai_finish_reason`'s "patch once" pattern.
    Setting the handler level to INFO ensures we see ``retry_failed`` records
    from ``weave.utils.retry``, which are the earliest signal of permanent
    upload failure.
    """
    global _storage_handler_attached
    if _storage_handler_attached:
        return
    handler = _StorageHandler(level=logging.INFO)
    weave_logger = logging.getLogger("weave")
    weave_logger.addHandler(handler)
    # Don't lower the logger's effective level globally — users may have
    # configured ``weave`` at WARNING. We only need to *receive* INFO records
    # if the logger is already configured to emit them; raising the logger's
    # level here would surprise users who rely on the default verbosity.
    _storage_handler_attached = True


def reset_storage_pressure() -> None:
    """Clear the storage-pressure flag and reason. Test/debug helper."""
    global _storage_pressure_reason
    _storage_pressure_flag.clear()
    with _storage_pressure_lock:
        _storage_pressure_reason = None


def make_client(api_key: str, project: str | None = None) -> OpenAI:
    """Create an OpenAI client configured for W&B Inference.

    The `project` argument, when provided, is forwarded to W&B for usage
    attribution and should be in the form `team/project`.
    """
    kwargs: dict = {
        "base_url": WB_INFERENCE_BASE_URL,
        "api_key": api_key,
    }
    if project:
        kwargs["project"] = project
    return OpenAI(**kwargs)


def list_models(client: OpenAI) -> list[str]:
    """Return all model IDs available on the connected W&B Inference account."""
    response = client.models.list()
    return sorted({m.id for m in response.data})


def _patch_weave_openai_finish_reason() -> None:
    """Make Weave's openai stream post-processor tolerant of ``finish_reason=None``.

    Weave's :mod:`weave.integrations.openai.openai_sdk` reconstructs a strict
    :class:`openai.types.ChatCompletion` from the accumulated streaming
    response and validates ``choices[].finish_reason`` against the OpenAI
    enum (``stop``/``length``/``tool_calls``/``content_filter``/``function_call``).
    W&B Inference's last chunk has ``finish_reason: null``, which trips the
    pydantic validator and prints a long traceback to stderr on every turn.

    We rewrite the integration's `openai_on_finish_post_processor` reference
    to a wrapper that mutates the incoming chunk in place: if any choice has
    ``finish_reason is None`` we set it to ``"tool_calls"`` when that choice
    has accumulated tool-call deltas and ``"stop"`` otherwise. The original
    function is then called and produces a valid trace. Idempotent: re-running
    after the patch is a no-op (we mark the wrapper with an attribute).
    """
    try:
        from weave.integrations.openai import openai_sdk  # type: ignore[import-not-found]
    except Exception:
        return

    original = getattr(openai_sdk, "openai_on_finish_post_processor", None)
    if original is None or getattr(original, "_wb_inference_patched", False):
        return

    def _patched(value: Any) -> Any:
        try:
            choices = getattr(value, "choices", None) or []
            for choice in choices:
                if getattr(choice, "finish_reason", "missing") is None:
                    delta = getattr(choice, "delta", None)
                    has_tool_calls = bool(
                        delta is not None and getattr(delta, "tool_calls", None)
                    )
                    choice.finish_reason = "tool_calls" if has_tool_calls else "stop"
        except Exception:
            pass
        return original(value)

    _patched._wb_inference_patched = True  # type: ignore[attr-defined]
    openai_sdk.openai_on_finish_post_processor = _patched


def init_weave(
    api_key: str, project: str | None = None
) -> tuple[WeaveClient, str, str]:
    """Initialize W&B Weave so every chat-completion call gets traced.

    Sets ``WANDB_API_KEY`` from ``api_key`` so :func:`weave.init` can
    authenticate non-interactively (it would otherwise prompt on stdin, which
    is broken in a Streamlit/desktop context), then calls
    ``weave.init(project)``. After this returns, **every supported SDK is
    auto-patched** — Weave 0.52+ ships native integrations for ``openai``,
    ``anthropic``, and ``google_genai`` (the three SDKs we use). Weave
    also ships integrations for several inference servers we don't
    list (``groq``, ``cerebras``, ``mistral``, etc.), but those are
    no-ops for us because we don't import their SDKs.

    All three dispatch paths in :func:`chat_streams.stream_chat` therefore
    log into the same trace tree without per-provider wiring beyond the
    existing ``@_op`` decorator on ``_stream_one_call``:

    - ``openai_native`` → patched ``openai.OpenAI`` (one trace per call)
    - ``openai_compat`` → patched ``openai.OpenAI`` with per-provider
      ``base_url`` (same auto-patch hook fires regardless of base URL)
    - ``anthropic_native`` → patched ``anthropic.Anthropic``
    - ``google_native`` → patched ``google.genai.Client``

    Reuses the same `team/project` string the user pastes in the sidebar (also
    used for W&B Inference usage attribution by ``make_client``); when empty,
    falls back to :data:`DEFAULT_WEAVE_PROJECT` under the user's default
    entity.

    Returns a ``(client, label, url)`` triple where:

    - ``label`` is the resolved ``entity/project`` string the UI displays
      (built from :attr:`WeaveClient.entity` / :attr:`WeaveClient.project`,
      so we get back whatever Weave's project-name slug-fixup actually used
      rather than echoing the user's input).
    - ``url`` is a deep link to the project's Weave traces page, built via
      :func:`weave.trace.urls.project_weave_root_url` so private W&B
      deployments and projects with URL-unsafe characters both work.
    """
    os.environ["WANDB_API_KEY"] = api_key
    target = project.strip() if project and project.strip() else DEFAULT_WEAVE_PROJECT
    # Patch BEFORE weave.init so the openai integration (which is wired up
    # during init / first openai call) picks up our tolerant post-processor.
    _patch_weave_openai_finish_reason()
    client = weave.init(target)
    # Now that the weave logger tree is in use, attach the storage-pressure
    # listener so any subsequent upload failure with a quota-shaped message
    # flips the prune flag. Idempotent across reconnects.
    _ensure_storage_handler_attached()
    label = f"{client.entity}/{client.project}"
    url = weave_urls.project_weave_root_url(client.entity, client.project)
    return client, label, url


def prune_oldest_calls(
    client: WeaveClient,
    *,
    batch_size: int = DEFAULT_PRUNE_BATCH_SIZE,
) -> dict[str, Any]:
    """Delete the oldest ``batch_size`` root calls from the active project.

    Used by :func:`prune_oldest_calls_if_pressured` after the storage-pressure
    flag flips. We restrict to root calls (``trace_roots_only=True``) because
    Weave deletes children transitively when a root is removed; trying to
    delete a child whose parent has already been deleted would be wasteful
    and risks confusing later traversal.

    Returns a JSON-serializable status dict mirroring the contract used by
    other agent events: ``status="ok"`` with ``deleted: int`` on success;
    ``status="error"`` with ``message: str`` on failure (the caller never
    raises).
    """
    if batch_size <= 0:
        batch_size = DEFAULT_PRUNE_BATCH_SIZE
    batch_size = min(batch_size, MAX_PRUNE_BATCH_SIZE)

    try:
        from weave.trace_server.trace_server_interface import (
            CallsDeleteReq,
            CallsFilter,
            SortBy,
        )

        calls_iter = client.get_calls(
            filter=CallsFilter(trace_roots_only=True),
            sort_by=[SortBy(field="started_at", direction="asc")],
            limit=batch_size,
            # ``id`` is always returned; asking for nothing else keeps the
            # query light when the project has many calls.
            columns=["id"],
        )
        call_ids = [c.id for c in calls_iter]
        if not call_ids:
            return {"status": "ok", "deleted": 0}
        client.server.calls_delete(
            CallsDeleteReq(
                project_id=client._project_id(),
                call_ids=call_ids,
            )
        )
        return {"status": "ok", "deleted": len(call_ids)}
    except Exception as e:  # noqa: BLE001 — surfaced verbatim to the UI
        return {"status": "error", "message": str(e)}


def prune_oldest_calls_if_pressured(
    *,
    batch_size: int = DEFAULT_PRUNE_BATCH_SIZE,
) -> dict[str, Any] | None:
    """Run :func:`prune_oldest_calls` only if Weave has flagged storage pressure.

    Called by ``agent.run_agent_turn`` at the end of every turn. Returns
    ``None`` when the flag is unset (so the agent can skip yielding an event
    altogether), or one of:

    - ``{"status": "skipped", "reason": "no_client"}`` — pressure flag set
      but Weave isn't initialized in this process.
    - ``{"status": "skipped", "reason": "debounced"}`` — another prune ran
      within the last :data:`PRUNE_DEBOUNCE_SECONDS`.
    - ``{"status": "ok", "deleted": N, "reason": <quota log message>}`` —
      successful prune; flag and reason cleared.
    - ``{"status": "error", "message": str}`` — query or delete raised; the
      flag is cleared so we don't fail in a loop, but the next storage log
      record will trip it again.

    On any non-``None`` return the pressure flag is cleared.
    """
    if not _storage_pressure_flag.is_set():
        return None

    global _last_prune_ts, _storage_pressure_reason
    now = time.monotonic()
    if now - _last_prune_ts < PRUNE_DEBOUNCE_SECONDS:
        # Keep the flag set so a later turn (after the debounce window) will
        # actually prune; just don't fire right now.
        return {"status": "skipped", "reason": "debounced"}

    client = weave_client_context.get_weave_client()
    if client is None:
        # Pressure was detected but Weave isn't initialized in this process,
        # which shouldn't happen in normal flow. Clear the flag so we don't
        # spin on it indefinitely.
        _storage_pressure_flag.clear()
        return {"status": "skipped", "reason": "no_client"}

    with _storage_pressure_lock:
        reason = _storage_pressure_reason

    result = prune_oldest_calls(client, batch_size=batch_size)
    _last_prune_ts = time.monotonic()
    _storage_pressure_flag.clear()
    with _storage_pressure_lock:
        _storage_pressure_reason = None

    if result.get("status") == "ok" and reason:
        result["reason"] = reason
    return result
