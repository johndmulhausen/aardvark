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
"""
from __future__ import annotations

import os
from typing import Any

import weave
from openai import OpenAI
from weave.trace.weave_client import WeaveClient

WB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"

# Project Weave logs traces under when the user has not set the optional
# `team/project` field in the sidebar. Weave creates this project under the
# user's default entity on first call, so no upfront setup is required.
DEFAULT_WEAVE_PROJECT = "wandb-coding-agent"


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


def init_weave(api_key: str, project: str | None = None) -> tuple[WeaveClient, str]:
    """Initialize W&B Weave so every OpenAI call gets traced.

    Sets ``WANDB_API_KEY`` from ``api_key`` so :func:`weave.init` can
    authenticate non-interactively (it would otherwise prompt on stdin, which
    is broken in a Streamlit/desktop context), then calls
    ``weave.init(project)``. After this returns, ``openai.OpenAI`` is patched
    and any chat-completion call from ``agent.py`` is captured as a child of
    the surrounding ``@weave.op``.

    Reuses the same `team/project` string the user pastes in the sidebar (also
    used for W&B Inference usage attribution by ``make_client``); when empty,
    falls back to :data:`DEFAULT_WEAVE_PROJECT` under the user's default
    entity.

    Returns a ``(client, project)`` pair where ``project`` is the resolved
    project string the UI can display to the user.
    """
    os.environ["WANDB_API_KEY"] = api_key
    target = project.strip() if project and project.strip() else DEFAULT_WEAVE_PROJECT
    # Patch BEFORE weave.init so the openai integration (which is wired up
    # during init / first openai call) picks up our tolerant post-processor.
    _patch_weave_openai_finish_reason()
    client = weave.init(target)
    return client, target
