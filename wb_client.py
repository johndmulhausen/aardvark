"""W&B Inference client wrapper.

Wraps the OpenAI Python SDK pointed at the W&B Inference base URL and provides
a thin helper for initializing W&B Weave so every chat-completion call that
flows through the OpenAI client is automatically traced.

Weave's contract: calling :func:`weave.init` patches ``openai.OpenAI`` at
import time so any subsequent ``client.chat.completions.create`` becomes a
child op of the currently active ``@weave.op`` (if any) and is logged with
inputs, outputs, latency, and token usage. The decorators in ``agent.py`` are
the parents that give those auto-traced calls a useful agent-loop hierarchy.
"""
from __future__ import annotations

import os

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
    client = weave.init(target)
    return client, target
