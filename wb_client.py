"""W&B Inference client wrapper.

Wraps the OpenAI Python SDK pointed at the W&B Inference base URL.
"""
from __future__ import annotations

from openai import OpenAI

WB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"


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
