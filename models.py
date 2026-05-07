"""Static metadata for the W&B Inference catalog.

This module is the single source of truth for model display labels, descriptions,
context windows, parameter counts, and **per-million-token pricing**. The
information is transcribed from two W&B docs pages:

- https://docs.wandb.ai/inference/models  (label / description / context / params)
- https://wandb.ai/site/pricing/inference   (input / output / cache prices)

Both ``streamlit_app.py`` (model-card UI under the chat input) and
``usage.py`` (cost computation for the usage dashboard) import from here. We
keep this dict in its own module rather than ``streamlit_app.py`` so non-UI
code (``usage.py``, ``app_pages/usage.py``) can read pricing without dragging
in Streamlit.

Adding a new model
------------------
1. Add a row to :data:`MODEL_METADATA` keyed by the exact API id returned by
   the ``/v1/models`` endpoint.
2. Fill in ``label`` / ``description`` / ``context`` / ``params`` from the
   "Available models" page.
3. Fill in ``input_price_per_1m`` / ``output_price_per_1m`` (and optionally
   ``cache_hit_price_per_1m``) from the pricing page. If the model is not yet
   listed (e.g. an experimental preview), leave the prices as ``None`` — the
   usage dashboard renders ``-`` for cost on those turns.
4. Set ``"weak_tool_calling": True`` only when the model is documented (or
   reliably observed) to emit tool-call intentions as plain text instead of
   structured ``tool_calls`` deltas — the chat page uses this flag to surface
   a one-line warning caption under the model card, so the user can see at a
   glance that "I'll write the file..." replies that don't actually edit are
   the model's fault, not ours. Default-omit the field for every other model:
   the absence of the key means "no warning".
"""
from __future__ import annotations

from typing import Any


MODEL_METADATA: dict[str, dict[str, Any]] = {
    "deepseek-ai/DeepSeek-V3.1": {
        "label": "DeepSeek V3.1",
        "description": "A large hybrid model that supports both thinking and non-thinking modes via prompt templates.",
        "context": "161k",
        "params": "37B-671B (Active-Total)",
        "input_price_per_1m": 0.55,
        "output_price_per_1m": 1.65,
    },
    "deepseek-ai/DeepSeek-V4-Flash": {
        "label": "DeepSeek V4-Flash (experimental)",
        "description": "DeepSeek V4-Flash is an MoE model with 1M context length great for coding, reasoning, and agentic workloads.",
        "context": "1000k",
        "params": "13B-284B (Active-Total)",
        "input_price_per_1m": 0.01,
        "output_price_per_1m": 0.01,
    },
    "google/gemma-4-31B-it": {
        "label": "Google Gemma 4 31B",
        "description": "Gemma 4 31B Dense is designed for advanced reasoning, agentic workflows, and longer context and is natively trained on 140+ languages.",
        "context": "262k",
        "params": "31B (Total)",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 1.25,
    },
    "ibm-granite/granite-4.1-8b": {
        "label": "IBM Granite 4.1 8B",
        "description": "Granite 4.1 8B is a long-context instruct model capable of enhanced tool calling, instruction following, and chat capabilities.",
        "context": "131k",
        "params": "8B (Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.10,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "label": "Meta Llama 3.3 70B",
        "description": "Multilingual model excelling in conversational tasks, detailed instruction-following, and coding.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.71,
        "output_price_per_1m": 0.71,
        # Documented multi-step tool-calling regression on this endpoint
        # (ai-dynamo/dynamo#8732, April 2026): the model returns unfilled
        # template placeholders instead of structured ``tool_calls`` deltas.
        "weak_tool_calling": True,
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "label": "Meta Llama 3.1 70B",
        "description": "Efficient conversational model optimized for responsive multilingual chatbot interactions.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 0.80,
        # Same Llama-3 family failure mode as 3.3-70B above; positioned by
        # W&B as "conversational" rather than agentic.
        "weak_tool_calling": True,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "label": "Meta Llama 3.1 8B",
        "description": "Efficient conversational model optimized for responsive multilingual chatbot interactions.",
        "context": "128k",
        "params": "8B (Total)",
        "input_price_per_1m": 0.22,
        "output_price_per_1m": 0.22,
        # Same family + smallest variant; tool-call hallucinations are
        # particularly common.
        "weak_tool_calling": True,
    },
    "microsoft/Phi-4-mini-instruct": {
        "label": "Microsoft Phi 4 Mini 3.8B",
        "description": "Compact, efficient model ideal for fast responses in resource-constrained environments.",
        "context": "128k",
        "params": "3.8B (Total)",
        "input_price_per_1m": 0.08,
        "output_price_per_1m": 0.35,
        # vLLM tool-parser issue (vllm-project/vllm#14682) where Phi-4 Mini
        # generates tool calls inside ``content`` but the parser returns an
        # empty ``tool_calls`` array; the W&B Inference catalog also doesn't
        # position this model for agentic use.
        "weak_tool_calling": True,
    },
    "MiniMaxAI/MiniMax-M2.5": {
        "label": "MiniMax M2.5",
        "description": "MoE model with a highly sparse architecture designed for high-throughput and low latency with strong coding capabilities.",
        "context": "197k",
        "params": "10B-230B (Active-Total)",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 1.20,
    },
    "moonshotai/Kimi-K2.6": {
        "label": "Moonshot AI Kimi K2.6",
        "description": "Kimi K2.6 is a multimodal Mixture-of-Experts language model featuring 32 billion activated parameters and a total of 1 trillion parameters.",
        "context": "262k",
        "params": "32B-1T (Active-Total)",
        # Not yet listed on the W&B pricing page; fall back to "no cost shown".
        "input_price_per_1m": None,
        "output_price_per_1m": None,
    },
    "moonshotai/Kimi-K2.5": {
        "label": "Moonshot AI Kimi K2.5",
        "description": "Kimi K2.5 is a multimodal Mixture-of-Experts language model featuring 32 billion activated parameters and a total of 1 trillion parameters.",
        "context": "262k",
        "params": "32B-1T (Active-Total)",
        "input_price_per_1m": 0.60,
        "output_price_per_1m": 3.00,
        "cache_hit_price_per_1m": 0.10,
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": {
        "label": "NVIDIA Nemotron 3 Super 120B",
        "description": "Nemotron 3 is a LatentMoE model designed to deliver strong agentic, reasoning, and conversational capabilities.",
        "context": "262k",
        "params": "12B-120B (Active-Total)",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.80,
    },
    "openai/gpt-oss-120b": {
        "label": "OpenAI GPT OSS 120B",
        "description": "Efficient Mixture-of-Experts model designed for high-reasoning, agentic and general-purpose use cases.",
        "context": "131k",
        "params": "5.1B-117B (Active-Total)",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
    },
    "openai/gpt-oss-20b": {
        "label": "OpenAI GPT OSS 20B",
        "description": "Lower latency Mixture-of-Experts model trained on OpenAI's Harmony response format with reasoning capabilities.",
        "context": "131k",
        "params": "3.6B-20B (Active-Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.20,
    },
    "OpenPipe/Qwen3-14B-Instruct": {
        "label": "OpenPipe Qwen3 14B Instruct",
        "description": "An efficient multilingual, dense, instruction-tuned model, optimized by OpenPipe for building agents with finetuning.",
        "context": "32.8k",
        "params": "14.8B (Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.22,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "label": "Qwen3.5 35B A3B",
        "description": "Qwen3.5-35B-A3B is an open-weights multimodal MoE model built for efficient, high-throughput inference across chat, reasoning, and agentic tasks.",
        "context": "262k",
        "params": "3B-35B (Active-Total)",
        "input_price_per_1m": 0.25,
        "output_price_per_1m": 1.25,
    },
    "Qwen/Qwen3-235B-A22B-Thinking-2507": {
        "label": "Qwen3 235B A22B Thinking-2507",
        "description": "High-performance Mixture-of-Experts model optimized for structured reasoning, math, and long-form generation.",
        "context": "262k",
        "params": "22B-235B (Active-Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
    },
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "label": "Qwen3 235B A22B-2507",
        "description": "Efficient multilingual, Mixture-of-Experts, instruction-tuned model, optimized for logical reasoning.",
        "context": "262k",
        "params": "22B-235B (Active-Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "label": "Qwen3 30B A3B",
        "description": "Qwen3-30B-A3B-Instruct-2507 is a 30.5B MoE instruction-tuned model with enhanced reasoning, coding, and long-context understanding.",
        "context": "262k",
        "params": "3.3B-30.5B (Active-Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.30,
    },
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
        "label": "Qwen3 Coder 480B A35B",
        "description": "Mixture-of-Experts model optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning.",
        "context": "262k",
        "params": "35B-480B (Active-Total)",
        "input_price_per_1m": 1.00,
        "output_price_per_1m": 1.50,
    },
    "zai-org/GLM-5.1": {
        "label": "Z.AI GLM 5.1",
        "description": "Powerful MoE model for long-horizon agentic engineering and advanced reasoning.",
        "context": "203k",
        "params": "40B-744B (Active-Total)",
        "input_price_per_1m": 1.40,
        "output_price_per_1m": 4.40,
        "cache_hit_price_per_1m": 0.26,
    },
    "Qwen/Qwen3.5-27B": {
        "label": "Qwen3.5 27B (experimental)",
        "description": "Qwen3.5-27B is a dense model from the Qwen3.5 family built for high performance across a large range of benchmarks.",
        "context": "262k",
        "params": "27B (Total)",
        # Experimental preview; no public pricing yet.
        "input_price_per_1m": None,
        "output_price_per_1m": None,
    },
}


def model_label(model_id: str) -> str:
    """Return the friendly display label for a model id, or its slug fallback.

    Falls back to the trailing slug of the id (the part after the final ``/``)
    when the model is not in :data:`MODEL_METADATA`. The dropdown in
    ``streamlit_app.py`` uses this so live ``/v1/models`` entries we don't
    recognize still get a readable label.
    """
    meta = MODEL_METADATA.get(model_id)
    return meta["label"] if meta else model_id.split("/")[-1]


def has_weak_tool_calling(model_id: str | None) -> bool:
    """Return True when ``model_id`` is documented to mishandle tool calling.

    "Mishandle" here means: the model often emits tool-call intentions as
    plain assistant text without producing the structured ``tool_calls``
    deltas the OpenAI streaming protocol requires, so the agent loop sees
    nothing to dispatch and the user sees a reply like "I'll write the
    file..." that didn't actually edit anything.

    Unknown / experimental ids and ids not present in
    :data:`MODEL_METADATA` return ``False`` (no warning shown). Only flip
    a model's ``weak_tool_calling`` to ``True`` when there's documented
    evidence (vendor docs, an upstream bug report, or a reproducible W&B
    Inference observation); see the module docstring's authoring rules.
    """
    if not model_id:
        return False
    meta = MODEL_METADATA.get(model_id)
    return bool(meta and meta.get("weak_tool_calling"))
