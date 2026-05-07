"""Static, curated metadata for the multi-provider model catalog.

This module is the single source of truth for **opinionated, hand-written
model metadata**: display labels, descriptions, context windows, parameter
counts, capability tags, and (where we have it) per-million-token pricing.
It is read by:

- ``model_catalog.py`` as the highest-priority source in the per-field
  precedence stack. Pricing comes from curated metadata first, then
  from OpenRouter's catalog (only for the ``openrouter:*`` namespace,
  where the user is paying OpenRouter directly so OpenRouter's price
  IS the direct rate). Descriptions come from curated first, then the
  provider's own ``/v1/models`` response (Google's ``description``,
  Anthropic's ``display_name``), then OpenRouter (again, only for
  ``openrouter:*``). When a qualified id has a curated entry here,
  those fields are used verbatim; the catalog never overwrites them.
- ``app_pages/chat.py`` for the inline model-card caption + the picker
  modal's row labels / descriptions / tag membership.
- ``usage.py`` for the per-turn cost compute (when curated pricing is
  set; otherwise the dashboard renders ``-`` for that turn's cost
  because we don't trust any other source for chat-completion pricing).

Qualified-id format
-------------------
Every entry is keyed by ``"<provider_id>:<raw_model_id>"`` where
``provider_id`` is one of :data:`providers.PROVIDERS` (today: ``wandb``,
``openai``, ``anthropic``, ``gemini``, ``openrouter``, ``mistral``,
``xai``) and ``raw_model_id`` is the model id the provider's API
accepts (e.g. ``gpt-4o``, ``claude-3-5-sonnet-20241022``,
``mistral-large-latest``).

Migrating from the old W&B-only schema
--------------------------------------
The previous schema keyed entries by bare model ids (``deepseek-
ai/DeepSeek-V3.1``). All such keys are now ``"wandb:<bare_id>"``. The
chat-file loader (``chats.load_all_chats``) rewrites bare ids on disk
to the new qualified form on the next save (Phase 2 migration).

Tags
----
Each entry carries a ``tags: list[str]`` field used for model-picker
tab membership. Allowed values (hand-curated, never auto-derived for
these three):

- ``"coding"`` — strong on code understanding / generation.
- ``"reasoning"`` — strong on multi-step reasoning / math.
- ``"frontier"`` — current-generation flagship from a major lab.

The ``model_catalog`` layer additionally derives ``"long_context"``
(>=200k context), ``"cheap"`` (<=$0.50/M output), and (for
``openrouter:*`` ids only) ``"multimodal"`` from OpenRouter's
architecture-modality flags.

Pricing
-------
``input_price_per_1m`` / ``output_price_per_1m`` are USD per million
tokens. Both are ``None`` when we don't have a verified curated price;
in that case the model is dropped from the picker by the strict
completeness gate (with the sole exception of ``openrouter:*`` ids,
which pick up pricing from OpenRouter's own catalog because the user
is paying OpenRouter directly).
``cache_hit_price_per_1m`` is optional and currently set only for
models that publish a cache-read price separately (e.g. Anthropic).

Weak tool calling
-----------------
Optional ``weak_tool_calling_issue_url: str`` field — a public bug or
docs URL — for entries documented to mishandle structured tool calling
(emitting tool-call intentions as plain assistant text instead of
``tool_calls`` deltas, which the agent loop has nothing to dispatch on,
producing the "I'll write the file..." replies that don't actually
edit). Presence of the field is what flips the model into the "weak"
bucket. The chat page reads :func:`weak_tool_calling_issue_url` to
render a small orange warning caption under the model card and uses
the same URL as the linked "known issue" inside the warning, so users
can verify the claim on the upstream tracker rather than trust our
wording. We deliberately link rather than recommend specific
alternative models in the caption text — alternative recommendations
age poorly as the catalog rotates, while a public bug URL is durable.

Today the flagged entries are:

- **Every W&B-served Llama 3.x deployment in the catalog** points at
  the shared module constant :data:`LLAMA_3X_TOOL_CALLING_ISSUE_URL`,
  which resolves to ``meta-llama/llama-models#229`` (an open issue
  filed in *Meta's own* repository, framed as a model behavior
  rather than a server-parser bug). The Llama 3.x family weights
  emit tool calls as plain JSON in ``content`` rather than using the
  ``tool_calls`` field, miss the ``<|python_tag|>`` token in
  multi-step flows, and miscall when a system message is present —
  all of which are model-side behaviors that every downstream
  inference server inherits. The same Llama models are also reachable
  via OpenRouter (which dispatches to whichever upstream backend it
  chooses); we don't flag OpenRouter Llama entries individually
  because OpenRouter's catalog doesn't have a stable per-model
  capability schema to extend. If a future provider demonstrates a
  robust parser-side workaround that fully mitigates the symptom,
  drop the field on that single entry — until then, flag.
- ``vllm-project/vllm#14682`` — vLLM tool-parser bug where Phi-4 Mini
  emits tool calls inside ``content`` and the parser returns an empty
  ``tool_calls`` array (set on the ``wandb:microsoft/Phi-4-mini-instruct``
  entry; not extended to other Phi deployments because the catalog
  doesn't list any today).

The bar for setting this field is "this model is *known* to misbehave
and the user can read about it" — vendor docs / upstream tracker /
filed observation. It's a user-facing warning; do not flip it on a
hunch.
"""
from __future__ import annotations

from typing import Any

from providers import PROVIDERS


# ---------------------------------------------------------------------------
# Shared "known issue" URLs for the ``weak_tool_calling_issue_url`` field.
#
# These exist as module constants (rather than inline string literals
# inside each metadata row) so a single edit propagates across every
# affected entry, and so it's obvious from reading the catalog that
# multiple rows share the same upstream cause. See the module
# docstring's "Weak tool calling" section for the policy on flagging.
# ---------------------------------------------------------------------------

# Meta's own tracker for the Llama 3.x family's tool-calling regression:
# the model frequently emits tool calls as plain JSON in ``content``
# instead of structured ``tool_calls`` deltas, drops the ``<|python_tag|>``
# token mid-loop, and miscalls when a system message is present. The
# behavior is rooted in the model weights, so every Llama 3.x deployment
# in the catalog (regardless of which provider serves it) inherits it
# until that provider proves a parser-side workaround. Cited on every
# Llama 3.x ``MODEL_METADATA`` entry below.
LLAMA_3X_TOOL_CALLING_ISSUE_URL = "https://github.com/meta-llama/llama-models/issues/229"

# vLLM tool-parser bug specific to Microsoft Phi-4 Mini: the model
# generates tool calls inside ``content`` but vLLM's parser returns
# an empty ``tool_calls`` array, surfacing to us as the same
# "describes-but-doesn't-edit" symptom. Only cited on the W&B-served
# Phi-4 Mini today (the catalog doesn't list any other Phi deployment).
PHI_4_MINI_TOOL_CALLING_ISSUE_URL = "https://github.com/vllm-project/vllm/issues/14682"


# ---------------------------------------------------------------------------
# Curated model metadata, keyed by qualified id ``<provider>:<raw_id>``.
#
# The keys here form the long tail of the picker's "Recommended" /
# "Coding" / "Reasoning" / "Frontier" tabs. The auto-tag layer in
# ``model_catalog`` will surface additional models the user has
# access to even when they're not curated here, but the curated rows
# are the ones we put our quality stamp on.
# ---------------------------------------------------------------------------
MODEL_METADATA: dict[str, dict[str, Any]] = {
    # =====================================================================
    # W&B Inference (provider_id = "wandb")
    # =====================================================================
    "wandb:deepseek-ai/DeepSeek-V3.1": {
        "provider_id": "wandb",
        "label": "DeepSeek V3.1",
        "description": "A large hybrid model that supports both thinking and non-thinking modes via prompt templates.",
        "context": "161k",
        "params": "37B-671B (Active-Total)",
        "input_price_per_1m": 0.55,
        "output_price_per_1m": 1.65,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "wandb:deepseek-ai/DeepSeek-V4-Flash": {
        "provider_id": "wandb",
        "label": "DeepSeek V4-Flash (experimental)",
        "description": "DeepSeek V4-Flash is an MoE model with 1M context length great for coding, reasoning, and agentic workloads.",
        "context": "1000k",
        "params": "13B-284B (Active-Total)",
        "input_price_per_1m": 0.01,
        "output_price_per_1m": 0.01,
        "tags": ["coding", "reasoning"],
    },
    "wandb:google/gemma-4-31B-it": {
        "provider_id": "wandb",
        "label": "Google Gemma 4 31B",
        "description": "Gemma 4 31B Dense is designed for advanced reasoning, agentic workflows, and longer context and is natively trained on 140+ languages.",
        "context": "262k",
        "params": "31B (Total)",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 1.25,
        "tags": ["reasoning"],
    },
    "wandb:ibm-granite/granite-4.1-8b": {
        "provider_id": "wandb",
        "label": "IBM Granite 4.1 8B",
        "description": "Granite 4.1 8B is a long-context instruct model capable of enhanced tool calling, instruction following, and chat capabilities.",
        "context": "131k",
        "params": "8B (Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.10,
        "tags": [],
    },
    "wandb:meta-llama/Llama-3.3-70B-Instruct": {
        "provider_id": "wandb",
        "label": "Meta Llama 3.3 70B",
        "description": "Multilingual model excelling in conversational tasks, detailed instruction-following, and coding.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.71,
        "output_price_per_1m": 0.71,
        "tags": ["coding"],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "wandb:meta-llama/Llama-3.1-70B-Instruct": {
        "provider_id": "wandb",
        "label": "Meta Llama 3.1 70B",
        "description": "Efficient conversational model optimized for responsive multilingual chatbot interactions.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 0.80,
        "tags": [],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "wandb:meta-llama/Llama-3.1-8B-Instruct": {
        "provider_id": "wandb",
        "label": "Meta Llama 3.1 8B",
        "description": "Efficient conversational model optimized for responsive multilingual chatbot interactions.",
        "context": "128k",
        "params": "8B (Total)",
        "input_price_per_1m": 0.22,
        "output_price_per_1m": 0.22,
        "tags": [],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "wandb:microsoft/Phi-4-mini-instruct": {
        "provider_id": "wandb",
        "label": "Microsoft Phi 4 Mini 3.8B",
        "description": "Compact, efficient model ideal for fast responses in resource-constrained environments.",
        "context": "128k",
        "params": "3.8B (Total)",
        "input_price_per_1m": 0.08,
        "output_price_per_1m": 0.35,
        "tags": [],
        "weak_tool_calling_issue_url": PHI_4_MINI_TOOL_CALLING_ISSUE_URL,
    },
    "wandb:MiniMaxAI/MiniMax-M2.5": {
        "provider_id": "wandb",
        "label": "MiniMax M2.5",
        "description": "MoE model with a highly sparse architecture designed for high-throughput and low latency with strong coding capabilities.",
        "context": "197k",
        "params": "10B-230B (Active-Total)",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 1.20,
        "tags": ["coding"],
    },
    "wandb:moonshotai/Kimi-K2.6": {
        "provider_id": "wandb",
        "label": "Moonshot AI Kimi K2.6",
        "description": "Kimi K2.6 is a multimodal Mixture-of-Experts language model featuring 32 billion activated parameters and a total of 1 trillion parameters.",
        "context": "262k",
        "params": "32B-1T (Active-Total)",
        "input_price_per_1m": None,
        "output_price_per_1m": None,
        "tags": ["frontier"],
    },
    "wandb:moonshotai/Kimi-K2.5": {
        "provider_id": "wandb",
        "label": "Moonshot AI Kimi K2.5",
        "description": "Kimi K2.5 is a multimodal Mixture-of-Experts language model featuring 32 billion activated parameters and a total of 1 trillion parameters.",
        "context": "262k",
        "params": "32B-1T (Active-Total)",
        "input_price_per_1m": 0.60,
        "output_price_per_1m": 3.00,
        "cache_hit_price_per_1m": 0.10,
        "tags": ["frontier"],
    },
    "wandb:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": {
        "provider_id": "wandb",
        "label": "NVIDIA Nemotron 3 Super 120B",
        "description": "Nemotron 3 is a LatentMoE model designed to deliver strong agentic, reasoning, and conversational capabilities.",
        "context": "262k",
        "params": "12B-120B (Active-Total)",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.80,
        "tags": ["reasoning"],
    },
    "wandb:openai/gpt-oss-120b": {
        "provider_id": "wandb",
        "label": "OpenAI GPT OSS 120B",
        "description": "Efficient Mixture-of-Experts model designed for high-reasoning, agentic and general-purpose use cases.",
        "context": "131k",
        "params": "5.1B-117B (Active-Total)",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
        "tags": ["reasoning"],
    },
    "wandb:openai/gpt-oss-20b": {
        "provider_id": "wandb",
        "label": "OpenAI GPT OSS 20B",
        "description": "Lower latency Mixture-of-Experts model trained on OpenAI's Harmony response format with reasoning capabilities.",
        "context": "131k",
        "params": "3.6B-20B (Active-Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.20,
        "tags": ["reasoning"],
    },
    "wandb:OpenPipe/Qwen3-14B-Instruct": {
        "provider_id": "wandb",
        "label": "OpenPipe Qwen3 14B Instruct",
        "description": "An efficient multilingual, dense, instruction-tuned model, optimized by OpenPipe for building agents with finetuning.",
        "context": "32.8k",
        "params": "14.8B (Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.22,
        "tags": [],
    },
    "wandb:Qwen/Qwen3.5-35B-A3B": {
        "provider_id": "wandb",
        "label": "Qwen3.5 35B A3B",
        "description": "Qwen3.5-35B-A3B is an open-weights multimodal MoE model built for efficient, high-throughput inference across chat, reasoning, and agentic tasks.",
        "context": "262k",
        "params": "3B-35B (Active-Total)",
        "input_price_per_1m": 0.25,
        "output_price_per_1m": 1.25,
        "tags": ["reasoning"],
    },
    "wandb:Qwen/Qwen3-235B-A22B-Thinking-2507": {
        "provider_id": "wandb",
        "label": "Qwen3 235B A22B Thinking-2507",
        "description": "High-performance Mixture-of-Experts model optimized for structured reasoning, math, and long-form generation.",
        "context": "262k",
        "params": "22B-235B (Active-Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
        "tags": ["reasoning"],
    },
    "wandb:Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "provider_id": "wandb",
        "label": "Qwen3 235B A22B-2507",
        "description": "Efficient multilingual, Mixture-of-Experts, instruction-tuned model, optimized for logical reasoning.",
        "context": "262k",
        "params": "22B-235B (Active-Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
        "tags": ["reasoning"],
    },
    "wandb:Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "provider_id": "wandb",
        "label": "Qwen3 30B A3B",
        "description": "Qwen3-30B-A3B-Instruct-2507 is a 30.5B MoE instruction-tuned model with enhanced reasoning, coding, and long-context understanding.",
        "context": "262k",
        "params": "3.3B-30.5B (Active-Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.30,
        "tags": ["reasoning", "coding"],
    },
    "wandb:Qwen/Qwen3-Coder-480B-A35B-Instruct": {
        "provider_id": "wandb",
        "label": "Qwen3 Coder 480B A35B",
        "description": "Mixture-of-Experts model optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning.",
        "context": "262k",
        "params": "35B-480B (Active-Total)",
        "input_price_per_1m": 1.00,
        "output_price_per_1m": 1.50,
        "tags": ["coding"],
    },
    "wandb:zai-org/GLM-5.1": {
        "provider_id": "wandb",
        "label": "Z.AI GLM 5.1",
        "description": "Powerful MoE model for long-horizon agentic engineering and advanced reasoning.",
        "context": "203k",
        "params": "40B-744B (Active-Total)",
        "input_price_per_1m": 1.40,
        "output_price_per_1m": 4.40,
        "cache_hit_price_per_1m": 0.26,
        "tags": ["coding", "reasoning"],
    },
    "wandb:Qwen/Qwen3.5-27B": {
        "provider_id": "wandb",
        "label": "Qwen3.5 27B (experimental)",
        "description": "Qwen3.5-27B is a dense model from the Qwen3.5 family built for high performance across a large range of benchmarks.",
        "context": "262k",
        "params": "27B (Total)",
        "input_price_per_1m": None,
        "output_price_per_1m": None,
        "tags": [],
    },

    # =====================================================================
    # OpenAI (provider_id = "openai")
    #
    # Pricing transcribed from openai.com/api/pricing (Feb 2026 snapshot).
    # The ``cache_hit_price_per_1m`` matches OpenAI's published cached-
    # input rate (typically 50% of the base input price for the o-series,
    # and 50% for GPT-4o).
    # =====================================================================
    "openai:o3": {
        "provider_id": "openai",
        "label": "o3",
        "description": "OpenAI's flagship reasoning model. Strong chain-of-thought before answering; excels at coding, math, and multi-step agentic loops. Supports adjustable reasoning_effort (low / medium / high).",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 8.00,
        "cache_hit_price_per_1m": 0.50,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "openai:o3-pro": {
        "provider_id": "openai",
        "label": "o3-pro",
        "description": "Premium-tier o3 with extended reasoning depth for the hardest problems. Significantly more expensive than o3; reserve for tasks where the extra thinking pays off.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 20.00,
        "output_price_per_1m": 80.00,
        "tags": ["frontier", "reasoning"],
    },
    "openai:o3-mini": {
        "provider_id": "openai",
        "label": "o3-mini",
        "description": "Compact reasoning model with adjustable reasoning_effort. Excellent value for coding, agentic workflows, and math where deep reasoning matters more than world knowledge.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 1.10,
        "output_price_per_1m": 4.40,
        "cache_hit_price_per_1m": 0.55,
        "tags": ["reasoning", "coding"],
    },
    "openai:o1": {
        "provider_id": "openai",
        "label": "o1",
        "description": "OpenAI's first-generation reasoning model. Still preferred by some workflows that pre-date o3, but the o3 family is faster and cheaper for most tasks.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 15.00,
        "output_price_per_1m": 60.00,
        "cache_hit_price_per_1m": 7.50,
        "tags": ["reasoning"],
    },
    "openai:gpt-4o": {
        "provider_id": "openai",
        "label": "GPT-4o",
        "description": "OpenAI's flagship multimodal chat model with vision and audio understanding. Strong general-purpose reasoning and coding with broad tool support; the default pick when you don't need o-series chain-of-thought.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 2.50,
        "output_price_per_1m": 10.00,
        "cache_hit_price_per_1m": 1.25,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "openai:gpt-4o-mini": {
        "provider_id": "openai",
        "label": "GPT-4o mini",
        "description": "Compact, low-latency multimodal model. Excellent value for everyday chat, simple coding, and high-throughput agentic loops where cost dominates.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
        "tags": ["coding"],
    },

    # =====================================================================
    # Anthropic (provider_id = "anthropic")
    #
    # Pricing transcribed from docs.anthropic.com/en/about-claude/pricing
    # (April 2026 snapshot). Cache-hit prices reflect the 0.1x multiplier
    # over base input price (the standard 5-minute cache rate). Claude
    # Opus 4.7 / 4.6 / 4.5 dropped pricing relative to Opus 4.1 / 4
    # ($5/$25 vs $15/$75) — the 4.x flagships are the value play.
    # =====================================================================
    "anthropic:claude-opus-4-7": {
        "provider_id": "anthropic",
        "label": "Claude Opus 4.7",
        "description": "Anthropic's flagship Claude. Best-in-class for the hardest reasoning, analysis, and long-form coding. New tokenizer (~35% more tokens for the same fixed text but better quality). Supports prompt caching for ~90% repeat-cost savings.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 5.00,
        "output_price_per_1m": 25.00,
        "cache_hit_price_per_1m": 0.50,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "anthropic:claude-sonnet-4-6": {
        "provider_id": "anthropic",
        "label": "Claude Sonnet 4.6",
        "description": "Anthropic's flagship Sonnet — the best balance of intelligence and speed for an agentic coding workflow. 1M-token context window. Prompt caching is the killer feature: ~90% cost cut on every turn after the first when the system prompt is cached.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "cache_hit_price_per_1m": 0.30,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "anthropic:claude-haiku-4-5": {
        "provider_id": "anthropic",
        "label": "Claude Haiku 4.5",
        "description": "Fastest current-generation Claude. Great for high-volume agentic loops where Haiku-class speed matters and full Sonnet isn't needed; still benefits from prompt caching.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 1.00,
        "output_price_per_1m": 5.00,
        "cache_hit_price_per_1m": 0.10,
        "tags": ["coding"],
    },
    "anthropic:claude-opus-4-1": {
        "provider_id": "anthropic",
        "label": "Claude Opus 4.1 (legacy)",
        "description": "Previous-generation Opus. Identical capability tier to Opus 4 but at the old $15/$75 pricing — most users will prefer the much cheaper Opus 4.7.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 15.00,
        "output_price_per_1m": 75.00,
        "cache_hit_price_per_1m": 1.50,
        "tags": ["reasoning"],
    },
    "anthropic:claude-3-5-sonnet-20241022": {
        "provider_id": "anthropic",
        "label": "Claude 3.5 Sonnet (legacy)",
        "description": "Older-generation Sonnet from October 2024. Kept for workflows pinned to a specific dated id; new chats should use Sonnet 4.6.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "cache_hit_price_per_1m": 0.30,
        "tags": ["coding"],
    },
    "anthropic:claude-3-5-haiku-20241022": {
        "provider_id": "anthropic",
        "label": "Claude 3.5 Haiku (legacy)",
        "description": "Older-generation Haiku from October 2024. Slightly cheaper input ($0.80 vs $1.00) than Haiku 4.5 but lower quality; kept for workflows pinned to a specific dated id.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 4.00,
        "cache_hit_price_per_1m": 0.08,
        "tags": ["coding"],
    },

    # =====================================================================
    # Google Gemini (provider_id = "gemini")
    #
    # Pricing transcribed from ai.google.dev/gemini-api/docs/pricing
    # (Standard tier, 2026 snapshot). Several Gemini models tier
    # pricing by prompt size (<=200k vs >200k); the curated price here
    # is the <=200k rate (the common case for an agent loop). Caches
    # use Google's "context caching" feature; the cache rate matches
    # the published 0.1x multiplier on standard input.
    # =====================================================================
    "gemini:gemini-3.1-pro-preview": {
        "provider_id": "gemini",
        "label": "Gemini 3.1 Pro",
        "description": "Google's flagship 3.x Pro — best-in-class for multimodal understanding, agentic capabilities, and vibe-coding. Massive context window. Native grounding with Google Search; supports context caching for repeat-cost savings.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 12.00,
        "cache_hit_price_per_1m": 0.20,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "gemini:gemini-3-flash-preview": {
        "provider_id": "gemini",
        "label": "Gemini 3 Flash",
        "description": "Most intelligent Gemini built for speed. Combines frontier intelligence with superior search and grounding — strong throughput for multimodal Q&A and agentic loops at a fraction of Pro pricing.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 3.00,
        "cache_hit_price_per_1m": 0.05,
        "tags": ["frontier", "coding"],
    },
    "gemini:gemini-3.1-flash-lite-preview": {
        "provider_id": "gemini",
        "label": "Gemini 3.1 Flash-Lite",
        "description": "Most cost-efficient Gemini, optimized for high-volume agentic tasks, translation, and simple data processing. Preview status — pricing and availability may shift before stable release.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.25,
        "output_price_per_1m": 1.50,
        "cache_hit_price_per_1m": 0.025,
        "tags": ["coding"],
    },
    "gemini:gemini-2.5-pro": {
        "provider_id": "gemini",
        "label": "Gemini 2.5 Pro",
        "description": "Stable-generation Gemini Pro with strong multipurpose reasoning and coding. Native grounding with Google Search and code execution available via the SDK.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
        "cache_hit_price_per_1m": 0.125,
        "tags": ["reasoning", "coding"],
    },
    "gemini:gemini-2.5-flash": {
        "provider_id": "gemini",
        "label": "Gemini 2.5 Flash",
        "description": "Stable hybrid reasoning Flash with thinking budgets and a 1M-token context. Solid value for everyday chat, multimodal Q&A, and high-volume agentic loops.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
        "cache_hit_price_per_1m": 0.03,
        "tags": ["coding"],
    },
    "gemini:gemini-2.5-flash-lite": {
        "provider_id": "gemini",
        "label": "Gemini 2.5 Flash-Lite",
        "description": "Smallest and most cost-effective stable Gemini, built for at-scale usage. Lower quality than Flash but unbeatable on per-token cost for simple chat / classification / extraction.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
        "cache_hit_price_per_1m": 0.01,
        "tags": [],
    },
    "gemini:gemini-2.0-flash": {
        "provider_id": "gemini",
        "label": "Gemini 2.0 Flash",
        "description": "Previous-generation Flash with multimodal input + audio output support. Mostly superseded by 2.5 Flash but kept for workflows that depend on its specific behavior.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
        "tags": [],
    },

    # =====================================================================
    # Mistral (provider_id = "mistral")
    #
    # Pricing transcribed from docs.mistral.ai (2026 lineup). The
    # December 2025 Large 3 release dropped flagship pricing 4x vs
    # the previous Large 2 ($0.50/$1.50 vs $2.00/$6.00). Magistral is
    # Mistral's reasoning family (similar role to o-series for OpenAI
    # or thinking models for Anthropic / Google).
    # =====================================================================
    "mistral:mistral-large-latest": {
        "provider_id": "mistral",
        "label": "Mistral Large 3",
        "description": "Mistral's flagship dense model. Strong reasoning + multilingual generation. The December 2025 release brought a 4x price cut vs Large 2 — now strongly competitive on cost.",
        "context": "256k",
        "params": "—",
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 1.50,
        "tags": ["frontier", "reasoning"],
    },
    "mistral:magistral-medium-latest": {
        "provider_id": "mistral",
        "label": "Magistral Medium",
        "description": "Mistral's frontier-class reasoning model — chain-of-thought before answering, similar to o-series. Best Mistral pick for hard coding, math, and multi-step agentic loops.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 5.00,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "mistral:magistral-small-latest": {
        "provider_id": "mistral",
        "label": "Magistral Small",
        "description": "Lightweight reasoning model from Mistral. Strong on domain-specific tasks where cost matters but you still want chain-of-thought.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 1.50,
        "tags": ["reasoning"],
    },
    "mistral:mistral-medium-3": {
        "provider_id": "mistral",
        "label": "Mistral Medium 3",
        "description": "Balanced-tier dense Mistral. ~90% of frontier performance at ~20% of frontier cost; a strong default pick when Large 3 is overkill.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 0.40,
        "output_price_per_1m": 2.00,
        "tags": ["coding"],
    },
    "mistral:codestral-latest": {
        "provider_id": "mistral",
        "label": "Codestral",
        "description": "Mistral's code-specialized model. Strong fill-in-the-middle and instruct-following on code; the natural pick from Mistral when the workload is code-heavy.",
        "context": "262k",
        "params": "22B (Total)",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 0.90,
        "tags": ["coding"],
    },
    "mistral:mistral-small-latest": {
        "provider_id": "mistral",
        "label": "Mistral Small 3.2",
        "description": "Mistral's most cost-efficient model. Sub-$0.10/M input — great for high-volume classification, extraction, and simple chat where Magistral or Medium would be wasted.",
        "context": "32k",
        "params": "—",
        "input_price_per_1m": 0.075,
        "output_price_per_1m": 0.20,
        "tags": [],
    },

    # =====================================================================
    # xAI (provider_id = "xai")
    #
    # Pricing transcribed from docs.x.ai/models (2026 lineup). The
    # Grok 4 family supplanted the Grok 2 series; the older Grok 2
    # entry is dropped because xAI has retired most of those endpoints.
    # Live Search is a native xAI feature (real-time web access at
    # query time) — not yet wired through this app's chat flow.
    # =====================================================================
    "xai:grok-4": {
        "provider_id": "xai",
        "label": "Grok 4",
        "description": "xAI's flagship Grok 4. Competitive on reasoning and conversational quality with function calling, structured outputs, and native Live Search (real-time web access).",
        "context": "256k",
        "params": "—",
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "tags": ["frontier", "reasoning"],
    },
    "xai:grok-4-fast": {
        "provider_id": "xai",
        "label": "Grok 4 Fast",
        "description": "Cost-efficient Grok 4 variant with a 2M-token context. Excellent for long-document analysis and high-volume agentic loops where the full Grok 4 price isn't justified.",
        "context": "2000k",
        "params": "—",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.50,
        "cache_hit_price_per_1m": 0.05,
        "tags": ["coding"],
    },

}


# ---------------------------------------------------------------------------
# Recommended models — anchor the picker's "Recommended" tab.
#
# Hand-curated, opinionated short list. Order matters: the picker
# renders these in this order before any auto-included models. When a
# user has not connected the provider for a given entry, that row is
# silently skipped (the model_catalog filter does that intersection).
# ---------------------------------------------------------------------------
RECOMMENDED_MODELS: list[str] = [
    "anthropic:claude-sonnet-4-6",
    "openai:o3",
    "gemini:gemini-3.1-pro-preview",
    "openai:gpt-4o",
    "anthropic:claude-opus-4-7",
    "wandb:deepseek-ai/DeepSeek-V3.1",
    "wandb:Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "anthropic:claude-haiku-4-5",
    "gemini:gemini-3-flash-preview",
    "openai:gpt-4o-mini",
    "openai:o3-mini",
]


# Allowed tag values. Kept here as a constant so model_catalog and the
# UI can reference them without typo risk.
#
# Curated-only tags (set hand-by-hand in MODEL_METADATA — never
# auto-derived): ``coding`` / ``reasoning`` / ``frontier``.
#
# Auto-derived tags (computed inside ``model_catalog._merge_one``
# from provider attributes + per-model curated capability flags +
# OpenRouter's architecture-modality arrays for ``openrouter:*``
# ids — overlay onto curated tags):
# - ``long_context`` — context >= 200k tokens.
# - ``cheap`` — output_price <= $0.50/M tokens.
# - ``multimodal`` — supports image input (legacy alias of ``vision``;
#   kept for back-compat with curated entries).
# - ``vision`` — chat model that accepts image input. Picker tab.
# - ``image_gen`` — model that GENERATES images. Picker tab.
# - ``audio_gen`` — model that GENERATES audio (TTS). Picker tab.
# - ``audio_in`` — model that accepts audio input but doesn't gen.
# - ``video_gen`` — model that GENERATES video. Picker tab.
# - ``video_in`` — model that accepts video input but doesn't gen.
# Without a per-provider machine-readable signal for the modality
# flags, the auto-derivation only fires for ``openrouter:*`` ids
# (where OpenRouter's catalog provides architecture data) and for
# any curated entry that explicitly sets ``mode`` or carries the
# matching tag. Other providers' modality tabs depend on curation.
ALLOWED_TAGS: tuple[str, ...] = (
    "coding",
    "reasoning",
    "long_context",
    "cheap",
    "frontier",
    "multimodal",
    "vision",
    "image_gen",
    "audio_gen",
    "audio_in",
    "video_gen",
    "video_in",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_qualified(model_id: str) -> bool:
    """Return True if ``model_id`` is in qualified ``provider:raw`` form.

    Heuristic: there's a ``:`` *before* any ``/`` in the string. Bare W&B
    ids like ``deepseek-ai/DeepSeek-V3.1`` slash-separate
    ``<owner>/<model>`` and have no colon, so they fail this check and
    get migrated by ``chats.load_all_chats`` on the next save.
    """
    if not isinstance(model_id, str) or not model_id:
        return False
    colon = model_id.find(":")
    slash = model_id.find("/")
    if colon == -1:
        return False
    if slash == -1:
        return True
    return colon < slash


def qualify(model_id: str, default_provider: str = "wandb") -> str:
    """Return ``model_id`` in qualified form, prefixing ``<default_provider>:`` if needed.

    Used by the chat-file loader during one-time migration of bare
    ids. New chats always store the qualified form directly.
    """
    if is_qualified(model_id):
        return model_id
    if not model_id:
        return model_id
    return f"{default_provider}:{model_id}"


def model_provider(qualified_id: str) -> str | None:
    """Return the ``provider_id`` for ``qualified_id`` or ``None`` on a bare id."""
    if not is_qualified(qualified_id):
        return None
    return qualified_id.split(":", 1)[0]


def unqualified(qualified_id: str) -> str:
    """Return the raw model id (everything after the first ``:``).

    For bare ids (no colon, e.g. legacy ``deepseek-ai/...``), returns
    the input verbatim.
    """
    if not is_qualified(qualified_id):
        return qualified_id
    return qualified_id.split(":", 1)[1]


def model_label(qualified_id: str) -> str:
    """Return the friendly display label for a qualified id, or a slug fallback.

    Falls back to the trailing slug of the raw id (the part after the
    final ``/``) when the model is not in :data:`MODEL_METADATA`. The
    picker uses this so live ``/v1/models`` entries we don't recognize
    still get a readable label.

    Bare (un-qualified) ids are tolerated for back-compat — a chat
    persisted before the qualified-id migration that passes through
    here before ``chats.load_all_chats`` rewrites it just gets the
    legacy slug treatment.
    """
    meta = MODEL_METADATA.get(qualified_id)
    if meta:
        return meta["label"]
    raw = unqualified(qualified_id)
    return raw.split("/")[-1] if raw else qualified_id


def models_with_tag(
    tag: str,
    *,
    available: set[str] | list[str] | None = None,
) -> list[str]:
    """Return curated qualified ids tagged with ``tag``.

    When ``available`` is provided, the result is intersected with it so
    callers can render "models that are both tagged X and reachable on
    a connected provider". The intersection is order-preserving with
    respect to :data:`MODEL_METADATA`'s insertion order, which is
    intentional: curated models surface in the order we listed them
    (W&B first, then OpenAI, etc.).

    The tag list itself is curated-only (per AGENTS.md): no auto-derived
    membership for ``coding`` / ``reasoning`` / ``frontier`` is checked
    here. The catalog layer is responsible for surfacing models with
    auto-derived tags (``long_context`` / ``cheap`` / ``fast`` /
    ``multimodal``) to the picker via its own membership API.
    """
    available_set = set(available) if available is not None else None
    out: list[str] = []
    for qid, meta in MODEL_METADATA.items():
        tags = meta.get("tags") or []
        if tag in tags:
            if available_set is None or qid in available_set:
                out.append(qid)
    return out


def is_known_provider(qualified_id: str) -> bool:
    """Return True if the provider half of ``qualified_id`` is in :data:`PROVIDERS`."""
    pid = model_provider(qualified_id)
    return pid is not None and pid in PROVIDERS


def weak_tool_calling_issue_url(qualified_id: str | None) -> str | None:
    """Return the public "known issue" URL for a model, or ``None``.

    The URL points to the bug / docs page that documents the model's
    failure to emit structured ``tool_calls`` deltas (the OpenAI
    streaming protocol's only channel for invoking tools), which
    manifests in the chat UI as "I'll write the file..." replies that
    produce no actual edit. Today these point at upstream issues on
    ``ai-dynamo/dynamo`` (W&B-served Llama 3.x) and ``vllm-project/vllm``
    (W&B-served Phi-4 Mini); see :data:`MODEL_METADATA` for the
    per-entry rows and the module docstring for the authoring rules.

    Tolerates bare (un-qualified) ids for back-compat with chats
    persisted before the qualified-id migration: a bare lookup misses
    on :data:`MODEL_METADATA` and we return ``None`` (which is the
    correct "no warning" answer — once the chat is saved, the
    persistence layer rewrites the id to qualified form and the next
    render picks up the warning naturally).

    The chat page renders the warning caption iff this returns
    non-None and uses the URL as the linked "known issue" inside the
    caption — that way users can verify the claim themselves rather
    than trust our wording, and the caption text never has to mention
    specific alternative model names that age poorly.
    """
    if not qualified_id:
        return None
    meta = MODEL_METADATA.get(qualified_id)
    if not meta:
        return None
    url = meta.get("weak_tool_calling_issue_url")
    return str(url) if isinstance(url, str) and url else None


def has_weak_tool_calling(qualified_id: str | None) -> bool:
    """Return True when ``qualified_id`` is documented to mishandle tool calling.

    Thin convenience wrapper over :func:`weak_tool_calling_issue_url`:
    the URL field is the single source of truth, and "flagged" is just
    "we have a public bug to point at." Callers that need the URL
    itself (e.g. to embed in a markdown link) should call
    :func:`weak_tool_calling_issue_url` directly to avoid the second
    dict lookup.
    """
    return weak_tool_calling_issue_url(qualified_id) is not None
