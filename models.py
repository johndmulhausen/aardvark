"""Static, curated metadata for the multi-provider model catalog.

This module is the single source of truth for **opinionated, hand-written
model metadata**: display labels, descriptions, context windows, parameter
counts, capability tags, and (where we have it) per-million-token pricing.
It is read by:

- ``model_catalog.py`` as the highest-priority source in the per-field
  precedence stack (curated > LiteLLM > OpenRouter > live API). When a
  qualified id has a curated entry here, those fields are used verbatim;
  the catalog never overwrites them.
- ``app_pages/chat.py`` for the inline model-card caption + the picker
  modal's row labels / descriptions / tag membership.
- ``usage.py`` for the per-turn cost compute (when curated pricing is
  set; otherwise it falls back to LiteLLM via the catalog).

Qualified-id format
-------------------
Every entry is keyed by ``"<provider_id>:<raw_model_id>"`` where
``provider_id`` is one of :data:`providers.PROVIDERS` (today: ``wandb``,
``openai``, ``anthropic``, ``gemini``, ``together``, ``groq``,
``fireworks``, ``mistral``, ``xai``, ``cerebras``, ``deepinfra``,
``openrouter``) and ``raw_model_id`` is the model id the provider's API
accepts (e.g. ``gpt-4o``, ``claude-3-5-sonnet-20241022``,
``meta-llama/Llama-3.3-70B-Instruct-Turbo``).

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
(>=200k context), ``"cheap"`` (<=$0.50/M output), ``"fast"`` (Groq /
Cerebras latency tier), and ``"multimodal"`` (LiteLLM
``supports_vision``) automatically when a curated ``tags`` list is
absent.

Pricing
-------
``input_price_per_1m`` / ``output_price_per_1m`` are USD per million
tokens. Both are ``None`` when we don't have a verified curated price;
in that case the catalog merge layer falls back to LiteLLM's
``model_prices_and_context_window.json`` for the same qualified id.
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

- **Every Llama 3.x deployment in the catalog** — across W&B,
  Together, Groq, Fireworks, Cerebras, and DeepInfra — points at the
  shared module constant :data:`LLAMA_3X_TOOL_CALLING_ISSUE_URL`,
  which resolves to ``meta-llama/llama-models#229`` (an open issue
  filed in *Meta's own* repository, framed as a model behavior
  rather than a server-parser bug). The Llama 3.x family weights
  emit tool calls as plain JSON in ``content`` rather than using the
  ``tool_calls`` field, miss the ``<|python_tag|>`` token in
  multi-step flows, and miscall when a system message is present —
  all of which are model-side behaviors that every downstream
  inference server inherits. Linking *all* Llama 3.x entries to the
  same Meta-tracker URL (rather than to a per-provider bug) is the
  most honest framing: it's a Llama family limitation, not a
  W&B-only quirk. If a future provider demonstrates a robust
  parser-side workaround that fully mitigates the symptom, drop the
  field on that single entry — until then, flag.
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
    # =====================================================================
    "openai:gpt-4o": {
        "provider_id": "openai",
        "label": "GPT-4o",
        "description": "OpenAI's flagship multimodal model with vision and audio understanding. Strong general-purpose reasoning and coding with broad tool support.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 2.50,
        "output_price_per_1m": 10.00,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "openai:gpt-4o-mini": {
        "provider_id": "openai",
        "label": "GPT-4o mini",
        "description": "Compact, low-latency multimodal model. Excellent value for everyday chat, simple coding, and high-throughput workloads.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
        "tags": ["coding"],
    },
    "openai:o1": {
        "provider_id": "openai",
        "label": "o1",
        "description": "OpenAI's first reasoning model — uses chain-of-thought before answering. Strong on math, science, and complex coding tasks.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 15.00,
        "output_price_per_1m": 60.00,
        "tags": ["frontier", "reasoning"],
    },
    "openai:o1-mini": {
        "provider_id": "openai",
        "label": "o1-mini",
        "description": "Smaller, cheaper, faster reasoning model. Strong on coding and STEM problems where deep reasoning matters more than world knowledge.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 1.10,
        "output_price_per_1m": 4.40,
        "tags": ["reasoning", "coding"],
    },
    "openai:o3-mini": {
        "provider_id": "openai",
        "label": "o3-mini",
        "description": "Next-gen compact reasoning model with adjustable reasoning_effort. Excellent for coding, agentic workflows, and math.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 1.10,
        "output_price_per_1m": 4.40,
        "tags": ["reasoning", "coding"],
    },
    "openai:gpt-4-turbo": {
        "provider_id": "openai",
        "label": "GPT-4 Turbo",
        "description": "Previous-generation GPT-4 model with vision support. Solid at long-form generation and coding when GPT-4o isn't available.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 10.00,
        "output_price_per_1m": 30.00,
        "tags": ["coding"],
    },

    # =====================================================================
    # Anthropic (provider_id = "anthropic")
    # =====================================================================
    "anthropic:claude-3-5-sonnet-20241022": {
        "provider_id": "anthropic",
        "label": "Claude 3.5 Sonnet",
        "description": "Anthropic's flagship balance of intelligence and speed. Strong on agentic coding, long-form reasoning, and tool use. Supports prompt caching for ~90% repeat-cost savings.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "cache_hit_price_per_1m": 0.30,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "anthropic:claude-3-5-haiku-20241022": {
        "provider_id": "anthropic",
        "label": "Claude 3.5 Haiku",
        "description": "Fast, lightweight Claude model. Great for high-volume tasks where Haiku-class speed matters and full Sonnet isn't needed.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 4.00,
        "cache_hit_price_per_1m": 0.08,
        "tags": ["coding"],
    },
    "anthropic:claude-3-opus-20240229": {
        "provider_id": "anthropic",
        "label": "Claude 3 Opus",
        "description": "Anthropic's previous flagship. Premium tier for the hardest reasoning, analysis, and writing tasks. Still preferred by some workflows that pre-date Sonnet 3.5.",
        "context": "200k",
        "params": "—",
        "input_price_per_1m": 15.00,
        "output_price_per_1m": 75.00,
        "tags": ["frontier", "reasoning"],
    },

    # =====================================================================
    # Google Gemini (provider_id = "gemini")
    # =====================================================================
    "gemini:gemini-2.5-pro": {
        "provider_id": "gemini",
        "label": "Gemini 2.5 Pro",
        "description": "Google's flagship multimodal model with strong reasoning and a massive context window. Native grounding with Google Search and code execution available via the SDK.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "gemini:gemini-2.5-flash": {
        "provider_id": "gemini",
        "label": "Gemini 2.5 Flash",
        "description": "Lower-latency Gemini sibling. Excellent throughput for everyday chat, multimodal Q&A, and high-volume agentic loops at a fraction of the Pro price.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
        "tags": ["coding"],
    },
    "gemini:gemini-2.0-flash": {
        "provider_id": "gemini",
        "label": "Gemini 2.0 Flash",
        "description": "Stable Flash-tier Gemini with multimodal input + audio output support. Good baseline for cost-effective coding and chat.",
        "context": "1000k",
        "params": "—",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
        "tags": [],
    },

    # =====================================================================
    # Together AI (provider_id = "together")
    # =====================================================================
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "provider_id": "together",
        "label": "Llama 3.3 70B Turbo",
        "description": "Together's optimized 70B Llama deployment. Strong open-weights generalist with low per-token pricing.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.88,
        "output_price_per_1m": 0.88,
        "tags": ["coding"],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "together:Qwen/Qwen2.5-Coder-32B-Instruct": {
        "provider_id": "together",
        "label": "Qwen2.5 Coder 32B",
        "description": "Open-weights model fine-tuned for code generation, completion, and review. Among the strongest OSS coding models at this size.",
        "context": "32k",
        "params": "32B (Total)",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 0.80,
        "tags": ["coding"],
    },
    "together:deepseek-ai/DeepSeek-V3": {
        "provider_id": "together",
        "label": "DeepSeek V3 (Together)",
        "description": "DeepSeek V3 served via Together AI. Frontier-tier coding and reasoning at OSS pricing.",
        "context": "131k",
        "params": "37B-671B (Active-Total)",
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 1.25,
        "tags": ["frontier", "reasoning", "coding"],
    },
    "together:meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
        "provider_id": "together",
        "label": "Llama 3.1 405B Turbo",
        "description": "The largest publicly-available Llama. Strong general-purpose capabilities; useful when you need OSS provenance at the high-parameter end.",
        "context": "128k",
        "params": "405B (Total)",
        "input_price_per_1m": 3.50,
        "output_price_per_1m": 3.50,
        "tags": ["frontier"],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },

    # =====================================================================
    # Groq (provider_id = "groq")
    # =====================================================================
    "groq:llama-3.3-70b-versatile": {
        "provider_id": "groq",
        "label": "Llama 3.3 70B (Groq)",
        "description": "Llama 3.3 70B running on Groq's LPU hardware — exceptionally fast token streaming. Same model weights as elsewhere; pay for the speed.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.59,
        "output_price_per_1m": 0.79,
        "tags": ["coding"],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "groq:llama-3.1-8b-instant": {
        "provider_id": "groq",
        "label": "Llama 3.1 8B Instant",
        "description": "Smallest Llama on Groq — sub-second responses for high-volume agentic workflows where latency dominates.",
        "context": "128k",
        "params": "8B (Total)",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.08,
        "tags": [],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "groq:mixtral-8x7b-32768": {
        "provider_id": "groq",
        "label": "Mixtral 8x7B (Groq)",
        "description": "Mistral's MoE model on Groq hardware. Solid quality at very high throughput for tool-calling agents.",
        "context": "32k",
        "params": "47B (Total)",
        "input_price_per_1m": 0.24,
        "output_price_per_1m": 0.24,
        "tags": [],
    },

    # =====================================================================
    # Fireworks AI (provider_id = "fireworks")
    # =====================================================================
    "fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct": {
        "provider_id": "fireworks",
        "label": "Llama 3.3 70B (Fireworks)",
        "description": "Fireworks-hosted Llama 3.3 70B with FireAttention for high-throughput inference. Production-grade SLAs.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.90,
        "output_price_per_1m": 0.90,
        "tags": ["coding"],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "fireworks:accounts/fireworks/models/qwen2p5-coder-32b-instruct": {
        "provider_id": "fireworks",
        "label": "Qwen2.5 Coder 32B (Fireworks)",
        "description": "Code-specialized Qwen2.5 model on Fireworks. Strong open-weights pick for coding agents.",
        "context": "32k",
        "params": "32B (Total)",
        "input_price_per_1m": 0.90,
        "output_price_per_1m": 0.90,
        "tags": ["coding"],
    },
    "fireworks:accounts/fireworks/models/deepseek-v3": {
        "provider_id": "fireworks",
        "label": "DeepSeek V3 (Fireworks)",
        "description": "DeepSeek V3 hosted on Fireworks. Competitive frontier-tier model at OSS pricing.",
        "context": "131k",
        "params": "37B-671B (Active-Total)",
        "input_price_per_1m": 0.90,
        "output_price_per_1m": 0.90,
        "tags": ["frontier", "reasoning", "coding"],
    },

    # =====================================================================
    # Mistral (provider_id = "mistral")
    # =====================================================================
    "mistral:mistral-large-latest": {
        "provider_id": "mistral",
        "label": "Mistral Large",
        "description": "Mistral's flagship dense model. Strong reasoning + multilingual generation.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 6.00,
        "tags": ["frontier", "reasoning"],
    },
    "mistral:mistral-small-latest": {
        "provider_id": "mistral",
        "label": "Mistral Small",
        "description": "Mistral's efficient small model. Great cost / quality balance for high-volume tasks.",
        "context": "32k",
        "params": "—",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.60,
        "tags": [],
    },
    "mistral:codestral-latest": {
        "provider_id": "mistral",
        "label": "Codestral",
        "description": "Mistral's code-specialized model. Strong fill-in-the-middle and instruct-following on code.",
        "context": "32k",
        "params": "22B (Total)",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.60,
        "tags": ["coding"],
    },

    # =====================================================================
    # xAI (provider_id = "xai")
    # =====================================================================
    "xai:grok-2-1212": {
        "provider_id": "xai",
        "label": "Grok 2",
        "description": "xAI's flagship Grok 2. Competitive on reasoning and conversational quality with native real-time web context support.",
        "context": "128k",
        "params": "—",
        "input_price_per_1m": 2.00,
        "output_price_per_1m": 10.00,
        "tags": ["frontier"],
    },

    # =====================================================================
    # Cerebras (provider_id = "cerebras")
    # =====================================================================
    "cerebras:llama-3.3-70b": {
        "provider_id": "cerebras",
        "label": "Llama 3.3 70B (Cerebras)",
        "description": "Llama 3.3 70B running on Cerebras' wafer-scale CS-3 hardware. Industry-leading throughput for long-context generation.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.85,
        "output_price_per_1m": 1.20,
        "tags": [],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "cerebras:llama3.1-8b": {
        "provider_id": "cerebras",
        "label": "Llama 3.1 8B (Cerebras)",
        "description": "Sub-second-latency Llama 3.1 8B on Cerebras hardware. Useful when you need LLM speed comparable to local inference.",
        "context": "128k",
        "params": "8B (Total)",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
        "tags": [],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },

    # =====================================================================
    # DeepInfra (provider_id = "deepinfra")
    # =====================================================================
    "deepinfra:meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "provider_id": "deepinfra",
        "label": "Llama 3.1 70B (DeepInfra)",
        "description": "Cost-optimized Llama 3.1 70B on DeepInfra. Among the cheapest pricing for this model size.",
        "context": "128k",
        "params": "70B (Total)",
        "input_price_per_1m": 0.35,
        "output_price_per_1m": 0.40,
        "tags": [],
        "weak_tool_calling_issue_url": LLAMA_3X_TOOL_CALLING_ISSUE_URL,
    },
    "deepinfra:Qwen/Qwen2.5-Coder-32B-Instruct": {
        "provider_id": "deepinfra",
        "label": "Qwen2.5 Coder 32B (DeepInfra)",
        "description": "DeepInfra-hosted Qwen2.5 Coder. Cheap option for OSS code generation.",
        "context": "32k",
        "params": "32B (Total)",
        "input_price_per_1m": 0.07,
        "output_price_per_1m": 0.16,
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
    "anthropic:claude-3-5-sonnet-20241022",
    "openai:gpt-4o",
    "gemini:gemini-2.5-pro",
    "openai:o3-mini",
    "wandb:deepseek-ai/DeepSeek-V3.1",
    "wandb:Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "openai:gpt-4o-mini",
    "anthropic:claude-3-5-haiku-20241022",
    "groq:llama-3.3-70b-versatile",
    "gemini:gemini-2.5-flash",
]


# Allowed tag values. Kept here as a constant so model_catalog and the
# UI can reference them without typo risk.
#
# Curated-only tags (set hand-by-hand in MODEL_METADATA — never
# auto-derived): ``coding`` / ``reasoning`` / ``frontier``.
#
# Auto-derived tags (computed inside ``model_catalog._merge_one``
# from LiteLLM flags + OpenRouter modality arrays + provider
# attributes — overlay onto curated tags):
# - ``long_context`` — context >= 200k tokens.
# - ``cheap`` — output_price <= $0.50/M tokens.
# - ``fast`` — provider is Groq or Cerebras (latency-tier hardware).
# - ``multimodal`` — supports image input (legacy alias of ``vision``;
#   kept for back-compat with curated entries).
# - ``vision`` — chat model that accepts image input. Picker tab.
# - ``image_gen`` — model that GENERATES images. Picker tab.
# - ``audio_gen`` — model that GENERATES audio (TTS). Picker tab.
# - ``audio_in`` — model that accepts audio input but doesn't gen.
# - ``video_gen`` — model that GENERATES video. Picker tab.
# - ``video_in`` — model that accepts video input but doesn't gen.
ALLOWED_TAGS: tuple[str, ...] = (
    "coding",
    "reasoning",
    "long_context",
    "fast",
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
