"""Provider-agnostic streaming chat completions.

Single entry point :func:`stream_chat` that dispatches on
``provider.kind`` and yields a normalized OpenAI-shape stream chunk
sequence to ``agent._stream_one_call``. Five-way dispatch:

- ``openai_native``: native ``openai.OpenAI`` against ``api.openai.com``.
  Active native feature: ``reasoning_effort`` for o-series models when
  ``model_options["reasoning_effort"]`` is set in the per-chat
  options dict. The raw OpenAI ``Stream`` is yielded through unchanged
  — agent.py speaks this shape directly.
- ``anthropic_native``: native :mod:`anthropic` SDK with translation.
  Messages are reshaped (system pulled out, ``role="tool"`` rewritten
  as a ``tool_result`` content block on a ``role="user"`` message),
  tools translated (OpenAI ``parameters`` → Anthropic ``input_schema``),
  and the streamed Anthropic events translated into synthesized
  OpenAI-shape chunks via :class:`SimpleNamespace`. **The system
  prompt is auto-wrapped with a ``cache_control: ephemeral`` block so
  every turn after the first hits the prompt cache** — ~90% repeat-
  cost savings on the long agent system prompt with zero UI surface.
- ``google_native``: native :mod:`google.genai` SDK with translation
  via ``google.genai.types``. Active native feature: Google Search
  grounding when ``model_options["grounding"]`` is True (adds
  ``Tool(google_search=GoogleSearch())`` to the call config).
- ``mistral_native``: native :mod:`mistralai` SDK. Mistral's chat-
  completions wire format is OpenAI-shape, so the translator is mostly
  a passthrough; the dispatch is kept separate so Mistral-specific
  kwargs (FIM, structured outputs, Magistral reasoning) can land in
  one branch without affecting other providers.
- ``xai_native``: ``openai.OpenAI`` against xAI's REST endpoint
  (officially OpenAI-compatible per docs.x.ai). Active native feature:
  Live Search when ``model_options["live_search"]`` is True (passes
  ``extra_body={"search_parameters": {...}}`` on the OpenAI-SDK call).
- ``openai_compat``: ``openai.OpenAI`` against the provider's own
  ``/v1/chat/completions`` (W&B Inference, OpenRouter). No
  provider-specific kwargs; same code path as ``openai_native`` minus
  the OpenAI-only feature flags.

The agent loop (``agent._stream_one_call``) consumes whatever this
yields without caring about provider. The contract is "the iterator
yields objects with ``.choices`` / ``.usage`` matching the OpenAI
streaming-chat schema". For non-OpenAI-shape paths we synthesize
those objects out of :class:`types.SimpleNamespace` so downstream
code does not have to differentiate.

``model_options`` is an optional dict of per-chat feature flags
threaded through from ``Chat.model_options``: ``reasoning_effort``
(low/medium/high for OpenAI o-series), ``grounding`` (bool for
Google), ``live_search`` (bool for xAI), and ``thinking`` (bool for
Anthropic extended thinking — hook-only today). Unknown keys are
silently ignored, so a future feature flag can flow through without
churn here.

This module is import-light: SDKs are imported lazily so an offline
build that ships only one provider's deps still loads cleanly.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterator

from providers import Provider


# ---------------------------------------------------------------------------
# OpenAI o-series detection — the ``reasoning_effort`` kwarg is only
# accepted by reasoning models. The id pattern is "o" followed by a
# digit (``o1``, ``o1-mini``, ``o3``, ``o3-mini``, ``o3-pro``, ``o4``,
# ``gpt-5-thinking-...``, etc.); we use a simple prefix check that
# matches the live OpenAI catalog as of 2026 and can be extended if
# OpenAI ships a different naming convention.
# ---------------------------------------------------------------------------
_VALID_REASONING_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high"})


def _is_openai_reasoning_model(model: str) -> bool:
    """Return True if ``model`` is an OpenAI o-series reasoning model."""
    if not isinstance(model, str) or not model:
        return False
    return (
        model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    )


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------
def stream_chat(
    *,
    provider: Provider,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model_options: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Yield OpenAI-shape stream chunks for any provider kind.

    ``model`` is the **raw** model id (``gpt-4o``, ``claude-sonnet-4-6``,
    ``deepseek-ai/DeepSeek-V3.1``); the qualified-id stripping happens
    in the agent loop before this is called.

    ``messages`` and ``tools`` arrive in OpenAI shape; the per-kind
    translators reshape them as needed for the native APIs and yield
    OpenAI-shape stream chunks back.

    ``client`` is the persistent client object stored in
    ``ss.clients[provider.id]`` — an ``openai.OpenAI`` for
    ``openai_native`` / ``openai_compat`` / ``xai_native``, an
    ``anthropic.Anthropic`` for ``anthropic_native``, a
    ``google.genai.Client`` for ``google_native``, a ``mistralai.Mistral``
    for ``mistral_native``. The connect flow in :mod:`actions` builds
    these via :func:`providers.make_provider_client` and stashes them
    in session state.

    ``model_options`` is the per-chat feature-flag dict (sourced from
    ``Chat.model_options``). Each per-kind translator pulls the keys
    it cares about; unknown keys are silently ignored.
    """
    opts = model_options or {}
    kind = provider.kind
    if kind == "openai_native":
        yield from _stream_openai(client, model, messages, tools, model_options=opts)
    elif kind == "openai_compat":
        yield from _stream_openai(client, model, messages, tools, model_options=None)
    elif kind == "xai_native":
        yield from _stream_xai_native(client, model, messages, tools, model_options=opts)
    elif kind == "anthropic_native":
        yield from _stream_anthropic_native(client, model, messages, tools, model_options=opts)
    elif kind == "google_native":
        yield from _stream_google_native(client, model, messages, tools, model_options=opts)
    elif kind == "mistral_native":
        yield from _stream_mistral_native(client, model, messages, tools, model_options=opts)
    else:
        raise ValueError(f"Unhandled provider kind: {kind!r}")


# ---------------------------------------------------------------------------
# OpenAI SDK path — used by ``openai_native`` and (without options)
# ``openai_compat``. Same call, same returned shape, same downstream
# consumer in ``agent._stream_one_call``. The two kinds stay separate at
# the dispatch level so OpenAI-specific kwargs only flow on the
# ``openai_native`` path.
# ---------------------------------------------------------------------------
def _stream_openai(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model_options: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Stream from an :class:`openai.OpenAI` client.

    No translation needed — the OpenAI SDK is the reference shape that
    ``agent._stream_one_call`` already consumes. ``openai_compat``
    providers (W&B Inference, OpenRouter) reach the same SDK via a
    ``base_url`` swap; no provider-specific normalization layer sits
    between this call and the network.

    When ``model_options`` is provided AND the model is an OpenAI
    o-series reasoning model AND ``model_options["reasoning_effort"]``
    is one of ``"low"`` / ``"medium"`` / ``"high"``, the kwarg is
    passed through to the call. ``openai_compat`` providers always
    pass ``model_options=None`` so the kwarg never leaks into a
    non-OpenAI request.
    """
    extra: dict[str, Any] = {}
    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"

    if model_options and _is_openai_reasoning_model(model):
        effort = model_options.get("reasoning_effort")
        if isinstance(effort, str) and effort in _VALID_REASONING_EFFORTS:
            extra["reasoning_effort"] = effort

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )
    yield from stream


# ---------------------------------------------------------------------------
# xAI native — uses the OpenAI SDK against xAI's REST endpoint, with
# Live Search wired through ``extra_body`` when the per-chat option is
# set. The xAI REST endpoint is officially OpenAI-compatible (per
# docs.x.ai), so the call shape is identical to ``_stream_openai``.
# ---------------------------------------------------------------------------
def _stream_xai_native(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model_options: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Stream from an :class:`openai.OpenAI` client pointed at api.x.ai.

    When ``model_options["live_search"]`` is True, ``extra_body`` is
    populated with xAI's ``search_parameters={"mode": "auto"}`` so the
    model can hit real-time web at query time. The ``mode="auto"``
    setting lets the model decide whether each turn needs a search
    (vs ``"on"`` which forces every turn to search and ``"off"`` which
    disables — ``"auto"`` is the right default for an agent loop).

    The default (``live_search`` not set or False) sends a normal
    OpenAI-shape request with no xAI extensions, so the cost model
    matches the picker's curated $/M-token figure for Grok models.
    Enabling Live Search adds per-search costs (xAI publishes those
    separately); the chat page exposes a toggle so the user opts in
    explicitly.
    """
    extra: dict[str, Any] = {}
    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"

    if model_options and bool(model_options.get("live_search")):
        # ``extra_body`` is the OpenAI SDK's hook for passing
        # provider-specific JSON fields through to the request body.
        # xAI's docs document ``search_parameters`` at the top level
        # of the request; ``extra_body`` lands it there verbatim.
        extra["extra_body"] = {"search_parameters": {"mode": "auto"}}

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )
    yield from stream


# ---------------------------------------------------------------------------
# Mistral native — uses the official ``mistralai`` SDK. Mistral's
# chat-completions wire format is OpenAI-shape, so the SSE chunks
# emitted by ``mistral.chat.stream(...)`` are themselves OpenAI-
# compatible. We synthesize :class:`SimpleNamespace` chunks defensively
# in case the SDK's response objects diverge from OpenAI's shape on
# some attribute path.
# ---------------------------------------------------------------------------
def _stream_mistral_native(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model_options: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Stream from a :class:`mistralai.Mistral` client.

    The ``mistralai`` SDK exposes ``client.chat.stream(...)`` which
    returns a context-managed iterator of SSE events. Each event has a
    ``.data`` attribute carrying an OpenAI-shape ``ChatCompletionChunk``.
    We re-yield the inner ``data`` chunks so the agent loop's
    OpenAI-shape consumer doesn't have to differentiate.

    ``model_options`` is currently unused for Mistral — the kind
    discriminator exists so future Mistral-only kwargs (FIM input,
    Magistral reasoning options) can flow through without churning
    the dispatch table.
    """
    extra: dict[str, Any] = {}
    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"

    # ``mistral.chat.stream`` returns a context manager wrapping an
    # SSE iterator. Each event carries ``.data`` — the actual OpenAI-
    # shape chunk. Some SDK versions yield the chunk directly; we
    # try ``.data`` first and fall through to the event itself.
    response = client.chat.stream(
        model=model,
        messages=messages,
        stream_options={"include_usage": True},
        **extra,
    )

    try:
        ctx = response.__enter__() if hasattr(response, "__enter__") else None
        iterator = ctx if ctx is not None else response
        for event in iterator:
            inner = getattr(event, "data", event)
            yield inner
    finally:
        if hasattr(response, "__exit__"):
            try:
                response.__exit__(None, None, None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Anthropic native — translate OpenAI shape ↔ Anthropic shape.
# ---------------------------------------------------------------------------
def _openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split OpenAI-shape ``messages`` into ``(system, anthropic_messages)``.

    Anthropic's ``messages.create`` API takes a top-level ``system``
    string (not a system message) and ``messages`` containing only
    ``user`` / ``assistant`` roles. ``role="tool"`` results are
    rewritten as ``user`` messages carrying a ``tool_result`` content
    block referencing the original ``tool_call_id``.
    """
    system_parts: list[str] = []
    out_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id") or ""
            text = content if isinstance(content, str) else json.dumps(content)
            out_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": text,
                    }
                ],
            })
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {})
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "input": args,
                })
            if not blocks:
                continue
            out_messages.append({"role": "assistant", "content": blocks})
            continue

        if role == "user":
            # Translate OpenAI-shape content arrays to Anthropic
            # blocks. Plain strings pass through unchanged.
            if isinstance(content, list):
                out_messages.append(
                    {"role": "user", "content": _openai_content_to_anthropic_blocks(content)}
                )
            else:
                out_messages.append({"role": "user", "content": content})
            continue

    system = "\n\n".join(p for p in system_parts if p)
    return system, out_messages


def _openai_content_to_anthropic_blocks(content: list[Any]) -> list[dict[str, Any]]:
    """Translate OpenAI-shape multimodal parts → Anthropic content blocks.

    Handles:

    - ``{"type": "text", "text": ...}`` → ``{"type": "text", "text": ...}``
      (same shape, pass-through).
    - ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
      → ``{"type": "image", "source": {"type": "base64", "media_type":
      "image/png", "data": "..."}}``.
    - PDF data URLs (``data:application/pdf;base64,...``) → Anthropic
      ``{"type": "document", "source": {...}}`` blocks (Claude 3.5+
      native PDF support).
    """
    out: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            out.append({"type": "text", "text": part.get("text") or ""})
        elif ptype == "image_url":
            url_obj = part.get("image_url") or {}
            url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
            if isinstance(url, str) and url.startswith("data:"):
                header, _, b64 = url.partition(",")
                mime = "image/png"
                if ":" in header:
                    mime = header.split(";")[0].split(":")[-1] or "image/png"
                if mime.startswith("application/pdf"):
                    out.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    })
                else:
                    out.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    })
    return out


def _openai_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Translate OpenAI ``tools`` schema → Anthropic ``tools`` schema.

    OpenAI shape: ``[{"type": "function", "function": {"name", "description", "parameters"}}]``.
    Anthropic shape: ``[{"name", "description", "input_schema"}]``.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out or None


def _stream_anthropic_native(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model_options: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Stream via the native ``anthropic.Anthropic`` client.

    Translates messages + tools to Anthropic shape, calls
    ``client.messages.stream(...)``, and emits OpenAI-shape chunks
    (synthesized via :class:`SimpleNamespace`) so the agent loop's
    OpenAI-shaped consumer doesn't need to know which provider
    produced the stream.

    **Auto prompt caching**: when ``system`` is non-empty, we wrap it
    as a single text block with ``cache_control={"type": "ephemeral"}``
    so every turn after the first hits Anthropic's prompt cache. The
    cost-side win is dramatic for an agent that re-sends the same long
    system prompt every turn — the cache-read rate is 10% of the base
    input price (i.e. ~90% savings on the cached portion). Cache
    breakpoint placement is Anthropic's recommended "single
    cache_control on the system block" pattern; the system message is
    by far the largest stable prefix in our setup.

    ``model_options["thinking"]`` is wired here as a hook (extended
    thinking via ``thinking={"type": "enabled", "budget_tokens": N}``)
    but currently unused — the chat page's UI surface for the toggle
    is reserved for a follow-up phase.
    """
    system, anth_messages = _openai_messages_to_anthropic(messages)
    anth_tools = _openai_tools_to_anthropic(tools)

    extra: dict[str, Any] = {}
    if anth_tools:
        extra["tools"] = anth_tools
    if system:
        # Wrap the system string in a single text block with
        # ``cache_control: ephemeral`` so the long agent system prompt
        # gets cached. The 5-minute cache duration is fine for an
        # interactive agent — every turn refreshes the cache. The
        # write costs 1.25x base input on the first turn, then every
        # subsequent turn reads at 0.1x base input — a single repeat
        # turn pays back the write, and agent sessions usually run
        # many turns.
        extra["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # ``client.messages.stream(...)`` returns a context manager. We
    # iterate the raw events to keep tool-call deltas mappable to
    # OpenAI's ``index`` semantics.
    stream_ctx = client.messages.stream(
        model=model,
        messages=anth_messages,
        max_tokens=4096,
        **extra,
    )

    # Tool-call accumulator: Anthropic tracks tool_use blocks by their
    # content-block index, but the input args arrive as a stream of
    # ``input_json_delta`` partial JSON strings inside the same block.
    # We materialize one OpenAI-shape tool_call per content block and
    # emit ``index`` matching the content-block index.
    tool_blocks: dict[int, dict[str, Any]] = {}
    pending_text_indexes: set[int] = set()
    final_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with stream_ctx as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                msg = getattr(event, "message", None)
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    final_usage["prompt_tokens"] = int(getattr(usage, "input_tokens", 0) or 0)
            elif etype == "content_block_start":
                idx = int(getattr(event, "index", 0) or 0)
                cb = getattr(event, "content_block", None)
                cb_type = getattr(cb, "type", None)
                if cb_type == "tool_use":
                    tool_blocks[idx] = {
                        "id": getattr(cb, "id", "") or "",
                        "name": getattr(cb, "name", "") or "",
                        "arguments": "",
                    }
                    yield _make_openai_tool_call_delta_chunk(
                        index=idx,
                        id_=tool_blocks[idx]["id"],
                        name=tool_blocks[idx]["name"],
                    )
                elif cb_type == "text":
                    pending_text_indexes.add(idx)
            elif etype == "content_block_delta":
                idx = int(getattr(event, "index", 0) or 0)
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    text = getattr(delta, "text", "") or ""
                    if text:
                        yield _make_openai_text_delta_chunk(text)
                elif dtype == "input_json_delta":
                    chunk = getattr(delta, "partial_json", "") or ""
                    if chunk and idx in tool_blocks:
                        tool_blocks[idx]["arguments"] += chunk
                        yield _make_openai_tool_call_delta_chunk(
                            index=idx,
                            arguments=chunk,
                        )
            elif etype == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                    final_usage["completion_tokens"] = out_tokens
            elif etype == "message_stop":
                pass

    final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]
    if final_usage["prompt_tokens"] or final_usage["completion_tokens"]:
        yield _make_openai_usage_chunk(**final_usage)


# ---------------------------------------------------------------------------
# Google Gemini native — translate OpenAI shape ↔ google.genai shape.
# ---------------------------------------------------------------------------
def _openai_messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str, list[Any]]:
    """Translate OpenAI ``messages`` → ``(system_instruction, gemini_contents)``.

    Gemini's API takes a ``system_instruction`` separate from the
    conversation contents. ``role="tool"`` becomes ``role="user"``
    with a ``function_response`` part. ``role="assistant"`` becomes
    ``role="model"`` (Gemini's terminology).
    """
    from google.genai import types

    system_parts: list[str] = []
    contents: list[Any] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        if role == "tool":
            # Gemini expects function-response parts; the function
            # name maps to the original tool's name. We don't have it
            # here directly (only ``tool_call_id``), so we rely on the
            # caller having seeded the ``name`` field on the tool
            # message — most callers do via the ``name`` key.
            text = content if isinstance(content, str) else json.dumps(content)
            tool_name = msg.get("name") or ""
            try:
                response_data = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                response_data = {"output": text}
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=tool_name,
                    response=response_data if isinstance(response_data, dict) else {"output": response_data},
                )],
            ))
            continue

        if role == "assistant":
            parts: list[Any] = []
            if isinstance(content, str) and content:
                parts.append(types.Part.from_text(text=content))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                parts.append(types.Part.from_function_call(
                    name=fn.get("name") or "",
                    args=args if isinstance(args, dict) else {},
                ))
            if not parts:
                continue
            contents.append(types.Content(role="model", parts=parts))
            continue

        if role == "user":
            if isinstance(content, str):
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)],
                ))
            elif isinstance(content, list):
                # Multimodal content array (Phase 6) — best-effort
                # translation of ``image_url`` parts. Plain text
                # parts pass through; non-text parts that we don't
                # know how to translate are dropped (the upstream
                # ``attachments.build_user_message`` should have
                # already handled provider-specific shaping when
                # capability flags allow it).
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text":
                            parts.append(types.Part.from_text(text=part.get("text") or ""))
                        elif ptype == "image_url":
                            url_obj = part.get("image_url") or {}
                            url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                            if isinstance(url, str) and url.startswith("data:"):
                                # data:image/png;base64,xxx OR
                                # data:application/pdf;base64,xxx — both
                                # ride the same ``Part.from_bytes``
                                # surface in google-genai; the mime
                                # type is what tells Gemini whether
                                # this is an image or a PDF.
                                header, _, b64 = url.partition(",")
                                mime = (
                                    header.split(";")[0].split(":")[-1]
                                    if ":" in header
                                    else "image/png"
                                )
                                import base64
                                try:
                                    parts.append(types.Part.from_bytes(
                                        data=base64.b64decode(b64),
                                        mime_type=mime,
                                    ))
                                except Exception:  # noqa: BLE001 — best-effort
                                    pass
                if parts:
                    contents.append(types.Content(role="user", parts=parts))

    system = "\n\n".join(system_parts)
    return system, contents


def _openai_tools_to_gemini(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    """Translate OpenAI ``tools`` → Gemini ``Tool(function_declarations=...)``."""
    if not tools:
        return None
    from google.genai import types

    declarations: list[Any] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        declarations.append(types.FunctionDeclaration(
            name=name,
            description=fn.get("description", ""),
            parameters=fn.get("parameters") or {"type": "object", "properties": {}},
        ))
    if not declarations:
        return None
    return [types.Tool(function_declarations=declarations)]


def _stream_google_native(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model_options: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Stream via the native ``google.genai.Client``.

    Translates messages + tools to Gemini shape, calls
    ``client.models.generate_content_stream(...)``, and emits
    OpenAI-shape chunks.

    **Active native feature**: when ``model_options["grounding"]`` is
    True, a ``Tool(google_search=GoogleSearch())`` is appended to the
    config so Gemini can hit Google Search at query time. Grounding
    incurs per-search costs above Google's free tier (5,000 requests
    per month shared across Gemini 3, then $14/1000 search queries
    per their published pricing) — the chat page exposes a toggle so
    the user opts in explicitly.

    ``Tool(google_search=...)`` can NOT be combined with
    ``Tool(function_declarations=...)`` in the same request (Google's
    API rejects mixed tool kinds). When both apply, function tools
    win — the agent loop's local tools (``read_file`` etc.) are too
    central to the workflow to suppress, and grounding is opt-in
    anyway. We surface this trade-off via a chat-page caption when
    grounding is requested but the agent's tools are also present.
    """
    from google.genai import types

    system, contents = _openai_messages_to_gemini(messages)
    g_tools = _openai_tools_to_gemini(tools)

    config_kwargs: dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system

    # Function tools take precedence — grounding is dropped silently
    # when both are requested, since the API can't accept both.
    if g_tools:
        config_kwargs["tools"] = g_tools
    elif model_options and bool(model_options.get("grounding")):
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )

    # Gemini emits tool calls as ``Candidate.content.parts[].function_call``
    # objects. We synthesize one OpenAI-shape tool_call per
    # function-call part, indexed in order of appearance.
    tool_call_seq = 0
    final_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for chunk in stream:
        candidates = getattr(chunk, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                fn_call = getattr(part, "function_call", None)
                if text:
                    yield _make_openai_text_delta_chunk(text)
                if fn_call is not None:
                    name = getattr(fn_call, "name", "") or ""
                    args = getattr(fn_call, "args", {}) or {}
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    # Gemini doesn't give us a stable tool-call id —
                    # synthesize one so the agent loop's ``id`` field
                    # is non-empty and tool_result messages can refer
                    # back to it. The ``call_<seq>`` form mirrors what
                    # OpenAI uses internally.
                    call_id = f"call_{tool_call_seq}"
                    yield _make_openai_tool_call_delta_chunk(
                        index=tool_call_seq,
                        id_=call_id,
                        name=name,
                        arguments=args_str,
                    )
                    tool_call_seq += 1

        # Usage metadata can land on any chunk for streaming responses;
        # we accumulate the final values and yield a usage chunk at the
        # end. ``usage_metadata`` is the canonical name in google-genai.
        usage = getattr(chunk, "usage_metadata", None)
        if usage is not None:
            final_usage["prompt_tokens"] = int(getattr(usage, "prompt_token_count", 0) or 0)
            final_usage["completion_tokens"] = int(
                getattr(usage, "candidates_token_count", 0)
                or getattr(usage, "completion_token_count", 0)
                or 0
            )

    final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]
    if final_usage["prompt_tokens"] or final_usage["completion_tokens"]:
        yield _make_openai_usage_chunk(**final_usage)


# ---------------------------------------------------------------------------
# OpenAI-shape chunk synthesis (used by the native paths only)
# ---------------------------------------------------------------------------
# We yield SimpleNamespace objects that quack like the OpenAI SDK's
# ChatCompletionChunk so ``agent._stream_one_call`` doesn't need to
# differentiate. The fields it reads are:
#
#   chunk.choices: list[Choice]    # may be empty (final usage chunk)
#   chunk.usage: Usage | None
#   choice.delta.content: str | None
#   choice.delta.tool_calls: list[ToolCallDelta] | None
#   tool_call_delta.index: int
#   tool_call_delta.id: str | None
#   tool_call_delta.function.name: str | None
#   tool_call_delta.function.arguments: str | None
#
# A SimpleNamespace with exactly those attributes is sufficient.
def _make_openai_text_delta_chunk(text: str) -> Any:
    """Return a fake ChatCompletionChunk carrying a single ``content`` delta."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text, tool_calls=None),
        )],
        usage=None,
    )


def _make_openai_tool_call_delta_chunk(
    *,
    index: int,
    id_: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    """Return a fake ChatCompletionChunk carrying a tool-call delta."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    tc = SimpleNamespace(index=index, id=id_, function=fn)
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=[tc]),
        )],
        usage=None,
    )


def _make_openai_usage_chunk(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> Any:
    """Return a fake ChatCompletionChunk carrying only a ``usage`` block.

    The agent loop short-circuits on ``not chunk.choices`` and reads
    ``chunk.usage`` directly; we return a chunk with ``choices=[]``
    and a populated ``usage`` field to mirror that contract.
    """
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


__all__ = ["stream_chat"]
