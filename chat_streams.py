"""Provider-agnostic streaming chat completions.

Single entry point :func:`stream_chat` that dispatches on
``provider.kind`` and yields a normalized OpenAI-shape stream chunk
sequence to ``agent._stream_one_call``. Four-way dispatch:

- ``openai_native``: ``ss.clients["openai"].chat.completions.create(...)``
  direct to ``api.openai.com``. The raw OpenAI Stream is yielded
  through unchanged — agent.py already speaks this shape.
- ``anthropic_native``: native :mod:`anthropic` SDK with translation.
  Messages are reshaped (system pulled out, ``role="tool"`` rewritten
  as a ``tool_result`` content block on a ``role="user"`` message),
  tools translated (OpenAI ``parameters`` → Anthropic ``input_schema``),
  and the streamed Anthropic events translated into synthesized
  OpenAI-shape chunks via :class:`SimpleNamespace`. Hooks ready (but
  unused in v1) for ``cache_control`` blocks on system / persistent
  context.
- ``google_native``: native :mod:`google.genai` SDK with translation
  via ``google.genai.types``. Hooks ready for ``tool_config`` and
  grounding (``google_search``, ``code_execution``).
- ``litellm_compat`` (the 9): ``litellm.completion(...)`` with
  ``stream=True`` and ``stream_options={"include_usage": True}``.
  LiteLLM normalizes responses to OpenAI shape, so the chunks pass
  through with no translation.

The agent loop (``agent._stream_one_call``) consumes whatever this
yields without caring about provider. The contract is "the iterator
yields objects with ``.choices`` / ``.usage`` matching the OpenAI
streaming-chat schema". For native paths we synthesize those objects
out of :class:`types.SimpleNamespace` so downstream code does not have
to differentiate.

This module is import-light: SDKs are imported lazily so an offline
build that ships only one provider's deps still loads cleanly.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterator

from providers import Provider


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------
def stream_chat(
    *,
    provider: Provider,
    api_key: str,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> Iterator[Any]:
    """Yield OpenAI-shape stream chunks for any provider kind.

    ``model`` is the **raw** model id (``gpt-4o``, ``claude-3-5-sonnet-...``,
    ``deepseek-ai/DeepSeek-V3.1``); the qualified-id stripping happens in
    the agent loop before this is called. For ``litellm_compat`` we
    additionally prepend ``provider.litellm_prefix`` so LiteLLM's
    namespacing (``together_ai/...``, ``groq/...``, ``wandb/...``)
    routes the request to the right backend.

    ``messages`` and ``tools`` arrive in OpenAI shape; the per-kind
    translators reshape them as needed for the native APIs and yield
    OpenAI-shape stream chunks back.
    """
    kind = provider.kind
    if kind == "openai_native":
        yield from _stream_openai_native(client, model, messages, tools)
    elif kind == "litellm_compat":
        yield from _stream_litellm(provider, api_key, model, messages, tools)
    elif kind == "anthropic_native":
        yield from _stream_anthropic_native(client, model, messages, tools)
    elif kind == "google_native":
        yield from _stream_google_native(client, model, messages, tools)
    else:
        raise ValueError(f"Unhandled provider kind: {kind!r}")


# ---------------------------------------------------------------------------
# OpenAI native + LiteLLM-compat (no translation needed — both speak
# OpenAI shape natively).
# ---------------------------------------------------------------------------
def _stream_openai_native(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> Iterator[Any]:
    """Direct stream from the native :class:`openai.OpenAI` client.

    No translation needed — the OpenAI SDK is the reference shape that
    ``agent._stream_one_call`` already consumes. Forward-future-proof:
    if the user's selected model is an o-series reasoning model and
    the messages list carries a ``reasoning_effort`` hint (e.g.
    spliced in by the system prompt), we'd lift it into the kwargs
    here. For v1 we just pass through; Phase 3.5 turns on the hooks.
    """
    extra: dict[str, Any] = {}
    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )
    yield from stream


def _stream_litellm(
    provider: Provider,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> Iterator[Any]:
    """Stream via :func:`litellm.completion`. No persistent client object.

    LiteLLM normalizes provider responses to OpenAI shape, so the
    chunks pass through unchanged. The ``model`` arg is prefixed with
    the provider's LiteLLM namespace (``together_ai/...``,
    ``groq/...``, etc.) so LiteLLM dispatches to the right backend.

    LiteLLM internally constructs an ``httpx`` request to
    ``provider.base_url`` using ``api_key``; no aggregation, no
    markup. The ``litellm.completion`` call is what makes LiteLLM
    direct-rather-than-marked-up — see :mod:`providers` for the
    library-vs-cloud distinction.
    """
    import litellm

    qualified_model = f"{provider.litellm_prefix}{model}"
    extra: dict[str, Any] = {}
    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"
    if provider.base_url:
        extra["api_base"] = provider.base_url

    stream = litellm.completion(
        model=qualified_model,
        api_key=api_key,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )
    yield from stream


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
) -> Iterator[Any]:
    """Stream via the native ``anthropic.Anthropic`` client.

    Translates messages + tools to Anthropic shape, calls
    ``client.messages.stream(...)``, and emits OpenAI-shape chunks
    (synthesized via :class:`SimpleNamespace`) so the agent loop's
    OpenAI-shaped consumer doesn't need to know which provider
    produced the stream.

    Hooks: ``cache_control`` blocks (Phase 3.5) would be added by
    extending the system + last persistent context with
    ``{"type": "ephemeral", "cache_control": {"type": "ephemeral"}}``;
    extended thinking via ``thinking={"type": "enabled", "budget_tokens": N}``
    is wired similarly.
    """
    system, anth_messages = _openai_messages_to_anthropic(messages)
    anth_tools = _openai_tools_to_anthropic(tools)

    extra: dict[str, Any] = {}
    if anth_tools:
        extra["tools"] = anth_tools
    if system:
        extra["system"] = system

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
) -> Iterator[Any]:
    """Stream via the native ``google.genai.Client``.

    Translates messages + tools to Gemini shape, calls
    ``client.models.generate_content_stream(...)``, and emits
    OpenAI-shape chunks. Hooks are ready for grounding (``Tool(google_search=...)``)
    and code execution but unused in v1.
    """
    from google.genai import types

    system, contents = _openai_messages_to_gemini(messages)
    g_tools = _openai_tools_to_gemini(tools)

    config_kwargs: dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system
    if g_tools:
        config_kwargs["tools"] = g_tools
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
