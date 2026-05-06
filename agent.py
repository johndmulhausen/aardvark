"""Agent loop driving W&B Inference chat completions with tool calling.

Exposes a generator, ``run_agent_turn``, that yields events the UI renders:

- ``{"type": "skills_loaded", "selected": [...], "unknown_slash": [...]}`` —
  emitted exactly once at the start of every turn so the UI can show which
  ``.cursor/skills`` were sliced into the system prompt this turn (and any
  ``/foo`` slash commands that didn't match a known skill).
- ``{"type": "assistant_text_delta", "content": ...}`` — a partial chunk of
  assistant text streamed token-by-token. The UI appends these to a live
  placeholder; they are NOT persisted to ``ui_turns`` for replay.
- ``{"type": "assistant_text", "content": ...}`` — the full text of an
  assistant message, emitted once after its deltas have streamed. This is
  what the UI replays from history on rerun.
- ``{"type": "tool_call", "id": ..., "name": ..., "args": ...}``
- ``{"type": "tool_result", "id": ..., "name": ..., "result": ...}``
- ``{"type": "error", "message": ...}``

The generator mutates the supplied ``messages`` list in place so the caller can
persist it across turns. Each model call is made with ``stream=True``; tool-call
deltas are accumulated by ``index`` across chunks before being dispatched, since
W&B Inference (like the OpenAI API) splits a single tool call's id, name, and
arguments across multiple chunks.

Tracing
-------
:func:`run_agent_turn` and :func:`_stream_one_call` are decorated with
``@weave.op`` so each turn shows up in W&B Weave as a single trace tree:
``run_agent_turn`` (kind=agent) → one ``_stream_one_call`` (kind=llm) per
inference round → the auto-patched ``client.chat.completions.create`` call
underneath. Tool dispatches are traced by ``tools.dispatch``. Weave init
happens in ``streamlit_app.py``'s connect flow via ``wb_client.init_weave``;
if the user is offline or init failed, ``@weave.op`` is a no-op and the
agent still runs unchanged.

The OpenAI ``client`` argument is stripped from logged inputs because it
embeds the user's W&B API key in its repr — the auto-traced child op
already captures the relevant request fields (model, messages, tools).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal

import weave
from openai import OpenAI

import mcp_servers
import project_context
from tools import ToolContext, dispatch, tools_for_mode


def _strip_client(inputs: dict[str, Any]) -> dict[str, Any]:
    """Drop the OpenAI ``client`` arg from a Weave op's logged inputs.

    The OpenAI client object's repr includes its API key; logging it would
    leak the user's W&B credentials into Weave traces. Everything we want to
    see in the trace (model, messages, tools) is captured by the auto-patched
    child op anyway.
    """
    return {k: v for k, v in inputs.items() if k != "client"}

SYSTEM_PROMPT = """\
You are Aardvark, a coding agent running inside a Streamlit app, powered by a model
served by the W&B Inference service. You help the user explore and modify
code in a local working directory.

Workflow:
1. When the user asks about the codebase, start with `list_files` to get oriented.
2. Use `read_file` before editing — never guess at file contents.
3. Prefer `edit_file` (single-string replacement) over `write_file` for small
   changes. Use `write_file` for new files or full rewrites.
4. Use `run_shell` whenever it helps: running tests, type checkers, linters,
   git commands, build scripts, or quick `grep`/`find` probes. Keep commands
   short, deterministic, and bounded; favor read-only commands over destructive
   ones unless the user has clearly asked for the latter.
5. After making edits, briefly summarize what you changed and why.

Rules:
- All paths are relative to the working directory. Do not use absolute paths
  or `..` segments that escape the directory; the executor will reject them.
- If a tool returns an error, read it carefully and adjust. Do not loop on the
  same failing call.
- Be concise. Do not narrate every internal step; let the tool call log speak
  for itself.
"""

SYSTEM_PROMPT_ASK = """\
You are Aardvark, a coding assistant running inside a Streamlit app, powered by
a model served by the W&B Inference service. The user is in "Ask only" mode.

You can ONLY explore the codebase using `list_files` and `read_file`. You
CANNOT modify files, write new files, or run shell commands — those tools are
not available to you in this mode.

Workflow:
1. Use `list_files` to get oriented when the user asks about the project.
2. Use `read_file` to inspect specific files before making claims about them.
3. Answer the user's question directly. If a change is needed, describe what
   would need to change rather than trying to change it; the user can switch
   to Agent mode if they want you to apply edits.

Rules:
- All paths are relative to the working directory; the executor rejects escapes.
- If a tool returns an error, read it carefully and adjust. Do not loop on the
  same failing call.
- Be concise. Do not narrate every internal step.
"""


def _short_model_name(model: str) -> str:
    return model.split("/")[-1]


@weave.op(
    name="stream_one_call",
    kind="llm",
    color="blue",
    postprocess_inputs=_strip_client,
)
def _stream_one_call(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Stream a single chat-completion call and yield UI events.

    Yields ``assistant_text_delta`` events while the model is producing text,
    accumulates any tool-call fragments by index, and at the end yields:

    - ``assistant_text`` with the full content if the model produced any text.
    - ``tool_call`` events for each fully-assembled tool call.

    Crucially, this also appends the assembled assistant message to
    ``messages`` so the next iteration of the agent loop sees the full
    conversation context. Tool *results* are appended by the caller after it
    dispatches each call; this helper does not run any tools itself.

    The yielded ``tool_call`` events carry ``raw_args`` (the unparsed JSON
    string the model produced) alongside the parsed ``args`` dict so the
    caller can pass ``raw_args`` straight to :func:`tools.dispatch` without
    re-serializing.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )

    content_parts: list[str] = []
    # Tool calls arrive as deltas keyed by ``index``. The id and name typically
    # land on the first delta for that index, then arguments stream in as a
    # sequence of string fragments that must be concatenated in order.
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            content_parts.append(delta.content)
            yield {"type": "assistant_text_delta", "content": delta.content}

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                slot = tool_calls_by_index.setdefault(
                    idx,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc_delta.id:
                    slot["id"] = tc_delta.id
                if tc_delta.function is not None:
                    if tc_delta.function.name:
                        slot["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        slot["function"]["arguments"] += tc_delta.function.arguments

    full_content = "".join(content_parts)
    tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]

    assistant_entry: dict[str, Any] = {"role": "assistant", "content": full_content}
    if tool_calls:
        assistant_entry["tool_calls"] = tool_calls
    messages.append(assistant_entry)

    if not tool_calls:
        yield {"type": "assistant_text", "content": full_content, "_final": True}
        return

    if full_content:
        yield {"type": "assistant_text", "content": full_content}

    for tc in tool_calls:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"] or "{}"
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed_args = {"_raw": raw_args}
        yield {
            "type": "tool_call",
            "id": tc["id"],
            "name": name,
            "args": parsed_args,
            "raw_args": raw_args,
        }


@weave.op(
    name="run_agent_turn",
    kind="agent",
    color="purple",
    postprocess_inputs=_strip_client,
)
def run_agent_turn(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    working_dir: Path,
    mode: Literal["agent", "ask"] = "agent",
) -> Iterator[dict[str, Any]]:
    """Drive one user turn through the model + tool loop.

    The caller is expected to have already appended the new user message to
    ``messages``. This function appends assistant and tool messages as the
    conversation progresses so the next turn has full context.

    ``mode`` selects the system prompt and which tools the model is offered:

    - ``"agent"``: full tool set (read, write, edit, shell).
    - ``"ask"``: read-only tools (``list_files``, ``read_file``) plus an
      ask-only system prompt; the model is told it cannot modify the project.

    The system message at index 0 is rewritten on every turn so a mid-chat
    mode switch takes effect immediately. The rewrite also folds in any
    project-context guidance (AGENTS.md, .cursor/rules, plus any skills
    matched by slash command or keyword in the latest user message); see
    :mod:`project_context`.

    Each model call is streamed; the function yields ``assistant_text_delta``
    events as tokens arrive, followed by a single ``assistant_text`` event
    once the message is complete (with the full content for replay), then
    any ``tool_call``/``tool_result`` pairs. The loop runs until the model
    returns a final answer with no tool calls (or a single iteration produces
    neither content nor tool calls, which is treated as an error).

    Before any of that, the function emits a single ``skills_loaded`` event
    so the UI can show which skills (and via which trigger) were spliced
    into the system prompt for this turn.
    """
    ctx = ToolContext(working_dir=working_dir)

    proj_ctx = project_context.scan(working_dir)
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                last_user = content
            break
    selection = project_context.select_skills_for_turn(last_user, proj_ctx)

    yield {
        "type": "skills_loaded",
        "selected": [
            {
                "slug": picked.skill.slug,
                "name": picked.skill.name,
                "scope": picked.skill.scope,
                "trigger_reason": picked.trigger_reason,
            }
            for picked in selection.selected
        ],
        "unknown_slash": list(selection.unknown_slash),
    }

    base_prompt = SYSTEM_PROMPT_ASK if mode == "ask" else SYSTEM_PROMPT
    addendum = project_context.build_system_addendum(proj_ctx, selection)
    system_content = base_prompt + addendum
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": system_content}
    else:
        messages.insert(0, {"role": "system", "content": system_content})

    # Tool list per turn = local tools + every connected MCP server's
    # tools. MCP tools are only exposed in Agent mode for v1: Ask mode's
    # contract is "purely sandbox-read", and we have no per-tool read-only
    # metadata yet.
    tools = list(tools_for_mode(mode))
    if mode != "ask":
        tools.extend(mcp_servers.get_registry().openai_tool_schemas())

    while True:
        try:
            stream_events = _stream_one_call(client, model, messages, tools)
            pending_tool_calls: list[dict[str, Any]] = []
            saw_final = False

            for event in stream_events:
                if event["type"] == "assistant_text" and event.pop("_final", False):
                    saw_final = True
                    yield event
                elif event["type"] == "tool_call":
                    pending_tool_calls.append(event)
                else:
                    yield event
        except Exception as e:
            yield {"type": "error", "message": f"Inference call failed: {e}"}
            return

        if saw_final:
            return

        if not pending_tool_calls:
            yield {
                "type": "error",
                "message": (
                    f"{_short_model_name(model)} returned no content and no "
                    "tool calls. Try rephrasing your request."
                ),
            }
            return

        for call_event in pending_tool_calls:
            yield call_event

            raw_args = call_event.pop("raw_args", "{}")
            tool_name = call_event["name"]
            if tool_name.startswith(mcp_servers.TOOL_NAME_PREFIX):
                result = mcp_servers.dispatch(tool_name, raw_args)
            else:
                result = dispatch(tool_name, raw_args, ctx)

            yield {
                "type": "tool_result",
                "id": call_event["id"],
                "name": tool_name,
                "result": result,
            }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_event["id"],
                    "content": json.dumps(result),
                }
            )
