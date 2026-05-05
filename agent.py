"""Agent loop driving W&B Inference chat completions with tool calling.

Exposes a generator, ``run_agent_turn``, that yields events the UI renders:

- ``{"type": "tool_call", "id": ..., "name": ..., "args": ...}``
- ``{"type": "tool_result", "id": ..., "name": ..., "result": ...}``
- ``{"type": "assistant_text", "content": ...}``
- ``{"type": "error", "message": ...}``

The generator mutates the supplied ``messages`` list in place so the caller can
persist it across turns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

from tools import TOOL_SCHEMAS, ToolContext, dispatch

SYSTEM_PROMPT = """\
You are Aardvark, a coding agent running inside a Streamlit app, powered by a model
served by the W&B Inference service. You help the user explore and modify
code in a local working directory.

Workflow:
1. When the user asks about the codebase, start with `list_files` to get oriented.
2. Use `read_file` before editing — never guess at file contents.
3. Prefer `edit_file` (single-string replacement) over `write_file` for small
   changes. Use `write_file` for new files or full rewrites.
4. After making edits, briefly summarize what you changed and why.
5. Only use `run_shell` if the user has enabled it. If a tool result says shell
   is disabled, do not retry — tell the user to enable it in the sidebar.

Rules:
- All paths are relative to the working directory. Do not use absolute paths
  or `..` segments that escape the directory; the executor will reject them.
- If a tool returns an error, read it carefully and adjust. Do not loop on the
  same failing call.
- Be concise. Do not narrate every internal step; let the tool call log speak
  for itself.
"""


def _short_model_name(model: str) -> str:
    return model.split("/")[-1]


def run_agent_turn(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    working_dir: Path,
    shell_enabled: bool,
    max_iters: int = 12,
) -> Iterator[dict[str, Any]]:
    """Drive one user turn through the model + tool loop.

    The caller is expected to have already appended the new user message to
    ``messages``. This function appends assistant and tool messages as the
    conversation progresses so the next turn has full context.
    """
    ctx = ToolContext(working_dir=working_dir, shell_enabled=shell_enabled)

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    for _ in range(max_iters):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as e:
            yield {"type": "error", "message": f"Inference call failed: {e}"}
            return

        choice = resp.choices[0]
        msg = choice.message

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            text = msg.content or ""
            yield {"type": "assistant_text", "content": text}
            return

        if msg.content:
            yield {"type": "assistant_text", "content": msg.content}

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {"_raw": raw_args}

            yield {
                "type": "tool_call",
                "id": tc.id,
                "name": name,
                "args": parsed_args,
            }

            result = dispatch(name, raw_args, ctx)

            yield {
                "type": "tool_result",
                "id": tc.id,
                "name": name,
                "result": result,
            }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    yield {
        "type": "error",
        "message": (
            f"Reached max_iters ({max_iters}) without a final answer from "
            f"{_short_model_name(model)}. Try increasing the limit or "
            "rephrasing your request."
        ),
    }
