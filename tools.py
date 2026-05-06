"""Tool schemas and executors for the code editing agent.

All tools are sandboxed to a `working_dir` provided at dispatch time. Paths that
escape the working directory are rejected. The schemas follow the OpenAI
function-calling format and are passed to the W&B Inference chat completions
endpoint via the `tools` parameter.

Tracing
-------
:func:`dispatch` is decorated with ``@weave.op(kind="tool")`` so each tool
invocation appears as a child of the surrounding ``run_agent_turn`` op in
W&B Weave, sitting between the inference calls that produced and consumed
it. When ``weave.init`` has not been called (no API key, init failed) the
decorator is a no-op and dispatch behaves identically.
"""
from __future__ import annotations

import difflib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import weave

SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build", ".next"}

READONLY_TOOL_NAMES: set[str] = {"list_files", "read_file"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories under a path inside the working "
                "directory. Returns a tree-style listing. Skips noisy folders "
                "like .git, node_modules, .venv, __pycache__."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the working directory. Defaults to '.'.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "How deep to recurse. Defaults to 3.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file. Returns content with each "
                "line prefixed by its 1-indexed line number, e.g. '   1|...'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file inside the working directory.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed first line to return. Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed last line to return (inclusive). Omit to read to EOF.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. Returns a "
                "unified diff showing the change (or '(new file)')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the working directory."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace the single occurrence of `old_string` with "
                "`new_string` in a file. Errors if `old_string` is missing or "
                "appears more than once. Returns a unified diff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the working directory."},
                    "old_string": {"type": "string", "description": "Exact text to find. Must be unique."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command inside the working directory. Returns "
                "stdout, stderr, and the exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Defaults to 30.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


@dataclass
class ToolContext:
    """Per-turn execution context for tool dispatch.

    Currently only carries the working directory; the ``_resolve_inside``
    sandbox check uses it to keep filesystem operations contained to the
    project the user picked. Shell commands run with this directory as their
    cwd.
    """
    working_dir: Path


def tools_for_mode(mode: str) -> list[dict[str, Any]]:
    """Return the OpenAI tool schemas exposed to the model for a given UI mode.

    - ``"agent"`` exposes every tool in :data:`TOOL_SCHEMAS`.
    - ``"ask"`` exposes only read-only tools (``list_files``, ``read_file``).

    Mode enforcement happens at the API boundary by simply withholding the
    schemas the model is not allowed to call.
    """
    if mode == "ask":
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in READONLY_TOOL_NAMES]
    return TOOL_SCHEMAS


def _resolve_inside(working_dir: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `working_dir`, rejecting escapes.

    Raises ValueError if the resolved path is not contained in `working_dir`.
    """
    base = working_dir.resolve()
    candidate = (base / rel_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as e:
        raise ValueError(
            f"Path '{rel_path}' resolves outside the working directory."
        ) from e
    return candidate


def _list_files(ctx: ToolContext, path: str = ".", max_depth: int = 3) -> dict[str, Any]:
    target = _resolve_inside(ctx.working_dir, path)
    if not target.exists():
        return {"error": f"Path does not exist: {path}"}
    if not target.is_dir():
        return {"error": f"Not a directory: {path}"}

    lines: list[str] = []

    def walk(p: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return
        entries = [e for e in entries if e.name not in SKIP_DIRS]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, depth + 1, prefix + extension)

    base_label = "." if path in ("", ".") else path.rstrip("/")
    lines.append(f"{base_label}/")
    walk(target, 1, "")
    return {"listing": "\n".join(lines)}


def _read_file(
    ctx: ToolContext,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    target = _resolve_inside(ctx.working_dir, path)
    if not target.exists():
        return {"error": f"File does not exist: {path}"}
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": f"File is not UTF-8 text: {path}"}

    all_lines = text.splitlines()
    total = len(all_lines)
    start = max(1, start_line)
    end = total if end_line is None else min(total, end_line)
    if start > total:
        return {"content": "", "total_lines": total, "note": "start_line beyond EOF"}

    width = max(4, len(str(end)))
    chunk = all_lines[start - 1 : end]
    rendered = "\n".join(f"{(start + i):>{width}}|{line}" for i, line in enumerate(chunk))
    return {"content": rendered, "total_lines": total, "shown_lines": [start, end]}


def _unified_diff(old: str, new: str, path: str) -> str:
    if old == new:
        return "(no change)"
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


def _write_file(ctx: ToolContext, path: str, content: str) -> dict[str, Any]:
    target = _resolve_inside(ctx.working_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    is_new = not target.exists()
    old = "" if is_new else target.read_text(encoding="utf-8")
    target.write_text(content, encoding="utf-8")
    diff = "(new file)" if is_new else _unified_diff(old, content, path)
    return {
        "ok": True,
        "path": path,
        "created": is_new,
        "bytes_written": len(content.encode("utf-8")),
        "diff": diff,
    }


def _edit_file(
    ctx: ToolContext,
    path: str,
    old_string: str,
    new_string: str,
) -> dict[str, Any]:
    target = _resolve_inside(ctx.working_dir, path)
    if not target.exists():
        return {"error": f"File does not exist: {path}"}
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    text = target.read_text(encoding="utf-8")
    occurrences = text.count(old_string)
    if occurrences == 0:
        return {"error": "old_string not found in file. Read the file again and retry."}
    if occurrences > 1:
        return {
            "error": (
                f"old_string appears {occurrences} times. Provide a longer, "
                "uniquely-identifying old_string."
            ),
        }
    new_text = text.replace(old_string, new_string, 1)
    target.write_text(new_text, encoding="utf-8")
    diff = _unified_diff(text, new_text, path)
    return {"ok": True, "path": path, "diff": diff}


def _run_shell(ctx: ToolContext, command: str, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.working_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s", "command": command}
    except Exception as e:
        return {"error": f"Failed to execute: {e}", "command": command}

    def _cap(s: str, n: int = 8000) -> str:
        if len(s) <= n:
            return s
        return s[:n] + f"\n... [truncated, {len(s) - n} more chars]"

    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": _cap(proc.stdout),
        "stderr": _cap(proc.stderr),
    }


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_files": _list_files,
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "run_shell": _run_shell,
}


@weave.op(name="dispatch_tool", kind="tool", color="green")
def dispatch(name: str, arguments_json: str, ctx: ToolContext) -> dict[str, Any]:
    """Run a tool call by name with JSON-encoded arguments.

    Errors during argument parsing or execution are returned as
    ``{"error": ...}`` rather than raised, so the model can recover.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON arguments: {e}"}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be a JSON object."}
    try:
        return fn(ctx, **args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
