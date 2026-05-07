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
import inspect
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import weave

# ---------------------------------------------------------------------------
# weave.op compat shim
# ---------------------------------------------------------------------------
# See ``mcp_servers._op`` for the full rationale: ``kind`` and ``color`` were
# added to ``weave.op`` partway through the 0.52 series; older installs raise
# ``TypeError`` at decorator-evaluation time. We feature-detect once at module
# load so older weave loses the UI categorization but still produces a correct
# trace tree. pyproject pins a recent-enough weave for fresh installs.
_WEAVE_OP_PARAMS = set(inspect.signature(weave.op).parameters)
_WEAVE_OP_DROP = {k for k in ("kind", "color") if k not in _WEAVE_OP_PARAMS}


def _op(*args: Any, **kwargs: Any) -> Any:
    """``@weave.op`` wrapper that drops kwargs unsupported by older weave."""
    for k in _WEAVE_OP_DROP:
        kwargs.pop(k, None)
    return weave.op(*args, **kwargs)

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
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate one or more images from a text prompt. Saves the "
                "resulting PNG file(s) into the chat's artifacts folder "
                "(.wb_artifacts/<chat_id>/) and returns their relative "
                "paths. Auto-selects an image-generation model on a "
                "connected provider (OpenAI / Google Gemini) when 'model' "
                "is omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the desired image."},
                    "model": {
                        "type": "string",
                        "description": "Optional qualified model id (e.g. 'openai:gpt-image-1').",
                    },
                    "size": {
                        "type": "string",
                        "description": "Image size, e.g. '1024x1024'. Defaults to '1024x1024'.",
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality tier ('standard' / 'hd'). Defaults to 'standard'.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "How many images to generate (1-4). Defaults to 1.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_speech",
            "description": (
                "Generate spoken audio from a text input. Saves the "
                "resulting audio file (mp3 by default) into the chat's "
                "artifacts folder and returns its relative path. "
                "Auto-selects a TTS model on a connected provider when "
                "'model' is omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to synthesize."},
                    "model": {
                        "type": "string",
                        "description": "Optional qualified model id (e.g. 'openai:gpt-4o-mini-tts').",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice preset (e.g. 'alloy', 'echo', 'fable'). Defaults to 'alloy'.",
                    },
                    "response_format": {
                        "type": "string",
                        "description": "Output format ('mp3' / 'wav' / 'opus' / 'flac'). Defaults to 'mp3'.",
                    },
                    "speed": {
                        "type": "number",
                        "description": "Playback speed (0.25-4.0). Defaults to 1.0.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": (
                "Generate a short video from a text prompt. Long-running "
                "operation (30-120s typical). Saves the resulting MP4 file "
                "into the chat's artifacts folder. Auto-selects a video-"
                "generation model on a connected provider when 'model' is "
                "omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the desired video."},
                    "model": {"type": "string", "description": "Optional qualified model id."},
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Approximate video length in seconds (1-30). Defaults to 5.",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Aspect ratio (e.g. '16:9', '9:16'). Defaults to '16:9'.",
                    },
                },
                "required": ["prompt"],
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

    ``chat_id`` is populated by ``chats.start_turn`` before dispatch so
    media-generation tools can write into the per-chat artifacts folder
    (``<working_dir>/.wb_artifacts/<chat_id>/``). It's optional because
    legacy / non-chat tool callers (e.g. the model_catalog refresh
    path) don't need it.

    ``provider_keys`` holds the per-provider API keys so media tools
    can dispatch to the right provider's image / audio / video API
    without re-importing Streamlit. Populated by ``chats.start_turn``
    from ``ss.provider_keys``; ``{}`` when no keys are configured.

    ``clients`` is the per-provider native client dict (Anthropic /
    Google / OpenAI native clients keyed by provider id). LiteLLM-
    routed providers carry ``None`` since LiteLLM is stateless.
    """
    working_dir: Path
    chat_id: str | None = None
    provider_keys: dict[str, str] | None = None
    clients: dict[str, Any] | None = None


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


# ---------------------------------------------------------------------------
# Media generation tools (Phase 5)
#
# All three tools auto-pick a default model on a connected provider when
# ``model`` is None, save output bytes into the chat's artifacts
# folder via ``_artifact_paths.artifact_path``, and return a result
# carrying ``kind`` + ``paths`` (or ``path``) + ``mime_type`` +
# ``cost_usd`` + ``model_used`` so the chat-page renderer can dispatch
# inline HTML5 previews and the usage dashboard can record per-mode
# spend without provider-specific knowledge.
# ---------------------------------------------------------------------------
def _resolve_media_model(
    ctx: ToolContext,
    requested: str | None,
    target_mode: str,
) -> tuple[Any, str | None] | None:
    """Resolve ``(provider, model_info)`` for a media generation call.

    ``requested`` may be a fully-qualified id, a bare id (assumed to
    be on the W&B Inference path for back-compat), or ``None`` (in
    which case we auto-pick the cheapest connected provider's model
    in ``target_mode``).

    Returns ``None`` when no connected provider supports the
    requested mode; the calling tool turns that into a clear
    ``{"error": ...}`` response.
    """
    import model_catalog
    from models import is_qualified, qualify

    if not ctx.clients or not ctx.provider_keys:
        return None

    target_qid: str | None = None
    if requested:
        target_qid = requested if is_qualified(requested) else qualify(requested, "wandb")
    else:
        info = model_catalog.default_model_for_mode(
            target_mode,  # type: ignore[arg-type]
            ctx.clients,
            prefer_cheapest=True,
        )
        if info is not None:
            target_qid = info.qualified_id

    if not target_qid:
        return None
    info = model_catalog.get_info(target_qid)
    if info is None:
        return None
    if info.mode != target_mode:
        return None
    return info, target_qid


def _generate_image(
    ctx: ToolContext,
    prompt: str,
    *,
    model: str | None = None,
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
) -> dict[str, Any]:
    """Generate ``n`` images via the OpenAI Images API (or compat)."""
    import _artifact_paths

    if not ctx.chat_id:
        return {"error": "generate_image requires an active chat context."}

    resolved = _resolve_media_model(ctx, model, "image_generation")
    if resolved is None:
        return {
            "error": (
                "No connected provider supports image generation. Configure "
                "OpenAI or another image-capable provider in Settings."
            ),
        }
    info, qid = resolved
    provider_id = info.provider_id

    api_key = (ctx.provider_keys or {}).get(provider_id, "")
    client = (ctx.clients or {}).get(provider_id)
    if not api_key:
        return {"error": f"No API key configured for {provider_id}."}

    n = max(1, min(int(n or 1), 4))
    paths_out: list[str] = []
    try:
        if provider_id == "openai" and client is not None:
            response = client.images.generate(
                model=info.raw_id,
                prompt=prompt,
                size=size,
                quality=quality if info.raw_id != "dall-e-3" or quality in {"standard", "hd"} else "standard",
                n=n,
            )
            for idx, item in enumerate(response.data):
                # The OpenAI API can return either ``b64_json`` (when
                # ``response_format="b64_json"`` was passed implicitly
                # for images APIs that always return base64, e.g.
                # gpt-image-1) or ``url`` (DALL-E 2/3). Handle both.
                import base64
                from urllib.request import urlopen
                b64 = getattr(item, "b64_json", None)
                url = getattr(item, "url", None)
                if b64:
                    raw = base64.b64decode(b64)
                elif url:
                    with urlopen(url, timeout=30) as r:
                        raw = r.read()
                else:
                    continue
                out = _artifact_paths.artifact_path(
                    ctx.working_dir, ctx.chat_id, kind="img", ext="png", idx=idx,
                )
                out.write_bytes(raw)
                paths_out.append(str(out.relative_to(ctx.working_dir)))
        elif provider_id == "gemini" and client is not None:
            # Google's image generation goes through ``client.models.generate_images``.
            response = client.models.generate_images(
                model=info.raw_id,
                prompt=prompt,
                config={"number_of_images": n},
            )
            for idx, gen in enumerate(getattr(response, "generated_images", []) or []):
                img = getattr(gen, "image", None)
                raw = getattr(img, "image_bytes", None)
                if not raw:
                    continue
                out = _artifact_paths.artifact_path(
                    ctx.working_dir, ctx.chat_id, kind="img", ext="png", idx=idx,
                )
                out.write_bytes(raw)
                paths_out.append(str(out.relative_to(ctx.working_dir)))
        else:
            return {
                "error": (
                    f"Image generation via {provider_id} is not yet "
                    "implemented. Try OpenAI or Google Gemini."
                ),
            }
    except Exception as e:  # noqa: BLE001 — surfaced verbatim to the model
        return {"error": f"Image generation failed: {e}"}

    if not paths_out:
        return {"error": "Image generation returned no images."}

    # Compute cost from the model's image_pricing slot when available.
    cost_usd: float | None = None
    if info.image_pricing:
        # Pick the closest matching size/quality key, falling back to
        # the cheapest known rate.
        key = f"{size}/{quality}"
        per_image = info.image_pricing.get(key) or min(info.image_pricing.values())
        cost_usd = round(per_image * len(paths_out), 6)

    return {
        "ok": True,
        "kind": "image",
        "paths": paths_out,
        "mime_type": "image/png",
        "model_used": qid,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": len(paths_out),
        "cost_usd": cost_usd,
    }


def _generate_speech(
    ctx: ToolContext,
    text: str,
    *,
    model: str | None = None,
    voice: str = "alloy",
    response_format: str = "mp3",
    speed: float = 1.0,
) -> dict[str, Any]:
    """Generate a speech audio file via OpenAI's TTS (or compat) API."""
    import _artifact_paths

    if not ctx.chat_id:
        return {"error": "generate_speech requires an active chat context."}

    resolved = _resolve_media_model(ctx, model, "audio_speech")
    if resolved is None:
        return {
            "error": (
                "No connected provider supports speech generation. Configure "
                "OpenAI or another TTS-capable provider in Settings."
            ),
        }
    info, qid = resolved
    provider_id = info.provider_id

    client = (ctx.clients or {}).get(provider_id)
    if provider_id == "openai" and client is None:
        return {"error": "OpenAI client not initialized."}

    ext = response_format if response_format in {"mp3", "wav", "opus", "aac", "flac"} else "mp3"
    out = _artifact_paths.artifact_path(ctx.working_dir, ctx.chat_id, kind="audio", ext=ext)
    try:
        if provider_id == "openai" and client is not None:
            with client.audio.speech.with_streaming_response.create(
                model=info.raw_id,
                voice=voice,
                input=text,
                response_format=ext,
                speed=speed,
            ) as response:
                response.stream_to_file(str(out))
        else:
            return {
                "error": (
                    f"Speech generation via {provider_id} is not yet "
                    "implemented. Try OpenAI."
                ),
            }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Speech generation failed: {e}"}

    cost_usd: float | None = None
    if info.tts_pricing_per_1m_chars is not None:
        cost_usd = round(len(text) / 1_000_000 * info.tts_pricing_per_1m_chars, 6)

    return {
        "ok": True,
        "kind": "audio",
        "path": str(out.relative_to(ctx.working_dir)),
        "mime_type": f"audio/{ext}",
        "model_used": qid,
        "voice": voice,
        "duration_seconds": None,
        "cost_usd": cost_usd,
    }


def _generate_video(
    ctx: ToolContext,
    prompt: str,
    *,
    model: str | None = None,
    duration_seconds: int = 5,
    aspect_ratio: str = "16:9",
) -> dict[str, Any]:
    """Generate a video via the provider's video API (best-effort).

    Video generation is a long-running operation (often 30–120s) and
    every provider has a different polling protocol. v1 invokes the
    provider's API and returns a clear "not yet implemented" error
    when the path isn't wired — the contract is in place, but we
    leave the provider-specific polling logic for a follow-up.
    """
    if not ctx.chat_id:
        return {"error": "generate_video requires an active chat context."}

    resolved = _resolve_media_model(ctx, model, "video_generation")
    if resolved is None:
        return {
            "error": (
                "No connected provider supports video generation. Configure "
                "OpenAI Sora or Google Veo in Settings."
            ),
        }
    # The actual video-generation flow per provider is non-trivial
    # (Sora is ``client.videos.generate(...)`` with polling; Veo is
    # ``client.models.generate_videos(...)`` with ``LongRunningOperation``).
    # For v1 we surface a clear error so the agent doesn't loop on it;
    # the model_catalog gate already makes sure ``model_used`` would be
    # accurate when this lands.
    return {
        "error": (
            "Video generation is wired through the model catalog but "
            "the per-provider polling logic isn't implemented yet. "
            "Track upstream progress at the provider's docs."
        ),
    }


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_files": _list_files,
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "run_shell": _run_shell,
    "generate_image": _generate_image,
    "generate_speech": _generate_speech,
    "generate_video": _generate_video,
}


@_op(name="dispatch_tool", kind="tool", color="green")
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
