"""AI text helpers for git commit messages and pull-request descriptions.

Pure module — no Streamlit imports — so it can be unit-tested in
isolation. The actual chat-completion call still routes through
:func:`agent.generate_text`, which is the single non-streaming caller
of ``client.chat.completions.create`` (per the AGENTS.md "single LLM
caller" rule). The diff blob fed to the model is built by
:func:`git_ops.combined_diff_for_paths` so the per-file walk + the
200 KB cap live in one place.

Public surface:

- :data:`COMMIT_MSG_SYSTEM` / :data:`PR_DESC_SYSTEM` — prompt templates.
- :func:`generate_commit_message` — one-shot conventional-commit
  message for the staged paths.
- :func:`generate_pr_description` — one-shot ``(title, body)`` tuple,
  with tolerant JSON parsing so a malformed model output still returns
  *something* the user can edit.
- :func:`is_deepseek_available` — trivial membership check the chat
  page calls before showing the AI affordances.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import git_ops
from agent import DEEPSEEK_MODEL, generate_text


COMMIT_MSG_SYSTEM = (
    "You are a senior engineer writing a git commit message.\n"
    "Output a single conventional-commit message:\n"
    "- First line: `<type>: <subject>` (<= 72 chars), where <type> is one of "
    "feat, fix, refactor, docs, test, chore, perf, style, build, ci.\n"
    "- Blank line.\n"
    "- 1-4 bullet body explaining the *why* of the change.\n"
    "Output the message text only. No code fences, no preamble, no quotes."
)


PR_DESC_SYSTEM = (
    "You are a senior engineer writing a pull-request description.\n"
    "Reply with strictly valid JSON of the shape "
    '`{"title": "...", "body": "..."}`.\n'
    "- title: <= 80 chars, no trailing period.\n"
    "- body: GitHub-flavored markdown with three sections (in order):\n"
    "  `## Summary` (1-3 bullets describing what and why),\n"
    "  `## Changes` (per-file or per-area bullets),\n"
    "  `## Test plan` (a checklist with `- [ ]` items).\n"
    "Do not wrap the JSON in code fences. Do not include any text before or "
    "after the JSON object."
)


def is_deepseek_available(models: list[str] | None) -> bool:
    """Return True if DeepSeek V4-Flash is in the connected account's model list.

    Accepts either qualified ids (``"wandb:deepseek-ai/DeepSeek-V4-Flash"``)
    or bare ids (``"deepseek-ai/DeepSeek-V4-Flash"``) — the chat page may
    pass ``ss.provider_models["wandb"]`` (raw ids) or the catalog's
    qualified ids depending on which call site is checking.
    """
    if not models:
        return False
    # ``DEEPSEEK_MODEL`` is the qualified ``wandb:<raw>`` form.
    raw_id = DEEPSEEK_MODEL.split(":", 1)[1] if ":" in DEEPSEEK_MODEL else DEEPSEEK_MODEL
    return DEEPSEEK_MODEL in models or raw_id in models


def generate_commit_message(
    client: Any,
    working_dir: Path,
    paths: list[str],
    *,
    api_key: str = "",
) -> str:
    """Ask DeepSeek for a conventional-commit message describing ``paths``.

    Returns ``""`` when the diff is empty (nothing to summarize) so the
    caller can fall back to a stub message instead of submitting an
    empty string. ``client`` is the user's W&B-configured
    ``openai.OpenAI`` instance (W&B Inference is ``openai_compat``,
    dispatched through the OpenAI SDK with W&B's ``base_url``);
    ``DEEPSEEK_MODEL`` is the qualified ``wandb:...`` id that pins
    routing to that provider. ``api_key`` is accepted for back-compat
    with older callers and is otherwise unused.
    """
    diff = git_ops.combined_diff_for_paths(working_dir, paths)
    if not diff.strip():
        return ""
    user = (
        "Write a conventional-commit message for the following diff:\n\n"
        "```diff\n" + diff + "\n```"
    )
    return generate_text(
        client=client,
        model=DEEPSEEK_MODEL,
        system=COMMIT_MSG_SYSTEM,
        user=user,
        max_tokens=500,
        api_key=api_key,
    )


def generate_pr_description(
    client: Any,
    working_dir: Path,
    paths: list[str],
    branch: str,
    base: str,
    *,
    api_key: str = "",
) -> tuple[str, str]:
    """Ask DeepSeek for a PR title + body. Returns ``(title, body)``.

    Tolerant JSON parsing: when the model emits invalid JSON we treat
    the whole response as the body and synthesize a title from its
    first non-empty line. That way the user always has *something*
    pre-filled in the GitHub compare URL even on a model misfire —
    they can still edit both fields before submitting the PR.
    """
    diff = git_ops.combined_diff_for_paths(working_dir, paths)
    if not diff.strip():
        return "", ""
    user = (
        f"Branch: `{branch}` -> `{base}`\n\n"
        "Summarize the following diff into a pull-request title and body:\n\n"
        "```diff\n" + diff + "\n```"
    )
    raw = generate_text(
        client=client,
        model=DEEPSEEK_MODEL,
        system=PR_DESC_SYSTEM,
        user=user,
        max_tokens=1500,
        api_key=api_key,
    )
    if not raw:
        return "", ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return (
                str(parsed.get("title") or "").strip(),
                str(parsed.get("body") or "").strip(),
            )
    except json.JSONDecodeError:
        pass
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    title = lines[0][:80] if lines else "Untitled change"
    return title, raw
