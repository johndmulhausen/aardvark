"""AI text helpers for git commit messages and pull-request descriptions.

Pure module — no Streamlit imports — so it can be unit-tested in
isolation. The actual chat-completion call still routes through
:func:`agent.generate_text`, which is the single non-streaming caller
of provider chat-completion APIs (per the AGENTS.md "single LLM
caller" rule). The diff blob fed to the model is built by
:func:`git_ops.combined_diff_for_paths` so the per-file walk + the
200 KB cap live in one place.

Each public helper takes an explicit ``model`` argument (the chat's
currently selected qualified id, ``<provider>:<raw>``) and the
matching per-provider client. The chat-page sync pipeline is the
caller — it resolves ``ss.clients[provider_id]`` and passes both in.
:func:`agent.generate_text` extracts the provider id from the
qualified model id and dispatches to the right SDK underneath, so
this module never has to care which provider the user picked.

That contract is what makes the Sync flow's commit messages + PR
descriptions track the user's selected chat model: a user on
Anthropic / OpenAI / Google / Mistral / xAI / W&B / OpenRouter all
get *their* provider's model writing the commit message, rather
than a forced cross-provider DeepSeek round-trip. Historically this
module hard-coded ``agent.DEEPSEEK_MODEL``; that gated AI commits to
W&B accounts that had DeepSeek listed and surprised users who picked
a different model for their conversation.

Public surface:

- :data:`COMMIT_MSG_SYSTEM` / :data:`PR_DESC_SYSTEM` — prompt templates.
- :func:`generate_commit_message(client, working_dir, paths, *, model)`.
- :func:`generate_pr_description(client, working_dir, paths, branch, base, *, model)`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import git_ops
from agent import generate_text


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


def generate_commit_message(
    client: Any,
    working_dir: Path,
    paths: list[str],
    *,
    model: str,
    api_key: str = "",
) -> str:
    """Ask ``model`` for a conventional-commit message describing ``paths``.

    Returns ``""`` when the diff is empty (nothing to summarize) so the
    caller can fall back to a stub message instead of submitting an
    empty string.

    ``client`` must be the per-provider client matching the qualified
    ``model`` id (the chat-page sync pipeline is responsible for
    resolving ``ss.clients[provider_id]`` and passing it here);
    :func:`agent.generate_text` extracts the provider id from
    ``model`` and dispatches to the right SDK underneath. ``api_key``
    is accepted for back-compat with older callers and is otherwise
    unused — the client carries its own credentials.
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
        model=model,
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
    model: str,
    api_key: str = "",
) -> tuple[str, str]:
    """Ask ``model`` for a PR title + body. Returns ``(title, body)``.

    Tolerant JSON parsing: when the model emits invalid JSON we treat
    the whole response as the body and synthesize a title from its
    first non-empty line. That way the user always has *something*
    pre-filled in the platform compare URL even on a model misfire —
    they can still edit both fields before submitting the PR.

    See :func:`generate_commit_message` for the ``client`` / ``model``
    routing contract.
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
        model=model,
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
