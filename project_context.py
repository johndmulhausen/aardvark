"""Auto-detect AGENTS.md, .cursor/rules, and .cursor/skills in the working dir.

This module is the single source of truth for "what guidance does this
project expose to the agent?". It scans the working directory (and the
user's ``~/.cursor/skills``) for known guidance files, decides which to
inject into the system prompt for a given turn, and produces both:

- a system-prompt addendum string (for ``agent.py``), and
- a UI-friendly summary dict (for ``streamlit_app.py``).

Two categories of guidance:

1. **Always-on, eagerly injected.** ``AGENTS.md`` / ``CLAUDE.md`` /
   ``CONVENTIONS.md`` at the working-dir root, plus every
   ``.cursor/rules/*.mdc``. Their full content (capped) is appended to the
   system prompt on every turn.

2. **Conditionally loaded.** Workspace ``.cursor/skills/**/SKILL.md`` and
   user ``~/.cursor/skills/**/SKILL.md``. Only loaded when one of two
   triggers fires:
   - **Slash command** — the user typed ``/<slug>`` anywhere in their
     message. Always wins, never gated by the per-turn cap.
   - **Keyword match** — the user's message contains a phrase mined from
     the skill's frontmatter ``description`` / explicit ``triggers``.
     Capped at :data:`MAX_AUTO_SKILLS` per turn.

The loading strategy is intentional: we do *not* rely on the model to
proactively read a skill. Skills are sliced into the system prompt before
the call so guarantee-of-visibility doesn't depend on model behavior.

Why no PyYAML?
--------------
The frontmatter we care about uses a strict subset of YAML — ``key:
value`` lines with optional list values written as ``[a, b, c]`` or as
``- item`` block sequences. A small hand-rolled parser is enough and lets
us avoid pulling PyYAML into ``pyproject.toml`` for two fields.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

# Per-turn cap on keyword-matched skills. Slash-invoked skills bypass this
# cap because the user explicitly asked for them.
MAX_AUTO_SKILLS = 5

# Per-skill content cap. Most skills are well under this; the limit just
# protects the context window from a pathologically long SKILL.md.
SKILL_CONTENT_CAP = 6_000

# Caps for eagerly-injected guidance files.
AGENTS_MD_CAP = 12_000
RULES_FILE_CAP = 4_000

# AGENTS.md filename variants we recognize at the working-dir root. The
# ordering matters because we list them in the system-prompt addendum in
# the same order — AGENTS.md takes precedence, then CLAUDE.md (Anthropic's
# convention), then CONVENTIONS.md (Aider).
AGENT_GUIDE_FILENAMES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    "CONVENTIONS.md",
)

# Slash-command regex. A slug must start with a letter or digit, may
# contain hyphens or underscores, and must follow whitespace or the start
# of the message — the latter rejects file paths like ``/usr/local/bin``
# from being parsed as commands.
_SLASH_RE = re.compile(r"(?:^|\s)/([a-z0-9][a-z0-9_-]{0,63})\b", re.IGNORECASE)

# Slug normalization for skill names. Same logic as the slash regex's
# allowed character class, applied to the frontmatter ``name`` value.
_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class GuidanceFile:
    """A single eagerly-injected guidance file.

    ``content`` is already truncated to the relevant cap; ``truncated`` is
    True when the original file was larger.
    """

    path: Path
    content: str
    truncated: bool

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Skill:
    """A discovered skill available for slash- or keyword-loading.

    ``content_loader`` is a thunk so we don't read every SKILL.md in the
    user's skills directory just to scan; the body is only read when a
    skill is actually selected for a turn.

    ``triggers`` is the union of three sources, in this order:

    1. The frontmatter ``triggers:`` list, if present.
    2. Phrases parsed from a ``Triggers:`` sentence inside ``description``
       (the convention several Cursor skills already use).
    3. Each non-trivial word from the skill's name (split on ``-``).

    Slash-command matching uses :attr:`slug` directly; keyword matching
    iterates :attr:`triggers`.
    """

    slug: str
    name: str
    path: Path
    description: str
    triggers: list[str]
    scope: Literal["workspace", "user"]
    content_loader: Callable[[], str]


@dataclass
class ProjectContext:
    """Result of a working-dir scan.

    ``slug_index`` maps each skill's slug to its :class:`Skill`. When a
    workspace skill and a user skill share a slug, the workspace skill
    wins (project context overrides global) and the conflict is recorded
    in :attr:`slug_conflicts`. The user-skill object is *also* removed
    from :attr:`user_skills` so it doesn't show up twice in the UI.
    """

    working_dir: Path
    agents_md: list[GuidanceFile] = field(default_factory=list)
    cursor_rules: list[GuidanceFile] = field(default_factory=list)
    workspace_skills: list[Skill] = field(default_factory=list)
    user_skills: list[Skill] = field(default_factory=list)
    slug_index: dict[str, Skill] = field(default_factory=dict)
    slug_conflicts: list[str] = field(default_factory=list)


@dataclass
class SelectedSkill:
    """A skill chosen for the current turn, plus the reason it was chosen."""

    skill: Skill
    trigger_reason: str


@dataclass
class SkillSelection:
    """The set of skills loaded into a single agent turn."""

    selected: list[SelectedSkill] = field(default_factory=list)
    unknown_slash: list[str] = field(default_factory=list)


# ----- frontmatter parsing -----

def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse a leading ``---`` YAML-ish block out of ``text``.

    Returns ``(metadata, body)``. Metadata is a dict with a few supported
    value types: scalars are kept as strings, ``[a, b]`` flow-style lists
    are split on commas and stripped, block sequences (``- item``) are
    accumulated until the next non-indented line. Anything we don't
    understand is ignored rather than raised — the goal is to extract a
    couple of fields, not validate the format.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=False)
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text
    meta: dict[str, object] = {}
    current_key: str | None = None
    current_list: list[str] | None = None
    for raw in lines[1:end_idx]:
        if current_list is not None and raw.lstrip().startswith("- "):
            current_list.append(raw.lstrip()[2:].strip().strip('"').strip("'"))
            continue
        if current_list is not None and raw.strip() == "":
            continue
        if current_list is not None:
            current_list = None
            current_key = None
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value == "":
            current_key = key
            current_list = []
            meta[key] = current_list
            continue
        if value.startswith("[") and value.endswith("]"):
            inside = value[1:-1].strip()
            if not inside:
                meta[key] = []
            else:
                items = [
                    item.strip().strip('"').strip("'")
                    for item in inside.split(",")
                    if item.strip()
                ]
                meta[key] = items
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        meta[key] = value
    body = "\n".join(lines[end_idx + 1:])
    return meta, body


# ----- slug + trigger extraction -----

def _slugify(value: str) -> str:
    """Lowercase + collapse non-alphanumerics into ``-``."""
    if not value:
        return ""
    return _SLUG_NON_ALNUM.sub("-", value.lower()).strip("-")


def _extract_triggers(
    name: str,
    description: str,
    frontmatter_triggers: object | None,
) -> list[str]:
    """Build the trigger phrase list from a skill's frontmatter.

    Sources in priority order (de-duplicated, in insertion order):

    1. Explicit ``triggers`` YAML field (string-or-list).
    2. The portion of ``description`` following ``Triggers:`` (case-insensitive),
       split on commas. This matches the convention several existing Cursor
       skills follow. We deliberately don't truncate at the first ``.`` because
       phrases like ``st.``, ``app.py``, and ``pyproject.toml`` legitimately
       contain dots.
    3. Each ≥3-character word from the skill's slug-style ``name``, split on
       ``-`` / ``_`` / whitespace. Gives a reasonable fallback for sub-skills
       that don't pack a ``Triggers:`` line into their description.

    Phrases shorter than 2 characters are dropped (avoids matching ``a``,
    ``i`` against any English message). All phrases are lowercased; the
    matching pass also lowercases the user message.
    """
    phrases: list[str] = []
    if isinstance(frontmatter_triggers, list):
        phrases.extend(str(t) for t in frontmatter_triggers if isinstance(t, (str, int)))
    elif isinstance(frontmatter_triggers, str):
        phrases.extend(p.strip() for p in frontmatter_triggers.split(",") if p.strip())

    if description:
        marker = re.search(r"\btriggers?\s*[:\-]", description, re.IGNORECASE)
        if marker:
            tail = description[marker.end():].strip().rstrip(".").strip()
            phrases.extend(
                p.strip().strip('"').strip("'") for p in tail.split(",") if p.strip()
            )

    if name:
        for word in re.split(r"[-_\s]+", name):
            word = word.strip()
            if len(word) >= 3:
                phrases.append(word)

    seen: set[str] = set()
    cleaned: list[str] = []
    for phrase in phrases:
        normalized = phrase.lower().strip()
        if len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


# ----- file detection -----

def _read_capped(path: Path, cap: int) -> tuple[str, bool]:
    """Read a text file, truncating to ``cap`` chars; second return is the
    flag for whether the file was larger than the cap."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "", False
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n\n[truncated, {len(text) - cap} more chars]", True


def _scan_agent_guides(working_dir: Path) -> list[GuidanceFile]:
    """Find AGENTS.md / CLAUDE.md / CONVENTIONS.md at the working-dir root."""
    out: list[GuidanceFile] = []
    for filename in AGENT_GUIDE_FILENAMES:
        path = working_dir / filename
        if not path.is_file():
            continue
        content, truncated = _read_capped(path, AGENTS_MD_CAP)
        if not content:
            continue
        out.append(GuidanceFile(path=path, content=content, truncated=truncated))
    return out


def _scan_cursor_rules(working_dir: Path) -> list[GuidanceFile]:
    """Find ``.cursor/rules/*.mdc`` files at the working-dir root."""
    rules_dir = working_dir / ".cursor" / "rules"
    if not rules_dir.is_dir():
        return []
    out: list[GuidanceFile] = []
    for path in sorted(rules_dir.glob("*.mdc")):
        if not path.is_file():
            continue
        content, truncated = _read_capped(path, RULES_FILE_CAP)
        if not content:
            continue
        out.append(GuidanceFile(path=path, content=content, truncated=truncated))
    return out


def _scan_skills(root: Path, scope: Literal["workspace", "user"]) -> list[Skill]:
    """Find every ``SKILL.md`` under ``root``.

    ``content_loader`` is a closure over ``path``; the SKILL.md body is
    only read when a skill is selected for a turn, not during scanning.
    """
    if not root.is_dir():
        return []
    out: list[Skill] = []
    for skill_path in sorted(root.glob("**/SKILL.md")):
        if not skill_path.is_file():
            continue
        try:
            text = skill_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        meta, _ = _parse_frontmatter(text)
        name = str(meta.get("name") or skill_path.parent.name)
        slug = _slugify(name)
        if not slug:
            continue
        description = str(meta.get("description") or "")
        triggers = _extract_triggers(name, description, meta.get("triggers"))

        def _make_loader(p: Path) -> Callable[[], str]:
            def _load() -> str:
                content, truncated = _read_capped(p, SKILL_CONTENT_CAP)
                if truncated and not content.endswith("more chars]"):
                    content += "\n[truncated]"
                return content
            return _load

        out.append(
            Skill(
                slug=slug,
                name=name,
                path=skill_path,
                description=description,
                triggers=triggers,
                scope=scope,
                content_loader=_make_loader(skill_path),
            )
        )
    return out


# ----- public API -----

def scan(working_dir: Path) -> ProjectContext:
    """Scan a working directory for guidance files and skills.

    The scan is fast (a handful of stat calls plus reading any present
    AGENTS.md / rules) and is safe to run on every turn — see
    ``agent.py``'s call site.
    """
    working_dir = Path(working_dir).expanduser().resolve()
    ctx = ProjectContext(working_dir=working_dir)
    if not working_dir.is_dir():
        return ctx

    ctx.agents_md = _scan_agent_guides(working_dir)
    ctx.cursor_rules = _scan_cursor_rules(working_dir)
    ctx.workspace_skills = _scan_skills(working_dir / ".cursor" / "skills", "workspace")
    user_skills = _scan_skills(Path.home() / ".cursor" / "skills", "user")

    slug_index: dict[str, Skill] = {}
    for skill in ctx.workspace_skills:
        slug_index[skill.slug] = skill

    deduped_user: list[Skill] = []
    conflicts: list[str] = []
    for skill in user_skills:
        if skill.slug in slug_index:
            conflicts.append(skill.slug)
            continue
        slug_index[skill.slug] = skill
        deduped_user.append(skill)

    ctx.user_skills = deduped_user
    ctx.slug_index = slug_index
    ctx.slug_conflicts = conflicts
    return ctx


def parse_slash_commands(message: str, ctx: ProjectContext) -> tuple[list[Skill], list[str]]:
    """Pull ``/<slug>`` tokens from a message.

    Returns ``(matched_skills, unknown_slugs)``. Duplicates within the
    message are collapsed.
    """
    if not message:
        return [], []
    seen: set[str] = set()
    matched: list[Skill] = []
    unknown: list[str] = []
    for raw_match in _SLASH_RE.findall(message):
        slug = raw_match.lower()
        if slug in seen:
            continue
        seen.add(slug)
        skill = ctx.slug_index.get(slug)
        if skill is None:
            unknown.append(slug)
        else:
            matched.append(skill)
    return matched, unknown


def match_skills(message: str, ctx: ProjectContext) -> list[tuple[Skill, str]]:
    """Find skills whose trigger phrases appear in ``message``.

    Returns a list of ``(skill, matched_phrase)`` tuples ordered by total
    trigger-match count (more matches → higher priority), with ties broken
    by skill name for stable ordering. Each skill appears at most once;
    the returned phrase is whichever trigger fired first.
    """
    if not message:
        return []
    haystack = message.lower()
    scored: list[tuple[int, str, Skill, str]] = []
    for skill in ctx.slug_index.values():
        first_phrase: str | None = None
        count = 0
        for phrase in skill.triggers:
            if not phrase:
                continue
            if _phrase_in_message(phrase, haystack):
                count += 1
                if first_phrase is None:
                    first_phrase = phrase
        if first_phrase is not None:
            scored.append((-count, skill.name.lower(), skill, first_phrase))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [(skill, phrase) for _, _, skill, phrase in scored]


def _phrase_in_message(phrase: str, haystack_lower: str) -> bool:
    """Whole-word(ish) match of ``phrase`` inside an already-lowercased
    message.

    We use a Unicode-friendly regex with ``\\b`` boundaries when the
    phrase consists of word characters. For phrases containing punctuation
    (e.g. ``"st."``) we fall back to a substring match because ``\\b``
    doesn't work after punctuation. This intentionally errs on the side
    of "match more" since false negatives mean the model never sees the
    relevant skill — a worse failure mode than accidentally loading a
    skill that turns out to be irrelevant.
    """
    if not phrase:
        return False
    if re.fullmatch(r"[\w\s]+", phrase):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        return re.search(pattern, haystack_lower) is not None
    return phrase in haystack_lower


def select_skills_for_turn(message: str, ctx: ProjectContext) -> SkillSelection:
    """Combine slash and keyword triggers into the final per-turn set.

    Slash-invoked skills always load (no cap). Keyword-matched skills fill
    the remaining budget up to :data:`MAX_AUTO_SKILLS`, skipping anything
    already pinned by a slash command.
    """
    selection = SkillSelection()
    slash_matched, unknown = parse_slash_commands(message, ctx)
    selection.unknown_slash = unknown

    pinned_slugs: set[str] = set()
    for skill in slash_matched:
        if skill.slug in pinned_slugs:
            continue
        pinned_slugs.add(skill.slug)
        selection.selected.append(SelectedSkill(skill=skill, trigger_reason="slash"))

    remaining = MAX_AUTO_SKILLS
    for skill, phrase in match_skills(message, ctx):
        if remaining <= 0:
            break
        if skill.slug in pinned_slugs:
            continue
        selection.selected.append(
            SelectedSkill(skill=skill, trigger_reason=f"keyword: {phrase}")
        )
        pinned_slugs.add(skill.slug)
        remaining -= 1

    return selection


def build_system_addendum(ctx: ProjectContext, selection: SkillSelection) -> str:
    """Build the per-turn system-prompt addendum.

    Layout (sections are omitted when empty):

    1. ``## Project guidance`` — full content of every AGENTS.md / CLAUDE.md
       / CONVENTIONS.md found at the working-dir root.
    2. ``## Project rules`` — full content of every ``.cursor/rules/*.mdc``.
    3. ``## Active skills for this turn`` — full content of every selected
       skill, each block prefaced with the trigger reason so the model
       knows the user explicitly asked (``slash``) versus an automated
       keyword match.
    """
    parts: list[str] = []

    if ctx.agents_md:
        parts.append("\n\n## Project guidance")
        parts.append(
            "These files describe how the user wants you to work in this project. "
            "Treat them as authoritative; if a file conflicts with general best "
            "practice, follow the file."
        )
        for guide in ctx.agents_md:
            try:
                rel = guide.path.relative_to(ctx.working_dir)
            except ValueError:
                rel = guide.path
            parts.append(f"\n### {rel}\n\n{guide.content}")

    if ctx.cursor_rules:
        parts.append("\n\n## Project rules")
        parts.append("Cursor `.cursor/rules/*.mdc` rules attached to this project.")
        for rule in ctx.cursor_rules:
            try:
                rel = rule.path.relative_to(ctx.working_dir)
            except ValueError:
                rel = rule.path
            parts.append(f"\n### {rel}\n\n{rule.content}")

    if selection.selected:
        parts.append("\n\n## Active skills for this turn")
        parts.append(
            "The user's message matched these skills. Each one is a procedural "
            "guide; consult them while answering and follow their instructions. "
            "If a skill says to read or edit specific files, do that before "
            "responding."
        )
        for picked in selection.selected:
            skill = picked.skill
            try:
                content = skill.content_loader()
            except Exception as e:
                content = f"[failed to load skill: {e}]"
            parts.append(
                f"\n### /{skill.slug} ({skill.scope}, trigger: {picked.trigger_reason})\n"
                f"Path: `{skill.path}`\n\n{content}"
            )

    if not parts:
        return ""
    return "".join(parts)


def summary(ctx: ProjectContext) -> dict[str, object]:
    """Render a UI-friendly summary of what was detected.

    Used by the sidebar "Project context" panel and the chat-controls
    Skills popover. Each skill entry includes its slug so the popover can
    show the slash command alongside the description.
    """
    def _guide(g: GuidanceFile) -> dict[str, object]:
        try:
            rel = str(g.path.relative_to(ctx.working_dir))
        except ValueError:
            rel = str(g.path)
        return {"name": g.path.name, "path": rel, "truncated": g.truncated}

    def _skill(s: Skill) -> dict[str, object]:
        try:
            rel = str(s.path.relative_to(ctx.working_dir))
        except ValueError:
            try:
                rel = str(s.path.relative_to(Path.home()))
                rel = "~/" + rel
            except ValueError:
                rel = str(s.path)
        return {
            "slug": s.slug,
            "name": s.name,
            "description": s.description,
            "triggers": list(s.triggers),
            "scope": s.scope,
            "path": rel,
        }

    return {
        "agents_md": [_guide(g) for g in ctx.agents_md],
        "cursor_rules": [_guide(g) for g in ctx.cursor_rules],
        "workspace_skills": [_skill(s) for s in ctx.workspace_skills],
        "user_skills": [_skill(s) for s in ctx.user_skills],
        "slug_conflicts": list(ctx.slug_conflicts),
        "all_skills": [_skill(s) for s in iter_all_skills(ctx)],
    }


def iter_all_skills(ctx: ProjectContext) -> Iterable[Skill]:
    """Iterate over workspace skills first, then user skills.

    Convenience for UI rendering — the popover wants a single list and
    we'd otherwise have to do this in the caller.
    """
    yield from ctx.workspace_skills
    yield from ctx.user_skills
