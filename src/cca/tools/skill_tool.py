"""Skill tool — invoke predefined skill templates stored as markdown files.

Skills design to support Claude Code's skill system (which user can share the existing settings):
  ~/.cca/skills/<name>/SKILL.md     (cca-native; wins on name collision)
  ~/.claude/skills/<name>/SKILL.md  (Claude Code shared; read-only)

Each SKILL.md has optional YAML frontmatter followed by a prompt body.
When invoked, the prompt body is submitted to the LLM as a user turn.

Example ~/.cca/skills/commit/SKILL.md:
---
name: commit
description: Generate and create a git commit for staged changes
---
Review the staged changes with `git diff --cached` and create a well-formed
commit. Follow the conventional commits format. Co-Authored-By: cca.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from cca.paths import SKILLS_DIR, CLAUDE_SKILLS_DIR
from cca.tools.base import Tool


class SkillEntry(NamedTuple):
    name: str
    description: str
    path: Path
    source: str  # "cca" or "claude"


def _parse_skill(path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, prompt_body)."""
    text = path.read_text()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            import yaml
            try:
                fm = yaml.safe_load(text[3:end]) or {}
            except Exception:
                fm = {}
            return fm, text[end + 4:].strip()
    return {}, text.strip()


def _skill_path(base_dir: Path, name: str) -> Path | None:
    """Return the SKILL.md path for a skill name if it exists."""
    p = base_dir / name / "SKILL.md"
    return p if p.exists() else None


def list_skills() -> list[SkillEntry]:
    """Return all available skills, cca-native first (wins on collision)."""
    seen: set[str] = set()
    entries: list[SkillEntry] = []

    for base_dir, source in [(SKILLS_DIR, "cca"), (CLAUDE_SKILLS_DIR, "claude")]:
        if not base_dir.exists():
            continue
        for skill_dir in sorted(base_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if name in seen:
                continue
            seen.add(name)
            try:
                fm, _ = _parse_skill(skill_file)
                description = fm.get("description", "")
            except Exception:
                description = ""
            entries.append(SkillEntry(name=name, description=description, path=skill_file, source=source))

    return entries


def skill_listing_block() -> str:
    """Return a system-prompt block listing available skills, or empty string."""
    skills = list_skills()
    if not skills:
        return ""
    lines = ["# Available skills"]
    lines.append(
        "You can invoke skills with the Skill tool. "
        "Skills are predefined prompt templates that extend your capabilities.\n"
    )
    for s in skills:
        tag = f" [cca]" if s.source == "cca" else ""
        desc = f" — {s.description}" if s.description else ""
        lines.append(f"- **{s.name}**{tag}{desc}")
    return "\n".join(lines)


class SkillTool(Tool):
    name = "Skill"
    description = (
        "Invoke a named skill — a predefined prompt template. "
        "Skills live in ~/.cca/skills/<name>/SKILL.md (cca-native) or "
        "~/.claude/skills/<name>/SKILL.md (Claude Code shared). "
        "cca-native skills take precedence over same-named Claude Code skills."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name (e.g. 'commit', 'review-pr').",
            },
            "args": {
                "type": "string",
                "description": "Optional arguments appended to the skill prompt.",
            },
        },
        "required": ["skill"],
    }

    async def execute(self, skill: str, args: str | None = None) -> str:
        # cca-native wins; fall back to Claude Code shared
        path = _skill_path(SKILLS_DIR, skill) or _skill_path(CLAUDE_SKILLS_DIR, skill)

        if path is None:
            available = [s.name for s in list_skills()]
            avail_str = ", ".join(available) if available else "(none)"
            return (
                f"Skill '{skill}' not found.\n"
                f"Available skills: {avail_str}\n\n"
                f"Create a cca skill at: {SKILLS_DIR / skill / 'SKILL.md'}"
            )

        fm, prompt_body = _parse_skill(path)
        if args:
            prompt_body = f"{prompt_body}\n\n{args}"

        source = "cca" if SKILLS_DIR in path.parents else "claude"
        return f"[Skill '{skill}' loaded from {source}]\n\n{prompt_body}"


class ToolSearchTool(Tool):
    """In this implementation all tools are always loaded, so ToolSearch
    acts as a schema lookup / documentation tool."""

    name = "ToolSearch"
    description = (
        "Look up the full schema and documentation for tools by name or keyword. "
        "Use 'select:Read,Edit' for exact lookup, or keywords to search. "
        "Returns complete parameter documentation."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Query to find tools. Use 'select:<name>[,<name>...]' for exact "
                    "lookup, or keywords to search by description."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, registry=None):
        self._registry = registry

    async def execute(self, query: str, max_results: int = 5) -> str:
        import json

        if self._registry is None:
            return "Error: ToolSearch requires a registry reference."

        schemas = self._registry.get_schemas()

        if query.startswith("select:"):
            names = [n.strip() for n in query[7:].split(",")]
            matched = [s for s in schemas if s.name in names]
        else:
            keywords = query.lower().split()
            matched = [
                s for s in schemas
                if any(kw in s.name.lower() or kw in s.description.lower() for kw in keywords)
            ]

        matched = matched[:max_results]

        if not matched:
            return f"No tools matched query: {query}"

        parts = []
        for s in matched:
            doc = (
                f"## {s.name}\n"
                f"{s.description}\n\n"
                f"**Input schema:**\n```json\n{json.dumps(s.input_schema, indent=2)}\n```"
            )
            parts.append(doc)
        return "\n\n---\n\n".join(parts)
