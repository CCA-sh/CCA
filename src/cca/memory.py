"""File-based memory system.

Two storage locations:
  Private:  ~/.cca/memory/           (user-specific, never committed)
  Team:     <project-root>/.cca/memory/  (shared, can be committed)

Each memory is a markdown file with YAML frontmatter. MEMORY.md in each
directory is a one-line-per-entry index.

Types: user | feedback | project | reference
Scopes: private | team
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from cca.paths import MEMORY_DIR, TEAM_MEMORY_SUBDIR


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _find_project_root(cwd: str | None = None) -> Path | None:
    """Walk up from cwd to find the nearest .git directory."""
    start = Path(cwd or ".").resolve()
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def get_team_memory_dir(cwd: str | None = None) -> Path | None:
    """Return the team memory directory if a project root exists."""
    root = _find_project_root(cwd)
    if root is None:
        return None
    return root / TEAM_MEMORY_SUBDIR


def get_private_memory_dir() -> Path:
    return MEMORY_DIR


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end].strip()
    body = text[end + 4:].strip()
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def _build_frontmatter(fm: dict) -> str:
    return "---\n" + yaml.dump(fm, default_flow_style=False).rstrip() + "\n---\n\n"


# ---------------------------------------------------------------------------
# Staleness warning
# ---------------------------------------------------------------------------

_MEMORY_REVIEW_INTERVAL = timedelta(days=7)


def _staleness_warning(path: Path) -> str:
    """Return a review note when a memory has not changed recently."""
    try:
        modified_at = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )
    except OSError:
        return ""

    if datetime.now(timezone.utc) - modified_at < _MEMORY_REVIEW_INTERVAL:
        return ""

    modified_date = modified_at.date().isoformat()
    return (
        "<system-reminder>"
        f"Memory review note: this entry was last modified on {modified_date}. "
        "Use it as background, but confirm details that depend on the repository's "
        "current files, commands, configuration, or dependencies."
        "</system-reminder>\n"
    )


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def load_index(cwd: str | None = None) -> str:
    """Return combined MEMORY.md index from private + team directories."""
    parts = []
    for directory in _all_memory_dirs(cwd):
        index_path = directory / "MEMORY.md"
        if index_path.exists():
            parts.append(index_path.read_text().strip())
    return "\n\n".join(parts) if parts else ""


def _update_index(directory: Path, filename: str, name: str, description: str) -> None:
    index_path = directory / "MEMORY.md"
    entry = f"- [{name}]({filename}.md) — {description}"

    if not index_path.exists():
        index_path.write_text(f"# Memory Index\n\n{entry}\n")
        return

    content = index_path.read_text()
    lines = [
        ln for ln in content.splitlines()
        if not re.search(rf"\({re.escape(filename)}\.md\)", ln)
    ]
    lines.append(entry)
    index_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# All memory directories (private + team)
# ---------------------------------------------------------------------------

def _all_memory_dirs(cwd: str | None = None) -> list[Path]:
    dirs = [MEMORY_DIR]
    team_dir = get_team_memory_dir(cwd)
    if team_dir and team_dir != MEMORY_DIR:
        dirs.append(team_dir)
    return dirs


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def save_memory(
    name: str,
    description: str,
    memory_type: str,
    body: str,
    filename: str | None = None,
    scope: str = "private",
    cwd: str | None = None,
) -> Path:
    """Write a memory file and update MEMORY.md.

    Args:
        name: Memory name.
        description: One-line description (used for index).
        memory_type: user | feedback | project | reference
        body: Markdown body content.
        filename: Override filename (without .md). Defaults to slugified name.
        scope: 'private' (default) or 'team'.
        cwd: Working directory for locating team memory dir.

    Returns:
        Path to the written memory file.
    """
    if scope == "team":
        directory = get_team_memory_dir(cwd)
        if directory is None:
            directory = MEMORY_DIR  # fallback to private if no git root
    else:
        directory = MEMORY_DIR

    directory.mkdir(parents=True, exist_ok=True)

    if filename is None:
        slug = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
        filename = f"{memory_type}_{slug}"

    path = directory / f"{filename}.md"

    fm = {
        "name": name,
        "description": description,
        "type": memory_type,
        "scope": scope,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(_build_frontmatter(fm) + body)
    _update_index(directory, filename, name, description)
    return path


def list_memories(cwd: str | None = None) -> list[dict]:
    """Return list of memory metadata dicts from all directories."""
    results = []
    for directory in _all_memory_dirs(cwd):
        if not directory.exists():
            continue
        for p in sorted(directory.glob("*.md")):
            if p.name == "MEMORY.md":
                continue
            text = p.read_text()
            fm, body = _parse_frontmatter(text)
            fm["_file"] = p.name
            fm["_path"] = p
            fm["_preview"] = body[:120]
            results.append(fm)
    return results


def delete_memory(filename: str, cwd: str | None = None) -> bool:
    """Delete a memory file and remove from index."""
    for directory in _all_memory_dirs(cwd):
        for candidate in [directory / filename, directory / f"{filename}.md"]:
            if candidate.exists():
                stem = candidate.stem
                candidate.unlink()
                index_path = directory / "MEMORY.md"
                if index_path.exists():
                    content = index_path.read_text()
                    lines = [
                        ln for ln in content.splitlines()
                        if not re.search(rf"\({re.escape(stem)}\.md\)", ln)
                    ]
                    index_path.write_text("\n".join(lines) + "\n")
                return True
    return False


# ---------------------------------------------------------------------------
# Memory manifest for extractor LLM
# ---------------------------------------------------------------------------

def build_memory_manifest(cwd: str | None = None) -> str:
    """Return a formatted manifest of all memory files.

    Entries that have not changed within the review interval include a
    repository-freshness reminder.
    """
    memories = list_memories(cwd)
    if not memories:
        return ""

    parts = []
    for m in memories:
        path: Path = m["_path"]
        warning = _staleness_warning(path)
        text = path.read_text()
        fm, body = _parse_frontmatter(text)
        parts.append(
            f"### {m['_file']}\n"
            f"{warning}"
            f"**Name:** {fm.get('name', '?')}\n"
            f"**Type:** {fm.get('type', '?')} | "
            f"**Scope:** {fm.get('scope', 'private')}\n"
            f"**Description:** {fm.get('description', '')}\n\n"
            f"{body}"
        )

    return "Available memories:\n\n" + "\n\n---\n\n".join(parts)
