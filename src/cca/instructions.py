"""Project instruction file discovery and loading.

  Global (lowest priority):
    ~/.cca/instructions.md

  Per-directory (outer → inner, so cwd wins):
    AGENTS.md        — project-shared, committed to source control
    AGENTS.local.md  — personal/private, should be gitignored

Lines starting with '@' are resolved as file imports (one level deep, no
recursive expansion), matching the import syntax described by /init.
"""

from __future__ import annotations

import os
from pathlib import Path


_GLOBAL_INSTRUCTIONS = Path.home() / ".cca" / "instructions.md"

# Shared file first so a personal override follows its sibling.
_PER_DIRECTORY_NAMES = ("AGENTS.md", "AGENTS.local.md")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_imports(text: str, base_dir: Path) -> str:
    """Expand '@path' import lines one level deep."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@"):
            ref = stripped[1:].strip()
            target = Path(os.path.expanduser(ref))
            if not target.is_absolute():
                target = (base_dir / target).resolve()
            if target.is_file():
                try:
                    out.append(target.read_text().rstrip())
                    continue
                except OSError:
                    pass  # fall through: keep original line
        out.append(line)
    return "\n".join(out)


def _read(path: Path) -> str | None:
    """Read, strip, and resolve imports. Returns None if missing/empty/unreadable."""
    if not path.is_file():
        return None
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    return _resolve_imports(raw, path.parent)


def _ancestor_chain(cwd: str) -> list[Path]:
    """Return [outermost, ..., cwd] stopping at home so we skip system dirs."""
    home = Path.home().resolve()
    start = Path(cwd).resolve()

    chain: list[Path] = []
    current = start
    while True:
        chain.append(current)
        parent = current.parent
        # Stop at home dir (inclusive) or at fs root
        if current == home or parent == current:
            break
        current = parent

    chain.reverse()  # outermost first → cwd last
    return chain


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect(cwd: str | None = None) -> tuple[str, list[str]]:
    """Discover and combine all instruction files for *cwd*.

    Returns ``(system_prompt_block, loaded_paths)`` where:

    - ``system_prompt_block`` is the formatted text ready to be injected into
      the system prompt (empty string when no files are found).
    - ``loaded_paths`` is the list of absolute path strings that were loaded,
      in the order they appear in the block (for startup display).

    Files are ordered outermost → innermost so that project-root instructions
    appear before subdirectory overrides, and per-directory AGENTS.local.md
    follows its AGENTS.md sibling: later (inner) content takes precedence when
    the model reads top-to-bottom.
    """
    work_dir = cwd or os.getcwd()
    sections: list[str] = []
    loaded: list[str] = []

    def _add(path: Path, text: str) -> None:
        sections.append(f"<cca-instructions path=\"{path}\">\n{text}\n</cca-instructions>")
        loaded.append(str(path))

    # 1. Global personal instructions (lowest priority)
    global_text = _read(_GLOBAL_INSTRUCTIONS)
    if global_text is not None:
        _add(_GLOBAL_INSTRUCTIONS, global_text)

    # 2. Walk ancestor directories outermost → cwd
    for directory in _ancestor_chain(work_dir):
        for name in _PER_DIRECTORY_NAMES:
            candidate = directory / name
            text = _read(candidate)
            if text is not None:
                _add(candidate, text)

    if not sections:
        return "", []

    body = "\n\n".join(sections)
    block = (
        "# Project Instructions\n\n"
        "The following instructions come from project and personal configuration\n"
        "files discovered in the working directory hierarchy. Inner (closer to\n"
        "the working directory) instructions take precedence.\n\n"
        + body
    )
    return block, loaded
