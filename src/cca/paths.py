"""Canonical filesystem paths for CCA.

All paths can be overridden via environment variables, useful for
CI runs, SFT isolation, or multi-instance setups.

Override map:
  CCA_DIR                overrides ~/.cca
  CCA_SESSIONS_DIR       overrides ~/.cca/sessions
  CCA_HISTORY_FILE       overrides ~/.cca/history.jsonl
  CCA_MEMORY_DIR         overrides ~/.cca/memory
  CCA_PLANS_DIR          overrides ~/.cca/plans
  CCA_SKILLS_DIR         overrides ~/.cca/skills
  CCA_TRANSCRIPTS_DIR    overrides ~/.cca/transcripts
  CCA_LLM_CALLS_DIR      overrides ~/.cca/llm-calls
  CCA_VARIANTS_DIR       overrides ~/.cca/variants
  CCA_TEAM_MEMORY_SUBDIR project-relative subpath for team memory (default: .cca/memory)
  CCA_WORKTREES_SUBDIR   project-relative subpath for worktrees  (default: .cca/worktrees)
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


# ── Global user data directory ────────────────────────────────────────────
CCA_DIR: Path = _env_path("CCA_DIR", Path.home() / ".cca")

# ── Subdirectories inside CCA_DIR ─────────────────────────────────────────
SESSIONS_DIR:    Path = _env_path("CCA_SESSIONS_DIR",    CCA_DIR / "sessions")
HISTORY_FILE:    Path = _env_path("CCA_HISTORY_FILE",    CCA_DIR / "history.jsonl")
MEMORY_DIR:      Path = _env_path("CCA_MEMORY_DIR",      CCA_DIR / "memory")
PLANS_DIR:       Path = _env_path("CCA_PLANS_DIR",       CCA_DIR / "plans")
SKILLS_DIR:      Path = _env_path("CCA_SKILLS_DIR",      CCA_DIR / "skills")
TRANSCRIPTS_DIR: Path = _env_path("CCA_TRANSCRIPTS_DIR", CCA_DIR / "transcripts")
LLM_CALLS_DIR:   Path = _env_path("CCA_LLM_CALLS_DIR",  CCA_DIR / "llm-calls")

# User-supplied randomization variants. Deliberately absent from ensure_dirs():
# it is created only when someone actually adds a variant, and shadows the
# bundled bank by variant id.
VARIANTS_DIR:    Path = _env_path("CCA_VARIANTS_DIR",   CCA_DIR / "variants")

# ── Project-relative subpath strings (appended to repo_root at runtime) ──
# These are strings, not absolute Paths, because they are joined with a
# dynamically-discovered repo root at call time.
TEAM_MEMORY_SUBDIR: str = os.environ.get("CCA_TEAM_MEMORY_SUBDIR", ".cca/memory")
WORKTREES_SUBDIR:   str = os.environ.get("CCA_WORKTREES_SUBDIR",   ".cca/worktrees")

# ── shared skills directory to support Claude Code users (read-only; lower priority than SKILLS_DIR) ──
CLAUDE_SKILLS_DIR: Path = Path.home() / ".claude" / "skills"


def ensure_dirs() -> None:
    """Create all required CCA data directories. Idempotent."""
    for d in [CCA_DIR, SESSIONS_DIR, MEMORY_DIR, PLANS_DIR, SKILLS_DIR, TRANSCRIPTS_DIR, LLM_CALLS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
