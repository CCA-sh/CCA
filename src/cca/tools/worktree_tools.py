"""Worktree tools: EnterWorktree and ExitWorktree.

Creates and manages isolated git worktrees under .cca/worktrees/.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from cca.paths import WORKTREES_SUBDIR
from cca.tools.base import Tool

# Session state
_active_worktree: dict | None = None
_original_cwd: str | None = None


class EnterWorktreeTool(Tool):
    name = "EnterWorktree"
    description = (
        "Create or switch into an isolated git worktree. "
        "Supply name to create a new worktree with that name, or path to enter "
        "an existing one. If neither is given, a random name is generated. "
        "Only use when explicitly instructed to work in a worktree."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name for a new worktree (mutually exclusive with path).",
            },
            "path": {
                "type": "string",
                "description": "Path to an existing worktree to enter (mutually exclusive with name).",
            },
        },
    }

    async def execute(
        self,
        name: str | None = None,
        path: str | None = None,
    ) -> str:
        global _active_worktree, _original_cwd

        if _active_worktree:
            return "Error: already in a worktree session. Exit first with ExitWorktree."

        cwd = Path.cwd()

        # Check we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=str(cwd)
        )
        if result.returncode != 0:
            return "Error: not in a git repository."
        repo_root = Path(result.stdout.strip())

        if path:
            # Enter existing worktree
            wt_path = Path(path)
            if not wt_path.exists():
                return f"Error: path does not exist: {path}"
            _original_cwd = str(cwd)
            _active_worktree = {"path": str(wt_path), "created": False, "name": wt_path.name}
            os.chdir(str(wt_path))
            return f"Entered existing worktree: {wt_path}"

        # Create new worktree
        wt_name = name or str(uuid.uuid4())[:8]
        wt_dir = repo_root / WORKTREES_SUBDIR / wt_name
        branch_name = f"worktree/{wt_name}"

        cmd = ["git", "worktree", "add", "-b", branch_name, str(wt_dir), "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        if result.returncode != 0:
            return f"Error creating worktree: {result.stderr.strip()}"

        _original_cwd = str(cwd)
        _active_worktree = {
            "path": str(wt_dir),
            "name": wt_name,
            "branch": branch_name,
            "created": True,
        }
        os.chdir(str(wt_dir))

        # Copy .cca/ project config into the worktree so it inherits settings.
        src_cca = repo_root / ".cca"
        dst_cca = wt_dir / ".cca"
        if src_cca.exists() and not dst_cca.exists():
            import shutil
            try:
                shutil.copytree(str(src_cca), str(dst_cca))
            except Exception:
                pass  # non-fatal; worktree still usable without config

        return (
            f"Created worktree '{wt_name}' at {wt_dir} on branch '{branch_name}'. "
            f"Working directory changed to {wt_dir}."
        )


class ExitWorktreeTool(Tool):
    name = "ExitWorktree"
    description = (
        "Exit the current worktree session and return to the original directory. "
        "action='keep': leave worktree on disk. "
        "action='remove': delete the worktree and its branch. "
        "Set discard_changes=true to force removal even with uncommitted changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["keep", "remove"],
                "description": "'keep' or 'remove' the worktree.",
            },
            "discard_changes": {
                "type": "boolean",
                "description": "Force remove even with uncommitted changes.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        discard_changes: bool = False,
    ) -> str:
        global _active_worktree, _original_cwd

        if not _active_worktree:
            return "No active worktree session."

        wt = _active_worktree
        original = _original_cwd or str(Path.home())

        if action == "keep" or not wt.get("created"):
            os.chdir(original)
            _active_worktree = None
            _original_cwd = None
            return f"Exited worktree '{wt['name']}'. Worktree preserved at {wt['path']}."

        # action == "remove"
        wt_path = wt["path"]

        # Check for uncommitted changes
        if not discard_changes:
            diff = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=wt_path
            )
            if diff.stdout.strip():
                return (
                    f"Refusing to remove: uncommitted changes in worktree.\n"
                    f"{diff.stdout.strip()}\n"
                    "Set discard_changes=true to force removal."
                )

        os.chdir(original)

        result = subprocess.run(
            ["git", "worktree", "remove", "--force", wt_path],
            capture_output=True, text=True,
        )

        # Try to delete branch too
        if wt.get("branch"):
            subprocess.run(
                ["git", "branch", "-D", wt["branch"]],
                capture_output=True,
            )

        _active_worktree = None
        _original_cwd = None

        if result.returncode != 0:
            return f"Warning: worktree remove returned error: {result.stderr.strip()}"
        return f"Removed worktree '{wt['name']}' and branch '{wt.get('branch', '')}'."
