"""Permission system for tool execution.

Tools fall into three categories:
  - always_allow: execute silently
  - prompt: ask the user before executing
  - never_allow: always deny
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Coroutine

from rich.console import Console
from rich.markup import escape

# Tools whose file_path argument must be inside cwd
_PATH_RESTRICTED_TOOLS = {"Edit", "Write", "NotebookEdit"}

# Tools that are safe and fast — no confirmation needed by default
DEFAULT_ALWAYS_ALLOW = {
    "Read",
    "Glob",
    "Grep",
    "ToolSearch",
    "WorkCreate",
    "WorkGet",
    "WorkList",
    "WorkUpdate",
    "JobOutput",
    "CronList",
    "AskUserQuestion",
    "ScheduleWakeup",
    "WebFetch",
    "WebSearch",
}

# Tools that are write / destructive — prompt by default
DEFAULT_PROMPT_TOOLS = {
    "Write",
    "Edit",
    "NotebookEdit",
    "Bash",
    "Agent",
    "CronCreate",
    "CronDelete",
    "Monitor",
    "JobCancel",
    "EnterWorktree",
    "ExitWorktree",
    "EnterPlanMode",
    "ExitPlanMode",
    "Skill",
}


class PermissionManager:
    def __init__(
        self,
        always_allow: set[str] | None = None,
        never_allow: set[str] | None = None,
        auto_approve: bool = False,
        console: Console | None = None,
        # permission_fn: async (tool_name) -> str
        # Used by the interactive UI to collect the user's answer.
        permission_fn: Callable[[str], Coroutine] | None = None,
        # Legacy compat — ignored when permission_fn is set
        suspend_fn: Callable | None = None,
        run_in_terminal_fn: Callable | None = None,
        # Working directory — write tools are restricted to paths inside this dir
        cwd: str | None = None,
        # Additional directories write tools may access (--add-dir)
        add_dirs: set[Path] | None = None,
    ):
        self.always_allow: set[str] = DEFAULT_ALWAYS_ALLOW | (always_allow or set())
        self.never_allow: set[str] = never_allow or set()
        self.auto_approve = auto_approve
        self._console = console or Console()
        self._permission_fn = permission_fn
        self._session_grants: dict[str, bool] = {}
        self._cwd: Path = Path(cwd or os.getcwd()).resolve()
        # CCA_DIR is always trusted — plan files, memory, sessions all live there.
        from cca.paths import CCA_DIR
        self._add_dirs: frozenset[Path] = frozenset(
            Path(d).resolve() for d in (add_dirs or [])
        ) | frozenset([CCA_DIR.resolve()])

    def update_cwd(self, cwd: str) -> None:
        self._cwd = Path(cwd).resolve()

    def _path_allowed(self, target: Path) -> bool:
        try:
            target.relative_to(self._cwd)
            return True
        except ValueError:
            pass
        for d in self._add_dirs:
            try:
                target.relative_to(d)
                return True
            except ValueError:
                pass
        return False

    async def check(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Return True if the tool call should proceed."""
        if tool_name in self.never_allow:
            self._console.print(f"[red]Denied:[/red] {tool_name} is in the never-allow list.")
            return False

        # Hard-block write tools during plan mode — regardless of auto_approve.
        # Exception: the plan file itself (inside PLANS_DIR) must remain writable,
        # since that is exactly what the agent is supposed to produce.
        from cca.tools.plan_tools import is_plan_mode_active
        if is_plan_mode_active() and tool_name in _PATH_RESTRICTED_TOOLS:
            fp = tool_input.get("file_path", "")
            if fp:
                from cca.paths import PLANS_DIR
                try:
                    Path(fp).resolve().relative_to(PLANS_DIR.resolve())
                    # Target is inside PLANS_DIR — allow it.
                except ValueError:
                    self._console.print(
                        f"[yellow]Blocked:[/yellow] {tool_name} is not allowed in plan mode. "
                        "Finish writing your plan and call ExitPlanMode first."
                    )
                    return False
            else:
                self._console.print(
                    f"[yellow]Blocked:[/yellow] {tool_name} is not allowed in plan mode. "
                    "Finish writing your plan and call ExitPlanMode first."
                )
                return False

        # Validate that write tool targets are inside cwd or an add_dirs entry.
        # Skipped in auto_approve mode — the user has explicitly opted in to full trust.
        if tool_name in _PATH_RESTRICTED_TOOLS and not self.auto_approve:
            fp = tool_input.get("file_path", "")
            if fp and not self._path_allowed(Path(fp).resolve()):
                allowed_dirs = [str(self._cwd)] + [str(d) for d in self._add_dirs]
                self._console.print(
                    f"[red]Denied:[/red] {tool_name} target is outside the allowed directories.\n"
                    f"  target:  {fp}\n"
                    f"  allowed: {', '.join(allowed_dirs)}"
                )
                return False

        if tool_name in self.always_allow:
            return True

        if self._session_grants.get(tool_name):
            return True

        if self.auto_approve:
            return True

        return await self._prompt_user(tool_name, tool_input)

    def _permission_body(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Build Rich-markup body for the permission Panel."""
        MAX_DIFF   = 25
        MAX_LINES  = 10

        if tool_name == "Edit":
            fp       = str(tool_input.get("file_path", "?"))
            old_str  = str(tool_input.get("old_string", ""))
            new_str  = str(tool_input.get("new_string", ""))
            old_lines = old_str.splitlines()
            new_lines = new_str.splitlines()
            removed, added = len(old_lines), len(new_lines)
            parts = [
                f"[bold]{escape(fp)}[/bold]",
                f"[dim]Added {added} line(s), removed {removed} line(s)[/dim]",
            ]
            diff_lines: list[str] = (
                [f"[red]- {escape(old_line)}[/red]" for old_line in old_lines]
                + [f"[green]+ {escape(new_line)}[/green]" for new_line in new_lines]
            )
            if len(diff_lines) > MAX_DIFF:
                diff_lines = diff_lines[:MAX_DIFF]
                diff_lines.append("[dim]…[/dim]")
            parts.extend(diff_lines)
            return "\n".join(parts)

        if tool_name == "Write":
            fp      = str(tool_input.get("file_path", "?"))
            content = str(tool_input.get("content", ""))
            c_lines = content.splitlines()
            parts = [
                f"[bold]{escape(fp)}[/bold]",
                f"[dim]{len(c_lines)} lines[/dim]",
            ]
            for line in c_lines[:MAX_LINES]:
                parts.append(f"[green]+ {escape(line)}[/green]")
            if len(c_lines) > MAX_LINES:
                parts.append(f"[dim]… {len(c_lines) - MAX_LINES} more line(s)[/dim]")
            return "\n".join(parts)

        PATH_KEYS = ("file_path", "path", "pattern", "command")
        primary_key = next((k for k in PATH_KEYS if k in tool_input), None)
        kv_lines: list[str] = []
        if primary_key:
            kv_lines.append(f"[bold]{escape(str(tool_input[primary_key]))}[/bold]")
            rest = {k: v for k, v in tool_input.items() if k != primary_key}
        else:
            rest = dict(tool_input)
        for k, v in rest.items():
            for i, seg in enumerate(str(v).splitlines() or [""]):
                prefix = f"  [dim]{k}:[/dim] " if i == 0 else "    "
                kv_lines.append(f"{prefix}{escape(seg)}")

        if len(kv_lines) > MAX_LINES:
            kv_lines = kv_lines[:MAX_LINES]
            kv_lines.append("[dim]…[/dim]")
        return "\n".join(kv_lines) if kv_lines else "[dim](no arguments)[/dim]"

    async def _prompt_user(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Interactively prompt for permission."""
        body = self._permission_body(tool_name, tool_input)

        if self._permission_fn is not None:
            # Print the panel above the Application via Rich console
            # (routed through _PTKRichOutput → print_formatted_text → patch_stdout)
            from rich.panel import Panel as _Panel
            self._console.print(_Panel(
                body,
                title=f"[yellow]Permission Needed — {escape(tool_name)}[/yellow]",
                border_style="yellow",
                padding=(0, 1),
            ))
            answer = await self._permission_fn(tool_name)
        else:
            # Fallback: write to /dev/tty (non-UI or test mode)
            answer = await self._ask_via_tty(tool_name, body)

        return self._handle_answer(tool_name, answer)

    async def confirm(self, title: str, body: str) -> bool:
        """One-off yes/no confirmation using the same prompt plumbing as tools."""
        if self._permission_fn is not None:
            from rich.panel import Panel as _Panel
            self._console.print(_Panel(
                body,
                title=f"[yellow]{escape(title)}[/yellow]",
                border_style="yellow",
                padding=(0, 1),
            ))
            answer = await self._permission_fn(title)
        else:
            answer = await self._ask_via_tty(title, body)
        return answer in ("y", "yes", "")

    async def _ask_via_tty(self, tool_name: str, body: str) -> str:
        from rich.panel import Panel as _Panel

        def _ask() -> str:
            width = shutil.get_terminal_size((80, 24)).columns
            with open("/dev/tty", "w") as tty_out:
                con = Console(file=tty_out, highlight=False, width=width)
                con.print(_Panel(
                    body,
                    title=f"[yellow]Permission Needed — {escape(tool_name)}[/yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                ))
                tty_out.write("Allow? [y]es / [n]o / [a]lways allow this tool / [!]never allow: ")
                tty_out.flush()
            with open("/dev/tty") as tty_in:
                return tty_in.readline().strip().lower()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _ask)

    def _handle_answer(self, tool_name: str, answer: str) -> bool:
        if answer in ("y", "yes", ""):
            return True
        if answer in ("a", "always"):
            self.always_allow.add(tool_name)
            self._session_grants[tool_name] = True
            # Persist to global config so the grant survives restarts.
            try:
                from cca.config import Config
                cfg = Config.load()
                if tool_name not in cfg.always_allow:
                    cfg.always_allow.append(tool_name)
                    cfg.save()
                    self._console.print(
                        f"[green]Always allowing {tool_name} — saved to config.[/green]"
                    )
                else:
                    self._console.print(f"[green]Always allowing {tool_name}.[/green]")
            except Exception:
                self._console.print(f"[green]Always allowing {tool_name} for this session.[/green]")
            return True
        if answer in ("!", "never"):
            self.never_allow.add(tool_name)
            self._console.print(f"[red]Never allowing {tool_name} for this session.[/red]")
            return False
        return False
