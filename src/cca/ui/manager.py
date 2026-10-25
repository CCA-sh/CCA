"""TerminalUI — wires InputBox, animation clock, and Rich Console together."""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto

from rich.console import Console

from cca.ui.components import (
    ANIMATION_INTERVAL,
    DONE_INDICATOR,
    ERR_INDICATOR,
    render_action_inline_ansi,
    spinner_frame,
)
from cca.ui.events import bus
from cca.ui.input_app import InputBox, make_dynamic_stdout

_MAX_BASH_LINES = 200


class _Phase(Enum):
    IDLE = auto()
    THINKING = auto()
    TOOL = auto()


@dataclass
class _ToolBox:
    label: str
    lines: list[str] = field(default_factory=list)
    total: int = 0
    has_error: bool = False
    started_at: float = field(default_factory=time.monotonic)


class TerminalUI:
    """Async context manager that manages the full terminal UI."""

    def __init__(self, model: str, cwd: str, vim_mode: bool = False, slash_commands: list[tuple[str, str]] | None = None) -> None:
        self._model = model
        self._think_level: str = "off"
        self._cwd = cwd
        self._vim_mode = vim_mode
        self._slash_commands = slash_commands
        self._phase = _Phase.IDLE
        self._tick = 0
        self._phase_start = 0.0
        self._last_token_time = 0.0

        # Thinking-phase state
        self._thinking_lines: list[str] = []
        self._thinking_total = 0
        self._thinking_buffer = ""

        # Tool-phase state — one entry per concurrent tool
        self._tool_boxes: dict[str, _ToolBox] = {}   # keyed by tc.id

        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._anim_task: asyncio.Task | None = None
        self._app_task: asyncio.Task | None = None
        self._input_box: InputBox | None = None

        # Exception that killed the Application mid-session, if any.
        self.app_error: BaseException | None = None

        self._console = Console(file=make_dynamic_stdout())

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "TerminalUI":
        from cca.config import CONFIG_DIR
        history_path = str(CONFIG_DIR / "history")

        self._input_box = InputBox(
            get_status=self._status_text,
            history_file=history_path,
            vim_mode=self._vim_mode,
            slash_commands=self._slash_commands,
        )
        self._app_task = asyncio.create_task(
            self._input_box.run_async(), name="ui-app"
        )
        # Dialogs can fire as early as session startup (e.g. the MCP trust
        # prompt), so don't hand the UI over until the Application is running —
        # and surface a failed startup instead of waiting forever.
        started = asyncio.ensure_future(self._input_box.wait_until_running())
        done, _ = await asyncio.wait(
            {started, self._app_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if self._app_task in done:
            started.cancel()
            self._app_task.result()
        # Startup succeeded — from here on, an Application death must be
        # surfaced instead of leaving the REPL running headless.
        self._app_task.add_done_callback(self._on_app_task_done)
        self._anim_task = asyncio.create_task(
            self._animation_loop(), name="ui-anim"
        )
        bus.on("bash_output_line", self._on_bash_line)
        return self

    async def __aexit__(self, *_) -> None:
        bus.off("bash_output_line", self._on_bash_line)
        if self._anim_task:
            self._anim_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._anim_task
        if self._input_box:
            self._input_box.exit()
        if self._app_task:
            self._app_task.cancel()
            # A crash was already reported via _on_app_task_done; re-raising it
            # here would mask the session's own outcome.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._app_task

    def _on_app_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.app_error = exc
            import sys
            print(f"\nInput UI crashed: {exc!r} — shutting down.", file=sys.__stderr__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def console(self) -> Console:
        return self._console

    async def get_input(self) -> str | None:
        assert self._input_box is not None
        return await self._input_box.get_input()

    async def request_permission(self, tool_name: str) -> str:
        """Collect the user's permission answer (panel already printed above Application)."""
        assert self._input_box is not None
        return await self._input_box.prompt_permission()

    async def ask_user(
        self,
        question: str,
        options: list[dict],
        multi: bool,
        header: str,
    ) -> str:
        """Display an AskUserQuestion panel then collect the answer through prompt_toolkit."""
        from rich.markup import escape
        from rich.panel import Panel

        lines: list[str] = [escape(question), ""]
        for i, opt in enumerate(options, 1):
            label = escape(opt.get("label", ""))
            desc = escape(opt.get("description", ""))
            lines.append(f"  [bold cyan]{i}.[/bold cyan] [bold]{label}[/bold]")
            if desc:
                lines.append(f"     [dim]{desc}[/dim]")
        lines.append("")
        lines.append("[dim]Or type your own answer[/dim]")

        self._console.print(Panel(
            "\n".join(lines),
            title=f"[cyan]Question — {escape(header)}[/cyan]",
            border_style="cyan",
            padding=(0, 1),
        ))

        assert self._input_box is not None
        return await self._input_box.prompt_question(options)

    async def wait_for_cancel(self) -> None:
        """Await the next Escape / Ctrl+C cancel signal."""
        assert self._input_box is not None
        await self._input_box.wait_for_cancel()

    def clear_cancel(self) -> None:
        """Reset the cancel flag before starting a new turn."""
        if self._input_box is not None:
            self._input_box.clear_cancel()

    # -- Thinking phase --------------------------------------------------

    def set_thinking(self) -> None:
        self._commit_thinking_box()   # no-op if already IDLE
        now = time.monotonic()
        self._phase = _Phase.THINKING
        self._phase_start = now
        self._last_token_time = now
        self._thinking_lines = []
        self._thinking_total = 0
        self._thinking_buffer = ""

    def set_idle(self, is_cancel: bool = False) -> None:
        if self._phase == _Phase.THINKING:
            self._commit_thinking_box(is_cancel=is_cancel)
        elif self._phase == _Phase.TOOL:
            # Commit any hanging tools so they don't disappear on cancel
            for tool_id, box in list(self._tool_boxes.items()):
                if is_cancel:
                    box.has_error = True
                self._commit_tool_box(tool_id, box)
            self._tool_boxes.clear()
            
        self._phase = _Phase.IDLE
        if self._input_box:
            self._input_box.clear_spinner()

    def on_token(self) -> None:
        self._last_token_time = time.monotonic()

    def add_text(self, text: str) -> None:
        """Feed streaming LLM text tokens into the live thinking box."""
        combined = self._thinking_buffer + text
        parts = combined.split("\n")
        self._thinking_buffer = parts[-1]
        for line in parts[:-1]:
            self._thinking_lines.append(line)
            self._thinking_total += 1
        if len(self._thinking_lines) > _MAX_BASH_LINES:
            self._thinking_lines = self._thinking_lines[-_MAX_BASH_LINES:]
        if self._phase == _Phase.THINKING:
            self._refresh()

    # -- Tool phase (one box per tool) -----------------------------------

    def start_tool(self, tool_id: str, label: str) -> None:
        """Register a new live box for one tool call."""
        self._tool_boxes[tool_id] = _ToolBox(label=label)
        self._phase = _Phase.TOOL
        self._refresh()

    def feed_tool_result(self, tool_id: str, lines: list[str]) -> None:
        """Populate a tool box from its result string (fallback for non-streaming tools)."""
        box = self._tool_boxes.get(tool_id)
        if box is None or box.total > 0 or not lines:
            return
        box.total = len(lines)
        box.lines = lines[-_MAX_BASH_LINES:]

    def set_tool_error(self, tool_id: str) -> None:
        """Mark a tool's box as errored before committing."""
        box = self._tool_boxes.get(tool_id)
        if box:
            box.has_error = True

    def finish_tool(self, tool_id: str) -> None:
        """Commit a tool's box to scroll history and remove it."""
        box = self._tool_boxes.get(tool_id)
        if box is None:
            return
        self._commit_tool_box(tool_id, box)
        del self._tool_boxes[tool_id]
        if not self._tool_boxes:
            self._phase = _Phase.IDLE
            if self._input_box:
                self._input_box.clear_spinner()
        else:
            self._refresh()

    # -- Token / model / cwd bookkeeping --------------------------------

    def update_tokens(self, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cache_read_tokens = cache_read
        self._cache_write_tokens = cache_write

    def update_model(self, model: str) -> None:
        self._model = model

    def update_think_level(self, level: str) -> None:
        self._think_level = level

    def update_cwd(self, cwd: str) -> None:
        self._cwd = cwd

    def _status_text(self) -> str:
        think_tag = f"({self._think_level})" if self._think_level != "off" else "(off)"
        status = f"{self._model}{think_tag}   {self._cwd}   ↑{self._input_tokens:,} ↓{self._output_tokens:,}"
        if self._cache_read_tokens:
            status += f" (cache {self._cache_read_tokens:,})"
        return status


    @asynccontextmanager
    async def suspend(self):
        yield

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _commit_thinking_box(self, is_cancel: bool = False) -> None:
        if self._phase != _Phase.THINKING or self._input_box is None:
            return
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import ANSI

        elapsed = time.monotonic() - self._phase_start
        lines = list(self._thinking_lines)
        if self._thinking_buffer:
            lines.append(self._thinking_buffer)

        rendered = render_action_inline_ansi(
            indicator=ERR_INDICATOR if is_cancel else DONE_INDICATOR,
            title="Thinking…",
            lines=lines,
            total=len(lines),
            has_error=is_cancel,
            elapsed=elapsed,
        )
        print_formatted_text(ANSI(rendered), end="\n")

    def _commit_tool_box(self, tool_id: str, box: _ToolBox) -> None:
        if self._input_box is None:
            return
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import ANSI

        elapsed = time.monotonic() - box.started_at
        rendered = render_action_inline_ansi(
            indicator=DONE_INDICATOR,
            title=box.label,
            lines=box.lines,
            total=box.total,
            has_error=box.has_error,
            elapsed=elapsed,
        )
        print_formatted_text(ANSI(rendered), end="\n")

    async def _animation_loop(self) -> None:
        while True:
            await asyncio.sleep(ANIMATION_INTERVAL)
            self._tick += 1
            self._refresh()

    def _on_bash_line(
        self, line: str, is_stderr: bool = False, tool_id: str | None = None
    ) -> None:
        box = self._tool_boxes.get(tool_id) if tool_id else None
        if box is None and self._tool_boxes:
            # Fallback: route to the only / first active box
            box = next(iter(self._tool_boxes.values()))
        if box is None:
            return
        if is_stderr:
            box.has_error = True
        box.lines.append(line)
        box.total += 1
        if len(box.lines) > _MAX_BASH_LINES:
            box.lines = box.lines[_MAX_BASH_LINES // 2:]
        self._refresh()

    def _refresh(self) -> None:
        if self._phase == _Phase.IDLE or self._input_box is None:
            return

        now = time.monotonic()
        frame = spinner_frame(self._tick)

        if self._phase == _Phase.THINKING:
            elapsed = now - self._phase_start
            display_lines = list(self._thinking_lines)
            if self._thinking_buffer:
                display_lines.append(self._thinking_buffer)
            rendered = render_action_inline_ansi(
                indicator=frame,
                title="Thinking…",
                lines=display_lines,
                total=len(display_lines),
                has_error=False,
                elapsed=elapsed,
                tick=self._tick,
            )
            self._input_box.set_spinner(rendered)
        else:
            parts: list[str] = []
            for box in self._tool_boxes.values():
                elapsed = now - box.started_at
                parts.append(render_action_inline_ansi(
                    indicator=frame,
                    title=box.label,
                    lines=box.lines,
                    total=box.total,
                    has_error=box.has_error,
                    elapsed=elapsed,
                    tick=self._tick,
                ))
            self._input_box.set_spinner("\n".join(parts))
