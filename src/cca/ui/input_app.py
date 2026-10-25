"""Persistent bottom input panel built on a prompt_toolkit Application."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, vi_navigation_mode
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import DynamicStyle, Style, merge_styles


_PALETTE_MAX_VISIBLE = 9


class UIStoppedError(RuntimeError):
    """The InputBox Application stopped while a dialog still needed an answer."""


# ---------------------------------------------------------------------------
# Rich-compatible output sink
# ---------------------------------------------------------------------------

class _PTKRichOutput:
    """Buffers Rich writes and flushes them via print_formatted_text."""

    def __init__(self, real_stdout) -> None:
        self._real = real_stdout
        self._buf: list[str] = []

    def write(self, text: str) -> int:
        self._buf.append(text)
        return len(text)

    def flush(self) -> None:
        if not self._buf:
            return
        data = "".join(self._buf)
        self._buf = []
        try:
            from prompt_toolkit import print_formatted_text
            print_formatted_text(ANSI(data), end="")
        except Exception:
            self._real.write(data)
            self._real.flush()

    def fileno(self) -> int:
        return self._real.fileno()

    def isatty(self) -> bool:
        return self._real.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._real, "errors", "replace")


def make_dynamic_stdout() -> _PTKRichOutput:
    return _PTKRichOutput(sys.stdout)


# ---------------------------------------------------------------------------
# InputBox
# ---------------------------------------------------------------------------

class InputBox:
    """Persistent input box at the bottom of the terminal."""

    def __init__(
        self,
        get_status: Callable[[], str],
        history_file: str | None = None,
        vim_mode: bool = False,
        slash_commands: list[tuple[str, str]] | None = None,
    ) -> None:
        self._get_status = get_status
        self._vim_mode = vim_mode
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._spinner_text = ""
        self._spinner_visible = False
        self._pending_prompts: list[str] = []

        # Set when a permission dialog is active; key bindings check this
        self._perm_queue: asyncio.Queue[str] | None = None

        # Set once the Application is running; dialogs raised during session
        # startup must wait for this before they can take keyboard focus.
        self._started = asyncio.Event()

        # Set when the Application stops (exit, crash, or double-Ctrl+C).
        # Dialog waits race against this so they never hang on a dead app.
        self._stopped = asyncio.Event()

        # Set when an AskUserQuestion dialog is active
        self._question_queue: asyncio.Queue[str] | None = None
        self._question_options: list[dict] = []
        self._question_n: int = 0

        # Set by Escape / first Ctrl+C so the REPL can cancel a running turn
        self._cancel_event = asyncio.Event()

        # Double-Ctrl+C quit tracking
        self._last_ctrlc: float = 0.0

        # Slash command palette state
        self._all_commands: list[tuple[str, str]] = slash_commands or []
        self._palette_active: bool = False
        self._palette_items: list[tuple[str, str]] = []
        self._palette_index: int = 0
        self._palette_scroll: int = 0

        history = FileHistory(history_file) if history_file else InMemoryHistory()

        self._buffer = Buffer(
            name="main",
            multiline=True,
            history=history,
            accept_handler=self._on_accept,
        )
        self._buffer.on_text_changed += self._on_buffer_changed

        # Single-line buffer used only while a permission dialog is shown
        self._perm_buffer = Buffer(
            name="perm",
            multiline=False,
            accept_handler=self._on_perm_accept,
        )

        # Single-line buffer used only while an AskUserQuestion dialog is shown
        self._question_buffer = Buffer(
            name="question",
            multiline=False,
            accept_handler=self._on_question_accept,
        )

        # ------------------------------------------------------------------
        # Key bindings
        # ------------------------------------------------------------------
        kb = KeyBindings()
        _in_perm = Condition(lambda: self._perm_queue is not None)
        _in_question = Condition(lambda: self._question_queue is not None)
        _in_dialog = _in_perm | _in_question
        _palette_open = Condition(lambda: self._palette_active)

        # Enter: accept palette selection, or submit buffer
        @kb.add("enter", filter=~_in_dialog & _palette_open)
        def _palette_accept(event) -> None:
            if self._palette_items:
                cmd = self._palette_items[self._palette_index][0]
                self._buffer.set_document(
                    self._buffer.document.__class__(cmd, len(cmd)),
                    bypass_readonly=True,
                )
            self._close_palette()
            self._buffer.validate_and_handle()

        @kb.add("enter", filter=~_in_dialog & ~_palette_open)
        def _submit(event) -> None:
            self._buffer.validate_and_handle()

        @kb.add("enter", filter=_in_perm)
        def _submit_perm(event) -> None:
            self._perm_buffer.validate_and_handle()

        @kb.add("enter", filter=_in_question & ~_in_perm)
        def _submit_question(event) -> None:
            self._question_buffer.validate_and_handle()

        # Option+Backspace on macOS usually sends escape + backspace (or escape + ctrl-h)
        # Delete word backwards
        @kb.add("escape", "backspace", filter=~_in_dialog)
        @kb.add("escape", "c-h", filter=~_in_dialog)
        def _delete_word_backward(event) -> None:
            b = event.current_buffer
            b.delete_before_cursor(count=b.document.find_start_of_previous_word(count=1) or 0)

        # Option+Delete on macOS (Forward delete word) usually sends escape + d
        @kb.add("escape", "d", filter=~_in_dialog)
        def _delete_word_forward(event) -> None:
            b = event.current_buffer
            b.delete(count=b.document.find_next_word_ends(count=1) or 0)

        # Option+Left / Option+Right (macOS sends escape+b / escape+f) — word jump.
        # Must be explicit here; our bare "escape" eager handler would otherwise fire first.
        @kb.add("escape", "b", filter=~_in_dialog)
        def _word_left(event) -> None:
            b = event.current_buffer
            b.cursor_left(count=b.document.find_start_of_previous_word(count=1) or 0)

        @kb.add("escape", "f", filter=~_in_dialog)
        def _word_right(event) -> None:
            b = event.current_buffer
            b.cursor_right(count=b.document.find_next_word_end(count=1) or 0)

        # Ctrl+J — literal newline (always available)
        @kb.add("c-j", filter=~_in_dialog)
        def _newline(event) -> None:
            self._buffer.insert_text("\n")

        if not vim_mode:
            # Escape+Enter — literal newline (only in emacs/normal mode, not while in a dialog)
            @kb.add("escape", "enter", eager=True, filter=~_in_dialog)
            def _newline_esc(event) -> None:
                self._buffer.insert_text("\n")

        # Ctrl+C: first press = clear input / cancel turn; second within 2 s = quit
        @kb.add("c-c")
        def _interrupt(event) -> None:
            now = time.monotonic()
            if now - self._last_ctrlc < 2.0:
                self._queue.put_nowait(None)
                if self._app and self._app.is_running:
                    self._app.exit()
                return

            self._last_ctrlc = now
            if self._buffer.text:
                self._buffer.text = ""
            if self._perm_queue is not None:
                self._perm_queue.put_nowait("n")
            if self._question_queue is not None:
                self._question_queue.put_nowait("")
            self._cancel_event.set()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            self._pending_prompts.clear()

        # Ctrl+D: quit when buffer is empty and no dialog is open
        @kb.add("c-d")
        def _eof(event) -> None:
            if not self._buffer.text and self._perm_queue is None:
                self._queue.put_nowait(None)

        if vim_mode:
            @kb.add("escape", filter=vi_navigation_mode, eager=True)
            def _vim_escape(event) -> None:
                self._close_palette()
                if self._perm_queue is not None:
                    self._perm_queue.put_nowait("n")
                if self._question_queue is not None:
                    self._question_queue.put_nowait("")
                self._cancel_event.set()
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        break
                self._pending_prompts.clear()
        else:
            @kb.add("escape", eager=True)
            def _escape(event) -> None:
                if self._palette_active:
                    self._close_palette()
                    return
                if self._perm_queue is not None:
                    self._perm_queue.put_nowait("n")
                if self._question_queue is not None:
                    self._question_queue.put_nowait("")
                self._cancel_event.set()
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        break
                self._pending_prompts.clear()

        # Up: palette navigate or history back
        @kb.add("up", filter=~_in_dialog)
        def _up(event) -> None:
            if self._palette_active and self._palette_items:
                self._palette_index = max(0, self._palette_index - 1)
                self._scroll_to_view()
                event.app.invalidate()
                return
            buf = event.current_buffer
            txt = buf.text
            first_nl = txt.find("\n")
            at_first_line = first_nl == -1 or buf.cursor_position <= first_nl
            if at_first_line:
                buf.history_backward(count=1)
            else:
                buf.cursor_up(count=1)

        # Down: palette navigate or history forward
        @kb.add("down", filter=~_in_dialog)
        def _down(event) -> None:
            if self._palette_active and self._palette_items:
                self._palette_index = min(len(self._palette_items) - 1, self._palette_index + 1)
                self._scroll_to_view()
                event.app.invalidate()
                return
            buf = event.current_buffer
            txt = buf.text
            last_nl = txt.rfind("\n")
            at_last_line = last_nl == -1 or buf.cursor_position > last_nl
            if at_last_line:
                buf.history_forward(count=1)
            else:
                buf.cursor_down(count=1)

        # ------------------------------------------------------------------
        # Layout helpers
        # ------------------------------------------------------------------
        def _palette_content():
            if not self._palette_active or not self._palette_items:
                return [("", "")]
            _, columns = self._terminal_size()
            cols = max(columns - 1, 10)
            total = len(self._palette_items)
            visible = self._palette_items[self._palette_scroll: self._palette_scroll + _PALETTE_MAX_VISIBLE]
            cmd_w = min(
                max((len(cmd) for cmd, _ in self._palette_items), default=8) + 2,
                max(1, cols - 4),
            )

            result: list[tuple[str, str]] = []

            # Header row
            label = " Commands "
            dash_left = "─" * 2
            dash_right = "─" * max(0, cols - len(label) - 2)
            result.append(("class:palette-border", f"{dash_left}{label}{dash_right}\n"))

            # Command rows
            for i, (cmd, desc) in enumerate(visible):
                abs_i = self._palette_scroll + i
                selected = abs_i == self._palette_index
                indicator = "▸" if selected else " "
                cmd_col = cmd[:cmd_w].ljust(cmd_w)
                max_desc = max(0, cols - cmd_w - 4)
                desc_col = desc[:max_desc] if len(desc) > max_desc else desc
                if selected:
                    result.append(("class:palette-sel-indicator", f" {indicator} "))
                    result.append(("class:palette-sel-cmd", cmd_col))
                    result.append(("class:palette-sel-desc", f" {desc_col}\n"))
                else:
                    result.append(("class:palette-indicator", f" {indicator} "))
                    result.append(("class:palette-cmd", cmd_col))
                    result.append(("class:palette-desc", f" {desc_col}\n"))

            # Footer row: match count + scroll hint + nav hint
            showing_end = min(self._palette_scroll + _PALETTE_MAX_VISIBLE, total)
            count_str = f" {self._palette_scroll + 1}–{showing_end} of {total} "
            nav_str = " ↑↓ navigate · ↵ select · Esc dismiss "
            nav_str = nav_str[:max(0, cols - len(count_str))]
            gap = max(0, cols - len(count_str) - len(nav_str))
            result.append(("class:palette-border", f"{count_str}{'─' * gap}{nav_str}\n"))

            return result

        def _pending_content():
            if not self._pending_prompts:
                return [("", "")]
            lines = []
            _, columns = self._terminal_size()
            cols = max(columns - 1, 10)
            for prompt in self._pending_prompts:
                lines.append(f"\033[34m╭{'─' * (cols - 2)}╮\033[0m")
                prompt_lines = prompt.split("\n")
                display_lines = prompt_lines[:4]
                for line in display_lines:
                    display_text = line if len(line) <= cols - 4 else line[:cols - 7] + "..."
                    lines.append(f"\033[34m│\033[0m {display_text.ljust(cols - 4)} \033[34m│\033[0m")
                if len(prompt_lines) > 4:
                    more_text = f"... and {len(prompt_lines) - 4} more lines"
                    lines.append(f"\033[34m│\033[0m \033[90m{more_text.ljust(cols - 4)}\033[0m \033[34m│\033[0m")
                lines.append(f"\033[34m╰{'─' * (cols - 2)}╯\033[0m")
            return ANSI("\n".join(lines) + "\n")

        def _spinner_content():
            if not self._spinner_text:
                return [("", "")]
            rows, _ = self._terminal_size()
            max_lines = max(1, rows - 5)
            lines = self._spinner_text.split("\n")
            text = "\n".join(lines[-max_lines:]) if len(lines) > max_lines else self._spinner_text
            return ANSI(text)

        def _status_content():
            from cca.tools.plan_tools import is_plan_mode_active
            status = self._get_status()
            parts: list[tuple[str, str]] = []
            if is_plan_mode_active():
                parts.append(("class:plan-badge", " ⚡ PLAN MODE "))
            if vim_mode:
                try:
                    from prompt_toolkit.vi_state import InputMode
                    mode = self._app.vi_state.input_mode
                    if mode == InputMode.INSERT:
                        indicator = " INSERT "
                    elif mode == InputMode.NAVIGATION:
                        indicator = " NORMAL "
                    elif mode == InputMode.REPLACE:
                        indicator = " REPLACE "
                    else:
                        indicator = " VISUAL "
                except Exception:
                    indicator = " VI "
                parts.append(("class:status-vim", indicator))
            parts.append(("class:status", " " + status))
            return parts

        def _get_line_prefix(line_number: int, wrap_count: int):
            if line_number == 0 and wrap_count == 0:
                return [("class:prompt", "❯ ")]
            return [("class:prompt-cont", "  ")]

        def _perm_line_prefix(line_number: int, wrap_count: int):
            return [("class:perm-prompt", "Allow? [y]es / [n]o / [a]lways / [!]never > ")]

        def _question_line_prefix(line_number: int, wrap_count: int):
            n = self._question_n
            return [("class:perm-prompt", f"Answer (1-{n} or free text) > ")]

        spinner_visible = Condition(lambda: self._spinner_visible)

        layout = Layout(
            HSplit([
                # Pending prompts area
                ConditionalContainer(
                    Window(
                        content=FormattedTextControl(_pending_content),
                        dont_extend_height=True,
                    ),
                    filter=Condition(lambda: len(self._pending_prompts) > 0) & ~_in_dialog,
                ),
                # Spinner area
                ConditionalContainer(
                    Window(
                        content=FormattedTextControl(_spinner_content),
                        dont_extend_height=True,
                    ),
                    filter=spinner_visible & ~_in_dialog,
                ),
                # Command palette — appears above the input panel when active
                ConditionalContainer(
                    Window(
                        content=FormattedTextControl(_palette_content),
                        dont_extend_height=True,
                    ),
                    filter=_palette_open & ~_in_dialog,
                ),
                # Normal multi-line input panel with one row of vertical padding.
                ConditionalContainer(
                    HSplit([
                        Window(
                            height=1,
                            dont_extend_height=True,
                            style="class:input-panel",
                        ),
                        Window(
                            content=BufferControl(buffer=self._buffer, focusable=True),
                            get_line_prefix=_get_line_prefix,
                            wrap_lines=True,
                            height=Dimension(min=1),
                            dont_extend_height=True,
                            style="class:input-panel",
                        ),
                        Window(
                            height=1,
                            dont_extend_height=True,
                            style="class:input-panel",
                        ),
                    ]),
                    filter=~_in_dialog,
                ),
                # Single-line permission input
                ConditionalContainer(
                    Window(
                        content=BufferControl(buffer=self._perm_buffer, focusable=True),
                        get_line_prefix=_perm_line_prefix,
                        height=1,
                        dont_extend_height=True,
                    ),
                    filter=_in_perm,
                ),
                # Single-line question input
                ConditionalContainer(
                    Window(
                        content=BufferControl(buffer=self._question_buffer, focusable=True),
                        get_line_prefix=_question_line_prefix,
                        height=1,
                        dont_extend_height=True,
                    ),
                    filter=_in_question & ~_in_perm,
                ),
                # Status bar
                Window(
                    content=FormattedTextControl(_status_content),
                    height=1,
                    dont_extend_height=True,
                ),
            ]),
            focused_element=self._buffer,
        )

        _base_style = Style.from_dict({
            "input-panel":           "bg:#262626",
            "prompt":                "bold #00bb00",
            "prompt-cont":           "#00bb00",
            "perm-prompt":           "",
            "spinner":               "#888888",
            "status":                "bg:#1c1c2e #666666",
            "status-vim":            "bg:#444466 bold #aaaaff",
            "plan-badge":            "bg:#1c1c2e #8f2d56",
            "palette-border":        "#444466",
            "palette-indicator":     "#555555",
            "palette-cmd":           "#88aaff",
            "palette-desc":          "#555566",
            "palette-sel-indicator": "bg:#1e3a5f bold #aaccff",
            "palette-sel-cmd":       "bg:#1e3a5f bold #ffffff",
            "palette-sel-desc":      "bg:#1e3a5f #8899bb",
        })
        # Overrides applied only while plan mode is active.
        _plan_style = Style.from_dict({
            "prompt":      "bold #8f2d56",
            "prompt-cont": "#8f2d56",
            "status":      "bg:#3d1020 #cc8899",
            "status-vim":  "bg:#5d1030 bold #ffb0cc",
            "plan-badge":  "bg:#8f2d56 bold #ffffff",
        })

        def _get_style() -> Style:
            from cca.tools.plan_tools import is_plan_mode_active
            typing_plan = self._buffer.text.lower().startswith("/plan")
            if is_plan_mode_active() or typing_plan:
                return merge_styles([_base_style, _plan_style])
            return _base_style

        self._app = Application(
            layout=layout,
            key_bindings=kb,
            editing_mode=EditingMode.VI if vim_mode else EditingMode.EMACS,
            style=DynamicStyle(_get_style),
            full_screen=False,
            refresh_interval=0.12,
            mouse_support=False,
        )

    # ------------------------------------------------------------------
    # Palette helpers
    # ------------------------------------------------------------------

    def _terminal_size(self) -> tuple[int, int]:
        """Return the dimensions used by prompt_toolkit's active renderer."""
        if hasattr(self, "_app"):
            size = self._app.output.get_size()
            return size.rows, size.columns
        return 24, 80

    def _on_buffer_changed(self, _) -> None:
        text = self._buffer.text
        if text.startswith("/") and " " not in text and self._all_commands:
            typed = text[1:]
            self._palette_items = [
                (cmd, desc) for cmd, desc in self._all_commands
                if cmd[1:].startswith(typed)
            ]
            self._palette_index = 0
            self._palette_scroll = 0
            self._palette_active = bool(self._palette_items)
        else:
            self._palette_active = False
            self._palette_items = []
        if hasattr(self, "_app") and self._app.is_running:
            self._app.invalidate()

    def _close_palette(self) -> None:
        self._palette_active = False
        self._palette_items = []
        if hasattr(self, "_app") and self._app.is_running:
            self._app.invalidate()

    def _scroll_to_view(self) -> None:
        if self._palette_index < self._palette_scroll:
            self._palette_scroll = self._palette_index
        elif self._palette_index >= self._palette_scroll + _PALETTE_MAX_VISIBLE:
            self._palette_scroll = self._palette_index - _PALETTE_MAX_VISIBLE + 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run_async(self) -> None:
        try:
            with patch_stdout():
                await self._app.run_async(pre_run=self._started.set)
        finally:
            # Wake everything blocked on this app: dialogs raise UIStoppedError,
            # get_input returns the quit sentinel, and a running turn is cancelled.
            self._stopped.set()
            self._cancel_event.set()

    async def wait_until_running(self) -> None:
        await self._started.wait()

    def exit(self) -> None:
        if self._app.is_running:
            self._app.exit()

    # ------------------------------------------------------------------
    # Normal input
    # ------------------------------------------------------------------

    async def _get_or_stopped(self, queue: asyncio.Queue):
        """Await a queue item, waking with None if the Application stops first."""
        getter = asyncio.ensure_future(queue.get())
        stopper = asyncio.ensure_future(self._stopped.wait())
        done, pending = await asyncio.wait(
            {getter, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        # An answer that raced the shutdown still wins.
        if getter in done:
            return getter.result()
        return None

    async def _require_running_app(self) -> None:
        """Wait for the Application to start; raise if it stopped instead."""
        started = asyncio.ensure_future(self._started.wait())
        stopper = asyncio.ensure_future(self._stopped.wait())
        done, pending = await asyncio.wait(
            {started, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if self._stopped.is_set():
            raise UIStoppedError("input UI is not running")

    async def get_input(self) -> str | None:
        """Block until Enter is pressed. Returns None on Ctrl-C/D or a dead app."""
        item = await self._get_or_stopped(self._queue)
        if item is not None and item in self._pending_prompts:
            self._pending_prompts.remove(item)
            if self._app and self._app.is_running:
                self._app.invalidate()
        return item

    def _on_accept(self, buffer: Buffer) -> bool:
        self._palette_active = False
        self._queue.put_nowait(buffer.text)
        if self._spinner_visible:
            self._pending_prompts.append(buffer.text)
        return False

    # ------------------------------------------------------------------
    # Permission dialog
    # ------------------------------------------------------------------

    def _on_perm_accept(self, buffer: Buffer) -> bool:
        text = buffer.text.strip().lower()
        if text.startswith("y"):
            answer = "y"
        elif text.startswith("a"):
            answer = "a"
        elif text.startswith("!") or text.startswith("never"):
            answer = "!"
        else:
            answer = "n"
        if self._perm_queue is not None:
            self._perm_queue.put_nowait(answer)
        return False

    async def prompt_permission(self) -> str:
        """Await the user's permission answer."""
        self._perm_queue = asyncio.Queue()
        self._perm_buffer.reset()
        try:
            await self._require_running_app()
            self._app.layout.focus(self._perm_buffer)
            self._app.invalidate()
            answer = await self._get_or_stopped(self._perm_queue)
            if answer is None:
                raise UIStoppedError("input UI stopped while awaiting permission")
            return answer
        finally:
            self._perm_queue = None
            if self._app.is_running:
                self._app.layout.focus(self._buffer)
                self._app.invalidate()

    # ------------------------------------------------------------------
    # Question dialog
    # ------------------------------------------------------------------

    def _on_question_accept(self, buffer: Buffer) -> bool:
        raw = buffer.text.strip()
        answer = raw
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(self._question_options):
                answer = self._question_options[idx]["label"]
        if self._question_queue is not None:
            self._question_queue.put_nowait(answer)
        return False

    async def prompt_question(self, options: list[dict]) -> str:
        """Await the user's answer to an AskUserQuestion prompt."""
        self._question_options = options
        self._question_n = len(options)
        self._question_queue = asyncio.Queue()
        self._question_buffer.reset()
        try:
            await self._require_running_app()
            self._app.layout.focus(self._question_buffer)
            self._app.invalidate()
            answer = await self._get_or_stopped(self._question_queue)
            if answer is None:
                raise UIStoppedError("input UI stopped while awaiting an answer")
            return answer
        finally:
            self._question_queue = None
            self._question_options = []
            self._question_n = 0
            if self._app.is_running:
                self._app.layout.focus(self._buffer)
                self._app.invalidate()

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def clear_cancel(self) -> None:
        """Reset the cancel signal (call before starting a new turn)."""
        self._cancel_event.clear()

    async def wait_for_cancel(self) -> None:
        """Await the next Escape or first-Ctrl+C cancel signal."""
        await self._cancel_event.wait()

    # ------------------------------------------------------------------
    # Spinner
    # ------------------------------------------------------------------

    def set_spinner(self, text: str) -> None:
        self._spinner_text = text
        self._spinner_visible = True
        if self._app.is_running:
            self._app.invalidate()

    def clear_spinner(self) -> None:
        if self._spinner_visible:
            self._spinner_visible = False
            self._spinner_text = ""
            if self._app.is_running:
                self._app.invalidate()
