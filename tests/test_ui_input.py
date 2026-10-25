"""Tests for the InputBox prompt_toolkit application, driven via pipe input."""
from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from cca.ui.input_app import InputBox, UIStoppedError
from cca.ui.manager import TerminalUI


class ResizableDummyOutput(DummyOutput):
    def __init__(self) -> None:
        self.size = Size(rows=24, columns=80)

    def get_size(self) -> Size:
        return self.size

    def get_rows_below_cursor_position(self) -> int:
        return self.size.rows


@pytest.fixture
async def pipe_session():
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            box = InputBox(get_status=lambda: "")
            yield box, pipe_input
            box.exit()


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_perm_prompt_raised_before_app_starts_still_receives_keys(pipe_session):
    """A permission dialog opened before the Application task has started
    (the MCP trust prompt during session startup) must still get keyboard focus."""
    box, pipe_input = pipe_session

    # Deliberately schedule the dialog *before* the app task ever runs,
    # reproducing session-startup ordering.
    perm_task = asyncio.create_task(box.prompt_permission())
    app_task = asyncio.create_task(box.run_async())

    await _wait_for(lambda: box._app.is_running)
    pipe_input.send_text("y\r")

    answer = await asyncio.wait_for(perm_task, timeout=2)
    assert answer == "y"

    box.exit()
    await app_task


async def test_terminal_ui_aenter_waits_for_app_to_run():
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            async with TerminalUI(model="m", cwd="/tmp") as ui:
                assert ui._input_box._app.is_running


async def test_terminal_ui_aenter_raises_when_app_fails_to_start(monkeypatch):
    async def broken_run_async(self) -> None:
        raise RuntimeError("no terminal")

    monkeypatch.setattr(InputBox, "run_async", broken_run_async)
    body_entered = False
    with pytest.raises(RuntimeError, match="no terminal"):
        async with asyncio.timeout(2):
            async with TerminalUI(model="m", cwd="/tmp"):
                body_entered = True
    assert not body_entered, "__aenter__ must raise before the session body runs"


async def test_resize_and_ctrl_b_preserve_prompt_contents():
    output = ResizableDummyOutput()
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            box = InputBox(get_status=lambda: "")
            app_task = asyncio.create_task(box.run_async())
            await _wait_for(lambda: box._app.is_running)
            input_window = next(
                window
                for window in box._app.layout.find_all_windows()
                if getattr(getattr(window, "content", None), "buffer", None)
                is box._buffer
            )

            pipe_input.send_text("hello\x02")
            await _wait_for(lambda: box._buffer.text == "hello")
            await _wait_for(lambda: box._buffer.cursor_position == 4)

            for rows, columns in [(18, 55), (35, 120), (20, 70)]:
                output.size = Size(rows=rows, columns=columns)
                box._app.invalidate()
                await asyncio.sleep(0.02)
                assert box._terminal_size() == (rows, columns)
                assert box._buffer.text == "hello"
                assert box._buffer.cursor_position == 4
                assert box._queue.empty()
                write_position = (
                    box._app.renderer._last_screen
                    .visible_windows_to_write_positions[input_window]
                )
                assert write_position.height == 1

            pipe_input.send_text("\r")
            assert await asyncio.wait_for(box.get_input(), timeout=2) == "hello"
            box.exit()
            await app_task


_QUESTION_OPTIONS = [{"label": "yes", "description": ""}, {"label": "no", "description": ""}]


async def _start_app(box: InputBox) -> asyncio.Task:
    app_task = asyncio.create_task(box.run_async())
    await _wait_for(lambda: box._app.is_running)
    return app_task


async def test_prompt_question_fails_fast_when_app_already_stopped(pipe_session):
    """A question raised after the Application stopped must error, not hang."""
    box, _ = pipe_session
    app_task = await _start_app(box)
    box.exit()
    await app_task

    with pytest.raises(UIStoppedError):
        async with asyncio.timeout(2):
            await box.prompt_question(_QUESTION_OPTIONS)


async def test_prompt_question_unblocks_when_app_exits_while_waiting(pipe_session):
    box, _ = pipe_session
    app_task = await _start_app(box)

    question_task = asyncio.create_task(box.prompt_question(_QUESTION_OPTIONS))
    await _wait_for(lambda: box._question_queue is not None)
    box.exit()
    await app_task

    with pytest.raises(UIStoppedError):
        await asyncio.wait_for(question_task, timeout=2)


async def test_prompt_permission_unblocks_when_app_exits_while_waiting(pipe_session):
    box, _ = pipe_session
    app_task = await _start_app(box)

    perm_task = asyncio.create_task(box.prompt_permission())
    await _wait_for(lambda: box._perm_queue is not None)
    box.exit()
    await app_task

    with pytest.raises(UIStoppedError):
        await asyncio.wait_for(perm_task, timeout=2)


async def test_get_input_returns_none_when_app_exits_while_waiting(pipe_session):
    """A dead Application must make get_input return None (quit) so the REPL exits."""
    box, _ = pipe_session
    app_task = await _start_app(box)

    input_task = asyncio.create_task(box.get_input())
    await asyncio.sleep(0.05)
    box.exit()
    await app_task

    assert await asyncio.wait_for(input_task, timeout=2) is None


async def test_app_exit_sets_cancel_event_so_running_turn_is_cancelled(pipe_session):
    box, _ = pipe_session
    app_task = await _start_app(box)

    cancel_task = asyncio.create_task(box.wait_for_cancel())
    await asyncio.sleep(0.05)
    box.exit()
    await app_task

    await asyncio.wait_for(cancel_task, timeout=2)


async def test_terminal_ui_records_app_crash_and_exits_cleanly():
    """An Application crash mid-session must be captured, and __aexit__ must not raise."""
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            async with asyncio.timeout(5):
                async with TerminalUI(model="m", cwd="/tmp") as ui:
                    ui._input_box._app.exit(exception=RuntimeError("boom"))
                    await _wait_for(lambda: ui.app_error is not None)
                    assert "boom" in str(ui.app_error)
            assert isinstance(ui.app_error, RuntimeError)


def test_normal_input_uses_borderless_background_panel():
    box = InputBox(get_status=lambda: "")
    windows = list(box._app.layout.find_all_windows())
    input_window = next(
        window
        for window in windows
        if getattr(getattr(window, "content", None), "buffer", None) is box._buffer
    )

    assert input_window.style == "class:input-panel"
    assert input_window.dont_extend_height()
    panel_windows = [
        window for window in windows if window.style == "class:input-panel"
    ]
    assert len(panel_windows) == 3
    assert panel_windows[0].height == 1
    assert panel_windows[1].height.min == 1
    assert panel_windows[2].height == 1
    assert all(window.style != "class:border" for window in windows)
