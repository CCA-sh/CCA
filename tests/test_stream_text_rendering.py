"""Buffered narration renders once, not once per parallel tool call.

A response that says "now the trainer, entrypoint and CLI script" and then
makes three Write calls raises three tool_use_start events. text_parts is
deliberately never cleared there — it has to survive into full_text and the
session file — so printing the whole buffer each time showed the same line
three times in a live view.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cca.llm.base import StreamEvent, ToolUseContent
from cca.repl import Repl


def _repl_with_stream(events):
    async def stream(**_kwargs):
        for event in events:
            yield event

    backend = MagicMock()
    backend.stream = stream
    registry = MagicMock()
    registry.get_schemas = MagicMock(return_value=[])
    registry._mcp_manager = None
    config = MagicMock()
    config.model = "litellm:fireworks_ai/kimi-k3"
    config.trace = False
    config.verbose = False
    config.hooks = {}
    config.max_tokens = 8192
    config.skills_enabled = False
    config.system_prompt_append = None
    config.vim = False
    perms = MagicMock()
    perms.check = AsyncMock(return_value=True)
    repl = Repl(backend=backend, registry=registry, config=config, permissions=perms,
                cwd="/tmp", print_mode=True, run_mode="print")
    printed: list[str] = []
    repl._console = type(
        "C", (),
        {"print": lambda _s, *a, **k: printed.append(
            getattr(a[0], "markup", str(a[0])) if a else "")},
    )()
    return repl, printed


def _tool(n):
    return StreamEvent(
        type="tool_use_start",
        data=ToolUseContent(id=f"t{n}", name="Write", input={}),
    )


@pytest.mark.asyncio
async def test_narration_before_parallel_tool_calls_prints_once():
    repl, printed = _repl_with_stream([
        StreamEvent(type="text_delta", data="Now the trainer and the CLI."),
        _tool(1), _tool(2), _tool(3),
        StreamEvent(type="message_end", data={}),
    ])

    _content, text, _stop = await repl._stream_response_impl("sys")

    assert text == "Now the trainer and the CLI.", "the text still reaches full_text"
    assert len(printed) == 1, f"narration printed {len(printed)} times: {printed}"


@pytest.mark.asyncio
async def test_text_arriving_after_a_tool_call_is_still_shown():
    repl, printed = _repl_with_stream([
        StreamEvent(type="text_delta", data="First. "),
        _tool(1),
        StreamEvent(type="text_delta", data="Second."),
        _tool(2),
        StreamEvent(type="message_end", data={}),
    ])

    _content, text, _stop = await repl._stream_response_impl("sys")

    assert text == "First. Second."
    assert len(printed) == 2
    assert "First." in printed[0] and "Second" in printed[1]
