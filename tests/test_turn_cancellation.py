"""Cancelling a turn must also cancel in-flight tool tasks, not orphan them."""
from __future__ import annotations

import asyncio

import pytest

from cca.config import Config
from cca.llm.base import ToolUseContent
from cca.repl import Repl


async def test_cancelling_execute_tools_cancels_child_tool_tasks(tmp_path):
    repl = Repl(backend=None, registry=None, config=Config(), permissions=None,
                cwd=str(tmp_path))

    started = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def slow_tool(tc):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            child_cancelled.set()
            raise

    repl._execute_one = slow_tool

    tool_call = ToolUseContent(id="t1", name="Bash", input={"command": "sleep 30"})
    turn = asyncio.create_task(repl._execute_tools([tool_call]))
    await asyncio.wait_for(started.wait(), timeout=2)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    await asyncio.wait_for(child_cancelled.wait(), timeout=2)
