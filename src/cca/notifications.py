"""Push-based notification queue — background jobs and agents push here."""

from __future__ import annotations

import asyncio
from typing import Any

_queue: asyncio.Queue | None = None
_loop: asyncio.AbstractEventLoop | None = None


def init(loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    """Initialize the queue. Call once from the running event loop."""
    global _queue, _loop
    _queue = asyncio.Queue()
    _loop = loop
    return _queue


def get_queue() -> asyncio.Queue:
    if _queue is None:
        raise RuntimeError("Notification queue not initialized. Call init() first.")
    return _queue


def push_threadsafe(notification: dict[str, Any]) -> None:
    """Push from a background thread (non-async)."""
    if _queue is None or _loop is None or _loop.is_closed():
        return
    try:
        _loop.call_soon_threadsafe(_queue.put_nowait, notification)
    except RuntimeError:
        # The loop can close between the check and scheduling during shutdown.
        return


async def push(notification: dict[str, Any]) -> None:
    """Push from an async context."""
    if _queue is None:
        return
    await _queue.put(notification)
