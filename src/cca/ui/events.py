"""Module-level EventBus for loose coupling between tools and UI.

Tools emit events without knowing about the UI layer.  The UI subscribes.
"""

from __future__ import annotations

import contextvars
from collections import defaultdict
from typing import Callable

# Set inside each tool's asyncio task so bash_output_line events can be routed
# to the correct per-tool box in TerminalUI.
current_tool_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_id", default=None
)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, handler: Callable) -> None:
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    def emit(self, event: str, **data) -> None:
        for h in list(self._handlers.get(event, [])):
            try:
                h(**data)
            except Exception:
                pass


# Singleton — import this everywhere instead of creating new instances.
bus = EventBus()
