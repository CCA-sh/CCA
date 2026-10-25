"""Tool registry — stores and dispatches tool calls."""

from __future__ import annotations

from typing import Any

from cca.llm.base import ToolSchema
from cca.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(
        self,
        exclude: set[str] | None = None,
        include: set[str] | None = None,
    ) -> list[ToolSchema]:
        """Return ToolSchema list, optionally filtered by include/exclude sets."""
        exclude = exclude or set()
        return [
            t.schema()
            for name, t in self._tools.items()
            if name not in exclude
            and (include is None or name in include)
        ]

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute a named tool and return its string result."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.execute(**tool_input)
        except TypeError as e:
            return f"Error: invalid arguments for {name}: {e}"
        except Exception:
            raise
