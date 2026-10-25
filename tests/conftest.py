"""Shared fixtures and helpers."""

from __future__ import annotations

import json
import pytest


class MockStreamingResponse:
    """Simulates the httpx streaming response from with_streaming_response.create."""

    def __init__(self, sse_lines: list[str]):
        self._lines = sse_lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_lines(self):
        for line in self._lines:
            yield line


def make_sse(chunks: list[dict]) -> list[str]:
    """Turn a list of chunk dicts into SSE data lines ending with [DONE]."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


def tool_call_sse(
    name: str,
    arguments: str,
    tool_id: str = "tc_001",
    thought_signature: str | None = None,
    split_signature: bool = False,
) -> list[str]:
    """
    Build SSE chunks for a single tool call.

    split_signature=True places thought_signature in a later chunk (after
    finish_reason), which is the problematic Gemini behaviour we need to handle.
    """
    first_func: dict = {"name": name, "arguments": ""}
    if thought_signature and not split_signature:
        first_func["thought_signature"] = thought_signature

    chunks = [
        # chunk 1: tool call start
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": tool_id, "function": first_func}]}}]},
        # chunk 2: arguments
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}, "finish_reason": "tool_calls"}]},
    ]

    if thought_signature and split_signature:
        # Gemini sends thought_signature AFTER finish_reason
        chunks.append(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"thought_signature": thought_signature}}]}}]}
        )

    return make_sse(chunks)
