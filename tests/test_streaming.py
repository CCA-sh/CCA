"""Unit tests for the OpenAI-backend SSE streaming parser.

No API key required — all responses are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cca.llm.base import Message, TextContent, ToolSchema, ToolUseContent
from cca.llm.openai_backend import OpenAIBackend
from tests.conftest import MockStreamingResponse, make_sse, tool_call_sse


def make_backend() -> OpenAIBackend:
    return OpenAIBackend(model="gemini-2.5-pro", api_key="fake", base_url="http://fake")


def _tool_schemas() -> list[ToolSchema]:
    return [
        ToolSchema(
            name="Bash",
            description="Run a shell command.",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        )
    ]


async def _collect(backend: OpenAIBackend, sse_lines: list[str]) -> list:
    """Run backend.stream() with mocked SSE and collect all events."""
    messages = [Message(role="user", content=[TextContent(text="run ls")])]
    mock_response = MockStreamingResponse(sse_lines)

    with patch.object(backend.client.chat.completions, "with_streaming_response") as mock_wsr:
        mock_wsr.create.return_value = mock_response
        events = []
        async for event in backend.stream(messages, _tool_schemas(), "system", max_tokens=256):
            events.append(event)
    return events


# ---------------------------------------------------------------------------
# Basic streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_delta_emitted():
    sse = make_sse([
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world"}}, {"finish_reason": "stop"}]},
    ])
    events = await _collect(make_backend(), sse)
    text = "".join(e.data for e in events if e.type == "text_delta")
    assert text == "Hello world"


@pytest.mark.asyncio
async def test_message_end_always_emitted():
    sse = make_sse([{"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}])
    events = await _collect(make_backend(), sse)
    assert any(e.type == "message_end" for e in events)


@pytest.mark.asyncio
async def test_stream_uses_max_completion_tokens_for_gpt5_models():
    backend = OpenAIBackend(model="gpt-5-mini", api_key="fake", base_url="http://fake")
    mock_response = MockStreamingResponse(make_sse([
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    ]))

    with patch.object(backend.client.chat.completions, "with_streaming_response") as mock_wsr:
        mock_wsr.create.return_value = mock_response
        events = []
        async for event in backend.stream(
            [Message(role="user", content=[TextContent(text="hello")])],
            [],
            "system",
            max_tokens=123,
        ):
            events.append(event)

    sent_kwargs = mock_wsr.create.call_args.kwargs
    assert sent_kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in sent_kwargs
    assert any(e.type == "message_end" for e in events)


# ---------------------------------------------------------------------------
# thought_signature capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_thought_signature_in_first_chunk():
    """thought_signature arrives in the same chunk as the tool name."""
    sse = tool_call_sse(
        name="Bash",
        arguments='{"command":"ls"}',
        thought_signature="sig-abc",
        split_signature=False,
    )
    events = await _collect(make_backend(), sse)
    completed = [e.data for e in events if e.type == "tool_use_complete"]
    assert len(completed) == 1
    assert completed[0].thought_signature == "sig-abc", (
        "thought_signature should be captured when it arrives in the first chunk"
    )
    assert completed[0].provider_state == {
        "provider": "openai-compatible",
        "tool_call_extra_content": {
            "google": {"thought_signature": "sig-abc"},
        },
    }


@pytest.mark.asyncio
async def test_thought_signature_after_finish_reason():
    """Gemini can send thought_signature in a chunk that arrives after finish_reason."""
    sse = tool_call_sse(
        name="Bash",
        arguments='{"command":"ls"}',
        thought_signature="sig-xyz",
        split_signature=True,
    )
    events = await _collect(make_backend(), sse)
    completed = [e.data for e in events if e.type == "tool_use_complete"]
    assert len(completed) == 1
    assert completed[0].thought_signature == "sig-xyz", (
        "thought_signature must be captured even when it arrives after finish_reason"
    )


@pytest.mark.asyncio
async def test_no_thought_signature_stored_as_none():
    """Non-thinking providers have no thought_signature; field should be None."""
    sse = tool_call_sse(name="Bash", arguments='{"command":"ls"}')
    events = await _collect(make_backend(), sse)
    completed = [e.data for e in events if e.type == "tool_use_complete"]
    assert completed[0].thought_signature is None


@pytest.mark.asyncio
async def test_multiple_tool_calls_each_capture_signature():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "tc0", "function": {"name": "Bash", "arguments": "", "thought_signature": "sig-0"}},
            {"index": 1, "id": "tc1", "function": {"name": "Read", "arguments": "", "thought_signature": "sig-1"}},
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"command":"ls"}'}},
            {"index": 1, "function": {"arguments": '{"file_path":"x.py"}'}},
        ]}, "finish_reason": "tool_calls"}]},
    ]
    sse = make_sse(chunks)
    events = await _collect(make_backend(), sse)
    completed = {e.data.name: e.data for e in events if e.type == "tool_use_complete"}
    assert completed["Bash"].thought_signature == "sig-0"
    assert completed["Read"].thought_signature == "sig-1"


@pytest.mark.asyncio
async def test_streaming_merges_split_opaque_tool_continuation_state():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "tc0",
            "function": {"name": "Bash", "arguments": ""},
            "extra_content": {"future": {"token": "opaque"}},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": '{"command":"ls"}'},
            "extra_content": {"future": {"sequence": 2}},
        }]}, "finish_reason": "tool_calls"}]},
    ]

    events = await _collect(make_backend(), make_sse(chunks))
    completed = [
        event.data for event in events if event.type == "tool_use_complete"
    ]

    assert completed[0].provider_state == {
        "provider": "openai-compatible",
        "tool_call_extra_content": {
            "future": {"token": "opaque", "sequence": 2},
        },
    }
