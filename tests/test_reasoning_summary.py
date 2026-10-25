"""Provider-neutral reasoning summary request, storage, and replay tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from cca.cli import main
from cca.llm.anthropic_backend import (
    _anthropic_thinking_params,
    _to_anthropic_messages,
)
from cca.llm.base import (
    Message, TextContent, ThinkingContent, ToolResultContent, ToolUseContent,
)
from cca.llm.openai_backend import (
    OpenAIBackend,
    _to_openai_messages,
    _to_responses_input,
)
from cca.repl import Repl
from tests.conftest import MockStreamingResponse, make_sse


def test_reasoning_block_round_trips_with_classification_and_state():
    original = Message(role="assistant", content=[
        ThinkingContent(
            thinking="Inspect the parser first.",
            reasoning_kind="summary",
            provider_state={
                "provider": "openai",
                "reasoning_items": [{
                    "type": "reasoning", "id": "rs_1", "summary": [],
                    "encrypted_content": "opaque",
                }],
            },
        ),
        ToolUseContent(
            id="call_1", name="Read", input={"path": "app.py"},
            thought_signature="gemini-signature",
        ),
    ])

    restored = Message.from_dict(original.to_dict())

    thinking = restored.content[0]
    tool = restored.content[1]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.reasoning_kind == "summary"
    assert thinking.provider_state["reasoning_items"][0]["encrypted_content"] == "opaque"
    assert isinstance(tool, ToolUseContent)
    assert tool.thought_signature == "gemini-signature"


def test_legacy_thinking_block_loads_as_unknown():
    restored = Message.from_dict({
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "legacy"}],
    })
    assert restored.content[0].reasoning_kind == "unknown"


def test_gemini_summary_uses_native_thinking_config_not_reasoning_effort():
    backend = OpenAIBackend(
        model="gemini-3.7-flash", provider="google", api_key="fake",
        base_url="http://fake",
    )

    kwargs = backend._build_thinking_kwargs("high", reasoning_summary=True)

    assert "reasoning_effort" not in kwargs
    config = kwargs["extra_body"]["extra_body"]["google"]["thinking_config"]
    assert config == {"thinking_level": "high", "include_thoughts": True}


def test_gemini_25_summary_maps_level_to_budget():
    backend = OpenAIBackend(
        model="gemini-2.5-flash", provider="google", api_key="fake",
        base_url="http://fake",
    )
    kwargs = backend._build_thinking_kwargs("medium", reasoning_summary=True)
    config = kwargs["extra_body"]["extra_body"]["google"]["thinking_config"]
    assert config == {"thinking_budget": 8192, "include_thoughts": True}


@pytest.mark.asyncio
async def test_gemini_37_request_omits_deprecated_sampling_parameters():
    backend = OpenAIBackend(
        model="gemini-3.7-flash", provider="google", api_key="fake",
        base_url="http://fake",
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                model_extra={}, reasoning_content=None, content="ok", tool_calls=[],
            ),
            finish_reason="stop",
        )],
        usage=None,
    )

    with patch.object(
        backend.client.chat.completions, "create",
        new=AsyncMock(return_value=response),
    ) as create:
        await backend.complete(
            [Message(role="user", content="hello")], [], "system",
            think_level="medium",
        )

    kwargs = create.await_args.kwargs
    assert kwargs["reasoning_effort"] == "medium"
    assert {"temperature", "top_p", "top_k", "candidate_count"}.isdisjoint(kwargs)


@pytest.mark.asyncio
async def test_non_streaming_tool_state_round_trips_through_child_style_calls():
    backend = OpenAIBackend(
        model="gemini-3.7-flash", provider="google", api_key="fake",
        base_url="http://fake",
    )
    tool_call = SimpleNamespace(
        id="call-glob",
        function=SimpleNamespace(
            name="Glob", arguments='{"pattern":"*"}', model_extra={},
        ),
        model_extra={
            "extra_content": {
                "google": {"thought_signature": "opaque-signature"},
            },
        },
    )
    first_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                model_extra={}, reasoning_content=None, content=None,
                tool_calls=[tool_call],
            ),
            finish_reason="tool_calls",
        )],
        usage=None,
    )
    second_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                model_extra={}, reasoning_content=None, content="done", tool_calls=[],
            ),
            finish_reason="stop",
        )],
        usage=None,
    )

    with patch.object(
        backend.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=[first_response, second_response]),
    ) as create:
        first = await backend.complete(
            [Message(role="user", content="inspect")], [], "system",
        )
        call = first.tool_calls[0]
        assert call.thought_signature == "opaque-signature"
        assert call.provider_state == {
            "provider": "google",
            "tool_call_extra_content": {
                "google": {"thought_signature": "opaque-signature"},
            },
        }
        await backend.complete([
            Message(role="user", content="inspect"),
            first.to_message(),
            Message(role="user", content=[ToolResultContent(
                tool_use_id="call-glob", name="Glob", content="file.swift",
            )]),
        ], [], "system")

    replayed = create.await_args_list[1].kwargs["messages"][2]["tool_calls"][0]
    assert replayed["extra_content"] == {
        "google": {"thought_signature": "opaque-signature"},
    }


@pytest.mark.asyncio
async def test_gemini_stream_routes_thought_content_away_from_final_text():
    backend = OpenAIBackend(
        model="gemini-3.6-flash", provider="google", api_key="fake",
        base_url="http://fake",
    )
    response = MockStreamingResponse(make_sse([
        {"choices": [{"delta": {
            "role": "assistant",
            "content": "Inspect the parser.",
            "extra_content": {"google": {"thought": True}},
        }}]},
        {"choices": [{"delta": {
            "role": "assistant",
            "content": "The parser is correct.",
        }}]},
        {"choices": [{"delta": {
            "role": "assistant",
            "extra_content": {"google": {"thought_signature": "opaque"}},
        }, "finish_reason": "stop"}]},
    ]))

    with patch.object(backend.client.chat.completions, "with_streaming_response") as raw:
        raw.create.return_value = response
        events = [event async for event in backend.stream(
            [Message(role="user", content=[TextContent(text="Check it")])],
            [], "system", think_level="high", reasoning_summary=True,
        )]

    assert "".join(e.data for e in events if e.type == "thinking_delta") == (
        "Inspect the parser."
    )
    assert "".join(e.data for e in events if e.type == "text_delta") == (
        "The parser is correct."
    )
    metadata = [e.data for e in events if e.type == "reasoning_metadata"]
    assert metadata[-1]["provider_state"]["thought_signature"] == "opaque"


def test_claude_summary_sets_display_and_preserves_signature():
    params = _anthropic_thinking_params(
        "claude-sonnet-4-6", "high", reasoning_summary=True
    )
    assert params["display"] == "summarized"

    messages = [Message(role="assistant", content=[ThinkingContent(
        thinking="Readable summary",
        signature="signature-1",
        reasoning_kind="summary",
        provider_state={
            "provider": "anthropic",
            "block_type": "thinking",
            "signature": "signature-1",
        },
    )])]
    assert _to_anthropic_messages(messages) == [{
        "role": "assistant",
        "content": [{
            "type": "thinking",
            "thinking": "Readable summary",
            "signature": "signature-1",
        }],
    }]


def test_openai_responses_replays_reasoning_item_and_tool_call_state():
    reasoning_item = {
        "type": "reasoning", "id": "rs_1", "summary": [],
        "encrypted_content": "opaque",
    }
    messages = [
        Message(role="assistant", content=[
            ThinkingContent(
                thinking="Summary", reasoning_kind="summary",
                provider_state={"provider": "openai", "reasoning_items": [reasoning_item]},
            ),
            ToolUseContent(
                id="call_1", name="Read", input={"path": "app.py"},
                provider_state={
                    "provider": "openai", "call_id": "call_1", "item_id": "fc_1",
                },
            ),
        ]),
    ]

    converted = _to_responses_input(messages)

    assert converted[0] == reasoning_item
    assert converted[1] == {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "Read", "arguments": '{"path": "app.py"}',
    }


class _AsyncEvents:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        async def iterate():
            for event in self.events:
                yield event
        return iterate()


@pytest.mark.asyncio
async def test_openai_responses_stream_emits_summary_state_text_and_usage():
    backend = OpenAIBackend(model="gpt-5.6", provider="openai", api_key="fake")
    reasoning_item = SimpleNamespace(
        type="reasoning",
        model_dump=lambda **_: {
            "type": "reasoning", "id": "rs_1", "summary": [],
            "encrypted_content": "opaque",
        },
    )
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        input_tokens_details=SimpleNamespace(cached_tokens=3, cache_write_tokens=2),
    )
    events = _AsyncEvents([
        SimpleNamespace(
            type="response.reasoning_summary_text.delta", item_id="rs_1",
            delta="Inspect first.",
        ),
        SimpleNamespace(type="response.output_item.done", item=reasoning_item),
        SimpleNamespace(type="response.output_text.delta", delta="Done."),
        SimpleNamespace(
            type="response.completed", response=SimpleNamespace(usage=usage),
        ),
    ])

    with patch.object(backend.client.responses, "create", AsyncMock(return_value=events)) as create:
        collected = [event async for event in backend.stream(
            [Message(role="user", content=[TextContent(text="Fix it")])],
            [], "system", think_level="high", reasoning_summary=True,
        )]

    sent = create.call_args.kwargs
    assert sent["reasoning"] == {"effort": "high", "summary": "auto"}
    assert sent["include"] == ["reasoning.encrypted_content"]
    assert sent["store"] is False
    assert "Inspect first." == "".join(
        event.data for event in collected if event.type == "thinking_delta"
    )
    state = [event.data for event in collected if event.type == "reasoning_metadata"][-1]
    assert state["provider_state"]["reasoning_items"][0]["encrypted_content"] == "opaque"
    ending = next(event.data for event in collected if event.type == "message_end")
    assert ending["cache_read_tokens"] == 3
    assert ending["cache_write_tokens"] == 2


def test_chat_conversion_does_not_leak_provider_state():
    messages = [Message(role="assistant", content=[
        ThinkingContent(
            thinking="summary", reasoning_kind="summary",
            provider_state={"provider": "openai", "encrypted_content": "opaque"},
        ),
        TextContent(text="answer"),
    ])]
    assert _to_openai_messages(messages) == [{"role": "assistant", "content": "answer"}]


def test_help_documents_reasoning_summary():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--reasoning-summary" in result.output


def test_llm_call_log_stores_structured_reasoning_and_legacy_text(tmp_path, monkeypatch):
    import json
    import cca.paths as paths

    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path)
    repl = Repl.__new__(Repl)
    repl._base_session_id = "base"
    repl._session_id = "session"
    repl._llm_call_idx = 0
    repl._features = SimpleNamespace(to_payload=lambda: {})
    repl._randomization_payload = None

    repl._log_llm_call(
        call_type="main", system="system", messages_snapshot=[], tools=[],
        params={"reasoning_summary": True}, response_text="Done.",
        tool_calls=[], thinking="Inspect first.", stop_reason="end_turn",
        input_tokens=10, output_tokens=5, cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning={
            "text": "Inspect first.", "kind": "summary",
            "requested": True, "captured": True,
        },
    )

    record = json.loads((tmp_path / "session_base.jsonl").read_text())
    assert record["response"]["thinking"] == "Inspect first."
    assert record["response"]["reasoning"] == {
        "text": "Inspect first.", "kind": "summary",
        "requested": True, "captured": True,
    }
