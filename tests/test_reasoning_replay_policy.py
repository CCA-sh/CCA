"""Which LiteLLM routes echo returned reasoning_content back on later calls."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai
import pytest
from rich.console import Console

from cca.llm.base import Message, TextContent, ThinkingContent, ToolUseContent
from cca.llm.factory import create_backend
from cca.llm.litellm_thinking import (
    get_litellm_thinking_profile,
    replays_reasoning_content,
)
from cca.llm.openai_backend import _to_openai_messages
from cca.llm.thinking import resolve_thinking
from cca.repl import Repl
from tests.conftest import MockStreamingResponse, make_sse

REPLAYED_ROUTES = [
    "fireworks_ai/deepseek-v4p1-flash",
    "fireworks_ai/kimi-k3",
    "fireworks_ai/glm-5p3",
    "fireworks_ai/qwen3p8-max",
    "fireworks_ai/deepseek-v4-pro-0813",
    "fireworks_ai/deepseek-v4-flash-0731",
    "fireworks_ai/minimax-m3",
    "fireworks_ai/nemotron-3-ultra-nvfp4",
    "fireworks_ai/inkling",
    "fireworks_ai/muse-glimmer-30b",
    "gemini/gemini-3.7-flash",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.1-pro-preview",
]

NOT_REPLAYED_ROUTES = [
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5",
    "anthropic/claude-fable-5",
    "anthropic/claude-fable-5-1",
    "groq/qwen/qwen3.8-27b",
    "groq/qwen/qwen3.6-27b",
    "gemini/gemma-4-31b-it",
    "xai/grok-4",
    "mistral/magistral-medium-latest",
    "bedrock/mistral.devstral-2-123b",
]


def _litellm_backend(route: str):
    return create_backend(
        f"litellm:{route}", api_key="test", base_url="https://proxy.example.test/v1"
    )


def _completion_handler(reasoning: str | None):
    def handler(request: httpx.Request) -> httpx.Response:
        message = {"role": "assistant", "content": "answer"}
        if reasoning:
            message["reasoning_content"] = reasoning
        return httpx.Response(200, json={
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    return handler


async def _complete(route: str, reasoning: str | None):
    backend = _litellm_backend(route)
    backend.client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://proxy.example.test/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_completion_handler(reasoning))
        ),
    )
    response = await backend.complete(
        [Message(role="user", content="hello")], [], "system", think_level="high"
    )
    await backend.client.close()
    return response


async def _stream_thinking_blocks(route: str, reasoning: str) -> list[dict]:
    backend = _litellm_backend(route)
    sse = make_sse([
        {"choices": [{"delta": {"reasoning_content": reasoning}}]},
        {"choices": [{"delta": {"content": "answer"}}, {"finish_reason": "stop"}]},
    ])
    with patch.object(backend.client.chat.completions, "with_streaming_response") as wsr:
        wsr.create.return_value = MockStreamingResponse(sse)
        events = [
            event
            async for event in backend.stream(
                [Message(role="user", content="hello")], [], "system", max_tokens=64
            )
        ]
    return [e.data for e in events if e.type == "reasoning_metadata"]


@pytest.mark.parametrize("route", REPLAYED_ROUTES)
def test_reasoning_is_replayed_for_routes_other_agents_replay(route):
    assert replays_reasoning_content(route) is True


@pytest.mark.parametrize("route", NOT_REPLAYED_ROUTES)
def test_reasoning_is_not_replayed_where_it_travels_another_way(route):
    assert replays_reasoning_content(route) is False


def test_unregistered_route_replays_unless_its_provider_uses_another_vehicle():
    assert get_litellm_thinking_profile("fireworks_ai/brand-new-model") is None
    assert replays_reasoning_content("fireworks_ai/brand-new-model") is True
    assert replays_reasoning_content("openai/brand-new-model") is False
    assert replays_reasoning_content("groq/brand-new-model") is False


@pytest.mark.parametrize(
    "route", ["fireworks_ai/inkling", "fireworks_ai/muse-glimmer-30b", "fireworks_ai/nemotron-3-ultra-nvfp4"]
)
def test_previously_unregistered_routes_now_receive_the_requested_effort(route):
    assert get_litellm_thinking_profile(route) is not None
    backend = _litellm_backend(route)
    assert backend.describe_request_settings("high")["thinking_parameters"] == {
        "reasoning_effort": "high",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }
    resolved = resolve_thinking(f"litellm:{route}", "high", "auto")
    assert (resolved.effective, resolved.source, resolved.warning) == ("high", "explicit", None)


def test_off_on_newly_registered_route_resolves_to_lowest_effort_not_none():
    resolved = resolve_thinking("litellm:fireworks_ai/inkling", "off", "auto")
    assert resolved.effective == "low"


def test_route_without_profile_still_reports_that_thinking_was_dropped():
    resolved = resolve_thinking("litellm:bedrock/mistral.devstral-2-123b", "high", "auto")
    assert resolved.effective == "off"
    assert resolved.source == "unsupported"
    assert "Requested high" in resolved.warning


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["fireworks_ai/kimi-k3", "fireworks_ai/glm-5p3", "gemini/gemini-3.8-flash"])
async def test_returned_reasoning_is_marked_for_replay(route):
    response = await _complete(route, "earlier plan")
    thinking = response.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.provider_state == {
        "provider": "litellm",
        "replay_as_reasoning_content": True,
    }
    replayed = _to_openai_messages([Message(role="assistant", content=response.content)])
    assert replayed[0]["reasoning_content"] == "earlier plan"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["anthropic/claude-sonnet-5", "openai/gpt-5.6-sol", "groq/qwen/qwen3.8-27b"])
async def test_returned_reasoning_is_kept_but_not_sent_back_where_replay_is_off(route):
    response = await _complete(route, "hidden rationale")
    thinking = response.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.thinking == "hidden rationale"
    assert thinking.provider_state is None
    replayed = _to_openai_messages([Message(role="assistant", content=response.content)])
    assert "reasoning_content" not in replayed[0]


@pytest.mark.asyncio
async def test_route_that_returns_no_reasoning_gets_nothing_to_replay():
    response = await _complete("fireworks_ai/kimi-k3", None)
    assert not any(isinstance(block, ThinkingContent) for block in response.content)
    replayed = _to_openai_messages([Message(role="assistant", content=response.content)])
    assert "reasoning_content" not in replayed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["fireworks_ai/kimi-k3", "fireworks_ai/glm-5p3", "fireworks_ai/inkling"])
async def test_streamed_reasoning_is_marked_for_replay(route):
    metadata = await _stream_thinking_blocks(route, "streamed plan")
    assert any(
        (item.get("provider_state") or {}).get("replay_as_reasoning_content") is True
        for item in metadata
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["anthropic/claude-sonnet-5", "openai/gpt-5.6-sol"])
async def test_streamed_reasoning_is_not_marked_where_replay_is_off(route):
    metadata = await _stream_thinking_blocks(route, "streamed plan")
    assert not any((item.get("provider_state") or {}).get("replay_as_reasoning_content") for item in metadata)


def test_tool_calls_and_replayed_reasoning_travel_together():
    messages = [Message(role="assistant", content=[
        ThinkingContent(
            thinking="why this call",
            provider_state={"provider": "litellm", "replay_as_reasoning_content": True},
        ),
        TextContent(text="running it"),
        ToolUseContent(id="tc_1", name="Bash", input={"command": "ls"}),
    ])]
    sent = _to_openai_messages(messages)[0]
    assert sent["reasoning_content"] == "why this call"
    assert sent["tool_calls"][0]["id"] == "tc_1"


@pytest.mark.parametrize(
    "route,expected",
    [("fireworks_ai/kimi-k3", True), ("gemini/gemini-3.8-flash", True), ("anthropic/claude-opus-5", False), ("openai/gpt-5.6-sol", False)],
)
def test_request_settings_report_whether_reasoning_is_replayed(route, expected):
    assert _litellm_backend(route).describe_request_settings("high")["reasoning_replay"] is expected


def test_non_litellm_provider_reports_no_reasoning_replay():
    backend = create_backend("openai:gpt-5.5", api_key="test")
    assert backend.describe_request_settings("high")["reasoning_replay"] is False


def _thinking_announcer(warning: str | None):
    stream = StringIO()
    stub = SimpleNamespace(
        _thinking_resolution=SimpleNamespace(warning=warning),
        _console=Console(file=stream, force_terminal=False, width=200),
    )
    Repl._announce_thinking_warning(stub)
    return stream.getvalue()


def test_unhonoured_thinking_request_is_announced():
    printed = _thinking_announcer("Requested high, but no adapter; using off.")
    assert "Requested high, but no adapter; using off." in printed


def test_honoured_thinking_request_prints_nothing():
    assert _thinking_announcer(None) == ""
