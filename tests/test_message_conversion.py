"""Unit tests for _to_openai_messages — especially thought_signature echo."""

from __future__ import annotations

import json

import pytest

from cca.llm.base import (
    Message,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from cca.llm.openai_backend import _to_openai_messages


def _tool_use(name: str, args: dict, thought_signature: str | None = None) -> ToolUseContent:
    return ToolUseContent(id=f"tc_{name}", name=name, input=args, thought_signature=thought_signature)


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------

def test_user_text_message():
    msgs = [Message(role="user", content=[TextContent(text="hello")])]
    result = _to_openai_messages(msgs)
    assert result == [{"role": "user", "content": "hello"}]


def test_assistant_text_message():
    msgs = [Message(role="assistant", content=[TextContent(text="hi there")])]
    result = _to_openai_messages(msgs)
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "hi there"


def test_deepseek_reasoning_content_is_replayed_when_marked():
    msgs = [Message(role="assistant", content=[
        ThinkingContent(
            thinking="provider reasoning state",
            provider_state={"replay_as_reasoning_content": True},
        ),
        TextContent(text="I will use a tool."),
        _tool_use("Read", {"file_path": "app.py"}),
    ])]

    result = _to_openai_messages(msgs)

    assert result[0]["reasoning_content"] == "provider reasoning state"
    assert result[0]["content"] == "I will use a tool."


def test_tool_result_becomes_tool_role():
    msgs = [Message(role="user", content=[ToolResultContent(tool_use_id="tc_x", content="output")])]
    result = _to_openai_messages(msgs)
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "tc_x"
    assert result[0]["content"] == "output"


# ---------------------------------------------------------------------------
# thought_signature echo
# ---------------------------------------------------------------------------

def _sig_from_tc(tc_entry: dict) -> str | None:
    """Extract thought_signature from the Gemini extra_content.google location."""
    return (tc_entry.get("extra_content") or {}).get("google", {}).get("thought_signature")


def test_thought_signature_included_when_present():
    tc = _tool_use("Bash", {"command": "ls"}, thought_signature="sig-abc")
    msgs = [Message(role="assistant", content=[tc])]
    result = _to_openai_messages(msgs)

    tc_entry = result[0]["tool_calls"][0]
    assert _sig_from_tc(tc_entry) == "sig-abc", (
        "thought_signature must be echoed in extra_content.google so Gemini 3 accepts it"
    )
    # Must NOT also appear in function dict (that location is ignored by Gemini)
    assert "thought_signature" not in tc_entry["function"], (
        "thought_signature must not be in function dict (Gemini ignores it there)"
    )


def test_thought_signature_omitted_when_none():
    tc = _tool_use("Bash", {"command": "ls"}, thought_signature=None)
    msgs = [Message(role="assistant", content=[tc])]
    result = _to_openai_messages(msgs)

    tc_entry = result[0]["tool_calls"][0]
    assert "extra_content" not in tc_entry, (
        "extra_content must not appear for providers that don't use thought_signature"
    )
    assert "thought_signature" not in tc_entry["function"], (
        "thought_signature must not appear in function dict either"
    )


def test_mixed_tools_each_have_correct_signature():
    tc_a = _tool_use("Bash", {"command": "ls"}, thought_signature="sig-a")
    tc_b = _tool_use("Read", {"file_path": "x.py"}, thought_signature="sig-b")
    msgs = [Message(role="assistant", content=[tc_a, tc_b])]
    result = _to_openai_messages(msgs)

    tool_calls = result[0]["tool_calls"]
    by_name = {tc["function"]["name"]: tc for tc in tool_calls}
    assert _sig_from_tc(by_name["Bash"]) == "sig-a"
    assert _sig_from_tc(by_name["Read"]) == "sig-b"


def test_adapter_owned_tool_state_is_replayed_without_runtime_interpretation():
    tc = ToolUseContent(
        id="tc_vendor",
        name="Read",
        input={"file_path": "x.py"},
        provider_state={
            "provider": "future-provider",
            "tool_call_extra_content": {
                "future-provider": {"opaque_continuation": "state-1"},
            },
        },
    )

    result = _to_openai_messages([Message(role="assistant", content=[tc])])

    assert result[0]["tool_calls"][0]["extra_content"] == {
        "future-provider": {"opaque_continuation": "state-1"},
    }


def test_arguments_round_trip():
    args = {"command": "echo hello", "timeout": 5000}
    tc = _tool_use("Bash", args)
    msgs = [Message(role="assistant", content=[tc])]
    result = _to_openai_messages(msgs)
    echoed = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
    assert echoed == args


# ---------------------------------------------------------------------------
# Multi-turn conversation ordering
# ---------------------------------------------------------------------------

def test_multi_turn_order_preserved():
    """user → assistant(tool_call) → user(tool_result) → assistant(text)"""
    msgs = [
        Message(role="user", content=[TextContent(text="run ls")]),
        Message(role="assistant", content=[_tool_use("Bash", {"command": "ls"}, "sig-1")]),
        Message(role="user", content=[ToolResultContent(tool_use_id="tc_Bash", content="file.py")]),
        Message(role="assistant", content=[TextContent(text="Done.")]),
    ]
    result = _to_openai_messages(msgs)
    roles = [m["role"] for m in result]
    assert roles == ["user", "assistant", "tool", "assistant"]
