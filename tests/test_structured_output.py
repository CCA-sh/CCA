"""`--json-schema`: constrain the final `--print` answer, or fail loudly.

The formatting phase runs *after* the agentic loop as its own tool-free call.
Its caller supplies the capability-resolved utility thinking level so models
that cannot disable thinking still receive a legal request.
"""

from __future__ import annotations

import json

import pytest

from cca.llm.base import CompletionResponse, Message, TextContent
from cca.llm.openai_backend import _is_unsupported_schema_error, _strict_json_schema
from cca.structured_output import (
    MAX_ATTEMPTS,
    StructuredOutputError,
    extract_json_object,
    format_as_schema,
    validate_against_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "satisfaction": {"type": "number", "minimum": 0, "maximum": 1},
        "continue": {"type": "boolean"},
        "tasks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompt", "satisfaction", "continue"],
    "additionalProperties": False,
}

VALID = {"prompt": "add a flag", "satisfaction": 0.4, "continue": True, "tasks": ["impl"]}


class _Backend:
    """Records how it was called and replays scripted replies."""

    def __init__(self, replies, mode_used="json_schema"):
        self._replies = list(replies)
        self._mode = mode_used
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return CompletionResponse(
            content=[TextContent(text=reply)],
            mode_used=self._mode,
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Here you go:\n{"a": 1}\nhope that helps',
])
def test_extract_json_object_handles_real_reply_shapes(text):
    assert extract_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '{"a": '])
def test_extract_json_object_returns_none_rather_than_raising(text):
    assert extract_json_object(text) is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_object_has_no_problems():
    assert validate_against_schema(VALID, SCHEMA) == []


def test_missing_required_field_is_reported():
    obj = {k: v for k, v in VALID.items() if k != "continue"}
    assert any("continue" in p for p in validate_against_schema(obj, SCHEMA))


def test_wrong_type_is_reported():
    assert any("satisfaction" in p for p in validate_against_schema(
        {**VALID, "satisfaction": "high"}, SCHEMA))


def test_a_boolean_does_not_satisfy_number():
    """bool subclasses int in Python; the checker must not accept True as 1."""
    assert any("satisfaction" in p for p in validate_against_schema(
        {**VALID, "satisfaction": True}, SCHEMA))


def test_numeric_bounds_are_enforced():
    assert any(">= 0" in p or "<= 1" in p for p in validate_against_schema(
        {**VALID, "satisfaction": 1.7}, SCHEMA))


def test_additional_properties_false_is_enforced():
    assert any("surprise" in p for p in validate_against_schema(
        {**VALID, "surprise": 1}, SCHEMA))


def test_array_item_types_are_checked():
    assert any("tasks[0]" in p for p in validate_against_schema(
        {**VALID, "tasks": [1]}, SCHEMA))


def test_enum_is_enforced():
    schema = {"type": "object", "properties": {"mood": {"enum": ["a", "b"]}},
              "required": ["mood"]}
    assert validate_against_schema({"mood": "a"}, schema) == []
    assert any("mood" in p for p in validate_against_schema({"mood": "z"}, schema))


def test_unknown_schema_keywords_are_not_treated_as_violations():
    """Better to accept something a real validator would accept than to reject
    a correct answer over a keyword this checker doesn't implement."""
    schema = {"type": "object", "properties": {"x": {"type": "string", "format": "uri"}},
              "required": ["x"], "allOf": [{"type": "object"}]}
    assert validate_against_schema({"x": "not-a-uri"}, schema) == []


# ---------------------------------------------------------------------------
# The formatting call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_returns_the_object_and_reports_the_mode():
    backend = _Backend([json.dumps(VALID)])
    result = await format_as_schema(
        backend, "add a flag please", SCHEMA, think_level="off"
    )

    assert result["object"] == VALID
    assert json.loads(result["text"]) == VALID
    assert result["mode_used"] == "json_schema"
    assert result["attempts"] == 1
    assert result["think_level"] == "off"


@pytest.mark.asyncio
async def test_format_call_uses_resolved_thinking_level_and_carries_the_schema():
    backend = _Backend([json.dumps(VALID)])
    await format_as_schema(backend, "answer", SCHEMA, think_level="low")

    call = backend.calls[0]
    assert call["tools"] == []
    assert call["think_level"] == "low"
    assert call["output_schema"] == SCHEMA
    assert isinstance(call["messages"][0], Message)


@pytest.mark.asyncio
async def test_format_retries_with_the_problems_then_succeeds():
    backend = _Backend(['{"prompt": "x"}', json.dumps(VALID)])
    result = await format_as_schema(
        backend, "answer", SCHEMA, think_level="off"
    )

    assert result["attempts"] == 2
    retry_text = backend.calls[1]["messages"][0].content[0].text
    assert "rejected" in retry_text and "satisfaction" in retry_text


@pytest.mark.asyncio
async def test_format_raises_instead_of_returning_something_invalid():
    backend = _Backend(['{"prompt": "x"}'])
    with pytest.raises(StructuredOutputError) as excinfo:
        await format_as_schema(backend, "answer", SCHEMA, think_level="off")

    assert excinfo.value.attempts == MAX_ATTEMPTS
    assert len(backend.calls) == MAX_ATTEMPTS
    assert any("satisfaction" in p for p in excinfo.value.problems)


@pytest.mark.asyncio
async def test_prompt_only_degradation_is_reported_not_hidden():
    backend = _Backend([json.dumps(VALID)], mode_used="prompt_only")
    result = await format_as_schema(
        backend, "answer", SCHEMA, think_level="off"
    )
    assert result["mode_used"] == "prompt_only"


# ---------------------------------------------------------------------------
# Provider-shape helpers
# ---------------------------------------------------------------------------


def test_strict_schema_requires_every_declared_property():
    """OpenAI strict mode 400s when `required` omits a declared property — the
    exact shape the sim harness' own schema had."""
    strict = _strict_json_schema(SCHEMA)
    assert set(strict["required"]) == set(SCHEMA["properties"])
    assert strict["additionalProperties"] is False
    # The caller's schema must not be mutated.
    assert SCHEMA["required"] == ["prompt", "satisfaction", "continue"]


def test_strict_schema_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
            },
        },
        "required": ["inner"],
    }
    strict = _strict_json_schema(schema)
    inner = strict["properties"]["inner"]
    assert set(inner["required"]) == {"a", "b"}
    assert inner["additionalProperties"] is False


class _Err(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("exc", [
    _Err("Invalid parameter: response_format is not supported", 400),
    _Err("json_schema response format unavailable for this model", 400),
    _Err("structured output not supported", 404),
])
def test_unsupported_schema_errors_are_recognized(exc):
    assert _is_unsupported_schema_error(exc)


@pytest.mark.parametrize("exc", [
    _Err("Rate limit exceeded", 429),
    _Err("Invalid API key", 401),
    _Err("context length exceeded, reduce response_format", 413),
    _Err("prompt is too long", 400),
])
def test_unrelated_errors_are_not_swallowed_as_degradation(exc):
    """A 429 retried unconstrained would hide a real failure behind a worse
    answer, so only genuine schema-support errors may degrade."""
    assert not _is_unsupported_schema_error(exc)


@pytest.mark.asyncio
async def test_native_anthropic_formatter_explicitly_disables_sonnet_5_thinking():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cca.llm.anthropic_backend import AnthropicBackend
    from cca.llm.base import Message

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend.model = "claude-sonnet-5"
    backend.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(
                type="tool_use",
                name="cca_structured_response",
                input={"ok": True},
            )],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
    )))

    result = await backend._complete_structured(
        [Message(role="user", content="format this")],
        "system",
        64,
        "off",
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )

    sent = backend.client.messages.create.await_args.kwargs
    assert sent["thinking"] == {"type": "disabled"}
    assert "output_config" not in sent
    assert result.mode_used == "forced_tool"
    assert json.loads(result.text) == {"ok": True}


# ---------------------------------------------------------------------------
# End-to-end through Repl.run_print
# ---------------------------------------------------------------------------


class _LoopBackend:
    """Streams a plain-text answer for the agentic turn, then formats on demand.

    The two phases are deliberately different calls: `stream` runs the loop,
    `complete` (with `output_schema`) runs the formatting step.
    """

    def __init__(self, answer, formatted):
        self.answer = answer
        self.formatted = formatted
        self.complete_calls: list[dict] = []

    async def stream(self, **kwargs):
        from cca.llm.base import StreamEvent

        yield StreamEvent(type="text_delta", data=self.answer)
        yield StreamEvent(type="message_end", data={
            "input_tokens": 1, "output_tokens": 1, "stop_reason": "end_turn",
        })

    async def complete(self, **kwargs):
        self.complete_calls.append(kwargs)
        return CompletionResponse(
            content=[TextContent(text=self.formatted)], mode_used="json_schema",
        )


class _Registry:
    _mcp_manager = None

    def get_schemas(self, **kwargs):
        return []

    def get(self, name):
        return None

    async def execute(self, name, tool_input):  # pragma: no cover - unused
        raise AssertionError("no tools in this test")


def _repl(tmp_path, monkeypatch, backend, schema, output_format="json"):
    from cca.config import Config
    from cca.permissions import PermissionManager
    from cca.repl import Repl

    monkeypatch.setattr("cca.sessions.save_session", lambda *args, **kwargs: None)
    return Repl(
        backend=backend,
        registry=_Registry(),
        config=Config(features={"memory": False, "compaction": False}),
        permissions=PermissionManager(auto_approve=True, cwd=str(tmp_path)),
        cwd=str(tmp_path),
        print_mode=True,
        run_mode="print",
        output_format=output_format,
        output_schema=schema,
    )


@pytest.mark.asyncio
async def test_run_print_returns_the_formatted_object_and_its_provenance(
    tmp_path, monkeypatch
):
    backend = _LoopBackend("I'd add a flag", json.dumps(VALID))
    repl = _repl(tmp_path, monkeypatch, backend, SCHEMA)

    payload = json.loads(await repl.run_print("what next?"))

    assert json.loads(payload["result"]) == VALID
    assert payload["structured_output"] == {
        "mode_used": "json_schema",
        "attempts": 1,
        "think_level": "off",
        "valid": True,
    }
    # The formatting phase is one extra call, and it sees the loop's answer.
    assert len(backend.complete_calls) == 1
    assert "I'd add a flag" in backend.complete_calls[0]["messages"][0].content[0].text


@pytest.mark.asyncio
async def test_run_print_without_a_schema_is_untouched(tmp_path, monkeypatch):
    """Every interactive and machine-mode run takes this path; it must not gain
    a formatting call or a structured_output key."""
    backend = _LoopBackend("I'd add a flag", json.dumps(VALID))
    repl = _repl(tmp_path, monkeypatch, backend, None)

    payload = json.loads(await repl.run_print("what next?"))

    assert payload["result"] == "I'd add a flag"
    assert "structured_output" not in payload
    assert backend.complete_calls == []


@pytest.mark.asyncio
async def test_run_print_raises_rather_than_printing_a_non_conforming_answer(
    tmp_path, monkeypatch
):
    backend = _LoopBackend("I'd add a flag", "sorry, I can't do that")
    repl = _repl(tmp_path, monkeypatch, backend, SCHEMA)

    with pytest.raises(StructuredOutputError):
        await repl.run_print("what next?")
