"""Retrying a refused structured-output call with neutral field names."""

from __future__ import annotations

import json

import pytest

from cca.llm.base import CompletionResponse, TextContent
from cca.structured_output import (
    _INSTRUCTION,
    MAX_ATTEMPTS,
    StructuredOutputError,
    format_as_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "internal_thought": {"type": "string"},
        "continue": {"type": "boolean"},
    },
    "required": ["prompt", "internal_thought", "continue"],
    "additionalProperties": False,
}
ANSWER = json.dumps({"prompt": "add a README", "internal_thought": "keep it short", "continue": True})


def reply(text: str, stop_reason: str = "end_turn", mode: str = "forced_tool") -> CompletionResponse:
    return CompletionResponse(content=[TextContent(text=text)], stop_reason=stop_reason, mode_used=mode)


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


def sent_text(call: dict) -> str:
    return call["messages"][0].content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["content_filter", "refusal"])
async def test_refused_call_is_retried_with_a_neutral_field_name_and_mapped_back(stop_reason):
    aliased = json.dumps({"prompt": "add a README", "internal_notes": "keep it short", "continue": True})
    backend = ScriptedBackend([reply("{}", stop_reason), reply(aliased)])

    result = await format_as_schema(backend, ANSWER, SCHEMA, think_level="off")

    assert result["object"] == json.loads(ANSWER)
    assert result["field_aliases"] == {"internal_thought": "internal_notes"}
    assert result["mode_used"] == "forced_tool"
    assert result["attempts"] == 2
    retry = backend.calls[1]
    assert "internal_thought" not in retry["output_schema"]["properties"]
    assert "internal_notes" in retry["output_schema"]["required"]
    assert "internal_thought" not in sent_text(retry) and "internal_notes" in sent_text(retry)
    assert "previous attempt was rejected" not in sent_text(retry)


@pytest.mark.asyncio
async def test_request_is_unchanged_when_the_provider_does_not_refuse():
    backend = ScriptedBackend([reply(ANSWER, mode="json_schema")])

    result = await format_as_schema(backend, ANSWER, SCHEMA, think_level="off")

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["output_schema"] is SCHEMA
    assert sent_text(call) == _INSTRUCTION.format(schema=json.dumps(SCHEMA, indent=2), answer=ANSWER)
    assert "field_aliases" not in result
    assert result["attempts"] == 1


@pytest.mark.asyncio
async def test_ordinary_retries_are_unchanged_and_never_alias():
    backend = ScriptedBackend([reply("not json at all"), reply(ANSWER)])

    result = await format_as_schema(backend, ANSWER, SCHEMA, think_level="off")

    assert result["attempts"] == 2
    assert "field_aliases" not in result
    assert all(call["output_schema"] is SCHEMA for call in backend.calls)
    assert "previous attempt was rejected: the reply was not a JSON object" in sent_text(backend.calls[1])


@pytest.mark.asyncio
async def test_a_refusal_that_persists_still_fails_after_the_same_number_of_attempts():
    backend = ScriptedBackend([reply("", "content_filter")])

    with pytest.raises(StructuredOutputError) as excinfo:
        await format_as_schema(backend, ANSWER, SCHEMA, think_level="off")

    assert excinfo.value.attempts == MAX_ATTEMPTS
    assert len(backend.calls) == MAX_ATTEMPTS
    assert "internal_thought" in backend.calls[0]["output_schema"]["properties"]
    assert all("internal_thought" not in c["output_schema"]["properties"] for c in backend.calls[1:])


@pytest.mark.asyncio
async def test_a_refusal_with_nothing_to_alias_behaves_as_before():
    plain = {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
    backend = ScriptedBackend([reply("", "content_filter"), reply("", "content_filter"), reply("", "content_filter")])

    with pytest.raises(StructuredOutputError) as excinfo:
        await format_as_schema(backend, json.dumps({"prompt": "x"}), plain, think_level="off")

    assert "the reply was not a JSON object" in str(excinfo.value)
    assert all(call["output_schema"] is plain for call in backend.calls)
    assert "previous attempt was rejected: the reply was not a JSON object" in sent_text(backend.calls[1])


@pytest.mark.asyncio
async def test_alias_avoids_colliding_with_an_existing_field():
    schema = {
        "type": "object",
        "properties": {"internal_thought": {"type": "string"}, "internal_notes": {"type": "string"}},
        "required": ["internal_thought", "internal_notes"],
    }
    answer = json.dumps({"internal_thought": "a", "internal_notes": "b"})
    backend = ScriptedBackend([reply("{}", "content_filter"), reply(json.dumps({"internal_notes_": "a", "internal_notes": "b"}))])

    result = await format_as_schema(backend, answer, schema, think_level="off")

    assert result["field_aliases"] == {"internal_thought": "internal_notes_"}
    assert result["object"] == {"internal_thought": "a", "internal_notes": "b"}


@pytest.mark.asyncio
async def test_answer_that_is_not_json_still_gets_its_field_names_translated():
    prose = 'Reply: {"prompt": "x", "internal_thought": "y", "continue": false}'
    aliased = json.dumps({"prompt": "x", "internal_notes": "y", "continue": False})
    backend = ScriptedBackend([reply("", "refusal"), reply(aliased)])

    result = await format_as_schema(backend, prose, SCHEMA, think_level="off")

    assert '"internal_notes"' in sent_text(backend.calls[1]) and '"internal_thought"' not in sent_text(backend.calls[1])
    assert result["object"]["internal_thought"] == "y"


@pytest.mark.asyncio
async def test_a_backend_reply_without_a_stop_reason_is_treated_as_not_refused():
    class Bare:
        text = ANSWER
        mode_used = "json_schema"

    class BareBackend:
        async def complete(self, **kwargs):
            return Bare()

    result = await format_as_schema(BareBackend(), ANSWER, SCHEMA, think_level="off")
    assert result["object"] == json.loads(ANSWER)
