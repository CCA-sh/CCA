"""Anthropic API backend (native tool use, streaming)."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import anthropic

from cca.llm.base import (
    CompletionResponse,
    ContentBlock,
    LLMBackend,
    Message,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolSchema,
    ToolUseContent,
)
from cca.llm.retry import call_with_retry, stream_with_retry
from cca.prompts import SYSTEM_PROMPT_CACHE_BOUNDARY

# Name of the synthetic tool used to constrain a structured reply. Namespaced
# so it cannot collide with a real tool in the agent's allowlist.
_STRUCTURED_TOOL_NAME = "cca_structured_response"


def _build_system_blocks(system: str) -> list[dict] | str:
    """Split system prompt at the cache boundary.

    Returns a list of content blocks with cache_control on the static prefix,
    or a plain string if no boundary is present (fallback for other agents).
    """
    if SYSTEM_PROMPT_CACHE_BOUNDARY not in system:
        return system
    static, dynamic = system.split(SYSTEM_PROMPT_CACHE_BOUNDARY, 1)
    blocks = []
    if static.strip():
        blocks.append({
            "type": "text",
            "text": static.rstrip(),
            "cache_control": {"type": "ephemeral"},
        })
    if dynamic.strip():
        blocks.append({"type": "text", "text": dynamic.lstrip()})
    return blocks if blocks else system


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    """Convert internal messages to Anthropic format."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
            continue

        blocks = []
        for b in msg.content:
            if isinstance(b, ThinkingContent):
                state = b.provider_state or {}
                if state.get("block_type") == "redacted_thinking":
                    blocks.append({
                        "type": "redacted_thinking",
                        "data": state.get("data", ""),
                    })
                else:
                    signature = state.get("signature") or b.signature
                    block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": b.thinking,
                    }
                    if signature:
                        block["signature"] = signature
                    blocks.append(block)
            elif isinstance(b, TextContent):
                blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseContent):
                blocks.append({
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.input,
                })
            elif isinstance(b, ToolResultContent):
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": b.content,
                }
                if b.is_error:
                    block["is_error"] = True
                blocks.append(block)

        # Merge consecutive user tool_result blocks
        result.append({"role": msg.role, "content": blocks})

    return result


def _model_supports_thinking(model: str) -> bool:
    """Return True for Claude models that support any form of extended thinking."""
    m = model.lower()
    # claude-3.7+ in the 3.x series
    if any(f"claude-3-{v}" in m for v in ("7", "8", "9")):
        return True
    # claude-4.x series: opus-4, sonnet-4, haiku-4
    if any(f"claude-{v}-4" in m for v in ("opus", "sonnet", "haiku")):
        return True
    # claude-5.x series: fable-5, mythos-5, etc.
    if re.search(r"claude-[a-z]+-5\b", m):
        return True
    return False


def _model_uses_adaptive_thinking(model: str) -> bool:
    """Return True for models that use type:'adaptive' instead of budget_tokens.

    Newer Claude models (opus/sonnet-4-6+, 5.x) support adaptive thinking
    controlled by a separate effort value. Claude 5 removes manual budgets;
    4.6 deprecates them, so CCA uses the durable adaptive form for both.
    """
    m = model.lower()
    # Claude Opus/Sonnet 4.6 and newer sub-versions.
    for family in ("opus", "sonnet"):
        if any(f"claude-{family}-4-{v}" in m for v in ("6", "7", "8", "9")):
            return True
    # claude 5.x family
    if re.search(r"claude-[a-z]+-5\b", m):
        return True
    return False


def _model_defaults_to_adaptive_thinking(model: str) -> bool:
    """Return True when omitting ``thinking`` does not mean thinking is off."""
    m = model.lower()
    return bool(re.search(r"claude-(?:opus|sonnet)-5\b", m))


def _anthropic_thinking_params(
    model: str,
    think_level: str,
    reasoning_summary: bool = False,
) -> dict | None:
    """Return the Anthropic `thinking` kwarg dict for the given level, or None."""
    if not _model_supports_thinking(model):
        return None
    if _model_uses_adaptive_thinking(model):
        if think_level == "off":
            # Claude Opus/Sonnet 5 enable adaptive thinking when this field is
            # omitted. An explicit disabled value is therefore required for
            # CCA's portable "off" setting to remain truthful. Other adaptive
            # families (for example always-thinking models) keep omission as
            # their safest portable behavior.
            if _model_defaults_to_adaptive_thinking(model):
                return {"type": "disabled"}
            return None
        params = {"type": "adaptive"}
        if reasoning_summary:
            params["display"] = "summarized"
        return params
    if think_level == "off":
        return None
    # Manual budget control
    budget_map = {"low": 4_000, "medium": 8_000, "high": 32_000}
    budget = budget_map.get(think_level, 8_000)
    params = {"type": "enabled", "budget_tokens": budget}
    if reasoning_summary:
        params["display"] = "summarized"
    return params


def _anthropic_thinking_kwargs(
    model: str,
    think_level: str,
    reasoning_summary: bool = False,
) -> dict[str, Any]:
    """Return complete Anthropic request controls for portable thinking.

    Adaptive thinking and effort are independent Anthropic fields. Keeping
    them together here prevents callers from forwarding ``adaptive`` while
    silently losing the user's requested low/medium/high effort level.
    """
    thinking = _anthropic_thinking_params(model, think_level, reasoning_summary)
    if thinking is None:
        return {}
    kwargs: dict[str, Any] = {"thinking": thinking}
    if _model_uses_adaptive_thinking(model) and think_level != "off":
        kwargs["output_config"] = {"effort": think_level}
    return kwargs


class AnthropicBackend(LLMBackend):
    """Backend using the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        base_url = kwargs.pop("base_url", None)
        # Retries are owned by cca.llm.retry, not the SDK — see the same
        # note in openai_backend.
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, max_retries=0
        )
        self.extra_kwargs = kwargs

    def supports_reasoning_summary(self) -> bool:
        return _model_supports_thinking(self.model)

    def describe_request_settings(
        self,
        think_level: str,
        reasoning_summary: bool = False,
    ) -> dict[str, Any]:
        thinking = _anthropic_thinking_kwargs(
            self.model, think_level, reasoning_summary
        )
        return {
            "adapter": type(self).__name__,
            "transport": "anthropic-messages",
            "max_tokens_parameter": "max_tokens",
            "thinking_parameters": thinking,
            "reasoning_summary": reasoning_summary,
        }

    def supports_output_schema(self) -> bool:
        return True

    async def _complete_structured(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
        think_level: str,
        output_schema: dict[str, Any],
    ) -> CompletionResponse:
        """Constrain the reply via a forced tool call.

        Anthropic has no `response_format`, but forcing a single tool makes the
        arguments conform to its input schema, which is the same guarantee
        wearing a different hat. The capability-resolved thinking policy is
        still passed so Claude 5 receives an explicit disabled value when the
        formatting utility asks for off rather than silently using its
        thinking-on default.
        """
        tool = ToolSchema(
            name=_STRUCTURED_TOOL_NAME,
            description="Return the response as structured fields. Call this exactly once.",
            input_schema=output_schema,
        )
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=_build_system_blocks(system),
            messages=_to_anthropic_messages(messages),
            tools=[tool.to_anthropic()],
            tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
        )
        kwargs.update(_anthropic_thinking_kwargs(self.model, think_level))
        response = await self.client.messages.create(**kwargs)
        text = ""
        for block in response.content:
            if block.type == "tool_use" and block.name == _STRUCTURED_TOOL_NAME:
                text = json.dumps(block.input)
                break
        else:
            # No forced call came back (a provider quirk, or a refusal). Hand
            # back whatever text there was and let the caller validate; do not
            # invent an object.
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
        usage = response.usage
        return CompletionResponse(
            content=[TextContent(text=text)],
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            },
            mode_used="forced_tool",
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int = 8192,
        think_level: str = "off",
        reasoning_summary: bool = False,
        output_schema: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        """Retry-wrapped `_complete_once` — see `cca.llm.retry`."""
        return await call_with_retry(
            lambda: self._complete_once(
                messages, tools, system, max_tokens, think_level,
                reasoning_summary, output_schema,
            ),
            what=f"complete[{self.model}]",
        )

    async def _complete_once(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int = 8192,
        think_level: str = "off",
        reasoning_summary: bool = False,
        output_schema: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        if output_schema is not None:
            return await self._complete_structured(
                messages, system, max_tokens, think_level, output_schema
            )
        anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = [t.to_anthropic() for t in tools] if tools else []

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=_build_system_blocks(system),
            messages=anthropic_messages,
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        thinking_params = _anthropic_thinking_kwargs(
            self.model, think_level, reasoning_summary
        )
        kwargs.update(thinking_params)

        response = await self.client.messages.create(**kwargs)

        content: list[ContentBlock] = []
        for block in response.content:
            if block.type == "thinking":
                signature = getattr(block, "signature", None)
                content.append(ThinkingContent(
                    thinking=block.thinking,
                    signature=signature,
                    reasoning_kind="summary" if reasoning_summary else "unknown",
                    provider_state={
                        "provider": "anthropic",
                        "block_type": "thinking",
                        **({"signature": signature} if signature else {}),
                    },
                ))
            elif block.type == "redacted_thinking":
                content.append(ThinkingContent(
                    thinking="",
                    reasoning_kind="opaque",
                    provider_state={
                        "provider": "anthropic",
                        "block_type": "redacted_thinking",
                        "data": block.data,
                    },
                ))
            elif block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseContent(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        return CompletionResponse(
            content=content,
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                "cache_write_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            },
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int = 8192,
        think_level: str = "off",
        reasoning_summary: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Retry-wrapped `_stream_once` — see `cca.llm.retry`."""
        async for event in stream_with_retry(
            lambda: self._stream_once(
                messages, tools, system, max_tokens, think_level, reasoning_summary
            ),
            what=f"stream[{self.model}]",
        ):
            yield event

    async def _stream_once(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int = 8192,
        think_level: str = "off",
        reasoning_summary: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = [t.to_anthropic() for t in tools] if tools else []

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=_build_system_blocks(system),
            messages=anthropic_messages,
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        thinking_params = _anthropic_thinking_params(
            self.model, think_level, reasoning_summary
        )
        if thinking_params:
            kwargs["thinking"] = thinking_params

        # Track in-progress content blocks
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        current_tool_json: str = ""
        in_thinking_block = False
        thinking_signature = ""
        _input_tokens = 0
        _output_tokens = 0
        _cache_read_tokens = 0
        _cache_write_tokens = 0
        _stop_reason = "end_turn"

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = event.type

                if etype == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage:
                        _input_tokens = getattr(usage, "input_tokens", 0)
                        # Cache fields arrive on message_start, not message_delta
                        _cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                        _cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

                elif etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        _output_tokens = getattr(usage, "output_tokens", 0)
                    _stop_reason = getattr(event, "delta", None)
                    if _stop_reason:
                        _stop_reason = getattr(_stop_reason, "stop_reason", "end_turn") or "end_turn"
                    else:
                        _stop_reason = "end_turn"

                elif etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_json = ""
                        yield StreamEvent(type="tool_use_start", data={
                            "id": block.id,
                            "name": block.name,
                        })
                    elif block.type == "thinking":
                        in_thinking_block = True
                        thinking_signature = ""
                        yield StreamEvent(type="reasoning_metadata", data={
                            "kind": "summary" if reasoning_summary else "unknown",
                            "provider_state": {
                                "provider": "anthropic",
                                "block_type": "thinking",
                            },
                        })
                    elif block.type == "redacted_thinking":
                        yield StreamEvent(type="reasoning_metadata", data={
                            "kind": "opaque",
                            "provider_state": {
                                "provider": "anthropic",
                                "block_type": "redacted_thinking",
                                "data": block.data,
                            },
                        })

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamEvent(type="text_delta", data=delta.text)
                    elif delta.type == "input_json_delta":
                        current_tool_json += delta.partial_json
                    elif delta.type == "thinking_delta":
                        yield StreamEvent(type="thinking_delta", data=delta.thinking)
                    elif delta.type == "signature_delta":
                        thinking_signature += delta.signature

                elif etype == "content_block_stop":
                    if in_thinking_block:
                        if thinking_signature:
                            yield StreamEvent(type="reasoning_metadata", data={
                                "provider_state": {
                                    "provider": "anthropic",
                                    "block_type": "thinking",
                                    "signature": thinking_signature,
                                },
                            })
                        in_thinking_block = False
                    elif current_tool_id is not None:
                        try:
                            tool_input = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        yield StreamEvent(
                            type="tool_use_complete",
                            data=ToolUseContent(
                                id=current_tool_id,
                                name=current_tool_name or "",
                                input=tool_input,
                            ),
                        )
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_json = ""

                elif etype == "message_stop":
                    yield StreamEvent(type="message_end", data={
                        "input_tokens": _input_tokens,
                        "output_tokens": _output_tokens,
                        "cache_read_tokens": _cache_read_tokens,
                        "cache_write_tokens": _cache_write_tokens,
                        "stop_reason": _stop_reason,
                    })
