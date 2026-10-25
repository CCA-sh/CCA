"""Abstract types and base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# ---------------------------------------------------------------------------
# Content block types
# ---------------------------------------------------------------------------

@dataclass
class TextContent:
    text: str
    type: str = field(default="text", init=False)


@dataclass
class ToolUseContent:
    id: str
    name: str
    input: dict[str, Any]
    thought_signature: str | None = None  # Gemini thinking models require this to be echoed back
    # Provider-owned opaque continuation data.  This is persisted for replay,
    # but must never be treated as readable reasoning or exported as SFT text.
    provider_state: dict[str, Any] | None = None
    type: str = field(default="tool_use", init=False)


@dataclass
class ToolResultContent:
    tool_use_id: str
    content: str
    is_error: bool = False
    name: str = ""  # tool name — required by Gemini's function_response.name
    type: str = field(default="tool_result", init=False)


@dataclass
class ThinkingContent:
    thinking: str
    # Anthropic returns a signature for multi-turn echoing; other backends don't.
    # Stored as-is when present; None for Gemini/OpenAI-compat reasoning.
    signature: str | None = None
    # Classification comes from the provider adapter, never text heuristics.
    # trace = directly emitted reasoning; summary = provider-produced summary;
    # rationale = ordinary prompted explanation; unknown = legacy/unclassified.
    reasoning_kind: str = "unknown"
    # Opaque state required by some providers for tool-use or multi-turn replay.
    # It is deliberately separate from the readable `thinking` field.
    provider_state: dict[str, Any] | None = None
    type: str = field(default="thinking", init=False)


ContentBlock = TextContent | ToolUseContent | ToolResultContent | ThinkingContent


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: list[ContentBlock] | str

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        role = d["role"]
        raw = d["content"]
        if isinstance(raw, str):
            return cls(role=role, content=raw)
        blocks: list[ContentBlock] = []
        for b in raw:
            t = b.get("type")
            if t == "text":
                blocks.append(TextContent(text=b["text"]))
            elif t == "tool_use":
                blocks.append(ToolUseContent(
                    id=b["id"], name=b["name"], input=b.get("input", {}),
                    thought_signature=b.get("thought_signature"),
                    provider_state=b.get("provider_state"),
                ))
            elif t == "tool_result":
                blocks.append(ToolResultContent(
                    tool_use_id=b["tool_use_id"],
                    content=b["content"],
                    is_error=b.get("is_error", False),
                    name=b.get("name", ""),
                ))
            elif t == "thinking":
                blocks.append(ThinkingContent(
                    thinking=b.get("thinking", ""),
                    signature=b.get("signature"),
                    reasoning_kind=b.get("reasoning_kind", "unknown"),
                    provider_state=b.get("provider_state"),
                ))
        return cls(role=role, content=blocks)

    def to_dict(self) -> dict:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        blocks = []
        for b in self.content:
            if isinstance(b, TextContent):
                blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseContent):
                block = {
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.input,
                }
                if b.thought_signature is not None:
                    block["thought_signature"] = b.thought_signature
                if b.provider_state is not None:
                    block["provider_state"] = b.provider_state
                blocks.append(block)
            elif isinstance(b, ToolResultContent):
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": b.content,
                }
                if b.is_error:
                    block["is_error"] = True
                blocks.append(block)
            elif isinstance(b, ThinkingContent):
                block = {"type": "thinking", "thinking": b.thinking}
                if b.signature is not None:
                    block["signature"] = b.signature
                if b.reasoning_kind != "unknown":
                    block["reasoning_kind"] = b.reasoning_kind
                if b.provider_state is not None:
                    block["provider_state"] = b.provider_state
                blocks.append(block)
        return {"role": self.role, "content": blocks}


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    type: str  # text_delta | tool_use_start | tool_use_complete | message_end | error
    data: Any = None


# ---------------------------------------------------------------------------
# Completion response
# ---------------------------------------------------------------------------

@dataclass
class CompletionResponse:
    content: list[ContentBlock]
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    # Which structured-output mechanism actually ran, when `output_schema` was
    # requested: "json_schema" (constrained decoding), "forced_tool", or
    # "prompt_only" (the provider rejected the constraint). None for every
    # ordinary call. Callers record it so a route that silently cannot be
    # constrained is visible rather than looking like a bad model.
    mode_used: str | None = None

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextContent))

    @property
    def tool_calls(self) -> list[ToolUseContent]:
        return [b for b in self.content if isinstance(b, ToolUseContent)]

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.content)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract base class for all LLM provider backends."""

    def supports_reasoning_summary(self) -> bool:
        """Whether this provider/model can return a readable reasoning summary."""
        return False

    def describe_request_settings(
        self,
        think_level: str,
        reasoning_summary: bool = False,
    ) -> dict[str, Any]:
        """Return safe, static controls that will be added to an LLM request."""
        return {
            "adapter": type(self).__name__,
            "thinking_parameters": {},
            "reasoning_summary": reasoning_summary,
        }

    def supports_output_schema(self) -> bool:
        """Whether `complete(output_schema=...)` constrains the reply at all.

        False means the caller must fall back to asking for JSON in the prompt
        and validating what comes back.
        """
        return False

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int = 8192,
        think_level: str = "off",  # "off" | "low" | "medium" | "high"
        reasoning_summary: bool = False,
        output_schema: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        """Non-streaming completion.

        `output_schema` is a JSON Schema the reply must conform to. Backends
        translate it into whatever their provider supports (constrained
        decoding, a forced tool call) and degrade to prompt-only when the
        provider rejects it; `CompletionResponse.text` carries the JSON either
        way. Ignored when None, which is every call in the agentic loop.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int = 8192,
        think_level: str = "off",  # "off" | "low" | "medium" | "high"
        reasoning_summary: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion, yielding StreamEvent objects."""
        ...
