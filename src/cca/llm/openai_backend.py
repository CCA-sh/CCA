"""OpenAI-compatible backend (covers OpenAI, Groq, Ollama, Mistral, Together, etc.)."""

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from typing import Any, AsyncIterator

import openai

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


def _tool_call_extra_content(*values: Any) -> dict[str, Any] | None:
    """Return adapter-owned continuation metadata from an SDK or raw payload."""
    merged: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            extras = value
        else:
            extras = getattr(value, "model_extra", {}) or {}
        extra_content = extras.get("extra_content")
        if isinstance(extra_content, dict):
            _merge_provider_state(merged, extra_content)
    return merged or None


def _merge_provider_state(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_provider_state(target[key], value)
        else:
            target[key] = deepcopy(value)


def _thought_signature(
    extra_content: dict[str, Any] | None,
    *fallbacks: Any,
) -> str | None:
    google = (extra_content or {}).get("google") or {}
    if isinstance(google, dict) and isinstance(google.get("thought_signature"), str):
        return google["thought_signature"]
    return next((value for value in fallbacks if isinstance(value, str) and value), None)


def _tool_provider_state(
    provider: str,
    extra_content: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not extra_content:
        return None
    return {
        "provider": provider or "openai-compatible",
        "tool_call_extra_content": deepcopy(extra_content),
    }


def _replay_tool_extra_content(tool_call: ToolUseContent) -> dict[str, Any] | None:
    state = tool_call.provider_state or {}
    stored = state.get("tool_call_extra_content")
    extra_content = deepcopy(stored) if isinstance(stored, dict) else None
    if tool_call.thought_signature:
        if extra_content is None:
            extra_content = {}
        google = extra_content.setdefault("google", {})
        if isinstance(google, dict):
            google.setdefault("thought_signature", tool_call.thought_signature)
    return extra_content


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    """Convert internal Message format to OpenAI chat format."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
            continue

        # Separate tool_result blocks (must become individual tool-role messages)
        tool_results = [b for b in msg.content if isinstance(b, ToolResultContent)]
        other_blocks = [b for b in msg.content if not isinstance(b, ToolResultContent)]

        if other_blocks:
            # Check for tool_use blocks
            tool_calls = [b for b in other_blocks if isinstance(b, ToolUseContent)]
            text_blocks = [b for b in other_blocks if isinstance(b, TextContent)]
            thinking_blocks = [
                b for b in other_blocks if isinstance(b, ThinkingContent)
            ]

            assistant_msg: dict[str, Any] = {"role": msg.role}
            if text_blocks:
                assistant_msg["content"] = "".join(b.text for b in text_blocks) or None
            else:
                assistant_msg["content"] = None

            replay_reasoning = "".join(
                block.thinking
                for block in thinking_blocks
                if (block.provider_state or {}).get(
                    "replay_as_reasoning_content"
                )
            )
            if replay_reasoning:
                assistant_msg["reasoning_content"] = replay_reasoning

            if tool_calls:
                openai_tool_calls = []
                for tc in tool_calls:
                    tc_entry: dict[str, Any] = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input),
                        },
                    }
                    extra_content = _replay_tool_extra_content(tc)
                    if extra_content:
                        tc_entry["extra_content"] = extra_content
                    openai_tool_calls.append(tc_entry)
                assistant_msg["tool_calls"] = openai_tool_calls
            result.append(assistant_msg)

        for tr in tool_results:
            tr_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": tr.content,
            }
            if tr.name:
                tr_msg["name"] = tr.name
            result.append(tr_msg)

    return result


def _to_responses_input(messages: list[Message]) -> list[dict]:
    """Convert internal messages to manually-managed Responses API input.

    Provider reasoning items are replayed unchanged before the assistant output
    they belong to.  Readable summaries alone are not sufficient continuation
    state and are never fabricated into reasoning items.
    """
    result: list[dict] = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
            continue

        text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
        thinking_blocks = [b for b in msg.content if isinstance(b, ThinkingContent)]
        tool_calls = [b for b in msg.content if isinstance(b, ToolUseContent)]
        tool_results = [b for b in msg.content if isinstance(b, ToolResultContent)]

        if msg.role == "assistant":
            for block in thinking_blocks:
                state = block.provider_state or {}
                items = state.get("reasoning_items")
                if not isinstance(items, list):
                    legacy_item = state.get("reasoning_item")
                    items = [legacy_item] if isinstance(legacy_item, dict) else []
                result.extend(item for item in items if isinstance(item, dict))
            if text_blocks:
                result.append({
                    "role": "assistant",
                    "content": "".join(b.text for b in text_blocks),
                })
            for tc in tool_calls:
                state = tc.provider_state or {}
                result.append({
                    "type": "function_call",
                    "call_id": state.get("call_id") or tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.input),
                    **({"id": state["item_id"]} if state.get("item_id") else {}),
                })
        else:
            if text_blocks:
                result.append({
                    "role": msg.role,
                    "content": "".join(b.text for b in text_blocks),
                })
            for tr in tool_results:
                result.append({
                    "type": "function_call_output",
                    "call_id": tr.tool_use_id,
                    "output": tr.content,
                })
    return result


def _to_responses_tools(tools: list[ToolSchema]) -> list[dict]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        for tool in tools
    ]


def _is_gemini_25(model: str) -> bool:
    """Gemini 2.5 series: uses thinkingBudget (integer tokens, 0 to disable)."""
    return "gemini-2.5" in model.lower()


def _is_gemini_3x(model: str) -> bool:
    """Gemini 3.x series (3.0 through 3.7): uses thinkingLevel (string level)."""
    m = model.lower()
    # Match gemini-3 followed by . or - (covers 3.0, 3.1, 3.5, 3-flash, etc.)
    return bool(re.search(r"gemini-3[.\-]", m))


def _is_openai_reasoning_model(model: str) -> bool:
    """Known OpenAI model families that expose reasoning controls."""
    from cca.llm.litellm_thinking import _is_openai_reasoning_route

    return _is_openai_reasoning_route(model)


def _google_model_supports_thinking(model: str) -> bool:
    """Return True for Gemini models that support extended thinking."""
    return _is_gemini_25(model) or _is_gemini_3x(model)


def _max_tokens_param_name(model: str) -> str:
    """Return the OpenAI token-limit parameter name for this model."""
    if _is_openai_reasoning_model(model):
        return "max_completion_tokens"
    return "max_tokens"


_UNSUPPORTED_SCHEMA_MARKERS = (
    "response_format",
    "json_schema",
    "structured output",
    "structured_output",
)

_UNSUPPORTED_TOOL_MARKERS = (
    "tool_choice",
    "tool choice",
    "tool calling",
    "tool use",
    "tools are not supported",
    "function calling",
)


def _is_unsupported_schema_error(exc: Exception) -> bool:
    """Whether `exc` means "this route can't do schemas", not "your call was bad".

    Providers signal an unsupported `response_format` with an ordinary 400, so
    the message is the only discriminator available. Errors that mention
    neither the parameter nor schemas — a rate limit, a bad key, a context
    overflow — must propagate untouched rather than being retried
    unconstrained, which would hide a real failure behind a worse answer.
    """
    # A TypeError never came from a provider: it means the call was built
    # wrong locally, and degrading to prompt_only would hide the bug behind
    # permanently worse output.
    if isinstance(exc, TypeError):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status is not None and str(status) not in ("400", "404", "422", "501"):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _UNSUPPORTED_SCHEMA_MARKERS)


def _is_unsupported_tool_error(exc: Exception) -> bool:
    """Whether a provider rejected forced tool use as an unsupported feature."""
    if isinstance(exc, TypeError):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status is not None and str(status) not in ("400", "404", "422", "501"):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _UNSUPPORTED_TOOL_MARKERS)


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return `schema` in the shape OpenAI's strict mode requires.

    Strict mode rejects a schema whose `required` omits any declared property,
    and one that does not set `additionalProperties: false` on every object.
    Optional fields are the common trip-up: they are made required here rather
    than dropped, so a model that has nothing to say for one can still answer
    (with an explicit null) instead of the whole request 400ing.
    """
    if not isinstance(schema, dict):
        return schema
    out = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            walk(value)

    walk(out)
    return out


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "",
        **kwargs,
    ):
        self.model = model
        self.provider = provider.lower()
        self.client = openai.AsyncOpenAI(
            api_key=api_key or "sk-placeholder",
            base_url=base_url,
            # Retries are owned by cca.llm.retry, not the SDK: its backoff caps
            # at ~8s, it cannot tell the caller that it is waiting, and its
            # sleeps would silently eat an external per-turn deadline.
            max_retries=0,
        )
        self.extra_kwargs = kwargs

    def supports_reasoning_summary(self) -> bool:
        if self.provider == "google":
            return _google_model_supports_thinking(self.model)
        if self.provider == "openai":
            return _is_openai_reasoning_model(self.model)
        return False

    def describe_request_settings(
        self,
        think_level: str,
        reasoning_summary: bool = False,
    ) -> dict[str, Any]:
        transport = (
            "openai-responses"
            if self.provider == "openai" and reasoning_summary
            else "openai-chat-completions"
        )
        return {
            "adapter": type(self).__name__,
            "transport": transport,
            "max_tokens_parameter": (
                "max_output_tokens" if transport == "openai-responses"
                else _max_tokens_param_name(self.model)
            ),
            "thinking_parameters": self._build_thinking_kwargs(
                think_level, reasoning_summary
            ),
            "reasoning_summary": reasoning_summary,
            "reasoning_replay": self._reasoning_replay_state() is not None,
        }

    def _reasoning_replay_state(self) -> dict[str, Any] | None:
        """Provider state marking returned reasoning to be echoed on later calls."""
        if self.provider != "litellm":
            return None
        from cca.llm.litellm_thinking import replays_reasoning_content

        if not replays_reasoning_content(self.model):
            return None
        return {"provider": "litellm", "replay_as_reasoning_content": True}

    def _build_thinking_kwargs(
        self,
        think_level: str,
        reasoning_summary: bool = False,
    ) -> dict:
        """Return extra kwargs to inject into the API call for thinking/reasoning.

        Gemini 2.5/3.x → top-level reasoning_effort ("low"|"medium"|"high")
                           via the OpenAI-compat endpoint; "off" sends nothing.
        OpenAI gpt-5   → top-level reasoning.effort ("none"|"low"|"medium"|"high")
        """
        if self.provider == "google" and _google_model_supports_thinking(self.model):
            if reasoning_summary:
                config: dict[str, Any] = {"include_thoughts": True}
                if _is_gemini_25(self.model):
                    config["thinking_budget"] = {
                        "low": 1_024,
                        "medium": 8_192,
                        "high": 24_576,
                    }.get(think_level, 8_192)
                else:
                    config["thinking_level"] = {
                        "low": "low",
                        "medium": "medium",
                        "high": "high",
                    }.get(think_level, "medium")
                # The OpenAI SDK's extra_body argument injects top-level keys.
                # Gemini expects a top-level `extra_body.google` extension.
                return {"extra_body": {"extra_body": {
                    "google": {"thinking_config": config}
                }}}
            if think_level == "off":
                return {}  # omit param; model uses its default reasoning level
            effort_map = {"low": "low", "medium": "medium", "high": "high"}
            effort = effort_map.get(think_level, "medium")
            return {"reasoning_effort": effort}

        if self.provider == "openai" and _is_openai_reasoning_model(self.model):
            effort_map = {"off": "none", "low": "low", "medium": "medium", "high": "high"}
            effort = effort_map.get(think_level, "medium")
            return {"reasoning": {"effort": effort}}

        if self.provider == "litellm":
            from cca.llm.litellm_thinking import get_litellm_thinking_profile

            profile = get_litellm_thinking_profile(self.model)
            if profile is None:
                return {}
            if profile.family == "Gemma":
                thinking_type = "disabled" if think_level == "off" else "enabled"
                return {"extra_body": {"thinking": {"type": thinking_type}}}
            if profile.family == "Claude":
                from cca.llm.anthropic_backend import _anthropic_thinking_kwargs

                anthropic_kwargs = _anthropic_thinking_kwargs(
                    self.model, think_level, reasoning_summary
                )
                if not anthropic_kwargs:
                    return {}
                # ``extra_body`` makes provider-native Anthropic fields
                # top-level fields in the OpenAI-compatible request. In
                # particular, adaptive ``thinking`` alone does not preserve
                # low/medium/high: Claude controls that via
                # ``output_config.effort``.
                extra_body: dict[str, Any] = dict(anthropic_kwargs)
                if self.model.lower().startswith("bedrock/"):
                    extra_body["allowed_openai_params"] = ["tools", "tool_choice"]
                return {"extra_body": extra_body}
            effort_map = {
                "off": "none",
                "low": "low",
                "medium": "medium",
                "high": "high",
            }
            if profile.effort_map:
                effort_map.update(profile.effort_map)
            effort = effort_map.get(think_level, "medium")
            allowed_params = ["reasoning_effort"]
            if self.model.lower().startswith("bedrock/"):
                allowed_params.extend(["tools", "tool_choice"])
            return {
                "reasoning_effort": effort,
                "extra_body": {
                    "allowed_openai_params": allowed_params,
                },
            }

        return {}

    def _responses_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int,
        think_level: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        effort = {
            "off": "none",
            "low": "low",
            "medium": "medium",
            "high": "high",
        }.get(think_level, "medium")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": _to_responses_input(messages),
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": effort, "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "stream": stream,
        }
        response_tools = _to_responses_tools(tools)
        if response_tools:
            kwargs["tools"] = response_tools
            kwargs["tool_choice"] = "auto"
        return kwargs

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
        """Constrain the reply to `output_schema`, degrading if unsupported.

        Tries native `response_format: json_schema` first — that is real
        constrained decoding, so the reply cannot be malformed. Support varies
        by route (OpenAI and Gemini have it; open-weight routes behind
        Fireworks/Groq vary by model), and an unsupported route answers with a
        400 rather than a silent downgrade. When native schema output is not
        available, a forced synthetic function call is tried before the final
        prompt-only compatibility fallback. The caller still validates every
        path — `structured_output.py` never assumes the schema was enforced.

        `mode_used` on the response records which path actually ran, so a
        model comparison can separate "the simulator was bad" from "this route
        could not be constrained".
        """
        oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        base: dict[str, Any] = dict(model=self.model, messages=oai_messages)
        base[_max_tokens_param_name(self.model)] = max_tokens
        base.update(self._build_thinking_kwargs(think_level, False))

        constrained = dict(base)
        constrained["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "cca_response",
                "strict": True,
                "schema": _strict_json_schema(output_schema),
            },
        }
        def with_litellm_allowlist(
            request: dict[str, Any], *parameters: str
        ) -> dict[str, Any]:
            if self.provider != "litellm":
                return request
            # litellm drops params a route does not advertise unless they are
            # explicitly allowed through, which would silently turn the
            # constrained call back into an unconstrained one.
            #
            # It belongs in `extra_body`, where `_build_thinking_kwargs` also
            # puts it: the openai SDK rejects any top-level kwarg it does not
            # define, so setting it directly raises TypeError before the
            # request is ever sent. `constrained` is a shallow copy of `base`,
            # so the nested dict is rebuilt rather than mutated in place —
            # otherwise the unconstrained fallback would inherit the entry.
            request = dict(request)
            extra_body = dict(request.get("extra_body") or {})
            allowed = list(extra_body.get("allowed_openai_params") or [])
            for parameter in parameters:
                if parameter not in allowed:
                    allowed.append(parameter)
            extra_body["allowed_openai_params"] = allowed
            request["extra_body"] = extra_body
            return request

        constrained = with_litellm_allowlist(constrained, "response_format")

        mode = "json_schema"
        try:
            response = await self.client.chat.completions.create(**constrained)
        except Exception as exc:  # noqa: BLE001 - provider-specific error types
            if not _is_unsupported_schema_error(exc):
                raise
            forced_tool = dict(base)
            forced_tool["tools"] = [{
                "type": "function",
                "function": {
                    "name": "cca_structured_response",
                    "description": (
                        "Return the response as structured fields. "
                        "Call this exactly once."
                    ),
                    "parameters": _strict_json_schema(output_schema),
                },
            }]
            forced_tool["tool_choice"] = {
                "type": "function",
                "function": {"name": "cca_structured_response"},
            }
            forced_tool = with_litellm_allowlist(
                forced_tool, "tools", "tool_choice"
            )
            mode = "forced_tool"
            try:
                response = await self.client.chat.completions.create(**forced_tool)
            except Exception as tool_exc:  # noqa: BLE001 - provider-specific errors
                if not _is_unsupported_tool_error(tool_exc):
                    raise
                mode = "prompt_only"
                response = await self.client.chat.completions.create(**base)

        choice = response.choices[0]
        text = choice.message.content or ""
        if mode == "forced_tool":
            tool_calls = getattr(choice.message, "tool_calls", None) or []
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                if getattr(function, "name", None) == "cca_structured_response":
                    text = getattr(function, "arguments", "") or ""
                    break
            else:
                # A route that ignored a forced choice did not constrain this
                # response. Preserve its content for validation but report the
                # honest mode so comparative callers can reject it.
                mode = "prompt_only"
        cache_read = 0
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cache_read = getattr(details, "cached_tokens", 0) or 0
        return CompletionResponse(
            content=[TextContent(text=text)],
            stop_reason=choice.finish_reason or "end_turn",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "cache_read_tokens": cache_read,
            },
            mode_used=mode,
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
        """Retry-wrapped `_complete_once` — see `cca.llm.retry`.

        The wrapper sits outside `_complete_structured` so its 400-driven
        unsupported-schema fallback still sees the provider's own exception,
        never a retry wrapper's.
        """
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
        if self.provider == "openai" and reasoning_summary:
            response = await self.client.responses.create(**self._responses_kwargs(
                messages, tools, system, max_tokens, think_level, stream=False
            ))
            content: list[ContentBlock] = []
            for item in response.output:
                if item.type == "reasoning":
                    summary = "\n".join(part.text for part in item.summary if part.text)
                    state = {
                        "provider": "openai",
                        "reasoning_items": [item.model_dump(exclude_none=True)],
                    }
                    content.append(ThinkingContent(
                        thinking=summary,
                        reasoning_kind="summary",
                        provider_state=state,
                    ))
                elif item.type == "message":
                    for part in item.content:
                        if part.type == "output_text" and part.text:
                            content.append(TextContent(text=part.text))
                elif item.type == "function_call":
                    try:
                        tool_input = json.loads(item.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}
                    content.append(ToolUseContent(
                        id=item.call_id,
                        name=item.name,
                        input=tool_input,
                        provider_state={
                            "provider": "openai",
                            "call_id": item.call_id,
                            **({"item_id": item.id} if item.id else {}),
                        },
                    ))
            usage = response.usage
            details = getattr(usage, "input_tokens_details", None) if usage else None
            return CompletionResponse(
                content=content,
                stop_reason="end_turn",
                usage={
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "cache_read_tokens": getattr(details, "cached_tokens", 0) or 0,
                    "cache_write_tokens": getattr(details, "cache_write_tokens", 0) or 0,
                },
            )

        oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        oai_tools = [t.to_openai() for t in tools] if tools else []

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=oai_messages,
        )
        kwargs[_max_tokens_param_name(self.model)] = max_tokens
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"
        kwargs.update(self._build_thinking_kwargs(think_level, reasoning_summary))

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        content: list[ContentBlock] = []
        msg_extra = getattr(msg, "model_extra", {}) or {}
        reasoning_text = (
            getattr(msg, "reasoning_content", None)
            or msg_extra.get("reasoning_content")
        )
        if reasoning_text:
            content.append(ThinkingContent(
                thinking=reasoning_text,
                reasoning_kind="summary" if reasoning_summary else "unknown",
                provider_state=self._reasoning_replay_state(),
            ))
        if msg.content:
            content.append(TextContent(text=msg.content))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                func_extra = getattr(tc.function, "model_extra", {}) or {}
                tc_extra = getattr(tc, "model_extra", {}) or {}
                extra_content = _tool_call_extra_content(tc, tc.function)
                thought_sig = _thought_signature(
                    extra_content,
                    func_extra.get("thought_signature"),
                    tc_extra.get("thought_signature"),
                )
                content.append(ToolUseContent(
                    id=tc.id,
                    name=tc.function.name,
                    input=tool_input,
                    thought_signature=thought_sig,
                    provider_state=_tool_provider_state(
                        self.provider, extra_content
                    ),
                ))

        cache_read = 0
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cache_read = getattr(details, "cached_tokens", 0) or 0

        return CompletionResponse(
            content=content,
            stop_reason=choice.finish_reason or "end_turn",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "cache_read_tokens": cache_read,
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
        """Retry-wrapped `_stream_once` — see `cca.llm.retry`.

        Only a failure before the first event is retried; after that the
        caller already holds partial output and a restart would duplicate it.
        """
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
        if self.provider == "openai" and reasoning_summary:
            async for event in self._stream_openai_responses(
                messages, tools, system, max_tokens, think_level
            ):
                yield event
            return

        oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        oai_tools = [t.to_openai() for t in tools] if tools else []

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=oai_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        kwargs[_max_tokens_param_name(self.model)] = max_tokens
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"
        kwargs.update(self._build_thinking_kwargs(think_level, reasoning_summary))

        # Accumulate tool call deltas.
        # Key is the tool call id when available (handles Gemini which sends all
        # parallel calls in one chunk without an index field), or the integer
        # index for OpenAI-style streaming where one call arrives across many chunks.
        tool_call_accum: dict[str | int, dict] = {}
        # Maps integer index → accumulator key, for OpenAI continuation chunks
        # that omit the id but repeat the index.
        _idx_to_key: dict[int, str | int] = {}
        _input_tokens = 0
        _output_tokens = 0
        _cache_read_tokens = 0

        import os
        import sys
        _debug = os.environ.get("CCA_DEBUG_SSE")
        _debug_usage = os.environ.get("CCA_DEBUG_USAGE") or _debug

        if _debug:
            # Show outgoing assistant messages so we can verify thought_signature echoing
            for m in oai_messages:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        func = tc.get("function", {})
                        sig = (tc.get("extra_content") or {}).get("google", {}).get("thought_signature")
                        sig_status = f"✓ {sig[:20]}..." if sig else "✗ None"
                        print(f"[SSE] outgoing assistant tool_call: {func.get('name')}  thought_signature={sig_status}", file=sys.stderr)

        def _extract_thought_sig(chunk: dict, choice: dict, delta: dict, tc_delta: dict) -> str | None:
            """Pull thought_signature from any location Gemini might put it."""
            func = tc_delta.get("function", {})
            # Real Gemini format: tc_delta["extra_content"]["google"]["thought_signature"]
            extra_content_sig = (
                (tc_delta.get("extra_content") or {})
                .get("google", {})
                .get("thought_signature")
            )
            return (
                extra_content_sig                       # real Gemini: extra_content.google.thought_signature
                or func.get("thought_signature")        # standard: inside function dict
                or tc_delta.get("thought_signature")    # fallback: at tool_call level
                or delta.get("thought_signature")       # fallback: at delta level
                or choice.get("thought_signature")      # fallback: at choice level
                or chunk.get("thought_signature")       # fallback: top-level chunk
            )

        # Use with_streaming_response to get raw SSE lines so provider-specific
        # extra fields (e.g. Gemini's thought_signature) are never dropped by pydantic.
        # Tool calls are emitted only at [DONE], not at finish_reason, because Gemini
        # can send thought_signature in chunks that arrive after finish_reason.
        async with self.client.chat.completions.with_streaming_response.create(**kwargs) as response:
            async for line in response.iter_lines():
                if _debug:
                    print(f"[SSE] {line}", file=sys.stderr)
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    # Emit in insertion order (dict preserves order in Python 3.7+)
                    for tc in tool_call_accum.values():
                        try:
                            tool_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        if _debug:
                            sig_status = f"✓ {tc['thought_signature'][:20]}..." if tc.get("thought_signature") else "✗ None"
                            print(f"[SSE] emit tool_use_complete: {tc['name']}  thought_signature={sig_status}", file=sys.stderr)
                        extra_content = tc.get("extra_content")
                        if tc.get("thought_signature"):
                            if not isinstance(extra_content, dict):
                                extra_content = {}
                            google = extra_content.setdefault("google", {})
                            if isinstance(google, dict):
                                google.setdefault(
                                    "thought_signature", tc["thought_signature"]
                                )
                        yield StreamEvent(
                            type="tool_use_complete",
                            data=ToolUseContent(
                                id=tc["id"],
                                name=tc["name"],
                                input=tool_input,
                                thought_signature=tc.get("thought_signature"),
                                provider_state=_tool_provider_state(
                                    self.provider,
                                    extra_content if isinstance(extra_content, dict) else None,
                                ),
                            ),
                        )
                    yield StreamEvent(type="message_end", data={
                        "input_tokens": _input_tokens,
                        "output_tokens": _output_tokens,
                        "cache_read_tokens": _cache_read_tokens,
                    })
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # Capture usage if the provider includes it in a chunk
                chunk_usage = chunk.get("usage") or {}
                if chunk_usage:
                    if _debug_usage:
                        print(f"[USAGE] raw chunk usage: {json.dumps(chunk_usage)}", file=sys.stderr)
                    if chunk_usage.get("prompt_tokens"):
                        _input_tokens = chunk_usage["prompt_tokens"]
                    if chunk_usage.get("completion_tokens"):
                        _output_tokens = chunk_usage["completion_tokens"]
                    details = chunk_usage.get("prompt_tokens_details") or {}
                    if details.get("cached_tokens"):
                        _cache_read_tokens = details["cached_tokens"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                google_extra = (
                    (delta.get("extra_content") or {}).get("google") or {}
                )
                google_thought = bool(google_extra.get("thought"))
                google_thought_signature = google_extra.get("thought_signature")

                if google_thought or (
                    reasoning_summary and google_thought_signature
                ):
                    provider_state = {"provider": "google"}
                    if google_thought_signature:
                        provider_state["thought_signature"] = google_thought_signature
                    yield StreamEvent(type="reasoning_metadata", data={
                        "kind": "summary" if reasoning_summary else "unknown",
                        "provider_state": provider_state,
                    })

                if delta.get("reasoning_content"):
                    provider_state = self._reasoning_replay_state()
                    yield StreamEvent(type="reasoning_metadata", data={
                        "kind": "summary" if reasoning_summary else "unknown",
                        **({"provider_state": provider_state} if provider_state else {}),
                    })
                    yield StreamEvent(type="thinking_delta", data=delta["reasoning_content"])

                if delta.get("content") and google_thought:
                    yield StreamEvent(type="thinking_delta", data=delta["content"])
                elif delta.get("content"):
                    yield StreamEvent(type="text_delta", data=delta["content"])

                tc_deltas = delta.get("tool_calls", [])
                for loop_i, tc_delta in enumerate(tc_deltas):
                    tc_id = tc_delta.get("id")
                    explicit_idx = tc_delta.get("index")
                    func = tc_delta.get("function", {})
                    sig = _extract_thought_sig(chunk, choice, delta, tc_delta)

                    # Resolve the accumulator key.
                    # Gemini sends all parallel calls in one chunk, each with a
                    # unique id but no index — use the id as the key.
                    # OpenAI streams one call across many chunks with a stable
                    # integer index (id only on the first chunk) — map index→key.
                    if tc_id and tc_id in tool_call_accum:
                        key = tc_id                          # continuation by id
                    elif tc_id:
                        key = tc_id                          # new call, id is key
                        if explicit_idx is not None:
                            _idx_to_key[explicit_idx] = key  # register for future index lookup
                    elif explicit_idx is not None and explicit_idx in _idx_to_key:
                        key = _idx_to_key[explicit_idx]      # OpenAI continuation chunk
                    elif explicit_idx is not None:
                        key = explicit_idx                   # new call, index is key
                        _idx_to_key[explicit_idx] = key
                    elif len(tool_call_accum) == 1:
                        key = next(iter(tool_call_accum))    # single call streaming: no id/index, append to only entry
                    else:
                        key = f"_pos_{loop_i}"               # last-resort positional key

                    if key not in tool_call_accum:
                        tool_call_accum[key] = {
                            "id": tc_id or str(uuid.uuid4()),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                            "thought_signature": sig,
                            "extra_content": _tool_call_extra_content(tc_delta),
                        }
                        yield StreamEvent(type="tool_use_start", data={
                            "id": tool_call_accum[key]["id"],
                            "name": tool_call_accum[key]["name"],
                        })
                    else:
                        if func.get("name"):
                            tool_call_accum[key]["name"] += func["name"]
                        if func.get("arguments"):
                            tool_call_accum[key]["arguments"] += func["arguments"]
                        if sig and not tool_call_accum[key]["thought_signature"]:
                            tool_call_accum[key]["thought_signature"] = sig
                        incoming_extra = _tool_call_extra_content(tc_delta)
                        if incoming_extra:
                            stored_extra = tool_call_accum[key].setdefault(
                                "extra_content", {}
                            )
                            _merge_provider_state(stored_extra, incoming_extra)

                # Handle chunks where Gemini sends thought_signature outside tool_calls
                # (e.g. a standalone delivery chunk with no tool_calls entries).
                if not tc_deltas and tool_call_accum:
                    loose_sig = (
                        delta.get("thought_signature")
                        or choice.get("thought_signature")
                        or chunk.get("thought_signature")
                    )
                    if loose_sig:
                        if _debug:
                            print(f"[SSE] loose thought_signature chunk: {loose_sig[:30]}...", file=sys.stderr)
                        for acc in tool_call_accum.values():
                            if not acc["thought_signature"]:
                                acc["thought_signature"] = loose_sig

    async def _stream_openai_responses(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str,
        max_tokens: int,
        think_level: str,
    ) -> AsyncIterator[StreamEvent]:
        """Stream OpenAI reasoning summaries through the Responses API."""
        stream = await self.client.responses.create(**self._responses_kwargs(
            messages, tools, system, max_tokens, think_level, stream=True
        ))
        reasoning_started: set[str] = set()
        async for event in stream:
            etype = event.type
            if etype == "response.reasoning_summary_text.delta":
                if event.item_id not in reasoning_started:
                    reasoning_started.add(event.item_id)
                    yield StreamEvent(type="reasoning_metadata", data={
                        "kind": "summary",
                        "provider_state": {"provider": "openai"},
                    })
                yield StreamEvent(type="thinking_delta", data=event.delta)
            elif etype == "response.output_text.delta":
                yield StreamEvent(type="text_delta", data=event.delta)
            elif etype == "response.output_item.added":
                item = event.item
                if item.type == "function_call":
                    yield StreamEvent(type="tool_use_start", data={
                        "id": item.call_id,
                        "name": item.name,
                    })
            elif etype == "response.output_item.done":
                item = event.item
                if item.type == "reasoning":
                    yield StreamEvent(type="reasoning_metadata", data={
                        "kind": "summary",
                        "provider_state": {
                            "provider": "openai",
                            "reasoning_items": [item.model_dump(exclude_none=True)],
                        },
                    })
                elif item.type == "function_call":
                    try:
                        tool_input = json.loads(item.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}
                    yield StreamEvent(
                        type="tool_use_complete",
                        data=ToolUseContent(
                            id=item.call_id,
                            name=item.name,
                            input=tool_input,
                            provider_state={
                                "provider": "openai",
                                "call_id": item.call_id,
                                **({"item_id": item.id} if item.id else {}),
                            },
                        ),
                    )
            elif etype == "response.completed":
                usage = event.response.usage
                details = getattr(usage, "input_tokens_details", None) if usage else None
                yield StreamEvent(type="message_end", data={
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "cache_read_tokens": getattr(details, "cached_tokens", 0) or 0,
                    "cache_write_tokens": getattr(details, "cache_write_tokens", 0) or 0,
                    "stop_reason": "end_turn",
                })
            elif etype == "response.incomplete":
                usage = event.response.usage
                details = getattr(usage, "input_tokens_details", None) if usage else None
                yield StreamEvent(type="message_end", data={
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "cache_read_tokens": getattr(details, "cached_tokens", 0) or 0,
                    "cache_write_tokens": getattr(details, "cache_write_tokens", 0) or 0,
                    "stop_reason": "incomplete",
                })
            elif etype == "response.failed":
                error = getattr(event.response, "error", None)
                message = getattr(error, "message", None) or "OpenAI Responses API failed"
                raise RuntimeError(message)
