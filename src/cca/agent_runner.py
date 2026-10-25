"""Depth-one subagent lifecycle with exact, separately stored trajectories."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import json
from typing import Any

from cca.agents import AgentSpec
from cca.prompts import environment_block
from cca.hooks import HookRunner
from cca.llm.base import (
    Message,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from cca.observability.semconv import model_attributes, response_model_attributes
from cca.trace_store import TraceIdentity, TraceStore


_ALWAYS_CHILD_DENIED = frozenset({
    "Agent",
    "AskUserQuestion",
    "JobOutput",
    "JobCancel",
    "EnterWorktree",
    "ExitWorktree",
    "EnterPlanMode",
    "ExitPlanMode",
})

_MUTATING_TOOLS = frozenset({
    "Write", "Edit", "NotebookEdit", "Bash", "CronCreate", "CronDelete",
    "Monitor", "JobCancel",
})

_TURN_CAP_HANDOFF_PROMPT = """\
You have reached the final reserved turn for this delegated run. Do not call tools.
Return a concise, truthful handoff to the parent with these sections:

1. Work completed
2. Files changed
3. Verification performed and its results
4. Remaining work or blockers
5. Risks or uncertainty

Do not claim the delegated task is complete unless the evidence supports that claim.
This response is a turn-cap handoff, not permission to continue working.
"""

_CONTEXT_CAP_HANDOFF_PROMPT = """\
You are approaching the model's context limit. Do not call tools. Return a concise,
truthful handoff to the parent with these sections:

1. Work completed
2. Files changed
3. Verification performed and its results
4. Remaining work or blockers
5. Risks or uncertainty

Do not claim the delegated task is complete unless the evidence supports that claim.
This response is a context-cap handoff, not permission to continue working.
"""


@dataclass
class AgentResult:
    status: str
    summary: str
    findings: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    agent_id: str = ""
    agent_name: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.error is None:
            payload.pop("error")
        return payload

    def to_tool_result(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)


@dataclass
class AgentRunContext:
    identity: TraceIdentity
    root_agent_id: str
    parent_session_id: str
    session_id: str
    cwd: str
    spec: AgentSpec
    description: str
    prompt: str
    trigger: str
    backend: Any
    registry: Any
    config: Any
    feature_set: Any
    trace_store: TraceStore
    catalog_hash: str
    thinking_level: str
    settings: dict[str, Any]
    permissions: Any = None
    parent_allowed_tools: set[str] | None = None
    parent_disallowed_tools: set[str] = field(default_factory=set)
    project_instructions: str = ""
    cancellation: asyncio.Event | None = None
    observer: Any = None
    parent_span: Any = None
    background: bool = False


class AgentRunner:
    """Run one child loop while preserving capability and trace invariants."""

    def __init__(self, context: AgentRunContext) -> None:
        self.context = context
        self._environment = ""
        self._call_idx = context.trace_store.next_call_idx(
            context.identity, child=True
        )
        hooks_config = (
            context.config.hooks
            if context.feature_set.feature("hooks") and context.config is not None
            else {}
        )
        self._hooks = HookRunner(
            hooks_config,
            session_id=context.session_id,
            cwd=context.cwd,
        )

    def _effective_tool_names(self) -> set[str]:
        available = set(getattr(self.context.registry, "_tools", {}))
        if self.context.parent_allowed_tools is not None:
            available &= self.context.parent_allowed_tools
        available -= self.context.parent_disallowed_tools
        if self.context.spec.tools is not None:
            available &= set(self.context.spec.tools)
        available -= set(self.context.spec.disallowed_tools)
        available -= _ALWAYS_CHILD_DENIED
        return available

    def _system_prompt(self) -> str:
        parts = [self.context.spec.prompt, self._environment, self.context.project_instructions]
        return "\n\n".join(part for part in parts if part)

    def _context_budget_state(
        self,
        messages: list[Message],
        system: str,
        tool_schemas: list[Any],
    ) -> dict[str, int] | None:
        """Return budget evidence when a final handoff should be reserved now."""
        from cca.llm.catalog import resolve_model

        selector = self.context.settings.get("model")
        model_info = resolve_model(selector) if isinstance(selector, str) else None
        context_window = model_info.context_window if model_info is not None else None
        if not isinstance(context_window, int) or context_window <= 0:
            return None
        serialized = json.dumps(
            {
                "system": system,
                "messages": [message.to_dict() for message in messages],
                "tools": [asdict(schema) for schema in tool_schemas],
            },
            ensure_ascii=False,
        )
        estimated_tokens = max(1, len(serialized) // 4)
        from cca.llm.model_caps import get_effective_max_output_tokens

        configured_output = getattr(self.context.config, "max_tokens", 8192)
        max_output = get_effective_max_output_tokens(
            selector,
            configured=(
                configured_output if isinstance(configured_output, int) else 8192
            ),
        )
        # Keep enough room for the handoff response and normal provider framing.
        reserve = max(12_288, max_output + 4_096)
        threshold = min(int(context_window * 0.85), max(1, context_window - reserve))
        if estimated_tokens < threshold:
            return None
        return {
            "estimated_tokens": estimated_tokens,
            "context_window": context_window,
            "handoff_threshold": threshold,
        }

    async def _execute_one(self, call: ToolUseContent, effective: set[str]) -> ToolResultContent:
        observer = self.context.observer
        if observer is None:
            return await self._execute_one_impl(call, effective)
        with observer.span(
            f"cca.tool.{call.name}",
            kind="tool",
            attributes={
                "cca.tool.name": call.name,
                "gen_ai.tool.call.id": call.id,
                "gen_ai.tool.name": call.name,
                "cca.agent_id": self.context.identity.agent_id,
                "cca.session_id": self.context.session_id,
                "cca.context_generation": (
                    self.context.identity.context_generation
                    if self.context.identity.context_generation is not None
                    else self.context.identity.generation
                ),
                "cca.parent_context_generation": (
                    self.context.identity.parent_context_generation
                ),
            },
            content={"gen_ai.tool.call.arguments": call.input},
        ) as span:
            result = await self._execute_one_impl(call, effective)
            if result.content.startswith("Tool blocked"):
                status = "blocked"
            elif result.content.startswith("Tool execution denied"):
                status = "denied"
            elif result.is_error:
                status = "error"
            else:
                status = "ok"
            span.finish(
                status=status,
                content={"gen_ai.tool.call.result": result.content},
            )
            return result

    async def _execute_one_impl(self, call: ToolUseContent, effective: set[str]) -> ToolResultContent:
        if call.name not in effective:
            return ToolResultContent(
                tool_use_id=call.id,
                content=f"Error: tool '{call.name}' is not allowed for this subagent.",
                is_error=True,
                name=call.name,
            )

        hook_pre = await self._hooks.pre_tool_use(call.name, call.input, call.id)
        if not hook_pre.allowed:
            return ToolResultContent(
                tool_use_id=call.id,
                content=f"Tool blocked by hook: {hook_pre.stderr or 'Blocked by hook'}",
                is_error=True,
                name=call.name,
            )
        if self.context.permissions is not None:
            allowed = await self.context.permissions.check(call.name, call.input)
            if not allowed:
                return ToolResultContent(
                    tool_use_id=call.id,
                    content="Tool execution denied by user.",
                    is_error=True,
                    name=call.name,
                )
        try:
            result = str(await self.context.registry.execute(call.name, call.input))
        except Exception as exc:
            result = f"Error: tool '{call.name}' raised: {exc}"

        hook_post = await self._hooks.post_tool_use(
            call.name, call.input, call.id, result
        )
        if hook_post.additional_context:
            result += f"\n\n<system-reminder>{hook_post.additional_context}</system-reminder>"
        is_error = result.startswith("Error:") or result.startswith("Tool blocked")
        return ToolResultContent(
            tool_use_id=call.id,
            content=result,
            is_error=is_error,
            name=call.name,
        )

    async def _execute_calls(
        self,
        calls: list[ToolUseContent],
        effective: set[str],
    ) -> list[ToolResultContent]:
        # Mutation is deliberately serial in a shared workspace. Read-only calls
        # can run concurrently, but results always rejoin in model call order.
        if any(call.name in _MUTATING_TOOLS for call in calls):
            return [await self._execute_one(call, effective) for call in calls]
        return list(await asyncio.gather(
            *(self._execute_one(call, effective) for call in calls)
        ))

    def _trace_call(
        self,
        *,
        system: str,
        messages_snapshot: list[dict[str, Any]],
        tool_schemas: list[Any],
        response: Any,
        call_type: str = "agent",
    ) -> None:
        from cca.trace_store import backend_provenance

        thinking = "".join(
            block.thinking for block in response.content
            if isinstance(block, ThinkingContent)
        )
        tool_calls = []
        for call in response.tool_calls:
            payload: dict[str, Any] = {
                "id": call.id,
                "name": call.name,
                "input": call.input,
            }
            if call.thought_signature is not None:
                payload["thought_signature"] = call.thought_signature
            if call.provider_state is not None:
                payload["provider_state"] = call.provider_state
            tool_calls.append(payload)
        usage = response.usage or {}
        record = {
            "ts": __import__("time").time_ns() // 1_000_000_000,
            "type": call_type,
            "session_id": self.context.session_id,
            "run_id": self.context.identity.agent_id,
            "continuation_reason": "initial",
            "model_provenance": backend_provenance(
                self.context.backend,
                self.context.settings.get("model_requested")
                or self.context.settings.get("model"),
                self.context.settings.get("model"),
            ),
            "features": self.context.feature_set.to_payload(),
            "settings": self.context.settings,
            "agent_definition": self.context.spec.to_payload(),
            "catalog_hash": self.context.catalog_hash,
            "trigger": self.context.trigger,
            "system": system,
            "messages": messages_snapshot,
            "tools": [asdict(schema) for schema in tool_schemas],
            "params": {
                "max_tokens": getattr(self.context.config, "max_tokens", 8192),
                "think_level": self.context.thinking_level,
            },
            "response": {
                "text": response.text,
                "tool_calls": tool_calls,
                "thinking": thinking,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.get("input_tokens", usage.get("input", 0)),
                "output_tokens": usage.get("output_tokens", usage.get("output", 0)),
                "cache_read_tokens": usage.get("cache_read_tokens", 0),
                "cache_write_tokens": usage.get("cache_write_tokens", 0),
            },
        }
        self.context.trace_store.append_call(
            self.context.identity,
            self._call_idx,
            record,
            child=True,
        )
        self._call_idx += 1

    def _save_session(self, messages: list[Message]) -> None:
        from cca.sessions import save_session

        save_session(
            session_id=self.context.session_id,
            cwd=self.context.cwd,
            messages=messages,
            display_override=f"[Subagent] {self.context.description}",
            parent_id=self.context.parent_session_id,
            system_prompt=self._system_prompt(),
            features=self.context.feature_set.to_payload(),
            settings=self.context.settings,
            state={
                "trace_id": self.context.identity.trace_id,
                "agent_id": self.context.identity.agent_id,
                "parent_agent_id": self.context.identity.parent_agent_id,
                "spawn_call_id": self.context.identity.spawn_call_id,
                "generation": self.context.identity.generation,
                "context_generation": (
                    self.context.identity.context_generation
                    if self.context.identity.context_generation is not None
                    else self.context.identity.generation
                ),
                "generation_reason": self.context.identity.generation_reason,
                "parent_context_generation": (
                    self.context.identity.parent_context_generation
                ),
                "next_call_idx": self._call_idx,
            },
        )

    async def _complete_observed(
        self,
        *,
        messages: list[Message],
        tools: list[Any],
        system: str,
        max_tokens: int,
        think_level: str,
        call_type: str,
    ) -> Any:
        observer = self.context.observer
        if observer is None:
            return await self.context.backend.complete(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
                think_level=think_level,
            )
        with observer.span(
            "cca.llm.call",
            kind="llm",
            attributes={
                "cca.llm.call_type": call_type,
                "cca.session_id": self.context.session_id,
                "cca.agent_id": self.context.identity.agent_id,
                "cca.context_generation": (
                    self.context.identity.context_generation
                    if self.context.identity.context_generation is not None
                    else self.context.identity.generation
                ),
                **model_attributes(self.context.settings.get("model")),
                "gen_ai.request.max_tokens": max_tokens,
                "cca.thinking_level": think_level,
            },
            content={
                "input.value": {
                    "messages": [message.to_dict() for message in messages]
                }
            },
        ) as span:
            response = await self.context.backend.complete(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
                think_level=think_level,
            )
            usage = response.usage or {}
            span.finish(
                status="ok",
                attributes={
                    "gen_ai.response.finish_reasons": [response.stop_reason],
                    **response_model_attributes(self.context.settings.get("model")),
                },
                metrics={
                    "gen_ai.usage.input_tokens": usage.get(
                        "input_tokens", usage.get("input", 0)
                    ),
                    "gen_ai.usage.output_tokens": usage.get(
                        "output_tokens", usage.get("output", 0)
                    ),
                    "gen_ai.usage.cache_read.input_tokens": usage.get(
                        "cache_read_tokens", 0
                    ),
                    "gen_ai.usage.cache_creation.input_tokens": usage.get(
                        "cache_write_tokens", 0
                    ),
                },
                content={"output.value": response.text},
            )
            return response

    async def run(self) -> AgentResult:
        observer = self.context.observer
        if observer is None:
            return await self._run_impl()
        with observer.span(
            "cca.agent.run",
            kind="agent",
            parent=self.context.parent_span,
            attributes={
                "cca.agent_id": self.context.identity.agent_id,
                "cca.parent_agent_id": self.context.identity.parent_agent_id,
                "cca.spawn_call_id": self.context.identity.spawn_call_id,
                "cca.agent.name": self.context.spec.name,
                "cca.agent.background": self.context.background,
                "cca.session_id": self.context.session_id,
                "cca.context_generation": (
                    self.context.identity.context_generation
                    if self.context.identity.context_generation is not None
                    else self.context.identity.generation
                ),
                "cca.parent_context_generation": (
                    self.context.identity.parent_context_generation
                ),
            },
            content={"input.value": self.context.prompt},
        ) as span:
            result = await self._run_impl()
            status = {
                "completed": "ok",
                "cancelled": "cancelled",
                "failed": "error",
            }.get(result.status, result.status)
            span.finish(
                status=status,
                attributes={"cca.agent.result_status": result.status},
                content={"output.value": result.summary},
            )
            return result

    async def _run_impl(self) -> AgentResult:
        context = self.context
        # Rendered once per run, off the event loop: it shells out to git, and
        # blocking here would stall sibling children running in parallel.
        self._environment = await asyncio.to_thread(environment_block, context.cwd)
        effective = self._effective_tool_names()
        tool_schemas = context.registry.get_schemas(include=effective)
        system = self._system_prompt()
        from cca.llm.model_caps import get_effective_max_output_tokens

        selector = context.settings.get("model")
        configured_output = getattr(context.config, "max_tokens", 8192)
        max_output = get_effective_max_output_tokens(
            selector if isinstance(selector, str) else "",
            configured=(
                configured_output if isinstance(configured_output, int) else 8192
            ),
        )
        messages = [Message(role="user", content=[TextContent(text=context.prompt)])]
        files_changed: list[str] = []
        verification: list[dict[str, Any]] = []

        context.trace_store.update_child(context.identity, status="running")
        context.trace_store.append_event(
            "agent.run.started",
            context.identity,
            agent_name=context.spec.name,
            effective_tools=sorted(effective),
            definition_hash=context.spec.definition_hash,
        )
        await self._hooks.session_start()

        try:
            # Reserve the last model call for a tool-free parent handoff. This
            # avoids spending the entire budget and then returning only a generic
            # failure with no synthesis of the work already performed.
            working_turns = max(0, context.spec.max_turns - 1)
            cap_status = "turn_cap"
            cap_prompt = _TURN_CAP_HANDOFF_PROMPT
            cap_evidence: dict[str, int] | None = None
            for _turn in range(working_turns):
                if context.cancellation is not None and context.cancellation.is_set():
                    result = AgentResult(
                        status="cancelled",
                        summary="Subagent was cancelled.",
                        files_changed=files_changed,
                        verification=verification,
                        agent_id=context.identity.agent_id,
                        agent_name=context.spec.name,
                    )
                    return self._finish(result, messages)

                cap_evidence = self._context_budget_state(
                    messages, system, tool_schemas
                )
                if cap_evidence is not None:
                    cap_status = "context_cap"
                    cap_prompt = (
                        _CONTEXT_CAP_HANDOFF_PROMPT
                        + "\nContext budget evidence: "
                        + json.dumps(cap_evidence, sort_keys=True)
                    )
                    context.trace_store.append_event(
                        "agent.context_cap.reached",
                        context.identity,
                        agent_name=context.spec.name,
                        **cap_evidence,
                    )
                    break

                snapshot = [message.to_dict() for message in messages]
                response = await self._complete_observed(
                    messages=messages,
                    tools=tool_schemas,
                    system=system,
                    max_tokens=max_output,
                    think_level=context.thinking_level,
                    call_type="agent",
                )
                self._trace_call(
                    system=system,
                    messages_snapshot=snapshot,
                    tool_schemas=tool_schemas,
                    response=response,
                )
                messages.append(response.to_message())

                if not response.tool_calls:
                    summary = response.text or "(no response)"
                    result = AgentResult(
                        status="completed",
                        summary=summary,
                        files_changed=sorted(set(files_changed)),
                        verification=verification,
                        agent_id=context.identity.agent_id,
                        agent_name=context.spec.name,
                    )
                    return self._finish(result, messages)

                tool_results = await self._execute_calls(response.tool_calls, effective)
                for call, tool_result in zip(response.tool_calls, tool_results):
                    if call.name in {"Write", "Edit", "NotebookEdit"}:
                        path = call.input.get("file_path")
                        if isinstance(path, str) and not tool_result.is_error:
                            files_changed.append(path)
                    if call.name == "Bash":
                        verification.append({
                            "command": call.input.get("command", ""),
                            "result": tool_result.content,
                            "passed": not tool_result.is_error and "[Exit code:" not in tool_result.content,
                        })
                messages.append(Message(role="user", content=tool_results))
                self._save_session(messages)

            if context.cancellation is not None and context.cancellation.is_set():
                result = AgentResult(
                    status="cancelled",
                    summary="Subagent was cancelled.",
                    files_changed=sorted(set(files_changed)),
                    verification=verification,
                    agent_id=context.identity.agent_id,
                    agent_name=context.spec.name,
                )
                return self._finish(result, messages)

            if cap_status == "turn_cap":
                final_budget = self._context_budget_state(
                    messages, system, tool_schemas
                )
                if final_budget is not None:
                    cap_status = "context_cap"
                    cap_evidence = final_budget
                    cap_prompt = (
                        _CONTEXT_CAP_HANDOFF_PROMPT
                        + "\nContext budget evidence: "
                        + json.dumps(final_budget, sort_keys=True)
                    )
                    context.trace_store.append_event(
                        "agent.context_cap.reached",
                        context.identity,
                        agent_name=context.spec.name,
                        **final_budget,
                    )

            messages.append(Message(
                role="user",
                content=[TextContent(text=cap_prompt)],
            ))
            finalization_error: str | None = None
            try:
                snapshot = [message.to_dict() for message in messages]
                response = await self._complete_observed(
                    messages=messages,
                    tools=[],
                    system=system,
                    max_tokens=max_output,
                    think_level=context.thinking_level,
                    call_type="agent_finalizer",
                )
                self._trace_call(
                    system=system,
                    messages_snapshot=snapshot,
                    tool_schemas=[],
                    response=response,
                    call_type="agent_finalizer",
                )
                messages.append(response.to_message())
                summary = response.text.strip() if response.text else ""
                if not summary:
                    summary = (
                        f"{cap_status} reached, but the reserved finalization call returned "
                        "no textual handoff. Inspect files_changed and verification."
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                finalization_error = str(exc)
                summary = (
                    f"{cap_status} reached and the reserved finalization call failed: "
                    f"{exc}. Inspect files_changed and verification."
                )

            result = AgentResult(
                status=cap_status,
                summary=summary,
                files_changed=sorted(set(files_changed)),
                verification=verification,
                error=(
                    cap_status
                    if finalization_error is None
                    else f"{cap_status}_finalization_failed: {finalization_error}"
                ),
                agent_id=context.identity.agent_id,
                agent_name=context.spec.name,
            )
            return self._finish(result, messages)
        except asyncio.CancelledError:
            result = AgentResult(
                status="cancelled",
                summary="Subagent was cancelled.",
                files_changed=sorted(set(files_changed)),
                verification=verification,
                agent_id=context.identity.agent_id,
                agent_name=context.spec.name,
            )
            self._finish(result, messages)
            raise
        except Exception as exc:
            result = AgentResult(
                status="failed",
                summary="Subagent failed before completing its task.",
                files_changed=sorted(set(files_changed)),
                verification=verification,
                error=str(exc),
                agent_id=context.identity.agent_id,
                agent_name=context.spec.name,
            )
            return self._finish(result, messages)
        finally:
            await self._hooks.session_end()

    def _finish(self, result: AgentResult, messages: list[Message]) -> AgentResult:
        self._save_session(messages)
        self.context.trace_store.update_child(
            self.context.identity,
            status=result.status,
            result=result.to_payload(),
        )
        self.context.trace_store.append_event(
            f"agent.run.{result.status}",
            self.context.identity,
            agent_name=self.context.spec.name,
            result=result.to_payload(),
        )
        return result
