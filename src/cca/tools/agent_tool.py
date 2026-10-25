"""Agent tool — dispatch CCA-native, depth-one subagents."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

from cca.agent_runner import AgentResult, AgentRunContext, AgentRunner
from cca.agents import (
    AgentCatalog, discover_agents, resolve_agent_name, resolve_subagent_settings,
)
from cca.tools.base import Tool
from cca.trace_store import TraceIdentity, TraceStore, generation_from_session_id


_background_agents: dict[str, dict[str, Any]] = {}


def get_background_agent(agent_id: str) -> dict[str, Any] | None:
    return _background_agents.get(agent_id)


def get_persisted_agent(
    agent_id: str,
    *,
    root_id: str | None = None,
) -> dict[str, Any] | None:
    """Retrieve a durable child result after the creating process has exited."""
    import json
    from cca.paths import LLM_CALLS_DIR

    manifest_paths = (
        [LLM_CALLS_DIR / f"session_{root_id}_agents.json"]
        if root_id is not None
        else LLM_CALLS_DIR.glob("session_*_agents.json")
    )
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entry = (manifest.get("agents") or {}).get(agent_id)
        if isinstance(entry, dict):
            return entry
    return None


def _public_run(entry: dict[str, Any]) -> dict[str, Any]:
    """Return lifecycle state without exposing tasks or cancellation primitives."""
    result = entry.get("result")
    if hasattr(result, "to_payload"):
        result = result.to_payload()
    return {
        key: value
        for key, value in {
            "agent_id": entry.get("agent_id"),
            "root_id": entry.get("root_id"),
            "trace_id": entry.get("trace_id"),
            "agent_name": entry.get("agent_name")
                or (entry.get("agent_definition") or {}).get("name"),
            "description": entry.get("description", ""),
            "status": entry.get("status", "unknown"),
            "done": entry.get("done", entry.get("status") in {
                "completed", "failed", "cancelled", "turn_cap", "context_cap",
            }),
            "definition_hash": entry.get("definition_hash")
                or (entry.get("agent_definition") or {}).get("definition_hash"),
            "catalog_hash": entry.get("catalog_hash"),
            "created_at": entry.get("created_at"),
            "updated_at": entry.get("updated_at"),
            "result": result,
            "error": entry.get("error"),
        }.items()
        if value is not None
    }


def list_agent_runs(root_id: str) -> list[dict[str, Any]]:
    """List durable and in-process child runs belonging to one root trace."""
    import json
    from cca.paths import LLM_CALLS_DIR

    runs: dict[str, dict[str, Any]] = {}
    manifest_path = LLM_CALLS_DIR / f"session_{root_id}_agents.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    for agent_id, raw in (manifest.get("agents") or {}).items():
        if not isinstance(raw, dict):
            continue
        run = _public_run({"agent_id": agent_id, "root_id": root_id, **raw})
        runs[agent_id] = run

    for agent_id, raw in tuple(_background_agents.items()):
        if raw.get("root_id") != root_id:
            continue
        in_process = _public_run(raw)
        durable = runs.get(agent_id, {})
        # The manifest knows when a scheduled child has actually entered its
        # runner; preserve that more precise state until the in-process task ends.
        if not in_process.get("done") and durable.get("status") == "running":
            in_process["status"] = "running"
        runs[agent_id] = {**durable, **in_process}

    return sorted(
        runs.values(),
        key=lambda run: (run.get("created_at") or 0, run.get("agent_id") or ""),
    )


def get_agent_run(agent_id: str, *, root_id: str) -> dict[str, Any] | None:
    """Retrieve one child only when it belongs to the requested root trace."""
    return next(
        (run for run in list_agent_runs(root_id) if run.get("agent_id") == agent_id),
        None,
    )


def stop_background_agent(agent_id: str, *, root_id: str | None = None) -> bool:
    entry = _background_agents.get(agent_id)
    if entry is None or entry.get("done"):
        return False
    if root_id is not None and entry.get("root_id") != root_id:
        return False
    entry["cancellation"].set()
    task = entry.get("task")
    if task is not None:
        task.cancel()
    return True


class AgentTool(Tool):
    name = "Agent"
    description = "Delegate a bounded task to a CCA subagent."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(
        self,
        backend=None,
        config=None,
        feature_set=None,
        *,
        cwd: str = "",
        permissions=None,
        catalog: AgentCatalog | None = None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._feature_set = feature_set
        self.cwd = cwd or __import__("os").getcwd()
        self.catalog = catalog or discover_agents(self.cwd)
        self.permissions = permissions
        self.parent_session_id: str | None = None
        self.root_agent_id: str | None = None
        self.trace_id: str | None = None
        self.root_trace_id: str | None = None
        self.parent_context_generation: int = 0
        self.thinking_request = "auto"
        self.parent_allowed_tools: set[str] | None = None
        self.parent_disallowed_tools: set[str] = set()
        self.project_instructions = ""
        self.observer = None
        settings = resolve_subagent_settings(config, self.catalog)
        self.default_agent = settings.default
        self.max_depth = settings.max_depth
        self.max_concurrency = settings.max_concurrency
        self.allow_background = settings.allow_background
        self.depth = 0
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._mutation_lock = asyncio.Lock()

        listing = "\n".join(
            f"- {name}: {self.catalog.specs[name].description}"
            for name in self.catalog.names()
        )
        self.description = (
            "Delegate one bounded task, or a joined batch of independent tasks, to "
            "specialized CCA subagents. Joined tasks are scheduled together, safely "
            "run concurrently when their profiles cannot mutate the workspace, return "
            "in declaration order, and rejoin before one parent synthesis. Children "
            "have isolated messages and cannot spawn another agent.\n"
            "Available agents:\n" + listing
        )
        task_properties = {
            "description": {
                "type": "string",
                "description": "Short 3-8 word task description.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Bounded objective, relevant scope and constraints, expected result, "
                    "verification evidence, and completion condition."
                ),
            },
            "agent": {
                "type": "string",
                "enum": self.catalog.names(),
                "description": "Specialization. Omit to use the session default.",
            },
            "subagent_type": {
                "type": "string",
                "description": "Deprecated compatibility alias for agent.",
            },
            "model": {
                "type": "string",
                "description": "Optional model alias or full model selector override.",
            },
        }
        self.input_schema = {
            "type": "object",
            "description": (
                "Provide description and prompt for one task, or tasks for a joined "
                "batch. Do not combine the two forms."
            ),
            "properties": {
                **task_properties,
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": task_properties,
                        "required": ["description", "prompt"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "Independent tasks to execute concurrently and join before "
                        "returning. Use instead of top-level description and prompt."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        }
        if self.allow_background:
            self.input_schema["properties"]["run_in_background"] = {
                "type": "boolean",
                "description": (
                    "Detach a single task and notify later. Never valid with tasks."
                ),
            }

    def set_default_agent(self, name: str) -> tuple[str, str]:
        """Change only this session's default and return (previous, resolved)."""
        spec = resolve_agent_name(self.catalog, name)
        if spec is None:
            raise ValueError(
                f"unknown agent '{name}'. Available: {', '.join(self.catalog.names())}"
            )
        previous = self.default_agent
        self.default_agent = spec.name
        return previous, spec.name

    def bind_parent(
        self,
        *,
        session_id: str,
        root_id: str,
        trace_id: str,
        root_agent_id: str,
        thinking_request: str,
        backend=None,
        context_generation: int | None = None,
        permissions=None,
        allowed_tools: set[str] | None = None,
        disallowed_tools: set[str] | None = None,
        project_instructions: str = "",
        cwd: str | None = None,
        observer=None,
    ) -> None:
        self.parent_session_id = session_id
        self.root_trace_id = root_id
        self.trace_id = trace_id
        self.root_agent_id = root_agent_id
        self.thinking_request = thinking_request
        if backend is not None:
            self._backend = backend
        self.parent_context_generation = (
            generation_from_session_id(session_id)
            if context_generation is None
            else context_generation
        )
        self.permissions = permissions
        self.parent_allowed_tools = allowed_tools
        self.parent_disallowed_tools = disallowed_tools or set()
        self.project_instructions = project_instructions
        if cwd:
            self.cwd = cwd
        if observer is not None:
            self.observer = observer

    def _resolve_backend(self, requested_model: str | None, spec) -> tuple[Any, str]:
        from cca.llm.catalog import split_selector
        from cca.llm.factory import create_backend

        backend = self._backend
        selector = (
            self._config.model if self._config is not None
            else getattr(backend, "model", "")
        )
        selection = requested_model or spec.model
        if not selection or selection == "inherit":
            return backend, selector

        aliases = {
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-6",
            "haiku": "claude-haiku-4-5-20251001",
        }
        chosen = aliases.get(selection, selection)
        if ":" in chosen:
            selector = chosen
        else:
            provider, _ = split_selector(selector)
            selector = f"{provider}:{chosen}"
        provider, _ = split_selector(selector)
        backend = create_backend(
            selector,
            api_key=self._config.get_api_key(provider) if self._config else None,
            base_url=self._config.base_url if self._config else None,
        )
        return backend, selector

    async def execute(
        self,
        description: str | None = None,
        prompt: str | None = None,
        agent: str | None = None,
        subagent_type: str | None = None,
        model: str | None = None,
        run_in_background: bool = False,
        tasks: list[dict[str, Any]] | None = None,
    ) -> str:
        if tasks is not None:
            single_values = (description, prompt, agent, subagent_type, model)
            if any(value is not None for value in single_values):
                return "Error: provide tasks or single-task fields, not both."
            if run_in_background:
                return "Error: joined task batches cannot run in the background."
            return await self._execute_batch(tasks)
        if not isinstance(description, str) or not description.strip():
            return "Error: description is required for a single subagent task."
        if not isinstance(prompt, str) or not prompt.strip():
            return "Error: prompt is required for a single subagent task."
        return await self._execute_single(
            description=description,
            prompt=prompt,
            agent=agent,
            subagent_type=subagent_type,
            model=model,
            run_in_background=run_in_background,
        )

    async def _execute_batch(self, tasks: list[dict[str, Any]]) -> str:
        if not isinstance(tasks, list) or not tasks:
            return "Error: tasks must be a non-empty list."
        normalized: list[dict[str, Any]] = []
        allowed = {"description", "prompt", "agent", "subagent_type", "model"}
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                return f"Error: tasks[{index}] must be an object."
            unknown = sorted(set(task) - allowed)
            if unknown:
                return (
                    f"Error: tasks[{index}] has unknown field(s): "
                    + ", ".join(unknown)
                )
            description = task.get("description")
            prompt = task.get("prompt")
            if not isinstance(description, str) or not description.strip():
                return f"Error: tasks[{index}].description is required."
            if not isinstance(prompt, str) or not prompt.strip():
                return f"Error: tasks[{index}].prompt is required."
            normalized.append(task)

        from cca.ui.events import current_tool_id
        group_id = current_tool_id.get() or f"agent-batch-{uuid.uuid4().hex[:12]}"

        async def execute_task(index: int, task: dict[str, Any]) -> dict[str, Any]:
            try:
                raw = await self._execute_single(
                    description=task["description"],
                    prompt=task["prompt"],
                    agent=task.get("agent"),
                    subagent_type=task.get("subagent_type"),
                    model=task.get("model"),
                    run_in_background=False,
                    spawn_group_id=group_id,
                    spawn_index=index,
                    force_foreground=True,
                )
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    payload = {"status": "failed", "error": str(raw)}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                payload = {"status": "failed", "error": str(exc)}
            return {
                "index": index,
                "description": task["description"],
                "result": payload,
            }

        results = await asyncio.gather(*(
            execute_task(index, task) for index, task in enumerate(normalized)
        ))
        completed = sum(
            item["result"].get("status") == "completed" for item in results
        )
        status = (
            "completed" if completed == len(results)
            else "failed" if completed == 0
            else "partial"
        )
        return json_result({
            "status": status,
            "mode": "joined",
            "group_id": group_id,
            "results": results,
        })

    async def _execute_single(
        self,
        *,
        description: str,
        prompt: str,
        agent: str | None = None,
        subagent_type: str | None = None,
        model: str | None = None,
        run_in_background: bool = False,
        spawn_group_id: str | None = None,
        spawn_index: int | None = None,
        force_foreground: bool = False,
    ) -> str:
        if self._backend is None:
            return "Error: Agent tool requires a backend to be configured."
        if self.depth >= self.max_depth:
            return f"Error: maximum subagent depth ({self.max_depth}) reached."
        if agent is not None and subagent_type is not None and agent != subagent_type:
            return "Error: provide agent or subagent_type, not conflicting values for both."

        requested_name = agent or subagent_type or self.default_agent
        spec = resolve_agent_name(self.catalog, requested_name)
        if spec is None:
            return (
                f"Error: unknown agent '{requested_name}'. Available agents: "
                + ", ".join(self.catalog.names())
            )
        run_in_background = False if force_foreground else (
            run_in_background or spec.background
        )
        if run_in_background and not self.allow_background:
            return "Error: background subagents are disabled by configuration."
        if spec.isolation == "worktree":
            return (
                "Error: worktree-isolated subagents are not enabled in this release; "
                "shared-workspace mutation is serialized."
            )

        backend, selector = self._resolve_backend(model, spec)
        from cca.features import resolve_features
        from cca.llm.thinking import resolve_thinking
        from cca.tools import build_registry

        feature_set = self._feature_set or resolve_features(self._config)
        requested_thinking = spec.effort or self.thinking_request
        configured_thinking = getattr(self._config, "default_thinking_level", "auto")
        thinking = resolve_thinking(selector, requested_thinking, configured_thinking)
        describe = getattr(backend, "describe_request_settings", None)
        backend_request = {
            "adapter": type(backend).__name__,
            "thinking_parameters": {},
            "reasoning_summary": False,
        }
        if callable(describe):
            try:
                described = describe(thinking.effective, False)
                if isinstance(described, dict):
                    backend_request = described
            except Exception:
                pass
        settings = {
            "run_mode": "subagent",
            "model": selector,
            "model_requested": model or spec.model or "inherit",
            "thinking": thinking.to_payload(),
            "backend_request": backend_request,
        }

        root_id = self.root_trace_id or re.sub(
            r"-comp\d+$", "", self.parent_session_id or uuid.uuid4().hex[:12]
        )
        root_agent_id = self.root_agent_id or root_id
        trace_id = self.trace_id or root_id
        child_agent_id = uuid.uuid4().hex[:12]
        from cca.ui.events import current_tool_id
        spawn_call_id = current_tool_id.get() or f"agent-{uuid.uuid4().hex[:12]}"
        parent_generation = self.parent_context_generation
        identity = TraceIdentity(
            trace_id=trace_id,
            agent_id=child_agent_id,
            parent_agent_id=root_agent_id,
            spawn_call_id=spawn_call_id,
            # Preserve the legacy parent generation while naming the child's
            # independent model-visible context generation explicitly.
            generation=parent_generation,
            context_generation=0,
            generation_reason="initial",
            parent_context_generation=parent_generation,
        )
        parent_session = self.parent_session_id or root_id
        session_id = f"{parent_session}-sub-{child_agent_id}"
        trace_store = TraceStore(root_id)
        trace_store.register_child(
            identity,
            root_agent_id=root_agent_id,
            agent_definition=spec.to_payload(),
            catalog_hash=self.catalog.catalog_hash,
            trigger="automatic",
            description=description,
            spawn_group_id=spawn_group_id,
            spawn_index=spawn_index,
        )
        trace_store.append_event(
            "agent.spawn.requested",
            identity,
            requested_agent=requested_name,
            resolved_agent=spec.name,
            description=description,
            prompt=prompt,
            background=run_in_background,
            **(
                {"spawn_group_id": spawn_group_id}
                if spawn_group_id is not None else {}
            ),
            **({"spawn_index": spawn_index} if spawn_index is not None else {}),
        )

        registry = build_registry(
            config=self._config,
            permissions=self.permissions,
            backend=backend,
            cwd=self.cwd,
            feature_set=feature_set,
        )
        # Defense in depth: children never receive or dispatch Agent even if a
        # custom definition asks for it or the parent feature set enables it.
        registry.deregister("Agent")
        cancellation = asyncio.Event()
        context = AgentRunContext(
            identity=identity,
            root_agent_id=root_agent_id,
            parent_session_id=parent_session,
            session_id=session_id,
            cwd=self.cwd,
            spec=spec,
            description=description,
            prompt=prompt,
            trigger="automatic",
            backend=backend,
            registry=registry,
            config=self._config,
            feature_set=feature_set,
            trace_store=trace_store,
            catalog_hash=self.catalog.catalog_hash,
            thinking_level=thinking.effective,
            settings=settings,
            permissions=self.permissions,
            parent_allowed_tools=self.parent_allowed_tools,
            parent_disallowed_tools=self.parent_disallowed_tools,
            project_instructions=self.project_instructions,
            cancellation=cancellation,
            observer=self.observer,
            parent_span=(self.observer.current_span if self.observer else None),
            background=run_in_background,
        )

        async def run_child():
            async with self._semaphore:
                runner = AgentRunner(context)
                # Any profile with inherited tools may mutate. Shared-workspace
                # mutation remains serial until worktree isolation is available.
                may_mutate = spec.tools is None or bool(
                    set(spec.tools) & {"Write", "Edit", "NotebookEdit", "Bash"}
                )
                if may_mutate:
                    async with self._mutation_lock:
                        return await runner.run()
                return await runner.run()

        if run_in_background:
            task = asyncio.create_task(run_child())
            entry: dict[str, Any] = {
                "agent_id": child_agent_id,
                "root_id": root_id,
                "trace_id": trace_id,
                "agent_name": spec.name,
                "description": description,
                "status": "queued",
                "done": False,
                "result": None,
                "definition_hash": spec.definition_hash,
                "catalog_hash": self.catalog.catalog_hash,
                "created_at": int(time.time()),
                "task": task,
                "cancellation": cancellation,
            }
            _background_agents[child_agent_id] = entry

            async def complete_background() -> None:
                try:
                    result = await task
                    entry.update(status=result.status, done=True, result=result)
                    from cca import notifications
                    await notifications.push({
                        "type": "agent_done",
                        "description": description,
                        "job_id": child_agent_id,
                        "result": result.to_tool_result(),
                    })
                except asyncio.CancelledError:
                    result = AgentResult(
                        status="cancelled",
                        summary="Subagent was cancelled.",
                        agent_id=child_agent_id,
                        agent_name=spec.name,
                    )
                    entry.update(status="cancelled", done=True, result=result)
                    persisted = get_persisted_agent(
                        child_agent_id, root_id=root_id
                    )
                    if not persisted or persisted.get("status") != "cancelled":
                        trace_store.update_child(
                            identity,
                            status="cancelled",
                            result=result.to_payload(),
                        )
                        trace_store.append_event(
                            "agent.run.cancelled",
                            identity,
                            agent_name=spec.name,
                            result=result.to_payload(),
                        )
                except Exception as exc:
                    entry.update(status="failed", done=True, error=str(exc))

            asyncio.create_task(complete_background())
            return json_result({
                "status": "queued",
                "agent_id": child_agent_id,
                "description": description,
            })

        result = await run_child()
        return result.to_tool_result()


def json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
