"""Subagent definitions, discovery, runtime, and trace contracts."""

from __future__ import annotations

import json

import pytest

from cca.agents import (
    AgentDefinitionError,
    SubagentConfigError,
    discover_agents,
    load_agent_file,
    resolve_agent_name,
    resolve_subagent_settings,
)
from cca.trace_store import TraceIdentity, TraceStore
from cca.llm.base import (
    CompletionResponse, Message, TextContent, ToolResultContent, ToolUseContent,
)


def test_builtin_catalog_is_cca_native_and_stable(tmp_path):
    catalog = discover_agents(tmp_path, global_dir=tmp_path / "global")
    assert catalog.names() == [
        "blueprint", "researcher", "reviewer", "risk-auditor", "scout",
        "verifier", "worker",
    ]
    assert catalog.get("worker").scope == "builtin"
    assert catalog.get("scout").tools == {
        "Read", "Glob", "Grep", "SearchInFile", "LSP",
    }
    assert {spec.max_turns for spec in catalog.specs.values()} == {40}
    assert len(catalog.catalog_hash) == 64


def test_discovery_reads_only_cca_agent_directories(tmp_path):
    repo = tmp_path / "repo"
    for stray in (repo / "agents", repo / ".config" / "agents"):
        stray.mkdir(parents=True)
        (stray / "stray.md").write_text(
            "---\nname: stray\ndescription: stray agent\n---\nDo stray work.\n"
        )

    catalog = discover_agents(repo, global_dir=tmp_path / "global")
    assert "stray" not in catalog.specs


def test_project_agent_overrides_global_and_builtin(tmp_path):
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "worker.md").write_text(
        "---\nname: worker\ndescription: global worker\ntools: Read\n---\nGlobal prompt.\n"
    )
    repo = tmp_path / "repo"
    project_dir = repo / ".cca" / "agents"
    project_dir.mkdir(parents=True)
    (project_dir / "worker.md").write_text(
        "---\nname: worker\ndescription: project worker\ntools: [Read, Grep]\n---\nProject prompt.\n"
    )

    catalog = discover_agents(repo, global_dir=global_dir)
    worker = catalog.get("worker")
    assert worker.description == "project worker"
    assert worker.prompt == "Project prompt."
    assert worker.scope == "project"


def test_nested_project_agent_wins(tmp_path):
    outer = tmp_path / ".cca" / "agents"
    inner = tmp_path / "packages" / "app" / ".cca" / "agents"
    outer.mkdir(parents=True)
    inner.mkdir(parents=True)
    for directory, prompt in ((outer, "Outer"), (inner, "Inner")):
        (directory / "specialist.md").write_text(
            "---\nname: specialist\ndescription: specialist\n---\n" + prompt
        )
    catalog = discover_agents(tmp_path / "packages" / "app", global_dir=tmp_path / "user-agents")
    assert catalog.get("specialist").prompt == "Inner"


def test_agent_file_supports_portable_field_aliases(tmp_path):
    path = tmp_path / "audit.md"
    path.write_text(
        "---\n"
        "name: audit\n"
        "description: audits a scope\n"
        "tools: Read, Grep, Bash\n"
        "disallowedTools: Write, Edit\n"
        "model: inherit\n"
        "maxTurns: 9\n"
        "effort: high\n"
        "color: purple\n"
        "---\n"
        "Audit carefully.\n"
    )
    spec, warnings = load_agent_file(path, scope="project")
    assert spec.tools == {"Read", "Grep", "Bash"}
    assert spec.disallowed_tools == {"Write", "Edit"}
    assert spec.max_turns == 9
    assert spec.effort == "high"
    assert warnings == [f"{path}: unsupported field 'color' ignored"]


@pytest.mark.parametrize("name", ["UPPER", "two_words", "-leading", "two--dash"])
def test_invalid_agent_names_are_rejected(tmp_path, name):
    path = tmp_path / "bad.md"
    path.write_text(
        f"---\nname: {name}\ndescription: invalid\n---\nPrompt.\n"
    )
    with pytest.raises(AgentDefinitionError, match="kebab-case"):
        load_agent_file(path, scope="project")


def test_invalid_discovered_file_warns_without_breaking_catalog(tmp_path):
    agent_dir = tmp_path / ".cca" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "bad.md").write_text("not frontmatter")
    catalog = discover_agents(tmp_path, global_dir=tmp_path / "global")
    assert catalog.get("worker") is not None
    assert "must start with YAML frontmatter" in catalog.warnings[0]


def test_legacy_names_are_not_resolved(tmp_path):
    catalog = discover_agents(tmp_path, global_dir=tmp_path / "global")
    assert resolve_agent_name(catalog, "Explore") is None
    assert resolve_agent_name(catalog, "scout").name == "scout"


def test_subagent_settings_validate_default_depth_and_concurrency(tmp_path):
    from cca.config import Config
    catalog = discover_agents(tmp_path, global_dir=tmp_path / "global")
    defaults = resolve_subagent_settings(Config(), catalog)
    assert defaults.allow_background is False
    settings = resolve_subagent_settings(
        Config(subagents={
            "default": "scout", "max_depth": 1,
            "max_concurrency": 5, "allow_background": False,
        }),
        catalog,
    )
    assert settings.default == "scout"
    assert settings.max_concurrency == 5
    assert settings.allow_background is False
    with pytest.raises(SubagentConfigError, match="max_depth"):
        resolve_subagent_settings(Config(subagents={"max_depth": 2}), catalog)
    with pytest.raises(SubagentConfigError, match="Unknown default"):
        resolve_subagent_settings(Config(subagents={"default": "missing"}), catalog)


def test_background_delegation_schema_is_opt_in(tmp_path):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    default_config = Config()
    default_tool = build_registry(
        config=default_config, cwd=str(tmp_path),
        feature_set=resolve_features(default_config),
    ).get("Agent")
    enabled_config = Config(subagents={"allow_background": True})
    enabled_tool = build_registry(
        config=enabled_config, cwd=str(tmp_path),
        feature_set=resolve_features(enabled_config),
    ).get("Agent")

    assert "run_in_background" not in default_tool.input_schema["properties"]
    assert "run_in_background" in enabled_tool.input_schema["properties"]


def test_definition_payload_contains_exact_prompt_and_provenance(tmp_path):
    catalog = discover_agents(tmp_path, global_dir=tmp_path / "global")
    payload = catalog.get("worker").to_payload()
    assert payload["prompt"]
    assert payload["definition_hash"]
    assert payload["source"] == "builtin"
    json.dumps(payload)


def test_trace_store_creates_no_child_artifacts_until_registration(tmp_path):
    store = TraceStore("root", directory=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert not store.manifest_path.exists()
    assert not store.event_path.exists()


def test_trace_store_separates_root_and_child_calls(tmp_path):
    store = TraceStore("root", directory=tmp_path)
    root = TraceIdentity(trace_id="trace", agent_id="root")
    child = TraceIdentity(
        trace_id="trace", agent_id="child", parent_agent_id="root",
        spawn_call_id="spawn-1",
    )
    store.append_call(root, 0, {"type": "main", "messages": []}, child=False)
    store.append_call(child, 0, {"type": "agent", "messages": []}, child=True)

    root_record = json.loads(store.root_call_path.read_text().splitlines()[0])
    child_record = json.loads(store.child_call_path("child").read_text().splitlines()[0])
    assert root_record["agent_id"] == "root"
    assert root_record["parent_agent_id"] is None
    assert child_record["agent_id"] == "child"
    assert child_record["parent_agent_id"] == "root"
    assert child_record["spawn_call_id"] == "spawn-1"
    assert store.next_call_idx(root, child=False) == 1
    assert store.next_call_idx(child, child=True) == 1


def test_manifest_is_created_only_for_invoked_child(tmp_path):
    store = TraceStore("root", directory=tmp_path)
    child = TraceIdentity(
        trace_id="trace", agent_id="child", parent_agent_id="root",
        spawn_call_id="spawn-1",
    )
    store.register_child(
        child,
        root_agent_id="root",
        agent_definition={"name": "scout", "prompt": "exact"},
        catalog_hash="catalog",
        trigger="automatic",
        description="map code",
    )
    store.update_child(child, status="completed", result={"summary": "done"})
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["agents"]["child"]["status"] == "completed"
    assert manifest["agents"]["child"]["agent_definition"]["prompt"] == "exact"


def test_job_output_retrieves_completed_agent_after_in_memory_state_is_gone(tmp_path, monkeypatch):
    import asyncio
    from cca import paths
    from cca.tools.agent_tool import _background_agents
    from cca.tools.job_tools import JobOutputTool

    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path)
    (tmp_path / "session_root_agents.json").write_text(json.dumps({
        "agents": {
            "durable-child": {
                "status": "completed",
                "description": "durable task",
                "result": {
                    "status": "completed",
                    "agent_id": "durable-child",
                    "summary": "persisted result",
                },
            },
        },
    }))
    _background_agents.pop("durable-child", None)
    result = json.loads(asyncio.run(JobOutputTool().execute(
        job_id="durable-child", block=False, timeout=1,
    )))
    assert result["summary"] == "persisted result"


def test_model_facing_job_controls_are_scoped_to_the_bound_root(monkeypatch):
    import asyncio
    from cca.tools.agent_tool import _background_agents
    from cca.tools.job_tools import JobCancelTool, JobOutputTool

    async def exercise():
        cancellation = asyncio.Event()
        _background_agents["foreign-agent"] = {
            "agent_id": "foreign-agent",
            "root_id": "root-b",
            "status": "running",
            "done": False,
            "description": "private foreign result",
            "cancellation": cancellation,
            "task": None,
        }
        try:
            output = await JobOutputTool(root_id="root-a").execute(
                job_id="foreign-agent", block=False, timeout=1,
            )
            stopped = await JobCancelTool(root_id="root-a").execute(
                job_id="foreign-agent"
            )
            assert "not found" in output
            assert "not found" in stopped
            assert cancellation.is_set() is False
        finally:
            _background_agents.pop("foreign-agent", None)

    asyncio.run(exercise())


class _AllowPermissions:
    never_allow: set[str] = set()

    async def check(self, tool_name, tool_input):
        return True


class _TwoTurnBackend:
    def __init__(self, tool_name="Write", tool_input=None):
        self.calls = 0
        self.tool_name = tool_name
        self.tool_input = tool_input or {}

    async def complete(self, messages, tools, system, max_tokens=8192, think_level="off"):
        self.calls += 1
        if self.calls == 1:
            return CompletionResponse(
                content=[ToolUseContent(
                    id="tool-1", name=self.tool_name, input=self.tool_input,
                )],
                usage={"input_tokens": 10, "output_tokens": 2},
            )
        return CompletionResponse(
            content=[TextContent(text="finished with evidence")],
            usage={"input_tokens": 12, "output_tokens": 4},
        )


class _TurnCapBackend:
    def __init__(self, target, *, fail_finalization=False):
        self.target = target
        self.fail_finalization = fail_finalization
        self.calls = 0
        self.tool_names: list[list[str]] = []

    async def complete(self, messages, tools, system, max_tokens=8192, think_level="off"):
        self.calls += 1
        self.tool_names.append([tool.name for tool in tools])
        if self.calls < 3:
            return CompletionResponse(
                content=[ToolUseContent(
                    id=f"read-{self.calls}",
                    name="Read",
                    input={"file_path": str(self.target)},
                )],
                usage={"input_tokens": 5, "output_tokens": 2},
            )
        if self.fail_finalization:
            raise RuntimeError("finalizer unavailable")
        return CompletionResponse(
            content=[TextContent(text=(
                "Work completed: inspected the target twice.\n"
                "Files changed: none.\n"
                "Verification: reads succeeded.\n"
                "Remaining work: delegated task is unfinished.\n"
                "Risks: more investigation is needed."
            ))],
            usage={"input_tokens": 8, "output_tokens": 10},
        )


def _redirect_runtime_paths(tmp_path, monkeypatch):
    from cca import paths, sessions
    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "llm-calls")
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sessions, "HISTORY_FILE", tmp_path / "history.jsonl")


def test_agent_tool_runs_worker_and_persists_exact_separate_trace(tmp_path, monkeypatch):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    target = tmp_path / "created.txt"
    backend = _TwoTurnBackend(
        "Write", {"file_path": str(target), "content": "created by worker\n"}
    )
    config = Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )

    result = json.loads(__import__("asyncio").run(tool.execute(
        description="create bounded file",
        prompt="Create the requested file and verify it.",
        agent="worker",
    )))
    assert result["status"] == "completed"
    assert result["agent_name"] == "worker"
    assert target.read_text() == "created by worker\n"
    assert result["files_changed"] == [str(target)]

    call_files = list((tmp_path / "llm-calls").glob("session_root_sub_*.jsonl"))
    assert len(call_files) == 1
    calls = [json.loads(line) for line in call_files[0].read_text().splitlines()]
    assert [call["call_idx"] for call in calls] == [0, 1]
    assert calls[0]["messages"][0]["content"][0]["text"] == (
        "Create the requested file and verify it."
    )
    assert calls[1]["response"]["text"] == "finished with evidence"
    assert all(call["agent_definition"]["definition_hash"] for call in calls)
    assert all(schema["name"] != "Agent" for schema in calls[0]["tools"])
    assert {call["context_generation"] for call in calls} == {0}
    assert {call["generation_reason"] for call in calls} == {"initial"}
    assert {call["parent_context_generation"] for call in calls} == {0}
    assert all(call["trace_id"] == "trace" for call in calls)
    assert all(call["model_provenance"]["backend"] == "_TwoTurnBackend" for call in calls)
    assert all(
        call["model_provenance"]["selector"] == config.model for call in calls
    )

    manifest = json.loads(
        (tmp_path / "llm-calls" / "session_root_agents.json").read_text()
    )
    child = next(iter(manifest["agents"].values()))
    assert child["status"] == "completed"
    assert child["result"]["summary"] == "finished with evidence"
    assert child["context_generation"] == 0
    assert child["parent_context_generation"] == 0

    session_files = list((tmp_path / "sessions").glob("*.json"))
    saved = json.loads(session_files[0].read_text())
    assert saved["messages"][-1]["role"] == "assistant"
    assert saved["messages"][-1]["content"][0]["text"] == "finished with evidence"


def test_parallel_children_finish_out_of_order_without_losing_or_reordering_results(
    tmp_path, monkeypatch,
):
    import asyncio
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)

    class Backend:
        def __init__(self):
            self.finished: list[str] = []

        async def complete(self, messages, tools, system, max_tokens=8192, think_level="off"):
            prompt = messages[0].content[0].text
            label = "slow" if "slow" in prompt else "fast"
            await asyncio.sleep(0.03 if label == "slow" else 0.001)
            self.finished.append(label)
            return CompletionResponse(content=[TextContent(text=f"{label} result")])

    backend = Backend()
    config = Config(subagents={"max_concurrency": 2})
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )

    async def run_both():
        return await asyncio.gather(
            tool.execute(
                description="slow child", prompt="perform the slow inspection",
                agent="scout",
            ),
            tool.execute(
                description="fast child", prompt="perform the fast inspection",
                agent="scout",
            ),
        )

    results = [json.loads(value) for value in asyncio.run(run_both())]
    assert backend.finished == ["fast", "slow"]
    assert [result["summary"] for result in results] == ["slow result", "fast result"]
    manifest = json.loads(
        (tmp_path / "llm-calls" / "session_root_agents.json").read_text()
    )
    assert len(manifest["agents"]) == 2
    assert {entry["status"] for entry in manifest["agents"].values()} == {"completed"}
    child_logs = list((tmp_path / "llm-calls").glob("session_root_sub_*.jsonl"))
    assert len(child_logs) == 2
    assert all(len(path.read_text().splitlines()) == 1 for path in child_logs)


def test_joined_batch_runs_concurrently_and_returns_in_declaration_order(
    tmp_path, monkeypatch,
):
    import asyncio
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry
    from cca.ui.events import current_tool_id

    _redirect_runtime_paths(tmp_path, monkeypatch)
    agent_dir = tmp_path / ".cca" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "detached-scout.md").write_text(
        "---\nname: detached-scout\ndescription: background profile probe\n"
        "tools: [Read, Glob, Grep]\nbackground: true\n---\nInspect the bounded scope.\n"
    )

    class Backend:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.finished: list[str] = []

        async def complete(self, messages, tools, system, max_tokens=8192, think_level="off"):
            prompt = messages[0].content[0].text
            label = "slow" if "slow" in prompt else "fast"
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.03 if label == "slow" else 0.001)
                self.finished.append(label)
                return CompletionResponse(content=[TextContent(text=f"{label} result")])
            finally:
                self.active -= 1

    backend = Backend()
    config = Config(subagents={"max_concurrency": 2, "allow_background": False})
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )
    assert "tasks" in tool.input_schema["properties"]
    assert "run_in_background" not in tool.input_schema["properties"]

    token = current_tool_id.set("batch-call")
    try:
        raw = asyncio.run(tool.execute(tasks=[
            {
                "description": "slow child",
                "prompt": "perform the slow inspection",
                "agent": "detached-scout",
            },
            {
                "description": "fast child",
                "prompt": "perform the fast inspection",
                "agent": "scout",
            },
        ]))
    finally:
        current_tool_id.reset(token)

    result = json.loads(raw)
    assert result["status"] == "completed"
    assert result["mode"] == "joined"
    assert result["group_id"] == "batch-call"
    assert backend.max_active == 2
    assert backend.finished == ["fast", "slow"]
    assert [item["result"]["summary"] for item in result["results"]] == [
        "slow result", "fast result",
    ]
    manifest = json.loads(
        (tmp_path / "llm-calls" / "session_root_agents.json").read_text()
    )
    entries = sorted(manifest["agents"].values(), key=lambda item: item["spawn_index"])
    assert [entry["spawn_group_id"] for entry in entries] == [
        "batch-call", "batch-call",
    ]
    assert [entry["description"] for entry in entries] == [
        "slow child", "fast child",
    ]
    from cca.trace_view import inspect_trace
    inspected = inspect_trace("root", directory=tmp_path / "llm-calls")
    inspected_entries = sorted(
        inspected["children"], key=lambda item: item["spawn_index"]
    )
    assert [item["spawn_group_id"] for item in inspected_entries] == [
        "batch-call", "batch-call",
    ]


def test_joined_batch_rejects_detached_or_ambiguous_inputs(tmp_path):
    import asyncio
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    config = Config()
    registry = build_registry(
        config=config, cwd=str(tmp_path), feature_set=resolve_features(config),
    )
    tool = registry.get("Agent")
    detached = asyncio.run(tool.execute(
        tasks=[{"description": "one", "prompt": "inspect"}],
        run_in_background=True,
    ))
    ambiguous = asyncio.run(tool.execute(
        description="one", prompt="inspect",
        tasks=[{"description": "two", "prompt": "review"}],
    ))
    assert "cannot run in the background" in detached
    assert "not both" in ambiguous


def test_child_round_trips_opaque_tool_state_and_remains_sft_exportable(
    tmp_path, monkeypatch,
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from cca.config import Config
    from cca.features import resolve_features
    from cca.llm.openai_backend import OpenAIBackend
    from cca.sft_export import export_trace
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    (tmp_path / "App.swift").write_text("struct App {}\n")
    backend = OpenAIBackend(
        model="gemini-3.7-flash", provider="google", api_key="fake",
        base_url="http://fake",
    )
    tool_call = SimpleNamespace(
        id="call-glob",
        function=SimpleNamespace(
            name="Glob", arguments='{"pattern":"*.swift"}', model_extra={},
        ),
        model_extra={
            "extra_content": {
                "google": {"thought_signature": "child-signature"},
            },
        },
    )
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    model_extra={}, reasoning_content=None, content=None,
                    tool_calls=[tool_call],
                ),
                finish_reason="tool_calls",
            )],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    model_extra={}, reasoning_content=None,
                    content="Reviewed App.swift.", tool_calls=[],
                ),
                finish_reason="stop",
            )],
            usage=None,
        ),
    ]
    config = Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )

    with patch.object(
        backend.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=responses),
    ) as create:
        result = json.loads(asyncio.run(tool.execute(
            description="review swift file",
            prompt="Inspect the Swift files and report what you reviewed.",
            agent="scout",
        )))

    assert result["status"] == "completed"
    assert result["summary"] == "Reviewed App.swift."
    replayed = create.await_args_list[1].kwargs["messages"][2]["tool_calls"][0]
    assert replayed["extra_content"] == {
        "google": {"thought_signature": "child-signature"},
    }
    children = export_trace(
        "root", directory=tmp_path / "llm-calls", view="children-separated",
    )
    serialized = json.dumps(children)
    assert len(children) == 1
    assert "child-signature" not in serialized
    assert "provider_state" not in serialized


def test_read_only_agent_rejects_hallucinated_write_at_dispatch(tmp_path, monkeypatch):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    target = tmp_path / "must-not-exist.txt"
    backend = _TwoTurnBackend(
        "Write", {"file_path": str(target), "content": "unsafe"}
    )
    config = Config()
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=resolve_features(config),
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )
    result = json.loads(__import__("asyncio").run(tool.execute(
        description="inspect only", prompt="Inspect without changes.", agent="scout",
    )))
    assert result["status"] == "completed"
    assert not target.exists()
    saved = json.loads(next((tmp_path / "sessions").glob("*.json")).read_text())
    denied = saved["messages"][2]["content"][0]
    assert denied["is_error"] is True
    assert "not allowed" in denied["content"]


def test_inherited_worker_never_receives_interactive_or_task_control_tools(
    tmp_path, monkeypatch,
):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)

    class Backend:
        def __init__(self):
            self.tool_names = []

        async def complete(self, messages, tools, system, max_tokens=8192, think_level="off"):
            self.tool_names.append({schema.name for schema in tools})
            return CompletionResponse(content=[TextContent(text="done")])

    backend = Backend()
    config = Config()
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=resolve_features(config),
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )
    result = json.loads(__import__("asyncio").run(tool.execute(
        description="safe worker", prompt="Return immediately.", agent="worker",
    )))
    assert result["status"] == "completed"
    assert {"Agent", "AskUserQuestion", "JobOutput", "JobCancel"}.isdisjoint(
        backend.tool_names[0]
    )


def test_turn_cap_reserves_tool_free_final_call_and_returns_handoff(tmp_path, monkeypatch):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.sft_export import export_trace
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    target = tmp_path / "target.txt"
    target.write_text("inspect me\n")
    agent_dir = tmp_path / ".cca" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "short-run.md").write_text(
        "---\nname: short-run\ndescription: exercises turn cap\n"
        "tools: [Read]\nmax_turns: 3\n---\nInspect until the cap.\n"
    )
    backend = _TurnCapBackend(target)
    config = Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )

    result = json.loads(__import__("asyncio").run(tool.execute(
        description="exercise turn cap",
        prompt="Keep inspecting so the turn cap is reached.",
        agent="short-run",
    )))
    assert backend.calls == 3
    assert backend.tool_names[0] == ["Read"]
    assert backend.tool_names[1] == ["Read"]
    assert backend.tool_names[2] == []
    assert result["status"] == "turn_cap"
    assert result["error"] == "turn_cap"
    assert "Remaining work: delegated task is unfinished" in result["summary"]

    call_path = next((tmp_path / "llm-calls").glob("session_root_sub_*.jsonl"))
    calls = [json.loads(line) for line in call_path.read_text().splitlines()]
    assert [call["call_idx"] for call in calls] == [0, 1, 2]
    assert [call["type"] for call in calls] == [
        "agent", "agent", "agent_finalizer",
    ]
    assert calls[-1]["tools"] == []
    assert "final reserved turn" in calls[-1]["messages"][-1]["content"][0]["text"]

    manifest = json.loads(
        (tmp_path / "llm-calls" / "session_root_agents.json").read_text()
    )
    child = next(iter(manifest["agents"].values()))
    assert child["status"] == "turn_cap"
    assert child["result"]["summary"] == result["summary"]
    assert export_trace(
        "root", directory=tmp_path / "llm-calls", view="children-separated"
    ) == []


def test_turn_cap_returns_structured_fallback_if_finalizer_fails(tmp_path, monkeypatch):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    target = tmp_path / "target.txt"
    target.write_text("inspect me\n")
    agent_dir = tmp_path / ".cca" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "short-run.md").write_text(
        "---\nname: short-run\ndescription: exercises turn cap\n"
        "tools: [Read]\nmax_turns: 3\n---\nInspect until the cap.\n"
    )
    backend = _TurnCapBackend(target, fail_finalization=True)
    config = Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )
    result = json.loads(__import__("asyncio").run(tool.execute(
        description="exercise failed finalizer",
        prompt="Keep inspecting so the turn cap is reached.", agent="short-run",
    )))
    assert result["status"] == "turn_cap"
    assert result["error"].startswith("turn_cap_finalization_failed:")
    assert "finalizer unavailable" in result["summary"]


def test_child_context_guard_returns_tool_free_handoff_before_provider_failure(
    tmp_path, monkeypatch,
):
    from cca.agent_runner import AgentRunner
    from cca.config import Config
    from cca.features import resolve_features
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    target = tmp_path / "target.txt"
    target.write_text("inspect me\n")
    agent_dir = tmp_path / ".cca" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "context-run.md").write_text(
        "---\nname: context-run\ndescription: exercises context guard\n"
        "tools: [Read]\nmax_turns: 3\n---\nInspect until context is nearly full.\n"
    )
    backend = _TurnCapBackend(target)
    config = Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    monkeypatch.setattr(
        AgentRunner,
        "_context_budget_state",
        lambda self, messages, system, tools: (
            None if len(messages) == 1 else {
                "estimated_tokens": 170_000,
                "context_window": 200_000,
                "handoff_threshold": 170_000,
            }
        ),
    )
    tool = registry.get("Agent")
    tool.bind_parent(
        session_id="root-comp0", root_id="root", trace_id="trace",
        root_agent_id="root", thinking_request="off",
        permissions=_AllowPermissions(), cwd=str(tmp_path),
    )

    result = json.loads(__import__("asyncio").run(tool.execute(
        description="exercise context guard",
        prompt="Inspect without overflowing context.",
        agent="context-run",
    )))
    assert result["status"] == "context_cap"
    assert result["error"] == "context_cap"
    assert backend.tool_names == [["Read"], []]
    call_path = next((tmp_path / "llm-calls").glob("session_root_sub_*.jsonl"))
    calls = [json.loads(line) for line in call_path.read_text().splitlines()]
    assert [call["type"] for call in calls] == ["agent", "agent_finalizer"]
    assert "context limit" in calls[-1]["messages"][-1]["content"][0]["text"]
    events = [
        json.loads(line)
        for line in (tmp_path / "llm-calls" / "session_root_agents.jsonl")
        .read_text().splitlines()
    ]
    assert any(event["event"] == "agent.context_cap.reached" for event in events)


def test_enabled_but_unused_writes_no_child_artifacts(tmp_path, monkeypatch):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    config = Config()
    features = resolve_features(config)
    registry = build_registry(config=config, cwd=str(tmp_path), feature_set=features)
    repl = Repl(
        backend=None, registry=registry, config=config, permissions=_AllowPermissions(),
        cwd=str(tmp_path), feature_set=features,
    )
    repl._log_llm_call(
        call_type="main", system="sys", messages_snapshot=[], tools=[], params={},
        response_text="done", tool_calls=[], thinking="", stop_reason="end_turn",
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0,
    )
    files = {path.name for path in (tmp_path / "llm-calls").iterdir()}
    assert files == {f"session_{repl._base_session_id}.jsonl"}
    record = json.loads(next((tmp_path / "llm-calls").iterdir()).read_text())
    assert record["multi_agent_capable"] is True


def test_disabled_scaffold_has_no_agent_tool_or_delegation_prompt(tmp_path, monkeypatch):
    from cca import paths
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "calls")
    config = Config(tools={"Agent": False})
    features = resolve_features(config)
    registry = build_registry(config=config, cwd=str(tmp_path), feature_set=features)
    repl = Repl(
        backend=None, registry=registry, config=config, permissions=_AllowPermissions(),
        cwd=str(tmp_path), feature_set=features,
    )
    assert registry.get("Agent") is None
    assert "# Delegation" not in repl._build_system()
    repl._log_llm_call(
        call_type="main", system="legacy", messages_snapshot=[], tools=[], params={},
        response_text="done", tool_calls=[], thinking="", stop_reason="end_turn",
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0,
    )
    record = json.loads(next((tmp_path / "calls").glob("*.jsonl")).read_text())
    assert record["multi_agent_capable"] is False
    assert not list((tmp_path / "calls").glob("*_agents.json*"))


def test_enabled_scaffold_exposes_named_activation_contract(tmp_path):
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    agent_dir = tmp_path / ".cca" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "activation-probe.md").write_text(
        "---\nname: activation-probe\ndescription: activation contract probe\n"
        "tools: [Read]\n---\nInspect one bounded scope.\n"
    )
    config = Config()
    features = resolve_features(config)
    registry = build_registry(config=config, cwd=str(tmp_path), feature_set=features)
    repl = Repl(
        backend=None, registry=registry, config=config, permissions=_AllowPermissions(),
        cwd=str(tmp_path), feature_set=features,
    )

    system = repl._build_system()
    assert "# Delegation" in system
    assert "bounded, independently verifiable task" in system
    assert "Do not delegate simple or tightly serial work" in system
    agent_tool = registry.get("Agent")
    assert "activation-probe" in agent_tool.input_schema["properties"]["agent"]["enum"]
    assert "activation-probe: activation contract probe" in agent_tool.description


def _terminal_call(agent_id, call_type, text="answer"):
    return {
        "schema_version": 2,
        "trace_id": "trace",
        "agent_id": agent_id,
        "parent_agent_id": None if call_type == "main" else "root",
        "call_idx": 0,
        "type": call_type,
        "system": "exact system",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "thinking", "thinking": "must not train"},
                {"type": "text", "text": "task"},
            ],
        }],
        "tools": [],
        "response": {"text": text, "tool_calls": []},
    }


def test_sft_export_views_are_separated_and_exclude_failed_children(tmp_path):
    from cca.sft_export import export_trace

    (tmp_path / "session_root.jsonl").write_text(
        json.dumps(_terminal_call("root", "main", "root answer")) + "\n"
    )
    for agent_id, text in (("good", "child answer"), ("failed", "bad answer")):
        (tmp_path / f"session_root_sub_{agent_id}.jsonl").write_text(
            json.dumps(_terminal_call(agent_id, "agent", text)) + "\n"
        )
    (tmp_path / "session_root_agents.json").write_text(json.dumps({
        "agents": {
            "good": {"status": "completed", "agent_id": "good"},
            "failed": {"status": "failed", "agent_id": "failed"},
        },
    }))

    root = export_trace("root", directory=tmp_path)
    children = export_trace("root", directory=tmp_path, view="children-separated")
    combined = export_trace("root", directory=tmp_path, view="all-separated")
    assert [item["actor"] for item in root] == ["root"]
    assert [item["agent_id"] for item in children] == ["good"]
    assert [item["actor"] for item in combined] == ["root", "child"]
    assert all(
        block.get("type") != "thinking"
        for message in children[0]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
    )
    assert children[0]["quality"] == "unverified"


def test_sft_export_strips_opaque_provider_state_from_tool_calls(tmp_path):
    from cca.sft_export import export_trace

    call = _terminal_call("root", "main", "finished")
    call["messages"] = [
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "agent-batch",
                "name": "Agent",
                "input": {"tasks": [{"description": "review", "prompt": "check"}]},
                "thought_signature": "opaque-signature",
                "provider_state": {
                    "provider": "future-provider",
                    "tool_call_extra_content": {"opaque": "continuation"},
                },
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "agent-batch",
                "content": '{"status":"completed"}',
            }],
        },
    ]
    (tmp_path / "session_root.jsonl").write_text(json.dumps(call) + "\n")

    exported = export_trace("root", directory=tmp_path)[0]
    tool_use = exported["messages"][1]["content"][0]
    assert tool_use["name"] == "Agent"
    assert "thought_signature" not in tool_use
    assert "provider_state" not in tool_use


def test_sft_export_rejects_nonterminal_tool_call(tmp_path):
    from cca.sft_export import SFTValidationError, export_trace

    call = _terminal_call("root", "main", "")
    call["response"] = {
        "text": "",
        "tool_calls": [{"id": "t1", "name": "Read", "input": {}}],
    }
    (tmp_path / "session_root.jsonl").write_text(json.dumps(call) + "\n")
    with pytest.raises(SFTValidationError, match="unresolved"):
        export_trace("root", directory=tmp_path)


def test_sft_call_export_preserves_every_compaction_generation(tmp_path):
    from cca.sft_export import export_trace

    before = _terminal_call("root", "main", "before answer")
    before.update({
        "call_idx": 0,
        "session_id": "root-comp0",
        "generation": 0,
        "context_generation": 0,
        "generation_reason": "initial",
    })
    before["messages"][0]["content"][1]["text"] = "before compaction"
    after = _terminal_call("root", "main", "after answer")
    after.update({
        "call_idx": 2,
        "session_id": "root-comp1",
        "generation": 1,
        "context_generation": 1,
        "generation_reason": "compaction",
    })
    after["messages"][0]["content"][1]["text"] = "after compaction"
    records = [
        before,
        {"schema_version": 2, "type": "compactor", "call_idx": 1},
        after,
    ]
    (tmp_path / "session_root.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    calls = export_trace("root", directory=tmp_path, format="calls")
    trajectories = export_trace("root", directory=tmp_path, format="trajectories")

    assert [item["provenance"]["call_indexes"] for item in calls] == [[0], [2]]
    assert [item["context_generation"] for item in calls] == [0, 1]
    assert [item["provenance"]["call_indexes"] for item in trajectories] == [[0], [2]]
    assert trajectories[1]["provenance"]["context_from_previous_generation"] is True
    assert "before compaction" in json.dumps(calls[0]["messages"])
    assert "after compaction" in json.dumps(calls[1]["messages"])


def test_call_export_is_lossless_when_trajectory_generation_ends_with_tool_use(tmp_path):
    from cca.sft_export import SFTValidationError, export_trace
    from cca.trace_view import inspect_trace

    tool_call = _terminal_call("root", "main", "")
    tool_call.update({
        "call_idx": 0,
        "context_generation": 0,
        "generation_reason": "initial",
    })
    tool_call["response"] = {
        "text": "",
        "tool_calls": [{"id": "tool-1", "name": "Read", "input": {"file_path": "x"}}],
    }
    completed = _terminal_call("root", "main", "finished")
    completed.update({
        "call_idx": 2,
        "context_generation": 1,
        "generation_reason": "compaction",
    })
    completed["messages"] = [
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use", "id": "tool-1", "name": "Read",
                "input": {"file_path": "x"},
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result", "tool_use_id": "tool-1",
                "content": "file contents",
            }],
        },
    ]
    records = [
        tool_call,
        {"schema_version": 2, "type": "compactor", "call_idx": 1},
        completed,
    ]
    (tmp_path / "session_root.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    calls = export_trace("root", directory=tmp_path, format="calls")
    assert [item["provenance"]["call_indexes"] for item in calls] == [[0], [2]]
    assert calls[0]["messages"][-1]["content"][0]["type"] == "tool_use"
    with pytest.raises(SFTValidationError, match="Use --format calls"):
        export_trace("root", directory=tmp_path, format="trajectories")

    status = inspect_trace("root", directory=tmp_path)["root"]["sft"]
    assert status["status"] == "partial"
    assert status["call_coverage_complete"] is True
    assert status["call_examples_ready"] == 2
    assert status["trajectory_exclusions"][0]["context_generation"] == 0


def test_historical_trace_without_generation_metadata_remains_exportable(tmp_path):
    from cca.sft_export import call_example_from_record, export_trace

    (tmp_path / "session_root.jsonl").write_text(
        json.dumps(_terminal_call("root", "main", "legacy answer")) + "\n"
    )
    example = export_trace("root", directory=tmp_path, format="calls")[0]
    assert example["context_generation"] == 0
    assert example["generation_reason"] == "initial"
    assert example["provenance"]["generation_inferred"] is True
    assert example["messages"][-1]["content"][0]["text"] == "legacy answer"

    legacy_child = _terminal_call("child", "agent", "legacy child")
    legacy_child.update({"generation": 4, "parent_agent_id": "root"})
    child_example = call_example_from_record(legacy_child, actor="child")
    assert child_example["context_generation"] == 0
    assert child_example["provenance"]["generation_inferred"] is True


def test_traces_export_cli(tmp_path):
    from click.testing import CliRunner
    from cca.cli import main

    calls = tmp_path / "calls"
    calls.mkdir()
    (calls / "session_root.jsonl").write_text(
        json.dumps(_terminal_call("root", "main")) + "\n"
    )
    output = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(main, [
        "traces", "export", "root", "--calls-dir", str(calls),
        "--output", str(output),
    ])
    assert result.exit_code == 0, result.output
    assert "Exported 1 trajectory" in result.output
    assert json.loads(output.read_text())["actor"] == "root"


def test_trace_inspection_visualizes_lineage_and_sft_readiness(tmp_path):
    from click.testing import CliRunner
    from cca.cli import main
    from cca.trace_view import inspect_trace

    (tmp_path / "session_root.jsonl").write_text(
        json.dumps(_terminal_call("root", "main", "root answer")) + "\n"
    )
    (tmp_path / "session_root_sub_good.jsonl").write_text(
        json.dumps(_terminal_call("good", "agent", "child answer")) + "\n"
    )
    (tmp_path / "session_root_agents.json").write_text(json.dumps({
        "schema_version": 2,
        "trace_id": "trace",
        "root_agent_id": "root",
        "agents": {
            "good": {
                "agent_id": "good",
                "parent_agent_id": "root",
                "spawn_call_id": "spawn-good",
                "status": "completed",
                "trigger": "automatic",
                "description": "inspect good path",
                "catalog_hash": "c" * 64,
                "agent_definition": {
                    "name": "scout",
                    "definition_hash": "d" * 64,
                },
                "result": {"summary": "child answer"},
            },
            "failed": {
                "agent_id": "failed",
                "parent_agent_id": "root",
                "spawn_call_id": "spawn-failed",
                "status": "failed",
                "catalog_hash": "c" * 64,
                "agent_definition": {
                    "name": "worker",
                    "definition_hash": "e" * 64,
                },
                "result": {"summary": "failed safely"},
            },
        },
    }))
    (tmp_path / "session_root_agents.jsonl").write_text("\n".join([
        json.dumps({
            "event": "agent.run.completed", "event_id": "event-1",
            "agent_id": "good", "ts": 1,
        }),
        json.dumps({
            "event": "agent.default.changed", "event_id": "event-2",
            "agent_id": "root", "origin": "operator_command", "ts": 2,
        }),
    ]) + "\n")

    summary = inspect_trace("root", directory=tmp_path)
    assert summary["clean_export_ready"] is True
    assert summary["root"]["sft"]["status"] == "ready"
    assert summary["children"][0]["agent_id"] == "failed"
    assert summary["children"][0]["sft"]["status"] == "excluded"
    assert summary["children"][1]["agent_id"] == "good"
    assert summary["children"][1]["sft"]["status"] == "ready"
    assert summary["children"][1]["reproducible"] is True
    assert summary["control_events"][0]["event"] == "agent.default.changed"

    result = CliRunner().invoke(main, [
        "traces", "inspect", "root", "--calls-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "clean export ready" in result.output
    assert "scout" in result.output
    assert "SFT excluded" in result.output

    json_result = CliRunner().invoke(main, [
        "traces", "inspect", "root", "--calls-dir", str(tmp_path), "--json",
    ])
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["children"][1]["agent_name"] == "scout"


def test_agents_cli_lists_native_catalog(tmp_path):
    from click.testing import CliRunner
    from cca.cli import main

    result = CliRunner().invoke(main, ["agents", "list", "--cwd", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "worker" in result.output
    assert "risk-auditor" in result.output
    assert "catalog" in result.output


def test_agents_import_is_explicit_and_normalized(tmp_path):
    from click.testing import CliRunner
    from cca.cli import main

    source = tmp_path / "portable.md"
    source.write_text(
        "---\nname: specialist\ndescription: focused specialist\n"
        "tools: Read, Grep\nmaxTurns: 7\ncolor: blue\n---\nDo focused work.\n"
    )
    project = tmp_path / "project"
    project.mkdir()
    result = CliRunner().invoke(main, [
        "agents", "import", str(source), "--cwd", str(project),
    ])
    assert result.exit_code == 0, result.output
    destination = project / ".cca" / "agents" / "specialist.md"
    text = destination.read_text()
    assert "max_turns: 7" in text
    assert "color:" not in text
    assert "Do focused work." in text
    assert "unsupported field 'color'" in result.output


def test_root_session_profile_replaces_base_prompt_but_keeps_project_context(tmp_path):
    from cca.agents import AgentSpec
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    (tmp_path / "AGENTS.md").write_text("Project-specific rule.")
    profile = AgentSpec(
        name="specialist", description="special root", prompt="ROOT PROFILE PROMPT",
        tools=frozenset({"Read", "Grep"}),
    )
    config = Config()
    features = resolve_features(config)
    registry = build_registry(config=config, cwd=str(tmp_path), feature_set=features)
    repl = Repl(
        backend=None, registry=registry, config=config, permissions=_AllowPermissions(),
        cwd=str(tmp_path), feature_set=features, agent_profile=profile,
        allowed_tools=set(profile.tools),
    )
    system = repl._build_system()
    assert system.startswith("ROOT PROFILE PROMPT")
    assert "Project-specific rule." in system
    assert "# Delegation" not in system
    assert repl._runtime_settings_payload()["agent_profile"]["name"] == "specialist"


def test_repl_resume_restores_compaction_state_and_monotonic_call_index(tmp_path, monkeypatch):
    from cca import paths, sessions
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "calls")
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sessions, "HISTORY_FILE", tmp_path / "history.jsonl")
    state = {
        "trace_id": "trace-stable",
        "agent_id": "root-stable",
        "message_index_offset": 33,
        "compaction_summary": "restored compacted context",
        "token_counts": {"input": 9, "output": 4},
    }
    sessions.save_session(
        "resume-comp2", str(tmp_path),
        [Message(role="user", content=[TextContent(text="continue")])],
        state=state,
    )
    store = TraceStore("resume", directory=tmp_path / "calls")
    identity = TraceIdentity(trace_id="trace-stable", agent_id="root-stable", generation=1)
    store.append_call(identity, 6, {"type": "main"}, child=False)

    config = Config()
    features = resolve_features(config)
    registry = build_registry(config=config, cwd=str(tmp_path), feature_set=features)
    repl = Repl(
        backend=None, registry=registry, config=config, permissions=_AllowPermissions(),
        cwd=str(tmp_path), feature_set=features,
    )
    __import__("asyncio").run(repl._cmd_resume("resume-comp2"))
    assert repl._trace_id == "trace-stable"
    assert repl._agent_id == "root-stable"
    assert repl._compaction_summary == "restored compacted context"
    assert repl._msg_index_offset == 33
    assert repl._llm_call_idx == 7
    assert repl._token_counts["input"] == 9


def test_compactor_trace_keeps_full_exact_input(tmp_path, monkeypatch):
    import asyncio
    from cca import config as config_mod, paths
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cca")
    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "calls")

    class Backend:
        async def complete(self, messages, tools, system, max_tokens=8192, think_level="off"):
            return CompletionResponse(content=[TextContent(
                text="<summary>" + ("complete summary state. " * 15) + "</summary>"
            )])

    config = Config(tools={"Agent": False})
    features = resolve_features(config)
    repl = Repl(
        backend=Backend(), registry=type("Registry", (), {"_tools": {}})(),
        config=config, permissions=_AllowPermissions(), cwd=str(tmp_path),
        feature_set=features,
    )
    marker = "FULL-COMPACTOR-INPUT-" + ("x" * 15_000)
    repl.messages = [
        Message(role="user" if i % 2 == 0 else "assistant", content=[
            TextContent(text=marker if i == 0 else f"message {i}")
        ])
        for i in range(16)
    ]
    asyncio.run(repl._compact_if_needed(check_type="force"))
    records = [
        json.loads(line)
        for line in (tmp_path / "calls" / f"session_{repl._base_session_id}.jsonl")
        .read_text().splitlines()
    ]
    compactor = next(record for record in records if record["type"] == "compactor")
    logged = compactor["messages"][0]["content"][0]["text"]
    assert marker in logged
    assert "FULL-COMPACTOR-INPUT-" in logged
    boundary = next(record for record in records if record["type"] == "trace_event")
    assert boundary["reason"] == "compaction"
    assert boundary["context_generation"] == 1
    assert repl._context_generation == 1


def test_compactor_failures_fall_back_to_bounded_emergency_slice(tmp_path, monkeypatch):
    import asyncio
    from cca import config as config_mod, paths
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import EMERGENCY_KEEP_MESSAGES, Repl

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cca")
    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "calls")

    class FailingBackend:
        async def complete(self, messages, tools, system, max_tokens=8192,
                           think_level="off"):
            raise RuntimeError("compactor unavailable")

    config = Config(tools={"Agent": False})
    features = resolve_features(config)
    repl = Repl(
        backend=FailingBackend(), registry=type("Registry", (), {"_tools": {}})(),
        config=config, permissions=_AllowPermissions(), cwd=str(tmp_path),
        feature_set=features,
    )
    original = [
        Message(role="user" if i % 2 == 0 else "assistant", content=[
            TextContent(text=f"message {i}")
        ])
        for i in range(EMERGENCY_KEEP_MESSAGES + 20)
    ]
    repl.messages = list(original)

    asyncio.run(repl._compact_if_needed(check_type="force"))

    assert len(repl.messages) <= EMERGENCY_KEEP_MESSAGES
    assert repl.messages[0].role == "user"
    assert repl.messages[-1].to_dict() == original[-1].to_dict()


def test_micro_compaction_archives_tool_result_before_snipping(tmp_path, monkeypatch):
    import asyncio
    from cca import config as config_mod, paths
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cca")
    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "llm-calls")
    config = Config(compact_threshold_upper=3_000)
    features = resolve_features(config)
    repl = Repl(
        backend=None, registry=type("Registry", (), {"_tools": {}})(),
        config=config, permissions=_AllowPermissions(), cwd=str(tmp_path),
        feature_set=features,
    )
    original = "BEGIN-EXACT\n" + ("middle-content\n" * 1_500) + "END-EXACT"
    repl.messages = [Message(role="user", content=[ToolResultContent(
        tool_use_id="tool-1", content=original,
    )])]
    repl._last_input_tokens = 5_000
    asyncio.run(repl._compact_if_needed(check_type="upper"))
    assert "Content snipped" in repl.messages[0].content[0].content
    assert repl._context_generation == 1
    assert repl._generation_reason == "compaction"
    transcript = next((tmp_path / "cca" / "transcripts").glob("*.jsonl")).read_text()
    assert original in json.loads(transcript)["content"][0]["content"]
    trace_records = [
        json.loads(line)
        for line in (tmp_path / "llm-calls" / f"session_{repl._base_session_id}.jsonl")
        .read_text().splitlines()
    ]
    assert trace_records[-1]["event"] == "trace.context.started"
    assert trace_records[-1]["compaction_mode"] == "micro"


def _interactive_agents_repl(tmp_path, monkeypatch, *, config=None):
    import io
    from rich.console import Console
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    _redirect_runtime_paths(tmp_path, monkeypatch)
    config = config or Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config, cwd=str(tmp_path), feature_set=features,
        permissions=_AllowPermissions(),
    )
    repl = Repl(
        backend=None, registry=registry, config=config,
        permissions=_AllowPermissions(), cwd=str(tmp_path), feature_set=features,
    )
    output = io.StringIO()
    repl._console = Console(file=output, force_terminal=False, width=160)
    return repl, output


def test_agents_interactive_catalog_is_frozen_and_read_only(tmp_path, monkeypatch):
    import asyncio

    project_agents = tmp_path / ".cca" / "agents"
    project_agents.mkdir(parents=True)
    (project_agents / "specialist.md").write_text(
        "---\nname: specialist\ndescription: project specialist\n"
        "tools: [Read, Grep]\nmax_turns: 7\n---\nExact specialist prompt.\n"
    )
    repl, output = _interactive_agents_repl(tmp_path, monkeypatch)
    original_messages = list(repl.messages)

    assert asyncio.run(repl._handle_slash("/agents")) is True
    assert asyncio.run(repl._handle_slash("/agents show specialist")) is True
    assert asyncio.run(repl._handle_slash("/agents help")) is True
    rendered = output.getvalue()
    assert "Max turns" in rendered
    assert "worker (default)" in rendered
    assert "40" in rendered
    assert "specialist" in rendered
    assert "project" in rendered
    assert "Exact specialist prompt." in rendered
    assert "Using CCA subagents" in rendered
    assert "/agents default <name>" in rendered
    assert "39 working turns" in rendered
    assert repl.messages == original_messages
    assert not (tmp_path / "llm-calls").exists()

    # Files written after startup cannot mutate this run's action-space snapshot.
    (project_agents / "late.md").write_text(
        "---\nname: late\ndescription: too late\n---\nLate prompt.\n"
    )
    output.seek(0)
    output.truncate(0)
    asyncio.run(repl._handle_slash("/agents"))
    assert "specialist" in output.getvalue()
    assert "late" not in output.getvalue()


def test_agents_interactive_list_works_when_dispatch_is_disabled(tmp_path, monkeypatch):
    import asyncio
    from cca.config import Config

    repl, output = _interactive_agents_repl(
        tmp_path, monkeypatch, config=Config(tools={"Agent": False})
    )
    asyncio.run(repl._handle_slash("/agents"))
    assert "worker" in output.getvalue()
    assert "disabled for model use" in output.getvalue()
    assert not (tmp_path / "llm-calls").exists()


def test_clear_starts_explicit_context_generation_without_child_artifacts(
    tmp_path, monkeypatch,
):
    import asyncio
    from cca.config import Config

    repl, _ = _interactive_agents_repl(
        tmp_path, monkeypatch, config=Config(tools={"Agent": False})
    )
    repl.messages = [Message(role="user", content=[TextContent(text="old context")])]
    assert asyncio.run(repl._handle_slash("/clear")) is True
    assert repl.messages == []
    assert repl._context_generation == 1
    assert repl._generation_reason == "clear"

    repl._log_llm_call(
        call_type="main", system="sys", messages_snapshot=[], tools=[], params={},
        response_text="fresh", tool_calls=[], thinking="", stop_reason="end_turn",
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0,
    )
    root_path = tmp_path / "llm-calls" / f"session_{repl._base_session_id}.jsonl"
    records = [json.loads(line) for line in root_path.read_text().splitlines()]
    assert records[0]["event"] == "trace.context.started"
    assert records[0]["reason"] == "clear"
    assert records[1]["context_generation"] == 1
    assert records[1]["generation_reason"] == "clear"
    assert records[1]["trace_id"] == repl._trace_id
    assert records[1]["agent_id"] == repl._agent_id
    assert not list((tmp_path / "llm-calls").glob("*_agents.json*"))
    from cca.trace_view import inspect_trace
    summary = inspect_trace(repl._base_session_id, directory=tmp_path / "llm-calls")
    assert summary["root"]["context_events"][0]["reason"] == "clear"
    assert summary["root"]["sft"]["context_generations"] == [1]


def test_model_switch_rebinds_inherited_subagent_backend(tmp_path, monkeypatch):
    import asyncio
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.tools import build_registry

    class Backend:
        def __init__(self, model):
            self.model = model

    old_backend = Backend("old-model")
    new_backend = Backend("new-model")
    config = Config(model="openai:old-model")
    features = resolve_features(config)
    registry = build_registry(
        config=config, backend=old_backend, cwd=str(tmp_path),
        permissions=_AllowPermissions(), feature_set=features,
    )
    repl = Repl(
        backend=old_backend, registry=registry, config=config,
        permissions=_AllowPermissions(), cwd=str(tmp_path), feature_set=features,
    )
    monkeypatch.setattr("cca.llm.factory.create_backend", lambda *args, **kwargs: new_backend)
    monkeypatch.setattr(config, "save", lambda: None)

    assert asyncio.run(repl._handle_slash("/model openai:new-model")) is True
    agent_tool = registry.get("Agent")
    inherited_backend, selector = agent_tool._resolve_backend(
        None, agent_tool.catalog.get("worker")
    )
    assert inherited_backend is new_backend
    assert selector == "openai:new-model"


def test_agents_default_is_durable_control_state_not_model_history(tmp_path, monkeypatch):
    import asyncio
    from cca.config import Config
    from cca.features import resolve_features
    from cca.repl import Repl
    from cca.sft_export import export_trace
    from cca.tools import build_registry

    repl, output = _interactive_agents_repl(tmp_path, monkeypatch)
    original_messages = list(repl.messages)
    asyncio.run(repl._handle_slash("/agents default scout"))

    agent_tool = repl.registry.get("Agent")
    assert agent_tool.default_agent == "scout"
    assert repl.messages == original_messages
    assert repl._session_state_payload()["subagent_default"] == "scout"
    assert repl._runtime_settings_payload()["subagents"]["default"] == "scout"
    assert "session only" in output.getvalue()

    event_path = tmp_path / "llm-calls" / f"session_{repl._base_session_id}_agents.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["agent.default.changed"]
    assert events[0]["previous_default"] == "worker"
    assert events[0]["new_default"] == "scout"
    assert events[0]["definition_hash"] == agent_tool.catalog.get("scout").definition_hash
    assert not (tmp_path / "llm-calls" / f"session_{repl._base_session_id}_agents.json").exists()

    # Root SFT stays literal; the operator action is replay-only provenance.
    root_path = tmp_path / "llm-calls" / f"session_{repl._base_session_id}.jsonl"
    root_path.write_text(json.dumps(_terminal_call(repl._agent_id, "main")) + "\n")
    example = export_trace(repl._base_session_id, directory=tmp_path / "llm-calls")[0]
    assert example["messages"][1]["content"][0]["text"] == "task"
    assert all("/agents" not in str(message) for message in example["messages"])
    assert example["provenance"]["control_events"][0]["new_default"] == "scout"

    # The event log restores the session default even if an older session state
    # snapshot did not yet contain it.
    config = Config()
    features = resolve_features(config)
    registry = build_registry(config=config, cwd=str(tmp_path), feature_set=features)
    resumed = Repl(
        backend=None, registry=registry, config=config, permissions=_AllowPermissions(),
        cwd=str(tmp_path), feature_set=features, session_id=repl._session_id,
        resume_state={"trace_id": repl._trace_id, "agent_id": repl._agent_id},
    )
    assert resumed.registry.get("Agent").default_agent == "scout"


def test_agents_lifecycle_views_are_scoped_and_stop_is_control_event(tmp_path, monkeypatch):
    import asyncio
    import contextlib
    from cca.tools.agent_tool import _background_agents

    repl, output = _interactive_agents_repl(tmp_path, monkeypatch)

    async def exercise():
        cancellation = asyncio.Event()
        task = asyncio.create_task(asyncio.Event().wait())
        agent_id = "live-child"
        _background_agents[agent_id] = {
            "agent_id": agent_id,
            "root_id": repl._base_session_id,
            "trace_id": repl._trace_id,
            "agent_name": "scout",
            "description": "inspect lifecycle",
            "status": "running",
            "done": False,
            "result": None,
            "definition_hash": "definition",
            "catalog_hash": "catalog",
            "created_at": 1,
            "task": task,
            "cancellation": cancellation,
        }
        _background_agents["foreign-child"] = {
            **_background_agents[agent_id],
            "agent_id": "foreign-child",
            "root_id": "another-root",
        }
        try:
            await repl._handle_slash("/agents running")
            await repl._handle_slash(f"/agents output {agent_id}")
            await repl._handle_slash("/agents output foreign-child")
            await repl._handle_slash(f"/agents stop {agent_id}")
            await asyncio.sleep(0)
            assert cancellation.is_set()
            assert task.cancelled()
        finally:
            _background_agents.pop(agent_id, None)
            _background_agents.pop("foreign-child", None)
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    rendered = output.getvalue()
    assert "live-child" in rendered
    assert "inspect lifecycle" in rendered
    assert '"status": "running"' in rendered
    assert "foreign-child' was not found in this session" in rendered
    assert "Cancellation requested" in rendered
    assert repl.messages == []

    event_path = tmp_path / "llm-calls" / f"session_{repl._base_session_id}_agents.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["agent.stop.requested"]
    assert events[0]["target_agent_id"] == "live-child"
