"""Contract and root-lifecycle tests for optional observability."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from cca.config import Config
from cca.llm.base import CompletionResponse, StreamEvent, TextContent, ToolUseContent
from cca.observability import (
    InMemoryExporter,
    ObservabilityConfigError,
    ObservabilitySettings,
    Observer,
    resolve_observability,
)
from cca.permissions import PermissionManager
from cca.repl import Repl


def _observer(*, capture: str = "metadata") -> tuple[Observer, InMemoryExporter]:
    exporter = InMemoryExporter()
    observer = Observer(
        ObservabilitySettings(enabled=True, destination="otlp", capture=capture),
        [exporter],
        common_attributes={"cca.trace_id": "trace-1"},
    )
    return observer, exporter


def test_disabled_resolution_is_a_true_noop(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    settings = resolve_observability({})
    observer = Observer(settings)

    with observer.span("ignored", kind="turn"):
        pass

    assert settings.enabled is False
    assert observer.enabled is False
    assert observer.export_errors == 0


def test_nested_spans_keep_explicit_parent_relationships():
    observer, exporter = _observer()

    with observer.span("session", kind="session") as session:
        with observer.span("turn", kind="turn") as turn:
            with observer.span("tool", kind="tool") as tool:
                pass

    assert exporter.started[0]["span_id"] == session.span_id
    assert exporter.started[1]["parent_span_id"] == session.span_id
    assert exporter.started[2]["parent_span_id"] == turn.span_id
    assert tool.parent_span_id == turn.span_id
    assert [event["status"] for event in exporter.finished] == ["ok", "ok", "ok"]


def test_content_capture_redacts_secrets_and_is_bounded():
    observer, exporter = _observer(capture="content")
    observer.settings = ObservabilitySettings(
        enabled=True,
        destination="otlp",
        capture="content",
        max_content_chars=100,
    )

    with observer.span(
        "tool",
        kind="tool",
        content={
            "input.value": {
                "api_key": "do-not-send",
                "command": "token=also-secret " + "x" * 200,
            }
        },
    ):
        pass

    captured = exporter.started[0]["attributes"]["input.value"]
    assert "do-not-send" not in captured
    assert "also-secret" not in captured
    assert "[REDACTED]" in captured
    assert "truncated" in captured


def test_oversized_content_stays_parseable_after_truncation():
    """Structural elision keeps payloads valid JSON so backends render a tree."""
    observer, exporter = _observer(capture="content")
    observer.settings = ObservabilitySettings(
        enabled=True, destination="otlp", capture="content", max_content_chars=4_000
    )
    messages = [{"role": "user", "content": "x" * 400} for _ in range(300)]

    with observer.span("llm", kind="llm", content={"input.value": {"messages": messages}}):
        pass

    captured = exporter.started[0]["attributes"]["input.value"]
    assert len(captured) < 10_000
    parsed = json.loads(captured)
    assert 0 < len(parsed["messages"]) < 300
    assert any("truncated" in str(item) for item in parsed["messages"])


def test_content_values_carry_mime_types_and_strings_are_not_double_encoded():
    observer, exporter = _observer(capture="content")

    with observer.span("turn", kind="turn", content={"input.value": "do the thing"}):
        pass
    with observer.span("llm", kind="llm", content={"input.value": {"messages": []}}):
        pass

    turn, llm = exporter.started[0]["attributes"], exporter.started[1]["attributes"]
    assert turn["input.value"] == "do the thing"
    assert turn["input.mime_type"] == "text/plain"
    assert llm["input.mime_type"] == "application/json"


def test_metadata_capture_emits_no_content_or_mime_attributes():
    observer, exporter = _observer(capture="metadata")

    with observer.span("turn", kind="turn", content={"input.value": "secret prompt"}):
        pass

    attributes = exporter.started[0]["attributes"]
    assert "input.value" not in attributes
    assert "input.mime_type" not in attributes


def test_inline_secret_redaction_consumes_the_auth_scheme():
    observer, exporter = _observer(capture="content")

    with observer.span(
        "tool", kind="tool", content={"input.value": "authorization: Bearer leak-me-123"}
    ):
        pass

    captured = exporter.started[0]["attributes"]["input.value"]
    assert "leak-me-123" not in captured
    assert "[REDACTED]" in captured


def test_model_attributes_split_provider_from_bare_model_id():
    from cca.observability import model_attributes, response_model_attributes

    attributes = model_attributes("anthropic:claude-opus-4-6")
    assert attributes["gen_ai.request.model"] == "claude-opus-4-6"
    assert attributes["gen_ai.system"] == "anthropic"
    assert attributes["gen_ai.provider.name"] == "anthropic"
    assert attributes["cca.model_selector"] == "anthropic:claude-opus-4-6"
    assert response_model_attributes("anthropic:claude-opus-4-6") == {
        "gen_ai.response.model": "claude-opus-4-6"
    }
    assert model_attributes(None) == {}


def test_capture_override_beats_configured_capture(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "top-secret")
    monkeypatch.setenv("WANDB_ENTITY", "team")

    settings = resolve_observability(
        {"enabled": True, "destination": "wandb", "capture": "metadata"},
        capture_override="content",
    )

    assert settings.capture == "content"


def test_wandb_resolution_uses_environment_without_exposing_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "top-secret")
    monkeypatch.setenv("WANDB_ENTITY", "team")

    settings = resolve_observability(
        {"enabled": True, "destination": "wandb", "project": "cca-tests"}
    )

    assert settings.endpoint == "https://trace.wandb.ai/otel/v1/traces"
    assert settings.headers == {"wandb-api-key": "top-secret"}
    assert "top-secret" not in str(settings.safe_payload())
    assert settings.safe_payload()["authentication"] == "env:WANDB_API_KEY"

    exact = resolve_observability({
        "enabled": True,
        "destination": "wandb",
        "endpoint": "https://self-hosted.example/v1/traces",
    })
    assert exact.endpoint == "https://self-hosted.example/v1/traces"


def test_enabled_wandb_requires_explicit_auth_and_entity(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    with pytest.raises(ObservabilityConfigError, match="WANDB_API_KEY"):
        resolve_observability({"enabled": True, "destination": "wandb"})


def test_repo_observability_config_merges_over_global(tmp_path, monkeypatch):
    from cca import config as config_module

    global_file = tmp_path / "global.yaml"
    global_file.write_text(
        "observability:\n  enabled: true\n  destination: otlp\n"
        "  endpoint: http://collector:4318/v1/traces\n"
    )
    project = tmp_path / "repo"
    (project / ".cca").mkdir(parents=True)
    (project / ".cca" / "config.yaml").write_text(
        "observability:\n  capture: content\n"
    )
    monkeypatch.setattr(config_module, "CONFIG_FILE", global_file)

    config = Config.load_merged(project)

    assert config.observability == {
        "enabled": True,
        "destination": "otlp",
        "endpoint": "http://collector:4318/v1/traces",
        "capture": "content",
    }


class _Backend:
    async def stream(self, **kwargs):
        yield StreamEvent(type="text_delta", data="done")
        yield StreamEvent(
            type="message_end",
            data={
                "input_tokens": 11,
                "output_tokens": 2,
                "cache_read_tokens": 3,
                "cache_write_tokens": 1,
                "stop_reason": "end_turn",
            },
        )


class _Registry:
    _mcp_manager = None

    def __init__(self):
        self._tools = {}

    def get_schemas(self, **kwargs):
        return []

    def get(self, name):
        return self._tools.get(name)

    async def execute(self, name, tool_input):
        await asyncio.sleep(tool_input.get("delay", 0))
        return f"{name}-ok"


async def test_print_mode_emits_session_turn_and_model_lifecycle(tmp_path, monkeypatch):
    observer, exporter = _observer()
    cfg = Config(features={"memory": False, "compaction": False})
    registry = _Registry()
    monkeypatch.setattr("cca.sessions.save_session", lambda *args, **kwargs: None)
    repl = Repl(
        backend=_Backend(),
        registry=registry,
        config=cfg,
        permissions=PermissionManager(auto_approve=True, cwd=str(tmp_path)),
        cwd=str(tmp_path),
        print_mode=True,
        run_mode="print",
        observer=observer,
        observability_settings=observer.settings,
    )

    result = await repl.run_print("hello")

    assert result == "done"
    by_kind = {record["kind"]: record for record in exporter.started}
    assert {"session", "turn", "llm"} <= set(by_kind)
    assert by_kind["turn"]["parent_span_id"] == by_kind["session"]["span_id"]
    assert by_kind["llm"]["parent_span_id"] == by_kind["turn"]["span_id"]
    llm_end = next(
        event for event in exporter.finished
        if event["span_id"] == by_kind["llm"]["span_id"]
    )
    assert llm_end["attributes"]["gen_ai.usage.output_tokens"] == 2
    assert exporter.closed is True


async def test_machine_observability_preserves_json_lines_stdout(tmp_path, monkeypatch, capsys):
    observer, exporter = _observer()
    repl = Repl(
        backend=_Backend(),
        registry=_Registry(),
        config=Config(features={"memory": False, "compaction": False}),
        permissions=PermissionManager(auto_approve=True, cwd=str(tmp_path)),
        cwd=str(tmp_path),
        run_mode="machine",
        observer=observer,
        observability_settings=observer.settings,
    )
    lines = iter(['{"type":"shutdown"}\n'])

    async def fake_readline():
        return next(lines, "")

    monkeypatch.setattr(repl, "_machine_readline", fake_readline)
    await repl.run_machine()

    stdout = capsys.readouterr().out.splitlines()
    assert stdout
    assert all(isinstance(json.loads(line), dict) for line in stdout)
    assert exporter.closed is True


async def test_parallel_tool_spans_share_turn_parent(tmp_path):
    observer, exporter = _observer()
    cfg = Config(features={"memory": False, "compaction": False})
    repl = Repl(
        backend=None,
        registry=_Registry(),
        config=cfg,
        permissions=PermissionManager(auto_approve=True, cwd=str(tmp_path)),
        cwd=str(tmp_path),
        observer=observer,
        observability_settings=observer.settings,
    )
    calls = [
        ToolUseContent(id="a", name="Read", input={"delay": 0.01}),
        ToolUseContent(id="b", name="Grep", input={"delay": 0}),
    ]

    with observer.span("turn", kind="turn") as turn:
        results = await repl._execute_tools(calls)

    tools = [record for record in exporter.started if record["kind"] == "tool"]
    assert len(tools) == 2
    assert {record["parent_span_id"] for record in tools} == {turn.span_id}
    assert [result.content for result in results] == ["Read-ok", "Grep-ok"]


async def test_subagent_run_links_agent_tool_to_child_model(tmp_path, monkeypatch):
    from cca import paths, sessions
    from cca.features import resolve_features
    from cca.tools import build_registry

    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "llm-calls")
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sessions, "HISTORY_FILE", tmp_path / "history.jsonl")

    class ChildBackend:
        async def complete(self, **kwargs):
            return CompletionResponse(
                content=[TextContent(text="child completed")],
                usage={"input_tokens": 7, "output_tokens": 3},
            )

    observer, exporter = _observer()
    config = Config()
    features = resolve_features(config)
    registry = build_registry(
        config=config,
        backend=ChildBackend(),
        cwd=str(tmp_path),
        permissions=PermissionManager(auto_approve=True, cwd=str(tmp_path)),
        feature_set=features,
    )
    agent_tool = registry.get("Agent")
    agent_tool.bind_parent(
        session_id="root-comp0",
        root_id="root",
        trace_id="trace-1",
        root_agent_id="root",
        thinking_request="off",
        permissions=PermissionManager(auto_approve=True, cwd=str(tmp_path)),
        cwd=str(tmp_path),
        observer=observer,
    )

    with observer.span("cca.tool.Agent", kind="tool") as tool_span:
        result = await agent_tool.execute(
            description="inspect",
            prompt="Inspect the project.",
            agent="scout",
        )

    assert "child completed" in result
    child = next(record for record in exporter.started if record["kind"] == "agent")
    model = next(record for record in exporter.started if record["kind"] == "llm")
    assert child["parent_span_id"] == tool_span.span_id
    assert model["parent_span_id"] == child["span_id"]
    assert child["attributes"]["cca.agent.background"] is False


def test_exporter_failures_never_escape_agent_code():
    class BrokenExporter:
        def start_span(self, **kwargs):
            raise RuntimeError("offline")

        def finish_span(self, token, **kwargs):
            raise AssertionError("no token should exist")

        def force_flush(self, timeout):
            raise RuntimeError("offline")

        def shutdown(self, timeout):
            raise RuntimeError("offline")

    settings = ObservabilitySettings(enabled=True, destination="otlp")
    observer = Observer(settings, [BrokenExporter()])

    with observer.span("turn", kind="turn"):
        pass
    observer.force_flush()
    observer.shutdown()

    assert observer.export_errors == 3


def test_optional_otel_adapter_constructs_when_installed():
    pytest.importorskip("opentelemetry.sdk")
    from cca.observability import create_observer

    observer = create_observer(ObservabilitySettings(
        enabled=True,
        destination="otlp",
        endpoint="http://127.0.0.1:9/v1/traces",
        sample_rate=0.0,
    ))
    with observer.span("cca.session", kind="session"):
        with observer.span("cca.turn", kind="turn"):
            pass
    observer.shutdown(0.1)
