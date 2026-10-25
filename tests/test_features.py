"""Tests for the feature-toggle resolver."""
import pytest

from cca.features import (
    FEATURE_DEFAULTS, OPTIONAL_TOOLS, CORE_TOOLS,
    FeatureConfigError, resolve_features,
)


class FakeConfig:
    def __init__(self, features=None, tools=None, presets=None, repo_keys=None):
        self.features = features or {}
        self.tools = tools or {}
        self.presets = presets or {}
        if repo_keys is not None:
            self._repo_toggle_keys = repo_keys


def test_defaults_all_on():
    fs = resolve_features(None)
    assert fs.is_default()
    assert fs.feature("memory") is True
    assert fs.feature("auto_approve") is False
    assert fs.tool_enabled("WebSearch") is True
    assert fs.tool_enabled("Read") is True          # core: always True
    assert fs.tool_enabled("EnterPlanMode") is True  # follows plan_mode
    assert fs.preset is None
    assert sorted(fs.enabled_tools()) == sorted(OPTIONAL_TOOLS)


def test_config_values_applied_with_sources():
    cfg = FakeConfig(features={"memory": False}, tools={"WebSearch": False},
                     repo_keys={"tools.WebSearch"})
    fs = resolve_features(cfg)
    assert fs.feature("memory") is False
    assert fs.tool_enabled("WebSearch") is False
    assert not fs.is_default()
    assert fs.sources["features.memory"] == "~/.cca/config.yaml"
    assert fs.sources["tools.WebSearch"] == ".cca/config.yaml"
    assert fs.sources["features.hooks"] == "default"


def test_unknown_feature_key_is_hard_error():
    with pytest.raises(FeatureConfigError, match="memmory"):
        resolve_features(FakeConfig(features={"memmory": False}))


def test_non_bool_value_is_hard_error():
    with pytest.raises(FeatureConfigError, match="memory"):
        resolve_features(FakeConfig(features={"memory": "no"}))


def test_core_tool_disable_is_hard_error():
    with pytest.raises(FeatureConfigError, match="core tool"):
        resolve_features(FakeConfig(tools={"Bash": False}))


def test_core_tool_true_is_allowed_noop():
    fs = resolve_features(FakeConfig(tools={"Bash": True}))
    assert fs.tool_enabled("Bash") is True


def test_plan_tool_in_tools_section_is_hard_error():
    with pytest.raises(FeatureConfigError, match="plan_mode"):
        resolve_features(FakeConfig(tools={"EnterPlanMode": False}))


def test_unknown_tool_name_is_hard_error():
    with pytest.raises(FeatureConfigError, match="WebSerch"):
        resolve_features(FakeConfig(tools={"WebSerch": False}))


def test_plan_mode_feature_controls_plan_tools():
    fs = resolve_features(FakeConfig(features={"plan_mode": False}))
    assert fs.tool_enabled("EnterPlanMode") is False
    assert fs.tool_enabled("ExitPlanMode") is False


def test_preset_replaces_config_not_merges():
    cfg = FakeConfig(features={"memory": True, "hooks": True},
                     tools={"WebSearch": True, "Agent": True})
    fs = resolve_features(cfg, preset="deterministic")
    assert fs.feature("memory") is False   # preset wins; config ignored entirely
    assert fs.tool_enabled("WebSearch") is False
    assert fs.preset == "deterministic"
    assert fs.sources["features.memory"] == "preset deterministic"


def test_deterministic_preset_golden():
    """Pin the exact preset contents so it cannot drift silently."""
    fs = resolve_features(None, preset="deterministic")
    assert {k: fs.feature(k) for k in FEATURE_DEFAULTS} == {
        "memory": False, "hooks": False, "mcp": False, "compaction": True,
        "disk_spill": False, "background_bash": True, "plan_mode": True,
        "auto_approve": True,
    }
    assert sorted(fs.enabled_tools()) == ["AskUserQuestion", "NotebookEdit"]


def test_sft_alias_matches_deterministic():
    a = resolve_features(None, preset="deterministic")
    b = resolve_features(None, preset="sft")
    assert {k: a.feature(k) for k in FEATURE_DEFAULTS} == {k: b.feature(k) for k in FEATURE_DEFAULTS}
    assert a.enabled_tools() == b.enabled_tools()


def test_user_defined_preset():
    cfg = FakeConfig(presets={"my_ablation": {"features": {"memory": False},
                                              "tools": {"WebSearch": False}}})
    fs = resolve_features(cfg, preset="my_ablation")
    assert fs.feature("memory") is False
    assert fs.feature("hooks") is True          # unlisted feature -> built-in default
    assert fs.tool_enabled("WebSearch") is False
    assert fs.tool_enabled("Agent") is True     # unlisted tool -> on


def test_user_preset_cannot_shadow_shipped_name():
    cfg = FakeConfig(presets={"deterministic": {"features": {}}})
    with pytest.raises(FeatureConfigError, match="shipped preset"):
        resolve_features(cfg, preset="deterministic")


def test_unknown_preset_is_hard_error():
    with pytest.raises(FeatureConfigError, match="determinstic"):
        resolve_features(None, preset="determinstic")


def test_preset_contents_are_validated():
    cfg = FakeConfig(presets={"bad": {"tools": {"Bash": False}}})
    with pytest.raises(FeatureConfigError, match="core tool"):
        resolve_features(cfg, preset="bad")


def test_to_payload_roundtrippable():
    fs = resolve_features(None, preset="deterministic")
    p = fs.to_payload()
    assert p["preset"] == "deterministic"
    assert p["features"]["memory"] is False
    assert "AskUserQuestion" in p["enabled_tools"]
    assert "WebSearch" not in p["enabled_tools"]


def test_summary_line_none_when_default_present_otherwise():
    assert resolve_features(None).summary_line() is None
    line = resolve_features(None, preset="deterministic").summary_line()
    assert "deterministic" in line
    cfg_line = resolve_features(FakeConfig(features={"memory": False})).summary_line()
    assert "memory" in cfg_line


def test_config_load_merged_merges_toggle_sections(tmp_path, monkeypatch):
    from cca import config as config_mod
    from cca.config import Config

    global_file = tmp_path / "global.yaml"
    global_file.write_text(
        "features:\n  memory: false\n  hooks: false\ntools:\n  WebSearch: false\n"
    )
    monkeypatch.setattr(config_mod, "CONFIG_FILE", global_file)

    repo = tmp_path / "repo"
    (repo / ".cca").mkdir(parents=True)
    (repo / ".cca" / "config.yaml").write_text(
        "features:\n  hooks: true\ntools:\n  Agent: false\n"
    )

    cfg = Config.load_merged(repo)
    assert cfg.features == {"memory": False, "hooks": True}   # per-key merge, repo wins
    assert cfg.tools == {"WebSearch": False, "Agent": False}
    assert cfg._repo_toggle_keys == {"features.hooks", "tools.Agent"}


def test_config_load_merged_end_to_end_with_resolver(tmp_path, monkeypatch):
    from cca import config as config_mod
    from cca.config import Config

    global_file = tmp_path / "global.yaml"
    global_file.write_text("features:\n  memory: false\n")
    monkeypatch.setattr(config_mod, "CONFIG_FILE", global_file)

    cfg = Config.load_merged(tmp_path)
    fs = resolve_features(cfg)
    assert fs.feature("memory") is False
    assert fs.sources["features.memory"] == "~/.cca/config.yaml"


def test_bash_tool_background_disabled_schema():
    from cca.tools.bash_tool import BashTool
    t = BashTool(allow_background=False)
    assert "run_in_background" not in t.input_schema["properties"]
    assert "run_in_background" not in t.description
    default = BashTool()
    assert "run_in_background" in default.input_schema["properties"]


def test_bash_tool_background_disabled_execute():
    import asyncio
    from cca.tools.bash_tool import BashTool
    t = BashTool(allow_background=False)
    result = asyncio.run(t.execute(command="echo hi", run_in_background=True))
    assert "disabled" in result.lower()


def _registry_names(feature_set):
    from cca.tools import build_registry
    reg = build_registry(config=None, feature_set=feature_set)
    return set(reg._tools)


def test_registry_default_contains_optional_tools():
    names = _registry_names(resolve_features(None))
    assert {"WebSearch", "Agent", "Monitor", "EnterWorktree", "WorkCreate",
            "EnterPlanMode", "ExitPlanMode", "ListMcpResourcesTool"} <= names


def test_public_tool_count_matches_builtin_catalog():
    from pathlib import Path
    from cca.features import BUILTIN_TOOL_COUNT, BUILTIN_TOOLS

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    blog = (root / "website/src/content/docs/blog/introducing-cca.md").read_text()

    assert BUILTIN_TOOL_COUNT == len(BUILTIN_TOOLS) == 33
    assert f"built--in%20tools-{BUILTIN_TOOL_COUNT}" in readme
    claim = f"{BUILTIN_TOOL_COUNT} available built-in tools"
    assert claim in readme
    assert claim in blog


def test_builtin_catalog_matches_fully_enabled_registry(monkeypatch):
    import cca.tools as tools_module
    from cca.config import Config
    from cca.features import BUILTIN_TOOLS

    config = Config(skills_enabled=True)
    monkeypatch.setattr(
        tools_module,
        "DETECTED_SERVERS",
        [{"language": "test", "command": "unused", "args": []}],
    )
    registry = tools_module.build_registry(
        config=config,
        feature_set=resolve_features(config),
    )

    assert set(registry._tools) == BUILTIN_TOOLS


def test_registry_deterministic_exact_action_space():
    names = _registry_names(resolve_features(None, preset="deterministic"))
    assert names == {
        "Read", "Write", "Edit", "Glob", "Grep", "SearchInFile", "Bash",
        "NotebookEdit", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode",
    }


def test_registry_deterministic_bash_keeps_background():
    from cca.tools import build_registry
    reg = build_registry(config=None, feature_set=resolve_features(None, preset="deterministic"))
    bash = reg.get("Bash")
    assert "run_in_background" in bash.input_schema["properties"]


def test_registry_bash_background_off_strips_parameter():
    from cca.tools import build_registry
    fs = resolve_features(FakeConfig(features={"background_bash": False}))
    bash = build_registry(config=None, feature_set=fs).get("Bash")
    assert "run_in_background" not in bash.input_schema["properties"]


def test_registry_mcp_feature_off_removes_static_mcp_tools():
    fs = resolve_features(FakeConfig(features={"mcp": False}))
    names = _registry_names(fs)
    assert not names & {"ListMcpResourcesTool", "ReadMcpResourceTool", "McpAuthTool"}


def test_registry_plan_mode_off_removes_plan_tools():
    fs = resolve_features(FakeConfig(features={"plan_mode": False}))
    names = _registry_names(fs)
    assert not names & {"EnterPlanMode", "ExitPlanMode"}


def test_save_session_writes_features_payload(tmp_path, monkeypatch):
    from cca import sessions
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sessions, "HISTORY_FILE", tmp_path / "history.jsonl")
    fs = resolve_features(None, preset="deterministic")
    sessions.save_session("feat-test", str(tmp_path), [], features=fs.to_payload())
    import json
    payload = json.loads((tmp_path / "feat-test.json").read_text())
    assert payload["features"]["preset"] == "deterministic"
    assert "enabled_tools" in payload["features"]


def test_repl_feature_set_gates_hooks(tmp_path):
    from cca.repl import Repl
    from cca.config import Config
    cfg = Config()
    cfg.hooks = {"session_start": [{"command": "echo hi"}]}
    fs = resolve_features(FakeConfig(features={"hooks": False}))
    repl = Repl(backend=None, registry=None, config=cfg, permissions=None,
                cwd=str(tmp_path), feature_set=fs)
    assert repl._hooks._hooks == {}
    assert repl._features.feature("hooks") is False


def test_cli_rejects_unknown_preset():
    from click.testing import CliRunner
    from cca.cli import main
    result = CliRunner().invoke(main, ["--preset", "nope", "--print", "hi"])
    assert result.exit_code == 1
    assert "Unknown preset" in result.output


def test_cca_features_command_shows_sources():
    from click.testing import CliRunner
    from cca.cli import main
    result = CliRunner().invoke(main, ["features", "--preset", "deterministic"])
    assert result.exit_code == 0
    assert "memory" in result.output
    assert "preset deterministic" in result.output
    assert "WebSearch" in result.output


def test_cca_features_command_default():
    from click.testing import CliRunner
    from cca.cli import main
    result = CliRunner().invoke(main, ["features"])
    assert result.exit_code == 0
    assert "default" in result.output


# ---------------------------------------------------------------------------
# Trace provenance: every persisted record carries the feature payload
# ---------------------------------------------------------------------------

def _make_repl(tmp_path, preset=None):
    from cca.repl import Repl
    from cca.config import Config
    fs = resolve_features(None, preset=preset)
    return Repl(backend=None, registry=None, config=Config(), permissions=None,
                cwd=str(tmp_path), feature_set=fs)


def test_llm_call_log_records_features(tmp_path, monkeypatch):
    import json
    from cca import paths
    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "llm-calls")

    repl = _make_repl(tmp_path, preset="deterministic")
    repl._log_llm_call(
        call_type="main", system="sys", messages_snapshot=[], tools=[],
        params={}, response_text="hi", tool_calls=[], thinking="",
        stop_reason="end_turn", input_tokens=1, output_tokens=1,
        cache_read_tokens=0, cache_write_tokens=0,
    )

    log_file = tmp_path / "llm-calls" / f"session_{repl._base_session_id}.jsonl"
    record = json.loads(log_file.read_text().splitlines()[0])
    assert record["features"]["preset"] == "deterministic"
    assert record["features"]["features"]["memory"] is False
    assert "enabled_tools" in record["features"]


def test_pre_compaction_save_records_features(tmp_path, monkeypatch):
    import asyncio
    from cca import config as config_mod, paths, sessions
    from cca.llm.base import (
        CompletionResponse, Message, TextContent,
    )
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cca")
    monkeypatch.setattr(paths, "LLM_CALLS_DIR", tmp_path / "llm-calls")

    saved_calls = []
    monkeypatch.setattr(
        sessions, "save_session",
        lambda *a, **kw: saved_calls.append(kw),
    )

    class FakeBackend:
        async def complete(self, messages, tools, system, max_tokens=8192,
                           think_level="off"):
            summary = "<summary>" + "compacted conversation state. " * 10 + "</summary>"
            return CompletionResponse(content=[TextContent(text=summary)])

    import types
    repl = _make_repl(tmp_path, preset="deterministic")
    repl.backend = FakeBackend()
    repl.registry = types.SimpleNamespace(_tools={})
    for i in range(16):
        role = "user" if i % 2 == 0 else "assistant"
        repl.messages.append(
            Message(role=role, content=[TextContent(text=f"message {i}")])
        )

    asyncio.run(repl._compact_if_needed(check_type="force"))

    assert saved_calls, "compaction never saved the pre-compaction session"
    assert saved_calls[0].get("features", {}).get("preset") == "deterministic"


def test_subagent_session_records_features(tmp_path, monkeypatch):
    import asyncio
    from cca import sessions
    from cca.config import Config
    from cca.tools import build_registry
    from cca.llm.base import (
        CompletionResponse, TextContent, ToolUseContent,
    )

    saved_calls = []
    monkeypatch.setattr(
        sessions, "save_session",
        lambda *a, **kw: saved_calls.append(kw),
    )

    class FakeBackend:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, tools, system, max_tokens=8192,
                           think_level="off"):
            self.calls += 1
            if self.calls == 1:
                return CompletionResponse(content=[
                    ToolUseContent(id="t1", name="NoSuchTool", input={}),
                ])
            return CompletionResponse(content=[TextContent(text="done")])

    fs = resolve_features(None)
    registry = build_registry(config=Config(), backend=FakeBackend(),
                              feature_set=fs)
    agent_tool = registry.get("Agent")
    agent_tool._backend = FakeBackend()

    asyncio.run(agent_tool.execute(description="test", prompt="do a thing"))

    assert saved_calls, "subagent never saved its session"
    assert "features" in saved_calls[0]
    assert saved_calls[0]["features"]["features"] == dict(FEATURE_DEFAULTS)
