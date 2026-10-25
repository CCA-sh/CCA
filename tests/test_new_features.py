"""Unit tests for features added in the new-features batch.

Covers (no API keys required — all external calls are mocked):
  - model_caps: per-model max_tokens table
  - repl helpers: _thinking_budget, _parse_token_budget
  - config: vim field persistence
  - partial compaction direction flags (parsing, not the full LLM call)
  - MCP tools: config loading, McpServerManager, DynamicMcpTool, static tools, McpAuthTool
  - LSP tool: server detection, tool properties
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# model_caps
# ---------------------------------------------------------------------------

from cca.llm.model_caps import (
    get_effective_max_output_tokens,
    get_model_max_output_tokens,
)


class TestModelCaps:
    def test_sonnet_4_6(self):
        default, upper = get_model_max_output_tokens("anthropic:claude-sonnet-4-6")
        assert default == 32_000
        assert upper == 128_000

    def test_opus_4_6(self):
        default, upper = get_model_max_output_tokens("anthropic:claude-opus-4-6")
        assert default == 64_000
        assert upper == 128_000

    def test_haiku_4(self):
        default, upper = get_model_max_output_tokens("anthropic:claude-haiku-4-5-20251001")
        assert default == 32_000
        assert upper == 64_000

    def test_claude_3_5_sonnet(self):
        default, upper = get_model_max_output_tokens("claude-3-5-sonnet-20241022")
        assert default == 8_192
        assert upper == 8_192

    def test_claude_3_opus(self):
        default, upper = get_model_max_output_tokens("claude-3-opus-20240229")
        assert default == 4_096
        assert upper == 4_096

    def test_gpt_4o(self):
        default, upper = get_model_max_output_tokens("openai:gpt-4o")
        assert default == 16_384

    def test_gemini(self):
        default, upper = get_model_max_output_tokens("google:gemini-2.5-pro")
        assert default == 8_192
        assert upper == 65_536

    def test_unknown_falls_back(self):
        default, upper = get_model_max_output_tokens("some:unknown-model-xyz")
        assert default > 0
        assert upper >= default

    def test_provider_prefix_stripped(self):
        a = get_model_max_output_tokens("anthropic:claude-sonnet-4-6")
        b = get_model_max_output_tokens("claude-sonnet-4-6")
        assert a == b

    def test_litellm_groq_qwen_38_uses_provider_limit(self):
        assert get_model_max_output_tokens(
            "litellm:groq/qwen/qwen3.8-27b"
        ) == (16_384, 16_384)

    @pytest.mark.parametrize(
        "model",
        ["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    )
    def test_litellm_gemini_gemma_4_uses_provider_limit(self, model):
        assert get_model_max_output_tokens(
            f"litellm:gemini/{model}"
        ) == (32_000, 32_768)

    def test_configured_tokens_are_clamped_to_model_limit(self):
        assert get_effective_max_output_tokens(
            "litellm:groq/qwen/qwen3.8-27b", configured=32_768
        ) == 16_384

    def test_default_sentinel_uses_model_default(self):
        assert get_effective_max_output_tokens(
            "openai:gpt-4o", configured=8_192
        ) == 16_384

    def test_tiny_override_is_clamped_to_floor(self):
        # Mirrors the observed incident: a mis-parsed inline token budget
        # (e.g. "+82" lifted from a diff hunk header) must never reach the
        # backend as the actual max_tokens for the call.
        assert get_effective_max_output_tokens(
            "litellm:fireworks_ai/minimax-m3", configured=8_192, override=82
        ) == 1_024

    def test_override_of_one_is_clamped_to_floor(self):
        assert get_effective_max_output_tokens(
            "litellm:anthropic/claude-sonnet-5", configured=8_192, override=1
        ) == 1_024

    def test_floor_never_exceeds_model_upper_limit(self):
        # A model whose own upper limit sits below the floor should clamp
        # to that limit, not inflate past what the model actually supports.
        assert get_effective_max_output_tokens(
            "some:tiny-model-xyz", configured=8_192, override=1
        ) <= get_model_max_output_tokens("some:tiny-model-xyz")[1]


# ---------------------------------------------------------------------------
# _thinking_budget and _parse_token_budget (module-level helpers in repl.py)
# ---------------------------------------------------------------------------

from cca.repl import (
    _keyword_think_level,
    _thinking_budget,
    _parse_token_budget,
    _parse_budget_amount,
)


class TestKeywordThinkLevel:
    def test_no_keyword_returns_none(self):
        assert _keyword_think_level("fix the bug") is None

    def test_ultrathink_keyword(self):
        assert _keyword_think_level("ultrathink about this") == "high"

    def test_think_harder_keyword(self):
        assert _keyword_think_level("think harder about this problem") == "high"

    def test_think_step_by_step(self):
        assert _keyword_think_level("think step by step please") == "high"

    def test_case_insensitive(self):
        assert _keyword_think_level("ULTRATHINK!") == "high"

    def test_think_it_through(self):
        assert _keyword_think_level("think it through") == "high"


class TestThinkingBudget:
    """Legacy wrapper — kept to ensure backward compat of the function signature."""

    def test_no_keyword_returns_zero(self):
        budget, mode = _thinking_budget("fix the bug")
        assert budget == 0
        assert mode == "enabled"

    def test_ultrathink_keyword(self):
        budget, mode = _thinking_budget("ultrathink about this")
        assert budget > 0
        assert mode == "adaptive"

    def test_think_harder_keyword(self):
        budget, mode = _thinking_budget("think harder about this problem")
        assert budget > 0
        assert mode == "adaptive"

    def test_think_step_by_step(self):
        budget, mode = _thinking_budget("think step by step please")
        assert mode == "adaptive"

    def test_case_insensitive(self):
        budget, mode = _thinking_budget("ULTRATHINK!")
        assert mode == "adaptive"


class TestParseBudgetAmount:
    """Bare-quantity parser used by the /budget slash command and the
    inline /budget <amount> trigger's captured group — no unit word,
    no floor, since the caller already knows this text is a budget amount."""

    def test_k_suffix(self):
        assert _parse_budget_amount("500k") == 500_000

    def test_m_suffix(self):
        assert _parse_budget_amount("2.5m") == 2_500_000

    def test_plain_integer(self):
        assert _parse_budget_amount("200000") == 200_000

    def test_case_insensitive_suffix(self):
        assert _parse_budget_amount("500K") == 500_000
        assert _parse_budget_amount("2M") == 2_000_000

    def test_not_a_number_returns_none(self):
        assert _parse_budget_amount("banana") is None
        assert _parse_budget_amount("") is None


class TestParseTokenBudget:
    """Inline trigger: only the literal `/budget <amount>` sigil, anywhere
    in the message — never a bare number or natural-language phrase."""

    def test_slash_budget_alone(self):
        assert _parse_token_budget("/budget 500k") == 500_000

    def test_slash_budget_with_decimal_and_m(self):
        assert _parse_token_budget("/budget 2.5m") == 2_500_000

    def test_slash_budget_plain_integer(self):
        assert _parse_token_budget("/budget 200000") == 200_000

    def test_slash_budget_embedded_at_start_of_longer_message(self):
        assert _parse_token_budget("/budget 500k now explain this codebase") == 500_000

    def test_slash_budget_embedded_at_end_of_longer_message(self):
        assert _parse_token_budget("explain this codebase /budget 500k") == 500_000

    def test_slash_budget_embedded_mid_message(self):
        assert _parse_token_budget(
            "..... /budget 500k this triggers?"
        ) == 500_000

    def test_no_match_returns_none(self):
        assert _parse_token_budget("no budget here") is None

    def test_natural_language_no_longer_triggers(self):
        # The old "use N tokens" / bare "N tokens" phrasing is removed —
        # /budget is the only trigger now.
        assert _parse_token_budget("use 500k tokens") is None
        assert _parse_token_budget("500k tokens") is None

    def test_bare_plus_number_is_rejected(self):
        # Previously accepted anywhere in the message — exactly what a diff
        # hunk header ("@@ -6,3 +6,5 @@") or a new-file header ("+++
        # b/path") would trip.
        assert _parse_token_budget("+200000") is None
        assert _parse_token_budget("+500k") is None
        assert _parse_token_budget("+2m") is None

    def test_bare_slash_word_without_budget_is_rejected(self):
        # Must be the literal word "budget", not just any "/word <number>".
        assert _parse_token_budget("/model 500k") is None
        assert _parse_token_budget("/limit 500k") is None

    def test_below_floor_is_rejected(self):
        assert _parse_token_budget("/budget 500") is None

    def test_real_diff_hunk_header_does_not_trigger(self):
        diff_text = (
            "diff --git a/.gitignore b/.gitignore\n"
            "index 15ee9eb..2b3d0c1 100644\n"
            "--- a/.gitignore\n"
            "+++ b/.gitignore\n"
            "@@ -6,3 +6,5 @@ __pycache__/\n"
            " *.egg-info/\n"
            " node_modules/\n"
            " .DS_Store\n"
            "+outputs/\n"
            "+data/*.pt\n"
        )
        assert _parse_token_budget(diff_text) is None

    def test_new_file_diff_header_does_not_trigger(self):
        diff_text = (
            "--- /dev/null\n"
            "+++ b/configs/train_config.yaml\n"
            "@@ -0,0 +1,20 @@\n"
            "+training:\n"
            "+  iterations: 5000\n"
        )
        assert _parse_token_budget(diff_text) is None

    def test_diff_with_small_hunk_numbers_does_not_trigger(self):
        # Regression for the observed incident: a hunk header like
        # "@@ -82,30 +82,27 @@" used to parse as a "+82" budget request.
        diff_text = (
            "--- a/multishot_video_dit/data/dataset_builder.py\n"
            "+++ b/multishot_video_dit/data/dataset_builder.py\n"
            "@@ -82,30 +82,27 @@ class MultiShotDatasetBuilder:\n"
            "     def build(self):\n"
            "+        pass\n"
        )
        assert _parse_token_budget(diff_text) is None


# ---------------------------------------------------------------------------
# Config: vim field
# ---------------------------------------------------------------------------

from cca.config import Config


class TestConfigVim:
    def test_vim_defaults_false(self):
        cfg = Config()
        assert cfg.vim is False

    def test_vim_roundtrip(self, tmp_path):
        cfg = Config(vim=True)
        cfg_file = tmp_path / "config.yaml"
        import yaml
        with open(cfg_file, "w") as f:
            yaml.dump(cfg.to_dict(), f)
        with open(cfg_file) as f:
            data = yaml.safe_load(f)
        loaded = Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})
        assert loaded.vim is True

    def test_vim_merge(self, tmp_path):
        import yaml
        global_cfg = Config(vim=False)
        repo_file = tmp_path / ".cca" / "config.yaml"
        repo_file.parent.mkdir(parents=True)
        with open(repo_file, "w") as f:
            yaml.dump({"vim": True}, f)
        # Simulate load_merged
        cfg = Config()
        with open(repo_file) as f:
            repo_data = yaml.safe_load(f)
        for k, v in repo_data.items():
            if k in Config.__dataclass_fields__:
                setattr(cfg, k, v)
        assert cfg.vim is True


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

from cca.tools.mcp_tools import (
    _load_mcp_config,
    McpServerManager,
    DynamicMcpTool,
    ListMcpResourcesTool,
    ReadMcpResourceTool,
    McpAuthTool,
    _load_tokens,
    _save_tokens,
)


class TestLoadMcpConfig:
    def test_returns_empty_when_no_file(self, tmp_path):
        result = _load_mcp_config(str(tmp_path))
        assert result == {}

    def test_reads_mcp_json(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "my-server": {"type": "stdio", "command": "my-server"}
            }
        }))
        result = _load_mcp_config(str(tmp_path))
        assert "my-server" in result
        assert result["my-server"]["command"] == "my-server"

    def test_walks_up_to_find_mcp_json(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {"s": {"type": "stdio", "command": "s"}}}))
        subdir = tmp_path / "sub" / "dir"
        subdir.mkdir(parents=True)
        result = _load_mcp_config(str(subdir))
        assert "s" in result

    def test_ignores_malformed_json(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text("not json!!!")
        result = _load_mcp_config(str(tmp_path))
        assert result == {}

    def test_returns_empty_when_no_mcpservers_key(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({"other": {}}))
        result = _load_mcp_config(str(tmp_path))
        assert result == {}


class TestMcpServerManager:
    def _make_mock_session(self, tools=None, resources=None):
        """Create a mock MCP ClientSession."""
        session = AsyncMock()
        tool_list = MagicMock()
        tool_list.tools = tools or []
        session.list_tools.return_value = tool_list
        resource_list = MagicMock()
        resource_list.resources = resources or []
        session.list_resources.return_value = resource_list
        session._exit_stack = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_start_returns_empty_without_config(self, tmp_path):
        mgr = McpServerManager(cwd=str(tmp_path))
        wrappers = await mgr.start()
        assert wrappers == []

    @pytest.mark.asyncio
    async def test_start_registers_tools(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {"srv": {"type": "stdio", "command": "fake-server"}}
        }))

        mock_tool = MagicMock()
        mock_tool.name = "do_thing"
        mock_tool.description = "Does a thing"
        mock_tool.inputSchema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}

        session = self._make_mock_session(tools=[mock_tool])

        with patch("cca.tools.mcp_tools._connect", new=AsyncMock(return_value=session)):
            mgr = McpServerManager(cwd=str(tmp_path), trust_all=True)
            wrappers = await mgr.start()

        assert len(wrappers) == 1
        assert wrappers[0].name == "mcp__srv__do_thing"

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self, tmp_path):
        mgr = McpServerManager()
        session = AsyncMock()
        content = MagicMock()
        content.text = "tool output"
        result = MagicMock()
        result.content = [content]
        session.call_tool.return_value = result
        mgr._sessions["srv"] = session

        out = await mgr.call_tool("srv", "do_thing", {"x": "hello"})
        assert out == "tool output"

    @pytest.mark.asyncio
    async def test_call_tool_missing_server(self):
        mgr = McpServerManager()
        out = await mgr.call_tool("nonexistent", "tool", {})
        assert "not connected" in out

    @pytest.mark.asyncio
    async def test_list_resources_no_servers(self):
        mgr = McpServerManager()
        out = await mgr.list_resources()
        assert "No MCP servers connected" in out

    @pytest.mark.asyncio
    async def test_list_resources_with_server(self):
        mgr = McpServerManager()
        session = AsyncMock()
        r = MagicMock()
        r.uri = "file:///foo.py"
        r.name = "foo.py"
        r.description = "A file"
        result = MagicMock()
        result.resources = [r]
        session.list_resources.return_value = result
        mgr._sessions["srv"] = session

        out = await mgr.list_resources("srv")
        assert "file:///foo.py" in out
        assert "foo.py" in out

    @pytest.mark.asyncio
    async def test_read_resource(self):
        mgr = McpServerManager()
        session = AsyncMock()
        c = MagicMock()
        c.text = "file content"
        result = MagicMock()
        result.contents = [c]
        session.read_resource.return_value = result
        mgr._sessions["srv"] = session

        out = await mgr.read_resource("srv", "file:///foo.py")
        assert out == "file content"

    @pytest.mark.asyncio
    async def test_read_resource_missing_server(self):
        mgr = McpServerManager()
        out = await mgr.read_resource("none", "file:///x")
        assert "not connected" in out

    @pytest.mark.asyncio
    async def test_stop_calls_exit_stack(self):
        mgr = McpServerManager()
        stack = AsyncMock()
        session = MagicMock()
        session._exit_stack = stack
        mgr._sessions["srv"] = session
        await mgr.stop()
        stack.aclose.assert_called_once()
        assert mgr._sessions == {}

    def test_server_names(self):
        mgr = McpServerManager()
        mgr._sessions = {"a": MagicMock(), "b": MagicMock()}
        assert set(mgr.server_names()) == {"a", "b"}

    def test_is_connected(self):
        mgr = McpServerManager()
        mgr._sessions["s"] = MagicMock()
        assert mgr.is_connected("s")
        assert not mgr.is_connected("x")


class TestDynamicMcpTool:
    def _make_tool(self, server="srv", tool_name="do_thing", desc="Does a thing", schema=None):
        mcp_tool = MagicMock()
        mcp_tool.name = tool_name
        mcp_tool.description = desc
        mcp_tool.inputSchema = schema or {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        manager = MagicMock()
        return DynamicMcpTool(server_name=server, mcp_tool=mcp_tool, manager=manager)

    def test_name(self):
        t = self._make_tool()
        assert t.name == "mcp__srv__do_thing"

    def test_description_includes_server(self):
        t = self._make_tool()
        assert "srv" in t.description
        assert "Does a thing" in t.description

    def test_input_schema(self):
        t = self._make_tool()
        schema = t.input_schema
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert "x" in schema["required"]

    def test_input_schema_empty_when_no_schema(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "noop"
        mcp_tool.description = ""
        mcp_tool.inputSchema = {}
        t = DynamicMcpTool(server_name="s", mcp_tool=mcp_tool, manager=MagicMock())
        schema = t.input_schema
        assert schema["properties"] == {}
        assert schema["required"] == []

    @pytest.mark.asyncio
    async def test_execute_delegates_to_manager(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "do_thing"
        mcp_tool.description = ""
        mcp_tool.inputSchema = {}
        manager = AsyncMock()
        manager.call_tool.return_value = "result text"
        t = DynamicMcpTool(server_name="srv", mcp_tool=mcp_tool, manager=manager)
        out = await t.execute(x="hello")
        assert out == "result text"
        manager.call_tool.assert_called_once_with("srv", "do_thing", {"x": "hello"})


class TestListMcpResourcesTool:
    @pytest.mark.asyncio
    async def test_execute_no_arg(self):
        manager = AsyncMock()
        manager.list_resources.return_value = "Server 'srv' resources (0):"
        t = ListMcpResourcesTool(manager)
        out = await t.execute()
        manager.list_resources.assert_called_with(None)

    @pytest.mark.asyncio
    async def test_execute_with_server(self):
        manager = AsyncMock()
        manager.list_resources.return_value = "ok"
        t = ListMcpResourcesTool(manager)
        await t.execute(server_name="my-server")
        manager.list_resources.assert_called_with("my-server")

    def test_name_and_schema(self):
        t = ListMcpResourcesTool(MagicMock())
        assert t.name == "ListMcpResourcesTool"
        assert "server_name" in t.input_schema["properties"]


class TestReadMcpResourceTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        manager = AsyncMock()
        manager.read_resource.return_value = "content"
        t = ReadMcpResourceTool(manager)
        out = await t.execute(server_name="srv", uri="file:///x")
        assert out == "content"
        manager.read_resource.assert_called_with("srv", "file:///x")

    def test_required_params(self):
        t = ReadMcpResourceTool(MagicMock())
        assert "server_name" in t.input_schema["required"]
        assert "uri" in t.input_schema["required"]


class TestMcpAuthTool:
    @pytest.fixture(autouse=True)
    def patch_token_file(self, tmp_path):
        token_file = tmp_path / "mcp-tokens.json"
        with patch("cca.tools.mcp_tools._TOKEN_FILE", token_file):
            yield token_file

    @pytest.mark.asyncio
    async def test_list_empty(self):
        t = McpAuthTool()
        out = await t.execute(action="list")
        assert "No MCP auth tokens" in out

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        t = McpAuthTool()
        out = await t.execute(action="set", server_name="my-srv", token="tok123")
        assert "stored" in out.lower()
        out2 = await t.execute(action="get", server_name="my-srv")
        assert "my-srv" in out2
        assert "tok123" not in out2  # secrets are never echoed

    @pytest.mark.asyncio
    async def test_set_and_list(self):
        t = McpAuthTool()
        await t.execute(action="set", server_name="my-srv", token="tok")
        out = await t.execute(action="list")
        assert "my-srv" in out

    @pytest.mark.asyncio
    async def test_delete(self):
        t = McpAuthTool()
        await t.execute(action="set", server_name="my-srv", token="tok")
        out = await t.execute(action="delete", server_name="my-srv")
        assert "deleted" in out.lower()
        out2 = await t.execute(action="get", server_name="my-srv")
        assert "No token" in out2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        t = McpAuthTool()
        out = await t.execute(action="get", server_name="nope")
        assert "No token" in out

    @pytest.mark.asyncio
    async def test_set_requires_server_name(self):
        t = McpAuthTool()
        out = await t.execute(action="set", token="tok")
        assert "Error" in out or "required" in out

    @pytest.mark.asyncio
    async def test_set_requires_token(self):
        t = McpAuthTool()
        out = await t.execute(action="set", server_name="srv")
        assert "Error" in out or "required" in out

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        t = McpAuthTool()
        out = await t.execute(action="frobnicate")
        assert "Unknown" in out or "frobnicate" in out

    def test_name_and_schema(self):
        t = McpAuthTool()
        assert t.name == "McpAuthTool"
        assert "action" in t.input_schema["required"]


# ---------------------------------------------------------------------------
# LSP tool
# ---------------------------------------------------------------------------

from cca.tools.lsp_tool import _detect_servers, LspTool, DETECTED_SERVERS


class TestLspToolDetection:
    def test_detect_servers_returns_list(self):
        assert isinstance(DETECTED_SERVERS, list)

    def test_detect_servers_with_mock(self):
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda b: f"/usr/bin/{b}" if b == "pyright-langserver" else None
            servers = _detect_servers()
        python_servers = [s for s in servers if s["language"] == "python"]
        assert len(python_servers) == 1
        assert python_servers[0]["binary"] == "pyright-langserver"

    def test_detect_servers_empty_when_none_found(self):
        with patch("shutil.which", return_value=None):
            servers = _detect_servers()
        assert servers == []

    def test_detect_servers_first_binary_wins(self):
        def which(b):
            return f"/usr/bin/{b}" if b in ("pyright-langserver", "pylsp") else None
        with patch("shutil.which", side_effect=which):
            servers = _detect_servers()
        python_srv = [s for s in servers if s["language"] == "python"]
        assert len(python_srv) == 1
        assert python_srv[0]["binary"] == "pyright-langserver"


class TestLspToolSchema:
    def _make_tool(self):
        fake_server = {
            "command": ["pyright-langserver", "--stdio"],
            "extensions": [".py"],
            "language": "python",
            "binary": "pyright-langserver",
        }
        with patch("cca.tools.lsp_tool.DETECTED_SERVERS", [fake_server]):
            return LspTool(cwd="/tmp")

    def test_name(self):
        t = self._make_tool()
        assert t.name == "LSP"

    def test_description_not_empty(self):
        t = self._make_tool()
        assert len(t.description) > 0

    def test_input_schema_has_operation(self):
        t = self._make_tool()
        assert "operation" in t.input_schema["properties"]

    def test_input_schema_has_file_path(self):
        t = self._make_tool()
        props = t.input_schema["properties"]
        # LSP tool uses "filePath" (camelCase) for the file path parameter
        assert "filePath" in props or "file_path" in props


# ---------------------------------------------------------------------------
# build_registry includes MCP static tools
# ---------------------------------------------------------------------------

from cca.tools import build_registry


class TestBuildRegistryMcp:
    def test_mcp_static_tools_registered(self, tmp_path):
        r = build_registry(cwd=str(tmp_path))
        assert "ListMcpResourcesTool" in r._tools
        assert "ReadMcpResourceTool" in r._tools
        assert "McpAuthTool" in r._tools

    def test_mcp_manager_attached(self, tmp_path):
        r = build_registry(cwd=str(tmp_path))
        assert hasattr(r, "_mcp_manager")
        assert isinstance(r._mcp_manager, McpServerManager)

    def test_no_dynamic_tools_without_mcp_json(self, tmp_path):
        r = build_registry(cwd=str(tmp_path))
        dynamic = [k for k in r._tools if k.startswith("mcp__")]
        assert dynamic == []


# ---------------------------------------------------------------------------
# Thinking-level backend helpers
# ---------------------------------------------------------------------------

from cca.llm.anthropic_backend import (
    _model_supports_thinking,
    _model_uses_adaptive_thinking,
    _anthropic_thinking_params,
    _anthropic_thinking_kwargs,
)
from cca.llm.openai_backend import (
    _is_gemini_25,
    _is_gemini_3x,
    _is_openai_reasoning_model,
    _google_model_supports_thinking,
)


class TestAnthropicThinkingParams:
    def test_off_returns_none(self):
        assert _anthropic_thinking_params("claude-sonnet-4-6", "off") is None

    def test_unsupported_model_returns_none(self):
        assert _anthropic_thinking_params("claude-3-5-sonnet-20241022", "high") is None

    def test_manual_model_low(self):
        p = _anthropic_thinking_params("claude-sonnet-4-5", "low")
        assert p == {"type": "enabled", "budget_tokens": 4_000}

    def test_manual_model_medium(self):
        p = _anthropic_thinking_params("claude-opus-4-5", "medium")
        assert p == {"type": "enabled", "budget_tokens": 8_000}

    def test_manual_model_high(self):
        p = _anthropic_thinking_params("claude-haiku-4-5-20251001", "high")
        assert p == {"type": "enabled", "budget_tokens": 32_000}

    def test_adaptive_model_returns_adaptive(self):
        p = _anthropic_thinking_params("claude-opus-4-8", "high")
        assert p == {"type": "adaptive"}

    def test_adaptive_model_off(self):
        assert _anthropic_thinking_params("claude-opus-4-8", "off") is None

    def test_sonnet_5_off_is_explicitly_disabled(self):
        assert _anthropic_thinking_params("claude-sonnet-5", "off") == {
            "type": "disabled"
        }

    def test_sonnet_5_enabled_level_keeps_effort(self):
        assert _anthropic_thinking_kwargs("claude-sonnet-5", "low") == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        }

    def test_claude_3_7_supports_thinking(self):
        assert _model_supports_thinking("claude-3-7-sonnet-20250219")

    def test_claude_3_5_does_not_support(self):
        assert not _model_supports_thinking("claude-3-5-sonnet-20241022")

    def test_opus_4_8_uses_adaptive(self):
        assert _model_uses_adaptive_thinking("claude-opus-4-8")

    def test_sonnet_4_6_uses_adaptive(self):
        assert _model_uses_adaptive_thinking("claude-sonnet-4-6")


class TestGeminiThinkingDetection:
    def test_gemini_25_pro(self):
        assert _is_gemini_25("gemini-2.5-pro")
        assert not _is_gemini_3x("gemini-2.5-pro")

    def test_gemini_25_flash(self):
        assert _is_gemini_25("gemini-2.5-flash")

    def test_gemini_35_flash(self):
        assert not _is_gemini_25("gemini-3.5-flash")
        assert _is_gemini_3x("gemini-3.5-flash")

    def test_gemini_37_flash(self):
        assert not _is_gemini_25("gemini-3.7-flash")
        assert _is_gemini_3x("gemini-3.7-flash")
        assert _google_model_supports_thinking("gemini-3.7-flash")

    def test_gemini_31_pro(self):
        assert _is_gemini_3x("gemini-3.1-pro")

    def test_gemini_3_flash(self):
        assert _is_gemini_3x("gemini-3-flash")

    def test_gemini_20_not_supported(self):
        assert not _google_model_supports_thinking("gemini-2.0-flash")

    def test_gemini_15_not_supported(self):
        assert not _google_model_supports_thinking("gemini-1.5-pro")

    def test_gemini_25_supports_thinking(self):
        assert _google_model_supports_thinking("gemini-2.5-pro")

    def test_gemini_35_supports_thinking(self):
        assert _google_model_supports_thinking("gemini-3.5-flash")


class TestOpenAIReasoningDetection:
    def test_gpt5(self):
        assert _is_openai_reasoning_model("gpt-5.5")

    def test_gpt54(self):
        assert _is_openai_reasoning_model("gpt-5.4")

    def test_gpt54_mini(self):
        assert _is_openai_reasoning_model("gpt-5.4-mini")

    def test_gpt4o_not_reasoning(self):
        assert not _is_openai_reasoning_model("gpt-4o")

    def test_gpt4_not_reasoning(self):
        assert not _is_openai_reasoning_model("gpt-4-turbo")


class TestOpenAIBackendThinkingKwargs:
    def _make_backend(self, model: str, provider: str = "openai"):
        from cca.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, api_key="sk-test", provider=provider)

    def test_gemini_25_off_sends_nothing(self):
        b = self._make_backend("gemini-2.5-flash", provider="google")
        kw = b._build_thinking_kwargs("off")
        assert kw == {}

    def test_gemini_25_high_uses_reasoning_effort(self):
        b = self._make_backend("gemini-2.5-pro", provider="google")
        kw = b._build_thinking_kwargs("high")
        assert kw == {"reasoning_effort": "high"}

    def test_gemini_35_off_sends_nothing(self):
        # Gemini 3.5 "off" → omit param (model still does minimal thinking internally)
        b = self._make_backend("gemini-3.5-flash", provider="google")
        kw = b._build_thinking_kwargs("off")
        assert kw == {}

    def test_gemini_35_high_uses_reasoning_effort(self):
        b = self._make_backend("gemini-3.5-flash", provider="google")
        kw = b._build_thinking_kwargs("high")
        assert kw == {"reasoning_effort": "high"}

    def test_gemini_35_low(self):
        b = self._make_backend("gemini-3.5-flash", provider="google")
        kw = b._build_thinking_kwargs("low")
        assert kw == {"reasoning_effort": "low"}

    def test_gemini_37_supported_efforts(self):
        b = self._make_backend("gemini-3.7-flash", provider="google")
        assert b._build_thinking_kwargs("low") == {"reasoning_effort": "low"}
        assert b._build_thinking_kwargs("medium") == {"reasoning_effort": "medium"}
        assert b._build_thinking_kwargs("high") == {"reasoning_effort": "high"}

    def test_gemini_20_not_supported_returns_empty(self):
        b = self._make_backend("gemini-2.0-flash", provider="google")
        kw = b._build_thinking_kwargs("high")
        assert kw == {}

    def test_gpt5_uses_reasoning_effort(self):
        b = self._make_backend("gpt-5.5", provider="openai")
        kw = b._build_thinking_kwargs("high")
        assert kw == {"reasoning": {"effort": "high"}}

    def test_gpt5_off_uses_none_effort(self):
        b = self._make_backend("gpt-5.5", provider="openai")
        kw = b._build_thinking_kwargs("off")
        assert kw == {"reasoning": {"effort": "none"}}

    def test_gpt4o_returns_empty(self):
        b = self._make_backend("gpt-4o", provider="openai")
        kw = b._build_thinking_kwargs("high")
        assert kw == {}
