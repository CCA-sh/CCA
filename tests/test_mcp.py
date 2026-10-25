"""Tests for MCP support: mode parity unit tests and stdio-server integration."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cca.repl import Repl


def _make_repl(**kwargs) -> Repl:
    """Minimal Repl for lifecycle unit tests (mirrors tests/test_machine_mode.py)."""
    backend = MagicMock()
    registry = MagicMock()
    registry.get_schemas = MagicMock(return_value=[])
    registry._mcp_manager = None
    config = MagicMock()
    config.model = "anthropic:claude-sonnet-4-6"
    config.trace = False
    config.verbose = False
    config.hooks = {}
    config.max_tokens = 8192
    config.skills_enabled = False
    config.system_prompt_append = None
    config.vim = False
    perms = MagicMock()
    perms.check = AsyncMock(return_value=True)
    return Repl(
        backend=backend,
        registry=registry,
        config=config,
        permissions=perms,
        cwd="/tmp",
        **kwargs,
    )


async def test_run_print_runs_session_startup_and_shutdown():
    repl = _make_repl(print_mode=True)
    calls: list[str] = []

    with patch.object(repl._hooks, "session_start", AsyncMock(side_effect=lambda: calls.append("session_start"))), \
         patch.object(repl._hooks, "session_end", AsyncMock(side_effect=lambda: calls.append("session_end"))), \
         patch.object(repl, "_start_mcp", AsyncMock(side_effect=lambda: calls.append("start_mcp"))), \
         patch.object(repl, "process_turn", AsyncMock(side_effect=lambda *a, **k: calls.append("turn") or "hi")):
        result = await repl.run_print("hello")

    assert result == "hi"
    assert calls == ["session_start", "start_mcp", "turn", "session_end"]


async def test_run_print_shutdown_stops_mcp_manager():
    repl = _make_repl(print_mode=True)
    manager = MagicMock()
    manager.stop = AsyncMock()
    repl.registry._mcp_manager = manager

    with patch.object(repl._hooks, "session_start", AsyncMock()), \
         patch.object(repl._hooks, "session_end", AsyncMock()), \
         patch.object(repl, "_start_mcp", AsyncMock()), \
         patch.object(repl, "process_turn", AsyncMock(return_value="ok")):
        await repl.run_print("hello")

    manager.stop.assert_awaited_once()


async def test_run_print_shutdown_runs_even_when_turn_raises():
    repl = _make_repl(print_mode=True)
    ended = AsyncMock()

    with patch.object(repl._hooks, "session_start", AsyncMock()), \
         patch.object(repl._hooks, "session_end", ended), \
         patch.object(repl, "_start_mcp", AsyncMock()), \
         patch.object(repl, "process_turn", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            await repl.run_print("hello")

    ended.assert_awaited_once()


async def test_run_machine_runs_session_startup_and_shutdown():
    repl = _make_repl(print_mode=False)
    calls: list[str] = []
    stdin_lines = [json.dumps({"type": "shutdown"})]

    async def fake_readline():
        if stdin_lines:
            return stdin_lines.pop(0) + "\n"
        return ""

    with patch.object(repl._hooks, "session_start", AsyncMock(side_effect=lambda: calls.append("session_start"))), \
         patch.object(repl._hooks, "session_end", AsyncMock(side_effect=lambda: calls.append("session_end"))), \
         patch.object(repl, "_start_mcp", AsyncMock(side_effect=lambda: calls.append("start_mcp"))), \
         patch.object(repl, "_machine_readline", fake_readline), \
         patch.object(repl, "_machine_writeline", lambda d: None):
        await repl.run_machine()

    assert calls == ["session_start", "start_mcp", "session_end"]


async def test_run_interactive_ui_runs_session_shutdown_on_quit():
    repl = _make_repl(print_mode=False)
    calls: list[str] = []
    manager = MagicMock()
    manager.stop = AsyncMock()
    repl.registry._mcp_manager = manager

    ui = MagicMock()
    ui.get_input = AsyncMock(return_value=None)  # user quits immediately
    repl._ui = ui

    with patch.object(repl._hooks, "session_start", AsyncMock(side_effect=lambda: calls.append("session_start"))), \
         patch.object(repl._hooks, "session_end", AsyncMock(side_effect=lambda: calls.append("session_end"))), \
         patch.object(repl, "_start_mcp", AsyncMock(side_effect=lambda: calls.append("start_mcp"))), \
         patch.object(repl, "_print_banner", MagicMock()):
        await repl.run_interactive()

    assert calls == ["session_start", "start_mcp", "session_end"]
    manager.stop.assert_awaited_once()


async def test_run_interactive_fallback_runs_session_shutdown_on_eof():
    repl = _make_repl(print_mode=False)
    calls: list[str] = []

    session = MagicMock()
    session.prompt_async = AsyncMock(side_effect=EOFError)

    with patch("prompt_toolkit.PromptSession", MagicMock(return_value=session)), \
         patch.object(repl._hooks, "session_start", AsyncMock(side_effect=lambda: calls.append("session_start"))), \
         patch.object(repl._hooks, "session_end", AsyncMock(side_effect=lambda: calls.append("session_end"))), \
         patch.object(repl, "_start_mcp", AsyncMock(side_effect=lambda: calls.append("start_mcp"))), \
         patch.object(repl, "_print_banner", MagicMock()):
        await repl.run_interactive()

    assert calls == ["session_start", "start_mcp", "session_end"]


# ---------------------------------------------------------------------------
# Integration: real stdio MCP server via .mcp.json
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

from cca.tools.mcp_tools import McpServerManager, _load_mcp_config

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"


def _write_mcp_json(directory: Path) -> None:
    (directory / ".mcp.json").write_text(json.dumps({
        "mcpServers": {
            "fixture": {
                "command": sys.executable,
                "args": [str(FIXTURE_SERVER)],
            }
        }
    }))


def test_load_mcp_config_walks_up_from_cwd(tmp_path):
    _write_mcp_json(tmp_path)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    cfg = _load_mcp_config(str(nested))
    assert "fixture" in cfg


def test_load_mcp_config_empty_when_absent(tmp_path):
    assert _load_mcp_config(str(tmp_path)) == {}


async def test_stdio_server_tools_connect_and_execute(trust_file, tmp_path):
    _write_mcp_json(tmp_path)
    manager = McpServerManager(cwd=str(tmp_path), trust_all=True)
    wrappers = await manager.start()
    try:
        assert manager.server_names() == ["fixture"]
        by_name = {w.name: w for w in wrappers}
        assert "mcp__fixture__add" in by_name

        add = by_name["mcp__fixture__add"]
        schema = add.input_schema
        assert set(schema["properties"]) == {"a", "b"}

        result = await add.execute(a=2, b=3)
        assert "5" in result

        resources = await manager.list_resources()
        assert "fixture://greeting" in resources

        content = await manager.read_resource("fixture", "fixture://greeting")
        assert "hello from fixture" in content
    finally:
        await manager.stop()
    assert manager.server_names() == []


# ---------------------------------------------------------------------------
# Token injection
# ---------------------------------------------------------------------------

from cca.tools import mcp_tools
from cca.tools.mcp_tools import McpAuthTool, _resolve_headers


def test_resolve_headers_injects_stored_token(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_load_tokens", lambda: {"github": "tok123"})
    headers = _resolve_headers("github", {"url": "https://x"})
    assert headers == {"Authorization": "Bearer tok123"}


def test_resolve_headers_explicit_authorization_wins(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_load_tokens", lambda: {"github": "tok123"})
    cfg = {"headers": {"authorization": "Bearer explicit"}}
    headers = _resolve_headers("github", cfg)
    assert headers == {"authorization": "Bearer explicit"}


def test_resolve_headers_none_when_no_headers_or_token(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_load_tokens", lambda: {})
    assert _resolve_headers("github", {}) is None


async def test_mcp_auth_get_does_not_echo_token(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_tools, "_TOKEN_FILE", tmp_path / "tokens.json")
    tool = McpAuthTool()
    await tool.execute(action="set", server_name="github", token="supersecret")
    result = await tool.execute(action="get", server_name="github")
    assert "supersecret" not in result
    assert "github" in result


# ---------------------------------------------------------------------------
# Trust store
# ---------------------------------------------------------------------------

from cca.tools import mcp_trust


@pytest.fixture
def trust_file(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_trust, "TRUST_FILE", tmp_path / "mcp-trust.json")
    return mcp_trust.TRUST_FILE


def test_trust_roundtrip(trust_file):
    cfg = {"command": "python", "args": ["srv.py"]}
    assert not mcp_trust.is_trusted("/proj/.mcp.json", "fixture", cfg)
    mcp_trust.record_trust("/proj/.mcp.json", "fixture", cfg)
    assert mcp_trust.is_trusted("/proj/.mcp.json", "fixture", cfg)
    assert mcp_trust.revoke_trust("/proj/.mcp.json", "fixture")
    assert not mcp_trust.is_trusted("/proj/.mcp.json", "fixture", cfg)
    assert not mcp_trust.revoke_trust("/proj/.mcp.json", "fixture")


def test_trust_invalidated_when_config_changes(trust_file):
    cfg = {"command": "python", "args": ["srv.py"]}
    mcp_trust.record_trust("/proj/.mcp.json", "fixture", cfg)
    edited = {"command": "python", "args": ["evil.py"]}
    assert not mcp_trust.is_trusted("/proj/.mcp.json", "fixture", edited)


def test_trust_is_per_config_path(trust_file):
    cfg = {"command": "python"}
    mcp_trust.record_trust("/proj-a/.mcp.json", "fixture", cfg)
    assert not mcp_trust.is_trusted("/proj-b/.mcp.json", "fixture", cfg)


# ---------------------------------------------------------------------------
# Trust gating in McpServerManager
# ---------------------------------------------------------------------------

async def test_untrusted_server_skipped_headless(trust_file, tmp_path):
    _write_mcp_json(tmp_path)
    manager = McpServerManager(cwd=str(tmp_path))
    wrappers = await manager.start()
    assert wrappers == []
    assert manager.server_names() == []
    assert manager.skipped_servers == ["fixture"]


async def test_trust_fn_accept_connects_and_persists(trust_file, tmp_path):
    _write_mcp_json(tmp_path)
    prompts: list[str] = []

    async def accept(name, cfg, config_path):
        prompts.append(name)
        return True

    manager = McpServerManager(cwd=str(tmp_path), trust_fn=accept)
    wrappers = await manager.start()
    try:
        assert prompts == ["fixture"]
        assert manager.server_names() == ["fixture"]
        assert any(w.name == "mcp__fixture__add" for w in wrappers)
    finally:
        await manager.stop()

    # Persisted: a later headless session connects without prompting.
    manager2 = McpServerManager(cwd=str(tmp_path))
    wrappers2 = await manager2.start()
    try:
        assert manager2.server_names() == ["fixture"]
        assert manager2.skipped_servers == []
    finally:
        await manager2.stop()


async def test_trust_fn_decline_skips_and_does_not_persist(trust_file, tmp_path):
    _write_mcp_json(tmp_path)

    async def decline(name, cfg, config_path):
        return False

    manager = McpServerManager(cwd=str(tmp_path), trust_fn=decline)
    assert await manager.start() == []
    assert manager.skipped_servers == ["fixture"]

    manager2 = McpServerManager(cwd=str(tmp_path))
    await manager2.start()
    assert manager2.skipped_servers == ["fixture"]


async def test_trust_all_connects_without_persisting(trust_file, tmp_path):
    _write_mcp_json(tmp_path)
    manager = McpServerManager(cwd=str(tmp_path), trust_all=True)
    wrappers = await manager.start()
    try:
        assert manager.server_names() == ["fixture"]
    finally:
        await manager.stop()
    cfg = _load_mcp_config(str(tmp_path))
    assert not mcp_trust.is_trusted(
        str(tmp_path / ".mcp.json"), "fixture", cfg["fixture"])


# ---------------------------------------------------------------------------
# REPL trust wiring
# ---------------------------------------------------------------------------

async def test_permission_manager_confirm_yes_and_no():
    from cca.permissions import PermissionManager

    answers = iter(["y", "n"])

    async def fake_permission_fn(title):
        return next(answers)

    pm = PermissionManager(permission_fn=fake_permission_fn, console=MagicMock())
    assert await pm.confirm("Trust?", "body") is True
    assert await pm.confirm("Trust?", "body") is False


async def test_start_mcp_reports_skipped_servers():
    repl = _make_repl(print_mode=True)
    manager = MagicMock()
    manager.start = AsyncMock(return_value=[])
    manager.server_names = MagicMock(return_value=[])
    manager.skipped_servers = ["fixture"]
    manager.trust_all = False
    repl.registry._mcp_manager = manager
    repl.permissions.auto_approve = False

    printed: list[str] = []
    with patch.object(repl, "_console", MagicMock(print=lambda *a, **k: printed.append(str(a[0])))):
        await repl._start_mcp()

    assert any("fixture" in line for line in printed)
    assert any("trust" in line.lower() for line in printed)


async def test_auto_approve_sets_trust_all():
    repl = _make_repl(print_mode=True)
    manager = MagicMock()
    manager.start = AsyncMock(return_value=[])
    manager.server_names = MagicMock(return_value=[])
    manager.skipped_servers = []
    manager.trust_all = False
    repl.registry._mcp_manager = manager
    repl.permissions.auto_approve = True

    await repl._start_mcp()
    assert manager.trust_all is True
