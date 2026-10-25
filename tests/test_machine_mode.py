"""Tests for --machine mode JSON-lines protocol in repl.py and cli.py."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cca.repl import Repl


def _make_repl(**kwargs) -> Repl:
    """Build a minimal Repl suitable for machine-mode unit tests."""
    backend = MagicMock()
    backend.stream = AsyncMock(return_value=_empty_stream())
    registry = MagicMock()
    registry.get_schemas = MagicMock(return_value=[])
    registry._mcp_manager = None  # _start_mcp checks for None and returns early
    config = MagicMock()
    config.model = "anthropic:claude-sonnet-4-6"
    config.trace = False
    config.verbose = False
    config.hooks = {}  # real dict so HookRunner works correctly
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
        print_mode=False,
        **kwargs,
    )


async def _empty_stream():
    """Yield nothing — simulates model returning empty response."""
    return
    yield  # make it an async generator


async def test_machine_shutdown_message():
    """Sending {"type": "shutdown"} cleanly terminates run_machine."""
    repl = _make_repl()
    stdin_lines = [json.dumps({"type": "shutdown"})]
    output_lines: list[dict] = []

    async def fake_readline():
        if stdin_lines:
            return stdin_lines.pop(0) + "\n"
        return ""

    with patch.object(repl, "_machine_readline", fake_readline):
        with patch.object(repl, "_machine_writeline", lambda d: output_lines.append(d)):
            await repl.run_machine()

    assert not any(d.get("type") == "error" for d in output_lines)


async def test_machine_ready_event_includes_resolved_settings():
    """The orchestrator can inspect effective controls before the first turn."""
    repl = _make_repl(run_mode="machine")
    stdin_lines = [json.dumps({"type": "shutdown"})]
    output_lines: list[dict] = []

    async def fake_readline():
        if stdin_lines:
            return stdin_lines.pop(0) + "\n"
        return ""

    with patch.object(repl, "_machine_readline", fake_readline):
        with patch.object(repl, "_machine_writeline", lambda d: output_lines.append(d)):
            await repl.run_machine()

    ready = next(d for d in output_lines if d["type"] == "ready")
    assert ready["settings"]["run_mode"] == "machine"
    assert ready["settings"]["model"]["selector"] == repl.config.model
    assert ready["settings"]["thinking"]["effective"] == "off"
    assert "authentication" in ready["settings"]["backend"]


async def test_machine_user_turn_emits_turn_complete():
    """{"type": "user_turn"} runs process_turn and emits turn_complete."""
    repl = _make_repl()
    stdin_lines = [
        json.dumps({"type": "user_turn", "text": "hello"}),
        json.dumps({"type": "shutdown"}),
    ]
    output_lines: list[dict] = []

    async def fake_readline():
        if stdin_lines:
            return stdin_lines.pop(0) + "\n"
        return ""

    async def fake_process_turn(text, return_text=False, harness_turn=False):
        return "done"

    with patch.object(repl, "_machine_readline", fake_readline):
        with patch.object(repl, "_machine_writeline", lambda d: output_lines.append(d)):
            with patch.object(repl, "process_turn", fake_process_turn):
                await repl.run_machine()

    types = [d["type"] for d in output_lines]
    assert "turn_complete" in types
    tc = next(d for d in output_lines if d["type"] == "turn_complete")
    assert "session_id" in tc
    assert "msg_index_end" in tc


def test_build_registry_accepts_plan_fn():
    """build_registry accepts separate ask_fn and plan_fn without error."""
    from cca.tools import build_registry
    cfg = MagicMock()
    cfg.search_provider = "none"
    cfg.search_api_key = None
    cfg.skills_enabled = False
    ask = AsyncMock()
    plan = AsyncMock()
    reg = build_registry(config=cfg, ask_fn=ask, plan_fn=plan)
    ask_tool = reg.get("AskUserQuestion")
    plan_tool = reg.get("ExitPlanMode")
    assert ask_tool._ask_fn is ask
    assert plan_tool._ask_fn is plan


def test_machine_flag_in_help():
    """--machine flag appears in cca --help output."""
    from click.testing import CliRunner
    from cca.cli import main
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--machine" in result.output


def test_machine_flag_parsed():
    """--machine flag is accepted without error (even if it can't actually run)."""
    from click.testing import CliRunner
    from cca.cli import main
    # Send a shutdown message on stdin so the machine loop exits immediately
    result = CliRunner().invoke(
        main,
        ["--machine"],
        input=json.dumps({"type": "shutdown"}) + "\n",
        catch_exceptions=False,
    )
    # Exit code 0 or 1 is acceptable; what must NOT happen is "No such option: --machine"
    assert "No such option" not in result.output


def test_ready_event_omits_randomization_by_default():
    repl = _make_repl()
    assert repl._randomization_payload is None


def test_ready_event_carries_randomization_when_active():
    from cca.randomize import resolve_randomization
    repl = _make_repl(randomization=resolve_randomization("all", 3))
    payload = repl._randomization_payload
    assert payload["seed"] == 3
    assert set(payload["variants"]) == {"system_prompt", "tool_descriptions", "compaction_prompt"}


def test_randomize_and_seed_flags_in_help():
    from click.testing import CliRunner
    from cca.cli import main
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--randomize" in result.output
    assert "--seed" in result.output


def test_unknown_randomize_axis_exits_nonzero():
    from click.testing import CliRunner
    from cca.cli import main
    result = CliRunner().invoke(main, ["--randomize", "bogus", "--print", "hi"])
    assert result.exit_code == 1
    assert "Unknown randomization axis" in result.output
