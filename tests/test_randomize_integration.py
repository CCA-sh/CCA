"""Randomization wired into Repl: prompt, schemas, compaction, and metadata sinks."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cca.llm.base import ToolSchema
from cca.prompts import COMPACTION_PROMPT, MAIN_SYSTEM_PROMPT_TEMPLATE, main_prompt
from cca.randomize import RandomizationPlan, resolve_randomization
from cca.repl import Repl


def _make_repl(**kwargs) -> Repl:
    backend = MagicMock()
    registry = MagicMock()
    registry.get_schemas = MagicMock(return_value=[
        ToolSchema("Read", "Read a file.", {"type": "object", "properties": {}}),
        ToolSchema("Bash", "Run a command.", {"type": "object", "properties": {}}),
    ])
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
    return Repl(backend=backend, registry=registry, config=config, permissions=perms,
                cwd="/tmp", print_mode=True, **kwargs)


# ---------------------------------------------------------------------------
# Default behavior is unchanged
# ---------------------------------------------------------------------------

def test_repl_without_randomization_uses_the_shipped_prompt():
    repl = _make_repl()
    assert repl._randomization.is_identity()
    assert repl._compaction_prompt == COMPACTION_PROMPT
    assert MAIN_SYSTEM_PROMPT_TEMPLATE.split("\n")[0] in repl._build_system()


def test_repl_without_randomization_writes_no_payload():
    assert _make_repl()._randomization_payload is None


def test_identity_leaves_schema_order_and_text_alone():
    repl = _make_repl()
    schemas = repl.registry.get_schemas()
    assert repl._randomization.apply_to_schemas(schemas) is schemas


def test_main_prompt_template_argument_defaults_to_the_shipped_prompt():
    assert main_prompt(cwd="/tmp") == main_prompt(cwd="/tmp", template=MAIN_SYSTEM_PROMPT_TEMPLATE)


def test_main_prompt_honours_a_supplied_template():
    rendered = main_prompt(cwd="/tmp", extra="X", template="A {env_info} B {extra}")
    assert rendered.startswith("A ") and rendered.endswith(" B X")


# ---------------------------------------------------------------------------
# Randomization takes effect
# ---------------------------------------------------------------------------

def test_randomized_system_prompt_replaces_the_template():
    plan = resolve_randomization("system_prompt", 3)
    repl = _make_repl(randomization=plan)
    built = repl._build_system()
    assert built != main_prompt(cwd="/tmp")
    # Placeholders were substituted, not left raw.
    assert "{env_info}" not in built and "{extra}" not in built


@pytest.mark.parametrize("seed", range(6))
def test_repl_uses_whichever_compaction_variant_was_selected(seed):
    plan = resolve_randomization("compaction_prompt", seed)
    assert _make_repl(randomization=plan)._compaction_prompt == plan.compaction_prompt


def test_some_seed_selects_a_non_baseline_compaction_prompt():
    """The baseline is in the pool, so only a sweep proves the axis has an effect."""
    assert any(
        resolve_randomization("compaction_prompt", seed).compaction_prompt != COMPACTION_PROMPT
        for seed in range(12)
    )


def test_explicit_system_prompt_override_beats_randomization():
    plan = resolve_randomization("system_prompt", 3)
    repl = _make_repl(randomization=plan, system_prompt="OVERRIDE")
    assert repl._build_system().startswith("OVERRIDE")


def test_project_instructions_still_reach_the_randomized_prompt():
    plan = resolve_randomization("system_prompt", 3)
    repl = _make_repl(randomization=plan)
    repl._project_instructions = "PROJECT-RULES"
    assert "PROJECT-RULES" in repl._build_system()


def test_randomized_payload_is_recorded():
    repl = _make_repl(randomization=resolve_randomization("all", 12))
    payload = repl._randomization_payload
    assert payload["seed"] == 12
    assert payload["seed_source"] == "explicit"
    assert payload["variants"]


# ---------------------------------------------------------------------------
# Metadata sinks
# ---------------------------------------------------------------------------

def _saved_payload(tmp_path, monkeypatch, randomization):
    monkeypatch.setattr("cca.paths.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("cca.sessions.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("cca.sessions.HISTORY_FILE", tmp_path / "history.jsonl")
    from cca.sessions import save_session
    save_session("sid", "/tmp", [], features={"preset": None}, randomization=randomization)
    return json.loads((tmp_path / "sid.json").read_text())


def test_session_file_omits_randomization_when_inactive(tmp_path, monkeypatch):
    """Backward compatibility: an ordinary run's file gains no new keys."""
    assert "randomization" not in _saved_payload(tmp_path, monkeypatch, None)


def test_session_file_records_randomization_when_active(tmp_path, monkeypatch):
    payload = {"seed": 5, "axes": ["tool_order"], "variants": {}}
    assert _saved_payload(tmp_path, monkeypatch, payload)["randomization"] == payload


def test_llm_call_log_carries_randomization(tmp_path, monkeypatch):
    monkeypatch.setattr("cca.paths.LLM_CALLS_DIR", tmp_path)
    repl = _make_repl(randomization=resolve_randomization("all", 9))
    repl._log_llm_call(
        call_type="turn", system="sys", messages_snapshot=[], tools=[], params={},
        response_text="", tool_calls=[], thinking="", stop_reason="end_turn",
        input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
    )
    record = json.loads((tmp_path / f"session_{repl._base_session_id}.jsonl").read_text())
    assert record["randomization"]["seed"] == 9


def test_llm_call_log_omits_randomization_when_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr("cca.paths.LLM_CALLS_DIR", tmp_path)
    repl = _make_repl()
    repl._log_llm_call(
        call_type="turn", system="sys", messages_snapshot=[], tools=[], params={},
        response_text="", tool_calls=[], thinking="", stop_reason="end_turn",
        input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
    )
    record = json.loads((tmp_path / f"session_{repl._base_session_id}.jsonl").read_text())
    assert "randomization" not in record


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_the_same_system_prompt():
    a = _make_repl(randomization=resolve_randomization("all", 77))._build_system()
    b = _make_repl(randomization=resolve_randomization("all", 77))._build_system()
    assert a == b


def test_different_seeds_diverge_across_a_sweep():
    prompts = {
        _make_repl(randomization=resolve_randomization("all", seed))._build_system()
        for seed in range(12)
    }
    assert len(prompts) > 1


@pytest.mark.parametrize("seed", range(8))
def test_every_seed_yields_a_renderable_prompt(seed):
    built = _make_repl(randomization=resolve_randomization("all", seed))._build_system()
    assert "{env_info}" not in built
    assert "<<CACHE_BOUNDARY>>" in built
    assert built.strip()
