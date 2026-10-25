"""Tests for the per-run randomization resolver."""
import random

import pytest

from cca.llm.base import ToolSchema
from cca.randomize import (
    AXES, RandomizationError, RandomizationPlan, parse_axes, resolve_randomization,
)


def _schemas() -> list[ToolSchema]:
    return [
        ToolSchema("Read", "Read a file.", {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Path."}},
        }),
        ToolSchema("Bash", "Run a command.", {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Command."}},
        }),
        ToolSchema("Glob", "Match files.", {"type": "object", "properties": {}}),
    ]


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

def test_none_spec_is_identity():
    plan = resolve_randomization(None, None)
    assert plan.is_identity()
    assert plan.seed is None
    assert plan.axes == frozenset()
    assert plan.summary_line() is None


@pytest.mark.parametrize("spec", ["off", "none", "", "OFF", "  off  "])
def test_off_specs_are_identity(spec):
    assert resolve_randomization(spec, 1).is_identity()


@pytest.mark.parametrize("spec", ["all", "ALL", "true"])
def test_all_spec_enables_every_axis(spec):
    assert resolve_randomization(spec, 1).axes == frozenset(AXES)


def test_comma_list_and_hyphen_normalization():
    plan = resolve_randomization("system_prompt,tool-order", 1)
    assert plan.axes == frozenset({"system_prompt", "tool_order"})


def test_unknown_axis_is_hard_error():
    with pytest.raises(RandomizationError, match="sysprompt"):
        parse_axes("sysprompt")


def test_unknown_axis_message_lists_valid_axes():
    with pytest.raises(RandomizationError) as excinfo:
        parse_axes("bogus")
    for axis in AXES:
        assert axis in str(excinfo.value)


def test_identity_does_no_bank_io(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("bank must not be touched when randomization is off")

    monkeypatch.setattr("cca.variants.load_axis", explode)
    monkeypatch.setattr("cca.variants.load_manifest", explode)
    assert resolve_randomization(None, None).is_identity()


# ---------------------------------------------------------------------------
# Seeds and determinism
# ---------------------------------------------------------------------------

def test_explicit_seed_is_recorded_as_explicit():
    plan = resolve_randomization("all", 4242)
    assert plan.seed == 4242
    assert plan.seed_source == "explicit"


def test_auto_seed_is_generated_and_labelled():
    plan = resolve_randomization("all", None)
    assert isinstance(plan.seed, int)
    assert plan.seed_source == "auto"


def test_same_seed_reproduces_the_same_plan():
    a = resolve_randomization("all", 7)
    b = resolve_randomization("all", 7)
    assert a.variant_ids == b.variant_ids
    assert a.system_prompt == b.system_prompt
    assert a.section_order == b.section_order
    assert a.compaction_prompt == b.compaction_prompt


def test_axis_choices_are_independent_of_the_axis_set():
    """Adding an axis must not shift which variant another axis picks."""
    narrow = resolve_randomization("system_prompt", 99)
    wide = resolve_randomization("system_prompt,tool_order,compaction_prompt", 99)
    assert narrow.variant_ids["system_prompt"] == wide.variant_ids["system_prompt"]
    assert narrow.system_prompt == wide.system_prompt


def test_bank_version_recorded():
    assert resolve_randomization("all", 1).bank_version == 1


# ---------------------------------------------------------------------------
# Schema application
# ---------------------------------------------------------------------------

def test_identity_plan_returns_schemas_untouched():
    schemas = _schemas()
    assert RandomizationPlan.identity().apply_to_schemas(schemas) is schemas


def test_reorder_is_a_permutation_and_deterministic():
    schemas = _schemas()
    plan = resolve_randomization("tool_order", 5)
    once = plan.apply_to_schemas(schemas)
    twice = plan.apply_to_schemas(schemas)
    assert [s.name for s in once] == [s.name for s in twice]
    assert sorted(s.name for s in once) == sorted(s.name for s in schemas)


def test_reorder_is_independent_of_input_order():
    plan = resolve_randomization("tool_order", 5)
    forward = plan.apply_to_schemas(_schemas())
    backward = plan.apply_to_schemas(list(reversed(_schemas())))
    assert [s.name for s in forward] == [s.name for s in backward]


def test_reorder_actually_reorders_for_some_seed():
    schemas = _schemas()
    original = [s.name for s in schemas]
    assert any(
        [s.name for s in resolve_randomization("tool_order", seed).apply_to_schemas(schemas)] != original
        for seed in range(30)
    )


def test_description_override_applies_only_to_named_tools():
    plan = RandomizationPlan(
        seed=1, seed_source="explicit", axes=frozenset({"tool_descriptions"}),
        tool_descriptions={"Read": {"description": "Fetch file contents.", "params": {}}},
    )
    out = {s.name: s for s in plan.apply_to_schemas(_schemas())}
    assert out["Read"].description == "Fetch file contents."
    assert out["Bash"].description == "Run a command."


def test_param_description_override_does_not_touch_the_registry_schema():
    schemas = _schemas()
    plan = RandomizationPlan(
        seed=1, seed_source="explicit", axes=frozenset({"tool_descriptions"}),
        tool_descriptions={"Read": {"description": None, "params": {"file_path": "Absolute path."}}},
    )
    out = {s.name: s for s in plan.apply_to_schemas(schemas)}
    assert out["Read"].input_schema["properties"]["file_path"]["description"] == "Absolute path."
    # The original schema object must be untouched — it is shared with the registry.
    assert schemas[0].input_schema["properties"]["file_path"]["description"] == "Path."


def test_unknown_param_in_override_is_ignored_not_injected():
    plan = RandomizationPlan(
        seed=1, seed_source="explicit", axes=frozenset({"tool_descriptions"}),
        tool_descriptions={"Read": {"description": None, "params": {"nonexistent": "x"}}},
    )
    out = {s.name: s for s in plan.apply_to_schemas(_schemas())}
    assert "nonexistent" not in out["Read"].input_schema["properties"]


def test_names_and_param_keys_are_never_mutated():
    """The load-bearing identifiers must survive every axis at every seed."""
    schemas = _schemas()
    expected_names = sorted(s.name for s in schemas)
    expected_params = {s.name: sorted((s.input_schema.get("properties") or {})) for s in schemas}
    for seed in range(25):
        out = resolve_randomization("all", seed).apply_to_schemas(schemas)
        assert sorted(s.name for s in out) == expected_names
        for schema in out:
            assert sorted((schema.input_schema.get("properties") or {})) == expected_params[schema.name]


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_payload_shape():
    payload = resolve_randomization("all", 11).to_payload()
    assert payload["seed"] == 11
    assert payload["seed_source"] == "explicit"
    assert sorted(payload["axes"]) == sorted(AXES)
    assert set(payload["variants"]) == {"system_prompt", "tool_descriptions", "compaction_prompt"}
    assert payload["section_order"][0] == "role"
    assert payload["section_order"][-1] == "env"


def test_summary_line_mentions_seed():
    assert "11" in resolve_randomization("all", 11).summary_line()
