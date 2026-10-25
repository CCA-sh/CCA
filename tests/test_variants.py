"""Quality gates for the shipped variant bank.

Parametrized over the whole bank, so a newly added variant is covered the moment it
lands. The contract logic lives in cca.variant_checks and is shared with
`cca variants verify`; these tests drive it per-variant for readable failures.
"""
import random

import pytest

from cca import variant_checks as checks
from cca import variants as vb
from cca.prompts import COMPACTION_PROMPT, MAIN_SYSTEM_PROMPT_TEMPLATE

MANIFEST = vb.load_manifest()
SCHEMAS = checks.registry_schemas()

ALL_VARIANTS = [v for axis in vb.CONTENT_AXES for v in vb.load_axis(axis)]
SYSTEM_PROMPT_VARIANTS = vb.load_axis("system_prompt")
COMPACTION_VARIANTS = vb.load_axis("compaction_prompt")

_ids = lambda variants: [f"{v.axis}/{v.id}" for v in variants]


# ---------------------------------------------------------------------------
# Contract gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS, ids=_ids(ALL_VARIANTS))
def test_variant_satisfies_its_contract(variant):
    assert checks.check_variant(variant, MANIFEST, SCHEMAS) == []


def test_whole_bank_verifies():
    """Exactly what `cca variants verify` runs in CI."""
    assert checks.check_bank() == []


def test_every_axis_ships_a_baseline():
    for axis in vb.CONTENT_AXES:
        assert vb.BASELINE_ID in [v.id for v in vb.load_axis(axis)]


def test_bank_adds_diversity():
    """Randomization against a one-entry bank is a silent no-op."""
    for axis in vb.CONTENT_AXES:
        assert len(vb.load_axis(axis)) > 1, f"{axis} bank adds no diversity"


# ---------------------------------------------------------------------------
# Baselines are the drift guard: editing prompts.py without the bank fails here
# ---------------------------------------------------------------------------

def test_system_prompt_baseline_is_byte_identical():
    text, _ = vb.render_sectioned(vb.get("system_prompt", vb.BASELINE_ID))
    assert text == MAIN_SYSTEM_PROMPT_TEMPLATE


def test_compaction_baseline_is_byte_identical():
    text, _ = vb.render_sectioned(vb.get("compaction_prompt", vb.BASELINE_ID))
    assert text == COMPACTION_PROMPT


def test_tool_descriptions_baseline_has_no_overrides():
    assert vb.tool_overrides(vb.get("tool_descriptions", vb.BASELINE_ID)) == {}


# ---------------------------------------------------------------------------
# Properties the contract checks cannot express per-variant
# ---------------------------------------------------------------------------

def test_variants_are_distinct_within_each_axis():
    for axis in ("system_prompt", "compaction_prompt"):
        rendered = {v.id: vb.render_sectioned(v)[0] for v in vb.load_axis(axis)}
        assert len(set(rendered.values())) == len(rendered), f"{axis} has duplicate variants"


def test_tool_description_variants_are_distinct():
    rendered = {v.id: repr(sorted(vb.tool_overrides(v).items())) for v in vb.load_axis("tool_descriptions")}
    assert len(set(rendered.values())) == len(rendered)


@pytest.mark.parametrize("variant", SYSTEM_PROMPT_VARIANTS, ids=_ids(SYSTEM_PROMPT_VARIANTS))
def test_system_prompt_declares_its_derivation(variant):
    assert "summary" in variant.meta
    if variant.id != vb.BASELINE_ID:
        assert variant.meta.get("derived_from") == vb.BASELINE_ID


@pytest.mark.parametrize("variant", COMPACTION_VARIANTS, ids=_ids(COMPACTION_VARIANTS))
def test_compaction_rendering_is_order_stable(variant):
    """The compaction prompt has ordered passes and an output spec; nothing may move."""
    plain, _ = vb.render_sectioned(variant)
    for seed in (1, 2, 3):
        shuffled, _ = vb.render_sectioned(variant, random.Random(seed))
        assert shuffled == plain


@pytest.mark.parametrize("variant", SYSTEM_PROMPT_VARIANTS, ids=_ids(SYSTEM_PROMPT_VARIANTS))
def test_system_prompt_shuffling_changes_order_for_some_seed(variant):
    baseline_order = vb.render_sectioned(variant)[1]
    assert any(
        vb.render_sectioned(variant, random.Random(seed))[1] != baseline_order
        for seed in range(20)
    ), f"{variant.id} has no shuffleable sections"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_missing_frontmatter_is_an_error():
    with pytest.raises(vb.VariantError, match="frontmatter"):
        vb._parse_variant("## @role\nhi\n", "system_prompt", "user", "bad.md")


def test_axis_mismatch_is_an_error():
    text = "---\nid: x\naxis: compaction_prompt\n---\n## @role\nhi\n"
    with pytest.raises(vb.VariantError, match="declares axis"):
        vb._parse_variant(text, "system_prompt", "user", "bad.md")


def test_unknown_variant_id_is_an_error():
    with pytest.raises(vb.VariantError, match="Unknown"):
        vb.get("system_prompt", "v99_nope")


def test_duplicate_section_names_are_an_error():
    text = "---\nid: x\naxis: system_prompt\n---\n## @a\none\n\n## @a\ntwo\n"
    with pytest.raises(vb.VariantError, match="duplicate"):
        vb.parse_sections(vb._parse_variant(text, "system_prompt", "user", "x.md"))


def test_misplaced_pin_is_an_error():
    text = "---\nid: x\naxis: system_prompt\n---\n## @a\none\n\n## @b {pinned=first}\ntwo\n"
    with pytest.raises(vb.VariantError, match="pinned=first"):
        vb.parse_sections(vb._parse_variant(text, "system_prompt", "user", "x.md"))


def test_line_continuations_are_joined():
    text = "---\nid: x\naxis: system_prompt\n---\n## @role\nwrapped \\\nline\n"
    variant = vb._parse_variant(text, "system_prompt", "user", "x.md")
    assert vb.parse_sections(variant)[0].text == "wrapped line"


def test_section_markers_are_stripped_from_output():
    text = "---\nid: x\naxis: system_prompt\n---\n## @a\none\n\n## @b\ntwo\n"
    rendered, _ = vb.render_sectioned(vb._parse_variant(text, "system_prompt", "user", "x.md"))
    assert rendered == "one\n\ntwo"
    assert "## @" not in rendered
