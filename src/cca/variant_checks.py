"""Contract checks for the variant bank, shared by `cca variants verify` and the tests.

Every check here defends one invariant that keeps randomization safe: a variant may
change how the scaffold reads, never what it permits, names, or requires.
"""

from __future__ import annotations

import random
import re

from cca import variants as bank
from cca.prompts import COMPACTION_PROMPT, MAIN_SYSTEM_PROMPT_TEMPLATE

# The prompt's idiom for naming a tool. Catches an invented "use the Search tool"
# while ignoring compound nouns such as "CLI tool" and "Large tool outputs".
_TOOL_MENTION_RE = re.compile(
    r"\b(?:the|The|use|Use|using|Using|via|with|call|Call|prefer|Prefer) ([A-Z][A-Za-z]+) tool\b")

_BASELINE_TEXT = {
    "system_prompt": MAIN_SYSTEM_PROMPT_TEMPLATE,
    "compaction_prompt": COMPACTION_PROMPT,
}


def registry_schemas() -> dict:
    from cca.tools import build_registry
    return {s.name: s for s in build_registry().get_schemas()}


def known_tool_names(schemas: dict) -> set[str]:
    # LSP and Skill are real tools that a default build may not register (LSP needs a
    # language server on PATH; Skill needs skills_enabled). Naming them is legitimate.
    return set(schemas) | {"LSP", "Skill", "Task"}


def check_variant(variant, manifest: dict, schemas: dict) -> list[str]:
    """Return a list of contract violations for one variant. Empty means it passes."""
    where = f"{variant.axis}/{variant.id}"
    if variant.axis == "tool_descriptions":
        return [f"{where}: {m}" for m in _check_tool_descriptions(variant, manifest, schemas)]
    try:
        text, _ = bank.render_sectioned(variant)
    except bank.VariantError as e:
        return [f"{where}: {e}"]
    checks = _check_sectioned(variant, text, manifest, schemas)
    return [f"{where}: {m}" for m in checks]


def _check_sectioned(variant, text: str, manifest: dict, schemas: dict) -> list[str]:
    rules = manifest.get(variant.axis) or {}
    problems: list[str] = []
    problems += _check_directives(text, rules)
    problems += _check_length(text, _BASELINE_TEXT[variant.axis], rules)
    problems += _check_pins(variant, rules)
    if variant.axis == "system_prompt":
        problems += _check_placeholders(text)
        problems += _check_tool_names(text, schemas)
    else:
        problems += _check_compaction_blocks(text)
    return problems


def _check_directives(text: str, rules: dict) -> list[str]:
    missing = [
        name for name, pattern in (rules.get("required_directives") or {}).items()
        if not re.search(pattern, text)
    ]
    return [f"dropped directives: {', '.join(sorted(missing))}"] if missing else []


def _check_length(text: str, baseline: str, rules: dict) -> list[str]:
    bounds = rules.get("length_ratio")
    if not bounds:
        return []
    low, high = bounds
    ratio = len(text) / len(baseline)
    if not low <= ratio <= high:
        return [f"length is {ratio:.2f}x the baseline, outside [{low}, {high}]"]
    return []


def _check_pins(variant, rules: dict) -> list[str]:
    """Pinned sections must stay put no matter how the middle is shuffled."""
    pins = rules.get("pinned") or {}
    if not pins:
        return []
    sections = bank.parse_sections(variant)
    names = sorted(s.name for s in sections)
    problems = []
    if sections[0].name != pins.get("first"):
        problems.append(f"first section is '{sections[0].name}', expected '{pins.get('first')}'")
    if sections[-1].name != pins.get("last"):
        problems.append(f"last section is '{sections[-1].name}', expected '{pins.get('last')}'")
    for seed in range(20):
        _, order = bank.render_sectioned(variant, random.Random(seed))
        if order[0] != pins.get("first") or order[-1] != pins.get("last"):
            problems.append(f"shuffle seed {seed} moved a pinned section")
            break
        if sorted(order) != names:
            problems.append(f"shuffle seed {seed} dropped or duplicated a section")
            break
    return problems


def _check_placeholders(text: str) -> list[str]:
    problems = []
    for token in ("{env_info}", "{extra}", "<<CACHE_BOUNDARY>>"):
        if text.count(token) != 1:
            problems.append(f"expected exactly one {token}, found {text.count(token)}")
    if problems:
        return problems
    # {extra} carries AGENTS.md, whose contract is "these override defaults", so it must
    # come last; everything volatile must sit after the cache boundary.
    if not text.index("<<CACHE_BOUNDARY>>") < text.index("{env_info}") < text.index("{extra}"):
        problems.append("cache boundary, {env_info} and {extra} are out of order")
    if not text.rstrip().endswith("{extra}"):
        problems.append("{extra} must be the final content of the prompt")
    try:
        text.format(env_info="ENV", extra="EXTRA")
    except (KeyError, IndexError, ValueError) as e:
        problems.append(f"does not render: unexpected brace placeholder ({e})")
    return problems


def _check_tool_names(text: str, schemas: dict) -> list[str]:
    mentioned = set(_TOOL_MENTION_RE.findall(text))
    if not mentioned:
        return ["names no tools at all — the tool-name check would be vacuous"]
    invented = mentioned - known_tool_names(schemas)
    return [f"names tools that do not exist: {', '.join(sorted(invented))}"] if invented else []


def _check_compaction_blocks(text: str) -> list[str]:
    """Only <summary> survives compaction, so its block structure is load-bearing.

    Both opening tags are also named in prose ("Every durable fact must appear in
    <summary>"), which leaves the closing tags as the only reliable anchors.
    """
    problems = []
    for tag in ("<scratchpad>", "</scratchpad>", "<summary>", "</summary>"):
        if tag not in text:
            problems.append(f"missing {tag}")
    if problems:
        return problems
    for tag in ("</scratchpad>", "</summary>"):
        if text.count(tag) != 1:
            problems.append(f"expected exactly one {tag}, found {text.count(tag)}")
    if text.index("</scratchpad>") > text.index("</summary>"):
        problems.append("the scratchpad block must close before the summary block")
    return problems


def _check_tool_descriptions(variant, manifest: dict, schemas: dict) -> list[str]:
    try:
        overrides = bank.tool_overrides(variant)
    except bank.VariantError as e:
        return [str(e)]
    low, high = (manifest.get("tool_descriptions") or {}).get("description_length_ratio", (0, 99))
    problems = []
    for name, override in overrides.items():
        if name not in schemas:
            problems.append(f"overrides unregistered tool '{name}'")
            continue
        schema = schemas[name]
        # A renamed parameter key would silently break execute(**tool_input).
        unknown = set(override["params"]) - set(schema.input_schema.get("properties") or {})
        if unknown:
            problems.append(f"{name} names parameters that do not exist: {', '.join(sorted(unknown))}")
        for param, param_text in override["params"].items():
            if not param_text.strip():
                problems.append(f"{name}.{param} has an empty description")
        description = override["description"]
        if description is None:
            continue
        if not description.strip():
            problems.append(f"{name} has an empty description")
            continue
        ratio = len(description) / len(schema.description)
        if not low <= ratio <= high:
            problems.append(f"{name} description is {ratio:.2f}x the built-in, outside [{low}, {high}]")
    return problems


def check_bank(axis: str | None = None) -> list[str]:
    """Run every contract check over the bank. Returns a flat list of failures."""
    manifest = bank.load_manifest()
    schemas = registry_schemas()
    failures: list[str] = []
    for name in ([axis] if axis else bank.CONTENT_AXES):
        variants = bank.load_axis(name)
        if len(variants) < 2:
            failures.append(f"{name}: bank has only {len(variants)} variant — randomization is a no-op")
        for variant in variants:
            failures.extend(check_variant(variant, manifest, schemas))
    failures.extend(_check_baselines())
    return failures


def _check_baselines() -> list[str]:
    """The baselines must reproduce prompts.py exactly, so editing one without the
    other is caught immediately rather than drifting silently."""
    problems = []
    for axis, expected in _BASELINE_TEXT.items():
        text, _ = bank.render_sectioned(bank.get(axis, bank.BASELINE_ID))
        if text != expected:
            problems.append(f"{axis}/{bank.BASELINE_ID}: no longer matches prompts.py verbatim")
    if bank.tool_overrides(bank.get("tool_descriptions", bank.BASELINE_ID)):
        problems.append(f"tool_descriptions/{bank.BASELINE_ID}: baseline must define no overrides")
    return problems
