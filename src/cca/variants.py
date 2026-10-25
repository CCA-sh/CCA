"""Variant bank — discovers, parses, and renders scaffold-surface variants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files as _package_files
from typing import Any, Iterator, Mapping

import yaml

from cca.paths import VARIANTS_DIR

# Axes backed by files on disk. `tool_order` is derived from a seed, not a bank.
CONTENT_AXES = ("system_prompt", "tool_descriptions", "compaction_prompt")

BASELINE_ID = "v00_baseline"

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_SECTION_RE = re.compile(r"^## @([\w-]+)[ \t]*(?:\{([^}]*)\})?[ \t]*$", re.M)
_PINNED_RE = re.compile(r"pinned\s*=\s*(first|last)")
# A trailing backslash joins with the next line, mirroring the `"""\` continuation
# style prompts.py itself uses. Lets variant authors wrap prose without changing
# the rendered text. Backslashes elsewhere in a line are untouched.
_CONTINUATION_RE = re.compile(r"\\\n")


class VariantError(ValueError):
    """Malformed variant file or unknown variant id. Always fatal."""


@dataclass(frozen=True)
class Section:
    name: str
    pinned: str | None  # "first" | "last" | None
    text: str


@dataclass(frozen=True)
class Variant:
    id: str
    axis: str
    source: str  # "bundled" | "user"
    meta: Mapping[str, Any]
    body: str  # text after frontmatter; empty for YAML-payload axes

    @property
    def summary(self) -> str:
        return str(self.meta.get("summary", ""))


def _bundled_root():
    # Directory name differs from this module's name on purpose: `cca/variants/`
    # as a data directory would collide with `cca/variants.py` in the importer.
    return _package_files("cca") / "variant_bank"


def _split_frontmatter(text: str, origin: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise VariantError(f"{origin}: missing '---' frontmatter block")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise VariantError(f"{origin}: frontmatter must be a mapping")
    return meta, text[match.end():]


def _parse_variant(text: str, axis: str, source: str, origin: str) -> Variant:
    if axis == "tool_descriptions":
        meta = yaml.safe_load(text) or {}
        if not isinstance(meta, dict):
            raise VariantError(f"{origin}: file must be a YAML mapping")
        body = ""
    else:
        meta, body = _split_frontmatter(text, origin)
    variant_id = meta.get("id")
    if not variant_id:
        raise VariantError(f"{origin}: missing 'id'")
    declared_axis = meta.get("axis")
    if declared_axis != axis:
        raise VariantError(
            f"{origin}: declares axis {declared_axis!r} but lives in the {axis!r} directory")
    return Variant(id=str(variant_id), axis=axis, source=source, meta=meta, body=body)


def _read_dir(root, axis: str, source: str) -> Iterator[Variant]:
    directory = root / axis
    try:
        if not directory.is_dir():
            return
        entries = list(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return
    for entry in sorted(entries, key=lambda e: e.name):
        if entry.name.startswith(".") or not entry.name.endswith((".md", ".yaml", ".yml")):
            continue
        yield _parse_variant(entry.read_text(encoding="utf-8"), axis, source, entry.name)


def load_axis(axis: str) -> list[Variant]:
    """All variants for one axis, user files shadowing bundled ones by id.

    Sorted by id so selection is reproducible regardless of filesystem order.
    """
    if axis not in CONTENT_AXES:
        raise VariantError(f"Unknown variant axis {axis!r}")
    by_id: dict[str, Variant] = {v.id: v for v in _read_dir(_bundled_root(), axis, "bundled")}
    by_id.update({v.id: v for v in _read_dir(VARIANTS_DIR, axis, "user")})
    if not by_id:
        raise VariantError(f"Variant bank for axis {axis!r} is empty")
    return sorted(by_id.values(), key=lambda v: v.id)


def get(axis: str, variant_id: str) -> Variant:
    for variant in load_axis(axis):
        if variant.id == variant_id:
            return variant
    known = ", ".join(v.id for v in load_axis(axis))
    raise VariantError(f"Unknown {axis} variant {variant_id!r}. Known: {known}")


def load_manifest() -> dict:
    return yaml.safe_load((_bundled_root() / "manifest.yaml").read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _join_continuations(text: str) -> str:
    return _CONTINUATION_RE.sub("", text)


def parse_sections(variant: Variant) -> list[Section]:
    """Split a sectioned variant body on its `## @name {pinned=...}` markers.

    The markers are scaffolding for ordering and are stripped from the output;
    only the section bodies reach the model.
    """
    marks = list(_SECTION_RE.finditer(variant.body))
    if not marks:
        raise VariantError(f"{variant.axis}/{variant.id}: no '## @name' sections found")
    sections = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(variant.body)
        attrs = mark.group(2) or ""
        pin = _PINNED_RE.search(attrs)
        sections.append(Section(
            name=mark.group(1),
            pinned=pin.group(1) if pin else None,
            text=_join_continuations(variant.body[mark.end():end]).strip("\n"),
        ))
    _validate_pins(variant, sections)
    return sections


def _validate_pins(variant: Variant, sections: list[Section]) -> None:
    """Structural sanity only. Which sections an axis *must* pin is manifest policy,
    enforced by the verifier — not every axis has reorderable parts."""
    names = [s.name for s in sections]
    if len(set(names)) != len(names):
        raise VariantError(f"{variant.axis}/{variant.id}: duplicate section names")
    for pin, expected_index in (("first", 0), ("last", len(sections) - 1)):
        pinned = [i for i, s in enumerate(sections) if s.pinned == pin]
        if len(pinned) > 1:
            raise VariantError(
                f"{variant.axis}/{variant.id}: {len(pinned)} sections marked pinned={pin}")
        if pinned and pinned[0] != expected_index:
            raise VariantError(
                f"{variant.axis}/{variant.id}: section '{sections[pinned[0]].name}' is "
                f"pinned={pin} but is not written {pin} in the file")


def render_sectioned(variant: Variant, rng=None) -> tuple[str, tuple[str, ...]]:
    """Render a sectioned variant, optionally shuffling its unpinned middle.

    Returns (text, section_order). Pinned sections never move: the role opener
    stays first and the environment block — which owns {env_info}/{extra} and
    the cache boundary — stays last.
    """
    sections = parse_sections(variant)
    head = [s for s in sections if s.pinned == "first"]
    tail = [s for s in sections if s.pinned == "last"]
    middle = [s for s in sections if s.pinned is None]
    if rng is not None:
        rng.shuffle(middle)
    ordered = head + middle + tail
    return "\n\n".join(s.text for s in ordered), tuple(s.name for s in ordered)


def tool_overrides(variant: Variant) -> dict[str, dict]:
    """Per-tool description overrides from a tool_descriptions variant.

    Deliberately partial: any tool absent here keeps its built-in description,
    so a variant need not be rewritten every time a tool is added.
    """
    tools = variant.meta.get("tools") or {}
    if not isinstance(tools, dict):
        raise VariantError(f"tool_descriptions/{variant.id}: 'tools' must be a mapping")
    overrides: dict[str, dict] = {}
    for name, spec in tools.items():
        if not isinstance(spec, dict):
            raise VariantError(f"tool_descriptions/{variant.id}: '{name}' must be a mapping")
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise VariantError(f"tool_descriptions/{variant.id}: '{name}.params' must be a mapping")
        overrides[name] = {
            "description": _join_continuations(spec["description"]).strip()
            if spec.get("description") else None,
            "params": {k: _join_continuations(str(v)).strip() for k, v in params.items()},
        }
    return overrides
