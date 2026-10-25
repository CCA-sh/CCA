"""Per-run randomization — resolves an axis spec and seed into a frozen RandomizationPlan."""

from __future__ import annotations

import copy
import random
import secrets
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from cca import variants as variant_bank

if TYPE_CHECKING:
    from cca.llm.base import ToolSchema

AXES = ("system_prompt", "tool_descriptions", "tool_order", "compaction_prompt")

_OFF_SPECS = frozenset({"", "off", "none", "false", "0"})
_ALL_SPECS = frozenset({"all", "true", "1"})


class RandomizationError(ValueError):
    """Invalid --randomize spec. Always fatal at startup.

    Same rationale as FeatureConfigError: a silently-ignored typo would poison an
    entire trace-collection run.
    """


@dataclass(frozen=True)
class RandomizationPlan:
    seed: int | None = None
    seed_source: str = "none"  # "auto" | "explicit" | "none"
    axes: frozenset[str] = frozenset()
    variant_ids: Mapping[str, str] = field(default_factory=dict)
    system_prompt: str | None = None
    section_order: tuple[str, ...] = ()
    compaction_prompt: str | None = None
    tool_descriptions: Mapping[str, dict] = field(default_factory=dict)
    bank_version: int | None = None

    @classmethod
    def identity(cls) -> "RandomizationPlan":
        return cls()

    def is_identity(self) -> bool:
        return not self.axes

    # -- application ------------------------------------------------------

    def apply_to_schemas(self, schemas: "list[ToolSchema]") -> "list[ToolSchema]":
        """Return schemas with randomized descriptions and ordering applied.

        Never touches tool names or input_schema property names — those are
        load-bearing identifiers wired through prompts, permissions, and the
        tool dispatcher.
        """
        if self.is_identity():
            return schemas
        result = [self._rewrite(s) for s in schemas]
        if "tool_order" in self.axes:
            result = self._reorder(result)
        return result

    def _rewrite(self, schema: "ToolSchema") -> "ToolSchema":
        override = self.tool_descriptions.get(schema.name)
        if not override:
            return schema
        description = override.get("description") or schema.description
        input_schema = schema.input_schema
        params = override.get("params") or {}
        if params:
            input_schema = copy.deepcopy(input_schema)
            properties = input_schema.get("properties") or {}
            for param, text in params.items():
                if param in properties:
                    properties[param]["description"] = text
        return replace(schema, description=description, input_schema=input_schema)

    def _reorder(self, schemas: "list[ToolSchema]") -> "list[ToolSchema]":
        # Permute over sorted names so the result depends only on the seed and the
        # name set — never on registration order.
        names = sorted(s.name for s in schemas)
        self._rng("tool_order").shuffle(names)
        rank = {name: i for i, name in enumerate(names)}
        return sorted(schemas, key=lambda s: rank[s.name])

    def _rng(self, stream: str) -> random.Random:
        return random.Random(f"{self.seed}:{stream}")

    # -- reporting --------------------------------------------------------

    def to_payload(self) -> dict:
        return {
            "seed": self.seed,
            "seed_source": self.seed_source,
            "bank_version": self.bank_version,
            "axes": sorted(self.axes),
            "variants": dict(self.variant_ids),
            "section_order": list(self.section_order),
        }

    def summary_line(self) -> str | None:
        if self.is_identity():
            return None
        chosen = ", ".join(f"{axis}={vid}" for axis, vid in sorted(self.variant_ids.items()))
        detail = f" ({chosen})" if chosen else ""
        return f"seed {self.seed} · {len(self.axes)} axes{detail}"


def parse_axes(spec: str | None) -> frozenset[str]:
    """Parse a --randomize spec into a validated axis set. None/off → empty."""
    if spec is None:
        return frozenset()
    normalized = spec.strip().lower()
    if normalized in _OFF_SPECS:
        return frozenset()
    if normalized in _ALL_SPECS:
        return frozenset(AXES)
    axes = set()
    for raw in normalized.split(","):
        name = raw.strip().replace("-", "_")
        if not name:
            continue
        if name not in AXES:
            raise RandomizationError(
                f"Unknown randomization axis '{raw.strip()}'. "
                f"Valid axes: {', '.join(sorted(AXES))} (or 'all' / 'off')")
        axes.add(name)
    if not axes:
        raise RandomizationError(
            f"Empty randomization spec. Valid axes: {', '.join(sorted(AXES))} (or 'all' / 'off')")
    return frozenset(axes)


def resolve_randomization(axes_spec: str | None = None, seed: int | None = None) -> RandomizationPlan:
    """Resolve a spec + seed into a frozen plan. No bank I/O when inactive."""
    axes = parse_axes(axes_spec)
    if not axes:
        return RandomizationPlan.identity()

    resolved_seed = secrets.randbits(32) if seed is None else seed
    manifest = variant_bank.load_manifest()

    variant_ids: dict[str, str] = {}
    system_prompt = None
    section_order: tuple[str, ...] = ()
    compaction_prompt = None
    tool_descriptions: dict[str, dict] = {}

    for axis in variant_bank.CONTENT_AXES:
        if axis not in axes:
            continue
        # One RNG stream per axis, so adding an axis never shifts another's choice.
        chosen = random.Random(f"{resolved_seed}:{axis}").choice(variant_bank.load_axis(axis))
        variant_ids[axis] = chosen.id
        if axis == "system_prompt":
            order_rng = random.Random(f"{resolved_seed}:system_prompt:order")
            system_prompt, section_order = variant_bank.render_sectioned(chosen, order_rng)
        elif axis == "compaction_prompt":
            compaction_prompt, _ = variant_bank.render_sectioned(chosen)
        else:
            tool_descriptions = variant_bank.tool_overrides(chosen)

    return RandomizationPlan(
        seed=resolved_seed,
        seed_source="explicit" if seed is not None else "auto",
        axes=axes,
        variant_ids=MappingProxyType(variant_ids),
        system_prompt=system_prompt,
        section_order=section_order,
        compaction_prompt=compaction_prompt,
        tool_descriptions=MappingProxyType(tool_descriptions),
        bank_version=manifest.get("bank_version"),
    )
