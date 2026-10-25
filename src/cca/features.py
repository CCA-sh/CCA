"""Feature toggles — resolves config/preset into an immutable FeatureSet."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

FEATURE_DEFAULTS: dict[str, bool] = {
    "memory": True,
    "hooks": True,
    "mcp": True,
    "compaction": True,
    "disk_spill": True,
    "background_bash": True,
    "plan_mode": True,
    "auto_approve": False,
}

CORE_TOOLS = frozenset({"Read", "Write", "Edit", "Glob", "Grep", "SearchInFile", "Bash"})

OPTIONAL_TOOLS = frozenset({
    "WebSearch", "WebFetch", "Agent", "Skill", "ToolSearch",
    "LSP", "NotebookEdit", "AskUserQuestion",
    "CronCreate", "CronDelete", "CronList",
    "Monitor", "ScheduleWakeup", "EnterWorktree", "ExitWorktree",
    "WorkCreate", "WorkGet", "WorkList", "WorkUpdate", "JobCancel", "JobOutput",
})

PLAN_TOOLS = frozenset({"EnterPlanMode", "ExitPlanMode"})

# CCA-shipped model-visible interfaces. Dynamic tools contributed by configured
# MCP servers are deliberately excluded from this product-level count.
MCP_BUILTIN_TOOLS = frozenset({
    "ListMcpResourcesTool", "ReadMcpResourceTool", "McpAuthTool",
})
BUILTIN_TOOLS = CORE_TOOLS | OPTIONAL_TOOLS | PLAN_TOOLS | MCP_BUILTIN_TOOLS
BUILTIN_TOOL_COUNT = len(BUILTIN_TOOLS)

_DETERMINISTIC = {
    "features": {
        "memory": False, "hooks": False, "mcp": False, "disk_spill": False,
        "background_bash": True, "compaction": True, "plan_mode": True,
        "auto_approve": True,
    },
    "tools": {name: False for name in OPTIONAL_TOOLS if name not in ("AskUserQuestion", "NotebookEdit")},
}

SHIPPED_PRESETS: dict[str, dict] = {
    "deterministic": _DETERMINISTIC,
    "sft": _DETERMINISTIC,  # undocumented alias
}


class FeatureConfigError(ValueError):
    """Invalid features:/tools:/presets: configuration. Always fatal at startup."""


@dataclass(frozen=True)
class FeatureSet:
    features: Mapping[str, bool]
    tools: Mapping[str, bool]  # complete map over OPTIONAL_TOOLS
    preset: str | None = None
    sources: Mapping[str, str] = field(default_factory=dict)

    def feature(self, name: str) -> bool:
        return self.features[name]

    def tool_enabled(self, name: str) -> bool:
        if name in PLAN_TOOLS:
            return self.features["plan_mode"]
        # Core tools and dynamically-registered names (MCP wrappers) are not
        # governed by the tools: section — they are always "enabled" here.
        return self.tools.get(name, True)

    def enabled_tools(self) -> list[str]:
        return sorted(n for n, on in self.tools.items() if on)

    def is_default(self) -> bool:
        return (self.preset is None
                and dict(self.features) == FEATURE_DEFAULTS
                and all(self.tools.values()))

    def to_payload(self) -> dict:
        return {
            "preset": self.preset,
            "features": dict(self.features),
            "enabled_tools": self.enabled_tools(),
        }

    def summary_line(self) -> str | None:
        """One-line description of deviations from defaults; None when default."""
        if self.is_default():
            return None
        if self.preset is not None:
            n_off = sum(1 for on in self.tools.values() if not on)
            return f"features: preset '{self.preset}' — {n_off} tools disabled · run /features for details"
        feats_off = [k for k, v in self.features.items() if v != FEATURE_DEFAULTS[k]]
        tools_off = [k for k, v in self.tools.items() if not v]
        parts = []
        if feats_off:
            parts.append(", ".join(f"{k} {'on' if self.features[k] else 'off'}" for k in sorted(feats_off)))
        if tools_off:
            parts.append(f"{len(tools_off)} tools off")
        return f"features: customized via config ({'; '.join(parts)}) · run /features for details"


def _validate_features(raw: dict) -> None:
    for key, value in raw.items():
        if key not in FEATURE_DEFAULTS:
            valid = ", ".join(sorted(FEATURE_DEFAULTS))
            raise FeatureConfigError(f"Unknown feature '{key}' in features: section. Valid keys: {valid}")
        if not isinstance(value, bool):
            raise FeatureConfigError(f"Feature '{key}' must be true or false, got {value!r}")


def _validate_tools(raw: dict) -> None:
    for name, value in raw.items():
        if name in PLAN_TOOLS:
            raise FeatureConfigError(
                f"'{name}' cannot be set in the tools: section — use features.plan_mode instead")
        if name in CORE_TOOLS:
            if value is False:
                raise FeatureConfigError(f"'{name}' is a core tool and cannot be disabled")
            continue
        if name not in OPTIONAL_TOOLS:
            valid = ", ".join(sorted(OPTIONAL_TOOLS))
            raise FeatureConfigError(f"Unknown tool '{name}' in tools: section. Valid names: {valid}")
        if not isinstance(value, bool):
            raise FeatureConfigError(f"Tool '{name}' must be true or false, got {value!r}")


def _lookup_preset(config: Any, name: str) -> dict:
    user_presets = dict(getattr(config, "presets", None) or {})
    for shadowed in user_presets:
        if shadowed in SHIPPED_PRESETS:
            raise FeatureConfigError(
                f"Preset '{shadowed}' in config collides with a shipped preset name")
    if name in SHIPPED_PRESETS:
        return SHIPPED_PRESETS[name]
    if name in user_presets:
        preset = user_presets[name]
        if not isinstance(preset, dict) or not set(preset) <= {"features", "tools"}:
            raise FeatureConfigError(
                f"Preset '{name}' must be a mapping with only 'features' and 'tools' keys")
        return preset
    known = ", ".join(sorted((set(SHIPPED_PRESETS) - {"sft"}) | set(user_presets)))
    raise FeatureConfigError(f"Unknown preset '{name}'. Known presets: {known}")


def resolve_features(config: Any = None, preset: str | None = None) -> FeatureSet:
    """Resolve config + optional preset into a validated, immutable FeatureSet.

    A preset REPLACES the config's features:/tools: sections entirely
    (replace-not-merge — see the design spec).
    """
    if preset is not None:
        source_label = f"preset {preset}"
        chosen = _lookup_preset(config, preset)
        raw_features = dict(chosen.get("features") or {})
        raw_tools = dict(chosen.get("tools") or {})
        repo_keys: set[str] = set()
    else:
        source_label = "~/.cca/config.yaml"
        raw_features = dict(getattr(config, "features", None) or {})
        raw_tools = dict(getattr(config, "tools", None) or {})
        repo_keys = set(getattr(config, "_repo_toggle_keys", None) or set())

    _validate_features(raw_features)
    _validate_tools(raw_tools)

    features = dict(FEATURE_DEFAULTS)
    sources: dict[str, str] = {f"features.{k}": "default" for k in FEATURE_DEFAULTS}
    for k, v in raw_features.items():
        features[k] = v
        key = f"features.{k}"
        sources[key] = ".cca/config.yaml" if key in repo_keys else source_label

    tools = {name: True for name in OPTIONAL_TOOLS}
    sources.update({f"tools.{n}": "default" for n in OPTIONAL_TOOLS})
    for n, v in raw_tools.items():
        if n in CORE_TOOLS:
            continue  # validated no-op (true only)
        tools[n] = v
        key = f"tools.{n}"
        sources[key] = ".cca/config.yaml" if key in repo_keys else source_label

    return FeatureSet(
        features=MappingProxyType(features),
        tools=MappingProxyType(tools),
        preset=preset,
        sources=MappingProxyType(sources),
    )
