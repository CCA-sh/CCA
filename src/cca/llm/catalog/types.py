"""Typed provider and model catalog records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCost:
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ModelCost":
        value = value or {}
        return cls(
            input=float(value.get("input", 0)),
            output=float(value.get("output", 0)),
            cache_read=float(value.get("cache_read", 0)),
            cache_write=float(value.get("cache_write", 0)),
        )


@dataclass(frozen=True)
class ThinkingCapabilities:
    mode: str
    efforts: tuple[str, ...]
    default_level: str | None = None
    effort_map: dict[str, str] = field(default_factory=dict)
    effort_routing: dict[str, str] = field(default_factory=dict)
    effort_budgets: dict[str, int] = field(default_factory=dict)
    supports_display: bool = False
    suppress_when_off: bool = False
    requires_effort: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ThinkingCapabilities | None":
        if not value or not value.get("mode") or not value.get("efforts"):
            return None
        return cls(
            mode=str(value["mode"]),
            efforts=tuple(str(item) for item in value["efforts"]),
            default_level=value.get("default_level"),
            effort_map={str(key): str(item) for key, item in value.get("effort_map", {}).items()},
            effort_routing={str(key): str(item) for key, item in value.get("effort_routing", {}).items()},
            effort_budgets={str(key): int(item) for key, item in value.get("effort_budgets", {}).items()},
            supports_display=bool(value.get("supports_display", False)),
            suppress_when_off=bool(value.get("suppress_when_off", False)),
            requires_effort=bool(value.get("requires_effort", False)),
        )


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    id: str
    name: str
    api: str
    context_window: int | None
    max_tokens: int | None
    reasoning: bool
    input: tuple[str, ...]
    supports_tools: bool | None
    supports_computer_use: bool
    thinking: ThinkingCapabilities | None
    cost: ModelCost

    @property
    def selector(self) -> str:
        return f"{self.provider}:{self.id}"

    @property
    def family(self) -> str:
        value = self.id.lower()
        families = (
            ("claude", "claude"),
            ("gpt", "openai"),
            ("o1", "openai"),
            ("o3", "openai"),
            ("o4", "openai"),
            ("gemini", "gemini"),
            ("gemma", "gemini"),
            ("grok", "grok"),
            ("glm", "glm"),
            ("kimi", "kimi"),
            ("moonshot", "kimi"),
            ("deepseek", "deepseek"),
            ("qwen", "qwen"),
            ("qwq", "qwen"),
            ("llama", "llama"),
            ("mistral", "mistral"),
            ("mixtral", "mistral"),
            ("devstral", "mistral"),
            ("codestral", "mistral"),
            ("minimax", "minimax"),
        )
        leaf = value.rsplit("/", 1)[-1]
        for prefix, family in families:
            if leaf.startswith(prefix) or f"/{prefix}" in value:
                return family
        return "other"

    @classmethod
    def from_dict(cls, provider: str, value: dict[str, Any]) -> "ModelInfo":
        supports_tools = value.get("supports_tools")
        reasoning = bool(value.get("reasoning", False))
        thinking = ThinkingCapabilities.from_dict(value.get("thinking"))
        if provider == "litellm" and not thinking:
            from cca.llm.litellm_thinking import get_litellm_thinking_profile

            profile = get_litellm_thinking_profile(str(value["id"]))
            if profile:
                reasoning = True
                thinking = ThinkingCapabilities(
                    mode=profile.mode,
                    efforts=profile.efforts,
                    default_level=profile.default_level,
                    effort_map=profile.effort_map,
                    requires_effort=profile.requires_effort,
                )
        return cls(
            provider=provider,
            id=str(value["id"]),
            name=str(value.get("name") or value["id"]),
            api=str(value.get("api", "openai-completions")),
            context_window=value.get("context_window"),
            max_tokens=value.get("max_tokens"),
            reasoning=reasoning,
            input=tuple(str(item) for item in value.get("input", ["text"])),
            supports_tools=bool(supports_tools) if supports_tools is not None else None,
            supports_computer_use=bool(value.get("supports_computer_use", False)),
            thinking=thinking,
            cost=ModelCost.from_dict(value.get("cost")),
        )


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    backend: str
    base_url: str | None
    env_vars: tuple[str, ...]
    default_model: str | None
    base_url_env_vars: tuple[str, ...] = ()
    keyless: bool = False
    source_provider: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProviderInfo":
        return cls(
            id=str(value["id"]),
            name=str(value.get("name") or value["id"]),
            backend=str(value.get("backend", "openai")),
            base_url=value.get("base_url"),
            base_url_env_vars=tuple(
                str(item) for item in value.get("base_url_env_vars", [])
            ),
            env_vars=tuple(str(item) for item in value.get("env_vars", [])),
            default_model=value.get("default_model"),
            keyless=bool(value.get("keyless", False)),
            source_provider=value.get("source_provider"),
        )
