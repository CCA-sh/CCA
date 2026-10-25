"""Resolve requested thinking policy into the level CCA sends to a backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cca.llm.catalog import resolve_model, split_selector


THINKING_LEVELS = ("auto", "off", "low", "medium", "high")
EFFECTIVE_THINKING_LEVELS = ("off", "low", "medium", "high")
CCA_DEFAULT_THINKING_LEVEL = "medium"


@dataclass(frozen=True)
class ThinkingResolution:
    """Requested, effective, and provenance information for thinking."""

    requested: str
    effective: str
    source: str
    support: str
    reason: str
    warning: str | None = None
    catalog_mode: str | None = None

    def to_payload(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def thinking_support(selector: str) -> tuple[str, str]:
    """Return CCA runtime support, not just catalog capability metadata."""
    from cca.llm.anthropic_backend import (
        _model_supports_thinking as _anthropic_supports,
        _model_uses_adaptive_thinking as _anthropic_adaptive,
    )
    from cca.llm.openai_backend import (
        _google_model_supports_thinking,
        _is_openai_reasoning_model,
    )

    provider, model_name = split_selector(selector)
    lowered = model_name.lower()
    if provider == "anthropic":
        if _anthropic_adaptive(lowered):
            return "adaptive", "thinking is adaptive; explicit levels select the same adaptive mode"
        if _anthropic_supports(lowered):
            return "full", "thinking supports low, medium, and high budgets"
        return "none", "the selected Anthropic model does not support extended thinking"
    if provider == "google":
        if _google_model_supports_thinking(lowered):
            return "full", "thinking supports explicit Gemini reasoning effort"
        return "none", "Gemini thinking requires a 2.5 or 3.x model"
    if provider == "openai":
        if _is_openai_reasoning_model(lowered):
            return "full", "reasoning supports explicit effort"
        return "none", "the selected OpenAI model does not support CCA reasoning controls"
    if provider == "litellm":
        from cca.llm.litellm_thinking import get_litellm_thinking_profile

        profile = get_litellm_thinking_profile(lowered)
        if profile:
            if profile.mode == "litellm-toggle":
                return (
                    "adaptive",
                    f"the routed {profile.family} model supports an on/off thinking mode; "
                    "explicit enabled levels select the same mode",
                )
            return (
                "full",
                f"the routed {profile.family} reasoning model supports explicit effort through LiteLLM",
            )
        return "none", "the selected LiteLLM route has no CCA thinking adapter"
    return "none", f"CCA has no thinking adapter for provider '{provider}'"


def resolve_thinking(
    selector: str,
    requested: str | None,
    configured_default: str = "auto",
) -> ThinkingResolution:
    """Resolve CLI/session intent without silently delegating to provider defaults.

    Precedence is explicit request, configured default, catalog default, then
    CCA's stable medium default for supported reasoning models. Models that say
    an effort is required cannot honor a literal off; CCA uses the lowest level
    its adapters can express and reports that fallback.
    """
    requested_value = (requested or "auto").lower()
    configured_value = (configured_default or "auto").lower()
    if requested_value not in THINKING_LEVELS:
        raise ValueError(f"Invalid thinking level: {requested_value}")
    if configured_value not in THINKING_LEVELS:
        raise ValueError(f"Invalid default_thinking_level: {configured_value}")

    support, support_reason = thinking_support(selector)
    model = resolve_model(selector)
    thinking = model.thinking if model else None
    catalog_mode = thinking.mode if thinking else None
    requires_effort = bool(thinking and thinking.requires_effort)

    if requested_value != "auto":
        candidate = requested_value
        source = "explicit"
        reason = "selected explicitly for this session"
    elif configured_value != "auto":
        candidate = configured_value
        source = "config"
        reason = "selected by default_thinking_level"
    elif thinking and thinking.default_level in EFFECTIVE_THINKING_LEVELS:
        candidate = thinking.default_level
        source = "catalog"
        reason = "selected by the bundled model default"
    elif support != "none":
        candidate = CCA_DEFAULT_THINKING_LEVEL
        source = "cca-default"
        reason = "selected by CCA's stable default for reasoning-capable models"
    else:
        candidate = "off"
        source = "unsupported"
        reason = support_reason

    if support == "none":
        warning = None
        if candidate != "off":
            warning = f"Requested {candidate}, but {support_reason}; using off."
        return ThinkingResolution(
            requested=requested_value,
            effective="off",
            source="unsupported",
            support=support,
            reason=support_reason,
            warning=warning,
            catalog_mode=catalog_mode,
        )

    if candidate == "off" and requires_effort:
        efforts = thinking.efforts if thinking else ()
        minimum = next((level for level in ("low", "medium", "high") if level in efforts), "low")
        return ThinkingResolution(
            requested=requested_value,
            effective=minimum,
            source="model-minimum",
            support=support,
            reason="the model requires a reasoning effort; true off is unavailable",
            warning=f"Thinking off is unavailable for {selector}; using {minimum}.",
            catalog_mode=catalog_mode,
        )

    return ThinkingResolution(
        requested=requested_value,
        effective=candidate,
        source=source,
        support=support,
        reason=reason,
        catalog_mode=catalog_mode,
    )
