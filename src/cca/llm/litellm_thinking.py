"""Thinking capability profiles for model IDs routed through LiteLLM.

LiteLLM exposes many provider-specific reasoning controls through its
OpenAI-compatible endpoint. Detection remains conservative: a family name
alone is insufficient when that family also ships non-reasoning, image, audio,
embedding, or safety models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LiteLLMThinkingProfile:
    family: str
    parameter: str = "reasoning_effort"
    mode: str = "litellm-effort"
    efforts: tuple[str, ...] = ("low", "medium", "high")
    default_level: str | None = None
    effort_map: dict[str, str] = field(default_factory=dict)
    # OpenAI-compatible thinking models return `reasoning_content` and expect it
    # echoed on later assistant messages; families whose reasoning travels
    # another way opt out explicitly.
    replay_reasoning_content: bool = True
    requires_effort: bool = False


_NON_CHAT_MARKERS = (
    "embedding",
    "embed-",
    "rerank",
    "image",
    "imagen",
    "dall-e",
    "whisper",
    "transcrib",
    "tts",
    "audio",
    "realtime",
    "live-preview",
    "moderation",
    "safeguard",
    "prompt-guard",
    "llama-guard",
    "ocr",
)


def _is_openai_reasoning_route(model: str) -> bool:
    return any(
        component.startswith(("gpt-5", "o1", "o3", "o4"))
        for component in re.split(r"[/.:]", model.lower())
    ) or "gpt-oss-" in model.lower()


def _is_claude_reasoning_route(model: str) -> bool:
    value = model.lower()
    return bool(
        re.search(r"claude-3[-.]7", value)
        or re.search(r"claude-(?:opus|sonnet|haiku)-4(?:\D|$)", value)
        or re.search(r"claude-4-(?:opus|sonnet|haiku)(?:\D|$)", value)
        or re.search(r"claude-[a-z]+-5(?:\D|$)", value)
    )


def _is_gemma_reasoning_route(model: str) -> bool:
    return bool(re.search(r"gemma-?4(?:\D|$)", model.lower()))


def _is_deepseek_reasoning_route(model: str) -> bool:
    value = model.lower()
    if "deepseek" not in value:
        return False
    return bool(
        "reasoner" in value
        or re.search(r"deepseek[^/]*(?:r1|v3[._-]?[12]|v4)", value)
    )


def _is_qwen_reasoning_route(model: str) -> bool:
    value = model.lower()
    if "qwen" not in value:
        return False
    return bool(
        "qwq" in value
        or "thinking" in value
        or "qwen3-coder" in value
        or re.search(r"qwen3(?:[p.-]?[5-9]|-30b|-32b)", value)
    )


def _is_kimi_reasoning_route(model: str) -> bool:
    value = model.lower()
    return bool(
        "kimi-k2-thinking" in value
        or re.search(r"kimi-k2[._-][5-9](?:\D|$)", value)
        or re.search(r"kimi-k3(?:\D|$)", value)
    )


def _minimax_reasoning_generation(model: str) -> int | None:
    match = re.search(r"minimax-m([23])(?:[._p-]?\d+)?(?:\D|$)", model.lower())
    return int(match.group(1)) if match else None


def _is_glm_reasoning_route(model: str) -> bool:
    value = model.lower()
    if "glm" not in value:
        return False
    return bool(
        re.search(r"glm-4[._p-]?[5-9](?:\D|$)", value)
        or re.search(r"glm-5(?:\D|$)", value)
        or re.search(r"glm-5(?:[._-]|p)[0-9]", value)
    )


def _is_grok_reasoning_route(model: str) -> bool:
    value = model.lower()
    if "non-reasoning" in value:
        return False
    return bool(
        re.search(r"grok-3-mini(?:\D|$)", value)
        or re.search(r"grok-4(?:\D|$)", value)
        or "grok-code-fast" in value
        or "grok-build" in value
    )


_FIREWORKS_OPEN_REASONING_FAMILIES = (
    ("nemotron", "Nemotron"),
    ("inkling", "Inkling"),
    ("muse-glimmer", "Muse"),
)

# Providers that carry reasoning some other way (signed blocks, opaque items,
# their own field), so an unregistered route behind them must not be sent
# `reasoning_content` on the strength of the field being present.
_REASONING_CONTENT_UNSAFE_PROVIDERS = (
    "openai/",
    "azure/",
    "anthropic/",
    "bedrock/",
    "mistral/",
    "groq/",
)


def _fireworks_open_reasoning_family(model: str) -> str | None:
    if not model.startswith("fireworks_ai/"):
        return None
    return next(
        (family for marker, family in _FIREWORKS_OPEN_REASONING_FAMILIES if marker in model),
        None,
    )


def get_litellm_thinking_profile(model: str) -> LiteLLMThinkingProfile | None:
    """Return a conservative reasoning profile for a LiteLLM route ID."""
    value = model.lower()
    if "*" in value or "non-reasoning" in value:
        return None
    if any(marker in value for marker in _NON_CHAT_MARKERS):
        return None
    # Mistral's adapter currently exposes the common effort control only for
    # Magistral. Models from other families hosted behind a ``mistral/`` route
    # must not inherit their native provider's capability by name alone.
    if value.startswith("mistral/") and "magistral" not in value:
        return None

    if _is_openai_reasoning_route(value):
        # Chat completions returns no readable reasoning to echo back.
        return LiteLLMThinkingProfile("OpenAI", replay_reasoning_content=False)
    if re.search(r"gemini-(?:2[.]5|3(?:[.-]|$))", value):
        # Gemini cannot disable thinking. LiteLLM renders
        # ``reasoning_effort=none`` as ``thinking_level: MINIMAL``, which
        # Gemini 3.x flash routes reject outright, including on CCA's internal
        # utility calls. Resolve an off request to the lowest enabled level,
        # matching the native ``google:`` provider profile.
        effort_map: dict[str, str] = {}
        if "gemini-3.8-flash" in value:
            # This route also renders ``reasoning_effort="none"``/``"low"`` as
            # ``thinking_level: MINIMAL``, which the model rejects outright.
            # ``off`` reaches here directly (unbumped) whenever the model is
            # not registered in CCA's catalog, e.g. CCA's internal utility
            # calls (structured-output formatting) request ``off`` explicitly
            # and this model has no catalog entry to trigger the usual
            # off-to-lowest-level bump. Map both straight to medium.
            effort_map = {"off": "medium", "low": "medium"}
        return LiteLLMThinkingProfile(
            "Gemini", requires_effort=True, effort_map=effort_map
        )
    if _is_gemma_reasoning_route(value):
        return LiteLLMThinkingProfile(
            "Gemma",
            parameter="thinking",
            mode="litellm-toggle",
            replay_reasoning_content=False,
        )
    if _is_claude_reasoning_route(value):
        # Anthropic continuity needs signed thinking blocks, not reasoning_content.
        return LiteLLMThinkingProfile(
            "Claude", parameter="thinking", replay_reasoning_content=False
        )
    if _is_deepseek_reasoning_route(value):
        return LiteLLMThinkingProfile(
            "DeepSeek", replay_reasoning_content=True
        )
    if _is_qwen_reasoning_route(value):
        # Groq exposes Qwen 3.6 as a binary thinking route. Its API accepts
        # ``none`` or ``default`` rather than the generic effort names. Keep
        # CCA's portable levels while mapping every enabled level to the one
        # enabled mode the route actually provides.
        if "groq/qwen/qwen3.6-27b" in value:
            return LiteLLMThinkingProfile(
                "Qwen",
                mode="litellm-toggle",
                default_level="medium",
                effort_map={
                    "off": "none",
                    "low": "default",
                    "medium": "default",
                    "high": "default",
                },
                replay_reasoning_content=False,
            )
        return LiteLLMThinkingProfile(
            "Qwen", replay_reasoning_content=not value.startswith("groq/")
        )
    if _is_kimi_reasoning_route(value):
        return LiteLLMThinkingProfile("Kimi")
    minimax_generation = _minimax_reasoning_generation(value)
    if minimax_generation is not None:
        return LiteLLMThinkingProfile(
            "MiniMax",
            replay_reasoning_content=True,
            requires_effort=minimax_generation == 2,
        )
    if _is_glm_reasoning_route(value):
        # GLM-5.3 is thinking-only. Fireworks rejects
        # ``reasoning_effort=none``, including on CCA's internal utility
        # calls, so resolve an off request to the lowest enabled level.
        requires_effort = bool(
            re.search(r"glm-5(?:[._-]|p)3(?:\D|$)", value)
        )
        return LiteLLMThinkingProfile(
            "GLM", requires_effort=requires_effort
        )
    if _is_grok_reasoning_route(value):
        return LiteLLMThinkingProfile("Grok", replay_reasoning_content=False)
    if "magistral" in value:
        return LiteLLMThinkingProfile("Magistral", replay_reasoning_content=False)
    open_family = _fireworks_open_reasoning_family(value)
    if open_family:
        # Whether these routes accept effort=none is unverified, so an off
        # request resolves to the lowest enabled level rather than risking a
        # rejected internal utility call.
        return LiteLLMThinkingProfile(open_family, requires_effort=True)
    return None


def replays_reasoning_content(model: str) -> bool:
    """Whether reasoning a route returns should be echoed back on later calls."""
    profile = get_litellm_thinking_profile(model)
    if profile is not None:
        return profile.replay_reasoning_content
    return not model.lower().startswith(_REASONING_CONTENT_UNSAFE_PROVIDERS)


__all__ = [
    "LiteLLMThinkingProfile",
    "get_litellm_thinking_profile",
    "replays_reasoning_content",
]
