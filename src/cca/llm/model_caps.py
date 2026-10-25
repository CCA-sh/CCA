"""Per-model output token caps.

Returns (default_tokens, upper_limit) for a given model string.
"""

from __future__ import annotations


def get_model_max_output_tokens(model: str) -> tuple[int, int]:
    """Return (default_max_tokens, upper_limit) for the given model string.

    The caller should use default_max_tokens for normal requests and
    upper_limit when escalating after a max_tokens truncation.
    """
    # Strip provider prefix (e.g. "anthropic:claude-sonnet-4-6" → "claude-sonnet-4-6")
    m = model.split(":", 1)[-1].lower()

    if "groq/qwen/qwen3.8-27b" in m:
        default, upper = 16_384, 16_384
    elif "gemini/gemma-4-26b-a4b-it" in m or "gemini/gemma-4-31b-it" in m:
        default, upper = 32_000, 32_768
    elif "opus-4-6" in m:
        default, upper = 64_000, 128_000
    elif "sonnet-4-6" in m:
        default, upper = 32_000, 128_000
    elif "opus-4-5" in m or "sonnet-4" in m or "haiku-4" in m:
        default, upper = 32_000, 64_000
    elif "opus-4-1" in m or "opus-4" in m:
        default, upper = 32_000, 32_000
    elif "3-7-sonnet" in m:
        default, upper = 32_000, 64_000
    elif "3-5-sonnet" in m or "3-5-haiku" in m:
        default, upper = 8_192, 8_192
    elif "claude-3-opus" in m:
        default, upper = 4_096, 4_096
    elif "claude-3-sonnet" in m:
        default, upper = 8_192, 8_192
    elif "claude-3-haiku" in m:
        default, upper = 4_096, 4_096
    # GPT / OpenAI models
    elif "gpt-4o" in m:
        default, upper = 16_384, 16_384
    elif "gpt-4" in m:
        default, upper = 8_192, 8_192
    elif "gpt-3.5" in m:
        default, upper = 4_096, 4_096
    # Gemini
    elif "gemini" in m:
        default, upper = 8_192, 65_536
    # Generic fallback
    else:
        default, upper = 32_000, 64_000

    from cca.llm.catalog import resolve_model
    catalog_model = resolve_model(model)
    if catalog_model and catalog_model.max_tokens:
        upper = catalog_model.max_tokens
        default = min(default, upper)
    return default, upper


MIN_OUTPUT_TOKENS = 1_024


def get_effective_max_output_tokens(
    model: str,
    configured: int = 8192,
    override: int | None = None,
) -> int:
    """Resolve CCA's requested output budget without exceeding the model cap.

    Never returns below `MIN_OUTPUT_TOKENS` (clamped to the model's own
    upper limit, if that's somehow smaller). This is a backstop against any
    caller — including a mis-parsed inline token-budget override — asking
    for a budget too small to leave room for an actual reply once a
    reasoning model spends part of it on hidden thinking.
    """
    model_default, model_upper = get_model_max_output_tokens(model)
    requested = override
    if requested is None:
        requested = configured if configured != 8192 else model_default
    resolved = min(requested, model_upper)
    floor = min(MIN_OUTPUT_TOKENS, model_upper)
    return max(resolved, floor)
