"""Factory for creating LLM backends from model strings.

Model string format:  provider:model-name
  anthropic:claude-sonnet-4-6
  openai:gpt-4o
  openai:gpt-4o-mini
  groq:llama-3.3-70b-versatile
  xai:grok-4
  zai:glm-5.2
  moonshot:kimi-k2.7-code
  ollama:llama3.2              (uses http://localhost:11434/v1)
  mistral:mistral-large-latest
  custom:my-model              (requires --base-url)
  litellm:openai/gpt-5.6-sol   (uses LITE_LLM_KEY and LITE_LLM_URL)

You can also supply just the model name for common known models:
  claude-sonnet-4-6  → anthropic
  gpt-4o             → openai
  llama3.2           → ollama
"""

from __future__ import annotations

import os

from cca.llm.base import LLMBackend
from cca.llm.catalog import get_provider, infer_provider, split_selector

_PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "",  # no key needed
}


def _infer_provider(model: str) -> str:
    return infer_provider(model)


def create_backend(
    model_string: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBackend:
    """Create and return the appropriate LLM backend.

    Args:
        model_string: Either "provider:model" or just "model".
        api_key: Override API key. Falls back to env var.
        base_url: Override base URL for OpenAI-compat providers.

    Returns:
        A configured LLMBackend instance.
    """
    from cca.llm.anthropic_backend import AnthropicBackend
    from cca.llm.openai_backend import OpenAIBackend

    provider, model = split_selector(model_string)
    provider_info = get_provider(provider)

    # Resolve API key
    if api_key is None:
        env_keys = provider_info.env_vars if provider_info else ()
        if not env_keys:
            legacy_env_key = _PROVIDER_ENV_KEYS.get(provider, "")
            env_keys = (legacy_env_key,) if legacy_env_key else ()
        api_key = next((os.environ[key] for key in env_keys if os.environ.get(key)), None)

    # Resolve base URL for OpenAI-compat providers
    if base_url is None:
        if provider_info:
            base_url = next(
                (
                    os.environ[key]
                    for key in provider_info.base_url_env_vars
                    if os.environ.get(key)
                ),
                provider_info.base_url,
            )
        else:
            base_url = _PROVIDER_BASE_URLS.get(provider)

    if base_url is None and provider_info and provider_info.base_url_env_vars:
        raise ValueError(
            f"{provider_info.name} has no default endpoint; set "
            f"{' or '.join(provider_info.base_url_env_vars)}, or pass --base-url"
        )

    backend_kind = provider_info.backend if provider_info else "openai"
    if backend_kind == "anthropic":
        anthropic_base_url = base_url if provider != "anthropic" else None
        return AnthropicBackend(model=model, api_key=api_key, base_url=anthropic_base_url)

    # Everything else uses the OpenAI-compat backend
    return OpenAIBackend(model=model, api_key=api_key, base_url=base_url, provider=provider)
