"""Lazy access to CCA's bundled provider and model metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Iterable

from cca.llm.catalog.types import ModelInfo, ProviderInfo


_DATA_PACKAGE = "cca.llm.catalog.data"

_MODEL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("grok-", "xai"),
    ("glm-", "zai"),
    ("kimi-", "moonshot"),
    ("moonshot-", "moonshot"),
    ("gemini-", "google"),
    ("llama", "ollama"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("gemma", "ollama"),
    ("qwen", "ollama"),
    ("phi", "ollama"),
    ("deepseek", "ollama"),
)


def _read_json(relative_path: str) -> object:
    resource = files(_DATA_PACKAGE).joinpath(relative_path)
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _provider_map() -> dict[str, ProviderInfo]:
    raw = _read_json("providers.json")
    if not isinstance(raw, list):
        raise ValueError("Bundled provider catalog must be a list")
    providers = [ProviderInfo.from_dict(item) for item in raw]
    return {provider.id: provider for provider in providers}


@lru_cache(maxsize=None)
def _model_map(provider: str) -> dict[str, ModelInfo]:
    if provider not in _provider_map():
        return {}
    resource = files(_DATA_PACKAGE).joinpath("models", f"{provider}.json")
    if not resource.is_file():
        return {}
    raw = json.loads(resource.read_text(encoding="utf-8"))
    # Some provider snapshots are stored in the OpenAI-compatible `/models`
    # response envelope so they can be refreshed without a bespoke transform.
    if isinstance(raw, dict):
        raw = raw.get("data")
    if not isinstance(raw, list):
        raise ValueError(
            f"Bundled model catalog for {provider} must be a list or a /models response"
        )
    models = [ModelInfo.from_dict(provider, item) for item in raw]
    return {model.id: model for model in models}


def list_providers() -> list[ProviderInfo]:
    return sorted(_provider_map().values(), key=lambda provider: provider.id)


def get_provider(provider: str) -> ProviderInfo | None:
    return _provider_map().get(provider.lower())


def list_models(
    provider: str | None = None,
    *,
    family: str | None = None,
    query: str | None = None,
) -> list[ModelInfo]:
    provider_ids: Iterable[str]
    if provider:
        provider_ids = (provider.lower(),)
    else:
        provider_ids = _provider_map()
    family_query = family.lower() if family else None
    text_query = query.lower() if query else None
    result: list[ModelInfo] = []
    for provider_id in provider_ids:
        for model in _model_map(provider_id).values():
            if family_query and model.family != family_query:
                continue
            if text_query and text_query not in model.selector.lower() and text_query not in model.name.lower():
                continue
            result.append(model)
    return sorted(result, key=lambda model: (model.provider, model.id.lower()))


def get_model(provider: str, model_id: str) -> ModelInfo | None:
    provider = provider.lower()
    model = _model_map(provider).get(model_id)
    if model is None and provider == "litellm":
        # Proxy routes are deployment-specific, so none are bundled; thinking
        # support is derived from the route name instead.
        return ModelInfo.from_dict(provider, {"id": model_id})
    return model


def infer_provider(model_id: str) -> str:
    lowered = model_id.lower()
    for prefix, provider in _MODEL_PREFIXES:
        if lowered.startswith(prefix):
            return provider
    return "openai"


def split_selector(selector: str) -> tuple[str, str]:
    if ":" in selector:
        provider, model_id = selector.split(":", 1)
        return provider.lower(), model_id
    return infer_provider(selector), selector


def resolve_model(selector: str) -> ModelInfo | None:
    provider, model_id = split_selector(selector)
    return get_model(provider, model_id)


def provider_model_count(provider: str) -> int:
    return len(_model_map(provider.lower()))
