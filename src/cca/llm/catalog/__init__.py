"""Public model catalog API."""

from cca.llm.catalog.registry import (
    get_model,
    get_provider,
    infer_provider,
    list_models,
    list_providers,
    provider_model_count,
    resolve_model,
    split_selector,
)
from cca.llm.catalog.types import ModelCost, ModelInfo, ProviderInfo, ThinkingCapabilities

__all__ = [
    "ModelCost",
    "ModelInfo",
    "ProviderInfo",
    "ThinkingCapabilities",
    "get_model",
    "get_provider",
    "infer_provider",
    "list_models",
    "list_providers",
    "provider_model_count",
    "resolve_model",
    "split_selector",
]
