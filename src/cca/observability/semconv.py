"""Map CCA model selectors onto GenAI semantic-convention span attributes."""

from __future__ import annotations

from typing import Any


def model_attributes(selector: str | None) -> dict[str, Any]:
    """Provider and bare model id for a `provider:model` selector.

    Backends match cost and model metadata on the bare id, so the selector
    prefix must not leak into `gen_ai.request.model`.
    """
    if not selector:
        return {}
    from cca.llm.catalog import split_selector

    provider, model_id = split_selector(str(selector))
    return {
        "gen_ai.system": provider,
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model_id,
        "cca.model_selector": str(selector),
    }


def response_model_attributes(selector: str | None) -> dict[str, Any]:
    if not selector:
        return {}
    from cca.llm.catalog import split_selector

    _, model_id = split_selector(str(selector))
    return {"gen_ai.response.model": model_id}
