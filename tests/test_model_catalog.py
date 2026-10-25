"""Bundled model catalog compatibility and CLI tests."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from cca.cli import main
from cca.config import Config
from cca.llm.anthropic_backend import AnthropicBackend
from cca.llm.catalog import (
    get_model,
    get_provider,
    infer_provider,
    list_models,
    list_providers,
    provider_model_count,
    resolve_model,
)
from cca.llm.factory import create_backend
from cca.llm.model_caps import get_model_max_output_tokens
from cca.llm.openai_backend import OpenAIBackend


def test_catalog_covers_requested_families():
    families = {model.family for model in list_models()}
    assert {"openai", "claude", "gemini", "grok", "glm", "kimi", "deepseek", "qwen"} <= families


def test_catalog_has_substantial_curated_coverage():
    assert len(list_providers()) >= 15
    assert sum(provider_model_count(provider.id) for provider in list_providers()) >= 500
    assert provider_model_count("anthropic") >= 20
    assert provider_model_count("openrouter") >= 200


def test_model_metadata_includes_thinking_and_limits():
    model = get_model("anthropic", "claude-opus-4-8")
    assert model is not None
    assert model.context_window == 1_000_000
    assert model.max_tokens == 128_000
    assert model.thinking is not None
    assert model.thinking.mode == "anthropic-adaptive"
    assert "max" in model.thinking.efforts


def test_gemini_37_flash_catalog_metadata():
    model = get_model("google", "gemini-3.7-flash")
    assert model is not None
    assert model.api == "google-generative-ai"
    assert model.context_window == 1_048_576
    assert model.max_tokens == 65_536
    assert model.supports_tools is True
    assert model.supports_computer_use is True
    assert model.cost.input == 0.75
    assert model.cost.output == 3.75
    assert model.cost.cache_read == 0.075
    assert model.thinking is not None
    assert model.thinking.mode == "google-level"
    assert model.thinking.efforts == ("low", "medium", "high")
    assert model.thinking.default_level == "medium"


def test_gemini_37_flash_openrouter_variants():
    standard = get_model("openrouter", "google/gemini-3.7-flash")
    batch = get_model("openrouter", "google/gemini-3.7-flash:batch")
    assert standard is not None
    assert batch is not None
    assert standard.context_window == batch.context_window == 1_048_576
    assert standard.max_tokens == batch.max_tokens == 65_536
    assert standard.supports_tools is batch.supports_tools is True
    assert standard.cost.input == 0.375
    assert standard.cost.output == 1.875
    assert batch.cost.input == 0.1875
    assert batch.cost.output == 0.9375
    assert standard.thinking is not None
    assert standard.thinking.efforts == ("low", "medium", "high")


def test_resolve_model_uses_provider_selector():
    model = resolve_model("openrouter:qwen/qwen3-coder")
    assert model is not None
    assert model.provider == "openrouter"
    assert model.family == "qwen"


def test_legacy_local_model_inference_is_preserved():
    assert infer_provider("qwen3:8b") == "ollama"
    assert infer_provider("deepseek-r1:14b") == "ollama"
    assert infer_provider("llama3.2") == "ollama"


def test_new_direct_family_inference():
    assert infer_provider("grok-4") == "xai"
    assert infer_provider("glm-5.2") == "zai"
    assert infer_provider("kimi-k2.7-code") == "moonshot"


def test_factory_uses_catalog_provider_endpoint():
    backend = create_backend("xai:grok-4", api_key="test")
    assert isinstance(backend, OpenAIBackend)
    assert str(backend.client.base_url) == "https://api.x.ai/v1/"


def test_factory_uses_anthropic_compatible_provider_endpoint():
    backend = create_backend("zai:glm-5.2", api_key="test")
    assert isinstance(backend, AnthropicBackend)
    assert str(backend.client.base_url) == "https://api.z.ai/api/anthropic/"


def test_litellm_provider_uses_key_and_custom_endpoint_environment(monkeypatch):
    monkeypatch.setenv("LITE_LLM_KEY", "proxy-test")
    monkeypatch.setenv("LITE_LLM_URL", "https://proxy.example.test/v1")

    cfg = Config()
    backend = create_backend(
        "litellm:azure_ai/gpt-5.6-sol",
        api_key=cfg.get_api_key("litellm"),
    )

    assert isinstance(backend, OpenAIBackend)
    assert cfg.get_api_key_with_source("litellm") == (
        "proxy-test",
        "env:LITE_LLM_KEY",
    )
    assert backend.model == "azure_ai/gpt-5.6-sol"
    assert backend.provider == "litellm"
    assert str(backend.client.base_url) == "https://proxy.example.test/v1/"


def test_litellm_provider_has_no_bundled_endpoint_or_routes():
    provider = get_provider("litellm")
    assert provider is not None
    assert provider.base_url is None
    assert provider.default_model is None
    assert provider.base_url_env_vars == ("LITE_LLM_URL",)
    assert provider.env_vars == ("LITE_LLM_KEY",)
    assert provider_model_count("litellm") == 0


def test_litellm_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("LITE_LLM_URL", raising=False)
    with pytest.raises(ValueError, match="LITE_LLM_URL"):
        create_backend("litellm:openai/gpt-5.6-sol", api_key="test")


def test_litellm_explicit_base_url_is_enough(monkeypatch):
    monkeypatch.delenv("LITE_LLM_URL", raising=False)
    backend = create_backend(
        "litellm:openai/gpt-5.6-sol", api_key="test", base_url="https://proxy.example.test/v1"
    )
    assert str(backend.client.base_url) == "https://proxy.example.test/v1/"


def test_litellm_routes_resolve_without_a_bundled_snapshot():
    model = get_model("litellm", "any-team/custom-route")
    assert model is not None
    assert model.selector == "litellm:any-team/custom-route"
    assert model.reasoning is False


def test_models_command_explains_litellm_routes_are_not_bundled():
    result = CliRunner().invoke(main, ["models", "litellm"])
    assert result.exit_code != 0
    assert "$LITE_LLM_URL/models" in result.output


@pytest.mark.parametrize(
    "model_id",
    [
        "azure_ai/gpt-5.6-sol",
        "gemini/gemini-3.7-flash",
        "vertex_ai/global/gemini-3.7-flash",
        "gemini/gemma-4-31b-it",
        "anthropic/claude-sonnet-4-6",
        "azure_ai/DeepSeek-V4-Pro",
        "groq/qwen/qwen3-32b",
        "azure_ai/Kimi-K2.7-Code",
        "fireworks_ai/minimax-m3",
        "fireworks_ai/deepseek-v4p1-flash",
        "fireworks_ai/glm-5p2",
        "xai/grok-4.3",
        "mistral/magistral-medium-latest",
    ],
)
def test_litellm_catalog_marks_major_thinking_families(model_id):
    model = get_model("litellm", model_id)
    assert model is not None
    assert model.reasoning is True
    assert model.thinking is not None
    assert model.thinking.efforts == ("low", "medium", "high")


def test_config_resolves_catalog_environment_keys(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "kimi-test")
    assert Config().get_api_key("moonshot") == "kimi-test"


def test_explicit_config_key_still_wins(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "environment")
    assert Config(api_keys={"xai": "configured"}).get_api_key("xai") == "configured"


def test_catalog_caps_enrich_existing_fallback_logic():
    assert get_model_max_output_tokens("anthropic:claude-opus-4-8") == (32_000, 128_000)
    assert get_model_max_output_tokens("xai:grok-4") == (32_000, 64_000)
    assert get_model_max_output_tokens(
        "litellm:fireworks_ai/deepseek-v4p1-flash"
    ) == (32_000, 64_000)


def test_models_command_lists_providers():
    result = CliRunner().invoke(main, ["models"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output
    assert "openrouter" in result.output


def test_models_command_lists_one_provider():
    result = CliRunner().invoke(main, ["models", "anthropic", "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert "anthropic:claude-" in result.output.replace("\n", "")
    assert "Showing 3 of" in result.output


def test_models_command_emits_model_json():
    result = CliRunner().invoke(main, ["models", "anthropic:claude-opus-4-8", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selector"] == "anthropic:claude-opus-4-8"
    assert payload["thinking"]["mode"] == "anthropic-adaptive"


def test_provider_metadata_keeps_existing_google_endpoint():
    provider = get_provider("google")
    assert provider is not None
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
