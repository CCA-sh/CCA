"""Thinking policy resolution and transparent runtime settings."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from click.testing import CliRunner

from cca.cli import main
from cca.config import Config
from cca.llm.factory import create_backend
from cca.llm.thinking import resolve_thinking, thinking_support
from cca.llm.base import Message, ThinkingContent
from cca.llm.openai_backend import _to_openai_messages
from cca.permissions import PermissionManager
from cca.repl import Repl


def test_gemini_unspecified_uses_stable_cca_default():
    resolved = resolve_thinking("google:gemini-3.6-flash", None, "auto")
    assert resolved.requested == "auto"
    assert resolved.effective == "medium"
    assert resolved.source == "cca-default"


def test_gemini_required_effort_reports_off_fallback():
    resolved = resolve_thinking("google:gemini-3.6-flash", "off", "auto")
    assert resolved.requested == "off"
    assert resolved.effective == "low"
    assert resolved.source == "model-minimum"
    assert "true off is unavailable" in resolved.reason
    assert resolved.warning


def test_configured_default_precedes_catalog_and_cca_default():
    resolved = resolve_thinking("google:gemini-3.6-flash", "auto", "high")
    assert resolved.effective == "high"
    assert resolved.source == "config"


def test_non_reasoning_model_resolves_auto_to_off():
    resolved = resolve_thinking("openai:gpt-4o", None, "auto")
    assert resolved.effective == "off"
    assert resolved.source == "unsupported"


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("azure_ai/gpt-5.6-sol", "OpenAI"),
        ("gemini/gemini-3.7-flash", "Gemini"),
        ("vertex_ai/global/gemini-3.7-flash", "Gemini"),
        ("anthropic/claude-sonnet-4-6", "Claude"),
        ("azure_ai/DeepSeek-V4-Pro", "DeepSeek"),
        ("groq/qwen/qwen3-32b", "Qwen"),
        ("azure_ai/Kimi-K2.7-Code", "Kimi"),
        ("fireworks_ai/minimax-m3", "MiniMax"),
        ("fireworks_ai/glm-5p2", "GLM"),
        ("xai/grok-4.3", "Grok"),
        ("mistral/magistral-medium-latest", "Magistral"),
    ],
)
def test_litellm_major_reasoning_families_honor_explicit_level(
    model, family
):
    resolved = resolve_thinking(
        f"litellm:{model}", "high", "auto"
    )
    assert resolved.requested == "high"
    assert resolved.effective == "high"
    assert resolved.support == "full"
    assert resolved.source == "explicit"
    assert resolved.warning is None
    support, support_reason = thinking_support(f"litellm:{model}")
    assert support == "full"
    assert family in support_reason


def test_litellm_gemma_uses_toggle_based_thinking_support():
    resolved = resolve_thinking(
        "litellm:gemini/gemma-4-31b-it", "medium", "auto"
    )
    assert resolved.effective == "medium"
    assert resolved.support == "adaptive"
    assert resolved.catalog_mode == "litellm-toggle"


def test_groq_qwen36_maps_portable_levels_to_binary_provider_mode():
    selector = "litellm:groq/qwen/qwen3.6-27b"
    resolved = resolve_thinking(selector, "high", "auto")

    assert resolved.effective == "high"
    assert resolved.support == "adaptive"
    assert resolved.catalog_mode == "litellm-toggle"

    backend = create_backend(selector, api_key="test", base_url="https://proxy.example.test/v1")
    assert backend.describe_request_settings("off")["thinking_parameters"] == {
        "reasoning_effort": "none",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }
    assert backend.describe_request_settings("high")["thinking_parameters"] == {
        "reasoning_effort": "default",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }


def test_groq_qwen38_keeps_distinct_high_provider_effort():
    selector = "litellm:groq/qwen/qwen3.8-27b"
    resolved = resolve_thinking(selector, "high", "auto")
    backend = create_backend(selector, api_key="test", base_url="https://proxy.example.test/v1")

    assert resolved.effective == "high"
    assert resolved.support == "full"
    assert backend.describe_request_settings("high")["thinking_parameters"] == {
        "reasoning_effort": "high",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }


def test_glm53_internal_off_resolves_to_required_minimum():
    selector = "litellm:fireworks_ai/glm-5p3"
    resolved = resolve_thinking(selector, "off", "auto")
    backend = create_backend(selector, api_key="test", base_url="https://proxy.example.test/v1")

    assert resolved.effective == "low"
    assert resolved.source == "model-minimum"
    assert resolved.warning
    assert backend.describe_request_settings(resolved.effective)[
        "thinking_parameters"
    ] == {
        "reasoning_effort": "low",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }


@pytest.mark.parametrize(
    "selector",
    [
        "litellm:gemini/gemini-3.7-flash",
        "litellm:vertex_ai/global/gemini-3.7-flash",
    ],
)
def test_litellm_gemini_internal_off_resolves_to_required_minimum(selector):
    resolved = resolve_thinking(selector, "off", "auto")
    backend = create_backend(selector, api_key="test", base_url="https://proxy.example.test/v1")

    assert resolved.effective == "low"
    assert resolved.source == "model-minimum"
    assert resolved.warning
    assert backend.describe_request_settings(resolved.effective)[
        "thinking_parameters"
    ] == {
        "reasoning_effort": "low",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }


@pytest.mark.parametrize(
    "model",
    [
        "groq/llama-3.3-70b-versatile",
        "xai/grok-4-fast-non-reasoning",
        "gemini/gemini-3.1-flash-image",
        "fireworks_ai/accounts/fireworks/models/qwen3-embedding-8b",
        "anthropic/*",
        "mistral/zai-glm-5-2",
    ],
)
def test_litellm_non_reasoning_routes_stay_unsupported(model):
    resolved = resolve_thinking(
        f"litellm:{model}", "high", "auto"
    )
    assert resolved.effective == "off"
    assert resolved.support == "none"
    assert "selected LiteLLM route" in resolved.reason


def test_backend_description_matches_gemini_request_control():
    backend = create_backend("google:gemini-3.6-flash", api_key="test")
    settings = backend.describe_request_settings("medium")
    assert settings["transport"] == "openai-chat-completions"
    assert settings["thinking_parameters"] == {"reasoning_effort": "medium"}


@pytest.mark.parametrize(
    "model",
    [
        "azure_ai/gpt-5.6-sol",
        "gemini/gemini-3.7-flash",
        "vertex_ai/global/gemini-3.7-flash",
        "azure_ai/DeepSeek-V4-Pro",
        "groq/qwen/qwen3-32b",
        "azure_ai/Kimi-K2.7-Code",
        "fireworks_ai/minimax-m3",
        "fireworks_ai/glm-5p2",
        "xai/grok-4.3",
        "mistral/magistral-medium-latest",
    ],
)
def test_litellm_backend_sends_reasoning_effort_for_supported_family(model):
    backend = create_backend(
        f"litellm:{model}",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    settings = backend.describe_request_settings("high")
    assert settings["thinking_parameters"] == {
        "reasoning_effort": "high",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }


@pytest.mark.parametrize(
    ("level", "thinking_type"),
    [
        ("off", "disabled"),
        ("low", "enabled"),
        ("medium", "enabled"),
        ("high", "enabled"),
    ],
)
def test_litellm_gemma_uses_thinking_toggle(level, thinking_type):
    backend = create_backend(
        "litellm:gemini/gemma-4-31b-it",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    assert backend.describe_request_settings(level)["thinking_parameters"] == {
        "extra_body": {"thinking": {"type": thinking_type}}
    }


@pytest.mark.parametrize(
    ("model", "level", "expected"),
    [
        (
            "anthropic/claude-sonnet-4-6",
            "high",
            {
                "extra_body": {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                }
            },
        ),
        (
            "anthropic/claude-opus-5",
            "medium",
            {
                "extra_body": {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "medium"},
                }
            },
        ),
        (
            "anthropic/claude-opus-5",
            "off",
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
    ],
)
def test_litellm_claude_uses_anthropic_thinking_payload(model, level, expected):
    backend = create_backend(
        f"litellm:{model}",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    assert backend.describe_request_settings(level)["thinking_parameters"] == expected


def test_litellm_routed_gpt_uses_max_completion_tokens():
    backend = create_backend(
        "litellm:azure_ai/gpt-5.6-sol",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    settings = backend.describe_request_settings("high")
    assert settings["max_tokens_parameter"] == "max_completion_tokens"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "azure_ai/gpt-5.6-sol",
        "gemini/gemini-3.7-flash",
        "vertex_ai/global/gemini-3.7-flash",
        "azure_ai/DeepSeek-V4-Pro",
        "groq/qwen/qwen3-32b",
        "azure_ai/Kimi-K2.7-Code",
        "fireworks_ai/minimax-m3",
        "fireworks_ai/glm-5p2",
        "xai/grok-4.3",
        "mistral/magistral-medium-latest",
    ],
)
async def test_litellm_supported_family_serializes_reasoning_request(model):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        })

    backend = create_backend(
        f"litellm:{model}",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    backend.client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://proxy.example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await backend.complete(
        [Message(role="user", content="hello")],
        [],
        "system",
        max_tokens=256,
        think_level="high",
    )
    await backend.client.close()

    assert captured["model"] == model
    assert captured["reasoning_effort"] == "high"
    assert captured["allowed_openai_params"] == ["reasoning_effort"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "level", "expected_thinking", "expected_effort"),
    [
        (
            "anthropic/claude-opus-5",
            "medium",
            {"type": "adaptive"},
            "medium",
        ),
        ("gemini/gemma-4-31b-it", "medium", {"type": "enabled"}, None),
    ],
)
async def test_litellm_serializes_native_thinking_request(
    model, level, expected_thinking, expected_effort
):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        })

    backend = create_backend(
        f"litellm:{model}",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    backend.client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://proxy.example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await backend.complete(
        [Message(role="user", content="hello")],
        [],
        "system",
        max_tokens=256,
        think_level=level,
    )
    await backend.client.close()

    assert captured["model"] == model
    assert captured["thinking"] == expected_thinking
    if expected_effort is None:
        assert "output_config" not in captured
    else:
        assert captured["output_config"] == {"effort": expected_effort}
    assert "reasoning_effort" not in captured
    assert "allowed_openai_params" not in captured


@pytest.mark.asyncio
async def test_litellm_sonnet_5_off_is_serialized_not_omitted():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "anthropic/claude-sonnet-5",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        })

    backend = create_backend(
        "litellm:anthropic/claude-sonnet-5",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    backend.client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://proxy.example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await backend.complete(
        [Message(role="user", content="hello")],
        [],
        "system",
        max_tokens=256,
        think_level="off",
    )
    await backend.client.close()

    assert captured["thinking"] == {"type": "disabled"}
    assert "output_config" not in captured


def test_litellm_bedrock_reasoning_allows_tool_parameters():
    backend = create_backend(
        "litellm:bedrock/us.anthropic.claude-sonnet-4-6",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    settings = backend.describe_request_settings("high")
    assert settings["thinking_parameters"] == {
        "extra_body": {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "allowed_openai_params": ["tools", "tool_choice"],
        }
    }


@pytest.mark.asyncio
async def test_litellm_deepseek_reasoning_is_retained_for_replay():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "chatcmpl-deepseek",
            "object": "chat.completion",
            "created": 1,
            "model": "azure_ai/DeepSeek-V4-Pro",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "continuation state",
                },
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        })

    backend = create_backend(
        "litellm:azure_ai/DeepSeek-V4-Pro",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    backend.client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://proxy.example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    response = await backend.complete(
        [Message(role="user", content="hello")],
        [],
        "system",
        think_level="high",
    )
    await backend.client.close()

    thinking = response.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.provider_state == {
        "provider": "litellm",
        "replay_as_reasoning_content": True,
    }
    replayed = _to_openai_messages([
        Message(role="assistant", content=response.content)
    ])
    assert replayed[0]["reasoning_content"] == "continuation state"


def test_litellm_minimax_reasoning_is_retained_for_replay():
    backend = create_backend(
        "litellm:fireworks_ai/minimax-m3",
        api_key="test",
        base_url="https://proxy.example.test/v1",
    )
    response = backend.describe_request_settings("medium")
    assert response["thinking_parameters"] == {
        "reasoning_effort": "medium",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }

    from cca.llm.litellm_thinking import get_litellm_thinking_profile

    profile = get_litellm_thinking_profile("fireworks_ai/minimax-m3")
    assert profile is not None
    assert profile.replay_reasoning_content is True


def test_litellm_minimax_m2_cannot_disable_required_reasoning():
    resolved = resolve_thinking(
        "litellm:fireworks_ai/accounts/fireworks/models/minimax-m2",
        "off",
        "auto",
    )
    assert resolved.effective == "low"
    assert resolved.source == "model-minimum"


def test_runtime_settings_are_safe_and_resolved(tmp_path):
    cfg = Config(
        model="google:gemini-3.6-flash",
        api_keys={"google": "super-secret"},
    )
    cfg._repo_config_path = None
    backend = create_backend(cfg.model, api_key=cfg.get_api_key("google"))
    resolved = resolve_thinking(cfg.model, None, cfg.default_thinking_level)
    registry = SimpleNamespace(get_schemas=lambda **_: [SimpleNamespace(name="Read")])
    repl = Repl(
        backend=backend,
        registry=registry,
        config=cfg,
        permissions=PermissionManager(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        think_level=resolved.effective,
        thinking_resolution=resolved,
        run_mode="print",
    )

    settings = repl._runtime_settings_payload()
    assert settings["thinking"]["requested"] == "auto"
    assert settings["thinking"]["effective"] == "medium"
    assert settings["backend"]["thinking_parameters"] == {
        "reasoning_effort": "medium"
    }
    assert settings["backend"]["authentication"] == "config"
    assert "super-secret" not in str(settings)
    assert settings["tools"]["active"] == ["Read"]


def test_litellm_runtime_settings_clamp_configured_output_to_model_limit(tmp_path):
    cfg = Config(
        model="litellm:groq/qwen/qwen3.8-27b",
        max_tokens=32_768,
        api_keys={"litellm": "super-secret"},
    )
    cfg._repo_config_path = None
    backend = create_backend(
        cfg.model, api_key=cfg.get_api_key("litellm"), base_url="https://proxy.example.test/v1"
    )
    resolved = resolve_thinking(cfg.model, None, cfg.default_thinking_level)
    repl = Repl(
        backend=backend,
        registry=SimpleNamespace(get_schemas=lambda **_: []),
        config=cfg,
        permissions=PermissionManager(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        think_level=resolved.effective,
        thinking_resolution=resolved,
        run_mode="print",
    )

    settings = repl._runtime_settings_payload()
    assert settings["max_tokens"] == {
        "configured": 32_768,
        "effective_default": 16_384,
        "model_default": 16_384,
        "model_upper_limit": 16_384,
    }


def test_cli_help_advertises_auto_and_settings_report():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "auto|off|low|medium|high" in result.output
    assert "--show-settings" in result.output
    assert "--observe" in result.output


def test_config_rejects_unknown_default_thinking_level(tmp_path, monkeypatch):
    from cca import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    result = CliRunner().invoke(
        main, ["config", "set", "default_thinking_level", "sometimes"]
    )
    assert result.exit_code != 0
    assert "must be one of" in result.output


def test_isolated_config_does_not_reimport_home_global_as_repo(tmp_path, monkeypatch):
    from cca import config as config_mod

    home = tmp_path / "home"
    project = home / "work" / "repo"
    home_config = home / ".cca" / "config.yaml"
    isolated_config = tmp_path / "isolated" / "config.yaml"
    project.mkdir(parents=True)
    home_config.parent.mkdir(parents=True)
    home_config.write_text("max_tokens: 12345\n")

    monkeypatch.setattr(config_mod, "CONFIG_FILE", isolated_config)
    monkeypatch.setattr(config_mod, "_HOME_CONFIG_FILE", home_config)

    cfg = Config.load_merged(project)
    assert cfg.max_tokens == 8192
    assert cfg._repo_config_path is None
