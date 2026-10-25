"""Generate CCA's curated model catalog from an oh-my-pi snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


PROVIDERS = (
    {
        "id": "anthropic",
        "name": "Anthropic",
        "backend": "anthropic",
        "base_url": "https://api.anthropic.com",
        "env_vars": ["ANTHROPIC_API_KEY"],
        "default_model": "claude-opus-4-8",
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "backend": "openai",
        "base_url": "https://api.openai.com/v1",
        "env_vars": ["OPENAI_API_KEY"],
        "default_model": "gpt-5.5",
    },
    {
        "id": "google",
        "name": "Google Gemini",
        "backend": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "default_model": "gemini-3.1-pro-preview",
    },
    {
        "id": "xai",
        "name": "xAI",
        "backend": "openai",
        "base_url": "https://api.x.ai/v1",
        "env_vars": ["XAI_API_KEY"],
        "default_model": "grok-4-fast-non-reasoning",
    },
    {
        "id": "zai",
        "name": "Z.AI",
        "backend": "anthropic",
        "base_url": "https://api.z.ai/api/anthropic",
        "env_vars": ["ZAI_API_KEY"],
        "default_model": "glm-5.2",
    },
    {
        "id": "moonshot",
        "name": "Moonshot AI",
        "backend": "openai",
        "base_url": "https://api.moonshot.ai/v1",
        "env_vars": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "default_model": "kimi-k2.7-code",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "backend": "openai",
        "base_url": "https://api.deepseek.com",
        "env_vars": ["DEEPSEEK_API_KEY"],
        "default_model": "deepseek-v4-pro",
    },
    {
        "id": "qwen-portal",
        "name": "Qwen Portal",
        "backend": "openai",
        "base_url": "https://portal.qwen.ai/v1",
        "env_vars": ["QWEN_PORTAL_API_KEY", "QWEN_OAUTH_TOKEN"],
        "default_model": "coder-model",
    },
    {
        "id": "alibaba-coding-plan",
        "name": "Alibaba Coding Plan",
        "backend": "openai",
        "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
        "env_vars": ["ALIBABA_CODING_PLAN_API_KEY"],
        "default_model": "qwen3.7-plus",
    },
    {
        "id": "groq",
        "name": "Groq",
        "backend": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "env_vars": ["GROQ_API_KEY"],
        "default_model": "openai/gpt-oss-120b",
    },
    {
        "id": "mistral",
        "name": "Mistral AI",
        "backend": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "env_vars": ["MISTRAL_API_KEY"],
        "default_model": "devstral-medium-latest",
    },
    {
        "id": "together",
        "name": "Together AI",
        "backend": "openai",
        "base_url": "https://api.together.xyz/v1",
        "env_vars": ["TOGETHER_API_KEY"],
        "default_model": "moonshotai/Kimi-K2.7-Code",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "backend": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "env_vars": ["OPENROUTER_API_KEY"],
        "default_model": "openai/gpt-5.5",
    },
    {
        "id": "fireworks",
        "name": "Fireworks AI",
        "backend": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_vars": ["FIREWORKS_API_KEY"],
        "default_model": "kimi-k2.7-code",
    },
    {
        "id": "cerebras",
        "name": "Cerebras",
        "backend": "openai",
        "base_url": "https://api.cerebras.ai/v1",
        "env_vars": ["CEREBRAS_API_KEY"],
        "default_model": "zai-glm-4.7",
    },
    {
        "id": "ollama",
        "name": "Ollama",
        "backend": "openai",
        "base_url": "http://localhost:11434/v1",
        "env_vars": [],
        "default_model": None,
        "keyless": True,
        "source_provider": None,
    },
    {
        "id": "lmstudio",
        "name": "LM Studio",
        "backend": "openai",
        "base_url": "http://localhost:1234/v1",
        "env_vars": [],
        "default_model": None,
        "keyless": True,
        "source_provider": None,
    },
)

OPENROUTER_FAMILY_PREFIXES = (
    "anthropic/",
    "deepseek/",
    "google/",
    "moonshotai/",
    "openai/",
    "qwen/",
    "x-ai/",
    "z-ai/",
)


def _source_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _thinking(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    rename = {
        "defaultLevel": "default_level",
        "effortMap": "effort_map",
        "effortRouting": "effort_routing",
        "effortBudgets": "effort_budgets",
        "supportsDisplay": "supports_display",
        "suppressWhenOff": "suppress_when_off",
        "requiresEffort": "requires_effort",
    }
    return {rename.get(key, key): item for key, item in value.items()}


def _model(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": value["id"],
        "name": value.get("name", value["id"]),
        "api": value.get("api", "openai-completions"),
        "reasoning": bool(value.get("reasoning", False)),
        "input": value.get("input", ["text"]),
        "cost": {
            "input": value.get("cost", {}).get("input", 0),
            "output": value.get("cost", {}).get("output", 0),
            "cache_read": value.get("cost", {}).get("cacheRead", 0),
            "cache_write": value.get("cost", {}).get("cacheWrite", 0),
        },
        "context_window": value.get("contextWindow"),
        "max_tokens": value.get("maxTokens"),
    }
    optional = {
        "supportsTools": "supports_tools",
        "supportsComputerUse": "supports_computer_use",
    }
    for source_key, target_key in optional.items():
        if source_key in value:
            result[target_key] = value[source_key]
    thinking = _thinking(value.get("thinking"))
    if thinking:
        result["thinking"] = thinking
    return result


def _provider_models(source: dict[str, Any], provider: dict[str, Any]) -> list[dict[str, Any]]:
    source_provider = provider.get("source_provider", provider["id"])
    if source_provider is None:
        return []
    raw_models = source.get(source_provider, {})
    if provider["id"] == "openrouter":
        raw_models = {
            model_id: value
            for model_id, value in raw_models.items()
            if model_id.lower().startswith(OPENROUTER_FAMILY_PREFIXES)
        }
    return [_model(raw_models[model_id]) for model_id in sorted(raw_models, key=str.lower)]


def generate(source_root: Path, output_root: Path) -> None:
    source_path = source_root / "packages" / "catalog" / "src" / "models.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    models_root = output_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    providers = []
    commit = _source_commit(source_root)
    for definition in PROVIDERS:
        provider = dict(definition)
        provider.setdefault("source_provider", provider["id"])
        models = _provider_models(source, provider)
        provider["model_count"] = len(models)
        providers.append(provider)
        (models_root / f"{provider['id']}.json").write_text(
            json.dumps(models, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    (output_root / "providers.json").write_text(
        json.dumps(providers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "SOURCE.json").write_text(
        json.dumps(
            {
                "project": "oh-my-pi",
                "repository": "https://github.com/can1357/oh-my-pi",
                "commit": commit,
                "source": "packages/catalog/src/models.json",
                "license": "MIT",
                "policy": "Selected direct providers plus requested model families on OpenRouter",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=repo_root.parent / "oh-my-pi")
    parser.add_argument("--output", type=Path, default=repo_root / "src" / "cca" / "llm" / "catalog" / "data")
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
