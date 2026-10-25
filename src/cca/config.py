"""Configuration management. Stores config at ~/.cca/config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Path constants live in paths.py; re-exported here for backward compatibility
# so existing `from cca.config import CONFIG_DIR` imports keep working.
from cca.paths import (  # noqa: E402
    CCA_DIR as CONFIG_DIR,
    MEMORY_DIR,
    SKILLS_DIR,
    PLANS_DIR,
    SESSIONS_DIR,
    TRANSCRIPTS_DIR,
)

CONFIG_FILE = CONFIG_DIR / "config.yaml"
# Project discovery walks toward the home directory. This path must remain a
# global config even when CCA_DIR points elsewhere for an isolated run.
_HOME_CONFIG_FILE = Path.home() / ".cca" / "config.yaml"


@dataclass
class Config:
    # Model specification: e.g. "anthropic:claude-sonnet-4-6"
    model: str = "anthropic:claude-sonnet-4-6"
    # Per-provider API keys (override env vars)
    api_keys: dict[str, str] = field(default_factory=dict)
    # Base URL for custom OpenAI-compat endpoints
    base_url: str | None = None
    # Max tokens per LLM call
    max_tokens: int = 8192
    # Thinking policy when --think is omitted: auto | off | low | medium | high
    default_thinking_level: str = "auto"
    # Tools that are always allowed without prompting the user
    always_allow: list[str] = field(default_factory=list)
    # Tools that are never allowed
    never_allow: list[str] = field(default_factory=list)
    # Search provider: "duckduckgo", "brave", "serper"
    search_provider: str = "duckduckgo"
    # Search API key (for brave/serper)
    search_api_key: str | None = None
    # System prompt additions
    system_prompt_append: str = ""
    # Verbose tool output
    verbose: bool = False
    # Trace mode: show all LLM inputs, outputs, and tool results
    trace: bool = False
    # Theme: "dark" | "light"
    theme: str = "dark"
    # Vi/vim keybindings in the REPL input
    vim: bool = False
    # Estimated token count at which context compaction is triggered
    compact_threshold: int = 150_000
    compact_threshold_upper: int = 200_000
    # Hooks: event → list of hook defs  {command, matcher?, timeout?, async?}
    hooks: dict = field(default_factory=dict)
    # Skills: load from ~/.cca/skills/ and ~/.claude/skills/ and list in system prompt
    skills_enabled: bool = False
    # Feature toggles: harness behaviors (features:) and per-tool availability (tools:)
    features: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    # User-defined presets: name → {features: {...}, tools: {...}}
    presets: dict = field(default_factory=dict)
    # Subagent runtime policy. Agent definitions live in ~/.cca/agents and
    # project .cca/agents directories rather than inline configuration.
    subagents: dict = field(default_factory=dict)
    # Optional agent profile for the root session (overridden by --agent).
    agent: str | None = None
    # Optional lifecycle observability. Empty/off by default; see Observability docs.
    observability: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        """Load global config from disk, returning defaults if missing."""
        if not CONFIG_FILE.exists():
            return cls()
        with open(CONFIG_FILE) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def find_repo_config(cls, cwd: "str | Path") -> "Path | None":
        """Walk up from cwd to find the nearest .cca/config.yaml."""
        path = Path(cwd).resolve()
        excluded_global_paths = {
            CONFIG_FILE.expanduser().resolve(),
            _HOME_CONFIG_FILE.resolve(),
        }
        while True:
            candidate = path / ".cca" / "config.yaml"
            if candidate.exists() and candidate.resolve() not in excluded_global_paths:
                return candidate
            parent = path.parent
            if parent == path:
                return None
            path = parent

    @classmethod
    def load_merged(cls, cwd: "str | Path | None" = None) -> "Config":
        """Load global config, then overlay the nearest repo-level config.

        Only keys present in the repo file are applied — the repo config is
        sparse, not a full copy.  api_keys dicts are merged (repo wins on
        conflicts, global fills in the rest).
        """
        cfg = cls.load()
        cfg._repo_config_path = None  # type: ignore[attr-defined]
        cfg._repo_toggle_keys = set()  # type: ignore[attr-defined]
        repo_path = cls.find_repo_config(cwd or Path.cwd())
        if repo_path is not None:
            with open(repo_path) as f:
                repo_data = yaml.safe_load(f) or {}
            for k, v in repo_data.items():
                if k not in cls.__dataclass_fields__:
                    continue
                if k in ("api_keys", "presets", "subagents", "observability") and isinstance(v, dict):
                    setattr(cfg, k, {**getattr(cfg, k), **v})
                elif k in ("features", "tools") and isinstance(v, dict):
                    setattr(cfg, k, {**getattr(cfg, k), **v})
                    cfg._repo_toggle_keys |= {f"{k}.{key}" for key in v}
                else:
                    setattr(cfg, k, v)
            cfg._repo_config_path = repo_path  # type: ignore[attr-defined]
        return cfg

    def save(self) -> None:
        """Persist to global config file."""
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def get_api_key(self, provider: str) -> str | None:
        """Return API key for provider, preferring config over env vars."""
        value, _ = self.get_api_key_with_source(provider)
        return value

    def get_api_key_with_source(self, provider: str) -> tuple[str | None, str]:
        """Return the key and a safe provenance label (never the key itself)."""
        normalized = provider.lower()
        if normalized in self.api_keys:
            return self.api_keys[normalized], "config"
        from cca.llm.catalog import get_provider
        provider_info = get_provider(normalized)
        if provider_info:
            for env_key in provider_info.env_vars:
                value = os.environ.get(env_key)
                if value:
                    return value, f"env:{env_key}"
            if provider_info.keyless:
                return None, "keyless"
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_key = env_map.get(normalized)
        if env_key and os.environ.get(env_key):
            return os.environ[env_key], f"env:{env_key}"
        return None, "missing"


def ensure_dirs() -> None:
    """Create all required config/data directories."""
    from cca.paths import ensure_dirs as _ensure
    _ensure()
