---
title: Configuration
sidebar:
  order: 1
  label: Getting started
---

All settings live in one YAML file — `~/.cca/config.yaml` — with optional per-project overrides. Set values from the command line; you never need to edit the file by hand.

```bash
cca config set model anthropic:claude-sonnet-4-6
cca config set-key anthropic sk-ant-...
cca config show      # view the merged config (global + repo)
cca config path      # show which files are in effect
```

## All config keys

| Key | Default | What it does |
|---|---|---|
| `model` | `anthropic:claude-sonnet-4-6` | Model to use — see [Models & providers](models-and-providers.md) |
| `api_keys` | `{}` | Per-provider API keys, set via `cca config set-key <provider> <key>` |
| `base_url` | — | Base URL override for OpenAI-compatible endpoints |
| `max_tokens` | `8192` | Max tokens per LLM response (override per turn — see [Thinking mode](thinking-mode.md)) |
| `default_thinking_level` | `auto` | Policy when `--think` is omitted: `auto`, `off`, `low`, `medium`, or `high`; `auto` resolves and reports a model-aware default |
| `always_allow` | `[]` | Tools that run without a permission prompt |
| `never_allow` | `[]` | Tools that are always denied |
| `search_provider` | `duckduckgo` | Web search backend: `duckduckgo`, `brave`, or `serper` |
| `search_api_key` | — | API key for Brave or Serper — see [Web search](../agent-features/web-search.md) |
| `system_prompt_append` | `""` | Text appended to the system prompt every session |
| `verbose` | `false` | Show full tool output inline |
| `trace` | `false` | Show all LLM inputs, outputs, and tool results |
| `theme` | `dark` | Terminal theme: `dark` or `light` |
| `vim` | `false` | Vi keybindings in the REPL input — see [Key bindings](../agent-features/key-bindings.md) |
| `compact_threshold` | `150000` | Estimated context tokens that trigger [compaction](../agent-features/compaction.md) at end of turn |
| `compact_threshold_upper` | `200000` | Hard threshold checked mid-task — compacts immediately |
| `hooks` | `{}` | Lifecycle hooks — see [Hooks](../agent-features/hooks.md) |
| `skills_enabled` | `false` | Load skills from `~/.cca/skills/` and enable the Skill tool |
| `agent` | — | Run the root session with this named [agent profile](../agent-features/subagents.md) |
| `subagents` | `{}` | Default profile, maximum depth/concurrency, and background policy for [subagents](../agent-features/subagents.md) |
| `observability` | `{}` (off) | Optional lifecycle export to W&B Weave or an OTLP backend — see [Observability](../agent-features/observability.md) |

Examples:

```bash
cca config set max_tokens 16384
cca config set default_thinking_level low
cca config set always_allow "Read,Glob,Grep"
cca config set system_prompt_append "Always prefer Python 3.12+ idioms."
```

Values are coerced to the key's type: booleans accept `true/1/yes`, lists are comma-separated, and mappings accept YAML/JSON object syntax.

## Per-project overrides

Add `--repo` to write into `.cca/config.yaml` in the current directory instead of the global file:

```bash
cca config set --repo model anthropic:claude-opus-4-7
cca config set-key --repo anthropic sk-ant-...
```

At startup, `cca` walks up from the working directory to the nearest `.cca/config.yaml` and overlays it on the global config. The repo file is **sparse** — only the keys it contains are overridden; everything else falls through to global. `api_keys` dictionaries are merged, with repo keys winning on conflict.

This means you can commit a `.cca/config.yaml` with a project's preferred model and tool allowlist while each developer keeps their API keys global.

## Precedence

From lowest to highest:

1. Built-in defaults
2. Global `~/.cca/config.yaml`
3. Repo `.cca/config.yaml` (nearest one up the directory tree)
4. CLI flags for the current run (`--model`, `--base-url`, `--verbose`, `--trace`, `--observe`, …)

API keys resolve config first, then the environment variables declared by the bundled provider catalog. Existing variables such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `TOGETHER_API_KEY`, and `OPENROUTER_API_KEY` retain their behavior; additional direct providers are documented under [Models & providers](models-and-providers.md#api-keys).

## Data locations

Session data, memory, and logs live under `~/.cca/` — see [Sessions & traces](../agent-features/sessions-and-traces.md) for the layout and the environment variables that relocate it.
