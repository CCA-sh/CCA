---
title: Models & providers
sidebar:
  order: 2
---

`cca` works with any major LLM API and with local models. Point it at a model once and it persists:

```bash
cca config set model anthropic:claude-sonnet-4-6
cca config set-key anthropic sk-ant-...
```

Or override for a single run: `cca --model google:gemini-3.5-flash`.

## Bundled catalog

CCA ships a curated model catalog covering the major OpenAI, Claude, Gemini,
Grok, GLM, Kimi, DeepSeek, and Qwen families. It also includes useful views of
those families through Groq, Mistral, Together, OpenRouter, Fireworks, Cerebras,
and Alibaba Coding Plan.

```bash
cca models                                      # provider summary
cca models anthropic                            # one provider
cca models qwen                                 # search IDs and names
cca models --family glm --reasoning             # cross-provider filtering
cca models anthropic:claude-opus-4-8            # detailed metadata
cca models openrouter --limit 300 --json        # machine-readable output
```

The catalog records model IDs, API family, context and output limits, input
modalities, pricing, reasoning support, and model-specific thinking modes and
efforts. `/models` provides a compact view inside the interactive REPL.

The catalog is not a whitelist: arbitrary `provider:model` strings and custom
OpenAI-compatible endpoints continue to work. A listed model can still require
account access or a provider plan, and catalog metadata does not imply that CCA
implements every provider-specific feature the upstream model advertises.

## Model strings

A model string is `provider:model-name`:

| Example | Provider |
|---|---|
| `anthropic:claude-sonnet-4-6` | Anthropic |
| `openai:gpt-4o` | OpenAI |
| `google:gemini-3.1-pro` | Google Gemini |
| `groq:llama-3.3-70b-versatile` | Groq |
| `mistral:mistral-large-latest` | Mistral AI |
| `together:...` | Together AI |
| `openrouter:...` | OpenRouter |
| `ollama:llama3.2` | Ollama (local, no key needed) |
| `lmstudio:my-model` | LM Studio (local) |
| `xai:grok-4` | xAI |
| `zai:glm-5.2` | Z.AI's Anthropic-compatible endpoint |
| `moonshot:kimi-k2.7-code` | Moonshot AI |
| `deepseek:deepseek-v4-pro` | DeepSeek |
| `qwen-portal:coder-model` | Qwen Portal |
| `fireworks:kimi-k2.7-code` | Fireworks AI |
| `cerebras:zai-glm-4.7` | Cerebras |
| `litellm:openai/gpt-5.6-sol` | LiteLLM proxy (OpenAI-compatible) |

You can also give just a model name — `cca --model claude-sonnet-4-6` — and the provider is inferred from the name prefix (`claude-` → anthropic, `gpt-` → openai, `gemini-` → google, `llama`/`qwen`/`deepseek`/… → ollama).

## API keys

Two ways to supply a key, either works:

```bash
cca config set-key google YOUR_GOOGLE_API_KEY     # stored in ~/.cca/config.yaml
export GOOGLE_API_KEY=...                          # or just export the env var
```

Config-stored keys take precedence over environment variables.

### Config values versus environment variables

Values under `api_keys` are always treated as literal API keys. They are not
expanded as environment-variable names. To use environment variables, omit the
provider from `api_keys` (or leave `api_keys: {}`) and export the variable before
starting `cca`:

```yaml
api_keys: {}
```

```bash
export GEMINI_API_KEY="..."
export LITE_LLM_KEY="..."
```

Do not put the variable names in `api_keys`; this sends the names themselves as
credentials instead of reading their values from the environment:

```yaml
# Wrong: these are interpreted as literal API keys
api_keys:
  google: GEMINI_API_KEY
  litellm: LITE_LLM_KEY
```

| Provider | Env var fallback |
|---|---|
| anthropic | `ANTHROPIC_API_KEY` |
| openai | `OPENAI_API_KEY` |
| google | `GOOGLE_API_KEY`, then `GEMINI_API_KEY` |
| groq | `GROQ_API_KEY` |
| mistral | `MISTRAL_API_KEY` |
| together | `TOGETHER_API_KEY` |
| openrouter | `OPENROUTER_API_KEY` |
| xai | `XAI_API_KEY` |
| zai | `ZAI_API_KEY` |
| moonshot | `MOONSHOT_API_KEY`, then `KIMI_API_KEY` |
| deepseek | `DEEPSEEK_API_KEY` |
| qwen-portal | `QWEN_PORTAL_API_KEY`, then `QWEN_OAUTH_TOKEN` |
| alibaba-coding-plan | `ALIBABA_CODING_PLAN_API_KEY` |
| fireworks | `FIREWORKS_API_KEY` |
| cerebras | `CEREBRAS_API_KEY` |
| litellm | `LITE_LLM_KEY` |
| ollama | none needed |

## LiteLLM proxy

The `litellm` provider talks to your own [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy).
It has no default endpoint, so set `LITE_LLM_URL` (or pass `--base-url`, which
takes precedence):

```bash
export LITE_LLM_KEY=sk-...
export LITE_LLM_URL=https://your-litellm-proxy.example.com/v1
cca --model litellm:openai/gpt-5.6-sol
```

Route names depend on how your proxy is configured, so CCA does not bundle a
list of them. Ask the proxy for its routes instead:

```bash
curl -H "Authorization: Bearer $LITE_LLM_KEY" "$LITE_LLM_URL/models"
```

Route names are passed to the proxy unchanged; the `litellm:` prefix is only
CCA's provider selector.
Routed GPT/o-series, Gemini, Claude, DeepSeek, Qwen/QwQ, Kimi, GLM, Grok,
and Magistral reasoning IDs support CCA's `--think low|medium|high` control
through LiteLLM's normalized `reasoning_effort` parameter. CCA conservatively
excludes non-reasoning and non-chat routes rather than sending unsupported
thinking fields to every ID containing a family name.

## Local models

Ollama and LM Studio work out of the box against their default local ports (`localhost:11434` and `localhost:1234`):

```bash
cca config set model ollama:llama3.2
```

## Any OpenAI-compatible server

For anything else that speaks the OpenAI chat API — vLLM, SGLang, a proxy, your own gateway — combine an `openai:` model string with a base URL:

```bash
cca --model openai:my-model --base-url http://localhost:8000/v1
# or persist it:
cca config set base_url http://localhost:8000/v1
```

## Switching models mid-session

Inside the REPL, `/model` shows the current model and `/model openai:gpt-4o` switches (and persists to config). The conversation continues with the new model — message history is provider-agnostic.

## Catalog source

The bundled metadata is a normalized selection from the MIT-licensed
`oh-my-pi` catalog. Its exact source commit and selection policy are stored in
`src/cca/llm/catalog/data/SOURCE.json`; license attribution is in
`THIRD_PARTY_NOTICES.md`. Maintainers can refresh it with:

Generation changes package data only. It does not rewrite `config.yaml`, alter
the selected model, or change CCA's established request-shaping rules.

## Related

- [Thinking mode](thinking-mode.md) — which models support extended reasoning, and how levels map to each provider's API
- [Configuration](index.md) — `max_tokens` and other per-model knobs
