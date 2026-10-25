---
title: Thinking mode
sidebar:
  order: 3
---

Some models can reason internally before replying — Anthropic calls it extended thinking, OpenAI calls it reasoning. `cca` exposes it as one knob that maps to whatever your current model supports:

```
/think high     # set the level for this session
/think          # show the current level and what your model supports
/think auto     # return to the resolved default
/think off      # request no reasoning
```

At launch: `cca --think high`, which also works with `--print` for headless runs.
If `--think` is omitted, the requested policy is `auto`.
Add `--reasoning-summary` when you also want the provider's readable summary
captured in the session and per-call trace:

```bash
cca --model google:gemini-3.6-flash --think high --reasoning-summary --print "Fix the tests"
```

The summary flag never changes the requested reasoning effort. It requires a
resolved non-off level and fails before the API call when the selected
provider/model does not support readable summaries.

## Levels

| Level | Meaning |
|---|---|
| `auto` | Resolve the model-aware default and report where it came from |
| `off` | Request no extended thinking |
| `low` | Light reasoning — faster, cheaper |
| `medium` | Balanced depth |
| `high` | Maximum reasoning for hard problems |

The level applies to every turn until changed. `auto` resolves in this order:

1. `default_thinking_level` from merged configuration, when it is not `auto`.
2. A bundled model-specific default, when the catalog declares one.
3. CCA's stable `medium` default for a reasoning-capable model.
4. `off` for a model whose CCA adapter has no reasoning controls.

Set a persistent preference with, for example,
`cca config set default_thinking_level low`. Keeping it at `auto` is recommended
because CCA re-resolves the policy whenever the model changes.

Some models require a reasoning effort and cannot honor a literal `off`. CCA
uses the lowest supported level in that case and reports the fallback, such as
`off -> low`; it does not omit the provider parameter and silently inherit an
unknown backend default.

## See what will be sent

Interactive startup shows the model, backend, endpoint, resolved thinking,
output-token limit, and permission mode. Use `/settings` for the complete safe
snapshot. For headless runs, a compact summary is written to stderr automatically;
add `--show-settings` for the full JSON snapshot:

```bash
cca --model google:gemini-3.6-flash --show-settings --print "Fix the tests"
```

The report distinguishes `requested`, `effective`, and `source`, and includes
the exact thinking field CCA will pass to the adapter (for example,
`{"reasoning_effort": "medium"}`). API keys are never printed. Machine mode
includes the same snapshot in its startup `ready` event.

## One-turn override

Include a keyword in any message to force `high` for just that turn:

> `ultrathink` · `think harder` · `think deeply` · `think longer` · `think more carefully` · `think step by step` · `think it through`

The next turn reverts to the session level.

## Model support

`cca` never silently ignores an unsupported combination. It reports an
unsupported request and resolves it to `off`; models that require reasoning are
reported with their lowest available fallback instead.

| Family | Models | Levels |
|---|---|---|
| Anthropic (token budget) | claude-3-7-sonnet, Claude 4.0–4.6 | `low` / `medium` / `high` map to thinking budgets of 4k / 8k / 32k tokens |
| Anthropic (adaptive) | claude-opus-4-7+, Claude 5.x | on/off only — any level enables adaptive thinking and the model chooses its own depth |
| Google Gemini | gemini-2.5-x, gemini-3.x | `low` / `medium` / `high` via `reasoning_effort` |
| OpenAI | gpt-5.5, gpt-5.4, gpt-5.4-mini | `low` / `medium` / `high` via `reasoning.effort` |
| LiteLLM routed models | GPT/o-series, Gemini, Claude, DeepSeek, Qwen/QwQ, Kimi, GLM, Grok, and Magistral reasoning routes | `low` / `medium` / `high` via LiteLLM's normalized `reasoning_effort` |

Not supported: Claude 3.5 and earlier, Gemini 2.0 and earlier, and non-5.x OpenAI models (gpt-4o, gpt-4-turbo, …). LiteLLM routes explicitly named `non-reasoning`, plus wildcard, image, audio/realtime, embedding, reranking, moderation/guard, and OCR routes, are also excluded from thinking detection.

When thinking is active, provider-returned reasoning text streams into the terminal
and is captured in the [LLM call log](../agent-features/sessions-and-traces.md#the-llm-call-log).
New logs store the unified `response.reasoning` object (`text`, `kind`, requested,
and captured status) and retain `response.thinking` for compatibility. `kind` is
`summary` for provider-generated summaries and `unknown` for legacy unclassified
reasoning. Opaque signatures/encrypted state are persisted only for conversation
continuity and are kept separate from readable text.

## Inline token budget

Separately from thinking, you can raise the **response token limit** (`max_tokens`, default 8192) for a single turn with `/budget <amount>` anywhere in a message:

```
/budget 500k
explain this codebase /budget 2m
```

Accepted forms: `/budget 500k`, `/budget 2.5m`, `/budget 100000`. `/budget` is a distinctive sigil rather than a natural-language phrase or a bare `+N`, so it can appear anywhere in the message without colliding with ordinary content — pasted code, diffs (a hunk header like `@@ -6,3 +6,5 @@` is never mistaken for a budget request), or prose. To set a budget that lasts for the session, run `/budget 500k` as its own command.
