---
title: Feature toggles & presets
sidebar:
  order: 4
---

Two config sections control exactly what is in your agent:

- `features:` — **harness behaviors**: what gets injected into the conversation, how permissions flow, how output is post-processed.
- `tools:` — the **action space**: which tools the model sees.

## Where configuration goes

Both sections live in your regular config files:

- `~/.cca/config.yaml` — global, applies everywhere.
- `.cca/config.yaml` in a repo — per-project overlay. Only the keys present in the repo file are applied; they win over the global file key by key.

Both sections are **sparse**: anything you don't list keeps its built-in default. You never enumerate everything — write only what you change. A missing section means "all defaults", which is why a fresh config file has neither.

## Two ways to configure

**1. Set keys directly in your config.** This is the right choice when you simply want your everyday agent to behave differently:

```yaml
# ~/.cca/config.yaml
features:
  memory: false
tools:
  WebSearch: false
  Agent: false
```

From now on, every `cca` run uses this shape. Nothing else is required.

**2. Define a named preset.** This is the right choice for an alternate setup you switch into — an ablation, a trace-collection run, a locked-down configuration you share with others:

```yaml
# ~/.cca/config.yaml
presets:
  my_ablation:
    features: {memory: false, mcp: false}
    tools: {WebSearch: false, Agent: false}
```

Apply it with `cca --preset my_ablation`, or set `CCA_PRESET=my_ablation` for orchestrators that cannot pass flags.

A preset **replaces** your config's `features:`/`tools:` sections entirely — no merging. Your config defines your default agent; `--preset X` ignores those two sections and uses exactly the preset instead. There are deliberately no per-key CLI override flags: one-off variations should be named presets, because "I ran `my_ablation`" is reproducible and a pile of CLI flags is not.

## Reference: `features:` keys

| Key               | Default | Controls                                                                                              |
| ----------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `memory`          | on      | Cross-session memory extraction and injection of memory facts into the system prompt                  |
| `hooks`           | on      | Lifecycle hook execution (`hooks:` config section)                                                    |
| `mcp`             | on      | MCP server loading — a tool *source*, so all MCP tools appear/disappear together                      |
| `compaction`      | on      | Automatic context compaction (explicit `/compact` always works)                                       |
| `disk_spill`      | on      | Persisting oversized tool output to disk and replacing it with a stub; off = full output stays inline |
| `background_bash` | on      | Bash's `run_in_background` parameter and the background notification queue                            |
| `plan_mode`       | on      | The `EnterPlanMode`/`ExitPlanMode` tools and plan-mode auto-toggling, as one unit                     |
| `auto_approve`    | off     | Skip all tool permission prompts (config equivalent of `--auto-approve`)                              |

## Reference: `tools:` names

Keys are the exact tool names the model sees, mapped to `true`/`false`. Unlisted tools default to **on**.

Toggleable tools (21): `WebSearch`, `WebFetch`, `Agent`, `Skill`, `ToolSearch`, `LSP`, `NotebookEdit`, `AskUserQuestion`, `CronCreate`, `CronDelete`, `CronList`, `Monitor`, `ScheduleWakeup`, `EnterWorktree`, `ExitWorktree`, `WorkCreate`, `WorkGet`, `WorkList`, `WorkUpdate`, `JobCancel`, `JobOutput`.

Not toggleable:

- **Core tools** — `Read`, `Write`, `Edit`, `Glob`, `Grep`, `SearchInFile`, `Bash`. Without these it is not a coding agent; setting one to `false` is a startup error.
- **Plan tools** — `EnterPlanMode`/`ExitPlanMode` are controlled by `features.plan_mode`, not here; listing them in `tools:` is a startup error.
- **Session recording** — always on; every session saves to `~/.cca/sessions/`.

## The `deterministic` preset

One preset ships with cca. `--preset deterministic` produces stationary, single-actor trajectories for trace collection: the system prompt is identical across rollouts of the same task, and the action space is fixed. It is:

- features: `memory`, `hooks`, `mcp`, `disk_spill` **off**; `compaction`, `plan_mode`, `auto_approve`, `background_bash` **on**
- tools: everything off except `AskUserQuestion` and `NotebookEdit`

The resulting action space is exactly: `Read`, `Write`, `Edit`, `Glob`, `Grep`, `SearchInFile`, `Bash`, `NotebookEdit`, `AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode`.

`background_bash` stays **on**: long-running commands are part of real coding work, so backgrounding belongs in the training distribution. The cost is that completion notices arrive as `<system-reminder>` turns the model did not produce, at timing-dependent points — so the preset no longer guarantees "nothing injected out-of-band". If your collection run needs that guarantee, define a preset that copies `deterministic` with `background_bash: false`.

Because a preset ignores the config files' `features:`/`tools:` sections entirely, `cca --preset deterministic` builds the identical agent on any machine.

**Warning:** `deterministic` implies `auto_approve: true` — it skips every tool permission prompt. It is intended for sandboxed trace-collection runs, not for interactive use on your own machine.

## Validation

Configuration mistakes are hard startup errors, never silent — a typo that silently did nothing could poison an entire collection run:

```
Feature config error: Unknown feature 'memmory' in features: section. Valid keys: auto_approve, background_bash, ...
```

The same applies to unknown tool names, non-boolean values, unknown preset names, disabling a core tool, and defining a preset whose name collides with a shipped one.

## Seeing what you're running

When anything deviates from defaults, cca announces it once at startup; when everything is default, nothing is printed.

- **Interactive** — one banner line, e.g. `features: preset 'deterministic' — 20 tools disabled · run /features for details`
- **`--print`** — the same line on stderr, keeping stdout clean for piping
- **`--machine`** — the initial `ready` event carries a `features` field with the full resolved state
- **Every session file** — `~/.cca/sessions/<id>.json` contains a `features` block recording the preset, feature flags, and exact enabled tool list the session ran under. This includes sessions archived before compaction and subagent (`-sub-*`) sessions.
- **Every LLM call record** — each line in `~/.cca/llm-calls/session_*.jsonl` carries the same `features` payload, captured at the moment of the call. Traces are self-describing: no join against session files is needed, and a session resumed under a different configuration shows exactly which calls ran under which config.

For the full breakdown — every key, its value, and where it came from (`default`, config file, or `preset <name>`) — run `cca features` (optionally with `--preset NAME` to preview one) or `/features` inside the REPL.

## Python usage

No CLI or env vars needed — resolve once and pass the result:

```python
from cca.config import Config
from cca.features import resolve_features
from cca.tools import build_registry

cfg = Config.load_merged("/path/to/project")
fs = resolve_features(cfg, preset="deterministic")
registry = build_registry(config=cfg, feature_set=fs)

fs.feature("memory")           # False
fs.enabled_tools()             # ['AskUserQuestion', 'NotebookEdit']
fs.sources["features.memory"]  # 'preset deterministic'
```
