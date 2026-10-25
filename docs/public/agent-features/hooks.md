---
title: Hooks
sidebar:
  order: 6
---

Hooks run your own shell commands at lifecycle events — format a file after every edit, audit every shell command before it runs, snapshot state before compaction. They live in your config file, so a repo can ship its own hooks in `.cca/config.yaml`.

The 80% case — auto-format after every edit:

```yaml
# ~/.cca/config.yaml (global) or .cca/config.yaml (per repo)
hooks:
  PostToolUse:
    - matcher: "Write|Edit"
      command: "ruff format $(jq -r .tool_input.file_path)"
      async: true          # fire-and-forget, doesn't block the agent
```

`/hooks` in the REPL lists everything configured.

## Events

| Event | Fires | Can block? |
|---|---|---|
| `SessionStart` | Once when a session opens | No |
| `SessionEnd` | Once when a session ends | No |
| `PreToolUse` | Before each tool call | **Yes** — exit 2 blocks the tool |
| `PostToolUse` | After each tool call | No, but can inject text into the result |
| `PreCompact` | Before [context compaction](compaction.md) | No |
| `PostCompact` | After compaction | No |

## Hook definition fields

| Field | Default | Meaning |
|---|---|---|
| `command` | — | Shell command to run |
| `matcher` | `"*"` | Glob pattern matched against the tool name (`PreToolUse`/`PostToolUse` only) |
| `timeout` | `10` | Seconds before the hook is killed |
| `async` | `false` | Run fire-and-forget; the agent doesn't wait for it |

## What your hook receives

Each hook gets a JSON object on stdin describing the event (and `CCA_HOOK_EVENT` in its environment):

```json
{
  "session_id": "20260701-143022-abc123-comp0",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {"command": "ls -la"},
  "tool_use_id": "toolu_01xxx"
}
```

`PostToolUse` adds `response` (the tool's output). `SessionStart`, `SessionEnd`, `PreCompact`, and `PostCompact` payloads carry `cwd` plus event-specific fields (`message_count`, `summary_length`).

## Controlling the agent from a hook

**Exit codes:**

| Event | Exit 0 | Exit 2 | Other |
|---|---|---|---|
| `PreToolUse` | Allow | **Block the tool** — stderr is shown to the model as the reason | Allow; stderr shown to the user |
| `PostToolUse` | Success | stderr is appended to the tool result the model sees | stderr shown to the user |
| `SessionStart` / `SessionEnd` / `PreCompact` / `PostCompact` | OK | Logged only | Logged only |

**Or return JSON on stdout** for finer control:

```json
{
  "decision": "block",
  "reason": "unsafe command detected",
  "additionalContext": "Use the safer make target instead."
}
```

`"decision": "block"` (or `"continue": false` with `"stopReason"`) blocks the tool; `additionalContext` is injected into the model's context even when allowing. If multiple hooks match an event, the first one that blocks wins; `additionalContext` from all of them is accumulated.

## Example: guard rails on Bash

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      command: "check-command.sh"
      timeout: 5
```

```bash
#!/bin/bash
# check-command.sh — block anything touching prod
cmd=$(jq -r .tool_input.command)
if [[ "$cmd" == *"--prod"* ]]; then
  echo "commands targeting prod are not allowed from the agent" >&2
  exit 2
fi
```

Hooks fire in all [run modes](run-modes.md): interactive, `--print`, and `--machine` sessions all run `SessionStart` at startup, and `--print` and `--machine` sessions run `SessionEnd` on exit.
