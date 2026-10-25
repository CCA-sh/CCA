---
title: Run modes
sidebar:
  order: 1
---

`cca` runs in three modes, depending on who is driving it: a human at a terminal (**interactive**), a script that wants one answer (**print**), or another program holding a multi-turn conversation (**machine**).

| | Interactive | `--print` | `--machine` |
|---|---|---|---|
| Who drives | Human | Script / one-shot | Orchestrator process |
| Turns | Many, conversational | One prompt, then exit | Many, via JSON protocol |
| Context across turns | Grows | Cold start each invocation | Grows |
| Slash commands | Yes | No (text sent to model verbatim) | Yes, via `slash_command` messages |
| MCP servers | Connected | Not connected | Connected |
| Permission prompts | Interactive dialog | Silently denied unless `--auto-approve` | Always auto-approved |
| Good for | Daily work | Benchmarks, cron jobs, CI | RL loops, agent sims, interactive benchmarks |
| Optional observability | Yes | Yes | Yes; stdout remains protocol-only |

## Interactive mode

```bash
cca                                        # start the REPL
cca --model google:gemini-3.5-flash        # one-session model override
cca --auto-approve                         # skip all tool permission prompts
cca --cwd /path/to/project                 # start in a specific directory
cca --continue                             # resume the latest session here
cca --trace                                # show raw LLM requests/responses
cca --verbose                              # show full tool output
```

The full terminal experience: streaming output, permission dialogs, [slash commands](repl-commands.md), [key bindings](key-bindings.md), plan mode, and MCP servers. `/help` lists all commands; `/doctor` checks your setup.

## Print mode

```bash
cca --print "summarize this repo"
cca --print "fix the failing test" --auto-approve
cca --print "explain main.py" --output-format json
```

Runs one prompt to completion and exits — no prompts, no UI. Useful flags:

| Flag | Effect |
|---|---|
| `--output-format text\|json\|stream-json` | Plain text (default), a final JSON object, or one JSON event per line as the run progresses |
| `--max-turns N` | Stop after N agentic turns (0 = unlimited) |
| `--system-prompt TEXT` | Replace the default system prompt entirely |
| `--append-system-prompt TEXT` | Append to the default system prompt |
| `--allowed-tools "Read,Grep"` | Restrict the toolset for this run |
| `--disallowed-tools "Bash"` | Block specific tools |
| `--add-dir DIR` | Let write tools touch an extra directory (repeatable) |
| `--think auto\|off\|low\|medium\|high` | Select the [thinking policy](../configuration/thinking-mode.md) for the run |
| `--show-settings` | Print the full safe, resolved runtime and backend settings to stderr |

Things to know before scripting with `--print`:

- **Pass `--auto-approve`** (or `--unsafe-auto-approve`) if the task needs Write/Edit/Bash. Without it there is no one to answer the permission prompt, so those tools are silently denied and the agent cannot change anything.
- **Plan approval is automatic** — if the agent enters plan mode, the plan is auto-approved and execution continues.
- **No MCP servers and no slash commands.** A prompt starting with `/` is sent to the model as literal text. Set thinking, model, and session via CLI flags instead (`--think`, `--model`, `--resume`).
- **Each invocation starts fresh** unless you chain them with `--continue` / `--resume <id>`.
- **Resolved settings are visible.** A compact model/thinking/token/permission summary is written to stderr, leaving stdout usable as program output. JSON output also contains a `settings` object.
- Sessions, traces, auto-compaction, and hooks on tool calls all work exactly as in interactive mode.

## Machine mode

```bash
echo '{"type": "user_turn", "text": "summarize this repo"}' | cca --machine
```

A persistent process speaking JSON-lines over stdin/stdout, built for driving `cca` from another program. Unlike `--print`, the process stays alive across turns with full growing context, MCP servers, hooks, and slash commands — feature parity with interactive mode, minus the human.

On startup, `cca` emits a `ready` event, then waits for commands:

**You send (one JSON object per line on stdin):**

```json
{"type": "user_turn", "text": "fix the auth bug"}
{"type": "slash_command", "command": "/plan add rate limiting"}
{"type": "answer", "id": "req-abc", "answers": {"Which DB?": "postgres"}}
{"type": "plan_decision", "id": "req-xyz", "approved": true, "feedback": "looks good"}
{"type": "shutdown"}
```

**cca emits (one JSON object per line on stdout):**

```json
{"type": "ready", "session_id": "...", "settings": {"model": {"selector": "..."}, "thinking": {"requested": "auto", "effective": "medium", "source": "cca-default"}}}
{"type": "turn_complete", "text": "...", "session_id": "...", "msg_index_end": 42}
{"type": "ask_user", "id": "req-abc", "questions": [...]}
{"type": "exit_plan_mode", "id": "req-xyz", "plan": "..."}
{"type": "error", "message": "..."}
```

The two blocking interactions a human would normally handle are routed through the protocol instead: when the agent calls `AskUserQuestion`, you receive an `ask_user` event and reply with an `answer`; when it finishes a plan, you receive `exit_plan_mode` and reply with a `plan_decision`. Tool permissions are always auto-approved — the orchestrator is trusted.

stderr stays human-readable for debugging; only stdout carries protocol events.
The `ready.settings` object is always present, so an orchestrator can record the
resolved model, adapter, endpoint, authentication source, thinking parameters,
token limits, permissions, tools, features, prompts, and configuration paths
before sending the first turn. Secret values are omitted.

Optional [observability](observability.md) uses a background exporter and never
writes telemetry to stdout, preserving the JSON-lines contract.

Every machine-mode session is recorded like any other — see [Sessions & traces](sessions-and-traces.md) for turning runs into data.
