---
title: REPL commands
sidebar:
  order: 10
---

Type `/` in the interactive REPL to open the command palette with autocomplete, or `/help` to list everything. Commands control the session — model, thinking, permissions, context — without leaving the conversation.

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/plan [task]` | Enter plan mode: the agent designs before touching code; write tools are blocked until you approve the plan |
| `/think [level]` | Show or set thinking policy: `auto` / `off` / `low` / `medium` / `high` — see [Thinking mode](../configuration/thinking-mode.md) |
| `/settings` | Print the complete safe, resolved runtime and backend settings |
| `/init` | Generate or improve `AGENTS.md` (project instructions) for this project |
| `/doctor` | Check API keys, model config, and tool availability |
| `/hooks` | List configured [hooks](hooks.md) |
| `/clear` | Clear conversation history |
| `/model [name]` | Show or switch model (persists to config) |
| `/models [provider-or-search]` | List bundled providers or search bundled model metadata |
| `/key [provider] [key]` | Show or set an API key |
| `/config` | Show current configuration |
| `/memory` | List stored [memories](memory.md) |
| `/work` | List session work items |
| `/tokens` | Show session token usage |
| `/budget [amount]` | Set or show the response token budget (e.g. `/budget 500k`) |
| `/compact [--prefix\|--tail] [instructions]` | Force [context compaction](compaction.md) now, optionally with custom instructions |
| `/resume [id]` | Load a previous session (omit the id to pick from a list) |
| `/review [target]` | Code review for a target path or the current directory |
| `/allowed-tools [tool]` | Toggle a tool in the always-allow list |
| `/mcp` | MCP server management |
| `/verbose` | Toggle verbose tool output |
| `/trace` | Toggle trace mode (show all LLM inputs/outputs) |
| `/vim` | Toggle vi keybindings |
| `/cwd [path]` | Show or change the working directory |
| `/skills [on\|off]` | List skills or toggle the skills system |
| `/quit` / `/exit` | Quit |

## Shell escape

Prefix any line with `!` to run it directly in your shell — the output appears in your terminal without going through the agent:

```
! git status
! pytest tests/ -x
```

## Inline message modifiers

Two things are parsed out of ordinary messages, no slash needed:

- **Thinking keywords** — `ultrathink`, `think harder`, `think step by step`, and similar phrases force high-level thinking for that one turn.
- **Token budgets** — `/budget <amount>` anywhere in a message (e.g. `/budget 500k`) raises the response token limit for that turn, the same as running `/budget 500k` as its own command. See [Thinking mode](../configuration/thinking-mode.md#inline-token-budget).

## Headless use

Slash commands are an interactive-mode feature. In `--print` mode use the equivalent CLI flags (`--think`, `--model`, `--resume`); in `--machine` mode send them as `slash_command` protocol messages. See [Run modes](run-modes.md).
