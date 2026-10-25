---
title: Permissions & tools
sidebar:
  order: 2
---

The agent works with a built-in toolset. Read-only tools run silently; anything that can change your system — writing files, running shell commands — asks first. You decide once, per session, or permanently.

## The permission prompt

When the agent wants to use a write/destructive tool, you get a prompt showing exactly what it's about to do (for edits, a diff preview). Answer with one key:

| Key | Effect |
|---|---|
| `y` (or Enter) | Allow this one call |
| `n` | Deny this call |
| `a` | Always allow this tool — **persisted to config**, survives restarts |
| `!` | Never allow this tool for the rest of the session |
| `Esc` | Deny and cancel the whole turn |

To skip prompts for specific tools permanently:

```bash
cca config set always_allow "Bash,Edit,Write"
```

To skip all prompts for a run: `cca --auto-approve` (alias: `--unsafe-auto-approve`). To hard-block tools: `cca config set never_allow "Bash"` or `--disallowed-tools "Bash"` for one run.

## Which tools prompt

**Never prompt (read-only / safe):** Read, Glob, Grep, ToolSearch, WebFetch, WebSearch, AskUserQuestion, ScheduleWakeup, WorkCreate/Get/List/Update, JobOutput, CronList.

**Prompt by default (write, process-launching, or destructive):** Write, Edit, NotebookEdit, Bash, Agent, Skill, LSP, Monitor, JobCancel, CronCreate, CronDelete, EnterWorktree/ExitWorktree, EnterPlanMode/ExitPlanMode.

## Directory sandboxing

Write tools (Write, Edit, NotebookEdit) are restricted to the working directory — a target outside it is denied even if you approve the tool. To grant extra directories:

```bash
cca --add-dir /path/to/other/project        # repeatable
```

`~/.cca` is always writable (plans, memory, and session data live there). The path check is skipped under `--auto-approve`.

## Plan mode

While plan mode is active (`/plan` or the agent entering it itself), all write tools are blocked regardless of permissions — the agent can only produce its plan file until you approve the plan via ExitPlanMode.

## Built-in tools

CCA uses short, action-oriented names for its model-visible tools so their purpose is clear in prompts, configuration, and saved traces. These names identify CCA's own implementations and stable interfaces; they do not imply drop-in compatibility with similarly named tools in other products.

| Tool | Description |
|---|---|
| `Read` | Read files (text, images, PDFs, notebooks) |
| `Write` | Create or overwrite files |
| `Edit` | Targeted string replacement |
| `Glob` | File pattern matching |
| `Grep` | Content search (ripgrep, with Python fallback) |
| `SearchInFile` | Search within a single file |
| `Bash` | Shell command execution |
| `Agent` | Spawn CCA profiles such as `scout`, `blueprint`, or `worker` |
| `WebFetch` / `WebSearch` | Fetch URLs and [search the web](web-search.md) |
| `NotebookEdit` | Edit Jupyter notebook cells |
| `WorkCreate/Get/List/Update` | Session work-item management |
| `JobOutput` / `JobCancel` | Inspect or cancel background Bash, monitor, and subagent jobs |
| `CronCreate/Delete/List` | Schedule recurring jobs |
| `Monitor` | Watch background process output |
| `EnterWorktree` / `ExitWorktree` | Git worktree isolation |
| `EnterPlanMode` / `ExitPlanMode` | Structured planning before implementation |
| `AskUserQuestion` | Structured multiple-choice questions to you |
| `ScheduleWakeup` | Self-pace looped agent runs |
| `ToolSearch` | Discover deferred tools |
| `LSP` | Query an auto-detected local language server for code intelligence |
| `Skill` | Invoke skills (only registered when `skills_enabled: true`) |

MCP servers add their own tools on top — manage them with `/mcp` in the REPL.

Restrict the toolset for a single run with `--allowed-tools "Read,Grep,Glob"` (allowlist) or `--disallowed-tools "Bash"` (blocklist).

## Headless behavior

In `--print` mode there is no one to answer prompts: without `--auto-approve`, prompting tools are **silently denied**. In `--machine` mode permissions are always auto-approved. See [Run modes](run-modes.md).
