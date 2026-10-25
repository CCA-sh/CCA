---
title: Memory
sidebar:
  order: 5
---

The agent remembers things between sessions — your preferences, feedback you've given, project context — as plain markdown files it writes and reads itself. No database, no opt-in: memories accumulate as you work and are loaded into future sessions automatically.

```bash
cca memory list           # list all stored memories
cca memory delete <name>  # delete one by filename (without .md)
```

Inside the REPL, `/memory` lists them.

## How memories are created

As a session grows, the agent periodically distills durable facts from the conversation — a correction you made, a constraint you mentioned, how you like output formatted — and saves each one as a file. Extraction only starts once a session has real substance (about 10k tokens) and re-runs incrementally as the conversation continues.

You can also just ask: "remember that I prefer pytest over unittest."

## Where they live

| Scope | Location | Use |
|---|---|---|
| Private | `~/.cca/memory/` | Yours only — preferences, personal setup. Never in a repo. |
| Team | `<project-root>/.cca/memory/` | Shared project context — can be committed so the whole team's agents know it. |

Each directory has a `MEMORY.md` index (one line per memory) that gets loaded into context each session; individual memory files are read on demand.

## Memory file format

One fact per file, markdown with YAML frontmatter:

```markdown
---
name: prefers-pytest
description: User prefers pytest over unittest for all new tests
type: feedback
---

Use pytest idioms (fixtures, parametrize) rather than unittest classes.
```

Types: `user` (who you are), `feedback` (guidance on how the agent should work), `project` (ongoing work and constraints), `reference` (pointers to external resources).

Because memories are plain files, editing them is just editing markdown — fix one, delete one, or check the team ones into git like any other file. Memories older than a day are flagged to the agent as potentially stale so it verifies before relying on them.
