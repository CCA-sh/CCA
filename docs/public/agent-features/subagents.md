---
title: Subagents
sidebar:
  order: 4
---

CCA can delegate bounded work to an isolated child context through the `Agent`
tool. The root agent remains responsible for decomposition, permissions,
verification, conflict handling, and the final answer. Children cannot spawn more
children; the supported hierarchy is deliberately `root → child`.

## Built-in profiles

| Agent | Purpose | File-edit tools |
|---|---|---|
| `worker` | Bounded implementation or repair | Yes |
| `scout` | Repository mapping and dependency tracing | No |
| `blueprint` | Architecture and implementation planning | No |
| `reviewer` | Independent correctness review | No; may run approved commands |
| `verifier` | Tests, builds, and failure reproduction | No; may run approved commands |
| `risk-auditor` | Security, privacy, dependency, and reliability analysis | No; may run approved commands |
| `researcher` | Primary documentation and technical research | No |

`worker` is the default. The built-in profiles have distinct tool policies; denied
tools are enforced again when a tool dispatches, not only by hiding their schemas
from the model. Commands still pass through the root permission and hook policy.

## Define an agent

Global definitions live in `~/.cca/agents/*.md`. Project definitions live in
`.cca/agents/*.md` and can be committed with the repository. Inner project
directories override outer definitions of the same name, and project definitions
override global definitions and built-ins.

```markdown
---
name: migration-reviewer
description: Reviews database migrations for correctness and rollback safety
tools: [Read, Glob, Grep, Bash]
disallowed_tools: [Write, Edit]
model: inherit
max_turns: 12
effort: medium
---

Review only the delegated migration. Report concrete defects with file and line
evidence, then list the checks you ran. Never edit the workspace.
```

CCA does not implicitly scan another product's configuration directories. Portable
Markdown definitions can be validated and explicitly normalized into CCA's native
registry:

```bash
cca agents list
cca agents validate ./reviewer.md
cca agents import ./reviewer.md                 # project .cca/agents/
cca agents import ./reviewer.md --scope global  # ~/.cca/agents/
```

Unknown fields produce warnings during explicit import and are not silently granted
runtime meaning.

Agents default to `max_turns: 40`. A turn is one child-model request, not one tool
call; a single response may request multiple tools. CCA reserves the final turn for
a tool-free handoff. If the child has not finished after 39 working turns, it must
summarize completed work, file changes, verification, remaining work, and risks for
the parent. The result is labeled `turn_cap`, not falsely marked complete, and is
excluded from clean SFT export by default.

## Joined parallel work

For several independent tasks that must feed one final answer, the root can send a
`tasks` array in one `Agent` call. CCA schedules the children together up to
`max_concurrency`, waits for every terminal result, and returns the result list in
the original declaration order. Read-only children can execute concurrently;
children whose profiles can mutate the shared workspace remain serialized until
worktree isolation is available. Fan-out therefore does not depend on a model or
provider supporting multiple function calls in one response.

```json
{
  "tasks": [
    {
      "agent": "reviewer",
      "description": "review implementation",
      "prompt": "Review the implementation and report concrete defects with evidence."
    },
    {
      "agent": "risk-auditor",
      "description": "audit security risks",
      "prompt": "Audit the same implementation for security and reliability risks."
    },
    {
      "agent": "verifier",
      "description": "verify release checks",
      "prompt": "Run the bounded release checks and report exact results."
    }
  ]
}
```

Joined batches cannot run in the background. A single foreground `Agent` call and
multiple foreground `Agent` calls emitted together continue to work as before.

## Configuration

```yaml
subagents:
  default: worker
  max_depth: 1
  max_concurrency: 3
  allow_background: false
```

Disable the entire subagent action space with:

```yaml
tools:
  Agent: false
```

With `Agent` disabled, CCA does not add the delegation policy or agent catalog to
the root scaffold. The shipped `deterministic` preset keeps it disabled.

Run the whole root session with a named profile using `cca --agent reviewer`, or
set `agent: reviewer` in global/project configuration. This is a root trajectory,
not a hidden child.

Background children are deliberately detached: they return an `agent_id` and may
cause a later continuation when they finish. Use `JobOutput` and `JobCancel` to
inspect or cancel them. Background delegation is opt-in: set
`subagents.allow_background: true` to expose its parameter in the model-visible
schema. Deterministic multi-agent collection should use foreground children or
joined batches because completion timing is otherwise nondeterministic.

## Interactive controls

The interactive `/agents` command uses the catalog snapshot frozen when the session
starts:

```text
/agents help                    # usage, profile, lifecycle, and trace guidance
/agents                         # list built-in, global, and project profiles
/agents show <name>             # exact source, prompt, tools, limits, and hashes
/agents default <name>          # choose the default for later delegations
/agents running                 # list children belonging to this root session
/agents output <agent-id>       # inspect current or persisted output without waiting
/agents stop <agent-id>         # cancel a child still active in this process
```

Catalog and lifecycle reads do not call a model or enter conversation history.
Default changes and stop requests are durable operator-control events stored outside
model messages. The default is session-local and is restored from session state or
the control log.

CCA does not reload agent files or enable/disable the `Agent` tool in place. Start a
new session to change the action space. Root profiles likewise remain a startup
choice through `cca --agent <name>`; an in-place root switch would mix system
scaffolds in one training trajectory.

## Traces

Each child has a separate exact LLM-call log and is linked to its parent by a
versioned manifest. Joined children additionally record a shared spawn-group ID and
their declaration index. Child calls are never inserted into the root's message
history as if the root made them. Interactive control events appear only as SFT
provenance, never as fabricated dialogue. See [Sessions & traces](sessions-and-traces.md)
for paths and separated SFT export views.

Children also reserve context for a final tool-free handoff. When the resolved model
catalog reports a context window and the child reaches the safety threshold, it
returns status `context_cap` with completed work, changed files, verification, and
remaining risks instead of waiting for the provider to reject an oversized request.
Context-capped runs remain available in raw traces but are excluded from clean SFT
views because their delegated task did not complete. This is a safety guard; full
child summarizing compaction is planned separately.
