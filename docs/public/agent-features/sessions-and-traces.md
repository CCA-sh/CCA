---
title: Sessions & Traces
sidebar:
  order: 3
---

Every `cca` conversation is automatically saved to disk — the resumable conversation state, plus a raw log of every LLM API call. You can resume any session later, and every trace is plain JSON you can read, replay, or turn into training data. Nothing needs to be enabled. For an optional live operational view, see [Observability](observability.md); it does not replace these files.

## Resume a session

```bash
cca --continue                 # resume the most recent session for this directory
cca --resume <id>              # resume a specific session by ID
cca sessions list              # list saved sessions (most recent first)
cca sessions delete <id>       # delete a session
```

Inside the REPL, `/resume` shows a picker of past sessions, and `/resume <id>` loads one directly.

`--continue` matches sessions by working directory, so each project resumes its own latest conversation.

## What's on disk

Everything lives under `~/.cca/`:

```
~/.cca/
├── history.jsonl                     # index: one line of metadata per session
├── sessions/<id>.json                # resumable conversation state, one file per session
├── llm-calls/session_<id>.jsonl      # raw LLM API calls: exact inputs and outputs
├── llm-calls/session_<id>_sub_<agent-id>.jsonl # separate child trajectories
├── llm-calls/session_<id>_agents.json # child lineage/status manifest, only when used
├── llm-calls/session_<id>_agents.jsonl # child lifecycle/operator control events
└── transcripts/session_<id>.jsonl    # pre-compaction message archive (only for long sessions)
```


| Store                            | What it's for                                                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `history.jsonl`                  | Metadata only (session ID, first user message, project directory, timestamp). This is what `sessions list`, `--continue`, and `/resume` read.                                  |
| `sessions/<id>.json`             | The full conversation as a flat JSON array in Anthropic message schema, plus the resolved runtime `settings`, rewritten at the end of every turn. This is what resuming loads. |
| `llm-calls/session_<id>.jsonl`   | One record per LLM API call — see below. This is the file to use for data extraction.                                                                                          |
| `transcripts/session_<id>.jsonl` | Written only when [compaction](compaction.md) fires: an archive of the messages that were summarized away, so nothing is ever lost.                                            |


## The LLM call log

For every API call the agent makes, one JSON line is appended with the **exact request and response at call time**:

```json
{
  "ts": 1751380000,
  "type": "main",
  "session_id": "20260701-143022-abc123-comp0",
  "call_idx": 3,
  "features": {
    "preset": "deterministic",
    "features": { "memory": false, "compaction": false, "...": "..." },
    "enabled_tools": ["Bash", "Edit", "Glob", "..."]
  },
  "system": "...full system prompt...",
  "messages": [ ...exact message array sent... ],
  "tools": [ ...tool schemas sent... ],
  "params": { ... },
  "response": {
    "text": "...",
    "tool_calls": [ ... ],
    "thinking": "...",
    "reasoning": {
      "text": "...readable provider summary...",
      "kind": "summary",
      "requested": true,
      "captured": true
    },
    "stop_reason": "tool_use",
    "input_tokens": 4200,
    "output_tokens": 310,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0
  }
}
```

Because each record is self-contained — system prompt, message snapshot, tool schemas, and the model's reply — every line is already an input/output pair. No reconstruction or diffing against the session file is needed.

For a main call, `params.thinking` records the requested policy, effective
level, and resolution source. `params.backend_request` records the concrete
adapter fields used for the call, such as `reasoning_effort` or Anthropic's
`thinking` budget. The session-level `settings` snapshot also records the model,
sanitized endpoint, authentication source, token limits, permissions, active
tools, features, prompt overrides, and configuration paths. API-key values and
endpoint credentials/query strings are never stored in this report.

`response.thinking` remains the legacy readable-text field. New records also carry
the structured `response.reasoning` form so dataset exporters can distinguish a
provider summary from a directly emitted trace without changing where the readable
text is stored. Provider signatures and encrypted continuation items may exist in
raw state, but exporters must remove them rather than treating them as training text.

The `type` field tells you which loop produced the call: `main` for regular agent calls, `compactor` (and `compactor_retry`) for the internal summarization calls made during compaction. Filter on `type == "main"` if you only want the agent's own turns.

The `features` field records the [feature-toggle configuration](../configuration/feature-toggles.md) the call ran under — the active preset (or `null`), all behavior flags, and the exact enabled tool list — captured at call time. Session files carry the same block, but the per-call stamp is authoritative: if a session is resumed under a different configuration, each trace line still shows the config that actually produced it. Filtering a collection run is one expression:

```bash
jq -c 'select(.features.preset == "deterministic" and .type == "main")' ~/.cca/llm-calls/*.jsonl
```

(Records written by older versions of cca predate this field — treat a missing `features` key as "configuration unknown".)

A `randomization` field appears alongside it **only** when the run used `[--randomize](../configuration/randomization.md)`, recording the seed and which prompt and tool-description variants were selected. The `system` and `tools` fields already contain the exact text that was sent, so this block is what lets you group, filter, or reproduce runs by variant:

```bash
jq -r 'select(.type == "main") | .randomization.variants.system_prompt' ~/.cca/llm-calls/*.jsonl | sort | uniq -c
```

A run without `--randomize` omits the key entirely.

## Sessions and compaction

When a long conversation is [compacted](compaction.md), the session ID gets a new suffix (`...-comp0` → `...-comp1`) and a new session file is written containing the trimmed conversation:

- The old `...-comp0.json` file is kept as-is — the full pre-compaction state is never deleted.
- The new `...-comp1.json` carries a `compacted_from_index` field: the message index where its array starts in the full history. Concatenating `comp0` messages up to that index with `comp1` reconstructs the complete timeline losslessly.
- The LLM call log is **one file per session chain** — the filename strips the `-compN` suffix — so all calls before and after compaction land in the same `.jsonl`, in order.

Resuming a post-compaction session loads only the trimmed conversation; the summarized-away messages remain available in the earlier session file and the transcript archive.

## Session ID format

```
20260701-143022-abc123-comp0
│        │      │      └── compaction counter (starts at 0)
│        │      └───────── uniqueness suffix
│        └──────────────── start time
└───────────────────────── start date
```

Subagent sessions append `-sub-<hex>` and record their parent session ID in `history.jsonl`.

## Subagent traces and RAW export

When the root invokes `Agent`, every child LLM call is stored in its own
`session_<root-id>_sub_<agent-id>.jsonl`. A versioned manifest records the exact
agent definition and hash, catalog hash, parent and spawn IDs, lifecycle status,
and structured result. Children created by one joined `Agent.tasks` call also record
the shared `spawn_group_id` and stable `spawn_index`; their individual calls remain
in separate files. Sessions that never invoke `Agent` create neither child logs nor
a manifest.

Interactive `/agents default` and `/agents stop` actions are appended to the event
log as operator-control records. They never enter the root message array. Read-only
catalog, status, and output commands create no trace artifacts. When control events
exist, the SFT exporter includes them under `provenance.control_events` so a run can
be replayed without presenting those actions as model dialogue.

The default exporter remains root-only:

```bash
cca traces inspect <root-id> # terminal tree: lineage, lifecycle, calls, SFT status
cca traces inspect <root-id> --json
cca traces export <root-id> --output root.jsonl
cca traces export <root-id> --format calls --output root-calls.jsonl
cca traces export <root-id> --view children-separated --output children.jsonl
cca traces export <root-id> --view all-separated --output all.jsonl
```

`traces inspect` is read-only. It reports root and child call counts/types, parent and
spawn IDs, definition/catalog hashes, lifecycle events, replay issues, orphan child
logs, exact call-export coverage, context generations, and safe or excluded
trajectories. Nonterminal children appear as explicitly excluded rather than being
presented as clean data.

`--format trajectories` remains the default and emits one structurally complete
trajectory per model-visible context generation. Compaction and `/clear` therefore
cannot make a final snapshot claim call indexes from an earlier context. If a
generation ends during a tool interaction, trajectory export rejects it with an
actionable message instead of silently truncating it.

`--format calls` is the lossless SFT view: every eligible policy-model call becomes
one example containing its exact request snapshot and clean assistant target. An
assistant tool call is a valid target; its result appears in the following call's
input. This view is the coverage authority across compaction boundaries.

Both formats validate monotonically increasing call indexes and tool-call/result
pairing. Failed, cancelled, queued, turn-capped, or context-capped children remain in
raw storage but are excluded from clean views. Provider thinking blocks and opaque
continuation state are retained in raw replay records but removed rather than treated
as training text. Root and child examples are always separate. For reproducible
multi-agent collection, use joined foreground batches so completion timing cannot
insert detached notification turns at nondeterministic positions.

New call records add model-visible context and execution provenance without changing
the recorded messages or responses:

```json
{
  "trace_id": "stable-root-trace",
  "agent_id": "stable-actor-id",
  "context_generation": 1,
  "generation_reason": "compaction",
  "run_id": "process-run-id",
  "continuation_reason": "initial",
  "model_provenance": {
    "requested_selector": "inherit",
    "selector": "anthropic:claude-sonnet-4-6",
    "backend": "AnthropicBackend",
    "verified_from_backend": true
  }
}
```

`trace_id` and `agent_id` stay stable across compaction, `/clear`, and resume.
`context_generation` advances only when model-visible context is transformed;
`generation_reason` is `initial`, `compaction`, or `clear`. Resume keeps the saved
context generation and records `continuation_reason: resume` under a new `run_id`.
Root context-boundary events live in the root call log, so a session that never
invokes `Agent` still creates no child manifest or child event file.

Compatibility has two explicit levels:

- With `tools.Agent: false`, root model inputs and SFT output retain the existing
single-agent scaffold. The `deterministic` preset uses this mode.
- With `Agent` enabled but unused, no child artifacts are created and root-only
  export remains structurally compatible, but the root trace is labeled
  `multi_agent_capable: true` because its action space includes delegation.
- Historical call logs remain read-only and exportable. Missing generation metadata
  is inferred from the legacy `generation` field or session ID and labeled
  `generation_inferred: true`; unknown execution details remain unknown rather than
  being invented.

## Relocating the data

The storage locations can be overridden with environment variables — useful for CI, benchmarking runs, or keeping experiment data separate:


| Variable            | Default                              |
| ------------------- | ------------------------------------ |
| `CCA_DIR`           | `~/.cca` (root for everything below) |
| `CCA_SESSIONS_DIR`  | `~/.cca/sessions`                    |
| `CCA_HISTORY_FILE`  | `~/.cca/history.jsonl`               |
| `CCA_LLM_CALLS_DIR` | `~/.cca/llm-calls`                   |
