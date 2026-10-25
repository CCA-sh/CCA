---
title: Compaction
sidebar:
  order: 4
---

Long sessions eventually outgrow the model's context window. Compaction keeps them going: when the conversation gets too large, `cca` summarizes the older part and continues with the summary plus the recent messages — automatically, without losing any data on disk.

You normally don't do anything. If you want control:

```
/compact                          # compact now
/compact focus on the API changes # compact now, with extra instructions for the summary
/tokens                           # see where you stand
```

## When it triggers

Two thresholds, both configurable in [Configuration](../configuration/index.md):

| Threshold | Default | Checked | Behavior |
|---|---|---|---|
| `compact_threshold` | 150,000 tokens | At the end of each turn | Compacts before the next turn starts |
| `compact_threshold_upper` | 200,000 tokens | After each batch of tool calls, mid-task | Compacts immediately, even in the middle of a task |

Token counts come from the API's own reported input size (falling back to estimation only before the first call), so the trigger tracks reality rather than a guess.

## What happens

1. **Cheap pass first (micro-compaction):** oversized old tool results are truncated. If that alone gets the context under target, no LLM call is made and nothing else changes.
2. Otherwise the full conversation is archived to `~/.cca/transcripts/`, an LLM call summarizes the older messages, and the session continues with the summary plus the most recent messages kept verbatim (roughly the last 40).
3. The session ID advances (`...-comp0` → `...-comp1`) and a new session file is written. The pre-compaction file is never deleted.
4. Recently read files are re-injected so the agent doesn't lose track of what it was working on.

`PreCompact` and `PostCompact` [hooks](hooks.md) fire around the operation.

## Manual compaction options

`/compact` accepts a direction flag and free-text instructions:

```
/compact                       # summarize the old prefix, keep recent messages (default)
/compact --prefix              # same as default, explicit
/compact --tail                # inverse: summarize the recent tail, keep the old prefix verbatim
/compact keep all file paths   # any remaining text is added to the summarization prompt
```

## Nothing is lost

Compaction trims the *in-memory* conversation, not your data:

- The full pre-compaction message array survives in the previous session file (`...-comp0.json`) and in the transcript archive.
- The new session file records `compacted_from_index` — the exact message index where it continues — so the complete timeline can be reconstructed losslessly by concatenation.
- The [LLM call log](sessions-and-traces.md) is one file per session chain and keeps recording across compactions, including the compactor's own summarization calls (`type: "compactor"`).

Every full or micro-compaction starts a new `context_generation` in that root log.
The stable trace and actor IDs do not change, and an explicit context-boundary record
states why the new generation began. A resumed process keeps the saved context
generation and receives a new run ID instead of pretending resume was compaction.

This matters when exporting training data: `cca traces export --format calls`
preserves every eligible policy-model call exactly once across compaction. The
default trajectory format emits safe per-generation trajectories and refuses a
generation that ends in the middle of a tool interaction rather than presenting an
incomplete example as clean.
