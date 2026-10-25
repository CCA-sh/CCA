---
title: Observability
sidebar:
  order: 4
---

CCA can send live lifecycle traces to an OpenTelemetry-compatible backend. It is
strictly opt in: a normal installation makes no observability network requests and
does not require the optional dependencies.

Observability complements CCA's local [sessions and traces](sessions-and-traces.md).
The local JSON files remain the durable, replayable source of truth; exported spans
provide a near-real-time operational view as work completes.

## What is traced

A trace follows CCA's own lifecycle rather than a provider-specific request format:

```text
session
└── turn
    ├── LLM call
    ├── tool call
    ├── Agent tool call
    │   └── subagent run
    │       ├── LLM call
    │       └── tool call
    ├── compaction
    │   └── LLM call
    └── memory extraction
        └── LLM call
```

Spans include status, duration, model and adapter, token/cache counts, run mode,
tool or agent name, and CCA's trace and agent IDs. Parallel tool calls share the
same turn parent. Subagents link to the `Agent` call that spawned them. Session
resume starts a new runtime session linked by the saved trace ID, while compaction
increments the context generation without fragmenting the trace chain.

Observability works in interactive, `--print`, and `--machine` modes. Machine-mode
stdout remains JSON-lines only; exporter failures never become protocol events.

## W&B Weave

Install the optional OpenTelemetry exporter, then provide W&B credentials through
environment variables:

```bash
pip install 'cca-cli[observability]'

# Secret from W&B User Settings > API keys. Do not commit this value.
export WANDB_API_KEY='...'

# Owner of the project: usually your W&B team slug. This is the <entity>
# in https://wandb.ai/<entity>/<project>/weave/traces, not the project name.
export WANDB_ENTITY='my-team-slug'

# Destination project name. Optional; CCA uses "cca" when this is unset.
export WANDB_PROJECT='cca-testing'

cca --observe wandb
cca --observe wandb --observe-capture content   # with full content captured
```

For example, with `WANDB_ENTITY='acme-ai'` and
`WANDB_PROJECT='cca-testing'`, open:

```text
https://wandb.ai/acme-ai/cca-testing/weave/traces
```

You can find the entity slug in W&B's left sidebar under **Teams**, or copy it
from the URL of an existing W&B project. Your API key must have access to that
entity. `WANDB_ENTITY` and `WANDB_PROJECT` answer different questions: the entity
owns the project, and the project groups the traces.

`--observe` is a one-run override. To enable it for every run, add this to the
global or repository `.cca/config.yaml`:

```yaml
observability:
  enabled: true
  destination: wandb
  entity: my-team-slug          # project owner; or set WANDB_ENTITY
  project: cca-testing          # project name; optional, defaults to cca
  capture: metadata
  sample_rate: 1.0
  flush_timeout: 2.0
```

CCA exports to W&B's OTLP trace endpoint and supplies Weave's thread and turn
attributes. The W&B API key is read only from `WANDB_API_KEY`; it is never written
to session files or the resolved-settings report.

Weave renders prompts, completions, and tool payloads only when they are exported,
so pair `--observe wandb` with `--observe-capture content`; with the default
`metadata` level Weave has nothing but timings and token counts to show. LLM spans
carry the provider in `gen_ai.system` and the bare model id in `gen_ai.request.model`
and `gen_ai.response.model`, which is what cost attribution matches on.

## Any OTLP backend

The agent lifecycle is provider neutral. Point the same instrumentation at an
OpenTelemetry collector or another OTLP/HTTP trace service:

```yaml
observability:
  enabled: true
  destination: otlp
  endpoint: http://localhost:4318/v1/traces
  capture: metadata
```

You may set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` instead of `endpoint`. Authentication
and routing for a generic collector should be handled by that collector or its local
network boundary in this first release.

Internally, lifecycle events go through a small exporter interface before the OTLP
mapping. This keeps W&B details out of the agent loop and leaves room for additional
native or custom destinations later without changing how sessions, turns, tools,
and subagents are instrumented.

## Privacy and reliability

The default `capture: metadata` sends lifecycle attributes and metrics, not prompts,
model responses, tool arguments, or tool results. Set `capture: content` only when
the destination is approved to receive repository and conversation content:

```yaml
observability:
  enabled: true
  destination: wandb
  entity: my-team
  capture: content
  max_content_chars: 100000
```

`--observe-capture content` sets the same level for a single run without editing
configuration.

Content capture applies recursive secret redaction and per-value size bounds before
export. Oversized values are reduced structurally — long strings are shortened and
long message lists keep their head and tail around a `…[truncated N items]` marker —
so an exported payload remains valid JSON that a viewer can render as a tree.
Redaction runs before truncation and is defense in depth, not a substitute for
reviewing what may leave your machine. CCA does not export provider-internal
reasoning as observability content.

Export is batched and fail-open. A slow or unavailable destination does not fail an
agent turn. CCA attempts a bounded flush during shutdown and records exporter error
counts on the session span when possible. Consequently, an abrupt process kill can
lose the newest remote spans; the unchanged local trace files remain available.

## Configuration reference


| Key                 | Default                                                    | Meaning                                                  |
| ------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| `enabled`           | `false`                                                    | Enables remote lifecycle export                          |
| `destination`       | `wandb` when enabled                                       | `wandb` or `otlp`                                        |
| `endpoint`          | W&B cloud endpoint or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | OTLP/HTTP traces URL                                     |
| `entity`            | `WANDB_ENTITY`                                             | Required W&B entity/team                                 |
| `project`           | `WANDB_PROJECT` or `cca`                                   | W&B project                                              |
| `capture`           | `metadata`                                                 | `metadata` or explicit `content` capture                 |
| `sample_rate`       | `1.0`                                                      | Trace sampling rate from `0.0` to `1.0`                  |
| `max_content_chars` | `100000`                                                   | Serialized character budget per captured content value   |
| `flush_timeout`     | `2.0`                                                      | Maximum shutdown flush time in seconds                   |
| `service_name`      | `cca`                                                      | OpenTelemetry service name                               |
