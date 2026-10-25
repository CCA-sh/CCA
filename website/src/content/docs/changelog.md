---
title: Changelog
---

Release notes for cca. Docs on this site always describe the latest
release.

## 0.3.2 — 2026-10-25

- Renamed session planning tools to `WorkCreate`, `WorkGet`, `WorkList`, and
  `WorkUpdate`.
- Renamed background lifecycle tools to `JobOutput` and `JobCancel`, with
  `job_id` identifiers.
- Replaced the `/tasks` command with `/work` and the automatic-approval warning
  alias with `--unsafe-auto-approve`.

## 0.3.1 — 2026-08-17

- Added native global and project subagents, seven built-in profiles, joined
  `Agent.tasks` fan-out/fan-in batches, root profiles, and durable background
  lifecycle controls.
- Added separated root/child traces, lineage and readiness inspection with `cca
  traces inspect`, plus validated trajectory and lossless call-level SFT exports.
- Added opt-in OpenTelemetry lifecycle tracing for W&B Weave and other OTLP/HTTP
  backends; observability remains completely inactive by default.
- Added a bundled model/provider catalog, `cca models` and `/models` browsing,
  model-aware thinking defaults, and resolved runtime/backend settings reports.
- Made detached background subagents opt-in and preserved opaque provider
  continuation state consistently across streaming and non-streaming child calls.
- Improved compaction provenance, context-generation safety, subagent permission
  inheritance, turn/context-cap handoffs, and clean SFT filtering.

## 0.3.0 — 2026-07-12

- Python imports now use `cca`; the PyPI distribution is now `cca-cli`.
- Added Git-backed skill management with `cca skills add`, `list`, and `remove`.
- Added explicit MCP server trust, Bearer-token authentication, and `/mcp
  trust|untrust` controls.
- Refined compaction, initialization, hook, plan-mode, and memory-staleness
  prompts.
- Fixed lifecycle handling in print and machine modes, prompt resizing, MCP
  loading and token exposure, and hangs when the input UI exits mid-turn.

## 0.2.4 — 2026-07-11

- Feature toggles & presets: configure features and tools per project, pick a
  preset with `--preset`, inspect the result with `cca features` or `/features`.
- Feature payload recorded with every LLM call and session save for provenance.
- Initial public website: documentation, blog, and this changelog.
- New boxed startup banner with the CCA ascii-art logo.

For the full commit-level history, see
[GitHub releases](https://github.com/CCA-sh/CCA/releases).
