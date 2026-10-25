# Changelog

All notable changes to **cca** are recorded here.

Format: `## [version] - YYYY-MM-DD` → bullet points under **Added / Changed / Fixed / Removed**.
Keep the newest version at the top. Update this file with every PR or meaningful commit.

---

## [0.3.2] - 2026-10-25

### Added
- LiteLLM proxy provider (`litellm:<route>`), configured through `LITE_LLM_KEY` and `LITE_LLM_URL`, with thinking-level passthrough for routed GPT/o-series, Gemini, Claude, DeepSeek, Qwen/QwQ, Kimi, GLM, Grok, and Magistral models
- `--observe-capture metadata|content` to set observability payload detail for a single run without editing configuration
- GenAI provider attributes on LLM spans (`gen_ai.system`, `gen_ai.provider.name`, `gen_ai.response.model`) so backends can attribute model cost
- `input.mime_type`/`output.mime_type` alongside captured content values, so viewers render structured payloads as a tree instead of an opaque string
- `reasoning_replay` in the resolved request settings and every call log, recording whether reasoning the route returns is echoed back on later calls
- Thinking profiles for the Fireworks Nemotron, Inkling, and Muse routes, which previously resolved to thinking `off` and sent no effort parameter
- CI workflow running the test suite on Python 3.11–3.13, `pip-audit` on `uv.lock`, and `npm audit` plus a build of the website, on every push, pull request, and weekly
- Dependabot updates for Python, website, and GitHub Actions dependencies, and a `SECURITY.md` describing how to report vulnerabilities

### Changed
- Session work tracking now uses `WorkCreate`, `WorkGet`, `WorkList`, and
  `WorkUpdate`; background execution uses `JobOutput` and `JobCancel`, with
  `job_id` identifiers and `/work` replacing the previous task-oriented names
- The warning alias for automatic tool approval is now
  `--unsafe-auto-approve`
- Oversized captured content is now reduced structurally — long strings shorten and long message lists keep head and tail around a `…[truncated N items]` marker — so exported payloads stay valid JSON instead of being sliced mid-serialization
- Raised the default `max_content_chars` from `8000` to `100000`, enough to carry a real message list under content capture
- Captured plain strings are exported unquoted rather than JSON-encoded
- LiteLLM routes now echo returned `reasoning_content` back on later assistant messages by default, as the other coding agents do. Families whose reasoning travels another way opt out: OpenAI, Claude, Gemma, Grok, Magistral, Groq routes, and unregistered routes behind OpenAI, Azure, Anthropic, Bedrock, Mistral, or Groq
- Project instructions now load from `AGENTS.md` and `AGENTS.local.md` at each directory level; `CCA.md`, `CCA.local.md`, and `CLAUDE.md` are no longer read, and `/init` writes `AGENTS.md`
- Built-in subagent prompts rewritten to be shorter and explicit about what each agent returns
- Minimum dependency versions raised past known advisories: `Pillow>=13.0.0`, `pypdf>=6.21.2`, `mcp>=1.30.0`, and `opentelemetry-sdk`/`opentelemetry-exporter-otlp-proto-http>=1.45.1`; the locked Python and website dependencies were upgraded to versions with no known advisories

### Fixed
- Fresh installs resolved `mcp` 2, which renamed the streamable HTTP client CCA uses, so HTTP MCP servers failed to connect; `mcp` is now capped below 2
- `cca sessions list` and `--continue` no longer read another tool's history file at `~/.claude/history.jsonl`, which listed sessions CCA could not load
- Emergency compaction now retains up to the 100 most recent messages at a
  text-user boundary instead of crashing on an undefined
  `COMPACT_MAX_MESSAGES` name when every LLM compactor attempt fails
- Gemini 2.5/3.x routes behind LiteLLM now resolve an internal `off` thinking request to the model minimum instead of sending `reasoning_effort=none`, which the proxy renders as `thinking_level: MINIMAL` and Gemini 3.x flash routes reject with a 400 — this failed every structured-output run on those routes
- `gen_ai.request.model` carried the full `provider:model` selector instead of the bare model id, preventing cost attribution from matching; the selector is now preserved separately in `cca.model_selector`
- Inline secret redaction consumed only the auth scheme, so `authorization: Bearer <token>` redacted `Bearer` and exported the token
- Kimi, GLM, Qwen, Gemini, Nemotron, Inkling, and Muse routes dropped their earlier reasoning from later requests; only DeepSeek and MiniMax replayed it
- `--machine` mode now prints the unsupported-thinking warning to stderr, as `--print` already did, instead of silently running with thinking off
- `--json-schema` no longer fails when the provider refuses the formatting call (`content_filter` or `refusal`, seen on claude-opus-5 and claude-fable-5 for a field named `internal_thought`): it retries once with reasoning-like top-level field names replaced by neutral aliases and maps them back before validating. Requests are unchanged when the provider does not refuse, and a successful retry reports `field_aliases`

### Removed
- Legacy subagent name aliases; agents are addressed only by their built-in names (`worker`, `scout`, `blueprint`, `reviewer`, `verifier`, `risk-auditor`, `researcher`)
- Read-only fallbacks for the pre-0.2 `~/.claude/sessions/`, `~/.claude/history.jsonl`, and `<project>/.claude/memory/` locations

---

## [0.3.1] - 2026-08-17

### Added
- Joined `Agent.tasks` fan-out/fan-in batches with runtime-managed concurrency, stable result ordering, shared spawn-group provenance, and one parent synthesis boundary
- CCA-native global/project subagent definitions, seven built-in profiles, depth-one traced execution, whole-session `--agent` profiles, durable completed-result retrieval, and validated separated SFT exports through `cca traces export`
- Trace-safe interactive `/agents` help/catalog inspection, session-default selection, and scoped background status/output/cancellation controls
- `cca traces inspect` terminal-tree and JSON views for lineage, lifecycle, replay diagnostics, and per-trajectory SFT readiness
- Lossless `cca traces export --format calls`, generation-safe trajectory export, explicit compaction/clear/run provenance, and call-coverage diagnostics
- Opt-in OpenTelemetry lifecycle tracing for W&B Weave and other OTLP/HTTP backends, with no observability network activity by default
- Curated bundled metadata for 558 models across 17 provider definitions, covering OpenAI, Claude, Gemini, Grok, GLM, Kimi, DeepSeek, Qwen, and selected popular routing providers
- `cca models` and `/models` catalog browsing, search, family/reasoning filters, detailed model inspection, and JSON output
- Catalog-backed endpoints and environment-key discovery for xAI, Z.AI, Moonshot, DeepSeek, Qwen Portal, Alibaba Coding Plan, Fireworks, and Cerebras
- Safe resolved runtime/backend settings at startup, through `/settings` and `--show-settings`, in machine `ready` events, JSON results, session files, and per-call traces

### Changed
- The OpenAI-compatible provider adapter now preserves approved opaque tool-continuation metadata through the same internal state contract in streaming and non-streaming root/child calls; clean SFT export strips that state generically
- Subagents now default to joined foreground execution; detached background execution is opt-in through `subagents.allow_background: true` and is omitted from the tool schema otherwise
- Subagent calls now inherit root tool filters, permissions, hooks, project context, and trace lineage; mutating children are serialized and nested delegation is blocked at dispatch
- Built-in and newly defined agents now default to 40 model turns; the final turn is reserved for a tool-free, structured parent handoff when the working budget is exhausted
- Compactor traces retain exact full inputs, micro-compaction archives tool results before mutation, and resumed sessions restore compaction metadata with monotonic root call indexes
- Subagents now rebind to `/model` backend changes, deny interactive/task-control tools, scope model-facing task controls to the root trace, and reserve context for a graceful `context_cap` handoff
- Exact catalog entries now enrich output-token upper limits while existing fallback and thinking request policies remain intact
- Unspecified thinking is now an explicit `auto` policy with deterministic, model-aware resolution: reasoning-capable models default to `medium`, while impossible `off` requests visibly resolve to the model minimum
- `--preset deterministic` (and its `sft` alias) now ships with `background_bash` **on**: `Bash` keeps its `run_in_background` parameter and background completions are delivered as `<system-reminder>` turns. The preset no longer guarantees a trajectory free of out-of-band injections; copy it with `background_bash: false` if you need that.

### Fixed
- OpenAI-compatible child runs now round-trip nested provider continuation state across tool calls instead of failing on the second request when an adapter requires opaque replay metadata

---

## [0.3.0] - 2026-07-12

### Added
- `cca skills add`, `cca skills list`, and `cca skills remove` commands for managing skills from Git repositories
- Per-configuration MCP trust store, interactive trust prompts, `/mcp trust|untrust`, and Bearer-token authentication
- Public MCP setup and usage guide

### Changed
- Python import package renamed to `cca`; distribution renamed to `cca-cli`
- Compaction, `/init`, hook, plan-mode, and memory-staleness prompts refined.
- Source distributions now include the public documentation linked from the README

### Fixed
- Session startup and shutdown now run hooks and MCP lifecycle handling in print and machine modes
- MCP server loading no longer exposes stored authentication tokens and now requires explicit trust
- Prompt boxes resize correctly, and sessions no longer hang when the input UI exits mid-turn

---

## [0.2.4] - 2026-07-11

### Added
- Feature toggles & presets: `features:`/`tools:`/`preset:` config fields, `--preset` flag (with `CCA_PRESET` fallback), `cca features` command and `/features` REPL command; tool registration gated behind the resolved `FeatureSet`, with a built-in `deterministic` preset
- Feature payload stamped into every LLM call record and session saves for provenance
- Public website (Astro Starlight): landing page, docs, blog, and changelog, deployed to GitHub Pages
- Boxed startup banner with CCA ascii-art logo and aligned session info grid

### Changed
- `SFT_MODE` removed in favour of the feature-toggle system

---

## [0.2.3] - 2026-06-28

### Added
- Interleaved tool narration for better visibility into agent actions

### Fixed
- Save pre-compaction session to disk before messages are wiped
- Compaction threshold set to 100k tokens, max output token to 32k
- LLM hallucination or malformed responses now retried or surfaced as error messages
- `cca --machine` mode crash when prompting user

---

## [0.2.2] - 2026-06-15

### Added
- `--machine` mode: JSON-lines session protocol for programmatic/non-interactive use
- `/thinking` slash command to toggle extended thinking (some model families not yet supported)
- `/skills` slash command to toggle individual skills on/off at runtime
- Prompt-caching support with a caching status indicator in the status bar

### Fixed
- Session export correctness in `--machine` mode
- `/plan` UI layout glitch

---

## [0.2.1] - 2026-06-08

### Added
- `/init` now auto-loads `CCA.md` from the project directory when present

### Fixed
- the duckduckgo searching package warning -> ddgs>=9.0.0
- Background bash completions and harness notifications now delivered as `<system-reminder>` turns instead of fake user speech; memory extractor skipped on harness-initiated turns (`harness_turn` flag on `process_turn`)

---

## [0.2.0] - 2026-06-03

### Added
- `/plan` slash command and inline plan mode
- `/compact` slash command for manual context compaction
- Inline command palette triggered by `/` with available tool list
- `Option+Backspace` / `Option+Delete` keybindings to delete whole words
- Dual-threshold compaction system (150 K lower / 200 K upper token limits)
- Tool result micro-compaction (snipping) for safe upper-limit checks
- Pending tasks displayed in a blue panel above the input area while agent is busy
- Turn limit raised in `--print` mode with a notification when the limit is hit
- System prompt snapshot stored in session files
- MCP support, LSP tool, vim keybindings, session resume, and new startup flags
- Hooks system, disk-backed tasks, token budget, provider-agnostic thinking
- Storage schema changed for sessions (flat JSON + `history.jsonl` index, timestamped IDs, parent-child linkage)
- `.cca/` config automatically copied into new git worktrees
- `ultrathink` keyword detection with `max_tokens` escalation
- Large tool results persisted to disk and re-injected after compaction
- Sliding shimmer effect on action label while running
- Aggregate tool result cap; `always-allow` grants persisted to config
- Compaction threshold made configurable (default 150 K tokens)

### Changed
- Ctrl+C: first press clears input, second press exits the app
- `'Pending Task'` label removed; pending prompts styled as standard blue boxes
- Compaction and plan-mode prompts updated
- System prompt split for Anthropic prompt caching
- Compaction message limits increased to reduce frequency
- Default lower compaction threshold changed to 100 K
- Only new input tokens counted per API call (not cumulative total)
- User input shown as a blue panel; token counts fixed for OpenAI-compatible streaming
- Permission dialog, config system, token tracking, and cancellation overhauled

### Fixed
- Planning mode auto-exit glitching
- `AskUserQuestion` collapse behaviour
- Thinking parser errors
- Git env info retry logic
- OpenAI API unknown-kwargs error
- Missing `ERR_INDICATOR` constant in `ui/components.py`
- All status/diagnostic output routed to stderr in `--print` mode
- Green result panel suppressed in `--print` mode (eliminated double output)
- Second Ctrl+C now reliably exits via `app.exit()` instead of `os.kill`
- Thinking/tools kept visible on cancel; "Cancelled" printed below them
- Robust history splitting during token compaction to maintain safe tool-boundary API requirements
- Message-count fallback for compaction removed
- Truncated identifiers and imports in `input_app.py` repaired
- Head-drop compaction retry; write tools restricted to working directory
- Ambiguous variable name `l` in diff generators

---

## [0.1.0] - initial release

### Added
- Initial commit: cca — Code CLI Agent
- Core UI with indentation-based layout (no boxes)
- Token count display

---

<!-- HOW TO ADD AN ENTRY
1. If bumping the version, add a new ## [x.y.z] - YYYY-MM-DD block at the top.
2. If still on the current version, append bullets to the appropriate section.
3. Use one of: Added / Changed / Fixed / Removed.
4. Keep bullets short (one line). Link to PRs/commits if helpful: ([#42](…)).
-->
