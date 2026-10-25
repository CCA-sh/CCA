"""Main REPL — the core agent loop.

Architecture:
  User input → messages → LLM (streaming) → tool calls → tool results → LLM → ...
  Continues until LLM produces no tool calls, then waits for next user input.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from cca import __version__
from cca.llm.base import (
    ContentBlock,
    Message,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from cca.llm.retry import LLMUnavailable
from cca.hooks import HookRunner
from cca.observability.semconv import (
    model_attributes as _model_attributes,
    response_model_attributes as _response_model_attributes,
)
from cca.permissions import PermissionManager

# Fallback console used when no TerminalUI is active (tests, --print mode, etc.)
_fallback_console = Console()
# In --print mode all status/diagnostic output goes to stderr so stdout stays clean.
_stderr_console = Console(stderr=True)

# Color scheme for --trace mode
_TRACE_COLORS = {
    "user":      ("cyan",  "USER"),
    "assistant": ("green", "ASSISTANT"),
}

def _render_trace_message(role: str, text: str) -> None:
    color, label = _TRACE_COLORS.get(role, ("white", role.upper()))
    _fallback_console.print(Panel(
        escape(text),
        title=f"[bold {color}]{label}[/bold {color}]",
        border_style=color,
        padding=(0, 1),
    ))

MAX_TURNS = 50        # interactive default: safety cap against runaway loops
MAX_TURNS_PRINT = 200  # headless default: batch tasks run to completion
EMERGENCY_KEEP_MESSAGES = 100  # retained only when every compactor attempt fails

# ---------------------------------------------------------------------------
# /init repository-guidance setup prompt
# ---------------------------------------------------------------------------

_INIT_PROMPT = """\
Prepare concise repository guidance for CCA. The durable outputs are a shared \
`AGENTS.md`, an optional private `AGENTS.local.md`, and—when it would add real value—\
repository hooks in `.cca/config.yaml`.

## Build an evidence brief

Inspect the repository before interviewing the user. Read any existing instruction \
files, contributor documentation, manifests, task runners, CI workflows, and \
formatter or linter configuration. Use an exploration subagent when the repository \
is large enough that parallel discovery will save time.

Reduce what you find to candidate instructions. Each candidate needs a source and \
must answer both questions:

1. Is this specific to the repository or the user's local workflow?
2. Would an agent plausibly choose the wrong command, location, convention, or \
sequence without being told?

Discard facts that are easy to rediscover, ordinary language conventions, broad \
quality advice, and inventories of the directory tree.

## Agree on the outputs

After the evidence pass, use AskUserQuestion to offer an informed choice among a \
shared project file, a private local file, or both. Recommend an option based on \
what already exists. Ask about hooks separately only when you found a concrete \
automation opportunity.

Collect missing information in one focused follow-up where possible. For shared \
guidance, ask only about undocumented team practices that affect implementation or \
verification. For personal guidance, ask which local workflow preferences, role \
context, or non-secret environment identifiers would help. Never request or record \
passwords, tokens, private keys, or other credentials.

## Assemble the selected guidance

Write shared, stable repository knowledge to `AGENTS.md` at the project root. Favor \
short operational statements covering commands that require special flags or \
ordering, architectural boundaries, non-default conventions, environment \
prerequisites, test selection, and contribution rules. Organize the file for quick \
scanning; omit a heading when there is nothing useful beneath it.

Write user-specific material only to `AGENTS.local.md`. Before creating that file, \
ensure the repository's `.gitignore` excludes it. Local paths, sandbox identifiers, \
and communication preferences are appropriate; secrets are not.

Preserve existing work:

- When a destination already exists, read it and make a minimal, reviewable edit \
instead of replacing it wholesale.
- Reuse useful rules from other agent configuration only after adapting them to \
CCA's files and actual repository behavior.

## Add hooks only when justified

If the user opts in, propose deterministic hooks tied to tools or lifecycle events \
that exist in CCA. Show the exact YAML before changing `.cca/config.yaml`, retain \
unrelated settings, and merge with existing hook entries. Matchers are shell-style \
glob patterns over one tool name; use separate entries when the same command should \
run for both `Write` and `Edit`.

Do not add a hook merely because a formatter is installed. Confirm that its command \
works with the file path supplied in the hook's JSON input and that automatic \
execution will not rewrite unrelated files.

## Validate and report

Re-read every changed file. Check that private guidance is ignored by Git, that any \
edited YAML still parses, and that each retained instruction is supported by repo \
evidence or an explicit user answer. Report the files changed, the decisions encoded \
in each, and any questions intentionally left unresolved.\
"""

# Single source of truth for slash commands — used by autocomplete and /help
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",          "Show this help"),
    ("/plan",          "Plan before implementing (e.g. /plan fix the auth bug)"),
    ("/think",         "Set thinking policy: auto | off | low | medium | high"),
    ("/settings",      "Show the exact resolved runtime/backend settings"),
    ("/init",          "Generate or improve AGENTS.md for this project"),
    ("/doctor",        "Check API keys, model config, and tool availability"),
    ("/hooks",         "List configured hooks"),
    ("/clear",         "Clear conversation history"),
    ("/model",         "Show or switch model (e.g. openai:gpt-4o)"),
    ("/models",        "List bundled providers or models"),
    ("/key",           "Show or set API key for provider"),
    ("/config",        "Show current configuration"),
    ("/memory",        "List stored memories"),
    ("/work",          "List session work items"),
    ("/agents",        "List and control this session's subagents"),
    ("/tokens",        "Show token usage"),
    ("/budget",        "Set/show token budget (e.g. /budget 500k)"),
    ("/compact",       "Force context compaction now"),
    ("/resume",        "Load a previous session (omit id to list)"),
    ("/review",        "Code review for target or current directory"),
    ("/allowed-tools", "Toggle tool in always-allow list"),
    ("/mcp",           "MCP server management"),
    ("/verbose",       "Toggle verbose tool output"),
    ("/trace",         "Toggle trace mode (show all LLM inputs/outputs)"),
    ("/vim",           "Toggle vim keybindings"),
    ("/cwd",           "Show or change working directory"),
    ("/skills",        "List skills or toggle skills on/off (e.g. /skills on)"),
    ("/features",      "Show resolved feature toggles and tool availability"),
    ("/quit",          "Quit"),
]

_THINK_LEVELS = ("auto", "off", "low", "medium", "high")


def _think_support(model: str) -> tuple[str, str]:
    """Return (support_type, human_description) for the given model string.

    support_type:
      "full"     — low/medium/high all work with distinct effect
      "adaptive" — thinking can be on/off only; all levels collapse to one mode
      "none"     — model does not support thinking at all
    """
    from cca.llm.thinking import thinking_support
    return thinking_support(model)

# Keyword in a user message forces a one-turn high-level override
_ULTRATHINK_KEYWORDS = (
    "ultrathink",
    "think harder",
    "think deeply",
    "think longer",
    "think more carefully",
    "think step by step",
    "think it through",
)


def _keyword_think_level(user_text: str) -> str | None:
    """Return "high" if the message contains an ultrathink keyword, else None."""
    lower = user_text.lower()
    if any(kw in lower for kw in _ULTRATHINK_KEYWORDS):
        return "high"
    return None


def _thinking_budget(user_text: str) -> tuple[int, str]:
    """Legacy helper kept for backward compatibility."""
    if _keyword_think_level(user_text):
        return 32_000, "adaptive"
    return 0, "enabled"


_TOKEN_BUDGET_MIN = 1_024

_BUDGET_MULTIPLIERS = {"k": 1_000, "m": 1_000_000}

_BUDGET_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([km])?", re.IGNORECASE)

_INLINE_BUDGET_RE = re.compile(
    r"(?:^|\s)/budget\s+(\d+(?:\.\d+)?)\s*([km])?\b",
    re.IGNORECASE,
)


def _parse_budget_amount(text: str) -> int | None:
    """Parse a bare quantity like "500k", "2.5m", "200000" — no unit word.

    Used where the caller already knows the text is a budget amount (the
    `/budget` slash command's argument, or the captured group of an inline
    `/budget <amount>` match) — there's no ambiguity left to guard against
    at this point, so no floor or keyword is applied here.
    """
    m = _BUDGET_AMOUNT_RE.fullmatch(text.strip())
    if not m:
        return None
    val = float(m.group(1)) * _BUDGET_MULTIPLIERS.get((m.group(2) or "").lower(), 1)
    return int(val)


def _parse_token_budget(text: str) -> int | None:
    """Parse an inline `/budget <amount>` directive anywhere in a message.

    Only the literal `/budget` sigil triggers this — never a bare number or
    natural-language phrase. Earlier versions matched a bare "+N" or a
    freestanding "N tokens" anywhere in the text, but both collide with
    ordinary content: unified diffs are full of "+N" sequences (hunk
    headers like "@@ -6,3 +6,5 @@", new-file headers like "+++ b/path"),
    and "N tokens" is a phrase that can appear in prose or pasted docs. A
    harness embedding a diff in a turn's prompt had a hunk number silently
    parsed as a token budget, collapsing that call's max_tokens to a
    handful of tokens and starving the model's reasoning of any room to
    answer in. `/budget` is not a phrase or pattern that occurs by
    coincidence in code, diffs, or natural language, so it's safe to
    recognize anywhere in the message rather than requiring a whole-message
    match — "explain this codebase /budget 500k" works as expected.

    Returns the token count as an integer, or None if no directive matched
    or the parsed value is too small to be a real request.
    """
    m = _INLINE_BUDGET_RE.search(text)
    if not m:
        return None
    budget = _parse_budget_amount(f"{m.group(1)}{m.group(2) or ''}")
    return budget if budget is not None and budget >= _TOKEN_BUDGET_MIN else None

# Memory extraction thresholds
_MEM_MIN_TOKENS_TO_INIT = 10_000
_MEM_MIN_TOKENS_BETWEEN_UPDATE = 5_000
_MEM_TOOL_CALLS_BETWEEN_UPDATES = 3


class Repl:
    def __init__(
        self,
        backend,
        registry,
        config,
        permissions: PermissionManager,
        cwd: str | None = None,
        print_mode: bool = False,
        ui=None,  # TerminalUI | None — injected by cli.py
        # Session resume
        resume_messages: "list[Message] | None" = None,
        session_id: str | None = None,
        resume_state: "dict | None" = None,
        # Output control
        output_format: str = "text",          # text | json | stream-json
        # JSON Schema the --print answer must satisfy; None disables the
        # formatting phase entirely (every interactive and machine-mode run).
        output_schema: "dict | None" = None,
        max_turns: int | None = None,         # None → use module-level MAX_TURNS
        # System prompt overrides
        system_prompt: str | None = None,     # replaces the base system prompt
        append_system_prompt: str | None = None,  # appended to the base
        # Tool filtering
        allowed_tools: "set[str] | None" = None,
        disallowed_tools: "set[str] | None" = None,
        # Thinking level
        think_level: str = "off",
        thinking_resolution=None,
        # Request a provider-generated readable reasoning summary.
        reasoning_summary: bool = False,
        # Print the complete safe runtime settings snapshot at startup.
        show_settings: bool = False,
        # interactive | print | machine
        run_mode: str = "interactive",
        # Machine-mode verbosity: emit intermediate events (tool_use, assistant_text)
        machine_verbose: bool = False,
        # Resolved feature toggles; None resolves defaults from config
        feature_set=None,
        # Resolved per-run randomization; None means no randomization
        randomization=None,
        # Optional named profile applied to the root session itself.
        agent_profile=None,
        # Optional lifecycle observer; default resolution is a true no-op.
        observability_settings=None,
        observer=None,
    ):
        self.backend = backend
        self.registry = registry
        self.config = config
        from cca.features import resolve_features
        from cca.randomize import RandomizationPlan
        self._features = feature_set if feature_set is not None else resolve_features(config)
        self._randomization = randomization if randomization is not None else RandomizationPlan.identity()
        self._agent_profile = agent_profile
        self.permissions = permissions
        self.cwd = cwd or os.getcwd()
        self.print_mode = print_mode
        self._ui = ui  # may be None (tests, --print mode)
        # Machine mode's stdout carries the JSON-lines protocol, so console
        # prose must never land there or an orchestrator silently drops it.
        self._console: Console = ui.console if ui is not None else (
            _stderr_console
            if print_mode or run_mode == "machine"
            else _fallback_console
        )
        # Freeze one resolved catalog for the interactive session. When Agent is
        # disabled this remains an inspection-only snapshot and never enters the
        # model scaffold or trace unless the user later enables Agent in a new run.
        from cca.agents import discover_agents
        from cca.tools.agent_tool import AgentTool
        active_agent_tool = (
            registry.get("Agent")
            if registry is not None and callable(getattr(registry, "get", None))
            else None
        )
        self._agent_catalog_snapshot = (
            active_agent_tool.catalog
            if isinstance(active_agent_tool, AgentTool)
            else discover_agents(self.cwd)
        )
        self._output_format = output_format
        self._output_schema = output_schema
        self._max_turns = max_turns
        self._system_prompt_override = system_prompt
        self._system_prompt_append = append_system_prompt
        self._allowed_tools = allowed_tools  # None = no restriction
        self._disallowed_tools = disallowed_tools or set()
        self.messages: list[Message] = list(resume_messages) if resume_messages else []
        self._token_counts: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._last_input_tokens = 0  # input tokens reported by the previous API call
        self._turn_count = 0
        self._notif_queue: asyncio.Queue | None = None
        self._trace_tools_shown: bool = False
        self._token_budget: int | None = None  # explicit token budget from user
        from cca.llm.thinking import ThinkingResolution
        self._thinking_resolution = thinking_resolution or ThinkingResolution(
            requested=think_level,
            effective=think_level,
            source="direct",
            support=_think_support(config.model)[0],
            reason="provided directly to Repl",
        )
        self._think_level: str = self._thinking_resolution.effective
        self._reasoning_summary = reasoning_summary
        self._show_settings = show_settings
        self._run_mode = run_mode
        self._machine_verbose: bool = machine_verbose
        # Throttle accounting, reset per turn by _process_turn_impl. Time spent
        # waiting on a busy provider is not the agent working and not the user
        # thinking, so it is reported separately rather than folded into the
        # turn's duration.
        self._throttle_wait_s: float = 0.0
        self._throttle_events: int = 0
        self._turn_throttled: bool = False
        from cca.llm import retry as _retry
        _retry.set_wait_observer(self._on_llm_wait)
        self._compaction_summary: str | None = None  # injected into system prompt after compaction
        from cca.prompts import COMPACTION_PROMPT as _COMPACTION_PROMPT
        self._compaction_prompt: str = self._randomization.compaction_prompt or _COMPACTION_PROMPT
        # None when randomization is off, which keeps the key out of session files
        # and call logs entirely rather than writing a null.
        self._randomization_payload: dict | None = (
            None if self._randomization.is_identity() else self._randomization.to_payload())
        import uuid as _uuid
        self._session_id: str = session_id or _uuid.uuid4().hex[:12]
        self._run_id: str = _uuid.uuid4().hex
        self._continuation_reason: str = "resume" if resume_state else "initial"
        # LLM call logging state
        self._base_session_id: str = re.sub(r'-comp\d+$', '', self._session_id)
        from cca.trace_store import TraceIdentity, TraceStore, generation_from_session_id
        self._trace_id: str = self._base_session_id
        self._agent_id: str = self._base_session_id
        self._context_generation: int = generation_from_session_id(self._session_id)
        self._generation_reason: str = (
            "compaction" if self._context_generation > 0 else "initial"
        )
        self._trace_store = TraceStore(self._base_session_id)
        self._root_trace_identity = TraceIdentity(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            generation=self._context_generation,
            context_generation=self._context_generation,
            generation_reason=self._generation_reason,
        )
        self._llm_call_idx: int = self._trace_store.next_call_idx(
            self._root_trace_identity, child=False
        )
        self._msg_index_offset: int = 0   # absolute message offset for compacted_from_index
        # Memory extraction state
        self._memory_facts: list[str] = []           # last extracted relevant facts
        self._extractor_messages: list[Message] = [] # stateful extractor conversation
        self._mem_tokens_at_last_extraction: int = 0
        self._mem_tool_calls_since_extraction: int = 0
        # Track recently-read files for post-compaction re-injection
        self._recently_read: list[str] = []          # file paths, most-recent-last, max 5
        self._post_compact_files: dict[str, str] = {}  # path → content, injected post-compact
        # Session directory for persisted tool results
        from cca.config import CONFIG_DIR
        self._tool_results_dir = CONFIG_DIR / "sessions" / self._session_id / "tool-results"
        # Hooks runner
        self._hooks = HookRunner(
            config.hooks if self._features.feature("hooks") else {},
            session_id=self._session_id,
            cwd=self.cwd,
        )
        # Project instruction files (AGENTS.md / AGENTS.local.md)
        # Loaded once at startup
        from cca.instructions import collect as _collect_instructions
        self._project_instructions, self._loaded_instruction_files = (
            _collect_instructions(self.cwd)
        )
        if resume_state:
            self._trace_id = str(resume_state.get("trace_id") or self._trace_id)
            self._agent_id = str(resume_state.get("agent_id") or self._agent_id)
            saved_generation = resume_state.get("context_generation")
            if not isinstance(saved_generation, int) or saved_generation < 0:
                saved_generation = generation_from_session_id(self._session_id)
            self._context_generation = saved_generation
            saved_reason = resume_state.get("generation_reason")
            self._generation_reason = (
                saved_reason
                if saved_reason in {"initial", "compaction", "clear"}
                else ("compaction" if saved_generation > 0 else "initial")
            )
            self._root_trace_identity = TraceIdentity(
                trace_id=self._trace_id,
                agent_id=self._agent_id,
                generation=self._context_generation,
                context_generation=self._context_generation,
                generation_reason=self._generation_reason,
            )
            self._llm_call_idx = self._trace_store.next_call_idx(
                self._root_trace_identity, child=False
            )
            token_counts = resume_state.get("token_counts")
            if isinstance(token_counts, dict):
                self._token_counts.update({
                    key: int(value) for key, value in token_counts.items()
                    if key in self._token_counts and isinstance(value, int)
                })
            summary = resume_state.get("compaction_summary")
            self._compaction_summary = summary if isinstance(summary, str) else None
            offset = resume_state.get("message_index_offset", 0)
            self._msg_index_offset = offset if isinstance(offset, int) and offset >= 0 else 0
            self._restore_subagent_default(resume_state)

        from cca.observability import (
            create_observer as _create_observer,
            resolve_observability as _resolve_observability,
        )
        raw_observability = getattr(config, "observability", {})
        if not isinstance(raw_observability, dict):
            raw_observability = {}
        self._observability_settings = (
            observability_settings
            if observability_settings is not None
            else (
                observer.settings
                if observer is not None
                else _resolve_observability(raw_observability)
            )
        )
        self._observer = observer or _create_observer(self._observability_settings)
        self._observer.add_common_attributes({
            "cca.trace_id": self._trace_id,
            "cca.run_id": self._run_id,
            "cca.run_mode": self._run_mode,
            "cca.agent_id": self._agent_id,
            "cca.session_id": self._session_id,
            "cca.context_generation": self._context_generation,
            "gen_ai.request.model": getattr(config, "model", None),
        })
        self._session_span = None
        self._turn_observation_index = 0

    def _session_state_payload(self) -> dict:
        payload = {
            "trace_id": self._trace_id,
            "agent_id": self._agent_id,
            # Keep the legacy field while making model-visible context epochs
            # explicit for new readers.
            "generation": self._context_generation,
            "context_generation": self._context_generation,
            "generation_reason": self._generation_reason,
            "next_call_idx": self._llm_call_idx,
            "message_index_offset": self._msg_index_offset,
            "compaction_summary": self._compaction_summary,
            "token_counts": dict(self._token_counts),
        }
        agent_tool = self._active_agent_tool()
        if agent_tool is not None:
            payload["subagent_default"] = agent_tool.default_agent
        return payload

    def _panel_border(self, default: str) -> str:
        """Return the plan mode colour when active, otherwise the default."""
        from cca.tools.plan_tools import is_plan_mode_active
        return "#8f2d56" if is_plan_mode_active() else default

    _BANNER_ART = """\
 ██████╗ ██████╗ █████╗
██╔════╝██╔════╝██╔══██╗
██║     ██║     ███████║
██║     ██║     ██╔══██║
╚██████╗╚██████╗██║  ██║
 ╚═════╝ ╚═════╝╚═╝  ╚═╝"""

    def _describe_backend_request(
        self, think_level: str, reasoning_summary: bool = False
    ) -> dict:
        """Describe request controls without requiring third-party test backends."""
        describe = getattr(self.backend, "describe_request_settings", None)
        if callable(describe):
            try:
                result = describe(think_level, reasoning_summary)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        return {
            "adapter": type(self.backend).__name__,
            "thinking_parameters": {},
            "reasoning_summary": reasoning_summary,
        }

    def _runtime_settings_payload(self) -> dict:
        """Return the safe, resolved settings that govern this running session."""
        from urllib.parse import urlsplit, urlunsplit
        from cca.config import CONFIG_FILE
        from cca.llm.catalog import get_provider, resolve_model, split_selector
        from cca.llm.thinking import resolve_thinking

        provider, model_id = split_selector(self.config.model)
        provider_info = get_provider(provider)
        model_info = resolve_model(self.config.model)

        endpoint = None
        client = getattr(self.backend, "client", None)
        raw_endpoint = getattr(client, "base_url", None)
        if raw_endpoint:
            try:
                parsed = urlsplit(str(raw_endpoint))
                host = parsed.hostname or ""
                if parsed.port:
                    host = f"{host}:{parsed.port}"
                endpoint = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
            except Exception:
                endpoint = "configured"
        elif provider_info:
            endpoint = provider_info.base_url

        auth_source = "unknown"
        resolver = getattr(self.config, "get_api_key_with_source", None)
        if callable(resolver):
            try:
                resolved = resolver(provider)
                if isinstance(resolved, tuple) and len(resolved) == 2:
                    auth_source = str(resolved[1])
            except Exception:
                pass

        model_default = self._model_default_max_tokens
        model_upper = self._model_upper_max_tokens
        configured_tokens = getattr(self.config, "max_tokens", 8192)
        from cca.llm.model_caps import get_effective_max_output_tokens
        effective_tokens = get_effective_max_output_tokens(
            self.config.model,
            configured=configured_tokens if isinstance(configured_tokens, int) else 8192,
        )

        backend_controls = self._describe_backend_request(
            self._think_level, self._reasoning_summary
        )

        configured_thinking = getattr(self.config, "default_thinking_level", "auto")
        if not isinstance(configured_thinking, str):
            configured_thinking = "auto"
        internal_thinking = resolve_thinking(
            self.config.model, "off", configured_thinking
        )

        active_tools: list[str] = []
        if self.registry is not None:
            try:
                active_tools = sorted(schema.name for schema in self.registry.get_schemas(
                    include=self._allowed_tools,
                    exclude=self._disallowed_tools or None,
                ))
            except Exception:
                pass

        permissions = {
            "mode": (
                "auto-approve" if getattr(self.permissions, "auto_approve", False)
                else "prompt"
            ),
            "always_allow": sorted(getattr(self.permissions, "always_allow", set()) or set()),
            "never_allow": sorted(getattr(self.permissions, "never_allow", set()) or set()),
        }
        agent_tool = self._active_agent_tool()
        subagent_runtime = None
        if agent_tool is not None:
            subagent_runtime = {
                "catalog_hash": agent_tool.catalog.catalog_hash,
                "default": agent_tool.default_agent,
                "max_depth": agent_tool.max_depth,
                "max_concurrency": agent_tool.max_concurrency,
                "allow_background": agent_tool.allow_background,
            }
        repo_config = getattr(self.config, "_repo_config_path", None)
        return {
            "run_mode": self._run_mode,
            "model": {
                "selector": self.config.model,
                "provider": provider,
                "id": model_id,
                "catalog_match": model_info is not None,
                "catalog_api": model_info.api if model_info else None,
            },
            "backend": {
                "endpoint": endpoint,
                "authentication": auth_source,
                **backend_controls,
            },
            "thinking": self._thinking_resolution.to_payload(),
            "internal_llm_calls": {
                "compaction": internal_thinking.to_payload(),
                "memory_extraction": internal_thinking.to_payload(),
                "subagents": "resolved from the subagent model and default_thinking_level",
            },
            "reasoning_summary": self._reasoning_summary,
            "max_tokens": {
                "configured": configured_tokens,
                "effective_default": effective_tokens,
                "model_default": model_default,
                "model_upper_limit": model_upper,
            },
            "output": {
                "format": self._output_format,
                "max_turns": self._max_turns,
            },
            "permissions": permissions,
            "tools": {
                "active": active_tools,
                "allowed_filter": sorted(self._allowed_tools) if self._allowed_tools else None,
                "disallowed_filter": sorted(self._disallowed_tools),
            },
            "features": self._features.to_payload(),
            "observability": self._observability_settings.safe_payload(),
            "randomization": self._randomization_payload,
            "agent_profile": (
                self._agent_profile.to_payload() if self._agent_profile is not None else None
            ),
            **({"subagents": subagent_runtime} if subagent_runtime is not None else {}),
            "prompts": {
                "system_override": self._system_prompt_override is not None,
                "system_append": bool(self._system_prompt_append or self.config.system_prompt_append),
                "instruction_files": [str(path) for path in self._loaded_instruction_files],
            },
            "config_files": {
                "global": str(CONFIG_FILE),
                "repo": str(repo_config) if repo_config else None,
            },
        }

    def _settings_summary(self) -> str:
        settings = self._runtime_settings_payload()
        thinking = settings["thinking"]
        requested = thinking["requested"]
        effective = thinking["effective"]
        thinking_text = effective if requested == effective else f"{requested}->{effective}"
        backend = settings["backend"]
        return (
            f"settings: mode={self._run_mode} model={self.config.model} "
            f"adapter={backend.get('adapter', type(self.backend).__name__)} "
            f"thinking={thinking_text} source={thinking['source']} "
            f"max_tokens={settings['max_tokens']['effective_default']} "
            f"permissions={settings['permissions']['mode']}"
            + (
                f" observe={self._observability_settings.destination}"
                if self._observability_settings.enabled else ""
            )
        )

    def _print_full_settings(self) -> None:
        import json
        self._console.print("[bold]Resolved runtime settings[/bold]")
        self._console.print(json.dumps(self._runtime_settings_payload(), indent=2))

    def _banner_thinking_status(self) -> "tuple[str, list[str]]":
        """Return (Thinking row markup, error lines to print after the banner).

        Resolution happened before backend construction, so this only renders it.
        """
        resolution = self._thinking_resolution
        requested = resolution.requested
        effective = resolution.effective
        color = "dim" if effective == "off" else "green"
        rendered = f"[{color}]{effective}[/{color}]"
        if requested != effective or requested == "auto":
            rendered = f"[cyan]{requested}[/cyan] → {rendered} [dim]({resolution.source})[/dim]"
        errors = [f"[yellow]{resolution.warning}[/yellow]"] if resolution.warning else []
        return rendered, errors

    def _banner_skills_status(self) -> str:
        if not self._features.tool_enabled("Skill"):
            return "[dim]off (disabled by feature config)[/dim]"
        if self.config.skills_enabled:
            from cca.tools.skill_tool import list_skills
            return f"[dim]on ({len(list_skills())} available)[/dim]"
        return "[dim]off (use /skills on to enable)[/dim]"

    def _print_banner(self) -> None:
        """Print the startup banner: ASCII logo, session facts, and key hints."""
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        header = Table.grid(padding=(0, 4))
        header.add_column()
        header.add_column(vertical="middle")
        header.add_row(
            Text(self._BANNER_ART, style="bold green"),
            Text(f"CCA v{__version__}\nYour Code CLI Agent.", style="dim italic"),
        )

        cwd_display = str(self.cwd)
        home = str(Path.home())
        if cwd_display == home or cwd_display.startswith(home + os.sep):
            cwd_display = "~" + cwd_display[len(home):]

        thinking_markup, think_errors = self._banner_thinking_status()
        runtime = self._runtime_settings_payload()

        info = Table.grid(padding=(0, 3))
        info.add_column(style="dim")
        info.add_column(overflow="fold")
        info.add_row("Model", f"[cyan]{self.config.model}[/cyan]")
        info.add_row(
            "Backend",
            f"[dim]{runtime['backend'].get('adapter', type(self.backend).__name__)} · "
            f"{runtime['model']['provider']}[/dim]",
        )
        if runtime["backend"].get("endpoint"):
            info.add_row("Endpoint", f"[dim]{escape(str(runtime['backend']['endpoint']))}[/dim]")
        info.add_row("CWD", escape(cwd_display))
        info.add_row("Thinking", thinking_markup)
        info.add_row(
            "Max output",
            f"[dim]{runtime['max_tokens']['effective_default']:,} default · "
            f"{runtime['max_tokens']['model_upper_limit']:,} upper[/dim]",
        )
        info.add_row("Permissions", f"[dim]{runtime['permissions']['mode']}[/dim]")
        if self._loaded_instruction_files:
            names = ", ".join(Path(p).name for p in self._loaded_instruction_files)
            info.add_row("Instructions", f"[dim]{escape(names)}[/dim]")
        summary = self._features.summary_line()
        if summary:
            # summary_line() is prefixed for bare printing; the row label covers it here
            info.add_row("Features", f"[yellow]{summary.removeprefix('features: ')}[/yellow]")
        randomize_summary = self._randomization.summary_line()
        if randomize_summary:
            info.add_row("Randomize", f"[yellow]{escape(randomize_summary)}[/yellow]")
        info.add_row("Skills", self._banner_skills_status())

        footer = Text.from_markup(
            "Type [bold]/help[/bold] for commands · Ctrl+C interrupt · Ctrl+D quit",
            style="dim",
        )
        self._console.print(Panel(
            Group(header, Text(), info, Text(), footer),
            border_style="dim",
            padding=(1, 3),
            expand=False,
        ))
        for line in think_errors:
            self._console.print(line)
        if self._show_settings:
            self._print_full_settings()
        self._console.print()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_interactive(self) -> None:
        """Start the interactive REPL."""
        from cca.config import CONFIG_DIR
        from cca import notifications

        loop = asyncio.get_running_loop()
        self._notif_queue = notifications.init(loop)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Interactive sessions can prompt for first-use trust of .mcp.json servers;
        # print/machine sessions leave trust_fn unset and skip untrusted servers.
        mcp_manager = getattr(self.registry, "_mcp_manager", None)
        if mcp_manager is not None:
            mcp_manager.trust_fn = self._mcp_trust_prompt

        await self._session_startup()

        self._print_banner()
        if self._ui:
            self._ui.update_think_level(self._think_level)

        if self._ui is not None:
            # ----------------------------------------------------------------
            # Interactive mode with full UI (InputBox Application)
            # ----------------------------------------------------------------
            try:
                while True:
                    await self._drain_notifications()

                    # Clear any stale cancel signal before waiting for input
                    self._ui.clear_cancel()

                    input_task = asyncio.create_task(self._ui.get_input())
                    notif_task = asyncio.create_task(self._notif_queue.get())

                    try:
                        done, pending = await asyncio.wait(
                            {input_task, notif_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        input_task.cancel()
                        notif_task.cancel()
                        self._console.print("\n[dim]Goodbye.[/dim]")
                        break

                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

                    if notif_task in done:
                        try:
                            notif = notif_task.result()
                            self._console.print()
                            await self._process_notification(notif)
                        except Exception:
                            pass
                        continue

                    try:
                        user_input = input_task.result()
                    except Exception:
                        continue

                    if user_input is None:   # Ctrl-C (first press) or Ctrl-D
                        # If a second Ctrl-C fires quickly the process exits via
                        # os.kill in the key binding; a single None here just
                        # means the user pressed Ctrl-C once while idle → quit.
                        self._console.print("\n[dim]Goodbye.[/dim]")
                        break

                    user_input = user_input.strip()
                    if not user_input:
                        continue

                    if user_input.startswith("!"):
                        await self._run_shell(user_input[1:].strip())
                        continue

                    display_input = user_input  # what to show in the echo panel
                    if user_input.startswith("/"):
                        try:
                            handled = await self._handle_slash(user_input)
                        except Exception as exc:
                            msg = str(exc) if str(exc) not in ("", "None") else type(exc).__name__
                            self._console.print(f"[red]Command failed:[/red] {escape(msg)}")
                            continue
                        if handled is True:
                            continue
                        elif isinstance(handled, str):
                            # Slash command wants a turn with a modified message;
                            # echo the original command, not the injected internals.
                            user_input = handled

                    # Echo user input above the scroll area before the turn starts
                    self._console.print(Panel(
                        escape(display_input),
                        border_style=self._panel_border("blue"),
                        padding=(0, 1),
                    ))

                    # Run the turn as a task so Escape/Ctrl+C can cancel it
                    self._ui.clear_cancel()
                    turn_task   = asyncio.create_task(self.process_turn(user_input))
                    cancel_task = asyncio.create_task(self._ui.wait_for_cancel())

                    try:
                        done, pending = await asyncio.wait(
                            {turn_task, cancel_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        turn_task.cancel()
                        cancel_task.cancel()
                        self._console.print("\n[dim]Goodbye.[/dim]")
                        break

                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

                    if cancel_task in done and turn_task not in done:
                        # User pressed Escape or Ctrl+C mid-turn
                        if self._ui:
                            self._ui.set_idle(is_cancel=True)
                        self._console.print("[dim]Cancelled.[/dim]")
                        # Drain the None that Ctrl+C put in the input queue
                        while not self._ui._input_box._queue.empty():
                            try:
                                self._ui._input_box._queue.get_nowait()
                            except Exception:
                                break
                    elif turn_task in done:
                        # Turn completed (normally or with an error) — always reset UI
                        if self._ui:
                            self._ui.set_idle()
                        exc = turn_task.exception()
                        if exc is not None:
                            msg = str(exc) if str(exc) not in ("", "None") else type(exc).__name__
                            self._console.print(f"[red]Error: {msg}[/red]")
            finally:
                # Close MCP sessions here, in the task that opened them —
                # otherwise the loop's asyncgen finalizer closes stdio_client
                # from another task and anyio raises a cancel-scope error.
                await self._session_shutdown()

        else:
            # ----------------------------------------------------------------
            # Fallback: no UI (tests / --print mode) — plain PromptSession
            # ----------------------------------------------------------------
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

            from prompt_toolkit.enums import EditingMode as _EditingMode
            session: PromptSession = PromptSession(
                history=FileHistory(str(CONFIG_DIR / "history")),
                auto_suggest=AutoSuggestFromHistory(),
                vi_mode=self.config.vim,
                editing_mode=_EditingMode.VI if self.config.vim else _EditingMode.EMACS,
            )

            try:
                while True:
                    await self._drain_notifications()

                    prompt_task = asyncio.create_task(session.prompt_async("cca> "))
                    notif_task  = asyncio.create_task(self._notif_queue.get())

                    try:
                        done, pending = await asyncio.wait(
                            {prompt_task, notif_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        prompt_task.cancel()
                        notif_task.cancel()
                        self._console.print("\n[dim]Goodbye.[/dim]")
                        break

                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

                    if notif_task in done:
                        try:
                            notif = notif_task.result()
                            self._console.print()
                            await self._process_notification(notif)
                        except Exception:
                            pass
                        continue

                    try:
                        user_input = prompt_task.result()
                    except KeyboardInterrupt:
                        continue
                    except EOFError:
                        self._console.print("\n[dim]Goodbye.[/dim]")
                        break
                    except Exception:
                        continue

                    user_input = user_input.strip()
                    if not user_input:
                        continue

                    if user_input.startswith("!"):
                        await self._run_shell(user_input[1:].strip())
                        continue

                    if user_input.startswith("/"):
                        try:
                            handled = await self._handle_slash(user_input)
                        except Exception as exc:
                            msg = str(exc) if str(exc) not in ("", "None") else type(exc).__name__
                            self._console.print(f"[red]Command failed:[/red] {escape(msg)}")
                            continue
                        if handled is True:
                            continue
                        elif isinstance(handled, str):
                            user_input = handled

                    await self.process_turn(user_input)
            finally:
                await self._session_shutdown()

    def _announce_thinking_warning(self) -> None:
        """Say on stderr when the requested thinking level cannot be honoured."""
        warning = self._thinking_resolution.warning
        if warning:
            self._console.print(f"[yellow]{warning}[/yellow]")

    async def run_print(self, prompt: str) -> str:
        """Non-interactive: run one turn and return formatted output."""
        from cca import notifications
        import json as _json
        self._notif_queue = notifications.init(asyncio.get_running_loop())
        await self._session_startup()
        self._console.print(f"[dim]{self._settings_summary()}[/dim]")
        self._announce_thinking_warning()
        if self._show_settings:
            self._print_full_settings()
        structured = None
        try:
            text = await self.process_turn(prompt, return_text=True)
            if self._turn_throttled:
                # Retries and the wait budget were spent without an answer.
                # Returning the partial text would hand a caller a truncated
                # reply indistinguishable from a complete one; the caller is
                # told instead, and can retry the whole call when the provider
                # has recovered.
                raise LLMUnavailable(
                    f"provider did not answer after {self._throttle_events} wait(s), "
                    f"{self._throttle_wait_s:.1f}s waited",
                    kind="throttle",
                    attempts=self._throttle_events,
                    waited_s=self._throttle_wait_s,
                )
            if self._output_schema is not None:
                # Reformat *after* the agentic loop in its own tool-free call.
                # Utility calls prefer thinking off, but capability resolution
                # raises that to the model's legal minimum when off is not
                # supported (for example, GLM-5.3).
                from cca.structured_output import format_as_schema

                structured = await format_as_schema(
                    self.backend,
                    text,
                    self._output_schema,
                    think_level=self._internal_think_level,
                )
                text = structured["text"]
        finally:
            await self._session_shutdown()
        if self._output_format == "json":
            payload = {
                "type": "result",
                "result": text,
                "session_id": self._session_id,
                "settings": self._runtime_settings_payload(),
            }
            if structured is not None:
                # Report how the constraint was actually applied. A route that
                # silently degraded to prompt_only produces worse output for
                # reasons that have nothing to do with the model's ability, and
                # a caller comparing models needs to be able to see that.
                payload["structured_output"] = {
                    "mode_used": structured["mode_used"],
                    "attempts": structured["attempts"],
                    "think_level": structured["think_level"],
                    "valid": True,
                }
            return _json.dumps(payload)
        # text and stream-json: return the text (stream-json already printed deltas)
        return text

    # ---------------------------------------------------------------------------
    # --machine mode: persistent JSON-lines session
    # ---------------------------------------------------------------------------

    async def run_machine(self) -> None:
        """Persistent machine mode: reads JSON-line commands from stdin,
        writes JSON-line events to stdout. stderr stays human-readable.

        Supported inbound message types:
          {"type": "user_turn", "text": "..."}
          {"type": "slash_command", "command": "/plan fix auth"}
          {"type": "answer", "id": "req-abc", "answers": {"Q?": "A"}}
          {"type": "plan_decision", "id": "req-xyz", "approved": true, "feedback": "ok"}
          {"type": "shutdown"}

        Emitted outbound message types (one JSON line per event):
          {"type": "ask_user", "id": "...", "questions": [...]}
          {"type": "exit_plan_mode", "id": "...", "plan": "..."}
          {"type": "turn_complete", "text": "...", "session_id": "...", "msg_index_end": N}
          {"type": "error", "message": "..."}
        """
        import json as _json
        from cca import notifications

        loop = asyncio.get_running_loop()
        self._notif_queue = notifications.init(loop)
        self._machine_pending: dict[str, asyncio.Event] = {}
        self._machine_answers: dict[str, object] = {}
        self._machine_shutdown = asyncio.Event()
        self._machine_inbox: asyncio.Queue = asyncio.Queue()

        await self._session_startup()

        self._machine_writeline({
            "type": "ready",
            "session_id": self._session_id,
            "settings": self._runtime_settings_payload(),
            "features": self._features.to_payload(),
            **({"randomization": self._randomization_payload}
               if self._randomization_payload else {}),
        })
        self._announce_thinking_warning()
        if self._show_settings:
            self._print_full_settings()

        # Continuously read stdin in the background so answer/plan_decision
        # messages arrive even while process_turn is running (they unblock
        # _machine_ask_fn / _machine_plan_fn via asyncio.Event).
        async def _stdin_reader():
            while not self._machine_shutdown.is_set():
                raw = await self._machine_readline()
                if not raw:
                    self._machine_shutdown.set()
                    return
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = _json.loads(raw)
                except _json.JSONDecodeError as e:
                    self._machine_writeline({"type": "error", "message": f"invalid JSON: {e}"})
                    continue

                mtype = msg.get("type")

                if mtype == "shutdown":
                    self._machine_shutdown.set()
                    return

                if mtype in ("answer", "plan_decision"):
                    req_id = msg.get("id", "")
                    self._machine_answers[req_id] = msg
                    ev = self._machine_pending.get(req_id)
                    if ev is not None:
                        ev.set()
                else:
                    await self._machine_inbox.put(msg)

        reader_task = asyncio.create_task(_stdin_reader())

        try:
            while not self._machine_shutdown.is_set():
                try:
                    msg = await asyncio.wait_for(
                        self._machine_inbox.get(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                mtype = msg.get("type")

                if mtype == "user_turn":
                    text = msg.get("text", "")
                    if not text:
                        self._machine_writeline({"type": "error", "message": "user_turn requires 'text'"})
                        continue
                    try:
                        result = await self.process_turn(text, return_text=True)
                    except Exception as exc:
                        self._machine_writeline({"type": "error", "message": str(exc)})
                        continue
                    self._machine_writeline({
                        "type": "turn_complete",
                        "text": result,
                        "session_id": self._session_id,
                        "msg_index_end": len(self.messages),
                        **self._throttle_payload(),
                    })

                elif mtype == "slash_command":
                    command = msg.get("command", "")
                    if not command:
                        self._machine_writeline({"type": "error", "message": "slash_command requires 'command'"})
                        continue
                    try:
                        handled = await self._handle_slash(command)
                        if isinstance(handled, str):
                            result = await self.process_turn(handled, return_text=True)
                            self._machine_writeline({
                                "type": "turn_complete",
                                "text": result,
                                "session_id": self._session_id,
                                "msg_index_end": len(self.messages),
                                **self._throttle_payload(),
                            })
                    except Exception as exc:
                        self._machine_writeline({"type": "error", "message": str(exc)})

                else:
                    self._machine_writeline({"type": "error", "message": f"unknown message type: {mtype!r}"})
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            await self._session_shutdown()

    async def _machine_readline(self) -> str:
        """Read one line from stdin asynchronously (overridable in tests)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, sys.stdin.readline)

    def _machine_writeline(self, obj: dict) -> None:
        """Write one JSON line to stdout (overridable in tests)."""
        import json as _json
        print(_json.dumps(obj), flush=True, file=sys.stdout)

    def _throttle_payload(self) -> dict:
        """Per-turn throttle accounting for a turn_complete event."""
        return {
            "throttled": self._turn_throttled,
            "throttle_wait_s": round(self._throttle_wait_s, 2),
            "throttle_events": self._throttle_events,
        }

    def _on_llm_wait(self, payload: dict) -> None:
        """Announce a throttle wait that is about to happen.

        Called by `cca.llm.retry` *before* it sleeps. In machine mode the
        harness owns a per-turn deadline it cannot otherwise distinguish from
        the agent being slow, so it is told first and can extend that deadline
        by `wait_s` instead of killing a process that is behaving correctly.
        """
        wait_s = float(payload.get("wait_s") or 0.0)
        self._throttle_wait_s += wait_s
        self._throttle_events += 1
        if self._run_mode == "machine":
            # The event carries everything the console line would say, and the
            # harness prints it on receipt — saying it twice on the same stderr
            # just makes a throttled sweep twice as hard to read.
            self._machine_writeline({"type": "throttled", **payload})
            return
        self._console.print(
            f"[dim yellow]LLM {payload.get('kind', 'busy')} — waiting "
            f"{wait_s:.1f}s before attempt {int(payload.get('attempt', 0)) + 1}"
            f" ({self._throttle_wait_s:.0f}s waited this turn)[/dim yellow]"
        )

    async def _machine_ask_fn(
        self,
        question: str,
        options: list,
        multi: bool,
        header: str,
    ) -> str:
        """ask_fn for machine mode — emits ask_user event, blocks until answer arrives."""
        import uuid as _uuid
        req_id = f"req-{_uuid.uuid4().hex[:8]}"
        ev = asyncio.Event()
        self._machine_pending[req_id] = ev
        self._machine_writeline({
            "type": "ask_user",
            "id": req_id,
            "questions": [{"question": question, "options": options, "multiSelect": multi, "header": header}],
        })
        await ev.wait()
        answer_msg = self._machine_answers.pop(req_id, {})
        answers = answer_msg.get("answers", {})
        if answers:
            return answers.get(question, next(iter(answers.values()), ""))
        return ""

    async def _machine_plan_fn(
        self,
        question: str,
        options: list,
        multi: bool,
        header: str,
    ) -> str:
        """ask_fn variant for ExitPlanMode in machine mode — emits exit_plan_mode event."""
        import uuid as _uuid
        from cca.tools.plan_tools import get_plan_file
        plan_file = get_plan_file()
        plan_text = plan_file.read_text() if plan_file and plan_file.exists() else ""
        req_id = f"req-{_uuid.uuid4().hex[:8]}"
        ev = asyncio.Event()
        self._machine_pending[req_id] = ev
        self._machine_writeline({
            "type": "exit_plan_mode",
            "id": req_id,
            "plan": plan_text,
        })
        await ev.wait()
        decision = self._machine_answers.pop(req_id, {})
        approved = decision.get("approved", True)
        return "Yes, proceed" if approved else "No, revise"

    # ------------------------------------------------------------------
    # Core turn processing
    # ------------------------------------------------------------------

    async def process_turn(
        self, user_input: str, return_text: bool = False, harness_turn: bool = False
    ) -> str:
        """Observe one user turn without changing the established turn loop."""
        self._turn_observation_index += 1
        with self._observer.span(
            "cca.turn",
            kind="turn",
            parent=self._session_span,
            attributes={
                "cca.turn.index": self._turn_observation_index,
                "cca.session_id": self._session_id,
                "cca.context_generation": self._context_generation,
                "cca.harness_turn": harness_turn,
            },
            content={"input.value": user_input},
        ) as span:
            result = await self._process_turn_impl(
                user_input,
                return_text=return_text,
                harness_turn=harness_turn,
            )
            span.finish(
                status="ok",
                attributes={"cca.message_count": len(self.messages)},
                content={"output.value": result},
            )
            return result

    async def _process_turn_impl(
        self, user_input: str, return_text: bool = False, harness_turn: bool = False
    ) -> str:
        # One turn is one wait budget: the harness deadline this must fit
        # inside is also per-turn.
        from cca.llm import retry as _retry
        _retry.budget().reset()
        self._throttle_wait_s = 0.0
        self._throttle_events = 0
        self._turn_throttled = False

        self.messages.append(Message(role="user", content=[TextContent(text=user_input)]))

        # Skip memory extraction for harness-initiated turns (no real user query)
        if not harness_turn and self._features.feature("memory"):
            await self._maybe_update_memory_facts(user_input)

        # Detect explicit token budget request (e.g. "use 500k tokens", "+2m")
        explicit_budget = _parse_token_budget(user_input)
        if explicit_budget is not None:
            self._token_budget = explicit_budget
            self._console.print(f"[dim]Token budget set to {explicit_budget:,}.[/dim]")

        # Per-message keyword may force "high" for this turn only
        keyword_level = _keyword_think_level(user_input)
        if keyword_level:
            support, desc = _think_support(self.config.model)
            if support == "none":
                self._console.print(
                    f"[yellow]Warning: thinking keyword detected but "
                    f"{self.config.model} does not support thinking — ignored.[/yellow]"
                )
                self._console.print(f"[yellow]{desc}[/yellow]")
                keyword_level = None
        effective_think_level = keyword_level or self._think_level
        _max_tok_override: int | None = self._token_budget
        _escalated = False

        final_text = ""
        _default_turns = MAX_TURNS_PRINT if self.print_mode else MAX_TURNS
        _turn_limit = self._max_turns if self._max_turns is not None else _default_turns
        _turn = 0
        while _turn < _turn_limit:
            _turn += 1
            system = self._build_system()
            content, text, stop_reason = await self._stream_response(
                system,
                think_level=effective_think_level,
                max_tokens_override=_max_tok_override,
            )
            effective_think_level = self._think_level  # keyword only applies to first turn
            _max_tok_override = self._token_budget  # respect persistent budget

            if text:
                final_text = text

            if self._machine_verbose and text:
                self._machine_writeline({"type": "assistant_text", "text": text[:500]})

            # If cut off by token limit, retry once with the model's upper limit
            if stop_reason == "max_tokens" and not _escalated:
                upper = self._model_upper_max_tokens
                current = _max_tok_override or self.config.max_tokens or self._model_default_max_tokens
                if upper > current:
                    _escalated = True
                    _max_tok_override = upper
                    self._console.print(
                        f"[dim]Response hit max_tokens ({current:,}); "
                        f"continuing with {upper:,} tokens…[/dim]"
                    )
                    continue

            # No tool calls — final response, print in a green panel (interactive only)
            if not any(isinstance(b, ToolUseContent) for b in content):
                if text and self._run_mode == "interactive":
                    from rich.markdown import Markdown as RichMarkdown
                    self._console.print(Panel(RichMarkdown(text), border_style=self._panel_border("green"), padding=(1, 1)))
                break

            # Execute tool calls
            tool_calls = [b for b in content if isinstance(b, ToolUseContent)]
            if self._machine_verbose:
                for tc in tool_calls:
                    inp = tc.input or {}
                    preview = {}
                    for k in ("command", "file_path", "pattern", "query", "description"):
                        if k in inp:
                            preview[k] = str(inp[k])[:200]
                    self._machine_writeline({
                        "type": "tool_use",
                        "name": tc.name,
                        "input_preview": preview,
                    })
            tool_results = await self._execute_tools(tool_calls)
            self._mem_tool_calls_since_extraction += len(tool_calls)
            self.messages.append(Message(role="user", content=tool_results))

            # Immediately compact if we hit the upper threshold within a task
            await self._compact_if_needed(check_type="upper")

            # Turn limit reached mid-task (still had tool calls)
            if _turn >= _turn_limit:
                if self.print_mode:
                    self._console.print(
                        f"[yellow]Reached turn limit ({_turn_limit}). "
                        f"Use --max-turns to increase.[/yellow]"
                    )
                else:
                    self._console.print(
                        f"\n[yellow]Reached turn limit ({_turn_limit}). "
                        f"Reply to continue, or /stop to cancel.[/yellow]\n"
                    )

        # Sanitize unclosed markdown code fences.  An odd number of triple-backtick
        # delimiters means the response ended inside a code block, which poisons
        # every subsequent assistant turn in a persistent session — the model
        # "thinks" it is still inside the open block and resumes mid-code.
        # Close the fence before persisting so downstream turns start clean.
        if final_text and final_text.count("```") % 2 == 1:
            self._console.print(
                "[dim yellow]⚠ Response ended with an unclosed code fence — closing automatically.[/dim yellow]"
            )
            final_text += "\n```"
            # Patch the in-memory assistant message so the saved session file
            # matches what we return to callers.
            if self.messages and self.messages[-1].role == "assistant":
                msg_content = self.messages[-1].content
                if isinstance(msg_content, list):
                    for block in reversed(msg_content):
                        if isinstance(block, TextContent):
                            block.text += "\n```"
                            break
                elif isinstance(msg_content, str):
                    self.messages[-1] = Message(
                        role="assistant", content=msg_content + "\n```"
                    )

        # Compact if we hit the lower threshold at the end of a task
        await self._compact_if_needed(check_type="lower")

        # Persist session after every turn so --continue / --resume work
        try:
            from cca.sessions import save_session
            save_session(self._session_id, self.cwd, self.messages,
                         system_prompt=self._build_system(),
                         token_counts=dict(self._token_counts),
                         compacted_from_index=self._msg_index_offset if self._msg_index_offset > 0 else None,
                         features=self._features.to_payload(),
                         randomization=self._randomization_payload,
                         settings=self._runtime_settings_payload(),
                         state=self._session_state_payload())
        except Exception:
            pass

        return final_text

    # ------------------------------------------------------------------
    # LLM streaming
    # ------------------------------------------------------------------

    @property
    def _model_default_max_tokens(self) -> int:
        from cca.llm.model_caps import get_model_max_output_tokens
        return get_model_max_output_tokens(self.config.model)[0]

    @property
    def _model_upper_max_tokens(self) -> int:
        from cca.llm.model_caps import get_model_max_output_tokens
        return get_model_max_output_tokens(self.config.model)[1]

    @property
    def _internal_think_level(self) -> str:
        """Effective level for utility calls that intentionally request off."""
        from cca.llm.thinking import resolve_thinking
        configured = getattr(self.config, "default_thinking_level", "auto")
        if not isinstance(configured, str):
            configured = "auto"
        return resolve_thinking(self.config.model, "off", configured).effective

    async def _complete_internal_observed(
        self,
        *,
        call_type: str,
        messages: list[Message],
        tools: list,
        system: str,
        max_tokens: int,
        think_level: str,
    ):
        """Observe utility model calls that do not use the streaming root path."""
        with self._observer.span(
            "cca.llm.call",
            kind="llm",
            attributes={
                "cca.llm.call_type": call_type,
                "cca.session_id": self._session_id,
                "cca.context_generation": self._context_generation,
                **_model_attributes(self.config.model),
                "gen_ai.request.max_tokens": max_tokens,
                "cca.thinking_level": think_level,
            },
            content={
                "input.value": {
                    "messages": [message.to_dict() for message in messages]
                }
            },
        ) as span:
            response = await self.backend.complete(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
                think_level=think_level,
            )
            usage = response.usage or {}
            span.finish(
                status="ok",
                attributes={
                    "gen_ai.response.finish_reasons": [response.stop_reason],
                    **_response_model_attributes(self.config.model),
                },
                metrics={
                    "gen_ai.usage.input_tokens": usage.get(
                        "input_tokens", usage.get("input", 0)
                    ),
                    "gen_ai.usage.output_tokens": usage.get(
                        "output_tokens", usage.get("output", 0)
                    ),
                    "gen_ai.usage.cache_read.input_tokens": usage.get(
                        "cache_read_tokens", 0
                    ),
                    "gen_ai.usage.cache_creation.input_tokens": usage.get(
                        "cache_write_tokens", 0
                    ),
                },
                content={"output.value": response.text},
            )
            return response

    def _log_llm_call(
        self,
        *,
        call_type: str,
        system: str,
        messages_snapshot: list,
        tools: list,
        params: dict,
        response_text: str,
        tool_calls: list,
        thinking: str,
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        reasoning: dict | None = None,
    ) -> None:
        """Append one LLM API call record to ~/.cca/llm-calls/session_<base_id>.jsonl."""
        import time as _time
        try:
            multi_agent_capable = self._agent_tool_visible()
            try:
                runtime_settings = self._runtime_settings_payload()
            except Exception:
                runtime_settings = None
            response = {
                "text": response_text,
                "tool_calls": tool_calls,
                # Legacy compatibility for existing extractors and run files.
                "thinking": thinking,
                "stop_reason": stop_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
            }
            if reasoning is not None:
                response["reasoning"] = reasoning
            record = {
                "ts": int(_time.time()),
                "type": call_type,
                "session_id": self._session_id,
                "run_id": getattr(self, "_run_id", self._session_id),
                "continuation_reason": getattr(
                    self, "_continuation_reason", "initial"
                ),
                "multi_agent_capable": multi_agent_capable,
                "features": self._features.to_payload(),
                **({"settings": runtime_settings} if runtime_settings else {}),
                **({"randomization": self._randomization_payload}
                   if self._randomization_payload else {}),
                "system": system,
                "messages": messages_snapshot,
                "tools": tools,
                "params": params,
                "response": response,
            }
            if multi_agent_capable and getattr(self, "registry", None) is not None:
                from cca.tools.agent_tool import AgentTool
                agent_tool = next(
                    (
                        tool for tool in getattr(self.registry, "_tools", {}).values()
                        if isinstance(tool, AgentTool)
                    ),
                    None,
                )
                if agent_tool is not None:
                    record["agent_catalog_hash"] = agent_tool.catalog.catalog_hash
            from cca.trace_store import (
                TraceIdentity, backend_provenance, generation_from_session_id,
            )
            backend = getattr(self, "backend", None)
            config = getattr(self, "config", None)
            requested_selector = getattr(config, "model", None)
            record["model_provenance"] = backend_provenance(
                backend, requested_selector
            )
            context_generation = getattr(
                self,
                "_context_generation",
                generation_from_session_id(self._session_id),
            )
            generation_reason = getattr(
                self,
                "_generation_reason",
                "compaction" if context_generation > 0 else "initial",
            )
            identity = TraceIdentity(
                trace_id=getattr(self, "_trace_id", self._base_session_id),
                agent_id=getattr(self, "_agent_id", self._base_session_id),
                generation=context_generation,
                context_generation=context_generation,
                generation_reason=generation_reason,
            )
            trace_store = getattr(self, "_trace_store", None)
            if trace_store is None:
                from cca.trace_store import TraceStore
                trace_store = TraceStore(self._base_session_id)
                self._trace_store = trace_store
            trace_store.append_call(
                identity,
                self._llm_call_idx,
                record,
                child=False,
            )
            self._llm_call_idx += 1
        except Exception:
            pass  # logging is non-fatal

    async def _stream_response(
        self,
        system: str,
        think_level: str = "off",
        max_tokens_override: int | None = None,
    ) -> tuple[list[ContentBlock], str, str]:
        """Observe one provider stream while retaining the existing implementation."""
        before = dict(self._token_counts)
        messages = [message.to_dict() for message in self.messages]
        from cca.llm.model_caps import get_effective_max_output_tokens
        max_tokens = get_effective_max_output_tokens(
            self.config.model,
            configured=self.config.max_tokens,
            override=max_tokens_override,
        )
        with self._observer.span(
            "cca.llm.call",
            kind="llm",
            attributes={
                "cca.llm.call_type": "main",
                "cca.session_id": self._session_id,
                "cca.context_generation": self._context_generation,
                **_model_attributes(self.config.model),
                "gen_ai.request.max_tokens": max_tokens,
                "cca.thinking_level": think_level,
            },
            content={"input.value": {"messages": messages}},
        ) as span:
            content, text, stop_reason = await self._stream_response_impl(
                system,
                think_level=think_level,
                max_tokens_override=max_tokens,
            )
            metrics = {
                "gen_ai.usage.input_tokens": max(
                    0, self._token_counts["input"] - before["input"]
                ),
                "gen_ai.usage.output_tokens": max(
                    0, self._token_counts["output"] - before["output"]
                ),
                "gen_ai.usage.cache_read.input_tokens": max(
                    0, self._token_counts["cache_read"] - before["cache_read"]
                ),
                "gen_ai.usage.cache_creation.input_tokens": max(
                    0, self._token_counts["cache_write"] - before["cache_write"]
                ),
            }
            span.finish(
                status="error" if stop_reason in ("error", "throttled") else "ok",
                attributes={
                    "gen_ai.response.finish_reasons": [stop_reason],
                    **_response_model_attributes(self.config.model),
                },
                metrics=metrics,
                content={"output.value": text},
            )
            return content, text, stop_reason

    async def _stream_response_impl(
        self,
        system: str,
        think_level: str = "off",
        max_tokens_override: int | None = None,
    ) -> tuple[list[ContentBlock], str, str]:
        """Stream one LLM response, printing text as it arrives.

        Returns (content_blocks, full_text, stop_reason).
        """
        if self.config.trace:
            self._trace_request(system, self.messages)

        from rich.markdown import Markdown as RichMarkdown

        if self._ui:
            self._ui.set_thinking()

        tool_schemas = self._randomization.apply_to_schemas(self.registry.get_schemas(
            include=self._allowed_tools,
            exclude=self._disallowed_tools or None,
        ))
        # Capture snapshots before the stream mutates state — needed for the call log.
        import json as _json_snap
        _messages_snapshot = [m.to_dict() for m in self.messages]
        from dataclasses import asdict as _asdict
        _tools_snapshot = [_asdict(ts) for ts in tool_schemas]

        content: list[ContentBlock] = []
        text_parts: list[str] = []
        # How much of `text_parts` has already been shown. A response with N
        # parallel tool calls raises N tool_use_start events, and text_parts is
        # deliberately never cleared (it has to survive into full_text), so
        # printing the whole buffer on each one repeated the same narration N
        # times on screen.
        rendered_parts = 0
        thinking_parts: list[str] = []
        reasoning_kind = "unknown"
        reasoning_provider_state: dict = {}
        in_thinking = False
        stop_reason = "end_turn"
        # Per-call token counts (raw values from message_end, not cumulative deltas).
        _call_input_tokens = 0
        _call_output_tokens = 0
        _call_cache_read = 0
        _call_cache_write = 0
        # Prefer explicit override → config value (if set above the default 8192)
        # → per-model default from the capability table.
        config_tok = self.config.max_tokens
        from cca.llm.model_caps import get_effective_max_output_tokens
        max_tok = get_effective_max_output_tokens(
            self.config.model,
            configured=config_tok,
            override=max_tokens_override,
        )
        request_reasoning_summary = self._reasoning_summary and think_level != "off"

        try:
            async for event in self.backend.stream(
                messages=self.messages,
                tools=tool_schemas,
                system=system,
                max_tokens=max_tok,
                think_level=think_level,
                reasoning_summary=request_reasoning_summary,
            ):
                if event.type == "reasoning_metadata":
                    metadata = event.data if isinstance(event.data, dict) else {}
                    if metadata.get("kind"):
                        reasoning_kind = metadata["kind"]
                    state = metadata.get("provider_state")
                    if isinstance(state, dict):
                        items = state.get("reasoning_items")
                        if isinstance(items, list):
                            reasoning_provider_state.setdefault(
                                "reasoning_items", []
                            ).extend(items)
                            state = {
                                key: value for key, value in state.items()
                                if key != "reasoning_items"
                            }
                        reasoning_provider_state.update(state)

                elif event.type == "thinking_delta":
                    if self.config.trace:
                        if not in_thinking:
                            self._console.print("[dim italic]thinking…[/dim italic]", end="")
                            in_thinking = True
                        self._console.print(f"[dim]{escape(event.data)}[/dim]", end="")
                    if self._ui:
                        self._ui.add_text(event.data)
                    thinking_parts.append(event.data)

                elif event.type == "text_delta":
                    if in_thinking:
                        self._console.print()
                        in_thinking = False
                    if self._ui:
                        self._ui.on_token()
                        if not text_parts:
                            # First text token: commit the thinking box so response
                            # text doesn't bleed into the thinking display.
                            self._ui.set_idle()
                    text_parts.append(event.data)
                    if self._output_format == "stream-json":
                        import json as _json
                        print(_json.dumps({
                            "type": "content_block_delta",
                            "delta": {"type": "text_delta", "text": event.data},
                        }), flush=True)

                elif event.type == "tool_use_start":
                    if in_thinking:
                        self._console.print()
                        in_thinking = False
                    # Render any buffered text before showing tool call panels.
                    # Do NOT clear text_parts — the accumulated text must survive
                    # into full_text so it is stored in content and self.messages.
                    # (Clearing here caused pre-tool narration to be shown on
                    # screen but silently dropped from the session file.)
                    if len(text_parts) > rendered_parts:
                        self._console.print(
                            RichMarkdown("".join(text_parts[rendered_parts:]))
                        )
                        rendered_parts = len(text_parts)

                elif event.type == "tool_use_complete":
                    tc: ToolUseContent = event.data
                    content.append(tc)

                elif event.type == "message_end":
                    if in_thinking:
                        self._console.print()
                        in_thinking = False
                    if event.data and isinstance(event.data, dict):
                        in_tok = event.data.get("input_tokens", 0)
                        out_tok = event.data.get("output_tokens", 0)
                        # Only count the delta: older messages are cached by the
                        # provider and don't re-contribute to cost each call.
                        in_delta = max(0, in_tok - self._last_input_tokens)
                        self._last_input_tokens = in_tok
                        self._token_counts["input"] += in_delta
                        self._token_counts["output"] += out_tok
                        self._token_counts["cache_read"] += event.data.get("cache_read_tokens", 0)
                        self._token_counts["cache_write"] += event.data.get("cache_write_tokens", 0)
                        stop_reason = event.data.get("stop_reason", "end_turn")
                        # Raw per-call counts for the call log.
                        _call_input_tokens = in_tok
                        _call_output_tokens = out_tok
                        _call_cache_read = event.data.get("cache_read_tokens", 0)
                        _call_cache_write = event.data.get("cache_write_tokens", 0)
                        if self._ui:
                            self._ui.update_tokens(
                                self._token_counts["input"],
                                self._token_counts["output"],
                                cache_read=self._token_counts["cache_read"],
                                cache_write=self._token_counts["cache_write"],
                            )
                    if self._ui:
                        self._ui.set_idle()

        except LLMUnavailable as e:
            # The provider, not the model: retries and the wait budget were
            # spent and it is still refusing. Reported apart from `error` so a
            # harness can treat it as "the API was busy" — worth waiting or
            # retrying — rather than as a failed agent turn.
            if len(text_parts) > rendered_parts:
                self._console.print(
                    RichMarkdown("".join(text_parts[rendered_parts:]))
                )
                text_parts = []
            if self._ui:
                self._ui.set_idle()
            self._turn_throttled = True
            label = "LLM throttled" if e.kind == "throttle" else "LLM unavailable"
            self._console.print(f"[red]{label}:[/red] {escape(str(e))}")
            return content, "", "throttled"

        except Exception as e:
            if len(text_parts) > rendered_parts:
                self._console.print(
                    RichMarkdown("".join(text_parts[rendered_parts:]))
                )
                text_parts = []
            if self._ui:
                self._ui.set_idle()
            self._console.print(f"[red]LLM error:[/red] {escape(str(e))}")
            return content, "", "error"

        full_text = "".join(text_parts)
        if full_text:
            content.insert(0, TextContent(text=full_text))

        reasoning_text = "".join(thinking_parts)
        if reasoning_text or reasoning_provider_state:
            signature = reasoning_provider_state.get("signature")
            content.insert(0, ThinkingContent(
                thinking=reasoning_text,
                signature=signature,
                reasoning_kind=reasoning_kind,
                provider_state=reasoning_provider_state or None,
            ))

        self.messages.append(Message(role="assistant", content=content))

        if self.config.trace:
            self._trace_tool_calls(content)

        self._log_llm_call(
            call_type="main",
            system=system,
            messages_snapshot=_messages_snapshot,
            tools=_tools_snapshot,
            params={
                "max_tokens": max_tok,
                "think_level": think_level,
                "reasoning_summary": request_reasoning_summary,
                "thinking": {
                    "session_requested": self._thinking_resolution.requested,
                    "effective": think_level,
                    "source": (
                        "message-keyword"
                        if think_level != self._think_level else self._thinking_resolution.source
                    ),
                },
                "backend_request": self._describe_backend_request(
                    think_level, request_reasoning_summary
                ),
            },
            response_text=full_text,
            tool_calls=[{"id": tc.id, "name": tc.name, "input": tc.input} for tc in content if isinstance(tc, ToolUseContent)],
            thinking="".join(thinking_parts),
            stop_reason=stop_reason,
            input_tokens=_call_input_tokens,
            output_tokens=_call_output_tokens,
            cache_read_tokens=_call_cache_read,
            cache_write_tokens=_call_cache_write,
            reasoning={
                "text": reasoning_text,
                "kind": reasoning_kind,
                "requested": request_reasoning_summary,
                "captured": bool(reasoning_text),
                **({"provider_state": reasoning_provider_state}
                   if reasoning_provider_state else {}),
            },
        )

        return content, full_text, stop_reason

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tools(
        self, tool_calls: list[ToolUseContent]
    ) -> list[ToolResultContent]:
        """Execute tool calls concurrently, one live box per tool."""
        import shutil as _shutil
        results: list[ToolResultContent] = []

        term_width = _shutil.get_terminal_size((80, 24)).columns
        label_max = max(20, term_width - 20)

        # Open a box for every tool before launching any of them
        if self._ui:
            for tc in tool_calls:
                label = self._tool_box_label(tc, label_max)
                self._ui.start_tool(tc.id, label)

        # Launch each tool as its own asyncio Task (copies context → each task
        # sets its own current_tool_id, routing bash output to the right box)
        task_map: dict[asyncio.Task, ToolUseContent] = {}
        for tc in tool_calls:
            task_map[asyncio.create_task(self._execute_one(tc))] = tc

        # Collect results; commit each box as soon as its tool finishes
        raw_results_map: dict[str, object] = {}
        pending = set(task_map.keys())
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    tc = task_map[task]
                    try:
                        raw: object = task.result()
                    except Exception as exc:
                        raw = exc
                    raw_results_map[tc.id] = raw
                    if self._ui:
                        if not isinstance(raw, Exception):
                            self._ui.feed_tool_result(tc.id, str(raw).splitlines())
                        if isinstance(raw, Exception) or "[py_compile ERROR]" in str(raw):
                            self._ui.set_tool_error(tc.id)
                        self._ui.finish_tool(tc.id)
        except asyncio.CancelledError:
            # Turn cancelled (Escape / Ctrl+C / dead UI) — don't orphan tools
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

        raw_results = [raw_results_map[tc.id] for tc in tool_calls]

        _PERSIST_CHARS = 50_000   # per-result threshold for disk persistence

        for tc, raw in zip(tool_calls, raw_results):
            if isinstance(raw, Exception):
                result_str = f"Error: {raw}"
                is_error = True
            else:
                result_str = str(raw)
                is_error = False

            # Track recently-read files for post-compaction re-injection.
            if tc.name == "Read" and not is_error:
                fp = tc.input.get("file_path", "")
                if fp:
                    if fp in self._recently_read:
                        self._recently_read.remove(fp)
                    self._recently_read.append(fp)
                    if len(self._recently_read) > 5:
                        self._recently_read = self._recently_read[-5:]

            # For large results, persist to disk instead of truncating.
            if self._features.feature("disk_spill") and len(result_str) > _PERSIST_CHARS:
                try:
                    self._tool_results_dir.mkdir(parents=True, exist_ok=True)
                    import uuid as _uuid2
                    fname = self._tool_results_dir / f"{_uuid2.uuid4().hex[:8]}.txt"
                    fname.write_text(result_str, encoding="utf-8", errors="replace")
                    _lines_full = result_str.splitlines()
                    preview = "\n".join(_lines_full[:20])
                    remaining = len(_lines_full) - 20
                    result_str = (
                        f"[Output persisted — {len(_lines_full):,} lines, "
                        f"{len(result_str):,} chars. Full output at: {fname}]\n"
                        f"[Use the Read tool to retrieve it if needed.]\n\n"
                        f"{preview}\n"
                        + (f"\n[… {remaining:,} more lines in persisted file]" if remaining > 0 else "")
                    )
                except Exception:
                    # Persistence failed — fall back to truncation
                    _lines = result_str.splitlines()
                    if len(_lines) > 300:
                        omitted = len(_lines) - 300
                        result_str = "\n".join(_lines[:300]) + f"\n[… {omitted} lines omitted …]"
            else:
                # Cap modest results at 300 lines to keep context tidy.
                _lines = result_str.splitlines()
                if len(_lines) > 300:
                    omitted = len(_lines) - 300
                    result_str = "\n".join(_lines[:300]) + f"\n[… {omitted} lines omitted …]"

            results.append(ToolResultContent(
                tool_use_id=tc.id,
                content=result_str,
                is_error=is_error,
                name=tc.name,
            ))

            if self.config.trace:
                self._trace_tool_result(tc.name, result_str, is_error)
            elif self.config.verbose:
                preview = result_str[:300] + ("..." if len(result_str) > 300 else "")
                self._console.print(f"  [dim]→ {escape(preview)}[/dim]")

        # --- Aggregate size cap: persist the largest results first ----------
        _AGGREGATE_LIMIT = 200_000
        aggregate = sum(len(r.content) for r in results)
        if aggregate > _AGGREGATE_LIMIT:
            # Sort by content size descending, persist until aggregate is small
            import uuid as _uuid3
            for r in sorted(results, key=lambda x: len(x.content), reverse=True):
                if aggregate <= _AGGREGATE_LIMIT:
                    break
                if "[Output persisted —" in r.content:
                    continue  # already persisted per-result
                try:
                    self._tool_results_dir.mkdir(parents=True, exist_ok=True)
                    fname = self._tool_results_dir / f"{_uuid3.uuid4().hex[:8]}.txt"
                    fname.write_text(r.content, encoding="utf-8", errors="replace")
                    n_lines = r.content.count("\n") + 1
                    n_chars = len(r.content)
                    aggregate -= n_chars
                    r.content = (
                        f"[Output persisted (aggregate limit) — {n_lines:,} lines, "
                        f"{n_chars:,} chars. Full output at: {fname}]\n"
                        f"[Use the Read tool to retrieve it if needed.]"
                    )
                    aggregate += len(r.content)
                except Exception:
                    pass  # non-fatal; leave content as-is

        return results

    _LABEL_SKIP_KEYS = frozenset({"description", "run_in_background", "timeout"})

    def _tool_box_label(self, tc: ToolUseContent, max_width: int = 200) -> str:
        """Build 'ToolName  key: value  …' label with smart truncation.

        For file paths that make the label too long, the path value is shortened
        from the left so the filename is always visible.  All other keys are kept.
        """
        name = tc.name
        inp = tc.input

        kv_pairs: list[tuple[str, str]] = [
            (k, str(v))
            for k, v in inp.items()
            if k not in self._LABEL_SKIP_KEYS
        ]

        def _build(pairs: list[tuple[str, str]]) -> str:
            if not pairs:
                return name
            return name + "  " + "  ".join(f"{k}: {v}" for k, v in pairs)

        label = _build(kv_pairs)
        if len(label) <= max_width:
            return label

        # Shorten path/url values from the left, keeping all keys visible.
        pairs = list(kv_pairs)
        for i, (k, v) in enumerate(pairs):
            if ("path" in k.lower() or "url" in k.lower()) and "/" in v and len(v) > 10:
                # Compute how much room this value can occupy
                other = [p for j, p in enumerate(pairs) if j != i]
                other_str = ("  " + "  ".join(f"{k2}: {v2}" for k2, v2 in other)) if other else ""
                avail = max_width - len(name) - 2 - len(f"{k}: ") - len(other_str)
                if avail < 6:
                    continue
                if len(v) <= avail:
                    continue   # this path is fine already
                # Keep as much of the END as possible, snapping to a '/' boundary
                tail = v[len(v) - avail + 3:]
                slash = tail.find("/")
                if slash > 0:
                    tail = tail[slash:]
                pairs[i] = (k, "..." + tail)
                label = _build(pairs)
                if len(label) <= max_width:
                    return label

        # Generic fallback: truncate at end
        return label[:max_width - 3] + "..."

    async def _execute_one(self, tc: ToolUseContent) -> str:
        """Observe tool lifecycle, including policy decisions and cancellation."""
        with self._observer.span(
            f"cca.tool.{tc.name}",
            kind="tool",
            attributes={
                "cca.tool.name": tc.name,
                "cca.context_generation": self._context_generation,
                "gen_ai.tool.call.id": tc.id,
                "gen_ai.tool.name": tc.name,
            },
            content={"gen_ai.tool.call.arguments": tc.input},
        ) as span:
            result = await self._execute_one_impl(tc)
            if result.startswith("Tool blocked"):
                status = "blocked"
            elif result.startswith("Tool execution denied"):
                status = "denied"
            elif result.startswith("Error:") or "[py_compile ERROR]" in result:
                status = "error"
            else:
                status = "ok"
            span.finish(
                status=status,
                content={"gen_ai.tool.call.result": result},
            )
            return result

    async def _execute_one_impl(self, tc: ToolUseContent) -> str:
        import subprocess as _sp
        from cca.ui.events import current_tool_id
        current_tool_id.set(tc.id)   # routes bash output to this tool's box

        # Pre-tool hook: may block the tool call
        hook_pre = await self._hooks.pre_tool_use(tc.name, tc.input, tc.id)
        if not hook_pre.allowed:
            reason = hook_pre.stderr or "Blocked by hook"
            return f"Tool blocked by hook: {reason}"

        allowed = await self.permissions.check(tc.name, tc.input)
        if not allowed:
            return "Tool execution denied by user."
        try:
            result = await self.registry.execute(tc.name, tc.input)
        except Exception as e:
            raise RuntimeError(f"Tool '{tc.name}' raised: {e}") from e

        if tc.name in ("Edit", "Write"):
            file_path = tc.input.get("file_path", "")
            if file_path.endswith(".py"):
                loop = asyncio.get_running_loop()
                chk = await loop.run_in_executor(
                    None,
                    lambda: _sp.run(
                        ["python3", "-m", "py_compile", file_path],
                        capture_output=True,
                        text=True,
                    ),
                )
                if chk.returncode != 0:
                    err = (chk.stderr or chk.stdout).strip()
                    result += f"\n\n[py_compile ERROR]\n{err}"

        # Post-tool hook: may inject additional context into the tool result
        hook_post = await self._hooks.post_tool_use(tc.name, tc.input, tc.id, result)
        if hook_post.additional_context:
            result += f"\n\n<system-reminder>\n{hook_post.additional_context}\n</system-reminder>"

        return result


    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _agent_tool_visible(self) -> bool:
        feature_tool_enabled = getattr(self._features, "tool_enabled", None)
        return bool(
            callable(feature_tool_enabled)
            and feature_tool_enabled("Agent")
            and getattr(self, "registry", None) is not None
            and self.registry.get("Agent") is not None
            and (
                getattr(self, "_allowed_tools", None) is None
                or "Agent" in self._allowed_tools
            )
            and "Agent" not in getattr(self, "_disallowed_tools", set())
        )

    def _build_system(self) -> str:
        from cca.prompts import environment_block, main_prompt
        from cca.memory import load_index

        if self._system_prompt_override is not None:
            base = self._system_prompt_override
        elif self._agent_profile is not None:
            extra_parts = [
                self._project_instructions,
                self.config.system_prompt_append,
                self._system_prompt_append,
            ]
            extra = "\n".join(part for part in extra_parts if part)
            base = (
                self._agent_profile.prompt
                + "\n\n" + environment_block(self.cwd)
                + (f"\n\n{extra}" if extra else "")
            )
        else:
            extra_parts = [
                self._project_instructions,
                self.config.system_prompt_append,
                self._system_prompt_append,
            ]
            extra = "\n".join(p for p in extra_parts if p)
            base = main_prompt(cwd=self.cwd, extra=extra,
                               template=self._randomization.system_prompt)

        if self._features.tool_enabled("Skill") and getattr(self.config, "skills_enabled", False):
            from cca.tools.skill_tool import skill_listing_block
            listing = skill_listing_block()
            if listing:
                base += f"\n\n{listing}"

        # This policy is present only in the explicitly multi-agent-capable
        # scaffold. With tools.Agent disabled, the legacy root prompt and action
        # space remain untouched.
        if self._agent_tool_visible():
            base += (
                "\n\n# Delegation\n"
                "Use the Agent tool only for a bounded, independently verifiable task "
                "when fresh context or parallel work is materially useful. Do not "
                "delegate simple or tightly serial work. The root agent owns "
                "decomposition, conflict handling, verification, and the final answer. "
                "When delegating, provide scope, constraints, expected evidence, and a "
                "clear completion condition. Use Agent.tasks for two or more independent "
                "tasks that must rejoin before one final answer. Background execution is "
                "only for deliberately detached work that may continue in a later turn."
            )

        if self._features.feature("memory"):
            memory_index = load_index(self.cwd)
            if memory_index:
                base += f"\n\n# Memory\n{memory_index}"

            if self._memory_facts:
                facts_text = "\n".join(f"- {f}" for f in self._memory_facts)
                base += f"\n\n<memory_facts>\n{facts_text}\n</memory_facts>"

        if self._compaction_summary:
            base += (
                "\n\n<compacted_context>\nThis session is being continued from a "
                "previous conversation that ran out of context. The summary below "
                "covers the earlier portion of the conversation.\n\n"
                f"{self._compaction_summary}\n\n"
                "Continue the conversation from where it left off without asking "
                "the user any further questions. Resume directly — do not "
                "acknowledge the summary, do not recap what was happening, do not "
                "preface with 'I'll continue' or similar. Pick up the last task "
                "as if the break never happened.\n"
                "</compacted_context>"
            )

        if self._post_compact_files:
            file_sections: list[str] = []
            for fp, content in self._post_compact_files.items():
                file_sections.append(f"<file path='{fp}'>\n{content}\n</file>")
            base += (
                "\n\n<recently_read_files>\n"
                "The following files were being read just before context compaction. "
                "Their current contents are provided so you can continue working "
                "without re-reading them.\n\n"
                + "\n\n".join(file_sections)
                + "\n</recently_read_files>"
            )

        # Re-inject plan mode instruction after compaction (or any call while active)
        from cca.tools.plan_tools import is_plan_mode_active, get_plan_file
        if is_plan_mode_active():
            plan_file = get_plan_file()
            plan_path = str(plan_file) if plan_file else "(unknown)"
            base += (
                "\n\n<system-reminder>\n"
                "Planning lock is active. Repository and environment mutations are "
                "outside the current task; use inspection tools to gather evidence. "
                f"The sole writable target is the active plan: {plan_path}\n\n"
                "Develop the plan as a working document rather than postponing it "
                "until research is finished. Begin with the likely approach, inspect "
                "the files that can confirm or disprove it, and update the document "
                "as conclusions change. Ask the user only about choices that cannot "
                "be settled from the repository.\n\n"
                "Before requesting approval, make the plan implementation-ready: "
                "name the affected components, describe the change sequence and "
                "interfaces, identify material risks, and give concrete verification "
                "steps. Add a compact Mermaid or ASCII diagram when branching, "
                "dependencies, or cross-component flow would otherwise be hard to "
                "follow. A straight-line edit does not need one.\n\n"
                "Use ExitPlanMode to submit the finished plan for the user's decision. "
                "If it is returned for revision, keep planning and update the same "
                "file. Do not begin implementation until approval releases the lock.\n"
                "</system-reminder>"
            )

        return base

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    async def _compact_if_needed(
        self,
        direction: str = "full",
        check_type: str = "lower",
    ) -> None:
        """LLM-based context compaction.

        direction:
          "full"   — summarise the old prefix, keep recent messages (default)
          "prefix" — same as "full" (alias for clarity)
          "tail"   — summarise the recent tail, keep the old prefix verbatim

        Tries to compact with one LLM call.  If that fails falls back to a
        two-pass split, then an emergency slice.
        """
        # Feature gate covers automatic triggers only; explicit /compact
        # (check_type="force") remains available.
        if check_type != "force" and not self._features.feature("compaction"):
            return
        import json as _json
        import re as _re
        from cca.config import CONFIG_DIR

        if check_type == "force":
            COMPACT_THRESHOLD = -1
        elif check_type == "upper":
            COMPACT_THRESHOLD = getattr(self.config, "compact_threshold_upper", 200_000)
        else:
            COMPACT_THRESHOLD = getattr(self.config, "compact_threshold", 100_000)
            
        if check_type == "upper":
            COMPACT_KEEP_RECENT = 50
        elif check_type == "force":
            COMPACT_KEEP_RECENT = 10
        else:
            COMPACT_KEEP_RECENT = 40

        # --- Check token budget: prefer API-reported count over estimation ----
        # _last_input_tokens is the total input token count from the most recent
        # API call; it accurately reflects the current context size. Fall back to
        # character estimation only if we have never received an API count yet.

        def _get_estimated_tokens():
            tot = 0
            for msg in self.messages:
                if isinstance(msg.content, str):
                    tot += len(msg.content) // 4
                else:
                    for blk in msg.content:
                        if isinstance(blk, TextContent):
                            tot += len(blk.text) // 4
                        elif isinstance(blk, ToolResultContent):
                            tot += len(blk.content) // 4
                        elif isinstance(blk, ToolUseContent):
                            tot += len(_json.dumps(blk.input)) // 4
            return tot

        if self._last_input_tokens > 0:
            total = self._last_input_tokens
        else:
            total = _get_estimated_tokens()

        if total <= COMPACT_THRESHOLD:
            return

        source = "API-reported" if self._last_input_tokens > 0 else "estimated"
        self._console.print(
            f"[dim]Context compaction triggered ({source}: {total:,} tokens).[/dim]"
        )

        # Archive the exact pre-mutation messages before micro-compaction can
        # snip tool results. This transcript is the lossless recovery source.
        transcript_dir = CONFIG_DIR / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"session_{self._session_id}.jsonl"
        try:
            with open(transcript_path, "a") as tf:
                for msg in self.messages:
                    tf.write(_json.dumps(msg.to_dict()) + "\n")
        except Exception:
            transcript_path = None

        # --- Micro-compaction (Snipping Tool Results) for upper limit mid-task ---
        if check_type == "upper":
            # Target buffer slightly below the upper threshold so we don't immediately trigger again
            target_buffer = int(COMPACT_THRESHOLD * 0.8)
            snipped_something = False
            
            # Iterate backward through messages to snip the largest tool results
            for msg_idx in range(len(self.messages)-1, max(-1, len(self.messages)-100), -1):
                if total <= target_buffer:
                    break
                
                msg = self.messages[msg_idx]
                if msg.role == "user" and not isinstance(msg.content, str):
                    for blk in msg.content:
                        if isinstance(blk, ToolResultContent) and len(blk.content) > 4000:
                            # Keep first 1000 and last 1000 chars, snip the middle
                            old_len = len(blk.content)
                            snip_msg = "\n...[Content snipped to save context space]...\n"
                            blk.content = blk.content[:1000] + snip_msg + blk.content[-1000:]
                            
                            # Update our estimated token count based on characters removed
                            chars_saved = old_len - len(blk.content)
                            tokens_saved = chars_saved // 4
                            total -= tokens_saved
                            snipped_something = True
            
            if snipped_something:
                self._console.print(f"[dim]Micro-compacted tool results. New estimated tokens: {total:,}.[/dim]")
                # If micro-compaction brought us under the buffer, we are done! No LLM needed.
                if total <= target_buffer:
                    previous_generation = self._context_generation
                    next_generation = previous_generation + 1
                    from cca.trace_store import TraceIdentity
                    next_identity = TraceIdentity(
                        trace_id=self._trace_id,
                        agent_id=self._agent_id,
                        generation=next_generation,
                        context_generation=next_generation,
                        generation_reason="compaction",
                    )
                    try:
                        self._trace_store.append_root_event(
                            "trace.context.started",
                            next_identity,
                            reason="compaction",
                            compaction_mode="micro",
                            previous_context_generation=previous_generation,
                            session_id=self._session_id,
                            transcript_path=(
                                str(transcript_path) if transcript_path else None
                            ),
                        )
                    except Exception:
                        pass
                    self._context_generation = next_generation
                    self._generation_reason = "compaction"
                    self._root_trace_identity = next_identity
                    self._observer.add_common_attributes({
                        "cca.context_generation": next_generation,
                    })
                    self._last_input_tokens = total
                    self._sync_agent_tool_settings()
                    return
                self._console.print("[dim]Micro-compaction insufficient, falling back to LLM summarization...[/dim]")

        await self._hooks.pre_compact(len(self.messages))

        # --- Find split: keep last N messages at a *text* user boundary ------
        def _is_text_user(msg: Message) -> bool:
            if msg.role != "user":
                return False
            if isinstance(msg.content, str):
                return True
            return any(isinstance(b, TextContent) for b in msg.content)

        if direction == "tail":
            # Keep the old prefix verbatim; summarise the recent tail.
            split = min(COMPACT_KEEP_RECENT, len(self.messages))
            # Walk backward to a text-user boundary so we don't sever tool_use from tool_result
            while split > 0 and not _is_text_user(self.messages[split]):
                split -= 1
            to_keep = self.messages[:split]
            to_summarize = self.messages[split:]
        else:
            # Default / "prefix": summarise the old prefix, keep recent tail.
            split = max(0, len(self.messages) - COMPACT_KEEP_RECENT)
            
            # Walk backward to find any user message
            while split > 0 and self.messages[split].role != "user":
                split -= 1
                
            # If the user message is a tool_result (not text), include the preceding assistant message too
            if split > 0 and not _is_text_user(self.messages[split]):
                split -= 1
                
            to_summarize = self.messages[:split]
            to_keep = self.messages[split:]
            
            # API requires messages to start with a 'user' role
            if to_keep and to_keep[0].role == "assistant":
                
                dummy = Message(role="user", content=[TextContent(text="[Continuing task after context compaction]")])
                to_keep.insert(0, dummy)

        if not to_summarize:
            self._console.print(
                f"[dim yellow]Nothing to compact — only {len(self.messages)} messages "
                f"(need >{COMPACT_KEEP_RECENT} to have something to summarize).[/dim yellow]"
            )
            return

        # --- Inner helpers ---------------------------------------------------

        _TOOL_RESULT_MAX = 2000
        _TOOL_RESULT_KEEP = 800

        def _truncate_for_compactor(text: str) -> str:
            """Truncate a tool result for the compaction LLM input."""
            if len(text) <= _TOOL_RESULT_MAX:
                return text
            snipped = len(text) - _TOOL_RESULT_KEEP * 2
            return (
                text[:_TOOL_RESULT_KEEP]
                + f"\n[...snipped {snipped:,} chars for compaction...]\n"
                + text[-_TOOL_RESULT_KEEP:]
            )

        def _render_msgs(msgs: list, prior_summary: str | None = None) -> str:
            """Render a message list to plain text for the compaction LLM."""
            parts: list[str] = []
            if prior_summary:
                parts.append(
                    "[PRIOR COMPACTION SUMMARY — incorporate this into your new summary]\n"
                    + prior_summary
                )
            for msg in msgs:
                role = msg.role.upper()
                if isinstance(msg.content, str):
                    parts.append(f"[{role}]\n{msg.content}")
                else:
                    for blk in msg.content:
                        if isinstance(blk, TextContent):
                            parts.append(f"[{role}]\n{blk.text}")
                        elif isinstance(blk, ToolUseContent):
                            parts.append(
                                f"[{role} TOOL_CALL: {blk.name}]\n"
                                + _json.dumps(blk.input, indent=2)
                            )
                        elif isinstance(blk, ToolResultContent):
                            tag = "TOOL_RESULT_ERROR" if blk.is_error else "TOOL_RESULT"
                            parts.append(f"[{tag}]\n{_truncate_for_compactor(blk.content)}")
            return "\n\n".join(parts)

        def _extract_summary(raw: str) -> str | None:
            """Extract <summary> from response, stripping the <scratchpad> block."""
            cleaned = _re.sub(r"<scratchpad>.*?</scratchpad>", "", raw, flags=_re.DOTALL)
            m = _re.search(r"<summary>(.*?)</summary>", cleaned, _re.DOTALL)
            self._console.print(
                f"[dim]_extract_summary: len={len(raw)}, has_tags={('<summary>' in raw)}, "
                f"regex_match={m is not None}, "
                f"match_len={len(m.group(1).strip()) if m else 0}[/dim]"
            )
            if not m:
                return None
            text = m.group(1).strip()
            if len(text) < 200:
                return None
            return text

        async def _call_compactor(text: str) -> str:
            with self._observer.span(
                "cca.compaction",
                kind="compaction",
                attributes={
                    "cca.compaction.check_type": check_type,
                    "cca.compaction.direction": direction,
                    "cca.compaction.message_count": len(self.messages),
                    "cca.context_generation": self._context_generation,
                },
            ) as span:
                summary = await _call_compactor_impl(text)
                span.finish(
                    status="ok",
                    attributes={"cca.compaction.summary_chars": len(summary)},
                )
                return summary

        async def _call_compactor_impl(text: str) -> str:
            """Send text to the LLM and return the extracted summary."""
            # Debug: log input size
            self._console.print(
                f"[dim]Compactor input: {len(text):,} chars (~{len(text)//4:,} tokens)[/dim]"
            )
            compactor_messages = [
                Message(role="user", content=[TextContent(text=text)])
            ]
            resp = await self._complete_internal_observed(
                call_type="compactor",
                messages=compactor_messages,
                tools=[],
                system=self._compaction_prompt,
                max_tokens=16_384,
                think_level=self._internal_think_level,
            )
            raw = resp.text.strip()
            self._log_llm_call(
                call_type="compactor",
                system=self._compaction_prompt,
                messages_snapshot=[{
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }],
                tools=[],
                params={
                    "max_tokens": 16_384,
                    "think_level": self._internal_think_level,
                    "thinking": {
                        "requested": "off",
                        "effective": self._internal_think_level,
                        "source": "internal-policy",
                    },
                    "backend_request": self._describe_backend_request(
                        self._internal_think_level, False
                    ),
                },
                response_text=raw,
                tool_calls=[],
                thinking="",
                stop_reason="end_turn",
                input_tokens=getattr(resp, "input_tokens", 0),
                output_tokens=getattr(resp, "output_tokens", 0),
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

            # Debug: dump raw response for inspection
            _debug_path = CONFIG_DIR / "compact_debug"
            _debug_path.mkdir(parents=True, exist_ok=True)
            import time as _time
            _debug_file = _debug_path / f"compact_{int(_time.time())}.txt"
            try:
                with open(_debug_file, "w") as _df:
                    _df.write(f"=== INPUT ({len(text)} chars) ===\n")
                    _df.write(text[:2000] + "\n...\n" + text[-500:] if len(text) > 2500 else text)
                    _df.write(f"\n\n=== RAW RESPONSE ({len(raw)} chars) ===\n")
                    _df.write(raw)
                    _df.write(f"\n\n=== EXTRACTION DEBUG ===\n")
                    _df.write(f"raw empty: {not raw}\n")
                    _df.write(f"has <scratchpad>: {'<scratchpad>' in raw}\n")
                    _df.write(f"has </scratchpad>: {'</scratchpad>' in raw}\n")
                    _df.write(f"has <summary>: {'<summary>' in raw}\n")
                    _df.write(f"has </summary>: {'</summary>' in raw}\n")
                    cleaned = _re.sub(r"<scratchpad>.*?</scratchpad>", "", raw, flags=_re.DOTALL)
                    _df.write(f"cleaned length: {len(cleaned)}\n")
                    m = _re.search(r"<summary>(.*?)</summary>", cleaned, _re.DOTALL)
                    _df.write(f"regex match: {m is not None}\n")
                    if m:
                        _df.write(f"match length: {len(m.group(1).strip())}\n")
                        _df.write(f"match passes 200 char check: {len(m.group(1).strip()) >= 200}\n")
                self._console.print(f"[dim]Debug: saved to {_debug_file}[/dim]")
            except Exception:
                pass

            summary = _extract_summary(raw) if raw else None
            if summary:
                return summary

            self._console.print(
                "[dim yellow]Compaction produced no valid <summary>; retrying with nudge…[/dim yellow]"
            )
            retry_text = (
                text
                + "\n\nIMPORTANT: Your previous attempt did not produce a valid "
                "handoff. You MUST wrap the surviving handoff in "
                "<summary></summary> tags. Prepare in a <scratchpad> block "
                "first, then write the full handoff with all of its sections."
            )
            retry_messages = [
                Message(role="user", content=[TextContent(text=retry_text)])
            ]
            resp2 = await self._complete_internal_observed(
                call_type="compactor_retry",
                messages=retry_messages,
                tools=[],
                system=self._compaction_prompt,
                max_tokens=16_384,
                think_level=self._internal_think_level,
            )
            raw2 = resp2.text.strip()
            self._log_llm_call(
                call_type="compactor_retry",
                system=self._compaction_prompt,
                messages_snapshot=[{
                    "role": "user",
                    "content": [{"type": "text", "text": retry_text}],
                }],
                tools=[],
                params={
                    "max_tokens": 16_384,
                    "think_level": self._internal_think_level,
                    "thinking": {
                        "requested": "off",
                        "effective": self._internal_think_level,
                        "source": "internal-policy",
                    },
                    "backend_request": self._describe_backend_request(
                        self._internal_think_level, False
                    ),
                },
                response_text=raw2,
                tool_calls=[],
                thinking="",
                stop_reason="end_turn",
                input_tokens=getattr(resp2, "input_tokens", 0),
                output_tokens=getattr(resp2, "output_tokens", 0),
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
            # Debug: dump retry response
            try:
                _debug_file2 = _debug_path / f"compact_{int(_time.time())}_retry.txt"
                with open(_debug_file2, "w") as _df2:
                    _df2.write(f"=== RETRY RAW RESPONSE ({len(raw2)} chars) ===\n")
                    _df2.write(raw2)
                self._console.print(f"[dim]Debug: retry saved to {_debug_file2}[/dim]")
            except Exception:
                pass
            summary2 = _extract_summary(raw2) if raw2 else None
            if summary2:
                return summary2

            if raw2:
                self._console.print(
                    "[dim yellow]Retry also lacked <summary> tags; using raw response.[/dim yellow]"
                )
                return raw2
            if raw:
                self._console.print(
                    "[dim yellow]Retry empty; falling back to first response.[/dim yellow]"
                )
                return raw
            raise RuntimeError("Compaction LLM returned empty response on both attempts")

        def _emergency_slice() -> None:
            """Last-resort: trim to the most-recent text-user-bounded slice."""
            sliced = self.messages[-EMERGENCY_KEEP_MESSAGES:]
            for i, msg in enumerate(sliced):
                if _is_text_user(msg):
                    self.messages = sliced[i:]
                    return
            # No text-user in the recent slice; scan the full history
            for i in range(len(self.messages) - 1, -1, -1):
                if _is_text_user(self.messages[i]):
                    self.messages = self.messages[i:]
                    return

        # --- Attempt 1: head-drop retry (drop oldest ¼ of history each time) ---
        # If the compaction request itself exceeds the model's context, drop the
        # oldest messages from the compaction input and retry, up to 3 times.
        summary: str | None = None
        last_exc: Exception | None = None

        self._console.print(
            f"[dim]Summarizing {len(to_summarize)} messages…[/dim]"
        )

        for _drop_pass in range(4):   # 0 = full history, 1-3 = progressive head-drop
            drop = (len(to_summarize) * _drop_pass) // 4
            attempt_msgs = to_summarize[drop:]
            if not attempt_msgs:
                break
            try:
                summary = await _call_compactor(
                    _render_msgs(attempt_msgs, self._compaction_summary)
                )
                break   # success
            except Exception as exc:
                last_exc = exc
                if _drop_pass < 3:
                    self._console.print(
                        f"[dim yellow]Compaction attempt {_drop_pass + 1} failed "
                        f"({exc!r}); dropping oldest {drop} messages and retrying…"
                        "[/dim yellow]"
                    )

        if summary is None:
            self._console.print(
                f"[dim yellow]All head-drop retries failed ({last_exc!r}); "
                "trying two-pass split…[/dim yellow]"
            )

            # --- Attempt 2: compact each half separately, chain the summaries -
            mid = max(1, len(to_summarize) // 2)
            first_summary: str | None = None

            try:
                self._console.print("[dim]Compacting first half…[/dim]")
                first_summary = await _call_compactor(
                    _render_msgs(to_summarize[:mid], self._compaction_summary)
                )
            except Exception as e2:
                self._console.print(
                    f"[dim red]First-half compaction failed ({e2!r}); emergency slice.[/dim red]"
                )
                _emergency_slice()
                return

            try:
                self._console.print("[dim]Compacting second half…[/dim]")
                summary = await _call_compactor(
                    _render_msgs(to_summarize[mid:], first_summary)
                )
            except Exception as e3:
                self._console.print(
                    f"[dim yellow]Second-half compaction failed ({e3!r}); using first-half summary.[/dim yellow]"
                )
                summary = first_summary

        # --- Finalise --------------------------------------------------------
        if transcript_path:
            summary += (
                f"\n\nIf you need specific details from before compaction, "
                f"read the full transcript at: {transcript_path}"
            )

        # Save the full pre-compaction session before messages are overwritten.
        # This writes xxx.json so sim and other tools can recover the API calls
        # that happened before this compaction event.
        try:
            from cca.sessions import save_session as _save_pre_compact
            _save_pre_compact(
                self._session_id,
                self.cwd,
                self.messages,
                system_prompt=self._build_system(),
                token_counts=dict(self._token_counts),
                features=self._features.to_payload(),
                randomization=self._randomization_payload,
                settings=self._runtime_settings_payload(),
                state=self._session_state_payload(),
            )
        except Exception:
            pass

        if direction == "tail":
            # Tail mode: old messages stay; append summary as a user note.
            tail_note = Message(
                role="user",
                content=[TextContent(
                    text=(
                        "<compacted_tail>\n"
                        + summary
                        + "\n</compacted_tail>"
                    )
                )],
            )
            self.messages = list(to_keep) + [tail_note]
            # Don't update _compaction_summary — the prefix context is intact.
        else:
            self._compaction_summary = summary
            self.messages = list(to_keep)
            # Track how many messages were summarized away so save_session can
            # write compacted_from_index for lossless comp chain reconstruction.
            self._msg_index_offset += split
        self._last_input_tokens = 0  # reset so delta starts fresh after truncation

        from cca.sessions import generate_session_id, mark_compacted_in_history
        previous_session_id = self._session_id
        next_session_id = generate_session_id(
            is_compaction=True, current_id=previous_session_id
        )
        next_generation = self._context_generation + 1
        from cca.trace_store import TraceIdentity
        next_identity = TraceIdentity(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            generation=next_generation,
            context_generation=next_generation,
            generation_reason="compaction",
        )
        try:
            self._trace_store.append_root_event(
                "trace.context.started",
                next_identity,
                reason="compaction",
                previous_context_generation=self._context_generation,
                previous_session_id=previous_session_id,
                session_id=next_session_id,
            )
        except Exception:
            # Context safety takes precedence over optional lifecycle metadata;
            # the next model call still carries the new generation explicitly.
            pass
        mark_compacted_in_history(previous_session_id)
        self._session_id = next_session_id
        self._context_generation = next_generation
        self._generation_reason = "compaction"
        self._root_trace_identity = next_identity
        self._observer.add_common_attributes({
            "cca.session_id": next_session_id,
            "cca.context_generation": next_generation,
        })
        self._sync_agent_tool_settings()

        await self._hooks.post_compact(len(summary))

        self._console.print(
            f"[dim green]Compaction complete: {len(summary):,} char summary.[/dim green]"
        )

        # Re-inject recently-read files so the model can continue without re-reading.
        _FILE_BUDGET = 50_000   # chars (~12.5K tokens)
        self._post_compact_files = {}
        budget = _FILE_BUDGET
        for fp in reversed(self._recently_read):  # most-recent first
            if budget <= 0:
                break
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="replace")
                if len(text) > budget:
                    text = text[:budget] + "\n[… truncated to fit post-compaction budget]"
                self._post_compact_files[fp] = text
                budget -= len(text)
            except Exception:
                pass  # file may have been deleted; skip silently

        self._console.print(
            f"[dim]Compacted {len(to_summarize)} messages → system prompt summary "
            f"+ {len(to_keep)} kept. Transcript: {transcript_path}[/dim]"
        )

    # ------------------------------------------------------------------
    # Memory extraction
    # ------------------------------------------------------------------

    def _estimate_tokens(self) -> int:
        import json as _json
        total = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                total += len(msg.content) // 4
            else:
                for blk in msg.content:
                    if isinstance(blk, TextContent):
                        total += len(blk.text) // 4
                    elif isinstance(blk, ToolResultContent):
                        total += len(blk.content) // 4
                    elif isinstance(blk, ToolUseContent):
                        total += len(_json.dumps(blk.input)) // 4
        return total

    async def _maybe_update_memory_facts(self, user_query: str) -> None:
        """Run the two-stage memory extraction if thresholds are met."""
        import json as _json
        from cca.memory import build_memory_manifest, list_memories
        from cca.prompts import MEMORY_EXTRACTOR_PROMPT

        def _log_memory_call(
            call_type: str,
            messages: list[Message],
            response,
            max_tokens: int,
        ) -> None:
            usage = response.usage or {}
            self._log_llm_call(
                call_type=call_type,
                system=MEMORY_EXTRACTOR_PROMPT,
                messages_snapshot=[message.to_dict() for message in messages],
                tools=[],
                params={
                    "max_tokens": max_tokens,
                    "think_level": self._internal_think_level,
                    "backend_request": self._describe_backend_request(
                        self._internal_think_level, False
                    ),
                },
                response_text=response.text,
                tool_calls=[
                    {"id": call.id, "name": call.name, "input": call.input}
                    for call in response.tool_calls
                ],
                thinking="".join(
                    block.thinking for block in response.content
                    if isinstance(block, ThinkingContent)
                ),
                stop_reason=response.stop_reason,
                input_tokens=usage.get("input_tokens", usage.get("input", 0)),
                output_tokens=usage.get("output_tokens", usage.get("output", 0)),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_write_tokens=usage.get("cache_write_tokens", 0),
            )

        if not list_memories(self.cwd):
            return

        estimated = self._estimate_tokens()

        # Don't run until the session has enough content to be worth it
        if estimated < _MEM_MIN_TOKENS_TO_INIT:
            return

        # Re-run if enough new tokens or tool calls since last extraction
        tokens_since = estimated - self._mem_tokens_at_last_extraction
        if (tokens_since < _MEM_MIN_TOKENS_BETWEEN_UPDATE
                and self._mem_tool_calls_since_extraction < _MEM_TOOL_CALLS_BETWEEN_UPDATES
                and self._extractor_messages):  # skip check on first run
            return

        # Build manifest on first call; reuse stateul conversation thereafter
        if not self._extractor_messages:
            manifest = build_memory_manifest(self.cwd)
            if not manifest:
                return
            self._extractor_messages = [
                Message(role="user", content=[TextContent(text=manifest)])
            ]
            # Prime the extractor with an acknowledgment turn
            try:
                init_resp = await self._complete_internal_observed(
                    call_type="memory_extractor_init",
                    messages=self._extractor_messages,
                    tools=[],
                    system=MEMORY_EXTRACTOR_PROMPT,
                    max_tokens=50,
                    think_level=self._internal_think_level,
                )
                _log_memory_call(
                    "memory_extractor_init",
                    self._extractor_messages,
                    init_resp,
                    50,
                )
                self._extractor_messages.append(
                    Message(role="assistant", content=[TextContent(text=init_resp.text or "Ready.")])
                )
            except Exception:
                self._extractor_messages = []
                return

        # Send the user query and get extracted facts
        query_msg = Message(role="user", content=[TextContent(text=user_query)])
        try:
            extractor_input = [*self._extractor_messages, query_msg]
            resp = await self._complete_internal_observed(
                call_type="memory_extractor",
                messages=extractor_input,
                tools=[],
                system=MEMORY_EXTRACTOR_PROMPT,
                max_tokens=2000,
                think_level=self._internal_think_level,
            )
            _log_memory_call("memory_extractor", extractor_input, resp, 2000)
            result = _json.loads(resp.text)
            facts = result.get("relevant_facts", [])
            if facts:
                self._memory_facts = facts
            # Extend stateful conversation
            self._extractor_messages.append(query_msg)
            self._extractor_messages.append(
                Message(role="assistant", content=[TextContent(text=resp.text)])
            )
            self._mem_tokens_at_last_extraction = estimated
            self._mem_tool_calls_since_extraction = 0
        except Exception:
            pass  # non-fatal: stale facts are better than crashing

    # ------------------------------------------------------------------
    # Trace display
    # ------------------------------------------------------------------

    def _trace_request(self, system: str, messages: list[Message]) -> None:
        """Show only the new messages being sent in this LLM call."""
        import json as _json

        if not self._trace_tools_shown:
            self._trace_tools_shown = True
            schemas = self._randomization.apply_to_schemas(self.registry.get_schemas())
            tool_lines = [
                f"  {s.name}: {s.description.splitlines()[0]}"
                for s in schemas
            ]
            self._console.print("\n[bold magenta]─── TOOLS[/bold magenta]", highlight=False)
            self._console.print("\n".join(tool_lines) or "  (none)")

        n = len(messages)
        self._console.print(
            f"\n[bold blue]─── INPUT[/bold blue] [dim]({n} msg{'s' if n != 1 else ''} total)[/dim]",
            highlight=False,
        )

        # Show only messages added since the last assistant response.
        last_asst = max(
            (i for i, m in enumerate(messages) if m.role == "assistant"),
            default=-1,
        )
        new_messages = messages[last_asst + 1:]

        for msg in new_messages:
            color, label = _TRACE_COLORS.get(msg.role, ("white", msg.role.upper()))
            content_blocks = msg.content if isinstance(msg.content, list) else [TextContent(text=msg.content)]
            for block in content_blocks:
                if isinstance(block, TextContent):
                    self._console.print(Panel(
                        escape(block.text),
                        title=f"[bold {color}]{label}[/bold {color}]",
                        border_style=color,
                        padding=(0, 1),
                    ))
                elif isinstance(block, ToolResultContent):
                    suffix = " [dim red]error[/dim red]" if block.is_error else ""
                    result_lines = block.content.splitlines()
                    if len(result_lines) > 50:
                        display = "\n".join(result_lines[:50]) + f"\n[… {len(result_lines) - 50} lines truncated in trace display]"
                    else:
                        display = block.content
                    self._console.print(Panel(
                        escape(display),
                        title=f"[bold {color}]{label}[/bold {color}] [dim]▸ tool_result{suffix}[/dim]",
                        border_style=color,
                        padding=(0, 1),
                    ))
                elif isinstance(block, ToolUseContent):
                    body = _json.dumps(block.input, indent=2)
                    sig = " [dim green]✓sig[/dim green]" if block.thought_signature else " [dim red]✗sig[/dim red]"
                    self._console.print(Panel(
                        escape(body),
                        title=f"[bold {color}]{label}[/bold {color}] [dim]▸ tool_call: {escape(block.name)}[/dim]{sig}",
                        border_style=color,
                        padding=(0, 1),
                    ))

        self._console.print("[bold green]─── OUTPUT[/bold green]", highlight=False)

    def _trace_tool_calls(self, content: list[ContentBlock]) -> None:
        import json as _json
        for block in content:
            if isinstance(block, ToolUseContent):
                body = _json.dumps(block.input, indent=2)
                sig = " [dim green]✓sig[/dim green]" if block.thought_signature else " [dim red]✗sig[/dim red]"
                self._console.print(Panel(
                    escape(body),
                    title=f"[bold yellow]CALLING → {escape(block.name)}[/bold yellow]{sig}",
                    border_style="yellow",
                    padding=(0, 1),
                ))

    def _trace_tool_result(self, name: str, result: str, is_error: bool) -> None:
        color = "red" if is_error else "magenta"
        lines = result.splitlines()
        if len(lines) > 50:
            display = "\n".join(lines[:50]) + f"\n[… {len(lines) - 50} lines truncated in trace display]"
        else:
            display = result
        self._console.print(Panel(
            escape(display),
            title=f"[bold {color}]RESULT ← {escape(name)}[/bold {color}]",
            border_style=color,
            padding=(0, 1),
        ))

    # ------------------------------------------------------------------
    # Notification handling
    # ------------------------------------------------------------------

    async def _drain_notifications(self) -> None:
        """Drain all pending notifications, batching monitor events."""
        if self._notif_queue is None or self._notif_queue.empty():
            return

        other: list[dict] = []
        monitor_batches: dict[str, dict] = {}

        while not self._notif_queue.empty():
            try:
                notif = self._notif_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if notif.get("type") == "monitor_event":
                mid = notif["monitor_id"]
                if mid not in monitor_batches:
                    monitor_batches[mid] = {"description": notif["description"], "lines": []}
                monitor_batches[mid]["lines"].append(notif["line"])
            else:
                other.append(notif)

        for mid, batch in monitor_batches.items():
            other.append({
                "type": "monitor_batch",
                "monitor_id": mid,
                "description": batch["description"],
                "message": "\n".join(batch["lines"]),
            })

        for notif in other:
            await self._process_notification(notif)

    async def _process_notification(self, notif: dict) -> None:
        """Deliver a background notification as a harness-initiated agent turn.

        The user-role message contains only a <system-reminder> block — no fake
        human speech. The model already knows from the system prompt that
        <system-reminder> tags come from the harness, not the user.
        """
        ntype = notif.get("type", "unknown")

        if ntype == "bash_done":
            job_id = notif["job_id"]
            rc = notif.get("returncode", "?")
            command = notif.get("command", "")
            output = (notif.get("output") or "").strip() or "(no output)"
            self._console.print(
                f"\n[bold yellow]◆ Background job {job_id} completed[/bold yellow] (exit {rc})"
            )
            inner = (
                f"Background Bash job {job_id} completed (exit {rc}).\n"
                f"Command: {command}\n"
                f"Output:\n{output}"
            )

        elif ntype == "agent_done":
            desc = notif["description"]
            job_id = notif.get("job_id", "")
            result = notif.get("result", "")
            self._console.print(f"\n[bold yellow]◆ Agent '{desc}' completed[/bold yellow]")
            inner = (
                f"Background agent '{desc}' (job_id={job_id}) completed.\n"
                f"Result:\n{result}"
            )

        elif ntype in ("monitor_event", "monitor_batch"):
            mid = notif.get("monitor_id", "?")
            desc = notif.get("description", "monitor")
            lines = notif.get("message", notif.get("line", ""))
            self._console.print(f"\n[bold yellow]◆ Monitor {mid}[/bold yellow] [{desc}]")
            inner = f"Monitor {mid} ({desc}):\n{lines}"

        elif ntype == "wakeup":
            reason = notif.get("reason", "")
            prompt = notif["prompt"]
            self._console.print(f"\n[bold yellow]◆ Wakeup:[/bold yellow] {reason}")
            inner = f"Scheduled wakeup: {reason}\n{prompt}"

        else:
            self._console.print(f"\n[bold yellow]◆ Notification:[/bold yellow] {ntype}")
            inner = f"Notification ({ntype}):\n{notif}"

        msg = f"<system-reminder>\n{inner}\n</system-reminder>"
        await self.process_turn(msg, harness_turn=True)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    async def _handle_slash(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            self._print_help()
            return True

        if command in ("/quit", "/exit", "/q"):
            self._console.print("[dim]Goodbye.[/dim]")
            sys.exit(0)

        if command == "/clear":
            previous_generation = self._context_generation
            next_generation = previous_generation + 1
            from cca.trace_store import TraceIdentity
            next_identity = TraceIdentity(
                trace_id=self._trace_id,
                agent_id=self._agent_id,
                generation=next_generation,
                context_generation=next_generation,
                generation_reason="clear",
            )
            try:
                self._trace_store.append_root_event(
                    "trace.context.started",
                    next_identity,
                    reason="clear",
                    previous_context_generation=previous_generation,
                    session_id=self._session_id,
                )
            except Exception as exc:
                self._console.print(
                    f"[red]Conversation was not cleared because its trace boundary "
                    f"could not be recorded:[/red] {escape(str(exc))}"
                )
                return True
            self.messages.clear()
            self._context_generation = next_generation
            self._generation_reason = "clear"
            self._root_trace_identity = next_identity
            self._observer.add_common_attributes({
                "cca.context_generation": next_generation,
            })
            self._sync_agent_tool_settings()
            self._compaction_summary = None
            self._msg_index_offset = 0
            self._last_input_tokens = 0
            self._post_compact_files = {}
            self._recently_read = []
            self._console.print("[dim]Conversation cleared.[/dim]")
            return True

        if command == "/model":
            if arg:
                self.config.model = arg
                from cca.llm.factory import create_backend
                from cca.llm.catalog import split_selector
                provider, _ = split_selector(arg)
                self.backend = create_backend(
                    arg,
                    api_key=self.config.get_api_key(provider),
                    base_url=self.config.base_url,
                )
                from cca.llm.thinking import resolve_thinking
                self._thinking_resolution = resolve_thinking(
                    arg,
                    self._thinking_resolution.requested,
                    self.config.default_thinking_level,
                )
                self._think_level = self._thinking_resolution.effective
                self._observer.add_common_attributes({
                    "gen_ai.request.model": arg,
                })
                self._sync_agent_tool_settings()
                self.config.save()
                if self._ui:
                    self._ui.update_model(arg)
                    self._ui.update_think_level(self._think_level)
                self._console.print(f"[green]Model switched to:[/green] {arg}")
            else:
                self._console.print(f"Current model: [cyan]{self.config.model}[/cyan]")
            return True

        if command == "/models":
            from cca.llm.catalog import get_provider, list_models, list_providers, provider_model_count
            if not arg:
                for provider in list_providers():
                    self._console.print(
                        f"  [cyan]{provider.id:<22}[/cyan] {provider_model_count(provider.id):>4}  {provider.name}"
                    )
                self._console.print("[dim]Use /models <provider-or-search> for model IDs.[/dim]")
                return True
            provider = arg if get_provider(arg) else None
            models = list_models(provider, query=None if provider else arg)
            for model in models[:50]:
                self._console.print(f"  [cyan]{model.selector}[/cyan]  {model.name}")
            if not models:
                self._console.print(f"[dim]No bundled models matched '{arg}'.[/dim]")
            elif len(models) > 50:
                self._console.print(f"[dim]Showing 50 of {len(models)} matches; use `cca models` for the full list.[/dim]")
            return True

        if command == "/think":
            await self._cmd_think(arg)
            return True

        if command == "/settings":
            self._print_full_settings()
            return True

        if command == "/key":
            # /key <provider> <api-key>
            parts = arg.split(None, 1) if arg else []
            if len(parts) == 2:
                provider, key = parts
                self.config.api_keys[provider.lower()] = key
                self.config.save()
                self._console.print(f"[green]API key for '{provider}' saved.[/green]")
            elif len(parts) == 1:
                provider = parts[0].lower()
                existing = self.config.api_keys.get(provider)
                if existing:
                    self._console.print(f"API key for '{provider}': [dim]{existing[:8]}…[/dim]")
                else:
                    self._console.print(f"[dim]No API key stored for '{provider}'.[/dim]")
            else:
                self._console.print("[dim]Usage: /key <provider> <api-key>[/dim]")
            return True

        if command == "/config":
            self._print_config()
            return True

        if command == "/memory":
            from cca.memory import list_memories
            mems = list_memories()
            if not mems:
                self._console.print("[dim]No memories stored.[/dim]")
            else:
                for m in mems:
                    self._console.print(f"  [{m.get('type','?')}] {m.get('name','?')} — {m.get('_file','')}")
            return True

        if command == "/work":
            from cca.tools.work_tools import _work_items
            if not _work_items:
                self._console.print("[dim]No work items.[/dim]")
            else:
                for work_id, work_item in _work_items.items():
                    self._console.print(
                        f"  [{work_id}] {work_item['status']:12} {work_item['subject']}"
                    )
            return True

        if command == "/agents":
            await self._cmd_agents(arg)
            return True

        if command == "/tokens":
            self._console.print(
                f"Session tokens — input: {self._token_counts['input']:,}  "
                f"output: {self._token_counts['output']:,}"
            )
            return True

        if command == "/compact":
            # Manual compaction — force regardless of token count.
            # Flags: --prefix (summarise old, keep recent) --tail (summarise recent, keep old)
            # Remaining arg text is appended to the compaction prompt as extra instructions.
            direction = "full"
            extra_arg = arg or ""
            for flag in ("--prefix", "--tail"):
                if extra_arg.startswith(flag):
                    direction = flag.lstrip("-")
                    extra_arg = extra_arg[len(flag):].strip()
                    break

            if extra_arg:
                orig_prompt = self._compaction_prompt
                self._compaction_prompt = orig_prompt + f"\n\nAdditional instructions: {extra_arg}\n"
                try:
                    await self._compact_if_needed(direction=direction, check_type="force")
                finally:
                    self._compaction_prompt = orig_prompt
            else:
                await self._compact_if_needed(direction=direction, check_type="force")
            return True

        if command == "/verbose":
            self.config.verbose = not self.config.verbose
            self._console.print(f"Verbose mode: {'on' if self.config.verbose else 'off'}")
            return True

        if command == "/trace":
            self.config.trace = not self.config.trace
            self._console.print(f"Trace mode: {'on' if self.config.trace else 'off'}")
            return True

        if command == "/vim":
            self.config.vim = not self.config.vim
            self.config.save()
            status = "on" if self.config.vim else "off"
            self._console.print(f"Vim mode: {status} (restart cca to apply)")
            return True

        if command == "/cwd":
            if arg:
                try:
                    # Slash commands are not interpreted by a shell. Accept the
                    # shell-style quoting users naturally type, while preserving
                    # unquoted paths that contain spaces for backward compatibility.
                    parsed = shlex.split(arg)
                    path_arg = parsed[0] if len(parsed) == 1 else arg
                    expanded = os.path.expandvars(os.path.expanduser(path_arg))
                    target = Path(expanded).resolve(strict=True)
                    if not target.is_dir():
                        raise NotADirectoryError(f"Not a directory: {target}")
                    os.chdir(target)
                except (OSError, RuntimeError, ValueError) as exc:
                    self._console.print(
                        f"[red]Cannot change working directory:[/red] {escape(str(exc))}"
                    )
                    return True

                self.cwd = str(target)
                self.permissions.update_cwd(self.cwd)
                if self._ui:
                    self._ui.update_cwd(self.cwd)
                self._sync_agent_tool_settings()
                self._console.print(f"Working directory: {self.cwd}")
            else:
                self._console.print(f"CWD: {self.cwd}")
            return True

        if command == "/init":
            await self._cmd_init()
            return True

        if command == "/doctor":
            await self._cmd_doctor()
            return True

        if command == "/hooks":
            self._cmd_hooks()
            return True

        if command == "/review":
            prompt = (
                f"Please review the code in {self.cwd}. Look for bugs, "
                "security issues, performance problems, and style improvements. "
                "Be concise and actionable."
            )
            if arg:
                prompt = f"Please review: {arg}"
            await self.process_turn(prompt)
            return True

        if command == "/allowed-tools":
            if arg:
                tool = arg.strip()
                if tool in self.permissions.always_allow:
                    self.permissions.always_allow.discard(tool)
                    self.config.always_allow = list(self.permissions.always_allow)
                    self.config.save()
                    self._console.print(f"[dim]Removed '{tool}' from always-allow list.[/dim]")
                else:
                    self.permissions.always_allow.add(tool)
                    self.config.always_allow = list(self.permissions.always_allow)
                    self.config.save()
                    self._console.print(f"[green]Added '{tool}' to always-allow list.[/green]")
            else:
                tools = sorted(self.permissions.always_allow)
                if tools:
                    self._console.print("Always-allowed tools: " + ", ".join(tools))
                else:
                    self._console.print("[dim]No tools in always-allow list.[/dim]")
            return True

        if command == "/budget":
            if arg:
                budget = _parse_budget_amount(arg)
                if budget:
                    self._token_budget = budget
                    self._console.print(f"[green]Token budget set to {budget:,}.[/green]")
                else:
                    self._console.print("[dim]Usage: /budget <number> (e.g. /budget 500k or /budget 200000)[/dim]")
            else:
                if self._token_budget:
                    self._console.print(f"Current token budget: {self._token_budget:,}")
                    self._console.print("[dim]/budget 0 to clear[/dim]")
                else:
                    self._console.print(f"[dim]No token budget set (default: {self.config.max_tokens:,}).[/dim]")
            return True

        if command == "/mcp":
            await self._cmd_mcp(arg)
            return True

        if command == "/resume":
            await self._cmd_resume(arg)
            return True

        if command == "/plan":
            result = await self._cmd_plan(arg)
            # None → no-arg case, already printed usage; True signals handled+no turn
            # str  → planning message to hand to the normal turn execution path
            return result if isinstance(result, str) else True

        if command == "/skills":
            await self._cmd_skills(arg)
            return True

        if command == "/features":
            await self._cmd_features()
            return True

        # Unknown slash command — let the LLM handle it
        return False

    # Amber — the plan mode brand colour; distinct from blue/green/yellow/cyan.
    _PLAN_COLOR = "#8f2d56"

    async def _cmd_plan(self, arg: str) -> str | None:
        """Per-turn plan mode: explore → write plan → user approves → implement.

        /plan          — activate plan mode and wait for the user to describe the task.
        /plan <task>   — activate plan mode and start planning immediately.

        Returns a str to hand to the normal turn machinery, or None if handled
        with no turn needed (banner already printed).
        """
        from rich.rule import Rule as _Rule
        from cca.tools.plan_tools import EnterPlanModeTool, is_plan_mode_active

        task = arg.strip()

        # Activate plan mode before the turn so hard enforcement is already on
        # when the agent's first token arrives.
        if not is_plan_mode_active():
            enter = EnterPlanModeTool()
            await enter.execute()

        if not task:
            # No task yet — show a vivid "you are now in plan mode" banner and wait.
            self._console.print()
            self._console.print(Panel(
                f"[bold {self._PLAN_COLOR}]Write tools are blocked[/bold {self._PLAN_COLOR}] "
                f"(Edit, Write, NotebookEdit) until you approve a plan.\n\n"
                "The agent will:\n"
                "  1. Explore the codebase\n"
                "  2. Write a plan for your review\n"
                "  3. Ask clarifying questions if needed\n"
                "  4. Implement only after you approve\n\n"
                "[dim]Type your task below and press Enter to begin.\n"
                f"Tip: [bold]/plan <task>[/bold] skips this step and starts immediately.[/dim]",
                title=f"[bold {self._PLAN_COLOR}]● PLAN MODE ACTIVE[/bold {self._PLAN_COLOR}]",
                border_style=self._PLAN_COLOR,
                padding=(1, 2),
            ))
            self._console.print()
            # Plan mode is now active; the next user message will have the plan
            # mode system prompt injected automatically via _build_system().
            return None

        # Task provided inline — show a slim activation banner then hand off to
        # the normal turn path (which will echo the user's /plan command first).
        self._console.print(_Rule(
            f"[bold {self._PLAN_COLOR}]⚡ PLAN MODE  ·  writes blocked until approval[/bold {self._PLAN_COLOR}]",
            style=self._PLAN_COLOR,
        ))

        # Task provided inline — return the planning message to the normal turn path.
        return (
            f"{task}\n\n"
            "<system-reminder>\n"
            "The user invoked /plan. You are already in plan mode "
            "(EnterPlanMode has been called). Write tools (Edit, Write, NotebookEdit) "
            "are hard-blocked until ExitPlanMode is approved.\n\n"
            "Your job now:\n"
            "1. Explore the codebase to understand the task.\n"
            "2. Write a complete implementation plan to the plan file.\n"
            "3. Call ExitPlanMode — the user will approve or reject.\n"
            "4. If approved, implement immediately in the same turn.\n"
            "You MAY use AskUserQuestion to clarify requirements before finalising "
            "the plan — the user is present and can answer.\n"
            "Do NOT attempt to edit files before ExitPlanMode is approved.\n"
            "</system-reminder>"
        )

    async def _cmd_think(self, arg: str) -> None:
        """/think [auto|off|low|medium|high] — show or set thinking policy."""
        level = arg.strip().lower()

        if not level:
            resolution = self._thinking_resolution
            color = "dim" if resolution.effective == "off" else "green"
            self._console.print(
                f"Thinking: requested [cyan]{resolution.requested}[/cyan] → "
                f"effective [{color}]{resolution.effective}[/{color}] "
                f"[dim]({resolution.source})[/dim]"
            )
            support, desc = _think_support(self.config.model)
            self._console.print(f"[dim]Model support: {desc}[/dim]")
            self._console.print(
                "[dim]Use /think auto|off|low|medium|high to change. "
                "'ultrathink' in a message forces high for that turn only.[/dim]"
            )
            return

        if level not in _THINK_LEVELS:
            self._console.print(
                f"[red]Unknown level '{level}'.[/red]  "
                f"Valid: {', '.join(_THINK_LEVELS)}"
            )
            return

        from cca.llm.thinking import resolve_thinking
        resolution = resolve_thinking(
            self.config.model,
            level,
            self.config.default_thinking_level,
        )
        self._thinking_resolution = resolution
        self._think_level = resolution.effective
        self._sync_agent_tool_settings()
        if self._ui:
            self._ui.update_think_level(self._think_level)
        color = "dim" if self._think_level == "off" else "green"
        self._console.print(
            f"Thinking: requested [cyan]{resolution.requested}[/cyan] → "
            f"effective [{color}]{resolution.effective}[/{color}] "
            f"[dim]({resolution.source})[/dim]"
        )
        if resolution.warning:
            self._console.print(f"[yellow]{resolution.warning}[/yellow]")
        if self._think_level != "off":
            self._console.print(
                "[dim]Applies to every turn until changed. "
                "'ultrathink' in a message overrides to high for that turn only.[/dim]"
            )

    async def _cmd_features(self) -> None:
        """/features — show resolved feature toggles and tool availability."""
        fs = self._features
        if fs.preset:
            self._console.print(f"Preset: [cyan]{fs.preset}[/cyan]")
        self._console.print("[bold]Features[/bold]")
        for name in sorted(fs.features):
            state = "[green]on[/green]" if fs.features[name] else "[red]off[/red]"
            self._console.print(f"  {name:<16} {state}  [dim]({fs.sources[f'features.{name}']})[/dim]")
        self._console.print("[bold]Tools[/bold]  [dim](core tools always on)[/dim]")
        for name in sorted(fs.tools):
            state = "[green]on[/green]" if fs.tools[name] else "[red]off[/red]"
            self._console.print(f"  {name:<16} {state}  [dim]({fs.sources[f'tools.{name}']})[/dim]")

    def _active_agent_tool(self):
        """Return this run's AgentTool, even when CLI filters hide it from the model."""
        if self.registry is None or not callable(getattr(self.registry, "get", None)):
            return None
        from cca.tools.agent_tool import AgentTool
        tool = self.registry.get("Agent")
        return tool if isinstance(tool, AgentTool) else None

    def _root_control_identity(self):
        """Build current root lineage for an operator control event."""
        from cca.trace_store import TraceIdentity, generation_from_session_id
        return TraceIdentity(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            generation=generation_from_session_id(self._session_id),
        )

    def _restore_subagent_default(self, state: dict | None = None) -> None:
        """Restore a session-local default from state or its durable control log."""
        agent_tool = self._active_agent_tool()
        if agent_tool is None:
            return
        requested = (state or {}).get("subagent_default")
        if not isinstance(requested, str):
            event = self._trace_store.latest_event("agent.default.changed")
            requested = event.get("new_default") if event else None
        if isinstance(requested, str) and self._agent_catalog_snapshot.get(requested):
            agent_tool.set_default_agent(requested)

    async def _cmd_agents(self, arg: str) -> None:
        """Trace-safe catalog and lifecycle controls for the frozen session catalog."""
        from rich.table import Table
        from cca.agents import resolve_agent_name
        from cca.tools.agent_tool import (
            get_background_agent,
            get_agent_run,
            list_agent_runs,
            stop_background_agent,
        )

        parts = arg.strip().split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "list"
        value = parts[1].strip() if len(parts) > 1 else ""
        if subcommand not in {
            "help", "list", "show", "default", "running", "output", "stop",
        }:
            self._console.print(
                "[red]Unknown /agents command.[/red] "
                "[dim]Use /agents help to see available commands.[/dim]"
            )
            return

        catalog = self._agent_catalog_snapshot
        agent_tool = self._active_agent_tool()
        current_default = agent_tool.default_agent if agent_tool is not None else None

        if subcommand == "help":
            from rich.markdown import Markdown as RichMarkdown
            guide = """\
# Using CCA subagents

The root agent can delegate bounded work to a specialized child. Children have
separate context and traces, cannot spawn more agents, and return their result to
the root for verification and synthesis.

## Discover and select profiles

- `/agents` — list the catalog frozen for this session.
- `/agents show <name>` — inspect a profile's prompt, source, tools, and limits.
- `/agents default <name>` — choose the profile used when delegation omits a name.
- Ask naturally for a profile when useful, for example: “Use the scout to map the
  authentication flow.” The root still decides when and how to delegate.

## Monitor background children

- `/agents running` — show children belonging to this root session.
- `/agents output <agent-id>` — read current or persisted output without adding it
  to model history.
- `/agents stop <agent-id>` — cancel a child still active in this process.

## Define and run profiles

- Project profiles: `.cca/agents/<name>.md`
- Global profiles: `~/.cca/agents/<name>.md`
- Root profile for a new session: `cca --agent <name>`
- Validate/import files: `cca agents validate` and `cca agents import`

The catalog does not reload in place; profile-file changes apply to a new session.
Root-profile switching and manual `/agents run` are intentionally not available yet.

## Turns and traces

Agents default to 40 model calls: 39 working turns plus one reserved, tool-free
handoff if the cap is reached. Child calls remain separate from root messages.
Read-only `/agents` commands create no model calls or SFT dialogue; default changes
and stop requests are stored only as control provenance.
"""
            self._console.print(Panel(
                RichMarkdown(guide),
                title="Subagent guide",
                border_style="cyan",
                padding=(1, 2),
            ))
            return

        if subcommand == "list":
            table = Table(title="Session agent catalog", box=None, pad_edge=False)
            table.add_column("Agent", style="cyan", no_wrap=True)
            table.add_column("Scope", no_wrap=True)
            table.add_column("Model", no_wrap=True)
            table.add_column("Max turns", justify="right", no_wrap=True)
            table.add_column("Purpose", overflow="fold")
            for name in catalog.names():
                spec = catalog.specs[name]
                markers = []
                if name == current_default:
                    markers.append("default")
                if self._agent_profile is not None and name == self._agent_profile.name:
                    markers.append("root")
                label = name + (f" ({', '.join(markers)})" if markers else "")
                table.add_row(
                    label,
                    spec.scope,
                    spec.model,
                    str(spec.max_turns),
                    spec.description,
                )
            self._console.print(table)
            state = "enabled" if self._agent_tool_visible() else "disabled for model use"
            self._console.print(
                f"[dim]{len(catalog.specs)} agents · {state} · "
                f"catalog {catalog.catalog_hash[:12]}[/dim]"
            )
            for warning in catalog.warnings:
                self._console.print(f"[yellow]Warning: {escape(warning)}[/yellow]")
            self._console.print(
                "[dim]Use /agents show <name> for details or /agents help for guidance.[/dim]"
            )
            return

        if subcommand == "show":
            if not value:
                self._console.print("[dim]Usage: /agents show <name>[/dim]")
                return
            spec = resolve_agent_name(catalog, value)
            if spec is None:
                self._console.print(f"[red]Unknown agent '{escape(value)}'.[/red]")
                return
            tools = "inherit" if spec.tools is None else ", ".join(sorted(spec.tools))
            denied = ", ".join(sorted(spec.disallowed_tools)) or "none"
            details = Table.grid(padding=(0, 2))
            details.add_column(style="dim", no_wrap=True)
            details.add_column(overflow="fold")
            details.add_row("Name", spec.name)
            details.add_row("Purpose", spec.description)
            details.add_row("Scope", spec.scope)
            details.add_row("Source", spec.source)
            details.add_row("Model", spec.model)
            details.add_row("Effort", spec.effort or "inherit")
            details.add_row("Max turns", str(spec.max_turns))
            details.add_row("Background", "yes" if spec.background else "no")
            details.add_row("Isolation", spec.isolation or "shared")
            details.add_row("Tools", tools)
            details.add_row("Denied", denied)
            details.add_row("Definition", spec.definition_hash)
            self._console.print(details)
            self._console.print(Panel(escape(spec.prompt), title="Exact prompt", border_style="dim"))
            return

        if subcommand == "default":
            if agent_tool is None or not self._agent_tool_visible():
                self._console.print(
                    "[yellow]Subagent dispatch is disabled for this session.[/yellow] "
                    "[dim]Start a new session with Agent enabled to change its default.[/dim]"
                )
                return
            if not value:
                self._console.print(
                    f"Current subagent default: [cyan]{agent_tool.default_agent}[/cyan]"
                )
                return
            spec = resolve_agent_name(catalog, value)
            if spec is None:
                self._console.print(f"[red]Unknown agent '{escape(value)}'.[/red]")
                return
            previous = agent_tool.default_agent
            if previous == spec.name:
                self._console.print(f"Subagent default is already [cyan]{spec.name}[/cyan].")
                return
            # The durable control event is written before runtime state changes.
            # A logging failure therefore cannot create an untraceable switch.
            try:
                event_id = self._trace_store.append_event(
                    "agent.default.changed",
                    self._root_control_identity(),
                    previous_default=previous,
                    new_default=spec.name,
                    catalog_hash=catalog.catalog_hash,
                    definition_hash=spec.definition_hash,
                    effective="next_dispatch",
                    origin="operator_command",
                )
            except OSError as exc:
                self._console.print(
                    f"[red]Could not record the default change; nothing changed: "
                    f"{escape(str(exc))}[/red]"
                )
                return
            agent_tool.set_default_agent(spec.name)
            self._console.print(
                f"Subagent default: [dim]{previous}[/dim] → [green]{spec.name}[/green] "
                f"[dim](session only, event {event_id[:8]})[/dim]"
            )
            return

        if subcommand == "running":
            runs = list_agent_runs(self._base_session_id)
            if not runs:
                self._console.print("[dim]No subagent runs in this session.[/dim]")
                return
            table = Table(title="Session subagent runs", box=None, pad_edge=False)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Agent", no_wrap=True)
            table.add_column("Status", no_wrap=True)
            table.add_column("Task", overflow="fold")
            for run in runs:
                table.add_row(
                    str(run.get("agent_id", "?")),
                    str(run.get("agent_name", "?")),
                    str(run.get("status", "unknown")),
                    str(run.get("description", "")),
                )
            self._console.print(table)
            return

        if not value:
            self._console.print(f"[dim]Usage: /agents {subcommand} <agent-id>[/dim]")
            return
        run = get_agent_run(value, root_id=self._base_session_id)
        if run is None:
            self._console.print(
                f"[red]Subagent '{escape(value)}' was not found in this session.[/red]"
            )
            return

        if subcommand == "output":
            import json
            result = run.get("result")
            payload = result if isinstance(result, dict) else {
                "agent_id": run.get("agent_id"),
                "agent_name": run.get("agent_name"),
                "description": run.get("description"),
                "status": run.get("status"),
            }
            self._console.print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
            return

        active = get_background_agent(value)
        if (
            active is None
            or active.get("done")
            or active.get("root_id") != self._base_session_id
        ):
            self._console.print(
                f"[yellow]Subagent {escape(value)} is not active in this process.[/yellow]"
            )
            return
        try:
            self._trace_store.append_event(
                "agent.stop.requested",
                self._root_control_identity(),
                target_agent_id=value,
                target_agent_name=run.get("agent_name"),
                previous_status=run.get("status"),
                origin="operator_command",
            )
        except OSError as exc:
            self._console.print(
                f"[red]Could not record the stop request; nothing changed: "
                f"{escape(str(exc))}[/red]"
            )
            return
        if not stop_background_agent(value, root_id=self._base_session_id):
            self._console.print(
                f"[yellow]Subagent {escape(value)} finished before cancellation.[/yellow]"
            )
            return
        self._console.print(f"[green]Cancellation requested for subagent {escape(value)}.[/green]")

    async def _cmd_skills(self, arg: str) -> None:
        """/skills [on|off] — show status/list, or toggle skills."""
        from cca.tools.skill_tool import list_skills

        if not self._features.tool_enabled("Skill"):
            self._console.print("[dim]Skills are disabled by feature config.[/dim]")
            return

        sub = arg.strip().lower()

        if sub in ("on", "off"):
            self.config.skills_enabled = sub == "on"
            self.config.save()
            # Re-register or deregister SkillTool without restart
            from cca.tools.skill_tool import SkillTool
            if self.config.skills_enabled:
                if not self.registry.get("Skill"):
                    self.registry.register(SkillTool())
                count = len(list_skills())
                self._console.print(f"Skills: [green]on[/green] [dim]({count} available)[/dim]")
            else:
                self.registry.deregister("Skill")
                self._console.print("Skills: [dim]off[/dim]")
            return

        # No arg — show status and list
        if not self.config.skills_enabled:
            self._console.print("Skills: [dim]off[/dim]  (use [bold]/skills on[/bold] to enable)")
            return

        skills = list_skills()
        if not skills:
            from cca.paths import SKILLS_DIR as _SKILLS_DIR
            self._console.print("Skills: [green]on[/green]  [dim]no skills installed[/dim]")
            self._console.print(f"[dim]Add skills to {_SKILLS_DIR}/<name>/SKILL.md or install via npx skills add[/dim]")
            return

        self._console.print(f"Skills: [green]on[/green]  ({len(skills)} available)\n")
        for s in skills:
            tag = " [dim][cca][/dim]" if s.source == "cca" else " [dim][claude][/dim]"
            desc = f"  [dim]{s.description}[/dim]" if s.description else ""
            self._console.print(f"  [cyan]{s.name}[/cyan]{tag}{desc}")

    async def _cmd_resume(self, arg: str) -> None:
        """Load a previous session's messages into the current conversation."""
        from cca.sessions import load_session_state, list_sessions

        if not arg:
            # Show a numbered list and prompt for a choice
            entries = list_sessions()
            if not entries:
                self._console.print("[dim]No saved sessions.[/dim]")
                return
            self._console.print("[bold]Saved sessions:[/bold]")
            for i, s in enumerate(entries, 1):
                self._console.print(
                    f"  [{i}] [cyan]{s.get('sessionId','?')}[/cyan]  {s.get('display','?')}  {s.get('project','?')}"
                )
            self._console.print("[dim]Use /resume <id> to load one.[/dim]")
            return

        sid = arg.strip()
        try:
            msgs, state = load_session_state(sid)
        except FileNotFoundError:
            self._console.print(f"[red]Session '{sid}' not found.[/red]")
            return

        self.messages = msgs
        self._session_id = sid
        self._base_session_id = re.sub(r'-comp\d+$', '', sid)
        self._trace_id = str(state.get("trace_id") or self._base_session_id)
        self._agent_id = str(state.get("agent_id") or self._base_session_id)
        from cca.trace_store import TraceIdentity, TraceStore, generation_from_session_id
        self._trace_store = TraceStore(self._base_session_id)
        saved_generation = state.get("context_generation")
        if not isinstance(saved_generation, int) or saved_generation < 0:
            saved_generation = generation_from_session_id(sid)
        self._context_generation = saved_generation
        saved_reason = state.get("generation_reason")
        self._generation_reason = (
            saved_reason
            if saved_reason in {"initial", "compaction", "clear"}
            else ("compaction" if saved_generation > 0 else "initial")
        )
        self._continuation_reason = "resume"
        identity = TraceIdentity(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            generation=self._context_generation,
            context_generation=self._context_generation,
            generation_reason=self._generation_reason,
        )
        self._root_trace_identity = identity
        self._llm_call_idx = self._trace_store.next_call_idx(identity, child=False)
        try:
            self._trace_store.append_root_event(
                "trace.run.resumed",
                identity,
                reason="resume",
                run_id=self._run_id,
                session_id=sid,
            )
        except Exception:
            pass
        summary = state.get("compaction_summary")
        self._compaction_summary = summary if isinstance(summary, str) else None
        offset = state.get("message_index_offset", 0)
        self._msg_index_offset = offset if isinstance(offset, int) and offset >= 0 else 0
        counts = state.get("token_counts")
        if isinstance(counts, dict):
            self._token_counts.update({
                key: int(value) for key, value in counts.items()
                if key in self._token_counts and isinstance(value, int)
            })
        self._post_compact_files = {}
        self._restore_subagent_default(state)
        if self._session_span is not None:
            self._session_span.finish(
                status="switched",
                attributes={"cca.session.switch_reason": "resume"},
            )
        self._observer.add_common_attributes({
            "cca.trace_id": self._trace_id,
            "cca.agent_id": self._agent_id,
            "cca.session_id": self._session_id,
            "cca.context_generation": self._context_generation,
        })
        self._session_span = self._observer.start(
            "cca.session",
            kind="session",
            attributes={
                "cca.session_id": self._session_id,
                "cca.continuation_reason": "resume",
                "cca.context_generation": self._context_generation,
            },
        )
        # Let AgentTool know the session ID so subagents link to us
        self._sync_agent_tool_settings()

        self._console.print(
            f"[green]Loaded session {sid}[/green] ({len(msgs)} messages). "
            "Conversation history restored."
        )

    async def _session_startup(self) -> None:
        """Shared startup for all run modes: SessionStart hooks, then MCP."""
        if self._session_span is None:
            self._session_span = self._observer.start(
                "cca.session",
                kind="session",
                attributes={
                    "cca.session_id": self._session_id,
                    "cca.continuation_reason": self._continuation_reason,
                    "cca.context_generation": self._context_generation,
                },
            )
        await self._hooks.session_start()
        await self._start_mcp()
        self._sync_agent_tool_settings()

    def _sync_agent_tool_settings(self) -> None:
        """Keep subagent provenance and thinking policy aligned with the parent."""
        if self.registry is None:
            return
        from cca.tools.agent_tool import AgentTool
        from cca.tools.job_tools import JobCancelTool, JobOutputTool
        tools = getattr(self.registry, "_tools", {})
        if not isinstance(tools, dict):
            return
        for tool in tools.values():
            if isinstance(tool, (JobOutputTool, JobCancelTool)):
                tool.bind_root(self._base_session_id)
            if isinstance(tool, AgentTool):
                tool.bind_parent(
                    session_id=self._session_id,
                    root_id=self._base_session_id,
                    trace_id=self._trace_id,
                    root_agent_id=self._agent_id,
                    thinking_request=self._thinking_resolution.requested,
                    backend=self.backend,
                    context_generation=self._context_generation,
                    permissions=self.permissions,
                    allowed_tools=self._allowed_tools,
                    disallowed_tools=self._disallowed_tools,
                    project_instructions=self._project_instructions,
                    cwd=self.cwd,
                    observer=self._observer,
                )

    async def _session_shutdown(self) -> None:
        """Shared shutdown: disconnect MCP servers, then SessionEnd hooks."""
        manager = getattr(self.registry, "_mcp_manager", None)
        if manager is not None:
            try:
                await manager.stop()
            except Exception:
                pass
        try:
            await self._hooks.session_end()
        except Exception:
            pass
        if self._session_span is not None:
            self._session_span.finish(
                status="ok",
                metrics={
                    "cca.tokens.input": self._token_counts["input"],
                    "cca.tokens.output": self._token_counts["output"],
                    "cca.observability.export_errors": self._observer.export_errors,
                },
            )
        try:
            timeout = self._observability_settings.flush_timeout
            await asyncio.wait_for(
                asyncio.to_thread(self._observer.shutdown, timeout),
                timeout=timeout + 0.5,
            )
        except Exception:
            pass

    async def _start_mcp(self) -> None:
        """Connect to trusted MCP servers and register dynamic tools."""
        manager = getattr(self.registry, "_mcp_manager", None)
        if manager is None:
            return
        manager.trust_all = manager.trust_all or self.permissions.auto_approve
        try:
            wrappers = await manager.start()
            for t in wrappers:
                self.registry.register(t)
            if wrappers:
                self._console.print(
                    f"[dim]MCP: connected {len(manager.server_names())} server(s), "
                    f"{len(wrappers)} tool(s) registered.[/dim]"
                )
            if manager.skipped_servers:
                names = ", ".join(manager.skipped_servers)
                self._console.print(
                    f"[yellow]MCP: skipped untrusted server(s): {names}.[/yellow] "
                    "[dim]Run interactively and approve the prompt, or use /mcp trust.[/dim]"
                )
        except Exception as e:
            self._console.print(f"[yellow]MCP startup warning:[/yellow] {e}")

    async def _mcp_trust_prompt(self, server_name: str, server_cfg: dict, config_path: str) -> bool:
        target = server_cfg.get("command") or server_cfg.get("url") or "?"
        args = " ".join(server_cfg.get("args", []))
        body = (
            f"[bold]{server_name}[/bold]  [dim]({config_path})[/dim]\n"
            f"  {target} {args}".rstrip()
            + "\n[dim]This launches an external process/connection with access to your environment.[/dim]"
        )
        return await self.permissions.confirm(f"Trust MCP server '{server_name}'?", body)

    async def _cmd_mcp(self, arg: str) -> None:
        """Handle the /mcp command."""
        manager = getattr(self.registry, "_mcp_manager", None)
        if manager is None:
            self._console.print("[dim]MCP manager not available.[/dim]")
            return

        parts = arg.split(maxsplit=1) if arg else []
        sub = parts[0].lower() if parts else "status"
        sub_arg = parts[1] if len(parts) > 1 else ""

        if sub in ("status", ""):
            servers = manager.server_names()
            if not servers:
                self._console.print("[dim]No MCP servers connected. Add servers to .mcp.json.[/dim]")
            else:
                self._console.print(f"Connected MCP servers: {', '.join(servers)}")
                tools = [t for t in self.registry._tools.values() if t.name.startswith("mcp__")]
                self._console.print(f"Registered MCP tools: {len(tools)}")

        elif sub == "list":
            # list tools, optionally for a specific server
            prefix = f"mcp__{sub_arg}__" if sub_arg else "mcp__"
            tools = [t for t in self.registry._tools.values() if t.name.startswith(prefix)]
            if not tools:
                self._console.print("[dim]No MCP tools found.[/dim]")
            else:
                for t in tools:
                    self._console.print(f"  {t.name}  —  {t.description}")

        elif sub == "resources":
            result = await manager.list_resources(sub_arg or None)
            self._console.print(result)

        elif sub == "trust":
            from cca.tools.mcp_tools import _locate_mcp_config, _read_mcp_config
            from cca.tools import mcp_trust
            config_path = _locate_mcp_config(self.cwd)
            if config_path is None:
                self._console.print("[dim]No .mcp.json found.[/dim]")
                return
            cfg = _read_mcp_config(config_path)
            targets = [sub_arg] if sub_arg else list(cfg)
            if not targets:
                self._console.print(f"[dim]No servers defined in {config_path}.[/dim]")
                return
            for name in targets:
                if name not in cfg:
                    self._console.print(f"[red]Unknown server '{name}' in {config_path}.[/red]")
                    continue
                mcp_trust.record_trust(str(config_path), name, cfg[name])
                self._console.print(f"[green]Trusted '{name}'[/green] [dim]({config_path})[/dim]")
            self._console.print("[dim]Run /mcp restart to connect.[/dim]")

        elif sub == "untrust":
            from cca.tools.mcp_tools import _locate_mcp_config
            from cca.tools import mcp_trust
            if not sub_arg:
                self._console.print("[dim]Usage: /mcp untrust <server>[/dim]")
                return
            config_path = _locate_mcp_config(self.cwd)
            if config_path is not None and mcp_trust.revoke_trust(str(config_path), sub_arg):
                self._console.print(
                    f"[yellow]Untrusted '{sub_arg}'.[/yellow] [dim]Run /mcp restart to disconnect.[/dim]")
            else:
                self._console.print(f"[dim]'{sub_arg}' was not trusted.[/dim]")

        elif sub == "restart":
            await manager.stop()
            try:
                wrappers = await manager.start()
                # Remove old dynamic tools and re-add
                old_keys = [k for k in list(self.registry._tools.keys()) if k.startswith("mcp__")]
                for k in old_keys:
                    self.registry._tools.pop(k, None)
                for t in wrappers:
                    self.registry.register(t)
                self._console.print(
                    f"[green]MCP restarted:[/green] {len(manager.server_names())} server(s), "
                    f"{len(wrappers)} tool(s)."
                )
            except Exception as e:
                self._console.print(f"[red]MCP restart failed:[/red] {e}")

        else:
            self._console.print(
                "[dim]Usage: /mcp [status|list [server]|resources [server]|trust [server]|untrust <server>|restart][/dim]"
            )

    async def _cmd_init(self) -> None:
        """Run the multi-phase AGENTS.md setup interview."""
        await self.process_turn(_INIT_PROMPT)

    async def _cmd_doctor(self) -> None:
        """Check configuration, API keys, and tool availability."""
        import shutil
        lines: list[str] = []
        ok = "[bold green]✓[/bold green]"
        warn = "[bold yellow]![/bold yellow]"
        err = "[bold red]✗[/bold red]"

        lines.append("[bold]cca doctor[/bold]\n")

        # Model config
        lines.append(f"  Model:  {self.config.model}")

        # API key check
        provider = self.config.model.split(":")[0] if ":" in self.config.model else "anthropic"
        key = self.config.get_api_key(provider)
        if key:
            lines.append(f"  {ok} API key for '{provider}': {key[:8]}…")
        else:
            lines.append(f"  {err} No API key for '{provider}' — set via /key or environment variable")

        # Config file
        from cca.config import CONFIG_FILE
        if CONFIG_FILE.exists():
            lines.append(f"  {ok} Config file: {CONFIG_FILE}")
        else:
            lines.append(f"  {warn} No config file (using defaults): {CONFIG_FILE}")

        # Hooks config
        if self.config.hooks:
            total = sum(len(v) for v in self.config.hooks.values() if isinstance(v, list))
            lines.append(f"  {ok} Hooks: {total} hook(s) configured")
        else:
            lines.append(f"  {warn} No hooks configured")

        # Tool checks
        for tool_name, exe in [("ripgrep (grep)", "rg"), ("git", "git"), ("python3", "python3")]:
            if shutil.which(exe):
                lines.append(f"  {ok} {tool_name}: {shutil.which(exe)}")
            else:
                lines.append(f"  {warn} {tool_name} not found in PATH")

        # Memory
        from cca.config import MEMORY_DIR
        mem_files = list(MEMORY_DIR.glob("*.md")) if MEMORY_DIR.exists() else []
        if mem_files:
            lines.append(f"  {ok} Memory: {len(mem_files)} file(s) in {MEMORY_DIR}")
        else:
            lines.append(f"  {warn} No memory files in {MEMORY_DIR}")

        # Session
        lines.append(f"\n  Session ID: {self._session_id}")
        lines.append(f"  CWD: {self.cwd}")
        lines.append(f"  Max tokens: {self.config.max_tokens:,}")
        lines.append(f"  Compact lower threshold: {getattr(self.config, 'compact_threshold', 100_000):,}")
        lines.append(f"  Compact upper threshold: {getattr(self.config, 'compact_threshold_upper', 200_000):,}")

        self._console.print("\n".join(lines))

    def _cmd_hooks(self) -> None:
        """List all configured hooks."""
        hooks = self.config.hooks
        if not hooks:
            self._console.print("[dim]No hooks configured.[/dim]")
            self._console.print(
                "\nAdd hooks to ~/.cca/config.yaml or .cca/config.yaml:\n"
                "  hooks:\n"
                "    PreToolUse:\n"
                "      - matcher: 'Bash'\n"
                "        command: 'my-logger.sh'\n"
                "    PostToolUse:\n"
                "      - command: 'audit.sh'\n"
                "        async: true\n"
            )
            return

        self._console.print("[bold]Configured hooks:[/bold]\n")
        for event, hook_list in hooks.items():
            self._console.print(f"  [bold cyan]{event}[/bold cyan]")
            for h in (hook_list or []):
                matcher = h.get("matcher", "*")
                cmd = h.get("command", "(no command)")
                timeout = h.get("timeout", 10)
                is_async = h.get("async", False)
                flags = []
                if is_async:
                    flags.append("async")
                flags_str = f"  [{', '.join(flags)}]" if flags else ""
                self._console.print(
                    f"    matcher=[cyan]{matcher}[/cyan]  "
                    f"timeout={timeout}s{flags_str}\n"
                    f"    cmd: [dim]{cmd}[/dim]"
                )
            self._console.print()

    def _print_help(self) -> None:
        cmd_width = max(len(cmd) for cmd, _ in SLASH_COMMANDS) + 2
        cmd_lines = "\n".join(
            f"  {cmd:<{cmd_width}}{desc}" for cmd, desc in SLASH_COMMANDS
        )
        self._console.print(f"""
[bold]cca — Code Agent CLI[/bold]

[bold]Slash commands:[/bold]
{cmd_lines}

[bold]Shell passthrough:[/bold]
  ! <command>    Run a shell command directly (e.g. ! git status)

[bold]Token budget (inline):[/bold]
  Use "use 500k tokens" or "+2m" in a message to set the budget for that turn.

[bold]Model format:[/bold]
  anthropic:claude-sonnet-4-6
  openai:gpt-4o
  groq:llama-3.3-70b-versatile
  ollama:llama3.2
  google:gemini-2.5-pro
""")

    def _print_config(self) -> None:
        repo_path = getattr(self.config, "_repo_config_path", None)
        if repo_path:
            self._console.print(f"  [dim]repo config:    {repo_path}[/dim]")
        self._console.print(f"  model:           {self.config.model}")
        self._console.print(f"  max_tokens:      {self.config.max_tokens}")
        self._console.print(f"  search_provider: {self.config.search_provider}")
        self._console.print(f"  verbose:         {self.config.verbose}")
        self._console.print(f"  auto_approve:    {self.permissions.auto_approve}")
        self._console.print(f"  always_allow:    {sorted(self.permissions.always_allow)}")

    # ------------------------------------------------------------------
    # Shell passthrough
    # ------------------------------------------------------------------

    async def _run_shell(self, command: str) -> None:
        import subprocess
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(command, shell=True, text=True,
                                   capture_output=True),
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
