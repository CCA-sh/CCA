"""CLI entry point.

Usage:
  cca --version                          # show version and exit
  cca                                    # interactive REPL
  cca --model google:gemini-3.1-pro-preview  # one-session model override
  cca --print "explain this repo"        # non-interactive single prompt
  cca --auto-approve                     # never prompt for tool permission
  cca --continue                         # resume most recent session for cwd
  cca --resume <id>                      # resume a specific session by ID
  cca --output-format json --print "..." # JSON output (text/json/stream-json)
  cca --max-turns 5 --print "..."        # limit agentic turns
  cca --system-prompt "You are..."       # replace default system prompt
  cca --append-system-prompt "..."       # append to default system prompt
  cca --allowed-tools "Read,Bash"        # restrict tools for this run
  cca --disallowed-tools "Bash"          # block specific tools for this run
  cca --add-dir /other/path              # allow writes to extra directory
  cca --think high --print "..."         # enable thinking for this run
  cca sessions list                      # show saved sessions
  cca models anthropic                   # browse bundled model metadata
  cca skills add owner/repo@name         # install a skill from GitHub
  cca skills list                        # show installed skills

Configuring a model and API key (persists across restarts):
  cca config set model google:gemini-3.1-pro-preview
  cca config set-key google AIzaSy...

  Model string format:  provider:model-name
    anthropic:claude-sonnet-4-6
    openai:gpt-4o
    google:gemini-3.1-pro-preview
    groq:llama-3.3-70b-versatile
    ollama:llama3.2

  Common providers and their env-var fallbacks (run `cca models` for all):
    anthropic  → ANTHROPIC_API_KEY
    openai     → OPENAI_API_KEY
    google     → GOOGLE_API_KEY
    groq       → GROQ_API_KEY
    mistral    → MISTRAL_API_KEY
    openrouter → OPENROUTER_API_KEY
    xai        → XAI_API_KEY
    moonshot   → MOONSHOT_API_KEY / KIMI_API_KEY
    deepseek   → DEEPSEEK_API_KEY
    ollama     → (no key needed)

Repo-level config (overrides global for the current project):
  cca config set --repo model openai:gpt-4o
  cca config set-key --repo anthropic sk-...

Other config commands:
  cca config show   # show merged config (global + repo)
  cca config path   # show config file locations
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console

from cca import __version__

console = Console(stderr=True)


def _parse_tools_arg(raw: str | None) -> set[str]:
    """Parse a comma- or space-separated tool list into a set of names."""
    if not raw:
        return set()
    import re
    return set(t for t in re.split(r"[,\s]+", raw.strip()) if t)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="cca", message="%(prog)s %(version)s")
@click.option("--model", "-m", default=None, help="Model to use, e.g. anthropic:claude-sonnet-4-6")
@click.option("--agent", "root_agent_name", default=None, metavar="NAME",
              help="Run the root session with a named CCA agent profile.")
@click.option("--print", "print_prompt", default=None, metavar="PROMPT",
              help="Run a single prompt non-interactively and print output.")
@click.option("--auto-approve", is_flag=True, default=False,
              help="Automatically approve all tool calls without prompting.")
@click.option("--unsafe-auto-approve", is_flag=True, default=False,
              help="Alias for --auto-approve; skip every tool permission prompt.")
@click.option("--base-url", default=None, help="Base URL for OpenAI-compat endpoint.")
@click.option("--cwd", default=None, help="Working directory (default: current directory).")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show full tool outputs.")
@click.option("--trace", "-t", is_flag=True, default=False, help="Show all LLM inputs, outputs, and tool results with color coding.")
# Session resume
@click.option("--continue", "resume_latest", is_flag=True, default=False,
              help="Resume the most recent session for this working directory.")
@click.option("--resume", "-r", "resume_id", default=None, metavar="ID",
              help="Resume a session by ID. Use 'cca sessions list' to see IDs.")
# Output control
@click.option("--output-format", "output_format",
              type=click.Choice(["text", "json", "stream-json"]), default="text",
              help="Output format for --print mode: text (default), json, or stream-json.")
@click.option("--max-turns", "max_turns", default=None, type=int,
              help="Stop after N agentic turns (0 = unlimited).")
@click.option("--json-schema", "json_schema", default=None, metavar="SCHEMA",
              help="With --print: constrain the final answer to this JSON Schema "
                   "(inline JSON, or @path to a file). The reply is reformatted in "
                   "one extra tool-free call and validated before it is printed.")
# System prompt
@click.option("--system-prompt", "system_prompt", default=None, metavar="TEXT",
              help="Replace the system prompt entirely.")
@click.option("--append-system-prompt", "append_system_prompt", default=None, metavar="TEXT",
              help="Append TEXT to the default system prompt.")
# Tool filtering
@click.option("--allowed-tools", "allowed_tools", default=None, metavar="TOOLS",
              help="Comma/space-separated allowlist of tools for this invocation.")
@click.option("--disallowed-tools", "disallowed_tools", default=None, metavar="TOOLS",
              help="Comma/space-separated blocklist of tools for this invocation.")
# Directory access
@click.option("--add-dir", "add_dirs", multiple=True, metavar="DIR",
              help="Extra directory write tools may access (repeatable).")
# Thinking
@click.option("--think", "think_level",
              type=click.Choice(["auto", "off", "low", "medium", "high"], case_sensitive=False),
              default=None,
              help="Set thinking policy for this run: auto|off|low|medium|high. "
                   "CCA reports the resolved level and any model fallback.")
@click.option("--reasoning-summary", is_flag=True, default=False,
              help="Request and persist the provider's readable reasoning summary. "
                   "Requires an effective low|medium|high level and a supported model.")
@click.option("--preset", "preset_name", default=None, metavar="NAME",
              help="Apply a feature preset (e.g. 'deterministic'). Replaces the "
                   "config files' features:/tools: sections entirely.")
@click.option("--randomize", "randomize_axes", is_flag=False, flag_value="all",
              default=None, metavar="AXES",
              help="Randomize the scaffold surface for this run. Bare flag enables all "
                   "axes; or pass a comma list of system_prompt, tool_descriptions, "
                   "tool_order, compaction_prompt (or 'off').")
@click.option("--seed", "seed", type=int, default=None,
              help="Seed for --randomize, making the run reproducible. "
                   "Omitted: a seed is generated and reported on stderr.")
@click.option("--machine", "machine_mode", is_flag=True, default=False,
              help="Persistent JSON-lines machine mode for orchestrator integration. "
                   "Reads turn commands from stdin; writes events to stdout. "
                   "Full feature parity with interactive mode (MCP, hooks, growing context).")
@click.option("--show-settings", is_flag=True, default=False,
              help="Print the full resolved runtime and backend settings to stderr.")
@click.option("--observe", "observe_destination",
              type=click.Choice(["wandb", "otlp"], case_sensitive=False),
              default=None, metavar="DESTINATION",
              help="Opt in to lifecycle observability for this run: wandb or otlp.")
@click.option("--observe-capture", "observe_capture",
              type=click.Choice(["metadata", "content"], case_sensitive=False),
              default=None, metavar="LEVEL",
              help="Observability payload detail: metadata (default) records timings, "
                   "models and token counts; content also records prompts, completions, "
                   "and tool arguments and results.")
@click.pass_context
def main(
    ctx: click.Context,
    model: str | None,
    root_agent_name: str | None,
    print_prompt: str | None,
    auto_approve: bool,
    unsafe_auto_approve: bool,
    base_url: str | None,
    cwd: str | None,
    verbose: bool,
    trace: bool,
    resume_latest: bool,
    resume_id: str | None,
    output_format: str,
    json_schema: str | None,
    max_turns: int | None,
    system_prompt: str | None,
    append_system_prompt: str | None,
    allowed_tools: str | None,
    disallowed_tools: str | None,
    add_dirs: tuple,
    think_level: str | None,
    reasoning_summary: bool,
    preset_name: str | None = None,
    randomize_axes: str | None = None,
    seed: int | None = None,
    machine_mode: bool = False,
    show_settings: bool = False,
    observe_destination: str | None = None,
    observe_capture: str | None = None,
) -> None:
    """cca — Code Agent CLI with pluggable LLM backends.

    \b
    Quick setup:
      cca config set model google:gemini-3.1-pro-preview
      cca config set-key google YOUR_API_KEY

    \b
    Model string format:  provider:model-name
      anthropic:claude-sonnet-4-6  openai:gpt-4o
      google:gemini-3.1-pro-preview  groq:llama-3.3-70b-versatile

    Run 'cca config --help' for all configuration options.
    """
    if ctx.invoked_subcommand is not None:
        return

    from cca.config import Config, ensure_dirs
    from cca.llm.factory import create_backend
    from cca.tools import build_registry
    from cca.permissions import PermissionManager
    from cca.repl import Repl

    ensure_dirs()

    # Working directory — resolved before config load so repo config is found
    work_dir = cwd or os.getcwd()
    if cwd:
        try:
            os.chdir(cwd)
            work_dir = str(Path(cwd).resolve())
        except OSError as e:
            console.print(f"[red]Cannot change to directory {cwd}: {e}[/red]")
            sys.exit(1)

    # Load config: global merged with repo-level (.cca/config.yaml up the tree)
    cfg = Config.load_merged(work_dir)
    from cca.agents import (
        SubagentConfigError, discover_agents, resolve_agent_name,
        resolve_subagent_settings,
    )
    agent_catalog = discover_agents(work_dir)
    selected_agent_name = root_agent_name or cfg.agent
    root_agent_profile = None
    if selected_agent_name:
        root_agent_profile = resolve_agent_name(agent_catalog, selected_agent_name)
        if root_agent_profile is None:
            console.print(
                f"[red]Unknown agent '{selected_agent_name}'.[/red] Available: "
                + ", ".join(agent_catalog.names())
            )
            sys.exit(1)
        if system_prompt is not None:
            raise click.UsageError("--agent and --system-prompt cannot be used together")
        if model is None and root_agent_profile.model != "inherit":
            profile_model = root_agent_profile.model
            aliases = {
                "sonnet": "anthropic:claude-sonnet-4-6",
                "opus": "anthropic:claude-opus-4-6",
                "haiku": "anthropic:claude-haiku-4-5-20251001",
            }
            resolved_profile_model = aliases.get(profile_model, profile_model)
            if ":" not in resolved_profile_model:
                from cca.llm.catalog import split_selector
                provider, _ = split_selector(cfg.model)
                resolved_profile_model = f"{provider}:{resolved_profile_model}"
            cfg.model = resolved_profile_model
    if model:
        cfg.model = model
    if base_url:
        cfg.base_url = base_url
    if verbose:
        cfg.verbose = True
    if trace:
        cfg.trace = True

    # Resolve feature toggles once; every surface consumes the same FeatureSet
    from cca.features import resolve_features, FeatureConfigError
    preset_name = preset_name or os.environ.get("CCA_PRESET") or None
    try:
        feature_set = resolve_features(cfg, preset=preset_name)
    except FeatureConfigError as e:
        console.print(f"[red]Feature config error:[/red] {e}")
        sys.exit(1)
    if feature_set.tool_enabled("Agent"):
        try:
            resolve_subagent_settings(cfg, agent_catalog)
        except SubagentConfigError as exc:
            console.print(f"[red]Subagent config error:[/red] {exc}")
            sys.exit(1)

    # Resolve randomization once, alongside the FeatureSet. Absent the flag this is
    # the identity plan and touches no files.
    from cca.randomize import resolve_randomization, RandomizationError
    randomize_axes = randomize_axes or os.environ.get("CCA_RANDOMIZE") or None
    if seed is None and os.environ.get("CCA_SEED"):
        try:
            seed = int(os.environ["CCA_SEED"])
        except ValueError:
            console.print("[red]Randomization error:[/red] CCA_SEED must be an integer")
            sys.exit(1)
    try:
        randomization = resolve_randomization(randomize_axes, seed)
    except RandomizationError as e:
        console.print(f"[red]Randomization error:[/red] {e}")
        sys.exit(1)
    # Always report an auto-generated seed, on stderr so --machine's stdout protocol
    # stays pure: without it a user who hits an interesting run cannot reproduce it.
    if not randomization.is_identity() and randomization.seed_source == "auto":
        print(f"randomize: axes={','.join(sorted(randomization.axes))} "
              f"seed={randomization.seed}", file=sys.stderr)

    # Observability is resolved independently from behavior feature toggles: it
    # must not change the agent's action space or deterministic presets.
    from cca.observability import (
        ObservabilityConfigError,
        resolve_observability,
    )
    try:
        observability_settings = resolve_observability(
            cfg.observability,
            destination_override=observe_destination,
            capture_override=observe_capture,
        )
    except ObservabilityConfigError as exc:
        raise click.UsageError(str(exc)) from exc

    effective_auto_approve = (
        auto_approve or unsafe_auto_approve or feature_set.feature("auto_approve")
    )

    # Parse tool filter lists
    allowed_set = _parse_tools_arg(allowed_tools) or None  # None = no restriction
    disallowed_set = _parse_tools_arg(disallowed_tools)
    if root_agent_profile is not None:
        if root_agent_profile.tools is not None:
            profile_tools = set(root_agent_profile.tools)
            allowed_set = profile_tools if allowed_set is None else allowed_set & profile_tools
        disallowed_set |= set(root_agent_profile.disallowed_tools)

    # Add dirs
    add_dirs_set = {Path(d).resolve() for d in add_dirs} if add_dirs else set()

    # Resolve max_turns (0 means unlimited → pass a large number; None → default)
    effective_max_turns: int | None = None
    if max_turns is not None:
        effective_max_turns = None if max_turns == 0 else max_turns

    # Build backend
    from cca.llm.catalog import split_selector
    provider, _ = split_selector(cfg.model)
    api_key = cfg.get_api_key(provider)
    try:
        backend = create_backend(cfg.model, api_key=api_key, base_url=cfg.base_url)
    except Exception as e:
        console.print(f"[red]Failed to create LLM backend:[/red] {e}")
        sys.exit(1)

    from cca.llm.thinking import resolve_thinking
    try:
        thinking_resolution = resolve_thinking(
            cfg.model,
            think_level,
            cfg.default_thinking_level,
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    if reasoning_summary:
        if thinking_resolution.effective == "off":
            raise click.UsageError(
                "--reasoning-summary requires an effective low|medium|high thinking level"
            )
        if not backend.supports_reasoning_summary():
            raise click.UsageError(
                f"--reasoning-summary is not supported by {cfg.model}"
            )

    # Session resume
    resume_messages: list = []
    resume_state: dict = {}
    resume_sid: str | None = None
    if resume_latest:
        from cca.sessions import find_latest_for_cwd, load_session_state
        sid = find_latest_for_cwd(work_dir)
        if sid is None:
            console.print("[yellow]No previous session found for this directory.[/yellow]")
        else:
            try:
                resume_messages, resume_state = load_session_state(sid)
                resume_sid = sid
                console.print(
                    f"[dim]Resuming session {sid} ({len(resume_messages)} messages).[/dim]"
                )
            except Exception as e:
                console.print(f"[yellow]Could not load session {sid}: {e}[/yellow]")
    elif resume_id is not None:
        from cca.sessions import load_session_state
        try:
            resume_messages, resume_state = load_session_state(resume_id)
            resume_sid = resume_id
            console.print(
                f"[dim]Resuming session {resume_id} ({len(resume_messages)} messages).[/dim]"
            )
        except FileNotFoundError:
            console.print(f"[red]Session '{resume_id}' not found.[/red]")
            sys.exit(1)

    # Delay constructing the optional exporter until normal startup validation
    # has succeeded, so an invalid model or resume request cannot leave a
    # background batch processor behind.
    from cca.observability import create_observer
    try:
        observer = create_observer(observability_settings)
    except ObservabilityConfigError as exc:
        raise click.UsageError(str(exc)) from exc

    # Resolve --json-schema before anything expensive starts: a malformed
    # schema should fail immediately with a usage error, not after a full
    # agentic turn has already been paid for.
    parsed_json_schema = None
    if json_schema is not None:
        if print_prompt is None:
            raise click.UsageError("--json-schema requires --print.")
        if output_format == "stream-json":
            # stream-json prints deltas as they arrive and the caller reads
            # those, so the reformatted object would be produced and then
            # dropped. Refuse rather than appear to work.
            raise click.UsageError(
                "--json-schema cannot be combined with --output-format stream-json; "
                "use --output-format json (or text)."
            )
        source = json_schema
        if source.startswith("@"):
            schema_path = Path(source[1:]).expanduser()
            try:
                source = schema_path.read_text()
            except OSError as exc:
                raise click.UsageError(f"--json-schema: cannot read {schema_path}: {exc}") from exc
        try:
            parsed_json_schema = json.loads(source)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"--json-schema is not valid JSON: {exc}") from exc
        if not isinstance(parsed_json_schema, dict):
            raise click.UsageError("--json-schema must be a JSON object.")

    # Tell plan tools whether we're running non-interactively so ExitPlanMode
    # auto-approves instead of blocking on stdin.
    # Machine mode must be False so ExitPlanMode routes through plan_fn.
    from cca.tools.plan_tools import set_non_interactive
    is_non_interactive = (effective_auto_approve or print_prompt is not None) and not machine_mode
    set_non_interactive(is_non_interactive)

    # Shared Repl kwargs
    repl_kwargs = dict(
        resume_messages=resume_messages or None,
        session_id=resume_sid,
        resume_state=resume_state or None,
        output_format=output_format,
        output_schema=parsed_json_schema,
        max_turns=effective_max_turns,
        system_prompt=system_prompt,
        append_system_prompt=append_system_prompt,
        allowed_tools=allowed_set,
        disallowed_tools=disallowed_set,
        think_level=thinking_resolution.effective,
        thinking_resolution=thinking_resolution,
        reasoning_summary=reasoning_summary,
        show_settings=show_settings,
        feature_set=feature_set,
        randomization=randomization,
        agent_profile=root_agent_profile,
        observability_settings=observability_settings,
        observer=observer,
    )

    if print_prompt:
        # Non-interactive: no spinner, no Live display
        registry = build_registry(config=cfg, backend=backend, cwd=work_dir,
                                  feature_set=feature_set)
        summary = feature_set.summary_line()
        if summary:
            click.echo(summary, err=True)
        perms = PermissionManager(
            always_allow=set(cfg.always_allow),
            never_allow=set(cfg.never_allow) | disallowed_set,
            auto_approve=effective_auto_approve,
            cwd=work_dir,
            add_dirs=add_dirs_set,
        )
        repl = Repl(
            backend=backend,
            registry=registry,
            config=cfg,
            permissions=perms,
            cwd=work_dir,
            print_mode=True,
            run_mode="print",
            **repl_kwargs,
        )
        async def run_print():
            result = await repl.run_print(print_prompt)
            if output_format != "stream-json":
                print(result)

        from cca.llm.retry import LLMUnavailable
        from cca.structured_output import StructuredOutputError
        try:
            asyncio.run(run_print())
        except LLMUnavailable as exc:
            # Exit 3, distinct from 1 (a real failure) and 2 (schema): the
            # provider was busy and nothing is wrong with the request, so a
            # caller can retry this call rather than treating it as a bad turn.
            console.print(f"[red]LLM throttled:[/red] {exc}")
            sys.exit(3)
        except StructuredOutputError as exc:
            # Exit non-zero with the reason on stderr rather than printing
            # something that does not satisfy the schema the caller asked for.
            # An orchestrator can tell "no valid answer" from "here is one".
            console.print(f"[red]--json-schema:[/red] {exc}")
            sys.exit(2)
    elif machine_mode:
        # Machine mode: persistent process, JSON-lines stdin/stdout.
        # ask_fn routes AskUserQuestion through the protocol.
        # plan_fn routes ExitPlanMode through exit_plan_mode events (not ask_user).
        # _non_interactive is False so ExitPlanMode calls ask_fn rather than auto-approving.
        async def run_machine():
            repl = Repl(
                backend=backend,
                registry=None,
                config=cfg,
                permissions=None,
                cwd=work_dir,
                print_mode=False,
                machine_verbose=verbose,
                run_mode="machine",
                **repl_kwargs,
            )
            registry = build_registry(
                config=cfg,
                backend=backend,
                cwd=work_dir,
                ask_fn=repl._machine_ask_fn,
                plan_fn=repl._machine_plan_fn,
                feature_set=feature_set,
            )
            perms = PermissionManager(
                always_allow=set(cfg.always_allow),
                never_allow=set(cfg.never_allow) | disallowed_set,
                auto_approve=True,  # machine mode always auto-approves tool permissions
                cwd=work_dir,
                add_dirs=add_dirs_set,
            )
            repl.registry = registry
            repl.permissions = perms
            await repl.run_machine()

        asyncio.run(run_machine())
    else:
        # Interactive: wrap everything in TerminalUI
        from cca.ui import TerminalUI

        async def run_interactive():
            from cca.repl import SLASH_COMMANDS
            async with TerminalUI(model=cfg.model, cwd=work_dir, vim_mode=cfg.vim, slash_commands=SLASH_COMMANDS) as ui:
                registry = build_registry(config=cfg, backend=backend, cwd=work_dir,
                                          ask_fn=ui.ask_user, feature_set=feature_set)
                perms = PermissionManager(
                    always_allow=set(cfg.always_allow),
                    never_allow=set(cfg.never_allow) | disallowed_set,
                    auto_approve=effective_auto_approve,
                    console=ui.console,
                    permission_fn=ui.request_permission,
                    cwd=work_dir,
                    add_dirs=add_dirs_set,
                )
                repl = Repl(
                    backend=backend,
                    registry=registry,
                    config=cfg,
                    permissions=perms,
                    cwd=work_dir,
                    print_mode=False,
                    ui=ui,
                    run_mode="interactive",
                    **repl_kwargs,
                )
                await repl.run_interactive()

        asyncio.run(run_interactive())


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------

@main.group()
def config():
    """Manage cca configuration."""


@config.command("show")
def config_show():
    """Show current configuration (global merged with repo if present)."""
    from cca.config import Config, CONFIG_FILE
    import yaml
    cfg = Config.load_merged()
    repo_path = getattr(cfg, "_repo_config_path", None)
    console.print(f"[dim]Global:[/dim] {CONFIG_FILE}")
    if repo_path:
        console.print(f"[dim]Repo:  [/dim] {repo_path}")
    console.print(yaml.dump(cfg.to_dict(), default_flow_style=False))


def _coerce(existing: object, value: str) -> object:
    if isinstance(existing, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, list):
        return [v.strip() for v in value.split(",")]
    if isinstance(existing, dict):
        import yaml
        parsed = yaml.safe_load(value)
        if not isinstance(parsed, dict):
            raise click.UsageError("mapping configuration values require YAML/JSON object syntax")
        return parsed
    return value


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--repo", is_flag=True, default=False,
              help="Write to .cca/config.yaml in the current directory.")
def config_set(key: str, value: str, repo: bool):
    """Set a configuration value. E.g.: cca config set model openai:gpt-4o"""
    from cca.config import Config
    cfg = Config.load()
    if not hasattr(cfg, key):
        console.print(f"[red]Unknown config key: {key}[/red]")
        sys.exit(1)
    coerced = _coerce(getattr(cfg, key), value)
    if key == "default_thinking_level":
        from cca.llm.thinking import THINKING_LEVELS
        normalized = str(coerced).lower()
        if normalized not in THINKING_LEVELS:
            raise click.UsageError(
                f"default_thinking_level must be one of: {', '.join(THINKING_LEVELS)}"
            )
        coerced = normalized
    if repo:
        _set_repo_key(key, coerced)
        console.print(f"[green]Set {key} = {coerced}[/green] [dim](repo)[/dim]")
    else:
        setattr(cfg, key, coerced)
        cfg.save()
        console.print(f"[green]Set {key} = {coerced}[/green]")


@config.command("set-key")
@click.argument("provider")
@click.argument("api_key")
@click.option("--repo", is_flag=True, default=False,
              help="Write to .cca/config.yaml in the current directory.")
def config_set_key(provider: str, api_key: str, repo: bool):
    """Store an API key for a provider. E.g.: cca config set-key anthropic sk-..."""
    from cca.config import Config
    if repo:
        existing_keys = {}
        import yaml
        import pathlib
        repo_file = pathlib.Path(".cca") / "config.yaml"
        if repo_file.exists():
            with open(repo_file) as f:
                existing_keys = (yaml.safe_load(f) or {}).get("api_keys", {})
        existing_keys[provider.lower()] = api_key
        _set_repo_key("api_keys", existing_keys)
        console.print(f"[green]API key for '{provider}' saved.[/green] [dim](repo)[/dim]")
    else:
        cfg = Config.load()
        cfg.api_keys[provider.lower()] = api_key
        cfg.save()
        console.print(f"[green]API key for '{provider}' saved.[/green]")


def _set_repo_key(key: str, value: object) -> None:
    """Write a single key into .cca/config.yaml in the current directory."""
    import yaml
    import pathlib
    repo_dir = pathlib.Path(".cca")
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo_file = repo_dir / "config.yaml"
    data: dict = {}
    if repo_file.exists():
        with open(repo_file) as f:
            data = yaml.safe_load(f) or {}
    data[key] = value
    with open(repo_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


@config.command("path")
def config_path():
    """Print config file paths (global and repo if present)."""
    from cca.config import Config, CONFIG_FILE
    console.print(f"global: {CONFIG_FILE}")
    repo_path = Config.find_repo_config(Path.cwd())
    if repo_path:
        console.print(f"repo:   {repo_path}")


@main.command("models")
@click.argument("selector", required=False)
@click.option("--provider", "provider_filter", default=None, help="Restrict results to a provider.")
@click.option("--family", default=None, help="Restrict results to a model family, e.g. claude or qwen.")
@click.option("--reasoning", is_flag=True, default=False, help="Show only reasoning-capable models.")
@click.option("--limit", default=100, type=click.IntRange(min=1), show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit machine-readable JSON.")
def models_cmd(
    selector: str | None,
    provider_filter: str | None,
    family: str | None,
    reasoning: bool,
    limit: int,
    as_json: bool,
):
    """Browse bundled models or inspect PROVIDER:MODEL."""
    import dataclasses
    import json
    from rich.table import Table
    from cca.llm.catalog import (
        get_provider,
        list_models,
        list_providers,
        provider_model_count,
        resolve_model,
    )

    if selector and ":" in selector:
        model = resolve_model(selector)
        if model is None:
            raise click.ClickException(f"Model not found in bundled catalog: {selector}")
        payload = dataclasses.asdict(model)
        payload["selector"] = model.selector
        payload["family"] = model.family
        if as_json:
            click.echo(json.dumps(payload, indent=2))
            return
        console.print(f"[bold]{model.selector}[/bold]  {model.name}")
        console.print(f"API: {model.api}   Family: {model.family}   Input: {', '.join(model.input)}")
        console.print(
            f"Context: {model.context_window or 'unknown'}   Max output: {model.max_tokens or 'unknown'}   "
            f"Reasoning: {'yes' if model.reasoning else 'no'}"
        )
        if model.thinking:
            console.print(
                f"Thinking: {model.thinking.mode} ({', '.join(model.thinking.efforts)})"
            )
        return

    selected_provider = provider_filter
    query = None
    if selector:
        if get_provider(selector):
            selected_provider = selector
        else:
            query = selector

    if not selected_provider and not family and not query and not reasoning:
        providers = list_providers()
        if as_json:
            payload = [
                {**dataclasses.asdict(provider), "model_count": provider_model_count(provider.id)}
                for provider in providers
            ]
            click.echo(json.dumps(payload, indent=2))
            return
        table = Table(title="Bundled model providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Name")
        table.add_column("Models", justify="right")
        table.add_column("Default model")
        for provider in providers:
            table.add_row(
                provider.id,
                provider.name,
                str(provider_model_count(provider.id)),
                provider.default_model or "dynamic",
            )
        console.print(table)
        console.print("[dim]Use `cca models <provider>`, `cca models <query>`, or `cca models provider:model`.[/dim]")
        return

    models = list_models(selected_provider, family=family, query=query)
    if reasoning:
        models = [model for model in models if model.reasoning]
    total = len(models)
    models = models[:limit]
    if as_json:
        payload = []
        for model in models:
            item = dataclasses.asdict(model)
            item["selector"] = model.selector
            item["family"] = model.family
            payload.append(item)
        click.echo(json.dumps(payload, indent=2))
        return
    if not models:
        if selected_provider == "litellm":
            raise click.ClickException(
                "LiteLLM routes come from your proxy and are not bundled; list them with "
                'curl -H "Authorization: Bearer $LITE_LLM_KEY" "$LITE_LLM_URL/models"'
            )
        raise click.ClickException("No bundled models matched")
    table = Table(title=f"Bundled models ({total})")
    table.add_column("Selector", style="cyan", overflow="fold")
    table.add_column("Name", overflow="fold")
    table.add_column("Context", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Thinking")
    for model in models:
        thinking = ",".join(model.thinking.efforts) if model.thinking else "-"
        table.add_row(
            model.selector,
            model.name,
            f"{model.context_window:,}" if model.context_window else "?",
            f"{model.max_tokens:,}" if model.max_tokens else "?",
            thinking,
        )
    console.print(table)
    if total > len(models):
        console.print(f"[dim]Showing {len(models)} of {total}; increase --limit to see more.[/dim]")


@main.command("features")
@click.option("--preset", "preset_name", default=None, metavar="NAME",
              help="Show the resolved state under this preset.")
@click.option("--cwd", default=None, help="Resolve repo config from this directory.")
def features_cmd(preset_name: str | None, cwd: str | None):
    """Show resolved feature toggles, tool availability, and their sources."""
    from cca.config import Config
    from cca.features import resolve_features, FeatureConfigError

    cfg = Config.load_merged(cwd or os.getcwd())
    preset_name = preset_name or os.environ.get("CCA_PRESET") or None
    try:
        fs = resolve_features(cfg, preset=preset_name)
    except FeatureConfigError as e:
        console.print(f"[red]Feature config error:[/red] {e}")
        sys.exit(1)

    console.print("[bold]Features[/bold]")
    for name in sorted(fs.features):
        state = "[green]on[/green]" if fs.features[name] else "[red]off[/red]"
        console.print(f"  {name:<16} {state}  [dim]({fs.sources[f'features.{name}']})[/dim]")

    console.print("\n[bold]Tools[/bold]  [dim](core tools always on: Read, Write, Edit, Glob, Grep, SearchInFile, Bash)[/dim]")
    for name in sorted(fs.tools):
        state = "[green]on[/green]" if fs.tools[name] else "[red]off[/red]"
        console.print(f"  {name:<16} {state}  [dim]({fs.sources[f'tools.{name}']})[/dim]")


# ---------------------------------------------------------------------------
# Randomization variant subcommands
# ---------------------------------------------------------------------------

@main.group()
def variants():
    """Inspect and verify the randomization variant bank."""


@variants.command("list")
@click.option("--axis", default=None, help="Restrict to one axis.")
def variants_list(axis: str | None):
    """List available variants per axis."""
    from cca import variants as bank

    for name in ([axis] if axis else bank.CONTENT_AXES):
        try:
            found = bank.load_axis(name)
        except bank.VariantError as e:
            console.print(f"[red]Variant error:[/red] {e}")
            sys.exit(1)
        console.print(f"[bold]{name}[/bold]")
        for variant in found:
            console.print(f"  {variant.id:<18} [dim]({variant.source})[/dim] {variant.summary}")


@variants.command("show")
@click.argument("axis")
@click.argument("variant_id")
@click.option("--seed", type=int, default=None, help="Shuffle sections with this seed.")
def variants_show(axis: str, variant_id: str, seed: int | None):
    """Print the fully rendered text of one variant."""
    import random
    from cca import variants as bank

    try:
        variant = bank.get(axis, variant_id)
        if axis == "tool_descriptions":
            for tool, override in bank.tool_overrides(variant).items():
                console.print(f"[bold]{tool}[/bold]: {override['description'] or '(built-in)'}")
            return
        text, _ = bank.render_sectioned(variant, random.Random(seed) if seed is not None else None)
    except bank.VariantError as e:
        console.print(f"[red]Variant error:[/red] {e}")
        sys.exit(1)
    click.echo(text)


@variants.command("verify")
@click.option("--axis", default=None, help="Restrict to one axis.")
def variants_verify(axis: str | None):
    """Check every variant against the manifest contracts. Exits 1 on failure."""
    from cca.variant_checks import check_bank

    failures = check_bank(axis)
    for message in failures:
        console.print(f"[red]FAIL[/red] {message}")
    if failures:
        console.print(f"\n[red]{len(failures)} problem(s) found.[/red]")
        sys.exit(1)
    console.print("[green]All variants pass.[/green]")


@variants.command("resolve")
@click.option("--seed", type=int, required=True, help="Seed to resolve.")
@click.option("--randomize", "axes_spec", default="all", help="Axis spec to resolve under.")
def variants_resolve(seed: int, axes_spec: str):
    """Show which variants a seed selects, without invoking the model."""
    from cca.randomize import resolve_randomization, RandomizationError

    try:
        plan = resolve_randomization(axes_spec, seed)
    except RandomizationError as e:
        console.print(f"[red]Randomization error:[/red] {e}")
        sys.exit(1)
    if plan.is_identity():
        console.print("[dim]No axes enabled — identity plan.[/dim]")
        return
    console.print(f"[bold]seed[/bold] {plan.seed}   [bold]axes[/bold] {', '.join(sorted(plan.axes))}")
    for axis, variant_id in sorted(plan.variant_ids.items()):
        console.print(f"  {axis:<20} {variant_id}")
    if plan.section_order:
        console.print(f"  {'section order':<20} {' → '.join(plan.section_order)}")


# ---------------------------------------------------------------------------
# Memory subcommands
# ---------------------------------------------------------------------------

@main.group()
def memory():
    """Manage persistent memories."""


@memory.command("list")
def memory_list():
    """List all stored memories."""
    from cca.memory import list_memories
    mems = list_memories()
    if not mems:
        console.print("[dim]No memories stored.[/dim]")
    else:
        for m in mems:
            console.print(f"  [{m.get('type','?')}] {m.get('name','?')} ({m.get('_file','')})")
            console.print(f"         {m.get('description','')}")


@memory.command("delete")
@click.argument("filename")
def memory_delete(filename: str):
    """Delete a memory by filename (without .md extension)."""
    from cca.memory import delete_memory
    if delete_memory(filename):
        console.print(f"[green]Deleted memory: {filename}[/green]")
    else:
        console.print(f"[red]Memory not found: {filename}[/red]")


# ---------------------------------------------------------------------------
# Sessions subcommands
# ---------------------------------------------------------------------------

@main.group()
def sessions():
    """List and manage saved sessions."""


@sessions.command("list")
@click.option("--limit", default=20, help="Max sessions to show.")
def sessions_list(limit: int):
    """List saved sessions, most recent first."""
    from cca.sessions import list_sessions
    from datetime import datetime
    entries = list_sessions(limit=limit)
    if not entries:
        console.print("[dim]No saved sessions.[/dim]")
        return
        
    console.print(f"{'ID':<40} {'Date/Time':<20} {'Description'}")
    console.print("-" * 85)
    for s in entries:
        sid = s.get('sessionId', '?')
        ts = s.get('timestamp', 0)
        dt = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S") if ts else '?'
        display = s.get('display', 'New Session')
        console.print(f"  [cyan]{sid:<38}[/cyan]  {dt:<20}  {display}")


@sessions.command("delete")
@click.argument("session_id")
def sessions_delete(session_id: str):
    """Delete a saved session by ID."""
    from cca.sessions import SESSIONS_DIR, HISTORY_FILE
    import json

    msg_file = SESSIONS_DIR / f"{session_id}.json"

    deleted = False
    if msg_file.exists():
        msg_file.unlink()
        deleted = True

    if HISTORY_FILE.exists():
        lines = []
        with open(HISTORY_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("sessionId") != session_id:
                        lines.append(line)
                    else:
                        deleted = True
                except Exception:
                    lines.append(line)
        with open(HISTORY_FILE, "w") as f:
            f.writelines(lines)
            
    if deleted:
        console.print(f"[green]Deleted session {session_id}.[/green]")
    else:
        console.print(f"[red]Session not found: {session_id}[/red]")


# ---------------------------------------------------------------------------
# Agent catalog subcommands
# ---------------------------------------------------------------------------

@main.group()
def agents():
    """Inspect and explicitly import CCA agent definitions."""


@agents.command("list")
@click.option("--cwd", default=None, help="Resolve project agents from this directory.")
def agents_list(cwd: str | None):
    """List the resolved native agent catalog and provenance."""
    from cca.agents import discover_agents
    catalog = discover_agents(cwd or os.getcwd())
    for name in catalog.names():
        spec = catalog.specs[name]
        console.print(
            f"[cyan]{name:<18}[/cyan] {spec.scope:<8} "
            f"[dim]{spec.definition_hash[:12]}[/dim]  {spec.description}"
        )
    for warning in catalog.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"[dim]catalog {catalog.catalog_hash}[/dim]")


@agents.command("validate")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def agents_validate(path: Path):
    """Validate one portable Markdown agent definition."""
    from cca.agents import AgentDefinitionError, load_agent_file
    try:
        spec, warnings = load_agent_file(path, scope="import")
    except AgentDefinitionError as exc:
        raise click.ClickException(str(exc)) from exc
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(
        f"[green]Valid agent '{spec.name}'[/green] "
        f"[dim]{spec.definition_hash}[/dim]"
    )


@agents.command("import")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--scope", type=click.Choice(["project", "global"]), default="project")
@click.option("--cwd", default=None, help="Project destination root.")
@click.option("--force", is_flag=True, default=False, help="Replace an existing definition.")
def agents_import(source: Path, scope: str, cwd: str | None, force: bool):
    """Explicitly normalize SOURCE into CCA's native agent registry."""
    from cca.agents import AgentDefinitionError, load_agent_file, render_agent_file
    from cca.paths import CCA_DIR
    try:
        spec, warnings = load_agent_file(source, scope="import")
    except AgentDefinitionError as exc:
        raise click.ClickException(str(exc)) from exc
    destination_dir = (
        CCA_DIR / "agents"
        if scope == "global"
        else Path(cwd or os.getcwd()).resolve() / ".cca" / "agents"
    )
    destination = destination_dir / f"{spec.name}.md"
    if destination.exists() and not force:
        raise click.ClickException(
            f"{destination} already exists; pass --force to replace it"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_agent_file(spec), encoding="utf-8")
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"[green]Imported '{spec.name}' to {destination}.[/green]")


# ---------------------------------------------------------------------------
# Trace export subcommands
# ---------------------------------------------------------------------------

@main.group()
def traces():
    """Validate and export exact root/subagent trajectories."""


@traces.command("inspect")
@click.argument("root_id")
@click.option("--calls-dir", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit the summary as JSON.")
def traces_inspect(root_id: str, calls_dir: Path | None, as_json: bool):
    """Visualize lineage, lifecycle, calls, and SFT readiness for ROOT_ID."""
    import json
    from rich.markup import escape
    from rich.tree import Tree
    from cca.trace_view import inspect_trace

    try:
        summary = inspect_trace(root_id, directory=calls_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return

    overall = (
        "[green]clean export ready[/green]"
        if summary["clean_export_ready"]
        else "[red]clean export not ready[/red]"
    )
    tree = Tree(f"[bold]Trace {escape(root_id)}[/bold] · {overall}")
    root = summary["root"]
    root_sft = root["sft"]
    root_style = {
        "ready": "green", "partial": "yellow", "excluded": "yellow",
        "invalid": "red",
    }.get(root_sft["status"], "white")
    root_node = tree.add(
        f"[bold]Root[/bold] {escape(str(root['agent_id']))} · "
        f"{root['calls']} records · SFT [{root_style}]{root_sft['status']}[/{root_style}]"
    )
    root_node.add(
        "types " + escape(
            ", ".join(f"{name}={count}" for name, count in root["call_types"].items())
            or "none"
        )
    )
    root_node.add(
        f"call coverage {root_sft.get('call_examples_ready', 0)}/"
        f"{root_sft.get('raw_model_calls', 0)} · "
        f"generations={len(root_sft.get('context_generations', []))} · "
        f"safe trajectories={root_sft.get('trajectories_ready', 0)}"
    )
    if root_sft.get("reason"):
        root_node.add(f"[red]{escape(str(root_sft['reason']))}[/red]")
    for event in root.get("context_events", []):
        mode = f"/{event['compaction_mode']}" if event.get("compaction_mode") else ""
        root_node.add(
            "context "
            f"{escape(str(event.get('previous_context_generation', '?')))}→"
            f"{escape(str(event.get('context_generation', '?')))} · "
            f"{escape(str(event.get('reason') or '?'))}{escape(mode)}"
        )
    for error in root.get("parse_errors", []):
        root_node.add(f"[red]{escape(error)}[/red]")

    children_node = tree.add(f"[bold]Children[/bold] ({len(summary['children'])})")
    for child in summary["children"]:
        sft = child["sft"]
        sft_style = {
            "ready": "green", "partial": "yellow", "excluded": "yellow",
            "invalid": "red",
        }.get(sft["status"], "white")
        label = (
            f"[cyan]{escape(child['agent_id'])}[/cyan] "
            f"{escape(str(child.get('agent_name') or '?'))} · "
            f"status={escape(child['status'])} · {child['calls']} records · "
            f"SFT [{sft_style}]{sft['status']}[/{sft_style}]"
        )
        child_node = children_node.add(label)
        if child.get("description"):
            child_node.add(escape(str(child["description"])))
        child_node.add(
            f"parent={escape(str(child.get('parent_agent_id') or '?'))} · "
            f"spawn={escape(str(child.get('spawn_call_id') or '?'))}"
        )
        if child.get("spawn_group_id") is not None:
            child_node.add(
                f"group={escape(str(child['spawn_group_id']))} · "
                f"index={escape(str(child.get('spawn_index', '?')))}"
            )
        child_node.add(
            f"definition={escape(str(child.get('definition_hash') or '?'))[:12]} · "
            f"catalog={escape(str(child.get('catalog_hash') or '?'))[:12]}"
        )
        if child.get("summary"):
            text = str(child["summary"])
            child_node.add("result: " + escape(text[:240] + ("…" if len(text) > 240 else "")))
        if sft.get("reason"):
            child_node.add(f"SFT reason: {escape(str(sft['reason']))}")
        child_node.add(
            f"call coverage {sft.get('call_examples_ready', 0)}/"
            f"{sft.get('raw_model_calls', 0)} · "
            f"generations={len(sft.get('context_generations', []))} · "
            f"safe trajectories={sft.get('trajectories_ready', 0)}"
        )
        for issue in child.get("reproducibility_issues", []):
            child_node.add(f"[red]replay issue: {escape(issue)}[/red]")

    events_node = tree.add(f"[bold]Events[/bold] ({len(summary['events'])})")
    for event in summary["events"]:
        target = event.get("target_agent_id") or event.get("agent_id") or "?"
        origin = f" · {event['origin']}" if event.get("origin") else ""
        events_node.add(
            f"{escape(str(event.get('event') or '?'))} · "
            f"agent={escape(str(target))}{escape(origin)}"
        )
    for error in summary.get("event_parse_errors", []):
        events_node.add(f"[red]{escape(error)}[/red]")
    for error in summary.get("manifest_errors", []):
        tree.add(f"[red]{escape(error)}[/red]")
    for orphan in summary.get("orphan_child_logs", []):
        tree.add(f"[red]Orphan child log: {escape(orphan)}[/red]")
    console.print(tree)


@traces.command("export")
@click.argument("root_id")
@click.option(
    "--view",
    type=click.Choice(["root-only", "children-separated", "all-separated"]),
    default="root-only",
    show_default=True,
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["calls", "trajectories"]),
    default="trajectories",
    show_default=True,
    help="Export every exact model call or safe per-generation trajectories.",
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--calls-dir", type=click.Path(path_type=Path), default=None)
def traces_export(
    root_id: str,
    view: str,
    export_format: str,
    output: Path,
    calls_dir: Path | None,
):
    """Export structurally validated SFT JSONL for ROOT_ID."""
    from cca.sft_export import SFTValidationError, export_trace, write_jsonl
    try:
        examples = export_trace(
            root_id, directory=calls_dir, view=view, format=export_format
        )
        count = write_jsonl(examples, output)
    except (FileNotFoundError, SFTValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    noun = "call example" if export_format == "calls" else "trajectory"
    console.print(f"[green]Exported {count} {noun}(s) to {output}.[/green]")


from cca.skills_cli import skills as _skills_group
main.add_command(_skills_group, name="skills")


if __name__ == "__main__":
    main()
