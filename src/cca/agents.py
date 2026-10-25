"""CCA-native subagent definitions and deterministic catalog discovery.

Agent definitions are Markdown files with YAML frontmatter.  CCA deliberately
discovers only its own ``.cca`` and ``~/.cca`` namespaces; integrations with
other agent formats belong at explicit import boundaries, never implicit
fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
DEFAULT_AGENT_MAX_TURNS = 40


class AgentDefinitionError(ValueError):
    """An agent definition is invalid and cannot safely be loaded."""


class SubagentConfigError(ValueError):
    """The subagents configuration cannot be enforced as written."""


def _tool_names(value: Any, field_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = [item.strip() for item in value if item.strip()]
    else:
        raise AgentDefinitionError(
            f"'{field_name}' must be a comma-separated string or a list of strings"
        )
    return frozenset(values)


@dataclass(frozen=True)
class AgentSpec:
    """One resolved, immutable subagent capability profile."""

    name: str
    description: str
    prompt: str
    tools: frozenset[str] | None = None
    disallowed_tools: frozenset[str] = frozenset()
    model: str = "inherit"
    effort: str | None = None
    max_turns: int = DEFAULT_AGENT_MAX_TURNS
    background: bool = False
    isolation: str | None = None
    source: str = "builtin"
    scope: str = "builtin"

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise AgentDefinitionError(
                f"Invalid agent name '{self.name}'; use lowercase kebab-case"
            )
        if not self.description.strip():
            raise AgentDefinitionError(f"Agent '{self.name}' requires a description")
        if not self.prompt.strip():
            raise AgentDefinitionError(f"Agent '{self.name}' requires a prompt body")
        if self.max_turns < 1:
            raise AgentDefinitionError(f"Agent '{self.name}' max_turns must be positive")
        if self.effort not in (None, "auto", "off", "low", "medium", "high"):
            raise AgentDefinitionError(
                f"Agent '{self.name}' effort must be auto, off, low, medium, or high"
            )
        if self.isolation not in (None, "worktree"):
            raise AgentDefinitionError(
                f"Agent '{self.name}' isolation must be 'worktree' when provided"
            )

    @property
    def definition_hash(self) -> str:
        payload = self.to_payload(include_provenance=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_payload(self, *, include_provenance: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "tools": sorted(self.tools) if self.tools is not None else None,
            "disallowed_tools": sorted(self.disallowed_tools),
            "model": self.model,
            "effort": self.effort,
            "max_turns": self.max_turns,
            "background": self.background,
            "isolation": self.isolation,
        }
        if include_provenance:
            payload.update({
                "source": self.source,
                "scope": self.scope,
                "definition_hash": self.definition_hash,
            })
        return payload


def _builtin_specs() -> tuple[AgentSpec, ...]:
    read_only = frozenset({"Read", "Glob", "Grep", "SearchInFile", "LSP"})
    read_web = read_only | {"WebSearch", "WebFetch"}
    verify = read_only | {"Bash"}
    no_mutation = frozenset({
        "Write", "Edit", "NotebookEdit", "Agent", "EnterWorktree",
        "ExitWorktree", "EnterPlanMode", "ExitPlanMode",
    })
    operational_denies = frozenset({
        "Agent", "EnterWorktree", "ExitWorktree", "EnterPlanMode",
        "ExitPlanMode", "CronCreate", "CronDelete", "Monitor",
        "ScheduleWakeup",
    })
    return (
        AgentSpec(
            name="worker",
            description=(
                "Handles a bounded implementation or repair task and returns changed "
                "files plus verification evidence."
            ),
            prompt=(
                "You are CCA's implementation worker. Complete only the bounded task "
                "the parent delegated. Your final message is the only thing the parent "
                "sees.\n"
                "- Read the relevant code before editing, follow its existing conventions, "
                "and leave unrelated code untouched.\n"
                "- Run the narrowest check that proves the change works.\n"
                "- If the task is blocked or ambiguous, stop and say what is missing "
                "instead of guessing.\n\n"
                "Finish with: files changed, checks run and their results, and anything "
                "left undone."
            ),
            disallowed_tools=operational_denies,
        ),
        AgentSpec(
            name="scout",
            description=(
                "Maps repository structure, traces dependencies, and gathers code evidence "
                "without changing the workspace."
            ),
            prompt=(
                "You are CCA's repository scout. Answer the delegated question from the "
                "code itself; you cannot modify the workspace. Your final message is the "
                "only thing the parent sees.\n"
                "- Match effort to the request: a quick lookup needs one or two searches; "
                "\"thorough\" means checking every plausible location and naming "
                "convention.\n"
                "- Run independent searches in parallel, then read the files that matter.\n\n"
                "Report findings with file:line references, separate what you confirmed "
                "from what you infer, and say what you looked for but did not find."
            ),
            tools=read_only,
            disallowed_tools=no_mutation,
        ),
        AgentSpec(
            name="blueprint",
            description=(
                "Produces a concrete architecture or implementation plan grounded in the "
                "current repository."
            ),
            prompt=(
                "You are CCA's design specialist. Read the relevant code, then write a "
                "plan someone can carry out without asking follow-up questions. Your "
                "final message is the only thing the parent sees. Structure it as:\n"
                "1. Approach: two or three sentences.\n"
                "2. Changes: each file or interface to touch, with file:line references.\n"
                "3. Steps: in dependency order.\n"
                "4. Risks and trade-offs, including alternatives you rejected and why.\n"
                "5. Verification: how to prove it works.\n\n"
                "Add a diagram only when the change has real structure, such as data "
                "flow or ordering."
            ),
            tools=read_web,
            disallowed_tools=no_mutation,
        ),
        AgentSpec(
            name="reviewer",
            description=(
                "Reviews a diff or selected code for high-confidence correctness and "
                "maintainability problems."
            ),
            prompt=(
                "You are CCA's independent code reviewer. Review only the requested "
                "change. Your final message is the only thing the parent sees.\n"
                "- Report only defects you can back with a concrete input or state that "
                "produces wrong behavior. Skip style preferences.\n"
                "- For each finding, give severity, file:line, impact, and a concrete fix.\n"
                "- If nothing meets that bar, say so plainly instead of padding the list."
            ),
            tools=verify,
            disallowed_tools=no_mutation,
        ),
        AgentSpec(
            name="verifier",
            description=(
                "Runs focused tests or builds, reproduces failures, and returns concise "
                "verification evidence without editing code."
            ),
            prompt=(
                "You are CCA's verification specialist. Your final message is the only "
                "thing the parent sees.\n"
                "- Run the smallest checks that answer the question, and report each "
                "exact command with its result.\n"
                "- Quote failing output verbatim instead of paraphrasing it.\n"
                "- Distinguish a new regression from a pre-existing failure or an "
                "environment problem (missing dependency, network, credentials). Never "
                "modify the workspace to find out."
            ),
            tools=verify,
            disallowed_tools=no_mutation,
        ),
        AgentSpec(
            name="risk-auditor",
            description=(
                "Examines security, privacy, dependency, and reliability risks using "
                "evidence-based read-only analysis."
            ),
            prompt=(
                "You are CCA's risk auditor. Analyze only the authorized scope. Your "
                "final message is the only thing the parent sees.\n"
                "- Look for exploitable security defects, privacy exposure, unsafe "
                "dependency behavior, and reliability hazards.\n"
                "- For each risk, give the evidence (file:line or command output), a "
                "realistic attack or failure path, its impact, and a mitigation.\n"
                "- Rank by real-world impact, and leave out theoretical issues with no "
                "plausible path to harm."
            ),
            tools=verify | {"WebSearch", "WebFetch"},
            disallowed_tools=no_mutation,
        ),
        AgentSpec(
            name="researcher",
            description=(
                "Researches current primary documentation and technical references without "
                "changing the workspace."
            ),
            prompt=(
                "You are CCA's technical researcher. Your final message is the only thing "
                "the parent sees.\n"
                "- Prefer primary sources: official docs, specifications, source code, "
                "release notes. Check which date or version each source applies to "
                "against today's date.\n"
                "- Separate documented facts from your own inference, and say when "
                "sources disagree.\n\n"
                "Return concise findings with a link for each claim."
            ),
            tools=read_web,
            disallowed_tools=no_mutation,
        ),
    )

_FRONTMATTER_KEYS = frozenset({
    "name", "description", "tools", "disallowedTools", "disallowed_tools",
    "model", "effort", "maxTurns", "max_turns", "background", "isolation",
})


def load_agent_file(path: Path, *, scope: str) -> tuple[AgentSpec, list[str]]:
    """Load one CCA agent Markdown file and return it plus non-fatal warnings."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentDefinitionError(f"Cannot read {path}: {exc}") from exc
    if not raw.startswith("---"):
        raise AgentDefinitionError(f"{path} must start with YAML frontmatter")
    parts = raw.split("---", 2)
    if len(parts) != 3:
        raise AgentDefinitionError(f"{path} has unterminated YAML frontmatter")
    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise AgentDefinitionError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise AgentDefinitionError(f"Frontmatter in {path} must be a mapping")

    unknown = sorted(set(metadata) - _FRONTMATTER_KEYS)
    warnings = [f"{path}: unsupported field '{key}' ignored" for key in unknown]
    name = metadata.get("name") or path.stem
    max_turns = metadata.get(
        "max_turns", metadata.get("maxTurns", DEFAULT_AGENT_MAX_TURNS)
    )
    if not isinstance(max_turns, int) or isinstance(max_turns, bool):
        raise AgentDefinitionError(f"{path}: max_turns must be an integer")
    background = metadata.get("background", False)
    if not isinstance(background, bool):
        raise AgentDefinitionError(f"{path}: background must be true or false")
    tools = _tool_names(metadata.get("tools"), "tools")
    denied = _tool_names(
        metadata.get("disallowed_tools", metadata.get("disallowedTools")),
        "disallowed_tools",
    ) or frozenset()
    return AgentSpec(
        name=str(name),
        description=str(metadata.get("description") or ""),
        prompt=parts[2].strip(),
        tools=tools,
        disallowed_tools=denied,
        model=str(metadata.get("model") or "inherit"),
        effort=(str(metadata["effort"]) if metadata.get("effort") is not None else None),
        max_turns=max_turns,
        background=background,
        isolation=metadata.get("isolation"),
        source=str(path.resolve()),
        scope=scope,
    ), warnings


@dataclass(frozen=True)
class AgentCatalog:
    specs: dict[str, AgentSpec]
    warnings: tuple[str, ...] = ()

    def get(self, name: str) -> AgentSpec | None:
        return self.specs.get(name)

    def names(self) -> list[str]:
        return sorted(self.specs)

    @property
    def catalog_hash(self) -> str:
        raw = json.dumps(
            {name: spec.definition_hash for name, spec in sorted(self.specs.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "catalog_hash": self.catalog_hash,
            "agents": [self.specs[name].to_payload() for name in self.names()],
        }


@dataclass(frozen=True)
class SubagentSettings:
    default: str = "worker"
    max_depth: int = 1
    max_concurrency: int = 3
    allow_background: bool = False


def resolve_subagent_settings(config: Any, catalog: AgentCatalog) -> SubagentSettings:
    raw = dict(getattr(config, "subagents", None) or {})
    valid = {"default", "max_depth", "max_concurrency", "allow_background"}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise SubagentConfigError(
            "Unknown subagents setting(s): " + ", ".join(unknown)
        )
    default = raw.get("default", "worker")
    if not isinstance(default, str) or resolve_agent_name(catalog, default) is None:
        raise SubagentConfigError(
            f"Unknown default subagent '{default}'. Available: {', '.join(catalog.names())}"
        )
    max_depth = raw.get("max_depth", 1)
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth not in (0, 1):
        raise SubagentConfigError("subagents.max_depth must be 0 or 1")
    max_concurrency = raw.get("max_concurrency", 3)
    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency < 1
    ):
        raise SubagentConfigError("subagents.max_concurrency must be a positive integer")
    allow_background = raw.get("allow_background", False)
    if not isinstance(allow_background, bool):
        raise SubagentConfigError("subagents.allow_background must be true or false")
    return SubagentSettings(
        default=resolve_agent_name(catalog, default).name,
        max_depth=max_depth,
        max_concurrency=max_concurrency,
        allow_background=allow_background,
    )


def _ancestor_chain(cwd: Path) -> Iterable[Path]:
    chain: list[Path] = []
    current = cwd.resolve()
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    return reversed(chain)


def discover_agents(
    cwd: str | Path,
    *,
    global_dir: Path | None = None,
) -> AgentCatalog:
    """Resolve built-ins, global definitions, then outer-to-inner project agents."""
    from cca.paths import CCA_DIR

    work_dir = Path(cwd).resolve()
    user_dir = (global_dir or (CCA_DIR / "agents")).resolve()
    resolved: dict[str, AgentSpec] = {spec.name: spec for spec in _builtin_specs()}
    warnings: list[str] = []

    sources: list[tuple[Path, str]] = [(user_dir, "global")]
    for directory in _ancestor_chain(work_dir):
        candidate = (directory / ".cca" / "agents").resolve()
        if candidate != user_dir:
            sources.append((candidate, "project"))

    for directory, scope in sources:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            try:
                spec, file_warnings = load_agent_file(path, scope=scope)
            except AgentDefinitionError as exc:
                warnings.append(str(exc))
                continue
            resolved[spec.name] = spec
            warnings.extend(file_warnings)

    return AgentCatalog(specs=resolved, warnings=tuple(warnings))


def resolve_agent_name(catalog: AgentCatalog, requested: str) -> AgentSpec | None:
    return catalog.get(requested)


def render_agent_file(spec: AgentSpec) -> str:
    """Render a normalized CCA agent definition for explicit imports."""
    metadata: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
    }
    if spec.tools is not None:
        metadata["tools"] = sorted(spec.tools)
    if spec.disallowed_tools:
        metadata["disallowed_tools"] = sorted(spec.disallowed_tools)
    if spec.model != "inherit":
        metadata["model"] = spec.model
    if spec.effort is not None:
        metadata["effort"] = spec.effort
    if spec.max_turns != DEFAULT_AGENT_MAX_TURNS:
        metadata["max_turns"] = spec.max_turns
    if spec.background:
        metadata["background"] = True
    if spec.isolation is not None:
        metadata["isolation"] = spec.isolation
    frontmatter = yaml.safe_dump(metadata, sort_keys=False).strip()
    return f"---\n{frontmatter}\n---\n\n{spec.prompt.rstrip()}\n"
