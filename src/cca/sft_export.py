"""Validated, separated SFT views derived from exact CCA call traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class SFTValidationError(ValueError):
    """A raw trajectory is collectable but not structurally safe for SFT."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _clean_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    cleaned: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") == "thinking":
            continue
        item = dict(block)
        item.pop("thought_signature", None)
        item.pop("provider_state", None)
        cleaned.append(item)
    return cleaned


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": message.get("role"), "content": _clean_content(message.get("content"))}
        for message in messages
        if isinstance(message, dict)
    ]


def _terminal_assistant(response: dict[str, Any]) -> dict[str, Any]:
    calls = response.get("tool_calls") or []
    if calls:
        raise SFTValidationError("trajectory ends with unresolved assistant tool calls")
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SFTValidationError("trajectory has no terminal assistant text")
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _call_assistant(response: dict[str, Any]) -> dict[str, Any]:
    """Return a clean assistant target, including tool-use targets."""
    content: list[dict[str, Any]] = []
    text = response.get("text")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    for raw in response.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        tool_id = raw.get("id")
        name = raw.get("name")
        tool_input = raw.get("input")
        if not isinstance(tool_id, str) or not isinstance(name, str):
            raise SFTValidationError("assistant tool call is missing id or name")
        content.append({
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": tool_input if isinstance(tool_input, dict) else {},
        })
    if not content:
        raise SFTValidationError("model call has no assistant text or tool calls")
    return {"role": "assistant", "content": content}


def _validate_tool_pairing(messages: list[dict[str, Any]]) -> None:
    outstanding: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if message.get("role") == "assistant":
            for block in content:
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    outstanding.add(block["id"])
        elif message.get("role") == "user":
            for block in content:
                if block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if tool_id not in outstanding:
                        raise SFTValidationError(
                            f"orphan tool_result for '{tool_id}'"
                        )
                    outstanding.remove(tool_id)
    if outstanding:
        raise SFTValidationError(
            "missing tool_result for: " + ", ".join(sorted(outstanding))
        )


def _model_calls(calls: list[dict[str, Any]], actor: str) -> list[dict[str, Any]]:
    call_type = "main" if actor == "root" else "agent"
    return [record for record in calls if record.get("type") == call_type]


def _context_generation(record: dict[str, Any]) -> tuple[int, bool]:
    explicit = record.get("context_generation")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit, False
    # Before context_generation existed, a child's legacy generation stored the
    # parent generation at spawn. Children had no compaction, so their own
    # model-visible context always began at zero.
    if record.get("parent_agent_id") and record.get("type") in {
        "agent", "agent_finalizer",
    }:
        return 0, True
    legacy = record.get("generation")
    if isinstance(legacy, int) and legacy >= 0:
        return legacy, True
    session_id = record.get("session_id")
    if isinstance(session_id, str):
        from cca.trace_store import generation_from_session_id
        return generation_from_session_id(session_id), True
    return 0, True


def _generation_reason(record: dict[str, Any], generation: int) -> tuple[str, bool]:
    reason = record.get("generation_reason")
    if reason in {"initial", "compaction", "clear"}:
        return str(reason), False
    return ("compaction" if generation > 0 else "initial"), True


def _validated_indexes(model_calls: list[dict[str, Any]]) -> list[int]:
    indexes = [record.get("call_idx") for record in model_calls]
    if any(not isinstance(index, int) for index in indexes):
        raise SFTValidationError("call_idx is missing or invalid")
    if indexes != sorted(set(indexes)):
        raise SFTValidationError("call_idx values are not strictly increasing")
    return indexes  # type: ignore[return-value]


def _provenance(record: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
    generation, generation_inferred = _context_generation(record)
    reason, reason_inferred = _generation_reason(record, generation)
    return {
        "source_schema_version": record.get("schema_version", 1),
        "features": record.get("features"),
        "settings": record.get("settings"),
        "model_provenance": record.get("model_provenance"),
        "catalog_hash": record.get("catalog_hash") or record.get("agent_catalog_hash"),
        "agent_definition": record.get("agent_definition"),
        "call_indexes": indexes,
        "context_generation": generation,
        "generation_reason": reason,
        "generation_inferred": generation_inferred or reason_inferred,
        "run_id": record.get("run_id"),
        "continuation_reason": record.get("continuation_reason"),
        "context_from_previous_generation": reason == "compaction",
    }


def call_example_from_record(
    record: dict[str, Any],
    *,
    actor: str,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one lossless supervised example from one exact model call."""
    if record.get("type") not in ({"main"} if actor == "root" else {"agent"}):
        raise SFTValidationError(f"record is not a {actor} policy-model call")
    indexes = _validated_indexes([record])
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        raise SFTValidationError("model call has no message snapshot")
    messages = _clean_messages(raw_messages)
    # Inputs must be complete. The assistant target may itself request tools;
    # their results correctly belong to the next supervised example.
    _validate_tool_pairing(messages)
    messages.append(_call_assistant(record.get("response") or {}))
    system = record.get("system")
    if not isinstance(system, str):
        raise SFTValidationError("model call has no exact system prompt")
    generation, _ = _context_generation(record)
    reason, _ = _generation_reason(record, generation)
    example = {
        "schema_version": 1,
        "format": "call",
        "trace_id": record.get("trace_id"),
        "agent_id": record.get("agent_id"),
        "parent_agent_id": record.get("parent_agent_id"),
        "actor": actor,
        "context_generation": generation,
        "generation_reason": reason,
        "training_eligible": True,
        "quality": "unverified",
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": record.get("tools") or [],
        "provenance": _provenance(record, indexes),
    }
    if lineage:
        example["lineage"] = lineage
    return example


def trajectory_from_calls(
    calls: list[dict[str, Any]],
    *,
    actor: str,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_calls = _model_calls(calls, actor)
    if not model_calls:
        raise SFTValidationError(f"no {actor} model calls")
    indexes = _validated_indexes(model_calls)
    generations = {_context_generation(record)[0] for record in model_calls}
    if len(generations) != 1:
        raise SFTValidationError(
            "trajectory contains multiple context generations: "
            + ", ".join(str(item) for item in sorted(generations))
        )

    terminal = model_calls[-1]
    raw_messages = terminal.get("messages")
    if not isinstance(raw_messages, list):
        raise SFTValidationError("terminal call has no message snapshot")
    messages = _clean_messages(raw_messages)
    messages.append(_terminal_assistant(terminal.get("response") or {}))
    _validate_tool_pairing(messages)

    system = terminal.get("system")
    if not isinstance(system, str):
        raise SFTValidationError("terminal call has no exact system prompt")
    generation = next(iter(generations))
    reason, _ = _generation_reason(terminal, generation)
    trajectory = {
        "schema_version": 1,
        "format": "trajectory",
        "trace_id": terminal.get("trace_id"),
        "agent_id": terminal.get("agent_id"),
        "parent_agent_id": terminal.get("parent_agent_id"),
        "actor": actor,
        "context_generation": generation,
        "generation_reason": reason,
        "training_eligible": True,
        "quality": "unverified",
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": terminal.get("tools") or [],
        "provenance": _provenance(terminal, indexes),
    }
    if lineage:
        trajectory["lineage"] = lineage
    return trajectory


def trajectory_examples_from_calls(
    calls: list[dict[str, Any]],
    *,
    actor: str,
    lineage: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build safe per-generation trajectories and explicit exclusions."""
    model_calls = _model_calls(calls, actor)
    if not model_calls:
        raise SFTValidationError(f"no {actor} model calls")
    _validated_indexes(model_calls)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in model_calls:
        generation, _ = _context_generation(record)
        grouped.setdefault(generation, []).append(record)
    examples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for generation, records in grouped.items():
        try:
            examples.append(trajectory_from_calls(
                records, actor=actor, lineage=lineage
            ))
        except SFTValidationError as exc:
            exclusions.append({
                "context_generation": generation,
                "call_indexes": [
                    record.get("call_idx") for record in records
                    if isinstance(record.get("call_idx"), int)
                ],
                "reason": str(exc),
            })
    return examples, exclusions


def analyze_sft_calls(
    calls: list[dict[str, Any]],
    *,
    actor: str,
    lifecycle_status: str | None = None,
) -> dict[str, Any]:
    """Report complete call coverage separately from trajectory safety."""
    model_calls = _model_calls(calls, actor)
    raw_indexes = [
        record.get("call_idx") for record in model_calls
        if isinstance(record.get("call_idx"), int)
    ]
    if actor == "child" and lifecycle_status != "completed":
        return {
            "status": "excluded",
            "reason": f"child lifecycle status is {lifecycle_status or 'unknown'}",
            "raw_model_calls": len(model_calls),
            "call_examples_ready": 0,
            "call_coverage_complete": False,
            "context_generations": sorted({
                _context_generation(record)[0] for record in model_calls
            }),
            "trajectories_ready": 0,
            "trajectory_exclusions": [],
        }
    call_errors: list[dict[str, Any]] = []
    ready_indexes: list[int] = []
    try:
        _validated_indexes(model_calls)
    except SFTValidationError as exc:
        call_errors.append({"call_idx": None, "reason": str(exc)})
    for record in model_calls:
        try:
            call_example_from_record(record, actor=actor)
            index = record.get("call_idx")
            if isinstance(index, int):
                ready_indexes.append(index)
        except SFTValidationError as exc:
            call_errors.append({"call_idx": record.get("call_idx"), "reason": str(exc)})
    try:
        trajectories, exclusions = trajectory_examples_from_calls(calls, actor=actor)
    except SFTValidationError as exc:
        trajectories, exclusions = [], [{"context_generation": None, "reason": str(exc)}]
    coverage_complete = (
        bool(model_calls)
        and not call_errors
        and ready_indexes == raw_indexes
        and len(ready_indexes) == len(model_calls)
    )
    if not coverage_complete:
        status = "invalid"
        reason = call_errors[0]["reason"] if call_errors else "no exportable model calls"
    elif exclusions:
        status = "partial"
        reason = "call coverage complete; one or more trajectory generations are unsafe"
    else:
        status = "ready"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "raw_model_calls": len(model_calls),
        "call_examples_ready": len(ready_indexes),
        "call_coverage_complete": coverage_complete,
        "call_errors": call_errors,
        "context_generations": sorted({
            _context_generation(record)[0] for record in model_calls
        }),
        "trajectories_ready": len(trajectories),
        "trajectory_exclusions": exclusions,
    }


def export_trace(
    root_id: str,
    *,
    directory: Path | None = None,
    view: str = "root-only",
    format: str = "trajectories",
) -> list[dict[str, Any]]:
    if view not in {"root-only", "children-separated", "all-separated"}:
        raise ValueError(f"Unknown SFT view: {view}")
    if format not in {"calls", "trajectories"}:
        raise ValueError(f"Unknown SFT format: {format}")
    if directory is None:
        from cca.paths import LLM_CALLS_DIR
        directory = LLM_CALLS_DIR
    directory = Path(directory)
    examples: list[dict[str, Any]] = []
    event_path = directory / f"session_{root_id}_agents.jsonl"
    control_events: list[dict[str, Any]] = []
    if event_path.exists():
        control_events = [
            event for event in _read_jsonl(event_path)
            if event.get("event") in {
                "agent.default.changed",
                "agent.stop.requested",
            }
        ]

    if view in {"root-only", "all-separated"}:
        root_path = directory / f"session_{root_id}.jsonl"
        if not root_path.exists():
            raise FileNotFoundError(root_path)
        root_calls = _read_jsonl(root_path)
        if format == "calls":
            root_model_calls = _model_calls(root_calls, "root")
            if not root_model_calls:
                raise SFTValidationError("no root model calls")
            _validated_indexes(root_model_calls)
            root_examples = [
                call_example_from_record(record, actor="root")
                for record in root_model_calls
            ]
        else:
            root_examples, exclusions = trajectory_examples_from_calls(
                root_calls, actor="root"
            )
            if exclusions:
                first = exclusions[0]
                raise SFTValidationError(
                    "root context generation "
                    f"{first.get('context_generation')} is not trajectory-safe: "
                    f"{first.get('reason')}. Use --format calls for lossless export."
                )
        for root in root_examples:
            if control_events:
                # Operator controls are replay provenance, never invented dialogue.
                root["provenance"]["control_events"] = control_events
            examples.append(root)

    if view in {"children-separated", "all-separated"}:
        manifest_path = directory / f"session_{root_id}_agents.json"
        if not manifest_path.exists():
            return examples
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        agents = manifest.get("agents") or {}
        for agent_id, lineage in sorted(agents.items()):
            # Failed, cancelled, queued, or turn-capped executions remain in raw
            # storage but are rejected from clean SFT views by default.
            if lineage.get("status") != "completed":
                continue
            child_path = directory / f"session_{root_id}_sub_{agent_id}.jsonl"
            if not child_path.exists():
                continue
            child_calls = _read_jsonl(child_path)
            if format == "calls":
                child_model_calls = _model_calls(child_calls, "child")
                if not child_model_calls:
                    raise SFTValidationError(f"no child model calls for {agent_id}")
                _validated_indexes(child_model_calls)
                examples.extend(
                    call_example_from_record(record, actor="child", lineage=lineage)
                    for record in child_model_calls
                )
            else:
                child_examples, exclusions = trajectory_examples_from_calls(
                    child_calls, actor="child", lineage=lineage
                )
                if exclusions:
                    first = exclusions[0]
                    raise SFTValidationError(
                        f"child {agent_id} context generation "
                        f"{first.get('context_generation')} is not trajectory-safe: "
                        f"{first.get('reason')}. Use --format calls for lossless export."
                    )
                examples.extend(child_examples)
    return examples


def write_jsonl(examples: Iterable[dict[str, Any]], output: Path) -> int:
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count
