"""Read-only summaries for inspecting root/subagent trace trees."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from cca.sft_export import analyze_sft_calls


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path.name}:{line_number}: {exc}")
                    continue
                if isinstance(value, dict):
                    records.append(value)
                else:
                    errors.append(f"{path.name}:{line_number}: record is not an object")
    except OSError as exc:
        errors.append(f"{path.name}: {exc}")
    return records, errors


def _sft_status(
    records: list[dict[str, Any]],
    *,
    actor: str,
    lifecycle_status: str | None = None,
) -> dict[str, Any]:
    return analyze_sft_calls(
        records,
        actor=actor,
        lifecycle_status=lifecycle_status,
    )


def _call_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    call_types = Counter(str(record.get("type", "unknown")) for record in records)
    indexes = [
        record["call_idx"] for record in records
        if isinstance(record.get("call_idx"), int)
    ]
    context_generations: set[int] = set()
    for record in records:
        if record.get("type") not in {"main", "agent", "agent_finalizer"}:
            continue
        value = record.get("context_generation")
        if not isinstance(value, int):
            value = (
                0
                if record.get("parent_agent_id")
                else record.get("generation", 0)
            )
        if isinstance(value, int):
            context_generations.add(value)
    return {
        "calls": len(records),
        "call_types": dict(sorted(call_types.items())),
        "call_indexes": indexes,
        "generations": sorted({
            record["generation"] for record in records
            if isinstance(record.get("generation"), int)
        }),
        "context_generations": sorted(context_generations),
    }


def inspect_trace(root_id: str, *, directory: Path | None = None) -> dict[str, Any]:
    """Return a JSON-serializable trace tree with SFT readiness diagnostics."""
    if directory is None:
        from cca.paths import LLM_CALLS_DIR
        directory = LLM_CALLS_DIR
    directory = Path(directory)
    root_path = directory / f"session_{root_id}.jsonl"
    manifest_path = directory / f"session_{root_id}_agents.json"
    event_path = directory / f"session_{root_id}_agents.jsonl"

    if not root_path.exists() and not manifest_path.exists() and not event_path.exists():
        raise FileNotFoundError(f"No trace artifacts found for root '{root_id}'")

    root_records, root_errors = _read_jsonl(root_path) if root_path.exists() else ([], [])
    root_sft = _sft_status(root_records, actor="root")
    root = {
        "agent_id": (
            next((record.get("agent_id") for record in root_records if record.get("agent_id")), root_id)
        ),
        **_call_summary(root_records),
        "sft": root_sft,
        "parse_errors": root_errors,
        "multi_agent_capable": any(
            bool(record.get("multi_agent_capable")) for record in root_records
        ),
        "context_events": [
            {
                "event": record.get("event"),
                "reason": record.get("reason"),
                "context_generation": record.get("context_generation"),
                "previous_context_generation": record.get(
                    "previous_context_generation"
                ),
                "session_id": record.get("session_id"),
                "run_id": record.get("run_id"),
                "compaction_mode": record.get("compaction_mode"),
            }
            for record in root_records
            if record.get("type") == "trace_event"
        ],
    }

    manifest_errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest_errors.append(f"{manifest_path.name}: manifest is not an object")
            manifest = {}
    except FileNotFoundError:
        manifest = {}
    except (OSError, json.JSONDecodeError) as exc:
        manifest_errors.append(f"{manifest_path.name}: {exc}")
        manifest = {}

    children: list[dict[str, Any]] = []
    manifested_ids: set[str] = set()
    for agent_id, raw in sorted((manifest.get("agents") or {}).items()):
        if not isinstance(raw, dict):
            continue
        manifested_ids.add(agent_id)
        child_path = directory / f"session_{root_id}_sub_{agent_id}.jsonl"
        records, parse_errors = _read_jsonl(child_path) if child_path.exists() else ([], [])
        definition = raw.get("agent_definition") or {}
        status = str(raw.get("status", "unknown"))
        reproducibility_issues: list[str] = []
        for field, value in (
            ("parent_agent_id", raw.get("parent_agent_id")),
            ("spawn_call_id", raw.get("spawn_call_id")),
            ("catalog_hash", raw.get("catalog_hash")),
            ("agent_definition", definition),
            ("definition_hash", definition.get("definition_hash")),
        ):
            if not value:
                reproducibility_issues.append(f"missing {field}")
        if not child_path.exists():
            reproducibility_issues.append("missing child call log")
        result = raw.get("result") if isinstance(raw.get("result"), dict) else None
        children.append({
            "agent_id": agent_id,
            "agent_name": definition.get("name"),
            "status": status,
            "trigger": raw.get("trigger"),
            "description": raw.get("description", ""),
            "parent_agent_id": raw.get("parent_agent_id"),
            "spawn_call_id": raw.get("spawn_call_id"),
            "spawn_group_id": raw.get("spawn_group_id"),
            "spawn_index": raw.get("spawn_index"),
            "catalog_hash": raw.get("catalog_hash"),
            "definition_hash": definition.get("definition_hash"),
            **_call_summary(records),
            "sft": _sft_status(records, actor="child", lifecycle_status=status),
            "reproducible": not reproducibility_issues and not parse_errors,
            "reproducibility_issues": [*reproducibility_issues, *parse_errors],
            "summary": result.get("summary") if result else None,
        })

    orphan_logs = []
    prefix = f"session_{root_id}_sub_"
    for path in sorted(directory.glob(f"{prefix}*.jsonl")):
        agent_id = path.name[len(prefix):-len(".jsonl")]
        if agent_id not in manifested_ids:
            orphan_logs.append(path.name)

    events, event_errors = _read_jsonl(event_path) if event_path.exists() else ([], [])
    event_summaries = [{
        "event": event.get("event"),
        "event_id": event.get("event_id"),
        "ts": event.get("ts"),
        "agent_id": event.get("agent_id"),
        "target_agent_id": event.get("target_agent_id"),
        "origin": event.get("origin"),
    } for event in events]
    controls = [
        event for event in event_summaries
        if event.get("event") in {"agent.default.changed", "agent.stop.requested"}
    ]
    completed_children = [child for child in children if child["status"] == "completed"]
    clean_export_ready = (
        root_sft["status"] == "ready"
        and root_sft["call_coverage_complete"]
        and not root_errors
        and not manifest_errors
        and not event_errors
        and not orphan_logs
        and all(
            child["sft"]["status"] == "ready"
            and child["sft"]["call_coverage_complete"]
            for child in completed_children
        )
        and all(child["reproducible"] for child in completed_children)
    )
    return {
        "schema_version": 1,
        "root_id": root_id,
        "directory": str(directory),
        "clean_export_ready": clean_export_ready,
        "root": root,
        "children": children,
        "events": event_summaries,
        "control_events": controls,
        "manifest_errors": manifest_errors,
        "event_parse_errors": event_errors,
        "orphan_child_logs": orphan_logs,
    }
