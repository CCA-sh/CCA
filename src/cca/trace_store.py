"""Serialized, lineage-aware storage for root and subagent traces."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any


_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class TraceIdentity:
    trace_id: str
    agent_id: str
    parent_agent_id: str | None = None
    spawn_call_id: str | None = None
    # ``generation`` is the original lineage field and remains on every record
    # for compatibility. ``context_generation`` names the model-visible context
    # epoch explicitly; they differ for children, whose legacy generation is the
    # parent generation at spawn while their own context starts at zero.
    generation: int = 0
    context_generation: int | None = None
    generation_reason: str | None = None
    parent_context_generation: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "spawn_call_id": self.spawn_call_id,
            "generation": self.generation,
            "context_generation": (
                self.generation
                if self.context_generation is None
                else self.context_generation
            ),
        }
        if self.generation_reason is not None:
            payload["generation_reason"] = self.generation_reason
        if self.parent_context_generation is not None:
            payload["parent_context_generation"] = self.parent_context_generation
        return payload


def generation_from_session_id(session_id: str) -> int:
    match = re.search(r"-comp(\d+)", session_id)
    return int(match.group(1)) if match else 0


def backend_provenance(
    backend: Any,
    requested_selector: str | None,
    resolved_selector: str | None = None,
) -> dict[str, Any]:
    """Describe the backend that will actually receive a model call.

    Configuration is intent; the backend instance is execution. Keeping both
    catches stale bindings such as a parent switching models while an Agent tool
    still holds the previous backend.
    """
    from cca.llm.catalog import split_selector

    requested = requested_selector or ""
    resolved = resolved_selector or requested
    resolved_provider, resolved_model = split_selector(resolved)
    actual_model = getattr(backend, "model", None)
    actual_provider = getattr(backend, "provider", None)
    if not isinstance(actual_provider, str) or not actual_provider:
        adapter = type(backend).__name__.lower()
        actual_provider = "anthropic" if "anthropic" in adapter else resolved_provider
    if isinstance(actual_model, str) and actual_model:
        actual_selector = f"{actual_provider}:{actual_model}"
        verified_from_backend = True
    else:
        actual_model = resolved_model or None
        actual_selector = resolved or None
        verified_from_backend = False
    return {
        "requested_selector": requested or None,
        "selector": actual_selector,
        "provider": actual_provider or None,
        "model": actual_model,
        "backend": type(backend).__name__,
        "verified_from_backend": verified_from_backend,
    }


class TraceStore:
    """Append-only call/event traces plus an atomically updated agent manifest."""

    schema_version = 2

    def __init__(self, root_id: str, *, directory: Path | None = None) -> None:
        if directory is None:
            from cca.paths import LLM_CALLS_DIR
            directory = LLM_CALLS_DIR
        self.root_id = root_id
        self.directory = Path(directory)

    @property
    def root_call_path(self) -> Path:
        return self.directory / f"session_{self.root_id}.jsonl"

    def child_call_path(self, agent_id: str) -> Path:
        return self.directory / f"session_{self.root_id}_sub_{agent_id}.jsonl"

    @property
    def event_path(self) -> Path:
        return self.directory / f"session_{self.root_id}_agents.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.directory / f"session_{self.root_id}_agents.json"

    def _append_json(self, path: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self._locked():
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()

    @contextmanager
    def _locked(self):
        """Serialize writers across threads and, where available, processes."""
        with _WRITE_LOCK:
            self.directory.mkdir(parents=True, exist_ok=True)
            # The existing root call log doubles as the process lock, avoiding
            # any new artifact in sessions that never invoke a subagent.
            lock_path = self.root_call_path
            with open(lock_path, "a", encoding="utf-8") as lock_handle:
                try:
                    import fcntl
                    fcntl.flock(lock_handle, fcntl.LOCK_EX)
                except ImportError:
                    fcntl = None  # type: ignore[assignment]
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle, fcntl.LOCK_UN)

    def append_call(
        self,
        identity: TraceIdentity,
        call_idx: int,
        record: dict[str, Any],
        *,
        child: bool,
    ) -> None:
        payload = {
            "schema_version": self.schema_version,
            **identity.to_payload(),
            "call_idx": call_idx,
            **record,
        }
        path = self.child_call_path(identity.agent_id) if child else self.root_call_path
        self._append_json(path, payload)

    def append_root_event(
        self,
        event_type: str,
        identity: TraceIdentity,
        **data: Any,
    ) -> str:
        """Append root lifecycle provenance without creating child artifacts."""
        event_id = uuid.uuid4().hex
        payload = {
            "schema_version": self.schema_version,
            "event_id": event_id,
            "ts": int(time.time()),
            "type": "trace_event",
            "event": event_type,
            **identity.to_payload(),
            **data,
        }
        self._append_json(self.root_call_path, payload)
        return event_id

    def append_event(
        self,
        event_type: str,
        identity: TraceIdentity,
        **data: Any,
    ) -> str:
        event_id = uuid.uuid4().hex
        payload = {
            "schema_version": self.schema_version,
            "event_id": event_id,
            "ts": int(time.time()),
            "event": event_type,
            **identity.to_payload(),
            **data,
        }
        self._append_json(self.event_path, payload)
        return event_id

    def latest_event(self, event_type: str) -> dict[str, Any] | None:
        """Return the newest matching event without creating trace artifacts."""
        if not self.event_path.exists():
            return None
        latest: dict[str, Any] | None = None
        try:
            with open(self.event_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and record.get("event") == event_type:
                        latest = record
        except OSError:
            return None
        return latest

    def next_call_idx(self, identity: TraceIdentity, *, child: bool) -> int:
        path = self.child_call_path(identity.agent_id) if child else self.root_call_path
        if not path.exists():
            return 0
        highest = -1
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("agent_id", identity.agent_id) == identity.agent_id:
                        idx = record.get("call_idx")
                        if isinstance(idx, int):
                            highest = max(highest, idx)
        except OSError:
            return 0
        return highest + 1

    def _read_manifest(self, trace_id: str, root_agent_id: str) -> dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {
            "schema_version": self.schema_version,
            "trace_id": trace_id,
            "root_agent_id": root_agent_id,
            "agents": {},
        }

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.manifest_path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def register_child(
        self,
        identity: TraceIdentity,
        *,
        root_agent_id: str,
        agent_definition: dict[str, Any],
        catalog_hash: str,
        trigger: str,
        description: str,
        spawn_group_id: str | None = None,
        spawn_index: int | None = None,
    ) -> None:
        with self._locked():
            manifest = self._read_manifest(identity.trace_id, root_agent_id)
            manifest.setdefault("agents", {})[identity.agent_id] = {
                **identity.to_payload(),
                "status": "queued",
                "trigger": trigger,
                "description": description,
                "catalog_hash": catalog_hash,
                "agent_definition": agent_definition,
                "created_at": int(time.time()),
                **(
                    {"spawn_group_id": spawn_group_id}
                    if spawn_group_id is not None else {}
                ),
                **(
                    {"spawn_index": spawn_index}
                    if spawn_index is not None else {}
                ),
            }
            self._write_manifest(manifest)

    def update_child(self, identity: TraceIdentity, **updates: Any) -> None:
        with self._locked():
            manifest = self._read_manifest(identity.trace_id, identity.parent_agent_id or "")
            agents = manifest.setdefault("agents", {})
            entry = agents.setdefault(identity.agent_id, identity.to_payload())
            entry.update(updates)
            entry["updated_at"] = int(time.time())
            self._write_manifest(manifest)
