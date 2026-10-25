"""Lifecycle spans, privacy policy, and exporter orchestration.

This module is dependency-free.  Exporters are optional adapters and all of
their runtime failures are isolated from agent execution.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Iterator, Protocol


_CURRENT_SPAN: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "cca_observability_span", default=None
)

_CAPTURE_LEVELS = frozenset({"metadata", "content"})
_DESTINATIONS = frozenset({"otlp", "wandb"})
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|secret|token)",
    re.IGNORECASE,
)
_INLINE_SECRET = re.compile(
    r"(?i)((?:api[_-]?key|authorization|password|secret|token)\s*[:=]\s*)"
    # An auth scheme must be consumed with the credential, or "Bearer" alone is
    # redacted and the token that follows it survives.
    r"(?:(?:bearer|basic|digest|token)\s+)?[^\s,;]+"
)
_TOKEN_LIKE = re.compile(
    r"\b(?:sk-(?:ant-|proj-)?|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{12,}\b"
)


class ObservabilityConfigError(ValueError):
    """Invalid or incomplete opt-in observability configuration."""


@dataclass(frozen=True)
class ObservabilitySettings:
    enabled: bool = False
    destination: str | None = None
    endpoint: str | None = None
    capture: str = "metadata"
    sample_rate: float = 1.0
    max_content_chars: int = 100_000
    flush_timeout: float = 2.0
    service_name: str = "cca"
    entity: str | None = None
    project: str | None = None
    _headers: tuple[tuple[str, str], ...] = field(
        default_factory=tuple, repr=False
    )

    @property
    def headers(self) -> dict[str, str]:
        return dict(self._headers)

    def safe_payload(self) -> dict[str, Any]:
        """Resolved configuration safe for session files and machine output."""
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": self.enabled,
            "destination": self.destination,
            "endpoint": self.endpoint,
            "capture": self.capture,
            "sample_rate": self.sample_rate,
            "max_content_chars": self.max_content_chars,
            "flush_timeout": self.flush_timeout,
            "service_name": self.service_name,
            "entity": self.entity,
            "project": self.project,
            "authentication": "env:WANDB_API_KEY"
            if self.destination == "wandb" and self.enabled
            else ("configured" if self._headers else "environment/default"),
        }


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ObservabilityConfigError(f"observability.{name} must be true or false")


def _normalise_wandb_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint
    if endpoint.endswith("/otel"):
        return endpoint + "/v1/traces"
    return endpoint + "/otel/v1/traces"


def resolve_observability(
    raw: dict[str, Any] | None,
    *,
    destination_override: str | None = None,
    capture_override: str | None = None,
) -> ObservabilitySettings:
    """Resolve opt-in configuration without importing an exporter dependency."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ObservabilityConfigError("observability must be a mapping")

    unknown = set(raw) - {
        "enabled", "destination", "exporter", "endpoint", "capture",
        "sample_rate", "max_content_chars", "flush_timeout", "service_name",
        "entity", "project",
    }
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ObservabilityConfigError(f"unknown observability setting(s): {names}")

    configured_enabled = raw.get("enabled", False)
    enabled = _as_bool(configured_enabled, name="enabled")
    if destination_override is not None:
        enabled = True

    destination = destination_override or raw.get("destination") or raw.get("exporter")
    if destination is not None:
        destination = str(destination).lower()
    if not enabled:
        return ObservabilitySettings()
    if destination is None:
        destination = "wandb"
    if destination not in _DESTINATIONS:
        raise ObservabilityConfigError(
            "observability.destination must be one of: otlp, wandb"
        )

    capture = str(capture_override or raw.get("capture", "metadata")).lower()
    if capture not in _CAPTURE_LEVELS:
        raise ObservabilityConfigError(
            "observability.capture must be one of: metadata, content"
        )
    try:
        sample_rate = float(raw.get("sample_rate", 1.0))
    except (TypeError, ValueError) as exc:
        raise ObservabilityConfigError(
            "observability.sample_rate must be a number from 0 to 1"
        ) from exc
    if not 0.0 <= sample_rate <= 1.0:
        raise ObservabilityConfigError(
            "observability.sample_rate must be a number from 0 to 1"
        )
    try:
        max_content_chars = int(raw.get("max_content_chars", 100_000))
        flush_timeout = float(raw.get("flush_timeout", 2.0))
    except (TypeError, ValueError) as exc:
        raise ObservabilityConfigError(
            "observability size and timeout settings must be numeric"
        ) from exc
    if max_content_chars < 0:
        raise ObservabilityConfigError(
            "observability.max_content_chars must be zero or greater"
        )
    if flush_timeout <= 0:
        raise ObservabilityConfigError(
            "observability.flush_timeout must be greater than zero"
        )

    endpoint = raw.get("endpoint")
    entity = raw.get("entity") or os.environ.get("WANDB_ENTITY")
    project = raw.get("project") or os.environ.get("WANDB_PROJECT")
    headers: dict[str, str] = {}
    if destination == "wandb":
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise ObservabilityConfigError(
                "W&B observability requires WANDB_API_KEY in the environment"
            )
        if not entity:
            raise ObservabilityConfigError(
                "W&B observability requires observability.entity or WANDB_ENTITY"
            )
        project = str(project or "cca")
        endpoint = endpoint or os.environ.get("WANDB_OTLP_ENDPOINT")
        endpoint = _normalise_wandb_endpoint(
            str(endpoint or "https://trace.wandb.ai")
        )
        headers["wandb-api-key"] = api_key
    else:
        endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        if not endpoint:
            raise ObservabilityConfigError(
                "OTLP observability requires observability.endpoint or "
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
            )

    return ObservabilitySettings(
        enabled=True,
        destination=destination,
        endpoint=str(endpoint),
        capture=capture,
        sample_rate=sample_rate,
        max_content_chars=max_content_chars,
        flush_timeout=flush_timeout,
        service_name=str(raw.get("service_name", "cca")),
        entity=str(entity) if entity else None,
        project=str(project) if project else None,
        _headers=tuple(headers.items()),
    )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = _INLINE_SECRET.sub(r"\1[REDACTED]", value)
        return _TOKEN_LIKE.sub("[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _truncate_text(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[:budget] + f"…[truncated {len(text) - budget} chars]"


def _cap_strings(value: Any, budget: int) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, budget)
    if isinstance(value, dict):
        return {key: _cap_strings(item, budget) for key, item in value.items()}
    if isinstance(value, list):
        return [_cap_strings(item, budget) for item in value]
    return value


def _cap_sequences(value: Any, keep: int) -> Any:
    if isinstance(value, dict):
        return {key: _cap_sequences(item, keep) for key, item in value.items()}
    if not isinstance(value, list):
        return value
    if len(value) > keep:
        head = max(1, keep // 2)
        tail = keep - head
        elided = [f"…[truncated {len(value) - keep} items]"]
        value = value[:head] + elided + (value[-tail:] if tail else [])
    return [_cap_sequences(item, keep) for item in value]


# Progressively harsher (sequence, string) caps.  Structural reduction keeps the
# exported payload parseable, which is what backends need to render a tree
# instead of an opaque blob.
_SHRINK_LADDER = ((64, 8192), (64, 2048), (32, 512), (16, 128), (8, 64), (4, 32), (2, 16))


def _shrink_to_fit(value: Any, limit: int) -> Any:
    candidate = value
    for sequence_keep, string_budget in _SHRINK_LADDER:
        candidate = _cap_sequences(_cap_strings(value, string_budget), sequence_keep)
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) <= limit:
            break
    return candidate


def _bounded_json(value: Any, limit: int) -> str:
    redacted = _redact(value)
    rendered = json.dumps(redacted, ensure_ascii=False, default=str)
    if len(rendered) <= limit:
        return rendered
    return json.dumps(
        _shrink_to_fit(redacted, limit), ensure_ascii=False, default=str
    )


def _bounded_content(value: Any, limit: int) -> str:
    """Plain strings stay unquoted; structured values become bounded JSON."""
    if isinstance(value, str):
        return _truncate_text(_redact(value), limit)
    return _bounded_json(value, limit)


def _attribute_value(value: Any) -> str | bool | int | float | list[Any]:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, bool, int, float)) for item in value
    ):
        return list(value)
    return json.dumps(_redact(value), ensure_ascii=False, default=str)


class LifecycleExporter(Protocol):
    def start_span(
        self,
        *,
        span_id: str,
        parent_token: Any,
        name: str,
        kind: str,
        started_at_ns: int,
        attributes: dict[str, Any],
    ) -> Any: ...

    def finish_span(
        self,
        token: Any,
        *,
        status: str,
        ended_at_ns: int,
        attributes: dict[str, Any],
    ) -> None: ...

    def force_flush(self, timeout: float) -> None: ...

    def shutdown(self, timeout: float) -> None: ...


class Span:
    """A destination-neutral handle for one observed operation."""

    def __init__(
        self,
        observer: "Observer",
        *,
        span_id: str,
        parent: "Span | None",
        name: str,
        kind: str,
        started_at_ns: int,
        started_mono_ns: int,
        tokens: list[tuple[LifecycleExporter, Any]],
    ) -> None:
        self.observer = observer
        self.span_id = span_id
        self.parent = parent
        self.name = name
        self.kind = kind
        self.started_at_ns = started_at_ns
        self.started_mono_ns = started_mono_ns
        self._tokens = tokens
        self._ended = False
        self._lock = threading.Lock()
        self._activation_token: contextvars.Token | None = None

    @property
    def parent_span_id(self) -> str | None:
        return self.parent.span_id if self.parent is not None else None

    def _token_for(self, exporter: LifecycleExporter) -> Any:
        for candidate, token in self._tokens:
            if candidate is exporter:
                return token
        return None

    def finish(
        self,
        *,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        ended_at_ns = time.time_ns()
        duration_ms = max(0.0, (time.monotonic_ns() - self.started_mono_ns) / 1_000_000)
        final = {
            "cca.status": status,
            "cca.duration_ms": duration_ms,
            **self.observer._prepare_attributes(attributes or {}),
            **self.observer._prepare_attributes(metrics or {}),
            **self.observer._prepare_content(content),
        }
        if error is not None:
            final["error.type"] = type(error).__name__
            if self.observer.settings.capture == "content":
                final["error.message"] = self.observer._bounded_content(str(error))
        for exporter, token in self._tokens:
            try:
                exporter.finish_span(
                    token,
                    status=status,
                    ended_at_ns=ended_at_ns,
                    attributes=final,
                )
            except Exception:
                self.observer._record_export_error()

    def __enter__(self) -> "Span":
        self._activation_token = _CURRENT_SPAN.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._activation_token is not None:
            _CURRENT_SPAN.reset(self._activation_token)
            self._activation_token = None
        if exc is None:
            self.finish()
        elif isinstance(exc, asyncio.CancelledError):
            self.finish(status="cancelled", error=exc)
        else:
            self.finish(status="error", error=exc)
        return False


class Observer:
    """Coordinates lifecycle exporters behind one fail-open interface."""

    def __init__(
        self,
        settings: ObservabilitySettings | None = None,
        exporters: list[LifecycleExporter] | None = None,
        *,
        common_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or ObservabilitySettings()
        self._exporters = list(exporters or [])
        self._common_attributes = self._prepare_attributes(common_attributes or {})
        self._export_errors = 0
        self._shutdown = False

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and bool(self._exporters)

    @property
    def export_errors(self) -> int:
        return self._export_errors

    @property
    def current_span(self) -> Span | None:
        span = _CURRENT_SPAN.get()
        return span if span is not None and span.observer is self else None

    def _record_export_error(self) -> None:
        self._export_errors += 1

    def add_common_attributes(self, attributes: dict[str, Any]) -> None:
        """Add stable run attributes after session identity has been resolved."""
        self._common_attributes.update(self._prepare_attributes(attributes))

    def _prepare_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): _attribute_value(value)
            for key, value in attributes.items()
            if value is not None
        }

    def _bounded_content(self, value: Any) -> str:
        return _bounded_content(value, self.settings.max_content_chars)

    def _prepare_content(self, content: dict[str, Any] | None) -> dict[str, Any]:
        if self.settings.capture != "content" or not content:
            return {}
        prepared: dict[str, Any] = {}
        for key, value in content.items():
            name = str(key)
            prepared[name] = self._bounded_content(value)
            # OpenInference pairs `<x>.value` with `<x>.mime_type`; without it
            # backends render serialized payloads as opaque strings.
            if name.endswith(".value"):
                prefix = name[: -len(".value")]
                prepared[f"{prefix}.mime_type"] = (
                    "text/plain" if isinstance(value, str) else "application/json"
                )
        return prepared

    def start(
        self,
        name: str,
        *,
        kind: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
    ) -> Span:
        if parent is None:
            parent = self.current_span
        span_id = uuid.uuid4().hex
        started_at_ns = time.time_ns()
        started_mono_ns = time.monotonic_ns()
        prepared = {
            **self._common_attributes,
            **self._prepare_attributes(attributes or {}),
            **self._prepare_content(content),
            "cca.span_id": span_id,
            "cca.parent_span_id": parent.span_id if parent is not None else None,
            "cca.span.kind": kind,
        }
        prepared = {key: value for key, value in prepared.items() if value is not None}
        tokens: list[tuple[LifecycleExporter, Any]] = []
        if not self._shutdown:
            for exporter in self._exporters:
                try:
                    token = exporter.start_span(
                        span_id=span_id,
                        parent_token=parent._token_for(exporter) if parent else None,
                        name=name,
                        kind=kind,
                        started_at_ns=started_at_ns,
                        attributes=prepared,
                    )
                    tokens.append((exporter, token))
                except Exception:
                    self._record_export_error()
        return Span(
            self,
            span_id=span_id,
            parent=parent,
            name=name,
            kind=kind,
            started_at_ns=started_at_ns,
            started_mono_ns=started_mono_ns,
            tokens=tokens,
        )

    @contextmanager
    def span(self, name: str, *, kind: str, **kwargs: Any) -> Iterator[Span]:
        handle = self.start(name, kind=kind, **kwargs)
        with handle:
            yield handle

    def force_flush(self, timeout: float | None = None) -> None:
        limit = timeout or self.settings.flush_timeout
        for exporter in self._exporters:
            try:
                exporter.force_flush(limit)
            except Exception:
                self._record_export_error()

    def shutdown(self, timeout: float | None = None) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        limit = timeout or self.settings.flush_timeout
        for exporter in self._exporters:
            try:
                exporter.shutdown(limit)
            except Exception:
                self._record_export_error()


class InMemoryExporter:
    """Deterministic exporter used by lifecycle contract tests."""

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []
        self.flushed = False
        self.closed = False

    def start_span(self, **record: Any) -> dict[str, Any]:
        token = dict(record)
        token["parent_span_id"] = (
            record["parent_token"].get("span_id")
            if isinstance(record.get("parent_token"), dict)
            else None
        )
        self.started.append(token)
        return token

    def finish_span(self, token: Any, **record: Any) -> None:
        self.finished.append({"span_id": token["span_id"], **record})

    def force_flush(self, timeout: float) -> None:
        self.flushed = True

    def shutdown(self, timeout: float) -> None:
        self.closed = True


def create_observer(
    settings: ObservabilitySettings,
    *,
    common_attributes: dict[str, Any] | None = None,
) -> Observer:
    if not settings.enabled:
        return Observer(settings, common_attributes=common_attributes)
    from cca.observability.otel import OpenTelemetryExporter

    return Observer(
        settings,
        [OpenTelemetryExporter(settings)],
        common_attributes=common_attributes,
    )
