"""Optional OpenTelemetry lifecycle exporter."""

from __future__ import annotations

from importlib import metadata
from typing import Any

from cca.observability.runtime import ObservabilityConfigError, ObservabilitySettings


class OpenTelemetryExporter:
    """Translate CCA lifecycle spans to batched OTLP spans."""

    def __init__(self, settings: ObservabilitySettings) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        except ImportError as exc:
            raise ObservabilityConfigError(
                "Observability requires optional dependencies. Install with: "
                "pip install 'cca-cli[observability]'"
            ) from exc

        try:
            version = metadata.version("cca-cli")
        except metadata.PackageNotFoundError:
            version = "development"
        resource_attributes: dict[str, Any] = {
            "service.name": settings.service_name,
            "service.version": version,
        }
        if settings.destination == "wandb":
            resource_attributes.update({
                "wandb.entity": settings.entity,
                "wandb.project": settings.project,
            })
        self._trace = trace
        self._provider = TracerProvider(
            resource=Resource.create(resource_attributes),
            sampler=ParentBased(TraceIdRatioBased(settings.sample_rate)),
        )
        exporter = OTLPSpanExporter(
            endpoint=settings.endpoint,
            headers=settings.headers or None,
            timeout=settings.flush_timeout,
        )
        self._provider.add_span_processor(BatchSpanProcessor(
            exporter,
            schedule_delay_millis=1_000,
        ))
        self._tracer = self._provider.get_tracer("cca.observability", version)
        self._settings = settings

    def start_span(
        self,
        *,
        span_id: str,
        parent_token: Any,
        name: str,
        kind: str,
        started_at_ns: int,
        attributes: dict[str, Any],
    ) -> Any:
        from opentelemetry.trace import SpanKind

        parent_context = (
            self._trace.set_span_in_context(parent_token)
            if parent_token is not None
            else None
        )
        otel_kind = SpanKind.CLIENT if kind == "llm" else SpanKind.INTERNAL
        mapped = dict(attributes)
        trace_id = mapped.get("cca.trace_id")
        if trace_id:
            mapped.setdefault("gen_ai.conversation.id", trace_id)
            if self._settings.destination == "wandb":
                mapped.setdefault("wandb.thread_id", trace_id)
        if kind == "turn" and self._settings.destination == "wandb":
            mapped["wandb.is_turn"] = True
        if kind == "llm":
            mapped.setdefault("gen_ai.operation.name", "chat")
        elif kind == "tool":
            mapped.setdefault("gen_ai.operation.name", "execute_tool")
        return self._tracer.start_span(
            name,
            context=parent_context,
            kind=otel_kind,
            attributes=mapped,
            start_time=started_at_ns,
        )

    def finish_span(
        self,
        token: Any,
        *,
        status: str,
        ended_at_ns: int,
        attributes: dict[str, Any],
    ) -> None:
        from opentelemetry.trace import Status, StatusCode

        for key, value in attributes.items():
            token.set_attribute(key, value)
        if status == "error":
            token.set_status(Status(StatusCode.ERROR))
        elif status == "ok":
            token.set_status(Status(StatusCode.OK))
        token.end(end_time=ended_at_ns)

    def force_flush(self, timeout: float) -> None:
        self._provider.force_flush(timeout_millis=max(1, int(timeout * 1000)))

    def shutdown(self, timeout: float) -> None:
        self.force_flush(timeout)
        self._provider.shutdown()
