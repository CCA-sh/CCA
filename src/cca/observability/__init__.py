"""Optional, provider-neutral lifecycle observability for CCA.

The public interface deliberately speaks in CCA concepts.  Destination-specific
translation belongs to exporters, so the agent loop never imports W&B or
OpenTelemetry directly.
"""

from cca.observability.runtime import (
    InMemoryExporter,
    LifecycleExporter,
    ObservabilityConfigError,
    ObservabilitySettings,
    Observer,
    Span,
    create_observer,
    resolve_observability,
)
from cca.observability.semconv import model_attributes, response_model_attributes

__all__ = [
    "model_attributes",
    "response_model_attributes",
    "InMemoryExporter",
    "LifecycleExporter",
    "ObservabilityConfigError",
    "ObservabilitySettings",
    "Observer",
    "Span",
    "create_observer",
    "resolve_observability",
]
