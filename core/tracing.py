import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("contexta.tracing")


@dataclass
class TracingState:
    enabled: bool
    exporter: str
    service_name: str
    endpoint: str | None


_TRACER = None
_STATE = TracingState(enabled=False, exporter="none", service_name="contexta-api", endpoint=None)


def initialize_tracing() -> TracingState:
    """Initialize optional OpenTelemetry tracing with OTLP export when configured."""
    global _TRACER, _STATE

    enabled = (os.getenv("TRACING_ENABLED", "false") or "false").strip().lower() == "true"
    service_name = (os.getenv("OTEL_SERVICE_NAME") or "contexta-api").strip()
    exporter = (os.getenv("TRACING_EXPORTER", "otlp") or "otlp").strip().lower()
    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip() or None

    if not enabled:
        _STATE = TracingState(enabled=False, exporter="none", service_name=service_name, endpoint=endpoint)
        _TRACER = None
        return _STATE

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

        if exporter == "otlp":
            if not endpoint:
                raise RuntimeError("OTEL_EXPORTER_OTLP_ENDPOINT must be set when TRACING_ENABLED=true")
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            span_exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(span_exporter))
        else:
            raise RuntimeError(f"Unsupported TRACING_EXPORTER: {exporter}")

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(service_name)
        _STATE = TracingState(enabled=True, exporter=exporter, service_name=service_name, endpoint=endpoint)
    except Exception as e:  # pragma: no cover - depends on optional package/runtime collector
        logger.warning(f"Tracing disabled due to initialization error: {e}")
        _TRACER = None
        _STATE = TracingState(enabled=False, exporter="none", service_name=service_name, endpoint=endpoint)

    return _STATE


def get_tracer():
    return _TRACER


def get_tracing_state() -> TracingState:
    return _STATE
