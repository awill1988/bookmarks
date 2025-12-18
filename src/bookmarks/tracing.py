import logging
import os

from opentelemetry import trace

_tracing_configured = False


def configure_tracing(project_name: str = "bookmarks") -> None:
    """
    Configure phoenix/openinference tracing.

    Tracing is enabled when either `BOOKMARKS_ENABLE_TRACING` is truthy or an OTLP endpoint is
    provided via `PHOENIX_COLLECTOR_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT`.
    """
    global _tracing_configured
    if _tracing_configured:
        return

    enabled = os.getenv("BOOKMARKS_ENABLE_TRACING", "").lower() in ("1", "true", "yes")
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not enabled and not endpoint:
        return

    try:
        from phoenix.otel import register
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("failed to import phoenix tracing: %s", exc)
        return

    project = os.getenv("PHOENIX_PROJECT_NAME") or project_name
    protocol = os.getenv("PHOENIX_COLLECTOR_PROTOCOL") or None
    batch = os.getenv("PHOENIX_OTEL_BATCH", "").lower() in ("1", "true", "yes")

    try:
        register(
            endpoint=endpoint,
            project_name=project,
            protocol=protocol,
            batch=batch,
            auto_instrument=False,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - passthrough for runtime issues
        logging.warning("failed to enable phoenix tracing: %s", exc)
        return

    _tracing_configured = True
    logging.info("tracing enabled (project=%s)", project)


def get_tracer(name: str):
    return trace.get_tracer(name)

