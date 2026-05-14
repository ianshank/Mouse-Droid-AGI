"""Telemetry subsystem exceptions."""

from __future__ import annotations


class TelemetryUnavailableError(RuntimeError):
    """Raised when the telemetry server cannot bind to any candidate port.

    The orchestrator catches this and degrades to metrics-only mode (periodic
    Prometheus text file export) so the rover continues running.
    """


class TelemetryConfigError(ValueError):
    """Raised during factory construction when telemetry configuration is invalid.

    Examples: ``auth_enabled=True`` but the token environment variable is unset.
    """
