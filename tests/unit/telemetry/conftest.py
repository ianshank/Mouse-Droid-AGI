"""Shared test helpers for tests/unit/telemetry/."""

from __future__ import annotations

from unittest.mock import AsyncMock


def _make_health_monitor(
    *,
    gpu_temp_c: float | None = None,
    gpu_load_pct: float | None = None,
) -> AsyncMock:
    """Build a mock health monitor whose ``check_health()`` returns a fixed payload.

    Defaults to the bare ``{"status": "ok"}`` payload; pass ``gpu_temp_c``/
    ``gpu_load_pct`` for call sites that also assert on GPU telemetry fields.
    """
    monitor = AsyncMock()
    payload: dict[str, object] = {"status": "ok"}
    if gpu_temp_c is not None:
        payload["gpu_temp_c"] = gpu_temp_c
    if gpu_load_pct is not None:
        payload["gpu_load_pct"] = gpu_load_pct
    monitor.check_health = AsyncMock(return_value=payload)
    return monitor
