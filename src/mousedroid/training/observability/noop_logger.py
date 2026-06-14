"""NoOp experiment logger — the always-available default."""

from __future__ import annotations

from typing import Any

from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)

# Opaque run-id sentinels (callers MUST treat run_id as opaque). Named so a
# format change has a single source of truth.
_NOOP_RUN_ID = "noop-run"
_NOOP_PHASE_PREFIX = "noop-phase-"


class NoOpExperimentLogger:
    """Every method is a silent no-op.

    Conforms structurally to :class:`ExperimentLoggerProtocol`. Returned by
    :func:`mousedroid.factory.build_experiment_logger` when observability is
    disabled OR when ``mlflow`` is not installed — so call sites can ALWAYS
    rely on a non-None logger and skip ``if logger is not None`` guards.
    """

    def start_run(
        self,
        *,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Return a stable run-id sentinel."""
        del run_name, params, tags
        return _NOOP_RUN_ID

    def log_params(self, params: dict[str, Any]) -> None:
        """Silent no-op."""
        del params

    def log_metric(self, key: str, value: Any, step: int | None = None) -> None:
        """Silent no-op."""
        del key, value, step

    def log_artifact(self, local_path: str) -> None:
        """Silent no-op."""
        del local_path

    def end_run(self, *, status: str = "FINISHED") -> None:
        """Silent no-op."""
        del status

    def start_phase(
        self,
        *,
        phase: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PhaseContext:
        """Return a PhaseContext with a stable id."""
        del params, tags
        return PhaseContext(run_id=f"{_NOOP_PHASE_PREFIX}{phase}", phase=phase)

    def log_phase_metric(
        self,
        ctx: PhaseContext,
        key: str,
        value: Any,
        step: int | None = None,
    ) -> None:
        """Silent no-op."""
        del ctx, key, value, step

    def log_phase_artifact(self, ctx: PhaseContext, local_path: str) -> None:
        """Silent no-op."""
        del ctx, local_path

    def end_phase(self, ctx: PhaseContext, *, status: str = "FINISHED") -> None:
        """Silent no-op."""
        del ctx, status


# Verify structural protocol conformance at import time so a method-signature
# drift fails fast.
_PROTOCOL_CHECK: ExperimentLoggerProtocol = NoOpExperimentLogger()
del _PROTOCOL_CHECK


__all__ = ["NoOpExperimentLogger"]
