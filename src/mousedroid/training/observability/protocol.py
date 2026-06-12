"""Protocol + dataclass for experiment-logger DI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseContext:
    """Opaque handle representing an active phase (child) run.

    Returned by :meth:`ExperimentLoggerProtocol.start_phase` and passed back
    to ``log_phase_metric`` / ``end_phase`` so the logger can route per-phase
    metrics to the right child run without per-call lookups. ``run_id`` is
    the backend's identifier (an MLflow run-id, or a ``noop-phase-<phase>``
    sentinel for the NoOp logger); callers MUST treat it as opaque.
    """

    run_id: str
    phase: str


@runtime_checkable
class ExperimentLoggerProtocol(Protocol):
    """Interface for an experiment logger threaded into the training pipeline.

    Two-tier scope:

    * ``start_run`` / ``log_params`` / ``log_metric`` / ``log_artifact`` /
      ``end_run`` — operate on the **parent (pipeline)** run.
    * ``start_phase`` / ``log_phase_metric`` / ``log_phase_artifact`` /
      ``end_phase`` — operate on a **child (phase)** run nested under the
      parent. The parent → child relation is implementation-defined (the
      MLflow concrete uses the ``mlflow.parentRunId`` tag).

    All methods are total — they MUST NOT raise on backend failure
    (network drop, malformed input, NaN). On failure they emit a structured
    warning and return without side effect, mirroring the
    ``Never raises on backend failure`` contract used by the LLM gateways.
    """

    # --- parent-run lifecycle -------------------------------------------------
    def start_run(
        self,
        *,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start the parent run and return its run-id.

        Args:
            run_name: Optional display name for the parent run. When ``None``
                the implementation falls back to its configured default run
                name (e.g. ``ExperimentLoggerConfig.run_name``) or a
                hard-coded sentinel such as ``"pipeline"``.
            params: Static hyperparameters logged immediately.
            tags: Key-value tags attached to the run.
        """
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log scalar params on the parent run (e.g. resolved config snapshot)."""
        ...

    def log_metric(self, key: str, value: Any, step: int | None = None) -> None:
        """Log a scalar metric on the parent run.

        ``value`` is coerced via :func:`to_finite_float`; NaN/Inf are
        skipped with a warning.
        """
        ...

    def log_artifact(self, local_path: str) -> None:
        """Upload a local file as a parent-run artifact (e.g. config.json)."""
        ...

    def end_run(self, *, status: str = "FINISHED") -> None:
        """Terminate the parent run.

        ``status`` MUST be one of ``FINISHED`` / ``FAILED`` / ``KILLED``.
        """
        ...

    # --- child (phase) lifecycle ---------------------------------------------
    def start_phase(
        self,
        *,
        phase: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PhaseContext:
        """Start a child run nested under the active parent run."""
        ...

    def log_phase_metric(
        self,
        ctx: PhaseContext,
        key: str,
        value: Any,
        step: int | None = None,
    ) -> None:
        """Log a scalar metric on the child run identified by ``ctx``."""
        ...

    def log_phase_artifact(self, ctx: PhaseContext, local_path: str) -> None:
        """Upload a local file as a child-run artifact."""
        ...

    def end_phase(self, ctx: PhaseContext, *, status: str = "FINISHED") -> None:
        """Terminate the child run identified by ``ctx``."""
        ...


# --- helpers --------------------------------------------------------------- #
def to_finite_float(value: Any) -> float | None:
    """Coerce ``value`` to a finite Python ``float`` or return ``None``.

    Handles ``int``, ``float``, numpy scalars, and torch zero-dim tensors.
    Returns ``None`` for NaN / Inf / non-numeric / dict / None — the caller
    is responsible for skipping the log call when ``None`` is returned.

    The helper exists so every code path that hits MLflow's ``log_metric``
    goes through one finite-check gate — a NaN slipping through silently
    creates curve gaps in the UI and wastes operator time. Pair with
    ``torch.no_grad()`` in the call site if the value comes from a tensor
    that still has its gradient graph attached (CLAUDE.md invariant #7).
    """
    if value is None or isinstance(value, str | bytes | dict | list | tuple | set):
        return None
    # Torch 0-dim tensor / numpy scalar both expose ``.item()``.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (RuntimeError, ValueError, TypeError):
            return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        _log.warning("experiment_logger_skipped_nonfinite", value=repr(value))
        return None
    return coerced


__all__ = ["ExperimentLoggerProtocol", "PhaseContext", "to_finite_float"]
