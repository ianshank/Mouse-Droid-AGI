"""MLflow-backed experiment logger using the ``MlflowClient`` OOP API.

Why ``MlflowClient`` and not the fluent ``mlflow.start_run`` API:

* No reliance on ``mlflow.active_run`` thread-local state — works under
  asyncio + thread pool dispatch without surprise.
* Idempotent / mockable / testable with a real client over a ``tmp_path``
  file backend (the recommended pattern per the project research notes).
* Symmetric with how every other backend wrapper in the codebase looks
  (e.g. :class:`AnthropicLLMGateway` wraps the ``anthropic`` SDK).

Imports ``mlflow`` lazily in :meth:`__init__` so the protocol module stays
import-safe when the ``[mlflow]`` extras are absent — the factory degrades
to :class:`NoOpExperimentLogger` in that case.

CLAUDE.md invariants honored:
* Protocol-DI (#1): conforms structurally to :class:`ExperimentLoggerProtocol`.
* No hardcoded values (#3): every knob comes from ``ExperimentLoggerConfig``.
* Structured logging (#4): all branches emit ``mlflow_logger_*`` events.
* Never raises on backend failure: catches ``Exception`` at every write
  boundary and degrades to a warning log + return (mirrors the LLM gateways).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from mousedroid.logging.setup import get_logger
from mousedroid.training.observability.protocol import (
    PhaseContext,
    _to_finite_float,
)

_log = get_logger(__name__)

_VALID_STATUSES: frozenset[str] = frozenset({"FINISHED", "FAILED", "KILLED"})
_PARENT_RUN_TAG = "mlflow.parentRunId"


class MlflowExperimentLogger:
    """Wraps :class:`mlflow.MlflowClient` for parent + nested-phase runs.

    Construction is the only place this class touches ``mlflow``; method
    bodies operate via the constructed client. The
    :class:`ExperimentLoggerProtocol` consumer never sees an mlflow type
    leak through.
    """

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
    ) -> None:
        """Build the underlying ``MlflowClient`` + resolve the experiment.

        Args:
            tracking_uri: MLflow tracking URI. ``file:`` URIs are pinned to
                an absolute path (the factory resolves CWD-relative paths
                BEFORE calling here so they survive working-dir changes).
            experiment_name: MLflow experiment name; created if missing.
            run_name: Optional default run name for the parent run.
        """
        # Lazy import so a project that never opts-in to mlflow does not pay
        # the import cost. The factory probes the extras availability and
        # degrades to NoOp when missing — so reaching this constructor
        # implies the extras are installed.
        from mlflow import MlflowClient

        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._default_run_name = run_name
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment_id: str = self._resolve_or_create_experiment(experiment_name)
        self._active_run_id: str | None = None
        _log.info(
            "mlflow_logger_initialised",
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            experiment_id=self._experiment_id,
        )

    # ---- experiment resolution ---------------------------------------------
    def _resolve_or_create_experiment(self, name: str) -> str:
        existing = self._client.get_experiment_by_name(name)
        if existing is not None:
            return cast(str, existing.experiment_id)
        # Bind to an annotated local rather than returning directly: under CI's
        # ``--ignore-missing-imports`` mlflow is untyped, so create_experiment is
        # ``Any`` and returning it trips ``no-any-return``; the annotation narrows
        # it. A ``cast`` would instead be flagged ``redundant-cast`` when mlflow IS
        # typed (e.g. a newer mlflow installed locally) — this form passes both.
        new_experiment_id: str = self._client.create_experiment(name)
        return new_experiment_id

    # ---- parent run --------------------------------------------------------
    def start_run(
        self,
        *,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a parent run and return its run-id.

        Args:
            run_name: Display name for the run.
            params: Static hyperparameters to log immediately (e.g. config snapshot).
            tags: Key-value tags attached to the run.

        Returns:
            The MLflow run-id string, or ``""`` if the backend call fails.
        """
        effective_name = run_name or self._default_run_name or "pipeline"
        try:
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                run_name=effective_name,
                tags=tags or {},
            )
        except Exception as exc:  # broad — never raise on backend failure  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_start_run_failed",
                error=f"{type(exc).__name__}:{exc}",
            )
            return ""
        self._active_run_id = cast(str, run.info.run_id)
        if params:
            self.log_params(params)
        return self._active_run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dict of params on the active parent run.

        Args:
            params: Key-value pairs. Values are coerced to ``str`` by MLflow.
        """
        if self._active_run_id is None:
            _log.warning("mlflow_logger_log_params_without_run")
            return
        for key, value in params.items():
            try:
                self._client.log_param(self._active_run_id, key, value)
            except Exception as exc:  # noqa: BLE001, RUF100
                _log.warning(
                    "mlflow_logger_log_param_failed",
                    key=key,
                    error=f"{type(exc).__name__}:{exc}",
                )

    def log_metric(self, key: str, value: Any, step: int | None = None) -> None:
        """Log a scalar metric on the active parent run.

        Args:
            key: Metric name.
            value: Numeric value. NaN/Inf are silently skipped.
            step: Optional training step for X-axis alignment.
        """
        if self._active_run_id is None:
            _log.warning("mlflow_logger_log_metric_without_run", key=key)
            return
        coerced = _to_finite_float(value)
        if coerced is None:
            return  # _to_finite_float already logged the skip
        try:
            self._client.log_metric(self._active_run_id, key, coerced, step=step)
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_log_metric_failed",
                key=key,
                error=f"{type(exc).__name__}:{exc}",
            )

    def log_artifact(self, local_path: str) -> None:
        """Upload a local file as a parent-run artifact.

        Args:
            local_path: Filesystem path to the file to upload.
        """
        if self._active_run_id is None:
            _log.warning("mlflow_logger_log_artifact_without_run", path=local_path)
            return
        if not Path(local_path).exists():
            _log.warning("mlflow_logger_artifact_missing", path=local_path)
            return
        try:
            self._client.log_artifact(self._active_run_id, local_path)
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_log_artifact_failed",
                path=local_path,
                error=f"{type(exc).__name__}:{exc}",
            )

    def end_run(self, *, status: str = "FINISHED") -> None:
        """Terminate the active parent run.

        Args:
            status: One of ``FINISHED`` / ``FAILED`` / ``KILLED``. Unknown
                values are normalised to ``FINISHED`` with a warning — never
                raises.
        """
        if self._active_run_id is None:
            return  # silent — nothing to end
        normalised = status if status in _VALID_STATUSES else "FINISHED"
        if normalised != status:
            _log.warning(
                "mlflow_logger_invalid_status_normalised",
                requested=status,
                normalised=normalised,
            )
        try:
            self._client.set_terminated(self._active_run_id, status=normalised)
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_end_run_failed",
                error=f"{type(exc).__name__}:{exc}",
            )
        finally:
            self._active_run_id = None

    # ---- child (phase) run -------------------------------------------------
    def start_phase(
        self,
        *,
        phase: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> PhaseContext:
        """Start a child run nested under the active parent run.

        Args:
            phase: Logical phase name (e.g. ``"rssm"``, ``"cql"``).
            params: Phase-specific hyperparameters logged immediately.
            tags: Additional key-value tags. ``mlflow.parentRunId`` and
                ``phase`` are set automatically.

        Returns:
            A :class:`PhaseContext` opaque handle. ``run_id`` is ``""`` on
            backend failure so callers can always use the returned object
            safely.
        """
        if self._active_run_id is None:
            _log.warning("mlflow_logger_start_phase_without_parent", phase=phase)
            return PhaseContext(run_id="", phase=phase)
        merged_tags = dict(tags or {})
        merged_tags[_PARENT_RUN_TAG] = self._active_run_id
        merged_tags.setdefault("phase", phase)
        try:
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                run_name=phase,
                tags=merged_tags,
            )
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_start_phase_failed",
                phase=phase,
                error=f"{type(exc).__name__}:{exc}",
            )
            return PhaseContext(run_id="", phase=phase)
        if params:
            for key, value in params.items():
                try:
                    self._client.log_param(run.info.run_id, key, value)
                except Exception as exc:  # noqa: BLE001, RUF100
                    _log.warning(
                        "mlflow_logger_phase_param_failed",
                        phase=phase,
                        key=key,
                        error=f"{type(exc).__name__}:{exc}",
                    )
        return PhaseContext(run_id=run.info.run_id, phase=phase)

    def log_phase_metric(
        self,
        ctx: PhaseContext,
        key: str,
        value: Any,
        step: int | None = None,
    ) -> None:
        """Log a scalar metric on the child run identified by ``ctx``.

        Args:
            ctx: Phase handle returned by :meth:`start_phase`.
            key: Metric name.
            value: Numeric value. NaN/Inf are silently skipped.
            step: Optional training step.
        """
        if not ctx.run_id:
            return
        coerced = _to_finite_float(value)
        if coerced is None:
            return
        try:
            self._client.log_metric(ctx.run_id, key, coerced, step=step)
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_log_phase_metric_failed",
                phase=ctx.phase,
                key=key,
                error=f"{type(exc).__name__}:{exc}",
            )

    def log_phase_artifact(self, ctx: PhaseContext, local_path: str) -> None:
        """Upload a local file as a child-run artifact.

        Args:
            ctx: Phase handle returned by :meth:`start_phase`.
            local_path: Filesystem path to the file to upload.
        """
        if not ctx.run_id:
            return
        if not Path(local_path).exists():
            _log.warning("mlflow_logger_phase_artifact_missing", path=local_path)
            return
        try:
            self._client.log_artifact(ctx.run_id, local_path)
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_log_phase_artifact_failed",
                phase=ctx.phase,
                error=f"{type(exc).__name__}:{exc}",
            )

    def end_phase(self, ctx: PhaseContext, *, status: str = "FINISHED") -> None:
        """Terminate the child run identified by ``ctx``.

        Args:
            ctx: Phase handle returned by :meth:`start_phase`.
            status: One of ``FINISHED`` / ``FAILED`` / ``KILLED``. Unknown
                values are normalised to ``FINISHED`` with a warning.
        """
        if not ctx.run_id:
            return
        normalised = status if status in _VALID_STATUSES else "FINISHED"
        if normalised != status:
            _log.warning(
                "mlflow_logger_invalid_phase_status_normalised",
                phase=ctx.phase,
                requested=status,
                normalised=normalised,
            )
        try:
            self._client.set_terminated(ctx.run_id, status=normalised)
        except Exception as exc:  # noqa: BLE001, RUF100
            _log.warning(
                "mlflow_logger_end_phase_failed",
                phase=ctx.phase,
                error=f"{type(exc).__name__}:{exc}",
            )


# Structural protocol-conformance check at import time. The placeholder
# instantiation is fenced behind ``mlflow`` availability so this module
# remains importable even when the extras are absent.
__all__ = ["MlflowExperimentLogger"]
