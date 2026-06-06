"""Experiment-logger subsystem for the training pipeline.

Defines :class:`ExperimentLoggerProtocol` and ships two implementations:

* :class:`~mousedroid.training.observability.noop_logger.NoOpExperimentLogger`
  — always available, byte-identical no-op (the default).
* :class:`~mousedroid.training.observability.mlflow_logger.MlflowExperimentLogger`
  — wraps :class:`mlflow.MlflowClient`. Concrete; imported lazily inside
  :func:`mousedroid.factory.build_experiment_logger` so callers can rely
  on the protocol without paying the mlflow import cost.

The factory returns ``NoOpExperimentLogger`` when ``cfg.observability`` is
``None`` or ``cfg.observability.experiment_logger.backend == "none"``, and
when the ``[mlflow]`` extras are not installed — preserving byte-identical
behavior to the pre-feature path.
"""

from __future__ import annotations

from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)

__all__ = ["ExperimentLoggerProtocol", "PhaseContext"]
