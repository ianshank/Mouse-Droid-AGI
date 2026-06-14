"""Tests for NoOpExperimentLogger — byte-identical no-op contract."""

from __future__ import annotations

import structlog.testing

from mousedroid.training.observability.noop_logger import NoOpExperimentLogger
from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)


def test_satisfies_protocol() -> None:
    """NoOp satisfies the runtime-checkable protocol."""
    assert isinstance(NoOpExperimentLogger(), ExperimentLoggerProtocol)


def test_start_run_returns_sentinel_id() -> None:
    """Returns a stable ``noop-run`` id so callers can stash it without crashing."""
    logger = NoOpExperimentLogger()
    run_id = logger.start_run(run_name="x", params={"a": 1}, tags={"b": "c"})
    assert run_id == "noop-run"


def test_start_phase_returns_phase_context() -> None:
    """Returns a PhaseContext with a stable id and the requested phase name."""
    logger = NoOpExperimentLogger()
    ctx = logger.start_phase(phase="rssm", params={"lr": 1e-3}, tags={"phase": "rssm"})
    assert isinstance(ctx, PhaseContext)
    assert ctx.phase == "rssm"
    assert ctx.run_id == "noop-phase-rssm"


def test_methods_emit_no_logs() -> None:
    """Every call is a silent no-op — NoOp must not spam structured logs."""
    with structlog.testing.capture_logs() as logs:
        logger = NoOpExperimentLogger()
        logger.start_run(run_name="x")
        logger.log_params({"a": 1})
        logger.log_metric("loss", 0.5, step=1)
        logger.log_artifact("/tmp/x.json")
        ctx = logger.start_phase(phase="p")
        logger.log_phase_metric(ctx, "loss", 0.4, step=1)
        logger.log_phase_artifact(ctx, "/tmp/y.json")
        logger.end_phase(ctx)
        logger.end_run()
    assert logs == []


def test_end_run_status_accepted_but_ignored() -> None:
    """All three legal statuses (FINISHED/FAILED/KILLED) are accepted, no raise."""
    logger = NoOpExperimentLogger()
    logger.start_run(run_name="x")
    logger.end_run(status="FAILED")  # must not raise
    logger.end_run(status="KILLED")  # must not raise
    logger.end_run(status="FINISHED")  # must not raise
