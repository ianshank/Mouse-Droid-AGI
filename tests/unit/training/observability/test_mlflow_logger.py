"""Tests for MlflowExperimentLogger — uses a real MlflowClient over tmp_path.

Per the writing-plans research, mocking ``MlflowClient`` itself loses the
ability to catch signature drift on MLflow upgrades. The right pattern is
a tmp_path-rooted file backend and a real client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")  # skip module entirely if extras missing
from mlflow import MlflowClient

from mousedroid.training.observability.mlflow_logger import (
    MlflowExperimentLogger,
)
from mousedroid.training.observability.protocol import (
    ExperimentLoggerProtocol,
    PhaseContext,
)


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'mlruns'}"


@pytest.fixture
def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def _build_logger(tracking_uri: str, experiment: str = "test-exp") -> MlflowExperimentLogger:
    return MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
    )


def test_satisfies_protocol(tracking_uri: str) -> None:
    assert isinstance(_build_logger(tracking_uri), ExperimentLoggerProtocol)


def test_start_run_creates_parent_run_with_params_and_tags(
    tracking_uri: str, client: MlflowClient
) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(
        run_name="pipeline-1",
        params={"phases_count": 4, "amp": True},
        tags={"track": "T"},
    )
    run = client.get_run(run_id)
    assert run.info.status == "RUNNING"
    assert run.info.run_name == "pipeline-1"
    # MLflow coerces param values to strings on read.
    assert run.data.params["phases_count"] == "4"
    assert run.data.params["amp"] == "True"
    assert run.data.tags["track"] == "T"
    logger.end_run()


def test_log_metric_records_step_history(tracking_uri: str, client: MlflowClient) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="metrics")
    for step, loss in enumerate([1.0, 0.8, 0.6]):
        logger.log_metric("loss", loss, step=step)
    logger.end_run()
    history = client.get_metric_history(run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(0, 1.0), (1, 0.8), (2, 0.6)]


def test_log_metric_skips_nonfinite_value(tracking_uri: str, client: MlflowClient) -> None:
    """NaN must not reach the store; the warning is recorded by _to_finite_float."""
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="nan")
    logger.log_metric("loss", float("nan"), step=0)
    logger.log_metric("loss", 0.5, step=1)
    logger.end_run()
    history = client.get_metric_history(run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(1, 0.5)]


def test_start_phase_nests_under_parent_via_tag(tracking_uri: str, client: MlflowClient) -> None:
    """Nested runs are tagged with mlflow.parentRunId — the canonical pattern."""
    logger = _build_logger(tracking_uri)
    parent_id = logger.start_run(run_name="pipe")
    ctx = logger.start_phase(phase="rssm")
    assert isinstance(ctx, PhaseContext)
    child = client.get_run(ctx.run_id)
    assert child.data.tags.get("mlflow.parentRunId") == parent_id
    assert child.data.tags.get("phase") == "rssm"
    logger.end_phase(ctx)
    logger.end_run()


def test_end_run_marks_status_finished(tracking_uri: str, client: MlflowClient) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="ok")
    logger.end_run()
    assert client.get_run(run_id).info.status == "FINISHED"


def test_end_run_status_failed_propagates(tracking_uri: str, client: MlflowClient) -> None:
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="boom")
    logger.end_run(status="FAILED")
    assert client.get_run(run_id).info.status == "FAILED"


def test_end_run_rejects_invalid_status_with_warning(tracking_uri: str) -> None:
    """An unknown status string is normalised to FINISHED with a warning, never raises."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    logger.end_run(status="GARBAGE")  # must not raise


def test_log_metric_before_start_run_is_safe(tracking_uri: str) -> None:
    """Calling log_metric without start_run is a silent no-op + warning."""
    logger = _build_logger(tracking_uri)
    logger.log_metric("loss", 0.5)  # must not raise


def test_end_phase_after_end_run_is_safe(tracking_uri: str) -> None:
    """End-of-life ordering robustness — a stale ctx never crashes the trainer."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    ctx = logger.start_phase(phase="p")
    logger.end_run()  # parent terminates first (unusual but possible on KeyboardInterrupt)
    logger.end_phase(ctx)  # must not raise
