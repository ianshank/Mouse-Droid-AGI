"""Tests for MlflowExperimentLogger — uses a real MlflowClient over tmp_path.

Per the writing-plans research, mocking ``MlflowClient`` itself loses the
ability to catch signature drift on MLflow upgrades. The right pattern is
a tmp_path-rooted file backend and a real client.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog.testing

pytest.importorskip("mlflow")  # skip module entirely if extras missing
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
    """NaN must not reach the store; the warning is recorded by to_finite_float."""
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


def test_end_run_rejects_invalid_status_with_warning(
    tracking_uri: str, client: MlflowClient
) -> None:
    """An unknown status string is normalised to FINISHED with a warning, never raises."""
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="x")
    logger.end_run(status="GARBAGE")  # must not raise
    assert client.get_run(run_id).info.status == "FINISHED"  # normalised, not GARBAGE


def test_log_metric_before_start_run_is_safe(tracking_uri: str) -> None:
    """Calling log_metric without start_run is a silent no-op + warning."""
    logger = _build_logger(tracking_uri)
    with structlog.testing.capture_logs() as logs:
        logger.log_metric("loss", 0.5)  # must not raise
    warn_logs = [e for e in logs if e["event"] == "mlflow_logger_log_metric_without_run"]
    assert len(warn_logs) == 1
    assert warn_logs[0]["log_level"] == "warning"
    assert warn_logs[0]["key"] == "loss"


def test_end_phase_after_end_run_is_safe(tracking_uri: str, client: MlflowClient) -> None:
    """End-of-life ordering robustness — a stale ctx never crashes the trainer."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    ctx = logger.start_phase(phase="p")
    logger.end_run()  # parent terminates first (unusual but possible on KeyboardInterrupt)
    logger.end_phase(ctx)  # must not raise
    # The child run itself must still be genuinely terminated, not just "didn't crash".
    assert client.get_run(ctx.run_id).info.status == "FINISHED"


# ---------------------------------------------------------------------------
# Blocker 2: reachable-path coverage
# ---------------------------------------------------------------------------


def test_resolve_existing_experiment_returns_same_id(tracking_uri: str) -> None:
    """Second construction with the same experiment_name reuses the existing id."""
    logger1 = _build_logger(tracking_uri, "shared-exp")
    logger2 = _build_logger(tracking_uri, "shared-exp")
    assert logger1._experiment_id == logger2._experiment_id


def test_resolve_experiment_handles_create_race(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost create-race (concurrent process created the experiment) re-resolves it."""

    class _Exp:
        experiment_id = "raced-id"

    logger = _build_logger(tracking_uri, "race-base")
    calls = {"n": 0}

    def _get(_name: str) -> object | None:
        calls["n"] += 1
        return None if calls["n"] == 1 else _Exp()  # absent, then present (other writer)

    def _create(_name: str) -> str:
        raise RuntimeError("experiment already exists")  # the race

    monkeypatch.setattr(logger._client, "get_experiment_by_name", _get)
    monkeypatch.setattr(logger._client, "create_experiment", _create)
    assert logger._resolve_or_create_experiment("raced") == "raced-id"


def test_resolve_experiment_reraises_genuine_failure(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine store failure (still absent after retry) re-raises → factory degrades to NoOp."""
    logger = _build_logger(tracking_uri, "fail-base")
    monkeypatch.setattr(logger._client, "get_experiment_by_name", lambda _name: None)

    def _create(_name: str) -> str:
        raise RuntimeError("tracking store unreachable")

    monkeypatch.setattr(logger._client, "create_experiment", _create)
    with pytest.raises(RuntimeError, match="tracking store unreachable"):
        logger._resolve_or_create_experiment("x")


def test_log_params_before_start_run_is_safe(tracking_uri: str) -> None:
    """log_params without an active run is a silent no-op + warning, never raises."""
    logger = _build_logger(tracking_uri)
    with structlog.testing.capture_logs() as logs:
        logger.log_params({"a": 1, "b": "two"})  # must not raise
    warn_logs = [e for e in logs if e["event"] == "mlflow_logger_log_params_without_run"]
    assert len(warn_logs) == 1
    assert warn_logs[0]["log_level"] == "warning"


def test_log_artifact_before_start_run_is_safe(tracking_uri: str) -> None:
    """log_artifact without an active run is a silent no-op + warning."""
    logger = _build_logger(tracking_uri)
    with structlog.testing.capture_logs() as logs:
        logger.log_artifact("/nonexistent/path/file.txt")  # must not raise
    warn_logs = [e for e in logs if e["event"] == "mlflow_logger_log_artifact_without_run"]
    assert len(warn_logs) == 1
    assert warn_logs[0]["log_level"] == "warning"
    assert warn_logs[0]["path"] == "/nonexistent/path/file.txt"


def test_log_artifact_missing_file_is_safe(tracking_uri: str, client: MlflowClient) -> None:
    """log_artifact with a non-existent file path logs a warning and returns."""
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="artifact-miss")
    with structlog.testing.capture_logs() as logs:
        logger.log_artifact("/path/does/not/exist.txt")  # must not raise
    logger.end_run()
    warn_logs = [e for e in logs if e["event"] == "mlflow_logger_artifact_missing"]
    assert len(warn_logs) == 1
    assert warn_logs[0]["log_level"] == "warning"
    assert warn_logs[0]["path"] == "/path/does/not/exist.txt"
    # "and returns" — no upload was ever attempted against the real backend.
    assert client.list_artifacts(run_id) == []


def test_log_artifact_uploads_real_file(
    tracking_uri: str, client: MlflowClient, tmp_path: Path
) -> None:
    """log_artifact with an existing file uploads it; list_artifacts confirms."""
    artifact_file = tmp_path / "model_card.txt"
    artifact_file.write_text("hello artifact")
    logger = _build_logger(tracking_uri)
    run_id = logger.start_run(run_name="artifact-ok")
    logger.log_artifact(str(artifact_file))
    logger.end_run()
    artifacts = client.list_artifacts(run_id)
    names = [a.path for a in artifacts]
    assert "model_card.txt" in names


def test_end_run_without_start_is_safe(tracking_uri: str) -> None:
    """end_run with no active run is a silent no-op."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    with patch.object(logger._client, "set_terminated") as mock_terminated:
        logger.end_run()  # must not raise
    # "no-op" means the backend is never touched, not just "did not raise".
    mock_terminated.assert_not_called()
    assert logger._active_run_id is None


def test_start_phase_without_parent_returns_empty_ctx(tracking_uri: str) -> None:
    """start_phase without a parent run returns a PhaseContext with run_id==""."""
    logger = _build_logger(tracking_uri)
    ctx = logger.start_phase(phase="orphan")
    assert ctx.run_id == ""
    assert ctx.phase == "orphan"


def test_log_phase_metric_with_empty_ctx_is_safe(tracking_uri: str) -> None:
    """log_phase_metric with an empty-id ctx is a silent no-op."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    ctx = PhaseContext(run_id="", phase="x")
    with patch.object(logger._client, "log_metric") as mock_log_metric:
        logger.log_phase_metric(ctx, "loss", 0.5)  # must not raise
    mock_log_metric.assert_not_called()


def test_log_phase_metric_skips_nan(tracking_uri: str, client: MlflowClient) -> None:
    """NaN is skipped by to_finite_float; only the finite value is stored."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="nan-phase")
    ctx = logger.start_phase(phase="nan-ph")
    logger.log_phase_metric(ctx, "loss", float("nan"), step=0)
    logger.log_phase_metric(ctx, "loss", 0.42, step=1)
    logger.end_phase(ctx)
    logger.end_run()
    history = client.get_metric_history(ctx.run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(1, 0.42)]


def test_log_phase_artifact_with_empty_ctx_is_safe(tracking_uri: str) -> None:
    """log_phase_artifact with an empty-id ctx is a silent no-op."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    ctx = PhaseContext(run_id="", phase="x")
    with patch.object(logger._client, "log_artifact") as mock_log_artifact:
        logger.log_phase_artifact(ctx, "/nonexistent/file.txt")  # must not raise
    mock_log_artifact.assert_not_called()


def test_log_phase_artifact_missing_file_is_safe(tracking_uri: str) -> None:
    """log_phase_artifact with a missing file logs a warning and returns."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="pa-miss")
    ctx = logger.start_phase(phase="miss-ph")
    with structlog.testing.capture_logs() as logs:
        logger.log_phase_artifact(ctx, "/no/such/file.bin")  # must not raise
    logger.end_phase(ctx)
    logger.end_run()
    warn_logs = [e for e in logs if e["event"] == "mlflow_logger_phase_artifact_missing"]
    assert len(warn_logs) == 1
    assert warn_logs[0]["log_level"] == "warning"
    assert warn_logs[0]["path"] == "/no/such/file.bin"


def test_log_phase_artifact_uploads_real_file(
    tracking_uri: str, client: MlflowClient, tmp_path: Path
) -> None:
    """log_phase_artifact with an existing file uploads it; list_artifacts confirms."""
    artifact_file = tmp_path / "phase_weights.pt"
    artifact_file.write_bytes(b"\x00\x01\x02")
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="pa-ok")
    ctx = logger.start_phase(phase="upload-ph")
    logger.log_phase_artifact(ctx, str(artifact_file))
    logger.end_phase(ctx)
    logger.end_run()
    artifacts = client.list_artifacts(ctx.run_id)
    names = [a.path for a in artifacts]
    assert "phase_weights.pt" in names


def test_end_phase_invalid_status_normalised(tracking_uri: str, client: MlflowClient) -> None:
    """An unknown status to end_phase is normalised to FINISHED with a warning."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="norm")
    ctx = logger.start_phase(phase="norm-ph")
    logger.end_phase(ctx, status="GARBAGE")  # must not raise
    child = client.get_run(ctx.run_id)
    assert child.info.status == "FINISHED"
    logger.end_run()


def test_start_phase_with_params_logs_them(tracking_uri: str, client: MlflowClient) -> None:
    """start_phase with params= logs the params on the child run immediately."""
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="with-params")
    ctx = logger.start_phase(phase="ph-with-params", params={"lr": 0.001, "epochs": 10})
    child = client.get_run(ctx.run_id)
    assert child.data.params["lr"] == "0.001"
    assert child.data.params["epochs"] == "10"
    logger.end_phase(ctx)
    logger.end_run()


def test_end_phase_with_empty_ctx_is_safe(tracking_uri: str) -> None:
    """end_phase with an empty-id ctx (from a failed start_phase) is a no-op."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    ctx = PhaseContext(run_id="", phase="orphan")
    with patch.object(logger._client, "set_terminated") as mock_terminated:
        logger.end_phase(ctx)  # must not raise
    mock_terminated.assert_not_called()


# ---------------------------------------------------------------------------
# Exception-path coverage — each except branch is exercised via patch.object
# on the internal _client so all other methods still use the real client.
# ---------------------------------------------------------------------------


def test_start_run_backend_failure_returns_empty_string(tracking_uri: str) -> None:
    """start_run except branch: returns "" and does not raise on client error."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    with patch.object(logger._client, "create_run", side_effect=RuntimeError("boom")):
        result = logger.start_run(run_name="fail")
    assert result == ""
    assert logger._active_run_id is None


def test_log_params_backend_failure_is_safe(tracking_uri: str) -> None:
    """log_params except branch: does not raise when log_param raises."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    with (
        patch.object(logger._client, "log_param", side_effect=RuntimeError("net")),
        structlog.testing.capture_logs() as logs,
    ):
        logger.log_params({"k": "v"})  # must not raise
    logger.end_run()
    failure_logs = [e for e in logs if e["event"] == "mlflow_logger_log_param_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["key"] == "k"
    assert failure_logs[0]["error_type"] == "RuntimeError"


def test_log_metric_backend_failure_is_safe(tracking_uri: str) -> None:
    """log_metric except branch: does not raise when log_metric raises."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    with (
        patch.object(logger._client, "log_metric", side_effect=RuntimeError("net")),
        structlog.testing.capture_logs() as logs,
    ):
        logger.log_metric("loss", 0.5)  # must not raise
    logger.end_run()
    failure_logs = [e for e in logs if e["event"] == "mlflow_logger_log_metric_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["key"] == "loss"
    assert failure_logs[0]["error_type"] == "RuntimeError"


def test_log_artifact_backend_failure_is_safe(tracking_uri: str, tmp_path: Path) -> None:
    """log_artifact except branch: does not raise when log_artifact raises."""
    from unittest.mock import patch

    artifact = tmp_path / "ckpt.txt"
    artifact.write_text("x")
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    with (
        patch.object(logger._client, "log_artifact", side_effect=RuntimeError("disk")),
        structlog.testing.capture_logs() as logs,
    ):
        logger.log_artifact(str(artifact))  # must not raise
    logger.end_run()
    failure_logs = [e for e in logs if e["event"] == "mlflow_logger_log_artifact_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["path"] == str(artifact)
    assert failure_logs[0]["error_type"] == "RuntimeError"


def test_end_run_backend_failure_still_clears_active_run(tracking_uri: str) -> None:
    """end_run except branch: _active_run_id is cleared even when set_terminated raises."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    with patch.object(logger._client, "set_terminated", side_effect=RuntimeError("term")):
        logger.end_run()  # must not raise
    assert logger._active_run_id is None


def test_start_phase_backend_failure_returns_empty_ctx(tracking_uri: str) -> None:
    """start_phase except branch: returns empty PhaseContext when create_run raises."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    with patch.object(logger._client, "create_run", side_effect=RuntimeError("net")):
        ctx = logger.start_phase(phase="rssm")
    assert ctx.run_id == ""
    assert ctx.phase == "rssm"
    logger.end_run()


def test_start_phase_param_logging_failure_is_safe(tracking_uri: str) -> None:
    """start_phase param-logging except branch: returns valid ctx even if log_param fails."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    with patch.object(logger._client, "log_param", side_effect=RuntimeError("net")):
        ctx = logger.start_phase(phase="rssm", params={"lr": 0.001})  # must not raise
    assert ctx.run_id != ""
    logger.end_phase(ctx)
    logger.end_run()


def test_log_phase_metric_backend_failure_is_safe(tracking_uri: str) -> None:
    """log_phase_metric except branch: does not raise when log_metric raises."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    ctx = logger.start_phase(phase="rssm")
    with (
        patch.object(logger._client, "log_metric", side_effect=RuntimeError("net")),
        structlog.testing.capture_logs() as logs,
    ):
        logger.log_phase_metric(ctx, "loss", 0.5)  # must not raise
    logger.end_phase(ctx)
    logger.end_run()
    failure_logs = [e for e in logs if e["event"] == "mlflow_logger_log_phase_metric_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["phase"] == "rssm"
    assert failure_logs[0]["key"] == "loss"
    assert failure_logs[0]["error_type"] == "RuntimeError"


def test_log_phase_artifact_backend_failure_is_safe(tracking_uri: str, tmp_path: Path) -> None:
    """log_phase_artifact except branch: does not raise when log_artifact raises."""
    from unittest.mock import patch

    artifact = tmp_path / "ph.txt"
    artifact.write_text("x")
    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    ctx = logger.start_phase(phase="rssm")
    with (
        patch.object(logger._client, "log_artifact", side_effect=RuntimeError("disk")),
        structlog.testing.capture_logs() as logs,
    ):
        logger.log_phase_artifact(ctx, str(artifact))  # must not raise
    logger.end_phase(ctx)
    logger.end_run()
    failure_logs = [e for e in logs if e["event"] == "mlflow_logger_log_phase_artifact_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["phase"] == "rssm"
    assert failure_logs[0]["error_type"] == "RuntimeError"


def test_end_phase_backend_failure_is_safe(tracking_uri: str) -> None:
    """end_phase except branch: does not raise when set_terminated raises."""
    from unittest.mock import patch

    logger = _build_logger(tracking_uri)
    logger.start_run(run_name="x")
    ctx = logger.start_phase(phase="rssm")
    with (
        patch.object(logger._client, "set_terminated", side_effect=RuntimeError("term")),
        structlog.testing.capture_logs() as logs,
    ):
        logger.end_phase(ctx)  # must not raise
    logger.end_run()
    failure_logs = [e for e in logs if e["event"] == "mlflow_logger_end_phase_failed"]
    assert len(failure_logs) == 1
    assert failure_logs[0]["log_level"] == "warning"
    assert failure_logs[0]["phase"] == "rssm"
    assert failure_logs[0]["error_type"] == "RuntimeError"


# ---- mlflow 3.x file-store compat guard (Sprint 3 / F-026) ----------------


def test_mlflow_allow_file_store_set_on_import() -> None:
    """Module-level guard sets MLFLOW_ALLOW_FILE_STORE unconditionally."""
    import os

    # The module has already been imported (top of this file), so the
    # setdefault has already fired.
    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE") == "true"
