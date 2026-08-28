"""Integration: the sqlite tracking backend actually works end to end (F-034).

Closes the gap that made F-034's own CI job vacuous. The ``mlflow-extras``
job installs ``sqlalchemy`` and ``alembic`` *specifically* because the
default ``tracking_uri`` is now ``sqlite:///...``, yet every other
mlflow-touching test drives a ``file:`` URI — so nothing exercised the
sqlite store and an mlflow/SQLAlchemy/Alembic API drift would have left
the job green. These tests open a real SQLite database through the real
``MlflowExperimentLogger`` and assert data round-trips.

Integration tier per ``.claude/skills/test-tier-mirror/SKILL.md``: several
modules wired together *through the factory* (``build_experiment_logger``),
plus a real backing store. Unit-tier coverage of ``_resolve_tracking_uri``'s
pure string behaviour lives in ``tests/unit/factory/test_factory_observability.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlflow")
# The sqlite tracking store is what this module exists to exercise; without
# these two, MlflowClient raises UnsupportedModelRegistryStoreURIException
# for any sqlite:/// URI (this is the exact failure F-034 added them for).
pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from mlflow import MlflowClient

from mousedroid.config.schema import (
    ExperimentLoggerConfig,
    ObservabilityConfig,
    Settings,
)
from mousedroid.factory import build_experiment_logger
from mousedroid.training.observability.mlflow_logger import MlflowExperimentLogger


@pytest.fixture
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test body with the process CWD inside ``tmp_path``.

    The default ``tracking_uri`` is CWD-relative, so exercising it honestly
    means controlling the working directory rather than rewriting the URI.

    ``monkeypatch.chdir`` rather than a hand-rolled ``os.chdir`` +
    try/finally: pytest restores the original directory during its own
    teardown phase, which keeps the process CWD stable for anything that
    writes relative paths at session scope (coverage data files among them).
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _settings_with_uri(tracking_uri: str) -> Settings:
    return Settings(mock_hardware=True).model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    tracking_uri=tracking_uri,
                    experiment_name="sqlite-backend-test",
                ),
            ),
        }
    )


def test_sqlite_backend_round_trips_a_run_through_the_factory(tmp_path: Path) -> None:
    """A run logged via the factory-built logger is readable back from SQLite.

    The end-to-end proof that sqlalchemy + alembic are wired correctly:
    mlflow creates and migrates the schema on first connect, and the run's
    params/metrics survive a fresh client reading the same database file.
    """
    db_path = tmp_path / "mlflow.db"
    logger = build_experiment_logger(_settings_with_uri(f"sqlite:///{db_path}"))
    assert isinstance(logger, MlflowExperimentLogger), (
        "factory degraded to NoOp -- the sqlite store failed to initialise"
    )

    run_id = logger.start_run(run_name="round-trip", params={"lr": "0.001"})
    assert run_id, "start_run returned an empty run-id (backend write failed)"
    logger.log_metric("loss", 0.25, step=1)
    logger.end_run()

    assert db_path.exists(), "no SQLite database file was created"

    # A *fresh* client, so this reads from the database rather than any
    # in-process state the logger may still hold.
    reader = MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    run = reader.get_run(run_id)
    assert run.data.params["lr"] == "0.001"
    assert run.data.metrics["loss"] == pytest.approx(0.25)
    assert run.info.status == "FINISHED"


def test_sqlite_phase_runs_nest_under_the_parent_run(tmp_path: Path) -> None:
    """Nested phase runs persist their parent tag through the sqlite store."""
    db_path = tmp_path / "phases.db"
    logger = build_experiment_logger(_settings_with_uri(f"sqlite:///{db_path}"))
    assert isinstance(logger, MlflowExperimentLogger)

    parent_id = logger.start_run(run_name="pipeline")
    ctx = logger.start_phase(phase="rssm", params={"epochs": "3"})
    logger.log_phase_metric(ctx, "phase_loss", 1.5, step=0)
    logger.end_phase(ctx)
    logger.end_run()

    reader = MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    child = reader.get_run(ctx.run_id)
    assert child.data.tags["mlflow.parentRunId"] == parent_id
    assert child.data.tags["phase"] == "rssm"
    assert child.data.metrics["phase_loss"] == pytest.approx(1.5)


def test_schema_default_resolves_to_an_absolute_db_under_the_launch_cwd(
    tmp_cwd: Path,
) -> None:
    """The CWD-relative default is pinned to an absolute path of the launch CWD.

    Scope note, deliberately narrow because the obvious stronger claim is
    false: pinning does NOT make the default CWD-independent. Two processes
    launched from different directories still get two different databases —
    each pins against its own CWD — and mlflow additionally caches its store
    per URI string, so an in-process ``chdir()`` is invisible either way.
    (Both behaviours were verified directly rather than assumed.)

    What the pin genuinely buys, and what this test therefore asserts, is
    that the *effective* database path is absolute and knowable: it is what
    ``experiment_logger_tracking_uri_resolved`` reports, which is how an
    operator hitting "no runs visible" identifies which database a given run
    actually went to. The operator-facing cure for the split itself is an
    absolute ``tracking_uri``, which the runbook now recommends.

    Uses the schema default (no ``tracking_uri`` override) so this breaks if
    the default ever changes shape.
    """
    settings = Settings(mock_hardware=True).model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    experiment_name="default-uri-test",
                ),
            ),
        }
    )
    assert settings.observability is not None
    assert settings.observability.experiment_logger.tracking_uri == "sqlite:///mlflow.db"

    logger = build_experiment_logger(settings)
    assert isinstance(logger, MlflowExperimentLogger)
    run_id = logger.start_run(run_name="default-uri")
    assert run_id
    logger.log_metric("value", 7.0, step=1)
    logger.end_run()

    # Pinned: the logger carries an absolute URI, not the relative literal.
    assert logger._tracking_uri != "sqlite:///mlflow.db"
    assert logger._tracking_uri.startswith("sqlite:///")
    pinned_path = Path(logger._tracking_uri[len("sqlite:///") :])
    assert pinned_path.is_absolute()
    # ...and it names the real file, under the CWD the factory ran in.
    assert pinned_path == (tmp_cwd / "mlflow.db").resolve()
    assert pinned_path.exists()

    reader = MlflowClient(tracking_uri=logger._tracking_uri)
    assert reader.get_run(run_id).data.metrics["value"] == pytest.approx(7.0)
