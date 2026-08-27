"""Tests for factory.build_experiment_logger."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import (
    ExperimentLoggerConfig,
    ObservabilityConfig,
    Settings,
)
from mousedroid.factory import _resolve_tracking_uri, build_experiment_logger
from mousedroid.training.observability import (
    ExperimentLoggerProtocol,
    NoOpExperimentLogger,
)


def _settings_with_logger(**overrides: object) -> Settings:
    base = Settings(mock_hardware=True)
    return base.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(**overrides),  # type: ignore[arg-type]
            ),
        }
    )


def test_no_observability_block_returns_noop() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.observability is None
    logger = build_experiment_logger(cfg)
    assert isinstance(logger, NoOpExperimentLogger)
    assert isinstance(logger, ExperimentLoggerProtocol)


def test_backend_none_returns_noop() -> None:
    cfg = _settings_with_logger(backend="none")
    assert isinstance(build_experiment_logger(cfg), NoOpExperimentLogger)


def test_backend_mlflow_returns_mlflow_logger_when_extras_present(tmp_path: object) -> None:
    pytest.importorskip("mlflow")
    cfg = _settings_with_logger(
        backend="mlflow",
        tracking_uri=f"file:{tmp_path}/mlruns",
        experiment_name="test",
    )
    logger = build_experiment_logger(cfg)
    from mousedroid.training.observability.mlflow_logger import MlflowExperimentLogger

    assert isinstance(logger, MlflowExperimentLogger)


def test_backend_mlflow_degrades_to_noop_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If mlflow is not importable the factory degrades cleanly with a warning."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("simulated absent extras")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Ensure no cached mlflow_logger module masks the import path.
    import sys

    for name in list(sys.modules):
        if name.startswith("mousedroid.training.observability.mlflow_logger") or name == "mlflow":
            sys.modules.pop(name, None)

    cfg = _settings_with_logger(backend="mlflow")
    logger = build_experiment_logger(cfg)
    assert isinstance(logger, NoOpExperimentLogger)


def test_backend_mlflow_degrades_to_noop_on_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ImportError construction failure (bad URI / store init) degrades to NoOp.

    Observability is best-effort — a broken tracking store must never crash the run.
    """
    pytest.importorskip("mlflow")
    import mousedroid.training.observability.mlflow_logger as mod

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated tracking-store init failure")

    monkeypatch.setattr(mod, "MlflowExperimentLogger", _boom)
    cfg = _settings_with_logger(
        backend="mlflow", tracking_uri="file:/nonexistent/mlruns", experiment_name="t"
    )
    logger = build_experiment_logger(cfg)
    assert isinstance(logger, NoOpExperimentLogger)


def test_relative_file_uri_is_resolved_to_absolute(tmp_path: object) -> None:
    """A ``file:./mlruns`` URI is pinned to an absolute path before construction."""
    pytest.importorskip("mlflow")
    monkey_cwd = str(tmp_path)
    import os

    saved_cwd = os.getcwd()
    try:
        os.chdir(monkey_cwd)
        cfg = _settings_with_logger(
            backend="mlflow",
            tracking_uri="file:./mlruns",
            experiment_name="abs",
        )
        logger = build_experiment_logger(cfg)
        from mousedroid.training.observability.mlflow_logger import MlflowExperimentLogger

        assert isinstance(logger, MlflowExperimentLogger)
        # Internal attribute access is fine in tests; the contract is "absolute".
        assert logger._tracking_uri.startswith("file:")
        assert "mlruns" in logger._tracking_uri
        # Crucially, the URI does NOT contain a relative ``./``.
        assert "./mlruns" not in logger._tracking_uri
    finally:
        os.chdir(saved_cwd)


# ---------------------------------------------------------------------------
# _resolve_tracking_uri directly (F-034) -- pure function, zero mlflow
# dependency, so these need no pytest.importorskip and always run. Kept
# separate from test_relative_file_uri_is_resolved_to_absolute above, which
# proves the same file: resolution but through the full
# build_experiment_logger -> MlflowExperimentLogger path; these prove the
# resolver's own contract in isolation, cheaply, for every URI scheme it
# must handle. Moved here from tests/regression/test_f034_mlflow_sqlite_aqa.py
# per test-tier-mirror/SKILL.md: AQA is for schema properties, not behaviour.
# ---------------------------------------------------------------------------


def test_resolve_tracking_uri_passes_sqlite_uris_through_unchanged() -> None:
    """The mlflow sqlite default's scheme is not mistaken for a file: URI.

    ``_resolve_tracking_uri`` only resolves ``file:`` URIs to an absolute
    path (so they survive a trainer's ``chdir()``); every other scheme,
    including the schema default, must pass through byte-identical.
    """
    assert _resolve_tracking_uri("sqlite:///mlflow.db") == "sqlite:///mlflow.db"


def test_resolve_tracking_uri_passes_http_and_absolute_sqlite_uris_through() -> None:
    """Belt-and-braces: remote and absolute-path URI schemes are also inert."""
    assert _resolve_tracking_uri("http://host:5000") == "http://host:5000"
    assert _resolve_tracking_uri("sqlite:////opt/mousedroid/mlflow.db") == (
        "sqlite:////opt/mousedroid/mlflow.db"
    )


def test_resolve_tracking_uri_still_resolves_file_uris_to_absolute_path() -> None:
    """Unchanged behaviour check: the file: URI resolution path still works.

    Proves the sqlite-default change did not accidentally touch this
    function's pre-existing behaviour for operators who explicitly opt
    back into the legacy file-store backend. Complements
    test_relative_file_uri_is_resolved_to_absolute above (which proves the
    same thing through the full factory+construction path) with a direct,
    dependency-free check of the resolver itself.
    """
    resolved = _resolve_tracking_uri("file:./mlruns")
    assert resolved.startswith("file:")
    # Path(...).name/.is_absolute() rather than string suffix/substring
    # checks -- Path.resolve() renders with the OS-native separator, so a
    # raw `resolved.endswith("/mlruns")` check is forward-slash-only and
    # fails on Windows, where the resolved path uses backslashes.
    resolved_path = Path(resolved.removeprefix("file:"))
    assert resolved_path.name == "mlruns"
    assert resolved_path.is_absolute()
