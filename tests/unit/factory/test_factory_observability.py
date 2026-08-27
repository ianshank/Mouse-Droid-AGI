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
    # Evict the cached mlflow packages so the simulated-absent import is the
    # one that runs. Two deliberate choices here, both previously wrong:
    #
    # * ``monkeypatch.delitem``, not a raw ``sys.modules.pop`` -- pop is
    #   permanent for the whole session, and dropping the top-level ``mlflow``
    #   package while its submodules (``mlflow.store``, ...) stayed cached left
    #   the next real import half-initialised, so a later test opening a
    #   genuine store died with "module 'mlflow' has no attribute 'store'".
    # * ``mousedroid.training.observability.mlflow_logger`` is deliberately
    #   NOT evicted. It imports mlflow lazily inside ``__init__`` rather than
    #   at module scope, so evicting it does nothing for this simulation --
    #   but re-importing it rebinds the *package attribute*
    #   ``mousedroid.training.observability.mlflow_logger``, which
    #   ``monkeypatch.undo()`` does not restore. The next test then patches a
    #   different module object than ``build_experiment_logger`` resolves, and
    #   its patch silently does nothing.
    import sys

    for name in list(sys.modules):
        if name == "mlflow" or name.startswith("mlflow."):
            monkeypatch.delitem(sys.modules, name, raising=False)

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


def _assert_pinned(resolved: str, prefix: str, expected_name: str) -> None:
    """Assert ``resolved`` is ``prefix`` + an absolute path named ``expected_name``.

    Path-object checks rather than string suffix/substring ones:
    ``Path.resolve()`` renders with the OS-native separator, so a raw
    ``endswith("/mlruns")`` check is forward-slash-only and fails on Windows
    (a real CI failure this suite has already hit once).
    """
    assert resolved.startswith(prefix)
    path = Path(resolved[len(prefix) :])
    assert path.name == expected_name
    assert path.is_absolute()


def test_resolve_tracking_uri_pins_the_relative_sqlite_default() -> None:
    """The schema default is pinned to an absolute path, like ``file:`` already was.

    Both local-store schemes now get the same treatment. Pinning does not make
    the default CWD-*independent* (a process launched elsewhere still pins
    against its own CWD), but it makes the effective database path absolute
    and therefore reportable -- which is what lets
    ``experiment_logger_tracking_uri_resolved`` tell an operator which
    database a run actually landed in.
    """
    _assert_pinned(_resolve_tracking_uri("sqlite:///mlflow.db"), "sqlite:///", "mlflow.db")


def test_resolve_tracking_uri_is_idempotent() -> None:
    """Re-resolving an already-pinned URI is a no-op, for both local schemes.

    Guards the rebuild step: ``sqlite:///`` + an absolute POSIX path yields
    the canonical four-slash form, which must not gain a fifth slash on a
    second pass.
    """
    for raw in ("sqlite:///mlflow.db", "file:./mlruns"):
        once = _resolve_tracking_uri(raw)
        assert _resolve_tracking_uri(once) == once


def test_resolve_tracking_uri_leaves_remote_and_in_memory_uris_alone() -> None:
    """Remote URIs and in-memory sqlite URIs carry no filesystem path to pin.

    Resolving them would corrupt the URI -- ``sqlite:///:memory:`` would
    become a bogus relative-file path named ``:memory:``.
    """
    for inert in (
        "http://host:5000",
        "https://host:5000",
        "databricks",
        "sqlite://",
        "sqlite:///:memory:",
    ):
        assert _resolve_tracking_uri(inert) == inert


def test_resolve_tracking_uri_matches_schemes_case_insensitively() -> None:
    """URI schemes are case-insensitive per RFC 3986, and mlflow lowercases them.

    A configured ``FILE:``/``SQLITE:///`` still selects the local store
    downstream, so matching case-sensitively here would silently skip the
    pin for exactly those URIs.
    """
    _assert_pinned(_resolve_tracking_uri("FILE:./mlruns"), "file:", "mlruns")
    _assert_pinned(_resolve_tracking_uri("SQLite:///mlflow.db"), "sqlite:///", "mlflow.db")


def test_resolve_tracking_uri_still_resolves_file_uris_to_absolute_path() -> None:
    """Unchanged-behaviour check: the pre-existing ``file:`` path still works.

    Proves widening the resolver to cover sqlite did not disturb the legacy
    backend's contract. Complements
    ``test_relative_file_uri_is_resolved_to_absolute`` above (same property
    through the full factory + construction path) with a direct,
    dependency-free check of the resolver itself.
    """
    _assert_pinned(_resolve_tracking_uri("file:./mlruns"), "file:", "mlruns")
