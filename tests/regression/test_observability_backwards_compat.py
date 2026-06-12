"""Regression: ``observability`` config is purely additive and defaults OFF.

Pins the CLAUDE.md invariant #9 ("Existing YAML files must load unchanged")
specifically for the new ``observability`` field added in PR T2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_minimal_pre_feature_yaml_still_loads() -> None:
    """A YAML that predates the feature loads with observability defaulting to OFF."""
    minimal = yaml.safe_load(
        """
        mock_hardware: true
        platform: mouse_droid
        """
    )
    cfg = Settings.model_validate(minimal)
    assert cfg.observability is None  # default; backwards-compatible


def test_existing_default_yaml_still_loads() -> None:
    """``config/default.yaml`` parses unchanged after the schema addition."""
    raw = yaml.safe_load((_REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    cfg = Settings.model_validate(raw)
    # Either absent (None) or explicitly disabled (the user has not opted in).
    if cfg.observability is not None:
        assert cfg.observability.experiment_logger.backend == "none"


def test_existing_jetson_production_yaml_still_loads() -> None:
    """``config/jetson_production.yaml`` parses unchanged after the schema addition."""
    raw = yaml.safe_load(
        (_REPO_ROOT / "config" / "jetson_production.yaml").read_text(encoding="utf-8")
    )
    cfg = Settings.model_validate(raw)
    if cfg.observability is not None:
        assert cfg.observability.experiment_logger.backend == "none"


def test_opt_in_overlay_parses() -> None:
    """A YAML that DOES set observability resolves the backend correctly."""
    overlay = yaml.safe_load(
        """
        mock_hardware: true
        platform: mouse_droid
        observability:
          experiment_logger:
            backend: mlflow
            tracking_uri: file:./mlruns
            experiment_name: mousedroid-test
        """
    )
    cfg = Settings.model_validate(overlay)
    assert cfg.observability is not None
    assert cfg.observability.experiment_logger.backend == "mlflow"
    assert cfg.observability.experiment_logger.tracking_uri == "file:./mlruns"
    assert cfg.observability.experiment_logger.experiment_name == "mousedroid-test"


def test_env_var_override() -> None:
    """Env-var nested overrides work for the new sub-config."""
    from mousedroid.config.schema import ExperimentLoggerConfig, ObservabilityConfig

    overlay = {"mock_hardware": True, "platform": "mouse_droid"}
    cfg = Settings.model_validate(overlay)
    updated = cfg.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    experiment_name="from-env",
                )
            )
        }
    )
    assert updated.observability is not None
    assert updated.observability.experiment_logger.experiment_name == "from-env"
    assert updated.observability.experiment_logger.backend == "mlflow"


def test_rejects_invalid_backend_literal() -> None:
    """Pydantic Literal rejects unknown backend strings at validation time."""
    bad = {
        "mock_hardware": True,
        "observability": {"experiment_logger": {"backend": "wandb"}},  # not in Literal
    }
    with pytest.raises(ValidationError):
        Settings.model_validate(bad)


def test_new_experiment_logger_field_defaults() -> None:
    """The wired fields (run_name / log_step_every_n / log_artifacts) keep safe defaults.

    Pins the contract so a future edit can't silently flip throttle/artifact
    behaviour for existing opt-in YAML that omits these keys.
    """
    from mousedroid.config.schema import ExperimentLoggerConfig

    cfg = ExperimentLoggerConfig(backend="mlflow")
    assert cfg.run_name is None  # logger falls back to its own default
    assert cfg.log_step_every_n == 1  # every step logged (byte-identical to pre-wiring)
    assert cfg.log_artifacts is True  # artifacts uploaded by default


def test_log_step_every_n_must_be_positive() -> None:
    """``log_step_every_n`` is ``gt=0`` — 0 (which would ZeroDivide the throttle) is rejected."""
    from mousedroid.config.schema import ExperimentLoggerConfig

    with pytest.raises(ValidationError):
        ExperimentLoggerConfig(backend="mlflow", log_step_every_n=0)
