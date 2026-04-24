"""Validation tests for :class:`GCPConfig` tightened fields.

These tests guard the pydantic ``model_validator`` that was added to
reject silently-empty cloud identifiers and the new ``metrics_labels``
field. Empty topic / bucket / robot IDs would otherwise fan out into
Pub/Sub publish attempts against empty paths at runtime.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import GCPConfig, GCPPubSubConfig, GCPStorageConfig


def _base_kwargs() -> dict[str, object]:
    """Return kwargs that form a valid GCPConfig."""
    return {"project_id": "proj", "robot_id": "droid-001"}


def test_defaults_are_valid() -> None:
    cfg = GCPConfig(**_base_kwargs())
    assert cfg.project_id == "proj"
    assert cfg.metrics_labels == {}


def test_empty_project_id_rejected() -> None:
    with pytest.raises(ValidationError):
        GCPConfig(project_id="", robot_id="droid-001")


def test_whitespace_robot_id_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        GCPConfig(project_id="proj", robot_id="  ")
    assert "robot_id" in str(excinfo.value)


def test_empty_telemetry_topic_rejected() -> None:
    pubsub = GCPPubSubConfig(telemetry_topic="", experience_topic="exp")
    with pytest.raises(ValidationError) as excinfo:
        GCPConfig(**_base_kwargs(), pubsub=pubsub)
    assert "pubsub.telemetry_topic" in str(excinfo.value)


def test_duplicate_topics_rejected() -> None:
    pubsub = GCPPubSubConfig(telemetry_topic="same", experience_topic="same")
    with pytest.raises(ValidationError) as excinfo:
        GCPConfig(**_base_kwargs(), pubsub=pubsub)
    assert "must differ" in str(excinfo.value)


def test_empty_bucket_rejected() -> None:
    storage = GCPStorageConfig(bucket="")
    with pytest.raises(ValidationError) as excinfo:
        GCPConfig(**_base_kwargs(), storage=storage)
    assert "storage.bucket" in str(excinfo.value)


def test_metrics_labels_accept_strings() -> None:
    cfg = GCPConfig(
        **_base_kwargs(),
        metrics_labels={"env": "prod", "region": "us-west1"},
    )
    assert cfg.metrics_labels["env"] == "prod"


def test_metrics_labels_reject_empty_value() -> None:
    with pytest.raises(ValidationError):
        GCPConfig(**_base_kwargs(), metrics_labels={"env": ""})


def test_metrics_labels_reject_non_string_value() -> None:
    with pytest.raises(ValidationError):
        GCPConfig(**_base_kwargs(), metrics_labels={"env": 123})  # type: ignore[dict-item]


def test_backwards_compatible_default_dict_factory() -> None:
    """Two instances must have independent label dicts (no shared mutable)."""
    a = GCPConfig(**_base_kwargs())
    b = GCPConfig(**_base_kwargs())
    a.metrics_labels["x"] = "y"
    assert "x" not in b.metrics_labels
