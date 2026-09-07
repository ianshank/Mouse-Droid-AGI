"""Automated Quality Assurance (AQA) — schema + allowlist hygiene for F-039."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.cloud.logging_sink import (
    CLOUD_LOG_ALLOWED_KEYS,
    CLOUD_LOG_REDACTED_KEYS,
    CloudLoggingSink,
)
from mousedroid.config.schema.gcp_cloud import GCPLoggingConfig


def test_queue_maxsize_has_description() -> None:
    """``GCPLoggingConfig.queue_maxsize`` carries a non-empty description."""
    info: FieldInfo = GCPLoggingConfig.model_fields["queue_maxsize"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_queue_maxsize_default_is_256() -> None:
    """Pinned off FieldInfo, not a live instance."""
    info: FieldInfo = GCPLoggingConfig.model_fields["queue_maxsize"]
    assert info.default == 256


def test_queue_maxsize_zero_raises_at_load() -> None:
    """A zero-depth queue is rejected at YAML-load time."""
    with pytest.raises(ValidationError, match="queue_maxsize"):
        GCPLoggingConfig(queue_maxsize=0)


def test_drain_timeout_s_has_description() -> None:
    info: FieldInfo = GCPLoggingConfig.model_fields["drain_timeout_s"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_drain_timeout_s_default_is_five() -> None:
    info: FieldInfo = GCPLoggingConfig.model_fields["drain_timeout_s"]
    assert info.default == 5.0


def test_queue_get_timeout_s_has_description() -> None:
    info: FieldInfo = GCPLoggingConfig.model_fields["queue_get_timeout_s"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_queue_get_timeout_s_default_is_point_two() -> None:
    info: FieldInfo = GCPLoggingConfig.model_fields["queue_get_timeout_s"]
    assert info.default == 0.2


def test_drain_timeout_s_zero_raises_at_load() -> None:
    with pytest.raises(ValidationError, match="drain_timeout_s"):
        GCPLoggingConfig(drain_timeout_s=0.0)


def test_queue_get_timeout_s_zero_raises_at_load() -> None:
    with pytest.raises(ValidationError, match="queue_get_timeout_s"):
        GCPLoggingConfig(queue_get_timeout_s=0.0)


def test_cloud_log_allowlist_is_operational_only() -> None:
    """Mission/NL keys must not be in the Cloud Logging allowlist."""
    forbidden = {
        "nl_command",
        "mission",
        "query",
        "command",
        "prompt",
        "text",
        "user_content",
        "nl",
        "content",
        "raw_text",
    }
    assert forbidden <= CLOUD_LOG_REDACTED_KEYS
    assert forbidden.isdisjoint(CLOUD_LOG_ALLOWED_KEYS)
    assert "loop_time_ms" in CLOUD_LOG_ALLOWED_KEYS
    assert "emergency" in CLOUD_LOG_ALLOWED_KEYS
    assert "robot_id" in CLOUD_LOG_ALLOWED_KEYS


def test_allowlist_is_module_frozenset_not_config() -> None:
    """Operators must not be able to widen the allowlist from YAML."""
    assert "cloud_log_allowed_keys" not in GCPLoggingConfig.model_fields
    assert isinstance(CLOUD_LOG_ALLOWED_KEYS, frozenset)
    assert isinstance(CLOUD_LOG_REDACTED_KEYS, frozenset)


def test_sink_exposes_flush_and_drop_count() -> None:
    """Queue drain and drop counter are real methods/properties."""
    assert callable(CloudLoggingSink.flush)
    assert isinstance(CloudLoggingSink.drop_count, property)
    assert isinstance(CloudLoggingSink.forward_failure_count, property)
