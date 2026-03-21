"""Tests for OpenClawConfig schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import OpenClawConfig, Settings

# -- Defaults ------------------------------------------------------------------


def test_defaults():
    c = OpenClawConfig()
    assert c.enabled is False
    assert c.api_endpoint == "http://localhost:8000"
    assert c.api_timeout_s == 5.0
    assert c.api_key is None
    assert c.goal_mode == "velocity"
    assert c.max_action_age_ms == 100.0
    assert c.poll_interval_s == 1.0
    assert c.fallback_to_cognitive is True
    assert c.connect_retries == 3
    assert c.connect_backoff_base == 2.0
    assert c.ws_enabled is False
    assert c.ws_reconnect_interval_s == 5.0


def test_observation_keys_default():
    c = OpenClawConfig()
    assert "distance_m" in c.observation_keys
    assert "battery_v" in c.observation_keys
    assert "safety" in c.observation_keys


# -- Validation ----------------------------------------------------------------


def test_api_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        OpenClawConfig(api_timeout_s=0)


def test_api_timeout_negative_rejected():
    with pytest.raises(ValidationError):
        OpenClawConfig(api_timeout_s=-1.0)


def test_max_action_age_must_be_positive():
    with pytest.raises(ValidationError):
        OpenClawConfig(max_action_age_ms=0)


def test_poll_interval_must_be_positive():
    with pytest.raises(ValidationError):
        OpenClawConfig(poll_interval_s=-0.5)


def test_connect_retries_must_be_positive():
    with pytest.raises(ValidationError):
        OpenClawConfig(connect_retries=0)


def test_goal_mode_valid_values():
    for mode in ("action", "velocity", "waypoint"):
        c = OpenClawConfig(goal_mode=mode)
        assert c.goal_mode == mode


def test_goal_mode_invalid():
    with pytest.raises(ValidationError):
        OpenClawConfig(goal_mode="invalid")


# -- Settings integration -----------------------------------------------------


def test_settings_has_openclaw_field():
    s = Settings(mock_hardware=True)
    assert hasattr(s, "openclaw")
    assert isinstance(s.openclaw, OpenClawConfig)


def test_settings_openclaw_disabled_by_default():
    s = Settings(mock_hardware=True)
    assert s.openclaw.enabled is False


def test_settings_openclaw_custom():
    s = Settings(
        mock_hardware=True,
        openclaw=OpenClawConfig(
            enabled=True,
            api_endpoint="http://custom:9000",
            api_timeout_s=2.0,
        ),
    )
    assert s.openclaw.enabled is True
    assert s.openclaw.api_endpoint == "http://custom:9000"
    assert s.openclaw.api_timeout_s == 2.0


def test_settings_backwards_compatible():
    """Settings without openclaw key should work (default factory)."""
    s = Settings(mock_hardware=True)
    assert s.openclaw.enabled is False
    assert s.openclaw.fallback_to_cognitive is True
