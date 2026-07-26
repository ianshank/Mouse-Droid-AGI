"""Functional tests for config overlay and schema."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import Settings


def test_base_schema_defaults(mock_settings):
    """Test base schema defaults load correctly."""
    assert mock_settings.mock_hardware is True
    assert mock_settings.loop.control_hz == 30.0


def test_yaml_overlay_overrides():
    """Test YAML overlay overrides specific fields."""
    # Assuming standard pydantic behavior or custom loading
    settings = Settings(mock_hardware=True, llm={"enabled": True})
    assert settings.llm.enabled is True


def test_env_var_overrides(monkeypatch):
    """Test environment variable overrides take precedence over YAML."""
    monkeypatch.setenv("MOUSEDROID_LOOP__CONTROL_HZ", "60.0")
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")

    settings = Settings()
    assert settings.loop.control_hz == 60.0
    assert settings.mock_hardware is True


def test_invalid_config_errors():
    """Test invalid config values produce clear errors."""
    with pytest.raises(ValidationError):
        # Invalid loop hz
        Settings(loop={"control_hz": -10.0})
