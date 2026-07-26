"""Shared fixtures for security tests."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings


@pytest.fixture
def secure_settings(mock_settings: Settings) -> Settings:
    """Settings instance with API key auth enabled."""
    secure = mock_settings.model_copy(deep=True)
    secure.telemetry.api_key = "test-secret-key-123"
    if secure.telemetry.auth is None:
        # Mock setting an auth config if it exists
        from mousedroid.config.schema import TelemetryAuthConfig

        secure.telemetry.auth = TelemetryAuthConfig(auth_enabled=True)
    else:
        secure.telemetry.auth.auth_enabled = True
    return secure
