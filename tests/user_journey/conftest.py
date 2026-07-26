"""Fixtures for user journey tests."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


@pytest.fixture
def user_journey_settings() -> Settings:
    """Provide a Settings instance tailored for user journey tests."""
    return Settings(mock_hardware=True)


@pytest.fixture
def mock_orchestrator(user_journey_settings: Settings):
    """Provide an orchestrator built with mock hardware for user journey testing."""
    return build_orchestrator(user_journey_settings)
