"""Shared test fixtures for MouseDroid test suite."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


@pytest.fixture(autouse=True)
def _mock_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock hardware is enabled for all tests."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")


@pytest.fixture
def mock_settings() -> Settings:
    """Create a Settings instance with mock hardware enabled."""
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    from mousedroid.config.schema import Settings

    return Settings(mock_hardware=True)
