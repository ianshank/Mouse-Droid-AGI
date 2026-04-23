"""Fixtures for Jetson-only end-to-end suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._jetson_hardware import is_jetson_host, load_jetson_runtime_settings

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


@pytest.fixture(autouse=True)
def _mock_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the root mock-hardware fixture for Jetson E2E tests."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "false")


@pytest.fixture
def runtime_settings() -> Settings:
    """Load real-hardware runtime settings for Jetson E2E tests."""
    if not is_jetson_host():
        pytest.skip("Jetson E2E tests require a Jetson host")

    return load_jetson_runtime_settings()
