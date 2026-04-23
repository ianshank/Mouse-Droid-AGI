"""Hardware test fixtures — overrides root mock-hardware autouse fixture.

The root ``tests/conftest.py`` sets ``MOUSEDROID_MOCK_HARDWARE=true`` for
every test via an ``autouse`` fixture.  Hardware integration tests must use
real hardware, so this conftest **reverses** that environment override.

Shared helpers (config loading, Jetson detection) live here to avoid
duplication across the hardware test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._jetson_hardware import is_jetson_host, load_jetson_runtime_settings

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the root mock-hardware fixture for hardware tests."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "false")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jetson_settings() -> Settings:
    """Load full ``Settings`` from the Jetson production config once per session.

    Falls back to ``config/default.yaml`` if the production config is missing
    so that tests can at least instantiate on non-Jetson hosts (they'll be
    skipped later by ``@pytest.mark.hardware`` / importorskip guards).
    """
    settings = load_jetson_runtime_settings()
    if not is_jetson_host():
        settings = settings.model_copy(update={"mock_hardware": True})

    return settings
