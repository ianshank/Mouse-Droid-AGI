"""Hardware test fixtures — overrides root mock-hardware autouse fixture.

The root ``tests/conftest.py`` sets ``MOUSEDROID_MOCK_HARDWARE=true`` for
every test via an ``autouse`` fixture.  Hardware integration tests must use
real hardware, so this conftest **reverses** that environment override.

Shared helpers (config loading, Jetson detection) live here to avoid
duplication across the hardware test modules.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


# ---------------------------------------------------------------------------
# Auto-use: ensure mock hardware is DISABLED for hardware tests
# ---------------------------------------------------------------------------

JETSON_PROD_CONFIG = os.getenv(
    "MOUSEDROID_JETSON_CONFIG",
    "config/jetson_production.yaml",
)


@pytest.fixture(autouse=True)
def _real_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override root conftest's ``_mock_hardware_env`` — disable mock mode.

    This fixture has the same ``autouse=True`` scope and is defined in a
    *more specific* conftest, so pytest will call it **instead of** the
    root-level one for every test collected under ``tests/hardware/``.
    """
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "false")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def is_jetson_host() -> bool:
    """Return True when running on a Jetson (Linux with ``/etc/nv_tegra_release``)."""
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").exists()


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
    import yaml

    from mousedroid.config.schema import Settings

    config_path = Path(JETSON_PROD_CONFIG)
    if not config_path.exists():
        config_path = Path("config/default.yaml")

    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    # On non-Jetson hosts, force mock mode so cross-field validators
    # (e.g. ultrasonic required when mock_hardware=false) do not block
    # fixture creation.  Tests are skipped later by @pytest.mark.hardware.
    if not is_jetson_host():
        raw["mock_hardware"] = True

    return Settings(**raw)
