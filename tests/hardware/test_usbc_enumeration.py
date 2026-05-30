"""Hardware smoke — verifies every required USB-C endpoint resolves.

Skipped on non-Jetson hosts via the shared ``jetson_settings`` fixture
that already keys off ``is_jetson_host()``.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.diagnostics.usbc import EndpointStatus, enumerate_usbc_devices

pytestmark = pytest.mark.hardware


def test_every_required_usbc_endpoint_resolves(jetson_settings: Settings) -> None:
    if jetson_settings.usbc_discovery is None or not jetson_settings.usbc_discovery.enabled:
        pytest.skip("usbc_discovery disabled for this overlay")

    if not jetson_settings.usbc_discovery.by_id_root.is_dir():
        pytest.skip(
            f"by_id_root {jetson_settings.usbc_discovery.by_id_root} "
            "not present (non-Jetson host or USB-C subsystem absent)"
        )

    results = enumerate_usbc_devices(jetson_settings.usbc_discovery)
    missing = [name for name, r in results.items() if r.status is EndpointStatus.MISSING]
    assert not missing, (
        f"USB-C endpoints missing under {jetson_settings.usbc_discovery.by_id_root}: "
        f"{missing}. Plug the rover into the Jetson USB-C port and re-run."
    )
