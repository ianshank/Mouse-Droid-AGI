"""Hardware smoke — verifies every required USB-C endpoint resolves.

Gated by the ``@pytest.mark.hardware`` marker AND an explicit
``is_jetson_host()`` skip guard. The shared ``jetson_settings`` fixture
already keys off ``is_jetson_host()``, but the explicit guard here
defends against a non-Jetson Linux host that happens to have
``/dev/serial/by-id`` populated (CodeRabbit finding 9): such a host would
still resolve the fixture and then assert on globs that have no meaning
off the rover.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.diagnostics.usbc import EndpointStatus, enumerate_usbc_devices
from tests._jetson_hardware import is_jetson_host

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not is_jetson_host(), reason="Jetson-only hardware test"),
]


def test_every_required_usbc_endpoint_resolves(jetson_settings: Settings) -> None:
    if jetson_settings.usbc_discovery is None or not jetson_settings.usbc_discovery.enabled:
        pytest.skip("usbc_discovery disabled for this overlay")

    if not jetson_settings.usbc_discovery.by_id_root.is_dir():
        pytest.skip(
            f"by_id_root {jetson_settings.usbc_discovery.by_id_root} "
            "not present (USB-C subsystem absent or pre-udev boot race)"
        )

    results = enumerate_usbc_devices(jetson_settings.usbc_discovery)
    missing = [name for name, r in results.items() if r.status is EndpointStatus.MISSING]
    assert not missing, (
        f"USB-C endpoints missing under {jetson_settings.usbc_discovery.by_id_root}: "
        f"{missing}. Plug the rover into the Jetson USB-C port and re-run."
    )
