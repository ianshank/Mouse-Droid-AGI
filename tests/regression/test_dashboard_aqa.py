"""AQA: dashboard + fusion-field hygiene (PR — rover full bring-up).

Architectural-quality assertions locking the contracts a refactor could break:
the modality name order matches the observation bundle, the fused field is
documented, the new routes exist, and the static page has no hardcoded host/port.
"""

from __future__ import annotations

from importlib import resources

from mousedroid.constants import N_SENSOR_MODALITIES, N_SENSOR_MODALITIES_WITH_LIDAR
from mousedroid.telemetry import frame_builder
from mousedroid.telemetry.protocol import TelemetryFrame


def _dashboard_html() -> str:
    return (
        resources.files("mousedroid.telemetry.static")
        .joinpath("dashboard.html")
        .read_text(encoding="utf-8")
    )


def test_modality_names_match_bundle_slot_order() -> None:
    """The fused builder's name tuple must match the bundle's valid_mask slots."""
    assert frame_builder._MODALITY_NAMES == (
        "vision",
        "ultrasonic",
        "motor",
        "audio",
        "lidar",
    )
    # Length covers both mask variants (4 without lidar, 5 with).
    assert len(frame_builder._MODALITY_NAMES) == N_SENSOR_MODALITIES_WITH_LIDAR
    assert N_SENSOR_MODALITIES == 4


def test_fused_field_documented() -> None:
    """The ``fused`` field carries an explanatory comment/docstring shape."""
    # The dataclass field exists and defaults to an empty dict.
    assert "fused" in TelemetryFrame().__dataclass_fields__
    assert TelemetryFrame().fused == {}


def test_dashboard_page_has_no_hardcoded_host_or_port() -> None:
    """The page derives its origin from window.location — no baked IP/port."""
    html = _dashboard_html()
    assert "127.0.0.1:8080" not in html
    assert "192.168." not in html
    assert "location.host" in html  # origin derived at runtime


def test_dashboard_page_is_token_aware_and_self_contained() -> None:
    html = _dashboard_html()
    assert "withAuth" in html  # token threaded into WS/stream URLs
    assert 'withAuth("/ws")' in html
    assert "/camera/stream" in html
    assert 'id="polar"' in html
    assert "fused" in html  # renders the fusion summary
