"""Smoke: sub-second sanity for the unified dashboard + fusion summary.

Imports the touched modules, confirms the static page parses, and checks the
frame builder emits a populated ``fused`` summary. Selected first in CI via
``-m smoke``.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_modules_import() -> None:
    import mousedroid.telemetry.frame_builder
    import mousedroid.telemetry.protocol
    import mousedroid.telemetry.server  # noqa: F401


def test_dashboard_html_present_and_well_formed() -> None:
    html = (
        resources.files("mousedroid.telemetry.static")
        .joinpath("dashboard.html")
        .read_text(encoding="utf-8")
    )
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "</html>" in html
    assert "MouseDroid — Dashboard" in html


def test_frame_builder_emits_fused() -> None:
    from mousedroid.safety.context import SafetyContext
    from mousedroid.sensing.bundle import MouseDroidObservationBundle
    from mousedroid.telemetry.frame_builder import build_telemetry_frame

    obs = MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.ones(4, dtype=np.float32),
        _distance_m=1.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(8, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    frame = build_telemetry_frame(
        obs, SafetyContext(is_emergency=False), loop_time_ms=1.0, tick_count=1
    )
    assert frame.fused["n_modalities"] == 4
    assert "modalities" in frame.fused
