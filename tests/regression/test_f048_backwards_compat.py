"""F-048 backwards-compat: /metrics unused render unchanged; harness None."""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.telemetry.metrics import MetricsRegistry


def test_metrics_byte_identical_without_isaac_family() -> None:
    out = MetricsRegistry(MetricsConfig()).render_prometheus()
    assert "isaac" not in out.lower()
    assert "track_isaac_sim" not in out


def test_harness_stays_none_on_default_settings() -> None:
    assert Settings(mock_hardware=True).harness is None
