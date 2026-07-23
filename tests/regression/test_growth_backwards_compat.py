"""Regression: the growth-distillation config is purely additive.

Guards the CLAUDE.md invariant "New config fields MUST have defaults; existing
YAML files must load unchanged". A Settings built without a ``growth`` block must
leave the field ``None`` (default-off, byte-identical), a legacy YAML without the
block must validate, and ``/metrics`` must render byte-identically until the first
distillation cycle.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.telemetry.metrics import MetricsRegistry

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_settings_without_block_is_none() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.growth is None


def test_legacy_yaml_loads_without_growth() -> None:
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    experience:
      path: /home/jetson/mousedroid_experience
    """
    s = Settings.model_validate(yaml.safe_load(legacy_yaml))
    assert s.growth is None
    assert s.experience.path == "/home/jetson/mousedroid_experience"


def test_block_present_round_trips() -> None:
    s = Settings.model_validate({"mock_hardware": True, "growth": {"enabled": True}})
    assert s.growth is not None
    assert s.growth.enabled is True
    # Unspecified fields fall back to defaults.
    assert s.growth.trigger_min_new_records == 500
    assert s.growth.alpha == 1.0
    assert s.growth.slot_dir == "growth_slot"


def test_shipped_default_yaml_still_parses() -> None:
    """The real shipped default config loads with the new field defaulting None."""
    data = yaml.safe_load((_REPO_ROOT / "config" / "default.yaml").read_text())
    s = Settings.model_validate(data)
    assert s.growth is None


def test_metrics_byte_identical_until_first_cycle() -> None:
    """A registry with no growth activity omits the growth family entirely."""
    reg = MetricsRegistry(MetricsConfig())
    assert "growth_distillations_total" not in reg.render_prometheus()
